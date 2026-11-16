//! Optional Q4 target head: group64 E4M3 weights with FP32 scales, using the existing native FP8G lane.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const tile = @import("cuda_tile.zig");
pub const image_sha = "0e3d6e28130225f7c7fb633d23b27f20f347e23b017599793d8e6d4e1a08a0ad";

pub const Shape = struct {
    n: usize,
    k: usize,
    npad: usize,
    codes: usize,
    scales: usize,
    pub fn make(n: usize, k: usize) !Shape {
        if (n == 0 or k == 0 or k % 128 != 0 or n > 131072 or k > 65536) return error.HeadFp8Shape;
        const npad = cuda.fp8.padded(n);
        return .{ .n = n, .k = k, .npad = npad, .codes = npad * k, .scales = npad * (k / 64) * 4 };
    }
};

pub const Kernels = struct {
    quant: cuda.Function,
    scales: cuda.Function,
    pub fn load(module: cuda.Module) !Kernels {
        return .{ .quant = try module.function("tf_glm_head_fp8_quant_g64_v1"), .scales = try module.function("tf_glm_head_fp8_scales_g64_v1") };
    }
    pub fn quantize(ks: Kernels, stream: cuda.Stream, shape: Shape, src: u64, codes: u64, scales: u64, status: u64) !void {
        if (src % 16 != 0 or codes % 16 != 0 or scales % 16 != 0 or status % 4 != 0) return error.HeadFp8Shape;
        const groups = shape.n * (shape.k / 64);
        var args: cuda.Args = .{};
        inline for (.{ src, @as(i64, @intCast(groups)), codes, scales, status }) |v| args.add(v);
        try cuda.launch.launch(ks.quant, .{ .grid = .{ .x = @intCast((groups + 3) / 4) }, .block = .{ .x = 128 } }, stream, &args);
    }
    pub fn packScales(ks: Kernels, stream: cuda.Stream, shape: Shape, src: u64, dst: u64) !void {
        var args: cuda.Args = .{};
        inline for (.{ src, dst, @as(c_int, @intCast(shape.n)), @as(c_int, @intCast(shape.k)), @as(c_int, @intCast(shape.npad)) }) |v| args.add(v);
        try cuda.launch.launch(ks.scales, .{ .grid = .{ .x = @intCast((shape.scales / 4 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !cuda.Module {
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_HEAD_FP8_IMAGE") orelse return error.HeadFp8ImageRequired;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.HeadFp8ImageHash;
    return cuda.Module.load(d, bytes);
}

pub fn pack(gpa: std.mem.Allocator, io: std.Io, w: *weights.Weights, lane: *const cuda.qmmf.Lane, stream: cuda.Stream, d: *const cuda.Driver) !void {
    if (w.head != .bf16 or (w.share.world != 2 and w.share.world != 4)) return error.HeadFp8Scope;
    const shape = try Shape.make(w.share.vocab.len(), w.config.hidden);
    const old = w.head.bf16;
    const at = tile.bufferAt(w, old) orelse return error.HeadFp8SharedWeights;
    var module = try load(gpa, io, d);
    defer module.unload();
    const ks = try Kernels.load(module);
    var packer = try cuda.modelopt.Packer.load(d);
    defer packer.unload();
    var raw = try cuda.DeviceBuffer.alloc(d, shape.n * shape.k);
    defer raw.free();
    var scales = try cuda.DeviceBuffer.alloc(d, shape.n * (shape.k / 64) * 4);
    defer scales.free();
    var status = try cuda.DeviceBuffer.alloc(d, 4);
    defer status.free();
    var codes = try cuda.DeviceBuffer.alloc(d, shape.codes);
    errdefer codes.free();
    var tiled_scales = try cuda.DeviceBuffer.alloc(d, shape.scales);
    errdefer tiled_scales.free();
    errdefer @import("cuda_load_owner.zig").drain(stream, d);
    const q = cuda.fp8.weight(codes.ptr, tiled_scales.ptr, shape.n, shape.k);
    try @import("cuda_proj.zig").fits(lane, q);
    try d.check(d.api.cuMemsetD8Async(status.ptr, 0, 4, stream.handle), "cuMemsetD8Async");
    try ks.quantize(stream, shape, old, raw.ptr, scales.ptr, status.ptr);
    try packer.fp8Lane(stream, raw.ptr, codes.ptr, shape.n, shape.k);
    try ks.packScales(stream, shape, scales.ptr, tiled_scales.ptr);
    try stream.synchronize();
    var invalid: u32 = 0;
    try cuda.memory.download(d, status.ptr, std.mem.asBytes(&invalid));
    if (invalid != 0) return error.HeadFp8InvalidWeights;
    try w.buffers.ensureUnusedCapacity(w.gpa, 1);
    tile.replace(w, at, codes);
    w.buffers.appendAssumeCapacity(tiled_scales);
    w.head = .{ .fp8g = q };
    std.log.info("rank {d}: GLM_HEAD_FP8 active=true row_group=64 scale=f32 n={d} k={d} bytes={d} image={s}", .{ w.share.rank, shape.n, shape.k, shape.codes + shape.scales, image_sha });
}
