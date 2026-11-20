//! Startup-only weighted fitting owner; inference consumes the unchanged affine pack layout.
const std = @import("std");
const cuda = @import("cuda");

// Deliberately unpinned: the draft cannot load until its actual compiled asset is qualified.
pub const image_sha = "c746268104167d9ec12d422a8f5f71d9c3ab9f81c7a5e583cc06e629ec22de86";
pub const View = struct { ptr: u64, len: usize };
pub const Buffers = struct { source: View, moments: View, words: View, scales: View, biases: View, status: View };

pub fn validateMap(data: *const @import("cuda_q8_moments.zig").Data, list: anytype, head_bf16: bool) !void {
    const head: usize = @intFromBool(head_bf16);
    if (list.len < head) return error.Q4CalibrationShape;
    const targets = list.len - head;
    if (data.target_count != targets or data.entries.len != targets + 1) return error.Q4CalibrationShape;
    for (list, 0..) |linear, i| _ = try data.linear(i, linear.n, linear.k);
}

pub const Owner = struct {
    module: cuda.Module,
    function: cuda.Function,

    pub fn deinit(self: *Owner) void {
        self.module.unload();
        self.* = undefined;
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, on: bool) !?Owner {
    if (!on) return null;
    if (image_sha.len != 64) return error.Q4ActivationImageUnbuilt;
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_Q4_ACTIVATION_IMAGE") orelse
        return error.Q4ActivationImageRequired;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.Q4ActivationImageHash;
    var module = try cuda.Module.load(d, bytes);
    errdefer module.unload();
    return .{ .module = module, .function = try module.function("tf_glm_q4_activation_refine_v1") };
}

fn fits(v: View, bytes: usize, alignment: u64) bool {
    return v.ptr != 0 and v.ptr % alignment == 0 and v.len >= bytes and
        bytes <= std.math.maxInt(u64) - v.ptr;
}

fn overlap(a: View, b: View) bool {
    return if (a.ptr <= b.ptr) b.ptr - a.ptr < a.len else a.ptr - b.ptr < b.len;
}

pub fn validate(n: usize, k: usize, b: Buffers) !usize {
    if (n == 0 or k == 0 or k % 64 != 0) return error.Q4ActivationShape;
    const count = try std.math.mul(usize, n, k);
    const groups = count / 64;
    if (groups > std.math.maxInt(u32)) return error.Q4ActivationShape;
    const sizes = [_]usize{ try std.math.mul(usize, count, 2), try std.math.mul(usize, k, 8), count / 2, try std.math.mul(usize, groups, 2), try std.math.mul(usize, groups, 2), 16 };
    const views = [_]View{ b.source, b.moments, b.words, b.scales, b.biases, b.status };
    const alignments = [_]u64{ 2, 8, 4, 2, 2, 4 };
    for (views, sizes, alignments) |v, bytes, alignment|
        if (!fits(v, bytes, alignment)) return error.Q4ActivationCapacity;
    for (views, sizes, 0..) |v, bytes, i| for (views[i + 1 ..], sizes[i + 1 ..]) |other, other_bytes| {
        if (overlap(.{ .ptr = v.ptr, .len = bytes }, .{ .ptr = other.ptr, .len = other_bytes }))
            return error.Q4ActivationAlias;
    };
    return groups;
}

pub fn launch(f: cuda.Function, stream: cuda.Stream, n: usize, k: usize, b: Buffers) !void {
    const groups = try validate(n, k, b);
    var args: cuda.Args = .{};
    inline for (.{ b.source.ptr, b.moments.ptr, @as(i64, @intCast(groups)), @as(i64, @intCast(k / 64)), b.words.ptr, b.scales.ptr, b.biases.ptr, b.status.ptr }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((groups + 127) / 128) }, .block = .{ .x = 128 } }, stream, &args);
}
