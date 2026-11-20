//! The shipped affine pack followed by weighted fitting, checked against the independent serial reference.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const limits = @import("core").contract.limits;
const fit = glm.affine_activation;
const ref = glm.q4_activation_reference;
pub const control_sha = "5663af38d908d48fc064e754e9a72f90fe36bf387debeb9e9fa8a5bff4e6605e";

pub fn run(gpu: check.Gpu, control_path: []const u8) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var original = try glm.affine_quant.load(gpu.gpa, gpu.io, gpu.d);
    defer original.unload();
    var candidate = (try fit.load(gpu.gpa, gpu.io, gpu.d, true)).?;
    defer candidate.deinit();
    var control = try loadControl(gpu, control_path);
    defer control.unload();
    defer stream.synchronize() catch @panic("weighted affine contract must drain before modules unload");
    const first = try original.function("tf_glm_q4_mse_v1");
    const bad = try control.function("tf_glm_q4_activation_fp32_control_v1");
    try resources(gpu, candidate.function, "tf_glm_q4_activation_refine_v1");
    try resources(gpu, bad, "tf_glm_q4_activation_fp32_control_v1");
    var improved: usize = 0;
    for ([_]usize{ 1, 3, 4, 5, 127, 128, 129, 4096 }) |n|
        improved += try shape(gpu, stream, first, candidate.function, n, 64, false);
    for ([_][2]usize{ .{ 3, 128 }, .{ 5, 192 }, .{ 7, 512 }, .{ 2, 4096 }, .{ 2, 12288 } }) |nk|
        improved += try shape(gpu, stream, first, candidate.function, nk[0], nk[1], false);
    _ = try shape(gpu, stream, first, candidate.function, 3, 128, true);
    try check.expect(improved > 0, "weighted fixture must actually change packs", .{});
    try stress(gpu, stream, candidate.function, bad);
    try invalid(gpu, stream, first, candidate.function);
    check.pass("Q4-ACTIVATION SUMMARY shapes14 repeats5 invalid6; actual FP32 and omitted-stage controls red", .{});
}

fn loadControl(gpu: check.Gpu, path: []const u8) !cuda.Module {
    if (control_sha.len != 64) return error.Q4ActivationControlUnbuilt;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, path, gpu.gpa, .limited(16 << 20), .@"16", null);
    defer gpu.gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), control_sha)) return error.Q4ActivationControlHash;
    return cuda.Module.load(gpu.d, bytes);
}

fn resources(gpu: check.Gpu, f: cuda.Function, symbol: []const u8) !void {
    const target = limits.target(try gpu.ctx.capability()) orelse return error.UnmeasuredGpu;
    const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)), .local = @intCast(try f.attribute(.local_size_bytes)) };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    const problems = try limits.cuda(.{ .name = "Q4 activation startup fit", .symbol = symbol, .image = "glm_q4_activation_refine", .threads = 128 }, usage, target, &log, gpu.gpa);
    std.debug.print("{s}RESOURCE {s} regs={d} local={d} bound={d}\n", .{
        log.items, symbol, usage.regs, usage.local, usage.bound,
    });
    try check.expect(problems == 0, "weighted affine live resources", .{});
}

fn shape(gpu: check.Gpu, stream: cuda.Stream, first: cuda.Function, f: cuda.Function, n: usize, k: usize, zero: bool) !usize {
    const groups = n * k / 64;
    const source = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(source);
    const importance = try gpu.gpa.alloc(f64, k);
    defer gpu.gpa.free(importance);
    for (importance, 0..) |*v, i| v.* = if (zero) 0 else if ((i % 64) % 3 == (i / 64) % 3) 100 else 0.01;
    const words = try gpu.gpa.alloc(u32, groups * 8);
    defer gpu.gpa.free(words);
    const scales = try gpu.gpa.alloc(u16, groups);
    defer gpu.gpa.free(scales);
    const biases = try gpu.gpa.alloc(u16, groups);
    defer gpu.gpa.free(biases);
    var stats: [4]u32 = @splat(0);
    for (0..groups) |g| {
        const row: *[64]u16 = source[g * 64 ..][0..64];
        for (row, 0..) |*h, j| h.* = glm.affine_reference.fixture(g, j);
        const old = try glm.affine_reference.quantize(row.*);
        const weights: *const [64]f64 = importance[(g % (k / 64)) * 64 ..][0..64];
        const result = try ref.fit(row, weights, .{ .words = old.words, .scale = old.scale, .bias = old.bias });
        try check.expect(std.math.isFinite(result.after) and result.after <= result.before, "independent weighted objective", .{});
        stats[1] += @intFromBool(result.pack.scale != old.scale or result.pack.bias != old.bias);
        stats[2] += @intFromBool(zero);
        stats[3] = @max(stats[3], result.rounds);
        @memcpy(words[g * 8 ..][0..8], &result.pack.words);
        scales[g] = result.pack.scale;
        biases[g] = result.pack.bias;
    }
    var rig: Rig = .{ .gpu = gpu, .s = stream };
    defer {
        stream.synchronize() catch @panic("weighted affine shape must drain before buffers free");
        rig.deinit();
    }
    const input = try rig.input(std.mem.sliceAsBytes(source));
    const moments = try rig.input(std.mem.sliceAsBytes(importance));
    const dw = try rig.output(words.len * 4);
    const ds = try rig.output(scales.len * 2);
    const db = try rig.output(biases.len * 2);
    const status = try rig.output(16);
    const b: fit.Buffers = .{ .source = .{ .ptr = input, .len = source.len * 2 }, .moments = .{ .ptr = moments, .len = importance.len * 8 }, .words = .{ .ptr = dw, .len = words.len * 4 }, .scales = .{ .ptr = ds, .len = scales.len * 2 }, .biases = .{ .ptr = db, .len = biases.len * 2 }, .status = .{ .ptr = status, .len = 16 } };
    for (0..5) |_| {
        try rig.zero(status, 16);
        try glm.affine_quant.quantize(first, stream, input, n, k, dw, ds, db, status);
        try fit.launch(f, stream, n, k, b);
        try check.sameBytes("weighted words", try rig.bytes(dw, words.len * 4), std.mem.sliceAsBytes(words));
        try check.sameBytes("weighted scales", try rig.bytes(ds, scales.len * 2), std.mem.sliceAsBytes(scales));
        try check.sameBytes("weighted biases", try rig.bytes(db, biases.len * 2), std.mem.sliceAsBytes(biases));
        try check.sameBytes("weighted counts", try rig.bytes(status, 16), std.mem.asBytes(&stats));
        try check.sameBytes("immutable source", try rig.bytes(input, source.len * 2), std.mem.sliceAsBytes(source));
        try check.sameBytes("immutable moments", try rig.bytes(moments, importance.len * 8), std.mem.sliceAsBytes(importance));
        try check.expect(try rig.intact(), "weighted all guards", .{});
    }
    if (stats[1] > 0) {
        try rig.zero(status, 16);
        try glm.affine_quant.quantize(first, stream, input, n, k, dw, ds, db, status);
        const same = std.mem.eql(u8, try rig.bytes(dw, words.len * 4), std.mem.sliceAsBytes(words)) and
            std.mem.eql(u8, try rig.bytes(ds, scales.len * 2), std.mem.sliceAsBytes(scales)) and
            std.mem.eql(u8, try rig.bytes(db, biases.len * 2), std.mem.sliceAsBytes(biases));
        try check.expect(!same, "actual omitted-refinement control red", .{});
    }
    check.pass("Q4 activation n{d} k{d} zero{} five repeats", .{ n, k, zero });
    return stats[1];
}

fn stress(gpu: check.Gpu, stream: cuda.Stream, f: cuda.Function, control: cuda.Function) !void {
    var source: [64]u16 = undefined;
    for (&source, 0..) |*v, i| v.* = ref.bf16(@floatFromInt(i % 4));
    const importance: [64]f64 = @splat(0x1p130);
    const baseline = ref.encode(&source, ref.bf16(0.25), ref.bf16(-0.125));
    const expected = try ref.fit(&source, &importance, baseline);
    try check.expect(expected.after == 0 and expected.before > std.math.floatMax(f32), "FP64 stress witness", .{});
    var rig: Rig = .{ .gpu = gpu, .s = stream };
    defer {
        stream.synchronize() catch @panic("weighted affine stress must drain before buffers free");
        rig.deinit();
    }
    const input = try rig.input(std.mem.asBytes(&source));
    const moments = try rig.input(std.mem.asBytes(&importance));
    const dw = try rig.output(32);
    const ds = try rig.output(2);
    const db = try rig.output(2);
    const status = try rig.output(16);
    const b: fit.Buffers = .{ .source = .{ .ptr = input, .len = 128 }, .moments = .{ .ptr = moments, .len = 512 }, .words = .{ .ptr = dw, .len = 32 }, .scales = .{ .ptr = ds, .len = 2 }, .biases = .{ .ptr = db, .len = 2 }, .status = .{ .ptr = status, .len = 16 } };
    for ([_]cuda.Function{ f, control }, 0..) |function, index| {
        const wb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = dw, .len = 32 };
        const sb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ds, .len = 2 };
        const bb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = db, .len = 2 };
        try wb.upload(0, std.mem.asBytes(&baseline.words));
        try sb.upload(0, std.mem.asBytes(&baseline.scale));
        try bb.upload(0, std.mem.asBytes(&baseline.bias));
        try rig.zero(status, 16);
        try fit.launch(function, stream, 1, 64, b);
        const equal = std.mem.eql(u8, try rig.bytes(dw, 32), std.mem.asBytes(&expected.pack.words)) and
            std.mem.eql(u8, try rig.bytes(ds, 2), std.mem.asBytes(&expected.pack.scale)) and
            std.mem.eql(u8, try rig.bytes(db, 2), std.mem.asBytes(&expected.pack.bias));
        try check.expect(equal == (index == 0), "actual FP32 moment/sum control must be red", .{});
        try check.expect(try rig.intact(), "precision stress guards", .{});
    }
    check.pass("Q4 activation finite FP64 stress and actual FP32 control red", .{});
}

fn invalid(gpu: check.Gpu, stream: cuda.Stream, first: cuda.Function, f: cuda.Function) !void {
    var source: [64]u16 = undefined;
    for (&source, 0..) |*v, i| v.* = glm.affine_reference.fixture(2, i);
    var importance: [64]f64 = @splat(1);
    var rig: Rig = .{ .gpu = gpu, .s = stream };
    defer {
        stream.synchronize() catch @panic("weighted affine invalid fixture must drain before buffers free");
        rig.deinit();
    }
    for (0..6) |panel| {
        rig.reset();
        for (&source, 0..) |*v, i| v.* = glm.affine_reference.fixture(2, i);
        @memset(&importance, 1);
        if (panel < 3) importance[17] = ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) })[panel] else source[17] = ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[panel - 3];
        const input = try rig.input(std.mem.asBytes(&source));
        const moments = try rig.input(std.mem.asBytes(&importance));
        const dw = try rig.output(32);
        const ds = try rig.output(16);
        const db = try rig.output(16);
        const status = try rig.output(16);
        try rig.zero(status, 16);
        try glm.affine_quant.quantize(first, stream, input, 1, 64, dw, ds, db, status);
        try fit.launch(f, stream, 1, 64, .{ .source = .{ .ptr = input, .len = 128 }, .moments = .{ .ptr = moments, .len = 512 }, .words = .{ .ptr = dw, .len = 32 }, .scales = .{ .ptr = ds, .len = 2 }, .biases = .{ .ptr = db, .len = 2 }, .status = .{ .ptr = status, .len = 16 } });
        try check.expect(std.mem.readInt(u32, (try rig.bytes(status, 16))[0..4], .little) != 0, "invalid weights/moments refuse publication", .{});
        try check.sameBytes("invalid source immutable", try rig.bytes(input, 128), std.mem.asBytes(&source));
        try check.sameBytes("invalid moments immutable", try rig.bytes(moments, 512), std.mem.asBytes(&importance));
        try check.expect(try rig.intact(), "invalid weighted guards", .{});
    }
}
