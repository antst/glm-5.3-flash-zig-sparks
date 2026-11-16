//! Explicit target affine Q4 qualification; the ordinary runner remains unchanged.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const fixed = @import("glm_mm_fixed_tests.zig");
const limits = @import("core").contract.limits;

pub fn run(gpu: check.Gpu, captured: bool) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var group = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer group.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer ops.unload();
    var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue.unload();
    var mm = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer mm.unload();
    defer stream.synchronize() catch @panic("affine target fixture must drain before module teardown");
    var qs = try glm.q4.Kernels.load(group, ops, gpu.ctx);
    try glm.q4.enableWide(&qs, group);
    try glm.q4.enableRows(&qs, group);
    var kernels = try glm.mm.Kernels.load(mm);
    kernels.affine = .{ .q4 = qs, .sums = try glue.function("tf_glm_group_sums") };
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var problems: usize = 0;
    for ([_]cuda.Function{ qs.group, qs.group_f32.?, qs.rows.?[0], qs.rows.?[1] }, 0..) |f, i| {
        const usage: limits.Usage = .{
            .regs = @intCast(try f.attribute(.num_regs)),
            .static = @intCast(try f.attribute(.shared_size_bytes)),
            .bound = @intCast(try f.attribute(.max_threads_per_block)),
            .local = @intCast(try f.attribute(.local_size_bytes)),
        };
        for ([_]u32{ 1, 2, 4, 8 }) |cluster| problems += try limits.cuda(.{
            .name = "GLM target affine Q4",
            .symbol = "qmm_group row16/64 BF16/FP32",
            .image = "qmm_group",
            .threads = 128,
            .dynamic = if (i < 2) 35328 else 43008,
            .cluster = cluster,
            .opt_in = true,
            .min_sm = 90,
        }, usage, target, &log, gpu.gpa);
        std.debug.print("RESOURCE affine-target i{d} regs{d} local{d} bound{d}\n", .{ i, usage.regs, usage.local, usage.bound });
    }
    std.debug.print("{s}", .{log.items});
    try check.expect(problems == 0, "affine target live resource gate", .{});
    if (!captured) try quantizer(gpu, stream, target);
    var red: usize = 0;
    for (fixed.target_shapes) |shape| {
        try fixed.shapeMode(gpu, &kernels, stream, 4, shape[0], shape[1], 5, &red, captured);
        check.pass("target affine N{d} K{d} captured={any}, FP64/row-alone/guards/repeat", .{ shape[0], shape[1], captured });
    }
    try check.expect(red > 0, "affine FP32 stress rejects reduced precision", .{});
    check.pass("AFFINE-TARGET SUMMARY shapes13 captured={any} FP16-red={d}", .{ captured, red });
}

fn quantizer(gpu: check.Gpu, stream: cuda.Stream, target: limits.Sm) !void {
    var module = try glm.affine_quant.load(gpu.gpa, gpu.io, gpu.d);
    defer module.unload();
    defer stream.synchronize() catch @panic("quantizer fixture must drain before module teardown");
    const f = try module.function("tf_glm_q4_mse_v1");
    const usage: limits.Usage = .{
        .regs = @intCast(try f.attribute(.num_regs)),
        .static = @intCast(try f.attribute(.shared_size_bytes)),
        .bound = @intCast(try f.attribute(.max_threads_per_block)),
        .local = @intCast(try f.attribute(.local_size_bytes)),
    };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    const problems = try limits.cuda(.{ .name = "Q4 affine MSE pack", .symbol = "tf_glm_q4_mse_v1", .image = "glm_q4_mse", .threads = 128 }, usage, target, &log, gpu.gpa);
    std.debug.print("{s}", .{log.items});
    try check.expect(problems == 0, "quantizer live resource gate", .{});
    for ([_]usize{ 1, 3, 4, 5, 17, 32, 41 }) |groups| {
        var rig: Rig = .{ .gpu = gpu, .s = stream };
        defer rig.deinit();
        const src = try gpu.gpa.alloc(u16, groups * 64);
        defer gpu.gpa.free(src);
        const words = try gpu.gpa.alloc(u32, groups * 8);
        defer gpu.gpa.free(words);
        const scales = try gpu.gpa.alloc(u16, groups);
        defer gpu.gpa.free(scales);
        const biases = try gpu.gpa.alloc(u16, groups);
        defer gpu.gpa.free(biases);
        for (0..groups) |g| {
            for (src[g * 64 ..][0..64], 0..) |*h, i| h.* = glm.affine_reference.fixture(g, i);
            const q = try glm.affine_reference.quantize(src[g * 64 ..][0..64].*);
            @memcpy(words[g * 8 ..][0..8], &q.words);
            scales[g] = q.scale;
            biases[g] = q.bias;
        }
        const input = try rig.input(std.mem.sliceAsBytes(src));
        const dw = try rig.output(words.len * 4);
        const ds = try rig.output(scales.len * 2);
        const db = try rig.output(biases.len * 2);
        const status = try rig.output(4);
        for (0..5) |_| {
            try rig.zero(status, 4);
            try glm.affine_quant.quantize(f, stream, input, groups, 64, dw, ds, db, status);
            try check.sameBytes("affine quantizer words", try rig.bytes(dw, words.len * 4), std.mem.sliceAsBytes(words));
            try check.sameBytes("affine quantizer stored scale", try rig.bytes(ds, scales.len * 2), std.mem.sliceAsBytes(scales));
            try check.sameBytes("affine quantizer stored bias", try rig.bytes(db, biases.len * 2), std.mem.sliceAsBytes(biases));
            try check.expect(std.mem.readInt(u32, (try rig.bytes(status, 4))[0..4], .little) == 0, "affine quantizer valid weights", .{});
            try check.expect(try rig.intact(), "affine quantizer all guards", .{});
            try check.sameBytes("affine quantizer input immutable", try rig.inputBytes(), std.mem.sliceAsBytes(src));
        }
    }
    var rig: Rig = .{ .gpu = gpu, .s = stream };
    defer rig.deinit();
    var bad: [64]u16 = @splat(0);
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f7f }) |h| {
        rig.reset();
        bad = @splat(0);
        bad[7] = h;
        if (h == 0x7f7f) bad[8] = 0xff7f;
        const src = try rig.input(std.mem.sliceAsBytes(&bad));
        const words = try rig.output(32);
        const scales = try rig.output(16);
        const biases = try rig.output(16);
        const status = try rig.output(4);
        try rig.zero(status, 4);
        try glm.affine_quant.quantize(f, stream, src, 1, 64, words, scales, biases, status);
        try check.expect(std.mem.readInt(u32, (try rig.bytes(status, 4))[0..4], .little) != 0, "affine quantizer refuses invalid weights", .{});
        try check.expect(try rig.intact(), "affine quantizer refusal guards", .{});
    }
    check.pass("affine MSE quantizer bytes,7 tail shapes*5repeats,4 invalid cases", .{});
}
