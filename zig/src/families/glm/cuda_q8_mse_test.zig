//! Independent encoded-code reconstruction and startup agreement for the quality experiment.
const std = @import("std");
const mse = @import("cuda_q8_mse.zig");
const legacy = @import("cuda_qpack.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

fn bf(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn error64(src: []const u16, words: []const u32, scales: []const u16) f64 {
    var total: f64 = 0;
    for (src, 0..) |h, i| {
        const shift: u5 = @intCast((i % 4) * 8);
        const q = @as(i32, @intCast((words[i / 4] >> shift) & 255)) - 128;
        const x: f32 = @bitCast(@as(u32, h) << 16);
        const sc: f64 = @as(f16, @bitCast(scales[i / 64]));
        const e = @as(f64, @floatFromInt(q)) * sc - x;
        total += e * e;
    }
    return total;
}

test "Q8 MSE: actual codes never increase group error and repeat exactly" {
    var input: [128 * 64]u16 = undefined;
    var random = std.Random.DefaultPrng.init(140017);
    for (&input, 0..) |*h, i| {
        const v: f32 = switch ((i / 64) % 8) {
            0 => 0,
            1 => -0.125,
            2 => if (i % 64 == 63) 1.375 else random.random().floatNorm(f32) * 0.05,
            3 => random.random().floatNorm(f32) * 0.25,
            4 => random.random().floatNorm(f32) * 1e-9,
            5 => random.random().floatNorm(f32) * 1024,
            6 => @as(f32, @floatFromInt(@as(i32, @intCast(i % 255)) - 127)) / 128,
            else => if (i % 2 == 0) -0.0 else 0.0,
        };
        h.* = bf(v);
    }
    const before = input;
    var old: [input.len / 4]u32 = undefined;
    var old_sc: [input.len / 64]u16 = undefined;
    legacy.quantize(8, &input, 128, 64, &old, &old_sc);
    var encoded: [old.len]u32 = undefined;
    var scales: [old_sc.len]u16 = undefined;
    const stat = try mse.quantize(&input, 128, 64, &encoded, &scales);
    try t.expectEqual(@as(usize, 128), stat.groups);
    try t.expect(stat.improved > 0);
    var independent_old: f64 = 0;
    var independent_new: f64 = 0;
    for (0..128) |g| {
        const row = input[g * 64 ..][0..64];
        const e0 = error64(row, old[g * 16 ..][0..16], old_sc[g..][0..1]);
        const e1 = error64(row, encoded[g * 16 ..][0..16], scales[g..][0..1]);
        try t.expect(std.math.isFinite(e1) and e1 <= e0);
        independent_old += e0;
        independent_new += e1;
    }
    try t.expectEqual(independent_old, stat.baseline_error);
    try t.expectEqual(independent_new, stat.selected_error);
    try t.expect(independent_new < independent_old);
    for (0..5) |_| {
        var again: [old.len]u32 = undefined;
        var again_sc: [old_sc.len]u16 = undefined;
        try t.expectEqualDeep(stat, try mse.quantize(&input, 64, 128, &again, &again_sc));
        try t.expectEqualSlices(u32, &encoded, &again);
        try t.expectEqualSlices(u16, &scales, &again_sc);
        try t.expectEqualSlices(u16, &before, &input);
    }
}

test "Q8 MSE: invalid shape weights and alias refuse before any output write" {
    var input: [64]u16 = @splat(bf(0.125));
    var words: [16]u32 = @splat(0xdeadbeef);
    var scales: [1]u16 = @splat(0xa55a);
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f7f }) |bad| {
        input[23] = bad;
        try t.expectError(error.Q8MseWeight, mse.quantize(&input, 1, 64, &words, &scales));
        try t.expectEqualSlices(u32, &@as([16]u32, @splat(0xdeadbeef)), &words);
        try t.expectEqual(@as(u16, 0xa55a), scales[0]);
    }
    input[23] = bf(0.125);
    try t.expectError(error.Q8MseShape, mse.quantize(&input, 0, 64, &words, &scales));
    try t.expectError(error.Q8MseShape, mse.quantize(&input, 1, 63, &words, &scales));
    try t.expectError(error.Q8MseShape, mse.quantize(&input, 1, 64, words[0..15], &scales));
    try t.expectError(error.Q8MseAlias, mse.quantize(&input, 1, 64, &words, input[0..1]));
}

test "Q8 MSE: strict cached setting scope and rank agreement leave OFF inert" {
    const Env = struct {
        var mode: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q8_MSE")) mode else null;
        }
    };
    defer Env.mode = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.mode = "0";
    try t.expectEqualDeep(off, try ranks.Settings.read(Env.get, &problem));
    try mse.require(off, 1, true);
    for ([_][]const u8{ "", "01", "true", "2" }) |bad| {
        Env.mode = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.mode = "1";
    var on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(on.q8_mse);
    try t.expectError(error.Q8MseScope, mse.require(on, 2, false));
    on.dense_bits = 8;
    try mse.require(on, 2, false);
    try t.expectError(error.Q8MseScope, mse.require(on, 4, false));
    try t.expectError(error.Q8MseScope, mse.require(on, 2, true));
    inline for (.{ "pages", "verify_graphs", "q4_affine_mse" }) |field| {
        var bad = on;
        @field(bad, field) = true;
        try t.expectError(error.Q8MseScope, mse.require(bad, 2, false));
    }
    var previous = on;
    previous.q8_mse = false;
    const FP = ranks.Fingerprint;
    const a = FP.of(2, 0, previous, 4096, false, @splat(0), @splat(0), @splat(0));
    const b = FP.of(2, 1, on, 4096, false, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.q8_mse, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_Q8_MSE", @import("cuda_rank_names.zig").explain(FP.Field.q8_mse));
}

test "Q8 MSE: experimental NVMe mode refuses before inspecting engine or allocating" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.settings.q8_mse = true;
    try t.expectError(error.Q8MseNvmeScope,
        @import("cuda_nvme_state.zig").State.open(&e, "unused", @splat(0), undefined));
}
