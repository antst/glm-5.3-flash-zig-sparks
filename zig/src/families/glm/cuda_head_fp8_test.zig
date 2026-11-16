//! Head-format opt-in and rank agreement; default target and affine-head16 modes remain available.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const affine = @import("cuda_tile_affine.zig");
const head = @import("cuda_head_fp8.zig");
const t = std.testing;

test "head FP8: strict cached format, precision scopes and complete fingerprint" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q4_HEAD_FP8")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    try affine.require(off, 1, true);
    for ([_][]const u8{ "", "true", "01", "+1", " 1" }) |bad| {
        Env.value = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    var on = try ranks.Settings.read(Env.get, &problem);
    try t.expectError(error.HeadFp8Scope, affine.require(on, 4, false));
    on.q4_affine_mse = true;
    on.dense_bits = 4;
    try t.expectError(error.AffineTargetScope, affine.require(on, 4, false));
    on.head_bits = 8;
    try affine.require(on, 4, false);
    try t.expect(@import("cuda_target_precision.zig").supported(on));
    var legacy = on;
    legacy.q4_head_fp8 = false;
    try t.expectError(error.AffineTargetScope, affine.require(legacy, 4, false));
    legacy.head_bits = 16;
    try affine.require(legacy, 4, false);
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, legacy, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 61), a.layout);
    try t.expectEqual(FP.Field.q4_head_fp8, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_Q4_HEAD_FP8", @import("cuda_rank_names.zig").explain(FP.Field.q4_head_fp8));
    on.q4_head_fp8 = false;
    on.head_bits = 16;
    try t.expect(std.meta.eql(legacy, on));
}

test "head FP8: storage geometry holds independent group64 scales and padded rows" {
    for ([_]usize{ 1, 65, 129, 38656, 38784 }) |n| {
        const q = try head.Shape.make(n, 4096);
        try t.expectEqual(@as(usize, (n + 127) / 128 * 128), q.npad);
        try t.expectEqual(q.npad * 4096, q.codes);
        try t.expectEqual(q.npad * 64 * 4, q.scales);
    }
    try t.expectError(error.HeadFp8Shape, head.Shape.make(0, 4096));
    try t.expectError(error.HeadFp8Shape, head.Shape.make(65537, 4096));
    try t.expectError(error.HeadFp8Shape, head.Shape.make(129, 4097));
}
