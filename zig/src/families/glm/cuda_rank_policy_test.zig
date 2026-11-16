//! A transport comparison must not silently change the DFlash MLP's producer or arithmetic.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const df = @import("dflash.zig");

test "dense segment startup mode is strict default-off and agreed before allocation" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DENSE_LATENT_SEGMENTS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(!off.dense_segments);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(on.dense_segments);
    const FP = ranks.Fingerprint;
    var all = [_]FP{
        FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0)),
        FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expectEqual(FP.Field.dense_segments, ranks.mismatch(&all).?.field);
    all[0] = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expect(ranks.mismatch(&all) == null);
    try std.testing.expectEqual(@as(u32, 61), all[0].layout);
    for ([_][]const u8{ "yes", "", "2", "false" }) |v| {
        Env.value = v;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_DENSE_LATENT_SEGMENTS: 0 or 1", problem);
    }
}

test "dense segment unsupported engine modes refuse before allocation or GPU calls" {
    const Engine = @import("cuda_engine.zig").Engine;
    for ([_]ranks.Settings{
        .{ .dense_segments = true, .pages = true },
        .{ .dense_segments = true, .kv_host = 1 },
        .{ .dense_segments = true, .verify_graphs = true },
    }) |settings| try std.testing.expectError(error.DenseScope, Engine.init(std.testing.allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = settings, .capacity = 4096, .mtp = true }));
    try std.testing.expectError(error.DenseScope, Engine.init(std.testing.allocator, undefined, undefined, undefined, "", undefined, null, 0, 2, .{ .settings = .{ .dense_segments = true }, .capacity = 4096, .mtp = true }));
}

test "prompt K experiment: TP4 transport and DFlash splitting are independent and agreed" {
    const Env = struct {
        var fast: ?[]const u8 = null;
        var mlp: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_ROCE")) return fast;
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_MLP_TP4")) return mlp;
            return null;
        }
    };
    var problem: []const u8 = "";
    for ([_][]const u8{ "0", "1" }) |fast| for ([_][]const u8{ "0", "1" }) |mlp| {
        Env.fast = fast;
        Env.mlp = mlp;
        const s = try ranks.Settings.read(&Env.get, &problem);
        try std.testing.expectEqual(std.mem.eql(u8, fast, "1"), s.roce_wide);
        try std.testing.expectEqual(std.mem.eql(u8, mlp, "1"), df.mlpSplit(4, s.dflash_mlp_wide).split);
    };
    Env.fast = "1";
    Env.mlp = null;
    try std.testing.expect(!df.mlpSplit(4, (try ranks.Settings.read(&Env.get, &problem)).dflash_mlp_wide).split);
    Env.mlp = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(&Env.get, &problem));
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, .{ .dflash_mlp_wide = true }, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(FP.Field.dflash_mlp_wide, ranks.mismatch(&.{ a, b }).?.field);
}

test "greedy batching: record is strict opt-in and fingerprinted before collective schedules differ" {
    const Get = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_GREEDY_BATCH")) value else null;
        }
    };
    defer Get.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Get.get, &problem);
    try std.testing.expect(!off.greedy_batch);
    Get.value = "0";
    try std.testing.expect(!(try ranks.Settings.read(Get.get, &problem)).greedy_batch);
    Get.value = "1";
    const on = try ranks.Settings.read(Get.get, &problem);
    try std.testing.expect(on.greedy_batch);
    Get.value = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Get.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_GREEDY_BATCH: 0 or 1", problem);
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, off, 8192, true, @splat(0), @splat(0), ranks.hashBytes("/c")),
        ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), ranks.hashBytes("/c")),
    };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .greedy_batch, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = ranks.Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), ranks.hashBytes("/c"));
    try std.testing.expect(ranks.mismatch(&all) == null);
}

test "shared64: tap rows cover the full verification arena while per-stream rows stay eight" {
    try std.testing.expectEqual(@as(usize, 64), @import("cuda_engine.zig").shared_rows);
    try std.testing.expectEqual(@as(usize, 64), df.STEP_ROWS);
    try std.testing.expectEqual(@as(usize, 16), @import("cuda_forward.zig").window_rows);
}

test "greedy batching: shared gather strict setting requires row batching and is agreed" {
    const Env = struct {
        var part: ?[]const u8 = "1";
        var shared: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_GREEDY_BATCH")) return part;
            if (std.mem.eql(u8, name, "TF_GLM_GREEDY_SHARED")) return shared;
            return null;
        }
    };
    var why: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &why);
    try std.testing.expect(!off.greedy_shared);
    Env.shared = "1";
    const on = try ranks.Settings.read(Env.get, &why);
    try std.testing.expect(on.greedy_shared);
    Env.shared = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &why));
    Env.shared = "1";
    Env.part = "0";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &why));
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, off, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(FP.Field.greedy_shared, ranks.mismatch(&.{ a, b }).?.field);
}
