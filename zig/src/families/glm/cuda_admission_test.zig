//! Strict startup admission choice, rank agreement and pre-allocation scope refusals.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const options = @import("cuda_admission_options.zig");

test "GLM admission wait: strict cached default and rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_ADMISSION_US")) value else null;
        }
    };
    defer Env.value = null;
    const t = std.testing;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(@as(u32, 0), off.admission_us);
    Env.value = "0";
    try t.expectEqualDeep(off, try ranks.Settings.read(Env.get, &problem));
    Env.value = "4000";
    const on = try ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(@as(u32, 4000), on.admission_us);
    Env.value = "0";
    try t.expectEqual(@as(u32, 4000), on.admission_us);
    var equal = on;
    equal.admission_us = 0;
    try t.expectEqualDeep(off, equal);
    const FP = ranks.Fingerprint;
    var all = [_]FP{
        FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0)),
        FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0)),
    };
    try t.expectEqual(@as(u32, 61), all[0].layout);
    try t.expectEqual(FP.Field.admission_us, ranks.mismatch(&all).?.field);
    try t.expectEqualStrings("TF_GLM_ADMISSION_US", @import("cuda_rank_names.zig").explain(FP.Field.admission_us));
    all[0] = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expect(ranks.mismatch(&all) == null);
    for ([_][]const u8{ "", "1", "3999", "4001", " 4000", "+4000", "4e3", "4000.0" }) |v| {
        Env.value = v;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_ADMISSION_US: 0 or 4000", problem);
    }
}

test "GLM admission wait: unsupported modes refuse before driver access" {
    const Engine = @import("cuda_engine.zig").Engine;
    const a = std.testing.allocator;
    const run = Engine.init;
    for ([_]ranks.Settings{
        .{ .admission_us = 4000, .pages = true },
        .{ .admission_us = 4000, .kv_host = 1 },
        .{ .admission_us = 4000, .dense_bits = 6 },
        .{ .admission_us = 4000, .head_bits = 8 },
    }) |s| try std.testing.expectError(error.AdmissionWaitScope, run(a, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = s, .capacity = 4096, .mtp = true, .drafter = true }));
    try std.testing.expectError(error.AdmissionWaitScope, run(a, undefined, undefined, undefined, "", undefined, null, 0, 2, .{ .settings = .{ .admission_us = 4000 }, .capacity = 4096, .mtp = true, .drafter = true }));
    try std.testing.expectError(error.AdmissionWaitScope, run(a, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .admission_us = 4000 }, .capacity = 4096, .mtp = true }));
    const fake = struct { settings: ranks.Settings = .{}, mtp: bool = true, drafter: bool = true, vision: ?u8 = null }{};
    try options.require(fake, 4);
}
