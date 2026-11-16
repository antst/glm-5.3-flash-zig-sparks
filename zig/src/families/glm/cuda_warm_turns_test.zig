//! Strict opt-in and rank agreement for actual cached next-turn prefill.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
test "GLM warm turns: strict default off and immutable rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_WARM_TURNS")) value else null;
        }
    };
    defer Env.value = null;
    const t = std.testing;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.warm_turns);
    Env.value = "0";
    try t.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "true", "01", "+1", " 1" }) |bad| {
        Env.value = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(on.warm_turns and !(try ranks.Settings.read(Env.get, &problem)).warm_turns);
    const a = ranks.Fingerprint.of(4, 0, off, 196608, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 196608, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 62), a.layout);
    try t.expectEqual(ranks.Fingerprint.Field.warm_turns, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_WARM_TURNS", @import("cuda_rank_names.zig").explain(ranks.Fingerprint.Field.warm_turns));
}
