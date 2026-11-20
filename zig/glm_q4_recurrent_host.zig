//! Principal role, loader ABI, strict setting and all-rank agreement controls; no device library.
const std = @import("std");
const glm = @import("glm");
const t = std.testing;

test {
    std.testing.refAllDecls(glm.q4_recurrent);
}

test "Q4 recurrent BF16: cached strict choice, inherited calibration scope, and all-rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_Q4_RECURRENT_BF16")) return value;
            return null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expect(off.q4_recurrent_bf16 == 0);
    Env.value = "0";
    try t.expectEqualDeep(off, try glm.ranks.Settings.read(Env.get, &problem));
    for ([_][]const u8{ "4", "01", "true", " 1" }) |bad| {
        Env.value = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    var on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(on.q4_recurrent_bf16 == 1);
    try t.expectError(error.Q4RecurrentBf16Scope, glm.q4_recurrent.require(on, 2));
    on.q4_affine_mse = true;
    on.q4_activation = true;
    on.dense_bits = 4;
    try glm.q4_recurrent.require(on, 2);
    try t.expectError(error.Q4RecurrentBf16Scope, glm.q4_recurrent.require(on, 4));
    const FP = glm.ranks.Fingerprint;
    var previous = on;
    previous.q4_recurrent_bf16 = 0;
    const a = FP.of(2, 0, previous, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 85), b.layout);
    try t.expectEqual(FP.Field.q4_recurrent_bf16, glm.ranks.mismatch(&.{ a, b }).?.field);
}
