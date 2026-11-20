//! Startup-agreed, default-off fixed-scale correlated rounding.
const moments = @import("cuda_q8_moments.zig");
pub const Choice = struct { on: bool = false, sha: [32]u8 = @splat(0) };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Choice {
    const on = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q8_CORRELATED"), problem, "TF_GLM_Q8_CORRELATED: 0 or 1");
    if (!on) return .{};
    const text = get("TF_GLM_Q8_CORRELATION_SHA256") orelse {
        problem.* = "Correlated Q8 requires TF_GLM_Q8_CORRELATION_SHA256";
        return error.BadSetting;
    };
    const sha = moments.digest(text) catch {
        problem.* = "TF_GLM_Q8_CORRELATION_SHA256: exactly64 hexadecimal digits";
        return error.BadSetting;
    };
    return .{ .on = true, .sha = sha };
}

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (!s.q8_correlated.on) return;
    if (world != 2 or !s.q8_activation or !s.q8_mse or s.dense_bits != 8 or s.head_bits != 16 or !s.tiled or
        !s.dflash_only_head or !o.drafter or !o.mtp or o.vision != null or (s.pages and !s.kv8) or s.kv_host != 0 or
        s.verify_graphs or s.q4_affine_mse or s.q4_activation) return error.Q8CorrelatedScope;
    const expected = try moments.digest("3f3566c55dfa5fff0b194fc53f8db0c6bb8a973f705d6e75e4219b6a919b64ce");
    if (!@import("std").mem.eql(u8, &s.calibration_sha, &expected)) return error.Q8CorrelatedScope;
}
