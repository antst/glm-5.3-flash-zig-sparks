//! Startup-agreed activation calibration identity; no file access on the disabled path.
const moments = @import("cuda_q8_moments.zig");
pub const Choice = struct { on: bool = false, sha: [32]u8 = @splat(0) };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Choice {
    const on = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q8_ACTIVATION"), problem, "TF_GLM_Q8_ACTIVATION: 0 or 1");
    if (!on) return .{};
    const text = get("TF_GLM_CALIBRATION_SHA256") orelse {
        problem.* = "TF_GLM_Q8_ACTIVATION requires TF_GLM_CALIBRATION_SHA256";
        return error.BadSetting;
    };
    const sha = moments.digest(text) catch {
        problem.* = "TF_GLM_CALIBRATION_SHA256: exactly64 hexadecimal digits";
        return error.BadSetting;
    };
    return .{ .on = true, .sha = sha };
}

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (!s.q8_activation) return;
    if (world != 2 or !s.q8_mse or s.dense_bits != 8 or s.head_bits != 16 or !s.tiled or
        !s.dflash_only_head or !o.drafter or !o.mtp or o.vision != null or s.pages or s.kv_host != 0 or
        s.verify_graphs or s.q4_affine_mse) return error.Q8ActivationScope;
}
