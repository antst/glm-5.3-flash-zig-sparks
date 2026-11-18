//! Optional calibrated affine packing; no calibration or asset access while disabled.
pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !bool {
    return @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_ACTIVATION"), problem, "TF_GLM_Q4_ACTIVATION: 0 or 1");
}

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (!s.q4_activation) return;
    if (world != 2 or !s.q4_affine_mse or s.q8_mse or s.q8_activation or s.dense_bits != 4 or
        s.head_bits != @as(u8, if (s.q4_head_fp8) 8 else 16) or !s.tiled or
        !s.dflash_only_head or !o.drafter or !o.mtp or o.vision != null or s.pages or s.kv_host != 0 or
        s.verify_graphs) return error.Q4ActivationScope;
}
