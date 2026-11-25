//! Engine startup scope checks before the first CUDA allocation.
const ranks = @import("cuda_ranks.zig");
const Options = @import("cuda_engine_types.zig").Options;
const prof_mod = @import("cuda_prof.zig");

pub fn validate(o: Options, world: usize) !bool {
    if (world > ranks.max_world) return error.TooManyRanks;
    try @import("cuda_q8_activation_options.zig").require(o, world);
    try @import("cuda_q8_correlated_options.zig").require(o, world);
    try @import("cuda_q4_ffn_q8.zig").require(o, world);
    try @import("cuda_q4_recurrent.zig").require(o.settings, world);
    try @import("cuda_q4_precision_families.zig").require(o.settings, world);
    try @import("cuda_tile_affine.zig").require(o.settings, world, o.vision != null);
    try @import("cuda_q8_mse.zig").require(o.settings, world, o.vision != null);
    try @import("cuda_admission_options.zig").require(o, world);
    try @import("cuda_nvfp4_graph.zig").require(o, world);
    try @import("cuda_decode_select.zig").require(o.settings);
    try @import("cuda_ffn_warm.zig").require(o, world);
    if (o.settings.exl3_fused_decode and ((world != 2 and world != 4) or o.settings.verify_graphs))
        return error.Exl3DecodeScope;
    if (o.settings.prewarm_sequence_slots and (!o.settings.retain_sequence_slots or (o.settings.pages and !o.settings.kv8) or
        o.vision != null or o.settings.kv_host != 0 or o.settings.verify_graphs)) return error.PrewarmSequenceScope;
    const profiled = try prof_mod.Prof.wanted();
    if (o.settings.ffn_graphs and ((world != 2 and world != 4) or (o.settings.pages and !o.settings.kv8) or o.settings.kv_host != 0 or
        o.vision != null or !@import("cuda_target_precision.zig").supported(o.settings) or profiled))
        return error.FfnGraphScope;
    try @import("cuda_dflash_head_mode.zig").require(o.settings, o.drafter, o.mtp, o.vision != null);
    try @import("cuda_verify_rows.zig").require(o.settings, world, o.vision != null, o.drafter, o.mtp);
    try @import("cuda_kda_options.zig").require(o.settings, world, o.vision != null, o.mtp);
    if (o.settings.retain_sequence_slots and ((o.settings.pages and !o.settings.kv8) or o.settings.verify_graphs))
        return error.RetainSequenceScope;
    // The dense owner uses ordinary attention for page tables.
    if (o.settings.dense_segments and ((world != 2 and world != 4) or (o.settings.pages and !o.settings.kv8) or
        o.settings.kv_host != 0 or o.settings.verify_graphs or o.vision != null)) return error.DenseScope;
    if (o.settings.verify_graphs and (o.settings.pages or o.settings.kv_host != 0 or
        o.settings.l2pf.on or o.vision != null or profiled)) return error.VerifyGraphScope; // before anything is allocated
    if (o.settings.dense_rows16 and (!o.settings.dense_segments or o.settings.verify_rows < 16))
        return error.DenseRowsScope;
    return profiled;
}
