//! The SM121 TP2 and TP4 prompt kernels. The built-in decode modules remain the default for every decode row.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const prompt_image = @import("cuda_prompt_image.zig");
const shard = @import("cuda_shard.zig");
const ranks = @import("cuda_ranks.zig");

pub const Images = struct {
    glue: cuda.Module,
    sparse: ?cuda.Module = null,
    prep: cuda.Module,
    step: cuda.Module,
    projection: ?cuda.Module = null,

    pub fn deinit(images: *Images) void {
        if (images.projection) |*m| m.unload();
        images.step.unload();
        images.prep.unload();
        if (images.sparse) |*m| m.unload();
        images.glue.unload();
    }
};

pub fn install(g: *glue.Kernels, k: *cuda.kda.Kernels, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, share: shard.Share, settings: ranks.Settings) !Images {
    if (!try prompt_image.optimized(ctx, share) or settings.pages or !settings.kv8)
        return error.PrefillOptimizedScope;
    var filters = try load(a, io, d, "TF_GLM_PREFILL_GLUE_IMAGE", "0671c791834fef7b9efcaf0cf2cb7fca1912790c26b6a2e78f029c8e54b026c5");
    errdefer filters.unload();
    var prep = try load(a, io, d, "TF_GLM_KDA_PREP_LANE4_IMAGE", "86a8fca9e8802a0fdd51aebe4b87f167938f27de1179240df72adca1e76cc1eb");
    errdefer prep.unload();
    var step = try load(a, io, d, "TF_GLM_KDA_ASYNC_STEP_IMAGE", "1598b2092588300fe6c0ace6234b4b36c2f3f23d77e6d8345b15ab64c4b9362a");
    errdefer step.unload();
    var projection: ?cuda.Module = null;
    errdefer if (projection) |*m| m.unload();
    if (!settings.absorb_mma) projection = try load(a, io, d, "TF_GLM_LATENT_ROW_STAGE_IMAGE", "6be1a1a57695feebb78c1fa91cf3361be24d408445e3237e59a621449a1f22c7");
    var wide_sparse: ?cuda.Module = null;
    errdefer if (wide_sparse) |*m| m.unload();
    if (share.world == 2)
        wide_sparse = try load(a, io, d, "TF_GLM_PREFILL_H32_IMAGE", "d8a23632ad033d1d4a0be4fbe5536f1d484cbfa3d7cb215c8716cd15103e52d8");
    const hc = try filters.function("tf_glm_hc_post_own");
    const sparse = if (wide_sparse) |m| try m.function("tf_glm_sparse_prompt_fp8_warp_key_prefetch_lines_h32") else try filters.function("tf_glm_sparse_prompt_fp8_warp_key_prefetch_lines");
    try sparse.allowDynamicShared(glue.compact_sparse_smem);
    const prep_kernel = try prep.function("tf_glm_kda_prep_lane4");
    const step_kernel = try step.function("tf_glm_kda_step_async_w4");
    if (projection) |m| g.latent_prompt = .{
        .absorb = try m.function("tf_glm_absorb_q_prompt_row_stage_4"),
        .expand = try m.function("tf_glm_expand_v_prompt_row_stage_1"),
    };
    g.hc_post_own = hc;
    g.sparse_prompt_fp8 = .{ .function = sparse, .heads = if (wide_sparse != null) 32 else 16 };
    k.prompt_prep = prep_kernel;
    k.prompt_step = step_kernel;
    std.log.info("GLM_PREFILL_OPT rank={d} sparse=16 kda=lane4+async own_partials=true row_stage={any}", .{ share.rank, projection != null });
    return .{ .glue = filters, .sparse = wide_sparse, .prep = prep, .step = step, .projection = projection };
}

fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, comptime name: [:0]const u8, sha: []const u8) !cuda.Module {
    const path = @import("sys").environment.Process.get(.{}, name) orelse return error.MissingPrefillImage;
    return prompt_image.load(a, io, d, path, sha);
}
