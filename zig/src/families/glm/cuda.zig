//! GLM-5.3-Flash on CUDA: the family's parts, ported from the Python family's CUDA path.

pub const experts = @import("cuda_experts.zig");
pub const exl3 = @import("cuda_exl3.zig");
pub const vision = @import("cuda_vision.zig");
pub const mm = @import("cuda_mm.zig");
pub const q8_activation = @import("cuda_q8_activation.zig");
pub const q8_mse = @import("cuda_q8_mse.zig");
pub const q8_moments = @import("cuda_q8_moments.zig");
pub const q8_activation_options = @import("cuda_q8_activation_options.zig");
pub const vision_config = @import("vision_config.zig");
pub const image_core = @import("image_core.zig");
/// An image's bytes to the tower's patches, as the Metal family prepares them (core's decoder).
pub const image = image_core.For(@import("core").picture);
pub const glue = @import("cuda_glue.zig");
pub const kda = @import("cuda_kda.zig");
pub const kda_segments = @import("cuda_kda_segments.zig");
pub const kda_options = @import("cuda_kda_options.zig");
pub const b16 = @import("cuda_b16.zig");
pub const latent = @import("cuda_latent.zig");
pub const latent_mma = @import("cuda_latent_mma.zig");
pub const dense_segments = @import("cuda_dense_segments.zig");
pub const sparse = @import("cuda_sparse.zig");
pub const index = @import("cuda_index.zig");
pub const decode_select = @import("cuda_decode_select.zig");
pub const moe = @import("cuda_moe.zig");
pub const kda_block = @import("cuda_kda_block.zig");
pub const dsa_block = @import("cuda_dsa_block.zig");
pub const mlp = @import("cuda_mlp.zig");
pub const shard = @import("cuda_shard.zig");
pub const weights = @import("cuda_weights.zig");
pub const tile = @import("cuda_tile.zig");
pub const forward = @import("cuda_forward.zig");
pub const shared = @import("cuda_shared.zig");
pub const verify_rows = @import("cuda_verify_rows.zig");
pub const relay = @import("cuda_relay.zig");
pub const config = @import("config.zig");
pub const quant = @import("cuda_quant.zig");
pub const q4 = @import("cuda_q4.zig");
pub const ranks = @import("cuda_ranks.zig");
pub const sampler = @import("cuda_sampler.zig");
pub const prompt_sink = @import("cuda_prompt_sink.zig");
pub const prompt_solo = @import("cuda_prompt_solo.zig");
pub const prompt_logprobs = @import("cuda_prompt_logprobs.zig");
pub const engine = @import("cuda_engine.zig");
pub const engine_prof = @import("cuda_engine_prof.zig");
pub const prompt_pair = @import("cuda_prompt_pair.zig");
pub const pages = @import("cuda_pages.zig");
pub const share = @import("cuda_share.zig");
pub const open_owner = @import("cuda_open_owner.zig");
pub const images = @import("cuda_images.zig");
pub const follow = @import("cuda_follow.zig");
pub const lanes = @import("cuda_lanes.zig");
pub const native = @import("cuda_native.zig");
pub const comm1107 = @import("cuda_comm1107.zig");
pub const copy_75c = @import("cuda_copy_75c.zig");
pub const snapshots = @import("cuda_snapshots.zig");
pub const dflash_head_mode = @import("cuda_dflash_head_mode.zig");
pub const mtp_load = @import("cuda_mtp_load.zig");
pub const kv_host = @import("cuda_kv_host.zig");
pub const disk_image = @import("cuda_disk_image.zig");
pub const disk_ranks = @import("cuda_disk_ranks.zig");
pub const nvme_image = @import("cuda_nvme_image.zig");
pub const nvme_state = @import("cuda_nvme_state.zig");
pub const nvme_options = @import("cuda_nvme_options.zig");
pub const nvme_progress = @import("cuda_nvme_progress.zig");
pub const nvme_wire = @import("cuda_nvme_wire.zig");

test {
    _ = @import("cuda_target_precision_test.zig");
    _ = @import("cuda_formats_csf0_test.zig");
    _ = @import("cuda_tile.zig");
    _ = @import("cuda_kv8_test.zig");
    _ = kv8_ref;
    _ = @import("cuda_prof.zig");
    _ = @import("cuda_load_owner.zig");
    _ = @import("cuda_images_interleave_test.zig");
    _ = @import("cuda_pages_test.zig");
    _ = @import("cuda_ranks_test.zig");
    _ = @import("cuda_rank_policy_test.zig");
    _ = @import("cuda_admission_test.zig");
    _ = @import("cuda_dflash_policy_test.zig");
    _ = @import("cuda_copy_75c_test.zig");
    _ = @import("dflash_select_test.zig");
    _ = @import("cuda_taps_tp_test.zig");
    _ = @import("cuda_roce_multi.zig");
    _ = @import("cuda_lanes_test.zig");
    _ = @import("dflash_load.zig");
    _ = prompt_solo;
    _ = @import("cuda_l2pf.zig");
    _ = prompt_pair;
    _ = @import("cuda_disk_image.zig");
    _ = experts;
    _ = exl3;
    _ = vision;
    _ = @import("vision_names.zig");
    _ = mm;
    _ = image_core;
    _ = glue;
    _ = kda;
    _ = b16;
    _ = latent;
    _ = sparse;
    _ = index;
    _ = moe;
    _ = kda_block;
    _ = dsa_block;
    _ = mlp;
    _ = shard;
    _ = weights;
    _ = forward;
    _ = relay;
    _ = quant;
    _ = q4;
    _ = ranks;
    _ = sampler;
    _ = prompt_logprobs;
    _ = engine;
    _ = follow;
    _ = lanes;
    _ = @import("cuda_zero_budget_test.zig");
    _ = native;
    _ = @import("cuda_qpack.zig");
    _ = share;
    _ = @import("cuda_mtp_chance.zig");
    _ = @import("cuda_engine_pick.zig");
    _ = @import("cuda_greedy_batch_test.zig");
    _ = @import("cuda_dense_segments.zig");
    _ = @import("cuda_dense_owner_test.zig");
    _ = @import("cuda_dense_h32_test.zig");
    _ = @import("cuda_dense16_test.zig");
    _ = @import("cuda_ffn_graph_test.zig");
    _ = @import("cuda_first_copy_test.zig");
    _ = @import("cuda_shared_pick.zig");
    _ = @import("cuda_sample_batch_test.zig");
    _ = @import("dflash_graph_test.zig");
    _ = @import("dflash_native_graph_test.zig");
    _ = @import("cuda_tp2_graph_settings_test.zig");
    _ = @import("cuda_shared_prof_test.zig");
    _ = @import("dflash_ingest_graph_test.zig");
    _ = @import("dflash_split_graph_test.zig");
    _ = @import("dflash_attention.zig");
    _ = @import("dflash_reference.zig");
    _ = @import("cuda_native_wire_layout.zig");
    _ = @import("cuda_verify_graph_test.zig");
    _ = @import("cuda_tile.zig");
    _ = @import("heat"); // upstream's heat bands, run where the GLM ranks use them
}

pub const dflash = @import("dflash.zig");
pub const dflash_candidates = @import("dflash_candidates.zig");
pub const dflash_load_owner = @import("dflash_load_owner.zig");
pub const dflash_load = @import("dflash_load.zig");
pub const taps = @import("cuda_taps.zig");

pub const prompts = @import("cuda_prompts.zig");
pub const prompt_hold = @import("cuda_prompt_hold.zig");
pub const prompt_wire = @import("cuda_prompt_wire.zig");
pub const structured = @import("cuda_structured.zig");

pub const kv8_ref = @import("cuda_kv8_ref.zig");

pub const dflash_attention = @import("dflash_attention.zig");

test {
    _ = @import("cuda_kda_segments.zig");
    _ = @import("cuda_kda_owner_test.zig");
    _ = @import("cuda_kda_h32_test.zig");
    _ = @import("cuda_kda_h32_launch_test.zig");
    _ = @import("cuda_kda_options.zig");
}

test {
    _ = @import("cuda_affine_target_test.zig");
}

pub const affine_target = @import("cuda_affine_target.zig");
pub const affine_quant = @import("cuda_affine_quant.zig");
pub const affine_reference = @import("cuda_affine_reference.zig");
test {
    _ = affine_reference;
}

pub const head_fp8 = @import("cuda_head_fp8.zig");
pub const projection = @import("cuda_proj.zig");
pub const head_fp8_reference = @import("cuda_head_fp8_reference.zig");
test {
    _ = @import("cuda_head_fp8_test.zig");
    _ = head_fp8_reference;
}

test {
    _ = @import("cuda_decode_select.zig");
    _ = latent_mma;
    _ = @import("cuda_nvfp4_mode_test.zig");
    _ = @import("cuda_warm_turns_test.zig");
}
