//! A TP rank's settings fingerprint, all-gathered before weights load: a mismatch stops every rank.
const std = @import("std");
const policy = @import("cuda_dflash_policy.zig");
const width = @import("cuda_verify_rows.zig");
const dense = @import("cuda_dense_segments.zig");
const cuda = @import("cuda");
const heat = @import("heat");
const l2pf = @import("cuda_l2pf.zig");
const wireOpens = @import("cuda_rank_options.zig").wireOpens;
pub const Settings = struct {
    absorb_mma: bool = false,
    kv8: bool = false, // --kv-cache fp8, supplied by the native options rather than an environment override
    pages: bool = false, // TF_KV_PAGES as `open` resolves it (with the world and --kv-cache): every sequence on pages
    prompt_fixed_k: bool = false, // TF_GLM_PROMPT_K: one (default) or shape (experimental fixed K slices)
    kda_segments: bool = false,
    kda_min_streams: u32 = 8,
    tiled: bool = true, // TF_GLM_MM: 1 (the bf16 linears on glm_mm, tiled at load) or 0
    split: bool = true, // TF_GLM_PREFILL_REDUCE: split (each rank glues its own rows), ring or gather
    ring: bool = false, // TF_GLM_PREFILL_REDUCE=ring: split, its exchanges passed along the ring (cuda_relay)
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE: 1 (decode windows on KDA's three-kernel chain) or 0
    kda_pad: bool = false, // TF_GLM_KDA_PAD: 1 (the fused KDA input padded to eight columns at load, for glm_mm) or 0
    pieces: u32 = 3, // TF_GLM_OVERLAP_PIECES: a prompt chunk's exchanges in this many row pieces (1: whole); unset, 8 under solo on the wire.
    solo: bool = false, // TF_GLM_CHUNK_SOLO: 4096 (prompt buffers for two chunks a pass, while no stream decodes) or 2048
    pair: bool = true, // TF_GLM_PROMPT_PAIR: 1 (while no stream decodes, a prompt step takes two chunks, each one's exchanges under the other's sites) or 0
    exl3_prompt_whole: bool = false,
    kv_live_half: bool = false,
    dflash_only_head: bool = false,
    nvfp4_fact: bool = false,
    dsel: bool = false, // TF_GLM_DSEL: upstream multi-CTA decode selection, agreed before startup
    index_split: u32 = 2048, // TF_GLM_INDEX_SPLIT: from this position on, rank 0 splits a prompt chunk's indexer rows (0: never)
    heat: ?heat.Bands = null, // TF_HEAT_HIGH and TF_HEAT_LOW (upstream's names): a prompt chunk waits on every rank's hottest zone
    layers: u32 = 0, // TF_GLM_LAYERS: the first this many layers only, a thin model for one-GPU checks (0: all)
    q8_mse: bool = false,
    q4_affine_mse: bool = false,
    q4_head_fp8: bool = false,
    q4_python_prompt: bool = false,
    q4_python_splits: bool = false,
    dense_bits: u8 = 16, // TF_GLM_DENSE_BITS: the dense linears packed at load to 4, 6 or 8 bits (16: bf16)
    head_bits: u8 = 16, // TF_GLM_HEAD_BITS: the head, likewise
    roce_multi: bool = false, // TF_GLM_ROCE_MULTI: 1 (three ranks or more: small all-gathers on fabric/tp's multirank RoCE beside NCCL) or 0
    roce_wide: bool = false, // TF_ROCE=1: the PR2205 fast wire at three ranks or more
    prewarm_sequence_slots: bool = false,
    retain_sequence_slots: bool = false, // TF_GLM_RETAIN_SEQUENCE_SLOTS=1: keep nonpaged slots1..7 on release
    warm_turns: bool = false, // TF_GLM_WARM_TURNS: background next-turn prefill when the prompt cache is enabled
    verify_rows: u8 = 8, // TF_GLM_VERIFY_ROWS: active8(default)/16/32 over physical32 storage
    copy_policy_75c: bool = false, // TF_GLM_COPY_POLICY_75C: request-local match8/reply16/latest-full proposals
    first_copy_tokens: u8 = 3, // TF_GLM_FIRST_COPY_TOKENS: 3/7,15 at width>=16,31 at width32
    sample_batch: bool = false, // TF_GLM_SAMPLE_BATCH=1: ordinary sampled rows use one padded byte gather
    greedy_shared: bool = false, // TF_GLM_GREEDY_SHARED=1: one decision gather for every greedy shared part
    exl3_fused_decode: bool = false,
    admission_us: u32 = 0, // TF_GLM_ADMISSION_US: idle whole prompts collect for 0 or 4000 microseconds
    ffn_warm: bool = false, // TF_GLM_FFN_WARM: capture target keys before readiness, never execute them
    ffn_graphs: bool = false, // TF_GLM_FFN_GRAPHS: compute-only target feed-forward graphs
    nvfp4_ffn_graphs: bool = false, // TF_GLM_NVFP4_FFN_GRAPHS: explicit NVFP4 capture scope
    dense_rows16: bool = false, // allow sixteen-row parts in the same dense table kernels
    dense_segments: bool = false, // TF_GLM_DENSE_LATENT_SEGMENTS=1: dense shared MLA launches across sequences
    greedy_batch: bool = false, // TF_GLM_GREEDY_BATCH=1: ordinary verify rows share one top1-record gather/readback
    verify_graphs: bool = false, // TF_GLM_VERIFY_GRAPHS=1: bounded raw-NCCL verify graphs, without side prefetch
    dflash_ingest_graphs: bool = false, // TF_GLM_DFLASH_INGEST_GRAPHS=1: pure ingest compute capture
    dflash_native_graph: bool = false, // whole-block graph on the borrowed native wire
    dflash_graphs: bool = false, // TF_GLM_DFLASH_GRAPHS=1: compute-only graphs for unsplit DFlash blocks
    dflash_confidence_only: bool = false, // TF_GLM_DFLASH_CONFIDENCE_ONLY strict startup opt-in
    dflash_confidence_floor: f64 = 0.7, // frozen TF_GLM_DRAFT_CONFIDENCE only under the new policy
    dflash_reference: bool = false, // TF_GLM_DFLASH_REFERENCE: pinned historical drafter arithmetic
    dflash_edge_06: bool = false, // TF_GLM_DFLASH_EDGE_06=1: selector edge coefficient0.6 (default1.0)
    dflash_attn_tp4: bool = false, // TF_GLM_DFLASH_ATTN_TP4=1: partition draft-block heads, full context rings
    dflash_mlp_wide: bool = false, // TF_GLM_DFLASH_MLP_TP4=1: a separate producer/arithmetic experiment
    kv_host: u8 = 0, // TF_GLM_KV_HOST=1: KV pages in host memory (1); with TF_GLM_KV_HOST_SHARED=1, one copy a host (2)
    kv_host_gib: f64 = 0, // TF_GLM_KV_HOST_GIB: the host pages' GiB (0: one full window)
    l2pf: l2pf.Options = .{}, // TF_GLM_L2PF and its knobs: a decode window's next weights read into L2 beside it
    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Settings {
        var s: Settings = .{};
        s.q8_mse = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q8_MSE"), problem, "TF_GLM_Q8_MSE: 0 or 1");
        s.absorb_mma = try @import("cuda_dense16.zig").flag(get("TF_GLM_ABSORB_MMA"), problem, "TF_GLM_ABSORB_MMA: 0 or 1");
        s.warm_turns = try @import("cuda_dense16.zig").flag(get("TF_GLM_WARM_TURNS"), problem, "TF_GLM_WARM_TURNS: 0 or 1");
        s.kv_live_half = try @import("cuda_dense16.zig").flag(get("TF_GLM_KV_LIVE_HALF"), problem, "TF_GLM_KV_LIVE_HALF: 0 or 1");
        s.dflash_only_head = try @import("cuda_dense16.zig").flag(get("TF_GLM_DFLASH_ONLY_HEAD"), problem, "TF_GLM_DFLASH_ONLY_HEAD: 0 or 1");
        s.nvfp4_fact = try @import("cuda_dense16.zig").flag(get("TF_GLM_NVFP4_FACT"), problem, "TF_GLM_NVFP4_FACT: 0 or 1");
        s.exl3_prompt_whole = try @import("cuda_dense16.zig").flag(get("TF_GLM_EXL3_PROMPT_WHOLE"), problem, "TF_GLM_EXL3_PROMPT_WHOLE: 0 or 1");
        s.dsel = try @import("cuda_dense16.zig").flag(get("TF_GLM_DSEL"), problem, "TF_GLM_DSEL: 0 or 1");
        s.q4_head_fp8 = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_HEAD_FP8"), problem, "TF_GLM_Q4_HEAD_FP8: 0 or 1");
        s.q4_affine_mse = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_AFFINE_MSE"), problem, "TF_GLM_Q4_AFFINE_MSE: 0 or 1");
        s.q4_python_prompt = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_PYTHON_PROMPT"), problem, "TF_GLM_Q4_PYTHON_PROMPT: 0 or 1");
        s.q4_python_splits = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_PYTHON_SPLITS"), problem, "TF_GLM_Q4_PYTHON_SPLITS: 0 or 1");
        const kda = try @import("cuda_kda_options.zig").read(get, problem);
        s.kda_segments = kda.on;
        s.kda_min_streams = kda.min_streams;
        s.dflash_reference = try @import("cuda_dense16.zig").flag(get("TF_GLM_DFLASH_REFERENCE"), problem, "TF_GLM_DFLASH_REFERENCE: 0 or 1");
        const p = try policy.read(get, problem);
        s.dflash_confidence_only = p.on;
        s.dflash_confidence_floor = p.floor;
        s.prewarm_sequence_slots = try @import("cuda_dense16.zig").flag(get("TF_GLM_PREWARM_SEQUENCE_SLOTS"), problem, "TF_GLM_PREWARM_SEQUENCE_SLOTS: 0 or 1");
        s.retain_sequence_slots = dense.enabled(get("TF_GLM_RETAIN_SEQUENCE_SLOTS")) catch {
            problem.* = "TF_GLM_RETAIN_SEQUENCE_SLOTS: 0 or 1";
            return error.BadSetting;
        };
        s.copy_policy_75c = dense.enabled(get("TF_GLM_COPY_POLICY_75C")) catch {
            problem.* = "TF_GLM_COPY_POLICY_75C: 0 or 1";
            return error.BadSetting;
        };
        s.dflash_native_graph = try @import("cuda_dense16.zig").flag(get("TF_GLM_DFLASH_NATIVE_GRAPH"), problem, "TF_GLM_DFLASH_NATIVE_GRAPH: 0 or 1");
        s.dense_rows16 = try @import("cuda_dense16.zig").flag(get("TF_GLM_DENSE_ROWS16"), problem, "TF_GLM_DENSE_ROWS16: 0 or 1");
        s.exl3_fused_decode = try @import("cuda_dense16.zig").flag(get("TF_GLM_EXL3_FUSED_DECODE"), problem, "TF_GLM_EXL3_FUSED_DECODE: 0 or 1");
        s.ffn_graphs = try @import("cuda_rank_options.zig").ffnGraphs(get, problem);
        s.nvfp4_ffn_graphs = try @import("cuda_dense16.zig").flag(get("TF_GLM_NVFP4_FFN_GRAPHS"), problem, "TF_GLM_NVFP4_FFN_GRAPHS: 0 or 1");
        s.admission_us = try @import("cuda_admission_options.zig").read(get("TF_GLM_ADMISSION_US"), problem);
        s.ffn_warm = try @import("cuda_dense16.zig").flag(get("TF_GLM_FFN_WARM"), problem, "TF_GLM_FFN_WARM: 0 or 1");
        const w = try width.read(get, problem);
        s.verify_rows = w.rows;
        s.first_copy_tokens = w.copy;
        s.dense_segments = dense.enabled(get("TF_GLM_DENSE_LATENT_SEGMENTS")) catch {
            problem.* = "TF_GLM_DENSE_LATENT_SEGMENTS: 0 or 1";
            return error.BadSetting;
        };
        s.prompt_fixed_k = try @import("cuda_rank_options.zig").promptK(get, problem);
        try @import("cuda_rank_options.zig").promptTail(get, problem);
        if (get("TF_GLM_MM")) |v| {
            if (std.mem.eql(u8, v, "0")) s.tiled = false else if (!std.mem.eql(u8, v, "1")) {
                problem.* = "TF_GLM_MM: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KDA_PAD")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kda_pad = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KDA_PAD: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_PREFILL_REDUCE")) |v| {
            if (std.mem.eql(u8, v, "split")) s.split = true else if (std.mem.eql(u8, v, "ring")) {
                s.split = true;
                s.ring = true;
            } else if (std.mem.eql(u8, v, "gather")) s.split = false else {
                problem.* = "TF_GLM_PREFILL_REDUCE: split, ring or gather";
                return error.BadSetting;
            }
        }
        s.kda_wide = try @import("cuda_rank_options.zig").kdaWide(get, problem);
        s.roce_multi = try @import("cuda_dense16.zig").flag(get("TF_GLM_ROCE_MULTI"), problem, "TF_GLM_ROCE_MULTI: 0 or 1");
        if (get("TF_ROCE")) |v| s.roce_wide = std.mem.eql(u8, v, "1");
        if (get("TF_GLM_VERIFY_GRAPHS")) |v| {
            if (std.mem.eql(u8, v, "1")) s.verify_graphs = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_VERIFY_GRAPHS: 0 or 1";
                return error.BadSetting;
            }
        }
        s.verify_graphs = (cuda.graph_cache.Settings{ .on = s.verify_graphs }).operator(get("TF_GRAPH_CACHE")).on;
        const flag = @import("cuda_dense16.zig").flag;
        s.dflash_ingest_graphs = try flag(get("TF_GLM_DFLASH_INGEST_GRAPHS"), problem, "TF_GLM_DFLASH_INGEST_GRAPHS: 0 or 1");
        s.dflash_graphs = try flag(get("TF_GLM_DFLASH_GRAPHS"), problem, "TF_GLM_DFLASH_GRAPHS: 0 or 1");
        if (get("TF_GLM_SAMPLE_BATCH")) |v| {
            if (std.mem.eql(u8, v, "1")) s.sample_batch = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_SAMPLE_BATCH: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_GREEDY_BATCH")) |v| {
            if (std.mem.eql(u8, v, "1")) s.greedy_batch = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_GREEDY_BATCH: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_GREEDY_SHARED")) |v| {
            if (std.mem.eql(u8, v, "1")) s.greedy_shared = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_GREEDY_SHARED: 0 or 1";
                return error.BadSetting;
            }
        }
        if (s.greedy_shared and !s.greedy_batch) {
            problem.* = "TF_GLM_GREEDY_SHARED requires TF_GLM_GREEDY_BATCH=1";
            return error.BadSetting;
        }
        if (get("TF_GLM_DFLASH_EDGE_06")) |v| {
            if (std.mem.eql(u8, v, "1")) s.dflash_edge_06 = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_DFLASH_EDGE_06: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_DFLASH_ATTN_TP4")) |v| {
            if (std.mem.eql(u8, v, "1")) s.dflash_attn_tp4 = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_DFLASH_ATTN_TP4: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_DFLASH_MLP_TP4")) |v| {
            if (std.mem.eql(u8, v, "1")) s.dflash_mlp_wide = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_DFLASH_MLP_TP4: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_CHUNK_SOLO")) |v| {
            if (std.mem.eql(u8, v, "4096")) s.solo = true else if (std.mem.eql(u8, v, "2048")) s.solo = false else {
                problem.* = "TF_GLM_CHUNK_SOLO: 2048 or 4096";
                return error.BadSetting;
            }
        }
        s.l2pf = try l2pf.Options.read(get, problem);
        if (get("TF_GLM_PROMPT_PAIR")) |v| {
            if (std.mem.eql(u8, v, "1")) s.pair = true else if (std.mem.eql(u8, v, "0")) s.pair = false else {
                problem.* = "TF_GLM_PROMPT_PAIR: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_OVERLAP_PIECES")) |v| {
            s.pieces = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.pieces < 1 or s.pieces > 8) {
                problem.* = "TF_GLM_OVERLAP_PIECES: 1 to 8";
                return error.BadSetting;
            }
        } else if (s.solo and s.roce_multi and wireOpens(get)) {
            s.pieces = 8; // TF_GLM_CHUNK_SOLO=4096 with TF_GLM_ROCE_MULTI=1: solo's doubled passes on the wire
        }
        if (get("TF_GLM_INDEX_SPLIT")) |v| {
            s.index_split = std.fmt.parseInt(u32, v, 10) catch {
                problem.* = "TF_GLM_INDEX_SPLIT: the position a prompt chunk's indexer rows split from, or 0 for never";
                return error.BadSetting;
            };
        }
        if (get("TF_GLM_LAYERS")) |v| {
            s.layers = std.fmt.parseInt(u32, v, 10) catch 0;
            if (s.layers == 0) {
                problem.* = "TF_GLM_LAYERS: the first how many layers, 1 or more";
                return error.BadSetting;
            }
        }
        inline for (.{ .{ "TF_GLM_DENSE_BITS", "dense_bits" }, .{ "TF_GLM_HEAD_BITS", "head_bits" } }) |e| if (get(e[0])) |v| {
            const b = std.fmt.parseInt(u8, v, 10) catch {
                problem.* = e[0] ++ ": 4, 6, 8 or 16";
                return error.BadSetting;
            };
            @field(s, e[1]) = switch (b) {
                4, 6, 8, 16 => b,
                0 => 16, // the old family's spelling of bf16
                else => {
                    problem.* = e[0] ++ ": 4, 6, 8 or 16";
                    return error.BadSetting;
                },
            };
        };
        s.heat = heat.bandsFrom(get("TF_HEAT_HIGH"), get("TF_HEAT_LOW")) catch {
            problem.* = "TF_HEAT_HIGH and TF_HEAT_LOW: both or neither, in degrees, with low at or under high";
            return error.BadSetting;
        };
        if (s.prompt_fixed_k and !s.tiled) {
            problem.* = "TF_GLM_PROMPT_K=shape requires TF_GLM_MM=1";
            return error.BadSetting;
        }
        if (get("TF_GLM_KV_HOST")) |v| {
            if (std.mem.eql(u8, v, "1")) s.kv_host = 1 else if (!std.mem.eql(u8, v, "0")) {
                problem.* = "TF_GLM_KV_HOST: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KV_HOST_SHARED")) |v| {
            if (std.mem.eql(u8, v, "1") and s.kv_host == 1) s.kv_host = 2 else if (!std.mem.eql(u8, v, "0")) {
                problem.* = if (std.mem.eql(u8, v, "1")) "TF_GLM_KV_HOST_SHARED=1 requires TF_GLM_KV_HOST=1" else "TF_GLM_KV_HOST_SHARED: 0 or 1";
                return error.BadSetting;
            }
        }
        if (get("TF_GLM_KV_HOST_GIB")) |v| {
            s.kv_host_gib = std.fmt.parseFloat(f64, v) catch 0;
            if (!(s.kv_host_gib > 0) or s.kv_host == 0) {
                problem.* = "TF_GLM_KV_HOST_GIB: the host pages' GiB, more than 0, with TF_GLM_KV_HOST=1";
                return error.BadSetting;
            }
        }
        // refused here, before the weights load: the wire's bootstrap meets at rank 0, its kernel comes from the image
        if (s.roce_multi and (get("TF_TP_MASTER") == null or get("TF_TP_KERNEL_IMAGE") == null)) {
            problem.* = "TF_GLM_ROCE_MULTI=1 requires TF_TP_MASTER and TF_TP_KERNEL_IMAGE";
            return error.BadSetting;
        }
        if (s.kda_pad and !s.tiled) {
            problem.* = "TF_GLM_KDA_PAD=1 requires TF_GLM_MM=1 (glm_mm is what reads the padded rows)";
            return error.BadSetting;
        }
        if (s.dflash_attn_tp4 and !s.dflash_mlp_wide) {
            problem.* = "TF_GLM_DFLASH_ATTN_TP4=1 requires TF_GLM_DFLASH_MLP_TP4=1";
            return error.BadSetting;
        }
        try @import("cuda_dense16.zig").graph(s, get("TF_GRAPH_CACHE"), problem);
        try @import("cuda_dense16.zig").check(s, problem);
        return s;
    }
};
/// The most ranks a world holds: rank 0's greedy pick gathers every rank's top-1 into one host array of this size.
pub const max_world = 16;
/// A setting by name (null when unset): a registered switch through the profile registry, the rest as before.
pub fn environment(name: [:0]const u8) ?[]const u8 { return @import("profile").lookup(name); }
/// Whether the group brings up its own RoCE path: not where TF_GLM_ROCE_MULTI=1 lends fabric/tp's wire (3+ ranks).
pub fn ownRoce(s: Settings, world: usize) bool { return !(s.roce_multi and world >= 3); }
pub const Fingerprint = extern struct {
    layout: u32 = 69,
    world: u32,
    rank: u32,
    kv8: u32 = 0,
    pages: u32 = 0, // TF_KV_PAGES: the engine's own sequence is made on pages at init, so every rank pages alike
    prompt_fixed_k: u32,
    kda_segments: u32 = 0,
    kda_min_streams: u32 = 8,
    tiled: u32,
    split: u32, // TF_GLM_PREFILL_REDUCE: 0 gather, 1 split, 2 ring
    kda_wide: u32,
    kda_pad: u32 = 0, // TF_GLM_KDA_PAD: every rank's KDA input rows carry one width
    pieces: u32,
    pair: u32, // TF_GLM_PROMPT_PAIR: rank 0's paired steps need the second chunk's buffers on every rank
    layers: u32, // TF_GLM_LAYERS (0: all)
    mtp: u32, // 1: the MTP head loads and runs (drafts on); 2: and DFlash2 (--drafter), whose rounds gather
    vision: u32 = 0, // --vision's image tokens (0: off): rank 0 sends image rows every rank must hold
    video_tokens: u32 = 0,
    max_images: u32 = 0,
    max_videos: u32 = 0,
    max_frames: u32 = 0,
    roce_multi: u32 = 0, // TF_GLM_ROCE_MULTI: every rank opens the borrowed wire or none does (a lone rank would wait on it)
    roce_wide: u32 = 0, // TF_ROCE=1: the PR2205 fast wire
    prewarm_sequence_slots: u32 = 0,
    retain_sequence_slots: u32 = 0,
    warm_turns: u32 = 0,
    verify_rows: u32 = 8,
    copy_policy_75c: u32 = 0,
    first_copy_tokens: u32 = 3,
    sample_batch: u32 = 0,
    greedy_shared: u32 = 0,
    exl3_fused_decode: u32 = 0,
    admission_us: u32 = 0,
    ffn_warm: u32 = 0,
    ffn_graphs: u32 = 0,
    nvfp4_ffn_graphs: u32 = 0,
    dense_rows16: u32 = 0,
    dense_segments: u32 = 0,
    exl3_prompt_whole: u32 = 0,
    kv_live_half: u32 = 0,
    dflash_only_head: u32 = 0,
    nvfp4_fact: u32 = 0,
    dsel: u32 = 0,
    greedy_batch: u32 = 0, // every rank batches the same verify draws or none does
    verify_graphs: u32 = 0,
    dflash_ingest_graphs: u32 = 0,
    dflash_native_graph: u32 = 0,
    dflash_graphs: u32 = 0,
    dflash_confidence_only: u32 = 0,
    dflash_confidence_floor: f64 = 0,
    dflash_reference: u32 = 0,
    dflash_edge_06: u32 = 0,
    dflash_attn_tp4: u32 = 0,
    dflash_mlp_wide: u32 = 0,
    kv_host: u32 = 0, // TF_GLM_KV_HOST(_SHARED): every rank sizes the pool alike and maps the shared file, or none does
    solo: u32 = 0, // TF_GLM_CHUNK_SOLO: rank 0's solo passes need the wider prompt buffers on every rank
    l2pf: [4]u32 = @splat(0), // TF_GLM_L2PF's words: every rank prefetches alike, so none waits on a slower one's windows
    q8_mse: u32 = 0,
    q4_affine_mse: u32 = 0,
    q4_head_fp8: u32 = 0,
    q4_python_prompt: u32 = 0,
    q4_python_splits: u32 = 0,
    bits: u32 = 16 << 8 | 16, // TF_GLM_DENSE_BITS << 8 | TF_GLM_HEAD_BITS: every rank's matmuls read the same codes
    capacity: u64, // tokens a sequence holds
    heat_high: f64, // the heat bands in degrees (0 and 0: off): every rank must wait alike
    heat_low: f64,
    kv_host_gib: f64 = 0, // TF_GLM_KV_HOST_GIB: each rank's row marks span the host pages rank 0 counts
    config: [32]u8, // sha256 of config.json
    program: [32]u8, // sha256 of this executable
    checkpoint: [64]u8, // hex digest of the checkpoint's metadata and every shard byte, through the digest cache
    absorb_mma: u32 = 0,
    pub const Field = enum {
        layout,
        rank,
        world,
        kv8,
        pages,
        prompt_fixed_k,
        kda_segments,
        kda_min_streams,
        tiled,
        split,
        kda_wide,
        kda_pad,
        pieces,
        pair,
        layers,
        mtp,
        vision,
        video_tokens,
        max_images,
        max_videos,
        max_frames,
        roce_multi,
        roce_wide,
        prewarm_sequence_slots,
        retain_sequence_slots,
        warm_turns,
        verify_rows,
        copy_policy_75c,
        first_copy_tokens,
        sample_batch,
        greedy_shared,
        exl3_fused_decode,
        admission_us,
        ffn_warm,
        ffn_graphs,
        nvfp4_ffn_graphs,
        dense_rows16,
        dense_segments,
        exl3_prompt_whole,
        kv_live_half,
        dflash_only_head,
        nvfp4_fact,
        dsel,
        greedy_batch,
        verify_graphs,
        dflash_ingest_graphs,
        dflash_native_graph,
        dflash_graphs,
        dflash_confidence_only,
        dflash_confidence_floor,
        dflash_reference,
        dflash_edge_06,
        dflash_attn_tp4,
        dflash_mlp_wide,
        kv_host,
        solo,
        l2pf,
        q8_mse,
        q4_affine_mse,
        q4_head_fp8,
        q4_python_prompt,
        q4_python_splits,
        bits,
        capacity,
        heat_high,
        heat_low,
        kv_host_gib,
        config,
        program,
        checkpoint,
        absorb_mma,
    };
    pub fn of(
        world: usize,
        rank: usize,
        s: Settings,
        capacity: usize,
        mtp: bool,
        config: [32]u8,
        program: [32]u8,
        checkpoint_digest: [32]u8,
    ) Fingerprint {
        const f: Fingerprint = .{
            .kv8 = @intFromBool(s.kv8),
            .pages = @intFromBool(s.pages),
            .world = @intCast(world),
            .rank = @intCast(rank),
            .prompt_fixed_k = @intFromBool(s.prompt_fixed_k),
            .kda_segments = @intFromBool(s.kda_segments),
            .kda_min_streams = s.kda_min_streams,
            .tiled = @intFromBool(s.tiled),
            .split = if (!s.split) 0 else if (s.ring) 2 else 1,
            .kda_wide = @intFromBool(s.kda_wide),
            .kda_pad = @intFromBool(s.kda_pad),
            .pieces = s.pieces,
            .pair = @intFromBool(s.pair),
            .layers = s.layers,
            .mtp = @intFromBool(mtp),
            .roce_multi = @intFromBool(s.roce_multi),
            .roce_wide = @intFromBool(s.roce_wide and world >= 3), // two ranks ask by default, so a lone TF_ROCE=1 there is no difference
            .prewarm_sequence_slots = @intFromBool(s.prewarm_sequence_slots),
            .retain_sequence_slots = @intFromBool(s.retain_sequence_slots),
            .warm_turns = @intFromBool(s.warm_turns),
            .verify_rows = s.verify_rows,
            .copy_policy_75c = @intFromBool(s.copy_policy_75c),
            .first_copy_tokens = s.first_copy_tokens,
            .sample_batch = @intFromBool(s.sample_batch),
            .greedy_shared = @intFromBool(s.greedy_shared),
            .exl3_fused_decode = @intFromBool(s.exl3_fused_decode),
            .admission_us = s.admission_us,
            .ffn_warm = @intFromBool(s.ffn_warm),
            .ffn_graphs = @intFromBool(s.ffn_graphs),
            .nvfp4_ffn_graphs = @intFromBool(s.nvfp4_ffn_graphs),
            .dense_rows16 = @intFromBool(s.dense_rows16),
            .dense_segments = @intFromBool(s.dense_segments),
            .exl3_prompt_whole = @intFromBool(s.exl3_prompt_whole),
            .kv_live_half = @intFromBool(s.kv_live_half),
            .dflash_only_head = @intFromBool(s.dflash_only_head),
            .nvfp4_fact = @intFromBool(s.nvfp4_fact),
            .dsel = @intFromBool(s.dsel),
            .greedy_batch = @intFromBool(s.greedy_batch),
            .verify_graphs = @intFromBool(s.verify_graphs),
            .dflash_ingest_graphs = @intFromBool(s.dflash_ingest_graphs),
            .dflash_native_graph = @intFromBool(s.dflash_native_graph),
            .dflash_graphs = @intFromBool(s.dflash_graphs),
            .dflash_confidence_only = @intFromBool(s.dflash_confidence_only),
            .dflash_confidence_floor = if (s.dflash_confidence_only) s.dflash_confidence_floor else 0,
            .dflash_reference = @intFromBool(s.dflash_reference),
            .dflash_edge_06 = @intFromBool(s.dflash_edge_06),
            .dflash_attn_tp4 = @intFromBool(s.dflash_attn_tp4),
            .dflash_mlp_wide = @intFromBool(s.dflash_mlp_wide and world >= 3),
            .kv_host = s.kv_host,
            .kv_host_gib = s.kv_host_gib,
            .solo = @intFromBool(s.solo),
            .l2pf = s.l2pf.words(),
            .q4_head_fp8 = @intFromBool(s.q4_head_fp8),
            .absorb_mma = @intFromBool(s.absorb_mma),
            .q4_python_prompt = @intFromBool(s.q4_python_prompt),
            .q4_python_splits = @intFromBool(s.q4_python_splits),
            .q8_mse = @intFromBool(s.q8_mse),
            .q4_affine_mse = @intFromBool(s.q4_affine_mse),
            .bits = @as(u32, s.dense_bits) << 8 | s.head_bits,
            .capacity = capacity,
            .heat_high = if (s.heat) |b| b.high_c else 0,
            .heat_low = if (s.heat) |b| b.low_c else 0,
            .config = config,
            .program = program,
            .checkpoint = std.fmt.bytesToHex(checkpoint_digest, .lower),
        };
        return f;
    }
};
pub const Mismatch = struct { field: Fingerprint.Field, rank: usize };
/// The first field where rank `r`'s fingerprint differs from rank 0's (or holds the wrong rank), null when all agree.
pub fn mismatch(all: []const Fingerprint) ?Mismatch {
    for (all, 0..) |f, r| {
        if (f.layout != all[0].layout) return .{ .field = .layout, .rank = r };
        if (f.rank != r) return .{ .field = .rank, .rank = r };
        const fields = @import("cuda_rank_names.zig").compared;
        inline for (fields) |name| {
            if (!std.mem.eql(u8, std.mem.asBytes(&@field(f, name)), std.mem.asBytes(&@field(all[0], name))))
                return .{ .field = @field(Fingerprint.Field, name), .rank = r };
        }
    }
    return null;
}
pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}
pub const hashBytes = @import("cuda_rank_options.zig").hashBytes;
pub fn gather(
    comptime T: type,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    rank: usize,
    mine: T,
    out: []T,
) !void {
    const size = @sizeOf(T);
    var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
    defer dev.free();
    try dev.uploadAsync(rank * size, std.mem.asBytes(&mine), stream.handle);
    const gathered = lib.api.ncclAllGather(dev.ptr + rank * size, dev.ptr, size, .u8, comm, stream.handle);
    try lib.check(gathered, "ncclAllGather");
    try stream.synchronize();
    try dev.download(0, std.mem.sliceAsBytes(out));
}
/// All-gathers `mine` over `comm`; any disagreement is error.RankMismatch on every rank, printing what differs.
pub fn agree(
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    mine: Fingerprint,
    out: []Fingerprint,
) !void {
    try gather(Fingerprint, d, lib, comm, stream, mine.rank, mine, out);
    if (mismatch(out)) |m| {
        std.debug.print("REFUSE rank settings: rank {d}'s {s} differs from rank 0's ({s})\n", .{ m.rank, @tagName(m.field), explain(m.field) });
        return error.RankMismatch;
    }
}
/// Before loading: direct split reaches every rank, so each pair is probed; the ring only passes rows to neighbours.
pub fn probeStart(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, s: Settings, rank: usize, world: usize) !void {
    if (s.split and !s.ring) try probeSplit(d, lib, comm, stream, rank, world);
}
/// Before loading, split's peers: one value each way with each, in rank order (no pair waits on a later one).
fn probeSplit(d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, rank: usize, world: usize) !void {
    var dev = try cuda.DeviceBuffer.alloc(d, 8);
    defer dev.free();
    for (0..world) |peer| if (peer != rank) {
        try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
        try lib.check(lib.api.ncclSend(dev.ptr, 1, .f32, @intCast(peer), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(dev.ptr + 4, 1, .f32, @intCast(peer), comm, stream.handle), "ncclRecv");
        lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd") catch return unreachablePeer(rank, peer);
        stream.synchronize() catch return unreachablePeer(rank, peer);
    };
}
pub var refused_pair: ?[2]usize = null;
fn unreachablePeer(rank: usize, peer: usize) error{SplitUnreachable} {
    if (@import("builtin").is_test) {
        refused_pair = .{ rank, peer };
        return error.SplitUnreachable;
    }
    std.debug.print("REFUSE TF_GLM_PREFILL_REDUCE=split: rank {d} cannot exchange with rank {d}; split sends every rank's rows to every other, and a fabric where some ranks reach only their neighbours (a ring of four without a switch) needs TF_GLM_PREFILL_REDUCE=ring or gather\n", .{ rank, peer });
    return error.SplitUnreachable;
}
const explain = @import("cuda_rank_names.zig").explain;
pub const mediaFingerprint = @import("cuda_rank_names.zig").mediaFingerprint;
/// The checkpoint digest a rank sends: its metadata and every shard byte, read again only for a file that changed.
pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}
