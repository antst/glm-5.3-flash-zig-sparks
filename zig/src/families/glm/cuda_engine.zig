//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const heat = @import("heat");
const exl3 = @import("cuda_exl3.zig");
const images_mod = @import("cuda_images.zig");
const tile = @import("cuda_tile.zig");
const mm_mod = @import("cuda_mm.zig");
const dflash = @import("dflash.zig");
const prof_mod = @import("cuda_prof.zig");
const pages_mod = @import("cuda_pages.zig");
const l2pf_mod = @import("cuda_l2pf.zig");
const prompt_pair = @import("cuda_prompt_pair.zig");
const verify_graph = @import("cuda_verify_graph.zig");
const sequence_slots = @import("cuda_sequence_slots.zig");
pub const Options = @import("cuda_engine_types.zig").Options;
pub const max_drafts = 3;
const pair_bytes = 16;
pub const prompt_rows = fw.prefill_rows;
pub const solo_rows = fw.solo_rows;
pub const max_together = @import("cuda_state.zig").prompt_heads;
pub const Prompt = struct { slot: u32, ids: []const u32 };
pub const shared_rows = 64;
pub const max_shared = 16;
const types = @import("cuda_engine_types.zig");
const snapshots = @import("cuda_snapshots.zig");
const prompt_solo = @import("cuda_prompt_solo.zig");
pub const Part = types.Part;
pub const Kept = types.Kept;
const Seq = types.Seq;
const Snap = types.Snap;
pub const Resume = types.Resume;
pub const Cancel = types.Cancel;
pub const Marks = types.Marks;
pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    group: ?*const cuda.tp_group.Group = null, // the main stream's all-gathers (RoCE on two ranks); null: lib and comm
    verify_graphs: ?*verify_graph.Graphs = null,
    dense_segments: ?@import("cuda_dense_owner.zig").Owner = null,
    decode_select: ?@import("cuda_decode_select.zig").Owner = null,
    kda_segments: ?@import("cuda_kda_owner.zig").Owner = null,
    images: ?*images_mod.Images = null, // --vision: a prompt pass's image rows (rank 0 encodes, the others receive)
    packs: mm_mod.Packed = .{}, // the linears packed at load and their bits (TF_GLM_DENSE_BITS, _HEAD_BITS)
    expert_fused_calls: u64 = 0,
    expert_fact_calls: u64 = 0,
    expert_prompt_whole_calls: u64 = 0,
    prompt_batch_calls: u64 = 0,
    drafter: ?*dflash.Drafter = null, // --drafter: DFlash2 (every rank runs its passes; rank 0 selects the drafts)
    profs: [fw.window_rows]?*prof_mod.Prof = @splat(null), // TF_GLM_PROF=1: verify sections' GPU time, by rows
    shared_prof: ?*@import("cuda_engine_prof.zig").Shared = null,
    profiling_prompt: ?*prof_mod.Prof = null,
    prompt_profs: [4]?*prof_mod.Prof = @splat(null), // and prompt chunks': to 64, 256, 1024, 2048 rows
    checkpoint_digest: [32]u8,
    settings: ranks.Settings,
    sequence_counts: sequence_slots.Counts = .{},
    row_counts: @import("cuda_verify_rows.zig").Counts = .{},
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    pair: ?prompt_pair.Pair = null, // TF_GLM_PROMPT_PAIR: a paired prompt step's second chunk
    bd: fw.Buffers, // verify windows
    bm: ?fw.Buffers = null, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    pairs: cuda.DeviceBuffer, // greedy records: one/rank, or [world][window_rows] when batching is agreed
    smp: sampler.Sampler,
    seqs: std.ArrayList(?*Seq) = .empty, // every stream's sequence by slot (the lane backend's), slot 0 made at init
    cur: *Seq = undefined, // the bound slot's: every step below runs on it
    slot: u32 = 0,
    capacity: usize,
    mtp: bool,
    heat_gate: heat.Gate, // upstream's bands (off unless set), read before each prompt chunk on the gathered maximum
    shared_buf: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    snap_pages: ?snapshots.pages.Pool = null,
    snap_setup: ?snapshots.Setup = null,
    next_snap: u64 = 1, // rank 0: the next id
    announce: u64 = 0, // rank 0: a state kept at this chunk's mark, for the other ranks to copy at its gate
    dropped: std.ArrayList(u32) = .empty, // rank 0: ids dropped since the other ranks last heard (cuda_follow)
    prompt_logprobs: ?prompt_lp.Want = null, // a prompt pass's rows asked for (every rank runs the head on them)
    prompt_bufs: ?prompt_lp.Buffers = null, // their head passes' buffers, made at the first pass that asks
    prompt_tried: bool = false, // that pass came: without buffers after it, the passes take the window's
    prompt_out: prompt_lp.Output, // a head pass's logprob words, then its picks, host-mapped (rank 0 reads them)
    structured: bool = false, // a structured output step: every rank's draws gather whole rows, rank 0 masks them
    pages: ?*pages_mod.Pages = null, // the other streams' KV pages and sequences (cuda_pages), once the server attached them
    index_split: bool = false, // the next prompt step's indexer rows split across the ranks: rank 0's choice, the chunk's flag 2
    index_splits: u64 = 0, // prompt steps whose indexer rows split (glm-split-check counts them)
    share_pages: bool = false, // TF_KV_SHARE: a kept state leaves its prefix's pages to the shared store that holds them
    l2pf: ?l2pf_mod.L2pf = null, // TF_GLM_L2PF: the prefetch's side stream, kernels and table, made after the weights tile
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
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
        if (o.settings.prewarm_sequence_slots and (!o.settings.retain_sequence_slots or o.settings.pages or
            o.vision != null or o.settings.kv_host != 0 or o.settings.verify_graphs)) return error.PrewarmSequenceScope;
        if (o.settings.ffn_graphs and ((world != 2 and world != 4) or o.settings.pages or o.settings.kv_host != 0 or
            o.vision != null or !@import("cuda_target_precision.zig").supported(o.settings) or prof_mod.Prof.wanted()))
            return error.FfnGraphScope;
        try @import("cuda_dflash_head_mode.zig").require(o.settings, o.drafter, o.mtp, o.vision != null);
        try @import("cuda_verify_rows.zig").require(o.settings, world, o.vision != null, o.drafter, o.mtp);
        try @import("cuda_kda_options.zig").require(o.settings, world, o.vision != null, o.mtp);
        if (o.settings.retain_sequence_slots and (o.settings.pages or o.settings.verify_graphs))
            return error.RetainSequenceScope;
        if (o.settings.dense_segments and ((world != 2 and world != 4) or o.settings.pages or
            o.settings.kv_host != 0 or o.settings.verify_graphs or o.vision != null)) return error.DenseScope;
        if (o.settings.verify_graphs and (o.settings.pages or o.settings.kv_host != 0 or
            o.settings.l2pf.on or o.vision != null or prof_mod.Prof.wanted())) return error.VerifyGraphScope; // before anything is allocated
        if (o.settings.dense_rows16 and (!o.settings.dense_segments or o.settings.verify_rows < 16))
            return error.DenseRowsScope;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24));
        var why: @import("core").modelopt.Why = .{};
        try @import("cuda_nvfp4_graph.zig").checkpoint(gpa, text, o.settings);
        const admitted = if (exl3.isExl3(gpa, text)) exl3.admit(gpa, text, &why) else @import("cuda_formats.zig").admit(gpa, text, &why);
        admitted catch |err| { // anything but NVFP4, MIXED_PRECISION in recipes read here or EXL3 is refused by name before loading
            std.log.err("{s}: {s}", .{ cfg_path, why.text() });
            return err;
        };
        const c = try config.parse(gpa, text);
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        const config_hash = try ranks.hashFile(a, io, cfg_path);
        const program_hash = try ranks.hashFile(a, io, "/proc/self/exe");
        const checkpoint_digest = try ranks.artifact(a, io, checkpoint);
        var fp = ranks.Fingerprint.of(
            world,
            rank,
            o.settings,
            o.capacity,
            o.mtp,
            config_hash,
            program_hash,
            checkpoint_digest,
        );
        if (o.vision) |lim| ranks.mediaFingerprint(&fp, lim);
        fp.mtp += @intFromBool(o.drafter);
        try ranks.agree(d, lib, comm, stream, fp, try a.alloc(ranks.Fingerprint, world));
        try ranks.probeStart(d, lib, comm, stream, o.settings, rank, world);
        const share = try shard.Share.of(&c, rank, world);
        const kb = @import("cuda_kda_block.zig");
        kb.pad_to = if (o.settings.kda_pad) kb.padWidth(c.kda_heads, world) else 0; // read once: rowWidth reads only this
        const spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(spans);
        for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
        const block32 = !exl3.isExl3(gpa, text) and try @import("cuda_experts.zig").wantsBlock32(gpa, text);
        try @import("cuda_nvfp4_mode.zig").require(o.settings.nvfp4_fact, exl3.isExl3(gpa, text), block32, world);
        var m = try fw.Modules.load(d, ctx, block32);
        errdefer m.deinit();
        try @import("cuda_latent_mma.zig").install(&m, gpa, io, d, ctx, o.settings.absorb_mma);
        try @import("cuda_exl3_whole_owner.zig").install(gpa, io, d, &m.exl3, o.settings.exl3_prompt_whole, exl3.isExl3(gpa, text));
        try @import("cuda_tile_affine.zig").prepare(&m, o.settings.q4_affine_mse);
        try @import("cuda_affine_prompt_owner.zig").selectSplits(&m, o.settings.q4_python_splits);
        try @import("cuda_affine_prompt_owner.zig").install(&m, gpa, io, d, o.settings.q4_python_prompt);
        if (o.settings.exl3_fused_decode) {
            if (!exl3.isExl3(gpa, text) or c.hidden != 4096 or
                !((world == 4 and share.moe.len() == 512) or (world == 2 and share.moe.len() == 1024)) or
                try ctx.attribute(.compute_capability_major) != 12 or try ctx.attribute(.compute_capability_minor) != 1)
                return error.Exl3DecodeScope;
            try @import("cuda_exl3_decode.zig").configure(gpa, io, d, &m.exl3);
        }
        const pair = prompt_pair.wanted(o.settings.pair, o.settings.solo, world);
        var pieces = try fw.Pieces.init(d, o.settings.pieces, pair);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .heat_gate = .{ .bands = o.settings.heat, .root = @import("cuda_engine_prof.zig").heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .checkpoint_digest = checkpoint_digest, .settings = o.settings, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .pairs = undefined, .smp = undefined, .prompt_out = undefined, .mtp = o.mtp };
        if (o.settings.exl3_fused_decode) e.m.exl3.decode_calls = &e.expert_fused_calls;
        const has_packs = o.settings.dense_bits != 16 or o.settings.head_bits != 16;
        if (has_packs and !o.settings.tiled) return error.PackedNeedsGlmMm;
        var tiler: @import("cuda_tile_affine.zig").Context = .{
            .gpa = gpa, .io = io, .affine = o.settings.q4_affine_mse, .head_fp8 = o.settings.q4_head_fp8,
            .lane = &e.m.lane, .qs = &e.m.q4, .mk = &e.m.mm, .s = stream, .d = d,
            .enabled = o.settings.tiled, .dense = o.settings.dense_bits, .head = o.settings.head_bits,
            .packs = &e.packs,
        };
        tiler.configure(o.settings);
        tiler.calibration = if (o.settings.q8_activation or o.settings.q4_activation) .{ .manifest = o.settings.calibration_sha, .checkpoint = checkpoint_digest } else null;
        e.w = try @import("cuda_load_owner.zig").finish(try weights.load(gpa, io, d, stream, e.m.tools(), checkpoint, c, share, layers, @import("cuda_dflash_head_mode.zig").selection(o.settings, o.mtp, o.drafter)), tiler, @import("cuda_tile_affine.zig").Context.apply);
        errdefer e.w.deinit();
        std.log.info("rank {d}: GLM_DFLASH_ONLY_HEAD enabled={any} target_mtp={any} borrowed_head={any}", .{ rank, o.settings.dflash_only_head, e.w.mtp != null, e.w.draftHead(o.settings.dflash_only_head) != null });
        @import("cuda_exl3_whole_owner.zig").attach(e);
        errdefer {
            gpa.free(e.packs.ptrs);
            gpa.free(e.packs.bits);
        }
        try @import("cuda_nvfp4_mode.zig").open(e);
        e.m.mm.prompt_fixed_k = o.settings.prompt_fixed_k;
        e.m.tiled = o.settings.tiled;
        if (o.settings.tiled) e.m.packs = &e.packs;
        if (o.settings.l2pf.want(try ctx.capability(), world)) e.l2pf = try l2pf_mod.L2pf.init(gpa, d, &e.w, o.settings.l2pf);
        errdefer if (e.l2pf) |*p| p.deinit();
        if (o.settings.pages) try pages_mod.ownSequence(e) else try e.bind(0); // on pages: no window, its pool places it
        errdefer {
            e.free(0);
            e.seqs.deinit(gpa);
        }
        e.bp = try fw.Buffers.init(d, &e.w, prompt_solo.bufferRows(o.settings.solo), true, o.capacity);
        errdefer e.bp.deinit();
        if (o.settings.solo and rank == 0) std.log.info("glm: prompt passes of {d} rows, {d} while another stream decodes", .{ fw.solo_rows, fw.prefill_rows });
        if (world > 1 and rank == 0) std.log.info("glm: a prompt chunk's exchanges in {d} row pieces", .{o.settings.pieces});
        if (pair) e.pair = try prompt_pair.Pair.init(d, &e.w, &e.bp, &e.pieces, o.capacity);
        errdefer if (e.pair) |*p| p.deinit();
        e.bd = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        if (!@import("cuda_dflash_head_mode.zig").omitted(e)) e.bm = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer if (e.bm) |*b| b.deinit();
        if (o.mtp) e.bs = try fw.Buffers.initShared(d, &e.w, shared_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        if (o.settings.dense_segments) {
            const b = if (e.bs) |*v| v else return error.DenseScope;
            e.dense_segments = try @import("cuda_dense_owner.zig").Owner.init(gpa, io, d, stream, &e.w, b.rows, b.nch);
            e.dense_segments.?.max_part_rows = if (o.settings.dense_rows16) 16 else 8;
        }
        errdefer if (e.dense_segments) |*o_dense| o_dense.deinit();
        try @import("cuda_decode_select.zig").open(e, gpa, io);
        errdefer @import("cuda_decode_select.zig").close(e);
        try @import("cuda_kda_options.zig").open(e);
        errdefer if (e.kda_segments) |*o_kda| o_kda.deinit();
        std.log.info("rank {d}: GLM_DENSE_SEGMENTS on={any} scratch_and_pinned={d}", .{ rank, e.dense_segments != null, if (e.dense_segments) |*o_dense| o_dense.bytes() else 0 });
        std.log.info("rank {d}: GLM_DENSE_ROWS16 requested={any} max_part_rows={d}", .{ rank, o.settings.dense_rows16, if (o.settings.dense_rows16) @as(u32, 16) else 8 });
        const pick_rows: usize = if (o.settings.greedy_shared) shared_rows else if (o.settings.greedy_batch) fw.window_rows else 1;
        e.pairs = try cuda.DeviceBuffer.alloc(d, pair_bytes * world * pick_rows);
        errdefer e.pairs.free();
        try d.check(d.api.cuMemsetD8_v2(e.pairs.ptr, 0, e.pairs.len), "cuMemsetD8");
        std.log.info("rank {d}: GLM_GREEDY_BATCH on={any} rows_cap={d} send_max={d} scratch={d}", .{ rank, o.settings.greedy_batch, pick_rows, pair_bytes * pick_rows, e.pairs.len });
        std.log.info("rank {d}: GLM_GREEDY_SHARED on={any} rows_cap={d}", .{ rank, o.settings.greedy_shared, pick_rows });
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        errdefer e.smp.deinit();
        if (o.settings.sample_batch) try e.smp.enableBatch(stream, spans, rank);
        const sample_bytes = if (e.smp.batch) |b| try b.plan.scratchBytes() else 0;
        std.log.info("rank {d}: GLM_SAMPLE_BATCH requested={any} active={any} rows_cap={d} scratch={d}", .{ rank, o.settings.sample_batch, e.smp.batch != null, fw.window_rows, sample_bytes });
        errdefer e.freeProfs();
        if (prof_mod.Prof.wanted()) for (&e.profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
        };
        if (prof_mod.Prof.wanted()) for (&e.prompt_profs) |*p| {
            p.* = try prof_mod.Prof.init(d, io);
            p.*.?.every = 4;
        };
        e.shared_prof = try @import("cuda_engine_prof.zig").Shared.init(gpa, d, io, prof_mod.Prof.wanted());
        e.prompt_out = try prompt_lp.Output.init(d);
        errdefer e.prompt_out.free();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        errdefer e.shared_buf.free();
        errdefer @import("cuda_load_owner.zig").drain(stream, d);
        if (o.settings.ffn_graphs) e.m.ffn_graphs = try @import("cuda_ffn_graph.zig").Graphs.createMode(gpa, d, rank, @import("cuda_nvfp4_graph.zig").wide(o.settings));
        errdefer @import("cuda_ffn_warm.zig").unwind(e);
        std.log.info("rank {d}: GLM_FFN_GRAPHS on={any} max={d} compute_only=true hit_vote=false nvfp4={any}", .{ rank, o.settings.ffn_graphs, @import("cuda_ffn_graph.zig").capacityFor(@import("cuda_nvfp4_graph.zig").wide(o.settings)), o.settings.nvfp4_ffn_graphs });
        if (o.settings.verify_graphs) e.verify_graphs = try verify_graph.Graphs.create(gpa, d);
        errdefer if (e.verify_graphs) |g| g.destroy(gpa);
        std.log.info("rank {d}: GLM_VERIFY_GRAPHS on={any} max=32 raw_nccl_only=true hit_vote=true", .{ rank, e.verify_graphs != null });
        try e.stream.synchronize();
        try @import("cuda_ffn_warm.zig").prepare(e);
        @import("cuda_verify_rows.zig").startup(o.settings, rank);
        sequence_slots.startup(e);
        return e;
    }
    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.prompt_batch_calls != 0)
            std.log.info("rank {d}: GLM_PROMPT_BATCH_CLOSE calls={d}", .{ e.share.rank, e.prompt_batch_calls });
        @import("cuda_exl3_whole_owner.zig").census(e);
        e.m.freeFfnGraphs();
        if (e.verify_graphs) |g| g.destroy(e.gpa);
        e.verify_graphs = null;
        e.gpa.free(e.packs.ptrs);
        e.gpa.free(e.packs.bits);
        if (e.drafter) |dr| dr.deinit();
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        if (e.snap_pages) |*p| p.deinit();
        e.dropped.deinit(e.gpa);
        e.shared_buf.free();
        e.freeProfs();
        if (e.prompt_bufs) |*b| b.free();
        e.prompt_out.free();
        e.smp.deinit();
        e.pairs.free();
        if (e.kda_segments) |*o_kda| o_kda.deinit();
        if (e.dense_segments) |*o_dense| o_dense.deinit();
        @import("cuda_nvfp4_mode.zig").close(e);
        @import("cuda_decode_select.zig").close(e);
        if (e.bs) |*b| b.deinit();
        if (e.bm) |*b| b.deinit();
        e.bd.deinit();
        if (e.pair) |*p| p.deinit();
        e.bp.deinit();
        if (e.pages) |p| p.deinit();
        e.freeSequences();
        if (e.l2pf) |*p| p.deinit();
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        e.stream.deinit();
        e.gpa.free(e.spans);
        @import("cuda_verify_rows.zig").close(e);
        const gpa = e.gpa;
        gpa.destroy(e);
    }
    const freeProfs = @import("cuda_engine_prof.zig").freeProfs;
    pub fn exchange(e: *const Engine) fw.Exchange {
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm, .group = e.group } };
    }
    pub fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        const im: ?fw.ImageRows = if (e.images) |v| .{ .v = v, .slot = e.slot } else null;
        return .{ .d = e.d, .m = &e.m, .decode_select = if (e.decode_select) |*owner| owner else null, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = prompt_pair.piecesOf(e, b), .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .images = im, .taps = if (e.drafter) |dr| dr.taps(b == &e.bp) else null, .l2pf = if (e.l2pf) |*p| p else null, .dense_segments = if (e.dense_segments) |*o_dense| o_dense else null, .kda_segments = if (e.kda_segments) |*o_kda| o_kda else null };
    }
    pub fn sharedView(e: *Engine, b: *fw.Buffers, segs: []const fw.Seg, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs }};
        return .{ .d = e.d, .m = &e.m, .decode_select = if (e.decode_select) |*owner| owner else null, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .taps = if (e.drafter) |dr| dr.taps(false) else null, .l2pf = if (e.l2pf) |*p| p else null, .dense_segments = if (e.dense_segments) |*o_dense| o_dense else null, .kda_segments = if (e.kda_segments) |*o_kda| o_kda else null };
    }
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        const made = e.seqs.items[slot] == null;
        const previous = if (e.images != null) e.cur else null;
        const previous_slot = if (e.images != null) e.slot else 0;
        errdefer {
            if (made) e.free(slot);
            if (previous) |q| {
                e.cur = q;
                e.slot = previous_slot;
            }
        }
        if (made) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = try fw.State.initHead(e.gpa, e.d, &e.w, e.capacity, e.settings.kv8, !@import("cuda_dflash_head_mode.zig").omitted(e)) };
            e.seqs.items[slot] = q;
            if (e.settings.retain_sequence_slots) e.sequence_counts.created += 1;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
        if (e.images) |v| try v.activate(slot);
    }
    pub const reserveSequenceSlots = sequence_slots.reserveLocal;
    /// Releases a sequence (optionally retained after drain), returns paged rows, then binds slot0.
    pub fn release(e: *Engine, slot: u32) void {
        if (e.images) |v| v.release(slot);
        const paged = if (e.pages) |p| p.paged(slot) else false;
        if (paged) e.pages.?.close(slot) else if (slot != 0 and !sequence_slots.retain(e, slot)) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }
    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        cuda.completion.drain(e.d, e.stream, "GLM sequence release did not drain");
        if (e.verify_graphs) |g| g.invalidate();
        q.st.deinit();
        if (e.settings.retain_sequence_slots) e.sequence_counts.destroyed += 1;
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }
    /// The actual deinit phase always destroys every slot, including retained sequences.
    pub fn freeSequences(e: *Engine) void {
        for (0..e.seqs.items.len) |i| e.free(i);
        sequence_slots.close(e);
        e.seqs.deinit(e.gpa);
    }
    /// Bytes one more stream's sequence takes on this rank (its state arena).
    pub fn seqBytes(e: *const Engine) u64 {
        return e.seqs.items[0].?.st.arena.buf.len;
    }
    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }
    /// A follower's begin: rank 0's next request; no image rows from an earlier one splice into it.
    pub fn beginFollowing(e: *Engine, sampling: ?sampler.Sampling) !void {
        if (e.images) |v| v.clear();
        return e.begin(sampling);
    }
    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
        if (e.drafter) |dr| try dr.restart(e.slot, 0);
    }
    /// decode.prefill: `ids` in 2,048-row chunks, then the first token from the last row; `from` restores a kept state.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) : (start += fw.prefill_rows) {
            if (try e.prefillChunk(ids[start..@min(ids.len, start + fw.prefill_rows + 1)], start, ids.len, null, cancel, marks)) |first| return first;
        }
    }
    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + e.settings.verify_rows >= e.cur.st.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
        try e.restoreState(r.id);
        return r.at;
    }
    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        const split = e.index_split; // this step's alone
        e.index_split = false;
        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
        if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
        const rows = @min(len - start, @as(usize, if (part.len > fw.prefill_rows + 1) fw.solo_rows else fw.prefill_rows));
        if (rows > fw.prefill_rows and e.pair != null) return prompt_pair.prefill(e, part, start, len, cancel, marks, split);
        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
        if (rows > fw.prefill_rows and !prompt_solo.allowed(e, start, rows)) return error.BadChunk;
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bp, &one);
        f.index_split = split;
        if (start + rows <= len -| @as(usize, @intCast(dflash.WIN))) f.taps = null; // none read (dflash.promptRun)
        const as_i32: []const i32 = @ptrCast(part);
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        f.prof = e.prompt_profs[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3];
        const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };
        if (f.prof) |p| try p.beginPrompt(e.stream, labels[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3], .{ .rank = e.share.rank, .slot = e.slot, .start = start, .rows = rows }, e.c.run, e.pieces.count);
        if (f.prof) |p| e.profiling_prompt = p;
        defer if (f.prof != null) {
            e.profiling_prompt = null;
        };
        defer if (f.prof) |p| p.end();
        errdefer if (f.prof) |p| {
            p.prompt.?.invalid = true;
        };
        var stamp = prof_mod.span(f.prof, e.stream, .gate);
        e.gate(cancel) catch |err| {
            if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
            return err;
        };
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .gate);
        try f.run(as_i32[0..rows]);
        e.index_splits += @intFromBool(split);
        stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
        if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
        stamp.end();
        stamp = prof_mod.span(f.prof, e.stream, .drafter);
        if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len);
        stamp.end();
        stamp = prof_mod.span(f.prof, e.stream, .draw);
        const last = start + rows == len;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, 0, &t);
            first = t[0];
        }
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .draw);
        if (last) try e.holdPrompt(e.bp.fnormed + (rows - 1) * e.c.hidden * 2);
        stamp = prof_mod.span(f.prof, e.stream, .mtp);
        if (e.mtp) try prompt_solo.mtpRows(e, &f, as_i32);
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .mtp);
        stamp = prof_mod.span(f.prof, e.stream, .commit);
        try f.commit(rows);
        stamp.end();
        if (f.prof) |p| p.mark(e.stream, .commit);
        return first;
    }
    pub const gate = @import("cuda_engine_gate.zig").gate;
    pub const shared = @import("cuda_engine_gate.zig").shared;
    pub const stateBytes = snapshots.bytes;
    pub const prepareSnapshots = snapshots.prepare;
    pub const keepState = snapshots.keep;
    pub const keepAs = snapshots.keepAs;
    pub const restoreState = snapshots.restore;
    pub const holds = snapshots.holds;
    pub const dropState = snapshots.drop;
    const freeSnap = snapshots.free;
    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
        try @import("cuda_verify_rows.zig").window(e.settings.verify_rows, ids.len, e.cur.sampling != null, e.structured);
        var one: [1]fw.Rank = undefined;
        var f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        f.prof = e.profs[ids.len - 1];
        if (f.prof) |p| try p.begin(e.stream, verifyLabel(ids.len));
        try @import("cuda_verify_forward.zig").run(e, &f, @ptrCast(ids));
        if (e.drafter) |dr| try dr.noteVerify(e.slot, 0, ids.len, at, true);
        e.cur.rows_at = e.bd.fnormed;
        try e.drawVerify(e.bd.logits, ids.len, at + 1, bits, out);
        if (f.prof) |p| p.end();
        @import("cuda_verify_rows.zig").solo(e, ids.len);
    }
    const verifyLabel = @import("cuda_engine_prof.zig").verifyLabel;
    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > e.settings.verify_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            try @import("cuda_verify_rows.zig").window(e.settings.verify_rows, p.ids.len, e.cur.sampling != null, e.structured);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = undefined;
        var f = e.sharedView(b, segs[0..parts.len], &one);
        const shared_prof = try @import("cuda_engine_prof.zig").beginShared(e, &f, rows, parts.len);
        errdefer if (shared_prof) |v| v.cancel();
        try @import("cuda_verify_forward.zig").run(e, &f, @ptrCast(ids[0..rows]));
        if (e.drafter) |dr| for (parts, segs[0..parts.len], 0..) |p, g, i| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, i == 0);
        try @import("cuda_shared_pick.zig").draw(e, b, parts, segs[0..parts.len], rows);
        if (shared_prof) |v| v.finish(e.stream);
        @import("cuda_verify_rows.zig").shared(e, parts);
    }
    /// DFlash2's round (every rank, the same `reqs`): each stream's kept rows into its context, then the block pass.
    pub fn draftBlock(e: *Engine, reqs: []dflash.Req) !void {
        const dr = e.drafter orelse return error.NoDrafter;
        for (reqs) |*q| q.pos = @intCast((e.seqs.items[q.slot] orelse return error.UnknownSlot).st.pos);
        try dr.draft(e, reqs);
    }
    pub const holdPrompt = @import("cuda_prompt_hold.zig").save;
    pub const prefillMany = @import("cuda_prompts.zig").prefillMany;
    /// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        try f.commit(n);
    }
    /// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
        return @import("cuda_engine_mtp.zig").draft(e, from_prompt, follow, depth, out, chances);
    }
    /// Ordinary greedy verification; all other draw callers retain their original path.
    pub fn drawVerify(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (e.settings.greedy_batch and !e.structured and e.cur.sampling == null)
            return @import("cuda_engine_pick.zig").greedyRows(e, logits, rows, out);
        return e.draw(logits, rows, first, bits, out);
    }
    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
        if (e.structured) return e.drawWhole(logits, rows, first, bits, out);
        if (e.cur.sampling) |s| {
            try e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
            return e.checkGathers(); // the draw read rows the forward's gathers made
        }
        for (0..rows) |r| out[r] = try e.greedy(logits + r * e.share.vocab.len() * 2);
    }
    pub const promptHead = @import("cuda_structured.zig").promptHead;
    pub const drawWhole = @import("cuda_structured.zig").drawWhole;
    pub const redrawFirst = @import("cuda_structured.zig").redrawFirst;
    /// decode.sample_rows' greedy pick: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    const greedy = @import("cuda_engine_pick.zig").greedy;
    const greedyChance = @import("cuda_engine_pick.zig").greedyChance;
    /// A failed RoCE gather (its rows hold nothing) or TF_GLM_KV_HOST gather past its slots, after a synchronize.
    pub fn checkGathers(e: *const Engine) !void {
        if (e.group) |g| try g.check();
        try @import("cuda_kv_host.zig").checkActive(); // a TF_GLM_KV_HOST gather past its slots
    }
    pub fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};
