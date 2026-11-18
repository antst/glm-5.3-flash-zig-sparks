//! One GLM rank owns its own local NVMe store, bounded pinned staging and source-credit slots.
const std = @import("std");
const cuda = @import("cuda");
const nvme = @import("core").nvme1107;
const spill = nvme.spill;
const image = @import("cuda_nvme_image.zig");
const Engine = @import("cuda_engine.zig").Engine;

pub const max_metadata = 128 << 10;
pub const max_pending = 2;
pub const block_tokens = 256;
pub const Config = struct {
    cap: u64,
    floor: u64,
    staging: u64,
    index: u64,
    min_tokens: u32 = 8192,
    lookahead: u1 = 1,
    serial_mode: bool = false,
};
pub var local: ?*State = null; // one native GLM engine per process

pub const State = struct {
    e: *Engine,
    root: [:0]u8,
    staging: cuda.HostBuffer,
    buffers: [4][1]spill.slots.Buffer,
    side: nvme.tier_state.Side,
    slots: [max_pending]u64 = @splat(0),
    tokens: std.ArrayList(u32) = .empty,
    token_id: u64 = 0,
    token_at: u32 = 0,

    pub fn open(e: *Engine, root: []const u8, compat: spill.Digest, cfg: Config) !*State {
        if (e.settings.q8_mse) return error.Q8MseNvmeScope;
        if (e.settings.q4_activation) return error.Q4ActivationNvmeScope;
        if (local != null or e.share_pages or root.len == 0 or root.len > 4096 or cfg.cap == 0 or cfg.index < 16 << 20 or cfg.staging > 2 << 30) return error.NvmeSettings;
        const max_at = (e.capacity -| @as(usize, @import("cuda_state.zig").window_rows)) / @import("cuda_native.zig").snap_grid * @import("cuda_native.zig").snap_grid;
        if (max_at == 0 or max_at > std.math.maxInt(u32) - 1) return error.NvmeSettings;
        const a = e.gpa;
        const s = try a.create(State);
        errdefer a.destroy(s);
        const own_root = try a.dupeSentinel(u8, root, 0);
        errdefer a.free(own_root);
        try std.Io.Dir.cwd().createDirPath(e.io, root);
        const img = image.of(e);
        var ps: std.ArrayList(nvme.layout.Part) = .empty;
        defer ps.deinit(a);
        try img.parts(e, @intCast(max_at), &ps, a);
        const l = try nvme.layout.Layout.init(ps.items, @intCast(max_at), block_tokens);
        const small_max = std.mem.alignForward(u64, max_metadata, 256) + l.smallBytes();
        const plan = if (cfg.serial_mode) try spill.slots.serialPlan(cfg.staging, l.blockBytes(), small_max) else try spill.slots.plan(cfg.staging, l.blockBytes(), small_max, max_pending, 1);
        var staging = try cuda.HostBuffer.alloc(e.d, @intCast(plan.bytes()));
        errdefer staging.free();
        if (@intFromPtr(staging.bytes.ptr) % 4096 != 0) return error.NvmeStagingAlignment;
        s.e = e;
        s.root = own_root;
        s.staging = staging;
        s.slots = @splat(0);
        s.tokens = .empty;
        s.token_id = 0;
        s.token_at = 0;
        var off: usize = 0;
        inline for (.{ .{ plan.write, plan.block_slot }, .{ plan.read, plan.block_slot }, .{ plan.small, plan.small_slot }, .{ plan.restore, plan.small_slot } }, 0..) |sizes, i| {
            if (plan.shared_small_restore and i == 3) s.buffers[i] = s.buffers[2] else {
                const bytes = sizes[0] * sizes[1];
                s.buffers[i][0] = @alignCast(s.staging.bytes[off..][0..bytes]);
                off += bytes;
            }
        }
        std.debug.assert(off == plan.bytes());
        const store = try spill.Store.open(a, e.io, .{ .root = s.root, .rank = @intCast(e.share.rank), .compat = @import("cuda_latent_mma.zig").compatibility(@import("cuda_snapshot_mode.zig").compatibilityFor(@import("cuda_affine_prompt_owner.zig").compatibilityWithSplits(compat, e.settings.q4_python_prompt, e.settings.q4_python_splits), e.settings.kv_live_half, e.drafter != null), e.settings.absorb_mma), .limits = .{ .max_tokens = @intCast(max_at + cfg.lookahead), .block_tokens = block_tokens, .block_bytes = l.blockBytes(), .max_small_bytes = small_max, .max_forks = 4096 }, .cap_bytes = cfg.cap, .floor_bytes = cfg.floor, .min_tokens = cfg.min_tokens, .index_bytes = @intCast(cfg.index), .readers = 2, .max_reads = 1, .staging = .{ .write = &s.buffers[0], .read = &s.buffers[1], .small = &s.buffers[2], .restore = &s.buffers[3], .shared_small_restore = plan.shared_small_restore } });
        s.side = nvme.tier_state.Side.init(a, e.io, store, img, if (cfg.serial_mode) 1 else max_pending, max_metadata);
        s.side.serial_read_wait_ns = if (cfg.serial_mode) 3 * std.time.ns_per_s else 0;
        s.side.lookahead = cfg.lookahead;
        s.side.defer_releases = true;
        std.log.info("GLM_NVME1107_OPEN rank={d} root={s} cap={d} staging={d} small_max={d} block={d} index={d} lookahead={d}", .{ e.share.rank, root, cfg.cap, plan.bytes(), small_max, l.blockBytes(), cfg.index, cfg.lookahead });
        std.log.info("NATIVE_TIERED_MEMORY rank={d} mode={s} pinned_bytes={d} metadata_limit={d} pending={d}", .{ e.share.rank, if (cfg.serial_mode) "serial" else "parallel", plan.bytes(), cfg.index, s.side.max_pending });
        local = s;
        return s;
    }

    pub fn close(s: *State) void {
        const a = s.e.gpa;
        s.side.deinit();
        const ct = s.side.store.counts();
        std.log.info("NATIVE_TIERED_CLOSE rank={d} pinned={d} index_used={d} index_peak={d} refused={d}", .{ s.e.share.rank, s.staging.bytes.len, ct.index_bytes, ct.index_peak, ct.metadata_refused });
        std.log.info("NATIVE_TIERED_IO rank={d} restored={d} loaded_bytes={d} block_reads={d} small_reads={d}", .{ s.e.share.rank, ct.loaded, ct.loaded_bytes, ct.read_block_calls, ct.read_small_calls });
        std.log.info("NATIVE_TIERED_WRITE rank={d} committed={d} staged={d} written={d} bytes={d} failed={d} cancelled={d}", .{ s.e.share.rank, ct.committed, ct.staged_blocks, ct.written_blocks, ct.written_bytes, ct.write_failed, ct.cancelled });
        s.side.store.close(); // join I/O before freeing staging; workers never call CUDA or the rank link
        s.tokens.deinit(a);
        s.staging.free();
        a.free(s.root);
        if (local == s) local = null;
        a.destroy(s);
    }

    /// Completion credits, not queued CUDA calls: every participant must finish before a source is recycled.
    pub fn completionMask(s: *const State) u64 {
        var mask: u64 = 0;
        for (s.slots, 0..) |id, i| if (id != 0 and s.side.completed(id)) {
            mask |= @as(u64, 1) << @intCast(i);
        };
        return mask;
    }
};
