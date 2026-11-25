//! A compact prefix of a rank's committed GLM state; live sequence addresses and kernel layouts stay fixed.
const std = @import("std");
const State = @import("cuda_state.zig").State;
const page_rows = @import("cuda_state.zig").page_rows;
const Config = @import("config.zig").Config;
const cache = @import("cuda_dsa_block.zig");
const sparse = @import("cuda_sparse.zig");

const walk = @import("core").state_walk.Device;

/// TF_GLM_KV_LIVE_HALF=1 (checked by cuda_ranks): a keep holds the live KDA half alone; a disk image must name it.
pub fn liveHalfEnabled() bool {
    return std.mem.eql(u8, @import("profile").lookup("TF_GLM_KV_LIVE_HALF") orelse return false, "1");
}

/// The half a plan keeps: `on` with every KDA layer on the same half gives that half; otherwise both (null).
pub fn halfOf(cur: []const u1, layers: usize, on: bool) ?u1 {
    if (!on or layers == 0 or cur.len != layers) return null;
    const h = cur[0];
    for (cur) |b| if (b != h) return null;
    return h;
}

/// `len` bytes from `live`, or on a paged sequence's pages `span` bytes a page `page` bytes apart, from byte `from`.
pub const Part = walk.Part;
pub const Plan = struct {
    parts: [6 * (@import("config.zig").max_layers + 1) + 4]Part = undefined,
    count: usize = 0,
    len: usize = 0,
    ids: []const u32 = &.{}, // a paged sequence's pages in row order
    sizing: bool = false, // `size`: the rows counted, not read, so no page need be mapped
    shared: usize = 0, // a paged sequence's first pages a page-sharing store holds: their rows stay on them
    half: ?u1 = null, // TF_GLM_KV_LIVE_HALF: the KDA half the plan carries (null: both, the arena head whole)

    /// The state at `at`; past `shared` pages only, on a paged sequence whose first pages the store holds.
    pub fn init(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize) !Plan {
        return initWith(st, c, at, head, shared, st.cur, false);
    }

    /// The immutable saved parities choose each layer's bank; the destination slot may now hold different ones.
    pub fn initWith(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, cur: []const u1, live: bool) !Plan {
        return make(st, c, at, head, shared, false, cur, live);
    }

    /// A diagnostic plan with the same parity on every recurrent layer.
    pub fn withHalf(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, half: ?u1) !Plan {
        const h = half orelse return init(st, c, at, head, shared);
        var cur: [@import("config.zig").max_layers]u1 = undefined;
        if (st.kda_layers > cur.len) return error.SnapshotShape;
        @memset(cur[0..st.kda_layers], h);
        return initWith(st, c, at, head, shared, cur[0..st.kda_layers], true);
    }

    /// A prefix's bytes at `at`, rows to `capacity` counted held or not (a paged sequence holds its request's alone).
    pub fn size(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize, shared: usize) !usize {
        return sizeWith(st, c, capacity, at, head, shared, false);
    }

    /// A live-bank payload has one fixed size for every parity vector, including a mixed-layer state.
    pub fn sizeWith(st: *const State, c: *const Config, capacity: usize, at: usize, head: ?usize, shared: usize, live: bool) !usize {
        var whole = st.*;
        whole.capacity = capacity;
        return (try make(&whole, c, at, head, shared, true, st.cur, live)).len;
    }

    fn make(st: *const State, c: *const Config, at: usize, head: ?usize, shared: usize, sizing: bool, cur: []const u1, live: bool) !Plan {
        if (at > st.capacity or st.caches.len > @import("config.zig").max_layers) return error.SnapshotShape;
        if (st.rings.len != 2 * (st.caches.len + 1)) return error.SnapshotShape;
        if (shared > 0 and (st.table == 0 or shared * page_rows > at)) return error.SnapshotShape; // pages below `at` only
        var p: Plan = .{ .ids = st.pages, .sizing = sizing, .shared = shared, .half = halfOf(cur, st.kda_layers, live) };
        // a paged sequence's caches are its pages: its fields up to them end at its first ring
        const first = st.snapshotPrefixEnd();
        if (first < st.arena.buf.ptr) return error.SnapshotShape;
        try p.prefix(st, first, cur, live);
        for (st.caches, 0..) |x, j| try p.rows(st, c, x, j, at);
        if (head) |n| {
            if (n > st.capacity) return error.SnapshotShape;
            try p.rows(st, c, st.mtp_caches, st.caches.len, n);
        }
        if (st.mtp_pos_dev < st.arena.buf.ptr or st.mtp_pos_dev - st.arena.buf.ptr > st.arena.buf.len)
            return error.SnapshotShape;
        const end = if (st.table != 0) st.table else st.arena.buf.ptr + st.arena.buf.len; // its table stays its own
        if (end < st.mtp_pos_dev) return error.SnapshotShape;
        try p.add(st, st.mtp_pos_dev, @intCast(end - st.mtp_pos_dev));
        return p;
    }

    /// Canonical prefix: before both physical banks, each live layer in order, then after both banks.
    fn prefix(p: *Plan, st: *const State, first: u64, cur: []const u1, live: bool) !void {
        if (!live) return p.add(st, st.arena.buf.ptr, @intCast(first - st.arena.buf.ptr));
        const bank = try recurrentBytes(st);
        const end = try std.math.add(u64, st.rec, try std.math.mul(u64, bank, 2));
        if (st.rec < st.arena.buf.ptr or end > first or cur.len != st.kda_layers) return error.SnapshotShape;
        try p.add(st, st.arena.buf.ptr, @intCast(st.rec - st.arena.buf.ptr));
        var layer: usize = 0;
        while (layer < cur.len) {
            var next = layer + 1;
            while (next < cur.len and cur[next] == cur[layer]) : (next += 1) {}
            try p.add(st, st.recAt(cur[layer], layer), try std.math.mul(usize, next - layer, st.rec_layer));
            layer = next;
        }
        try p.add(st, end, @intCast(first - end));
    }

    fn rows(p: *Plan, st: *const State, c: *const Config, x: cache.Caches, j: usize, at: usize) !void {
        const latent_row = @as(usize, c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1);
        const latent = try std.math.mul(usize, at, latent_row);
        const ring = try std.math.mul(usize, @min(at, st.ring_rows), st.ring_row);
        const pooled = try std.math.mul(usize, @min(at / sparse.pool + 2, x.pk_rows), @as(usize, c.i_dim) * 2);
        try p.addCache(st, x, x.lc, latent, page_rows * latent_row, x.ps);
        if (x.ls != 0) try p.addCache(st, x, x.ls, try std.math.mul(usize, at, @sizeOf(f32)), page_rows * @sizeOf(f32), x.ps);
        try p.add(st, st.rings[2 * j], ring);
        try p.add(st, st.rings[2 * j + 1], ring);
        try p.addCache(st, x, x.pk, pooled, page_rows / sparse.pool * @as(usize, c.i_dim) * 2, if (x.pks != 0) x.pks else x.ps);
    }

    /// A cache's rows: in the arena, or on the sequence's pages, `span` bytes a page (past the store's `shared`).
    fn addCache(p: *Plan, st: *const State, x: cache.Caches, live: u64, len: usize, span: usize, page: u64) !void {
        if (x.pt == 0) {
            if (len != 0 and st.linear != null and (live == x.lc or live == x.ls)) return p.outside(live, len);
            return p.add(st, live, len);
        }
        const from = @min(len, p.shared * span);
        if (len == from) return;
        if ((!p.sizing and (len - 1) / span >= p.ids.len) or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len - from);
        p.parts[p.count] = .{ .live = live, .len = len - from, .span = span, .page = page, .from = from };
        p.count += 1;
    }

    /// The shared latent ranges are already present at their destination. Keep their canonical offsets.
    pub fn skipAliased(p: *Plan, st: *const State) void {
        for (p.parts[0..p.count]) |*part| {
            for (st.caches) |cache_part| {
                if (part.live == cache_part.lc or part.live == cache_part.ls) {
                    part.skip = true;
                    break;
                }
            }
        }
    }

    /// Where a part's bytes from `done` on sit, as many as lie in a row there.
    pub fn piece(p: *const Plan, part: Part, done: usize) Part {
        return walk.piece(p.ids, part, done);
    }

    fn add(p: *Plan, st: *const State, live: u64, len: usize) !void {
        if (len == 0) return;
        if (live < st.arena.buf.ptr) return error.SnapshotShape;
        const off = live - st.arena.buf.ptr;
        if (off > st.arena.buf.len or len > st.arena.buf.len - off or p.count == p.parts.len)
            return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }

    /// An immutable side buffer appended to the target's own state, copied under the same lease lifetime.
    pub fn outside(p: *Plan, live: u64, len: usize) !void {
        if (live == 0 or p.count == p.parts.len) return error.SnapshotShape;
        p.len = try std.math.add(usize, p.len, len);
        p.parts[p.count] = .{ .live = live, .len = len };
        p.count += 1;
    }
};

/// One physical recurrent bank, independently of the current layer parities.
pub fn recurrentBytes(st: *const State) !usize {
    if (st.cur.len != st.kda_layers) return error.SnapshotShape;
    return std.math.mul(usize, st.kda_layers, st.rec_layer);
}
