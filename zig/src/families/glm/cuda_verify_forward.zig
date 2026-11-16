//! Host preparation and state-key construction surround the unchanged verify compute body.
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const vg = @import("cuda_verify_graph.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const dsa = @import("cuda_dsa_block.zig");

pub fn keyOf(f: *const fw.Forward, rows: usize) !vg.Key {
    if (f.ranks.len != 1 or rows == 0 or rows > 32) return error.VerifyGraphScope;
    const r = f.ranks[0];
    if (r.b.prefill or f.images != null or f.prof != null or f.l2pf != null or f.index_split)
        return error.VerifyGraphScope;
    if (f.exchange == .nccl) if (f.exchange.nccl.group) |g| {
        if (g.fast != null or g.multi != null) return error.VerifyGraphTransport;
    };
    const count = if (r.segs.len > 0) r.segs.len else 1;
    if (count > vg.max_parts) return error.VerifyGraphScope;
    var key: vg.Key = .{ .scratch = r.b.arena.buf.ptr, .rows = rows, .count = count };
    for (key.parts[0..count], 0..) |*part, i| {
        const g: fw.Seg = if (r.segs.len > 0) r.segs[i] else .{ .st = r.st, .row0 = 0, .rows = rows };
        const st = g.st;
        if (st.cur.len > 64 or st.caches.len > vg.max_dsa or g.rows > fw.window_rows) return error.VerifyGraphScope;
        part.* = .{
            .state = @intFromPtr(st),
            .arena = st.arena.buf.ptr,
            .pos_device = st.pos_dev,
            .base = st.base,
            .row0 = g.row0,
            .rows = g.rows,
        };
        for (st.cur, 0..) |parity, j| part.parity |= @as(u64, parity) << @as(u6, @intCast(j));
        var di: usize = 0;
        for (r.w.layers) |l| switch (l.mixer) {
            .kda => {},
            .mla => |a| {
                const cache = st.caches[di];
                if (cache.pt != 0 or cache.ps != 0 or cache.pks != 0) return error.VerifyGraphScope;
                const np = sparse.poolBucket(st.pos, g.rows, cache.pk_rows - 2);
                part.attention[di] = .{
                    .cache = cache,
                    .dense = st.pos < a.dense_limit,
                    .sparse = dsa.sparseRows(a, st.pos, g.rows),
                    .chunks = @min(latent.chunksFor(st.pos + g.rows), r.b.nch),
                    .pools = np,
                    .score_blocks = (sparse.scoreColumns(st.pos, g.rows, np) + 63) / 64,
                };
                di += 1;
            },
        };
    }
    return key;
}

pub fn run(e: anytype, f: *const fw.Forward, ids: []const i32) !void {
    const graphs = e.verify_graphs orelse return f.run(ids);
    try f.prepare(ids); // token H2D, ready/rebase and host state updates execute before every replay
    const key = try keyOf(f, ids.len);
    const Context = struct {
        e: @TypeOf(e),
        f: *const fw.Forward,
        rows: usize,

        fn compute(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try c.f.compute(c.rows);
        }

        fn agree(ptr: *anyopaque, ok: bool) anyerror!bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            return try c.e.shared(if (ok) 0 else 1) == 0;
        }
    };
    var c: Context = .{ .e = e, .f = f, .rows = ids.len };
    _ = try graphs.run(key, .{
        .engine = @intFromPtr(e),
        .weights = @intFromPtr(&e.w),
        .modules = @intFromPtr(&e.m),
        .stream = @intFromPtr(e.stream.handle),
        .comm = @intFromPtr(e.comm),
        .taps = if (f.taps) |t| t.out else 0,
        .tap_function = if (f.taps) |t| @intFromPtr(t.f.handle) else 0,
    }, e.stream, .{ .ctx = &c, .all = Context.agree }, .{ .ctx = &c, .run = Context.compute });
}
