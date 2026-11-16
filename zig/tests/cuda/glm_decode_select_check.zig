//! Upstream planted checks using the separate selector module and the unchanged old-glue reference.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const sp = glm.sparse;
const parent = @import("glm_index_check.zig");
const Case = parent.Case;
const Bufs = parent.Bufs;
const zeros = parent.zeros;
const fromHost = parent.fromHost;
const scratch = parent.scratch;
const same = parent.same;
const step = parent.step;
const bf16 = parent.bf16;
const ms = parent.ms;
const d = 128;

const DCase = struct { pos: usize, rows: usize, cap: usize };
const dcases = [_]DCase{
    .{ .pos = 65536 - 32, .rows = 31, .cap = 16384 },
    .{ .pos = 262144 - 32, .rows = 32, .cap = 65536 },
    .{ .pos = 1048576 - 48, .rows = 48, .cap = 262144 },
    .{ .pos = 4 * 4096 - 1, .rows = 1, .cap = 8192 }, // 4,096 pools: the merge takes the row whole
    .{ .pos = 4 * 4096 + 3, .rows = 2, .cap = 8192 }, // the first row whole, the second past the merge's candidates
    .{ .pos = 131072 - 1, .rows = 1, .cap = 216080 },
    .{ .pos = 131072 - 4, .rows = 4, .cap = 216080 },
    .{ .pos = 204800 - 16, .rows = 16, .cap = 216080 },
    .{ .pos = 864256 - 1, .rows = 1, .cap = 216080 },
    .{ .pos = 864256 - 16, .rows = 16, .cap = 216080 }, // the production context: a bucket clipped to the capacity
};

/// How many rows took each path of the decode select (whole row, candidates, whole row after them, ties at the floor).
var dpaths = [4]usize{ 0, 0, 0, 0 };

/// The decode select (cuda_sparse.decodeSelect) against tf_glm_select_rows byte for byte: planted, then selectTokens.
pub fn run(gpu: check.Gpu, s: cuda.Stream, ops: cuda.Module, ks: *const glm.glue.Kernels, rand: std.Random) !usize {
    const gpa = gpu.gpa;
    try check.expect(try @import("contract_limits.zig").live(gpu, &.{"glm_decode_select"}) == 0, "decode select live limits", .{});
    const old: glm.glue.Set = .{ .glue = ks };
    var owner = try glm.decode_select.Owner.init(gpa, gpu.io, gpu.d, s, 48, 1 << 21, 0);
    defer owner.deinit();
    defer cuda.completion.drain(gpu.d, s, "decode select module fixture");
    const new: glm.glue.Set = .{ .glue = ks, .decode_select = &owner };
    try check.expect(try owner.scan.attribute(.max_threads_per_block) >= 256, "scan block limit", .{});
    try check.expect(try owner.merge.attribute(.max_threads_per_block) >= 1024, "merge block limit", .{});
    const special = [_]f32{ std.math.nan(f32), -std.math.nan(f32), 0.0, -0.0, std.math.inf(f32), -std.math.inf(f32) };
    var fails: usize = 0;
    for (dcases) |c| {
        const np = sp.poolBucket(c.pos, c.rows, c.cap);
        var bufs: Bufs = .empty;
        defer {
            for (bufs.items) |*b| b.free();
            bufs.deinit(gpa);
        }
        defer cuda.completion.drain(gpu.d, s, "decode select case");
        const p0: i32 = @intCast(c.pos);
        const dpos = try fromHost(gpu, &bufs, std.mem.asBytes(&p0));
        const scores = try zeros(gpu, &bufs, c.rows * np * 4);
        const pa = try zeros(gpu, &bufs, c.rows * sp.topk_pools * 8);
        const pb = try zeros(gpu, &bufs, c.rows * sp.topk_pools * 8);
        const path = try zeros(gpu, &bufs, c.rows * 4);
        const host = try gpa.alloc(f32, c.rows * np);
        defer gpa.free(host);
        for (0..6) |kind| {
            for (0..c.rows) |r| {
                const vis = (c.pos + r + 1) / 4;
                const limit = @min(np, @max(vis, sp.topk_pools));
                const row = host[r * np ..][0..np];
                for (row, 0..) |*x, i| {
                    // past the limit: no select may read
                    x.* = if (i >= limit) 1e9 else if (i >= vis) -std.math.inf(f32) else switch (kind) {
                        0 => rand.floatNorm(f32),
                        // the top level crowds the floor
                        1 => @as(f32, @floatFromInt(rand.uintLessThan(u32, 16))) * 0.25,
                        2 => 0.0, // every visible pool tied
                        3 => if (rand.uintLessThan(u32, 8) == 0) special[rand.uintLessThan(usize, special.len)] else rand.floatNorm(f32),
                        4 => rand.floatNorm(f32),
                        else => if (i % 3 == 0) 1.0 else if (i % 997 == 0) 2.0 + rand.float(f32) else rand.floatNorm(f32) * 0.1,
                    };
                }
                if (kind == 4) { // every sampled run high: the floor overshoots and the candidates fall short of K
                    for (0..4096) |q| {
                        const j = (q / 8) * limit / 512 / 8 * 8 + q % 8;
                        if (j < vis) row[j] = 100.0 + @as(f32, @floatFromInt(q));
                    }
                }
            }
            step(.{ .cap = c.cap, .pos = c.pos, .rows = c.rows }, np, "decode select: planted scores");
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = scores, .len = c.rows * np * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(host));
            var sel: cuda.Args = .{};
            inline for (.{ scores, pa, @as(i64, @intCast(np)), dpos, @as(c_int, sp.topk_pools) }) |v| sel.add(v);
            try cuda.launch.launch(old.glue.select_rows, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 128 } }, s, &sel);
            if (!try owner.selectPath(s, scores, pb, dpos, c.pos, c.rows, np, false, path)) {
                var fallback: cuda.Args = .{};
                inline for (.{ scores, pb, @as(i64, @intCast(np)), dpos, @as(c_int, sp.topk_pools) }) |v| fallback.add(v);
                try cuda.launch.launch(old.glue.select_rows, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 128 } }, s, &fallback);
            }
            try s.synchronize();
            var taken: [48]i32 = undefined;
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = path, .len = c.rows * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(taken[0..c.rows]));
            for (taken[0..c.rows], 0..) |q, r| {
                if (q < 0 or q > 3) {
                    std.debug.print("FAIL np {d} row {d} kind {d}: the decode select reported path {d}\n", .{ np, r, kind, q });
                    return error.TestFailed;
                }
                dpaths[@intCast(q)] += 1;
            }
            const f = try same(gpu, "decode select pools", pa, pb, c.rows * sp.topk_pools * 8, np, .{ .cap = c.cap, .pos = c.pos, .rows = c.rows });
            if (f != 0) std.debug.print("  (kind {d}, paths {any})\n", .{ kind, taken[0..c.rows] });
            fails += f;
        }
        // selectTokens on tf_glm_scores' rows: today's set against the decode select's, pools and token lists
        const qi = try gpa.alloc(u16, c.rows * 32 * d);
        defer gpa.free(qi);
        for (qi) |*x| x.* = bf16(rand.floatNorm(f32));
        const wts = try gpa.alloc(u16, c.rows * 32);
        defer gpa.free(wts);
        for (wts) |*x| x.* = bf16(rand.floatNorm(f32));
        const pk = try gpa.alloc(u16, c.cap * d);
        defer gpa.free(pk);
        for (pk) |*x| x.* = bf16(rand.floatNorm(f32) * 2);
        step(.{ .cap = c.cap, .pos = c.pos, .rows = c.rows }, np, "decode select: selectTokens, both sets");
        const dq = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(qi));
        const dw = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(wts));
        const dk = try fromHost(gpu, &bufs, std.mem.sliceAsBytes(pk));
        const sa = try scratch(gpu, &bufs, c.rows, np);
        const sb = try scratch(gpu, &bufs, c.rows, np);
        const tok_bytes = c.rows * sp.width * 4;
        const ta = try zeros(gpu, &bufs, tok_bytes);
        const tb = try zeros(gpu, &bufs, tok_bytes);
        const ca = try zeros(gpu, &bufs, c.rows * 4);
        const cb = try zeros(gpu, &bufs, c.rows * 4);
        var ms_old: f64 = 0;
        var ms_new: f64 = 0;
        for (0..4) |rep| {
            const t0 = std.Io.Timestamp.now(gpu.io, .awake);
            try sp.selectTokens(old, ops, s, dq, dw, 32, dk, .{}, dpos, c.pos, c.rows, 32, d, c.cap, sa, ta, ca);
            try s.synchronize();
            const t1 = std.Io.Timestamp.now(gpu.io, .awake);
            try sp.selectTokens(new, ops, s, dq, dw, 32, dk, .{}, dpos, c.pos, c.rows, 32, d, c.cap, sb, tb, cb);
            try s.synchronize();
            const t2 = std.Io.Timestamp.now(gpu.io, .awake);
            if (rep > 0) {
                ms_old += ms(t0, t1) / 3;
                ms_new += ms(t1, t2) / 3;
            }
        }
        const dc: Case = .{ .cap = c.cap, .pos = c.pos, .rows = c.rows };
        fails += try same(gpu, "decode scores", sa.scores, sb.scores, c.rows * np * 4, np, dc);
        fails += try same(gpu, "decode pools", sa.pools, sb.pools, c.rows * sp.topk_pools * 8, np, dc);
        fails += try same(gpu, "decode tokens", ta, tb, tok_bytes, np, dc);
        fails += try same(gpu, "decode counts", ca, cb, c.rows * 4, np, dc);
        std.debug.print("decode np {d} pos {d} rows {d}: selectTokens {d:.3} ms, with the decode select {d:.3} ms (host-timed, a launch each)\n", .{ np, c.pos, c.rows, ms_old, ms_new });
    }
    try check.expect(owner.calls > 0, "new selector must execute", .{});
    for (dpaths, 0..) |n, path| try check.expect(n > 0, "decode path {d} missing", .{path});
    std.debug.print("decode-select contract paths {any}, calls {d}\n", .{ dpaths, owner.calls });
    return fails;
}
