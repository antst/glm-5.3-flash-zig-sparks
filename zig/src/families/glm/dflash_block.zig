//! DFlash2's device passes: the context ingest (taps through fc into the rings) and the block (embed to candidates).
const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const df = @import("dflash.zig");
const Drafter = df.Drafter;
const D = df.D;
const BLOCK = df.BLOCK;

fn go(dr: *const Drafter, f: cuda.Function, grid: cuda.Dim3, threads: u32, e: anytype, a: *cuda.Args) !void {
    try cuda_pdl.launchIf(dr.pdl, f, .{ .grid = grid, .block = .{ .x = threads } }, e.stream, a);
}

fn set(e: anytype) glue.Set {
    return .{ .glue = &e.m.glue, .mm = &e.m.mm };
}

/// out[m,n] = x[m,k] . w^T; bf16 or explicit FP32 partials, Q4 in at most16-row launches, otherwise glm_mm.
fn lin(dr: *const Drafter, e: anytype, x: u64, w: df.Lin, out: u64, m: usize, wide: bool) !void {
    return linWithSums(dr, e, x, w, out, m, wide, null);
}

fn linWithSums(dr: *const Drafter, e: anytype, x: u64, w: df.Lin, out: u64, m: usize, wide: bool, supplied: ?u64) !void {
    switch (w) {
        .bf16 => |t| try mm.matmul(&e.m.mm, e.stream, x, t.k, t, out, dr.part, m, wide, false),
        .q4 => |q| {
            const sums = supplied orelse dr.gs;
            if (supplied == null) {
                var a: cuda.Args = .{};
                a.add(x);
                a.add(@as(i32, @intCast(q.k)));
                a.add(@as(i32, @intCast(q.k)));
                a.add(sums);
                try go(dr, dr.k.gsum, .{ .x = @intCast(m) }, 128, e, &a);
            }
            var r0: usize = 0;
            while (r0 < m) : (r0 += 64) {
                const rows = @min(64, m - r0);
                const at = out + r0 * q.n * @as(usize, if (wide) 4 else 2);
                try q4.matmulRows(&e.m.q4, e.stream, x + r0 * q.k * 2, q.k, sums + r0 * (q.k / 64) * 4, q, at, rows, wide);
            }
        },
    }
}

fn linB(dr: *const Drafter, e: anytype, x: u64, w: mm.Tiled, out: u64, m: usize, wide: bool) !void {
    try lin(dr, e, x, .{ .bf16 = w }, out, m, wide);
}

pub fn copy(e: anytype, dst: u64, src: u64, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
}

pub fn upload(dr: *const Drafter, e: anytype, dst: u64, bytes: []const u8) !void {
    _ = dr;
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, e.stream.handle), "cuMemcpyHtoDAsync");
}

/// After the pass: the stream idle, a failed RoCE gather's error raised, `src` read into `dst`.
pub fn download(dr: *const Drafter, e: anytype, dst: []u8, src: u64) !void {
    _ = dr;
    try e.stream.synchronize();
    if (e.group) |g| try g.check();
    try cuda.memory.download(e.d, src, std.mem.sliceAsBytes(dst)[0..dst.len]);
}

/// Every rank's `bytes` at buf + rank * bytes into every rank's buf, rank order (the engine's gathers' route).
fn gather(e: anytype, buf: u64, bytes: usize) !void {
    if (e.spans.len == 1) return;
    if (e.group) |g| return g.allGather(e.stream, buf + e.share.rank * bytes, buf, bytes);
    try e.lib.check(e.lib.api.ncclAllGather(buf + e.share.rank * bytes, buf, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
}

/// `N` context rows (cin, cpos, crings uploaded) through fc, the hidden norm and each layer's k, v into the rings.
pub fn ingestPass(dr: *Drafter, e: anytype, N: usize) !void {
    try lin(dr, e, dr.cin, dr.fc, dr.c1, N, false);
    try glue.rmsnorm(set(e), e.stream, dr.c1, D, dr.hnorm, dr.ctx, D, null, df.EPS, N, D);
    for (dr.layers, 0..) |l, li| {
        try lin(dr, e, dr.ctx, l.kv, dr.ckv, N, false);
        var a: cuda.Args = .{};
        a.add(dr.ckv);
        a.add(l.kn);
        a.add(dr.cpos);
        a.add(dr.crings);
        a.add(@as(i32, @intCast(li)));
        a.add(df.EPS);
        a.add(df.THETA);
        try go(dr, dr.k.ctx_kv, .{ .x = @intCast(N), .y = 8 }, 64, e, &a);
    }
}

fn conv(dr: *const Drafter, e: anytype, z: u64, which: i32, base: u64, out: u64, add: bool, M: usize) !void {
    var a: cuda.Args = .{};
    a.add(z);
    a.add(dr.dyn);
    a.add(which);
    a.add(base);
    a.add(out);
    a.add(@as(i32, @intFromBool(add)));
    try go(dr, dr.k.conv, .{ .x = @intCast(M), .y = D / 1024 }, 256, e, &a);
}

/// The block pass for `S` streams (tables uploaded): embed, 5 layers, norm, each rank's 16 best a row, gathered.
pub fn pass(dr: *Drafter, e: anytype, S: usize, rank0: bool) !void {
    try compute(dr, e, S);
    try finish(dr, e, S, rank0);
}

/// Compute through this rank's candidates; unsplit MLPs make this body free of communication and host reads.
pub fn compute(dr: *Drafter, e: anytype, S: usize) !void {
    try embed(dr, e, S);
    for (dr.layers, 0..) |_, li| {
        try beforeAttention(dr, e, S, li);
        if (dr.attention != null) try gatherLayer(dr, e, S);
        try afterAttention(dr, e, S, li);
        try beforeMlp(dr, e, S, li);
        if (dr.split) try gatherLayer(dr, e, S);
        try afterGather(dr, e, S, li);
    }
    try candidates(dr, e, S);
}

fn embed(dr: *Drafter, e: anytype, S: usize) !void {
    const M = S * BLOCK;
    try glue.embedB16(set(e), e.stream, dr.ids, e.w.embed, dr.x, M, D, 1);
}

fn beforeGather(dr: *Drafter, e: anytype, S: usize, li: usize) !void {
    if (dr.attention != null) return error.BadDrafterGraphStage;
    try beforeAttention(dr, e, S, li);
    try afterAttention(dr, e, S, li);
    try beforeMlp(dr, e, S, li);
}

fn beforeAttention(dr: *Drafter, e: anytype, S: usize, li: usize) !void {
    const M = S * BLOCK;
    const l = dr.layers[li];
    const local = dr.attention;
    const qh = if (local) |a| a.plan.q_heads else df.QH;
    const kvh = if (local) |a| a.plan.kv_heads else 8;
    const qkv_width = if (local) |a| a.plan.fusedWidth() else df.QKV;
    // attention, wrapped in the attention conv
    try glue.rmsnorm(set(e), e.stream, dr.x, D, l.in_norm, dr.y, D, null, df.EPS, M, D);
    try lin(dr, e, dr.y, l.a_proj, dr.dyn, M, false);
    try conv(dr, e, dr.y, 0, l.a_base, dr.y2, false, M);
    try lin(dr, e, dr.y2, l.qkv, dr.qkv, M, false);
    {
        var a: cuda.Args = .{};
        a.add(dr.qkv);
        a.add(@as(i32, @intCast(qkv_width)));
        a.add(l.qn);
        a.add(l.kn);
        a.add(dr.bpos);
        a.add(df.EPS);
        a.add(df.THETA);
        a.add(dr.q);
        a.add(dr.kb);
        a.add(dr.vb);
        try go(dr, if (local) |v| v.kernels.block_qkv else dr.k.block_qkv, .{ .x = @intCast(M), .y = @intCast(qh + kvh) }, 64, e, &a);
    }
    if (dr.python_attention) |p| {
        try p.run(dr, e, S, li);
    } else {
        var a: cuda.Args = .{};
        for ([_]u64{ dr.q, dr.kb, dr.vb, dr.rings, dr.starts, dr.froms }) |p| a.add(p);
        a.add(@as(i32, @intCast(li)));
        a.add(df.SCALE);
        for ([_]u64{ dr.pm, dr.pl, dr.pacc }) |p| a.add(p);
        if (local) |v| a.add(@as(i32, @intCast(v.plan.kv0 / df.HD)));
        try go(dr, if (local) |v| v.kernels.attn_part else dr.k.attn_part, .{ .x = @intCast(S), .y = @intCast(kvh), .z = df.CH }, 256, e, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ dr.pm, dr.pl, dr.pacc, dr.att }) |p| b.add(p);
        try go(dr, if (local) |v| v.kernels.attn_merge else dr.k.attn_merge, .{ .x = @intCast(M), .y = @intCast(qh) }, df.HD, e, &b);
    }
    if (local != null) {
        const bytes = M * D * 4;
        try lin(dr, e, dr.att, l.o, dr.sums + e.share.rank * bytes, M, true);
    } else try lin(dr, e, dr.att, l.o, dr.a, M, false);
}

fn afterAttention(dr: *Drafter, e: anytype, S: usize, li: usize) !void {
    if (dr.attention != null) try sum(dr, e, S);
    try conv(dr, e, dr.a, 1, dr.layers[li].a_base + 2 * D * 2, dr.x, true, S * BLOCK);
}

fn beforeMlp(dr: *Drafter, e: anytype, S: usize, li: usize) !void {
    const M = S * BLOCK;
    const l = dr.layers[li];
    // the MLP, wrapped in the MLP conv
    try glue.rmsnorm(set(e), e.stream, dr.x, D, l.post_norm, dr.y, D, null, df.EPS, M, D);
    try lin(dr, e, dr.y, l.m_proj, dr.dyn, M, false);
    try conv(dr, e, dr.y, 0, l.m_base, dr.y2, false, M);
    try lin(dr, e, dr.y2, l.gu, dr.gu, M, false);
    const supplied: ?u64 = if (dr.compat & 1 != 0) dr.gs else null;
    if (supplied) |sums| {
        try glue.swiglu(set(e), e.stream, dr.gu, dr.act, sums, 1e30, M, dr.ff);
    } else {
        var a: cuda.Args = .{};
        a.add(dr.gu);
        a.add(dr.act);
        a.add(@as(i32, @intCast(dr.ff)));
        try go(dr, dr.k.swiglu, .{ .x = @intCast(M), .y = @intCast((dr.ff + 1023) / 1024) }, 256, e, &a);
    }
    if (dr.split) {
        const bytes = M * D * 4;
        try linWithSums(dr, e, dr.act, l.down, dr.sums + e.share.rank * bytes, M, true, supplied);
    } else try linWithSums(dr, e, dr.act, l.down, dr.a, M, false, supplied);
}

pub fn gatherLayer(dr: *Drafter, e: anytype, S: usize) !void {
    try gather(e, dr.sums, S * BLOCK * D * 4);
}

fn afterGather(dr: *Drafter, e: anytype, S: usize, li: usize) !void {
    const M = S * BLOCK;
    const l = dr.layers[li];
    if (dr.split) try sum(dr, e, S);
    try conv(dr, e, dr.a, 1, l.m_base + 2 * D * 2, dr.x, true, M);
}

fn sum(dr: *Drafter, e: anytype, S: usize) !void {
    const M = S * BLOCK;
    var a: cuda.Args = .{};
    a.add(dr.sums);
    a.add(dr.a);
    a.add(@as(i32, @intCast(M)));
    a.add(@as(i32, @intCast(dr.world)));
    try go(dr, dr.k.sumn, .{ .x = @intCast(M), .y = D / 1024 }, 256, e, &a);
}

fn candidates(dr: *Drafter, e: anytype, S: usize) !void {
    const M = S * BLOCK;
    // every row through the drafts' 4-bit head (this rank's vocabulary), its 16 best, gathered; rank 0's projections
    try glue.rmsnorm(set(e), e.stream, dr.x, D, dr.norm, dr.hn, D, dr.xs, df.EPS, M, D);
    const head = e.w.draftHead(@import("cuda_dflash_head_mode.zig").omitted(e)).?;
    var r0: usize = 0;
    while (r0 < M) : (r0 += 64) {
        const rows = @min(64, M - r0);
        try q4.matmulRows(&e.m.q4, e.stream, dr.hn + r0 * D * 2, D, dr.xs + r0 * (D / 64) * 4, head, dr.logits + r0 * head.n * 2, rows, false);
    }
    const chunks = (head.n + 4095) / 4096;
    const nw = M * df.CW;
    {
        var a: cuda.Args = .{};
        a.add(dr.logits);
        a.add(@as(i32, @intCast(head.n)));
        a.add(@as(i32, @intCast(e.share.vocab.lo)));
        a.add(dr.pv);
        a.add(dr.pi);
        try go(dr, dr.k.top16_part, .{ .x = @intCast(M), .y = @intCast(chunks) }, 256, e, &a);
        var b: cuda.Args = .{};
        b.add(dr.pv);
        b.add(dr.pi);
        b.add(@as(i32, @intCast(chunks)));
        b.add(dr.gtk + e.share.rank * nw * 4);
        try go(dr, dr.k.top16_merge, .{ .x = @intCast(M) }, 256, e, &b);
    }
}

/// Six compute stages surround the five eager MLP gathers; no communication is recorded in these bodies.
pub fn computeStage(dr: *Drafter, e: anytype, S: usize, stage: usize) !void {
    if (!dr.split or dr.layers.len != 5 or stage > 5) return error.BadDrafterGraphStage;
    if (stage == 0) try embed(dr, e, S) else try afterGather(dr, e, S, stage - 1);
    if (stage < dr.layers.len) try beforeGather(dr, e, S, stage) else try candidates(dr, e, S);
}

/// Candidate communication and rank 0's projection keep their eager order after the compute body.
pub fn finish(dr: *Drafter, e: anytype, S: usize, rank0: bool) !void {
    const M = S * BLOCK;
    try gather(e, dr.gtk, M * df.CW * 4);
    if (rank0) try linB(dr, e, dr.hn, dr.hproj, dr.g, M, dr.compat & 2 == 0);
}

/// Eleven pure compute stages surround alternating O/down gathers for the TP4 attention-plus-MLP path.
pub fn computeAttentionStage(dr: *Drafter, e: anytype, S: usize, stage: usize) !void {
    if (dr.attention == null or !dr.split or stage > 10) return error.BadDrafterGraphStage;
    if (stage == 0) {
        try embed(dr, e, S);
        return beforeAttention(dr, e, S, 0);
    }
    const previous = (stage - 1) / 2;
    if (stage % 2 == 1) {
        try afterAttention(dr, e, S, previous);
        return beforeMlp(dr, e, S, previous);
    }
    try afterGather(dr, e, S, previous);
    if (stage < 10) try beforeAttention(dr, e, S, stage / 2) else try candidates(dr, e, S);
}
