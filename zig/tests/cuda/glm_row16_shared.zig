//! Actual shared.begin/mixer and partial commits for eight unequal parts summing to64 rows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const H = 16;
const N = H * 128;
const D = 4096;
const L = 3;
const physical = glm.forward.window_rows;

fn floats(r: *Rig, n: usize, scale: f32) ![]f32 {
    const out = try r.a.alloc(f32, n);
    for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
    return out;
}

fn copy(d: *const cuda.Driver, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
    try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "row16 mixed D2D");
}

fn equal(r: *Rig, a: u64, b: u64, bytes: usize, digest: *std.crypto.hash.sha2.Sha256) !void {
    const got = try r.back(a, bytes);
    try check.sameBytes("row16 mixed: actual shared/private projection, saves and commit", got, try r.back(b, bytes));
    digest.update(got);
}

fn fixture(gpu: check.Gpu, m: *const glm.forward.Modules, s: cuda.Stream, comptime wide32: bool) ![32]u8 {
    const widths: []const usize = if (wide32) &.{ 32, 17, 15 } else &.{ 16, 15, 9, 8, 7, 5, 3, 1 };
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(1664);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        cuda.completion.drain(gpu.d, s, "row16 mixed Rig teardown");
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var layers: [L]glm.weights.Layer = undefined;
    var w: glm.weights.Weights = .{ .gpa = r.a, .config = .{ .layers = L, .experts = 12 }, .share = undefined };
    w.share = try glm.shard.Share.of(&w.config, 0, 4);
    w.layers = &layers;
    const kw = glm.kda_block.rowWidth(H);
    const proj = try r.dev(std.mem.sliceAsBytes(try r.bfs(kw * D, 0.015625)));
    const fb = try r.dev(std.mem.sliceAsBytes(try r.bfs(N * 128, 0.03125)));
    const gb = try r.dev(std.mem.sliceAsBytes(try r.bfs(N * 128, 0.03125)));
    const norm = try r.dev(std.mem.sliceAsBytes(try r.bfs(128, 1)));
    for (&layers) |*l| l.mixer = .{ .kda = .{
        .proj = .{ .bf16 = proj },
        .fb = fb,
        .gb = gb,
        .o = .{ .bf16 = 0 },
        .conv_w = try r.dev(std.mem.sliceAsBytes(try floats(&r, 3 * N * 4, 0.2))),
        .a_log = try r.dev(std.mem.sliceAsBytes(try floats(&r, H, 1))),
        .dt_bias = try r.dev(std.mem.sliceAsBytes(try floats(&r, N, 0.2))),
        .norm_w = norm,
        .eps = 1e-6,
        .lower = -5,
        .heads = H,
        .dims = D,
    } };
    var shared = try glm.forward.Buffers.initShared(gpu.d, &w, 64, 128);
    defer shared.deinit();
    var lone = try glm.forward.Buffers.init(gpu.d, &w, physical, false, 128);
    defer lone.deinit();
    var states: [16]glm.forward.State = undefined;
    var made: usize = 0;
    defer {
        cuda.completion.drain(gpu.d, s, "row16 mixed State/buffer teardown");
        for (states[0..made]) |*st| st.deinit();
    }
    for (&states) |*st| {
        st.* = try glm.forward.State.initFormat(r.a, gpu.d, &w, 128, true);
        made += 1;
        try st.arena.buf.fill8(0xa5, s.handle);
        try st.setMtpLen(s, 0);
    }
    var segs: [widths.len]glm.forward.Seg = undefined;
    var row0: usize = 0;
    for (widths, 0..) |rows, i| {
        const parity: u1 = @truncate(i);
        const cv = try r.dev(std.mem.sliceAsBytes(try r.bfs(L * 3 * 3 * N, 0.5)));
        const rec = try r.dev(std.mem.sliceAsBytes(try floats(&r, L * H * 128 * 128, 0.02)));
        for ([_]usize{ i, i + 8 }) |j| {
            const st = &states[j];
            @memset(st.cur, parity);
            try copy(gpu.d, s, st.conv, cv, L * 3 * 3 * N * 2);
            try copy(gpu.d, s, st.recAt(parity, 0), rec, L * st.rec_layer);
            try st.setPos(s, 19 + i);
            st.pending = rows;
            st.prompt = false;
        }
        segs[i] = .{ .st = &states[i], .row0 = row0, .rows = rows };
        row0 += rows;
    }
    try check.expect(row0 == 64 and physical >= widths[0], "row-window mixed: coherent physical/shared64", .{});
    const input = try r.bfs(64 * D, 0.5);
    const xd = try r.dev(std.mem.sliceAsBytes(input));
    try copy(gpu.d, s, shared.normed, xd, input.len * 2);
    const ranks = [_]glm.forward.Rank{.{ .w = &w, .st = &states[0], .b = &shared, .segs = &segs }};
    const f: glm.forward.Forward = .{ .d = gpu.d, .m = m, .stream = s, .ranks = &ranks, .exchange = .local };
    try glm.shared.begin(ranks[0], 64, s);
    for (layers, 0..) |layer, li| {
        try glm.shared.mixer(&f, ranks[0], layer, li, 0, 64, null);
        for (segs, 0..) |g, i| {
            const ref = &states[i + 8];
            const at = li * physical * N;
            var scratch = lone.kda;
            scratch.p = ref.proj + li * ref.proj_layer * 2;
            try copy(gpu.d, s, scratch.p, shared.kproj + g.row0 * kw * 2, g.rows * kw * 2);
            try copy(gpu.d, s, scratch.ka, shared.kda.ka + g.row0 * N * 2, g.rows * N * 2);
            try copy(gpu.d, s, scratch.kg, shared.kda.kg + g.row0 * N * 2, g.rows * N * 2);
            scratch.saves.state_out = ref.recAt(1 - ref.cur[li], li);
            scratch.saves.k = ref.save_k + at * 4;
            scratch.saves.v = ref.save_v + at * 2;
            scratch.saves.g = ref.save_g + at * 4;
            scratch.saves.b = ref.save_b + li * physical * H * 4;
            try glm.kda_block.chainRows(&m.kda, s, layer.mixer.kda, scratch, g.rows, ref.conv + li * 3 * ref.conv_c * 2, ref.recAt(ref.cur[li], li), true);
            try equal(&r, shared.kda.saves.out + g.row0 * N * 2, scratch.saves.out, g.rows * N * 2, &digest);
            try equal(&r, g.st.proj + li * g.st.proj_layer * 2, scratch.p, g.rows * kw * 2, &digest);
            for ([_][3]u64{
                .{ g.st.save_k + at * 4, scratch.saves.k, g.rows * N * 4 },
                .{ g.st.save_v + at * 2, scratch.saves.v, g.rows * N * 2 },
                .{ g.st.save_g + at * 4, scratch.saves.g, g.rows * N * 4 },
                .{ g.st.save_b + li * physical * H * 4, scratch.saves.b, g.rows * H * 4 },
            }) |v| try equal(&r, v[0], v[1], v[2], &digest);
        }
    }
    for (segs, 0..) |g, i| {
        const keep = if (i == 0) 9 else @max(@as(usize, 1), g.rows - 1);
        for ([_]usize{ i, i + 8 }) |j| {
            const own = [_]glm.forward.Rank{.{ .w = &w, .st = &states[j], .b = &lone }};
            var one = f;
            one.ranks = &own;
            try one.commit(keep);
        }
        const ref = &states[i + 8];
        try equal(&r, g.st.recAt(g.st.cur[0], 0), ref.recAt(ref.cur[0], 0), L * ref.rec_layer, &digest);
        try equal(&r, g.st.conv, ref.conv, L * 3 * ref.conv_c * 2, &digest);
        try check.expect(g.st.pos == ref.pos and g.st.pending == 0 and ref.pending == 0, "row16 mixed: actual private partial commits", .{});
    }
    return digest.finalResult();
}

fn qualify(gpu: check.Gpu, comptime wide32: bool) !void {
    var m = try glm.forward.Modules.load(gpu.d, gpu.ctx, null);
    defer m.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    defer cuda.completion.drain(gpu.d, s, "row16 mixed modules teardown");
    const first = try fixture(gpu, &m, s, wide32);
    for (0..4) |_| try check.sameBytes("row16 mixed: five-repeat output/save/commit SHA", &first, &(try fixture(gpu, &m, s, wide32)));
}

pub fn run(gpu: check.Gpu) !void {
    try qualify(gpu, false);
    check.pass("GLM_ROW16_SHARED: parts16/15/9/8/7/5/3/1 total64; actual shared/private saves/commits; repeats", .{});
}

pub fn run32(gpu: check.Gpu) !void {
    try check.expect(physical == 32, "row32 shared: physical32 required", .{});
    try qualify(gpu, true);
    check.pass("GLM_ROW32_SHARED: parts32/17/15 total64; actual begin/mixer, private saves/commits; five repeats", .{});
}
