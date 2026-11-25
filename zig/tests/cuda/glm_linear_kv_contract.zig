//! The original FP8 write and attention kernels consume remapped CUDA virtual addresses.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const lw = glm.linear_latent.row_bytes;

fn position(r: *Rig, pos: usize) !u64 {
    const p: i32 = @intCast(pos);
    return r.dev(std.mem.asBytes(&p));
}

fn partials(r: *Rig, nch: usize, rows: usize, heads: usize) !glm.latent.Partials {
    return .{
        .po = try r.zeros(nch * rows * heads * lw * 4),
        .pm = try r.zeros(nch * rows * heads * 4),
        .pl = try r.zeros(nch * rows * heads * 4),
    };
}

fn fixture(gpu: check.Gpu, s: cuda.Stream, set: glm.glue.Set, pool: *cuda.linear_pages.Pool, heads: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(5375 + heads);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const boundary = pool.page_bytes / lw;
    if (boundary < 16 or boundary > 32_768) return error.LinearKvTestGeometry;
    const rows: usize = if (set.prompt) glm.latent.prompt_rows else 8;
    const cut = boundary - 3;
    const keys = cut + rows;
    var origin = try glm.linear_latent.View.init(pool, keys);
    defer origin.deinit() catch {};
    var branch = try glm.linear_latent.View.init(pool, keys);
    defer branch.deinit() catch {};
    defer cuda.completion.drain(gpu.d, s, "GLM linear KV test teardown");

    const input = try r.bfs(keys * lw, 1);
    const data = try r.dev(std.mem.sliceAsBytes(input));
    const cache = try r.zeros(try glm.latent.cacheBytes(keys, lw, true));
    const scales = cache + keys * lw;
    const zero = try position(&r, 0);
    const at = try position(&r, cut);

    try origin.prepareWrite(0, cut);
    try glm.latent.latentWriteFormat(set, s, data, lw, origin.codes.base, origin.scales.base, .{}, zero, cut, lw);
    try glm.latent.latentWriteFormat(set, s, data, lw, cache, scales, .{}, zero, keys, lw);
    try s.synchronize();
    try branch.borrow(&origin, cut);
    try branch.prepareWrite(cut, keys);
    try glm.latent.latentWriteFormat(
        set,
        s,
        data + cut * lw * 2,
        lw,
        branch.codes.base,
        branch.scales.base,
        .{},
        at,
        rows,
        lw,
    );
    try check.sameBytes("linear KV codes", try r.back(branch.codes.base, keys * lw), try r.back(cache, keys * lw));
    try check.sameBytes("linear KV scales", try r.back(branch.scales.base, keys * 4), try r.back(scales, keys * 4));
    try check.sameBytes("linear KV shared prefix", try r.back(origin.codes.base, cut * lw), try r.back(cache, cut * lw));

    const q = try r.bfs(rows * heads * lw, 1);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const output_bytes = rows * heads * lw * 2;
    const nch = glm.latent.chunksFor(keys);
    const dense = try partials(&r, if (set.prompt) 1 else nch, rows, heads);
    const sparse = try partials(&r, 1, rows, heads);
    const ref = try r.zeros(output_bytes);
    const got = try r.zeros(output_bytes);
    const tag = try r.zeros(4);
    if (!set.prompt) {
        try glm.latent.attentionFormat(set, s, dq, cache, scales, .{}, at, dense, ref, tag, rows, heads, lw, nch, 0.125, 16);
        try glm.latent.attentionFormat(set, s, dq, branch.codes.base, branch.scales.base, .{}, at, dense, got, tag, rows, heads, lw, nch, 0.125, 16);
        const dense_bytes = try r.back(got, output_bytes);
        try written(dense_bytes);
        try check.sameBytes("linear KV dense attention", dense_bytes, try r.back(ref, output_bytes));
    }

    const width: usize = if (set.prompt) 2048 else 8;
    const tokens = try r.a.alloc(i32, rows * width);
    const counts = try r.a.alloc(i32, rows);
    @memset(counts, @intCast(width));
    for (0..rows) |i| {
        const last = cut + i;
        for (tokens[i * width ..][0..width], 0..) |*token, j| token.* = @intCast(j * last / (width - 1));
    }
    const dt = try r.dev(std.mem.sliceAsBytes(tokens));
    const dc = try r.dev(std.mem.sliceAsBytes(counts));
    const ref_sparse = try r.zeros(output_bytes);
    const got_sparse = try r.zeros(output_bytes);
    try glm.latent.sparseAttentionFormat(set, s, dq, cache, scales, .{}, dt, dc, width, sparse, ref_sparse, rows, heads, lw, 0.125);
    try glm.latent.sparseAttentionFormat(set, s, dq, branch.codes.base, branch.scales.base, .{}, dt, dc, width, sparse, got_sparse, rows, heads, lw, 0.125);
    const sparse_bytes = try r.back(got_sparse, output_bytes);
    try written(sparse_bytes);
    try check.sameBytes("linear KV sparse attention", sparse_bytes, try r.back(ref_sparse, output_bytes));
    const normal: Kv = .{ .codes = cache, .scales = scales };
    const paged: Kv = .{ .codes = branch.codes.base, .scales = branch.scales.base };
    var work: Work = .{ .s = s, .set = set, .q = dq, .pos = at, .tokens = dt, .counts = dc, .rows = rows, .heads = heads, .width = width, .parts = sparse, .nch = nch, .out = got_sparse, .tag = tag };
    try work.timing(normal, paged);
    if (!set.prompt) {
        work.width = 0;
        work.parts = dense;
        work.out = got;
        try work.timing(normal, paged);
    }
}

const Kv = struct { codes: u64, scales: u64 };
const Work = struct {
    s: cuda.Stream,
    set: glm.glue.Set,
    q: u64,
    pos: u64,
    tokens: u64,
    counts: u64,
    rows: usize,
    heads: usize,
    width: usize,
    parts: glm.latent.Partials,
    nch: usize,
    out: u64,
    tag: u64,

    fn launch(w: Work, kv: Kv) !void {
        if (w.width == 0)
            return glm.latent.attentionFormat(w.set, w.s, w.q, kv.codes, kv.scales, .{}, w.pos, w.parts, w.out, w.tag, w.rows, w.heads, lw, w.nch, 0.125, 16);
        return glm.latent.sparseAttentionFormat(w.set, w.s, w.q, kv.codes, kv.scales, .{}, w.tokens, w.counts, w.width, w.parts, w.out, w.rows, w.heads, lw, 0.125);
    }

    fn timing(w: Work, normal: Kv, paged: Kv) !void {
        var start = try cuda.Event.init(w.s.d, true);
        defer start.deinit();
        var end = try cuda.Event.init(w.s.d, true);
        defer end.deinit();
        for (0..3) |_| {
            try w.launch(normal);
            try w.launch(paged);
        }
        var ns: [2][9]f32 = undefined;
        const cases = [_]Kv{ normal, paged };
        for (0..9) |i| for (0..2) |j| {
            const k = (i + j) % 2;
            try start.record(w.s);
            for (0..5) |_| try w.launch(cases[k]);
            try end.record(w.s);
            try end.synchronize();
            ns[k][i] = try start.elapsedMs(end) * (1000.0 / 5.0);
        };
        for (&ns) |*samples| std.mem.sort(f32, samples, {}, comptime std.sort.asc(f32));
        const label = if (w.set.prompt) "compact-prefill" else if (w.width == 0) "dense-decode" else "sparse-decode";
        std.debug.print("LINEAR_KV_TIMING case={s} H={d} R={d} contiguous_us={d:.3} mapped_us={d:.3}\n", .{ label, w.heads, w.rows, ns[0][4], ns[1][4] });
    }
};

fn written(raw: []const u8) !void {
    var i: usize = 0;
    while (i < raw.len) : (i += 2) {
        const bits = std.mem.readInt(u16, raw[i..][0..2], .little);
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        try check.expect(std.math.isFinite(value), "linear KV attention output word {d} is unwritten or nonfinite", .{i / 2});
    }
}

pub fn run(gpu: check.Gpu) !void {
    return testWindows(gpu, null, false);
}

pub fn testWindows(gpu: check.Gpu, paths: ?[3][]const u8, long: bool) !void {
    var vmm = try cuda.arena.Vmm.open(gpu.d, gpu.ctx.device, if (long) 4 << 30 else 1 << 30);
    defer vmm.close();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var linear: cuda.arena.Vmm.Linear = .{ .vmm = &vmm, .stream = s };
    const extent = if (long) @max(8 << 20, vmm.granularity) else vmm.granularity;
    var pool = try cuda.linear_pages.Pool.init(gpu.gpa, linear.backend(), extent, if (long) 80 else 12);
    defer pool.deinit() catch {};
    var module = if (paths) |files|
        try glm.prompt_image.load(gpu.gpa, gpu.io, gpu.d, files[0], "383422501cd782e8cc30851b7da2e49e20cb2a2c25959a688b94b3629521c3fd")
    else
        try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer module.unload();
    const kernels = try glm.glue.Kernels.load(module);
    for ([_]usize{ 16, 32 }) |heads| try runFixture(gpu, s, .{ .glue = &kernels }, &pool, heads, long);
    if (paths) |files| {
        const hs = [_]usize{ 16, 32 };
        const names = [_][:0]const u8{ "tf_glm_sparse_prompt_fp8_warp_key_prefetch_lines", "tf_glm_sparse_prompt_fp8_warp_key_prefetch_lines_h32" };
        const shas = [_][]const u8{ "0671c791834fef7b9efcaf0cf2cb7fca1912790c26b6a2e78f029c8e54b026c5", "d8a23632ad033d1d4a0be4fbe5536f1d484cbfa3d7cb215c8716cd15103e52d8" };
        for (hs, names, shas, files[1..]) |heads, name, sha, path| {
            var compact = try glm.prompt_image.load(gpu.gpa, gpu.io, gpu.d, path, sha);
            defer compact.unload();
            const f = try compact.function(name);
            try f.allowDynamicShared(glm.glue.compact_sparse_smem);
            var k = kernels;
            k.sparse_prompt_fp8 = .{ .function = f, .heads = @intCast(heads) };
            try runFixture(gpu, s, .{ .glue = &k, .prompt = true }, &pool, heads, long);
        }
    }
    if (long) {
        check.pass("GLM_LINEAR_KV_LONG: sampled FP8 writes and sparse H16/H32 outputs match exactly", .{});
    } else {
        check.pass("GLM_LINEAR_KV: FP8 write and dense/sparse attention exact on remapped H16 and H32", .{});
    }
}

fn runFixture(gpu: check.Gpu, s: cuda.Stream, set: glm.glue.Set, pool: *cuda.linear_pages.Pool, heads: usize, long: bool) !void {
    if (!long) return fixture(gpu, s, set, pool, heads);
    for ([_]usize{ 299_992, 1_047_696 }) |keys| try longSparse(gpu, s, set, pool, heads, keys);
}

fn longSparse(gpu: check.Gpu, s: cuda.Stream, set: glm.glue.Set, pool: *cuda.linear_pages.Pool, heads: usize, keys: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(112071 + heads);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mapped = try glm.linear_latent.View.init(pool, keys);
    defer mapped.deinit() catch {};
    defer cuda.completion.drain(gpu.d, s, "GLM long linear KV test teardown");

    const cache = try r.zeros(try glm.latent.cacheBytes(keys, lw, true));
    const normal: Kv = .{ .codes = cache, .scales = cache + keys * lw };
    const paged: Kv = .{ .codes = mapped.codes.base, .scales = mapped.scales.base };
    try mapped.prepareWrite(0, keys);
    // Repeat a generated block to bound fixture RAM while initializing every key.
    const write_rows = 8_192;
    const input = try r.bfs(write_rows * lw, 1);
    const data = try r.dev(std.mem.sliceAsBytes(input));
    const offsets = try r.a.alloc(i32, std.math.divCeil(usize, keys, write_rows) catch unreachable);
    for (offsets, 0..) |*value, i| value.* = @intCast(i * write_rows);
    const positions = try r.dev(std.mem.sliceAsBytes(offsets));
    for (offsets, 0..) |offset, i| {
        const pos = positions + i * @sizeOf(i32);
        const count = @min(write_rows, keys - @as(usize, @intCast(offset)));
        for ([_]Kv{ normal, paged }) |kv|
            try glm.latent.latentWriteFormat(set, s, data, lw, kv.codes, kv.scales, .{}, pos, count, lw);
    }

    const width = 2_048;
    const rows: usize = if (set.prompt) glm.latent.prompt_rows else 8;
    const tokens = try r.a.alloc(i32, rows * width);
    const counts = try r.a.alloc(i32, rows);
    @memset(counts, width);
    for (0..rows) |i| {
        const last = keys - rows + i;
        for (tokens[i * width ..][0..width], 0..) |*value, j| {
            const low = j * last / width;
            const high = (j + 1) * last / width;
            value.* = @intCast(r.rng.intRangeAtMost(usize, low, high - 1));
        }
    }
    for ([_]usize{ 0, pool.page_bytes / lw - 1, pool.page_bytes / lw, keys - 1 }) |row| {
        try check.sameBytes("long linear KV codes", try r.back(normal.codes + row * lw, lw), try r.back(paged.codes + row * lw, lw));
        try check.sameBytes("long linear KV scales", try r.back(normal.scales + row * 4, 4), try r.back(paged.scales + row * 4, 4));
    }

    const output_bytes = rows * heads * lw * 2;
    const q = try r.bfs(rows * heads * lw, 1);
    var work: Work = .{
        .s = s,
        .set = set,
        .q = try r.dev(std.mem.sliceAsBytes(q)),
        .pos = 0,
        .tokens = try r.dev(std.mem.sliceAsBytes(tokens)),
        .counts = try r.dev(std.mem.sliceAsBytes(counts)),
        .rows = rows,
        .heads = heads,
        .width = width,
        .parts = try partials(&r, if (set.prompt) 1 else glm.latent.chunksFor(width), rows, heads),
        .nch = 0,
        .out = try r.zeros(output_bytes),
        .tag = 0,
    };
    try work.launch(normal);
    const ref = try r.back(work.out, output_bytes);
    work.out = try r.zeros(output_bytes);
    try work.launch(paged);
    const got = try r.back(work.out, output_bytes);
    try written(got);
    try check.sameBytes("long linear KV sparse attention", ref, got);
    std.debug.print("LINEAR_KV_SPARSE_CONTEXT {d} extent={d}\n", .{ keys, pool.page_bytes });
    try work.timing(normal, paged);
    try mapped.codes.clear();
    try mapped.scales.clear();
    try pool.trim();
}
