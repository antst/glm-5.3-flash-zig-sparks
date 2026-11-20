//! Original versus TensorFold tiled decode scoring through GLM's actual selector, with and without page tables.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const fixture = @import("glm_index_check.zig");
const sparse = glm.sparse;
const dim = 128;
const pools_per_page = 128;
const key_bytes = pools_per_page * dim * 2;
const page_bytes = key_bytes + 4096;

const Arm = struct {
    route: glm.glue.Set,
    work: sparse.Scratch,
    tokens: u64,
    counts: u64,
};

fn exercise(gpu: check.Gpu, s: cuda.Stream, ops: cuda.Module, original: glm.glue.Set, updated: glm.glue.Set, c: fixture.Case) !void {
    var bufs: fixture.Bufs = .empty;
    defer {
        for (bufs.items) |*x| x.free();
        bufs.deinit(gpu.gpa);
    }
    defer cuda.completion.drain(gpu.d, s, "GLM decode scorer comparison");
    var host = std.heap.ArenaAllocator.init(gpu.gpa);
    defer host.deinit();
    const a = host.allocator();
    var rng = std.Random.DefaultPrng.init(c.pos + c.rows);
    const random = rng.random();
    const heads = c.heads;
    const query = try a.alloc(u16, c.rows * heads * dim);
    for (query) |*x| x.* = fixture.bf16(random.floatNorm(f32));
    const weights = try a.alloc(u16, c.rows * heads);
    for (weights) |*x| x.* = fixture.bf16(random.floatNorm(f32));
    const q = try fixture.fromHost(gpu, &bufs, std.mem.sliceAsBytes(query));
    const w = try fixture.fromHost(gpu, &bufs, std.mem.sliceAsBytes(weights));
    const pos_host: i32 = @intCast(c.pos);
    const pos = try fixture.fromHost(gpu, &bufs, std.mem.asBytes(&pos_host));
    const pages = c.cap / pools_per_page;
    const ids = try a.alloc(i32, pages);
    const all_keys = try a.alloc(u16, if (c.paged) pages * page_bytes / 2 else c.cap * dim);
    @memset(all_keys, 0);
    for (ids, 0..) |*id, i| id.* = @intCast(pages - 1 - i);
    for (0..pages) |i| {
        const at = if (c.paged) @as(usize, @intCast(ids[i])) * (page_bytes / 2) else i * (key_bytes / 2);
        for (all_keys[at..][0 .. key_bytes / 2]) |*x| x.* = fixture.bf16(random.float(f32) * 4 - 2);
    }
    const k = try fixture.fromHost(gpu, &bufs, std.mem.sliceAsBytes(all_keys));
    const table: glm.latent.Table = if (c.paged)
        .{ .pt = try fixture.fromHost(gpu, &bufs, std.mem.sliceAsBytes(ids)), .ps = page_bytes, .page_rows = 512, .logical = true }
    else
        .{};
    const np = sparse.poolBucket(c.pos, c.rows, c.cap);
    const nb = (sparse.scoreColumns(c.pos, c.rows, np) + 63) / 64;
    try check.expect(updated.index.?.takesDecode(c.rows, np, nb, k, table.ps), "DSA scorer fixture must select the candidate", .{});
    var arms: [2]Arm = undefined;
    for (&arms, [_]glm.glue.Set{ original, updated }) |*arm, route| arm.* = .{
        .route = route,
        .work = try fixture.scratch(gpu, &bufs, c.rows, np),
        .tokens = try fixture.zeros(gpu, &bufs, c.rows * sparse.width * 4),
        .counts = try fixture.zeros(gpu, &bufs, c.rows * 4),
    };
    var begin = try cuda.Event.init(gpu.d, true);
    defer begin.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var ms: [2][8]f64 = undefined;
    for (0..10) |repetition| for (0..2) |i| {
        const index = (i + repetition) % arms.len;
        const arm = arms[index];
        try begin.record(s);
        try sparse.selectTokens(arm.route, ops, s, q, w, heads, k, table, pos, c.pos, c.rows, heads, dim, c.cap, arm.work, arm.tokens, arm.counts);
        try end.record(s);
        try end.synchronize();
        if (repetition >= 2) ms[index][repetition - 2] = try cuda.Event.elapsedMs(begin, end);
    };
    var mismatches: usize = 0;
    mismatches += try fixture.same(gpu, "scores", arms[0].work.scores, arms[1].work.scores, c.rows * np * 4, np, c);
    mismatches += try fixture.same(gpu, "selected pools", arms[0].work.pools, arms[1].work.pools, c.rows * sparse.topk_pools * 8, np, c);
    mismatches += try fixture.same(gpu, "tokens", arms[0].tokens, arms[1].tokens, c.rows * sparse.width * 4, np, c);
    mismatches += try fixture.same(gpu, "counts", arms[0].counts, arms[1].counts, c.rows * 4, np, c);
    try check.expect(mismatches == 0, "GLM tiled decode score path changed output", .{});
    const counts = try a.alloc(i32, c.rows);
    try cuda.memory.download(gpu.d, arms[1].counts, std.mem.sliceAsBytes(counts));
    for (counts) |n| try check.expect(n > 0 and n <= sparse.width, "DSA selector returned no valid token list", .{});
    std.debug.print("GLM_DECODE_SCORE_CASE pos={d} rows={d} blocks={d} paged={any} old_ms={d:.4} new_ms={d:.4} exact=true\n", .{ c.pos, c.rows, nb, c.paged, check.median(&ms[0]), check.median(&ms[1]) });
}

pub fn run(gpu: check.Gpu) !void {
    var glue = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue.unload();
    const kernels = try glm.glue.Kernels.load(glue);
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    var index = try glm.index.Kernels.load(gpu.d, cuda.kernels.glm_index, .both);
    defer index.unload();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    defer cuda.completion.drain(gpu.d, stream, "GLM decode scorer modules");
    const original: glm.glue.Set = .{ .glue = &kernels, .index = &index };
    var updated = original;
    updated.dscore = true;
    const cases = [_]fixture.Case{
        .{ .cap = 8192, .pos = 16384, .rows = 8, .paged = true },
        .{ .cap = 132096, .pos = 524288, .rows = 4 },
        .{ .cap = 262144, .pos = 1040384, .rows = 8 },
        .{ .cap = 262144, .pos = 1040384, .rows = 32, .paged = true },
    };
    for (cases) |c| try exercise(gpu, stream, ops, original, updated, c);
    check.pass("GLM_DSA_DECODE_SCORES: four decode geometries, contiguous and shuffled pages, exact", .{});
}
