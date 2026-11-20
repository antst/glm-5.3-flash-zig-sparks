//! Byte comparison of the paged CUDA FP8 cache format with the contiguous format on the same device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;
const latent = glm.latent;
const width = 512;
const heads = 32;
const page_rows = 512;
const codes_per_page = page_rows * width;
const scales_per_page = page_rows * @sizeOf(f32);
const pitch = codes_per_page + scales_per_page + 4096;
const page_ids = [_]i32{ 3, 0, 4 };

fn position(r: *Rig, n: usize) !u64 {
    const pos: i32 = @intCast(n);
    return r.dev(std.mem.asBytes(&pos));
}

fn partials(r: *Rig, chunks: usize, rows: usize) !latent.Partials {
    const n = chunks * rows * heads;
    return .{ .po = try r.zeros(n * width * 4), .pm = try r.zeros(n * 4), .pl = try r.zeros(n * 4) };
}

fn checkOutput(r: *Rig, plain: u64, paged: u64, rows: usize, masked: bool) !void {
    const row_bytes = heads * width * 2;
    const a = try r.back(plain, rows * row_bytes);
    try check.sameBytes("paged FP8 attention", try r.back(paged, a.len), a);
    for (0..rows) |row| {
        const data = a[row * row_bytes ..][0..row_bytes];
        if (masked and row == 1) {
            for (data) |b| try check.expect(b == 0xff, "masked attention wrote output", .{});
        } else {
            var at: usize = 0;
            while (at < data.len) : (at += 2) {
                const bits = std.mem.readInt(u16, data[at..][0..2], .little);
                const f: f32 = @bitCast(@as(u32, bits) << 16);
                try check.expect(std.math.isFinite(f), "attention output is unwritten or non-finite", .{});
            }
        }
    }
}

fn exercise(r: *Rig, set: glm.glue.Set, pos: usize) !void {
    const rows = 4;
    const keys = pos + rows;
    const stride = width + 8;
    const input = try r.dev(std.mem.sliceAsBytes(try r.bfs(keys * stride, 1)));
    const flat = try r.zeros(try latent.cacheBytes(keys, width, true));
    const scales = flat + keys * width;
    const paged = try r.zeros(5 * pitch);
    const ps = paged + codes_per_page;
    const t: latent.Table = .{ .pt = try r.dev(std.mem.asBytes(&page_ids)), .ps = pitch, .page_rows = page_rows, .logical = true };
    try latent.latentWriteFormat(set, r.s, input, stride, flat, scales, .{}, try position(r, 0), keys, width);
    var at: usize = 0;
    while (at < keys) {
        const n = @min(keys - at, 173);
        try latent.latentWriteFormat(set, r.s, input + at * stride * 2, stride, paged, ps, t, try position(r, at), n, width);
        at += n;
    }
    const contiguous = try r.back(flat, keys * width);
    const contiguous_scales = try r.back(scales, keys * 4);
    for (0..5) |physical| {
        const logical = std.mem.indexOfScalar(i32, &page_ids, @intCast(physical));
        const first = if (logical) |i| i * page_rows else keys;
        const count: usize = if (first < keys) @min(page_rows, keys - first) else 0;
        const raw = try r.back(paged + physical * pitch, pitch);
        if (count != 0) {
            try check.sameBytes("paged FP8 codes", raw[0 .. count * width], contiguous[first * width ..][0 .. count * width]);
            try check.sameBytes("paged FP8 scales", raw[codes_per_page..][0 .. count * 4], contiguous_scales[first * 4 ..][0 .. count * 4]);
        }
        for (raw[count * width .. codes_per_page]) |b| try check.expect(b == 0xff, "FP8 codes touched an unused row", .{});
        for (raw[codes_per_page + count * 4 ..]) |b| try check.expect(b == 0xff, "FP8 scales touched an unused row or guard", .{});
    }
    const q = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * heads * width, 1)));
    const bytes = rows * heads * width * 2;
    const nch = latent.chunksFor(keys);
    const dpos = try position(r, pos);
    for ([_]usize{ 16, 32 }) |block| {
        const a = try r.zeros(bytes);
        const b = try r.zeros(bytes);
        try latent.attentionFormat(set, r.s, q, flat, scales, .{}, dpos, try partials(r, nch, rows), a, try r.zeros(4), rows, heads, width, nch, 0.125, block);
        try latent.attentionFormat(set, r.s, q, paged, ps, t, dpos, try partials(r, nch, rows), b, try r.zeros(4), rows, heads, width, nch, 0.125, block);
        try checkOutput(r, a, b, rows, false);
    }
    const selected = 523;
    const counts = [_]i32{ selected, 0, selected, 7 };
    var tokens: [rows * selected]i32 = @splat(-1);
    for (counts, 0..) |count, row| for (0..@intCast(count)) |j| {
        tokens[row * selected + j] = @intCast((j * 89 + row * 173) % keys);
    };
    const tok = try r.dev(std.mem.asBytes(&tokens));
    const cnt = try r.dev(std.mem.asBytes(&counts));
    const a = try r.zeros(bytes);
    const b = try r.zeros(bytes);
    const sparse_chunks = latent.chunksFor(selected);
    try latent.sparseAttentionFormat(set, r.s, q, flat, scales, .{}, tok, cnt, selected, try partials(r, sparse_chunks, rows), a, rows, heads, width, 0.125);
    try latent.sparseAttentionFormat(set, r.s, q, paged, ps, t, tok, cnt, selected, try partials(r, sparse_chunks, rows), b, rows, heads, width, 0.125);
    try checkOutput(r, a, b, rows, true);
    try check.sameBytes("FP8 table immutable", try r.back(t.pt, @sizeOf(@TypeOf(page_ids))), std.mem.asBytes(&page_ids));
}

pub fn run(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer module.unload();
    const k = try glm.glue.Kernels.load(module);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(117);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    defer cuda.completion.drain(gpu.d, s, "paged FP8 fixture teardown");
    for ([_]usize{ 510, 1022 }) |pos| try exercise(&r, .{ .glue = &k }, pos);
    check.pass("GLM_PAGED_FP8: shuffled pages, codes/scales/guards and 16/32-head dense plus sparse attention bytes", .{});
}
