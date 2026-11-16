//! Original BF16/FP8 dense and sparse attention at row16, exact reconstructed caches and row-alone bytes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Rig = qr.RigT;
const H = 16;
const LW = 512;
const guard_rows = 7;

fn position(r: *Rig, value: usize) !u64 {
    const p: i32 = @intCast(value);
    return r.dev(std.mem.asBytes(&p));
}

fn partials(r: *Rig, nch: usize, rows: usize) !glm.latent.Partials {
    return .{ .po = try r.zeros(nch * rows * H * LW * 4), .pm = try r.zeros(nch * rows * H * 4), .pl = try r.zeros(nch * rows * H * 4) };
}

fn guards(r: *Rig, base: u64, rows: usize) !void {
    const size = guard_rows * H * LW * 2;
    for ([_]u64{ base, base + (guard_rows + rows) * H * LW * 2 }) |at|
        for (try r.back(at, size)) |v| try check.expect(v == 0xff, "row16 attention: outer guards", .{});
}

fn written(raw: []const u8, count_zero: bool) bool {
    if (count_zero) {
        for (raw) |v| if (v != 0xff) return false;
        return true;
    }
    var i: usize = 0;
    while (i < raw.len) : (i += 2) {
        const b = std.mem.readInt(u16, raw[i..][0..2], .little);
        if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, b) << 16)))) return false;
    }
    return true;
}

test "row16 attention oracle refuses unwritten poison except explicit count-zero rows" {
    const poison = [_]u8{ 0xff, 0xff, 0xff, 0xff };
    const zero = [_]u8{ 0, 0, 0, 0 };
    try std.testing.expect(!written(&poison, false));
    try std.testing.expect(written(&poison, true));
    try std.testing.expect(written(&zero, false));
    try std.testing.expect(!written(&zero, true));
}

fn fixture(gpu: check.Gpu, set: glm.glue.Set, s: cuda.Stream, rows: usize, pos: usize) ![32]u8 {
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(1616 + rows + pos);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    defer cuda.completion.drain(gpu.d, s, "row16 attention fixture teardown");
    const keys = pos + rows;
    const capacity = keys + 17;
    const c = try r.bfs(keys * LW, 1);
    const cd = try r.dev(std.mem.sliceAsBytes(c));
    const fp = try r.zeros(try glm.latent.cacheBytes(capacity, LW, true));
    const scales = fp + capacity * LW;
    try glm.latent.latentWriteFormat(set, s, cd, LW, fp, scales, .{}, try position(&r, 0), keys, LW);
    const decoded = try r.a.alloc(u16, capacity * LW);
    @memset(decoded, 0x7fc0);
    const codes = try r.a.alloc(u8, keys * LW);
    const sc = try r.a.alloc(f32, keys);
    for (0..keys) |row| {
        sc[row] = try glm.kv8_ref.pack(c[row * LW ..][0..LW], codes[row * LW ..][0..LW]);
        for (codes[row * LW ..][0..LW], 0..) |code, col|
            decoded[row * LW + col] = glm.kv8_ref.decoded(code, sc[row]);
    }
    try check.sameBytes("row16 attention: native FP8 codes", try r.back(fp, keys * LW), codes);
    try check.sameBytes("row16 attention: native FP8 scales", try r.back(scales, keys * 4), std.mem.sliceAsBytes(sc));
    const bf = try r.dev(std.mem.sliceAsBytes(decoded));
    const q = try r.bfs((rows + 2 * guard_rows) * H * LW, 1);
    const qbase = try r.dev(std.mem.sliceAsBytes(q));
    const dq = qbase + guard_rows * H * LW * 2;
    const nch = glm.latent.chunksFor(keys);
    const width = 2051;
    const tokens = try r.a.alloc(i32, rows * width);
    @memset(tokens, -1);
    const counts = try r.a.alloc(i32, rows);
    for (0..rows) |row| {
        counts[row] = if (row % 7 == 0) 0 else @intCast(@min(@min(keys, width), 47 + row * 173));
        for (0..@intCast(counts[row])) |col|
            tokens[row * width + col] = @intCast(col * (keys - 1) / @max(@as(usize, 1), @as(usize, @intCast(counts[row])) - 1));
    }
    const dt = try r.dev(std.mem.sliceAsBytes(tokens));
    const dc = try r.dev(std.mem.sliceAsBytes(counts));
    const out_len = (rows + 2 * guard_rows) * H * LW * 2;
    for ([_]bool{ false, true }) |sparse| {
        const fo = try r.zeros(out_len);
        const bo = try r.zeros(out_len);
        const f = fo + guard_rows * H * LW * 2;
        const b = bo + guard_rows * H * LW * 2;
        if (sparse) {
            const nc = glm.latent.chunksFor(width);
            try glm.latent.sparseAttentionFormat(set, s, dq, fp, scales, .{}, dt, dc, width, try partials(&r, nc, rows), f, rows, H, LW, 0.125);
            try glm.latent.sparseAttention(set, s, dq, bf, .{}, dt, dc, width, try partials(&r, nc, rows), b, rows, H, LW, 0.125);
        } else {
            const at = try position(&r, pos);
            try glm.latent.attentionFormat(set, s, dq, fp, scales, .{}, at, try partials(&r, nch, rows), f, try r.zeros(4), rows, H, LW, nch, 0.125, 16);
            try glm.latent.attention(set, s, dq, bf, .{}, at, try partials(&r, nch, rows), b, try r.zeros(4), rows, H, LW, nch, 0.125, 16);
        }
        const wide = try r.back(f, rows * H * LW * 2);
        for (0..rows) |row| try check.expect(written(wide[row * H * LW * 2 ..][0 .. H * LW * 2], sparse and counts[row] == 0), "row16 attention: positive write or explicit count-zero preservation", .{});
        digest.update(wide);
        try check.sameBytes("row16 attention: FP8 vs reconstructed BF16", wide, try r.back(b, wide.len));
        for (0..rows) |row| {
            const one = try r.zeros(H * LW * 2);
            if (sparse) {
                try glm.latent.sparseAttentionFormat(set, s, dq + row * H * LW * 2, fp, scales, .{}, dt + row * width * 4, dc + row * 4, width, try partials(&r, glm.latent.chunksFor(width), 1), one, 1, H, LW, 0.125);
            } else {
                try glm.latent.attentionFormat(set, s, dq + row * H * LW * 2, fp, scales, .{}, try position(&r, pos + row), try partials(&r, glm.latent.chunksFor(pos + row + 1), 1), one, try r.zeros(4), 1, H, LW, glm.latent.chunksFor(pos + row + 1), 0.125, 16);
            }
            try check.sameBytes("row16 attention: row-alone identity", try r.back(one, H * LW * 2), wide[row * H * LW * 2 ..][0 .. H * LW * 2]);
        }
        try guards(&r, fo, rows);
        try guards(&r, bo, rows);
    }
    try check.sameBytes("row16 attention: queries immutable", try r.back(qbase, q.len * 2), std.mem.sliceAsBytes(q));
    try check.sameBytes("row16 attention: original BF16 cache immutable", try r.back(cd, c.len * 2), std.mem.sliceAsBytes(c));
    try check.sameBytes("row16 attention: indices immutable", try r.back(dt, tokens.len * 4), std.mem.sliceAsBytes(tokens));
    try check.sameBytes("row16 attention: counts immutable", try r.back(dc, counts.len * 4), std.mem.sliceAsBytes(counts));
    return digest.finalResult();
}

fn qualify(gpu: check.Gpu, widths: []const usize) !void {
    var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer m.unload();
    const k = try glm.glue.Kernels.load(m);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    defer cuda.completion.drain(gpu.d, s, "row16 attention module teardown");
    for (widths) |rows| for ([_]usize{ 497, 2047 }) |pos| {
        const first = try fixture(gpu, .{ .glue = &k }, s, rows, pos);
        for (0..4) |_| try check.sameBytes("row16 attention: five-repeat output SHA", &first, &(try fixture(gpu, .{ .glue = &k }, s, rows, pos)));
    };
}

pub fn run(gpu: check.Gpu) !void {
    try qualify(gpu, &.{ 9, 15, 16 });
    check.pass("GLM_ROW16_ATTENTION: H16 R9/15/16, BF16/FP8 dense+sparse, row-alone bytes", .{});
}

pub fn run32(gpu: check.Gpu) !void {
    try qualify(gpu, &.{ 17, 31, 32 });
    check.pass("GLM_ROW32_ATTENTION: H16 R17/31/32, five repeats, BF16/FP8 dense+sparse, chunk edges, row-alone bytes", .{});
}
