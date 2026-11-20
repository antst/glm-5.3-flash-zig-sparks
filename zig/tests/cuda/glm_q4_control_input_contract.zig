//! Original full-width BF16 arithmetic vs separate, single-slice control tail.
//! This tests the overwrite mechanism; quantized QKV quality is a separate model gate.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const old = @import("glm_mm_reference.zig");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const D = 4096;
const H = 32;
const C = 288;
const N = 12576;
const FA = 12288;
const MAX = 2048;
const ROWS = [_]usize{ 1, 8, 16, 32, 64, 129, 2048 };

pub fn run(gpu: check.Gpu, rank: usize) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer module.unload();
    var ops_module = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops_module.unload();
    defer s.synchronize() catch @panic("control contract stream did not drain before module unload");
    var ref = try old.Kernels.load(module);
    var ks = try glm.mm.Kernels.load(module);
    ref.pdl = false;
    ks.pdl = false;
    const copy = try ops_module.function("tf_glm_copy_rows");
    const set: glm.glue.Set = .{ .glue = undefined, .mm = &ks, .copy_rows = copy };
    for (ks.by_rows, 0..) |f, i| {
        const needed: i32 = if (i < 2) 128 else 256;
        try check.expect(try f.attribute(.max_threads_per_block) >= needed, "live BF16 control resource", .{});
        std.debug.print("Q4_CONTROL_RESOURCE bucket={d} regs={d} shared={d} local={d}\n", .{
            i, try f.attribute(.num_regs), try f.attribute(.shared_size_bytes), try f.attribute(.local_size_bytes),
        });
    }
    for ([_]bool{ false, true }) |stress| for ([_]usize{ D, D + 64 }) |stride|
        for ([_]bool{ false, true }) |prompt| try panel(gpu, s, set, &ref, rank, stress, stride, prompt);
    std.debug.print("Q4_CONTROL_INPUT_PASS rank={d} cases=56 repeats=5 negative_split=2\n", .{rank});
}

fn bf(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn put(bytes: []u8, i: usize, value: u16) void {
    std.mem.writeInt(u16, bytes[i * 2 ..][0..2], value, .little);
}
fn generated(a: std.mem.Allocator, count: usize, seed: u32) ![]u8 {
    const data = try a.alloc(u8, count * 2);
    var x = seed;
    for (0..count) |i| {
        x = x *% 1664525 +% 1013904223;
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((x >> 16) % 255)) - 127)) / 512;
        put(data, i, bf(v));
    }
    return data;
}
fn tileRef(ref: *const old.Kernels, s: cuda.Stream, rig: *Rig, raw: u64, n: usize, k: usize) !u64 {
    const dst = try rig.scratch(n * k * 2);
    try old.tile(ref, s, raw, dst, n, k);
    return dst;
}

fn panel(gpu: check.Gpu, s: cuda.Stream, set: glm.glue.Set, ref: *const old.Kernels, rank: usize, stress: bool, stride: usize, prompt: bool) !void {
    var selected = set;
    selected.prompt = prompt;
    const a = gpu.gpa;
    const raw_w = try generated(a, N * D, 2345 + @as(u32, @intCast(rank)));
    defer a.free(raw_w);
    const raw_x = try generated(a, MAX * stride, 8765);
    defer a.free(raw_x);
    const raw_fb = try generated(a, D * 128, 2468);
    defer a.free(raw_fb);
    const raw_gb = try generated(a, D * 128, 8642);
    defer a.free(raw_gb);
    if (stress) {
        // A real split-K rounding counterexample: SK1=1, SK8=0 (the small term vanishes in slice1).
        @memset(raw_w[FA * D * 2 ..][0 .. D * 2], 0);
        put(raw_w, FA * D, 0x4c80); // +2^26
        put(raw_w, FA * D + 512, 0xcc80); // -2^26
        put(raw_w, FA * D + 640, 0x3f80); // +1
        for (0..MAX) |r| for (0..D) |k| put(raw_x, r * stride + k, 0x3f80);
    }
    var rig: Rig = .{ .gpu = gpu, .s = s };
    var layers: [1]glm.weights.Layer = undefined;
    var w: glm.weights.Weights = .{ .gpa = a, .config = undefined, .share = undefined, .layers = &layers };
    w.share.world = 2;
    w.share.rank = rank;
    defer {
        s.synchronize() catch @panic("control contract stream did not drain before buffer free");
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(a);
        rig.deinit();
    }
    const raw = try rig.input(raw_w);
    const x = try rig.input(raw_x);
    const fb = try tileRef(ref, s, &rig, try rig.input(raw_fb), D, 128);
    const gb = try tileRef(ref, s, &rig, try rig.input(raw_gb), D, 128);
    layers[0].mixer = .{ .kda = .{ .proj = .{ .bf16 = raw }, .fb = fb, .gb = gb, .o = .{ .bf16 = 0 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = H, .dims = D } };
    try glm.q4_control_input.prepare(&w, set.mm.?, s, gpu.d);
    // Read the original raw tail before full tiling, exactly as startup does.
    const tiled = try tileRef(ref, s, &rig, raw, N, D);
    layers[0].mixer.kda.proj.bf16 = tiled;
    var scratch: glm.kda_block.Scratch = undefined;
    scratch.p = try rig.output(MAX * N * 2);
    scratch.ka = try rig.output(MAX * D * 2);
    scratch.kg = try rig.output(MAX * D * 2);
    scratch.mm_part = try rig.scratch(8 * 64 * N * 4);
    scratch.tmp = try rig.scratch(MAX * C * 2);
    scratch.tmp_bytes = MAX * C * 2;
    const rp = try rig.output(MAX * N * 2);
    const ra = try rig.output(MAX * D * 2);
    const rg = try rig.output(MAX * D * 2);
    for (ROWS) |rows| {
        var first: [32]u8 = undefined;
        for (0..5) |repeat| {
            for ([_]struct { ptr: u64, len: usize }{
                .{ .ptr = scratch.p, .len = MAX * N * 2 },  .{ .ptr = scratch.ka, .len = MAX * D * 2 },
                .{ .ptr = scratch.kg, .len = MAX * D * 2 }, .{ .ptr = rp, .len = MAX * N * 2 },
                .{ .ptr = ra, .len = MAX * D * 2 },         .{ .ptr = rg, .len = MAX * D * 2 },
            }) |b| {
                const buf: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = b.ptr, .len = b.len };
                try buf.fill8(0xff, s.handle);
            }
            try old.matmul(ref, s, x, stride, .{ .w = tiled, .n = N, .k = D }, rp, scratch.mm_part, rows, false, prompt);
            try old.matmul(ref, s, rp + FA * 2, N, .{ .w = fb, .n = D, .k = 128 }, ra, scratch.mm_part, rows, false, prompt);
            try old.matmul(ref, s, rp + (FA + 128) * 2, N, .{ .w = gb, .n = D, .k = 128 }, rg, scratch.mm_part, rows, false, prompt);
            try glm.kda_block.project(selected, s, layers[0].mixer.kda, scratch, x, stride, rows);
            var digest = std.crypto.hash.sha2.Sha256.init(.{});
            inline for (.{ .{ scratch.p, rp, N }, .{ scratch.ka, ra, D }, .{ scratch.kg, rg, D } }) |pair| {
                const count = rows * pair[2] * 2;
                const expected = try a.dupe(u8, try rig.bytes(pair[1], count));
                defer a.free(expected);
                const actual = try rig.bytes(pair[0], count);
                try finite(actual);
                try check.sameBytes("original full projection/gates vs restored tail", actual, expected);
                digest.update(actual);
                const tail = try rig.bytes(pair[0] + count, (MAX - rows) * pair[2] * 2);
                for (tail) |v| if (v != 0xff) return error.ControlOutputTailOverwrite;
            }
            var hash: [32]u8 = undefined;
            digest.final(&hash);
            if (repeat == 0) first = hash else try check.sameBytes("control five-repeat full-output SHA", &hash, &first);
        }
        if (stress and !prompt and rows == 1) try wrongSplit(&rig, set.mm.?, x, stride, layers[0].mixer.kda.control_tail, scratch.tmp, scratch.mm_part);
        std.debug.print("Q4_CONTROL_CASE_PASS rank={d} stress={any} stride={d} prompt={any} rows={d} sha={s}\n", .{ rank, stress, stride, prompt, rows, std.fmt.bytesToHex(first, .lower) });
    }
    try check.sameBytes("control original input immutable", try rig.bytes(x, raw_x.len), raw_x);
    try check.sameBytes("control original weights immutable", try rig.bytes(raw, raw_w.len), raw_w);
    try check.expect(try rig.intact(), "control all allocation guards", .{});
}

fn finite(bytes: []const u8) !void {
    var nonzero = false;
    var i: usize = 0;
    while (i < bytes.len) : (i += 2) {
        const v: f32 = @bitCast(@as(u32, std.mem.readInt(u16, bytes[i..][0..2], .little)) << 16);
        if (!std.math.isFinite(v)) return error.NonfiniteControlOutput;
        nonzero = nonzero or v != 0;
    }
    if (!nonzero) return error.MissingControlOutput;
}
fn wrongSplit(rig: *Rig, ks: *const glm.mm.Kernels, x: u64, stride: usize, w: u64, out: u64, part: u64) !void {
    try check.expect(glm.mm.splitK(C, D) == 8 and glm.mm.splitK(N, D) == 1, "actual split dispatch", .{});
    try glm.mm.matmulOneSlice(ks, rig.s, x, stride, .{ .w = w, .n = C, .k = D }, out, part, 1, false, false);
    const good = std.mem.readInt(u16, (try rig.bytes(out, 2))[0..2], .little);
    try glm.mm.matmul(ks, rig.s, x, stride, .{ .w = w, .n = C, .k = D }, out, part, 1, false, false);
    const bad = std.mem.readInt(u16, (try rig.bytes(out, 2))[0..2], .little);
    try check.expect(good == 0x3f80 and bad != good, "real default split loses cancellation witness", .{});
    std.debug.print("Q4_CONTROL_NEGATIVE_SPLIT_PASS good={x} wrong={x}\n", .{ good, bad });
}
