//! Prompt K policies: each GPU row alone and batched, guarded tails, and exact dyadic FP64 references.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
pub const Gpu = check.Gpu;
const pool = 136;
const guard = 128;
const marker = 0xa5;
const counts = [_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
pub const graph_counts = rows_block: {
    var rows: [64]usize = undefined;
    for (&rows, 1..) |*v, i| v.* = i;
    break :rows_block rows;
};
const CapturedCall = struct {
    mk: *const glm.mm.Kernels,
    x: u64,
    stride: usize,
    w: glm.mm.Tiled,
    out: u64,
    part: u64,
    rows: usize,
    wide: bool,
    fn body(ptr: *anyopaque, stream: cuda.Stream) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        try glm.mm.matmul(c.mk, stream, c.x, c.stride, c.w, c.out, c.part, c.rows, c.wide, true);
    }
};

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
fn value(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}
fn integer(bits: u8, col: usize, k: usize) i32 {
    const half: usize = if (bits == 16) 8 else @as(usize, 1) << @intCast(bits - 1);
    return @as(i32, @intCast((col * 17 + k * 13 + k / 64) % (2 * half))) - @as(i32, @intCast(half));
}
fn scale(col: usize, group: usize) f32 {
    return 1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast(4 + (col + group) % 3))));
}
fn pack(dst: []u32, codes: [16]u32, bits: u8) void {
    @memset(dst, 0);
    for (codes, 0..) |code, i| {
        if (bits == 6 and i == 15) {
            for (0..3) |j| dst[j] |= ((code >> @as(u5, @intCast(2 * j))) & 3) << 30;
        } else {
            const per: usize = 32 / bits;
            dst[i / per] |= code << @as(u5, @intCast((i % per) * bits));
        }
    }
}
fn guards(bytes: []const u8, used: usize) !void {
    for (bytes[0..guard]) |byte| try check.expect(byte == marker, "glm-mm-fixed: prefix guard", .{});
    for (bytes[guard + used ..]) |byte| try check.expect(byte == marker, "glm-mm-fixed: tail guard", .{});
}
// Fixture rows repeat every3, weight columns every48/192/768; cache only this proven periodic oracle.
fn reference(a: std.mem.Allocator, x: []const u16, w: []const u16, n: usize, k: usize, stride: usize, wide: bool, bits: u8, red: ?*usize) ![]u8 {
    const size: usize = if (wide) 4 else 2;
    const period: usize = if (bits == 8) 768 else if (bits == 6) 192 else 48;
    const columns = @min(n, period);
    const cache = try a.alloc(f64, 3 * columns);
    defer a.free(cache);
    for (0..3) |row| for (0..columns) |col| {
        var sum: f64 = 0;
        var lower: f16 = 0;
        for (0..k / 64) |group| {
            var dot: f64 = 0;
            for (0..64) |j| dot += value(x[row * stride + group * 64 + j]) * value(w[col * k + group * 64 + j]);
            sum += dot;
            lower = @floatCast(@as(f64, lower) + dot);
        }
        cache[row * columns + col] = sum;
        if (wide and red != null and @as(f64, lower) != sum) red.?.* += 1;
    };
    const out = try a.alloc(u8, pool * n * size);
    errdefer a.free(out);
    for (0..pool) |row| for (0..n) |col| {
        const sum = cache[(row % 3) * columns + col % period];
        if (wide) std.mem.writeInt(u32, out[(row * n + col) * size ..][0..4], @bitCast(@as(f32, @floatCast(sum))), .little) else std.mem.writeInt(u16, out[(row * n + col) * size ..][0..2], bf(@floatCast(sum)), .little);
    };
    return out;
}

fn shape(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream, bits: u8, n: usize, k: usize) !void {
    return shapeRuns(gpu, modules, s, bits, n, k, 1, null);
}

fn shapeRuns(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream, bits: u8, n: usize, k: usize, repeats: usize, red: ?*usize) !void {
    return shapeMode(gpu, modules, s, bits, n, k, repeats, red, false);
}

pub fn shapeMode(gpu: check.Gpu, modules: *const glm.mm.Kernels, s: cuda.Stream, bits: u8, n: usize, k: usize, repeats: usize, red: ?*usize, captured: bool) !void {
    const a = gpu.gpa;
    const affine = modules.affine != null;
    const stride = k + 8;
    const hw = try a.alloc(u16, n * k);
    defer a.free(hw);
    const hs = try a.alloc(u16, n * (k / 64));
    defer a.free(hs);
    const words = try a.alloc(u32, if (bits == 16) 4 else n * k * bits / 32);
    defer a.free(words);
    for (0..n) |col| {
        for (0..k / 64) |group| hs[col * (k / 64) + group] = if (affine) bf(scale(col, group)) else @bitCast(@as(f16, @floatCast(scale(col, group))));
        for (0..k) |j| hw[col * k + j] = bf(@as(f32, @floatFromInt(integer(bits, col, j))) * scale(col, j / 64));
        if (bits != 16) for (0..k / 16) |chunk| {
            var codes: [16]u32 = undefined;
            for (&codes, 0..) |*code, i| code.* = @intCast(integer(bits, col, chunk * 16 + i) + (@as(i32, 1) << @as(u5, @intCast(bits - 1))));
            const at = (col * (k / 16) + chunk) * (bits / 2);
            pack(words[at..][0 .. bits / 2], codes, bits);
        };
    }
    const hx = try a.alloc(u16, pool * stride);
    defer a.free(hx);
    @memset(hx, 0x7fc0);
    for (0..pool) |row| for (0..k) |j| {
        const q = @as(i32, @intCast((row * 5 + j * 3 + 1) % 15)) - 7;
        hx[row * stride + j] = bf(@as(f32, @floatFromInt(q)) / 32);
    };
    var source = try cuda.DeviceBuffer.fromHost(gpu.d, if (bits == 16) std.mem.sliceAsBytes(hw) else std.mem.sliceAsBytes(words));
    defer source.free();
    var scales = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hs));
    defer scales.free();
    const sizes = glm.q4.bytes(n, k);
    const tile_bytes = if (affine) sizes[0] else n * k * @as(usize, if (bits == 16) 2 else bits) / @as(usize, if (bits == 16) 1 else 8);
    const scale_bytes = if (affine) sizes[1] else hs.len * 2;
    const hb = try a.alloc(u16, hs.len);
    defer a.free(hb);
    for (hb, 0..) |*b, i| b.* = bf(-8 * scale(i / (k / 64), i % (k / 64)));
    var biases = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hb));
    defer biases.free();
    var tiled_biases = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + scale_bytes);
    defer tiled_biases.free();
    var tiled = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + tile_bytes);
    defer tiled.free();
    var tiled_scales = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + scale_bytes);
    defer tiled_scales.free();
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    const tw: glm.mm.Tiled = .{ .w = tiled.ptr + guard, .s = if (bits == 16) 0 else tiled_scales.ptr + guard, .bits = bits, .n = n, .k = k, .bias = if (affine) tiled_biases.ptr + guard else 0, .npad = if (affine) glm.q4.npadOf(n) else 0 };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + @max(16, @max(glm.mm.partBytes(pool, tw), if (affine) glm.affine_target.scratchBytes(pool, k) else 0)));
    defer part.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + pool * n * 4);
    defer out.free();
    var solo = try cuda.DeviceBuffer.alloc(gpu.d, out.len);
    defer solo.free();
    defer s.synchronize() catch @panic("GLM prompt gate retains device work until completion");
    try tiled.fill8(marker, s.handle);
    try tiled_scales.fill8(marker, s.handle);
    try part.fill8(marker, s.handle);
    try tiled_biases.fill8(marker, s.handle);
    if (affine) try glm.q4.pack(&modules.affine.?.q4, s, source.ptr, scales.ptr, biases.ptr, .{ .w = tw.w, .s = tw.s, .b = tw.bias, .n = n, .k = k, .npad = tw.npad }) else if (bits == 16) try glm.mm.tile(modules, s, source.ptr, tw.w, n, k) else try glm.mm.tilePacked(modules, s, bits, source.ptr, scales.ptr, tw.w, tw.s, n, k);
    try s.synchronize();
    const before_tiles = try a.alloc(u8, tiled.len);
    defer a.free(before_tiles);
    const before_biases = try a.alloc(u8, tiled_biases.len);
    defer a.free(before_biases);
    try tiled_biases.download(0, before_biases);
    try guards(before_biases, scale_bytes);
    const before_scales = try a.alloc(u8, tiled_scales.len);
    defer a.free(before_scales);
    try tiled.download(0, before_tiles);
    try tiled_scales.download(0, before_scales);
    try guards(before_tiles, tile_bytes);
    try guards(before_scales, scale_bytes);
    const got = try a.alloc(u8, out.len);
    defer a.free(got);
    const alone = try a.alloc(u8, solo.len);
    defer a.free(alone);
    for ([_]bool{ false, true }) |wide| {
        const size: usize = if (wide) 4 else 2;
        const want = try reference(a, hx, hw, n, k, stride, wide, bits, red);
        defer a.free(want);
        for ([_]bool{ false, true }) |fixed| {
            var mk = modules.*;
            mk.prompt_fixed_k = fixed;
            try solo.fill8(marker, s.handle);
            for (0..pool) |row| try glm.mm.matmul(&mk, s, x.ptr + row * stride * 2, stride, tw, solo.ptr + guard + row * n * size, part.ptr + guard, 1, wide, true);
            try s.synchronize();
            try solo.download(0, alone);
            try guards(alone, want.len);
            try check.sameBytes("glm-mm-fixed: solo against FP64", alone[guard..][0..want.len], want);
            const row_counts: []const usize = if (captured) &graph_counts else &counts;
            for (row_counts) |rows| for ([_]usize{ 0, 1, 7 }) |offset| {
                var command: CapturedCall = .{ .mk = &mk, .x = x.ptr + offset * stride * 2, .stride = stride, .w = tw, .out = out.ptr + guard, .part = part.ptr + guard, .rows = rows, .wide = wide };
                var engine: cuda.graph_cache.CudaEngine = .{ .d = gpu.d };
                const api = engine.engine();
                const exec: ?*anyopaque = if (captured) blk: {
                    try s.synchronize();
                    break :blk try api.capture(api.ctx, s, .{ .ctx = &command, .run = CapturedCall.body });
                } else null;
                defer {
                    s.synchronize() catch @panic("Q4 captured work must drain before graph/fixture teardown");
                    if (exec) |e| api.free(api.ctx, e);
                }
                for (0..repeats) |_| {
                    try out.fill8(marker, s.handle);
                    if (exec) |e| try api.launch(api.ctx, e, s) else try CapturedCall.body(&command, s);
                    try s.synchronize();
                    try out.download(0, got);
                    const bytes = rows * n * size;
                    const at = offset * n * size;
                    try guards(got, bytes);
                    try check.sameBytes("glm-mm-fixed: batched against FP64", got[guard..][0..bytes], want[at..][0..bytes]);
                    try check.sameBytes("glm-mm-fixed: batched against each row alone", got[guard..][0..bytes], alone[guard + at ..][0..bytes]);
                }
            };
        }
    }
    if (red != null and n == 2048 and k == 128) try nonfinite(gpu, modules, s, tw, hw, hx, stride, &x, &out, &part, got, captured);
    const after_tiles = try a.alloc(u8, tiled.len);
    defer a.free(after_tiles);
    const after_scales = try a.alloc(u8, tiled_scales.len);
    defer a.free(after_scales);
    const after_part = try a.alloc(u8, part.len);
    defer a.free(after_part);
    try tiled.download(0, after_tiles);
    try tiled_scales.download(0, after_scales);
    try part.download(0, after_part);
    try check.sameBytes("target Q4 packed weights immutable/guards", before_tiles, after_tiles);
    try check.sameBytes("target Q4 tiled scales immutable/guards", before_scales, after_scales);
    try guards(after_part, part.len - 2 * guard);
    const bias_back = try a.alloc(u8, tiled_biases.len);
    defer a.free(bias_back);
    try tiled_biases.download(0, bias_back);
    try check.sameBytes("target affine biases immutable/guards", bias_back, before_biases);
    const input_back = try a.alloc(u8, x.len);
    defer a.free(input_back);
    try x.download(0, input_back);
    try check.sameBytes("glm-mm-fixed: input immutable", input_back, std.mem.sliceAsBytes(hx));
    const words_back = try a.alloc(u8, source.len);
    defer a.free(words_back);
    try source.download(0, words_back);
    try check.sameBytes("glm-mm-fixed: raw weights immutable", words_back, if (bits == 16) std.mem.sliceAsBytes(hw) else std.mem.sliceAsBytes(words));
    const scales_back = try a.alloc(u8, scales.len);
    defer a.free(scales_back);
    try scales.download(0, scales_back);
    try check.sameBytes("target Q4 raw scales immutable", scales_back, std.mem.sliceAsBytes(hs));
}
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer module.unload();
    const kernels = try glm.mm.Kernels.load(module);
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_][2]usize{ .{ 64, 512 }, .{ 128, 2048 } }) |dims| try shape(gpu, &kernels, s, bits, dims[0], dims[1]);
    check.pass("glm-mm-fixed: both policies, bf16/4/6/8 bits, FP32/bf16 output, every solo/offset/bucket/tail cell", .{});
}

fn nonfinite(gpu: check.Gpu, mk: *const glm.mm.Kernels, stream: cuda.Stream, w: glm.mm.Tiled, hw: []const u16, hx: []u16, stride: usize, x: *cuda.DeviceBuffer, out: *cuda.DeviceBuffer, part: *cuda.DeviceBuffer, got: []u8, captured: bool) !void {
    const saved = [3]u16{ hx[0], hx[stride + 1], hx[2 * stride + 2] };
    defer for (0..3) |r| {
        hx[r * stride + r] = saved[r];
    };
    hx[0] = 0x7fc0;
    hx[stride + 1] = 0x7f80;
    hx[2 * stride + 2] = 0xff80;
    try x.upload(0, std.mem.sliceAsBytes(hx));
    const expected = try gpu.gpa.alloc(f64, 3 * w.n);
    defer gpu.gpa.free(expected);
    for (0..3) |row| for (0..w.n) |col| {
        var sum: f64 = 0;
        for (0..w.k) |j| sum += value(hx[row * stride + j]) * value(hw[col * w.k + j]);
        if (w.bias != 0) {
            sum = 0;
            for (0..w.k / 64) |group| {
                const sc: f64 = scale(col, group);
                var dot: f64 = 0;
                var xs: f64 = 0;
                for (0..64) |i| {
                    const j = group * 64 + i;
                    const v = value(hx[row * stride + j]);
                    dot += v * (value(hw[col * w.k + j]) / sc + 8);
                    xs += v;
                }
                sum += dot * sc + xs * (-8 * sc);
            }
        }
        expected[row * w.n + col] = sum;
    };
    const solo = try gpu.gpa.alloc(u8, 3 * w.n * 4);
    defer gpu.gpa.free(solo);
    for ([_]bool{ false, true }) |fixed| for ([_]bool{ false, true }) |wide| {
        var selected = mk.*;
        selected.prompt_fixed_k = fixed;
        const size: usize = if (wide) 4 else 2;
        const used = 3 * w.n * size;
        try out.fill8(marker, stream.handle);
        for (0..3) |r| try glm.mm.matmul(&selected, stream, x.ptr + r * stride * 2, stride, w, out.ptr + guard + r * w.n * size, part.ptr + guard, 1, wide, true);
        try stream.synchronize();
        try out.download(0, got);
        try guards(got, used);
        @memcpy(solo[0..used], got[guard..][0..used]);
        var command: CapturedCall = .{ .mk = &selected, .x = x.ptr, .stride = stride, .w = w, .out = out.ptr + guard, .part = part.ptr + guard, .rows = 3, .wide = wide };
        var engine: cuda.graph_cache.CudaEngine = .{ .d = gpu.d };
        const api = engine.engine();
        const exec: ?*anyopaque = if (captured) blk: {
            try stream.synchronize();
            break :blk try api.capture(api.ctx, stream, .{ .ctx = &command, .run = CapturedCall.body });
        } else null;
        defer {
            stream.synchronize() catch @panic("Q4 nonfinite captured work must drain before graph teardown");
            if (exec) |e| api.free(api.ctx, e);
        }
        for (0..5) |_| {
            try out.fill8(marker, stream.handle);
            if (exec) |e| try api.launch(api.ctx, e, stream) else try CapturedCall.body(&command, stream);
            try stream.synchronize();
            try out.download(0, got);
            try guards(got, used);
            try check.sameBytes("target Q4 nonfinite repeats/row-alone", got[guard..][0..used], solo[0..used]);
            for (expected, 0..) |e, i| {
                const at = guard + i * size;
                const v: f32 = if (wide) @bitCast(std.mem.readInt(u32, got[at..][0..4], .little)) else @bitCast(@as(u32, std.mem.readInt(u16, got[at..][0..2], .little)) << 16);
                try check.expect(if (std.math.isNan(e)) std.math.isNan(v) else std.math.isInf(v) and std.math.signbit(v) == std.math.signbit(e), "target Q4 NaN/Inf reference class", .{});
            }
        }
        if (exec) |e| {
            for (0..3) |r| hx[r * stride + r] = saved[r];
            const finite_ref = try reference(gpu.gpa, hx, hw, w.n, w.k, stride, wide, w.bits, null);
            defer gpu.gpa.free(finite_ref);
            try x.upload(0, std.mem.sliceAsBytes(hx));
            for (0..5) |_| {
                try out.fill8(marker, stream.handle);
                try api.launch(api.ctx, e, stream);
                try stream.synchronize();
                try out.download(0, got);
                try guards(got, used);
                try check.sameBytes("target Q4 same graph reads restored finite inputs", got[guard..][0..used], finite_ref[0..used]);
            }
            hx[0] = 0x7fc0;
            hx[stride + 1] = 0x7f80;
            hx[2 * stride + 2] = 0xff80;
            try x.upload(0, std.mem.sliceAsBytes(hx));
        }
    };
    const nonfinite_input = try gpu.gpa.alloc(u8, x.len);
    defer gpu.gpa.free(nonfinite_input);
    try x.download(0, nonfinite_input);
    try check.sameBytes("target Q4 nonfinite input immutable before restore", nonfinite_input, std.mem.sliceAsBytes(hx));
    for (0..3) |r| hx[r * stride + r] = saved[r];
    try x.upload(0, std.mem.sliceAsBytes(hx));
    check.pass("target Q4 nonfinite NaN/+Inf/-Inf,2policies*2outputs*5repeats/row-alone", .{});
}

pub const target_shapes = [_][2]usize{
    .{ 6416, 4096 }, .{ 2048, 128 }, .{ 4096, 2048 },  .{ 2048, 4096 },
    .{ 4096, 1536 }, .{ 160, 4096 }, .{ 6144, 4096 },  .{ 4096, 3072 },
    .{ 1024, 4096 }, .{ 4096, 512 }, .{ 38784, 4096 }, .{ 38656, 4096 },
    .{ 4096, 8192 },
};

pub fn targetLaunch(i: usize) @import("core").contract.limits.CudaLaunch {
    std.debug.assert(i < 4);
    const bm: usize = @as(usize, 16) << @intCast(i);
    return .{ .name = "GLM target Q4", .symbol = "runtime-resolved", .image = "glm_mm", .threads = if (bm <= 32) 128 else 256, .dynamic = glm.mm.sharedBytes(4, bm), .opt_in = true };
}

/// Explicit large-shape Q4 gate, never part of the default runner.
pub fn targetQ4(gpu: check.Gpu) !void {
    return targetMode(gpu, false);
}

pub fn targetQ4Graphs(gpu: check.Gpu) !void {
    return targetMode(gpu, true);
}

fn targetMode(gpu: check.Gpu, captured: bool) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer module.unload();
    const kernels = try glm.mm.Kernels.load(module);
    const limits = @import("core").contract.limits;
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var problems: usize = 0;
    for (kernels.packed_rows[0], 0..) |f, i| {
        const bm: usize = @as(usize, 16) << @intCast(i);
        const threads: u32 = if (bm <= 32) 128 else 256;
        const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)), .local = @intCast(try f.attribute(.local_size_bytes)) };
        const decl = targetLaunch(i);
        const enabled = try f.attribute(.max_dynamic_shared_size_bytes);
        try check.expect(enabled >= decl.dynamic, "target Q4 function has actual shared-memory opt-in", .{});
        problems += try limits.cuda(decl, usage, target, &log, gpu.gpa);
        std.debug.print("RESOURCE targetQ4 BM{d} sm{d} threads{d} regs{d} static{d} local{d} bound{d} dynamic{d} enabled{d} opt_in=true\n", .{ bm, cap, threads, usage.regs, usage.static, usage.local, usage.bound, decl.dynamic, enabled });
    }
    std.debug.print("{s}", .{log.items});
    try check.expect(problems == 0, "target Q4 live resource gate", .{});
    var red: usize = 0;
    for (target_shapes) |dims| {
        try shapeMode(gpu, &kernels, s, 4, dims[0], dims[1], 5, &red, captured);
        check.pass("target Q4 N{d} K{d}, captured={any} rowcounts={d} offsets3 policies2 outputs2 repeats5", .{ dims[0], dims[1], captured, if (captured) graph_counts.len else counts.len });
    }
    try check.expect(red > 0, "target Q4 FP32 fixture rejects grouped FP16 accumulation", .{});
    check.pass("glm-target-q4-contract captured={any} shapes13 repeatcalls={d}, FP64/solo/guards/input; CPU_FP16_reference_diff={d}", .{ captured, 13 * (if (captured) graph_counts.len else counts.len) * 3 * 2 * 2 * 5, red });
}

test "target Q4 oracle: periodic row and column formulas match all fixture cells" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| {
        const period: usize = if (bits == 8) 768 else if (bits == 6) 192 else 48;
        for (0..1000) |col| for (0..128) |j| {
            try std.testing.expectEqual(integer(bits, col, j), integer(bits, col % period, j));
            try std.testing.expectEqual(scale(col, j / 64), scale(col % period, j / 64));
        };
    }
    for (0..pool) |row| for (0..128) |j| try std.testing.expectEqual((row * 5 + j * 3 + 1) % 15, ((row % 3) * 5 + j * 3 + 1) % 15);
}
