//! Explicit real-head geometry, packing, FP64, row-alone, replay and precision checks.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("glm_head_fp8_oracle.zig");
const c = core.contract.spec;
const K = ref.K;

fn digest(bytes: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &result, .{});
    return result;
}

fn resource(gpu: check.Gpu, f: cuda.Function, name: []const u8, threads: u32, dynamic: u32, cluster: u32) !void {
    const limits = core.contract.limits;
    const target = limits.target(try gpu.ctx.capability()) orelse return error.UnmeasuredGpu;
    const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .local = @intCast(try f.attribute(.local_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)) };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    const launch: limits.CudaLaunch = .{ .name = name, .symbol = name, .image = "pinned head operators", .threads = threads, .dynamic = dynamic, .cluster = cluster, .opt_in = dynamic > 0 };
    const problems = try limits.cuda(launch, usage, target, &log, gpu.gpa);
    std.debug.print("{s}HEAD_RESOURCE {s} regs={d} shared={d} local={d} dynamic={d} cluster={d}\n", .{ log.items, name, usage.regs, usage.static, usage.local, dynamic, cluster });
    try check.expect(problems == 0, "head live resource limits", .{});
    log.clearRetainingCapacity();
    var wrong = usage;
    wrong.bound = threads - 1;
    const refused = try limits.cuda(launch, wrong, target, &log, gpu.gpa);
    try check.expect(refused > 0 and std.mem.indexOf(u8, log.items, "__launch_bounds__") != null, "head planted bad resource metadata refused", .{});
    std.debug.print("HEAD_LIMIT_CONTROL {s} expected=RED problems={d}\n", .{ name, refused });
}

const Weights = struct {
    r: Rig,
    n: usize,
    w: cuda.qmmf.Weight,
    oracle: ref.Oracle,
    original: [32]u8,

    fn init(gpu: check.Gpu, s: cuda.Stream, ks: glm.head_fp8.Kernels, packer: *const cuda.modelopt.Packer, n: usize) !Weights {
        var r: Rig = .{ .gpu = gpu, .s = s };
        errdefer r.deinit();
        var oracle = try ref.Oracle.init(gpu.gpa, n);
        errdefer oracle.deinit();
        errdefer cuda.completion.drain(gpu.d, s, "head weight preparation did not drain");
        const shape = try glm.head_fp8.Shape.make(n, K);
        const src = try gpu.gpa.alloc(u16, n * K);
        defer gpu.gpa.free(src);
        for (0..n) |j| {
            for (src[j * K ..][0..K], 0..) |*v, k| v.* = ref.weight(j, k);
        }
        const original = digest(std.mem.sliceAsBytes(src));
        const input = try r.input(std.mem.sliceAsBytes(src));
        const raw = try r.output(n * K);
        const scale = try r.output(n * (K / 64) * 4);
        const codes = try r.output(shape.codes);
        const scales = try r.output(shape.scales);
        const status = try r.scratch(4);
        try ks.quantize(s, shape, input, raw, scale, status);
        try check.expect(std.mem.readInt(u32, (try r.bytes(status, 4))[0..4], .little) == 0, "finite head accepted", .{});
        const raw_bytes = try gpu.gpa.dupe(u8, try r.bytes(raw, n * K));
        defer gpu.gpa.free(raw_bytes);
        for (0..n) |j| try check.sameBytes("all head quantized row codes", raw_bytes[j * K ..][0..K], oracle.codes[(j % ref.period) * K ..][0..K]);
        const scale_bytes = try r.bytes(scale, n * (K / 64) * 4);
        for (0..n) |j| for (0..K / 64) |g| try check.expect(std.mem.readInt(u32, scale_bytes[(j * (K / 64) + g) * 4 ..][0..4], .little) ==
            @as(u32, @bitCast(oracle.scales[(j % ref.period) * (K / 64) + g])), "all head scales exact", .{});
        try packer.fp8Lane(s, raw, codes, n, K);
        try ks.packScales(s, shape, scale, scales);
        const want = try gpu.gpa.alloc(u8, shape.codes);
        defer gpu.gpa.free(want);
        cuda.fp8.packCodes(want, raw_bytes, n, K);
        try check.sameBytes("entire head GPU packing equals CPU permutation", try r.bytes(codes, shape.codes), want);
        const packed_scales = try r.bytes(scales, shape.scales);
        for (0..shape.scales / 4) |i| {
            const j = (i / (64 * (K / 64))) * 64 + i % 64;
            const group = (i / 64) % (K / 64);
            const value: f32 = if (j < n) oracle.scales[(j % ref.period) * (K / 64) + group] else 1;
            try check.expect(std.mem.readInt(u32, packed_scales[i * 4 ..][0..4], .little) == @as(u32, @bitCast(value)), "all independent group64 scales and padding exact", .{});
        }
        try check.expect(try r.intact(), "all head packing guards", .{});
        return .{ .r = r, .n = n, .w = cuda.fp8.weight(codes, scales, n, K), .oracle = oracle, .original = original };
    }

    fn finish(w: *Weights) !void {
        try check.expect(std.mem.eql(u8, &w.original, &digest(try w.r.inputBytes())), "BF16 source head immutable", .{});
        try check.expect(try w.r.intact() and !w.r.overlap(), "head buffers guarded and disjoint", .{});
    }

    fn deinit(w: *Weights) void {
        cuda.completion.drain(w.r.gpu.d, w.r.s, "head weights freed before completion");
        w.r.deinit();
        w.oracle.deinit();
    }
};

const Call = struct {
    lane: *const cuda.qmmf.Lane,
    w: cuda.qmmf.Weight,
    x: u64,
    stride: usize,
    out: u64,
    rows: usize,
    elem: c.Elem,

    fn go(v: Call, stream: cuda.Stream) !void {
        const set: glm.glue.Set = .{ .glue = undefined, .linear = .{ .lane = v.lane } };
        try glm.projection.matmul(set, stream, .{ .fp8g = v.w }, v.x, v.stride, v.out, 0, v.rows, v.w.n, K, v.elem == .f32);
    }
};

fn guarded(bytes: []const u8, payload: usize) !void {
    for (bytes[0..16]) |v| try check.expect(v == 0xff, "head output prefix", .{});
    for (bytes[16 + payload ..]) |v| try check.expect(v == 0xff, "head output suffix", .{});
}

fn agrees(w: *const Weights, bytes: []const u8, rows: usize, elem: c.Elem, nonfinite: bool) bool {
    for (0..rows) |r| for (0..w.n) |j| {
        const got = c.read(elem, bytes, r * w.n + j);
        const want = w.oracle.want(r, j, nonfinite);
        if (c.classOf(got) != c.classOf(want)) return false;
        if (std.math.isFinite(want)) {
            const bound = w.oracle.bound(r, j, elem);
            if (!std.math.isFinite(got) or !std.math.isFinite(bound) or @abs(got - want) > bound) return false;
        }
    };
    return true;
}

fn panel(w: *Weights, lane: *const cuda.qmmf.Lane, rows: usize, stride: usize, elem: c.Elem, nonfinite: bool, controls: bool) !void {
    const gpu = w.r.gpu;
    const s = w.r.s;
    var r: Rig = .{ .gpu = gpu, .s = s };
    defer r.deinit();
    defer cuda.completion.drain(gpu.d, s, "head panel did not drain");
    const x = try gpu.gpa.alloc(u16, 8 + rows * stride + 8);
    defer gpu.gpa.free(x);
    @memset(x, 0x7fc1);
    for (0..rows) |row| {
        for (x[8 + row * stride ..][0..K], 0..) |*v, k| v.* = ref.input(row, k);
    }
    if (nonfinite) {
        for (0..@min(rows, 3)) |row| x[8 + row * stride + 17] =
            if (row == 0) 0x7fc1 else if (row == 1) 0x7f80 else 0xff80;
    }
    const dx = try r.input(std.mem.sliceAsBytes(x));
    const size = c.size(elem);
    const payload = rows * w.n * size;
    const slot = try r.output(payload + 64);
    const call: Call = .{ .lane = lane, .w = w.w, .x = dx + 16, .stride = stride, .out = slot + 16, .rows = rows, .elem = elem };
    const first = try gpu.gpa.alloc(u8, payload);
    defer gpu.gpa.free(first);
    try cuda.graph.beginCapture(s, .thread_local);
    call.go(s) catch |err| {
        if (cuda.graph.endCapture(s)) |v| {
            var bad = v;
            bad.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(s);
    defer graph.deinit();
    var exec = try graph.instantiate();
    defer exec.deinit();
    defer cuda.completion.drain(gpu.d, s, "head graph owners did not drain");
    for ([_]bool{ false, true }) |captured| for (0..5) |repeat| {
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = slot, .len = payload + 64 }).fill8(0xff, s.handle);
        if (captured) try exec.launchOn(s) else try call.go(s);
        const all = try r.bytes(slot, payload + 64);
        try guarded(all, payload);
        const got = all[16..][0..payload];
        try check.expect(agrees(w, got, rows, elem, nonfinite), "full-head FP64/class bound n={d} R={d}", .{ w.n, rows });
        if (!captured and repeat == 0) @memcpy(first, got) else try check.sameBytes("head five eager and five graph repeats exact", got, first);
    };
    if (!nonfinite) {
        const one = try r.output(w.n * size + 64);
        for (0..rows) |row| {
            var single = call;
            single.x += row * stride * 2;
            single.rows = 1;
            single.out = one + 16;
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = one, .len = w.n * size + 64 }).fill8(0xff, s.handle);
            try single.go(s);
            const got = try r.bytes(one, w.n * size + 64);
            try guarded(got, w.n * size);
            try check.sameBytes("every head row alone equals batched", got[16..][0 .. w.n * size], first[row * w.n * size ..][0 .. w.n * size]);
        }
    }
    if (controls) {
        var wrong = call;
        wrong.w.scale = 1.5;
        try wrong.go(s);
        try check.expect(!agrees(w, try r.bytes(wrong.out, payload), rows, elem, false), "wrong tensor scale control red", .{});
        wrong = call;
        wrong.stride = K;
        try wrong.go(s);
        try check.expect(!agrees(w, try r.bytes(wrong.out, payload), rows, elem, false), "wrong stride control red", .{});
    }
    try check.sameBytes("all head input rows and padding immutable", try r.inputBytes(), std.mem.sliceAsBytes(x));
    try check.expect(try r.intact() and !r.overlap(), "head call guards/disjoint", .{});
    std.debug.print("HEAD_CASE N={d} R={d} stride={d} out={s} nonfinite={any} eager=5 graph=5 exact=1\n", .{ w.n, rows, stride, @tagName(elem), nonfinite });
}

fn invalidWeights(gpu: check.Gpu, s: cuda.Stream, ks: glm.head_fp8.Kernels) !void {
    var r: Rig = .{ .gpu = gpu, .s = s };
    defer r.deinit();
    defer cuda.completion.drain(gpu.d, s, "invalid head fixture did not drain");
    var src: [3 * 128]u16 = @splat(0x3f80);
    src[0] = 0x7fc1;
    src[128] = 0x7f80;
    src[256] = 0xff80;
    const input = try r.input(std.mem.sliceAsBytes(&src));
    const raw = try r.output(src.len);
    const scale = try r.output(24);
    const status = try r.scratch(4);
    try ks.quantize(s, try glm.head_fp8.Shape.make(3, 128), input, raw, scale, status);
    try check.expect(std.mem.readInt(u32, (try r.bytes(status, 4))[0..4], .little) == 1, "nonfinite head weights rejected", .{});
    for (try r.bytes(raw, src.len), 0..) |v, i| {
        const want: u8 = if (i % 128 < 64) 0 else 0x7e;
        try check.expect(v == want, "only rejected group64 sanitized, adjacent finite group written", .{});
    }
    try check.sameBytes("nonfinite source weights immutable", try r.inputBytes(), std.mem.sliceAsBytes(&src));
    try check.expect(try r.intact(), "nonfinite weight guard", .{});
}

fn stress(gpu: check.Gpu, s: cuda.Stream, lane: *const cuda.qmmf.Lane, ks: glm.head_fp8.Kernels, packer: *const cuda.modelopt.Packer, control: cuda.Module) !void {
    const n = 38784;
    var r: Rig = .{ .gpu = gpu, .s = s };
    defer r.deinit();
    defer cuda.completion.drain(gpu.d, s, "head FP32 stress did not drain");
    const raw = try gpu.gpa.alloc(u8, n * K);
    defer gpu.gpa.free(raw);
    const shape = try glm.head_fp8.Shape.make(n, K);
    const scales = try gpu.gpa.alloc(f32, n * (K / 64));
    defer gpu.gpa.free(scales);
    @memset(scales, 1);
    const ds = try r.input(std.mem.sliceAsBytes(scales));
    const codes = try r.output(shape.codes);
    const ts = try r.output(shape.scales);
    const out = try r.output(n * 4);
    const reduced = try r.output(16 * 4);
    for (0..2) |mode| {
        var x: [K]u16 = undefined;
        var w: [K]u16 = undefined;
        for (0..K) |k| {
            const at = k % 64;
            x[k] = c.toBf16(if (mode == 0) 4096 else if (at == 0) 4096 else if (at < 32) 1 else if (at == 32) -4096 else -1);
            w[k] = c.toBf16(if (mode == 0 and at >= 32) -1 else 1);
        }
        for (raw, 0..) |*v, k| v.* = if (mode == 0 and k % 64 >= 32) 0xb8 else 0x38;
        const dx = try r.input(std.mem.sliceAsBytes(&x));
        const dw = try r.input(std.mem.sliceAsBytes(&w));
        const dc = try r.input(raw);
        const before = digest(try r.inputBytes());
        try packer.fp8Lane(s, dc, codes, n, K);
        try ks.packScales(s, shape, ds, ts);
        var call: Call = .{ .lane = lane, .w = cuda.fp8.weight(codes, ts, n, K), .x = dx, .stride = K, .out = out, .rows = 1, .elem = .f32 };
        for ([_]c.Elem{ .bf16, .f32 }) |elem| for (0..5) |_| {
            call.elem = elem;
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = out, .len = n * 4 }).fill8(0xff, s.handle);
            try call.go(s);
            const got = try r.bytes(out, n * 4);
            for (0..n) |j| try check.expect(c.read(elem, got, j) == 0, "actual FP32 head cancellation", .{});
            if (elem == .bf16) for (got[n * 2 ..]) |v|
                try check.expect(v == 0xff, "BF16 stress output width", .{});
        };
        for ([_][:0]const u8{ "tf_head_half_control", "tf_head_bfloat_control" }, 0..) |name, index| {
            if (mode == 0 and index == 1) continue;
            const f = try control.function(name);
            try resource(gpu, f, name, 32, 0, 1);
            var args: cuda.Args = .{};
            inline for (.{ dx, dw, reduced, @as(c_int, 16), @as(c_int, K) }) |v| args.add(v);
            try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, s, &args);
            const bad = try r.bytes(reduced, 16 * 4);
            for (0..16) |j| try check.expect(!std.math.isFinite(c.read(.f32, bad, j)) or @abs(c.read(.f32, bad, j)) > 1, "actual reduced accumulation control red", .{});
        }
        try check.expect(std.mem.eql(u8, &before, &digest(try r.inputBytes())), "stress inputs immutable", .{});
    }
    try check.expect(try r.intact(), "head stress guards", .{});
    check.pass("HEAD_FP32_STRESS two cancellation panels, FP16 and BF16 controls red", .{});
}

pub fn run(gpu: check.Gpu, args: []const [:0]const u8) !void {
    if (args.len != 1) return error.MissingHeadControlImage;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var module = try glm.head_fp8.load(gpu.gpa, gpu.io, gpu.d);
    defer module.unload();
    const ks = try glm.head_fp8.Kernels.load(module);
    var packer = try cuda.modelopt.Packer.load(gpu.d);
    defer packer.unload();
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, args[0], gpu.gpa, .limited(8 << 20), .@"16", null);
    defer gpu.gpa.free(bytes);
    var control = try cuda.Module.load(gpu.d, bytes);
    defer control.unload();
    defer cuda.completion.drain(gpu.d, stream, "head module owners did not drain");
    try resource(gpu, ks.quant, "head quantizer", 128, 0, 1);
    try resource(gpu, ks.scales, "head scale packing", 256, 0, 1);
    try resource(gpu, packer.fp8_lane, "head code packing", 256, 0, 1);
    for ([_]u32{ 16, 32, 64 }, 0..) |bm, ti| for (0..2) |cl| for ([_]bool{ false, true }) |wide| {
        const f = (if (wide) lane.f32_fns else lane.fns)[3][ti * 2 + cl];
        try resource(gpu, f, "head FP8G", 128, cuda.qmmf.sharedBytes(.fp8g, bm), if (cl == 1) 8 else 1);
    };
    try invalidWeights(gpu, stream, ks);
    var cases: usize = 0;
    for ([_]usize{ 127, 128, 129, 38656, 38784 }) |n| {
        var weights = try Weights.init(gpu, stream, ks, &packer, n);
        defer weights.deinit();
        for ([_]usize{ 1, 3, 8, 16, 17, 32, 33, 63, 64 }) |rows| for ([_]usize{ K, K + 64 }) |stride|
            for ([_]c.Elem{ .bf16, .f32 }) |elem| {
                try panel(&weights, &lane, rows, stride, elem, false, n == 129 and rows == 3 and stride > K);
                cases += 1;
            };
        if (n == 129) for ([_]c.Elem{ .bf16, .f32 }) |elem| {
            try panel(&weights, &lane, 3, K + 64, elem, true, false);
            cases += 1;
        };
        try weights.finish();
    }
    try stress(gpu, stream, &lane, ks, &packer, control);
    try check.expect(cases == 182, "head case census", .{});
    check.pass("HEAD_FP8_SUMMARY cases={d} repeats={d} packing/FP64/rowalone/graphs/guards/precision", .{ cases, cases * 10 });
}
