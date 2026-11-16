//! GLM's EXL3 routed experts (cuda_exl3.run, decode and prompt) against float64 matrices from core's EXL3 decoder.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const fmt = core.exl3_format;
const h = qr.helpers;
const Rig = qr.RigT;

const E = 12; // routed experts (the shared one is id E, skipped)
const D = 512; // dims
const N = 256; // the rank's width of each expert
const slots = glm.exl3.top_k + 1;
const limit: f32 = 7.0;

/// One weight set: every expert's trellis words and scales as the loader lays them out, and their float64 matrices.
const Set = struct {
    gt: u64,
    ut: u64,
    dt: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    k2: u64,
    shx: bool,
    wg: []f64, // [E][D][N]
    wu: []f64,
    wd: []f64, // [E][N][D]

    fn experts(s: Set) glm.exl3.Experts {
        return .{ .k2 = s.k2, .gt = s.gt, .ut = s.ut, .dt = s.dt, .suh_g = s.suh_g, .suh_u = s.suh_u, .svh_g = s.svh_g, .svh_u = s.svh_u, .suh_d = s.suh_d, .svh_d = s.svh_d, .shx = s.shx };
    }
};

fn signs(r: *Rig, n: usize, scale: f16) ![]f16 {
    const out = try r.a.alloc(f16, n);
    for (out) |*v| v.* = if (r.rng.boolean()) scale else -scale;
    return out;
}

/// `E` random K x N matrices: trellis words, suh (shared by every expert when `one`), svh; device copies and float64.
fn matrices(r: *Rig, k: usize, n: usize, shared_suh: ?[]const f16, out_suh: *u64, out_svh: *u64, w: ?[]f64) !u64 {
    const words = k / 16 * (n / 16) * 64;
    const tr = try r.a.alloc(i16, E * words);
    for (tr) |*v| v.* = r.rng.int(i16);
    const suh = try r.a.alloc(f16, E * k);
    const svh = try r.a.alloc(f16, E * n);
    const scratch: []f64 = if (w != null) try r.a.alloc(f64, 2 * k * n) else &.{};
    for (0..E) |e| {
        if (shared_suh) |s| @memcpy(suh[e * k ..][0..k], s) else @memcpy(suh[e * k ..][0..k], try signs(r, k, 1.0));
        @memcpy(svh[e * n ..][0..n], try signs(r, n, 0.02));
        if (w) |want| fmt.dequantize(.{ .halves = 8 }, .mcg, tr[e * words ..][0..words], k / 16, n / 16, suh[e * k ..][0..k], svh[e * n ..][0..n], want[e * k * n ..][0 .. k * n], scratch);
    }
    out_suh.* = try r.dev(std.mem.sliceAsBytes(suh));
    out_svh.* = try r.dev(std.mem.sliceAsBytes(svh));
    const base = try r.dev(std.mem.sliceAsBytes(tr));
    const ptrs = try r.a.alloc(u64, E);
    for (ptrs, 0..) |*ptr, e| ptr.* = base + e * words * 2;
    return r.dev(std.mem.sliceAsBytes(ptrs));
}

fn weights(r: *Rig, one: bool) !Set {
    return weightsShape(r, one, D, N, true);
}

fn weightsShape(r: *Rig, one: bool, dims: usize, width: usize, want: bool) !Set {
    var s: Set = undefined;
    s.shx = one;
    const k2: [E]i32 = @splat(8);
    s.k2 = try r.dev(std.mem.sliceAsBytes(&k2));
    s.wg = if (want) try r.a.alloc(f64, E * dims * width) else &.{};
    s.wu = if (want) try r.a.alloc(f64, E * dims * width) else &.{};
    s.wd = if (want) try r.a.alloc(f64, E * width * dims) else &.{};
    const rot = if (one) try signs(r, dims, 1.0) else null;
    s.gt = try matrices(r, dims, width, rot, &s.suh_g, &s.svh_g, if (want) s.wg else null);
    s.ut = try matrices(r, dims, width, rot, &s.suh_u, &s.svh_u, if (want) s.wu else null);
    s.dt = try matrices(r, width, dims, null, &s.suh_d, &s.svh_d, if (want) s.wd else null);
    return s;
}

/// Each row's 8 distinct routed picks, then the shared expert's id.
fn picks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E]bool = @splat(false);
        for (0..slots - 1) |j| {
            var e = r.rng.uintLessThan(usize, E);
            while (seen[e]) e = r.rng.uintLessThan(usize, E);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E;
    }
    return out;
}

const Run = struct { ey: u64, rows: usize };

/// The family's dispatch on `x` (bf16 [rows][D]) and `pk`: ey [rows][slots][D] fp32, the shared slot left as filled.
fn run(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, s: Set, x: []const u16, pk: []const i32, rows: usize, prompt: bool) !Run {
    return runShape(r, k, ek, s, x, pk, rows, prompt, D, N);
}

fn runShape(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, s: Set, x: []const u16, pk: []const i32, rows: usize, prompt: bool, dims: usize, width: usize) !Run {
    const P = rows * glm.exl3.top_k;
    const tile: usize = if (prompt) glm.experts.staged_tile else glm.experts.tile;
    const wide = P > cuda.grouped.small;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E, tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) P else 1) * 4),
        .hist = try r.zeros((if (wide) (P + 1023) / 1024 * E else 1) * 4),
    };
    const sz = glm.exl3.scratchBytes(rows, dims, width, prompt, s.shx);
    const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const nbytes = rows * slots * dims * 4;
    const guarded = try r.a.alloc(u8, nbytes + 512);
    @memset(guarded, 0x5a);
    @memset(guarded[256..][0..nbytes], 0);
    const guard_base = try r.dev(guarded);
    const ey = guard_base + 256;
    const xd = try r.zeros(P * width * 2);
    const xp = try r.dev(std.mem.sliceAsBytes(x));
    const pp = try r.dev(std.mem.sliceAsBytes(pk));
    try glm.exl3.run(k, ek, r.s, s.experts(), sc, plan, xp, dims, pp, slots, rows, E, dims, width, limit, prompt, ey, xd);
    try r.s.synchronize();
    try check.sameBytes("EXL3 input unchanged", std.mem.sliceAsBytes(x), try r.back(xp, x.len * 2));
    try check.sameBytes("EXL3 picks unchanged", std.mem.sliceAsBytes(pk), try r.back(pp, pk.len * 4));
    const guards = try r.back(guard_base, nbytes + 512);
    try check.sameBytes("EXL3 leading guard", guarded[0..256], guards[0..256]);
    try check.sameBytes("EXL3 trailing guard", guarded[0..256], guards[nbytes + 256 ..]);
    return .{ .ey = ey, .rows = rows };
}

/// The kernels' SwiGLU: gate and up rounded to bf16 and limited, silu to bf16, the product to bf16.
fn act(g: f64, u: f64) f64 {
    const gg = @min(h.bfOf(h.bf16Of(@floatCast(g))), limit);
    const uu = @max(@min(h.bfOf(h.bf16Of(@floatCast(u))), limit), -limit);
    return h.bfOf(h.bf16Of(@floatCast(h.bfOf(h.bf16Of(@floatCast(gg / (1 + @exp(-gg))))) * uu)));
}

/// Float64 slot rows for routed slots (the shared slot's want is NaN: checked apart) and their tolerances.
fn reference(r: *Rig, s: Set, x: []const u16, pk: []const i32, rows: usize, want: []f64, tol: []f64) !void {
    const a = try r.a.alloc(f64, N);
    for (0..rows) |row| for (0..slots - 1) |j| {
        const e: usize = @intCast(pk[row * slots + j]);
        for (0..N) |n| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..D) |kk| {
                const xv = h.bfOf(x[row * D + kk]);
                g += xv * s.wg[(e * D + kk) * N + n];
                u += xv * s.wu[(e * D + kk) * N + n];
            }
            a[n] = act(g, u);
        }
        for (0..D) |n| {
            var y: f64 = 0;
            var mag: f64 = 0;
            for (0..N) |kk| {
                y += a[kk] * s.wd[(e * N + kk) * D + n];
                mag += @abs(a[kk] * s.wd[(e * N + kk) * D + n]);
            }
            const at = (row * slots + j) * D + n;
            want[at] = y;
            tol[at] = mag * (1.0 / 64.0) + 1e-6; // fp16 rotated inputs, bf16 activations, fp16 down inputs
        }
    };
}

/// Routed slots within tolerance; the shared slot untouched (the fill's bytes).
fn expect(r: *Rig, name: []const u8, out: Run, s: Set, x: []const u16, pk: []const i32) !void {
    const n = out.rows * slots * D;
    const want = try r.a.alloc(f64, n);
    const tol = try r.a.alloc(f64, n);
    try reference(r, s, x, pk, out.rows, want, tol);
    const raw = try r.back(out.ey, n * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (0..out.rows) |row| for (0..slots) |j| for (0..D) |c| {
        const i = (row * slots + j) * D + c;
        const bits = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
        if (j == slots - 1) { // the shared slot: the fill's bytes
            try check.expect(bits == 0xffffffff, "{s}: the shared slot left alone", .{name});
            continue;
        }
        const g: f64 = @as(f32, @bitCast(bits));
        const q = @abs(g - want[i]) / tol[i];
        if (!(q <= most)) {
            most = q;
            at = i;
        }
    };
    r.worst = @max(r.worst, most);
    std.debug.print("RESULT {s} ({d} rows): largest difference {d:.3} of its tolerance (element {d})\n", .{ name, out.rows, most, at });
    try check.expect(most <= 1, "{s} within its float64 tolerance", .{name});
}

pub fn run_(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), false);
    defer ek.unload();
    for ([_]bool{ true, false }) |one| {
        const s = try weights(&r, one);
        for ([_]usize{ 1, 3, 8 }) |rows| { // decode windows
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            const got = try run(&r, &k, &ek, s, x, pk, rows, false);
            try expect(&r, if (one) "glm-exl3 decode, one rotation" else "glm-exl3 decode, a rotation an expert", got, s, x, pk);
            if (rows > 1) { // every row's bits as alone
                const all = try r.back(got.ey, rows * slots * D * 4);
                for (0..rows) |row| {
                    const alone = try run(&r, &k, &ek, s, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, false);
                    const mine = try r.back(alone.ey, slots * D * 4);
                    try check.expect(std.mem.eql(u8, mine[0 .. (slots - 1) * D * 4], all[row * slots * D * 4 ..][0 .. (slots - 1) * D * 4]), "glm-exl3 decode: row {d} of {d} equals the row alone", .{ row, rows });
                }
            }
        }
        for ([_]usize{ 1, 40, 200, glm.exl3.chunk_rows + 6 }) |rows| { // prompt chunks, the last in two calls
            const x = try r.bfs(rows * D, 1.0);
            const pk = try picks(&r, rows);
            const got = try run(&r, &k, &ek, s, x, pk, rows, true);
            try expect(&r, if (one) "glm-exl3 prompt, one rotation" else "glm-exl3 prompt, a rotation an expert", got, s, x, pk);
            if (rows > glm.exl3.chunk_rows) { // the rows either side of the calls' boundary keep their bits as alone
                const all = try r.back(got.ey, rows * slots * D * 4);
                for ([_]usize{ 0, glm.exl3.chunk_rows - 1, glm.exl3.chunk_rows, rows - 1 }) |row| {
                    const alone = try run(&r, &k, &ek, s, x[row * D ..][0..D], pk[row * slots ..][0..slots], 1, true);
                    const mine = try r.back(alone.ey, slots * D * 4);
                    try check.expect(std.mem.eql(u8, mine[0 .. (slots - 1) * D * 4], all[row * slots * D * 4 ..][0 .. (slots - 1) * D * 4]), "glm-exl3 prompt: row {d} of {d} equals the row alone", .{ row, rows });
                }
            }
        }
    }
    { // a NaN in one row's input: that row's routed slots NaN, every other row finite
        const s = try weights(&r, true);
        const rows = 4;
        const x = try r.bfs(rows * D, 1.0);
        x[2 * D + 17] = 0x7fc0;
        const pk = try picks(&r, rows);
        for ([_]bool{ false, true }) |prompt| {
            const got = try run(&r, &k, &ek, s, x, pk, rows, prompt);
            const raw = try r.back(got.ey, rows * slots * D * 4);
            for (0..rows) |row| for (0..slots - 1) |j| for (0..D) |c| {
                const v: f32 = @bitCast(std.mem.readInt(u32, raw[((row * slots + j) * D + c) * 4 ..][0..4], .little));
                try check.expect(if (row == 2) std.math.isNan(v) else std.math.isFinite(v), "glm-exl3 {s}: a NaN stays in its row ({d})", .{ if (prompt) "prompt" else "decode", row });
            };
        }
    }
    check.pass("glm-exl3: decode and prompt slot rows within {d:.3} of their float64 tolerance, rows as alone, NaN kept to its row", .{r.worst});
}

pub fn fused(gpu: Gpu, path: []const u8) !void {
    return fusedWidth(gpu, path, 512);
}

pub fn fusedTp2(gpu: Gpu, path: []const u8) !void {
    return fusedWidth(gpu, path, 1024);
}

fn fusedWidth(gpu: Gpu, path: []const u8, width: usize) !void {
    const dims = 4096;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var baseline = try glm.exl3.Kernels.load(gpu.d);
    defer baseline.unload();
    var candidate = try glm.exl3.Kernels.load(gpu.d);
    defer candidate.unload();
    try glm.exl3.loadFusedDecode(gpu.gpa, gpu.io, gpu.d, path, &candidate);
    var calls: u64 = 0;
    candidate.decode_calls = &calls;
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), false);
    defer ek.unload();
    var prng = std.Random.DefaultPrng.init(197);
    for ([_]bool{ true, false }) |one| {
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer arena.deinit();
        var rig: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
        defer {
            for (rig.bufs.items) |*b| b.free();
            rig.bufs.deinit(gpu.gpa);
        }
        defer cuda.completion.drain(gpu.d, stream, "EXL3 fused fixture did not drain");
        const s = try weightsShape(&rig, one, dims, width, false);
        for ([_]usize{ 1, 3, 8, 15, 16, 17, 31, 32, 48, 63, 64, 4 }, 0..) |rows, ci| {
            const mark = rig.bufs.items.len;
            const x = try rig.bfs(rows * dims, 1.0);
            const nonfinite = ci == 11;
            if (nonfinite) {
                x[0] = 0x7fc0;
                x[dims + 1] = 0x7f80;
                x[2 * dims + 2] = 0xff80;
            }
            const pk = try picks(&rig, rows);
            const expected = try runShape(&rig, &baseline, &ek, s, x, pk, rows, false, dims, width);
            const n = rows * slots * dims * 4;
            const old = try rig.back(expected.ey, n);
            freeAfter(&rig, mark);
            for (0..5) |_| {
                const actual = try runShape(&rig, &candidate, &ek, s, x, pk, rows, false, dims, width);
                const got = try rig.back(actual.ey, n);
                try check.expect(std.mem.eql(u8, old, got), "EXL3 fused decode: rows {d} shared_input={any} full slot bytes", .{ rows, one });
                for (0..rows) |r| for (0..(slots - 1) * dims) |col| {
                    const index = (r * slots * dims + col) * 4;
                    const value: f32 = @bitCast(std.mem.readInt(u32, got[index..][0..4], .little));
                    try check.expect(if (nonfinite and r < 3) std.math.isNan(value) else std.math.isFinite(value),
                        "EXL3 fused decode wrote expected finite/nonfinite routed rows", .{});
                };
                freeAfter(&rig, mark);
            }
        }
    }
    try check.expect(calls == 120, "EXL3 fused decode active for all 120 positive runs", .{});
    check.pass("glm-exl3-fused-contract width={d}: 24 cases and 120 exact repeats", .{width});
}

fn freeAfter(rig: *Rig, mark: usize) void {
    while (rig.bufs.items.len > mark) {
        var b = rig.bufs.pop().?;
        b.free();
    }
}

pub fn whole(gpu: Gpu, path: []const u8) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var baseline = try glm.exl3.Kernels.load(gpu.d);
    defer baseline.unload();
    var candidate = try glm.exl3.Kernels.load(gpu.d);
    defer candidate.unload();
    try glm.exl3.loadWhole(gpu.gpa, gpu.io, gpu.d, path, &candidate);
    var calls: u64 = 0;
    candidate.whole_calls = &calls;
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)), false);
    defer ek.unload();
    for ([_]cuda.Function{ candidate.whole.?.kernels.rotate, candidate.whole.?.kernels.gateup, candidate.whole.?.kernels.down }, [_]u32{ 32, 256, 256 }) |f, threads| {
        const limits = core.contract.limits;
        const target = limits.target(try gpu.ctx.capability()) orelse return error.UnmeasuredGpu;
        const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .local = @intCast(try f.attribute(.local_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)) };
        var log: std.ArrayList(u8) = .empty;
        defer log.deinit(gpu.gpa);
        const problems = try limits.cuda(.{ .name = "EXL3 whole prompt", .symbol = "runtime-resolved", .image = "separate whole module", .threads = threads }, usage, target, &log, gpu.gpa);
        std.debug.print("{s}EXL3 WHOLE LIVE threads={d} registers={d} shared={d}\n", .{ log.items, usage.bound, usage.regs, usage.static });
        try check.expect(problems == 0, "EXL3 whole live resource limits", .{});
    }
    var cases: usize = 0;
    var expected_calls: u64 = 0;
    for ([_]bool{ true, false }) |shared| {
        for ([_]usize{ 1, 63, 64, 65, 1023, 1024, 1025, 2048, 4095, 4096, 4102 }) |rows| {
            try wholeCase(gpu, stream, &baseline, &candidate, &ek, D, N, rows, shared, false);
            expected_calls += 5 * ((rows + 4095) / 4096);
            cases += 1;
        }
        try wholeCase(gpu, stream, &baseline, &candidate, &ek, D, N, 4, shared, true);
        expected_calls += 5;
        cases += 1;
    }
    for ([_]usize{ 512, 1024 }) |width| for ([_]usize{ 1, 65, 1025, 2054, 4102 }) |rows| {
        try wholeCase(gpu, stream, &baseline, &candidate, &ek, 4096, width, rows, true, false);
        expected_calls += 5 * ((rows + 4095) / 4096);
        cases += 1;
    };
    try check.expect(calls == expected_calls, "EXL3 whole successful call census", .{});
    check.pass("glm-exl3-whole-contract {d} cases five repeats, direct bytes and shared-slot guards", .{cases});
}

fn wholeCase(gpu: Gpu, stream: cuda.Stream, baseline: *const glm.exl3.Kernels, candidate: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, dims: usize, width: usize, rows: usize, shared: bool, nonfinite: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(991);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    defer cuda.completion.drain(gpu.d, stream, "whole prompt fixture did not drain");
    const s = try weightsShape(&r, shared, dims, width, false);
    const x = try r.bfs(rows * dims, 1.0);
    if (nonfinite) {
        x[0] = 0x7fc0;
        x[dims + 1] = 0x7f80;
        x[2 * dims + 2] = 0xff80;
    }
    const pk = try picks(&r, rows);
    var first: []u8 = &.{};
    for (0..6) |rep| {
        var temporary: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer temporary.deinit();
        r.a = temporary.allocator();
        defer r.a = arena.allocator();
        const mark = r.bufs.items.len;
        defer freeAfter(&r, mark);
        defer cuda.completion.drain(gpu.d, stream, "whole prompt fixture did not drain");
        const raw = try wholeOutput(&r, if (rep == 0) baseline else candidate, ek, s, x, pk, rows, dims, width);
        if (rep == 0) first = try arena.allocator().dupe(u8, raw) else try check.sameBytes("whole prompt versus unchanged 1024-row module", first, raw);
        for (0..rows) |row| for (0..slots) |slot| for (0..dims) |col| {
            const at = 256 + ((row * slots + slot) * dims + col) * 4;
            const bits = std.mem.readInt(u32, raw[at..][0..4], .little);
            if (slot == 8) {
                try check.expect(bits == 0x4b123456, "EXL3 shared slot untouched", .{});
            } else if (!nonfinite) {
                try check.expect(bits != 0x4b123456 and std.math.isFinite(@as(f32, @bitCast(bits))), "EXL3 finite routed output written", .{});
            }
        };
    }
    std.debug.print("EXL3 WHOLE BYTES PASS D={d} I={d} R={d} shx={} nonfinite={} repeats=5\n", .{ dims, width, rows, shared, nonfinite });
}
fn wholeOutput(r: *Rig, k: *const glm.exl3.Kernels, ek: *const glm.experts.Kernels, s: Set, x: []const u16, pk: []const i32, rows: usize, dims: usize, width: usize) ![]u8 {
    const P = rows * slots;
    const tile: usize = glm.experts.staged_tile;
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(P * 4),
        .items = try r.zeros(cuda.grouped.maxItems(P, E + 1, tile) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros(P * 4),
        .hist = try r.zeros((P + 1023) / 1024 * (E + 1) * 4),
    };
    const sz = glm.exl3.scratchBytesMode(rows, dims, width, true, s.shx, k.whole != null);
    const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
    const output_bytes = rows * slots * dims * 4;
    const sentinel = try r.a.alloc(u32, output_bytes / 4 + 128);
    @memset(sentinel, 0x4b123456);
    const base = try r.dev(std.mem.sliceAsBytes(sentinel));
    const input_bytes = std.mem.sliceAsBytes(x);
    const picks_bytes = std.mem.sliceAsBytes(pk);
    const xp = try r.dev(input_bytes);
    const pp = try r.dev(picks_bytes);
    try glm.exl3.run(k, ek, r.s, s.experts(), sc, plan, xp, dims, pp, slots, rows, E, dims, width, limit, true, base + 256, 0);
    const raw = try r.back(base, output_bytes + 512);
    try check.sameBytes("whole input unchanged", input_bytes, try r.back(xp, input_bytes.len));
    try check.sameBytes("whole picks unchanged", picks_bytes, try r.back(pp, picks_bytes.len));
    try check.sameBytes("whole leading guard", std.mem.sliceAsBytes(sentinel)[0..256], raw[0..256]);
    try check.sameBytes("whole trailing guard", std.mem.sliceAsBytes(sentinel)[0..256], raw[256 + output_bytes ..]);
    return raw;
}
