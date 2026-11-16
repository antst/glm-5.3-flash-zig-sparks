//! Independent stage formulas and exact-sum stress for the whole-prompt EXL3 launchers.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const fmt = core.exl3_format;

pub const Kind = enum { rotate, gateup, down };
pub const experts = 8;
pub const slots = 9;
pub const had: f64 = @as(f32, 0.08838834764831844055);

fn depth(s: c.Shape) usize {
    return s.k / 16 + 16 + 14;
}
const rotate_shapes = [_]c.Shape{ .{ .rows = 4, .n = 128, .k = 4096 }, .{ .rows = 4096, .n = 128, .k = 4096 } };
const gu_shapes = [_]c.Shape{ .{ .rows = 4, .n = 1024, .k = 4096 }, .{ .rows = 65, .n = 512, .k = 4096 }, .{ .rows = 4096, .n = 512, .k = 4096 } };
const down_shapes = [_]c.Shape{ .{ .rows = 4, .n = 4096, .k = 512 }, .{ .rows = 65, .n = 4096, .k = 1024 }, .{ .rows = 4096, .n = 4096, .k = 512 } };

pub fn spec(kind: Kind, shared: bool) c.Spec {
    return .{
        .name = switch (kind) {
            .rotate => "cuda EXL3 whole prompt rotate",
            .gateup => if (shared) "cuda EXL3 whole prompt gateup shared9" else "cuda EXL3 whole prompt gateup paired1",
            .down => "cuda EXL3 whole prompt down direct9",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 3,
        .out = if (kind == .down) .f32 else .f16,
        .replaces = "the unchanged shipped EXL3 prompt stage on identical routed members",
        .nan = .{ .stated = "same stage bytes, with finite and planted nonfinite outputs" },
        .tiles = .{ .{ .size = 64 }, .{ .size = 256, .step = 256, .min = 256 }, .{ .size = 256, .step = 256, .min = 256 } },
        .served = switch (kind) {
            .rotate => &rotate_shapes,
            .gateup => &gu_shapes,
            .down => &down_shapes,
        },
    };
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}
fn value(v: u16, kind: Kind) f64 {
    return if (kind == .rotate) c.fromBf16(v) else @as(f16, @bitCast(v));
}

pub const Fixture = struct {
    a: std.mem.Allocator,
    kind: Kind,
    shape: c.Shape,
    fill: g.Fill,
    x: []u16,
    trellis: [2][]i16,
    scales: [3][]f16,
    weights: [2][]f16 = @splat(&.{}),

    pub fn valid(kind: Kind, s: c.Shape) bool {
        return s.rows > 0 and s.rows <= 4096 and s.k > 0 and s.k % 256 == 0 and s.n > 0 and
            (kind == .rotate or (s.n % 256 == 0 and s.n <= 4096));
    }

    pub fn init(a: std.mem.Allocator, kind: Kind, s: c.Shape, fill: g.Fill, seed: u64) !Fixture {
        var f: Fixture = .{ .a = a, .kind = kind, .shape = s, .fill = fill, .x = &.{}, .trellis = @splat(&.{}), .scales = @splat(&.{}) };
        errdefer f.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        f.x = try a.alloc(u16, s.rows * s.k);
        for (f.x) |*v| v.* = if (kind == .rotate) c.toBf16(rng.float(f32) * 2 - 1) else @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
        for (&f.scales, 0..) |*sc, i| {
            sc.* = try a.alloc(f16, if (kind == .rotate) s.k else s.n);
            for (sc.*) |*v| v.* = if (fill == .stress) switch (kind) {
                .rotate => 256,
                .gateup => if (i < 2) 0x1p-15 else 1,
                .down => 1,
            } else (if (rng.boolean()) @as(f16, 1) else -1) * (if (kind == .gateup and i < 2) @as(f16, 0.02) else 1);
        }
        if (kind != .rotate) for (&f.trellis) |*t| {
            t.* = try a.alloc(i16, s.k / 16 * (s.n / 16) * 64);
            for (t.*) |*v| v.* = if (fill == .stress) 0 else rng.int(i16);
        };
        if (fill == .stress) {
            for (0..s.rows) |r| {
                const row = f.x[r * s.k ..][0..s.k];
                const span = if (kind == .rotate) 128 else s.k;
                var base: usize = 0;
                while (base < s.k) : (base += span) {
                    const h = span / 2;
                    for (0..h) |i| {
                        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 16));
                        row[base + i] = if (kind == .rotate) c.toBf16(v) else @bitCast(@as(f16, @floatCast(v * 256)));
                        row[base + h + i] = if (kind == .rotate) c.toBf16(-v) else @bitCast(@as(f16, @floatCast(-v * 256)));
                    }
                    for ([_]usize{ 0, 32, h, h + 32 }) |i| row[base + i] = if (kind == .rotate) c.toBf16(64) else @bitCast(@as(f16, 64 * 256));
                    for ([_]usize{ 1, 33, h + 1 }) |i| row[base + i] = if (kind == .rotate) c.toBf16(-64) else @bitCast(@as(f16, -64 * 256));
                    row[base + h + 33] = if (kind == .rotate) c.toBf16(-65) else @bitCast(@as(f16, -65 * 256));
                }
            }
        } else if (fill == .nonfinite) {
            const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (plants, 0..) |v, r| if (r < s.rows) {
                f.x[r * s.k + r] = if (kind == .rotate) c.toBf16(v) else @bitCast(@as(f16, @floatCast(v)));
            };
        }
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.a.free(f.x);
        for (f.trellis) |t| f.a.free(t);
        for (f.scales) |s| f.a.free(s);
        for (f.weights) |w| f.a.free(w);
        f.* = undefined;
    }

    pub fn count(f: Fixture) usize {
        return f.shape.rows * (if (f.kind == .rotate) f.shape.k else slots * f.shape.n);
    }

    fn weight(f: *Fixture, index: usize) ![]const f16 {
        if (f.weights[index].len == 0) {
            f.weights[index] = try f.a.alloc(f16, f.shape.k * f.shape.n);
            fmt.unpack(.{ .halves = 8 }, .mcg, f.trellis[index], f.shape.k / 16, f.shape.n / 16, f.weights[index]);
        }
        return f.weights[index];
    }

    fn sum(acc: ?c.Accum, order: g.Order, t: []const f64) f64 {
        if (acc) |a| return core.contract.rows.sumIn(a, order, t);
        var v: f64 = 0;
        for (t) |x| v += x;
        return v;
    }

    /// Sylvester H128 without its normalizer: every signed input term is independent of the CUDA butterflies.
    fn hadamard(v: []f64, acc: ?c.Accum, order: g.Order) void {
        var input: [128]f64 = undefined;
        var terms: [128]f64 = undefined;
        var base: usize = 0;
        while (base < v.len) : (base += 128) {
            @memcpy(&input, v[base..][0..128]);
            for (0..128) |j| {
                for (&terms, input, 0..) |*t, x, i| t.* = if (@popCount(i & j) % 2 == 0) x else -x;
                v[base + j] = sum(acc, order, &terms) * had;
            }
        }
    }

    fn projection(f: *Fixture, matrix: usize, row: usize, out: []f64, acc: ?c.Accum, order: g.Order) !void {
        const terms = try f.a.alloc(f64, f.shape.k);
        defer f.a.free(terms);
        if (f.fill == .stress) {
            const w: f64 = fmt.codebookValue(.mcg, 0);
            std.debug.assert(w == 59.0 / 32.0);
            for (terms, f.x[row * f.shape.k ..][0..f.shape.k]) |*t, x| t.* = value(x, f.kind) * w;
            @memset(out, sum(acc, order, terms));
        } else {
            const w = try f.weight(matrix);
            for (out, 0..) |*v, j| {
                for (terms, 0..) |*t, i| t.* = value(f.x[row * f.shape.k + i], f.kind) * @as(f64, w[i * f.shape.n + j]);
                v.* = sum(acc, order, terms);
            }
        }
        hadamard(out, acc, order);
    }

    pub fn evaluate(f: *Fixture, out: []f64, acc: ?c.Accum, order: g.Order) !void {
        const n = if (f.kind == .rotate) f.shape.k else f.shape.n;
        const work = try f.a.alloc(f64, 2 * n);
        defer f.a.free(work);
        const z = work[0..n];
        const u = work[n..];
        for (0..f.shape.rows) |r| {
            if (f.kind == .rotate) {
                for (z, 0..) |*v, i| v.* = value(f.x[r * n + i], f.kind) * @as(f64, f.scales[0][i]);
                hadamard(z, acc, order);
                @memcpy(out[r * n ..][0..n], z);
                continue;
            }
            try f.projection(0, r, z, acc, order);
            if (f.kind == .gateup) {
                try f.projection(1, r, u, acc, order);
                for (z, u, 0..) |*v, up, i| {
                    const gg = @min(bf(v.* * @as(f64, f.scales[0][i])), 7);
                    const uu = std.math.clamp(bf(up * @as(f64, f.scales[1][i])), -7, 7);
                    v.* = bf(bf(gg / (1 + @exp(-gg))) * uu) * @as(f64, f.scales[2][i]);
                }
                hadamard(z, acc, order);
            } else for (z, 0..) |*v, i| v.* *= @as(f64, f.scales[0][i]);
            for (0..slots) |slot| {
                const dest = out[(r * slots + slot) * n ..][0..n];
                if (slot == experts) @memset(dest, 1) else @memcpy(dest, z);
            }
        }
    }

    pub fn reference(f: *Fixture, out: []f64, mag: []f64) !void {
        try f.evaluate(out, null, .sequential);
        @memset(mag, 0);
    }
};

const Host = struct {
    a: std.mem.Allocator,
    kind: Kind,
    f: ?Fixture = null,
    bytes: []u8 = &.{},
    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }
    fn prepare(ctx: *anyopaque, s: c.Shape, fill: g.Fill, seed: u64) anyerror!bool {
        const h = self(ctx);
        if (h.f) |*f| f.deinit();
        h.f = null;
        h.f = try Fixture.init(h.a, h.kind, s, fill, seed);
        return true;
    }
    fn run(ctx: *anyopaque) anyerror!void {
        const h = self(ctx);
        const f = &h.f.?;
        const out = try h.a.alloc(f64, f.count());
        defer h.a.free(out);
        try f.evaluate(out, .f32, .sequential);
        h.a.free(h.bytes);
        const elem = spec(h.kind, true).out;
        h.bytes = try h.a.alloc(u8, out.len * c.size(elem));
        for (out, 0..) |v, i| if (elem == .f32) {
            std.mem.writeInt(u32, h.bytes[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little);
        } else std.mem.writeInt(u16, h.bytes[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
    }
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        return self(ctx).bytes;
    }
    fn inputs(_: *anyopaque) anyerror![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) anyerror!bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(ctx: *anyopaque) usize {
        return self(ctx).f.?.count();
    }
    fn reference(ctx: *anyopaque, out: []f64, mag: []f64) anyerror!void {
        return self(ctx).f.?.reference(out, mag);
    }
    fn simulate(ctx: *anyopaque, acc: c.Accum, order: g.Order, out: []f64) anyerror!void {
        return self(ctx).f.?.evaluate(out, acc, order);
    }
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .simulate = simulate };
};

test "EXL3 whole gates: independent stress proves both reduced accumulators fail in all three orders" {
    const old = g.quiet;
    g.quiet = true;
    defer g.quiet = old;
    for ([_]Kind{ .rotate, .gateup, .down }) |kind| {
        var host: Host = .{ .a = std.testing.allocator, .kind = kind };
        defer {
            if (host.f) |*f| f.deinit();
            host.a.free(host.bytes);
        }
        const kernel: g.Kernel = .{ .spec = spec(kind, true), .ctx = &host, .vt = &Host.vt };
        const verdict = try g.sums(host.a, kernel, std.hash.Wyhash.hash(0x6b63, kernel.spec.name));
        try std.testing.expectEqual(g.Verdict.pass, verdict);
    }
}
