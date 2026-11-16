//! Scalar quantizer oracle: explicit FP32 operations, BF16 parameters, and a separately checked FP64 error.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
pub const Group = struct { words: [8]u32, scale: u16, bias: u16, error64: f64 };

fn even(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    return if (d > 0.5 or (d == 0.5 and @mod(f, 2) != 0)) f + 1 else f;
}
fn code(x: f32, scale: f32, bias: f32) u32 {
    return @intFromFloat(even(@max(0, @min(15, (x - bias) / scale))));
}

pub fn quantize(src: [64]u16) !Group {
    @setFloatMode(.strict);
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (src) |h| {
        const v = c.fromBf16(h);
        if (!std.math.isFinite(v)) return error.NonfiniteWeight;
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    const clips = [_]f32{ 1, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67, 0.64, 0.61, 0.58, 0.55 };
    var result: Group = undefined;
    var best: f32 = std.math.inf(f32);
    for (clips, 0..) |clip, index| {
        const l: f32 = lo * clip;
        const h: f32 = hi * clip;
        const sb = c.toBf16(@max((h - l) / 15, 1e-8));
        const bb = c.toBf16(l);
        const s = c.fromBf16(sb);
        const b = c.fromBf16(bb);
        if (!std.math.isFinite(s) or !std.math.isFinite(b) or s <= 0) return error.WeightRange;
        var partial: [32]f32 = undefined;
        for (&partial, 0..) |*v, lane| {
            const x = c.fromBf16(src[2 * lane]);
            const y = c.fromBf16(src[2 * lane + 1]);
            const px: f32 = @as(f32, @floatFromInt(code(x, s, b))) * s;
            const py: f32 = @as(f32, @floatFromInt(code(y, s, b))) * s;
            const ex: f32 = (px + b) - x;
            const ey: f32 = (py + b) - y;
            v.* = ex * ex + ey * ey;
        }
        var step: usize = 16;
        while (step > 0) : (step /= 2) {
            const before = partial;
            for (0..32 - step) |lane| partial[lane] = before[lane] + before[lane + step];
        }
        if (!std.math.isFinite(partial[0])) return error.WeightRange;
        if (index == 0 or partial[0] < best) {
            best = partial[0];
            result.scale = sb;
            result.bias = bb;
        }
    }
    const s = c.fromBf16(result.scale);
    const b = c.fromBf16(result.bias);
    result.words = @splat(0);
    result.error64 = 0;
    for (src, 0..) |h, i| {
        const q = code(c.fromBf16(h), s, b);
        result.words[i / 8] |= q << @as(u5, @intCast(4 * (i % 8)));
        const err: f64 = @as(f64, @floatFromInt(q)) * s + b - c.fromBf16(h);
        result.error64 += err * err;
    }
    return result;
}

pub fn fixture(group: usize, at: usize) u16 {
    const q: i32 = @as(i32, @intCast((group * 37 + at * 13) % 97)) - 48;
    return switch (group % 8) {
        0 => c.toBf16(0),
        1 => c.toBf16(-0.125),
        2 => c.toBf16(@as(f32, @floatFromInt(q)) / 128 + 0.25),
        3 => c.toBf16(if (at == 63) @as(f32, 2) else @as(f32, @floatFromInt(q)) / 1024),
        4 => c.toBf16(@as(f32, @floatFromInt(q)) * 0x1p-120),
        5 => c.toBf16(@as(f32, @floatFromInt(q)) * 0x1p30),
        else => c.toBf16(@as(f32, @floatFromInt(q)) / 256),
    };
}

test "affine target: scalar MSE oracle stores affine parameters, ties and finite error" {
    for (0..40) |g| {
        var src: [64]u16 = undefined;
        for (&src, 0..) |*h, i| h.* = fixture(g, i);
        const q = try quantize(src);
        try std.testing.expect(std.math.isFinite(q.error64));
        try std.testing.expect(c.fromBf16(q.scale) > 0);
        if (g % 8 == 0 or g % 8 == 1) try std.testing.expectEqual(@as(f64, 0), q.error64);
    }
    var bad: [64]u16 = @splat(0);
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |h| {
        bad[17] = h;
        try std.testing.expectError(error.NonfiniteWeight, quantize(bad));
    }
    bad = @splat(0x7f7f);
    bad[0] = 0xff7f;
    try std.testing.expectError(error.WeightRange, quantize(bad));
    try std.testing.expectEqual(@as(f32, 2), even(2.5));
    try std.testing.expectEqual(@as(f32, 4), even(3.5));
}
