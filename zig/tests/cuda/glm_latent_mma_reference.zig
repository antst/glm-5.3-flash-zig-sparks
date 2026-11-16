//! Independent periodic BF16 operands, FP64 dots and FP32/BF16 error bounds for both latent projection layouts.
const std = @import("std");
const c = @import("core").contract.spec;
pub const Fill = enum { ordinary, rounding, stress, nonfinite };

pub fn input(row: usize, k: usize, fill: Fill) u16 {
    if (fill == .nonfinite and row % 4 < 3 and k == 13)
        return if (row % 4 == 0) 0x7fc1 else if (row % 4 == 1) 0x7f80 else 0xff80;
    if (fill == .stress) return c.toBf16(switch (k % 4) {
        0 => 4096, 1 => 1, 2 => -4096, else => 0,
    });
    const v: f32 = @floatFromInt(@as(i32, @intCast((k * 17 + row % 4 * 13) % 31)) - 15);
    const scale: f32 = if (fill == .rounding)
        std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(k % 13)) - 6)) else 1.0 / 128.0;
    return c.toBf16(v * scale);
}

pub fn weight(head: usize, n: usize, k: usize, fill: Fill) u16 {
    if (fill == .stress or fill == .nonfinite) return c.toBf16(@as(f32, @floatFromInt(1 + n % 8)) / 16);
    const v: f32 = @floatFromInt(@as(i32, @intCast((k * 7 + n % 8 * 11 + head % 2 * 3) % 29)) - 14);
    return c.toBf16(v / 64);
}

pub const Dot = struct { value: f64, magnitude: f64 };

pub fn dot(row: usize, head: usize, n: usize, cols: usize, fill: Fill) Dot {
    var result: Dot = .{ .value = 0, .magnitude = 0 };
    for (0..cols) |k| {
        const term = c.fromBf16(input(row, k, fill)) * c.fromBf16(weight(head, n, k, fill));
        result.value += term;
        result.magnitude += @abs(term);
    }
    return result;
}

pub fn bound(d: Dot, cols: usize) f64 {
    const u: f64 = 0x1p-24;
    const gamma = @as(f64, @floatFromInt(cols)) * u / (1 - @as(f64, @floatFromInt(cols)) * u);
    const sum_error = gamma * d.magnitude;
    return sum_error + (1.0 / 256.0) * (@abs(d.value) + sum_error) + 0x1p-133;
}

pub fn accepts(value: f64, reference: Dot, cols: usize) bool {
    if (c.classOf(value) != c.classOf(reference.value)) return false;
    if (!std.math.isFinite(reference.value)) return true;
    return std.math.isFinite(value) and @abs(value - reference.value) <= bound(reference, cols);
}

test "latent MMA reference: periodic FP64 dots, nonfinite classes and reduced-precision red control" {
    const t = std.testing;
    for ([_]usize{ 256, 512 }) |cols| {
        const exact = dot(0, 0, 0, cols, .stress);
        try t.expectEqual(@as(f64, @floatFromInt(cols / 4)) / 16, exact.value);
        var reduced: f16 = 0;
        for (0..cols) |k| reduced = @floatCast(@as(f64, reduced) +
            c.fromBf16(input(0, k, .stress)) * c.fromBf16(weight(0, 0, k, .stress)));
        try t.expect(!accepts(@floatCast(reduced), exact, cols));
        try t.expect(accepts(c.fromBf16(c.toBf16(@floatCast(exact.value))), exact, cols));
        try t.expectEqual(dot(0, 0, 0, cols, .ordinary).value, dot(4, 2, 8, cols, .ordinary).value);
    }
    try t.expect(std.math.isNan(dot(0, 0, 0, 256, .nonfinite).value));
    try t.expectEqual(std.math.inf(f64), dot(1, 0, 0, 256, .nonfinite).value);
    try t.expectEqual(-std.math.inf(f64), dot(2, 0, 0, 256, .nonfinite).value);
    try t.expect(!accepts(std.math.nan(f64), dot(3, 0, 0, 256, .nonfinite), 256));
}
