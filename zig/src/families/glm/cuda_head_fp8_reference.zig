//! Independent finite E4M3FN rounding and row128 quantization reference for the optional target head.
const std = @import("std");
const c = @import("core").contract.spec;
pub const Group = struct { codes: [128]u8, scale: f32 };

pub fn value(bits: u8) f32 {
    const e = (bits >> 3) & 15;
    const m = bits & 7;
    const abs: f32 = if (e == 0) @as(f32, @floatFromInt(m)) / 512 else (1 + @as(f32, @floatFromInt(m)) / 8) * @as(f32, @bitCast((@as(u32, e) + 120) << 23));
    return if (e == 15 and m == 7) std.math.nan(f32) else if (bits & 128 != 0) -abs else abs;
}

pub fn encode(x: f32) !u8 {
    if (!std.math.isFinite(x)) return error.NonfiniteWeight;
    const sign: u8 = if (std.math.signbit(x)) 128 else 0;
    const mag = @abs(x);
    if (mag >= 448) return sign | 126;
    var lo: u8 = 0;
    var hi: u8 = 126;
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        if (value(mid) <= mag) lo = mid else hi = mid;
    }
    const a = @as(f64, mag) - value(lo);
    const b = @as(f64, value(hi)) - mag;
    return sign | (if (a < b or (a == b and lo & 1 == 0)) lo else hi);
}

pub fn quantize(src: [128]u16) !Group {
    @setFloatMode(.strict);
    var amax: f32 = 0;
    for (src) |h| {
        const v = c.fromBf16(h);
        if (!std.math.isFinite(v)) return error.NonfiniteWeight;
        amax = @max(amax, @abs(v));
    }
    var out: Group = .{ .codes = undefined, .scale = @max(amax * @as(f32, @bitCast(@as(u32, 0x3b124925))), 1e-30) };
    for (src, &out.codes) |h, *q| q.* = try encode(@max(-448, @min(448, c.fromBf16(h) / out.scale)));
    return out;
}

pub fn fixture(group: usize, col: usize) u16 {
    const q: i32 = @as(i32, @intCast((group * 17 + col * 29) % 251)) - 125;
    const v: f32 = switch (group % 6) {
        0 => 0,
        1 => @as(f32, @floatFromInt(q)) / 64,
        2 => @as(f32, @floatFromInt(q)) * 0x1p-120,
        3 => @as(f32, @floatFromInt(q)) * 0x1p100,
        4 => if (col == 127) 12.5 else @as(f32, @floatFromInt(q)) / 1024,
        else => @as(f32, @floatFromInt(q)) / 256,
    };
    return c.toBf16(v);
}

test "head FP8: finite rounding covers every code, sign, midpoint and row128 scale" {
    var scalar: [128]u16 = @splat(c.toBf16(3));
    try std.testing.expectEqual(@as(u32, 0x3bdb6db8), @as(u32, @bitCast((try quantize(scalar)).scale)));
    scalar[0] = 0x7fc1;
    try std.testing.expectError(error.NonfiniteWeight, quantize(scalar));
    for (0..256) |i| {
        const bits: u8 = @intCast(i);
        if (bits & 127 == 127) continue;
        try std.testing.expectEqual(bits, try encode(value(bits)));
    }
    for (0..126) |i| {
        const lo: u8 = @intCast(i);
        const mid: f32 = (value(lo) + value(lo + 1)) / 2;
        try std.testing.expectEqual(if (lo & 1 == 0) lo else lo + 1, try encode(mid));
    }
    for (0..18) |g| {
        var src: [128]u16 = undefined;
        for (&src, 0..) |*v, i| v.* = fixture(g, i);
        const q = try quantize(src);
        try std.testing.expect(q.scale >= 1e-30 and std.math.isFinite(q.scale));
        for (q.codes) |code| try std.testing.expect(std.math.isFinite(value(code)));
    }
    try std.testing.expectError(error.NonfiniteWeight, encode(std.math.inf(f32)));
}
