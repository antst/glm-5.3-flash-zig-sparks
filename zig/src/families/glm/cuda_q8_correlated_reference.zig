//! CPU contract reference for fixed-scale, group64 correlated Q8 rounding.
const std = @import("std");
pub const Stats = struct { improved: bool, changed: usize, before: f64, after: f64 };
pub fn tri(i: usize, j: usize) usize {
    const a = @min(i, j);
    const b = @max(i, j);
    return a * (129 - a) / 2 + b - a;
}
pub fn decode(words: []const u32, i: usize) i32 {
    return @as(i32, @intCast((words[i / 4] >> @as(u5, @intCast(8 * (i % 4)))) & 255)) - 128;
}
fn value(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}
fn nearest(x: f64) i32 {
    const bounded = @max(-127, @min(127, x));
    const low = @floor(bounded);
    const fraction = bounded - low;
    return @intFromFloat(if (fraction > 0.5 or (fraction == 0.5 and @mod(low, 2) != 0)) low + 1 else low);
}
fn objective(w: *const [64]f64, q: *const [64]i32, g: []const f64, ridge: f64, scale: f64) f64 {
    @setFloatMode(.strict);
    var total: f64 = 0;
    for (0..64) |i| {
        const di = @as(f64, @floatFromInt(q[i])) * scale - w[i];
        total += (g[tri(i, i)] + ridge) * di * di;
        for (i + 1..64) |j| {
            const dj = @as(f64, @floatFromInt(q[j])) * scale - w[j];
            total += 2 * g[tri(i, j)] * di * dj;
        }
    }
    return total;
}
fn overlap(a: []const u8, b: []const u8) bool {
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}
/// Scales are immutable. Production must start from the exact old C18 pack.
/// The two ascending sweeps minimize the same damped Gram objective.
pub fn refine(src: []const u16, g: []const f64, scale_bits: u16, words: []u32) !Stats {
    @setFloatMode(.strict);
    if (src.len != 64 or g.len != 2080 or words.len != 16) return error.Q8CorrelatedShape;
    const output = std.mem.sliceAsBytes(words);
    if (overlap(output, std.mem.sliceAsBytes(src)) or overlap(output, std.mem.sliceAsBytes(g))) return error.Q8CorrelatedAlias;
    const scale: f64 = @as(f16, @bitCast(scale_bits));
    if (scale < 0 or !std.math.isFinite(scale)) return error.Q8CorrelatedScale;
    var w: [64]f64 = undefined;
    var q: [64]i32 = undefined;
    for (src, 0..) |h, i| {
        w[i] = value(h);
        q[i] = decode(words, i);
        if (!std.math.isFinite(w[i])) return error.Q8CorrelatedWeight;
        if (q[i] < -127 or q[i] > 127) return error.Q8CorrelatedSeed;
    }
    for (g) |v| if (!std.math.isFinite(v)) return error.Q8CorrelatedMoment;
    var trace: f64 = 0;
    for (0..64) |i| {
        if (g[tri(i, i)] < 0) return error.Q8CorrelatedMoment;
        trace += g[tri(i, i)];
    }
    if (!std.math.isFinite(trace)) return error.Q8CorrelatedMoment;
    if (trace == 0) {
        for (g) |v| if (v != 0) return error.Q8CorrelatedMoment;
        return .{ .improved = false, .changed = 0, .before = 0, .after = 0 };
    }
    const ridge = 1e-4 * trace / 64;
    const before = objective(&w, &q, g, ridge, scale);
    if (!std.math.isFinite(before) or before < 0) return error.Q8CorrelatedObjective;
    if (scale == 0) return .{ .improved = false, .changed = 0, .before = before, .after = before };
    var gradient: [64]f64 = @splat(0);
    for (0..64) |i| for (0..64) |j| {
        const h = g[tri(i, j)] + (if (i == j) ridge else @as(f64, 0));
        gradient[i] += h * (@as(f64, @floatFromInt(q[j])) * scale - w[j]);
    };
    for (0..2) |_| for (0..64) |i| {
        const diagonal = g[tri(i, i)] + ridge;
        const proposed = @as(f64, @floatFromInt(q[i])) - gradient[i] / (scale * diagonal);
        if (!std.math.isFinite(proposed)) return error.Q8CorrelatedObjective;
        const next = nearest(proposed);
        const delta = @as(f64, @floatFromInt(next - q[i])) * scale;
        const change = 2 * delta * gradient[i] + delta * delta * diagonal;
        if (!(change < 0)) continue;
        q[i] = next;
        for (0..64) |j| gradient[j] += (g[tri(j, i)] + (if (i == j) ridge else @as(f64, 0))) * delta;
    };
    const after = objective(&w, &q, g, ridge, scale);
    if (!std.math.isFinite(after) or after < 0) return error.Q8CorrelatedObjective;
    if (!(after < before)) return .{ .improved = false, .changed = 0, .before = before, .after = before };
    var changed: usize = 0;
    for (0..16) |word| {
        var bits: u32 = 0;
        for (0..4) |j| {
            const i = word * 4 + j;
            changed += @intFromBool(q[i] != decode(words, i));
            bits |= @as(u32, @intCast(q[i] + 128)) << @as(u5, @intCast(8 * j));
        }
        words[word] = bits;
    }
    return .{ .improved = true, .changed = changed, .before = before, .after = after };
}
