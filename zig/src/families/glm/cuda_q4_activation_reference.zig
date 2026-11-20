//! Independent serial startup-fit reference; not linked into serving.
const std = @import("std");

pub const Pack = struct { words: [8]u32, scale: u16, bias: u16 };
pub const Result = struct { pack: Pack, before: f64, after: f64, rounds: u32 };

pub fn value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (bits & 0x7f800000 == 0x7f800000) return @truncate(bits >> 16);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

pub fn neighbors(center: u16) [3]u16 {
    return .{ center, if (center & 0x7fff == 0) 0x8001 else center - 1, if (center & 0x7fff == 0) 1 else if (center == 65535) center else center + 1 };
}

pub fn code(v: f32, scale: f32, bias: f32) u32 {
    @setFloatMode(.strict);
    const difference: f32 = v - bias;
    const z = @max(0, @min(15, difference / scale));
    const floor = @floor(z);
    const fraction = z - floor;
    return @intFromFloat(if (fraction > 0.5 or (fraction == 0.5 and @mod(floor, 2) != 0)) floor + 1 else floor);
}

pub fn encode(row: *const [64]u16, scale: u16, bias: u16) Pack {
    var result: Pack = .{ .words = @splat(0), .scale = scale, .bias = bias };
    for (row, 0..) |x, j| result.words[j / 8] |= code(value(x), value(scale), value(bias)) << @intCast(4 * (j % 8));
    return result;
}

pub fn score(row: *const [64]u16, importance: *const [64]f64, scale: u16, bias: u16) f64 {
    @setFloatMode(.strict);
    var error_value: f64 = 0;
    for (row, importance) |x, weight| {
        const q: f64 = @floatFromInt(code(value(x), value(scale), value(bias)));
        const delta = (q * @as(f64, value(scale)) + @as(f64, value(bias))) - @as(f64, value(x));
        error_value += weight * (delta * delta);
    }
    return error_value;
}

pub fn coefficients(row: *const [64]u16, importance: *const [64]f64, scale: u16, bias: u16) ?[2]f64 {
    @setFloatMode(.strict);
    var total: f64 = 0;
    var sq: f64 = 0;
    var sx: f64 = 0;
    var sqq: f64 = 0;
    var sqx: f64 = 0;
    for (row, importance) |x, weight| {
        const v: f64 = value(x);
        const q: f64 = @floatFromInt(code(value(x), value(scale), value(bias)));
        total += weight;
        sq += weight * q;
        sx += weight * v;
        sqq += weight * (q * q);
        sqx += weight * (q * v);
    }
    const determinant = total * sqq - sq * sq;
    if (!(total > 0) or !(determinant > 0) or !std.math.isFinite(determinant)) return null;
    const s = (total * sqx - sq * sx) / determinant;
    const b = (sx - s * sq) / total;
    if (!(s > 0) or !std.math.isFinite(s) or !std.math.isFinite(b)) return null;
    return .{ s, b };
}

pub fn fit(row: *const [64]u16, importance: *const [64]f64, baseline: Pack) !Result {
    @setFloatMode(.strict);
    if (!(value(baseline.scale) > 0) or !std.math.isFinite(value(baseline.scale)) or
        !std.math.isFinite(value(baseline.bias))) return error.AffineActivationBaseline;
    var total: f64 = 0;
    for (row, importance) |x, weight| {
        if (!std.math.isFinite(value(x))) return error.AffineActivationInput;
        if (!std.math.isFinite(weight) or weight < 0) return error.AffineActivationMoment;
        total += weight;
    }
    if (!std.meta.eql(baseline, encode(row, baseline.scale, baseline.bias))) return error.AffineActivationBaseline;
    const before = score(row, importance, baseline.scale, baseline.bias);
    if (!std.math.isFinite(total) or !std.math.isFinite(before)) return error.AffineActivationObjective;
    var result: Result = .{ .pack = baseline, .before = before, .after = before, .rounds = 0 };
    if (total == 0) return result;
    for (0..2) |iteration| {
        result.rounds = @intCast(iteration + 1);
        const prior = result.pack;
        const fitted = coefficients(row, importance, prior.scale, prior.bias) orelse break;
        const scales = neighbors(bf16(@floatCast(fitted[0])));
        const biases = neighbors(bf16(@floatCast(fitted[1])));
        for (scales) |s| {
            if (!(value(s) > 0) or !std.math.isFinite(value(s))) continue;
            for (biases) |b| {
                if (!std.math.isFinite(value(b))) continue;
                const candidate = score(row, importance, s, b);
                if (candidate < result.after) {
                    result.after = candidate;
                    result.pack = encode(row, s, b);
                }
            }
        }
        if (std.meta.eql(prior, result.pack)) break;
    }
    return result;
}

test "affine input moments: exact affine optimum and unchanged zero-weight pack" {
    var row: [64]u16 = undefined;
    for (&row, 0..) |*v, i| v.* = bf16(@floatFromInt(i % 4));
    const baseline = encode(&row, bf16(0.25), bf16(-0.125));
    const weights: [64]f64 = @splat(1);
    const result = try fit(&row, &weights, baseline);
    try std.testing.expectEqual(@as(f64, 1), result.before);
    try std.testing.expectEqual(@as(f64, 0), result.after);
    try std.testing.expectEqual(bf16(0.25), result.pack.scale);
    try std.testing.expectEqual(bf16(0), result.pack.bias);
    for (result.pack.words) |word| try std.testing.expectEqual(@as(u32, 0xc840c840), word);
    const zero: [64]f64 = @splat(0);
    const unchanged = try fit(&row, &zero, baseline);
    try std.testing.expect(std.meta.eql(baseline, unchanged.pack));
    try std.testing.expectEqual(@as(u32, 0), unchanged.rounds);
}

test "affine input moments: weighted normal equation agrees with exact rational coefficients" {
    var row: [64]u16 = undefined;
    var weights: [64]f64 = undefined;
    const values = [_]f32{ 0, 1.03125, 1.96875, 3 };
    const importance = [_]f64{ 1, 1, 8, 1 };
    for (&row, &weights, 0..) |*v, *w, i| {
        v.* = bf16(values[i % 4]);
        w.* = importance[i % 4];
    }
    const baseline = encode(&row, bf16(0.25), bf16(0));
    const normal = coefficients(&row, &weights, baseline.scale, baseline.bias).?;
    const exact_s: f64 = 1959.0 / 7936.0;
    const exact_b: f64 = 3.0 / 992.0;
    try std.testing.expectApproxEqAbs(exact_s, normal[0], 1e-15);
    try std.testing.expectApproxEqAbs(exact_b, normal[1], 1e-15);
    const result = try fit(&row, &weights, baseline);
    try std.testing.expect(result.after < result.before);
}

test "affine input moments: signed-zero neighbors and invalid moments cannot become a fit" {
    try std.testing.expectEqual([3]u16{ 0, 0x8001, 1 }, neighbors(0));
    try std.testing.expectEqual([3]u16{ 0x8000, 0x8001, 1 }, neighbors(0x8000));
    const row: [64]u16 = @splat(bf16(1));
    var weights: [64]f64 = @splat(1);
    const baseline = encode(&row, bf16(0.25), bf16(0));
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        weights[17] = bad;
        try std.testing.expectError(error.AffineActivationMoment, fit(&row, &weights, baseline));
    }
}
