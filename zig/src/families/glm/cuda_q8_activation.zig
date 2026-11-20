//! Startup-only input-moment weighting, with the same group64 Q8 representation and inference kernels.
const std = @import("std");
const mse = @import("cuda_q8_mse.zig");

pub const Stats = struct {
    groups: usize = 0,
    improved: usize = 0,
    baseline_error: f64 = 0,
    selected_error: f64 = 0,
};

fn overlap(a: []const u8, b: []const u8) bool {
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}

pub fn score(row: []const u16, importance: []const f64, scale: f32) f64 {
    @setFloatMode(.strict);
    var err: f64 = 0;
    for (row, importance) |h, weight| {
        const v = mse.value(h);
        const delta = @as(f64, @floatFromInt(mse.code(v, scale))) * scale - v;
        err += weight * (delta * delta);
    }
    return err;
}

/// Retain the existing MSE pack unless a strictly lower independent weighted objective is found.
pub fn quantize(src: []const u16, n: usize, k: usize, importance: []const f64, words: []u32, scales: []u16) !Stats {
    @setFloatMode(.strict);
    if (k == 0 or k % 64 != 0 or importance.len != k) return error.Q8ActivationShape;
    const ib = std.mem.sliceAsBytes(importance);
    if (overlap(ib, std.mem.sliceAsBytes(src)) or overlap(ib, std.mem.sliceAsBytes(words)) or
        overlap(ib, std.mem.sliceAsBytes(scales))) return error.Q8ActivationAlias;
    for (importance) |v| if (!std.math.isFinite(v) or v < 0) return error.Q8ActivationMoment;
    _ = try mse.quantize(src, n, k, words, scales);
    var result: Stats = .{};
    const per = k / 64;
    for (scales, 0..) |*stored, g| {
        const row = src[g * 64 ..][0..64];
        const weights = importance[(g % per) * 64 ..][0..64];
        const original = stored.*;
        var best: f32 = @as(f16, @bitCast(original));
        var err = score(row, weights, best);
        result.baseline_error += err;
        var amax: f32 = 0;
        for (row) |h| amax = @max(amax, @abs(mse.value(h)));
        const raw: f16 = @floatCast(amax / 127);
        const raw_error = score(row, weights, raw);
        if (raw_error < err) {
            err = raw_error;
            best = raw;
            stored.* = @bitCast(raw);
        }
        for (0..2) |_| {
            var dot: f64 = 0;
            var norm: f64 = 0;
            for (row, weights) |h, weight| {
                const v = mse.value(h);
                const q: f64 = @floatFromInt(mse.code(v, best));
                dot += weight * (@as(f64, v) * q);
                norm += weight * (q * q);
            }
            if (norm == 0) break;
            const fitted: f16 = @floatCast(dot / norm);
            const bits: u16 = @bitCast(fitted);
            if (bits == 0 or bits >= 0x7c00) break;
            for ([_]u16{ bits, bits - 1, bits + 1 }) |candidate| {
                if (candidate == 0 or candidate >= 0x7c00) continue;
                const sc: f32 = @as(f16, @bitCast(candidate));
                const e = score(row, weights, sc);
                if (e < err) {
                    err = e;
                    best = sc;
                    stored.* = candidate;
                }
            }
        }
        if (stored.* != original) {
            result.improved += 1;
            for (row, 0..) |h, i| {
                const q: u32 = @intCast(mse.code(mse.value(h), best) + 128);
                const at = g * 16 + i / 4;
                const shift: u5 = @intCast(8 * (i % 4));
                words[at] = (words[at] & ~(@as(u32, 255) << shift)) | (q << shift);
            }
        }
        result.groups += 1;
        result.selected_error += err;
    }
    return result;
}
