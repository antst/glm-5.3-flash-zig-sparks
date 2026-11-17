//! Startup-only Q8 scale refinement; the original codes/scale remain a candidate in each group.
const std = @import("std");
const legacy = @import("cuda_qpack.zig");

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

fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn code(v: f32, scale: f32) i32 {
    if (scale == 0) return 0;
    const x = @max(-127, @min(127, v / scale));
    const f = @floor(x);
    const d = x - f;
    return @intFromFloat(if (d > 0.5 or (d == 0.5 and @mod(f, 2) != 0)) f + 1 else f);
}

fn score(row: []const u16, scale: f32) f64 {
    var err: f64 = 0;
    for (row) |h| {
        const v = value(h);
        const delta = @as(f64, @floatFromInt(code(v, scale))) * scale - v;
        err += delta * delta;
    }
    return err;
}

pub fn require(s: anytype, world: usize, vision: bool) !void {
    if (!s.q8_mse) return;
    if (world != 2 or vision or s.dense_bits != 8 or s.head_bits != 16 or !s.tiled or
        s.q4_affine_mse or s.pages or s.kv_host != 0 or s.verify_graphs) return error.Q8MseScope;
}

/// Validate before writing, then keep legacy bytes unless a lower FP64 error is found.
pub fn quantize(src: []const u16, n: usize, k: usize, words: []u32, scales: []u16) !Stats {
    @setFloatMode(.strict);
    if (n == 0 or k == 0 or k % 64 != 0) return error.Q8MseShape;
    const count = std.math.mul(usize, n, k) catch return error.Q8MseShape;
    if (src.len != count or words.len != count / 4 or scales.len != count / 64) return error.Q8MseShape;
    const input = std.mem.sliceAsBytes(src);
    const codes = std.mem.sliceAsBytes(words);
    const factors = std.mem.sliceAsBytes(scales);
    if (overlap(input, codes) or overlap(input, factors) or
        overlap(codes, factors)) return error.Q8MseAlias;
    for (src) |h| if (!std.math.isFinite(value(h))) return error.Q8MseWeight;
    // FP16 scale overflow is not a valid existing GPU pack; refuse before publishing any codes.
    for (0..count / 64) |g| {
        var amax: f32 = 0;
        for (src[g * 64 ..][0..64]) |h| amax = @max(amax, @abs(value(h)));
        const s: f16 = @floatCast(amax / 127);
        if (!std.math.isFinite(s)) return error.Q8MseWeight;
    }
    legacy.quantize(8, src, n, k, words, scales);
    var result: Stats = .{};
    for (scales, 0..) |*stored, g| {
        const row = src[g * 64 ..][0..64];
        var best = @as(f32, @as(f16, @bitCast(stored.*)));
        var err = score(row, best);
        result.baseline_error += err;
        const original = stored.*;
        for (0..2) |_| {
            var dot: f64 = 0;
            var norm: f64 = 0;
            for (row) |h| {
                const v = value(h);
                const q: f64 = @floatFromInt(code(v, best));
                dot += @as(f64, v) * q;
                norm += q * q;
            }
            if (norm == 0) break;
            const fitted: f16 = @floatCast(dot / norm);
            const bits: u16 = @bitCast(fitted);
            if (bits == 0 or bits >= 0x7c00) break;
            const candidates = [_]u16{ bits, bits - 1, bits + 1 };
            for (candidates) |candidate| {
                if (candidate == 0 or candidate >= 0x7c00) continue;
                const sc: f32 = @as(f16, @bitCast(candidate));
                const e = score(row, sc);
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
                const q: u32 = @intCast(code(value(h), best) + 128);
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
