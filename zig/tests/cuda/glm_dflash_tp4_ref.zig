//! TP4 own-head byte layouts and an independent softmax diagnostic; no device or model state.
const std = @import("std");
const c = @import("core").contract.spec;
pub const hd = 128;
pub const block = 8;
pub const chunks = 16;
pub const win = 2048;
pub const full_q = 32;
pub const full_kv = 8;
pub const local_q = 8;
pub const local_kv = 2;
pub const layers = 2;
pub const ring_cells = layers * 2 * win * full_kv * hd;
pub const Case = struct { start: i64, from: i64, layer: usize };
pub const cases = [_]Case{
    .{ .start = 0, .from = 0, .layer = 0 },
    .{ .start = 7, .from = 0, .layer = 1 },
    .{ .start = 2051, .from = 0, .layer = 1 },
    .{ .start = 4100, .from = 4097, .layer = 0 },
};

pub fn counts(start: i64, from: i64, j: usize) struct { lo: i64, context: usize, total: usize, piece: usize } {
    const lo = @max(0, @max(from, start + @as(i64, @intCast(j)) - (win - 1)));
    const ctx: usize = @intCast(@max(0, start - lo));
    return .{ .lo = lo, .context = ctx, .total = ctx + block, .piece = (ctx + block + chunks - 1) / chunks };
}

/// Rank-local Q, K and V rows packed from the original full fused rows; tail padding is caller-owned.
pub fn compact(dst: []u16, ld: usize, src: []const u16, full_ld: usize, rows: usize, rank: usize) !void {
    if (rank >= 4 or ld < 1536 or full_ld < 6144 or dst.len < rows * ld or src.len < rows * full_ld) return error.Geometry;
    for (0..rows) |r| {
        @memcpy(dst[r * ld ..][0..1024], src[r * full_ld + rank * 1024 ..][0..1024]);
        @memcpy(dst[r * ld + 1024 ..][0..256], src[r * full_ld + 4096 + rank * 256 ..][0..256]);
        @memcpy(dst[r * ld + 1280 ..][0..256], src[r * full_ld + 5120 + rank * 256 ..][0..256]);
    }
}

/// Extract one contiguous head share per row from q/k/v, chunk scalars, chunk vectors or final attention.
pub fn select(dst: []u8, src: []const u8, rows: usize, full_heads: usize, local_heads: usize, rank: usize, head_bytes: usize) !void {
    if (rank >= 4 or full_heads != local_heads * 4 or dst.len != rows * local_heads * head_bytes or src.len != rows * full_heads * head_bytes) return error.Geometry;
    for (0..rows) |r| @memcpy(dst[r * local_heads * head_bytes ..][0 .. local_heads * head_bytes], src[(r * full_heads + rank * local_heads) * head_bytes ..][0 .. local_heads * head_bytes]);
}

pub fn grid(ok: bool, qkv_heads: u32, part_heads: u32, merge_heads: u32) !void {
    if (!ok or qkv_heads != 10 or part_heads != 2 or merge_heads != 8) return error.Geometry;
}

pub const Diagnostic = struct { max_abs: f64 = 0, max_rel: f64 = 0, outputs: usize = 0 };

/// FP64 diagnostic of attention AFTER qkv norm/rope; no bound/pass claim for nonlinear fp32 arithmetic.
pub fn attention(q: []const u16, kb: []const u16, vb: []const u16, ring: []const u16, got: []const u16, rank: usize, case: Case, scale: f32) !Diagnostic {
    if (rank >= 4 or q.len != block * 1024 or kb.len != block * 256 or vb.len != block * 256 or got.len != q.len or ring.len != ring_cells) return error.Geometry;
    var stats: Diagnostic = .{};
    const kvw = full_kv * hd;
    const base = case.layer * 2 * win * kvw;
    for (0..block) |r| for (0..local_q) |h| {
        const ct = counts(case.start, case.from, r);
        var scores: [win + block]f64 = undefined;
        var maximum: f64 = -std.math.inf(f64);
        for (scores[0..ct.total], 0..) |*score, n| {
            var dot: f64 = 0;
            for (0..hd) |d| {
                const key = if (n < ct.context) ring[base + @as(usize, @intCast(@mod(ct.lo + @as(i64, @intCast(n)), win))) * kvw + (rank * 2 + h / 4) * hd + d] else kb[(n - ct.context) * 256 + (h / 4) * hd + d];
                dot += @as(f64, c.fromBf16(q[r * 1024 + h * hd + d])) * @as(f64, c.fromBf16(key));
            }
            score.* = dot * @as(f64, scale);
            maximum = @max(maximum, score.*);
        }
        var denom: f64 = 0;
        for (scores[0..ct.total]) |*score| {
            score.* = @exp(score.* - maximum);
            denom += score.*;
        }
        for (0..hd) |d| {
            var numerator: f64 = 0;
            for (scores[0..ct.total], 0..) |weight, n| {
                const value = if (n < ct.context) ring[base + win * kvw + @as(usize, @intCast(@mod(ct.lo + @as(i64, @intCast(n)), win))) * kvw + (rank * 2 + h / 4) * hd + d] else vb[(n - ct.context) * 256 + (h / 4) * hd + d];
                numerator += weight * @as(f64, c.fromBf16(value));
            }
            const want = numerator / denom;
            const have: f64 = c.fromBf16(got[r * 1024 + h * hd + d]);
            if (!std.math.isFinite(have) or !std.math.isFinite(want)) return error.NonfiniteDiagnostic;
            const delta = @abs(have - want);
            stats.max_abs = @max(stats.max_abs, delta);
            stats.max_rel = @max(stats.max_rel, delta / @max(@abs(want), 1e-30));
            stats.outputs += 1;
        }
    };
    return stats;
}
