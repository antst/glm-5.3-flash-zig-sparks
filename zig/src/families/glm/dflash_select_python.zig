//! Explicit finite-input diagnostic of the frozen Python selector's ordering, score and beta2 confidence.
const std = @import("std");
const TOPK = 16;
const RANK = 256;
const CW = 32;
const Candidate = struct { id: u32, value: f32 };

fn bf(x: u16) f64 {
    const f: f32 = @bitCast(@as(u32, x) << 16);
    return f;
}

/// The download occupied the first half of this same host allocation; expand backwards without alias loss.
pub fn widen(values: []f32) void {
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(values));
    var i = values.len;
    while (i != 0) {
        i -= 1;
        values[i] = @bitCast(@as(u32, halves[i]) << 16);
    }
}

fn before(a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

pub fn select(dr: anytype, stream: usize, rows: usize, req: anytype) void {
    var prev: usize = req.anchor;
    for (1..8) |t| {
        const row = stream * 8 + t;
        var candidates: [TOPK]Candidate = undefined;
        var n: usize = 0;
        for (0..dr.world) |rank| {
            const block = dr.cand[(rank * rows + row) * CW ..][0..CW];
            for (0..TOPK) |j| {
                if (block[j] < 0) continue;
                const c: Candidate = .{ .id = @intCast(block[j]), .value = @bitCast(block[TOPK + j]) };
                var at: usize = 0;
                while (at < n and !before(c, candidates[at])) : (at += 1) {}
                if (at == TOPK) continue;
                if (n < TOPK) n += 1;
                var end = n - 1;
                while (end > at) : (end -= 1) candidates[end] = candidates[end - 1];
                candidates[at] = c;
            }
        }
        if (n == 0) {
            for (t - 1..7) |j| {
                req.out.?[j] = @intCast(prev);
                if (req.probs) |p| p[j] = 0;
            }
            return;
        }
        var pg: [RANK]f64 = undefined;
        for (&pg, 0..) |*p, k| p.* = bf(dr.pred[prev * RANK + k]) * @as(f64, dr.gh[row * RANK + k]);
        var scores: [TOPK]f64 = undefined;
        var best: usize = 0;
        for (candidates[0..n], 0..) |c, j| {
            var dot: f64 = 0;
            for (pg, 0..) |p, k| dot += p * bf(dr.succ[@as(usize, c.id) * RANK + k]);
            scores[j] = @as(f64, c.value) + 0.6 * dot;
            if (scores[j] > scores[best]) best = j;
        }
        var z: f64 = 0;
        for (scores[0..n]) |s| z += @exp((s - scores[best]) / 2.0);
        prev = candidates[best].id;
        req.out.?[t - 1] = @intCast(prev);
        if (req.probs) |p| p[t - 1] = 1.0 / z;
    }
}

test "Python draft diagnostic: BF16 host expansion preserves all bits and aliases safely" {
    var values: [65536]f32 = undefined;
    const halves = std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(&values));
    for (0..65536) |i| halves[i] = @intCast(i);
    widen(&values);
    for (values, 0..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i)) << 16, @as(u32, @bitCast(v)));
}

test "Python draft diagnostic: actual rank stride, beta2 and coefficient with changing predecessors" {
    const Fake = struct { world: usize, cand: []i32, gh: []f32, pred: []u16, succ: []u16 };
    const Request = struct { anchor: u32, out: ?[]u32, probs: ?[]f64 };
    const rows = 16;
    var words: [4 * rows * CW]i32 = @splat(-1);
    var gh: [rows * RANK]f32 = @splat(0);
    var pred: [3 * RANK]u16 = @splat(0);
    var succ: [64 * RANK]u16 = @splat(0);
    pred[0] = 0x3f80;
    pred[RANK] = 0x3f80;
    succ[2 * RANK] = 0x3f80;
    for (0..rows) |row| {
        gh[row * RANK] = 1;
        words[row * CW] = 1;
        words[row * CW + TOPK] = @bitCast(@as(f32, 0.75));
        words[(3 * rows + row) * CW] = 2;
        words[(3 * rows + row) * CW + TOPK] = @bitCast(@as(f32, 0));
    }
    var dr: Fake = .{ .world = 4, .cand = &words, .gh = &gh, .pred = &pred, .succ = &succ };
    for (0..2) |stream| {
        var out: [7]u32 = undefined;
        var probs: [7]f64 = undefined;
        select(&dr, stream, rows, Request{ .anchor = 0, .out = &out, .probs = &probs });
        for (out, probs) |id, p| {
            try std.testing.expectEqual(@as(u32, 1), id);
            try std.testing.expectApproxEqAbs(1.0 / (1.0 + @exp(-0.15 / 2.0)), p, 1e-15);
        }
    }
    // Equal logit membership is selected by token ID, including the cutoff after all four ranks.
    @memset(&pred, 0);
    for (0..4) |rank| for (0..rows) |row| for (0..16) |j| {
        words[(rank * rows + row) * CW + j] = @intCast(63 - rank * 16 - j);
        words[(rank * rows + row) * CW + TOPK + j] = @bitCast(@as(f32, 0));
    };
    var out: [7]u32 = undefined;
    var probs: [7]f64 = undefined;
    select(&dr, 0, rows, Request{ .anchor = 0, .out = &out, .probs = &probs });
    for (out, probs) |id, p| {
        try std.testing.expectEqual(@as(u32, 0), id);
        try std.testing.expectEqual(@as(f64, 1.0 / 16.0), p);
    }
}
