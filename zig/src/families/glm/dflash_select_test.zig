//! CPU-only calls to the actual selector; no driver, modules, stream, weights or graph construction.
const std = @import("std");
const df = @import("dflash.zig");
const actual = @import("dflash_select.zig");
const legacy = @import("dflash_select_original_test.zig");
const ranks = @import("cuda_ranks.zig");
const M = 2 * df.BLOCK;

const Fixture = struct {
    cand: [4 * M * df.CW]i32 = @splat(-1),
    gh: [M * df.RANK]f32 = @splat(0),
    pred: [3 * df.RANK]u16 = @splat(0),
    succ: [3 * df.RANK]u16 = @splat(0),

    fn init(f: *Fixture, world: usize) void {
        f.* = .{};
        for (0..3) |i| f.pred[i * df.RANK] = 0x3f80; // BF16 1.0 for every possible predecessor.
        f.succ[2 * df.RANK] = 0x3f80;
        for (0..M) |r| {
            f.gh[r * df.RANK] = 1;
            f.put(0, r, 0, 1, 0.75);
            f.put(world - 1, r, if (world == 1) 1 else 0, 2, 0);
        }
    }

    fn put(f: *Fixture, rank: usize, row: usize, j: usize, id: i32, value: f32) void {
        const at = rank * M * df.CW + row * df.CW;
        f.cand[at + j] = id;
        f.cand[at + df.TOPK + j] = @bitCast(value);
    }

    fn drafter(f: *Fixture, world: usize, on: bool) df.Drafter {
        // select reads exactly these host fields; undefined device owners must remain untouched.
        var dr: df.Drafter = undefined;
        dr.world = world;
        dr.temp = 1;
        dr.edge_06 = on;
        dr.cand = f.cand[0 .. world * M * df.CW];
        dr.gh = &f.gh;
        dr.pred = &f.pred;
        dr.succ = &f.succ;
        return dr;
    }
};

fn req(out: *[7]u32, probs: *[7]f64) df.Req {
    return .{ .slot = 0, .pos = 0, .n = 0, .anchor = 0, .out = out, .probs = probs };
}

fn bf64(x: u16) f64 {
    const v: f32 = @bitCast(@as(u32, x) << 16);
    return v;
}

test "DFlash edge: actual chain flips on0.6 and matches independent old/preweighted FP64 scores" {
    var f: Fixture = .{};
    for ([_]usize{ 1, 2, 4 }) |world| for (0..2) |i| for ([_]bool{ false, true }) |on| {
        f.init(world);
        var dr = f.drafter(world, on);
        for ([_]f64{ 0.5, 1, 2 }) |temp| {
            dr.temp = temp;
            var out: [7]u32 = undefined;
            var probs: [7]f64 = undefined;
            actual.select(&dr, i, M, req(&out, &probs));
            var prev: usize = 0;
            const edge: f64 = if (on) 0.6 else 1;
            for (0..7) |t| {
                const row = i * df.BLOCK + t + 1;
                var dot: f64 = 0;
                var preweighted: f64 = 0;
                for (0..df.RANK) |k| {
                    const pg = bf64(f.pred[prev * df.RANK + k]) * @as(f64, f.gh[row * df.RANK + k]);
                    dot += bf64(f.succ[2 * df.RANK + k]) * pg;
                    preweighted += (bf64(f.succ[2 * df.RANK + k]) * edge) * pg;
                }
                const old_score = edge * dot;
                try std.testing.expectEqual(old_score, preweighted);
                const best: u32 = if (old_score > 0.75) 2 else 1;
                const gap = @abs(old_score - 0.75);
                const chance = 1.0 / (1.0 + @exp(-gap / temp));
                try std.testing.expectEqual(best, out[t]);
                try std.testing.expectApproxEqAbs(chance, probs[t], 1e-7);
                prev = best;
            }
        }
    };
}

test "DFlash edge: OFF is bit-exact to frozen selector for ties NaNs signedzero and changing predecessors" {
    var f: Fixture = .{};
    for ([_]usize{ 1, 2, 4 }) |world| for (0..2) |i| for (0..5) |pattern| {
        f.init(world);
        for (0..M) |r| {
            if (pattern == 1) {
                f.put(0, r, 0, 1, 0);
                f.succ[2 * df.RANK] = 0;
            } else if (pattern == 2) {
                f.put(0, r, 0, 1, std.math.nan(f32));
            } else if (pattern == 3) {
                f.put(0, r, 0, 1, -0.0);
                f.succ[2 * df.RANK] = 0x8000;
                f.gh[r * df.RANK] = -0.0;
            } else if (pattern == 4) {
                f.gh[r * df.RANK] = if (r % 2 == 0) 0.25 else -0.5;
                f.pred[df.RANK] = 0x4000; // BF162: later picks depend on the previous candidate.
            }
        }
        var dr = f.drafter(world, false);
        var out: [7]u32 = undefined;
        var old_out: [7]u32 = undefined;
        var probs: [7]f64 = undefined;
        var old_probs: [7]f64 = undefined;
        actual.select(&dr, i, M, req(&out, &probs));
        legacy.select(&dr, i, M, req(&old_out, &old_probs));
        try std.testing.expectEqualSlices(u32, &old_out, &out);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&old_probs), std.mem.asBytes(&probs));
    };
}

test "DFlash edge: strict defaultOFF startup parse and all-rank disagreement are explicit" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH_EDGE_06")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(!off.dflash_edge_06);
    Env.value = "0";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_edge_06);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(on.dflash_edge_06);
    for ([_][]const u8{ "", "yes", "true", "0.6", "1.0", "2" }) |bad| {
        Env.value = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_DFLASH_EDGE_06: 0 or 1", problem);
    }
    const FP = ranks.Fingerprint;
    var all = [_]FP{
        FP.of(4, 0, off, 8192, true, @splat(0), @splat(0), @splat(0)),
        FP.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expectEqual(ranks.Mismatch{ .field = .dflash_edge_06, .rank = 1 }, ranks.mismatch(&all).?);
    all[0] = FP.of(4, 0, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expect(ranks.mismatch(&all) == null);
    try std.testing.expectEqual(@as(u32, 84), all[0].layout);
}
