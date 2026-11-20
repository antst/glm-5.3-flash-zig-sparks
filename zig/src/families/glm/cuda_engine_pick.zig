//! The existing rank-gathered greedy pick, kept separate from engine lifetime and prompt diagnostics.
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const chance = @import("cuda_mtp_chance.zig");
const Engine = @import("cuda_engine.zig").Engine;
const pair_bytes = 16;
pub fn greedy(e: *Engine, logits: u64) !u32 {
    return (try pick(e, logits, false)).token;
}

/// greedy with the pick's chance from every rank's softmax sum (tf_glm_top1_sums), for --mtp-confidence's drafts.
pub fn greedyChance(e: *Engine, logits: u64) !chance.Pick {
    return pick(e, logits, true);
}

fn pick(e: *Engine, logits: u64, sums: bool) !chance.Pick {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda_pdl.launch(if (sums) e.m.glue.top1_sums else e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * pair_bytes]);
    return chance.pick(all[0..world], sums);
}

/// Existing top1 blocks followed by one gather/readback for ordinary greedy verification.
pub fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    return run(@import("cuda_forward.zig").window_rows, e, logits, rows, out);
}

pub fn greedySharedRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    return run(@import("cuda_engine.zig").shared_rows, e, logits, rows, out);
}

fn run(comptime cap: usize, e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    if (rows == 0 or rows > cap or out.len < rows) return error.WindowTooWide;
    const world = e.share.world;
    if (world == 0 or world > ranks.max_world or e.share.rank >= world) return error.BadWorld;
    const bytes = rows * pair_bytes;
    if (e.pairs.len < world * bytes) return error.PickScratchTooSmall;
    // This call's rank stride is its actual rows, not the allocation's maximum rows.
    const mine = e.pairs.ptr + e.share.rank * bytes;
    for (0..rows) |row| {
        var a: cuda.Args = .{};
        a.add(logits + row * e.share.vocab.len() * 2);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(mine + row * pair_bytes);
        try cuda_pdl.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    }
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world * cap][4]u32 = undefined;
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.sliceAsBytes(all[0 .. world * rows]));
    try selectRows(cap, all[0 .. world * rows], world, rows, out);
}

/// Retain chance.pick's rank order; the flat gather is [world][actual_rows][4] words.
fn selectRows(comptime cap: usize, all: []const [4]u32, world: usize, rows: usize, out: []u32) !void {
    if (rows == 0 or rows > cap or out.len < rows) return error.WindowTooWide;
    if (world == 0 or world > ranks.max_world or all.len != world * rows) return error.BadWorld;
    var rank_words: [ranks.max_world][4]u32 = undefined;
    for (0..rows) |row| {
        for (0..world) |rank| rank_words[rank] = all[rank * rows + row];
        out[row] = chance.pick(rank_words[0..world], false).token;
    }
}

test "greedy batching: host view uses actual-row rank strides and writes only requested outputs" {
    const cap = @import("cuda_forward.zig").window_rows;
    var records: [ranks.max_world * cap][4]u32 = undefined;
    for (1..ranks.max_world + 1) |world| for (1..cap + 1) |rows| {
        for (0..world) |rank| for (0..rows) |row| {
            const value: f32 = if (rank == row % world) 1 else 0;
            records[rank * rows + row] = .{ @bitCast(value), @intCast(1000 * row + rank), 0, 0 };
        };
        var out: [cap + 2]u32 = @splat(0xdeadbeef);
        try selectRows(cap, records[0 .. world * rows], world, rows, out[1 .. rows + 1]);
        try std.testing.expectEqual(@as(u32, 0xdeadbeef), out[0]);
        try std.testing.expectEqual(@as(u32, 0xdeadbeef), out[rows + 1]);
        for (0..rows) |row| try std.testing.expectEqual(@as(u32, @intCast(1000 * row + row % world)), out[row + 1]);
    };
}

test "greedy batching: host view preserves original NaN rank order and signed-zero tie rule" {
    const records = [_][4]u32{
        .{ @bitCast(std.math.nan(f32)), 99, 0, 0 }, .{ @bitCast(@as(f32, -0.0)), 7, 0, 0 },
        .{ @bitCast(std.math.inf(f32)), 1, 0, 0 },  .{ @bitCast(@as(f32, 0.0)), 4, 0, 0 },
        .{ @bitCast(@as(f32, 10)), 2, 0, 0 },       .{ @bitCast(@as(f32, -0.0)), 9, 0, 0 },
    };
    var out: [2]u32 = undefined;
    try selectRows(8, &records, 3, 2, &out);
    try std.testing.expectEqualSlices(u32, &.{ 99, 4 }, &out);
    try std.testing.expectError(error.WindowTooWide, selectRows(8, &records, 3, 0, &out));
    try std.testing.expectError(error.WindowTooWide, selectRows(8, &records, 3, 2, out[0..1]));
    try std.testing.expectError(error.BadWorld, selectRows(8, &records, 0, 2, &out));
    try std.testing.expectError(error.BadWorld, selectRows(8, records[0..5], 3, 2, &out));
}
