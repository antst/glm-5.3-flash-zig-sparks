//! Actual H32 argument, scratch and startup-cache tests; no CUDA libraries are opened.
const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense_segments.zig");
const t = std.testing;

const Fake = struct {
    var calls: usize = 0;
    var rows: usize = 0;
    fn u(p: ?[*]?*anyopaque, i: usize) u64 {
        return @as(*align(1) const u64, @ptrCast(p.?[i].?)).*;
    }
    fn launch(_: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32, smem: u32, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(x == rows and by == 1 and bz == 1 and u(p, 5) == rows and u(p, 6) == 32);
        if (calls == 0) {
            std.debug.assert(y == 2 and z == 1 and bx == 256 and smem == @import("cuda_glue.zig").chunksSmem(16));
        } else std.debug.assert(y == 32 and z == 1 and bx == 128 and smem == 0);
        calls += 1;
        return 0;
    }
};

fn plan(lengths: []const usize) !dense.Plan {
    var segs: [8]dense.Segment = undefined;
    for (lengths, 0..) |rows, i| segs[i] = .{ .rows = rows, .host_pos = i * 4, .capacity = 2048, .cache = 0x10000000 + i * 0x02000000, .scales = 0x11000000 + i * 0x02000000, .position = 0x11800000 + i * 0x02000000 };
    return dense.Plan.make(segs[0..lengths.len], .{ .heads = 32, .max_part_rows = 16, .dense_limit = 2048, .scratch_rows = 64, .scratch_chunks = 4 });
}

fn storage(rows: usize) dense.Storage {
    const stat = rows * 32 * 4;
    return .{ .table = .{ .ptr = 0x30000000, .len = rows * 32 }, .qa = .{ .ptr = 0x40000000, .len = rows * 32 * 512 * 2 }, .po = .{ .ptr = 0x50000000, .len = stat * 512 }, .pm = .{ .ptr = 0x60000000, .len = stat }, .pl = .{ .ptr = 0x70000000, .len = stat }, .out = .{ .ptr = 0x80000000, .len = rows * 32 * 512 * 2 } };
}

test "dense H32: actual launch covers both head tiles and full global partial layout" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const k: dense.Kernels = .{ .chunks = .{ .d = &d, .handle = @ptrFromInt(1) }, .chunks_fp8 = .{ .d = &d, .handle = @ptrFromInt(2) }, .merge = .{ .d = &d, .handle = @ptrFromInt(3) } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_][]const usize{ &.{1}, &.{8}, &.{ 3, 6 }, &.{16}, &.{ 16, 16 }, &.{ 16, 16, 16, 16 }, &.{ 8, 8, 8, 8, 8, 8, 8, 8 } }) |lens| {
        const p = try plan(lens);
        try t.expectEqual(@as(usize, 32), p.heads);
        Fake.rows = p.rows;
        Fake.calls = 0;
        try k.launch(stream, &p, storage(p.rows), 1.0 / 16.0);
        try t.expectEqual(@as(usize, 2), Fake.calls);
        var bad = storage(p.rows);
        bad.qa.len /= 2;
        Fake.calls = 0;
        try t.expectError(error.DenseScratch, k.launch(stream, &p, bad, 1.0 / 16.0));
        try t.expectEqual(@as(usize, 0), Fake.calls);
        bad = storage(p.rows);
        bad.po.len /= 2;
        try t.expectError(error.DenseScratch, k.launch(stream, &p, bad, 1.0 / 16.0));
        try t.expectEqual(@as(usize, 0), Fake.calls);
    }
}

test "dense H32: malformed head count and unsupported parts refuse before enqueue" {
    var p = try plan(&.{ 8, 8 });
    p.heads = 24;
    try t.expectError(error.DenseShape, dense.validate(&p, storage(p.rows)));
    try t.expectError(error.DenseRows, plan(&.{17}));
    try t.expectError(error.DenseRows, plan(&.{ 16, 16, 16, 16, 1 }));
}

test "FFN warm wide cache is independent of optional expert fusion" {
    const graph = @import("cuda_nvfp4_graph.zig");
    const Settings = @import("cuda_ranks.zig").Settings;
    const legacy: Settings = .{ .ffn_graphs = true, .verify_rows = 32 };
    try t.expect(!graph.wide(legacy));
    var warm = legacy;
    warm.ffn_warm = true;
    try t.expect(graph.wide(warm));
    try t.expectEqual(@as(u32, 4608), @import("cuda_ffn_graph.zig").capacityFor(graph.wide(warm)));
    const actual = .{ .settings = warm, .drafter = true, .mtp = true, .vision = @as(?*anyopaque, null) };
    try @import("cuda_ffn_warm.zig").require(actual, 2);
    try t.expectError(error.FfnWarmScope, @import("cuda_ffn_warm.zig").require(actual, 1));
    warm.ffn_graphs = false;
    try t.expectError(error.FfnWarmScope, @import("cuda_ffn_warm.zig").require(.{ .settings = warm, .drafter = true, .mtp = true, .vision = @as(?*anyopaque, null) }, 2));
}
