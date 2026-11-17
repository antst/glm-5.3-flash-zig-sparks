const std = @import("std");
const cuda = @import("cuda");
const seg = @import("cuda_kda_segments.zig");
const n = 32 * 128;
const pitch = 3 * n + 256 + 32;

fn view(slot: usize, bytes: usize) seg.View {
    return .{ .ptr = 0x10000000 + slot * 0x1000000, .len = bytes };
}

fn parameters() seg.Parameters {
    return .{ .heads = 32, .p_stride = pitch, .b_off = 3 * n + 256, .a_stride = n, .g_stride = n, .conv_w = view(0,
        3 * n * 4 * 4), .a_log = view(1, 32 * 4), .dt_bias = view(2, n * 4), .norm_w = view(3, 128 * 2), .lower =
        -3, .eps = 1e-6 };
}

fn part(i: usize, row0: usize, rows: usize) seg.Segment {
    const at = 16 + i * 8;
    return .{ .row0 = row0, .rows = rows, .p = view(at, 8 * pitch * 2), .conv = view(at + 1, 3 * 3 * n * 2),
        .state_in = view(at + 2, 32 * 128 * 128 * 4), .state_out = view(at + 3, 32 * 128 * 128 * 4), .k = view(at +
        4, 8 * n * 4), .v = view(at + 5, 8 * n * 2), .g = view(at + 6, 8 * n * 4), .b = view(at + 7, 8 * 32 * 4) };
}

fn storage() seg.Storage {
    return .{ .table = view(96, 8 * @sizeOf(seg.Row)), .a = view(97, 64 * n * 2), .gate = view(98, 64 * n * 2), .q =
        view(99, 64 * n * 4), .y = view(100, 64 * n * 2), .out = view(101, 64 * n * 2) };
}

test "KDA TP2: H32 actual launch ABI and every admitted segment count use full shared scratch" {
    const Fake = struct {
        var calls: usize = 0;
        var count: u32 = 0;
        var most: u32 = 0;
        var fail_at: usize = 0;
        fn integer(p: ?[*]?*anyopaque, at: usize) i32 {
            return @as(*align(1) const i32, @ptrCast(p.?[at].?)).*;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32, shared: u32, _:
            cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(integer(args, 1) == count and integer(args, 2) == 32);
            std.debug.assert(z == count and by == 1 and bz == 1 and shared == 0);
            const which = @intFromPtr(f);
            if (which == 2) {
                std.debug.assert(x == 32 and y == 8 and bx == 128);
            } else {
                std.debug.assert(x == most and y == 32 and bx == (if (which == 1) @as(u32, 512) else 128));
                if (which == 1) std.debug.assert(integer(args, 3) == pitch and integer(args, 4) == 3 * n + 256);
            }
            calls += 1;
            return if (calls == fail_at) cuda.abi.error_not_ready else 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const kernels: seg.Kernels = .{ .prep = .{ .d = &d, .handle = @ptrFromInt(1) }, .step = .{ .d = &d, .handle =
        @ptrFromInt(2) }, .out = .{ .d = &d, .handle = @ptrFromInt(3) } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var parts: [8]seg.Segment = undefined;
    for (1..9) |count| for (0..3) |style| {
        var rows: usize = 0;
        for (parts[0..count], 0..) |*s, i| {
            const length = switch (style) {
                0 => 1,
                1 => 8,
                else => (i * 5 + 3) % 8 + 1,
            };
            s.* = part(i, rows, length);
            rows += length;
        }
        const plan = try seg.Plan.make(parts[0..count], parameters(), .{});
        try std.testing.expectEqual(rows, plan.rows);
        Fake.count = @intCast(count);
        Fake.most = @intCast(plan.most);
        Fake.calls = 0;
        Fake.fail_at = 0;
        try kernels.launch(stream, &plan, storage());
        try std.testing.expectEqual(@as(usize, 3), Fake.calls);
        Fake.calls = 0;
        Fake.fail_at = 2;
        try std.testing.expectError(error.NotReady, kernels.launch(stream, &plan, storage()));
        try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    };
    const plan = try seg.Plan.make(&.{part(0, 0, 8)}, parameters(), .{});
    var bad = storage();
    bad.q.len = 8 * 16 * 128 * 4;
    Fake.calls = 0;
    try std.testing.expectError(error.KdaStorage, kernels.launch(stream, &plan, bad));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    var short = part(0, 0, 8);
    short.state_in.len = 16 * 128 * 128 * 4;
    try std.testing.expectError(error.KdaStorage, seg.Plan.make(&.{short}, parameters(), .{}));
}
