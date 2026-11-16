//! Stage policy preserves compute/gather order in capture, replay, eager fallback and failure.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("dflash_graph.zig");
const gc = cuda.graph_cache;
const stream: cuda.Stream = .{ .d = undefined, .handle = null };
const binding: graph.Binding = .{ .engine = 1, .weights = 2, .scratch = 3 };

const Trace = struct {
    events: [22]u8 = undefined,
    n: usize = 0,
    gap: usize = 0,
    captures: usize = 0,
    freed: usize = 0,
    tails: usize = 0,
    fail_capture: ?usize = null,
    fail_gather: ?usize = null,
    fail_launch: bool = false,
    entries: [200]u8 = undefined,

    fn add(t: *Trace, value: u8) void {
        t.events[t.n] = value;
        t.n += 1;
    }
    fn capture(ptr: *anyopaque, _: cuda.Stream, body: gc.Body) anyerror!*anyopaque {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        const stage: *Stage = @ptrCast(@alignCast(body.ctx));
        if (t.fail_capture == stage.which) return error.RecordedCaptureFailure;
        const i = t.captures;
        t.captures += 1;
        t.entries[i] = stage.which;
        return &t.entries[i];
    }
    fn launch(ptr: *anyopaque, exec: *anyopaque, _: cuda.Stream) anyerror!void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.fail_launch) return error.RecordedLaunchFailure;
        t.add(@as(*u8, @ptrCast(exec)).*);
    }
    fn free(ptr: *anyopaque, _: *anyopaque) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        t.freed += 1;
    }
    fn between(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.fail_gather == t.gap) return error.RecordedGatherFailure;
        t.add(@intCast(10 + t.gap));
        t.gap += 1;
    }
    fn tail(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.gap != 5 and t.gap != 10) return error.WrongGatherCount;
        t.add(99);
        t.tails += 1;
    }
    fn graphs(t: *Trace) graph.Graphs {
        return .{ .driver = undefined, .cache = .{
            .gpa = std.testing.allocator,
            .engine = .{ .ctx = t, .capture = capture, .launch = launch, .free = free },
            .settings = .{ .max = 96, .hold = true },
            .cap = 96,
        } };
    }
    fn run(t: *Trace, g: *graph.Graphs, streams: usize) !void {
        return t.runStages(g, streams, false);
    }
    fn runStages(t: *Trace, g: *graph.Graphs, streams: usize, attention: bool) !void {
        t.n = 0;
        t.gap = 0;
        var contexts: [11]Stage = undefined;
        var stages: [11]gc.Body = undefined;
        const count: usize = if (attention) 11 else 6;
        for (contexts[0..count], stages[0..count], 0..) |*context, *body, i| {
            context.* = .{ .trace = t, .which = @intCast(i) };
            body.* = .{ .ctx = context, .run = Stage.run };
        }
        if (attention) try g.runAttention(stream, streams, binding, stages, .{ .ctx = t, .run = between }, .{ .ctx = t, .run = tail }) else try g.runSplit(stream, streams, binding, stages[0..6].*, .{ .ctx = t, .run = between }, .{ .ctx = t, .run = tail });
    }
};

const Stage = struct {
    trace: *Trace,
    which: u8,
    fn run(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const stage: *Stage = @ptrCast(@alignCast(ptr));
        stage.trace.add(stage.which);
    }
};

const expected = [_]u8{ 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 99 };

test "DFlash graphs: all split stages replay around exactly five eager gathers and a candidate tail" {
    var t: Trace = .{};
    var g = t.graphs();
    defer g.cache.deinit();
    for (0..2) |_| for (1..17) |streams| {
        try t.run(&g, streams);
        try std.testing.expectEqualSlices(u8, &expected, t.events[0..t.n]);
    };
    try std.testing.expectEqual(@as(usize, 96), t.captures);
    try std.testing.expectEqual(@as(usize, 96), g.cache.count());
    try std.testing.expectEqual(@as(usize, 32), t.tails);
}

test "DFlash graphs: split capture failure and cache off preserve the entire eager schedule" {
    var t: Trace = .{ .fail_capture = 2 };
    var g = t.graphs();
    defer g.cache.deinit();
    try t.run(&g, 8);
    try std.testing.expectEqualSlices(u8, &expected, t.events[0..t.n]);
    try std.testing.expect(g.cache.failed);
    try std.testing.expectEqual(@as(usize, 2), t.freed);
    g.cache.settings.on = false;
    try t.run(&g, 4);
    try std.testing.expectEqualSlices(u8, &expected, t.events[0..t.n]);
}

test "DFlash graphs: split gather or launch error suppresses every later compute stage and tail" {
    var t: Trace = .{ .fail_gather = 1 };
    var g = t.graphs();
    defer g.cache.deinit();
    try std.testing.expectError(error.RecordedGatherFailure, t.run(&g, 4));
    try std.testing.expectEqualSlices(u8, &.{ 0, 10, 1 }, t.events[0..t.n]);
    try std.testing.expectEqual(@as(usize, 0), t.tails);
    t.fail_gather = null;
    t.fail_launch = true;
    try std.testing.expectError(error.RecordedLaunchFailure, t.run(&g, 4));
    try std.testing.expectEqual(@as(usize, 0), t.n);
    try std.testing.expectEqual(@as(usize, 0), t.tails);
}

test "DFlash attention: eleven stages replay around ten alternating partial gathers" {
    var t: Trace = .{};
    var g = t.graphs();
    g.cache.settings.max = 176;
    g.cache.cap = 176;
    defer g.cache.deinit();
    for (0..2) |_| for (1..17) |streams| {
        try t.runStages(&g, streams, true);
        try std.testing.expectEqual(@as(usize, 22), t.n);
        for (0..11) |stage| {
            try std.testing.expectEqual(@as(u8, @intCast(stage)), t.events[stage * 2]);
            if (stage < 10) try std.testing.expectEqual(@as(u8, @intCast(stage + 10)), t.events[stage * 2 + 1]);
        }
        try std.testing.expectEqual(@as(u8, 99), t.events[21]);
    };
    try std.testing.expectEqual(@as(usize, 176), t.captures);
    try std.testing.expectEqual(@as(usize, 176), g.cache.count());
    try std.testing.expectEqual(@as(usize, 32), t.tails);
}
