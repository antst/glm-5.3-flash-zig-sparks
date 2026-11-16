//! The actual DFlash graph adapter with a recording cache engine, without kernels or a CUDA context.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("dflash_graph.zig");
const gc = cuda.graph_cache;
const ranks = @import("cuda_ranks.zig");
const s: cuda.Stream = .{ .d = undefined, .handle = null };
const binding: graph.Binding = .{ .engine = 1, .weights = 2, .scratch = 3 };

const Fake = struct {
    captures: usize = 0,
    launches: usize = 0,
    frees: usize = 0,
    live: usize = 0,
    eager: usize = 0,
    tails: usize = 0,
    table: usize = 0,
    observed: usize = 0,
    fail_capture: bool = false,
    fail_launch: bool = false,
    fail_body: bool = false,
    fail_tail: bool = false,
    entries: [16]u8 = @splat(0),

    fn engine(f: *Fake) gc.Engine {
        return .{ .ctx = f, .capture = capture, .launch = launch, .free = free };
    }
    fn capture(p: *anyopaque, _: cuda.Stream, _: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.captures += 1;
        if (f.fail_capture) return error.RecordedCaptureFailure;
        f.live += 1;
        return &f.entries[f.captures - 1];
    }
    fn launch(p: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(p));
        if (f.fail_launch) return error.RecordedLaunchFailure;
        f.launches += 1;
        f.observed = f.table;
    }
    fn free(p: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.frees += 1;
        f.live -= 1;
    }
    fn body(p: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(p));
        if (f.fail_body) return error.RecordedBodyFailure;
        f.eager += 1;
        f.observed = f.table;
    }
    fn tail(p: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(p));
        if (f.fail_tail) return error.RecordedTailFailure;
        f.tails += 1;
        if (f.observed != f.table) return error.StaleTable;
    }
    fn graphs(f: *Fake) graph.Graphs {
        return .{ .driver = undefined, .cache = .{
            .gpa = std.testing.allocator,
            .engine = f.engine(),
            .settings = .{ .max = 16, .min_keep = 16, .hold = true },
            .cap = 16,
        } };
    }
    fn run(f: *Fake, g: *graph.Graphs, streams: usize, b: graph.Binding, split: bool) !void {
        try g.run(s, streams, b, split, .{ .ctx = f, .run = body }, .{ .ctx = f, .run = tail });
    }
};

test "DFlash graphs: all16 shapes capture once, replay fresh tables and keep every tail outside capture" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    for (0..3) |round| for (1..17) |n| {
        f.table = round * 100 + n;
        try f.run(&g, n, binding, false);
        try std.testing.expectEqual(f.table, f.observed);
        try std.testing.expectEqual(round * 16 + n, f.tails);
    };
    try std.testing.expectEqual(@as(usize, 16), f.captures);
    try std.testing.expectEqual(@as(usize, 48), f.launches);
    try std.testing.expectEqual(@as(usize, 0), f.eager);
    try std.testing.expectEqual(@as(usize, 16), g.cache.count());
    g.cache.dropAll();
    try std.testing.expectEqual(@as(usize, 16), f.frees);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

test "DFlash graphs: changed allocation ownership is rejected before any graph or collective tail" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    try f.run(&g, 1, binding, false);
    inline for (.{ "engine", "weights", "scratch" }) |field| {
        var changed = binding;
        @field(changed, field) += 1;
        try std.testing.expectError(error.DrafterGraphStorageChanged, f.run(&g, 1, changed, false));
    }
    try std.testing.expectEqual(@as(usize, 1), f.launches);
    try std.testing.expectEqual(@as(usize, 1), f.tails);
    for ([_]usize{ 0, 17 }) |n| try std.testing.expectError(error.BadDrafterGraphStreams, f.run(&g, n, binding, false));
}

test "DFlash graphs: split MLP and cache-off retain eager compute then the same tail" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    try f.run(&g, 2, binding, true);
    g.cache.settings.on = false;
    try f.run(&g, 4, binding, false);
    try std.testing.expectEqual(@as(usize, 0), f.captures);
    try std.testing.expectEqual(@as(usize, 2), f.eager);
    try std.testing.expectEqual(@as(usize, 2), f.tails);
}

test "DFlash graphs: capture failure drops graphs and runs one eager compute before each tail" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    try f.run(&g, 1, binding, false);
    f.fail_capture = true;
    f.table = 99;
    try f.run(&g, 2, binding, false);
    try f.run(&g, 1, binding, false);
    try std.testing.expect(g.cache.failed);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 2), f.eager);
    try std.testing.expectEqual(@as(usize, 3), f.tails);
}

test "DFlash graphs: launch and eager failures never publish a collective tail; tail errors propagate" {
    var f: Fake = .{ .fail_launch = true };
    var g = f.graphs();
    defer g.cache.deinit();
    try std.testing.expectError(error.RecordedLaunchFailure, f.run(&g, 1, binding, false));
    try std.testing.expectEqual(@as(usize, 0), f.tails);
    f.fail_body = true;
    try std.testing.expectError(error.RecordedBodyFailure, f.run(&g, 1, binding, true));
    try std.testing.expectEqual(@as(usize, 0), f.tails);
    f.fail_body = false;
    f.fail_tail = true;
    try std.testing.expectError(error.RecordedTailFailure, f.run(&g, 1, binding, true));
}

test "DFlash graphs: strict opt-in is fingerprinted before weights load" {
    const Get = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH_GRAPHS")) value else null;
        }
    };
    defer Get.value = null;
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(Get.get, &problem)).dflash_graphs);
    Get.value = "0";
    try std.testing.expect(!(try ranks.Settings.read(Get.get, &problem)).dflash_graphs);
    Get.value = "1";
    const on = try ranks.Settings.read(Get.get, &problem);
    try std.testing.expect(on.dflash_graphs);
    Get.value = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Get.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_GRAPHS: 0 or 1", problem);
    const a = ranks.Fingerprint.of(4, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(ranks.Fingerprint.Field.dflash_graphs, ranks.mismatch(&.{ a, b }).?.field);
}
