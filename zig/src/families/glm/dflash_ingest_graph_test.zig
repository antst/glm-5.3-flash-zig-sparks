//! Ingest graph ownership and bounded shape reuse through the actual adapter, without device work.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("dflash_ingest_graph.zig");
const gc = cuda.graph_cache;
const ranks = @import("cuda_ranks.zig");
const stream: cuda.Stream = .{ .d = undefined, .handle = null };
const binding: @import("dflash_graph.zig").Binding = .{ .engine = 1, .weights = 2, .scratch = 3 };

const Record = struct {
    captures: usize = 0,
    launches: usize = 0,
    freed: usize = 0,
    eager: usize = 0,
    table: u64 = 0,
    seen: u64 = 0,
    fail_capture: bool = false,
    fail_launch: bool = false,
    fail_body: bool = false,
    handles: [64]u8 = undefined,

    fn capture(ptr: *anyopaque, _: cuda.Stream, _: gc.Body) anyerror!*anyopaque {
        const r: *Record = @ptrCast(@alignCast(ptr));
        if (r.fail_capture) return error.RecordedCaptureFailure;
        const i = r.captures;
        r.captures += 1;
        return &r.handles[i];
    }
    fn launch(ptr: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const r: *Record = @ptrCast(@alignCast(ptr));
        if (r.fail_launch) return error.RecordedLaunchFailure;
        r.launches += 1;
        r.seen = r.table;
    }
    fn free(ptr: *anyopaque, _: *anyopaque) void {
        const r: *Record = @ptrCast(@alignCast(ptr));
        r.freed += 1;
    }
    fn body(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const r: *Record = @ptrCast(@alignCast(ptr));
        if (r.fail_body) return error.RecordedBodyFailure;
        r.eager += 1;
        r.seen = r.table;
    }
    fn graphs(r: *Record) graph.Graphs {
        return .{ .driver = undefined, .cache = .{
            .gpa = std.testing.allocator,
            .engine = .{ .ctx = r, .capture = capture, .launch = launch, .free = free },
            .settings = .{ .max = 32, .min_keep = 0, .hold = true },
            .cap = 32,
        } };
    }
    fn run(r: *Record, g: *graph.Graphs, rows: usize, owner: @TypeOf(binding)) !void {
        try g.run(stream, rows, owner, .{ .ctx = r, .run = body });
    }
};

test "DFlash ingest graphs: replay reads current table contents and invalid dimensions enqueue nothing" {
    var r: Record = .{};
    var g = r.graphs();
    defer g.cache.deinit();
    const cases = [_]usize{ 1, 8, 16, 32, 128, 2048 };
    for (0..3) |round| for (cases) |rows| {
        r.table = round * 4096 + rows;
        try r.run(&g, rows, binding);
        try std.testing.expectEqual(r.table, r.seen);
    };
    try std.testing.expectEqual(cases.len, r.captures);
    try std.testing.expectEqual(3 * cases.len, r.launches);
    for ([_]usize{ 0, 2049, std.math.maxInt(usize) }) |rows|
        try std.testing.expectError(error.DrafterContextTooWide, r.run(&g, rows, binding));
    try std.testing.expectEqual(3 * cases.len, r.launches);
    inline for (.{ "engine", "weights", "scratch" }) |field| {
        var changed = binding;
        @field(changed, field) += 1;
        try std.testing.expectError(error.DrafterGraphStorageChanged, r.run(&g, 1, changed));
    }
    g.cache.dropAll();
    try std.testing.expectEqual(cases.len, r.freed);
}

test "DFlash ingest graphs: cap remains bounded and capture failure keeps eager compute" {
    var r: Record = .{};
    var g = r.graphs();
    defer g.cache.deinit();
    for (1..34) |rows| try r.run(&g, rows, binding);
    try std.testing.expect(g.cache.count() <= 32);
    try std.testing.expect(r.captures - r.freed <= 32);
    r.fail_capture = true;
    r.table = 9001;
    try r.run(&g, 2048, binding);
    try std.testing.expect(g.cache.failed);
    try std.testing.expectEqual(r.captures, r.freed);
    try std.testing.expectEqual(r.table, r.seen);
    const eager = r.eager;
    try r.run(&g, 1, binding);
    try std.testing.expectEqual(eager + 1, r.eager);
}

test "DFlash ingest graphs: launch errors propagate and disabled-body failure never retries" {
    var r: Record = .{ .fail_launch = true };
    var g = r.graphs();
    defer g.cache.deinit();
    try std.testing.expectError(error.RecordedLaunchFailure, r.run(&g, 1, binding));
    try std.testing.expectEqual(@as(usize, 0), r.eager);
    g.cache.settings.on = false;
    r.fail_body = true;
    try std.testing.expectError(error.RecordedBodyFailure, r.run(&g, 1, binding));
    try std.testing.expectEqual(@as(usize, 0), r.eager);
}

test "DFlash ingest graphs: strict opt-in and rank agreement precede allocation" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_INGEST_GRAPHS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_ingest_graphs);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(on.dflash_ingest_graphs);
    Env.value = "0";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_ingest_graphs);
    Env.value = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_INGEST_GRAPHS: 0 or 1", problem);
    const a = ranks.Fingerprint.of(4, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(ranks.Fingerprint.Field.dflash_ingest_graphs, ranks.mismatch(&.{ a, b }).?.field);
}
