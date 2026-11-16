//! Recording policy for complete-block graphs; real kernels and wire remain device gates.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("dflash_native_graph.zig");
const gc = cuda.graph_cache;
const ranks = @import("cuda_ranks.zig");
const s: cuda.Stream = .{ .d = undefined, .handle = null };
const binding: graph.Binding = .{ .engine = 1, .weights = 2, .scratch = 3, .stream = 4, .comm = 5, .group = 6, .modules = 7 };

const Fake = struct {
    captures: usize = 0,
    launches: usize = 0,
    frees: usize = 0,
    votes: usize = 0,
    bodies: usize = 0,
    table: usize = 0,
    observed: usize = 0,
    fail_capture: bool = false,
    fail_launch: bool = false,
    fail_vote: bool = false,
    peer_room: bool = true,
    peer_capture: bool = true,
    entries: [8]u8 = @splat(0),

    fn capture(ptr: *anyopaque, stream: cuda.Stream, captured_body: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.fail_capture) return error.Capture;
        try captured_body.run(captured_body.ctx, stream);
        f.captures += 1;
        return &f.entries[f.captures - 1];
    }
    fn launch(ptr: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.fail_launch) return error.Launch;
        f.launches += 1;
        f.observed = f.table;
    }
    fn free(ptr: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.frees += 1;
    }
    fn body(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.bodies += 1;
        f.observed = f.table;
    }
    fn vote(ptr: *anyopaque, ok: bool) anyerror!bool {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.votes += 1;
        if (f.fail_vote) return error.Vote;
        return ok and (if (f.votes % 2 == 1) f.peer_room else f.peer_capture);
    }
    fn owner(f: *Fake) graph.Native {
        return .{ .driver = undefined, .rank = 0, .cache = .{
            .gpa = std.testing.allocator,
            .engine = .{ .ctx = f, .capture = capture, .launch = launch, .free = free },
            .settings = .{ .max = 8, .hold = true },
            .cap = 8,
        } };
    }
    fn run(f: *Fake, n: *graph.Native, streams: usize, b: graph.Binding) !gc.Outcome {
        return n.run(streams, b, s, .{ .ctx = f, .all = vote }, .{ .ctx = f, .run = body });
    }
};

test "DFlash graphs: native complete cache has eight keys, two miss votes and no hit votes or evictions" {
    var f: Fake = .{};
    var n = f.owner();
    defer n.cache.deinit();
    for (0..4) |round| for (1..9) |streams| {
        f.table = 100 * round + streams;
        const outcome = try f.run(&n, streams, binding);
        try std.testing.expectEqual(if (round == 0) gc.Outcome.captured else gc.Outcome.replayed, outcome);
        try std.testing.expectEqual(f.table, f.observed);
    };
    try std.testing.expectEqual(@as(usize, 8), f.captures);
    try std.testing.expectEqual(@as(usize, 32), f.launches);
    try std.testing.expectEqual(@as(usize, 16), f.votes);
    try std.testing.expectEqual(@as(usize, 8), f.bodies);
    try std.testing.expectEqual(@as(usize, 8), n.cache.count());
    try std.testing.expectEqual(@as(u64, 0), n.cache.stats.evicted);
    try std.testing.expectEqual(@as(u64, 0), n.cache.stats.dropped);
}

test "DFlash graphs: native room and capture failure fall back together, launch and vote errors propagate" {
    var f: Fake = .{ .peer_capture = false };
    var n = f.owner();
    defer n.cache.deinit();
    try std.testing.expectEqual(gc.Outcome.eager, try f.run(&n, 1, binding));
    try std.testing.expect(n.cache.failed);
    try std.testing.expectEqual(@as(usize, 1), f.frees);
    try std.testing.expectEqual(gc.Outcome.eager, try f.run(&n, 1, binding));
    try std.testing.expectEqual(@as(usize, 2), f.votes);
    var r: Fake = .{ .peer_room = false };
    var room = r.owner();
    defer room.cache.deinit();
    try std.testing.expectEqual(gc.Outcome.eager, try r.run(&room, 1, binding));
    try std.testing.expectEqual(@as(usize, 0), r.captures);
    var e: Fake = .{ .fail_vote = true };
    var errors = e.owner();
    defer errors.cache.deinit();
    try std.testing.expectError(error.Vote, e.run(&errors, 1, binding));
    e.fail_vote = false;
    e.fail_launch = true;
    try std.testing.expectError(error.Launch, e.run(&errors, 1, binding));
}

test "DFlash graphs: native scope payload and direct-owner changes refuse before enqueue" {
    try graph.scope(4, true, true, true);
    try std.testing.expectError(error.DFlashNativeGraphScope, graph.scope(2, true, true, true));
    try std.testing.expectError(error.DFlashNativeGraphScope, graph.scope(4, false, true, true));
    try std.testing.expectError(error.DFlashNativeGraphScope, graph.scope(4, true, false, true));
    try std.testing.expectEqual(@as(usize, 1048576), try graph.payload(8));
    for ([_]usize{ 0, 9, 16 }) |streams| try std.testing.expectError(error.DFlashNativeGraphStreams, graph.payload(streams));
    var f: Fake = .{};
    var n = f.owner();
    defer n.cache.deinit();
    _ = try f.run(&n, 1, binding);
    inline for (@typeInfo(graph.Binding).@"struct".field_names) |field| {
        var changed = binding;
        @field(changed, field) += 1;
        try std.testing.expectError(error.DFlashNativeGraphOwnerChanged, f.run(&n, 1, changed));
    }
    try std.testing.expectEqual(@as(usize, 1), f.launches);
}

test "DFlash graphs: native strict flag and dependencies are agreed in fingerprint40" {
    const Env = struct {
        var value: ?[]const u8 = null;
        var deps = true;
        var cache: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_NATIVE_GRAPH")) return value;
            if (std.mem.eql(u8, name, "TF_GRAPH_CACHE")) return cache;
            for ([_][]const u8{ "TF_GLM_DFLASH_GRAPHS", "TF_GLM_DFLASH_ATTN_TP4", "TF_GLM_DFLASH_MLP_TP4" }) |key|
                if (std.mem.eql(u8, name, key)) return if (deps) "1" else "0";
            return null;
        }
    };
    defer {
        Env.value = null;
        Env.deps = true;
        Env.cache = null;
    }
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(!off.dflash_native_graph);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 63), a.layout);
    try std.testing.expectEqual(FP.Field.dflash_native_graph, ranks.mismatch(&.{ a, b }).?.field);
    Env.deps = false;
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    Env.deps = true;
    Env.cache = "0";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    Env.cache = null;
    for ([_][]const u8{ "yes", "", "2" }) |v| {
        Env.value = v;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
}
