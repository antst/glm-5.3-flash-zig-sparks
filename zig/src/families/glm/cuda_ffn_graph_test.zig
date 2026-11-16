const std = @import("std");
const cuda = @import("cuda");
const graph = @import("cuda_ffn_graph.zig");
const gc = cuda.graph_cache;
const t = std.testing;
const stream: cuda.Stream = .{ .d = undefined, .handle = null };
const key: graph.Key = .{ .layer = 1, .rows = 8, .weights = 2, .scratch = 3, .input = 4, .output = 5 };

const Fake = struct {
    captures: usize = 0,
    launches: usize = 0,
    frees: usize = 0,
    bodies: usize = 0,
    capture_error: bool = false,
    launch_error: bool = false,
    tokens: [4]u8 = @splat(0),

    fn capture(ptr: *anyopaque, _: cuda.Stream, _: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.capture_error) return error.Capture;
        f.captures += 1;
        return &f.tokens[f.captures - 1];
    }
    fn launch(ptr: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.launch_error) return error.Launch;
        f.launches += 1;
    }
    fn free(ptr: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.frees += 1;
    }
    fn body(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.bodies += 1;
    }
    fn owner(f: *Fake) graph.Graphs {
        return .{ .gpa = t.allocator, .driver = undefined, .rank = 0, .cache = .{
            .gpa = t.allocator,
            .engine = .{ .ctx = f, .capture = capture, .launch = launch, .free = free },
            .settings = .{ .max = 2, .hold = true },
            .cap = 2,
        } };
    }
    fn run(f: *Fake, g: *graph.Graphs, k: graph.Key) !gc.Outcome {
        return g.run(k, stream, .{ .ctx = f, .run = body });
    }
};

test "FFN graphs: stable pointer keys replay, full cache retains entries and runs new shapes eagerly" {
    var f: Fake = .{};
    var g = f.owner();
    defer g.cache.deinit();
    try t.expect(g.miss(key));
    try t.expectEqual(gc.Outcome.captured, try f.run(&g, key));
    for (0..10) |_| try t.expectEqual(gc.Outcome.replayed, try f.run(&g, key));
    var next = key;
    next.rows = 16;
    try t.expectEqual(gc.Outcome.captured, try f.run(&g, next));
    next.output += 1;
    try t.expect(!g.miss(next));
    try t.expectEqual(gc.Outcome.eager, try f.run(&g, next));
    try t.expectEqual(@as(usize, 2), g.cache.count());
    try t.expectEqual(@as(usize, 0), f.frees);
    try t.expectEqual(@as(u64, 1), g.full_eager);
    try t.expectEqual(gc.Outcome.replayed, try f.run(&g, key));
    try t.expect(g.cache.agree == null);
}

test "FFN graphs: capture failure disables capture and launch errors never run eager output" {
    var f: Fake = .{ .capture_error = true };
    var g = f.owner();
    defer g.cache.deinit();
    try t.expectEqual(gc.Outcome.eager, try f.run(&g, key));
    try t.expect(g.cache.failed);
    try t.expectEqual(@as(usize, 1), f.bodies);
    f.capture_error = false;
    try t.expectEqual(gc.Outcome.eager, try f.run(&g, key));
    try t.expectEqual(@as(usize, 0), f.captures);
    g.cache.failed = false;
    try t.expectEqual(gc.Outcome.captured, try f.run(&g, key));
    f.launch_error = true;
    try t.expectError(error.Launch, f.run(&g, key));
    try t.expectEqual(@as(usize, 2), f.bodies);
}

test "FFN graphs: actual forward routes only nonprofiled complete decode FFNs through the owner" {
    const fw = @import("cuda_forward.zig");
    const weights = @import("cuda_weights.zig");
    const Hooks = struct {
        var syncs: usize = 0;
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            syncs += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = Hooks.sync;
    var f: Fake = .{};
    var g = f.owner();
    defer g.cache.deinit();
    var m: fw.Modules = undefined;
    m.ffn_graphs = &g;
    var w: weights.Weights = undefined;
    var b: fw.Buffers = undefined;
    b.prefill = false;
    b.arena.buf.ptr = 0x10000;
    b.normed = 0x20000;
    var st: fw.State = undefined;
    const r: fw.Rank = .{ .w = &w, .b = &b, .st = &st };
    var forward: fw.Forward = .{ .d = &d, .m = &m, .stream = .{ .d = &d, .handle = null }, .ranks = &.{r}, .exchange = .local };
    var l: weights.Layer = undefined;
    l.index = 3;
    l.ffn = .{ .dense = undefined };
    Hooks.syncs = 0;
    try forward.ffn(r, l, 16, 0x30000);
    try forward.ffn(r, l, 16, 0x30000);
    try t.expectEqual(@as(usize, 1), Hooks.syncs);
    try t.expectEqual(@as(usize, 1), f.captures);
    try t.expectEqual(@as(usize, 2), f.launches);
    try t.expectEqual(@as(usize, 0), f.bodies);
    try t.expectEqual(@as(u64, 2), g.dense_calls);
    try t.expectEqual(@as(u64, 0), g.exl3_calls);
    f.launch_error = true;
    try t.expectError(error.Launch, forward.ffn(r, l, 16, 0x30000));
    try t.expectEqual(@as(u64, 2), g.dense_calls);
    f.launch_error = false;
    for (1..65) |rows| try t.expect(graph.eligible(false, false, rows, 0x30000));
    try t.expect(!graph.eligible(true, false, 16, 0x30000));
    try t.expect(!graph.eligible(false, true, 16, 0x30000));
    try t.expect(!graph.eligible(false, false, 16, null));
    try t.expect(!graph.eligible(false, false, 65, 0x30000));
    g.modules = 42;
    try t.expectError(error.FfnGraphOwnerChanged, forward.ffn(r, l, 16, 0x30000));
    try t.expectEqual(@as(usize, 2), f.launches);
    g.modules = null;
    var moe: @import("cuda_moe.zig").Layer = undefined;
    moe.exl3 = std.mem.zeroes(@import("cuda_exl3.zig").Experts);
    l.ffn = .{ .moe = moe };
    try forward.ffn(r, l, 16, 0x30000);
    try t.expectEqual(@as(u64, 1), g.exl3_calls);
    moe.exl3 = null;
    l.ffn = .{ .moe = moe };
    try forward.ffn(r, l, 16, 0x30000);
    try t.expectEqual(@as(u64, 1), g.other_moe_calls);
    try t.expectEqual(@as(u64, 2), g.dense_calls);
}

test "FFN graphs: strict default-off settings and fingerprint agreement" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var value: ?[]const u8 = null;
        var cache: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GRAPH_CACHE")) return cache;
            return if (std.mem.eql(u8, name, "TF_GLM_FFN_GRAPHS")) value else null;
        }
    };
    defer {
        Env.value = null;
        Env.cache = null;
    }
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.ffn_graphs);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    Env.cache = "off";
    try t.expect(!(try ranks.Settings.read(Env.get, &problem)).ffn_graphs);
    Env.cache = null;
    const a = ranks.Fingerprint.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const z = ranks.Fingerprint.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(ranks.Fingerprint.Field.ffn_graphs, ranks.mismatch(&.{ a, z }).?.field);
    for ([_][]const u8{ "", "true", "2", "1 " }) |v| {
        Env.value = v;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
}

test "FFN graphs: actual module owner frees held graphs exactly once and startup scopes refuse before allocation" {
    const fw = @import("cuda_forward.zig");
    var f: Fake = .{};
    const g = try t.allocator.create(graph.Graphs);
    g.* = f.owner();
    _ = try f.run(g, key);
    var second = key;
    second.rows = 64;
    _ = try f.run(g, second);
    var m: fw.Modules = undefined;
    m.ffn_graphs = g;
    m.freeFfnGraphs();
    try t.expect(m.ffn_graphs == null);
    try t.expectEqual(@as(usize, 2), f.frees);
    m.freeFfnGraphs();
    try t.expectEqual(@as(usize, 2), f.frees);
    const Engine = @import("cuda_engine.zig").Engine;
    try t.expectError(error.FfnGraphScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 3, .{ .settings = .{ .ffn_graphs = true }, .capacity = 4096, .mtp = true }));
    try t.expectError(error.FfnGraphScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .ffn_graphs = true, .pages = true }, .capacity = 4096, .mtp = true }));
}

test "FFN graphs: fused mode capacity is bounded while legacy creation stays512" {
    const old = try graph.Graphs.create(t.allocator, undefined, 0);
    defer old.destroy();
    const fused = try graph.Graphs.createMode(t.allocator, undefined, 0, true);
    defer fused.destroy();
    try t.expectEqual(@as(usize, 512), old.cache.cap);
    try t.expectEqual(@as(usize, 512), old.cache.settings.max);
    try t.expectEqual(@as(usize, 4608), fused.cache.cap);
    try t.expectEqual(@as(usize, 4608), fused.cache.settings.max);
    try t.expect(fused.cache.cap >= 45 * 64);
    try t.expect(fused.cache.agree == null and fused.cache.room == null);
    const Engine = @import("cuda_engine.zig").Engine;
    try t.expectError(error.Exl3DecodeScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 3, .{ .settings = .{ .exl3_fused_decode = true, .ffn_graphs = true }, .capacity = 4096, .mtp = true }));
    try t.expectError(error.Exl3DecodeScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .exl3_fused_decode = true, .verify_graphs = true }, .capacity = 4096, .mtp = true }));
}

comptime {
    _ = @import("cuda_ffn_warm_test.zig");
    _ = @import("cuda_nvfp4_graph_test.zig");
}
