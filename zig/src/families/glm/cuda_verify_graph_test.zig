//! Actual key construction and cache coordination with recording graph executables; no CUDA execution.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("cuda_verify_graph.zig");
const forward = @import("cuda_verify_forward.zig");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const dsa = @import("cuda_dsa_block.zig");
const ranks = @import("cuda_ranks.zig");
const gc = cuda.graph_cache;
const s: cuda.Stream = .{ .d = undefined, .handle = null };
const binding: graph.Binding = .{ .engine = 1, .weights = 2, .modules = 3, .stream = 4, .comm = 5, .taps = 6, .tap_function = 7 };

const Fake = struct {
    captures: usize = 0,
    launches: usize = 0,
    frees: usize = 0,
    eager: usize = 0,
    votes: usize = 0,
    peer_miss: bool = false,
    fail_capture: bool = false,
    fail_launch: bool = false,
    fail_vote: bool = false,
    entries: [100]u8 = @splat(0),

    fn capture(p: *anyopaque, _: cuda.Stream, _: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(p));
        if (f.fail_capture) return error.RecordedCaptureFailure;
        f.captures += 1;
        return &f.entries[f.captures - 1];
    }
    fn launch(p: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(p));
        if (f.fail_launch) return error.RecordedLaunchFailure;
        f.launches += 1;
    }
    fn free(p: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.frees += 1;
    }
    fn body(p: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.eager += 1;
    }
    fn vote(p: *anyopaque, ok: bool) anyerror!bool {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.votes += 1;
        if (f.fail_vote) return error.RecordedVoteFailure;
        if (f.peer_miss) {
            f.peer_miss = false;
            return false;
        }
        return ok;
    }
    fn graphs(f: *Fake) graph.Graphs {
        return .{ .driver = undefined, .cache = .{
            .gpa = std.testing.allocator,
            .engine = .{ .ctx = f, .capture = capture, .launch = launch, .free = free },
            .settings = .{ .max = 32, .hold = true },
            .cap = 32,
        } };
    }
    fn run(f: *Fake, g: *graph.Graphs, k: graph.Key) !gc.Outcome {
        return g.run(k, binding, s, .{ .ctx = f, .all = vote }, .{ .ctx = f, .run = body });
    }
};

test "GLM verify graphs: rank miss replaces a local hit before capture votes; hits vote once" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    const k: graph.Key = .{ .scratch = 10, .rows = 8, .count = 1 };
    try std.testing.expectEqual(gc.Outcome.captured, try f.run(&g, k));
    try std.testing.expectEqual(@as(usize, 3), f.votes);
    try std.testing.expectEqual(gc.Outcome.replayed, try f.run(&g, k));
    try std.testing.expectEqual(@as(usize, 4), f.votes);
    f.peer_miss = true;
    try std.testing.expectEqual(gc.Outcome.captured, try f.run(&g, k));
    try std.testing.expectEqual(@as(usize, 2), f.captures);
    try std.testing.expectEqual(@as(usize, 1), f.frees);
    try std.testing.expectEqual(@as(usize, 7), f.votes);
}

test "GLM verify graphs: bounded cache, generation invalidation, capture and launch errors" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    for (1..34) |n| _ = try f.run(&g, .{ .scratch = n, .rows = 1, .count = 1 });
    try std.testing.expectEqual(@as(usize, 32), g.cache.count());
    g.invalidate();
    try std.testing.expectEqual(@as(u64, 1), g.generation);
    try std.testing.expectEqual(@as(usize, 33), f.frees);
    const k: graph.Key = .{ .scratch = 50, .rows = 1, .count = 1 };
    f.fail_launch = true;
    try std.testing.expectError(error.RecordedLaunchFailure, f.run(&g, k));
    f.fail_launch = false;
    f.fail_vote = true;
    try std.testing.expectError(error.RecordedVoteFailure, f.run(&g, k));
    f.fail_vote = false;
    g.invalidate();
    f.fail_capture = true;
    try std.testing.expectEqual(gc.Outcome.eager, try f.run(&g, k));
    const votes = f.votes;
    try std.testing.expectEqual(gc.Outcome.eager, try f.run(&g, k));
    try std.testing.expectEqual(votes, f.votes);
    try std.testing.expectEqual(@as(usize, 2), f.eager);
}

test "GLM verify graphs: immutable owner bindings reject replacement" {
    var f: Fake = .{};
    var g = f.graphs();
    defer g.cache.deinit();
    const k: graph.Key = .{ .scratch = 1, .rows = 1, .count = 1 };
    _ = try f.run(&g, k);
    inline for (@typeInfo(graph.Binding).@"struct".field_names) |field| {
        var changed = binding;
        @field(changed, field) += 1;
        try std.testing.expectError(error.VerifyGraphOwnerChanged, g.run(k, changed, s, .{ .ctx = &f, .all = Fake.vote }, .{ .ctx = &f, .run = Fake.body }));
    }
    try std.testing.expectEqual(@as(usize, 1), f.launches);
}

test "GLM verify graphs: actual keys include parity, allocation, rebase and attention launch geometry" {
    var cur = [_]u1{ 0, 1 };
    var caches = [_]dsa.Caches{.{ .lc = 100, .ik = 200, .ig = 300, .pk = 400, .pk_rows = 65538 }};
    var st: fw.State = undefined;
    st.arena.buf.ptr = 1000;
    st.pos = 100;
    st.pos_dev = 500;
    st.base = 0;
    st.cur = &cur;
    st.caches = &caches;
    var b: fw.Buffers = undefined;
    b.arena.buf.ptr = 2000;
    b.prefill = false;
    b.nch = 1024;
    var a: dsa.Layer = undefined;
    a.dense_limit = 2048;
    var layer: weights.Layer = undefined;
    layer.mixer = .{ .mla = a };
    var w: weights.Weights = undefined;
    w.layers = @as(*[1]weights.Layer, @ptrCast(&layer));
    var rs = [_]fw.Rank{.{ .w = &w, .st = &st, .b = &b }};
    const fwrd: fw.Forward = .{ .d = undefined, .m = undefined, .stream = s, .ranks = &rs, .exchange = .local };
    const original = try forward.keyOf(&fwrd, 4);
    st.pos = 101; // device position changes inside identical host launch geometry
    try std.testing.expectEqualDeep(original, try forward.keyOf(&fwrd, 4));
    cur[0] = 1;
    try std.testing.expect(!std.meta.eql(original, try forward.keyOf(&fwrd, 4)));
    cur[0] = 0;
    st.base = 4;
    caches[0].ik = 196;
    try std.testing.expect(!std.meta.eql(original, try forward.keyOf(&fwrd, 4)));
    st.base = 0;
    caches[0].ik = 200;
    st.arena.buf.ptr += 1;
    try std.testing.expect(!std.meta.eql(original, try forward.keyOf(&fwrd, 4)));
    st.arena.buf.ptr -= 1;
    for ([_]usize{ 511, 2045, 2048, 4093, 5000 }) |pos| {
        st.pos = pos;
        try std.testing.expect(!std.meta.eql(original, try forward.keyOf(&fwrd, 4)));
    }
    var segs = [_]fw.Seg{ .{ .st = &st, .row0 = 0, .rows = 2 }, .{ .st = &st, .row0 = 2, .rows = 2 } };
    rs[0].segs = &segs;
    const shared = try forward.keyOf(&fwrd, 4);
    segs[0].rows = 1;
    segs[1].row0 = 1;
    segs[1].rows = 3;
    try std.testing.expect(!std.meta.eql(shared, try forward.keyOf(&fwrd, 4)));
    b.prefill = true;
    try std.testing.expectError(error.VerifyGraphScope, forward.keyOf(&fwrd, 4));
    b.prefill = false;
    caches[0].pt = 1;
    try std.testing.expectError(error.VerifyGraphScope, forward.keyOf(&fwrd, 4));
}

test "GLM verify graphs: strict default-off setting and operator disable are agreed" {
    const Get = struct {
        var value: ?[]const u8 = null;
        var operator: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_VERIFY_GRAPHS")) return value;
            if (std.mem.eql(u8, name, "TF_GRAPH_CACHE")) return operator;
            return null;
        }
    };
    defer Get.value = null;
    defer Get.operator = null;
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(Get.get, &problem)).verify_graphs);
    Get.value = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Get.get, &problem));
    Get.value = "1";
    const on = try ranks.Settings.read(Get.get, &problem);
    try std.testing.expect(on.verify_graphs);
    Get.operator = "off";
    try std.testing.expect(!(try ranks.Settings.read(Get.get, &problem)).verify_graphs);
    const a = ranks.Fingerprint.of(4, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(ranks.Fingerprint.Field.verify_graphs, ranks.mismatch(&.{ a, b }).?.field);
}
