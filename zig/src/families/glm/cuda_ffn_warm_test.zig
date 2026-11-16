const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;
const graph = @import("cuda_ffn_graph.zig");
const warm = @import("cuda_ffn_warm.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const Engine = @import("cuda_engine.zig").Engine;
const weights = @import("cuda_weights.zig");
const t = std.testing;
const no_stream: cuda.Stream = .{ .d = undefined, .handle = null };
const key: graph.Key = .{ .layer = 0, .rows = 16, .weights = 1, .scratch = 2, .input = 3, .output = 4 };

const Fake = struct {
    captures: usize = 0,
    launches: usize = 0,
    freed: usize = 0,
    eager: usize = 0,
    fail: bool = false,
    body_addr: ?usize = null,
    require_drain: bool = false,
    fn capture(ptr: *anyopaque, _: cuda.Stream, b: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.fail) return error.Capture;
        f.captures += 1;
        f.body_addr = @intFromPtr(b.run);
        return @ptrFromInt(4096 + f.captures * 16);
    }
    fn launch(ptr: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.launches += 1;
    }
    fn free(ptr: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.require_drain) std.debug.assert(Hooks.syncs > 0);
        f.freed += 1;
    }
    fn body(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        f.eager += 1;
    }
    fn owner(f: *Fake) graph.Graphs {
        return .{ .gpa = t.allocator, .driver = undefined, .rank = 2, .cache = .{
            .gpa = t.allocator,
            .cap = 4096,
            .settings = .{ .max = 4096, .hold = true },
            .engine = .{ .ctx = f, .capture = capture, .launch = launch, .free = free },
        } };
    }
};

test "FFN graphs: warm prepares without numerical execution and first actual hit launches once" {
    var f: Fake = .{};
    var g = f.owner();
    defer g.cache.deinit();
    const b: gc.Body = .{ .ctx = &f, .run = Fake.body };
    try g.prepare(key, no_stream, b);
    try g.prepare(key, no_stream, b);
    try t.expectEqual(@as(usize, 1), f.captures);
    try t.expectEqual(@as(usize, 0), f.launches + f.eager);
    try t.expectEqual(@as(u64, 1), g.warm_captured);
    try t.expectEqual(gc.Outcome.replayed, try g.run(key, no_stream, b));
    try t.expectEqual(@as(usize, 1), f.launches);
    try t.expectEqual(@as(usize, 0), f.eager);
    try t.expectEqual(@as(u64, 1), g.cache.stats.replayed);
}

test "FFN graphs: warm capture and allocation failure never execute eager or publish an entry" {
    var f: Fake = .{ .fail = true };
    var g = f.owner();
    defer g.cache.deinit();
    const b: gc.Body = .{ .ctx = &f, .run = Fake.body };
    try t.expectError(error.Capture, g.prepare(key, no_stream, b));
    try t.expectEqual(@as(usize, 0), f.captures + f.launches + f.eager + g.cache.count());
    f.fail = false;
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    g.cache.entries.deinit(g.cache.gpa);
    g.cache.entries = .empty;
    g.cache.gpa = failing.allocator();
    try t.expectError(error.OutOfMemory, g.prepare(key, no_stream, b));
    try t.expectEqual(@as(usize, 0), f.captures + f.launches + f.eager + g.cache.count());
}

test "FFN graphs: warm refuses disabled failed full and changed fingerprint owners" {
    var f: Fake = .{};
    var g = f.owner();
    defer g.cache.deinit();
    const b: gc.Body = .{ .ctx = &f, .run = Fake.body };
    g.cache.settings.on = false;
    try t.expectError(error.FfnWarmCacheScope, g.prepare(key, no_stream, b));
    g.cache.settings.on = true;
    g.cache.failed = true;
    try t.expectError(error.FfnWarmCacheScope, g.prepare(key, no_stream, b));
    g.cache.failed = false;
    g.cache.settings.max = 0;
    try t.expectError(error.FfnWarmCapacity, g.prepare(key, no_stream, b));
    g.cache.settings.max = 4096;
    g.cache.fingerprint = 3;
    try t.expectError(error.FfnWarmFingerprint, g.prepare(key, no_stream, b));
    try t.expectEqual(@as(usize, 0), f.captures + f.launches + f.eager);
}

const Hooks = struct {
    var syncs: usize = 0;
    var mem_reads: usize = 0;
    fn mem(free: *usize, total: *usize) callconv(.c) cuda.abi.Result {
        free.* = 1 << 30;
        total.* = 2 << 30;
        mem_reads += 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return 0;
    }
};
fn buffers(rows: usize, ptr: u64) fw.Buffers {
    var b: fw.Buffers = undefined;
    b.rows = rows;
    b.prefill = false;
    b.normed = ptr + 0x1000;
    b.gathered = ptr + 0x2000;
    b.arena.buf.ptr = ptr;
    return b;
}

test "FFN graphs: physical32 warm captures active16/32 solo and all shared keys without launches" {
    for ([_]u8{ 16, 32 }) |active| try warmRows(active);
}

fn warmRows(active: u8) !void {
    var f: Fake = .{};
    var g = f.owner();
    defer g.cache.deinit();
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = Hooks.sync;
    d.api.cuMemGetInfo_v2 = Hooks.mem;
    var e: Engine = undefined;
    e.d = &d;
    e.settings = .{ .ffn_warm = true, .verify_rows = active };
    e.stream = .{ .d = &d, .handle = null };
    e.m.ffn_graphs = &g;
    e.share.rank = 2;
    e.w.share.rank = 2;
    e.w.config.hidden = 4096;
    var layers: [2]weights.Layer = undefined;
    for (&layers, 0..) |*l, i| {
        l.index = @intCast(i);
        l.ffn = .{ .dense = undefined };
    }
    e.w.layers = &layers;
    var seq: @TypeOf(e.cur.*) = undefined;
    e.cur = &seq;
    e.images = null;
    e.slot = 0;
    e.drafter = null;
    e.l2pf = null;
    e.dense_segments = null;
    e.kda_segments = null;
    e.pair = null;
    e.group = null;
    e.lib = undefined;
    e.comm = null;
    e.expert_fused_calls = 0;
    e.bd = buffers(fw.window_rows, 0x10000);
    e.bs = buffers(64, 0x20000);
    Hooks.syncs = 0;
    Hooks.mem_reads = 0;
    const count: usize = 2 * (@as(usize, active) + 64);
    try warm.prepare(&e);
    try t.expectEqual(count, f.captures);
    try t.expectEqual(@as(usize, 0), f.launches + f.eager);
    try t.expectEqual((count + 1), Hooks.syncs);
    try t.expectEqual(@as(usize, 2), Hooks.mem_reads);
    for (g.cache.entries.keys()) |k| {
        const b = if (k.scratch == e.bd.arena.buf.ptr) &e.bd else &e.bs.?;
        try t.expect(k.rows >= 1 and k.rows <= b.rows and k.layer < 2);
        try t.expectEqual(b.normed, k.input);
        try t.expectEqual(b.gathered + 2 * k.rows * 4096 * 4, k.output);
        try t.expectEqual(@intFromPtr(&e.w), k.weights);
    }
    var one: [1]fw.Rank = undefined;
    const forward = e.view(&e.bd, &one);
    try forward.ffn(one[0], layers[0], 16, fw.Forward.slot(one[0], 16));
    try t.expectEqual(count, f.captures);
    try t.expectEqual(@as(usize, 1), f.launches);
    try t.expectEqual((count + 1), Hooks.syncs);
    try t.expectEqual(@as(u64, 1), g.dense_calls);
    var other = layers[0];
    other.index = 3;
    const before = f.body_addr;
    try forward.ffn(one[0], other, 16, fw.Forward.slot(one[0], 16));
    try t.expectEqual(before, f.body_addr);
    try t.expectEqual((count + 1), f.captures);
}

test "FFN graphs: warm OFF is inert and strict flag fingerprint scope agrees" {
    var e: Engine = undefined;
    e.settings = .{};
    try warm.prepare(&e);
    warm.unwind(&e);
    const Get = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_FFN_WARM")) value else null;
        }
    };
    var problem: []const u8 = "";
    try t.expect(!(try ranks.Settings.read(Get.get, &problem)).ffn_warm);
    Get.value = "0";
    try t.expect(!(try ranks.Settings.read(Get.get, &problem)).ffn_warm);
    Get.value = "1";
    const on = try ranks.Settings.read(Get.get, &problem);
    try t.expect(on.ffn_warm);
    Get.value = "true";
    try t.expectError(error.BadSetting, ranks.Settings.read(Get.get, &problem));
    const hash: [32]u8 = @splat(0);
    var a = ranks.Fingerprint.of(4, 0, .{}, 4096, true, hash, hash, hash);
    var b = ranks.Fingerprint.of(4, 1, on, 4096, true, hash, hash, hash);
    try t.expectEqual(ranks.Fingerprint.Field.ffn_warm, ranks.mismatch(&.{ a, b }).?.field);
    a.ffn_warm = 1;
    b.ffn_warm = 1;
    try t.expect(ranks.mismatch(&.{ a, b }) == null);
    const Options = struct { settings: ranks.Settings, drafter: bool = true, mtp: bool = true, vision: ?u8 = null };
    var o: Options = .{ .settings = .{ .ffn_warm = true, .ffn_graphs = true, .exl3_fused_decode = true, .verify_rows = 16 } };
    try warm.require(o, 4);
    o.settings.verify_rows = 32;
    try warm.require(o, 4);
    o.settings.verify_rows = 16;
    try t.expectError(error.FfnWarmScope, warm.require(o, 3));
    o.drafter = false;
    try t.expectError(error.FfnWarmScope, warm.require(o, 4));
    o.drafter = true;
    o.settings.dense_bits = 4;
    try warm.require(o, 4);
    o.settings.dense_bits = 6;
    try t.expectError(error.FfnWarmScope, warm.require(o, 4));
}

test "FFN graphs: warm init unwind drains then frees published executables and clears owner" {
    var f: Fake = .{ .require_drain = true };
    const g = try t.allocator.create(graph.Graphs);
    g.* = f.owner();
    try g.prepare(key, no_stream, .{ .ctx = &f, .run = Fake.body });
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = Hooks.sync;
    var e: Engine = undefined;
    e.settings = .{ .ffn_warm = true };
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.m.ffn_graphs = g;
    Hooks.syncs = 0;
    warm.unwind(&e);
    try t.expectEqual(@as(usize, 1), Hooks.syncs);
    try t.expectEqual(@as(usize, 1), f.freed);
    try t.expect(e.m.ffn_graphs == null);
    try t.expectError(error.FfnWarmScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .capacity = 4096, .mtp = true, .settings = .{ .ffn_warm = true } }));
}
