//! Shared-forward profiler ownership and real binding with virtual events, never a CUDA library.
const std = @import("std");
const cuda = @import("cuda");
const owner = @import("cuda_engine_prof.zig");
const Prof = @import("cuda_prof.zig").Prof;
const Engine = @import("cuda_engine.zig").Engine;
const Forward = @import("cuda_forward.zig").Forward;
const t = std.testing;

const Fake = struct {
    var made: usize = 0;
    var live: usize = 0;
    var records: usize = 0;
    var syncs: usize = 0;
    var reads: usize = 0;
    var fail_record = false;
    var clock: f32 = 0;
    var times: [514]f32 = undefined;
    fn reset() void {
        made = 0;
        live = 0;
        records = 0;
        syncs = 0;
        reads = 0;
        fail_record = false;
        clock = 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuEventCreate = create;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = sync;
        d.api.cuEventElapsedTime = elapsed;
        return d;
    }
    fn create(out: *cuda.abi.Event, flags: c_uint) callconv(.c) c_int {
        std.debug.assert(flags == 0 and made < 513);
        made += 1;
        live += 1;
        out.* = @ptrFromInt(made);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
        live -= 1;
        return 0;
    }
    fn record(event: cuda.abi.Event, stream: cuda.abi.Stream) callconv(.c) c_int {
        std.debug.assert(@intFromPtr(stream) == 7);
        records += 1;
        if (fail_record) return cuda.abi.error_not_ready;
        times[@intFromPtr(event)] = clock;
        return 0;
    }
    fn sync(_: cuda.abi.Event) callconv(.c) c_int {
        syncs += 1;
        return 0;
    }
    fn elapsed(out: *f32, start: cuda.abi.Event, end: cuda.abi.Event) callconv(.c) c_int {
        reads += 1;
        out.* = times[@intFromPtr(end)] - times[@intFromPtr(start)];
        return 0;
    }
};

test "GLM shared profile: OFF performs no allocation event clock sync or binding" {
    Fake.reset();
    var d: cuda.Driver = undefined;
    try t.expect(try owner.Shared.init(t.failing_allocator, &d, t.io, false) == null);
    var e: Engine = undefined;
    e.shared_prof = null;
    var f: Forward = undefined;
    f.prof = null;
    try t.expect(try owner.beginShared(&e, &f, 0, 0) == null);
    try t.expect(f.prof == null);
    try t.expectEqual(@as(usize, 0), Fake.made + Fake.records + Fake.syncs + Fake.reads);
}

test "GLM shared profile: actual binding reports each R S and awake start with stable owner label" {
    Fake.reset();
    var d = Fake.driver();
    var e: Engine = undefined;
    e.profs = @splat(null);
    e.prompt_profs = @splat(null);
    e.shared_prof = try owner.Shared.init(t.allocator, &d, t.io, true);
    errdefer owner.freeProfs(&e);
    e.stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    e.share.rank = 2;
    var f: Forward = undefined;
    f.prof = null;
    var previous: i96 = 0;
    for ([_][2]usize{ .{ 32, 4 }, .{ 37, 8 }, .{ 64, 8 } }, 1..) |shape, serial| {
        const before = std.Io.Clock.awake.now(t.io).toNanoseconds();
        const p = (try owner.beginShared(&e, &f, shape[0], shape[1])).?;
        const after = std.Io.Clock.awake.now(t.io).toNanoseconds();
        try t.expect(f.prof == p.prof and p.prof.prompt == null);
        try t.expectEqual(@as(u64, 1), p.prof.every);
        try t.expect(p.prof.t0 >= before and p.prof.t0 <= after and p.prof.t0 >= previous);
        previous = p.prof.t0;
        var label: [192]u8 = undefined;
        try t.expectEqualStrings(try std.fmt.bufPrint(&label, "verify_shared rank=2 R={d} S={d} seq={d} tstart_ns={d}", .{ shape[0], shape[1], serial, p.prof.t0 }), p.prof.label);
        Fake.clock += 3;
        f.prof.?.mark(e.stream, .embed);
        Fake.clock += 4;
        f.prof.?.mark(e.stream, .kda);
        Fake.clock += 4;
        f.prof.?.mark(e.stream, .head);
        Fake.clock += 2;
        p.finish(e.stream);
        try t.expect(!p.prof.open);
        try t.expectEqual(@as(u64, 0), p.prof.calls); // every1 emitted and reset, no cross-shape averages
        try t.expectEqual(@as(f64, 0), p.prof.gpu_ms);
        for (p.prof.sections) |ms| try t.expectEqual(@as(f64, 0), ms);
    }
    try t.expectEqual(@as(usize, 513), Fake.made);
    try t.expectEqual(@as(usize, 15), Fake.records); // begin +3target marks +draw, three forwards
    try t.expectEqual(@as(usize, 3), Fake.syncs);
    try t.expectEqual(@as(usize, 15), Fake.reads); // four sections +whole-call timing each
    owner.freeProfs(&e);
    try t.expect(e.shared_prof == null);
    try t.expectEqual(@as(usize, 0), Fake.live);
}

test "GLM shared profile: begin or body failure and missing terminal event publish no completed call" {
    Fake.reset();
    var d = Fake.driver();
    const p = (try owner.Shared.init(t.allocator, &d, t.io, true)).?;
    defer p.deinit();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    Fake.fail_record = true;
    try t.expectError(error.NotReady, p.begin(stream, 0, 40, 8));
    try t.expect(!p.prof.open and p.prof.n == 0);
    Fake.fail_record = false;
    try p.begin(stream, 0, 40, 8);
    const serial = p.serial;
    try t.expectError(error.SharedProfileOpen, p.begin(stream, 0, 40, 8));
    try t.expectEqual(serial, p.serial);
    p.prof.mark(stream, .head);
    p.cancel(); // same no-wait unwind used by verifyShared's errdefer
    try t.expect(!p.prof.open and p.prof.n == 0 and p.prof.calls == 0);
    try p.begin(stream, 0, 48, 8);
    p.prof.mark(stream, .head);
    Fake.fail_record = true;
    p.finish(stream);
    try t.expect(!p.prof.open and p.prof.n == 0 and p.prof.calls == 0);
    try t.expectEqual(@as(usize, 0), Fake.syncs + Fake.reads);
}

test "GLM shared profile: creation unwind and invalid geometry are event-free" {
    const Factory = struct {
        fn fail(_: *const cuda.Driver, _: std.Io) !*Prof {
            return error.InjectedProfileInit;
        }
    };
    Fake.reset();
    var d = Fake.driver();
    try t.expectError(error.InjectedProfileInit, owner.Shared.initWith(t.allocator, &d, t.io, true, Factory.fail));
    const p = (try owner.Shared.init(t.allocator, &d, t.io, true)).?;
    defer p.deinit();
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(7) };
    for ([_][2]usize{ .{ 0, 1 }, .{ 65, 8 }, .{ 4, 0 }, .{ 64, @import("cuda_engine.zig").max_shared + 1 }, .{ 4, 5 } }) |bad|
        try t.expectError(error.SharedProfileShape, p.begin(stream, 0, bad[0], bad[1]));
    try t.expectEqual(@as(usize, 0), Fake.records + Fake.syncs + Fake.reads);
    try t.expectEqual(@as(u64, 0), p.serial);
}
