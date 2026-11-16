const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense_segments.zig");
const Owner = @import("cuda_dense_owner.zig").Owner;

test "dense segment actual shared verify constructor binds its table owner and original segment order" {
    const Engine = @import("cuda_engine.zig").Engine;
    const fw = @import("cuda_forward.zig");
    var e: Engine = undefined;
    var d: cuda.Driver = undefined;
    e.d = &d;
    e.lib = @ptrFromInt(0x1000);
    e.comm = null;
    e.group = null;
    e.stream = .{ .d = &d, .handle = null };
    e.settings = .{};
    e.drafter = null;
    e.l2pf = null;
    e.dense_segments = null;
    e.kda_segments = null;
    var b: fw.Buffers = undefined;
    var a: fw.State = undefined;
    var z: fw.State = undefined;
    const segs = [_]fw.Seg{ .{ .st = &a, .row0 = 0, .rows = 3 }, .{ .st = &z, .row0 = 3, .rows = 6 } };
    var one: [1]fw.Rank = undefined;
    const off = e.sharedView(&b, &segs, &one);
    try std.testing.expect(off.dense_segments == null);
    e.dense_segments = @as(Owner, undefined);
    const on = e.sharedView(&b, &segs, &one);
    try std.testing.expect(on.dense_segments.? == &e.dense_segments.?);
    try std.testing.expect(on.ranks[0].segs.ptr == &segs);
    try std.testing.expect(on.ranks[0].w == &e.w and on.ranks[0].b == &b);
    try std.testing.expectEqual(@as(usize, 6), on.ranks[0].segs[1].rows);
    try std.testing.expect(on.ranks[0].segs[1].st == &z);
}

test "dense segment owner refreshes actual row tables once and drains pinned staging before reuse" {
    const Fake = struct {
        var uploads: usize = 0;
        var records: usize = 0;
        var waits: usize = 0;
        var fail_record = false;
        var copied: [4096]u8 align(16) = undefined;
        fn copy(dst: u64, src: ?*const anyopaque, len: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(dst == 0x80000000 and len == copied.len and s == null);
            @memcpy(&copied, @as([*]const u8, @ptrCast(src.?))[0..len]);
            uploads += 1;
            return 0;
        }
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            records += 1;
            return if (fail_record) cuda.abi.error_not_ready else 0;
        }
        fn wait(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
            waits += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copy;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventSynchronize = Fake.wait;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var staging: [4096]u8 align(16) = @splat(0);
    var plans: [2]?dense.Plan = @splat(null);
    var owner: Owner = .{
        .gpa = std.testing.allocator,
        .kernels = undefined,
        .module = undefined,
        .stream = stream,
        .tables = .{ .d = &d, .ptr = 0x80000000, .len = staging.len },
        .partials = undefined,
        .staging = .{ .d = &d, .bytes = &staging },
        .uploaded = .{ .d = &d, .handle = null },
        .plans = &plans,
        .rows = 64,
        .chunks = 6,
        .stat_bytes = 0,
    };
    const Cache = struct { pt: u64 = 0, ps: u64 = 0, pks: u64 = 0, lc: u64, ls: u64 };
    const State = struct { caches: []const Cache, pos: usize, capacity: usize = 4096, pos_dev: u64 };
    const Seg = struct { st: *State, row0: usize, rows: usize };
    const Mixer = union(enum) { mla: struct { dense_limit: usize }, kda: void };
    const Layer = struct { mixer: Mixer };
    const layers = [_]Layer{ .{ .mixer = .{ .mla = .{ .dense_limit = 1024 } } }, .{ .mixer = .kda }, .{ .mixer = .{ .mla = .{ .dense_limit = 2048 } } } };
    const ca = [_]Cache{ .{ .lc = 0x10000000, .ls = 0x11000000 }, .{ .lc = 0x12000000, .ls = 0x13000000 } };
    const cb = [_]Cache{ .{ .lc = 0x20000000, .ls = 0x21000000 }, .{ .lc = 0x22000000, .ls = 0x23000000 } };
    var a: State = .{ .caches = &ca, .pos = 509, .pos_dev = 0x30000000 };
    var b: State = .{ .caches = &cb, .pos = 511, .pos_dev = 0x31000000 };
    var segs = [_]Seg{ .{ .st = &a, .row0 = 0, .rows = 3 }, .{ .st = &b, .row0 = 3, .rows = 6 } };
    var buffers: struct { prefill: bool } = .{ .prefill = false };
    const weights = .{ .layers = &layers };
    const rank = .{ .b = &buffers, .w = &weights, .segs = &segs };
    Fake.uploads = 0;
    Fake.records = 0;
    Fake.waits = 0;
    try owner.prepare(rank, 9);
    try std.testing.expectEqual(@as(usize, 1), Fake.uploads);
    try std.testing.expectEqual(@as(usize, 1), Fake.records);
    try std.testing.expectEqual(@as(usize, 0), Fake.waits);
    for (0..2) |layer| {
        const table: [*]align(1) const dense.Row = @ptrCast(&Fake.copied[layer * 2048]);
        for (0..9) |r| {
            const st = if (r < 3) &a else &b;
            try std.testing.expectEqual(st.caches[layer].lc, table[r].cache);
            try std.testing.expectEqual(st.pos_dev, table[r].position);
            try std.testing.expectEqual(@as(i32, if (r < 3) 0 else 3), table[r].row0);
        }
    }
    a.pos = 1021;
    b.pos = 1023;
    b.pos_dev = 0x32000000;
    try owner.prepare(rank, 9);
    try std.testing.expect(plans[0] == null and plans[1] != null);
    try std.testing.expectEqual(@as(usize, 1), Fake.waits);
    try std.testing.expectEqual(@as(usize, 2), Fake.uploads);
    try std.testing.expectEqual(b.pos_dev, plans[1].?.table[8].position);
    buffers.prefill = true;
    try owner.prepare(rank, 9);
    try std.testing.expect(plans[0] == null and plans[1] == null);
    try std.testing.expectEqual(@as(usize, 2), Fake.waits);
    try std.testing.expectEqual(@as(usize, 2), Fake.uploads);
    buffers.prefill = false;
    a.pos = 509;
    b.pos = 511;
    segs[0].rows = 9;
    segs[1].row0 = 9;
    segs[1].rows = 16;
    try owner.prepare(rank, 25);
    try std.testing.expect(plans[0] == null and plans[1] == null);
    try std.testing.expectEqual(@as(usize, 2), Fake.uploads);
    owner.max_part_rows = 16;
    try owner.prepare(rank, 25);
    try std.testing.expect(plans[0] != null and plans[1] != null);
    try std.testing.expectEqual(@as(usize, 16), plans[0].?.most);
    try std.testing.expectEqual(@as(i32, 9), plans[1].?.table[24].row0);
    try std.testing.expectEqual(b.pos_dev, plans[1].?.table[24].position);
    try std.testing.expectEqual(@as(usize, 3), Fake.uploads);
    owner.max_part_rows = 8;
    segs[0].rows = 3;
    segs[1].row0 = 3;
    segs[1].rows = 6;
    a.pos = 1021;
    b.pos = 1023;
    segs[1].row0 = 4;
    try std.testing.expectError(error.BadWindow, owner.prepare(rank, 9));
    try std.testing.expectEqual(@as(usize, 3), Fake.uploads);
    segs[1].row0 = 3;
    Fake.fail_record = true;
    defer Fake.fail_record = false;
    try std.testing.expectError(error.NotReady, owner.prepare(rank, 9));
    try std.testing.expect(owner.unusable);
    try std.testing.expectEqual(@as(usize, 4), Fake.uploads);
    try std.testing.expectError(error.DenseUnavailable, owner.prepare(rank, 9));
    try std.testing.expectEqual(@as(usize, 4), Fake.uploads);
}
