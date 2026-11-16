const std = @import("std");
const cuda = @import("cuda");
const seg = @import("cuda_kda_segments.zig");
const Owner = @import("cuda_kda_owner.zig").Owner;
const Block = @import("cuda_kda_block.zig");

test "KDA shared owner builds real state tables and executes one original projection copy per sequence" {
    const Fake = struct {
        var uploads: usize = 0;
        var waits: usize = 0;
        var copies: usize = 0;
        var launches: usize = 0;
        var fail_record = false;
        var uploaded: [1152]u8 align(16) = undefined;
        fn copyH(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.debug.assert(dst == 0x90000000 and n == uploaded.len);
            @memcpy(&uploaded, @as([*]const u8, @ptrCast(src.?))[0..n]);
            uploads += 1;
            return 0;
        }
        fn copyD(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            const second = copies % 2 == 1;
            std.debug.assert(dst == (if (second) @as(u64, 0x30000000) else 0x10000000));
            std.debug.assert(src == 0xa0000000 + @as(u64, if (second) 3 * 6416 * 2 else 0));
            std.debug.assert(n == @as(usize, if (second) 6 else 3) * 6416 * 2);
            copies += 1;
            return 0;
        }
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return if (fail_record) cuda.abi.error_not_ready else 0;
        }
        fn wait(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
            waits += 1;
            return 0;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(z == 2);
            const id = @intFromPtr(f);
            std.debug.assert(if (id == 2) (x == 16 and y == 8 and bx == 128) else (x == 6 and y == 16 and bx == (if (id == 1) @as(u32, 512) else 128)));
            launches += 1;
            return 0;
        }
    };
    const State = struct {
        cur: [2]u1 = .{ 0, 1 },
        heads: usize = 16,
        proj: u64,
        proj_layer: usize = @import("cuda_state.zig").window_rows * 6416,
        conv: u64,
        conv_c: usize = 3 * 2048,
        save_k: u64,
        save_v: u64,
        save_g: u64,
        save_b: u64,
        rec: u64,
        pub fn recAt(s: @This(), parity: u1, li: usize) u64 {
            return s.rec + (@as(usize, parity) * 2 + li) * 1048576;
        }
    };
    const Piece = struct { st: *State, row0: usize, rows: usize };
    var states = [_]State{
        .{ .proj = 0x10000000, .conv = 0x11000000, .rec = 0x12000000, .save_k = 0x13000000, .save_v = 0x14000000, .save_g = 0x15000000, .save_b = 0x16000000 },
        .{ .proj = 0x30000000, .conv = 0x31000000, .rec = 0x32000000, .save_k = 0x33000000, .save_v = 0x34000000, .save_g = 0x35000000, .save_b = 0x36000000 },
    };
    const pieces = [_]Piece{ .{ .st = &states[0], .row0 = 0, .rows = 3 }, .{ .st = &states[1], .row0 = 3, .rows = 6 } };
    var layer: Block.Layer = undefined;
    layer.heads = 16;
    layer.conv_w = 0x70000000;
    layer.a_log = 0x71000000;
    layer.dt_bias = 0x72000000;
    layer.norm_w = 0x73000000;
    layer.lower = -3;
    layer.eps = 1e-6;
    const Mixer = union(enum) { kda: Block.Layer, mla: void };
    const Layer = struct { mixer: Mixer };
    const layers = [_]Layer{ .{ .mixer = .{ .kda = layer } }, .{ .mixer = .mla }, .{ .mixer = .{ .kda = layer } } };
    const weights = .{ .layers = &layers };
    var buffers = .{ .prefill = @as(bool, false), .rows = @as(usize, 64), .kproj = @as(u64, 0xa0000000), .kda = .{ .wide = .{ .q = @as(u64, 0xb0000000), .y = @as(u64, 0xc0000000) }, .ka = @as(u64, 0xd0000000), .kg = @as(u64, 0xe0000000), .saves = .{ .out = @as(u64, 0xf0000000) } } };
    const rank = .{ .w = &weights, .b = &buffers, .segs = &pieces };
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copyH;
    d.api.cuMemcpyDtoDAsync_v2 = Fake.copyD;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventSynchronize = Fake.wait;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var plans: [2]?seg.Plan = @splat(null);
    var staging: [1152]u8 align(16) = @splat(0);
    var o: Owner = .{ .gpa = std.testing.allocator, .module = undefined, .stream = stream, .kernels = .{ .prep = .{ .d = &d, .handle = @ptrFromInt(1) }, .step = .{ .d = &d, .handle = @ptrFromInt(2) }, .out = .{ .d = &d, .handle = @ptrFromInt(3) } }, .tables = .{ .d = &d, .ptr = 0x90000000, .len = staging.len }, .staging = .{ .d = &d, .bytes = &staging }, .uploaded = .{ .d = &d, .handle = null }, .plans = &plans, .rows = 64 };
    Fake.uploads = 0;
    Fake.waits = 0;
    Fake.copies = 0;
    Fake.launches = 0;
    try o.prepare(rank, 9);
    try std.testing.expectEqual(@as(usize, 0), Fake.uploads);
    try std.testing.expect(plans[0] == null);
    try std.testing.expectEqual(@as(usize, 0), Fake.launches);
    o.min_segments = 2;
    try o.prepare(rank, 9);
    try std.testing.expectEqual(@as(usize, 1), Fake.uploads);
    for (0..2) |li| {
        const rows: [*]align(1) const seg.Row = @ptrCast(&Fake.uploaded[li * 576]);
        for (0..2) |i| {
            try std.testing.expectEqual(states[i].recAt(states[i].cur[li], li), rows[i].state_in);
            try std.testing.expectEqual(states[i].proj + li * @import("cuda_state.zig").window_rows * 6416 * 2, rows[i].p);
            try std.testing.expectEqual(states[i].save_k + li * @import("cuda_state.zig").window_rows * 2048 * 4, rows[i].k);
        }
    }
    const forward = .{ .d = &d, .stream = stream, .kda_wide = true };
    try std.testing.expect(try o.run(forward, rank, layer, 0, 9));
    try std.testing.expectEqual(@as(usize, 2), Fake.copies);
    try std.testing.expectEqual(@as(usize, 3), Fake.launches);
    states[0].cur[0] = 1;
    try o.prepare(rank, 9);

    try std.testing.expectEqual(states[0].recAt(1, 0), plans[0].?.table[0].state_in);
    try std.testing.expectEqual(@as(usize, 1), Fake.waits);
    Fake.fail_record = true;
    defer Fake.fail_record = false;
    try std.testing.expectError(error.NotReady, o.prepare(rank, 9));
    try std.testing.expectError(error.KdaSegmentsUnavailable, o.prepare(rank, 9));
    try std.testing.expectEqual(@as(usize, 3), Fake.uploads);
}

test "KDA adaptive actual solo and shared verify constructors bind the owned table" {
    const Engine = @import("cuda_engine.zig").Engine;
    const fw = @import("cuda_forward.zig");
    var e: Engine = undefined;
    var driver: cuda.Driver = undefined;
    e.d = &driver;
    e.lib = @ptrFromInt(0x1000);
    e.comm = null;
    e.group = null;
    e.stream = .{ .d = &driver, .handle = null };
    e.settings = .{};
    e.drafter = null;
    e.images = null;
    e.pair = null;
    e.l2pf = null;
    e.dense_segments = null;
    e.kda_segments = null;
    var a: fw.State = undefined;
    var z: fw.State = undefined;
    const segs = [_]fw.Seg{ .{ .st = &a, .row0 = 0, .rows = 3 }, .{ .st = &z, .row0 = 3, .rows = 6 } };
    var one: [1]fw.Rank = undefined;
    try std.testing.expect(e.sharedView(&e.bd, &segs, &one).kda_segments == null);
    e.kda_segments = @as(Owner, undefined);
    const shared = e.sharedView(&e.bd, &segs, &one);
    try std.testing.expect(shared.kda_segments.? == &e.kda_segments.?);
    try std.testing.expect(shared.ranks[0].segs.ptr == &segs);
    try std.testing.expect(shared.ranks[0].segs[1].st == &z);
    var seq: @typeInfo(@FieldType(Engine, "cur")).pointer.child = undefined;
    e.cur = &seq;
    const solo = e.view(&e.bd, &one);
    try std.testing.expect(solo.kda_segments.? == &e.kda_segments.?);
}
