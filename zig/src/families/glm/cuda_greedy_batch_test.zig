//! The real greedy-row picker against recording CUDA/NCCL callbacks; no dynamic libraries or devices.
const std = @import("std");
const cuda = @import("cuda");
const pick = @import("cuda_engine_pick.zig");
const Engine = @import("cuda_engine.zig").Engine;
const ranks = @import("cuda_ranks.zig");
const cap = @import("cuda_engine.zig").shared_rows;
const t = std.testing;
const Kind = enum { top, gather, sync, check, download };
const Fail = enum { none, top, gather, sync, check, download };
const logits_base = 0x10000000;
const stream_at: cuda.abi.Stream = @ptrFromInt(0x1234);
const top_at: cuda.abi.Function = @ptrFromInt(0x5678);

const Fake = struct {
    var world: usize = 4;
    var rank: usize = 0;
    var rows: usize = 1;
    var width: usize = 38784;
    var fail: Fail = .none;
    var calls: [cap + 4]Kind = undefined;
    var n: usize = 0;
    var launched: usize = 0;
    var records: [ranks.max_world * cap][4]u32 align(16) = undefined;

    fn reset(w: usize, q: usize, r: usize, failure: Fail) void {
        world = w;
        rank = q;
        rows = r;
        width = if (q < 2) 38784 else 38656;
        fail = failure;
        n = 0;
        launched = 0;
    }
    fn note(kind: Kind) void {
        calls[n] = kind;
        n += 1;
    }
    fn record(q: usize, r: usize) [4]u32 {
        const value: f32 = if (q == r % world) 2 else 1;
        return .{ @bitCast(value), @intCast(1000 * r + q), 0xaabbccdd, 0x12345678 };
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(f == top_at and s == stream_at);
        std.debug.assert(gx == 1 and gy == 1 and gz == 1 and bx == 1024 and by == 1 and bz == 1 and shared == 0);
        const args = params.?;
        const input = @as(*align(1) const u64, @ptrCast(args[0].?)).*;
        const v = @as(*align(1) const i64, @ptrCast(args[1].?)).*;
        const lo = @as(*align(1) const u32, @ptrCast(args[2].?)).*;
        const output = @as(*align(1) const u64, @ptrCast(args[3].?)).*;
        std.debug.assert(input == logits_base + launched * width * 2 and v == width and lo == 17);
        std.debug.assert(output == @intFromPtr(&records) + (rank * rows + launched) * 16);
        note(.top);
        if (fail == .top and launched == 1) return cuda.abi.error_not_ready;
        records[rank * rows + launched] = record(rank, launched);
        launched += 1;
        return 0;
    }
    fn gather(mine: u64, all: u64, bytes: usize, kind: cuda.nccl.DataType, _: cuda.nccl.Comm, s: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(s == stream_at and kind == .u8 and launched == rows);
        std.debug.assert(bytes == rows * 16 and all == @intFromPtr(&records));
        std.debug.assert(mine == all + rank * bytes);
        for (0..rows) |r| std.debug.assert(std.mem.eql(u32, &records[rank * rows + r], &record(rank, r)));
        note(.gather);
        if (fail == .gather) return 1;
        for (0..world) |q| for (0..rows) |r| {
            records[q * rows + r] = record(q, r);
        };
        return 0;
    }
    fn sync(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        std.debug.assert(s == stream_at and calls[n - 1] == .gather);
        note(.sync);
        return if (fail == .sync) cuda.abi.error_not_ready else 0;
    }
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, bytes: usize) callconv(.c) cuda.abi.Result {
        std.debug.assert(src == @intFromPtr(&records) and bytes == world * rows * 16);
        std.debug.assert(calls[n - 1] == .sync or calls[n - 1] == .check);
        note(.download);
        if (fail == .download) return cuda.abi.error_not_ready;
        const target: [*]u8 = @ptrCast(dst.?);
        @memcpy(target[0..bytes], std.mem.sliceAsBytes(records[0 .. world * rows]));
        return 0;
    }
    fn fits(_: *anyopaque, _: usize) bool {
        return true;
    }
    fn grouped(_: *anyopaque, s: cuda.abi.Stream, mine: u64, all: u64, bytes: usize) !void {
        if (gather(mine, all, bytes, .u8, null, s) != 0) return error.NcclFailed;
    }
    fn check(_: *anyopaque) !void {
        std.debug.assert(calls[n - 1] == .sync);
        note(.check);
        if (fail == .check) return error.InjectedGatherCheck;
    }
    fn noOp(_: *anyopaque) void {}

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(launch);
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxGetDevice = device;
        d.api.cuDeviceGetAttribute = attribute;
        d.api.cuMemcpyDtoH_v2 = download;
        return d;
    }
    fn library() cuda.nccl.Library {
        var lib: cuda.nccl.Library = undefined;
        lib.api.ncclAllGather = gather;
        return lib;
    }
    fn engine(d: *const cuda.Driver, lib: *const cuda.nccl.Library) Engine {
        var e: Engine = undefined;
        e.d = d;
        e.lib = lib;
        e.comm = null;
        e.group = null;
        e.share.world = world;
        e.share.rank = rank;
        e.share.vocab = .{ .lo = 17, .hi = 17 + width };
        e.pairs = .{ .d = d, .ptr = @intFromPtr(&records), .len = world * cap * 16, .borrowed = true };
        e.stream = .{ .d = d, .handle = stream_at };
        e.m.glue.top1 = .{ .d = d, .handle = top_at };
        e.profiling_prompt = null;
        return e;
    }
    fn group(lib: cuda.nccl.Library) cuda.tp_group.Group {
        return .{ .lib = lib, .comm = null, .world = world, .rank = rank, .multi = .{ .ctx = &n, .fits = fits, .gather = grouped, .check = check, .abort = noOp, .deinit = noOp } };
    }
};

test "greedy batching: real picker launches original row blocks then one gather and readback" {
    var d = Fake.driver();
    const lib = Fake.library();
    defer cuda.download_stage.forget();
    for ([_]bool{ false, true }) |borrowed| for ([_]usize{ 1, 4 }) |world| for (0..world) |rank| {
        Fake.reset(world, rank, 1, .none);
        var e = Fake.engine(&d, &lib);
        var g = Fake.group(lib);
        if (borrowed) e.group = &g;
        for ([_]usize{ 8, 1, 7, 2, 6, 3, 5, 4, 9, 15, 16, 17, 31, 32, 33, 63, 64 }) |rows| {
            Fake.reset(world, rank, rows, .none);
            var out: [cap + 2]u32 = @splat(0xdeadbeef);
            if (rows <= @import("cuda_forward.zig").window_rows) try pick.greedyRows(&e, logits_base, rows, out[1 .. rows + 1]) else try pick.greedySharedRows(&e, logits_base, rows, out[1 .. rows + 1]);
            try t.expectEqual(rows + @as(usize, if (borrowed) 4 else 3), Fake.n);
            for (Fake.calls[0..rows]) |kind| try t.expectEqual(Kind.top, kind);
            const tail: []const Kind = if (borrowed) &.{ .gather, .sync, .check, .download } else &.{ .gather, .sync, .download };
            try t.expectEqualSlices(Kind, tail, Fake.calls[rows..Fake.n]);
            for (0..rows) |r| try t.expectEqual(@as(u32, @intCast(1000 * r + r % world)), out[r + 1]);
            try t.expectEqual(@as(u32, 0xdeadbeef), out[0]);
            try t.expectEqual(@as(u32, 0xdeadbeef), out[rows + 1]);
            Fake.reset(world, rank, 1, .none);
            try t.expectEqual(@as(u32, 0), try pick.greedy(&e, logits_base));
        }
    };
}

test "greedy batching: launch gather sync check and download errors do not publish tokens" {
    var d = Fake.driver();
    const lib = Fake.library();
    defer cuda.download_stage.forget();
    for ([_]Fail{ .top, .gather, .sync, .check, .download }) |fail| {
        Fake.reset(4, 2, 3, fail);
        var e = Fake.engine(&d, &lib);
        var g = Fake.group(lib);
        if (fail == .check) e.group = &g;
        var out: [3]u32 = @splat(0xdeadbeef);
        const expected = switch (fail) {
            .gather => error.NcclFailed,
            .check => error.InjectedGatherCheck,
            else => error.NotReady,
        };
        try t.expectError(expected, pick.greedyRows(&e, logits_base, 3, &out));
        try t.expectEqualSlices(u32, &@as([3]u32, @splat(0xdeadbeef)), &out);
        const last: Kind = switch (fail) {
            .top => .top,
            .gather => .gather,
            .sync => .sync,
            .check => .check,
            .download => .download,
            .none => unreachable,
        };
        try t.expectEqual(last, Fake.calls[Fake.n - 1]);
    }
}

test "greedy batching: invalid rows rank and scratch refuse before any enqueue" {
    var d = Fake.driver();
    const lib = Fake.library();
    Fake.reset(4, 0, 1, .none);
    var e = Fake.engine(&d, &lib);
    var out: [cap + 1]u32 = undefined;
    try t.expectError(error.WindowTooWide, pick.greedyRows(&e, logits_base, 0, &out));
    try t.expectError(error.WindowTooWide, pick.greedyRows(&e, logits_base, cap + 1, &out));
    try t.expectError(error.WindowTooWide, pick.greedyRows(&e, logits_base, 2, out[0..1]));
    e.share.rank = 4;
    try t.expectError(error.BadWorld, pick.greedyRows(&e, logits_base, 1, &out));
    e.share.rank = 0;
    e.share.world = 0;
    try t.expectError(error.BadWorld, pick.greedyRows(&e, logits_base, 1, &out));
    e.share.world = ranks.max_world + 1;
    try t.expectError(error.BadWorld, pick.greedyRows(&e, logits_base, 1, &out));
    e.share.world = 4;
    e.pairs.len = 63;
    try t.expectError(error.PickScratchTooSmall, pick.greedyRows(&e, logits_base, 1, &out));
    try t.expectEqual(@as(usize, 0), Fake.n);
}

test "greedy batching: actual shared dispatcher scatters uneven parts after one full-round gather" {
    const fw = @import("cuda_forward.zig");
    const Seq = @import("cuda_engine_types.zig").Seq;
    var d = Fake.driver();
    const lib = Fake.library();
    defer cuda.download_stage.forget();
    for ([_]usize{ 9, 64 }) |rows| {
        Fake.reset(4, 3, rows, .none);
        var e = Fake.engine(&d, &lib);
        e.settings.greedy_batch = true;
        e.settings.greedy_shared = true;
        e.structured = false;
        e.images = null;
        e.c.hidden = 4096;
        var states: [8]Seq = undefined;
        var slots: [8]?*Seq = undefined;
        var parts: [8]@import("cuda_engine_types.zig").Part = undefined;
        var segs: [8]fw.Seg = undefined;
        var outputs: [8][18]u32 = @splat(@splat(0xdeadbeef));
        var ids: [16]u32 = @splat(1);
        const count: usize = if (rows == 9) 2 else 4;
        var row0: usize = 0;
        for (0..count) |i| {
            const n: usize = if (rows == 9) (if (i == 0) 3 else 6) else 16;
            states[i].sampling = null;
            states[i].st.pos = 117 * i;
            slots[i] = &states[i];
            parts[i] = .{ .slot = @intCast(i), .ids = ids[0..n], .out = outputs[i][1 .. n + 1] };
            segs[i] = .{ .st = &states[i].st, .row0 = row0, .rows = n };
            row0 += n;
        }
        e.seqs = .empty;
        e.seqs.items = slots[0..count];
        e.seqs.capacity = count;
        var b: fw.Buffers = undefined;
        b.logits = logits_base;
        b.fnormed = 0x40000000;
        try @import("cuda_shared_pick.zig").draw(&e, &b, parts[0..count], segs[0..count], rows);
        try t.expectEqual(rows + 3, Fake.n);
        for (0..count) |i| {
            for (0..segs[i].rows) |j| {
                const r = segs[i].row0 + j;
                try t.expectEqual(@as(u32, @intCast(1000 * r + r % 4)), outputs[i][j + 1]);
            }
            try t.expectEqual(@as(u32, 0xdeadbeef), outputs[i][0]);
            try t.expectEqual(@as(u32, 0xdeadbeef), outputs[i][segs[i].rows + 1]);
            try t.expectEqual(b.fnormed + segs[i].row0 * e.c.hidden * 2, states[i].rows_at);
        }
        try t.expectEqual(@as(u32, @intCast(count - 1)), e.slot);
    }
}
