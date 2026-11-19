//! Principal tests call the actual loader, projector, one-slice dispatch and buffer constructors.
const std = @import("std");
const cuda = @import("cuda");
const control = @import("cuda_q4_control_input.zig");
const kda = @import("cuda_kda_block.zig");
const mm = @import("cuda_mm.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

fn layer() kda.Layer {
    return .{ .proj = .{ .bf16 = 0x10000000 }, .fb = 0x20000000, .gb = 0x22000000, .o = .{ .bf16 = 0x24000000 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = 32, .dims = 4096 };
}

const Fake = struct {
    var allocations: usize = 0;
    var freed: usize = 0;
    var syncs: usize = 0;
    var calls: usize = 0;
    var fail: bool = false;
    var last_size: usize = 0;
    var last_alloc: u64 = 0;
    var tile_src: u64 = 0;
    var tile_dst: u64 = 0;
    var dims: [16][4]i32 = undefined; // rows,N,K,slices
    var output: [16]u64 = undefined;
    var functions: [16]usize = undefined;

    fn reset() void {
        allocations = 0;
        freed = 0;
        syncs = 0;
        calls = 0;
        fail = false;
        last_alloc = 0x40000000;
        last_size = 0;
    }
    fn alloc(p: *u64, size: usize) callconv(.c) cuda.abi.Result {
        allocations += 1;
        last_alloc += std.mem.alignForward(u64, @intCast(size), 4096) + 4096;
        p.* = last_alloc;
        last_size = size;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        freed += 1;
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zeroSync(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return 0;
    }
    fn value(comptime T: type, p: ?*anyopaque) T {
        return @as(*align(1) const T, @ptrCast(p.?)).*;
    }
    fn launch(f: cuda.abi.Function, _: u32, _: u32, gz: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const p = params.?;
        const at = calls;
        calls += 1;
        functions[at] = @intFromPtr(f);
        if (@intFromPtr(f) == 1) {
            tile_src = value(u64, p[0]);
            tile_dst = value(u64, p[1]);
            dims[at] = .{ 0, value(i32, p[2]), value(i32, p[3]), 0 };
        } else if (@intFromPtr(f) == 2) {
            dims[at] = .{ value(i32, p[4]), value(i32, p[5]), @intCast(value(i64, p[3])), 0 };
            output[at] = value(u64, p[2]);
        } else {
            dims[at] = .{ value(i32, p[7]), value(i32, p[8]), value(i32, p[9]), value(i32, p[10]) };
            output[at] = value(u64, p[4]);
            t.expectEqual(@as(u32, 1), gz) catch @panic("unexpected K slices");
        }
        return if (fail) cuda.abi.error_not_ready else 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = zeroSync;
        d.api.cuMemsetD8Async = zero;
        d.api.cuStreamSynchronize = sync;
        d.api.cuLaunchKernel = launch;
        return d;
    }
    fn kernels(d: *const cuda.Driver) mm.Kernels {
        var k: mm.Kernels = undefined;
        k.affine = null;
        k.pdl = false;
        k.prompt_fixed_k = false;
        k.tile = .{ .d = d, .handle = @ptrFromInt(1) };
        for (&k.by_rows, 0..) |*f, i| f.* = .{ .d = d, .handle = @ptrFromInt(10 + i) };
        k.prompt = .{ .d = d, .handle = @ptrFromInt(14) };
        return k;
    }
};

test "Q4 input control: parser preserves0/1 and agrees mode2 before load" {
    const E = struct {
        var v: ?[]const u8 = null;
        fn get(n: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, n, "TF_GLM_Q4_RECURRENT_BF16")) v else null;
        }
    };
    defer E.v = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(E.get, &problem);
    try t.expectEqual(@as(u8, 0), off.q4_recurrent_bf16);
    E.v = "0";
    try t.expectEqualDeep(off, try ranks.Settings.read(E.get, &problem));
    for ([_][]const u8{ "4", "02", "-1", "true", " 2" }) |bad| {
        E.v = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(E.get, &problem));
    }
    E.v = "2";
    var on = try ranks.Settings.read(E.get, &problem);
    E.v = "0";
    try t.expectEqual(@as(u8, 2), on.q4_recurrent_bf16);
    on.q4_affine_mse = true;
    on.q4_activation = true;
    on.dense_bits = 4;
    try @import("cuda_q4_recurrent.zig").require(on, 2);
    const F = ranks.Fingerprint;
    const a = F.of(2, 0, off, 8192, true, @splat(0), @splat(0), @splat(0));
    var baseline = on;
    baseline.q4_recurrent_bf16 = 1;
    const b = F.of(2, 0, baseline, 8192, true, @splat(0), @splat(0), @splat(0));
    const c = F.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 81), c.layout);
    try t.expectEqual(F.Field.q4_recurrent_bf16, ranks.mismatch(&.{ b, c }).?.field);
    try t.expectEqual(@as(u32, 0), a.q4_recurrent_bf16);
}

test "Q4 input control: actual loader tiles only original tail and transfers ownership after drain" {
    Fake.reset();
    var d = Fake.driver();
    const k = Fake.kernels(&d);
    var layers: [1]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = layer() };
    var w: weights.Weights = .{ .gpa = t.allocator, .config = undefined, .share = undefined, .layers = &layers };
    w.share.world = 2;
    w.share.rank = 0;
    defer {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(t.allocator);
    }
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    try control.prepare(&w, &k, s, &d);
    try t.expectEqual(@as(usize, 2359296), Fake.last_size);
    try t.expectEqual(@as(u64, 0x10000000 + 3 * 32 * 128 * 4096 * 2), Fake.tile_src);
    try t.expectEqual(Fake.last_alloc, Fake.tile_dst);
    try t.expectEqualDeep([4]i32{ 0, 288, 4096, 0 }, Fake.dims[0]);
    try t.expectEqual(@as(usize, 1), Fake.syncs);
    try t.expectEqual(@as(usize, 1), w.buffers.items.len);
    try t.expectEqual(Fake.last_alloc, layers[0].mixer.kda.control_tail);
    try t.expectEqual(@as(u64, 0x10000000), layers[0].mixer.kda.proj.bf16);
    try t.expectError(error.Q4ControlInputScope, control.prepare(&w, &k, s, &d));
    Fake.fail = true;
    layers[0].mixer.kda.control_tail = 0;
    try t.expectError(error.NotReady, control.prepare(&w, &k, s, &d));
    try t.expectEqual(@as(u64, 0), layers[0].mixer.kda.control_tail);
    try t.expectEqual(@as(usize, 1), w.buffers.items.len);
    try t.expectEqual(@as(usize, 1), Fake.freed);
}

test "Q4 input control: actual project is full projection then one-slice tail copy then both gates" {
    var d = Fake.driver();
    const kernels = Fake.kernels(&d);
    var glue: @import("cuda_glue.zig").Kernels = undefined;
    const set: @import("cuda_glue.zig").Set = .{ .glue = &glue, .mm = &kernels, .copy_rows = .{ .d = &d, .handle = @ptrFromInt(2) } };
    var l = layer();
    l.control_tail = 0x30000000;
    var s: kda.Scratch = undefined;
    s.p = 0x50000000;
    s.ka = 0x60000000;
    s.kg = 0x70000000;
    s.mm_part = 0x80000000;
    s.tmp = 0x90000000;
    for ([_]usize{ 1, 8, 16, 32, 64 }) |rows| {
        Fake.reset();
        s.tmp_bytes = control.scratchBytes(rows, l.heads);
        try kda.project(set, .{ .d = &d, .handle = null }, l, s, 0xA0000000, 4096, rows);
        try t.expectEqual(@as(usize, 5), Fake.calls);
        try t.expectEqualDeep([4]i32{ @intCast(rows), 12576, 4096, 1 }, Fake.dims[0]);
        try t.expectEqualDeep([4]i32{ @intCast(rows), 288, 4096, 1 }, Fake.dims[1]);
        try t.expectEqual(@as(usize, 2), Fake.functions[2]);
        try t.expectEqual(s.p + 3 * 32 * 128 * 2, Fake.output[2]);
        try t.expectEqualDeep([4]i32{ @intCast(rows), 4096, 128, 1 }, Fake.dims[3]);
        try t.expectEqualDeep(Fake.dims[3], Fake.dims[4]);
        try t.expectEqual(s.ka, Fake.output[3]);
        try t.expectEqual(s.kg, Fake.output[4]);
        Fake.reset();
        s.tmp_bytes -= 1;
        try t.expectError(error.Q4ControlInputScope, control.overwrite(set, .{ .d = &d, .handle = null }, l, s, 0xA0000000, 4096, rows));
        try t.expectEqual(@as(usize, 0), Fake.calls);
    }
    try t.expectEqual(@as(usize, 8), mm.splitK(288, 4096));
}

test "Q4 input control: actual shared/prefill/twin buffers own bounded independent temporaries" {
    Fake.reset();
    var d = Fake.driver();
    const config: @import("config.zig").Config = .{};
    const share = try @import("cuda_shard.zig").Share.of(&config, 0, 2);
    var layers: [1]weights.Layer = undefined;
    layers[0].mixer = .{ .kda = layer() };
    layers[0].ffn = .{ .dense = .{ .w = .{ .bf16 = .{ .gu = 17, .down = 17 } }, .width = share.dense.len(), .dims = config.hidden, .limit = 0 } };
    const w: weights.Weights = .{ .gpa = t.allocator, .config = config, .share = share, .layers = &layers, .affine_target = true };
    var off = try state.Buffers.init(&d, &w, 64, true, 8192);
    defer off.deinit();
    var off_twin = try state.Buffers.initTwin(&d, &w, &off, 8192);
    defer off_twin.deinit();
    try t.expectEqual(@as(usize, 0), off.kda.tmp_bytes);
    try t.expectEqual(off.kda.tmp, off_twin.kda.tmp);
    layers[0].mixer.kda.control_tail = 0x30000000;
    var on = try state.Buffers.init(&d, &w, 64, true, 8192);
    defer on.deinit();
    var twin = try state.Buffers.initTwin(&d, &w, &on, 8192);
    defer twin.deinit();
    var shared = try state.Buffers.initShared(&d, &w, 64, 8192);
    defer shared.deinit();
    for ([_]*const state.Buffers{ &on, &twin, &shared }) |b| {
        try t.expectEqual(@as(usize, 64 * 288 * 2), b.kda.tmp_bytes);
        try t.expect(b.kda.tmp >= b.arena.buf.ptr and b.kda.tmp + b.kda.tmp_bytes <= b.arena.buf.ptr + b.arena.buf.len);
    }
    try t.expect(on.kda.tmp != twin.kda.tmp and on.kda.tmp != shared.kda.tmp);
}
