//! Record real affine target launches and cached settings without a CUDA library.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const affine = @import("cuda_affine_target.zig");
const tile = @import("cuda_tile_affine.zig");
const t = std.testing;

test "affine target: strict OFF, scope and rank fingerprint" {
    const Env = struct {
        var mode: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q4_AFFINE_MSE")) mode else null;
        }
    };
    defer Env.mode = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.mode = "0";
    try t.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "true", "01", " 1", "2" }) |bad| {
        Env.mode = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.mode = "1";
    var on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(on.q4_affine_mse);
    try t.expectError(error.AffineTargetScope, tile.require(on, 4, false));
    on.dense_bits = 4;
    try tile.require(on, 4, false);
    try tile.require(off, 1, true);
    try tile.require(on, 2, false);
    try t.expectError(error.AffineTargetScope, tile.require(on, 3, false));
    try t.expectError(error.AffineTargetScope, tile.require(on, 4, true));
    inline for (.{ "pages", "verify_graphs" }) |field| {
        var bad = on;
        @field(bad, field) = true;
        try t.expectError(error.AffineTargetScope, tile.require(bad, 4, false));
    }
    const FP = ranks.Fingerprint;
    var s0 = on;
    s0.q4_affine_mse = false;
    const a = FP.of(4, 0, s0, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.q4_affine_mse, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_Q4_AFFINE_MSE", @import("cuda_rank_names.zig").explain(FP.Field.q4_affine_mse));
    try tile.prepare(@as(*@import("cuda_forward.zig").Modules, undefined), false);
}

test "affine target: storage registry distinguishes affine scale and bias from symmetric Q4" {
    const pointers = [_]u64{0x100000};
    const bits = [_]u8{4};
    const legacy: mm.Packed = .{ .ptrs = &pointers, .bits = &bits };
    var p = legacy;
    p.affine = true;
    const sizes = q4.bytes(160, 4096);
    const w = p.tiled(pointers[0], 160, 4096);
    try t.expectEqual(pointers[0] + sizes[0], w.s);
    try t.expectEqual(w.s + sizes[1], w.bias);
    try t.expectEqual(@as(usize, 256), w.npad);
    try t.expectEqual(@as(u64, 0), legacy.tiled(pointers[0], 160, 4096).bias);
    try t.expectEqual(@as(u8, 16), p.tiled(0x900000, 160, 4096).bits);
    for ([_]usize{ 1, 8, 16, 64, 128, 2048 }) |rows|
        try t.expectEqual(@as(usize, @min(rows, 64)) * 4096 / 64 * 4, affine.scratchBytes(rows, 4096));
}

const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };
const Call = struct { kind: usize, x: u64, xs: u64, out: u64, rows: usize, ldx: usize, k: usize };
const Fake = struct {
    var calls: [128]Call = undefined;
    var count: usize = 0;
    var fail_at: ?usize = null;
    fn record(f: cuda.abi.Function, gx: u32, gy: u32, bz: u32, smem: u32, args: ?[*]?*anyopaque) cuda.abi.Result {
        std.debug.assert(bz == 1 and count < calls.len);
        const a = args.?;
        const kind = @intFromPtr(f);
        var c: Call = .{ .kind = kind, .x = read(u64, a[0]), .xs = read(u64, a[1]), .out = 0, .rows = 0, .ldx = 0, .k = 0 };
        if (kind == 1) {
            c.rows = gx;
            c.ldx = @intCast(read(i64, a[2]));
            c.k = @as(usize, @intCast(read(i64, a[3]))) * 64;
            std.debug.assert(gy == (c.k / 64 + 15) / 16 and smem == 0);
        } else {
            const parts = read(Parts, a[2]);
            std.debug.assert(parts.count == 1 and parts.p[0].biases == 0x300000);
            c.out = parts.p[0].out;
            c.rows = @intCast(read(c_int, a[3]));
            c.k = @intCast(read(c_int, a[4]));
            c.ldx = @intCast(read(c_int, a[5]));
            std.debug.assert(gy == 1 and smem == (if (c.rows <= 16) @as(u32, 35328) else 43008));
        }
        calls[count] = c;
        count += 1;
        return if (fail_at == count - 1) cuda.abi.error_not_ready else 0;
    }
    fn read(comptime T: type, p: ?*anyopaque) T {
        return @as(*align(1) const T, @ptrCast(p.?)).*;
    }
    fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(gz == 1 and bx == 128 and by == 1);
        return record(f, gx, gy, bz, smem, args);
    }
    fn extended(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return record(f, c.grid_x, c.grid_y, c.block_z, c.shared_bytes, args);
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = extended;
        return d;
    }
};

test "affine target: actual matmul windows, per-buffer sums, strides and failure stops" {
    const d = Fake.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const fun = struct {
        fn at(driver: *const cuda.Driver, n: usize) cuda.Function {
            return .{ .d = driver, .handle = @ptrFromInt(n) };
        }
    }.at;
    var kernels: mm.Kernels = undefined;
    const q: q4.Kernels = .{ .group = fun(&d, 2), .group_f32 = fun(&d, 3), .rows = .{ fun(&d, 4), fun(&d, 5) }, .pack_dense = undefined, .transpose = undefined, .gb10 = true };
    kernels.affine = .{ .q4 = q, .sums = fun(&d, 1) };
    const w: mm.Tiled = .{ .w = 0x100000, .s = 0x200000, .bias = 0x300000, .bits = 4, .n = 160, .k = 4096, .npad = 256 };
    for ([_]usize{ 1, 8, 16, 17, 32, 63, 64, 65, 127, 128, 2048 }) |rows| {
        for ([_]bool{ false, true }) |wide| {
            Fake.count = 0;
            Fake.fail_at = null;
            try mm.matmul(&kernels, stream, 0x400000, 4160, w, 0x800000, 0x700000, rows, wide, true);
            try t.expectEqual((rows + 63) / 64 * 2, Fake.count);
            var row: usize = 0;
            for (0..Fake.count / 2) |i| {
                const a = Fake.calls[2 * i];
                const b = Fake.calls[2 * i + 1];
                const n = @min(rows - row, 64);
                try t.expectEqual(@as(usize, 1), a.kind);
                try t.expectEqual(n, a.rows);
                try t.expectEqual(@as(u64, 0x400000 + row * 4160 * 2), a.x);
                try t.expectEqual(a.x, b.x);
                try t.expectEqual(@as(u64, 0x700000), b.xs);
                try t.expectEqual(@as(u64, 0x800000 + row * 160 * @as(usize, if (wide) 4 else 2)), b.out);
                try t.expectEqual(@as(usize, if (n <= 16) 2 else 4) + @intFromBool(wide), b.kind);
                row += n;
            }
        }
    }
    for (0..4) |fail| {
        Fake.count = 0;
        Fake.fail_at = fail;
        try t.expectError(error.NotReady, mm.matmul(&kernels, stream, 0x400000, 4160, w, 0x800000, 0x700000, 128, true, false));
        try t.expectEqual(fail + 1, Fake.count);
    }
    Fake.count = 0;
    Fake.fail_at = null;
    try t.expectError(error.BadAffineShape, mm.matmul(&kernels, stream, 0x400000, 4160, w, 0x800000, 0, 64, false, false));
    try t.expectEqual(@as(usize, 0), Fake.count);
}

test "affine target: actual paired buffers retain independent sum scratch only when ON" {
    const FakeMemory = struct {
        var next: u64 = 0x100000000;
        var live: usize = 0;
        fn alloc(out: *cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
            out.* = next;
            next += std.mem.alignForward(u64, n, 256) + 256;
            live += 1;
            return 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            live -= 1;
            return 0;
        }
        fn zero(_: cuda.abi.DevicePtr, _: u8, _: usize) callconv(.c) cuda.abi.Result {
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = FakeMemory.alloc;
    d.api.cuMemFree_v2 = FakeMemory.free;
    d.api.cuMemsetD8_v2 = FakeMemory.zero;
    const config: @import("config.zig").Config = .{};
    var w: @import("cuda_weights.zig").Weights = .{
        .gpa = t.allocator,
        .config = config,
        .share = try @import("cuda_shard.zig").Share.of(&config, 0, 4),
    };
    for ([_]bool{ false, true }) |on| {
        w.affine_target = on;
        var a = try @import("cuda_state.zig").Buffers.init(&d, &w, 2048, true, 4096);
        defer a.deinit();
        var b = try @import("cuda_state.zig").Buffers.initTwin(&d, &w, &a, 4096);
        defer b.deinit();
        try t.expectEqual(on, a.kda.mm_part != b.kda.mm_part);
        inline for (.{ "kda", "dsa", "dense", "moe" }) |field|
            try t.expectEqual(b.kda.mm_part, @field(b, field).mm_part);
        try t.expect(b.kda.mm_part >= (if (on) b.arena.buf.ptr else a.arena.buf.ptr));
        try t.expect(b.kda.mm_part < (if (on) b.arena.buf.ptr + b.arena.buf.len else a.arena.buf.ptr + a.arena.buf.len));
    }
    try t.expectEqual(@as(usize, 0), FakeMemory.live);
}
