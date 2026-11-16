//! Q4's actual loader and launch calls against recording Driver callbacks, without libraries or a GPU.
const std = @import("std");
const cuda = @import("cuda");
const q4 = @import("cuda_q4.zig");
const t = std.testing;
const narrow: cuda.abi.Function = @ptrFromInt(0x1000);
const wide: cuda.abi.Function = @ptrFromInt(0x2000);
const stream_at: cuda.abi.Stream = @ptrFromInt(0x3000);
const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };

const Fake = struct {
    var missing: bool = false;
    var lookup_error: bool = false;
    var attr_error: bool = false;
    var launch_error: bool = false;
    var lookups: usize = 0;
    var wide_lookups: usize = 0;
    var attrs: usize = 0;
    var launches: usize = 0;
    var cap: c_int = 121;
    var f: cuda.abi.Function = null;
    var grid: u32 = 0;
    var cluster: u32 = 1;
    var pdl: bool = false;
    var part: Parts = undefined;
    var scalar: [5]c_int = undefined;
    var pointers: [2]u64 = undefined;

    fn reset() void {
        missing = false;
        lookup_error = false;
        attr_error = false;
        launch_error = false;
        lookups = 0;
        wide_lookups = 0;
        attrs = 0;
        launches = 0;
        cap = 121;
    }
    fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
        const n = std.mem.span(name);
        lookups += 1;
        if (std.mem.eql(u8, n, "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii")) {
            wide_lookups += 1;
            if (missing) return cuda.abi.error_not_found;
            if (lookup_error) return cuda.abi.error_not_ready;
            out.* = wide;
        } else if (std.mem.eql(u8, n, "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii")) {
            out.* = narrow;
        } else if (std.mem.indexOf(u8, n, "ILi64ELi64ELi64ELi1ELi4ELi4E") != null) {
            if (missing) return cuda.abi.error_not_found;
            out.* = @ptrFromInt(if (std.mem.indexOf(u8, n, "ELb1E") != null) @as(usize, 0x8000) else 0x7000);
        } else if (std.mem.eql(u8, n, "tf_pack_dense")) {
            out.* = @ptrFromInt(0x4000);
        } else {
            std.debug.assert(std.mem.eql(u8, n, "tf_transpose_pad16"));
            out.* = @ptrFromInt(0x5000);
        }
        return 0;
    }
    fn attr(fun: cuda.abi.Function, kind: cuda.abi.FunctionAttribute, bytes: c_int) callconv(.c) cuda.abi.Result {
        std.debug.assert((fun == narrow or fun == wide or fun == @as(cuda.abi.Function, @ptrFromInt(0x7000)) or fun == @as(cuda.abi.Function, @ptrFromInt(0x8000))) and kind == .max_dynamic_shared_size_bytes and (bytes == 35328 or bytes == 43008));
        attrs += 1;
        return if (attr_error) cuda.abi.error_not_ready else 0;
    }
    fn device(out: *c_int, kind: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = switch (kind) {
            .compute_capability_major => @divTrunc(cap, 10),
            .compute_capability_minor => @mod(cap, 10),
            else => unreachable,
        };
        return 0;
    }
    fn record(fun: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, s: cuda.abi.Stream, params: ?[*]?*anyopaque) cuda.abi.Result {
        std.debug.assert(gy == 1 and gz == 1 and bx == 128 and by == 1 and bz == 1 and (smem == 35328 or smem == 43008) and s == stream_at);
        f = fun;
        grid = gx;
        launches += 1;
        const p = params.?;
        for (0..2) |i| pointers[i] = @as(*align(1) const u64, @ptrCast(p[i].?)).*;
        part = @as(*align(1) const Parts, @ptrCast(p[2].?)).*;
        for (&scalar, 3..) |*v, i| v.* = @as(*align(1) const c_int, @ptrCast(p[i].?)).*;
        return if (launch_error) cuda.abi.error_not_ready else 0;
    }
    fn launch(fun: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, s: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        cluster = 1;
        pdl = false;
        return record(fun, gx, gy, gz, bx, by, bz, smem, s, p);
    }
    fn extended(cfg: *const cuda.abi.LaunchConfig, fun: cuda.abi.Function, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        cluster = 1;
        pdl = false;
        for (cfg.attrs.?[0..cfg.num_attrs]) |a| switch (a.id) {
            .cluster_dimension => {
                std.debug.assert(a.value.cluster_dim.y == 1 and a.value.cluster_dim.z == 1);
                cluster = a.value.cluster_dim.x;
            },
            .programmatic_stream_serialization => pdl = a.value.int == 1,
            else => unreachable,
        };
        return record(fun, cfg.grid_x, cfg.grid_y, cfg.grid_z, cfg.block_x, cfg.block_y, cfg.block_z, cfg.shared_bytes, cfg.stream, p);
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuModuleGetFunction = get;
        d.api.cuFuncSetAttribute = attr;
        d.api.cuDeviceGetAttribute = device;
        d.api.cuLaunchKernel = launch;
        d.api.cuLaunchKernelEx = extended;
        return d;
    }
    fn load(d: *const cuda.Driver) !q4.Kernels {
        const m: cuda.Module = .{ .d = d, .handle = @ptrFromInt(0x6000) };
        const ctx: cuda.Context = .{ .d = d, .device = 0, .handle = null };
        return q4.Kernels.load(m, m, &ctx);
    }
};

test "GLM Q4 wide: BF16 load never asks an old asset for the new symbol" {
    Fake.reset();
    Fake.missing = true;
    const d = Fake.driver();
    var ks = try Fake.load(&d);
    try t.expect(ks.group_f32 == null);
    try t.expectEqual(@as(usize, 3), Fake.lookups);
    try t.expectEqual(@as(usize, 0), Fake.wide_lookups);
    try t.expectEqual(@as(usize, 1), Fake.attrs);
    const s: cuda.Stream = .{ .d = &d, .handle = stream_at };
    const w: q4.Q4 = .{ .w = 1, .s = 2, .b = 3, .n = 4096, .npad = 4096, .k = 1024 };
    try t.expectError(error.Q4F32NotEnabled, q4.matmulWide(&ks, s, 4, 1024, 5, w, 6, 1));
    try t.expectEqual(@as(usize, 3), Fake.lookups);
    try t.expectEqual(@as(usize, 0), Fake.launches);
    try q4.matmul(&ks, s, 4, 1024, 5, w, 6, 1);
    try t.expectEqual(narrow, Fake.f);
}

test "GLM Q4 wide: explicit symbol and attribute failures do not enable or launch" {
    for (0..3) |failure| {
        Fake.reset();
        const d = Fake.driver();
        var ks = try Fake.load(&d);
        Fake.missing = failure == 0;
        Fake.lookup_error = failure == 1;
        Fake.attr_error = failure == 2;
        const m: cuda.Module = .{ .d = &d, .handle = null };
        if (failure == 0) try t.expectError(error.MissingQ4F32Kernel, q4.enableWide(&ks, m)) else try t.expectError(error.NotReady, q4.enableWide(&ks, m));
        try t.expect(ks.group_f32 == null);
        try t.expectEqual(@as(usize, 0), Fake.launches);
        try t.expectEqual(@as(usize, if (failure == 2) 2 else 1), Fake.attrs);
    }
    Fake.reset();
    const d = Fake.driver();
    const other = Fake.driver();
    var ks = try Fake.load(&d);
    try t.expectError(error.Q4ModuleDriverMismatch, q4.enableWide(&ks, .{ .d = &other, .handle = null }));
    try t.expectEqual(@as(usize, 0), Fake.wide_lookups);
}

test "GLM Q4 wide: actual BF16 and FP32 launch ABI agrees across rows strides and split K" {
    Fake.reset();
    const d = Fake.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = stream_at };
    for ([_]c_int{ 121, 120, 90 }) |cap| {
        Fake.cap = cap;
        var ks = try Fake.load(&d);
        try q4.enableWide(&ks, .{ .d = &d, .handle = null });
        for ([_][2]usize{ .{ 4096, 1024 }, .{ 4096, 4096 }, .{ 64, 4096 }, .{ 65, 1024 }, .{ 127, 512 }, .{ 129, 2048 } }) |shape| {
            const w: q4.Q4 = .{ .w = 0x10000, .s = 0x20000, .b = 0x30000, .n = shape[0], .npad = q4.npadOf(shape[0]), .k = shape[1] };
            for (1..17) |rows| for ([_]usize{ w.k, w.k + 64 }) |ldx| {
                for ([_]bool{ false, true }) |f32_out| {
                    if (f32_out) try q4.matmulWide(&ks, s, 0x40002, ldx, 0x50000, w, 0x60000, rows) else try q4.matmul(&ks, s, 0x40002, ldx, 0x50000, w, 0x60000, rows);
                    try t.expectEqual(if (f32_out) wide else narrow, Fake.f);
                    try t.expectEqualSlices(u64, &.{ 0x40002, 0x50000 }, &Fake.pointers);
                    try t.expectEqual(@as(c_int, 1), Fake.part.count);
                    const p = Fake.part.p[0];
                    try t.expectEqual(w.w, p.w);
                    try t.expectEqual(w.s, p.scales);
                    try t.expectEqual(w.b, p.biases);
                    try t.expectEqual(@as(u64, 0x60000), p.out);
                    try t.expectEqual(@as(c_int, @intCast(w.n)), p.n);
                    try t.expectEqual(@as(c_int, @intCast(w.npad)), p.npad);
                    const sk = q4.splitK(w.n, w.k);
                    const tiles = (w.n + 63) / 64;
                    try t.expectEqual(@as(c_int, @intCast(sk)), p.sk);
                    try t.expectEqual(@as(c_int, @intCast(tiles)), p.tiles);
                    try t.expectEqual(@as(c_int, 0), p.first);
                    const zero: Part = std.mem.zeroes(Part);
                    for (Fake.part.p[1..]) |unused| try t.expectEqualDeep(zero, unused);
                    try t.expectEqualSlices(c_int, &.{ @intCast(rows), @intCast(w.k), @intCast(if (rows == 1) w.k else ldx), 1, @intCast(sk) }, &Fake.scalar);
                    try t.expectEqual(@as(u32, @intCast(tiles * sk)), Fake.grid);
                    try t.expectEqual(@as(u32, @intCast(sk)), Fake.cluster);
                    try t.expectEqual(cap == 121, Fake.pdl);
                }
            };
        }
    }
    try t.expectEqual(@as(usize, 3 * 6 * 16 * 2 * 2), Fake.launches);
}

test "GLM Q4 wide: invalid rows and launch failures never fall back" {
    Fake.reset();
    const d = Fake.driver();
    var ks = try Fake.load(&d);
    try q4.enableWide(&ks, .{ .d = &d, .handle = null });
    const s: cuda.Stream = .{ .d = &d, .handle = stream_at };
    const w: q4.Q4 = .{ .w = 1, .s = 2, .b = 3, .n = 4096, .npad = 4096, .k = 1024 };
    for ([_]usize{ 0, 17 }) |rows| {
        try t.expectError(error.WindowTooWide, q4.matmulWide(&ks, s, 4, 1024, 5, w, 6, rows));
        try t.expectError(error.WindowTooWide, q4.matmul(&ks, s, 4, 1024, 5, w, 6, rows));
    }
    try t.expectEqual(@as(usize, 0), Fake.launches);
    Fake.launch_error = true;
    try t.expectError(error.NotReady, q4.matmulWide(&ks, s, 4, 1024, 5, w, 6, 1));
    try t.expectEqual(@as(usize, 1), Fake.launches);
    try t.expectEqual(wide, Fake.f);
}


test "GLM Q4 wide: row tile launch keeps old narrow ABI and actual global row stride" {
    Fake.reset();
    const d = Fake.driver();
    var ks = try Fake.load(&d);
    try q4.enableWide(&ks, .{ .d = &d, .handle = null });
    try q4.enableRows(&ks, .{ .d = &d, .handle = null });
    const stream: cuda.Stream = .{ .d = &d, .handle = stream_at };
    for ([_][2]usize{ .{ 64, 512 }, .{ 65, 1024 }, .{ 4096, 1024 }, .{ 4096, 3072 }, .{ 6144, 4096 } }) |shape| {
        const w: q4.Q4 = .{ .w = 1, .s = 2, .b = 3, .n = shape[0], .k = shape[1], .npad = q4.npadOf(shape[0]) };
        for (1..65) |rows| for ([_]bool{ false, true }) |fp32_out| {
            try q4.matmulRows(&ks, stream, 4, w.k + 64, 5, w, 6, rows, fp32_out);
            const expected = if (rows <= 16) (if (fp32_out) wide else narrow) else
                @as(cuda.abi.Function, @ptrFromInt(if (fp32_out) @as(usize, 0x8000) else 0x7000));
            try t.expectEqual(expected, Fake.f);
            try t.expectEqual(@as(c_int, @intCast(rows)), Fake.scalar[0]);
            try t.expectEqual(@as(c_int, @intCast(if (rows == 1) w.k else w.k + 64)), Fake.scalar[2]);
            try t.expectEqual(@as(c_int, 1), Fake.scalar[3]);
            try t.expectEqual(@as(u32, @intCast((w.n + 63) / 64 * q4.splitK(w.n, w.k))), Fake.grid);
            try t.expectEqual(@as(u64, 6), Fake.part.p[0].out);
        };
    }
    const w: q4.Q4 = .{ .w = 1, .s = 2, .b = 3, .n = 64, .k = 512, .npad = 128 };
    const before = Fake.launches;
    for ([_]usize{ 0, 65 }) |rows|
        try t.expectError(error.WindowTooWide, q4.matmulRows(&ks, stream, 4, 512, 5, w, 6, rows, false));
    try t.expectEqual(before, Fake.launches);
}

test "GLM Q4 wide: optional row setup never publishes partial handles" {
    Fake.reset();
    const d = Fake.driver();
    var ks = try Fake.load(&d);
    Fake.missing = true;
    try t.expectError(error.MissingQ4RowsKernel, q4.enableRows(&ks, .{ .d = &d, .handle = null }));
    try t.expect(ks.rows == null);
    Fake.missing = false;
    Fake.attr_error = true;
    try t.expectError(error.NotReady, q4.enableRows(&ks, .{ .d = &d, .handle = null }));
    try t.expect(ks.rows == null);
}
