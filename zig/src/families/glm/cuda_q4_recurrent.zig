//! Architectural precision boundary: retain KDA decay/output-gate linears in their original BF16 path.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const tile = @import("cuda_tile.zig");
const mm = @import("cuda_mm.zig");

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !u8 {
    const raw = get("TF_GLM_Q4_RECURRENT_BF16") orelse return 0;
    if (std.mem.eql(u8, raw, "0")) return 0;
    if (std.mem.eql(u8, raw, "1")) return 1;
    if (std.mem.eql(u8, raw, "2")) return 2;
    if (std.mem.eql(u8, raw, "3")) return 3;
    problem.* = "TF_GLM_Q4_RECURRENT_BF16: 0, 1, 2 or 3";
    return error.BadSetting;
}

pub fn require(settings: anytype, world: usize) !void {
    if (settings.q4_recurrent_bf16 == 0) return;
    if (settings.q4_recurrent_bf16 > 3 or world != 2 or !settings.q4_affine_mse or !settings.q4_activation or settings.dense_bits != 4 or !settings.tiled)
        return error.Q4RecurrentBf16Scope;
}

/// Field identity selects the architectural role, never matrix shape, layer ID or tensor values.
pub fn isControl(w: *const weights.Weights, field: *const u64) bool {
    for (w.layers) |*layer| switch (layer.mixer) {
        .kda => |*k| if (field == &k.fb or field == &k.gb) return true,
        .mla => {},
    };
    return false;
}

/// The complete recurrent input, selected by field address. Other projections are never retained here.
pub fn isInput(w: *const weights.Weights, field: *const u64) bool {
    for (w.layers) |*layer| switch (layer.mixer) {
        .kda => |*k| if (k.proj == .bf16 and field == &k.proj.bf16) return true,
        .mla => {},
    };
    return false;
}

pub fn isRetained(w: *const weights.Weights, field: *const u64, mode: u8) bool {
    return mode != 0 and (isControl(w, field) or (mode == 3 and isInput(w, field)));
}

/// Exactly the existing BF16-head tiling sequence, using the same reusable loader scratch.
pub fn retain(m: *const mm.Kernels, stream: cuda.Stream, d: *const cuda.Driver, l: tile.Lin, tmp: u64) !void {
    try mm.tile(m, stream, l.field.*, tmp, l.n, l.k);
    try d.check(d.api.cuMemcpyDtoDAsync_v2(l.field.*, tmp, l.n * l.k * 2, stream.handle), "cuMemcpyDtoDAsync");
    try stream.synchronize();
}

test "Q4 recurrent BF16: actual loader roles use field identity, and omitted packed pointers dispatch BF16" {
    const t = std.testing;
    var layers: [3]weights.Layer = undefined;
    for (&layers) |*layer| {
        layer.mixer = .{ .kda = .{ .proj = .{ .bf16 = 17 }, .fb = 17, .gb = 17, .o = .{ .bf16 = 17 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = 32, .dims = 4096 } };
    }
    var w: weights.Weights = undefined;
    w.layers = &layers;
    const stranger: u64 = 17;
    try t.expect(!isControl(&w, &stranger));
    for (&layers) |*layer| {
        const k = &layer.mixer.kda;
        try t.expect(isControl(&w, &k.fb) and isControl(&w, &k.gb));
        try t.expect(!isControl(&w, &k.proj.bf16) and !isControl(&w, &k.o.bf16));
    }
    const packed_set: mm.Packed = .{ .ptrs = &.{ 64, 128 }, .bits = &.{ 4, 4 }, .affine = true };
    const retained = packed_set.tiled(17, 4096, 128);
    try t.expectEqual(@as(u8, 16), retained.bits);
    try t.expectEqual(@as(u64, 0), retained.bias);
    try t.expectEqual(@as(u8, 4), packed_set.tiled(64, 4096, 4096).bits);
}

test "Q4 recurrent BF16: actual retained tile/copy/drain ABI preserves full BF16 bytes and order" {
    const Fake = struct {
        var step: usize = 0;
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 0), step) catch @panic("tile order");
            std.testing.expectEqual(@as(c_uint, 256), bx) catch @panic("tile block");
            std.testing.expectEqual(@as(c_uint, 128), gx) catch @panic("tile grid");
            const p = params.?;
            std.testing.expectEqual(@as(u64, 256), @as(*align(1) const u64, @ptrCast(p[0].?)).*) catch @panic("tile source");
            std.testing.expectEqual(@as(u64, 2048), @as(*align(1) const u64, @ptrCast(p[1].?)).*) catch @panic("tile scratch");
            step += 1;
            return 0;
        }
        fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 1), step) catch @panic("copy order");
            std.testing.expectEqual(@as(u64, 256), dst) catch @panic("copy destination");
            std.testing.expectEqual(@as(u64, 2048), src) catch @panic("copy source");
            std.testing.expectEqual(@as(usize, 1048576), bytes) catch @panic("BF16 payload");
            step += 1;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 2), step) catch @panic("drain order");
            step += 1;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Fake.launch;
    driver.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    driver.api.cuStreamSynchronize = Fake.sync;
    var kernels: mm.Kernels = undefined;
    kernels.tile = .{ .d = &driver, .handle = @ptrFromInt(16) };
    var field: u64 = 256;
    Fake.step = 0;
    try retain(&kernels, .{ .d = &driver, .handle = @ptrFromInt(32) }, &driver, .{ .field = &field, .n = 4096, .k = 128 }, 2048);
    try std.testing.expectEqual(@as(usize, 3), Fake.step);
}

test "Q4 recurrent BF16: whole input mode preserves old modes and actual linear field classification" {
    const t = std.testing;
    const config: @import("config.zig").Config = .{};
    const share = try @import("cuda_shard.zig").Share.of(&config, 0, 2);
    var layers: [3]weights.Layer = undefined;
    for (&layers) |*l| {
        l.mixer = .{ .kda = .{ .proj = .{ .bf16 = 17 }, .fb = 17, .gb = 17, .o = .{ .bf16 = 17 }, .conv_w = 17, .a_log = 17, .dt_bias = 17, .norm_w = 17, .eps = 1e-6, .lower = -5, .heads = 32, .dims = 4096 } };
        l.ffn = .{ .dense = .{ .w = .{ .bf16 = .{ .gu = 17, .down = 17 } }, .width = 64, .dims = 4096, .limit = 0 } };
    }
    var w: weights.Weights = .{ .gpa = t.allocator, .config = config, .share = share, .layers = &layers };
    var linears: std.ArrayList(tile.Lin) = .empty;
    defer linears.deinit(t.allocator);
    try tile.linears(&w, &linears);
    try t.expectEqual(@as(usize, 18), linears.items.len);
    const stranger: u64 = 17;
    for (0..4) |mode| {
        var count: usize = 0;
        for (linears.items) |l| {
            const on = isRetained(&w, l.field, @intCast(mode));
            if (mode < 3) try t.expectEqual(mode != 0 and isControl(&w, l.field), on);
            count += @intFromBool(on);
        }
        try t.expectEqual(([_]usize{ 0, 6, 6, 9 })[mode], count);
        try t.expect(!isRetained(&w, &stranger, @intCast(mode)));
    }
    for (&layers) |*l| {
        const k = &l.mixer.kda;
        try t.expect(isInput(&w, &k.proj.bf16));
        for ([_]*const u64{ &k.fb, &k.gb, &k.o.bf16, &k.conv_w, &k.a_log, &k.dt_bias, &k.norm_w }) |p|
            try t.expect(!isInput(&w, p));
    }
    const E = struct {
        var value: ?[]const u8 = "3";
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q4_RECURRENT_BF16")) value else null;
        }
    };
    defer E.value = null;
    var problem: []const u8 = "";
    const ranks = @import("cuda_ranks.zig");
    var on = try ranks.Settings.read(E.get, &problem);
    try t.expectEqual(@as(u8, 3), on.q4_recurrent_bf16);
    E.value = "4";
    try t.expectError(error.BadSetting, ranks.Settings.read(E.get, &problem));
    on.q4_affine_mse = true;
    on.q4_activation = true;
    on.dense_bits = 4;
    try require(on, 2);
    try t.expectError(error.Q4RecurrentBf16Scope, require(on, 4));
    var old = on;
    old.q4_recurrent_bf16 = 2;
    const a = ranks.Fingerprint.of(2, 0, old, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 86), b.layout);
    try t.expectEqual(ranks.Fingerprint.Field.q4_recurrent_bf16, ranks.mismatch(&.{ a, b }).?.field);
    const sizes = @import("cuda_q4.zig").bytes(12576, 4096);
    try t.expectEqual(@as(usize, 2510094336), 34 * (12576 * 4096 * 2 - sizes[0] - sizes[1] - sizes[2]));
}

test "Q4 recurrent BF16: actual full recurrent input retain tiles copies and drains 103022592 bytes" {
    const F = struct {
        var step: usize = 0;
        fn launch(_: cuda.abi.Function, gx: u32, _: u32, _: u32, bx: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 0), step) catch @panic("full input tile order");
            std.testing.expectEqual(@as(u32, 12576), gx) catch @panic("full input tile grid");
            std.testing.expectEqual(@as(u32, 256), bx) catch @panic("full input tile block");
            const p = params.?;
            std.testing.expectEqual(@as(i32, 12576), @as(*align(1) const i32, @ptrCast(p[2].?)).*) catch @panic("full input N");
            std.testing.expectEqual(@as(i32, 4096), @as(*align(1) const i32, @ptrCast(p[3].?)).*) catch @panic("full input K");
            step += 1;
            return 0;
        }
        fn copy(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 1), step) catch @panic("full input copy order");
            std.testing.expectEqual(@as(u64, 256), dst) catch @panic("full input dst");
            std.testing.expectEqual(@as(u64, 2048), src) catch @panic("full input src");
            std.testing.expectEqual(@as(usize, 103022592), n) catch @panic("full input payload");
            step += 1;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            std.testing.expectEqual(@as(usize, 2), step) catch @panic("full input drain order");
            step += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = F.launch;
    d.api.cuMemcpyDtoDAsync_v2 = F.copy;
    d.api.cuStreamSynchronize = F.sync;
    var kernels: mm.Kernels = undefined;
    kernels.tile = .{ .d = &d, .handle = @ptrFromInt(16) };
    var ptr: u64 = 256;
    F.step = 0;
    try retain(&kernels, .{ .d = &d, .handle = @ptrFromInt(32) }, &d, .{ .field = &ptr, .n = 12576, .k = 4096 }, 2048);
    try std.testing.expectEqual(@as(usize, 3), F.step);
}
