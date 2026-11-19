//! Architectural precision boundary: retain KDA decay/output-gate linears in their original BF16 path.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const tile = @import("cuda_tile.zig");
const mm = @import("cuda_mm.zig");

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !bool {
    return @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_RECURRENT_BF16"), problem, "TF_GLM_Q4_RECURRENT_BF16: 0 or 1");
}

pub fn require(settings: anytype, world: usize) !void {
    if (!settings.q4_recurrent_bf16) return;
    if (world != 2 or !settings.q4_affine_mse or !settings.q4_activation or settings.dense_bits != 4 or !settings.tiled)
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
