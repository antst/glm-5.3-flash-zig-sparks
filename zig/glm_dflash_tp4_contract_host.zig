//! TP4 comparison-command inventory, pure layout controls and semantic compilation without a device.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ref = @import("tests/cuda/glm_dflash_tp4_ref.zig");
const fixture = @import("tests/cuda/glm_dflash_tp4_contract.zig");
const t = std.testing;
var run_device = false;

test "DFlash TP4 contract: real entrypoint semantically compiles without execution" {
    if (run_device) try fixture.run(undefined);
}

test "DFlash TP4 contract: actual resolver refuses missing new symbol before work" {
    const Fake = struct {
        var calls: usize = 0;
        var fail_at: usize = 0;
        fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            const names = [_][]const u8{ "df_block_qkv_tp4_v1", "df_attn_part_tp4_v1", "df_attn_merge_tp4_v1" };
            std.debug.assert(std.mem.eql(u8, std.mem.span(name), names[calls]));
            calls += 1;
            if (calls == fail_at) return cuda.abi.error_not_found;
            out.* = @ptrFromInt(0x1000 * calls);
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Fake.get;
    const m: cuda.Module = .{ .d = &d, .handle = null };
    for (1..4) |failure| {
        Fake.calls = 0;
        Fake.fail_at = failure;
        try t.expectError(error.MissingDFlashAttentionKernel, fixture.Fns.load(m, true));
        try t.expectEqual(failure, Fake.calls);
    }
    Fake.calls = 0;
    Fake.fail_at = 0;
    const f = try fixture.Fns.load(m, true);
    try t.expectEqual(@as(usize, 3), Fake.calls);
    try t.expectEqual(@as(usize, 0x1000), @intFromPtr(f.qkv.handle));
    try t.expectEqual(@as(usize, 0x3000), @intFromPtr(f.merge.handle));
}

test "DFlash TP4 contract: local fused rows and every own-head byte view retain padding" {
    const a = t.allocator;
    for ([_]usize{ 8, 32, 64 }) |rows| {
        const full = try a.alloc(u16, rows * 6208);
        defer a.free(full);
        const local = try a.alloc(u16, rows * 1600);
        defer a.free(local);
        for (full, 0..) |*v, j| v.* = @truncate(j);
        for (0..4) |rank| {
            @memset(local, 0xbabe);
            try ref.compact(local, 1600, full, 6208, rows, rank);
            for (0..rows) |r| {
                try t.expectEqualSlices(u16, full[r * 6208 + rank * 1024 ..][0..1024], local[r * 1600 ..][0..1024]);
                try t.expectEqualSlices(u16, full[r * 6208 + 4096 + rank * 256 ..][0..256], local[r * 1600 + 1024 ..][0..256]);
                try t.expectEqualSlices(u16, full[r * 6208 + 5120 + rank * 256 ..][0..256], local[r * 1600 + 1280 ..][0..256]);
                for (local[r * 1600 + 1536 ..][0..64]) |v| try t.expectEqual(@as(u16, 0xbabe), v);
            }
            for ([_][3]usize{ .{ 32, 8, 256 }, .{ 8, 2, 256 }, .{ 32, 8, 64 }, .{ 32, 8, 8192 } }) |shape| {
                const source = try a.alloc(u8, rows * shape[0] * shape[2]);
                defer a.free(source);
                const guard = try a.alloc(u8, rows * shape[1] * shape[2] + 2);
                defer a.free(guard);
                for (source, 0..) |*v, j| v.* = @truncate(j * 7 + j / 257);
                @memset(guard, 0xad);
                const out = guard[1 .. guard.len - 1];
                try ref.select(out, source, rows, shape[0], shape[1], rank, shape[2]);
                for (0..rows) |r| try t.expectEqualSlices(u8, source[(r * shape[0] + rank * shape[1]) * shape[2] ..][0 .. shape[1] * shape[2]], out[r * shape[1] * shape[2] ..][0 .. shape[1] * shape[2]]);
                try t.expectEqual(@as(u8, 0xad), guard[0]);
                try t.expectEqual(@as(u8, 0xad), guard[guard.len - 1]);
            }
        }
    }
}

test "DFlash TP4 contract: old grids refuse and empty/window/cropped counts are exercised" {
    try ref.grid(true, 10, 2, 8);
    try t.expectError(error.Geometry, ref.grid(true, 40, 8, 32));
    try t.expectError(error.Geometry, ref.grid(false, 10, 2, 8));
    for (0..8) |j| {
        const empty = ref.counts(0, 0, j);
        try t.expectEqual(@as(usize, 0), empty.context);
        try t.expectEqual(@as(usize, 8), empty.total);
        try t.expectEqual(@as(usize, 1), empty.piece);
        const wrap = ref.counts(2051, 0, j);
        try t.expectEqual(@as(i64, @intCast(4 + j)), wrap.lo);
        try t.expectEqual(2047 - j, wrap.context);
        const cropped = ref.counts(4100, 4097, j);
        try t.expectEqual(@as(usize, 3), cropped.context);
    }
}

test "DFlash TP4 contract: independent FP64 softmax diagnostic uses rank-offset full context heads" {
    const a = t.allocator;
    const ring = try a.alloc(u16, ref.ring_cells);
    defer a.free(ring);
    @memset(ring, 0);
    const q: [8 * 1024]u16 = @splat(0);
    const k: [8 * 256]u16 = @splat(0);
    var v: [8 * 256]u16 = undefined;
    var out: [8 * 1024]u16 = undefined;
    const bf = core.contract.spec.toBf16;
    for (0..2) |layer| for (0..2048) |slot| for (0..8) |head| {
        const value = bf(1 + @as(f32, @floatFromInt(head)) / 8);
        @memset(ring[(layer * 2 * 2048 + 2048 + slot) * 1024 + head * 128 ..][0..128], value);
    };
    for ([_]usize{ 0, 3 }) |rank| {
        for (0..8) |r| {
            for (0..2) |h| @memset(v[r * 256 + h * 128 ..][0..128], bf(1 + @as(f32, @floatFromInt(rank * 2 + h)) / 8));
            for (0..8) |h| @memset(out[r * 1024 + h * 128 ..][0..128], bf(1 + @as(f32, @floatFromInt(rank * 2 + h / 4)) / 8));
        }
        const stats = try ref.attention(&q, &k, &v, ring, &out, rank, ref.cases[1], 0.08838835);
        try t.expectEqual(@as(usize, 8192), stats.outputs);
        try t.expectEqual(@as(f64, 0), stats.max_abs);
        try t.expectEqual(@as(f64, 0), stats.max_rel);
    }
    out[0] = 0x7fc1;
    try t.expectError(error.NonfiniteDiagnostic, ref.attention(&q, &k, &v, ring, &out, 3, ref.cases[1], 0.08838835));
}
