//! Record the real optional prompt encoder without allocating CUDA objects.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");
const prompt = @import("cuda_affine_prompt.zig");

const Fake = struct {
    var calls: usize = 0;
    var fail: bool = false;
    var rows: usize = 0;
    var wide: bool = false;
    fn read(comptime T: type, ptr: ?*anyopaque) T {
        return @as(*align(1) const T, @ptrCast(ptr.?)).*;
    }
    fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(@intFromPtr(f) == (if (wide) @as(usize, 22) else 11));
        std.debug.assert(gx == ((rows + 127) / 128) * 16 and gy == 1 and gz == 1);
        std.debug.assert(bx == 256 and by == 1 and bz == 1 and smem == prompt.shared_bytes);
        const a = args.?;
        for ([_]u64{ 0x400000, 0x100000, 0x200000, 0x300000, 0x800000 }, 0..) |v, i|
            std.debug.assert(read(u64, a[i]) == v);
        for ([_]usize{ rows, 2048, 4096, 2048, 4160, 8 }, 5..) |v, i|
            std.debug.assert(read(c_int, a[i]) == v);
        calls += 1;
        return if (fail) cuda.abi.error_not_ready else 0;
    }
};

test "affine target: optional prompt actual ABI, row boundaries, output kinds and early refusal" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const ks: prompt.Kernels = .{ .functions = .{
        .{ .d = &d, .handle = @ptrFromInt(11) }, .{ .d = &d, .handle = @ptrFromInt(22) },
    } };
    var mk: mm.Kernels = undefined;
    mk.affine = .{ .q4 = undefined, .sums = undefined, .prompt = ks };
    const w: mm.Tiled = .{ .w = 0x100000, .s = 0x200000, .bias = 0x300000, .bits = 4, .n = 2048, .k = 4096, .npad = 2048 };
    Fake.calls = 0;
    Fake.fail = false;
    for ([_]usize{ 1, 16, 17, 63, 64, 65, 127, 128, 129, 2048 }) |rows| for ([_]bool{ false, true }) |wide| {
        Fake.rows = rows;
        Fake.wide = wide;
        try mm.matmul(&mk, s, 0x400000, 4160, w, 0x800000, 0, rows, wide, true);
    };
    try std.testing.expectEqual(@as(usize, 20), Fake.calls);
    const q: @import("cuda_q4.zig").Q4 = .{ .w = w.w, .s = w.s, .b = w.bias, .n = w.n, .k = w.k, .npad = w.npad };
    for ([_]usize{ 0, 2049 }) |rows|
        try std.testing.expectError(error.BadAffinePromptShape, ks.matmul(s, 0x400000, 4160, q, 0x800000, rows, false));
    try std.testing.expectError(error.BadAffinePromptShape, ks.matmul(s, 0x400001, 4160, q, 0x800000, 1, false));
    try std.testing.expectEqual(@as(usize, 20), Fake.calls);
    Fake.fail = true;
    Fake.rows = 1;
    Fake.wide = true;
    try std.testing.expectError(error.NotReady, mm.matmul(&mk, s, 0x400000, 4160, w, 0x800000, 0, 1, true, true));
    try std.testing.expectEqual(@as(usize, 21), Fake.calls);
    Fake.fail = false;
}
