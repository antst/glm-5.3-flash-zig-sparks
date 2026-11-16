//! Record the actual optional selector's dispatch and failures without a CUDA library.
const std = @import("std");
const cuda = @import("cuda");
const select = @import("cuda_decode_select.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

const Fake = struct {
    var count: usize = 0;
    var fail: usize = 0;
    var bad: bool = false;
    var rows: u32 = 0;
    var np: i64 = 0;
    var ctas: u32 = 0;

    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }

    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        count += 1;
        const scan = @intFromPtr(f) == 1;
        bad = bad or @intFromPtr(stream) != 0x80 or gz != 1 or by != 1 or bz != 1 or shared != 0;
        bad = bad or gx != (if (scan) ctas else rows) or gy != (if (scan) rows else @as(u32, 1));
        bad = bad or bx != (if (scan) @as(u32, 256) else @as(u32, 1024));
        bad = bad or arg(i64, args, if (scan) 1 else 2) != np;
        bad = bad or arg(u64, args, 0) != 0x2000 or arg(u64, args, if (scan) 2 else 3) != 0x4000;
        bad = bad or arg(c_int, args, if (scan) 3 else 4) != 512;
        bad = bad or arg(u64, args, 5) != 0x9000 or arg(i64, args, 6) != @divTrunc(np + 31, 32);
        if (!scan) bad = bad or arg(u64, args, 1) != 0x3000 or arg(u64, args, 7) != 0;
        return if (count == fail) 600 else 0;
    }
};

test "decode select: actual scan merge arguments, guarded fallback and no tail after failure" {
    var d: cuda.Driver = undefined;
    d.func_name = null;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x80) };
    var owner: select.Owner = .{ .module = undefined, .scan = .{ .d = &d, .handle = @ptrFromInt(1) }, .merge = .{ .d = &d, .handle = @ptrFromInt(2) }, .bits = .{ .d = &d, .ptr = 0x9000, .len = try select.bitsBytes(48, 524288) }, .stream = stream, .np = 524288, .rows = 48, .rank = 0 };
    for ([_]usize{ 1, 4, 8, 15, 16, 31, 32, 48 }) |rows| for ([_]usize{ 8192, 65536, 524288 }) |np| {
        Fake.count = 0;
        Fake.bad = false;
        Fake.fail = 0;
        Fake.rows = @intCast(rows);
        Fake.np = @intCast(np);
        Fake.ctas = @intCast(select.scanCtas(1048576, rows, np));
        try t.expect(try owner.select(stream, 0x2000, 0x3000, 0x4000, 1048576, rows, np, false));
        try t.expectEqual(@as(usize, 2), Fake.count);
        try t.expect(!Fake.bad);
    };
    try t.expectEqual(@as(u64, 24), owner.calls);
    Fake.count = 0;
    try t.expect(!try owner.select(stream, 0, 0, 0, 0, 16, 65536, true));
    try t.expect(!try owner.select(stream, 0, 0, 0, 0, 16, 4096, false));
    try t.expect(!try owner.select(stream, 0, 0, 0, 1048576, 49, 65536, false));
    try t.expectEqual(@as(usize, 0), Fake.count);
    try t.expectError(error.DselCapacity, owner.select(stream, 0x2000, 0x3000, 0x4000, 1048576, 16, 1048576, false));
    try t.expectError(error.DselAddress, owner.select(stream, 0x2001, 0x3000, 0x4000, 1048576, 16, 65536, false));
    try t.expectEqual(@as(usize, 0), Fake.count);
    for ([_]usize{ 1, 2 }) |stage| {
        Fake.count = 0;
        Fake.fail = stage;
        Fake.rows = 16;
        Fake.np = 65536;
        Fake.ctas = @intCast(select.scanCtas(1048576, 16, 65536));
        try t.expectError(error.NotReady, owner.select(stream, 0x2000, 0x3000, 0x4000, 1048576, 16, 65536, false));
        try t.expectEqual(stage, Fake.count);
        try t.expectEqual(@as(u64, 24), owner.calls);
    }
}

test "decode select: strict default-off mode and all-rank disagreement before collective work" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DSEL")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.dsel);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(on.dsel);
    const F = ranks.Fingerprint;
    var all = [_]F{ F.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0)), F.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0)) };
    try t.expectEqual(F.Field.dsel, ranks.mismatch(&all).?.field);
    try t.expectEqualStrings("TF_GLM_DSEL", @import("cuda_rank_names.zig").explain(F.Field.dsel));
    all[0] = F.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expect(ranks.mismatch(&all) == null);
    for ([_][]const u8{ "yes", "", "2", "false" }) |value| {
        Env.value = value;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
}
