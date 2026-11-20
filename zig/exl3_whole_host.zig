//! Whole-prompt dispatch through the actual GLM adapter, with no CUDA context or image execution.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const ex = cuda.exl3_experts;
const t = std.testing;
const Fake = struct {
    const Call = struct { id: usize, grid: [3]u32, args: [20]u64 = @splat(0) };
    var calls: [128]Call = undefined;
    var n: usize = 0;
    var copied: usize = 0;
    var fail: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        calls[n] = .{ .id = id, .grid = .{ gx, gy, gz } };
        const sizes: []const u8 = switch (id) {
            31 => &.{ 8, 4, 8, 8, 4 },
            32 => &.{ 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4 },
            33 => &.{ 8, 8, 8, 8, 8, 8, 8, 4, 4, 4 },
            20 => &.{ 8, 4, 4, 4, 8, 8, 8 },
            21 => &.{ 8, 4, 4, 8, 8 },
            22 => &.{ 4, 4, 4, 8, 8, 8 },
            23 => &.{ 8, 4, 4, 8, 8, 8 },
            else => &.{},
        };
        for (sizes, 0..) |size, i| {
            const bytes: [*]const u8 = @ptrCast(args.?[i].?);
            calls[n].args[i] = if (size == 8) std.mem.readInt(u64, bytes[0..8], .little) else std.mem.readInt(u32, bytes[0..4], .little);
        }
        n += 1;
        return if (id == fail) 600 else cuda.abi.success;
    }
    fn copy(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copied += 1;
        return cuda.abi.success;
    }
    fn errorName(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
        out.* = "CUDA_ERROR_NOT_READY";
        return cuda.abi.success;
    }
    fn reset() void {
        n = 0;
        copied = 0;
        fail = 0;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
const Rig = struct {
    d: cuda.Driver = undefined,
    k: ex.Experts = undefined,
    ek: glm.experts.Kernels = undefined,
    w: glm.exl3.Experts = .{ .gt = 100, .ut = 200, .dt = 300, .k2 = 400, .suh_g = 500, .suh_u = 600, .svh_g = 700, .svh_u = 800, .suh_d = 900, .svh_d = 1000, .shx = true },
    count: u64 = 0,
    fn init(r: *Rig, whole: bool) void {
        Fake.reset();
        r.d.api.cuLaunchKernel = Fake.launch;
        r.d.api.cuMemcpy2DAsync_v2 = Fake.copy;
        r.d.api.cuGetErrorName = Fake.errorName;
        r.k = .{ .mod = undefined, .group = function(&r.d, 1), .rot_in = function(&r.d, 2), .grouped = @splat(function(&r.d, 3)), .gateup = function(&r.d, 4), .down_combine = function(&r.d, 5) };
        r.k.prompt = .{ .rotate = function(&r.d, 11), .gateup = function(&r.d, 12), .down = function(&r.d, 13) };
        if (whole) {
            r.k.whole = .{ .module = undefined, .kernels = .{ .rotate = function(&r.d, 31), .gateup = function(&r.d, 32), .down = function(&r.d, 33) } };
            r.k.whole_calls = &r.count;
        }
        r.ek = .{ .nvfp4 = undefined, .b32 = null, .router = .{ .plan_small = function(&r.d, 20), .plan_rank = function(&r.d, 21), .plan_offsets = function(&r.d, 22), .plan_scatter = function(&r.d, 23), .plan_rows = function(&r.d, 24), .plan_rank_rows = function(&r.d, 25), .plan_scatter_rows = function(&r.d, 26) } };
    }
    fn run(r: *Rig, rows: usize, dims: usize, width: usize, prompt: bool) !void {
        const sc: glm.exl3.Scratch = .{ .picks = 0x100000000, .xg = 0, .xu = 0, .z = 0 };
        const p: cuda.grouped.Plan = .{ .items = 10000, .members = 20000, .counts = 30000, .rank = 40000, .hist = 50000 };
        try glm.exl3.run(&r.k, &r.ek, .{ .d = &r.d, .handle = null }, r.w, sc, p, 0x200000000, dims + 8, 0x300000000, 9, rows, 256, dims, width, 10, prompt, 0x400000000, 0);
    }
};
test "whole prompt: real adapter uses direct nine-slot planes and shared-id skip across 4096 boundary" {
    for ([_]bool{ false, true }) |shared| for ([_]usize{ 512, 4096 }) |dims|
        for ([_]usize{ 512, 1024 }) |width| for ([_]usize{ 1, 63, 64, 65, 1023, 1024, 1025, 2048, 4095, 4096, 4097 }) |rows| {
            var r: Rig = .{};
            r.init(true);
            r.w.shx = shared;
            try r.run(rows, dims, width, true);
            try t.expectEqual(@as(usize, 0), Fake.copied);
            const calls = (rows + 4095) / 4096;
            try t.expectEqual(@as(u64, calls), r.count);
            var gu: usize = 0;
            var down: usize = 0;
            for (Fake.calls[0..Fake.n]) |c| {
                if (c.id == 32) {
                    const at = gu * 4096;
                    const n: usize = @min(@as(usize, 4096), rows - at);
                    try t.expectEqual(@as(u64, 256), c.args[16]);
                    try t.expectEqual(@as(u64, if (shared) 9 else 1), c.args[13]);
                    try t.expectEqual(@as(u32, @intCast(width / 128)), c.grid[0]);
                    try t.expectEqual(@as(u32, @intCast(cuda.grouped.maxItems(n * 9, 257, 64))), c.grid[1]);
                    gu += 1;
                }
                if (c.id == 33) {
                    try t.expectEqual(@as(u64, 0x400000000 + down * 4096 * 9 * dims * 4), c.args[6]);
                    try t.expectEqual(@as(u64, 256), c.args[9]);
                    down += 1;
                }
                try t.expect(c.id != 11 and c.id != 12 and c.id != 13);
            }
            try t.expectEqual(calls, gu);
            try t.expectEqual(calls, down);
        };
}
test "whole prompt: OFF keeps old allocation bytes; ON scratch is local and bounded for every buffer size" {
    for ([_]usize{ 1, 16, 64, 1024, 2048, 4096, 4097 }) |rows| for ([_]bool{ false, true }) |shared|
        for ([_]bool{ false, true }) |prompt| {
            const old = glm.exl3.scratchBytes(rows, 4096, 512, prompt, shared);
            try t.expectEqual(old, glm.exl3.scratchBytesMode(rows, 4096, 512, prompt, shared, false));
            const now = glm.exl3.scratchBytesMode(rows, 4096, 512, prompt, shared, true);
            if (!prompt) try t.expectEqual(old, now);
            try t.expect(now[0] > 0);
        };
    var r: Rig = .{};
    r.init(false);
    try r.run(1025, 4096, 512, true);
    try t.expectEqual(@as(usize, 4), Fake.copied);
    r.init(true);
    try r.run(4, 4096, 512, false);
    for (Fake.calls[0..Fake.n]) |c| try t.expect(c.id != 31 and c.id != 32 and c.id != 33);
}
test "whole prompt: launch failure does not publish a call or enqueue down" {
    var r: Rig = .{};
    r.init(true);
    Fake.fail = 32;
    try t.expectError(error.NotReady, r.run(2048, 4096, 512, true));
    try t.expectEqual(@as(u64, 0), r.count);
    for (Fake.calls[0..Fake.n]) |c| try t.expect(c.id != 33);
}
test "whole prompt: flag is strict, defaults OFF and is agreed by all ranks" {
    const Reader = struct {
        var value: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_EXL3_PROMPT_WHOLE")) value else null;
        }
    };
    var problem: []const u8 = "";
    var settings = try glm.ranks.Settings.read(Reader.get, &problem);
    try t.expect(!settings.exl3_prompt_whole);
    Reader.value = "1";
    settings = try glm.ranks.Settings.read(Reader.get, &problem);
    try t.expect(settings.exl3_prompt_whole);
    const F = glm.ranks.Fingerprint;
    const digest: [32]u8 = @splat(0);
    var all = [_]F{ F.of(2, 0, settings, 4096, true, digest, digest, digest), F.of(2, 1, settings, 4096, true, digest, digest, digest) };
    try t.expectEqual(@as(u32, 85), all[0].layout);
    try t.expect(glm.ranks.mismatch(&all) == null);
    all[1].exl3_prompt_whole = 0;
    try t.expectEqual(F.Field.exl3_prompt_whole, glm.ranks.mismatch(&all).?.field);
    Reader.value = "true";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Reader.get, &problem));
}
