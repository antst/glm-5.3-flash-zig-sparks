//! Optional prompt-only BF16 tensor-core latent projections with immutable module ownership.
const std = @import("std");
const cuda = @import("cuda");
pub const image_sha = "34e82c3d34080e41408fa4e2de24d229ffcbce55d3ad24bae953ffeeb9608650";
pub const shared = [2]u32{ 3 * (128 * 40 + 32 * 136) * 2, 3 * (128 * 40 + 128 * 40) * 2 };
pub const Kind = enum { absorb, expand };

pub fn eligible(prompt: bool, rows: usize, ptrs: [3]u64) bool {
    if (!prompt or rows < 64) return false;
    for (ptrs) |p| if (p == 0 or p % 16 != 0) return false;
    return true;
}

pub const Kernels = struct {
    absorb: cuda.Function,
    expand: cuda.Function,
    enqueued: ?*u64 = null,

    pub fn load(module: cuda.Module) !Kernels {
        const k: Kernels = .{
            .absorb = try module.function("tf_glm_latent_mma_kn"),
            .expand = try module.function("tf_glm_latent_mma_nk"),
        };
        try k.absorb.allowDynamicShared(shared[0]);
        try k.expand.allowDynamicShared(shared[1]);
        return k;
    }

    pub fn launch(k: *const Kernels, kind: Kind, stream: cuda.Stream, x: u64, xr: usize, w: u64,
        out: u64, stride: usize, rows: usize, heads: usize) !void
    {
        const n: usize = if (kind == .absorb) 512 else 256;
        const cols: usize = if (kind == .absorb) 256 else 512;
        if (rows == 0 or heads == 0 or xr < heads * cols or stride < heads * n or
            xr % 8 != 0 or stride % 8 != 0) return error.LatentMmaShape;
        for ([_]u64{ x, w, out }) |p| if (p == 0 or p % 16 != 0) return error.LatentMmaAlignment;
        var a: cuda.Args = .{};
        inline for (.{ x, @as(i64, @intCast(xr)), w, out, @as(i64, @intCast(stride)),
            @as(i64, @intCast(rows)), @as(c_int, @intCast(n)), @as(c_int, @intCast(cols)) }) |v| a.add(v);
        const index: usize = if (kind == .absorb) 0 else 1;
        try cuda.launch.launch(if (kind == .absorb) k.absorb else k.expand,
            .{ .grid = .{ .x = @intCast(n / 128), .y = @intCast((rows + 127) / 128),
                .z = @intCast(heads) }, .block = .{ .x = 256 }, .shared = shared[index] }, stream, &a);
        if (k.enqueued) |count| count.* += 1;
    }
};

pub const Owner = struct {
    gpa: std.mem.Allocator,
    module: cuda.Module,
    kernels: Kernels,
    enqueued: *u64,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !Owner {
        const path = @import("sys").environment.Process.get(.{}, "TF_GLM_LATENT_MMA_IMAGE") orelse
            return error.LatentMmaImageRequired;
        const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(32 << 20), .@"16", null);
        defer gpa.free(bytes);
        try checkImage(bytes);
        var module = try cuda.Module.load(d, bytes);
        errdefer module.unload();
        const count = try gpa.create(u64);
        errdefer gpa.destroy(count);
        count.* = 0;
        var kernels = try Kernels.load(module);
        kernels.enqueued = count;
        return .{ .gpa = gpa, .module = module, .kernels = kernels, .enqueued = count };
    }

    pub fn deinit(owner: *Owner) void {
        std.log.info("GLM_ABSORB_MMA_CLOSE host_enqueues={d} image={s}", .{ owner.enqueued.*, image_sha });
        owner.module.unload();
        owner.gpa.destroy(owner.enqueued);
        owner.* = undefined;
    }
};

pub fn checkImage(bytes: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.LatentMmaImageHash;
}

pub fn install(m: anytype, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver,
    ctx: *const cuda.Context, on: bool) !void
{
    if (!on) return;
    if (m.latent_mma != null or try ctx.capability() != 121) return error.LatentMmaScope;
    m.latent_mma = try Owner.init(gpa, io, d);
    std.log.info("GLM_ABSORB_MMA active=true prompt_rows_min=64 accum=fp32 image={s}", .{image_sha});
}

pub fn compatibility(base: [32]u8, on: bool) [32]u8 {
    if (!on) return base;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&base);
    hash.update("glm-target-prefill-latent-mma-v1");
    hash.update(image_sha);
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return digest;
}

test "latent MMA: prompt routing boundary and optional owner OFF preserve the old path" {
    const t = std.testing;
    for ([_]usize{ 1, 32, 63, 64, 65, 128, 2048 }) |rows| {
        try t.expectEqual(rows >= 64, eligible(true, rows, .{ 16, 32, 48 }));
        try t.expect(!eligible(false, rows, .{ 16, 32, 48 }));
        try t.expect(!eligible(true, rows, .{ 16, 33, 48 }));
    }
    var inert: @import("cuda_forward.zig").Modules = undefined;
    try install(&inert, undefined, undefined, undefined, undefined, false);
    try t.expectError(error.LatentMmaImageHash, checkImage("not the pinned module"));
    const base: [32]u8 = @splat(17);
    try t.expectEqualSlices(u8, &base, &compatibility(base, false));
    try t.expect(!std.mem.eql(u8, &base, &compatibility(base, true)));
}

test "latent MMA: cached strict setting and rank agreement precede model load" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_ABSORB_MMA")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "true", "01", " 1", "+1" }) |bad| {
        Env.value = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(on.absorb_mma);
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 85), a.layout);
    try std.testing.expectEqual(FP.Field.absorb_mma, ranks.mismatch(&.{ a, b }).?.field);
    try std.testing.expectEqualStrings("TF_GLM_ABSORB_MMA", @import("cuda_rank_names.zig").explain(FP.Field.absorb_mma));
}

test "latent MMA: actual Forward set and latent calls retain SIMT decode and route prompt tails to the new module" {
    const Fake = struct {
        var calls: usize = 0;
        var want_mma: bool = false;
        var rows: usize = 0;
        var heads: usize = 0;
        var fail: bool = false;
        fn u(p: ?[*]?*anyopaque, i: usize) u64 {
            return @as(*align(1) const u64, @ptrCast(p.?[i].?)).*;
        }
        fn small(p: ?[*]?*anyopaque, i: usize) i32 {
            return @as(*align(1) const i32, @ptrCast(p.?[i].?)).*;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32,
            smem: u32, stream: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result
        {
            const id = @intFromPtr(f);
            std.debug.assert(by == 1 and bz == 1 and stream == null and u(p, 0) == 16);
            if (want_mma) {
                const kn = id == 1;
                const n: usize = if (kn) 512 else 256;
                const k: usize = if (kn) 256 else 512;
                std.debug.assert((id == 1 or id == 2) and bx == 256 and smem == shared[@intFromBool(!kn)]);
                std.debug.assert(x == n / 128 and y == (rows + 127) / 128 and z == heads);
                std.debug.assert(u(p, 1) == heads * k and u(p, 2) == 32 and u(p, 3) == 48);
                std.debug.assert(u(p, 4) == heads * n and u(p, 5) == rows);
                std.debug.assert(small(p, 6) == n and small(p, 7) == k);
            } else {
                const rb = @min(rows, 64);
                std.debug.assert((id == 11 or id == 12) and bx == 128 and smem == 0);
                std.debug.assert(x == heads and y == 16 and z == (rows + rb - 1) / rb);
                std.debug.assert(u(p, 1) == 32 and u(p, 2) == 48 and u(p, 3) == rows);
                std.debug.assert(u(p, 4) == heads and u(p, 5) == rb);
            }
            calls += 1;
            return if (fail) cuda.abi.error_not_ready else 0;
        }
    };
    const fw = @import("cuda_forward.zig");
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var m: fw.Modules = undefined;
    m.index = null;
    m.tiled = false;
    m.packs = null;
    m.copy_rows = undefined;
    m.glue.absorb_q = .{ .d = &d, .handle = @ptrFromInt(11) };
    m.glue.expand_v = .{ .d = &d, .handle = @ptrFromInt(12) };
    var b: fw.Buffers = undefined;
    var rank = [_]fw.Rank{.{ .w = undefined, .st = undefined, .b = &b }};
    var f: fw.Forward = undefined;
    f.m = &m;
    f.ranks = &rank;
    f.decode_select = null;
    Fake.calls = 0;
    Fake.fail = false;
    var count: u64 = 0;
    for ([_]bool{ false, true }) |on| {
        m.latent_mma = if (on) .{ .module = undefined, .gpa = undefined, .enqueued = &count, .kernels = .{
            .absorb = .{ .d = &d, .handle = @ptrFromInt(1) },
            .expand = .{ .d = &d, .handle = @ptrFromInt(2) },
            .enqueued = &count,
        } } else null;
        for ([_]bool{ false, true }) |prompt| for ([_]usize{ 1, 16, 32 }) |heads| {
            for ([_]usize{ 1, 32, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
                b.prefill = prompt;
                Fake.want_mma = on and prompt and rows >= 64;
                Fake.heads = heads;
                Fake.rows = rows;
                try @import("cuda_latent.zig").absorbQ(f.set(), s, 16, 32, 48, rows, heads, 256, 512);
                try @import("cuda_latent.zig").expandV(f.set(), s, 16, 32, 48, rows, heads, 256, 512);
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 216), Fake.calls);
    try std.testing.expectEqual(@as(u64, 36), count);
    const before = count;
    Fake.fail = true;
    try std.testing.expectError(error.NotReady,
        @import("cuda_latent.zig").absorbQ(f.set(), s, 16, 32, 48, Fake.rows, Fake.heads, 256, 512));
    try std.testing.expectEqual(before, count);
}
