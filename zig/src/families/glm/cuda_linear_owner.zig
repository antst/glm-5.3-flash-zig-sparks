//! Rank-owned virtual address pool for ordinary GLM kernels' FP8 latent arrays.
const std = @import("std");
const cuda = @import("cuda");
const latent = @import("cuda_linear_latent.zig");
const dsa = @import("cuda_dsa_block.zig");

const slots = @import("cuda_sequence_slots.zig").keep_slots;

pub fn open(e: anytype, ctx: *const cuda.Context) !void {
    if (!e.c.exl3) return error.LinearKvScope;
    const layers = @import("cuda_state.zig").mlaLayers(&e.w);
    const owner = try Owner.init(e.gpa, ctx, e.stream, e.capacity, layers);
    owner.on_demand = e.settings.linear_paging;
    std.log.info("GLM_LINEAR_KV_ACTIVE rank={d} extent={d} per_state={d} on_demand={any}", .{
        e.share.rank, owner.pool.page_bytes, owner.per_state, owner.on_demand,
    });
    e.linear = owner;
}

pub const Owner = struct {
    gpa: std.mem.Allocator,
    vmm: cuda.arena.Vmm,
    adapter: cuda.arena.Vmm.Linear,
    pool: cuda.linear_pages.Pool,
    capacity: usize,
    layers: usize,
    per_state: usize,
    on_demand: bool = false,
    forks: u64 = 0,
    generation: u64 = 0,
    retain_slots: u32 = slots,
    spare: [slots][]latent.View = undefined,
    spare_count: usize = 0,

    pub fn init(gpa: std.mem.Allocator, ctx: *const cuda.Context, s: cuda.Stream, capacity: usize, layers: usize) !*Owner {
        if (layers == 0) return error.LinearKvScope;
        const owner = try gpa.create(Owner);
        errdefer gpa.destroy(owner);
        owner.* = .{
            .gpa = gpa,
            .vmm = try cuda.arena.Vmm.open(ctx.d, ctx.device, 256 << 30),
            .adapter = undefined,
            .pool = undefined,
            .capacity = capacity,
            .layers = layers,
            .per_state = undefined,
        };
        errdefer owner.vmm.close();
        owner.adapter = .{ .vmm = &owner.vmm, .stream = s };
        const extent = @max(8 << 20, owner.vmm.granularity);
        const per_layer = try bytesPerLayer(extent, capacity);
        owner.per_state = try std.math.mul(usize, per_layer, layers);
        const max_extents = try std.math.mul(usize, owner.per_state / extent, slots);
        owner.pool = try cuda.linear_pages.Pool.init(gpa, owner.adapter.backend(), extent, max_extents);
        return owner;
    }

    pub fn deinit(o: *Owner) void {
        for (o.spare[0..o.spare_count]) |views| {
            o.release(views);
            o.gpa.free(views);
        }
        o.pool.deinit() catch |err| {
            std.log.err("GLM linear KV pool teardown: {s}", .{@errorName(err)});
            return;
        };
        o.vmm.close();
        o.gpa.destroy(o);
    }

    /// Charge both plane-rounding margins per layer and the speculative write window.
    pub fn leaseCost(o: *const Owner) @import("core").kv_pool.PerRequest {
        return .{
            .layout = .{
                .fixed_bytes = 2 * (o.pool.page_bytes - 1) * o.layers,
                .token_bytes = (latent.row_bytes + latent.scale_bytes) * o.layers,
                .page_tokens = 0,
                .snap_bytes = 0,
            },
            .lookahead = @import("cuda_state.zig").window_rows,
            .scratch = 0,
        };
    }

    /// The admission budget includes the rounded physical extents, independently of their virtual aliases.
    pub fn bytesPerLayer(extent: usize, rows: usize) !usize {
        if (!std.math.isPowerOfTwo(extent) or rows == 0) return error.PageGeometry;
        const data = try std.math.mul(usize, rows, latent.row_bytes);
        const scales = try std.math.mul(usize, rows, latent.scale_bytes);
        const a = try std.math.add(usize, data, extent - 1);
        const b = try std.math.add(usize, scales, extent - 1);
        return std.math.add(usize, a & ~(extent - 1), b & ~(extent - 1));
    }

    /// First serving gate: map at slot reservation, before any rank enters the inference collectives.
    pub fn make(o: *Owner, caches: []dsa.Caches) ![]latent.View {
        if (o.spare_count != 0) {
            o.spare_count -= 1;
            const views = o.spare[o.spare_count];
            attach(views, caches);
            return views;
        }
        const views = try o.gpa.alloc(latent.View, caches.len);
        errdefer o.gpa.free(views);
        var done: usize = 0;
        errdefer o.release(views[0..done]);
        for (views) |*v| {
            v.* = try latent.View.init(&o.pool, o.capacity);
            done += 1;
            if (!o.on_demand) {
                try v.prepareWrite(0, o.capacity);
                try zero(o.adapter.stream, v.codes.base, o.capacity * latent.row_bytes);
                try zero(o.adapter.stream, v.scales.base, o.capacity * latent.scale_bytes);
            }
        }
        attach(views, caches);
        return views;
    }

    fn attach(views: []const latent.View, caches: []dsa.Caches) void {
        for (views, caches) |v, *c| {
            c.lc = v.codes.base;
            c.ls = v.scales.base;
        }
    }

    /// The private sequence arena can come and go. Keep its empty virtual ranges for a later slot.
    pub fn discard(o: *Owner, views: []latent.View) void {
        if (o.on_demand and o.pool.healthy and o.spare_count < o.spare.len) {
            for (views) |*view| {
                view.clear() catch break;
            } else {
                o.spare[o.spare_count] = views;
                o.spare_count += 1;
                return;
            }
        }
        o.release(views);
        o.gpa.free(views);
    }

    /// Initialization uses the same synchronous zeroing API as an ordinary state arena.
    fn zero(s: cuda.Stream, address: u64, bytes: usize) !void {
        try s.d.check(s.d.api.cuMemsetD8_v2(address, 0, bytes), "cuMemsetD8_v2");
    }

    pub fn release(_: *Owner, views: []latent.View) void {
        for (views) |*v| v.deinit() catch |err| {
            std.log.err("GLM linear KV view teardown: {s}", .{@errorName(err)});
        };
    }
};

test "linear latent admission counts rounded backing bytes, no alias charge" {
    const extent = 8 << 20;
    const t = std.testing;
    try t.expectEqual(@as(usize, 16 << 20), try Owner.bytesPerLayer(extent, 4_096));
    try t.expectEqual(@as(usize, 520 << 20), try Owner.bytesPerLayer(extent, 1_048_576));
    try t.expectEqual(@as(usize, 528 << 20), try Owner.bytesPerLayer(extent, 1_048_577));
    try t.expectError(error.Overflow, Owner.bytesPerLayer(extent, std.math.maxInt(usize)));
}

test "linear latent: a cold private arena reuses the returned virtual windows" {
    const Mock = struct {
        fn reserve(p: *anyopaque, bytes: usize) anyerror!u64 {
            const next: *u64 = @ptrCast(@alignCast(p));
            const start = next.*;
            next.* += bytes;
            return start;
        }
        fn create(_: *anyopaque, _: usize) anyerror!u64 {
            unreachable;
        }
        fn release(_: *anyopaque, _: u64, _: usize) anyerror!void {
            unreachable;
        }
        fn map(_: *anyopaque, _: u64, _: usize, _: u64) anyerror!void {
            unreachable;
        }
        fn copy(_: *anyopaque, _: u64, _: u64, _: usize) anyerror!void {
            unreachable;
        }
    };
    const t = std.testing;
    var next: u64 = 4096;
    var o: Owner = .{
        .gpa = t.allocator,
        .vmm = undefined,
        .adapter = undefined,
        .pool = try cuda.linear_pages.Pool.init(t.allocator, .{
            .ptr = &next,
            .granularity = 8,
            .reserve = Mock.reserve,
            .create = Mock.create,
            .release = Mock.release,
            .map = Mock.map,
            .unmap = Mock.release,
            .copy = Mock.copy,
        }, 8, 128),
        .capacity = 4,
        .layers = 1,
        .per_state = 2064,
        .on_demand = true,
    };
    var caches: [1]dsa.Caches = undefined;
    o.discard(try o.make(&caches));
    const base = caches[0].lc;
    const reserved = next;
    for (0..64) |_| {
        o.discard(try o.make(&caches));
        try t.expectEqual(base, caches[0].lc);
    }
    try t.expectEqual(reserved, next);
    try t.expectEqual(@as(usize, 1), o.spare_count);
    o.release(o.spare[0]);
    t.allocator.free(o.spare[0]);
    try o.pool.deinit();
}
