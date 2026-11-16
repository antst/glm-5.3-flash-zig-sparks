//! Compute-only target FFNs: immutable weights/scratch, no captured collectives or rank votes.
const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");

pub const Key = struct { layer: usize, rows: usize, weights: usize, scratch: u64, input: u64, output: u64 };
pub const Cache = gc.Cache(Key);
pub const capacity = 512;
pub const fused_capacity = 4608; // 45 layers across active solo32 and shared64 keys, with bounded spare room.
pub fn capacityFor(fused: bool) u32 {
    return if (fused) fused_capacity else capacity;
}

pub fn eligible(prefill: bool, profiled: bool, rows: usize, output: ?u64) bool {
    return !prefill and !profiled and rows >= 1 and rows <= 64 and output != null;
}

pub const Graphs = struct {
    gpa: std.mem.Allocator,
    driver: gc.CudaEngine,
    cache: Cache,
    rank: usize,
    stream: ?usize = null,
    modules: ?usize = null,
    warm_captured: u64 = 0,
    warm_host_exl3: u64 = 0,
    full_eager: u64 = 0,
    dense_calls: u64 = 0,
    exl3_calls: u64 = 0,
    other_moe_calls: u64 = 0,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize) !*Graphs {
        return createMode(gpa, d, rank, false);
    }

    pub fn createMode(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize, fused: bool) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .gpa = gpa, .driver = .{ .d = d }, .cache = undefined, .rank = rank };
        const settings: gc.Settings = .{ .max = capacityFor(fused), .min_keep = 0, .hold = true };
        g.cache = Cache.init(gpa, g.driver.engine(), settings);
        g.cache.settings = settings;
        g.cache.cap = settings.max;
        g.cache.name = "glm-ffn-compute";
        return g;
    }

    /// The engine drains first, then destroys this cache before any weights, buffers or modules.
    pub fn destroy(g: *Graphs) void {
        const s = g.cache.stats;
        std.log.info("rank {d}: GLM_FFN_GRAPHS_CLOSE captured={d} replayed={d} eager={d} full_eager={d} held={d} failed={any} dense_calls={d} exl3_calls={d} other_moe_calls={d}", .{
            g.rank,        s.captured,   s.replayed,        s.eager, g.full_eager, g.cache.count(), g.cache.failed,
            g.dense_calls, g.exl3_calls, g.other_moe_calls,
        });
        if (g.warm_captured != 0) std.log.info("rank {d}: GLM_FFN_WARM_CLOSE warm_captured={d} request_captured={d} warm_host_exl3={d}", .{ g.rank, g.warm_captured, s.captured - g.warm_captured, g.warm_host_exl3 });
        g.cache.deinit();
        g.gpa.destroy(g);
    }

    pub fn miss(g: *const Graphs, key: Key) bool {
        return g.cache.settings.on and !g.cache.failed and !g.cache.has(key) and g.cache.count() < g.cache.settings.max;
    }

    /// Record and upload only: failure propagates, with no eager fallback or graph execution.
    pub fn prepare(g: *Graphs, key: Key, stream: cuda.Stream, body: gc.Body) !void {
        const c = &g.cache;
        if (!c.settings.on or c.failed or !c.settings.hold or c.agree != null or c.room != null or c.meter != null)
            return error.FfnWarmCacheScope;
        if (c.fingerprint) |f| if (f != 0) return error.FfnWarmFingerprint;
        if (c.has(key)) return;
        if (c.count() >= c.settings.max) return error.FfnWarmCapacity;
        try c.entries.ensureUnusedCapacity(c.gpa, 1);
        const exec = try c.engine.capture(c.engine.ctx, stream, body);
        c.tick += 1;
        c.entries.putAssumeCapacityNoClobber(key, .{ .exec = exec, .used = c.tick });
        c.fingerprint = 0;
        c.stats.captured += 1;
        g.warm_captured += 1;
    }

    /// The caller drains before a miss; hits enqueue only the graph, and a full cache keeps new keys eager.
    pub fn run(g: *Graphs, key: Key, stream: cuda.Stream, body: gc.Body) !gc.Outcome {
        if (!g.cache.has(key) and g.cache.count() >= g.cache.settings.max) {
            try body.run(body.ctx, stream);
            g.full_eager += 1;
            return .eager;
        }
        return g.cache.run(key, stream, 0, body);
    }
};

const Context = struct {
    f: *const fw.Forward,
    r: fw.Rank,
    l: weights.Layer,
    rows: usize,
    output: u64,
    fn body(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        try c.f.ffnBody(c.r, c.l, c.rows, c.output);
    }
};

pub fn prepare(g: *Graphs, f: *const fw.Forward, r: fw.Rank, l: weights.Layer, rows: usize, output: u64) !void {
    return operation(g, f, r, l, rows, output, true);
}

pub fn run(g: *Graphs, f: *const fw.Forward, r: fw.Rank, l: weights.Layer, rows: usize, output: u64) !void {
    return operation(g, f, r, l, rows, output, false);
}

fn operation(g: *Graphs, f: *const fw.Forward, r: fw.Rank, l: weights.Layer, rows: usize, output: u64, comptime warming: bool) !void {
    const stream = @intFromPtr(f.stream.handle);
    const modules = @intFromPtr(f.m);
    if (g.stream) |old| if (old != stream) return error.FfnGraphOwnerChanged;
    if (g.modules) |old| if (old != modules) return error.FfnGraphOwnerChanged;
    g.stream = stream;
    g.modules = modules;
    const key: Key = .{ .layer = l.index, .rows = rows, .weights = @intFromPtr(r.w), .scratch = r.b.arena.buf.ptr, .input = r.b.normed, .output = output };
    if (g.miss(key)) try f.stream.synchronize();
    var c: Context = .{ .f = f, .r = r, .l = l, .rows = rows, .output = output };
    if (warming) return g.prepare(key, f.stream, .{ .ctx = &c, .run = Context.body });
    _ = try g.run(key, f.stream, .{ .ctx = &c, .run = Context.body });
    switch (l.ffn) {
        .dense => g.dense_calls += 1,
        .moe => |m| if (m.exl3 != null) {
            g.exl3_calls += 1;
        } else {
            g.other_moe_calls += 1;
        },
    }
}
