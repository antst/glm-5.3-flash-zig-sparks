//! Bounded compute-only DFlash graphs; changing tables, collectives and host selection stay outside capture.
const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;

pub const max_streams = 16;
const Cache = gc.Cache(u8);
pub const split_stages = 6;
pub const max_split_graphs = max_streams * split_stages;
pub const attention_stages = 11;
pub const max_attention_graphs = max_streams * attention_stages;

/// These allocations and their engine own every direct pointer captured by the compute body.
pub const Binding = struct { engine: usize, weights: u64, scratch: u64 };

pub const Graphs = struct {
    driver: gc.CudaEngine,
    cache: Cache,
    binding: ?Binding = null,
    native: ?*@import("dflash_native_graph.zig").Native = null,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, split: bool, attention: bool) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .driver = .{ .d = d }, .cache = undefined };
        const limit: u32 = if (attention) max_attention_graphs else if (split) max_split_graphs else max_streams;
        g.cache = Cache.init(gpa, g.driver.engine(), .{ .max = limit, .min_keep = 0, .hold = true });
        g.cache.name = "glm-dflash-compute";
        return g;
    }

    pub fn destroy(g: *Graphs, gpa: std.mem.Allocator) void {
        if (g.native) |n| n.destroy(gpa);
        g.cache.deinit();
        gpa.destroy(g);
    }

    /// The caller drains the owning stream before destroying graphs, scratch, weights or kernel modules.
    pub fn run(g: *Graphs, s: cuda.Stream, streams: usize, binding: Binding, split: bool, body: gc.Body, tail: gc.Body) !void {
        if (streams == 0 or streams > max_streams) return error.BadDrafterGraphStreams;
        if (split or !g.cache.settings.on) {
            try body.run(body.ctx, s);
        } else {
            if (g.binding) |b| if (!std.meta.eql(b, binding)) return error.DrafterGraphStorageChanged;
            g.binding = binding;
            _ = try g.cache.run(@intCast(streams), s, 0, body);
        }
        try tail.run(tail.ctx, s);
    }

    /// Compute stage i runs, then its eager gather; the final stage precedes the ordinary candidate tail.
    pub fn runSplit(g: *Graphs, s: cuda.Stream, streams: usize, binding: Binding, stages: [split_stages]gc.Body, between: gc.Body, tail: gc.Body) !void {
        return g.runPieces(s, streams, binding, &stages, between, tail);
    }

    pub fn runAttention(g: *Graphs, s: cuda.Stream, streams: usize, binding: Binding, stages: [attention_stages]gc.Body, between: gc.Body, tail: gc.Body) !void {
        return g.runPieces(s, streams, binding, &stages, between, tail);
    }

    fn runPieces(g: *Graphs, s: cuda.Stream, streams: usize, binding: Binding, stages: []const gc.Body, between: gc.Body, tail: gc.Body) !void {
        if (streams == 0 or streams > max_streams) return error.BadDrafterGraphStreams;
        if (g.binding) |b| if (!std.meta.eql(b, binding)) return error.DrafterGraphStorageChanged;
        g.binding = binding;
        for (stages, 0..) |body, i| {
            const key: u8 = @intCast(i * max_streams + streams - 1);
            _ = try g.cache.run(key, s, 0, body);
            if (i + 1 < stages.len) try between.run(between.ctx, s);
        }
        try tail.run(tail.ctx, s);
    }
};

/// The captured body has no communication when split is false; the tail runs after every launch or eager fallback.
pub fn pass(g: *Graphs, dr: anytype, e: anytype, streams: usize, rank0: bool) !void {
    if (g.native) |n| return @import("dflash_native_graph.zig").pass(n, dr, e, streams, rank0);
    if (dr.attention != null) return @import("dflash_attn_graph.zig").pass(g, dr, e, streams, rank0);
    if (dr.split) return @import("dflash_split_graph.zig").pass(g, dr, e, streams, rank0);
    const Context = struct {
        dr: @TypeOf(dr),
        e: @TypeOf(e),
        streams: usize,
        rank0: bool,

        fn compute(p: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try @import("dflash_block.zig").compute(c.dr, c.e, c.streams);
        }

        fn finish(p: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(p));
            try @import("dflash_block.zig").finish(c.dr, c.e, c.streams, c.rank0);
        }
    };
    var c: Context = .{ .dr = dr, .e = e, .streams = streams, .rank0 = rank0 };
    try g.run(e.stream, streams, .{ .engine = @intFromPtr(e), .weights = dr.weights.ptr, .scratch = dr.scratch.ptr }, dr.split, .{ .ctx = &c, .run = Context.compute }, .{ .ctx = &c, .run = Context.finish });
}
