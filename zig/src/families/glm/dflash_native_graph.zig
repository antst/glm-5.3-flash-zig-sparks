//! One immutable owner's complete DFlash block, including the borrowed native gathers.
const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;
const Cache = gc.Cache(u8);

pub const Binding = struct {
    engine: usize,
    weights: u64,
    scratch: u64,
    stream: usize,
    comm: usize,
    group: usize,
    modules: usize,
};

pub fn scope(world: usize, split: bool, attention: bool, graphs: bool) !void {
    if (!((world == 4 and attention) or (world == 2 and !attention)) or !split or !graphs) return error.DFlashNativeGraphScope;
}

/// Family startup checks the actual world before opening rank links or allocating the model.
pub fn require(world: usize, settings: anytype, active_drafter: bool) !void {
    if (!settings.dflash_native_graph) return;
    if (!active_drafter or (world != 2 and world != 4)) return error.DFlashNativeGraphScope;
    const split = @import("dflash.zig").mlpSplit(world, settings.dflash_mlp_wide).split;
    try scope(world, split, settings.dflash_attn_tp4, settings.dflash_graphs);
}

pub fn payload(streams: usize) !usize {
    if (streams == 0 or streams > 8) return error.DFlashNativeGraphStreams;
    return streams * 8 * 4096 * 4;
}

pub const Native = struct {
    driver: gc.CudaEngine,
    cache: Cache,
    binding: ?Binding = null,
    rank: usize,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, rank: usize) !*Native {
        const n = try gpa.create(Native);
        n.* = .{ .driver = .{ .d = d }, .cache = undefined, .rank = rank };
        const settings: gc.Settings = .{ .max = 8, .min_keep = 0, .hold = true };
        n.cache = Cache.init(gpa, n.driver.engine(), settings);
        n.cache.settings = settings; // effective enable was agreed before allocation
        n.cache.cap = 8;
        n.cache.name = "glm-dflash-native-block";
        return n;
    }

    pub fn destroy(n: *Native, gpa: std.mem.Allocator) void {
        const s = n.cache.stats;
        std.log.info("rank {d}: GLM_DFLASH_NATIVE_GRAPH_CLOSE captured={d} replayed={d} eager={d} evicted={d} dropped={d} held={d}", .{
            n.rank, s.captured, s.replayed, s.eager, s.evicted, s.dropped, n.cache.count(),
        });
        n.cache.deinit();
        gpa.destroy(n);
    }

    /// Exactly eight keys, never evicted or invalidated: coordinated misses keep every rank's hit set identical.
    pub fn run(n: *Native, streams: usize, binding: Binding, stream: cuda.Stream, vote: gc.Agree, body: gc.Body) !gc.Outcome {
        _ = try payload(streams);
        if (n.binding) |old| if (!std.meta.eql(old, binding)) return error.DFlashNativeGraphOwnerChanged;
        n.binding = binding;
        n.cache.agree = vote;
        return n.cache.run(@intCast(streams), stream, 0, body);
    }
};

pub fn pass(n: *Native, dr: anytype, e: anytype, streams: usize, rank0: bool) !void {
    try scope(dr.world, dr.split, dr.attention != null, true);
    const bytes = try payload(streams);
    const group = e.group orelse return error.DFlashNativeGraphWire;
    if (!@import("cuda_comm1107.zig").graphReady(group, e.stream, bytes)) return error.DFlashNativeGraphWire;
    if (rank0 != (dr.rank == 0)) return error.DFlashNativeGraphRank;
    const Context = struct {
        dr: @TypeOf(dr),
        e: @TypeOf(e),
        streams: usize,
        rank0: bool,

        fn whole(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try @import("dflash_block.zig").pass(c.dr, c.e, c.streams, c.rank0);
        }
        fn agree(ptr: *anyopaque, ok: bool) anyerror!bool {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            const all = try c.e.shared(if (ok) 0 else 1) == 0;
            try c.e.group.?.check();
            return all;
        }
    };
    var c: Context = .{ .dr = dr, .e = e, .streams = streams, .rank0 = rank0 };
    _ = try n.run(streams, .{
        .engine = @intFromPtr(e),
        .weights = dr.weights.ptr,
        .scratch = dr.scratch.ptr,
        .stream = @intFromPtr(e.stream.handle),
        .comm = @intFromPtr(e.comm),
        .group = @intFromPtr(group),
        .modules = @intFromPtr(&e.m),
    }, e.stream, .{ .ctx = &c, .all = Context.agree }, .{ .ctx = &c, .run = Context.whole });
}
