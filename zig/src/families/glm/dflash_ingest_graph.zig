//! Context ingest captures compute only; tap copies, position/ring tables and slot bookkeeping remain eager.
const std = @import("std");
const cuda = @import("cuda");
const gc = cuda.graph_cache;
const Binding = @import("dflash_graph.zig").Binding;
pub const Cache = gc.Cache(u16);

pub const Graphs = struct {
    driver: gc.CudaEngine,
    cache: Cache,
    binding: ?Binding = null,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Graphs {
        const g = try gpa.create(Graphs);
        g.* = .{ .driver = .{ .d = d }, .cache = undefined };
        g.cache = Cache.init(gpa, g.driver.engine(), .{ .max = 32, .min_keep = 0, .hold = true });
        g.cache.name = "glm-dflash-ingest";
        return g;
    }

    pub fn destroy(g: *Graphs, gpa: std.mem.Allocator) void {
        g.cache.deinit();
        gpa.destroy(g);
    }

    pub fn run(g: *Graphs, stream: cuda.Stream, rows: usize, binding: Binding, body: gc.Body) !void {
        if (rows == 0 or rows > 2048) return error.DrafterContextTooWide;
        if (g.binding) |old| if (!std.meta.eql(old, binding)) return error.DrafterGraphStorageChanged;
        g.binding = binding;
        _ = try g.cache.run(@intCast(rows), stream, 0, body);
    }
};

pub fn pass(g: *Graphs, dr: anytype, e: anytype, rows: usize) !void {
    const Context = struct {
        dr: @TypeOf(dr),
        e: @TypeOf(e),
        rows: usize,
        fn compute(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try @import("dflash_block.zig").ingestPass(c.dr, c.e, c.rows);
        }
    };
    var c: Context = .{ .dr = dr, .e = e, .rows = rows };
    try g.run(e.stream, rows, .{
        .engine = @intFromPtr(e),
        .weights = dr.weights.ptr,
        .scratch = dr.scratch.ptr,
    }, .{ .ctx = &c, .run = Context.compute });
}
