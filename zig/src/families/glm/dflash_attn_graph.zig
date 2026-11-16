//! TP4 attention and MLP compute surround ten eager partial gathers.
const cuda = @import("cuda");
const graph = @import("dflash_graph.zig");
const block = @import("dflash_block.zig");

pub fn pass(g: *graph.Graphs, dr: anytype, e: anytype, streams: usize, rank0: bool) !void {
    const Context = struct {
        dr: @TypeOf(dr),
        e: @TypeOf(e),
        streams: usize,
        rank0: bool,
        stage: usize,

        fn compute(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try block.computeAttentionStage(c.dr, c.e, c.streams, c.stage);
        }
        fn gather(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try block.gatherLayer(c.dr, c.e, c.streams);
        }
        fn finish(ptr: *anyopaque, _: cuda.Stream) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            try block.finish(c.dr, c.e, c.streams, c.rank0);
        }
    };
    var contexts: [graph.attention_stages]Context = undefined;
    var stages: [graph.attention_stages]cuda.graph_cache.Body = undefined;
    for (&contexts, &stages, 0..) |*ctx, *body, i| {
        ctx.* = .{ .dr = dr, .e = e, .streams = streams, .rank0 = rank0, .stage = i };
        body.* = .{ .ctx = ctx, .run = Context.compute };
    }
    try g.runAttention(e.stream, streams, .{ .engine = @intFromPtr(e), .weights = dr.weights.ptr, .scratch = dr.scratch.ptr }, stages, .{ .ctx = &contexts[0], .run = Context.gather }, .{ .ctx = &contexts[0], .run = Context.finish });
}
