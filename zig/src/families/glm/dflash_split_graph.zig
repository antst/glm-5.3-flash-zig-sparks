//! Split MLP compute is captured in six stages; its five FP32 partial gathers stay eager.
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
            try block.computeStage(c.dr, c.e, c.streams, c.stage);
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
    var contexts: [graph.split_stages]Context = undefined;
    var stages: [graph.split_stages]cuda.graph_cache.Body = undefined;
    for (&contexts, &stages, 0..) |*ctx, *body, i| {
        ctx.* = .{ .dr = dr, .e = e, .streams = streams, .rank0 = rank0, .stage = i };
        body.* = .{ .ctx = ctx, .run = Context.compute };
    }
    try g.runSplit(e.stream, streams, .{ .engine = @intFromPtr(e), .weights = dr.weights.ptr, .scratch = dr.scratch.ptr }, stages, .{ .ctx = &contexts[0], .run = Context.gather }, .{ .ctx = &contexts[0], .run = Context.finish });
}
