//! Capture-only initialization of the identical per-layer target FFN keys; no numerical launches.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("cuda_ffn_graph.zig");
const fw = @import("cuda_forward.zig");

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (!s.ffn_warm) return;
    if ((world != 2 and world != 4) or !o.drafter or !o.mtp or o.vision != null or !s.ffn_graphs or !@import("cuda_nvfp4_graph.zig").wide(s) or
        !s.tiled or !@import("cuda_target_precision.zig").supported(s) or (s.verify_rows != 16 and s.verify_rows != 32) or
        (s.pages and !s.kv8) or s.kv_host != 0 or s.verify_graphs) return error.FfnWarmScope;
}

/// A failed init must drain uploaded executables before the modules and buffers unwind.
pub fn unwind(e: anytype) void {
    if (!e.settings.ffn_warm) return;
    cuda.completion.drain(e.d, e.stream, "GLM FFN warm uploads did not drain");
    e.m.freeFfnGraphs();
}

pub fn prepare(e: anytype) !void {
    if (!e.settings.ffn_warm) return;
    const g = e.m.ffn_graphs orelse return error.FfnWarmScope;
    const shared = if (e.bs) |*b| b else return error.FfnWarmScope;
    const solo: usize = e.settings.verify_rows;
    if (e.bd.rows != fw.window_rows or solo > e.bd.rows or shared.rows != 64 or e.bd.prefill or shared.prefill)
        return error.FfnWarmRows;
    const count = e.w.layers.len * (solo + shared.rows);
    if (count > g.cache.settings.max) return error.FfnWarmCapacity;
    std.log.info("rank {d}: GLM_FFN_WARM_START planned={d} capture_only=true", .{ e.share.rank, count });
    const free_before = try freeBytes(e.d);
    const before = e.expert_fused_calls;
    defer g.warm_host_exl3 = e.expert_fused_calls - before;
    try buffer(e, g, &e.bd, solo);
    try buffer(e, g, shared, shared.rows);
    try e.stream.synchronize();
    if (g.warm_captured != count or g.cache.count() != count) return error.FfnWarmCount;
    const free_after = try freeBytes(e.d);
    std.log.info("rank {d}: GLM_FFN_WARM_DONE captured={d} launches=0 eager=0 held={d} free_before={d} free_after={d}", .{ e.share.rank, g.warm_captured, g.cache.count(), free_before, free_after });
}

fn buffer(e: anytype, g: *graph.Graphs, b: *fw.Buffers, limit: usize) !void {
    var one: [1]fw.Rank = undefined;
    const f = e.view(b, &one);
    const r = one[0];
    for (1..limit + 1) |rows| {
        const output = fw.Forward.slot(r, rows);
        for (e.w.layers) |layer| try graph.prepare(g, &f, r, layer, rows, output);
    }
}

fn freeBytes(d: *const cuda.Driver) !usize {
    var free: usize = 0;
    var total: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo FFN warm");
    return free;
}
