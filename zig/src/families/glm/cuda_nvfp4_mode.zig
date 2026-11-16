//! Explicit NVFP4 decode tile selection, fixed before warm-up and graph capture.
const std = @import("std");

pub fn require(on: bool, exl3: bool, block32: bool, world: usize) !void {
    if (on and (exl3 or block32 or world != 4)) return error.Nvfp4FactScope;
}

pub fn open(e: anytype) !void {
    if (!e.settings.nvfp4_fact) return;
    var count: usize = 0;
    for (e.w.layers) |layer| switch (layer.ffn) {
        .dense => {},
        .moe => |m| {
            if (m.exl3 != null or m.routed.format != .nvfp4 or m.routed.relu2 or
                m.routed.dims != 4096 or m.routed.width != 512) return error.Nvfp4FactScope;
            count += 1;
        },
    };
    if (count == 0) return error.Nvfp4FactScope;
    e.m.experts.fact = true;
    e.m.experts.fact_calls = &e.expert_fact_calls;
    std.log.info("rank {d}: GLM_NVFP4_FACT active=true layers={d} tile=64", .{ e.share.rank, count });
}

pub fn close(e: anytype) void {
    if (e.settings.nvfp4_fact)
        std.log.info("rank {d}: GLM_NVFP4_FACT_CLOSE host_gate_up_calls={d}", .{ e.share.rank, e.expert_fact_calls });
}
