//! Gather greedy decisions once for all shared parts; mixed or structured draws retain per-part dispatch.
const std = @import("std");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const Engine = engine.Engine;

pub fn draw(e: *Engine, b: *fw.Buffers, parts: []const engine.Part, segs: []const fw.Seg, rows: usize) !void {
    const all = eligible(e, parts);
    var picked: [engine.shared_rows]u32 = undefined;
    if (all) {
        for (parts, segs) |p, g| if (p.out.len < g.rows) return error.WindowTooWide;
        try @import("cuda_engine_pick.zig").greedySharedRows(e, b.logits, rows, picked[0..rows]);
    }
    for (parts, segs) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        if (all) @memcpy(p.out[0..g.rows], picked[g.row0..][0..g.rows]) else try e.drawVerify(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
    }
}

fn eligible(e: *const Engine, parts: []const engine.Part) bool {
    if (!e.settings.greedy_shared or !e.settings.greedy_batch or e.structured) return false;
    for (parts) |p| if (e.seqs.items[p.slot].?.sampling != null) return false;
    return true;
}

test "greedy batching: shared routing keeps mixed sampled and structured draws on the original path" {
    const Seq = @import("cuda_engine_types.zig").Seq;
    var a: Seq = undefined;
    var b: Seq = undefined;
    a.sampling = null;
    b.sampling = null;
    var seqs = [_]?*Seq{ &a, &b };
    var e: Engine = undefined;
    e.settings.greedy_batch = true;
    e.settings.greedy_shared = true;
    e.structured = false;
    e.seqs = .empty;
    e.seqs.items = &seqs;
    e.seqs.capacity = 2;
    const parts = [_]engine.Part{
        .{ .slot = 0, .ids = &.{1}, .out = &.{} },
        .{ .slot = 1, .ids = &.{2}, .out = &.{} },
    };
    try std.testing.expect(eligible(&e, &parts));
    b.sampling = .{ .temperature = 1, .seed = 42 };
    try std.testing.expect(!eligible(&e, &parts));
    b.sampling = null;
    e.structured = true;
    try std.testing.expect(!eligible(&e, &parts));
    e.structured = false;
    e.settings.greedy_shared = false;
    try std.testing.expect(!eligible(&e, &parts));
}
