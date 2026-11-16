//! Forward preparation stays outside capture; compute retains the original launch order.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const shared = @import("cuda_shared.zig");
const prof_mod = @import("cuda_prof.zig");

pub fn prepare(f: *const fw.Forward, ids: []const i32) !void {
    const R = ids.len;
    if ((f.dense_segments != null or f.kda_segments != null) and f.ranks.len != 1) return error.DenseScope;
    for (f.ranks) |r| {
        if (R == 0 or R > r.b.rows) return error.BadWindow;
        if (r.segs.len == 0) {
            try r.st.ready(f.stream, R);
            r.st.pending = R;
            r.st.prompt = r.b.prefill;
        } else try shared.begin(r, R, f.stream);
        if (f.dense_segments) |o| try o.prepare(r, R);
        if (f.kda_segments) |o| try o.prepare(r, R);
        try cuda.copy_source.enqueue(f.d, r.b.ids, std.mem.sliceAsBytes(ids), f.stream);
    }
}

pub fn compute(f: *const fw.Forward, R: usize) !void {
    const c = &f.ranks[0].w.config;
    const embed_stamp = prof_mod.span(f.prof, f.stream, .embed);
    for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
    if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
    embed_stamp.end();
    if (f.prof) |p| p.mark(f.stream, .embed);
    var cut: [fw.max_pieces][2]usize = undefined;
    const pieces = if (f.pieces) |p| (if (f.ranks[0].b.prefill) fw.ranges(R, p.count, &cut) else 0) else 0;
    // overlap.Overlap.begin: pieced with two pieces or more, and split takes a chunk of one piece too
    if (pieces > 1 or (pieces == 1 and f.splits(R))) try f.pieced(f.pieces.?, R, cut[0..pieces]) else try f.whole(R);
    try f.finish(R);
    if (f.prof) |p| p.mark(f.stream, .head);
}

pub fn run(f: *const fw.Forward, ids: []const i32) !void {
    try prepare(f, ids);
    try compute(f, ids.len);
}
