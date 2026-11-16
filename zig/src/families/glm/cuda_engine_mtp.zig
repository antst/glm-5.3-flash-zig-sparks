//! Target MTP chains keep their original operation order; external DFlash uses its own block path.
const glm = @import("cuda.zig");
const Engine = glm.engine.Engine;

pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
    if (!e.mtp or e.w.mtp == null) return error.NoDraftHead;
    if (depth > glm.engine.max_drafts) return error.TooDeep;
    const bm = if (e.bm) |*b| b else return error.NoDraftHead;
    var one: [1]glm.forward.Rank = undefined;
    const f = e.view(bm, &one);
    const D = e.c.hidden;
    const hidden = if (from_prompt) @import("cuda_prompt_hold.zig").input(e) else e.cur.rows_at;
    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
    e.cur.in_head = 0;
    const next: []const i32 = @ptrCast(follow);
    var s0: usize = 0;
    while (s0 < next.len) : (s0 += glm.forward.window_rows) {
        const part = next[s0..@min(next.len, s0 + glm.forward.window_rows)];
        try f.mtpMode(part, &.{hidden + s0 * D * 2}, depth > 0);
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + part.len);
    }
    for (0..depth) |j| {
        if (chances != null and !e.structured and e.cur.sampling == null) {
            const p = try @import("cuda_engine_pick.zig").greedyChance(e, bm.logits);
            out[j] = p.token;
            chances.?[j] = p.chance;
        } else {
            try e.draw(bm.logits, 1, e.cur.st.pos + 1 + j, 0, out[j..][0..1]);
            if (chances) |c| c[j] = 1; // a sampled or structured draft keeps chance 1: its window runs whole
        }
        if (j + 1 < depth) {
            try f.mtp(@ptrCast(out[j..][0..1]), &.{bm.fnormed});
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
            e.cur.in_head += 1;
        }
    }
}
