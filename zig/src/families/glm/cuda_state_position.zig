//! Host and device cursors at the same sequence position.
const cuda = @import("cuda");
const State = @import("cuda_state.zig").State;

/// State.set_mtp_len: rows in the MTP layer's caches (the next MTP step writes from here).
pub fn setMtpLen(st: *State, stream: cuda.Stream, n: usize) !void {
    st.mtp_len = n;
    const v: i32 = @intCast(n);
    try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.mtp_pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
}

pub fn setPos(st: *State, stream: cuda.Stream, pos: usize) !void {
    st.pos = pos;
    const v: i32 = @intCast(pos);
    try st.arena.buf.d.check(st.arena.buf.d.api.cuMemsetD32Async(st.pos_dev, @bitCast(v), 1, stream.handle), "cuMemsetD32Async");
}
