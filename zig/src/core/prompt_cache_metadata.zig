//! Entry metadata ownership, including shared immutable token histories and later resume histories.
const pc = @import("prompt_cache.zig");

pub fn free(s: *pc.Store, e: *pc.Entry) void {
    if (s.disk) |r| pc.rows.write.dropped(r, e);
    s.family.vtable.drop(s.family.ptr, e.saved);
    if (e.token_lease) |l| l.release(s.gpa) else s.gpa.free(if (e.own.len > 0) e.own else e.tokens);
    s.gpa.free(e.decode_spans);
    s.gpa.free(e.images);
    dropLast(s, e);
    e.ext.deinit(s.gpa);
    s.gpa.destroy(e);
}

fn dropLast(s: *pc.Store, e: *pc.Entry) void {
    if (e.last_lease) |l| l.release(s.gpa) else if (e.own.len == 0 or e.last.ptr != e.own.ptr) s.gpa.free(e.last);
    e.last_lease = null;
}

pub fn resumed(s: *pc.Store, e: *pc.Entry, prompt: []const u32) void {
    const lease = s.token_pool.take(s.gpa, prompt) catch return;
    dropLast(s, e);
    e.last = lease.words;
    e.last_lease = lease;
}

pub fn abandon(s: *pc.Store, e: *pc.Entry, saved: bool) void {
    if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
    if (e.token_lease) |lease| lease.release(s.gpa) else {
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
    }
    s.gpa.destroy(e);
}
