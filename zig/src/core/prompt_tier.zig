//! Optional bounded family tiers; all communication and staging run at host boundaries.
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");

pub const Tier = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        /// Metadata and local source pins only. A keep may run inside a collective's mark hook.
        queued: *const fn (*anyopaque, *const pc.Entry, []const u32) void,
        /// On the engine thread, before a new backend pass: atomically restore or retain `have`.
        recall: *const fn (*anyopaque, *pc.Store, []const u32, u32, []const u32, []const modes.Span, ?*pc.Entry) ?*pc.Entry,
        /// Between backend operations only; bounded staging, completions and peer protocol progress.
        pump: *const fn (*anyopaque, u64) void,
        pending: *const fn (*anyopaque) bool,
        /// After a host-boundary reservation, settle retired source leases needed by the next pass's marks.
        room: ?*const fn (*anyopaque, u64) void = null,
        /// Drain or cancel all pins before cache entries and the family's state arena are released.
        stop: *const fn (*anyopaque, u64) void,
        destroy: *const fn (*anyopaque) void,
    };
};

pub fn queued(s: *pc.Store, e: *const pc.Entry, starts: []const u32) void {
    if (s.tier) |t| t.vtable.queued(t.ptr, e, starts);
}

pub fn recall(s: *pc.Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*pc.Entry) ?*pc.Entry {
    const t = s.tier orelse return have;
    return t.vtable.recall(t.ptr, s, prompt, born, starts, spans, have);
}

pub fn pump(s: *pc.Store, budget_ns: u64) void {
    if (s.tier) |t| t.vtable.pump(t.ptr, budget_ns);
}

pub fn pending(s: *const pc.Store) bool {
    return if (s.tier) |t| t.vtable.pending(t.ptr) else false;
}

pub fn prepareRoom(s: *pc.Store, bytes: u64) void {
    if (s.tier) |t| if (t.vtable.room) |f| f(t.ptr, bytes);
}

pub fn stop(s: *pc.Store, budget_ns: u64) void {
    if (s.tier) |t| t.vtable.stop(t.ptr, budget_ns);
}

pub fn destroy(s: *pc.Store) void {
    if (s.tier) |t| t.vtable.destroy(t.ptr);
    s.tier = null;
}

fn fits(s: *const pc.Store, at: u32, bytes: u64) bool {
    if (bytes > s.budget or s.held > s.budget or s.entries.items.len >= s.most) return false;
    const f = s.family.vtable;
    if (f.trim == null) if (f.reuses) |reuse| if (reuse(s.family.ptr, at)) return true;
    const room = s.budget - s.held;
    const spare = s.spare();
    return spare <= room and bytes <= room - spare;
}

/// Reserve before any rank reads a byte; `have` remains available if the tier refuses or fails.
pub fn makeRoom(s: *pc.Store, have: ?*pc.Entry, prompt: []const u32, born: u32, at: u32, bytes: u64) bool {
    if (bytes > s.budget or s.most == 0) return false;
    while (!fits(s, at, bytes)) {
        if (s.family.vtable.trim) |trim| {
            const before = s.spare();
            trim(s.family.ptr, s.budget -| s.held -| bytes);
            if (s.spare() < before) continue;
        }
        prepareRoom(s, bytes);
        if (fits(s, at, bytes)) return true;
        s.remove(@import("prompt_cache_others.zig").victim(s, have, prompt, born, bytes, 1) orelse return false);
        s.counts.evicted += 1;
    }
    return true;
}

/// The coordinator has committed every rank. Publish owned metadata last; the caller still owns `saved` on failure.
pub fn adopt(s: *pc.Store, saved: pc.Saved, prompt: []const u32, born: u32, at: u32, spans: []const modes.Span, shared: bool) !*pc.Entry {
    const n = @as(usize, at) + s.rules.lookahead;
    if (n > prompt.len) return error.TierPrefix;
    const e = try s.gpa.create(pc.Entry);
    errdefer s.gpa.destroy(e);
    const lease = try s.token_pool.take(s.gpa, prompt);
    errdefer lease.release(s.gpa);
    const tokens = lease.words[0..n];
    const route = try modes.prefix(s.gpa, spans, at);
    errdefer s.gpa.free(route);
    const last = lease.words;
    const f = s.family.vtable;
    const bytes = if (f.charged) |charged| charged(s.family.ptr, saved) else f.bytes(s.family.ptr, at);
    e.* = .{ .tokens = tokens, .at = at, .saved = saved, .bytes = bytes, .born = born, .used = s.clock, .last = last, .decode_spans = route, .shared = shared, .own = last, .token_lease = lease };
    try @import("prompt_cache_index.zig").add(s, e);
    @import("prompt_cache_policy.zig").link(s, e);
    s.held += bytes;
    return e;
}
