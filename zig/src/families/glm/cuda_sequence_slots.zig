//! Bounded opt-in retention of existing nonpaged sequence slots; no allocation or slot selection here.
const std = @import("std");
const cuda = @import("cuda");

pub const keep_slots = 8; // slot0 already stays; only1..7 are newly retained
const State = @import("cuda_state.zig").State;
const Seq = @import("cuda_engine_types.zig").Seq;

pub fn make(e: anytype, slot: u32) !void {
    if (e.settings.linear_paging and slot >= e.linear.?.retain_slots) try e.linear.?.pool.trim();
    const q = try e.gpa.create(Seq);
    errdefer e.gpa.destroy(q);
    q.* = .{ .st = if (e.settings.linear_cache)
        try State.initLinear(e.gpa, e.d, &e.w, e.capacity, e.linear orelse return error.LinearKvScope)
    else
        try State.initHead(e.gpa, e.d, &e.w, e.capacity, e.settings.kv8, !@import("cuda_dflash_head_mode.zig").omitted(e)) };
    e.seqs.items[slot] = q;
    if (e.settings.retain_sequence_slots) e.sequence_counts.created += 1;
}

pub const Counts = struct { created: u64 = 0, retained_releases: u64 = 0, destroyed: u64 = 0 };

pub fn reserveLeader(x: @import("cuda_follow.zig").Leader, streams: u32) !void {
    if (!x.e.settings.prewarm_sequence_slots or x.e.settings.pages) return;
    try x.reserveSlots(streams);
    const lazy = if (x.e.settings.linear_paging) x.e.linear.?.per_state else 0;
    std.log.info("GLM_PREWARM_SEQUENCE_SLOTS admitted={d} per_rank_bytes={d}", .{ streams, (x.e.seqBytes() - lazy) * streams });
}

/// Materialize only the slots admitted by the existing CUDA budget, before taking a request.
pub fn reserveLocal(e: anytype, streams: u32) !void {
    if (!e.settings.prewarm_sequence_slots or !e.settings.retain_sequence_slots or e.pages != null or
        e.images != null or e.slot != 0 or streams == 0 or streams > keep_slots)
        return error.BadSequenceReserve;
    defer {
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }
    for (1..streams) |slot| try e.bind(@intCast(slot));
    if (e.settings.linear_paging) e.linear.?.retain_slots = streams;
}

/// Called only by release's nonpaged branch. The original free path still owns every other slot.
pub fn retain(e: anytype, slot: u32) bool {
    if (!e.settings.retain_sequence_slots or slot == 0 or slot >= keep_slots) return false;
    if (e.settings.linear_paging and slot >= e.linear.?.retain_slots) return false;
    if (e.seqs.items[slot] == null) return false;
    cuda.completion.drain(e.d, e.stream, "GLM sequence release did not drain");
    if (e.verify_graphs) |g| g.invalidate();
    e.sequence_counts.retained_releases += 1;
    return true;
}

pub fn startup(e: anytype) void {
    if (!e.settings.retain_sequence_slots) return;
    std.log.info("GLM_RETAIN_SEQUENCE_SLOTS rank={d} on=true slots={d} state_creates={d} retained_releases={d}", .{ e.share.rank, keep_slots, e.sequence_counts.created, e.sequence_counts.retained_releases });
}

/// Called after the actual Engine deinit loop freed every nonnull sequence, before its list is dropped.
pub fn close(e: anytype) void {
    if (!e.settings.retain_sequence_slots) return;
    var live: usize = 0;
    for (e.seqs.items) |q| live += @intFromBool(q != null);
    std.log.info("GLM_RETAIN_SEQUENCE_SLOTS_CLOSE rank={d} state_creates={d} retained_releases={d} state_destroys={d} live_slots={d}", .{ e.share.rank, e.sequence_counts.created, e.sequence_counts.retained_releases, e.sequence_counts.destroyed, live });
}
