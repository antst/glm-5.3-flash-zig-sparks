//! Make stable linear aliases writable before any rank enters the forward kernels.
const std = @import("std");
const cuda = @import("cuda");
const State = @import("cuda_state.zig").State;

const Range = struct { st: *State, start: usize, end: usize };

pub fn one(e: anytype, start: usize, rows: usize) !void {
    if (!e.settings.linear_paging) return;
    if (!needs(&e.cur.st, start + rows)) return;
    try prepare(e, &.{.{ .st = &e.cur.st, .start = start, .end = start + rows }});
}

pub fn many(e: anytype, segs: anytype) !void {
    if (!e.settings.linear_paging) return;
    var ranges: [@import("cuda_engine.zig").max_shared]Range = undefined;
    var used: usize = 0;
    for (segs) |v| {
        const end = v.st.pos + v.rows;
        if (!needs(v.st, end)) continue;
        ranges[used] = .{ .st = v.st, .start = v.st.pos, .end = end };
        used += 1;
    }
    if (used != 0) try prepare(e, ranges[0..used]);
}

fn needs(st: *State, end: usize) bool {
    return end > st.linear.?.writable;
}

fn map(range: Range) !void {
    const state = &range.st.linear.?;
    if (range.end > range.st.capacity) return error.PastCapacity;
    for (state.views) |*view| try view.prepareWrite(range.start, range.end);
}

fn prepare(e: anytype, ranges: []const Range) !void {
    var failed: ?anyerror = null;
    for (ranges) |range| map(range) catch |err| {
        failed = err;
        break;
    };
    // Positions and cache marks are rank commands. Publish a new writable limit only after all peers map it.
    if (try e.shared(if (failed == null) 0 else 1) != 0) return failed orelse error.PeerLinearKv;
    for (ranges) |range| {
        const state = &range.st.linear.?;
        const extent = state.owner.pool.page_bytes / @import("cuda_linear_latent.zig").row_bytes;
        state.writable = @min(range.st.capacity, std.mem.alignForward(usize, range.end, extent));
    }
}

pub fn reset(seq: anytype, stream: cuda.Stream) !void {
    const state = if (seq.st.linear) |*value| value else return;
    if (!state.owner.on_demand) return;
    state.writable = 0;
    state.resumed = null;
    state.owner.generation += 1;
    state.generation = state.owner.generation;
    for (state.views) |*view| {
        if (view.codes.mapped != 0 or view.scales.mapped != 0 or !state.owner.pool.healthy) break;
    } else return;
    try stream.synchronize();
    for (state.views) |*view| try view.clear();
}

pub fn release(e: anytype, slot: u32) void {
    if (!e.settings.linear_paging) return;
    const seq = e.seqs.items[slot] orelse return;
    reset(seq, e.stream) catch |err| {
        seq.st.linear.?.owner.pool.healthy = false;
        std.log.err("GLM linear KV release: {s}", .{@errorName(err)});
    };
}

pub fn close(owner: *@import("cuda_linear_owner.zig").Owner, rank: usize) void {
    if (owner.on_demand) std.log.info("GLM_LINEAR_KV_CLOSE rank={d} shared_prefix_restores={d}", .{ rank, owner.forks });
    owner.deinit();
}
