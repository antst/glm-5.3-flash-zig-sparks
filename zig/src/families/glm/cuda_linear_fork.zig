//! Reuse the latent prefix of an active GLM sequence; the prompt store retains its ordinary disk image.
const types = @import("cuda_engine_types.zig");
const work = @import("cuda_linear_work.zig");

pub fn origin(e: anytype) ?types.Origin {
    if (!e.settings.linear_paging) return null;
    return .{ .slot = e.slot, .generation = e.cur.st.linear.?.generation };
}

const Candidate = struct { slot: usize, seq: *types.Seq };

fn source(e: anytype, id: u64, snap: *const types.Snap) ?Candidate {
    if (snap.origin) |saved| if (saved.slot != e.slot and saved.slot < e.seqs.items.len) {
        if (e.seqs.items[saved.slot]) |q| {
            if (q.st.linear.?.generation == saved.generation and q.st.pos >= snap.pos)
                return .{ .slot = saved.slot, .seq = q };
        }
    };
    for (e.seqs.items, 0..) |maybe, slot| {
        const q = maybe orelse continue;
        if (q != e.cur and q.st.linear.?.resumed == id and q.st.pos >= snap.pos)
            return .{ .slot = slot, .seq = q };
    }
    return null;
}

pub fn restore(e: anytype, id: u64, snap: *const types.Snap) !bool {
    if (!e.settings.linear_paging) return false;
    var failure: ?anyerror = null;
    work.reset(e.cur, e.stream) catch |err| {
        failure = err;
    };
    const parent = source(e, id, snap);
    const copy = @import("cuda_sequence_slots.zig").keep_slots + 1;
    const failed = copy + 1;
    const choice = try e.shared(@floatFromInt(if (failure != null) failed else if (parent) |q| q.slot + 1 else copy));
    if (choice >= @as(f64, @floatFromInt(failed))) return failure orelse error.PeerLinearKv;
    if (choice == @as(f64, @floatFromInt(copy))) {
        try work.one(e, 0, snap.pos);
        return false;
    }
    if (parent) |q| {
        if (choice == @as(f64, @floatFromInt(q.slot + 1))) {
            share(e, q.seq, snap.pos) catch |err| {
                failure = err;
            };
        } else failure = error.LinearKvPeersDisagree;
    } else failure = error.LinearKvPeersDisagree;
    if (try e.shared(if (failure == null) 0 else 1) != 0) return failure orelse error.PeerLinearKv;
    e.linear.?.forks += 1;
    return true;
}

fn share(e: anytype, q: *types.Seq, rows: usize) !void {
    const from = &q.st.linear.?;
    const to = &e.cur.st.linear.?;
    // Only mapping or COW boundaries synchronize; subsequent kernels reuse their stable addresses.
    try e.stream.synchronize();
    from.writable = 0;
    for (to.views, from.views) |*child, *parent| try child.borrow(parent, rows);
}

pub fn restored(e: anytype, id: u64) void {
    if (e.settings.linear_paging) e.cur.st.linear.?.resumed = id;
}
