//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const fsio = @import("store_types.zig").fsio;
const mut = @import("store_types.zig").mut;
const Digest = @import("store_types.zig").Digest;
const Slot = @import("store_types.zig").Slot;
const ReadId = @import("store_types.zig").ReadId;
const ReadyBlock = @import("store_types.zig").ReadyBlock;
const ReadStatus = @import("store_types.zig").ReadStatus;
const ReadError = @import("store_types.zig").ReadError;
const ReadDone = @import("store_types.zig").ReadDone;
const SmallView = @import("store_types.zig").SmallView;
const Counters = @import("store_types.zig").Counters;
const ReadJob = @import("store_types.zig").ReadJob;
const freeRead = Store.freeRead;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const now = Store.now;
const dropBlock = Store.dropBlock;
const remove = Store.remove;
const tree = Store.tree;
const victims = Store.victims;
const convBytes = Store.convBytes;
const cancel = Store.cancel;
const pins = Store.pins;
const readJob = Store.readJob;
const assign = Store.assign;
const giveRead = Store.giveRead;
const BlockReadError = Store.BlockReadError;

pub fn readFailed(s: *Store, r: *ReadJob, name: Digest, e0: BlockReadError) void {
    const e = if (mut(.lost_is_damage) and e0 == error.Missing) error.Damaged else e0;
    switch (e) {
        error.Missing => {
            r.lost.append(s.gpa, name) catch {};
            s.counters.missing_errors += 1;
            if (r.err == null) r.err = .missing;
        },
        error.Damaged => {
            r.bad.append(s.gpa, name) catch {};
            s.counters.crc_errors += 1;
            if (r.err == null) r.err = .damaged;
        },
        error.Io => if (r.err == null) {
            r.err = .io;
        },
    }
}

pub fn readReady(s: *Store, id: ReadId, upto: ?u32, out: []ReadyBlock) struct { n: usize, status: ReadStatus } {
    s.lockS();
    defer s.unlockS();
    const r = s.readJob(id) orelse return .{ .n = 0, .status = .failed };
    const stop = if (mut(.loadpart_past_upto)) r.total else @min(upto orelse r.total, r.total);
    var n: usize = 0;
    while (r.next_hand < stop and n < out.len) {
        const i = r.next_hand;
        switch (r.state[i]) {
            .ready => {},
            .failed => return .{ .n = n, .status = .failed },
            else => break,
        }
        r.state[i] = .handed;
        r.next_hand += 1;
        const b = r.point.blocks[i];
        out[n] = .{ .index = i, .first = i * s.cfg.limits.block_tokens, .tokens = b.tokens, .data = s.slots.bytes(r.slot[i])[0..@intCast(b.bytes)], .slot = r.slot[i] };
        s.counters.loaded_bytes += b.bytes;
        n += 1;
    }
    if (r.err != null and r.next_hand < stop and r.state[r.next_hand] != .ready) return .{ .n = n, .status = .failed };
    if (r.next_hand >= r.total) return .{ .n = n, .status = .done };
    if (r.next_hand >= stop) return .{ .n = n, .status = .done };
    return .{ .n = n, .status = if (r.state[r.next_hand] == .ready) .more else .waiting };
}

pub fn readRelease(s: *Store, slot: Slot) void {
    s.lockS();
    defer s.unlockS();
    const holder = s.slots.holder(slot);
    for (s.reads.items) |r| if (r.id == holder) {
        r.held -= 1;
        break;
    };
    s.slots.give(slot);
    if (holder != 0) for (s.reads.items) |r| if (r.id == holder) {
        s.maybeFreeRead(r);
        break;
    };
    s.assign();
}

pub const Recycler = struct {
    ctx: ?*anyopaque = null,
    recycle: ?*const fn (ctx: ?*anyopaque) void = null,
};

pub fn readAwait(s: *Store, id: ReadId, index: u32, timeout_ns: u64, rc: Recycler) bool {
    const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + timeout_ns;
    while (true) {
        if (rc.recycle) |f| if (!mut(.hang_no_recycle_in_wait)) f(rc.ctx);
        s.lockS();
        const r = s.readJob(id) orelse {
            s.unlockS();
            return false;
        };
        if (index >= r.total) {
            s.unlockS();
            return false;
        }
        const st = r.state[index];
        if (st == .ready or st == .handed) {
            s.unlockS();
            return true;
        }
        if (st == .failed or r.ended or (r.err != null and st == .idle)) {
            s.unlockS();
            return false;
        }
        const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
        if (left <= 0) {
            s.unlockS();
            return false;
        }
        s.done.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@min(left, 20 * std.time.ns_per_ms)), .clock = .awake } }) catch {};
        s.unlockS();
    }
}

pub fn readSmallView(s: *Store, id: ReadId) ?SmallView {
    s.lockS();
    defer s.unlockS();
    const r = s.readJob(id) orelse return null;
    if (r.small != .ready) return null;
    return .{ .bytes = s.slots.bytes(r.small_slot)[0..@intCast(r.point.small_bytes)], .sections = r.point.sections };
}

pub const Failure = struct { err: ?ReadError, bad: []const Digest, lost: []const Digest };

pub fn readFailure(s: *Store, id: ReadId) Failure {
    s.lockS();
    defer s.unlockS();
    const r = s.readJob(id) orelse return .{ .err = .cancelled, .bad = &.{}, .lost = &.{} };
    return .{ .err = r.err, .bad = r.bad.items, .lost = r.lost.items };
}

pub const EndError = error{ Unknown, Incomplete, Failed, OutOfMemory };

pub const Outcome = enum { commit, abort, cancel };

pub fn endRead(s: *Store, id: ReadId, outcome: Outcome) EndError!?ReadDone {
    s.lockS();
    defer s.unlockS();
    const r = s.readJob(id) orelse return error.Unknown;
    var done: ?ReadDone = null;
    var result: EndError!?ReadDone = null;
    if (outcome == .commit) {
        if (r.err != null) {
            result = error.Failed;
        } else if (r.next_hand < r.total or r.small != .ready) {
            result = error.Incomplete;
        } else {
            const p = r.point;
            const lineage = p.lineage.clone(s.metadata.parent) catch return error.OutOfMemory;
            done = .{ .key = p.key, .at = p.at, .shared = p.shared, .group = p.group, .has_raw = p.has_raw, .lineage = lineage, .origin = p.origin };
            s.counters.loaded += 1;
            s.counters.loaded_tokens += p.at;
            result = done;
        }
        if (done == null) return result; // nothing ended: the caller aborts it next
    } else if (outcome == .abort) s.counters.load_failed += 1;
    r.ended = true;
    if (r.small != .queued) s.slots.give(r.small_slot); // (else its reader gives it back)
    if (s.reading.getPtr(r.point.key)) |v| {
        v.* -= 1;
        if (v.* == 0) _ = s.reading.remove(r.point.key);
    }
    // blocks read but never handed: their slots back now (a block still reading gives its own back)
    for (r.state, 0..) |st, i| if (st == .ready) {
        s.giveRead(r, r.slot[i]);
        r.state[i] = .failed;
    };
    s.maybeFreeRead(r);
    s.assign();
    return done;
}

pub fn maybeFreeRead(s: *Store, r: *ReadJob) void {
    if (!r.ended or r.disk != 0 or r.held != 0) return;
    for (s.reads.items, 0..) |x, k| if (x == r) {
        _ = s.reads.orderedRemove(k);
        break;
    };
    const p = r.point;
    p.pins -= 1;
    if (p.pins == 0 and p.removed) p.deinit(s.gpa);
    freeRead(s.gpa, r);
}

pub fn forget(s: *Store, key: Digest, bad: []const Digest, lost: []const Digest) void {
    const gpa = s.gpa;
    s.lockS();
    defer s.unlockS();
    var gone: std.AutoHashMapUnmanaged(Digest, void) = .empty;
    defer gone.deinit(gpa);
    for (bad) |b| {
        gone.put(gpa, b, {}) catch {};
        if (mut(.damage_not_widened)) continue;
        if (s.have.get(b)) |info| if (s.inodes.get(info.ino)) |ino| {
            for (ino.names.items) |nm| gone.put(gpa, nm, {}) catch {};
        };
    }
    if (!mut(.lost_names_kept)) for (lost) |b| gone.put(gpa, b, {}) catch {};
    var drop: std.ArrayList(Digest) = .empty;
    defer drop.deinit(gpa);
    drop.append(gpa, key) catch {};
    var it = s.points.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr, &key)) continue;
        for (e.value_ptr.*.blocks) |b| if (gone.contains(b.name)) {
            drop.append(gpa, e.key_ptr.*) catch {};
            break;
        };
    }
    // every one goes, read or not: a point being read is unlisted now and freed when its last read ends (those
    for (drop.items) |k| {
        if (mut(.forget_skips_reading) and s.reading.contains(k) and !std.mem.eql(u8, &k, &key)) continue;
        s.remove(k, "dropped");
    }
    var git = gone.keyIterator();
    while (git.next()) |b| s.dropBlock(b.*);
}

pub fn stepReads(s: *Store, ids: []const ReadId, turn: *usize, budget_ns: u64, ctx: ?*anyopaque, copy: *const fn (ctx: ?*anyopaque, id: ReadId, b: ReadyBlock) bool) usize {
    if (ids.len == 0) return 0;
    const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
    const k = if (mut(.budget_same_first)) 0 else turn.* % ids.len;
    turn.* +%= 1;
    var n: usize = 0;
    for (0..ids.len) |x| {
        const id = ids[(k + x) % ids.len];
        const own = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
        while (std.Io.Clock.awake.now(s.io).toNanoseconds() < (if (mut(.budget_each_read)) own else deadline)) {
            var one: [1]ReadyBlock = undefined;
            const got = s.readReady(id, null, &one);
            if (got.n == 0) break;
            n += 1;
            if (!copy(ctx, id, one[0])) return n;
        }
    }
    return n;
}

/// A constant-cost worker snapshot for the pressure watchdog; no index walk or disk operation.
pub fn writeProgress(s: *Store) u64 {
    s.lockS();
    defer s.unlockS();
    return s.counters.staged_blocks +% s.counters.written_blocks +% s.counters.committed;
}

pub fn counts(s: *Store) Counters {
    s.lockS();
    defer s.unlockS();
    var x = s.counters;
    x.points = s.points.count();
    x.blocks = s.have.count();
    x.stored_bytes = s.stored_bytes;
    x.queued_bytes = s.queued_bytes;
    x.pending_jobs = s.jobs_by_id.count();
    x.reads = s.reads.items.len;
    x.staging_bytes = s.slots.heldBytes();
    x.staging_peak_bytes = s.slots.peakBytes();
    x.disk_free_bytes = s.disk_free;
    var it = s.points.valueIterator();
    while (it.next()) |pp| if (pp.*.recency_dirty) {
        x.recency_pending += 1;
    };
    x.index_bytes = s.metadata.used.load(.monotonic);
    x.index_peak = s.metadata.peak.load(.monotonic);
    x.index_limit = s.metadata.limit;
    x.metadata_refused += s.metadata.refused.load(.monotonic);
    return x;
}

pub fn refreshDiskFree(s: *Store) void {
    const free = fsio.diskFree(&s.hooks, s.blocks_fd);
    s.lockS();
    defer s.unlockS();
    s.disk_free = free;
}

pub fn isCrashed(s: *Store) bool {
    s.lockS();
    defer s.unlockS();
    return s.crashed;
}

pub const Summary = struct {
    conversations: u64 = 0,
    shared_states: u64 = 0,
    forks: u64 = 0,
    turns_kept_max: u64 = 0,
    conversation_bytes_max: u64 = 0,
    conversation_tokens_max: u64 = 0,
};

pub fn summary(s: *Store) Summary {
    const gpa = s.gpa;
    s.lockS();
    defer s.unlockS();
    var out: Summary = .{};
    var t = s.tree() catch return out;
    defer t.deinit(gpa);
    var v = s.victims(null) catch return out;
    defer v.deinit(gpa);
    for (v.items) |x| switch (x.why) {
        .pruned_forks => out.forks += 1,
        .pruned_shared => out.shared_states += 1,
        else => {},
    };
    var convs: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer convs.deinit(gpa);
    for (t.conv) |cv| if (cv) |x| {
        convs.put(gpa, x, {}) catch {};
    };
    out.conversations = convs.count();
    var cit = convs.keyIterator();
    var keys: std.ArrayList(Digest) = .empty;
    defer keys.deinit(gpa);
    while (cit.next()) |cv| {
        keys.clearRetainingCapacity();
        var tokens: u64 = 0;
        for (t.keys, 0..) |k, j| if (t.conv[j] == cv.*) {
            keys.append(gpa, k) catch {};
            tokens = @max(tokens, s.points.get(k).?.at);
        };
        out.conversation_bytes_max = @max(out.conversation_bytes_max, s.convBytes(keys.items));
        out.conversation_tokens_max = @max(out.conversation_tokens_max, tokens);
        // its active path's length (from its newest leaf)
        var newest: ?Digest = null;
        var used: i96 = std.math.minInt(i96);
        for (keys.items) |k| {
            const p = s.points.get(k).?;
            if (p.used > used) {
                used = p.used;
                newest = k;
            }
        }
        var len: u64 = 0;
        var cur = newest;
        while (cur) |k| {
            const p = s.points.get(k) orelse break;
            if (p.shared) break;
            len += 1;
            cur = s.parent.get(k) orelse null;
        }
        out.turns_kept_max = @max(out.turns_kept_max, len);
    }
    return out;
}
