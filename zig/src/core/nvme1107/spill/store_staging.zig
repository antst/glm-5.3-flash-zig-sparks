//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const fsio = @import("store_types.zig").fsio;
const mut = @import("store_types.zig").mut;
const Digest = @import("store_types.zig").Digest;
const Source = @import("store_types.zig").Source;
const JobId = @import("store_types.zig").JobId;
const StageReq = @import("store_types.zig").StageReq;
const CancelReason = @import("store_types.zig").CancelReason;
const Completion = @import("store_types.zig").Completion;
const GuardResult = @import("store_types.zig").GuardResult;
const WriteJob = @import("store_types.zig").WriteJob;
const open = Store.open;
const freeJob = Store.freeJob;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const now = Store.now;
const pointName = Store.pointName;
const ref = Store.ref;
const gc = Store.gc;
const queue = Store.queue;
const complete = Store.complete;
const remove = Store.remove;
const req = Store.req;
const staged = Store.staged;
const dropInflight = Store.dropInflight;

pub fn guard(s: *Store, source: Source, from: u32, out: []StageReq) GuardResult {
    s.lockS();
    defer s.unlockS();
    var r: GuardResult = .{};
    var i: usize = 0;
    while (i < s.fifo.items.len) {
        const j = s.fifo.items[i];
        i += 1;
        if (j.state != .open or j.source.id != source.id) continue;
        if (j.source.generation != source.generation) {
            s.cancelJob(j, .source_gone);
            r.cancelled += 1;
            i = 0;
            continue;
        }
        // the blocks left reaching past `from`
        var needed: usize = 0;
        for (j.new.items[j.next..]) |idx| {
            const b = j.point.blocks[idx];
            if (idx * s.cfg.limits.block_tokens + b.tokens > from) needed += 1;
        }
        if (needed == 0) continue;
        const left = j.new.items.len - j.next;
        if (left > out.len - r.staged or left > s.slots.available(.write) or mut(.guard_cancels_only)) {
            s.cancelJob(j, .guard);
            r.cancelled += 1;
            i = 0;
            continue;
        }
        while (j.next < j.new.items.len) {
            out[r.staged] = s.req(j, s.slots.take(.write, j.id).?);
            r.staged += 1;
            s.counters.guard_staged += 1;
        }
        j.listed = false;
        for (s.fifo.items, 0..) |x, k| if (x == j) {
            _ = s.fifo.orderedRemove(k);
            break;
        };
        i = 0;
    }
    return r;
}

pub fn cancel(s: *Store, job: JobId) bool {
    s.lockS();
    defer s.unlockS();
    const j = s.jobs_by_id.get(job) orelse return false;
    if (j.state != .open) return j.state != .publishing;
    s.cancelJob(j, .explicit);
    return true;
}

pub fn cancelJob(s: *Store, j: *WriteJob, why: CancelReason) void {
    j.state = .cancelled;
    j.cancel = why;
    switch (why) {
        .guard => s.counters.cancelled_guard += 1,
        .source_gone => s.counters.cancelled_source += 1,
        else => {},
    }
    s.release(j);
    s.settle(j);
}

pub fn release(s: *Store, j: *WriteJob) void {
    if (j.listed) {
        for (s.fifo.items, 0..) |x, k| if (x == j) {
            _ = s.fifo.orderedRemove(k);
            break;
        };
        j.listed = false;
    }
    for (j.new.items[j.next..]) |idx| {
        const name = j.point.blocks[idx].name;
        if (!s.handOver(j, name)) {
            if (s.inflight.get(name)) |owner| if (owner == j.id) {
                _ = s.inflight.remove(name);
            };
        }
    }
    j.next = j.new.items.len;
    if (j.tail) |ti| if (j.tail_slot != null and j.capture != .pending) {
        s.slots.give(j.tail_slot.?);
        j.tail_slot = null;
        s.dropInflight(j, ti);
    };
    s.recheckParked();
}

pub fn handOver(s: *Store, from: *WriteJob, name: Digest) bool {
    var it = s.jobs_by_id.valueIterator();
    while (it.next()) |kk| {
        const k = kk.*;
        if (k == from or k.state != .open) continue;
        for (k.point.blocks, 0..) |b, idx| {
            if (!std.mem.eql(u8, &b.name, &name) or b.tokens != s.cfg.limits.block_tokens) continue;
            s.fifo.ensureUnusedCapacity(s.gpa, 1) catch return false;
            k.new.append(s.gpa, @intCast(idx)) catch return false;
            s.inflight.put(s.gpa, name, k.id) catch {
                _ = k.new.pop();
                return false;
            };
            if (!k.listed) {
                s.fifo.appendAssumeCapacity(k);
                k.listed = true;
            }
            // its bytes are now `k`'s to write: reserved with `k` (the giver's reservation ends with it)
            if (!mut(.handover_unreserved)) {
                k.size += b.bytes;
                s.queued_bytes += b.bytes;
            }
            return true;
        }
    }
    return false;
}

pub fn settle(s: *Store, j: *WriteJob) void {
    switch (j.state) {
        .open => if (j.next >= j.new.items.len and j.outstanding == 0 and j.capture == .ok and j.tail_slot == null) {
            j.state = .publishing;
            j.shared_at_write = j.point.shared;
            s.queue(.{ .point = j });
        },
        .publishing => {},
        .failed, .cancelled => if (j.outstanding == 0 and j.capture != .pending) s.endJob(j, false),
    }
}

pub fn endJob(s: *Store, j: *WriteJob, published: bool) void {
    if (j.listed) {
        for (s.fifo.items, 0..) |x, k| if (x == j) {
            _ = s.fifo.orderedRemove(k);
            break;
        };
    }
    for (s.parked.items, 0..) |x, k| if (x == j) {
        _ = s.parked.swapRemove(k);
        break;
    };
    if (j.tail_slot) |t| s.slots.give(t);
    s.slots.give(j.small_slot);
    var it = s.inflight.iterator();
    var drop: std.ArrayList(Digest) = .empty;
    defer drop.deinit(s.gpa);
    while (it.next()) |e| if (e.value_ptr.* == j.id) drop.append(s.gpa, e.key_ptr.*) catch {};
    for (drop.items) |d| _ = s.inflight.remove(d);
    _ = s.waiting.remove(j.point.key);
    _ = s.jobs_by_id.remove(j.id);
    s.queued_bytes -= j.size;
    s.ref(j.point.blocks, -1);
    s.gc(j.point.blocks);
    const key = j.point.key;
    if (published) {
        s.complete(.{ .published = .{ .job = j.id, .key = key, .at = j.point.at } });
    } else if (j.state == .cancelled) {
        s.counters.cancelled += 1;
        s.complete(.{ .cancelled = .{ .job = j.id, .key = key, .why = j.cancel } });
    } else {
        s.counters.write_failed += 1;
        s.complete(.{ .failed = .{ .job = j.id, .key = key, .why = j.fail } });
    }
    freeJob(s.gpa, j, !published);
    s.recheckParked();
    s.done.broadcast(s.io);
}

pub fn recheckParked(s: *Store) void {
    var i: usize = 0;
    while (i < s.parked.items.len) {
        const j = s.parked.items[i];
        if (s.blocksSettled(j)) {
            _ = s.parked.swapRemove(i);
            j.parked = false;
            s.queue(.{ .point = j });
        } else i += 1;
    }
}

pub fn blocksSettled(s: *Store, j: *WriteJob) bool {
    for (j.point.blocks) |b| if (!s.have.contains(b.name) and s.inflight.contains(b.name)) return false;
    return true;
}

pub fn poll(s: *Store, out: []Completion) usize {
    s.lockS();
    defer s.unlockS();
    const n = @min(out.len, s.completions.items.len);
    @memcpy(out[0..n], s.completions.items[0..n]);
    std.mem.copyForwards(Completion, s.completions.items, s.completions.items[n..]);
    s.completions.shrinkRetainingCapacity(s.completions.items.len - n);
    return n;
}

pub fn pendingStage(s: *Store) usize {
    s.lockS();
    defer s.unlockS();
    var n: usize = 0;
    var it = s.jobs_by_id.valueIterator();
    while (it.next()) |j| {
        if (waitsOnEngine(j.*)) n += 1;
    }
    return n;
}

pub const Drain = struct { done: bool, unpublished: usize };

pub fn waitsOnEngine(j: *const WriteJob) bool {
    if (j.engine > 0 or j.capture == .pending) return true;
    return j.state == .open and j.next < j.new.items.len;
}

pub fn drain(s: *Store, timeout_ns: u64) Drain {
    const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + timeout_ns;
    s.lockS();
    defer s.unlockS();
    while (true) {
        var engine_wait: usize = 0;
        var writer_wait: usize = 0;
        var it = s.jobs_by_id.valueIterator();
        while (it.next()) |jj| {
            const j = jj.*;
            if (waitsOnEngine(j)) engine_wait += 1 else writer_wait += 1;
        }
        if (writer_wait == 0 and mut(.drain_skips_rewrites)) return .{ .done = engine_wait == 0, .unpublished = engine_wait };
        if (writer_wait == 0) {
            s.sync_sent += 1;
            const want = s.sync_sent;
            s.queue(.{ .sync = want });
            while (s.sync_done < want) {
                const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
                if (left <= 0 or s.crashed) return .{ .done = false, .unpublished = engine_wait };
                s.done.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@min(left, 50 * std.time.ns_per_ms)), .clock = .awake } }) catch {};
            }
            return .{ .done = engine_wait == 0, .unpublished = engine_wait };
        }
        const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
        if (left <= 0 or s.crashed) return .{ .done = false, .unpublished = engine_wait + writer_wait };
        s.done.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@min(left, 50 * std.time.ns_per_ms)), .clock = .awake } }) catch {};
    }
}

pub fn flushRecency(s: *Store, timeout_ns: u64) bool {
    const deadline = std.Io.Clock.awake.now(s.io).toNanoseconds() + timeout_ns;
    s.lockS();
    defer s.unlockS();
    if (s.stopping or s.crashed) return false;
    s.recency_sync_sent += 1;
    const want = s.recency_sync_sent;
    s.queue(.{ .recency_sync = want });
    while (s.recency_sync_done < want) {
        const left = deadline - std.Io.Clock.awake.now(s.io).toNanoseconds();
        if (left <= 0 or s.crashed) return false;
        s.done.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@min(left, 50 * std.time.ns_per_ms)), .clock = .awake } }) catch {};
    }
    return s.recency_sync_failed < want;
}

pub fn dirtyRecency(s: *Store) bool {
    var it = s.points.valueIterator();
    while (it.next()) |pp| if (pp.*.recency_dirty) return true;
    return false;
}

pub fn flushRecencyLocked(s: *Store, force: bool) bool {
    if (s.crashed) return false;
    const awake_ns = std.Io.Clock.awake.now(s.io).toNanoseconds();
    if (!force and awake_ns < s.recency_next) return true;
    s.recency_epoch +%= 1;
    if (s.recency_epoch == 0) {
        var reset = s.points.valueIterator();
        while (reset.next()) |pp| pp.*.recency_epoch = 0;
        s.recency_epoch = 1;
    }
    const cutoff = s.clock;
    const Item = struct { key: Digest, used: i96 };
    var ok = true;
    while (true) {
        var batch: [32]Item = undefined;
        var n: usize = 0;
        var it = s.points.valueIterator();
        while (it.next()) |pp| {
            const p = pp.*;
            if (!p.recency_dirty or p.recency_epoch == s.recency_epoch or p.used > cutoff) continue;
            p.recency_epoch = s.recency_epoch;
            batch[n] = .{ .key = p.key, .used = p.used };
            n += 1;
            if (n == batch.len) break;
        }
        if (n == 0) break;
        var written: [32]bool = @splat(false);
        s.unlockS();
        for (batch[0..n], 0..) |item, i| {
            var name: [80]u8 = undefined;
            fsio.touchDurable(s.points_fd, pointName(&name, &item.key, ".man"), item.used) catch continue;
            written[i] = true;
        }
        s.lockS();
        for (batch[0..n], written[0..n]) |item, success| {
            const p = s.points.get(item.key) orelse continue;
            if (success) {
                s.counters.recency_writes += 1;
                if (p.used == item.used) p.recency_dirty = false;
            } else {
                s.counters.recency_errors += 1;
                ok = false;
            }
        }
    }
    const retry = @max(std.time.ns_per_s, @min(s.cfg.recency_interval_ns, 60 * std.time.ns_per_s));
    const wait = if (ok) s.cfg.recency_interval_ns else retry;
    s.recency_next = std.Io.Clock.awake.now(s.io).toNanoseconds() + wait;
    return ok;
}

pub fn pins(s: *Store, source: Source, from: u32) bool {
    s.lockS();
    defer s.unlockS();
    for (s.fifo.items) |j| {
        if (j.state != .open or j.source.id != source.id) continue;
        for (j.new.items[j.next..]) |idx| if (idx * s.cfg.limits.block_tokens + j.point.blocks[idx].tokens > from) return true;
    }
    return false;
}

pub fn pinsAny(s: *Store, ids: []const u64, out: []bool) void {
    std.debug.assert(out.len == ids.len);
    @memset(out, false);
    if (ids.len == 0) return;
    s.lockS();
    defer s.unlockS();
    for (s.fifo.items) |j| {
        if (j.state != .open) continue;
        const i = std.mem.indexOfScalar(u64, ids, j.source.id) orelse continue;
        if (out[i]) continue;
        for (j.new.items[j.next..]) |idx| if (idx * s.cfg.limits.block_tokens + j.point.blocks[idx].tokens > 0) {
            out[i] = true;
            break;
        };
    }
}
