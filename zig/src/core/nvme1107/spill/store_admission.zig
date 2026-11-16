//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const format = @import("store_types.zig").format;
const mut = @import("store_types.zig").mut;
const history = @import("store_types.zig").history;
const Digest = @import("store_types.zig").Digest;
const Slot = @import("store_types.zig").Slot;
const Keep = @import("store_types.zig").Keep;
const JobId = @import("store_types.zig").JobId;
const KeepResult = @import("store_types.zig").KeepResult;
const StageReq = @import("store_types.zig").StageReq;
const Ticket = @import("store_types.zig").Ticket;
const WriteJob = @import("store_types.zig").WriteJob;
const open = Store.open;
const destroy = Store.destroy;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const now = Store.now;
const chainOf = Store.chainOf;
const makePoint = Store.makePoint;
const ref = Store.ref;
const gc = Store.gc;
const queue = Store.queue;
const remove = Store.remove;
const makeRoom = Store.makeRoom;
const trimOwn = Store.trimOwn;
const release = Store.release;
const settle = Store.settle;
const recheckParked = Store.recheckParked;
const touch = Store.touch;

pub fn keep(s: *Store, k: Keep) KeepResult {
    const gpa = s.gpa;
    const l = s.cfg.limits;
    const at: u32 = @intCast(@min(k.keyed.len, std.math.maxInt(u32)));
    if (at < s.cfg.min_tokens or at == 0) return s.refuse(.short);
    if (!s.validKeep(k)) return s.refuse(.invalid);
    // hashing runs before the lock (the writer and readers never wait on it), one pass over the ids: the chain,
    const ch = chainOf(gpa, k.keyed, l.block_tokens) catch return s.refuse(.no_staging);
    var ch_owned = true;
    defer if (ch_owned) gpa.free(ch);
    const key = format.pointKeyChained(ch, k.keyed, l.block_tokens, k.media);
    if (s.alreadyKept(key, k.shared)) return .stored;
    const reserve = @min(@as(usize, 64 << 20), s.cfg.index_bytes / 4);
    const required = @as(usize, at) * 8 + reserve;
    if (required > s.cfg.index_bytes -| s.metadata.used.load(.monotonic)) return s.refuse(.metadata);
    var lineage = k.lineage.clone(gpa) catch return s.refuse(.no_staging);
    const p = s.makePoint(k.keyed, ch, k.media, lineage, .{
        .key = key,
        .at = at,
        .shared = k.shared,
        .group = k.group,
        .created_ns = 0,
        .has_raw = k.raw != null,
        .mutable_tail = at % l.block_tokens != 0,
        .origin = k.origin,
    }) catch {
        lineage.deinit(gpa);
        return s.refuse(.no_staging);
    };
    ch_owned = false;
    var ok = false;
    defer if (!ok) p.deinit(gpa);
    // sections, laid out on 256-byte boundaries
    const secs = gpa.alloc(format.Section, k.sections.len) catch return s.refuse(.no_staging);
    p.sections = secs;
    var off: u64 = 0;
    for (k.sections, secs) |spec, *sec| {
        off = std.mem.alignForward(u64, off, 256);
        sec.* = .{ .name = format.Name.of(spec.name) catch unreachable, .offset = off, .len = spec.len, .crc = 0 };
        off += spec.len;
    }
    p.small_bytes = off;
    p.ids_bytes = @as(u64, at) * 4 * (if (k.raw != null) @as(u64, 2) else 1);
    const has_tail = at % l.block_tokens != 0;
    const full = l.fullBlockBytes(k.lineage.part).?; // (validKeep: a part where the store needs one)
    for (p.blocks, 0..) |*b, i| b.bytes = if (has_tail and i + 1 == p.blocks.len) k.tail_bytes else full;
    const ids = format.encodeIds(gpa, k.keyed, k.raw) catch return s.refuse(.no_staging);
    var ids_owned = true;
    defer if (ids_owned) gpa.free(ids);
    s.lockS();
    defer s.unlockS();
    s.counters.keeps += 1;
    if (s.points.contains(key) or s.waiting.contains(key)) { // (only the engine thread keeps: never, in practice)
        s.counters.stored_skips += 1;
        return .stored;
    }
    p.created_ns = @intCast(s.now());
    var size: u64 = 0;
    var new: std.ArrayList(u32) = .empty;
    defer if (!ok) new.deinit(gpa);
    var tail: ?u32 = null;
    for (p.blocks, 0..) |b, i| {
        if (s.have.contains(b.name) or s.inflight.contains(b.name)) continue;
        size += b.bytes;
        if (has_tail and i + 1 == p.blocks.len and !mut(.tail_staged_later)) tail = @intCast(i) else new.append(gpa, @intCast(i)) catch return s.refuseOom();
    }
    const need = size + p.small_bytes + p.ids_bytes + history.overhead(at, k.raw != null) + format.header_bytes + 80 * p.blocks.len;
    // GLM 0110's pin: the point holds its references before room is made, so making room never takes a block it
    var pinned = false;
    defer if (!ok and pinned) {
        s.ref(p.blocks, -1);
        s.gc(p.blocks);
    };
    if (!mut(.keep_unpinned)) {
        s.ref(p.blocks, 1);
        pinned = true;
    }
    s.trimOwn(p, need);
    const room = s.floorShort(need);
    const over = (s.stored_bytes + s.queued_bytes + need) -| s.cfg.cap_bytes;
    if (@max(room, over) > s.stored_bytes) {
        s.counters.no_space += 1;
        return .no_space;
    }
    if (room > 0 or over > 0) {
        s.makeRoom(@max(room, over), key);
        if (s.floorShort(need) > 0 or s.stored_bytes + s.queued_bytes + need > s.cfg.cap_bytes) {
            s.counters.no_space += 1;
            return .no_space;
        }
    }
    if (s.queued_bytes + size > s.cfg.backlog_bytes) {
        s.counters.backlog_refused += 1;
        return .backlog;
    }
    const id = s.next_id;
    const small_slot = s.slots.take(.small, id) orelse {
        s.counters.no_staging += 1;
        return .no_staging;
    };
    var tail_slot: ?Slot = null;
    if (tail != null) {
        tail_slot = s.slots.take(.write, id) orelse {
            s.slots.give(small_slot);
            s.counters.no_staging += 1;
            return .no_staging;
        };
    }
    const j = gpa.create(WriteJob) catch {
        s.slots.give(small_slot);
        if (tail_slot) |t| s.slots.give(t);
        return s.refuseOom();
    };
    j.* = .{ .id = id, .point = p, .source = k.source, .new = new, .tail = tail, .tail_slot = tail_slot, .small_slot = small_slot, .size = need, .ids = ids };
    if (!pinned) { // (the keep_unpinned mutant: references taken only now)
        s.ref(p.blocks, 1);
        pinned = true;
    }
    s.registerJob(j) catch {
        gpa.destroy(j);
        s.slots.give(small_slot);
        if (tail_slot) |t| s.slots.give(t);
        return s.refuseOom();
    };
    ok = true;
    ids_owned = false;
    s.next_id += 1;
    s.counters.queued += 1;
    const small = s.slots.bytes(small_slot);
    return .{ .queued = .{
        .job = id,
        .small = small[0..@intCast(p.small_bytes)],
        .sections = p.sections,
        .tail = if (tail_slot) |t| s.slots.bytes(t)[0..@intCast(k.tail_bytes)] else null,
    } };
}

pub fn refuseOom(s: *Store) KeepResult {
    s.counters.no_staging += 1;
    return .no_staging;
}

pub fn refuse(s: *Store, why: std.meta.Tag(KeepResult)) KeepResult {
    s.lockS();
    defer s.unlockS();
    s.counters.keeps += 1;
    switch (why) {
        .short => s.counters.short += 1,
        .invalid => s.counters.invalid += 1,
        .metadata => s.counters.metadata_refused += 1,
        else => s.counters.no_staging += 1,
    }
    return switch (why) {
        .short => .short,
        .invalid => .invalid,
        .metadata => .metadata,
        else => .no_staging,
    };
}

pub fn alreadyKept(s: *Store, key: Digest, shared: bool) bool {
    s.lockS();
    defer s.unlockS();
    if (s.points.get(key)) |p| {
        s.counters.keeps += 1;
        p.used = s.now();
        s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
        if (shared and !p.shared and !mut(.no_upgrade)) {
            p.shared = true;
            s.counters.shared_marked += 1;
            if (!mut(.upgrade_not_durable)) s.queue(.{ .shared = key });
        }
        s.counters.stored_skips += 1;
        return true;
    }
    if (s.waiting.get(key)) |j| {
        s.counters.keeps += 1;
        if (shared and !j.point.shared and !mut(.waiting_not_flagged) and !mut(.no_upgrade)) {
            j.point.shared = true; // its manifest takes the flag (or is rewritten after, if already written)
            s.counters.shared_marked += 1;
        }
        s.counters.stored_skips += 1;
        return true;
    }
    return false;
}

pub fn validKeep(s: *Store, k: Keep) bool {
    const l = s.cfg.limits;
    if (k.keyed.len > l.max_tokens) return false;
    if (k.raw) |r| if (r.len != k.keyed.len) return false;
    if (k.media.len > l.max_media or k.sections.len > l.max_sections or k.lineage.forks.len > l.max_forks) return false;
    if (l.fullBlockBytes(k.lineage.part) == null) return false; // (owner-sharded: every point names its part)
    // pictures: in order, inside the point, the keyed ids made from their digests and the raw ids beside them
    if (!format.mediaIds(k.keyed, k.raw, k.media)) return false;
    var total: u64 = 0;
    for (k.sections) |sec| {
        if (sec.name.len == 0 or sec.name.len > 32) return false;
        total = std.mem.alignForward(u64, total, 256) + sec.len;
    }
    if (total > l.max_small_bytes or total > s.small_slot) return false;
    const has_tail = k.keyed.len % l.block_tokens != 0;
    if (has_tail and (k.tail_bytes == 0 or k.tail_bytes > l.block_bytes)) return false;
    if (!has_tail and k.tail_bytes != 0) return false;
    return true;
}

pub fn floorShort(s: *Store, need: u64) u64 {
    if (mut(.floor_ignored)) return 0;
    const want = s.cfg.floor_bytes + need + s.queued_bytes;
    return want -| (s.disk_free + s.pending_free);
}

pub fn registerJob(s: *Store, j: *WriteJob) !void {
    const gpa = s.gpa;
    try s.jobs_by_id.ensureUnusedCapacity(gpa, 1);
    try s.waiting.ensureUnusedCapacity(gpa, 1);
    try s.fifo.ensureUnusedCapacity(gpa, 1);
    try s.inflight.ensureUnusedCapacity(gpa, @intCast(j.new.items.len + 1));
    s.jobs_by_id.putAssumeCapacity(j.id, j);
    s.waiting.putAssumeCapacity(j.point.key, j);
    for (j.new.items) |i| s.inflight.putAssumeCapacity(j.point.blocks[i].name, j.id);
    if (j.tail) |i| s.inflight.putAssumeCapacity(j.point.blocks[i].name, j.id);
    s.queued_bytes += j.size; // (its references were taken by `keep` before it made room)
    s.fifo.appendAssumeCapacity(j);
}

pub fn captured(s: *Store, job: JobId, ok: bool) void {
    s.lockS();
    defer s.unlockS();
    const j = s.jobs_by_id.get(job) orelse return;
    if (j.capture != .pending) return;
    j.capture = if (ok) .ok else .failed;
    if (!ok and j.state == .open) {
        j.state = .failed;
        j.fail = .capture;
    }
    if (j.tail) |ti| {
        const slot = j.tail_slot.?;
        j.tail_slot = null;
        if (j.state == .open) {
            j.outstanding += 1;
            s.queue(.{ .block = .{ .job = j, .index = ti, .slot = slot } });
        } else s.slots.give(slot);
    }
    s.settle(j);
}

pub fn stage(s: *Store, out: []StageReq) usize {
    s.lockS();
    defer s.unlockS();
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.fifo.items.len and n < out.len) {
        const j = s.fifo.items[i];
        if (j.state != .open) {
            i += 1;
            continue;
        }
        while (j.next < j.new.items.len and n < out.len) {
            const slot = s.slots.take(.write, j.id) orelse return n;
            out[n] = s.req(j, slot);
            n += 1;
        }
        if (j.next >= j.new.items.len) {
            j.listed = false;
            _ = s.fifo.orderedRemove(i);
        } else i += 1;
    }
    return n;
}

pub fn req(s: *Store, j: *WriteJob, slot: Slot) StageReq {
    const idx = j.new.items[j.next];
    j.next += 1;
    j.outstanding += 1;
    j.engine += 1;
    s.counters.staged_blocks += 1;
    const b = j.point.blocks[idx];
    return .{
        .ticket = .{ .job = j.id, .index = idx, .slot = slot },
        .source = j.source,
        .first = idx * s.cfg.limits.block_tokens,
        .tokens = b.tokens,
        .dst = s.slots.bytes(slot)[0..@intCast(b.bytes)],
    };
}

pub fn staged(s: *Store, t: Ticket, ok: bool) void {
    s.lockS();
    defer s.unlockS();
    const j = s.jobs_by_id.get(t.job) orelse {
        s.slots.give(t.slot);
        return;
    };
    j.engine -= 1;
    if (!ok and j.state == .open) {
        j.state = .failed;
        j.fail = .stage;
        s.release(j);
    }
    if (j.state == .open or (ok and s.stillNamed(j, t.index))) {
        s.queue(.{ .block = .{ .job = j, .index = t.index, .slot = t.slot } });
    } else {
        s.slots.give(t.slot);
        j.outstanding -= 1;
        s.dropInflight(j, t.index);
        s.settle(j);
    }
}

pub fn stillNamed(s: *Store, j: *WriteJob, index: u32) bool {
    return (s.refs.get(j.point.blocks[index].name) orelse 0) > 1;
}

pub fn dropInflight(s: *Store, j: *WriteJob, index: u32) void {
    const name = j.point.blocks[index].name;
    if (s.inflight.get(name)) |owner| if (owner == j.id) {
        _ = s.inflight.remove(name);
        s.recheckParked();
    };
}
