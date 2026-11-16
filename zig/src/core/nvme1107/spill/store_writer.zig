//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const format = @import("store_types.zig").format;
const fsio = @import("store_types.zig").fsio;
const mut = @import("store_types.zig").mut;
const history = @import("store_types.zig").history;
const Digest = @import("store_types.zig").Digest;
const Slot = @import("store_types.zig").Slot;
const Source = @import("store_types.zig").Source;
const WriteFailure = @import("store_types.zig").WriteFailure;
const WriteJob = @import("store_types.zig").WriteJob;
const Task = @import("store_types.zig").Task;
const open = Store.open;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const now = Store.now;
const pointName = Store.pointName;
const blockName = Store.blockName;
const addBlock = Store.addBlock;
const ref = Store.ref;
const queue = Store.queue;
const adopt = Store.adopt;
const trimPath = Store.trimPath;
const stillNamed = Store.stillNamed;
const dropInflight = Store.dropInflight;
const release = Store.release;
const settle = Store.settle;
const endJob = Store.endJob;
const blocksSettled = Store.blocksSettled;
const dirtyRecency = Store.dirtyRecency;
const flushRecencyLocked = Store.flushRecencyLocked;
const manifestOf = Store.manifestOf;
const rewriteShared = Store.rewriteShared;
const touch = Store.touch;

pub fn writerLoop(s: *Store) void {
    s.lockS();
    defer s.unlockS();
    while (true) {
        _ = s.flushRecencyLocked(false);
        while (s.task_head >= s.tasks.items.len and !s.stopping) {
            if (!s.dirtyRecency()) {
                s.work.waitUncancelable(s.io, &s.mutex);
            } else {
                const left = @max(1, s.recency_next - std.Io.Clock.awake.now(s.io).toNanoseconds());
                s.work.waitTimeout(s.io, &s.mutex, .{ .duration = .{ .raw = .fromNanoseconds(left), .clock = .awake } }) catch {};
            }
            _ = s.flushRecencyLocked(false);
        }
        if (s.task_head >= s.tasks.items.len) {
            _ = s.flushRecencyLocked(true);
            break;
        }
        const t = s.tasks.items[s.task_head];
        s.task_head += 1;
        if (s.task_head == s.tasks.items.len) {
            s.tasks.clearRetainingCapacity();
            s.task_head = 0;
        }
        if (s.stopping) {
            switch (t) {
                .block, .point => {
                    s.dropTask(t);
                    continue;
                },
                else => {},
            }
        }
        if (s.crashed) {
            s.dropTask(t);
            continue;
        }
        s.busy += 1;
        s.runTask(t);
        s.busy -= 1;
        s.done.broadcast(s.io);
    }
}

pub fn dropTask(s: *Store, t: Task) void {
    switch (t) {
        .block => |b| {
            s.slots.give(b.slot);
            b.job.outstanding -= 1;
            if (b.job.state == .open) {
                b.job.state = .failed;
                b.job.fail = .crashed;
            }
            s.dropInflight(b.job, b.index);
            if (!s.crashed) s.settle(b.job);
        },
        .point => |j| if (!s.crashed) {
            j.state = .failed;
            j.fail = .crashed;
            s.endJob(j, false);
        },
        .sync => |g| s.sync_done = @max(s.sync_done, g),
        .recency_sync => |g| {
            s.recency_sync_done = @max(s.recency_sync_done, g);
            s.recency_sync_failed = @max(s.recency_sync_failed, g);
        },
        else => {},
    }
    s.done.broadcast(s.io);
}

pub fn runTask(s: *Store, t: Task) void {
    switch (t) {
        .block => |b| s.writeBlock(b.job, b.index, b.slot),
        .point => |j| s.writePoint(j),
        .shared => |k| s.rewriteShared(k),
        .touch => |x| {
            var nb: [80]u8 = undefined;
            const name = pointName(&nb, &x.key, ".man");
            s.unlockS();
            fsio.touchAt(s.points_fd, name, x.ns);
            s.lockS();
        },
        .unlink_block => |x| {
            var nb: [80]u8 = undefined;
            const name = blockName(&nb, &x.name);
            s.unlockS();
            fsio.unlinkAt(s.blocks_fd, name);
            const free = fsio.diskFree(&s.hooks, s.blocks_fd);
            s.lockS();
            s.disk_free = free;
            s.pending_free -|= x.bytes;
        },
        .unlink_point => |x| {
            s.unlockS();
            s.unlinkPointFiles(x.key);
            s.histories.?.release(x.roots) catch {};
            s.histories.?.collect() catch {};
            const free = fsio.diskFree(&s.hooks, s.blocks_fd);
            s.lockS();
            s.disk_free = free;
            s.pending_free -|= x.bytes;
            s.refreshHistoryBytes();
        },
        .sync => |g| s.sync_done = @max(s.sync_done, g),
        .recency_sync => |g| {
            if (!s.flushRecencyLocked(true)) s.recency_sync_failed = @max(s.recency_sync_failed, g);
            s.recency_sync_done = @max(s.recency_sync_done, g);
        },
    }
}

pub fn refreshHistoryBytes(s: *Store) void {
    const current = s.histories.?.bytes;
    s.stored_bytes = (s.stored_bytes -| s.history_bytes) + current;
    s.history_bytes = current;
}

pub fn unlinkPointFiles(s: *Store, key: Digest) void {
    var nb: [80]u8 = undefined;
    fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".man")); // the manifest first: the point is gone at once
    fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".st"));
    fsio.unlinkAt(s.points_fd, pointName(&nb, &key, ".ids"));
}

pub fn crash(s: *Store, step: fsio.Step, name: []const u8) bool {
    if (!fsio.crashAt(&s.hooks, step, name)) return false;
    s.lockS();
    s.crashed = true;
    s.unlockS();
    return true;
}

pub fn writeBlock(s: *Store, j: *WriteJob, index: u32, slot: Slot) void {
    if (j.state != .open and !s.stillNamed(j, index)) { // cancelled meanwhile and nobody else names it
        s.slots.give(slot);
        j.outstanding -= 1;
        s.dropInflight(j, index);
        s.settle(j);
        return;
    }
    const b = j.point.blocks[index];
    const data = s.slots.bytes(slot);
    const len: usize = @intCast(b.bytes);
    s.unlockS();
    const crc = format.crc32(data[0..len]);
    const sha = format.sha256(data[0..len]);
    s.lockS();
    var same: ?Digest = null;
    if (s.content.get(sha)) |other| {
        if (s.have.get(other)) |info| if (info.crc == crc and info.bytes == b.bytes) {
            same = other;
        };
    }
    if (same == null and mut(.dedup_without_sha)) {
        var hit = s.have.iterator();
        while (hit.next()) |e| if (e.value_ptr.*.crc == crc and e.value_ptr.*.bytes == b.bytes) {
            same = e.key_ptr.*;
            break;
        };
    }
    s.unlockS();
    var nb: [80]u8 = undefined;
    const name = blockName(&nb, &b.name);
    var linked = false;
    var failed: ?WriteFailure = null;
    var crashed = false;
    if (same) |other| {
        var ob: [80]u8 = undefined;
        var tb: [160]u8 = undefined;
        const tmp = s.writer.tmpName(&tb, name);
        if (fsio.linkAt(s.blocks_fd, blockName(&ob, &other), tmp)) {
            if (s.crash(.linked, name)) {
                crashed = true;
            } else if (fsio.renameAt(&s.hooks, s.blocks_fd, tmp, name)) {
                linked = true;
            } else |_| fsio.unlinkAt(s.blocks_fd, tmp);
        } else |_| {} // gone meanwhile: written below instead
    }
    if (!linked and !crashed) {
        _ = s.writer.put(s.blocks_fd, name, .{ .aligned = .{ .buf = data, .len = len } }) catch |e| {
            if (e == error.Crashed) {
                s.lockS();
                s.crashed = true;
                s.unlockS();
                crashed = true;
            } else failed = if (e == error.NoSpace) .no_space else .io;
        };
    }
    var ino: u64 = 0;
    if (failed == null and !crashed) {
        if (fsio.statAt(s.blocks_fd, name)) |st| ino = st.ino else |_| failed = .io;
    }
    const free = fsio.diskFree(&s.hooks, s.blocks_fd);
    s.lockS();
    s.disk_free = free;
    s.counters.direct_refused = s.writer.direct_refused.load(.monotonic);
    s.slots.give(slot);
    j.outstanding -= 1;
    if (crashed) return;
    if (failed) |why| {
        if (j.state == .open) {
            j.state = .failed;
            j.fail = why;
            s.release(j);
        }
    } else {
        s.addBlock(b.name, crc, sha, ino, b.bytes) catch {};
        if (linked) {
            s.counters.dedup_blocks += 1;
            s.counters.dedup_bytes += b.bytes;
        } else {
            s.counters.written_blocks += 1;
            s.counters.written_bytes += b.bytes;
        }
    }
    s.dropInflight(j, index);
    s.settle(j);
}

pub fn writePoint(s: *Store, j: *WriteJob) void {
    const gpa = s.gpa;
    const p = j.point;
    if (j.state != .publishing) {
        s.settle(j);
        return;
    }
    if (!s.blocksSettled(j)) { // a block another write owns is still on its way
        j.parked = true;
        s.parked.append(gpa, j) catch {};
        return;
    }
    for (p.blocks) |*b| {
        const info = s.have.get(b.name) orelse return s.failPoint(j, .missing_block);
        b.crc = info.crc;
        b.sha = info.sha;
    }
    const shared = p.shared;
    const small = s.slots.bytes(j.small_slot);
    s.unlockS();
    var why: ?WriteFailure = null;
    var crashed = false;
    var nb: [80]u8 = undefined;
    var nb2: [80]u8 = undefined;
    var nb3: [80]u8 = undefined;
    var man_bytes: u64 = 0;
    var ids_encoded: ?history.Encoded = null;
    defer if (ids_encoded) |v| gpa.free(v.bytes);
    p.ids_crc = format.crc32(j.ids);
    for (p.sections) |*sec| sec.crc = format.crc32(small[@intCast(sec.offset)..][0..@intCast(sec.len)]);
    p.small_crc = format.crc32(small[0..@intCast(p.small_bytes)]);
    publish: {
        ids_encoded = s.histories.?.encode(&s.writer, j.ids, p.at, p.has_raw) catch |e| {
            if (e == error.Crashed) crashed = true else why = if (e == error.NoSpace) .no_space else .io;
            break :publish;
        };
        fsio.syncDir(&s.hooks, s.blocks_fd, "blocks") catch {
            why = .io;
            break :publish;
        };
        if (s.crash(.dir_synced, "blocks")) {
            crashed = true;
            break :publish;
        }
        var m = manifestOf(p, shared, s.cfg.limits.block_tokens, s.cfg.owner_layout);
        const raw = m.encode(gpa, s.cfg.compat) catch {
            why = .io;
            break :publish;
        };
        defer gpa.free(raw);
        man_bytes = raw.len;
        const files = [_]struct { name: [:0]const u8, src: fsio.Source }{
            .{ .name = pointName(&nb, &p.key, ".ids"), .src = .{ .parts = &.{ids_encoded.?.bytes} } },
            .{ .name = pointName(&nb2, &p.key, ".st"), .src = .{ .aligned = .{ .buf = small, .len = @intCast(p.small_bytes) } } },
            .{ .name = pointName(&nb3, &p.key, ".man"), .src = .{ .parts = &.{raw} } },
        };
        const order: [3]usize = if (mut(.manifest_not_last)) .{ 2, 0, 1 } else .{ 0, 1, 2 };
        for (order) |fi| {
            if (fi == 2) { // the manifest last: the files it names durable first
                fsio.syncDir(&s.hooks, s.points_fd, "points") catch {
                    why = .io;
                    break :publish;
                };
                if (s.crash(.dir_synced, "points")) {
                    crashed = true;
                    break :publish;
                }
            }
            _ = s.writer.put(s.points_fd, files[fi].name, files[fi].src) catch |e| {
                if (e == error.Crashed) crashed = true else why = if (e == error.NoSpace) .no_space else .io;
                break :publish;
            };
        }
        fsio.syncDir(&s.hooks, s.points_fd, "points") catch {
            why = .io;
            break :publish;
        };
        if (s.crash(.published, "points")) {
            crashed = true;
            break :publish;
        }
    }
    const free = fsio.diskFree(&s.hooks, s.blocks_fd);
    s.lockS();
    s.disk_free = free;
    if (crashed) {
        s.crashed = true;
        return;
    }
    if (why) |w| {
        s.unlockS();
        s.unlinkPointFiles(p.key);
        s.histories.?.collect() catch {};
        s.lockS();
        s.refreshHistoryBytes();
        return s.failPoint(j, w);
    }
    for (p.blocks) |b| if (!s.have.contains(b.name)) {
        s.queue(.{ .unlink_point = .{ .key = p.key, .bytes = 0 } }); // a block it names was dropped while it was written
        return s.failPoint(j, .missing_block);
    };
    p.ids_disk_bytes = ids_encoded.?.bytes.len;
    p.history_roots = ids_encoded.?.roots;
    p.files = man_bytes + p.ids_disk_bytes + p.small_bytes;
    p.used = s.now();
    s.points.put(gpa, p.key, p) catch {
        s.queue(.{ .unlink_point = .{ .key = p.key, .bytes = 0 } });
        return s.failPoint(j, .io);
    };
    s.ref(p.blocks, 1); // its own reference; the waiting one goes in endJob
    s.adopt(p);
    s.histories.?.pin(p.history_roots);
    s.stored_bytes += p.files;
    s.refreshHistoryBytes();
    s.counters.committed += 1;
    s.counters.written_small_bytes += p.small_bytes;
    s.endJob(j, true);
    if (p.shared and !shared and !mut(.written_no_rewrite)) s.queue(.{ .shared = p.key }); // made shared while it was written
    s.trimPath(p);
}

pub fn failPoint(s: *Store, j: *WriteJob, why: WriteFailure) void {
    j.state = .failed;
    j.fail = why;
    s.endJob(j, false);
}
