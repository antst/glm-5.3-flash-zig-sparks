//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const c = @import("store_types.zig").c;
const format = @import("store_types.zig").format;
const fsio = @import("store_types.zig").fsio;
const slots_mod = @import("store_types.zig").slots_mod;
const mut = @import("store_types.zig").mut;
const Digest = @import("store_types.zig").Digest;
const Media = @import("store_types.zig").Media;
const Slot = @import("store_types.zig").Slot;
const ReadId = @import("store_types.zig").ReadId;
const Query = @import("store_types.zig").Query;
const Hit = @import("store_types.zig").Hit;
const Point = @import("store_types.zig").Point;
const ReadJob = @import("store_types.zig").ReadJob;
const close = Store.close;
const destroy = Store.destroy;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const now = Store.now;
const pointName = Store.pointName;
const blockName = Store.blockName;
const queue = Store.queue;
const pins = Store.pins;
const crash = Store.crash;
const readFailed = Store.readFailed;
const maybeFreeRead = Store.maybeFreeRead;

pub fn manifestOf(p: *const Point, shared: bool, block_tokens: u32, layout: format.OwnerLayout) format.Manifest {
    return .{
        .key = p.key,
        .at = p.at,
        .block_tokens = block_tokens,
        .owner_layout = layout,
        .flags = .{ .shared = shared, .has_raw = p.has_raw, .mutable_tail = p.mutable_tail },
        .group = p.group,
        .created_ns = p.created_ns,
        .lineage = p.lineage,
        .origin = p.origin,
        .ids_bytes = p.ids_bytes,
        .ids_crc = p.ids_crc,
        .small_bytes = p.small_bytes,
        .small_crc = p.small_crc,
        .media = p.media,
        .blocks = p.blocks,
        .sections = p.sections,
    };
}

pub fn rewriteShared(s: *Store, key: Digest) void {
    const gpa = s.gpa;
    const p = s.points.get(key) orelse return;
    var m = manifestOf(p, p.shared, s.cfg.limits.block_tokens, s.cfg.owner_layout);
    const raw = m.encode(gpa, s.cfg.compat) catch return;
    defer gpa.free(raw);
    var nb: [80]u8 = undefined;
    const name = pointName(&nb, &key, ".man");
    s.unlockS();
    var crashed = false;
    const ok = blk: {
        if (mut(.rewrite_not_put)) {
            const fd = c.openat(s.points_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0));
            if (fd < 0) break :blk false;
            fsio.pwriteAll(&s.hooks, fd, raw, 0) catch {};
            _ = c.close(fd);
        } else _ = s.writer.put(s.points_fd, name, .{ .parts = &.{raw} }) catch |e| {
            crashed = e == error.Crashed;
            break :blk false;
        };
        if (!mut(.rewrite_no_dir_fsync)) fsio.syncDir(&s.hooks, s.points_fd, "points") catch break :blk false;
        if (s.crash(.rewritten, name)) {
            crashed = true;
            break :blk false;
        }
        break :blk true;
    };
    s.lockS();
    if (crashed) {
        s.crashed = true;
        return;
    }
    if (ok) s.counters.shared_rewrites += 1;
    if (ok and s.points.get(key) == p) s.queue(.{ .touch = .{ .key = key, .ns = p.used } }); // its last use, kept
    if (s.points.get(key) != p and !mut(.rewrite_resurrects)) {
        s.unlockS();
        fsio.unlinkAt(s.points_fd, name);
        s.lockS();
    }
}

pub fn lookup(s: *Store, q: Query) ?Hit {
    const gpa = s.gpa;
    const block = s.cfg.limits.block_tokens;
    var need: usize = 0;
    {
        s.lockS();
        defer s.unlockS();
        s.counters.disk_lookups += 1;
        var it = s.points.valueIterator();
        while (it.next()) |pp| if (eligible(pp.*, q)) {
            need = @max(need, pp.*.chain.len);
        };
    }
    const chain = gpa.alloc(Digest, need) catch return null;
    defer gpa.free(chain);
    var prev: Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash("tf-spill/ids", &prev, .{});
    for (chain, 0..) |*d, i| {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(&prev);
        h.update(std.mem.sliceAsBytes(q.keyed[i * block .. (i + 1) * block]));
        h.final(&prev);
        d.* = prev;
    }
    s.lockS();
    defer s.unlockS();
    var best: ?*Point = null;
    var it = s.points.valueIterator();
    while (it.next()) |pp| {
        const p = pp.*;
        if (!eligible(p, q) or p.chain.len > chain.len) continue;
        if (best != null and (p.at < best.?.at or (p.at == best.?.at and p.used <= best.?.used))) continue;
        var same = true;
        for (p.chain, chain[0..p.chain.len]) |a, b| if (!std.mem.eql(u8, &a, &b)) {
            same = false;
            break;
        };
        if (!same) continue;
        const whole = p.chain.len * block;
        if (!std.mem.eql(u32, p.tail, q.keyed[whole..p.at])) continue;
        if (!mediaMatch(p, q.media)) continue;
        if (!s.complete_(p)) continue;
        best = p;
    }
    const b = best orelse return null;
    s.counters.disk_hits += 1;
    return .{ .key = b.key, .at = b.at, .shared = b.shared, .group = b.group, .part = b.lineage.part };
}

pub fn eligible(p: *const Point, q: Query) bool {
    if (p.at <= q.longer_than or p.at > q.max_at or p.at > q.keyed.len) return false;
    return q.grid == 0 or p.at % q.grid == 0;
}

pub fn mediaMatch(p: *const Point, media: []const Media) bool {
    var i: usize = 0;
    for (media) |m| {
        if (m.start >= p.at) continue;
        if (i >= p.media.len or !p.media[i].eql(m)) return false;
        i += 1;
    }
    return i == p.media.len;
}

pub fn complete_(s: *Store, p: *const Point) bool {
    for (p.blocks) |b| if (!s.have.contains(b.name)) return false;
    return true;
}

pub fn presence(s: *Store, key: Digest) bool {
    s.lockS();
    defer s.unlockS();
    const p = s.points.get(key) orelse return false;
    return s.complete_(p);
}

pub fn partOf(s: *Store, key: Digest) ?struct { part: ?u32 } {
    s.lockS();
    defer s.unlockS();
    const p = s.points.get(key) orelse return null;
    return .{ .part = p.lineage.part };
}

pub fn touch(s: *Store, key: Digest) void {
    s.lockS();
    defer s.unlockS();
    const p = s.points.get(key) orelse return;
    p.used = s.now();
    s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
}

pub fn markShared(s: *Store, key: Digest) void {
    s.lockS();
    defer s.unlockS();
    if (s.points.get(key)) |p| {
        if (!p.shared) {
            p.shared = true;
            s.counters.shared_marked += 1;
            s.queue(.{ .shared = key });
        }
    } else if (s.waiting.get(key)) |j| j.point.shared = true;
}

pub const BeginError = error{ Missing, NoSlot, TooMany, OutOfMemory };

pub fn beginRead(s: *Store, key: Digest) BeginError!ReadId {
    const gpa = s.gpa;
    s.lockS();
    defer s.unlockS();
    const p = s.points.get(key) orelse return error.Missing;
    if (!s.complete_(p)) return error.Missing;
    if (s.liveReads() >= s.cfg.max_reads) {
        s.counters.read_deferred += 1;
        return error.TooMany;
    }
    const id = s.next_id;
    const small_slot = s.slots.take(.restore, id) orelse {
        s.counters.read_deferred += 1;
        return error.NoSlot;
    };
    errdefer s.slots.give(small_slot);
    const r = try gpa.create(ReadJob);
    errdefer gpa.destroy(r);
    const n: u32 = @intCast(p.blocks.len);
    r.* = .{ .id = id, .point = p, .total = n, .state = try gpa.alloc(ReadJob.BState, n), .slot = undefined, .small_slot = small_slot, .t0 = std.Io.Clock.awake.now(s.io).toNanoseconds() };
    errdefer gpa.free(r.state);
    r.slot = try gpa.alloc(Slot, n);
    errdefer gpa.free(r.slot);
    @memset(r.state, .idle);
    try s.reads.append(gpa, r);
    errdefer _ = s.reads.pop();
    const gop = try s.reading.getOrPut(gpa, key);
    gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
    p.pins += 1;
    s.next_id += 1;
    p.used = s.now();
    s.queue(.{ .touch = .{ .key = key, .ns = p.used } });
    r.disk += 1;
    s.rtasks.append(gpa, .{ .job = r, .index = null, .slot = small_slot }) catch {
        r.disk -= 1;
        r.small = .failed;
        r.err = .io;
    };
    s.assign();
    s.rwork.broadcast(s.io);
    return id;
}

pub fn readJob(s: *Store, id: ReadId) ?*ReadJob {
    for (s.reads.items) |r| if (r.id == id and !r.ended) return r;
    return null;
}

pub fn readsFull(s: *Store) bool {
    s.lockS();
    defer s.unlockS();
    return s.liveReads() >= s.cfg.max_reads or s.slots.available(.restore) == 0;
}

pub fn liveReads(s: *Store) usize {
    var n: usize = 0;
    for (s.reads.items) |r| n += @intFromBool(!r.ended);
    return n;
}

pub fn assign(s: *Store) void {
    const n = s.reads.items.len;
    while (s.slots.available(.read) > 0 and n > 0) {
        var pick: ?*ReadJob = null;
        for (0..n) |k| {
            const r = s.reads.items[(s.read_turn + k) % n];
            if (r.ended or r.err != null or r.next_assign >= r.total) continue;
            if (pick == null or r.held < pick.?.held) pick = r;
        }
        const r = pick orelse break;
        s.read_turn +%= 1;
        const slot = s.slots.take(.read, r.id) orelse break;
        const i = r.next_assign;
        r.next_assign += 1;
        r.state[i] = .reading;
        r.slot[i] = slot;
        r.disk += 1;
        r.held += 1;
        s.rtasks.append(s.gpa, .{ .job = r, .index = i, .slot = slot }) catch {
            r.disk -= 1;
            r.held -= 1;
            r.state[i] = .failed;
            r.err = .io;
            s.slots.give(slot);
            continue;
        };
    }
    s.rwork.broadcast(s.io);
}

pub fn giveRead(s: *Store, r: *ReadJob, slot: Slot) void {
    s.slots.give(slot);
    r.held -= 1;
}

pub fn readerLoop(s: *Store) void {
    s.lockS();
    defer s.unlockS();
    while (true) {
        while (s.rtask_head >= s.rtasks.items.len and !s.stopping) s.rwork.waitUncancelable(s.io, &s.mutex);
        if (s.stopping) break;
        const t = s.rtasks.items[s.rtask_head];
        s.rtask_head += 1;
        if (s.rtask_head == s.rtasks.items.len) {
            s.rtasks.clearRetainingCapacity();
            s.rtask_head = 0;
        }
        const r = t.job;
        if (r.ended or (r.err != null and t.index != null)) {
            if (t.index) |i| {
                s.giveRead(r, t.slot);
                r.state[i] = .failed;
            } else if (r.ended) s.slots.give(t.slot); // the small slot: its reader gives it back once ended
            r.disk -= 1;
            s.maybeFreeRead(r);
            s.assign();
            continue;
        }
        if (t.index) |i| {
            const b = r.point.blocks[i];
            const buf = s.slots.bytes(t.slot);
            s.unlockS();
            const got = s.readBlock(b, buf);
            s.lockS();
            s.counters.read_block_calls += 1;
            r.disk -= 1;
            if (r.ended) {
                s.giveRead(r, t.slot);
                r.state[i] = .failed;
            } else if (got) |_| {
                r.state[i] = .ready;
            } else |e| {
                r.state[i] = .failed;
                s.giveRead(r, t.slot); // (never handed: its slot comes back now, or the job could never end)
                s.readFailed(r, b.name, e);
            }
        } else {
            const buf = s.slots.bytes(t.slot);
            const p = r.point;
            s.unlockS();
            const ok = s.readSmall(p, buf);
            s.lockS();
            s.counters.read_small_calls += 1;
            r.disk -= 1;
            if (r.ended) {
                s.slots.give(t.slot);
            } else if (ok) r.small = .ready else {
                r.small = .failed;
                if (r.err == null) r.err = .small;
                s.counters.crc_errors += 1;
            }
        }
        if (r.ended) {
            s.maybeFreeRead(r);
            s.assign();
        }
        s.done.broadcast(s.io);
    }
}

pub const BlockReadError = error{ Missing, Damaged, Io };

pub fn readBlock(s: *Store, b: format.BlockRef, buf: slots_mod.Buffer) BlockReadError!void {
    var nb: [80]u8 = undefined;
    var f = fsio.openRead(s.blocks_fd, blockName(&nb, &b.name), s.cfg.direct and !s.hooks.no_direct) catch |e| return if (e == error.Missing) error.Missing else error.Io;
    defer f.close();
    const st = fsio.statFd(f.fd) catch return error.Io;
    if (st.size != b.bytes) return error.Damaged;
    const len: usize = @intCast(b.bytes);
    const got = f.read(buf, 0, len) catch |e| return if (e == error.EndOfFile) error.Damaged else error.Io;
    std.debug.assert(got.ptr == buf.ptr);
    if (!f.direct) fsio.dropPages(&s.hooks, f.fd, "block");
    if (format.crc32(buf[0..len]) != b.crc) return error.Damaged;
}

pub fn readSmall(s: *Store, p: *const Point, buf: slots_mod.Buffer) bool {
    var nb: [80]u8 = undefined;
    var f = fsio.openRead(s.points_fd, pointName(&nb, &p.key, ".st"), s.cfg.direct and !s.hooks.no_direct) catch return false;
    defer f.close();
    const st = fsio.statFd(f.fd) catch return false;
    if (st.size != p.small_bytes) return false;
    const len: usize = @intCast(p.small_bytes);
    if (len == 0) return p.small_crc == format.crc32(&.{});
    _ = f.read(buf, 0, len) catch return false;
    if (!f.direct) fsio.dropPages(&s.hooks, f.fd, "small");
    if (format.crc32(buf[0..len]) != p.small_crc) return false;
    for (p.sections) |sec| if (format.crc32(buf[@intCast(sec.offset)..][0..@intCast(sec.len)]) != sec.crc) return false;
    return true;
}
