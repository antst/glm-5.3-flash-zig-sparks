//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const format = @import("store_types.zig").format;
const fsio = @import("store_types.zig").fsio;
const mut = @import("store_types.zig").mut;
const history = @import("store_types.zig").history;
const Allocator = @import("store_types.zig").Allocator;
const Digest = @import("store_types.zig").Digest;
const Lineage = @import("store_types.zig").Lineage;
const Media = @import("store_types.zig").Media;
const Completion = @import("store_types.zig").Completion;
const Point = @import("store_types.zig").Point;
const Task = @import("store_types.zig").Task;
const destroy = Store.destroy;
const lockS = Store.lockS;
const unlockS = Store.unlockS;
const pointName = Store.pointName;
const victims = Store.victims;
const pins = Store.pins;
const touch = Store.touch;
const readSmall = Store.readSmall;

pub fn recover(s: *Store, key: Digest, present: *const std.AutoHashMapUnmanaged(Digest, fsio.Stat), unsure: *const std.AutoHashMapUnmanaged(Digest, void)) !*Point {
    const gpa = s.gpa;
    const l = s.cfg.limits;
    var nb: [80]u8 = undefined;
    const raw = try fsio.readSmall(gpa, s.points_fd, pointName(&nb, &key, ".man"), l.maxManifest());
    defer gpa.free(raw);
    var m = try format.Manifest.parse(gpa, raw, s.cfg.compat, l);
    defer {
        gpa.free(m.media);
        gpa.free(m.blocks);
        gpa.free(m.sections);
    }
    var lineage_owned = true;
    defer if (lineage_owned) gpa.free(m.lineage.forks);
    if (!std.mem.eql(u8, &m.key, &key) or m.owner_layout != s.cfg.owner_layout) return error.Corrupt;
    const man_st = try fsio.statAt(s.points_fd, pointName(&nb, &key, ".man"));
    const ids_file = try fsio.readSmall(gpa, s.points_fd, pointName(&nb, &key, ".ids"), history.fileBytes(m.at, m.flags.has_raw));
    defer gpa.free(ids_file);
    const ids_raw = try s.histories.?.decode(ids_file, m.at, m.flags.has_raw);
    defer gpa.free(ids_raw);
    if (ids_raw.len != m.ids_bytes or format.crc32(ids_raw) != m.ids_crc) return error.Corrupt;
    const small_st = try fsio.statAt(s.points_fd, pointName(&nb, &key, ".st"));
    if (small_st.size != m.small_bytes) return error.Corrupt;
    const keyed = try gpa.alloc(u32, m.at);
    defer gpa.free(keyed);
    @memcpy(std.mem.sliceAsBytes(keyed), ids_raw[0 .. m.at * 4]);
    const ch = try chainOf(gpa, keyed, l.block_tokens);
    var ch_owned = true;
    defer if (ch_owned) gpa.free(ch);
    if (!std.mem.eql(u8, &format.pointKeyChained(ch, keyed, l.block_tokens, m.media), &key)) return error.Corrupt;
    if (m.media.len > 0) { // a point with pictures: its keyed and raw ids agree with its media (version 3)
        const raw_ids = try gpa.alloc(u32, m.at);
        defer gpa.free(raw_ids);
        @memcpy(std.mem.sliceAsBytes(raw_ids), ids_raw[m.at * 4 ..][0 .. m.at * 4]);
        if (!format.mediaIds(keyed, raw_ids, m.media)) return error.Corrupt;
    }
    const spans = try gpa.alloc(format.Span, m.blocks.len);
    defer gpa.free(spans);
    format.blockNames(m.lineage, ch, keyed, l.block_tokens, spans);
    for (m.blocks, spans) |b, sp| {
        if (!std.mem.eql(u8, &b.name, &sp.name)) return error.Corrupt;
        const st = present.get(b.name) orelse return if (unsure.contains(b.name)) error.Io else error.Missing;
        if (st.size != b.bytes and !mut(.recover_no_size_check)) return error.Corrupt;
    }
    const p = try s.makePoint(keyed, ch, m.media, m.lineage, .{
        .key = key,
        .at = m.at,
        .shared = m.flags.shared,
        .group = m.group,
        .created_ns = m.created_ns,
        .has_raw = m.flags.has_raw,
        .mutable_tail = m.flags.mutable_tail,
        .origin = m.origin,
    });
    lineage_owned = false; // the point owns them now
    ch_owned = false;
    errdefer p.deinit(gpa);
    @memcpy(p.blocks, m.blocks);
    gpa.free(p.sections);
    p.sections = try gpa.dupe(format.Section, m.sections);
    p.history_roots = try history.History.parse(ids_file, m.at, m.flags.has_raw);
    s.histories.?.pin(p.history_roots);
    p.ids_disk_bytes = ids_file.len;
    p.ids_bytes = m.ids_bytes;
    p.ids_crc = m.ids_crc;
    p.small_bytes = m.small_bytes;
    p.small_crc = m.small_crc;
    p.used = man_st.mtime_ns;
    p.files = man_st.size + p.ids_disk_bytes + m.small_bytes;
    return p;
}

pub const PointInit = struct {
    key: Digest,
    at: u32,
    shared: bool,
    group: u64,
    created_ns: i64,
    has_raw: bool,
    mutable_tail: bool,
    origin: ?format.Origin,
};

pub fn chainOf(gpa: Allocator, keyed: []const u32, block: u32) ![]Digest {
    var ch: std.ArrayList(Digest) = .empty;
    errdefer ch.deinit(gpa);
    try format.chain(gpa, keyed, block, &ch);
    return ch.toOwnedSlice(gpa);
}

pub fn makePoint(s: *Store, keyed: []const u32, ch: []Digest, media: []const Media, lineage: Lineage, i: PointInit) !*Point {
    const gpa = s.gpa;
    const block = s.cfg.limits.block_tokens;
    const p = try gpa.create(Point);
    errdefer gpa.destroy(p);
    const whole = ch.len * block;
    const tail = try gpa.dupe(u32, keyed[whole..]);
    errdefer gpa.free(tail);
    const md = try gpa.dupe(Media, media);
    errdefer gpa.free(md);
    const nblocks = std.math.divCeil(usize, keyed.len, block) catch unreachable;
    const spans = try gpa.alloc(format.Span, nblocks);
    defer gpa.free(spans);
    format.blockNames(lineage, ch, keyed, block, spans);
    const blocks = try gpa.alloc(format.BlockRef, nblocks);
    errdefer gpa.free(blocks);
    for (blocks, spans) |*b, sp| b.* = .{ .name = sp.name, .tokens = sp.end - sp.start, .bytes = 0, .crc = 0, .sha = format.zero_digest };
    const chain = try s.chain_prefixes.take(ch);
    p.* = .{
        .key = i.key,
        .at = i.at,
        .shared = i.shared,
        .group = i.group,
        .used = 0,
        .created_ns = i.created_ns,
        .has_raw = i.has_raw,
        .mutable_tail = i.mutable_tail,
        .lineage = lineage,
        .origin = i.origin,
        .chain = chain.items,
        .chain_lease = chain,
        .tail = tail,
        .media = md,
        .blocks = blocks,
        .sections = &.{},
        .ids_bytes = 0,
        .ids_crc = 0,
        .small_bytes = 0,
        .small_crc = 0,
    };
    return p;
}

pub fn addBlock(s: *Store, name: Digest, crc: u32, sha: Digest, ino: u64, bytes: u64) !void {
    const gpa = s.gpa;
    if (s.have.get(name)) |old| {
        if (old.ino != ino or old.bytes != bytes or old.crc != crc or !std.mem.eql(u8, &old.sha, &sha))
            return error.Corrupt;
        return;
    }
    try s.have.ensureUnusedCapacity(gpa, 1);
    try s.content.ensureUnusedCapacity(gpa, 1);
    const existing = s.inodes.get(ino);
    if (existing == null) try s.inodes.ensureUnusedCapacity(gpa, 1);
    const node = existing orelse try gpa.create(@import("store_types.zig").Inode);
    errdefer if (existing == null) gpa.destroy(node);
    if (existing == null) node.* = .{ .crc = crc, .sha = sha, .ino = ino, .bytes = bytes } else {
        if (node.bytes != bytes or node.crc != crc or !std.mem.eql(u8, &node.sha, &sha)) return error.Corrupt;
    }
    try node.names.ensureUnusedCapacity(gpa, 1);
    node.names.appendAssumeCapacity(name);
    s.have.putAssumeCapacity(name, node);
    if (existing == null) {
        s.inodes.putAssumeCapacity(ino, node);
        s.stored_bytes += bytes;
    }
    if (!s.content.contains(sha)) s.content.putAssumeCapacity(sha, name);
}

pub fn dropBlock(s: *Store, name: Digest) void {
    const info = (s.have.fetchRemove(name) orelse return).value;
    var freed: u64 = 0;
    if (s.inodes.get(info.ino)) |ino| {
        for (ino.names.items, 0..) |n, i| if (std.mem.eql(u8, &n, &name)) {
            _ = ino.names.swapRemove(i);
            break;
        };
        if (s.content.get(info.sha)) |cur| if (std.mem.eql(u8, &cur, &name)) {
            if (ino.names.items.len > 0) s.content.putAssumeCapacity(info.sha, ino.names.items[0]) else _ = s.content.remove(info.sha);
        };
        if (ino.names.items.len == 0) {
            s.stored_bytes -= ino.bytes;
            freed = ino.bytes;
            ino.names.deinit(s.gpa);
            _ = s.inodes.remove(info.ino);
            s.gpa.destroy(ino);
        }
    }
    s.pending_free += freed;
    s.queue(.{ .unlink_block = .{ .name = name, .bytes = freed } });
}

pub fn ref(s: *Store, blocks: []const format.BlockRef, k: i32) void {
    for (blocks) |b| {
        const gop = s.refs.getOrPut(s.gpa, b.name) catch continue;
        const cur: i64 = if (gop.found_existing) gop.value_ptr.* else 0;
        const v = cur + k;
        if (v > 0) gop.value_ptr.* = @intCast(v) else _ = s.refs.remove(b.name);
    }
}

pub fn gc(s: *Store, blocks: []const format.BlockRef) void {
    for (blocks) |b| {
        if (s.have.contains(b.name) and !s.refs.contains(b.name) and !s.inflight.contains(b.name)) s.dropBlock(b.name);
    }
}

pub fn queue(s: *Store, t: Task) void {
    if (t == .touch) {
        if (s.points.get(t.touch.key)) |p| {
            if (p.recency_dirty) return;
            p.recency_dirty = true;
            s.work.signal(s.io);
        }
        return;
    }
    s.tasks.append(s.gpa, t) catch {
        std.log.err("spill: rank {d}: writer queue allocation failed", .{s.cfg.rank});
        return;
    };
    s.work.signal(s.io);
}

pub fn complete(s: *Store, x: Completion) void {
    s.completions.append(s.gpa, x) catch {};
}

pub fn remove(s: *Store, key: Digest, comptime why: []const u8) void {
    const kv = s.points.fetchRemove(key) orelse return;
    const p = kv.value;
    s.ref(p.blocks, -1);
    const up = if (s.parent.fetchRemove(key)) |e| e.value else null;
    var it = s.parent.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.*) |v| if (std.mem.eql(u8, &v, &key)) {
            e.value_ptr.* = up;
        };
    }
    s.stored_bytes -= p.files;
    s.pending_free += p.files;
    s.queue(.{ .unlink_point = .{ .key = key, .bytes = p.files, .roots = p.history_roots } });
    @field(s.counters, why) += 1;
    s.gc(p.blocks);
    if (p.pins > 0) p.removed = true else p.deinit(s.gpa);
}

pub fn extends(s: *Store, p: *const Point, q: *const Point) bool {
    _ = s;
    if (q.at >= p.at) return false;
    const k = q.chain.len;
    if (k > p.chain.len) return false;
    for (q.chain, p.chain[0..k]) |a, b| if (!std.mem.eql(u8, &a, &b)) return false;
    if (q.tail.len == 0) return true;
    if (k != p.chain.len) return false;
    return std.mem.eql(u32, p.tail[0..q.tail.len], q.tail);
}

pub fn parentOf(s: *Store, p: *const Point) ?Digest {
    var best: ?*Point = null;
    var it = s.points.valueIterator();
    while (it.next()) |qq| {
        const q = qq.*;
        if (q == p) continue;
        if ((best == null or q.at > best.?.at) and s.extends(p, q)) best = q;
    }
    return if (best) |b| b.key else null;
}

pub fn adopt(s: *Store, p: *Point) void {
    s.parent.put(s.gpa, p.key, s.parentOf(p)) catch {};
    var it = s.points.valueIterator();
    while (it.next()) |qq| {
        const q = qq.*;
        if (q == p or !s.extends(q, p)) continue;
        const cur = s.parent.get(q.key) orelse null;
        if (cur == null or s.points.get(cur.?).?.at < p.at) s.parent.put(s.gpa, q.key, p.key) catch {};
    }
}

pub const Tree = struct {
    keys: []Digest,
    conv: []?usize, // a conversation id per key (null: a shared point)
    index: std.AutoHashMapUnmanaged(Digest, usize),

    pub fn deinit(t: *Tree, gpa: Allocator) void {
        gpa.free(t.keys);
        gpa.free(t.conv);
        t.index.deinit(gpa);
    }
};

pub fn find(uf: []usize, x: usize) usize {
    var r = x;
    while (uf[r] != r) r = uf[r];
    var y = x;
    while (uf[y] != r) {
        const nx = uf[y];
        uf[y] = r;
        y = nx;
    }
    return r;
}

pub fn tree(s: *Store) !Tree {
    const gpa = s.gpa;
    const n = s.points.count();
    const keys = try gpa.alloc(Digest, n);
    errdefer gpa.free(keys);
    var index: std.AutoHashMapUnmanaged(Digest, usize) = .empty;
    errdefer index.deinit(gpa);
    var it = s.points.keyIterator();
    var i: usize = 0;
    while (it.next()) |k| : (i += 1) {
        keys[i] = k.*;
        try index.put(gpa, k.*, i);
    }
    std.mem.sort(Digest, keys, {}, struct {
        pub fn lt(_: void, a: Digest, b: Digest) bool {
            return std.mem.order(u8, &a, &b) == .lt;
        }
    }.lt);
    for (keys, 0..) |k, j| index.putAssumeCapacity(k, j);
    const uf = try gpa.alloc(usize, n);
    defer gpa.free(uf);
    for (uf, 0..) |*u, j| u.* = j;
    const conv = try gpa.alloc(?usize, n);
    errdefer gpa.free(conv);
    var groups: std.AutoHashMapUnmanaged(u64, usize) = .empty;
    defer groups.deinit(gpa);
    for (keys, 0..) |k, j| {
        const p = s.points.get(k).?;
        if (p.shared) {
            conv[j] = null;
            continue;
        }
        var top = j;
        var cur = s.parent.get(k) orelse null;
        while (cur) |ck| {
            const q = s.points.get(ck) orelse break;
            if (q.shared) break;
            top = index.get(ck).?;
            cur = s.parent.get(ck) orelse null;
        }
        conv[j] = top;
    }
    for (keys, 0..) |k, j| {
        const top = conv[j] orelse continue;
        const p = s.points.get(k).?;
        if (p.group == 0) continue;
        const gop = try groups.getOrPut(gpa, p.group);
        if (!gop.found_existing) gop.value_ptr.* = top else {
            const a = find(uf, top);
            const b = find(uf, gop.value_ptr.*);
            if (a != b) uf[@max(a, b)] = @min(a, b);
        }
    }
    for (conv) |*x| if (x.*) |t| {
        x.* = find(uf, t);
    };
    return .{ .keys = keys, .conv = conv, .index = index };
}

pub const Why = enum { pruned_turns, pruned_forks, pruned_conversations, pruned_shared };

pub const Victim = struct { key: Digest, why: Why };

pub fn evictionOrder(s: *Store) ![]Victim {
    s.lockS();
    defer s.unlockS();
    var v = try s.victims(null);
    defer v.deinit(s.gpa);
    return s.metadata.parent.dupe(Victim, v.items);
}
