//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const c = @import("store_types.zig").c;
const format = @import("store_types.zig").format;
const fsio = @import("store_types.zig").fsio;
const slots_mod = @import("store_types.zig").slots_mod;
const mut = @import("store_types.zig").mut;
const history = @import("store_types.zig").history;
const Budget = @import("store_types.zig").Budget;
const Allocator = @import("store_types.zig").Allocator;
const Digest = @import("store_types.zig").Digest;
const Config = @import("store_types.zig").Config;
const Point = @import("store_types.zig").Point;
const WriteJob = @import("store_types.zig").WriteJob;
const ReadJob = @import("store_types.zig").ReadJob;
const recover = Store.recover;
const addBlock = Store.addBlock;
const ref = Store.ref;
const parentOf = Store.parentOf;
const release = Store.release;
const pins = Store.pins;
const writerLoop = Store.writerLoop;
const readerLoop = Store.readerLoop;

pub fn open(gpa: Allocator, io: std.Io, cfg: Config) !*Store {
    const l = cfg.limits;
    if (cfg.index_bytes < (1 << 20)) return error.BadConfig;
    if (cfg.worker_stack_bytes < 64 << 10 or cfg.worker_stack_bytes > 8 << 20) return error.BadConfig;
    if (l.block_tokens == 0 or l.block_bytes == 0 or l.max_tokens == 0 or cfg.cap_bytes == 0) return error.BadConfig;
    if (l.foreign_block_bytes > l.block_bytes or (l.foreign_block_bytes != 0 and cfg.owner_layout != .owner_sharded)) return error.BadConfig;
    if (!std.math.isFinite(cfg.quota) or cfg.quota < 0 or cfg.quota > 1) return error.BadConfig;
    const s = try gpa.create(Store);
    s.* = .{
        .gpa = gpa,
        .metadata = Budget.init(gpa, cfg.index_bytes),
        .io = io,
        .cfg = cfg,
        .hooks = cfg.hooks,
        .block_slot = std.mem.alignForward(usize, @intCast(l.block_bytes), fsio.alignment),
        .small_slot = std.mem.alignForward(usize, @intCast(@max(l.max_small_bytes, 1)), fsio.alignment),
    };
    s.gpa = s.metadata.allocator();
    s.chain_prefixes = .{ .gpa = s.gpa, .io = io };
    s.writer = fsio.Writer.init(&s.hooks, cfg.direct);
    errdefer s.destroy();
    s.slots = try slots_mod.Slots.init(s.gpa, cfg.staging, s.block_slot, s.small_slot);
    s.root_fd = try fsio.openDir(cfg.root);
    var nb: [64]u8 = undefined;
    const top = try namespaceName(&nb, cfg.compat);
    try fsio.ensureDirAt(s.root_fd, top);
    s.top_fd = try fsio.openDirAt(s.root_fd, top);
    var rb: [32]u8 = undefined;
    const rank = std.fmt.bufPrintSentinel(&rb, "rank{d}", .{cfg.rank}, 0) catch unreachable;
    try fsio.ensureDirAt(s.top_fd, rank);
    s.rank_fd = try fsio.openDirAt(s.top_fd, rank);
    s.lock = fsio.Lock.take(s.rank_fd) catch |e| if (mut(.owner_lock_ignored) and e == error.Busy) fsio.Lock{ .fd = -1 } else return if (e == error.Busy) error.StoreBusy else e;
    try fsio.ensureDirAt(s.rank_fd, "blocks");
    try fsio.ensureDirAt(s.rank_fd, "points");
    s.blocks_fd = try fsio.openDirAt(s.rank_fd, "blocks");
    s.points_fd = try fsio.openDirAt(s.rank_fd, "points");
    s.histories = try history.History.open(s.gpa, s.rank_fd, @max(1, cfg.index_bytes / 4 / 128));
    s.pruneBuilds(top, rank);
    try s.scan();
    try s.histories.?.finishRecovery(s.counters.recover_deferred == 0);
    s.history_bytes = s.histories.?.bytes;
    s.stored_bytes += s.history_bytes;
    s.recency_next = std.Io.Clock.awake.now(io).toNanoseconds() + cfg.recency_interval_ns;
    s.disk_free = fsio.diskFree(&s.hooks, s.blocks_fd);
    try s.reader_threads.ensureTotalCapacity(s.gpa, @max(cfg.readers, 1));
    s.writer_thread = try std.Thread.spawn(.{ .stack_size = cfg.worker_stack_bytes }, writerLoop, .{s});
    for (0..@max(cfg.readers, 1)) |_| {
        const t = std.Thread.spawn(.{ .stack_size = cfg.worker_stack_bytes }, readerLoop, .{s}) catch |e| {
            s.stopThreads();
            return e;
        };
        s.reader_threads.appendAssumeCapacity(t);
    }
    return s;
}

pub fn stopThreads(s: *Store) void {
    s.mutex.lockUncancelable(s.io);
    s.stopping = true;
    s.work.broadcast(s.io);
    s.rwork.broadcast(s.io);
    s.mutex.unlock(s.io);
    if (s.writer_thread) |t| t.join();
    s.writer_thread = null;
    for (s.reader_threads.items) |t| t.join();
    s.reader_threads.clearRetainingCapacity();
}

pub fn close(s: *Store) void {
    s.stopThreads();
    s.destroy();
}

pub fn destroy(s: *Store) void {
    const owner = s.metadata.parent;
    const gpa = s.gpa;
    var jit = s.jobs_by_id.valueIterator();
    while (jit.next()) |j| freeJob(gpa, j.*, true);
    s.jobs_by_id.deinit(gpa);
    for (s.reads.items) |r| {
        r.point.pins -= 1;
        if (r.point.pins == 0 and r.point.removed) r.point.deinit(gpa);
        freeRead(gpa, r);
    }
    s.reads.deinit(gpa);
    var pit = s.points.valueIterator();
    while (pit.next()) |p| p.*.deinit(gpa);
    s.points.deinit(gpa);
    s.have.deinit(gpa);
    s.content.deinit(gpa);
    var iit = s.inodes.valueIterator();
    while (iit.next()) |x| {
        x.*.names.deinit(gpa);
        gpa.destroy(x.*);
    }
    s.inodes.deinit(gpa);
    s.chain_prefixes.deinit();
    s.refs.deinit(gpa);
    s.inflight.deinit(gpa);
    s.parent.deinit(gpa);
    s.reading.deinit(gpa);
    s.waiting.deinit(gpa);
    s.fifo.deinit(gpa);
    s.parked.deinit(gpa);
    s.tasks.deinit(gpa);
    s.rtasks.deinit(gpa);
    s.completions.deinit(gpa);
    s.reader_threads.deinit(gpa);
    s.slots.deinit(gpa);
    if (s.histories) |*h| h.deinit();
    std.debug.assert(s.metadata.used.load(.monotonic) == 0);
    if (s.lock.fd >= 0) s.lock.release();
    for ([_]c.fd_t{ s.points_fd, s.blocks_fd, s.rank_fd, s.top_fd, s.root_fd }) |fd| if (fd >= 0) {
        _ = c.close(fd);
    };
    owner.destroy(s);
}

pub fn freeJob(gpa: Allocator, j: *WriteJob, with_point: bool) void {
    j.new.deinit(gpa);
    gpa.free(j.ids);
    if (with_point) j.point.deinit(gpa);
    gpa.destroy(j);
}

pub fn freeRead(gpa: Allocator, r: *ReadJob) void {
    gpa.free(r.state);
    gpa.free(r.slot);
    r.bad.deinit(gpa);
    r.lost.deinit(gpa);
    gpa.destroy(r);
}

pub fn namespaceName(buf: []u8, compat: Digest) ![:0]const u8 {
    const h = format.hex(&compat);
    return std.fmt.bufPrintSentinel(buf, "native-spill-{s}", .{h[0..16]}, 0) catch unreachable;
}

pub fn lockS(s: *Store) void {
    s.mutex.lockUncancelable(s.io);
}

pub fn unlockS(s: *Store) void {
    s.mutex.unlock(s.io);
}

pub fn now(s: *Store) i96 {
    const t = std.Io.Clock.real.now(s.io).toNanoseconds();
    s.clock = @max(t, s.clock + 1); // strictly increasing: last use orders every point
    return s.clock;
}

pub fn pruneBuilds(s: *Store, mine: []const u8, rank: [:0]const u8) void {
    const gpa = s.gpa;
    var names = fsio.list(gpa, s.root_fd) catch return;
    defer fsio.freeList(gpa, &names);
    const Other = struct { name: [:0]const u8, mtime: i96 };
    var others: std.ArrayList(Other) = .empty;
    defer others.deinit(gpa);
    for (names.items) |n| {
        if (!std.mem.startsWith(u8, n, "native-spill-") or std.mem.eql(u8, n, mine)) continue;
        if (fsio.isSymlinkAt(s.root_fd, n)) { // never ours: the link goes, its target is left alone
            fsio.unlinkAt(s.root_fd, n);
            continue;
        }
        const st = fsio.statAt(s.root_fd, n) catch continue;
        others.append(gpa, .{ .name = n, .mtime = st.mtime_ns }) catch return;
    }
    std.mem.sort(Other, others.items, {}, struct {
        pub fn newer(_: void, a: Other, b: Other) bool {
            return a.mtime > b.mtime;
        }
    }.newer);
    if (others.items.len <= s.cfg.keep_builds) return;
    for (others.items[s.cfg.keep_builds..]) |o| {
        const dir = fsio.openDirAt(s.root_fd, o.name) catch continue;
        defer _ = c.close(dir);
        if (fsio.isSymlinkAt(dir, rank)) fsio.unlinkAt(dir, rank);
        const sub = fsio.openDirAt(dir, rank) catch {
            _ = c.unlinkat(s.root_fd, o.name, c.AT.REMOVEDIR); // empty once every rank pruned its own
            continue;
        };
        defer _ = c.close(sub);
        var lock = fsio.Lock.take(sub) catch continue; // live (or unlockable): left alone
        defer lock.release();
        fsio.removeTree(gpa, dir, rank); // (its owner.lock goes too, while held)
        s.counters.builds_pruned += 1;
        _ = c.unlinkat(s.root_fd, o.name, c.AT.REMOVEDIR);
    }
}

pub fn pointName(buf: []u8, key: *const Digest, comptime suffix: []const u8) [:0]const u8 {
    const h = format.hex(key);
    return std.fmt.bufPrintSentinel(buf, "{s}" ++ suffix, .{h}, 0) catch unreachable;
}

pub fn blockName(buf: []u8, name: *const Digest) [:0]const u8 {
    const h = format.hex(name);
    return std.fmt.bufPrintSentinel(buf, "{s}.blk", .{h}, 0) catch unreachable;
}

pub fn scan(s: *Store) !void {
    const gpa = s.gpa;
    for ([_]c.fd_t{ s.points_fd, s.blocks_fd }) |dir| {
        var names = try fsio.list(gpa, dir);
        defer fsio.freeList(gpa, &names);
        for (names.items) |n| if (n[0] == '.' and !mut(.scan_keeps_temps)) {
            fsio.unlinkAt(dir, n);
            s.counters.temps_removed += 1;
        };
    }
    var present: std.AutoHashMapUnmanaged(Digest, fsio.Stat) = .empty;
    defer present.deinit(gpa);
    var unsure: std.AutoHashMapUnmanaged(Digest, void) = .empty; // blocks whose stat failed, not as missing
    defer unsure.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(Digest, void) = .empty; // points left on disk unread
    defer kept.deinit(gpa);
    {
        var names = try fsio.list(gpa, s.blocks_fd);
        defer fsio.freeList(gpa, &names);
        for (names.items) |n| {
            s.scanned(n);
            const d = if (std.mem.endsWith(u8, n, ".blk")) format.parseHex(n[0 .. n.len - 4]) else null;
            if (d == null) {
                fsio.unlinkAt(s.blocks_fd, n);
                s.counters.orphans_removed += 1;
                continue;
            }
            const st = fsio.statAt(s.blocks_fd, n) catch |e| {
                if (e != error.Missing) try unsure.put(gpa, d.?, {});
                continue;
            };
            try present.put(gpa, d.?, st);
        }
    }
    var names = try fsio.list(gpa, s.points_fd);
    defer fsio.freeList(gpa, &names);
    for (names.items) |n| {
        s.scanned(n);
        if (!std.mem.endsWith(u8, n, ".man")) continue;
        const key = format.parseHex(n[0 .. n.len - 4]) orelse continue;
        const p = s.recover(key, &present, &unsure) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Corrupt, error.Foreign, error.Missing, error.TooLarge, error.Short => null,
            else => if (mut(.recover_drops_on_io)) null else {
                try kept.put(gpa, key, {});
                s.counters.recover_deferred += 1;
                continue;
            },
        };
        if (p) |pt| {
            s.points.put(gpa, key, pt) catch |err| {
                pt.deinit(gpa);
                return err;
            };
            s.counters.recovered_points += 1;
        } else s.counters.recovered_dropped += 1;
    }
    // files of no point, then blocks no point names
    for (names.items) |n| {
        const dot = std.mem.indexOfScalar(u8, n, '.') orelse n.len;
        const key = format.parseHex(n[0..dot]);
        if (key == null or !(s.points.contains(key.?) or kept.contains(key.?))) {
            fsio.unlinkAt(s.points_fd, n);
            s.counters.orphans_removed += 1;
        }
    }
    var pit = s.points.valueIterator();
    while (pit.next()) |pp| {
        const p = pp.*;
        for (p.blocks) |b| {
            if (!s.have.contains(b.name)) try s.addBlock(b.name, b.crc, b.sha, present.get(b.name).?.ino, b.bytes);
        }
        s.stored_bytes += p.files;
        s.ref(p.blocks, 1);
    }
    // blocks no point names (kept while a point was left unread: they may be its)
    if (kept.count() == 0 and unsure.count() == 0) {
        var bit = present.iterator();
        while (bit.next()) |e| {
            if (s.have.contains(e.key_ptr.*)) continue;
            var bb: [80]u8 = undefined;
            fsio.unlinkAt(s.blocks_fd, blockName(&bb, e.key_ptr));
            s.counters.orphans_removed += 1;
        }
    }
    // parents, shortest first
    var order: std.ArrayList(*Point) = .empty;
    defer order.deinit(gpa);
    pit = s.points.valueIterator();
    while (pit.next()) |p| try order.append(gpa, p.*);
    std.mem.sort(*Point, order.items, {}, struct {
        pub fn shorter(_: void, a: *Point, b: *Point) bool {
            return a.at < b.at;
        }
    }.shorter);
    for (order.items) |p| try s.parent.put(gpa, p.key, s.parentOf(p));
    s.clock = 0;
    for (order.items) |p| s.clock = @max(s.clock, p.used);
}

pub fn scanned(s: *Store, name: []const u8) void {
    if (s.hooks.progress) |p| _ = p.fetchAdd(1, .monotonic);
    if (s.hooks.scanned) |f| f(s.hooks.ctx, name);
}
