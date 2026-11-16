//! Shared implementation helpers for store.
const source_api = @import("store_test.zig");
const std = source_api.std;
const store = source_api.store;
const Allocator = source_api.Allocator;
const block_tokens = source_api.block_tokens;
const bpt = source_api.bpt;
const CheckFailed = source_api.CheckFailed;
const ensure = source_api.ensure;
const H = source_api.H;
const write = source_api.write;
const idsOf = source_api.idsOf;
const lin = source_api.lin;
const waitUntil = source_api.waitUntil;
const Pending = source_api.Pending;
const slots = source_api.slots;
const Slow = source_api.Slow;
const format = source_api.format;
const keepIds = source_api.keepIds;
const pump = source_api.pump;
const restore = source_api.restore;
const manifestShared = source_api.manifestShared;
const countFiles = source_api.countFiles;
const crash_points = source_api.crash_points;
const c = source_api.c;
const Digest = source_api.Digest;
const block_bytes = source_api.block_bytes;
const blockFile = source_api.blockFile;
const pointFile = source_api.pointFile;
const settle = source_api.settle;
const io = source_api.io;
const max_tokens = source_api.max_tokens;
const small_max = source_api.small_max;
const Opts = source_api.Opts;
const Store = source_api.Store;
const cat = source_api.cat;
const fileExists = source_api.fileExists;
const fsio = source_api.fsio;

pub fn checkHang(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "hang", .read_slots = 4 });
    defer h.deinit();
    const n = 12 * block_tokens;
    const ids = try idsOf(gpa, n, 501);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(50), .source = 500 });
    const id = try h.s.beginRead(key);
    var pend: Pending = .{ .h = h };
    defer pend.held.deinit(gpa);
    const out = try gpa.alloc(u8, n * bpt);
    defer gpa.free(out);
    var ok = true;
    for (0..12) |i| {
        if (!h.s.readAwait(id, @intCast(i), 2 * std.time.ns_per_s, .{ .ctx = &pend, .recycle = Pending.recycle })) {
            ok = false;
            break;
        }
        var rb: [1]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, @intCast(i + 1), &rb);
        if (got.n != 1) {
            ok = false;
            break;
        }
        @memcpy(out[rb[0].first * bpt ..][0..rb[0].data.len], rb[0].data);
        try pend.held.append(gpa, rb[0].slot); // its copy's event not complete yet
    }
    Pending.recycle(&pend);
    if (!ok) _ = h.s.endRead(id, .abort) catch {};
    try ensure(ok, @src());
    _ = waitUntil(struct {
        pub fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000);
    var done = (h.s.endRead(id, .commit) catch return CheckFailed) orelse return CheckFailed;
    done.lineage.deinit(gpa);
    try ensure(std.mem.eql(u8, out, h.src(500)[0 .. n * bpt]), @src());
}

pub fn checkCarve(gpa: Allocator) !void {
    const per = 3 * slots.alignment;
    const pre = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), 2 * per + per / 2);
    defer gpa.free(pre);
    try ensure(slots.extraBytes(&.{pre}, per, 7) == 5 * per, @src());
    try ensure(slots.extraBytes(&.{pre}, per, 2) == 0, @src());
}

pub fn checkBudget(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "budget", .read_slots = 16, .restore_slots = 3, .max_reads = 3 });
    defer h.deinit();
    var ids: [3]store.ReadId = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, 5 * block_tokens, 80 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i });
        ids[i] = try h.s.beginRead(key);
    }
    for (ids) |id| for (0..5) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src());
    var slow: Slow = .{ .h = h };
    defer slow.calls.deinit(gpa);
    var turn: usize = 0;
    _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
    // copies counted, not wall time (a loaded machine stretches each 30 ms sleep, which let the every-read-its-own-
    const first_calls = slow.calls.items.len;
    var firsts: [3]store.ReadId = undefined;
    for (0..3) |k| {
        slow.calls.clearRetainingCapacity();
        _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
        firsts[k] = if (slow.calls.items.len > 0) slow.calls.items[0] else 0;
    }
    for (ids) |id| _ = h.s.endRead(id, .cancel) catch {};
    try ensure(first_calls >= 1 and first_calls <= 2, @src()); // (every read the whole budget: 3 or more)
    std.mem.sort(store.ReadId, &firsts, {}, std.sort.asc(store.ReadId));
    var want = ids;
    std.mem.sort(store.ReadId, &want, {}, std.sort.asc(store.ReadId));
    try ensure(std.mem.eql(store.ReadId, &firsts, &want), @src());
}

pub fn crashed(h: *H) bool {
    return h.s.isCrashed();
}

pub fn checkCrashes(gpa: Allocator) !void {
    return checkCrashesMode(gpa, false);
}

pub fn checkCrashesMode(gpa: Allocator, serial: bool) !void {
    for (crash_points, 0..) |cp, ci| {
        const h = try H.init(gpa, .{ .tag = "crash", .shared_small_restore = serial });
        defer h.deinit();
        const a = try idsOf(gpa, 2 * block_tokens, 900 + ci);
        defer gpa.free(a);
        const ka = try write(h, a, .{ .lineage = lin(90), .source = 90 });
        const b = try idsOf(gpa, 3 * block_tokens + 3, 950 + ci);
        defer gpa.free(b);
        const src_b: u64 = if (cp.mode == .link) 90 else 91; // link: B's rows are A's (other ids, other names)
        var kb = format.pointKey(b, block_tokens, &.{});
        if (cp.mode == .rewrite) kb = try write(h, b, .{ .lineage = lin(91), .source = src_b });
        h.hk.crash_step = cp.step;
        h.hk.crash_suffix = cp.suffix;
        h.hk.crash_skip = cp.skip;
        if (cp.mode == .rewrite) {
            h.s.markShared(kb);
        } else {
            try ensure(keepIds(h, b, .{ .lineage = lin(91), .source = src_b }) == .queued, @src());
            pump(h);
        }
        if (!waitUntil(crashed, h, 3000)) {
            if (!source_api.quiet) std.debug.print("     no crash at {t} {s} (+{d}, {t})\n", .{ cp.step, cp.suffix, cp.skip, cp.mode });
            return CheckFailed;
        }
        const temps_before = countFiles(gpa, h.s.blocks_fd, true) + countFiles(gpa, h.s.points_fd, true);
        h.hk.crash_step = null;
        try h.reopen();
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, true) == 0, @src());
        if (temps_before > 0) try ensure(h.s.counts().temps_removed >= temps_before, @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count(), @src());
        try ensure(countFiles(gpa, h.s.points_fd, false) == 3 * h.s.points.count(), @src());
        const got = (try restore(h, ka)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(90)[0 .. 2 * block_tokens * bpt]), @src());
        const published = cp.mode == .rewrite or (cp.step == .renamed and std.mem.eql(u8, cp.suffix, ".man")) or cp.step == .published;
        try ensure(h.s.presence(kb) == published, @src());
        if (published) {
            const gb = (try restore(h, kb)) orelse return CheckFailed;
            defer gpa.free(gb);
            try ensure(std.mem.eql(u8, gb, h.src(src_b)[0 .. b.len * bpt]), @src());
        }
        if (cp.mode == .rewrite) { // whole either way; durably shared once the rewrite's directory fsync ran
            const on_disk = manifestShared(h, kb) orelse return CheckFailed;
            try ensure((h.s.points.get(kb) orelse return CheckFailed).shared == on_disk, @src());
            if (cp.step == .rewritten) try ensure(on_disk, @src());
            if (cp.step == .written or cp.step == .synced) try ensure(!on_disk, @src());
        }
    }
}

pub fn checkRecovery(gpa: Allocator) !void {
    return checkRecoveryMode(gpa, false);
}

pub fn checkRecoveryMode(gpa: Allocator, serial: bool) !void {
    const h = try H.init(gpa, .{ .tag = "recover", .shared_small_restore = serial });
    defer h.deinit();
    var keys: [4]Digest = undefined;
    for (0..4) |i| {
        const x = try idsOf(gpa, 2 * block_tokens + 4, 300 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(30 + i)), .source = 300 + i });
    }
    const p0 = h.s.points.get(keys[0]) orelse return CheckFailed;
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const bname = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, p0.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, bname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    _ = c.ftruncate(fd, block_bytes - 100);
    _ = c.close(fd);
    var nb2: [96]u8 = undefined;
    var zb2: [96]u8 = undefined;
    _ = c.unlinkat(h.s.points_fd, try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[1], ".st")}, 0), 0);
    const mname = try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[2], ".man")}, 0);
    const mfd = c.openat(h.s.points_fd, mname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    const junk: [4]u8 = @splat(0x33);
    _ = c.pwrite(mfd, &junk, 4, 40);
    _ = c.close(mfd);
    const tfd = c.openat(h.s.blocks_fd, ".stale.blk.1.1.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    try h.reopen();
    try ensure(!h.s.points.contains(keys[0]) and !h.s.points.contains(keys[1]) and !h.s.points.contains(keys[2]), @src());
    try ensure(h.s.points.contains(keys[3]), @src());
    try ensure(h.s.counts().recovered_dropped == 3 and h.s.counts().temps_removed >= 1, @src());
    try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count() and h.s.have.count() == 3, @src());
    const got = (try restore(h, keys[3])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(303)[0 .. (2 * block_tokens + 4) * bpt]), @src());
}

pub fn forgeCrc(data: []u8, target: u32) void {
    var table: [256]u32 = undefined;
    for (&table, 0..) |*t, i| {
        var x: u32 = @intCast(i);
        for (0..8) |_| x = if (x & 1 != 0) (x >> 1) ^ 0xEDB88320 else x >> 1;
        t.* = x;
    }
    const head = data[0 .. data.len - 4];
    const before = ~std.hash.Crc32.hash(head); // the register after the head
    var x: u32 = ~target; // the register wanted after the last 4 bytes
    for (0..4) |_| {
        const top: u8 = @truncate(x >> 24);
        var idx: u32 = 0;
        for (table, 0..) |t, i| if (@as(u8, @truncate(t >> 24)) == top) {
            idx = @intCast(i);
            break;
        };
        x = ((x ^ table[idx]) << 8) | idx;
    }
    std.mem.writeInt(u32, data[data.len - 4 ..][0..4], x ^ before, .little);
}

pub fn checkDedupSha(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "dedup" });
    defer h.deinit();
    const ia = try idsOf(gpa, block_tokens, 41);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, block_tokens, 42);
    defer gpa.free(ib);
    const ka = try write(h, ia, .{ .lineage = lin(41), .source = 41 });
    const want = std.hash.Crc32.hash(h.src(41)[0..block_bytes]);
    const rows = h.src(42);
    @memcpy(rows[0..block_bytes], h.src(41)[0..block_bytes]);
    rows[17] ^= 0x5a;
    forgeCrc(rows[0..block_bytes], want);
    try ensure(std.hash.Crc32.hash(rows[0..block_bytes]) == want, @src());
    const kb = try write(h, ib, .{ .lineage = lin(42), .source = 42 });
    const pa = h.s.points.get(ka) orelse return CheckFailed;
    const pb = h.s.points.get(kb) orelse return CheckFailed;
    try ensure(h.s.have.get(pa.blocks[0].name).?.ino != h.s.have.get(pb.blocks[0].name).?.ino, @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, rows[0..block_bytes]), @src());
}

pub fn checkGuard(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guard" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 7);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(5)[0 .. 4 * block_tokens * bpt]);
    defer gpa.free(kept);
    try ensure(keepIds(h, ids, .{ .lineage = lin(7), .source = 5 }) == .queued, @src());
    try ensure(h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    var any: [3]bool = undefined;
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (pins for many sources, one lock)
    try ensure(!any[0] and any[1] and !any[2], @src());
    var out: [8]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 5, .generation = 1 }, 0, &out);
    try ensure(g.staged == 4 and g.cancelled == 0, @src());
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (every block staged: nothing left to pin, as pins says)
    try ensure(!any[1] and any[1] == h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    for (out[0..g.staged]) |r| @memcpy(r.dst, h.src(5)[r.first * bpt ..][0..r.dst.len]); // queued before the change
    @memset(h.src(5)[0 .. 4 * block_tokens * bpt], 0xee); // the change
    for (out[0..g.staged]) |r| h.s.staged(r.ticket, true);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
}

pub fn checkGuardCancel(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guardc" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 8);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(8), .source = 6 }) == .queued, @src());
    var out: [2]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 6, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    const ids2 = try idsOf(gpa, 2 * block_tokens, 9);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(9), .source = 7, .gen = 1 }) == .queued, @src());
    var o2: [8]store.StageReq = undefined;
    const g2 = h.s.guard(.{ .id = 7, .generation = 2 }, 0, &o2);
    try ensure(g2.cancelled == 1, @src());
    var comp: [4]store.Completion = undefined;
    const n = h.s.poll(&comp);
    try ensure(n == 2 and comp[0] == .cancelled and comp[0].cancelled.why == .guard and comp[1].cancelled.why == .source_gone, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_bytes == 0 and ct.points == 0 and ct.pending_jobs == 0, @src());
}

pub fn checkStagingCap(gpa: Allocator) !void {
    const o: Opts = .{ .tag = "stagecap", .write_slots = 3, .read_slots = 2, .small_slots = 2, .restore_slots = 1 };
    const cap: u64 = (o.write_slots + o.read_slots) * block_bytes + (o.small_slots + o.restore_slots) * small_max;
    const h = try H.init(gpa, o);
    defer h.deinit();
    const ids = try idsOf(gpa, max_tokens, 3100); // 64 blocks: 256 KiB against 44 KiB of staging
    defer gpa.free(ids);
    try ensure(max_tokens * bpt > 4 * cap, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(31), .source = 310 }) == .queued, @src());
    var reqs: [16]store.StageReq = undefined;
    var staged: usize = 0;
    var i: usize = 0;
    while (i < 20000 and h.s.pendingStage() > 0) : (i += 1) {
        const n = h.s.stage(&reqs);
        try ensure(n <= o.write_slots, @src());
        try ensure(h.s.counts().staging_bytes <= cap, @src());
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(310)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
        staged += n;
        if (n == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try ensure(staged == max_tokens / block_tokens, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure(h.s.presence(key), @src());
    const got = (try restore(h, key)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(310)[0 .. max_tokens * bpt]), @src());
    // a second large point guarded before any of it staged: more blocks left than free write slots, so it is cancelled
    const ids2 = try idsOf(gpa, max_tokens, 3101);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(32), .source = 311 }) == .queued, @src());
    var out: [64]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 311, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_peak_bytes <= cap and ct.staging_peak_bytes >= block_bytes, @src());
    try ensure(ct.staging_bytes == 0 and ct.guard_staged == 0 and ct.cancelled_guard == 1, @src());
}

pub fn checkEvictionOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "evict", .turns = 2 });
    defer h.deinit();
    const sys = try idsOf(gpa, 2 * block_tokens, 1000);
    defer gpa.free(sys);
    const r1 = try idsOf(gpa, block_tokens, 1001);
    defer gpa.free(r1);
    const r2 = try idsOf(gpa, block_tokens, 1002);
    defer gpa.free(r2);
    const r3 = try idsOf(gpa, block_tokens, 1003);
    defer gpa.free(r3);
    const r4 = try idsOf(gpa, block_tokens, 1004);
    defer gpa.free(r4);
    const b1 = try cat(gpa, sys, r1);
    defer gpa.free(b1);
    const a1 = try cat(gpa, sys, r2);
    defer gpa.free(a1);
    const a2f = try cat(gpa, a1, r3);
    defer gpa.free(a2f);
    const a2 = try cat(gpa, a1, r4);
    defer gpa.free(a2);
    const ks = try write(h, sys, .{ .lineage = lin(1), .source = 1, .shared = true });
    const kb1 = try write(h, b1, .{ .lineage = lin(2), .source = 2 });
    const ka1 = try write(h, a1, .{ .lineage = lin(3), .source = 3 });
    const ka2f = try write(h, a2f, .{ .lineage = lin(4), .source = 4 });
    const ka2 = try write(h, a2, .{ .lineage = lin(5), .source = 5 });
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    const W = Store.Why;
    const want = [_]struct { k: Digest, w: W }{
        .{ .k = ka2f, .w = .pruned_forks },
        .{ .k = kb1, .w = .pruned_conversations },
        .{ .k = ka2, .w = .pruned_conversations },
        .{ .k = ka1, .w = .pruned_conversations },
        .{ .k = ks, .w = .pruned_shared },
    };
    try ensure(order.len == want.len, @src());
    for (order, want) |o, w| try ensure(std.mem.eql(u8, &o.key, &w.k) and o.why == w.w, @src());
    const sm = h.s.summary();
    try ensure(sm.conversations == 2 and sm.shared_states == 1 and sm.forks == 1 and sm.turns_kept_max == 2, @src());
}

pub fn checkTurns(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "turns", .turns = 2 });
    defer h.deinit();
    const t3 = try idsOf(gpa, 3 * block_tokens, 1100);
    defer gpa.free(t3);
    const k1 = try write(h, t3[0..block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k2 = try write(h, t3[0 .. 2 * block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k3 = try write(h, t3, .{ .lineage = lin(1), .source = 11 });
    try ensure(!h.s.points.contains(k1) and h.s.points.contains(k2) and h.s.points.contains(k3), @src());
    const ct = h.s.counts();
    try ensure(ct.trimmed_turns == 1 and ct.blocks == 3 and ct.staged_blocks == 3, @src());
    const got = (try restore(h, k3)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(11)[0 .. 3 * block_tokens * bpt]), @src());
}

pub fn checkFork(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fork" });
    defer h.deinit();
    const p = try idsOf(gpa, 3 * block_tokens, 1200);
    defer gpa.free(p);
    const more = try idsOf(gpa, 2 * block_tokens, 1201);
    defer gpa.free(more);
    const q = try cat(gpa, p[0 .. 2 * block_tokens], more);
    defer gpa.free(q);
    _ = try write(h, p, .{ .lineage = lin(12), .source = 12 });
    const before = h.s.counts().staged_blocks;
    var forked = try lin(12).fork(gpa, @splat(77), 2 * block_tokens);
    defer forked.deinit(gpa);
    const kq = try write(h, q, .{ .lineage = forked, .source = 12 });
    try ensure(h.s.counts().staged_blocks - before == 2, @src());
    const got = (try restore(h, kq)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(12)[0 .. 4 * block_tokens * bpt]), @src());
}

pub fn checkGroups(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "groups" });
    defer h.deinit();
    const sys = try idsOf(gpa, block_tokens, 1300);
    defer gpa.free(sys);
    _ = try write(h, sys, .{ .lineage = lin(13), .source = 13, .shared = true });
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1301 + i);
        defer gpa.free(x);
        const y = try cat(gpa, sys, x);
        defer gpa.free(y);
        _ = try write(h, y, .{ .lineage = lin(@intCast(14 + i)), .source = 14 + i, .group = if (i < 2) 7 else 0 });
    }
    try ensure(h.s.summary().conversations == 2, @src());
}

pub fn checkQuota(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "quota", .cap = 400 << 10, .quota = 0.1, .turns = 0 });
    defer h.deinit();
    const other = try idsOf(gpa, 2 * block_tokens, 1400);
    defer gpa.free(other);
    const ko = try write(h, other, .{ .lineage = lin(40), .source = 40 });
    const conv = try idsOf(gpa, 8 * block_tokens, 1401);
    defer gpa.free(conv);
    var keys: [8]Digest = undefined;
    for (1..9) |t| {
        const r = keepIds(h, conv[0 .. t * block_tokens], .{ .lineage = lin(41), .source = 41 });
        if (r != .queued) break;
        try ensure(settle(h).done, @src());
        keys[t - 1] = format.pointKey(conv[0 .. t * block_tokens], block_tokens, &.{});
    }
    const ct = h.s.counts();
    try ensure(ct.quota_trims > 0 and ct.pruned_turns > 0, @src());
    try ensure(h.s.points.contains(ko), @src());
    try ensure(h.s.points.contains(keys[7]), @src());
}

pub fn checkFloor(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "floor", .floor = 1 << 20, .free = (1 << 20) + 200 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 4 * block_tokens, 1500);
    defer gpa.free(a);
    const ka = try write(h, a, .{ .lineage = lin(50), .source = 50 });
    // the disk fills with other files: the next write evicts A first
    h.hk.free = (1 << 20) + 2 * 1024;
    h.s.refreshDiskFree();
    const b = try idsOf(gpa, 2 * block_tokens, 1501);
    defer gpa.free(b);
    try ensure(keepIds(h, b, .{ .lineage = lin(51), .source = 51 }) == .queued, @src());
    try ensure(!h.s.points.contains(ka), @src());
    _ = settle(h);
    // full of other things: nothing could make room, the write is refused
    h.hk.free = 1 << 10;
    h.s.refreshDiskFree();
    const c2 = try idsOf(gpa, 2 * block_tokens, 1502);
    defer gpa.free(c2);
    try ensure(keepIds(h, c2, .{ .lineage = lin(52), .source = 52 }) == .no_space, @src());
}

pub fn checkCap(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cap", .cap = 12 * 1024 });
    defer h.deinit();
    var keys: [3]Digest = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1600 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(60 + i)), .source = 60 + i });
    }
    try ensure(!h.s.points.contains(keys[0]) and h.s.points.contains(keys[1]) and h.s.points.contains(keys[2]), @src());
    try ensure(h.s.counts().pruned_conversations >= 1 and h.s.counts().stored_bytes <= h.cfg.cap_bytes, @src());
}

pub fn checkOwner(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "owner" });
    defer h.deinit();
    const tfd = c.openat(h.s.blocks_fd, ".pending.blk.9.9.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    if (Store.open(gpa, io, h.cfg)) |other| {
        other.close();
        try ensure(false, @src());
    } else |e| try ensure(e == error.StoreBusy, @src());
    try ensure(fileExists(h.s.blocks_fd, ".pending.blk.9.9.tmp"), @src());
}

pub fn checkPrune(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "prune", .keep_builds = 0 });
    defer h.deinit();
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    try fsio.ensureDirAt(rfd, "native-spill-00000000deadbeef");
    const ofd = try fsio.openDirAt(rfd, "native-spill-00000000deadbeef");
    defer _ = c.close(ofd);
    try fsio.ensureDirAt(ofd, "rank0");
    const r0 = try fsio.openDirAt(ofd, "rank0");
    var live = try fsio.Lock.take(r0);
    try h.reopen();
    try ensure(fileExists(ofd, "rank0"), @src());
    live.release();
    _ = c.close(r0);
    try h.reopen();
    try ensure(!fileExists(rfd, "native-spill-00000000deadbeef"), @src());
}
