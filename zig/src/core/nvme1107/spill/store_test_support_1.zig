//! Shared implementation helpers for store.
const source_api = @import("store_test.zig");
const std = source_api.std;
const CheckFailed = source_api.CheckFailed;
const store = source_api.store;
const block_tokens = source_api.block_tokens;
const bpt = source_api.bpt;
const secs = source_api.secs;
const H = source_api.H;
const KO = source_api.KO;
const Store = source_api.Store;
const io = source_api.io;
const format = source_api.format;
const Digest = source_api.Digest;
const Allocator = source_api.Allocator;
const Lineage = source_api.Lineage;
const fsio = source_api.fsio;
const c = source_api.c;
const block_bytes = source_api.block_bytes;

pub fn ensure(ok: bool, src: std.builtin.SourceLocation) !void {
    if (ok) return;
    if (!source_api.quiet) std.debug.print("     check failed at {s}:{d}\n", .{ src.file, src.line });
    return CheckFailed;
}

pub fn smallByte(source: u64, at: usize, i: usize) u8 {
    return @truncate(i *% 131 +% source *% 7 +% at);
}

pub fn keepIds(h: *H, ids: []const u32, o: KO) store.KeepResult {
    const at = ids.len;
    const tail_tokens = at % block_tokens;
    const r = h.s.keep(.{
        .keyed = ids,
        .raw = if (o.raw) |r| r[0..ids.len] else null,
        .media = o.media,
        .lineage = o.lineage,
        .shared = o.shared,
        .group = o.group,
        .sections = &secs,
        .tail_bytes = tail_tokens * bpt,
        .source = .{ .id = o.source, .generation = o.gen },
    });
    if (r == .queued) {
        const q = r.queued;
        for (q.sections) |sec| {
            for (0..@intCast(sec.len)) |i| q.small[@intCast(sec.offset + i)] = smallByte(o.source, at, i);
        }
        if (q.tail) |t| @memcpy(t, h.src(o.source)[(at - tail_tokens) * bpt .. at * bpt]);
        if (o.capture) h.s.captured(q.job, true);
    }
    return r;
}

pub fn pump(h: *H) void {
    var reqs: [16]store.StageReq = undefined;
    while (true) {
        const n = h.s.stage(&reqs);
        if (n == 0) return;
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(r.source.id)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
    }
}

pub fn settle(h: *H) store.Store.Drain {
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        pump(h);
        if (h.s.pendingStage() == 0) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return h.s.drain(5 * std.time.ns_per_s);
}

pub fn write(h: *H, ids: []const u32, o: KO) !Digest {
    const r = keepIds(h, ids, o);
    try ensure(r == .queued, @src());
    const d = settle(h);
    try ensure(d.done, @src());
    const key = format.pointKey(ids, block_tokens, o.media);
    try ensure(h.s.presence(key), @src());
    return key;
}

pub fn restore(h: *H, key: Digest) !?[]u8 {
    const id = h.s.beginRead(key) catch return null;
    const p = h.s.points.get(key) orelse return null;
    const at = p.at;
    const out = try h.gpa.alloc(u8, at * bpt);
    errdefer h.gpa.free(out);
    var i: u32 = 0;
    const total: u32 = @intCast(p.blocks.len);
    while (i < total) {
        if (!h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{})) break;
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        if (got.status == .failed and got.n == 0) break;
        for (rb[0..got.n]) |b| {
            @memcpy(out[b.first * bpt ..][0..b.data.len], b.data);
            h.s.readRelease(b.slot);
            i += 1;
        }
    }
    var small_ok = false;
    for (0..2500) |_| {
        if (h.s.readSmallView(id) != null) {
            small_ok = true;
            break;
        }
        if (h.s.readFailure(id).err != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    if (i < total or !small_ok) {
        _ = h.s.endRead(id, .abort) catch {};
        h.gpa.free(out);
        return null;
    }
    var done = (h.s.endRead(id, .commit) catch {
        h.gpa.free(out);
        return null;
    }) orelse {
        h.gpa.free(out);
        return null;
    };
    done.lineage.deinit(h.gpa);
    return out;
}

pub fn restoreFails(h: *H, key: Digest) !bool {
    const id = h.s.beginRead(key) catch return true;
    const total: u32 = @intCast((h.s.points.get(key) orelse return true).blocks.len);
    var failed = false;
    var i: u32 = 0;
    while (i < total) {
        _ = h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{});
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
        i += @intCast(got.n);
        if (got.status == .failed) {
            failed = true;
            break;
        }
        if (got.n == 0 and got.status == .waiting and h.s.readFailure(id).err != null) {
            failed = true;
            break;
        }
    }
    const f = h.s.readFailure(id);
    if (f.err != null) failed = true;
    if (failed) {
        const bad = try h.gpa.dupe(Digest, f.bad);
        defer h.gpa.free(bad);
        const lost = try h.gpa.dupe(Digest, f.lost);
        defer h.gpa.free(lost);
        h.s.forget(key, bad, lost);
    }
    _ = h.s.endRead(id, if (failed) .abort else .cancel) catch {};
    return failed;
}

pub fn idsOf(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

pub fn cat(gpa: Allocator, a: []const u32, b: []const u32) ![]u32 {
    const out = try gpa.alloc(u32, a.len + b.len);
    @memcpy(out[0..a.len], a);
    @memcpy(out[a.len..], b);
    return out;
}

pub fn lin(n: u8) Lineage {
    return .{ .root = @splat(n) };
}

pub fn manifestShared(h: *H, key: Digest) ?bool {
    var nb: [80]u8 = undefined;
    const h64 = format.hex(&key);
    const name = std.fmt.bufPrintSentinel(&nb, "{s}.man", .{h64}, 0) catch return null;
    const raw = fsio.readSmall(h.gpa, h.s.points_fd, name, 1 << 20) catch return null;
    defer h.gpa.free(raw);
    var m = format.Manifest.parse(h.gpa, raw, h.cfg.compat, h.cfg.limits) catch return null;
    defer m.deinit(h.gpa);
    return m.flags.shared;
}

pub fn fileExists(dirfd: c.fd_t, name: []const u8) bool {
    var nb: [128]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&nb, "{s}", .{name}, 0) catch return false;
    _ = fsio.statAt(dirfd, z) catch return false;
    return true;
}

pub fn blockFile(buf: []u8, name: Digest) []const u8 {
    const x = format.hex(&name);
    return std.fmt.bufPrint(buf, "{s}.blk", .{x}) catch unreachable;
}

pub fn pointFile(buf: []u8, key: Digest, suffix: []const u8) []const u8 {
    const x = format.hex(&key);
    return std.fmt.bufPrint(buf, "{s}{s}", .{ x, suffix }) catch unreachable;
}

pub fn countFiles(gpa: Allocator, dirfd: c.fd_t, temps_only: bool) usize {
    var names = fsio.list(gpa, dirfd) catch return 0;
    defer fsio.freeList(gpa, &names);
    var n: usize = 0;
    for (names.items) |x| {
        if (temps_only and x[0] != '.') continue;
        n += 1;
    }
    return n;
}

pub fn waitUntil(cond: anytype, ctx: anytype, ms: u32) bool {
    var i: u32 = 0;
    while (i < ms) : (i += 2) {
        if (cond(ctx)) return true;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    return cond(ctx);
}

pub fn checkTail(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "tail" });
    defer h.deinit();
    const n = 2 * block_tokens + 6;
    const ids = try idsOf(gpa, n, 1);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(1)[0 .. n * bpt]);
    defer gpa.free(kept);
    const r = keepIds(h, ids, .{ .lineage = lin(1), .source = 1 });
    try ensure(r == .queued, @src());
    const had_tail = r.queued.tail != null;
    // the stream goes on decoding in its extent past the keep: the tail block's rows change
    var prng = std.Random.DefaultPrng.init(9);
    prng.random().bytes(h.src(1)[2 * block_tokens * bpt .. (n + 8) * bpt]);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
    try ensure(had_tail and h.s.counts().staged_blocks == 2, @src());
}

pub fn checkUpto(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upto" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(2), .source = 2 });
    const id = try h.s.beginRead(key);
    for (0..3) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src()); // (readers finish in any order)
    var rb: [8]store.ReadyBlock = undefined;
    const a = h.s.readReady(id, 2, &rb);
    try ensure(a.n == 2 and a.status == .done, @src());
    for (rb[0..a.n]) |b| h.s.readRelease(b.slot);
    const b2 = h.s.readReady(id, null, &rb);
    try ensure(b2.n == 1 and b2.status == .done and rb[0].index == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(waitUntil(struct {
        pub fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000), @src());
    const v = h.s.readSmallView(id).?;
    try ensure(v.sections.len == 2 and std.mem.eql(u8, v.sections[0].name.slice(), "ring"), @src());
    for (v.sections) |sec| for (0..@intCast(sec.len)) |i| {
        try ensure(v.bytes[@intCast(sec.offset + i)] == smallByte(2, 3 * block_tokens, i), @src());
    };
    var done = (try h.s.endRead(id, .commit)) orelse return CheckFailed;
    defer done.lineage.deinit(gpa);
    try ensure(done.at == 3 * block_tokens and done.lineage.eql(lin(2)), @src());
}

pub fn checkLastUse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lastuse" });
    defer h.deinit();
    const a = try idsOf(gpa, block_tokens, 2100);
    defer gpa.free(a);
    const b = try idsOf(gpa, block_tokens, 2101);
    defer gpa.free(b);
    const ka = try write(h, a, .{ .lineage = lin(21), .source = 21 });
    const kb = try write(h, b, .{ .lineage = lin(22), .source = 22 });
    std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
    h.s.touch(ka);
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try h.reopen();
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    try ensure(order.len == 2 and std.mem.eql(u8, &order[0].key, &kb) and std.mem.eql(u8, &order[1].key, &ka), @src());
}

pub fn checkRefusals(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "refuse", .small_slots = 1, .backlog = 3 * block_bytes + 2048 });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2200);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids[0..8], .{ .lineage = lin(1), .source = 1 }) == .short, @src());
    const bad = h.s.keep(.{ .keyed = ids, .lineage = lin(1), .sections = &.{.{ .name = "a-section-name-longer-than-32-bytes", .len = 8 }}, .source = .{ .id = 1, .generation = 1 } });
    try ensure(bad == .invalid, @src());
    const tail = h.s.keep(.{ .keyed = ids[0 .. block_tokens + 3], .lineage = lin(1), .sections = &secs, .tail_bytes = 0, .source = .{ .id = 1, .generation = 1 } });
    try ensure(tail == .invalid, @src());
    const r1 = keepIds(h, ids, .{ .lineage = lin(1), .source = 1, .capture = false });
    try ensure(r1 == .queued, @src());
    const other = try idsOf(gpa, block_tokens, 2201);
    defer gpa.free(other);
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .no_staging, @src());
    h.s.captured(r1.queued.job, true);
    try ensure(settle(h).done, @src());
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .queued, @src());
    try ensure(settle(h).done, @src());
    const big = try idsOf(gpa, 4 * block_tokens, 2202);
    defer gpa.free(big);
    try ensure(keepIds(h, big, .{ .lineage = lin(3), .source = 3 }) == .backlog, @src());
    const ct = h.s.counts();
    try ensure(ct.short == 1 and ct.invalid == 2 and ct.no_staging == 1 and ct.backlog_refused == 1 and ct.staging_bytes == 0, @src());
    try ensure(ct.committed == 2 and ct.points == 2, @src());
}

pub fn checkSharedUpgrade(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upgrade" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 51);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(51), .source = 51 });
    h.hk.clear();
    try ensure(keepIds(h, ids, .{ .lineage = lin(51), .source = 51, .shared = true }) == .stored, @src());
    try ensure(h.s.points.get(key).?.shared, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
    // the rewrite's own events: written to a temporary name, fsynced, pages dropped, renamed, the folder fsynced
    h.hk.mutex.lockUncancelable(io);
    var saw: [8]fsio.Event = undefined;
    var k: usize = 0;
    for (h.hk.events.items) |e| if (std.mem.endsWith(u8, e.name[0..e.len], ".man") or std.mem.eql(u8, e.name[0..e.len], "points")) {
        if (k < saw.len) saw[k] = e.ev;
        k += 1;
    };
    h.hk.mutex.unlock(io);
    try ensure(k == 5 and std.mem.eql(fsio.Event, saw[0..5], &.{ .write, .fsync, .drop_pages, .rename, .dir_fsync }), @src());
    try h.reopen();
    const p = h.s.points.get(key) orelse return CheckFailed;
    try ensure(p.shared, @src());
}

pub fn checkSharedWaiting(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "waiting" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 52);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52 }) == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52, .shared = true }) == .stored, @src());
    try ensure(settle(h).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

pub fn markWhileWriting(h: *H) void {
    const ids = h.rows.get(0xdead).?; // the ids stashed as bytes
    const s: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, ids));
    _ = keepIds(h, s, .{ .lineage = lin(53), .source = 53, .shared = true });
}

pub fn checkSharedDuringWrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "during" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 53);
    defer gpa.free(ids);
    try h.rows.put(gpa, 0xdead, try gpa.dupe(u8, std.mem.sliceAsBytes(ids)));
    h.hk.action_step = .written;
    h.hk.action_suffix = ".ids";
    h.hk.action = markWhileWriting;
    const key = try write(h, ids, .{ .lineage = lin(53), .source = 53 });
    try ensure(h.hk.action == null, @src()); // (it ran)
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

pub fn dropDuring(h: *H) void {
    h.s.forget(source_api.drop_key, &.{}, &.{});
}

pub fn checkDropDuringRewrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "resurrect" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 54);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(54), .source = 54 });
    source_api.drop_key = key;
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = dropDuring;
    try ensure(keepIds(h, ids, .{ .lineage = lin(54), .source = 54, .shared = true }) == .stored, @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    try ensure(h.hk.action == null and !h.s.points.contains(key), @src());
    var nb: [96]u8 = undefined;
    try ensure(!fileExists(h.s.points_fd, pointFile(&nb, key, ".man")), @src());
}

pub fn slowRewrite(_: *H) void {
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
}

pub fn checkDrainWaitsRewrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "drainrw" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 55);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(55), .source = 55 });
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = slowRewrite;
    try ensure(keepIds(h, ids, .{ .lineage = lin(55), .source = 55, .shared = true }) == .stored, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
}

pub fn linked(h: *H) ![3]Digest {
    const gpa = h.gpa;
    const ia = try idsOf(gpa, 2 * block_tokens, 61);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 62);
    defer gpa.free(ib);
    const more = try idsOf(gpa, block_tokens, 63);
    defer gpa.free(more);
    const ic = try cat(gpa, ia, more);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(1), .source = 60 });
    const kb = try write(h, ib, .{ .lineage = lin(2), .source = 60 });
    const kc = try write(h, ic, .{ .lineage = lin(1), .source = 60 });
    return .{ ka, kb, kc };
}

pub fn checkForgetLost(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lost" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    try ensure(h.s.counts().dedup_blocks == 2, @src());
    for (0..2) |i| try ensure(h.s.have.get(pa.blocks[i].name).?.ino == h.s.have.get(pb.blocks[i].name).?.ino, @src());
    const pb_names = [2]Digest{ pb.blocks[0].name, pb.blocks[1].name };
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    _ = c.unlinkat(h.s.blocks_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0), 0);
    try ensure(try restoreFails(h, k[0]), @src());
    try ensure(!h.s.points.contains(k[0]) and !h.s.points.contains(k[2]), @src());
    try ensure(h.s.points.contains(k[1]), @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    for (pb_names) |n| try ensure(fileExists(h.s.blocks_fd, blockFile(&nb, n)), @src());
    const got = (try restore(h, k[1])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(60)[0 .. 2 * block_tokens * bpt]), @src());
}

pub fn checkForgetDamaged(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "damaged" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    const b0 = pb.blocks[0].name;
    // C names A's blocks and is being read when A's read fails: it goes too (unlisted, freed when its read ends)
    const rc = try h.s.beginRead(k[2]);
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try ensure(fd >= 0, @src());
    const junk: [64]u8 = @splat(0xff);
    _ = c.pwrite(fd, &junk, junk.len, 0);
    _ = c.close(fd);
    try ensure(try restoreFails(h, k[0]), @src());
    const gone = !h.s.points.contains(k[0]) and !h.s.points.contains(k[1]) and !h.s.points.contains(k[2]);
    _ = h.s.endRead(rc, .cancel) catch {};
    try ensure(gone, @src());
    try ensure(h.s.counts().points == 0, @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    var nb2: [96]u8 = undefined;
    try ensure(!fileExists(h.s.blocks_fd, blockFile(&nb2, b0)), @src());
}

pub fn checkPages(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "pages" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 7, 71);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(71), .source = 71 });
    var nb: [96]u8 = undefined;
    for ([_][]const u8{ ".ids", ".st", ".man" }) |suffix| {
        const want = pointFile(&nb, key, suffix);
        var seq: [4]fsio.Event = undefined;
        var k: usize = 0;
        h.hk.mutex.lockUncancelable(io);
        for (h.hk.events.items) |e| if (std.mem.eql(u8, e.name[0..e.len], want) and (e.ev == .fsync or e.ev == .drop_pages)) {
            if (k < seq.len) seq[k] = e.ev;
            k += 1;
        };
        h.hk.mutex.unlock(io);
        try ensure(k == 2 and seq[0] == .fsync and seq[1] == .drop_pages, @src());
    }
}

pub fn checkPublishOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "order" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 72);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(72), .source = 72 });
    var nb: [96]u8 = undefined;
    const man = pointFile(&nb, key, ".man");
    var last_blk: ?usize = null;
    var blocks_sync: ?usize = null;
    var ids_ren: ?usize = null;
    var st_ren: ?usize = null;
    var points_sync: ?usize = null;
    var man_ren: ?usize = null;
    h.hk.mutex.lockUncancelable(io);
    for (h.hk.events.items, 0..) |e, i| {
        const n = e.name[0..e.len];
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".blk")) last_blk = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "blocks") and blocks_sync == null) blocks_sync = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".ids")) ids_ren = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".st")) st_ren = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "points") and points_sync == null and st_ren != null) points_sync = i;
        if (e.ev == .rename and std.mem.eql(u8, n, man)) man_ren = i;
    }
    h.hk.mutex.unlock(io);
    const all = last_blk != null and blocks_sync != null and ids_ren != null and st_ren != null and points_sync != null and man_ren != null;
    try ensure(all, @src());
    try ensure(last_blk.? < blocks_sync.? and blocks_sync.? < ids_ren.? and ids_ren.? < st_ren.? and st_ren.? < points_sync.? and points_sync.? < man_ren.?, @src());
}
