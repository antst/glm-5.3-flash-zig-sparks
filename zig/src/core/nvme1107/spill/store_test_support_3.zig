//! Shared implementation helpers for store.
const source_api = @import("store_test.zig");
const std = source_api.std;
const format = source_api.format;
const store = source_api.store;
const Allocator = source_api.Allocator;
const Digest = source_api.Digest;
const block_tokens = source_api.block_tokens;
const ensure = source_api.ensure;
const H = source_api.H;
const keepIds = source_api.keepIds;
const write = source_api.write;
const idsOf = source_api.idsOf;
const lin = source_api.lin;
const c = source_api.c;
const CheckFailed = source_api.CheckFailed;
const pointFile = source_api.pointFile;
const fsio = source_api.fsio;
const bpt = source_api.bpt;
const settle = source_api.settle;
const restore = source_api.restore;
const countFiles = source_api.countFiles;
const waitUntil = source_api.waitUntil;
const Store = source_api.Store;
const io = source_api.io;
const block_bytes = source_api.block_bytes;
const max_tokens = source_api.max_tokens;
const blockFile = source_api.blockFile;
const checks = source_api.checks;

pub fn checkLookup(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lookup", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(gpa, 5 * block_tokens, 1700);
    defer gpa.free(ids);
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(1) }};
    const raw = try gpa.dupe(u32, ids);
    defer gpa.free(raw);
    for (3..7) |i| { // the picture's keyed ids made from its digest, its raw ids the placeholder
        ids[i] = format.pictureId(&m[0].digest, @intCast(i - 3));
        raw[i] = 99;
    }
    // a point whose ids do not agree with its media is refused at keep (and would be at recovery)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m }) == .invalid, @src());
    var forged = m;
    forged[0].digest[20] ^= 1; // (the first 8 bytes seed the ids: the same ids, another picture past them)
    const kf = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &forged, .raw = raw });
    var seeded = m;
    seeded[0].digest[0] ^= 1; // (a seed byte changed: its ids no longer derive from it)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &seeded, .raw = raw }) == .invalid, @src());
    var placeholder = try gpa.dupe(u32, raw);
    defer gpa.free(placeholder);
    placeholder[4] = ids[4]; // a raw id inside the picture that is a picture's keyed id, not a model id
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = placeholder }) == .invalid, @src());
    const k2 = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const k4 = try write(h, ids[0 .. 4 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const q = struct {
        pub fn at(hh: *H, x: store.Query) ?Digest {
            return if (hh.s.lookup(x)) |hit| hit.key else null;
        }
    };
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50 }), k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50, .longer_than = 40 }) == null, @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 80, .grid = 64 }), k4), @src());
    var other = m;
    other[0].digest[0] = 2;
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
    // the same shortened ids from another payload (only the full digest tells them apart): its own point, never m's
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(!std.meta.eql(kf, k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &.{}, .max_at = 80 }) == null, @src());
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[20] +%= 1;
    try ensure(std.meta.eql(q.at(h, .{ .keyed = changed, .media = &m, .max_at = 80 }), null), @src());
    // across a restart: the points with pictures recover (their ids checked against their media) and match as before
    try h.reopen();
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
}

pub fn rewritePointFile(h: *H, key: Digest, suffix: []const u8, bytes: []const u8) !void {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, suffix)}, 0);
    const fd = c.openat(h.s.points_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0) return CheckFailed;
    defer _ = c.close(fd);
    if (c.pwrite(fd, bytes.ptr, bytes.len, 0) != @as(isize, @intCast(bytes.len))) return CheckFailed;
}

pub fn checkPictureRecovery(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "picrecover", .turns = 0 });
    defer h.deinit();
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(7) }};
    const n = 2 * block_tokens;
    var keys: [3]Digest = undefined;
    var keyed: [3][]u32 = undefined;
    for (0..3) |i| {
        keyed[i] = try idsOf(gpa, n, 1900 + i);
        const raw = try gpa.dupe(u32, keyed[i]);
        defer gpa.free(raw);
        for (3..7) |j| {
            keyed[i][j] = format.pictureId(&m[0].digest, @intCast(j - 3));
            raw[j] = 99;
        }
        keys[i] = try write(h, keyed[i], .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i, .media = &m, .raw = raw });
    }
    defer for (keyed) |k| gpa.free(k);
    for (0..2) |i| {
        var nb: [96]u8 = undefined;
        var zb: [96]u8 = undefined;
        const ids_raw = try fsio.readSmall(gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".ids")}, 0), 1 << 20);
        defer gpa.free(ids_raw);
        try ensure(ids_raw.len >= 2 * n * 4, @src());
        const raw_ids: []align(1) u32 = std.mem.bytesAsSlice(u32, ids_raw[n * 4 ..][0 .. n * 4]);
        if (i == 0) raw_ids[4] = keyed[i][4] else raw_ids[10] +%= 1; // a picture id inside, another model id outside
        try rewritePointFile(h, keys[i], ".ids", ids_raw);
        const man = try fsio.readSmall(gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".man")}, 0), 1 << 20);
        defer gpa.free(man);
        var mf = try format.Manifest.parse(gpa, man, h.cfg.compat, h.cfg.limits);
        defer mf.deinit(gpa);
        const same = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(same);
        try ensure(std.mem.eql(u8, same, man), @src()); // (the re-encoding is the manifest's own bytes)
        mf.ids_crc = format.crc32(ids_raw);
        const fixed = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(fixed);
        try rewritePointFile(h, keys[i], ".man", fixed);
    }
    try h.reopen();
    try ensure(h.s.counts().recovered_dropped == 2, @src());
    try ensure(!h.s.presence(keys[0]) and !h.s.presence(keys[1]) and h.s.presence(keys[2]), @src());
    for (0..2) |i| try ensure(h.s.lookup(.{ .keyed = keyed[i], .media = &m, .max_at = 80 }) == null, @src());
    const hit = h.s.lookup(.{ .keyed = keyed[2], .media = &m, .max_at = 80 }) orelse return CheckFailed;
    try ensure(std.mem.eql(u8, &hit.key, &keys[2]), @src());
}

pub fn checkWriteFaults(gpa: Allocator) !void {
    {
        const h = try H.init(gpa, .{ .tag = "enospc" });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens, 1800);
        defer gpa.free(ids);
        h.hk.nospace_writes = 1;
        try ensure(keepIds(h, ids, .{ .lineage = lin(80), .source = 80 }) == .queued, @src());
        _ = settle(h);
        var comp: [4]store.Completion = undefined;
        const n = h.s.poll(&comp);
        try ensure(n == 1 and comp[0] == .failed and comp[0].failed.why == .no_space, @src());
        try ensure(!h.s.presence(format.pointKey(ids, block_tokens, &.{})), @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, false) == 0, @src());
        try ensure(h.s.counts().staging_bytes == 0, @src());
    }
    inline for (.{ false, true }) |nd| {
        const h = try H.init(gpa, .{ .tag = "short", .no_direct = nd });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens + 9, 1801);
        defer gpa.free(ids);
        h.hk.short_intr = true;
        const key = try write(h, ids, .{ .lineage = lin(81), .source = 81 });
        h.hk.short_intr = false;
        try h.reopen();
        const got = (try restore(h, key)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(81)[0 .. ids.len * bpt]), @src());
    }
}

pub fn checkCancelRead(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cancelread", .read_slots = 3 });
    defer h.deinit();
    const ids = try idsOf(gpa, 8 * block_tokens, 1900);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(90), .source = 90 });
    const id = try h.s.beginRead(key);
    var rb: [1]store.ReadyBlock = undefined;
    _ = h.s.readAwait(id, 0, 5 * std.time.ns_per_s, .{});
    const got = h.s.readReady(id, null, &rb);
    _ = try h.s.endRead(id, .cancel);
    for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
    try ensure(waitUntil(struct {
        pub fn f(hh: *H) bool {
            const ct = hh.s.counts();
            return ct.staging_bytes == 0 and hh.s.reads.items.len == 0;
        }
    }.f, h, 3000), @src());
    try ensure(!h.s.reading.contains(key), @src());
}

pub fn checkFairReads(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fair", .read_slots = 2 });
    defer h.deinit();
    var ids: [2]store.ReadId = undefined;
    for (0..2) |i| {
        const x = try idsOf(gpa, 6 * block_tokens, 2000 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(100 + i)), .source = 100 + i });
        ids[i] = try h.s.beginRead(key);
    }
    // the first restore took both slots (the second was not there yet); the slots it gives back go to the second
    try ensure(h.s.readAwait(ids[0], 0, 5 * std.time.ns_per_s, .{}) and h.s.readAwait(ids[0], 1, 5 * std.time.ns_per_s, .{}), @src());
    var rb: [2]store.ReadyBlock = undefined;
    const got = h.s.readReady(ids[0], null, &rb);
    try ensure(got.n == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(h.s.readAwait(ids[1], 0, 5 * std.time.ns_per_s, .{}), @src());
    h.s.readRelease(rb[1].slot);
    for (ids) |id| _ = try h.s.endRead(id, .cancel);
    try ensure(waitUntil(struct {
        pub fn f(hh: *H) bool {
            return hh.s.counts().staging_bytes == 0;
        }
    }.f, h, 3000), @src());
}

pub fn checkLayout(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "layout" });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2300);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(23), .source = 23 });
    h.cfg.owner_layout = .owner_sharded;
    try h.reopen();
    try ensure(!h.s.presence(key) and h.s.counts().recovered_dropped == 1, @src());
}

pub fn checkOwnerParts(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "parts" });
    defer h.deinit();
    const foreign = block_bytes / 4;
    h.cfg.owner_layout = .owner_sharded;
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.limits.own_part = 0;
    try h.reopen();
    const ids = try idsOf(gpa, 2 * block_tokens, 2500);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(25), .source = 25 }) == .invalid, @src()); // (no part)
    const own = try write(h, ids, .{ .lineage = lin(25).inPart(0), .source = 25 });
    const other_ids = try idsOf(gpa, 3 * block_tokens, 2501);
    defer gpa.free(other_ids);
    const other = try write(h, other_ids, .{ .lineage = lin(26).inPart(1), .source = 26 });
    for (h.s.points.get(own).?.blocks) |b| try ensure(b.bytes == block_bytes, @src());
    for (h.s.points.get(other).?.blocks) |b| try ensure(b.bytes == foreign, @src());
    // the same ids from another part: other names (never the first point's blocks by name)
    const pa = h.s.points.get(own).?;
    var spans: [4]format.Span = undefined;
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(gpa);
    try format.chain(gpa, ids, block_tokens, &ch);
    format.blockNames(lin(25).inPart(1), ch.items, ids, block_tokens, spans[0..2]);
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(!std.mem.eql(u8, &b.name, &sp.name), @src());
    format.blockNames(lin(25).inPart(1).inPart(0), ch.items, ids, block_tokens, spans[0..2]); // (re-salted: replaced)
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(std.mem.eql(u8, &b.name, &sp.name), @src());
    // the restored rows are the stored sizes: a foreign block is its source's first bytes of the block's rows
    const got = (try restore(h, other)) orelse return CheckFailed;
    defer gpa.free(got);
    for (0..3) |k| try ensure(std.mem.eql(u8, got[k * block_tokens * bpt ..][0..foreign], h.src(26)[k * block_tokens * bpt ..][0..foreign]), @src());
    try h.reopen();
    try ensure(h.s.presence(own) and h.s.presence(other), @src());
    try ensure(h.s.partOf(own).?.part == 0 and h.s.partOf(other).?.part == 1 and h.s.partOf(@splat(9)) == null, @src());
    const hit = h.s.lookup(.{ .keyed = other_ids, .max_at = max_tokens }) orelse return CheckFailed;
    try ensure(hit.part == 1 and std.mem.eql(u8, &hit.key, &other), @src());
    // a rank owning part 1 with these files: the sizes are not its own, both points damaged
    h.cfg.limits.own_part = 1;
    try h.reopen();
    try ensure(!h.s.presence(own) and !h.s.presence(other) and h.s.counts().recovered_dropped == 2, @src());
    // a foreign size past the block's, or one on a replicated store, is no configuration
    h.closeStore();
    h.cfg.limits.foreign_block_bytes = block_bytes + 1;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.owner_layout = .replicated;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
}

pub fn checkKeepReuse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "reuse", .cap = 24 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 3 * block_tokens, 2400);
    defer gpa.free(a);
    const ka1 = try write(h, a[0 .. 2 * block_tokens], .{ .lineage = lin(24), .source = 24 });
    for (0..2) |i| {
        const x = try idsOf(gpa, block_tokens, 2401 + i);
        defer gpa.free(x);
        _ = try write(h, x, .{ .lineage = lin(@intCast(25 + i)), .source = 25 + i });
    }
    const before = h.s.counts();
    const ka2 = try write(h, a, .{ .lineage = lin(24), .source = 24 });
    const after = h.s.counts();
    try ensure(!h.s.points.contains(ka1) and after.pruned_conversations > before.pruned_conversations, @src()); // (the pressure)
    try ensure(after.staged_blocks - before.staged_blocks == 1 and after.stored_bytes <= h.cfg.cap_bytes, @src());
    const got = (try restore(h, ka2)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(24)[0 .. 3 * block_tokens * bpt]), @src());
}

pub fn checkHandOver(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "handover" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2600);
    defer gpa.free(ids);
    const r1 = keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(28), .source = 28 });
    try ensure(r1 == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(28), .source = 28 }) == .queued, @src());
    try ensure(h.s.cancel(r1.queued.job), @src());
    try ensure(h.s.counts().queued_bytes >= 3 * block_bytes, @src()); // the second write's 3 blocks, all reserved
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(28)[0 .. 3 * block_tokens * bpt]), @src());
    try ensure(h.s.counts().queued_bytes == 0, @src());
}

pub fn checkBadConfig(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "badcfg" });
    defer h.deinit();
    h.closeStore();
    for ([_]f64{ std.math.nan(f64), -0.1, 1.5, std.math.inf(f64) }) |q| {
        var cfg = h.cfg;
        cfg.quota = q;
        if (Store.open(gpa, io, cfg)) |s2| {
            s2.close();
            try ensure(false, @src());
        } else |e| try ensure(e == error.BadConfig, @src());
    }
}

pub fn chmodAt(dirfd: c.fd_t, name: []const u8, mode: c.mode_t) !void {
    var zb: [128]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0);
    if (c.fchmodat(dirfd, z, mode, 0) != 0) return error.Unexpected;
}

pub fn checkRecoverTransient(gpa: Allocator) !void {
    if (c.geteuid() == 0) return; // (permissions do not stop root)
    const h = try H.init(gpa, .{ .tag = "transient" });
    defer h.deinit();
    const ia = try idsOf(gpa, 2 * block_tokens + 3, 2500);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 2501);
    defer gpa.free(ib);
    const ic = try idsOf(gpa, block_tokens, 2502);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(25), .source = 250 });
    const kb = try write(h, ib, .{ .lineage = lin(26), .source = 251 });
    const kc = try write(h, ic, .{ .lineage = lin(27), .source = 252 });
    const files = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    var nb: [96]u8 = undefined;
    const b0 = (h.s.points.get(kb) orelse return CheckFailed).blocks[0].name;
    try chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0);
    try chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0);
    try h.reopen();
    const ct = h.s.counts();
    const deferred = !h.s.points.contains(ka) and !h.s.points.contains(kb) and h.s.points.contains(kc);
    const kept = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0o600) catch {}; // (gone if recovery deleted them)
    chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0o600) catch {};
    try ensure(deferred and ct.recover_deferred == 2 and ct.recovered_dropped == 0, @src());
    try ensure(kept == files, @src());
    try h.reopen();
    try ensure(h.s.points.contains(ka) and h.s.points.contains(kb) and h.s.points.contains(kc), @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(251)[0 .. 2 * block_tokens * bpt]), @src());
}

pub fn checkNamed(name: []const u8) *const fn (Allocator) anyerror!void {
    for (checks) |ch| if (std.mem.eql(u8, ch.name, name)) return ch.f;
    @panic("no such check");
}
