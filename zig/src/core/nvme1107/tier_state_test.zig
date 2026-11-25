//! The actual tier adapter and disk workers, with immutable family images in host memory.
const std = @import("std");
const tier = @import("tier_state.zig");
const fixture = @import("spill/store_test.zig");
const spill = @import("spill/root.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const t = std.testing;

test "NVMe adapter: coalesced progress reuses one delayed worker slot without a host command per block" {
    const Slow = struct {
        fn event(_: ?*anyopaque, ev: @import("spill/fsio.zig").Event, _: []const u8) void {
            if (ev == .write) std.Io.sleep(io, .fromMilliseconds(20), .awake) catch {};
        }
    };
    const h = try fixture.H.init(a, .{ .tag = "coalesced-progress", .write_slots = 1 });
    defer h.deinit();
    h.s.cfg.hooks.event = Slow.event;
    var m: Memory = .{};
    defer m.deinit();
    const at = 8 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 117);
    defer a.free(ids);
    try m.source(1, at);
    const expected = try a.dupe(u8, m.map.get(1).?.bytes);
    defer a.free(expected);
    var side = tier.Side.init(a, io, h.s, m.image(), 1, 1024);
    defer side.deinit();
    side.defer_releases = true;
    try side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = "wait-loop", .lineage = .{ .root = @splat(117) } });
    Memory.drop(&m, 1);
    if (@hasDecl(tier.Side, "pumpWait")) side.pumpWait(100 * std.time.ns_per_ms) else side.pump(100 * std.time.ns_per_ms);
    try t.expect(h.s.counts().staged_blocks >= 2);
    try settle(&side);
    try t.expect(m.map.contains(1));
    try side.release(1);
    try t.expect(!m.map.contains(1));
    const key = spill.format.pointKey(ids, fixture.block_tokens, &.{});
    try h.reopen();
    side.store = h.s;
    try side.beginRead(key, 2, at, "wait-loop");
    try t.expectEqual(tier.Status.ready, try readUntil(&side));
    try side.commitRead();
    try t.expectEqualSlices(u8, expected, m.map.get(2).?.bytes);
}

const Memory = struct {
    const Entry = struct { bytes: []u8, pins: u32 = 0, dropped: bool = false, sealed: bool = false };
    map: std.AutoHashMapUnmanaged(u64, Entry) = .empty,
    copies: usize = 0,

    fn self(p: *anyopaque) *Memory {
        return @ptrCast(@alignCast(p));
    }
    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(tier.Part), alloc: std.mem.Allocator) !void {
        try out.appendSlice(alloc, &.{ .{ .bytes = 300 }, .{ .bytes = @as(u64, at) * fixture.bpt, .row = fixture.bpt }, .{ .bytes = 200 } });
    }
    fn take(p: *anyopaque, id: u64, ranges: []const tier.Range, dst: []u8) !void {
        const m = self(p);
        const src = (m.map.get(id) orelse return error.SourceGone).bytes;
        var off: usize = 0;
        for (ranges) |r| {
            if (r.off > src.len or r.len > src.len - r.off or r.len > dst.len - off) return error.BadRange;
            @memcpy(dst[off..][0..@intCast(r.len)], src[@intCast(r.off)..][0..@intCast(r.len)]);
            off += @intCast(r.len);
        }
        if (off != dst.len) return error.BadRange;
        m.copies += 1;
    }
    fn put(p: *anyopaque, id: u64, ranges: []const tier.Range, src: []const u8) !void {
        const m = self(p);
        const e = m.map.getPtr(id) orelse return error.SourceGone;
        if (!e.sealed) return error.Unsealed;
        var off: usize = 0;
        for (ranges) |r| {
            if (r.off > e.bytes.len or r.len > e.bytes.len - r.off or r.len > src.len - off) return error.BadRange;
            @memcpy(e.bytes[@intCast(r.off)..][0..@intCast(r.len)], src[off..][0..@intCast(r.len)]);
            off += @intCast(r.len);
        }
        if (off != src.len) return error.BadRange;
        m.copies += 1;
    }
    fn pin(p: *anyopaque, id: u64) !void {
        const e = self(p).map.getPtr(id) orelse return error.SourceGone;
        e.pins += 1;
    }
    fn unpin(p: *anyopaque, id: u64) void {
        const m = self(p);
        const e = m.map.getPtr(id).?;
        std.debug.assert(e.pins > 0);
        e.pins -= 1;
        if (e.pins == 0 and e.dropped) m.remove(id);
    }
    fn blank(p: *anyopaque, id: u64, at: u32) !void {
        const m = self(p);
        if (m.map.contains(id)) return error.Duplicate;
        const bytes = try a.alloc(u8, 500 + @as(usize, at) * fixture.bpt);
        errdefer a.free(bytes);
        @memset(bytes, 0xee);
        try m.map.put(a, id, .{ .bytes = bytes });
    }
    fn seal(p: *anyopaque, id: u64, small: []const u8) !void {
        if (small.len < 8 or !std.mem.eql(u8, small[0..8], "GLMTEST1")) return error.BadWords;
        const e = self(p).map.getPtr(id) orelse return error.SourceGone;
        e.sealed = true;
    }
    fn drop(p: *anyopaque, id: u64) void {
        const m = self(p);
        const e = m.map.getPtr(id) orelse return;
        e.dropped = true;
        if (e.pins == 0) m.remove(id);
    }
    fn remove(m: *Memory, id: u64) void {
        a.free(m.map.fetchRemove(id).?.value.bytes);
    }
    fn deinit(m: *Memory) void {
        var it = m.map.valueIterator();
        while (it.next()) |e| {
            std.debug.assert(e.pins == 0);
            a.free(e.bytes);
        }
        m.map.deinit(a);
    }
    fn image(m: *Memory) tier.Image {
        return .{ .ptr = m, .parts = parts, .take = take, .put = put, .pin = pin, .unpin = unpin, .blank = blank, .seal = seal, .drop = drop };
    }
    fn source(m: *Memory, id: u64, at: u32) !void {
        try blank(m, id, at);
        const e = m.map.getPtr(id).?;
        for (e.bytes, 0..) |*v, i| v.* = @truncate(i *% 13 +% id);
        @memcpy(e.bytes[0..8], "GLMTEST1");
        e.sealed = true;
    }
};

fn settle(s: *tier.Side) !void {
    const end = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (s.pending()) {
        if (std.Io.Clock.awake.now(io).toNanoseconds() >= end) return error.WriterTimeout;
        s.pump(std.time.ns_per_ms);
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn readUntil(s: *tier.Side) !tier.Status {
    const end = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (true) {
        const status = s.pollRead();
        if (status != .pending) return status;
        if (std.Io.Clock.awake.now(io).toNanoseconds() >= end) return error.ReaderTimeout;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "NVMe adapter: a cache eviction cannot recycle a queued source; restart restores every byte" {
    const h = try fixture.H.init(a, .{ .tag = "image-adapter" });
    defer h.deinit();
    var m: Memory = .{};
    defer m.deinit();
    const at = 4 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 41);
    defer a.free(ids);
    try m.source(1, at);
    const want = try a.dupe(u8, m.map.get(1).?.bytes);
    defer a.free(want);
    const meta = "canonical-spans-and-chunk-starts";
    var side = tier.Side.init(a, io, h.s, m.image(), 2, 1024);
    defer side.deinit();
    try side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = meta, .lineage = .{ .root = @splat(42) } });
    try t.expectEqual(@as(usize, 0), m.copies); // a mark must not call CUDA or issue peer work
    Memory.drop(&m, 1);
    try t.expect(m.map.contains(1));
    try settle(&side);
    try t.expect(!m.map.contains(1));
    const key = spill.format.pointKey(ids, fixture.block_tokens, &.{});
    try t.expect(h.s.presence(key));
    try h.reopen();
    side.store = h.s;
    try side.beginRead(key, 7, at, meta);
    try t.expectEqual(tier.Status.ready, try readUntil(&side));
    try t.expectEqualSlices(u8, want, m.map.get(7).?.bytes);
    try side.commitRead();
    try t.expectEqual(@as(u64, 1), side.counts.restores);
}

test "NVMe adapter: a different row-arithmetic map is refused before copying any payload" {
    const h = try fixture.H.init(a, .{ .tag = "image-modes" });
    defer h.deinit();
    var m: Memory = .{};
    defer m.deinit();
    const at = 2 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 42);
    defer a.free(ids);
    try m.source(1, at);
    var side = tier.Side.init(a, io, h.s, m.image(), 2, 1024);
    defer side.deinit();
    try side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = "prompt-route", .lineage = .{ .root = @splat(43) } });
    try settle(&side);
    const copies = m.copies;
    try side.beginRead(spill.format.pointKey(ids, fixture.block_tokens, &.{}), 8, at, "decode-route");
    try t.expectEqual(tier.Status.failed, try readUntil(&side));
    try t.expectEqual(error.NvmeArithmetic, side.read.?.failed.?);
    try t.expectEqual(copies, m.copies);
    try t.expectError(error.NvmeNotReady, side.commitRead());
    side.abortRead();
    try t.expect(!m.map.contains(8));
}

test "NVMe adapter: backlog and metadata are bounded; cancellation releases each source once" {
    const h = try fixture.H.init(a, .{ .tag = "image-bounds" });
    defer h.deinit();
    var m: Memory = .{};
    defer m.deinit();
    const at = fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 43);
    defer a.free(ids);
    try m.source(1, at);
    try m.source(2, at);
    var side = tier.Side.init(a, io, h.s, m.image(), 1, 8);
    defer side.deinit();
    const note: tier.Note = .{ .id = 1, .at = at, .tokens = ids, .metadata = "mode", .lineage = .{ .root = @splat(44) } };
    var bad = note;
    bad.metadata = "more than eight bytes";
    try t.expectError(error.NvmeBacklog, side.enqueue(bad));
    try side.enqueue(note);
    bad = note;
    bad.id = 2;
    try t.expectError(error.NvmeBacklog, side.enqueue(bad));
    Memory.drop(&m, 1);
    side.stop(0);
    try t.expect(!side.pending());
    try t.expect(!m.map.contains(1));
    try t.expectEqual(@as(u32, 0), m.map.get(2).?.pins);
    try t.expect(!h.s.presence(spill.format.pointKey(ids, fixture.block_tokens, &.{})));
}

test "NVMe adapter: a fast rank retains its evicted source until the delayed rank returns its credit" {
    const h0 = try fixture.H.init(a, .{ .tag = "credit-fast" });
    defer h0.deinit();
    const h1 = try fixture.H.init(a, .{ .tag = "credit-slow" });
    defer h1.deinit();
    var m0: Memory = .{};
    defer m0.deinit();
    var m1: Memory = .{};
    defer m1.deinit();
    var s0 = tier.Side.init(a, io, h0.s, m0.image(), 1, 1024);
    defer s0.deinit();
    var s1 = tier.Side.init(a, io, h1.s, m1.image(), 1, 1024);
    defer s1.deinit();
    s0.defer_releases = true;
    s1.defer_releases = true;
    const at = 4 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 91);
    defer a.free(ids);
    const note: tier.Note = .{ .id = 1, .at = at, .tokens = ids, .metadata = "route", .lineage = .{ .root = @splat(91) } };
    try m0.source(1, at);
    try m1.source(1, at);
    try s0.enqueue(note);
    try s1.enqueue(note);
    Memory.drop(&m0, 1);
    Memory.drop(&m1, 1);
    try settle(&s0); // the other rank has not even staged its first byte
    try t.expect(s0.completed(1) and !s1.completed(1));
    try t.expectEqual(@as(u32, 1), m0.map.get(1).?.pins);
    try t.expectError(error.NvmeNotReleased, s1.release(1));
    try m0.source(2, at);
    var next = note;
    next.id = 2;
    try t.expectError(error.NvmeBacklog, s0.enqueue(next)); // held credits consume the bounded queue
    try settle(&s1);
    try t.expect(s0.completed(1) and s1.completed(1));
    try s0.release(1);
    try s1.release(1);
    try t.expect(!m0.map.contains(1) and !m1.map.contains(1));
    try t.expectError(error.NvmeNotReleased, s0.release(1));
    try s0.enqueue(next);
    s0.stop(0);
    try t.expectEqual(@as(u32, 0), m0.map.get(2).?.pins);
}

test "NVMe adapter: the lookahead token keys the checkpoint but adds no state row" {
    const h = try fixture.H.init(a, .{ .tag = "key-lookahead" });
    defer h.deinit();
    var m: Memory = .{};
    defer m.deinit();
    var side = tier.Side.init(a, io, h.s, m.image(), 2, 1024);
    defer side.deinit();
    side.lookahead = 1;
    const at = 4 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at + 1, 92);
    defer a.free(ids);
    try m.source(1, at);
    const want = try a.dupe(u8, m.map.get(1).?.bytes);
    defer a.free(want);
    try side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = "route", .lineage = .{ .root = @splat(92) } });
    try settle(&side);
    const key = spill.format.pointKey(ids, fixture.block_tokens, &.{});
    const hit = h.s.lookup(.{ .keyed = ids, .max_at = at + 1, .longer_than = 0 }).?;
    try t.expectEqual(at + 1, hit.at);
    ids[at] +%= 1;
    try t.expect(h.s.lookup(.{ .keyed = ids, .max_at = at + 1, .longer_than = 0 }) == null);
    ids[at] -%= 1;
    try side.beginRead(key, 2, at, "route");
    try t.expectError(error.NvmeNotReady, side.commitRead());
    try t.expectEqual(tier.Status.ready, try readUntil(&side));
    try t.expectEqualSlices(u8, want, m.map.get(2).?.bytes);
    side.abortRead(); // ready alone never publishes the restored state
    try t.expect(!m.map.contains(2));
    try side.beginRead(key, 3, at, "route");
    try t.expectEqual(tier.Status.ready, try readUntil(&side));
    try side.commitRead();
    try t.expectEqualSlices(u8, want, m.map.get(3).?.bytes);
}

test "NVMe adapter: storage corruption is unlisted before abort, while a route mismatch preserves its point" {
    const h = try fixture.H.init(a, .{ .tag = "corrupt-adapter" });
    defer h.deinit();
    var m: Memory = .{};
    defer m.deinit();
    var side = tier.Side.init(a, io, h.s, m.image(), 2, 1024);
    defer side.deinit();
    const at = 4 * fixture.block_tokens;
    const ids = try spill.format.testIds(a, at, 93);
    defer a.free(ids);
    try m.source(1, at);
    try side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = "route", .lineage = .{ .root = @splat(93) } });
    try settle(&side);
    const key = spill.format.pointKey(ids, fixture.block_tokens, &.{});
    try side.beginRead(key, 2, at, "other-route");
    try t.expectEqual(tier.Status.failed, try readUntil(&side));
    side.abortRead();
    try t.expect(h.s.presence(key));
    const block = h.s.points.get(key).?.blocks[0].name;
    var filename: [80]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&filename, "{s}.blk", .{spill.format.hex(&block)}, 0);
    const fd = std.c.openat(h.s.blocks_fd, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try t.expect(fd >= 0);
    const junk: [64]u8 = @splat(0xff);
    const wrote = std.c.pwrite(fd, &junk, junk.len, 0);
    _ = std.c.close(fd);
    try t.expectEqual(@as(isize, junk.len), wrote);
    try side.beginRead(key, 3, at, "route");
    try t.expectEqual(tier.Status.failed, try readUntil(&side));
    try t.expectEqual(error.NvmeRead, side.read.?.failed.?);
    side.forgetRead(key);
    side.abortRead();
    try t.expect(!h.s.presence(key));
    try t.expect(!m.map.contains(3));
}

test "NVMe adapter: serial staging keeps two queued checkpoints without a second writer" {
    const Slow = struct {
        fn event(_: ?*anyopaque, ev: @import("spill/fsio.zig").Event, _: []const u8) void {
            if (ev == .write) std.Io.sleep(io, .fromMilliseconds(125), .awake) catch {};
        }
    };
    const h = try fixture.H.init(a, .{ .tag = "serial-queued-marks", .write_slots = 1 });
    defer h.deinit();
    h.s.cfg.hooks.event = Slow.event;
    var m: Memory = .{};
    defer m.deinit();
    var side = tier.Side.init(a, io, h.s, m.image(), 2, 1024);
    defer side.deinit();
    side.serial_writes = true;
    side.defer_releases = true;

    const at = 2 * fixture.block_tokens;
    const first = try spill.format.testIds(a, at, 418);
    defer a.free(first);
    const second = try spill.format.testIds(a, at, 419);
    defer a.free(second);
    try m.source(1, at);
    try m.source(2, at);
    const want = try a.dupe(u8, m.map.get(2).?.bytes);
    defer a.free(want);
    try side.enqueue(.{ .id = 1, .at = at, .tokens = first, .metadata = "first", .lineage = .{ .root = @splat(18) } });
    try side.enqueue(.{ .id = 2, .at = at, .tokens = second, .metadata = "second", .lineage = .{ .root = @splat(19) } });
    try t.expectError(error.NvmeBacklog, side.enqueue(.{ .id = 3, .at = at, .tokens = second, .metadata = "extra", .lineage = .{ .root = @splat(20) } }));
    Memory.drop(&m, 1);
    Memory.drop(&m, 2);

    side.pump(25 * std.time.ns_per_ms);
    try t.expectEqual(@as(usize, 2), side.notes.items.len);
    try t.expect(side.notes.items[0].job != null and side.notes.items[1].job == null);
    try t.expectEqual(@as(u64, 1), h.s.counts().pending_jobs);
    try settle(&side);
    try t.expect(side.completed(1) and side.completed(2));
    try t.expectEqual(@as(u64, 2), h.s.counts().committed);
    try t.expectEqual(@as(u64, 0), side.counts.refused);
    try side.release(1);
    try side.release(2);
    try t.expect(!m.map.contains(1) and !m.map.contains(2));

    const key = spill.format.pointKey(second, fixture.block_tokens, &.{});
    try h.reopen();
    side.store = h.s;
    try side.beginRead(key, 3, at, "second");
    try t.expectEqual(tier.Status.ready, try readUntil(&side));
    try side.commitRead();
    try t.expectEqualSlices(u8, want, m.map.get(3).?.bytes);
}
