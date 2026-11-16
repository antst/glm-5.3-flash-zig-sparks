//! Host checks for format.
const source_api = @import("format.zig");
const std = source_api.std;
const Digest = source_api.Digest;
const namespace_version = source_api.namespace_version;
const header_bytes = source_api.header_bytes;
const fork_bytes = source_api.fork_bytes;
const media_bytes = source_api.media_bytes;
const trailer_bytes = source_api.trailer_bytes;
const Limits = source_api.Limits;
const Media = source_api.Media;
const picture_ids = source_api.picture_ids;
const pictureId = source_api.pictureId;
const mediaIds = source_api.mediaIds;
const Lineage = source_api.Lineage;
const Span = source_api.Span;
const chain = source_api.chain;
const pointKeyChained = source_api.pointKeyChained;
const pointKey = source_api.pointKey;
const crc32 = source_api.crc32;
const Flags = source_api.Flags;
const Manifest = source_api.Manifest;
const testing = source_api.testing;
const testNames = source_api.testNames;
const testIds = source_api.testIds;
const sampleManifest = source_api.sampleManifest;
const textManifest = source_api.textManifest;

test "block names: shared prefixes of one computation share names, forks diverge at the cut's block" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 3 * 16 + 5, 1);
    defer gpa.free(ids);
    const root: Lineage = .{ .root = @splat(7) };
    var a: [4]Span = undefined;
    try testNames(root, ids, &a);
    var b: [3]Span = undefined;
    try testNames(root, ids[0..48], &b);
    for (0..3) |i| try testing.expectEqualSlices(u8, &a[i].name, &b[i].name);
    try testing.expectEqual(@as(u32, 48), a[3].start);
    try testing.expectEqual(@as(u32, 53), a[3].end);
    // a tail's name covers its length: the same ids one shorter is another name
    var c: [4]Span = undefined;
    try testNames(root, ids[0..52], &c);
    try testing.expect(!std.mem.eql(u8, &a[3].name, &c[3].name));
    // a fork at 32: blocks before it keep their names, the block holding it and every later one are new
    var forked = try root.fork(gpa, @splat(9), 32);
    defer forked.deinit(gpa);
    var f: [4]Span = undefined;
    try testNames(forked, ids, &f);
    try testing.expectEqualSlices(u8, &a[0].name, &f[0].name);
    try testing.expectEqualSlices(u8, &a[1].name, &f[1].name);
    try testing.expect(!std.mem.eql(u8, &a[2].name, &f[2].name));
    try testing.expect(!std.mem.eql(u8, &a[3].name, &f[3].name));
    // another computation of the same ids shares nothing by name
    var o: [4]Span = undefined;
    try testNames(.{ .root = @splat(8) }, ids, &o);
    for (0..4) |i| try testing.expect(!std.mem.eql(u8, &a[i].name, &o[i].name));
}

test "full block sizes: a replicated store's every point block_bytes; an owner-sharded one's by its part" {
    const rep: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    for ([_]?u32{ null, 0, 3 }) |p| try std.testing.expectEqual(@as(?u64, 4096), rep.fullBlockBytes(p));
    var sh = rep;
    sh.foreign_block_bytes = 1024;
    sh.own_part = 1;
    try std.testing.expectEqual(@as(?u64, 4096), sh.fullBlockBytes(1));
    try std.testing.expectEqual(@as(?u64, 1024), sh.fullBlockBytes(0));
    try std.testing.expectEqual(@as(?u64, null), sh.fullBlockBytes(null));
}

test "part salt: replaced, never stacked; forks untouched" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 32, 2);
    defer gpa.free(ids);
    var l = try (Lineage{ .root = @splat(1) }).fork(gpa, @splat(2), 16);
    defer l.deinit(gpa);
    const p1 = l.inPart(1);
    const p1again = p1.inPart(1);
    const p0 = p1.inPart(0);
    try testing.expect(p1.eql(p1again));
    try testing.expect(!p1.eql(p0));
    try testing.expect(p0.eql(l.inPart(0)));
    var x: [2]Span = undefined;
    var y: [2]Span = undefined;
    var z: [2]Span = undefined;
    try testNames(p1, ids, &x);
    try testNames(p1again, ids, &y);
    try testNames(l, ids, &z);
    try testing.expectEqualSlices(u8, &x[0].name, &y[0].name);
    try testing.expect(!std.mem.eql(u8, &x[0].name, &z[0].name)); // an unsharded store's names are unchanged
}

test "chain and point key" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 40, 3);
    defer gpa.free(ids);
    var c1: std.ArrayList(Digest) = .empty;
    defer c1.deinit(gpa);
    try chain(gpa, ids, 16, &c1);
    try testing.expectEqual(@as(usize, 2), c1.items.len);
    var c2: std.ArrayList(Digest) = .empty;
    defer c2.deinit(gpa);
    try chain(gpa, ids[0..32], 16, &c2);
    try testing.expectEqualSlices(u8, &c1.items[1], &c2.items[1]);
    const m: Media = .{ .start = 3, .len = 4, .digest = @splat(5) };
    const k1 = pointKey(ids, 16, &.{});
    const k2 = pointKey(ids, 16, &.{m});
    try testing.expect(!std.mem.eql(u8, &k1, &k2));
    var m2 = m;
    m2.digest[0] = 6;
    try testing.expect(!std.mem.eql(u8, &k2, &pointKey(ids, 16, &.{m2})));
    try testing.expectEqualSlices(u8, &k2, &pointKeyChained(c1.items, ids, 16, &.{m}));
    try testing.expectEqualSlices(u8, &k1, &pointKeyChained(c1.items, ids, 16, &.{}));
    // a tail's ids count: one id changed after the last whole block is another key
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[39] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    changed[39] -%= 1;
    changed[3] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(ids[0..39], 16, &.{})));
}

test "manifest round trip and bounded parse" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(0xab);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    defer back.deinit(gpa);
    try testing.expectEqual(m.at, back.at);
    try testing.expectEqual(m.group, back.group);
    try testing.expect(back.flags.shared and back.flags.has_raw);
    try testing.expect(back.lineage.eql(m.lineage));
    try testing.expectEqual(@as(u32, 32), back.origin.?.at);
    try testing.expectEqualStrings("raw", back.sections[1].name.slice());
    try testing.expect(back.media[0].eql(m.media[0]));
    try testing.expectEqual(@as(u64, 100), back.blocks[2].bytes);

    // another store, another geometry: foreign; any flipped byte: corrupt (the trailer CRC), never a crash
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, @splat(0xac), l));
    var l2 = l;
    l2.block_tokens = 32;
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, compat, l2));
    const bad = try gpa.dupe(u8, raw);
    defer gpa.free(bad);
    for (0..bad.len) |i| {
        bad[i] ^= 0x41;
        if (Manifest.parse(gpa, bad, compat, l)) |*ok| {
            var x = ok.*;
            x.deinit(gpa);
            return error.TestUnexpectedResult;
        } else |_| {}
        bad[i] ^= 0x41;
    }
    // truncation at every length is refused before allocation
    for (0..raw.len) |n| try testing.expect(std.meta.isError(Manifest.parse(gpa, raw[0..n], compat, l)));
    // a count past the limits is refused even when the bytes would add up
    var l3 = l;
    l3.max_media = 0;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l3));
    // a full block of another size is damage
    var l4 = l;
    l4.block_bytes = 8192;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l4));
}

test "a huge count in a forged header with a valid CRC is refused before allocating" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(1);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    // n_media at offset 8+4+4+32+32+4+4+4 = 92
    std.mem.writeInt(u32, raw[92..96], 0xffff_fff0, .little);
    const body = raw.len - trailer_bytes;
    std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
    var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    try testing.expectError(error.Corrupt, Manifest.parse(failing.allocator(), raw, compat, l));
}

test "manifest versions: a text point is written and read as version 2, a point with pictures as version 3" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5a);
    // text: version 2, 40-byte media entries never written, the bytes those of the format before version 3
    var t = try textManifest(gpa, l);
    defer t.deinit(gpa);
    const traw = try t.encode(gpa, compat);
    defer gpa.free(traw);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, traw[8..12], .little));
    var tback = try Manifest.parse(gpa, traw, compat, l);
    tback.deinit(gpa);
    // pictures: version 3, 48-byte entries (kind, scheme)
    var p = try sampleManifest(gpa, l);
    defer p.deinit(gpa);
    const praw = try p.encode(gpa, compat);
    defer gpa.free(praw);
    try testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, praw[8..12], .little));
    try testing.expectEqual(traw.len + media_bytes, praw.len);
    var pback = try Manifest.parse(gpa, praw, compat, l);
    defer pback.deinit(gpa);
    try testing.expect(pback.media[0].eql(p.media[0]));
    // the compat digest names the namespace, not the manifest version: version 3 keeps a store's text points
    try testing.expectEqual(@as(u32, 3), namespace_version);

    const Fix = struct {
        pub fn crc(raw: []u8) void {
            const body = raw.len - trailer_bytes;
            std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
        }
    };
    const bad = try gpa.dupe(u8, praw);
    defer gpa.free(bad);
    // a version 2 manifest with media (the scheme of its ids unknown): foreign, even with its CRC made good
    std.mem.writeInt(u32, bad[8..12], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    // a version 3 manifest without media, or without raw ids: corrupt
    const t3 = try gpa.dupe(u8, traw);
    defer gpa.free(t3);
    std.mem.writeInt(u32, t3[8..12], 3, .little);
    Fix.crc(t3);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, t3, compat, l));
    @memcpy(bad, praw);
    const flags_at = 8 + 4 + 4 + 32 + 32 + 4 + 4;
    var fl: Flags = @bitCast(std.mem.readInt(u32, bad[flags_at..][0..4], .little));
    fl.has_raw = false;
    std.mem.writeInt(u32, bad[flags_at..][0..4], @bitCast(fl), .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    // another kind or scheme: foreign; a span out of order or past the window: corrupt
    const media_at = header_bytes + p.lineage.forks.len * fork_bytes;
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 8 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 12 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 4 ..][0..4], 2000, .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at..][0..4], 37, .little); // at the cut, not before it
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
}

test "manifest with pictures: two overlapping spans are refused, a span straddling the cut is not" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5b);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    gpa.free(m.media);
    const media = try gpa.alloc(Media, 2);
    m.media = media;
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    media[1] = .{ .start = 30, .len = 20, .digest = @splat(8) }; // [30, 50) runs past the cut at 37
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    back.deinit(gpa);
    media[1].start = 5; // inside the first span
    const raw2 = try m.encode(gpa, compat);
    defer gpa.free(raw2);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw2, compat, l));
}

test "manifest fuzz: random edits of a version 3 manifest never crash and never parse into an inconsistent point" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5c);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var prng = std.Random.DefaultPrng.init(0x7069637475726573);
    const rnd = prng.random();
    const buf = try gpa.alloc(u8, raw.len + 64);
    defer gpa.free(buf);
    var parsed: usize = 0;
    for (0..20_000) |_| {
        // a copy, a few random bytes changed (in the media entries half the time), a random length, the CRC made...
        var n = raw.len;
        @memcpy(buf[0..n], raw);
        const media_at = header_bytes + m.lineage.forks.len * fork_bytes;
        for (0..rnd.intRangeAtMost(usize, 1, 4)) |_| {
            const i = if (rnd.boolean()) media_at + rnd.uintLessThan(usize, media_bytes) else rnd.uintLessThan(usize, n);
            buf[i] = rnd.int(u8);
        }
        if (rnd.uintLessThan(u8, 8) == 0) n = rnd.uintLessThan(usize, raw.len + 64);
        if (n >= trailer_bytes and rnd.boolean()) {
            const body = n - trailer_bytes;
            std.mem.writeInt(u32, buf[body..][0..4], crc32(buf[0..body]), .little);
            std.mem.writeInt(u32, buf[body + 4 ..][0..4], 0, .little);
        }
        var got = Manifest.parse(gpa, buf[0..n], compat, l) catch continue;
        defer got.deinit(gpa);
        parsed += 1;
        // whatever parses is consistent: spans in order inside the point and the window, version and media agree
        var end: u64 = 0;
        for (got.media) |x| {
            try testing.expect(x.len > 0 and x.start >= end and x.start < got.at and @as(u64, x.start) + x.len <= l.max_tokens);
            end = @as(u64, x.start) + x.len;
        }
        try testing.expect(got.media.len == 0 or got.flags.has_raw);
        try testing.expect(got.at > 0 and got.at <= l.max_tokens and got.blocks.len == std.math.divCeil(u32, got.at, 16) catch unreachable);
    }
    try testing.expect(parsed > 0); // (the CRC-made-good edits of unchecked bytes, e.g. a digest, still parse)
}

test "media ids: keyed ids derive from the digest, raw ids hold model ids there and equal the keyed ones elsewhere" {
    const d: Digest = @splat(3);
    var keyed = [_]u32{ 5, 6, 0, 0, 0, 7, 0, 0 };
    var raw = [_]u32{ 5, 6, 99, 99, 99, 7, 99, 99 };
    for (2..5) |i| keyed[i] = pictureId(&d, @intCast(i - 2));
    for (6..8) |i| keyed[i] = pictureId(&d, @intCast(i - 6)); // a second picture straddling the end
    const m = [_]Media{ .{ .start = 2, .len = 3, .digest = d }, .{ .start = 6, .len = 10, .digest = d } };
    try testing.expect(mediaIds(&keyed, &raw, &m));
    try testing.expect(!mediaIds(&keyed, null, &m)); // pictures need their raw ids
    try testing.expect(!mediaIds(&keyed, &raw, m[0..1])); // a picture the media does not name
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[1], m[0] })); // out of order
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[0], .{ .start = 4, .len = 1, .digest = d } })); // overlapping
    var r2 = raw;
    r2[0] = 4;
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // raw differs outside a picture
    r2 = raw;
    r2[3] = keyed[3];
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // a picture id where a model id belongs
    var k2 = keyed;
    k2[3] +%= 1;
    try testing.expect(!mediaIds(&k2, &raw, &m)); // an id not made from the digest
    try testing.expect(mediaIds(&.{ 1, 2, 3 }, null, &.{})); // text
    try testing.expect(!mediaIds(&.{ 1, picture_ids, 3 }, null, &.{})); // a picture id without its media
}
