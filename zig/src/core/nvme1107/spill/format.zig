//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block layout, lineage and fsync order) spill tier:
pub const std = @import("std");
pub const builtin = @import("builtin");
pub const Sha256 = std.crypto.hash.sha2.Sha256;
pub const Crc32 = std.hash.Crc32;
pub const Allocator = std.mem.Allocator;

comptime {
    // ids and fields are hashed and stored as their little-endian bytes
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

pub const Digest = [32]u8;
pub const zero_digest: Digest = @splat(0);

/// The native spill format's manifest version (bumped on any byte change below). Version 3 (pictures): a point with
pub const version: u32 = 3;
/// The version a point without media is written as (and the oldest one read).
pub const text_version: u32 = 2;
/// Shared token-history references require a fresh namespace; legacy stores are never interpreted as this format.
pub const namespace_version: u32 = 3;
pub const magic = "TFSPILL\x02".*;
/// The manifest's fixed header.
pub const header_bytes = 256;
pub const fork_bytes = 20;
pub const media_bytes = 48; // version 3: start, len, kind, scheme, SHA-256
pub const media_v2_bytes = 40; // version 2 (never written with media: such a manifest is foreign)
/// A version-3 media entry's kind: a picture (an image's span of the prompt).
pub const media_picture: u32 = 1;
/// The scheme a picture's keyed ids are made with: `pictureId` (recipe 0006's ids, seeded by the digest's first...
pub const picture_scheme: u32 = 1;
pub const block_ref_bytes = 80;
pub const section_bytes = 56;
pub const trailer_bytes = 8;

/// How a rank's planes are laid out on disk. Replicated and owner-sharded stores never share a namespace (the compat
pub const OwnerLayout = enum(u32) { replicated = 0, owner_sharded = 1 };

/// Bounds every parse checks before it allocates, derived from the context window and the model descriptor.
pub const Limits = struct {
    /// Most tokens a point may hold (the context window).
    max_tokens: u32,
    /// Tokens a block (the engine's chunk rows: 2048).
    block_tokens: u32,
    /// A full block's bytes on this rank (exact: a full block of any other length is damaged).
    block_bytes: u64,
    /// Most small-state bytes a point carries.
    max_small_bytes: u64,
    /// `owner_sharded` (0118): a full block's bytes on this rank for a point whose part (`Lineage.part`) another...
    foreign_block_bytes: u64 = 0,
    /// `owner_sharded`: the part this rank owns (a point of that part has full blocks of `block_bytes`).
    own_part: u32 = 0,
    max_media: u32 = 4096,
    max_sections: u32 = 256,
    max_forks: u32 = 4096,

    /// A full block's exact bytes for a point of part `part` (its lineage's salt): `block_bytes`, or on an
    pub fn fullBlockBytes(l: Limits, part: ?u32) ?u64 {
        if (l.foreign_block_bytes == 0) return l.block_bytes;
        const p = part orelse return null;
        return if (p == l.own_part) l.block_bytes else l.foreign_block_bytes;
    }

    pub fn maxBlocks(l: Limits) u32 {
        return std.math.divCeil(u32, l.max_tokens, l.block_tokens) catch 0;
    }

    /// The largest manifest these limits admit.
    pub fn maxManifest(l: Limits) u64 {
        return header_bytes + @as(u64, l.max_forks) * fork_bytes + @as(u64, l.max_media) * media_bytes +
            @as(u64, l.maxBlocks()) * block_ref_bytes + @as(u64, l.max_sections) * section_bytes + trailer_bytes;
    }
};

/// A picture inside a point: its prompt positions and the full SHA-256 of its payload (recipe 0006's per-position
pub const Media = extern struct {
    start: u32,
    len: u32,
    digest: Digest,

    pub fn eql(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

/// Keyed ids at or above this are pictures' (`pictureId`); vocabulary ids stay below it.
pub const picture_ids: u32 = 0x8000_0000;

/// Scheme 1's keyed id for position `j` of a picture whose payload's SHA-256 is `digest` (equal to
pub fn pictureId(digest: *const Digest, j: u32) u32 {
    const seed = std.mem.readInt(u64, digest[0..8], .little);
    return picture_ids | (@as(u32, @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&j)))) & 0x7fff_ffff);
}

/// Whether a point's ids agree with its media: spans in order, not overlapping, not empty, each starting inside the
pub fn mediaIds(keyed: []const u32, raw: ?[]const u32, media: []const Media) bool {
    if (raw) |r| if (r.len != keyed.len) return false;
    if (media.len > 0 and raw == null) return false;
    var at: usize = 0;
    for (media) |m| {
        if (m.len == 0 or m.start < at or m.start >= keyed.len) return false;
        const stop: usize = @intCast(@min(@as(u64, m.start) + m.len, keyed.len));
        if (!textIds(keyed[at..m.start], if (raw) |r| r[at..m.start] else null)) return false;
        for (keyed[m.start..stop], 0..) |id, j| if (id != pictureId(&m.digest, @intCast(j))) return false;
        if (raw) |r| for (r[m.start..stop]) |id| if (id >= picture_ids) return false;
        at = @intCast(@as(u64, m.start) + m.len); // (past the ids when the picture straddles the cut: the loop ends)
    }
    if (at < keyed.len) return textIds(keyed[at..], if (raw) |r| r[at..] else null);
    return true;
}

pub fn textIds(keyed: []const u32, raw: ?[]const u32) bool {
    for (keyed) |id| if (id >= picture_ids) return false;
    if (raw) |r| if (!std.mem.eql(u32, keyed, r)) return false;
    return true;
}

/// One fork of a lineage: a computation that resumed at `at` and computed its own rows from there.
pub const Fork = struct { name: [16]u8, at: u32 };

/// The computation a state's rows belong to. Two prefills of the same ids may differ in their numerical history, so
pub const Lineage = struct {
    root: [16]u8,
    part: ?u32 = null,
    forks: []const Fork = &.{},

    /// A fresh prefill's lineage: `boot` (this process's nonce) and its own sequence number.
    pub fn fresh(boot: [16]u8, seq: u64) Lineage {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage-root");
        h.update(&boot);
        h.update(std.mem.asBytes(&seq));
        const d = h.finalResult();
        return .{ .root = d[0..16].* };
    }

    /// `l` resumed at `at` by computation `name` (gpa-owned forks; free with `deinit`).
    pub fn fork(l: Lineage, gpa: Allocator, name: [16]u8, at: u32) !Lineage {
        const forks = try gpa.alloc(Fork, l.forks.len + 1);
        @memcpy(forks[0..l.forks.len], l.forks);
        forks[l.forks.len] = .{ .name = name, .at = at };
        return .{ .root = l.root, .part = l.part, .forks = forks };
    }

    /// `l` for rows that live in part `part` of an owner-sharded pool: the part replaces any earlier one (a lineage
    pub fn inPart(l: Lineage, part: u32) Lineage {
        return .{ .root = l.root, .part = part, .forks = l.forks };
    }

    pub fn clone(l: Lineage, gpa: Allocator) !Lineage {
        return .{ .root = l.root, .part = l.part, .forks = try gpa.dupe(Fork, l.forks) };
    }

    pub fn deinit(l: *Lineage, gpa: Allocator) void {
        gpa.free(l.forks);
        l.forks = &.{};
    }

    pub fn eql(a: Lineage, b: Lineage) bool {
        if (!std.mem.eql(u8, &a.root, &b.root) or a.part != b.part or a.forks.len != b.forks.len) return false;
        for (a.forks, b.forks) |x, y| if (x.at != y.at or !std.mem.eql(u8, &x.name, &y.name)) return false;
        return true;
    }
};

/// One block of a point: tokens [start, end) and its chained name.
pub const Span = struct { start: u32, end: u32, name: Digest };

pub fn idsBytes(ids: []const u32) []const u8 {
    return std.mem.sliceAsBytes(ids);
}

/// Each block of `ids` under `lineage`: whole blocks, then the tail [k * block, n) if any. A name covers the lineage
pub fn blockNames(lineage: Lineage, ids_chain: []const Digest, ids: []const u32, block: u32, out: []Span) void {
    std.debug.assert(block > 0 and out.len == std.math.divCeil(usize, ids.len, block) catch unreachable);
    std.debug.assert(ids_chain.len == ids.len / block);
    var head: Digest = undefined;
    {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage");
        h.update(&lineage.root);
        if (lineage.part) |p| {
            h.update("#part");
            h.update(std.mem.asBytes(&p));
        }
        h.final(&head);
    }
    var s: usize = 0;
    var i: usize = 0;
    while (s < ids.len) : (i += 1) {
        const e = @min(ids.len, s + block);
        var h = Sha256.init(.{});
        h.update("tf-spill/block");
        h.update(&head);
        for (lineage.forks) |f| {
            if (s <= f.at and f.at < e) {
                h.update("|fork");
                h.update(&f.name);
                h.update(std.mem.asBytes(&f.at));
            }
        }
        const whole = e - s == block;
        if (whole) {
            h.update("|ids");
            h.update(&ids_chain[i]);
        } else {
            const end: u32 = @intCast(e);
            h.update("|tail");
            h.update(idsBytes(ids[s..e]));
            h.update(std.mem.asBytes(&end));
        }
        var d: Digest = undefined;
        h.final(&d);
        out[i] = .{ .start = @intCast(s), .end = @intCast(e), .name = d };
        if (whole) head = d;
        s = e;
    }
}

pub fn chainStep(prev: *Digest, ids: []const u32) void {
    var h = Sha256.init(.{});
    h.update(prev);
    h.update(idsBytes(ids));
    h.final(prev);
}

pub fn chainSeed() Digest {
    var d: Digest = undefined;
    Sha256.hash("tf-spill/ids", &d, .{});
    return d;
}

/// The id chain of `ids` (no lineage) at each whole block's end: two id lists share a prefix of k whole blocks
pub fn chain(gpa: Allocator, ids: []const u32, block: u32, out: *std.ArrayList(Digest)) !void {
    try out.ensureUnusedCapacity(gpa, ids.len / block);
    var prev = chainSeed();
    var s: usize = 0;
    while (s + block <= ids.len) : (s += block) {
        chainStep(&prev, ids[s .. s + block]);
        out.appendAssumeCapacity(prev);
    }
}

/// A point's key from its id chain (`chain(keyed, block)`): its length, the block size, the last whole block's chain
pub fn pointKeyChained(ids_chain: []const Digest, keyed: []const u32, block: u32, media: []const Media) Digest {
    std.debug.assert(block > 0 and ids_chain.len == keyed.len / block);
    const last = if (ids_chain.len > 0) ids_chain[ids_chain.len - 1] else chainSeed();
    return keyOf(last, keyed, ids_chain.len * block, block, media);
}

/// A point's key from its keyed ids alone (the chain computed here, nothing allocated): equals `pointKeyChained`.
pub fn pointKey(keyed: []const u32, block: u32, media: []const Media) Digest {
    std.debug.assert(block > 0);
    var last = chainSeed();
    const whole = keyed.len / block;
    for (0..whole) |i| chainStep(&last, keyed[i * block ..][0..block]);
    return keyOf(last, keyed, whole * block, block, media);
}

pub fn keyOf(last: Digest, keyed: []const u32, whole: usize, block: u32, media: []const Media) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/point");
    const n: u32 = @intCast(keyed.len);
    h.update(std.mem.asBytes(&n));
    h.update(std.mem.asBytes(&block));
    h.update(&last);
    h.update(idsBytes(keyed[whole..]));
    const m: u32 = @intCast(media.len);
    h.update(std.mem.asBytes(&m));
    if (m > 0) { // (version 3: the scheme the keyed ids were made with; a text point's key is unchanged)
        h.update("|pictures");
        h.update(std.mem.asBytes(&picture_scheme));
    }
    for (media) |x| {
        h.update(std.mem.asBytes(&x.start));
        h.update(std.mem.asBytes(&x.len));
        h.update(&x.digest);
    }
    return h.finalResult();
}

/// The compat digest a store's directory is named by: the caller's common digest (build, kit, model, tokenizer,
pub fn compatDigest(common: Digest, layout: OwnerLayout, block_tokens: u32, block_bytes: u64) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/compat");
    h.update(std.mem.asBytes(&namespace_version));
    h.update(&common);
    const l: u32 = @backingInt(layout);
    h.update(std.mem.asBytes(&l));
    h.update(std.mem.asBytes(&block_tokens));
    h.update(std.mem.asBytes(&block_bytes));
    return h.finalResult();
}

pub fn hex(d: *const Digest) [64]u8 {
    return std.fmt.bytesToHex(d.*, .lower);
}

pub fn parseHex(s: []const u8) ?Digest {
    if (s.len != 64) return null;
    var d: Digest = undefined;
    _ = std.fmt.hexToBytes(&d, s) catch return null;
    return d;
}

pub fn crc32(bytes: []const u8) u32 {
    return Crc32.hash(bytes);
}

pub fn sha256(bytes: []const u8) Digest {
    var d: Digest = undefined;
    Sha256.hash(bytes, &d, .{});
    return d;
}

/// A small-state section's name: at most 32 bytes.
pub const Name = struct {
    bytes: [32]u8 = @splat(0),
    len: u8 = 0,

    pub fn of(s: []const u8) !Name {
        if (s.len == 0 or s.len > 32) return error.BadSectionName;
        var n: Name = .{ .len = @intCast(s.len) };
        @memcpy(n.bytes[0..s.len], s);
        return n;
    }

    pub fn slice(n: *const Name) []const u8 {
        return n.bytes[0..n.len];
    }
};

/// One block a point names: its chained name, its tokens and bytes, CRC-32 (checked on every read) and content
pub const BlockRef = struct { name: Digest, tokens: u32, bytes: u64, crc: u32, sha: Digest };

/// A named small-state section inside the point's `.st` file.
pub const Section = struct { name: Name, offset: u64, len: u64, crc: u32 };

pub const Flags = packed struct(u32) {
    shared: bool = false,
    has_raw: bool = false,
    /// The last block was captured from rows still being written (a mutable tail; native points are chunk-aligned).
    mutable_tail: bool = false,
    _: u29 = 0,
};

/// Where a point's computation came from: the point it resumed (informational; eviction uses the id tree).
pub const Origin = struct { key: Digest, at: u32 };

/// A point's manifest: the last file written, the only one that makes a point readable.
pub const Manifest = @import("format_support_1.zig").Manifest;

/// A bounded little-endian cursor over a buffer whose size the caller checked first.
pub const Cursor = struct {
    buf: []u8,
    at: usize = 0,

    pub fn bytes(c: *Cursor, b: []const u8) void {
        @memcpy(c.buf[c.at..][0..b.len], b);
        c.at += b.len;
    }

    pub fn int(c: *Cursor, comptime T: type, v: T) void {
        std.mem.writeInt(T, c.buf[c.at..][0..@sizeOf(T)], v, .little);
        c.at += @sizeOf(T);
    }

    pub fn take(c: *Cursor, n: usize) []const u8 {
        defer c.at += n;
        return c.buf[c.at..][0..n];
    }

    pub fn get(c: *Cursor, comptime T: type) T {
        defer c.at += @sizeOf(T);
        return std.mem.readInt(T, c.buf[c.at..][0..@sizeOf(T)], .little);
    }
};

/// A point's `.ids` file: its keyed ids, then its raw ids when they differ (pictures), little-endian u32.
pub fn encodeIds(gpa: Allocator, keyed: []const u32, raw: ?[]const u32) ![]u8 {
    const n = keyed.len * 4 * (if (raw != null) @as(usize, 2) else 1);
    const out = try gpa.alloc(u8, n);
    @memcpy(out[0 .. keyed.len * 4], idsBytes(keyed));
    if (raw) |r| {
        std.debug.assert(r.len == keyed.len);
        @memcpy(out[keyed.len * 4 ..], idsBytes(r));
    }
    return out;
}

// -- tests ---------------------------------------------------------------------------------------------------------
pub const testing = std.testing;

pub fn testNames(lineage: Lineage, ids: []const u32, out: []Span) !void {
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(testing.allocator);
    try chain(testing.allocator, ids, 16, &ch);
    blockNames(lineage, ch.items, ids, 16, out);
}

pub fn testIds(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

pub fn sampleManifest(gpa: Allocator, l: Limits) !Manifest {
    const blocks = try gpa.alloc(BlockRef, 3);
    for (blocks, 0..) |*b, i| b.* = .{ .name = @splat(@intCast(i)), .tokens = if (i < 2) 16 else 5, .bytes = if (i < 2) l.block_bytes else 100, .crc = 7, .sha = @splat(3) };
    const sections = try gpa.alloc(Section, 2);
    sections[0] = .{ .name = try Name.of("ring"), .offset = 0, .len = 64, .crc = 1 };
    sections[1] = .{ .name = try Name.of("raw"), .offset = 64, .len = 32, .crc = 2 };
    const media = try gpa.alloc(Media, 1);
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    const forks = try gpa.alloc(Fork, 1);
    forks[0] = .{ .name = @splat(4), .at = 16 };
    return .{
        .key = @splat(1),
        .at = 37,
        .block_tokens = 16,
        .flags = .{ .shared = true, .has_raw = true },
        .group = 99,
        .created_ns = 12345,
        .lineage = .{ .root = @splat(2), .part = 1, .forks = forks },
        .origin = .{ .key = @splat(8), .at = 32 },
        .ids_bytes = 37 * 8,
        .ids_crc = 5,
        .small_bytes = 96,
        .small_crc = 6,
        .media = media,
        .blocks = blocks,
        .sections = sections,
    };
}

// -- pictures (version 3) --------------------------------------------------------------------------------------------

/// `sampleManifest` without media: a text point (version 2).
pub fn textManifest(gpa: Allocator, l: Limits) !Manifest {
    var m = try sampleManifest(gpa, l);
    gpa.free(m.media);
    m.media = &.{};
    m.flags.has_raw = false;
    m.ids_bytes = 37 * 4;
    return m;
}

test {
    _ = @import("format_cases_1.zig");
}
