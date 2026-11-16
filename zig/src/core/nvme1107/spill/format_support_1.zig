//! Shared implementation helpers for format.
const source_api = @import("format.zig");
const std = source_api.std;
const Allocator = source_api.Allocator;
const Digest = source_api.Digest;
const zero_digest = source_api.zero_digest;
const version = source_api.version;
const text_version = source_api.text_version;
const magic = source_api.magic;
const header_bytes = source_api.header_bytes;
const fork_bytes = source_api.fork_bytes;
const media_bytes = source_api.media_bytes;
const media_v2_bytes = source_api.media_v2_bytes;
const media_picture = source_api.media_picture;
const picture_scheme = source_api.picture_scheme;
const block_ref_bytes = source_api.block_ref_bytes;
const section_bytes = source_api.section_bytes;
const trailer_bytes = source_api.trailer_bytes;
const OwnerLayout = source_api.OwnerLayout;
const Limits = source_api.Limits;
const Media = source_api.Media;
const Fork = source_api.Fork;
const Lineage = source_api.Lineage;
const crc32 = source_api.crc32;
const BlockRef = source_api.BlockRef;
const Section = source_api.Section;
const Flags = source_api.Flags;
const Origin = source_api.Origin;
const Cursor = source_api.Cursor;

pub const Manifest = struct {
    key: Digest,
    at: u32,
    block_tokens: u32,
    flags: Flags = .{},
    group: u64 = 0,
    created_ns: i64 = 0,
    owner_layout: OwnerLayout = .replicated,
    owner_epoch: u32 = 0,
    lineage: Lineage,
    origin: ?Origin = null,
    ids_bytes: u64,
    ids_crc: u32,
    small_bytes: u64,
    small_crc: u32,
    media: []const Media = &.{},
    blocks: []const BlockRef = &.{},
    sections: []const Section = &.{},

    pub fn deinit(m: *Manifest, gpa: Allocator) void {
        gpa.free(m.media);
        gpa.free(m.blocks);
        gpa.free(m.sections);
        gpa.free(m.lineage.forks);
        m.* = undefined;
    }

    /// The version this manifest is written as: 3 with media, else 2 (a text point's bytes as before).
    pub fn versionOf(m: *const Manifest) u32 {
        return if (m.media.len > 0) version else text_version;
    }

    pub fn encodedLen(m: *const Manifest) u64 {
        return header_bytes + m.lineage.forks.len * fork_bytes + m.media.len * media_bytes +
            m.blocks.len * block_ref_bytes + m.sections.len * section_bytes + trailer_bytes;
    }

    /// The manifest's bytes (gpa-owned).
    pub fn encode(m: *const Manifest, gpa: Allocator, compat: Digest) ![]u8 {
        const out = try gpa.alloc(u8, @intCast(m.encodedLen()));
        @memset(out, 0);
        var w: Cursor = .{ .buf = out };
        w.bytes(&magic);
        w.int(u32, m.versionOf());
        w.int(u32, header_bytes);
        w.bytes(&compat);
        w.bytes(&m.key);
        w.int(u32, m.at);
        w.int(u32, m.block_tokens);
        w.int(u32, @bitCast(m.flags));
        w.int(u32, @intCast(m.media.len));
        w.int(u32, @intCast(m.blocks.len));
        w.int(u32, @intCast(m.sections.len));
        w.int(u32, @intCast(m.lineage.forks.len));
        w.int(u32, @backingInt(m.owner_layout));
        w.int(u32, m.owner_epoch);
        w.int(u64, m.group);
        w.int(i64, m.created_ns);
        w.bytes(&m.lineage.root);
        w.int(u32, @intFromBool(m.lineage.part != null));
        w.int(u32, m.lineage.part orelse 0);
        w.int(u32, @intFromBool(m.origin != null));
        w.int(u32, if (m.origin) |o| o.at else 0);
        w.bytes(if (m.origin) |*o| &o.key else &zero_digest);
        w.int(u64, m.ids_bytes);
        w.int(u32, m.ids_crc);
        w.int(u32, m.small_crc);
        w.int(u64, m.small_bytes);
        std.debug.assert(w.at <= header_bytes);
        w.at = header_bytes;
        for (m.lineage.forks) |f| {
            w.bytes(&f.name);
            w.int(u32, f.at);
        }
        for (m.media) |x| {
            w.int(u32, x.start);
            w.int(u32, x.len);
            w.int(u32, media_picture);
            w.int(u32, picture_scheme);
            w.bytes(&x.digest);
        }
        for (m.blocks) |b| {
            w.bytes(&b.name);
            w.int(u32, b.tokens);
            w.int(u32, b.crc);
            w.int(u64, b.bytes);
            w.bytes(&b.sha);
        }
        for (m.sections) |s| {
            w.bytes(&s.name.bytes);
            w.int(u64, s.offset);
            w.int(u64, s.len);
            w.int(u32, s.crc);
            w.int(u32, s.name.len);
        }
        w.int(u32, crc32(out[0..w.at]));
        w.int(u32, 0);
        std.debug.assert(w.at == out.len);
        return out;
    }

    pub const ParseError = error{ Foreign, Corrupt, OutOfMemory };

    /// The manifest in `raw` (a whole file), checked against `compat` and `limits`: sizes and counts before any
    pub fn parse(gpa: Allocator, raw: []const u8, compat: Digest, limits: Limits) ParseError!Manifest {
        if (raw.len < header_bytes + trailer_bytes or raw.len > limits.maxManifest()) return error.Corrupt;
        var r: Cursor = .{ .buf = @constCast(raw) };
        if (!std.mem.eql(u8, r.take(8), &magic)) return error.Foreign;
        const ver = r.get(u32);
        if ((ver != version and ver != text_version) or r.get(u32) != header_bytes) return error.Foreign;
        if (!std.mem.eql(u8, r.take(32), &compat)) return error.Foreign;
        var m: Manifest = .{ .key = r.take(32)[0..32].*, .at = 0, .block_tokens = 0, .lineage = .{ .root = @splat(0) }, .ids_bytes = 0, .ids_crc = 0, .small_bytes = 0, .small_crc = 0 };
        m.at = r.get(u32);
        m.block_tokens = r.get(u32);
        m.flags = @bitCast(r.get(u32));
        const n_media = r.get(u32);
        const n_blocks = r.get(u32);
        const n_sections = r.get(u32);
        const n_forks = r.get(u32);
        const layout = r.get(u32);
        m.owner_epoch = r.get(u32);
        m.group = r.get(u64);
        m.created_ns = r.get(i64);
        m.lineage.root = r.take(16)[0..16].*;
        const has_part = r.get(u32);
        const part = r.get(u32);
        const has_origin = r.get(u32);
        const origin_at = r.get(u32);
        const origin_key = r.take(32)[0..32].*;
        m.ids_bytes = r.get(u64);
        m.ids_crc = r.get(u32);
        m.small_crc = r.get(u32);
        m.small_bytes = r.get(u64);
        if (m.block_tokens != limits.block_tokens) return error.Foreign;
        m.owner_layout = std.enums.fromInt(OwnerLayout, layout) orelse return error.Foreign;
        if (m.flags._ != 0 or has_part > 1 or has_origin > 1) return error.Corrupt;
        if (m.at == 0 or m.at > limits.max_tokens) return error.Corrupt;
        if (n_blocks != std.math.divCeil(u32, m.at, m.block_tokens) catch unreachable) return error.Corrupt;
        if (n_media > limits.max_media or n_sections > limits.max_sections or n_forks > limits.max_forks) return error.Corrupt;
        // version 2 points are text points (no build wrote one with media; its ids name no scheme: foreign),...
        if (ver == text_version and n_media > 0) return error.Foreign;
        if (ver == version and (n_media == 0 or !m.flags.has_raw)) return error.Corrupt;
        const ids_want = @as(u64, m.at) * 4 * (if (m.flags.has_raw) @as(u64, 2) else 1);
        if (m.ids_bytes != ids_want or m.small_bytes > limits.max_small_bytes) return error.Corrupt;
        const total = header_bytes + @as(u64, n_forks) * fork_bytes + @as(u64, n_media) * (if (ver == version) @as(u64, media_bytes) else media_v2_bytes) +
            @as(u64, n_blocks) * block_ref_bytes + @as(u64, n_sections) * section_bytes + trailer_bytes;
        if (raw.len != total) return error.Corrupt;
        const body = raw.len - trailer_bytes;
        if (std.mem.readInt(u32, raw[body..][0..4], .little) != crc32(raw[0..body])) return error.Corrupt;
        if (std.mem.readInt(u32, raw[body + 4 ..][0..4], .little) != 0) return error.Corrupt;
        if (has_part == 1) m.lineage.part = part;
        if (has_origin == 1) m.origin = .{ .key = origin_key, .at = origin_at };
        r.at = header_bytes;

        const forks = try gpa.alloc(Fork, n_forks);
        errdefer gpa.free(forks);
        for (forks) |*f| {
            f.name = r.take(16)[0..16].*;
            f.at = r.get(u32);
            if (f.at > limits.max_tokens) return error.Corrupt;
        }
        const media = try gpa.alloc(Media, n_media);
        errdefer gpa.free(media);
        var media_end: u64 = 0;
        for (media) |*x| {
            x.start = r.get(u32);
            x.len = r.get(u32);
            const kind = r.get(u32);
            const scheme = r.get(u32);
            x.digest = r.take(32)[0..32].*;
            if (kind != media_picture or scheme != picture_scheme) return error.Foreign;
            // in order, not overlapping, starting inside the point (one may run past its cut), inside the window
            if (x.len == 0 or x.start < media_end or x.start >= m.at or @as(u64, x.start) + x.len > limits.max_tokens) return error.Corrupt;
            media_end = @as(u64, x.start) + x.len;
        }
        // (a part-less point in an owner-sharded store, or a foreign block size where none is kept: damaged)
        const full = limits.fullBlockBytes(m.lineage.part) orelse return error.Corrupt;
        if (full > limits.block_bytes) return error.Corrupt;
        const blocks = try gpa.alloc(BlockRef, n_blocks);
        errdefer gpa.free(blocks);
        for (blocks, 0..) |*b, i| {
            b.name = r.take(32)[0..32].*;
            b.tokens = r.get(u32);
            b.crc = r.get(u32);
            b.bytes = r.get(u64);
            b.sha = r.take(32)[0..32].*;
            const last = i + 1 == n_blocks;
            const want = if (!last) m.block_tokens else m.at - @as(u32, @intCast(i)) * m.block_tokens;
            if (b.tokens != want) return error.Corrupt;
            if (b.tokens == m.block_tokens) {
                if (b.bytes != full) return error.Corrupt;
            } else if (b.bytes == 0 or b.bytes > limits.block_bytes) return error.Corrupt;
        }
        const sections = try gpa.alloc(Section, n_sections);
        errdefer gpa.free(sections);
        var end: u64 = 0;
        for (sections) |*s| {
            s.name.bytes = r.take(32)[0..32].*;
            s.offset = r.get(u64);
            s.len = r.get(u64);
            s.crc = r.get(u32);
            const nl = r.get(u32);
            if (nl == 0 or nl > 32) return error.Corrupt;
            s.name.len = @intCast(nl);
            for (s.name.bytes[nl..]) |c| if (c != 0) return error.Corrupt;
            const stop = std.math.add(u64, s.offset, s.len) catch return error.Corrupt;
            if (s.offset < end or stop > m.small_bytes) return error.Corrupt;
            end = stop;
        }
        m.lineage.forks = forks;
        m.media = media;
        m.blocks = blocks;
        m.sections = sections;
        return m;
    }
};
