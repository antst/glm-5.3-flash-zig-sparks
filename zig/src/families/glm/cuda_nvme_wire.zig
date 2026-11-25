//! Finite all-rank NVMe commands on the existing GLM follow link, outside backend passes.
const std = @import("std");
const follow = @import("cuda_follow.zig");
const ranks = @import("cuda_ranks.zig");
const disk = @import("cuda_disk_ranks.zig");
const state = @import("cuda_nvme_state.zig");
const progress = @import("cuda_nvme_progress.zig");
const nvme = @import("core").nvme1107;
const Engine = @import("cuda_engine.zig").Engine;
const snapshots = @import("cuda_snapshots.zig");
pub const base: u32 = 0x11070000;
pub const failed: u64 = 1 << 63;
pub const Op = enum(u32) { open = base, note_tokens, note_metadata, pump, release, cancel, read_begin, read_poll, read_commit, read_abort, stop, close, read_publish };
pub const Open = extern struct { compat: [32]u8, cap: u64, floor: u64, staging: u64, index: u64, min_tokens: u32, lookahead: u32, root_bytes: u32, reserved: u32 = 0 };

fn side() !*state.State {
    return state.local orelse error.NvmeNotOpen;
}

pub fn isOurs(flags: u32) bool {
    return flags >= base and flags <= @backingInt(Op.read_publish);
}

/// A local storage refusal joins the status exchange. Only communication errors break the command protocol.
fn apply(e: *Engine, h: follow.Header, ids: []const u32) !u64 {
    if (!isOurs(h.flags)) return error.NvmeProtocol;
    const op: Op = @fromBackingInt(h.flags);
    switch (op) {
        .open => {
            const bytes = std.mem.sliceAsBytes(ids);
            if (bytes.len < @sizeOf(Open)) return error.NvmeProtocol;
            const o = std.mem.bytesToValue(Open, bytes[0..@sizeOf(Open)]);
            if (o.reserved > 1 or o.lookahead > 1 or o.root_bytes == 0 or o.root_bytes > 4096 or o.root_bytes > bytes.len - @sizeOf(Open)) return error.NvmeProtocol;
            _ = try state.State.open(e, bytes[@sizeOf(Open)..][0..o.root_bytes], o.compat, .{ .cap = o.cap, .floor = o.floor, .staging = o.staging, .index = o.index, .min_tokens = o.min_tokens, .lookahead = @intCast(o.lookahead), .serial_mode = o.reserved == 1 });
            return 0;
        },
        .close => {
            if (state.local) |s| s.close();
            return 0;
        },
        else => {},
    }
    const s = try side();
    // Leader.send owns Header.slot (the bound sequence). Source-credit slots use top_k >> 1;
    // the low bit is the note's shared-history flag, never an engine sequence number.
    const slot = h.top_k >> 1;
    switch (op) {
        .note_tokens => {
            if (h.a == 0 or h.seed > std.math.maxInt(u32) or ids.len > s.side.store.cfg.limits.max_tokens or slot >= s.slots.len) return error.NvmeProtocol;
            s.tokens.clearRetainingCapacity();
            try s.tokens.appendSlice(e.gpa, ids);
            s.token_id = h.a;
            s.token_at = @intCast(h.seed);
        },
        .note_metadata => {
            const bytes = std.mem.sliceAsBytes(ids);
            if (slot >= s.slots.len or s.token_id != h.a or h.seed != s.token_at or h.depth > state.max_metadata or bytes.len < 16 or h.depth > bytes.len - 16) return error.NvmeProtocol;
            const meta = bytes[16..][0..h.depth];
            const pages = try pageIds(e, ids, 16 + @as(usize, h.depth), s.token_at);
            const root = bytes[0..16].*;
            if (s.slots[slot] != 0) return error.NvmeProtocol;
            const existing = for (s.side.notes.items) |p| {
                if (p.note.id == h.a) break p.note;
            } else null;
            if (existing) |note| {
                if (!std.mem.eql(u32, note.tokens, s.tokens.items) or !std.mem.eql(u8, note.metadata, meta) or !std.mem.eql(u8, &note.lineage.root, &root) or note.at != s.token_at or note.shared != (h.top_k & 1 != 0)) return error.NvmeProtocol;
                if (e.share_pages and !std.mem.eql(u32, pages, (e.snaps.get(h.a) orelse return error.NoKeptState).tier_ids))
                    return error.NvmeProtocol;
            } else {
                if (e.share_pages) try snapshots.attachTierIds(e, h.a, pages);
                errdefer if (e.share_pages) {
                    if (e.snaps.getPtr(h.a)) |kept| snapshots.clearTierIds(e, kept);
                };
                try s.side.enqueue(.{ .id = h.a, .at = s.token_at, .tokens = s.tokens.items, .metadata = meta, .lineage = .{ .root = root }, .shared = h.top_k & 1 != 0 });
            }
            s.slots[slot] = h.a;
            s.token_id = 0;
            s.tokens.clearRetainingCapacity();
        },
        .pump => {
            if (h.seed > 10 * std.time.ns_per_ms or h.depth > 1) return error.NvmeProtocol;
            if (h.depth == 1) s.side.pumpWait(h.seed) else s.side.pump(h.seed);
            comptime std.debug.assert(state.max_pending <= 16);
            return progress.packet(s.completionMask(), s.side.store.writeProgress());
        },
        .release => {
            if (slot >= s.slots.len or s.slots[slot] != h.a) return error.NvmeProtocol;
            try s.side.release(h.a);
            s.slots[slot] = 0;
        },
        .cancel => {
            if (slot >= s.slots.len) return error.NvmeProtocol;
            s.side.cancelNote(h.a);
            if (s.side.completed(h.a)) try s.side.release(h.a);
            s.slots[slot] = 0;
            s.token_id = 0;
            s.tokens.clearRetainingCapacity();
        },
        .read_begin => {
            const bytes = std.mem.sliceAsBytes(ids);
            if (bytes.len < 32 or h.depth > state.max_metadata or h.depth > bytes.len - 32 or h.seed > std.math.maxInt(u32)) return error.NvmeProtocol;
            const at: u32 = @intCast(h.seed);
            const pages = try pageIds(e, ids, 32 + @as(usize, h.depth), at);
            try s.side.beginRead(bytes[0..32].*, h.a, at, bytes[32..][0..h.depth]);
            errdefer s.side.abortRead();
            if (e.share_pages) try snapshots.attachTierIds(e, h.a, pages);
        },
        .read_poll => {
            const status = s.side.pollRead();
            if (status == .failed) if (s.side.read) |r| if (r.failed) |err| if (err == error.NvmeRead) return 3;
            return @backingInt(status);
        },
        .read_commit => try s.side.commitRead(),
        .read_publish => {
            if (!e.share_pages or ids.len != 0) return error.NvmeProtocol;
            const kept = e.snaps.getPtr(h.a) orelse return error.NoKeptState;
            if (kept.len == 0 or kept.tier_ids.len != kept.shared) return error.NvmeProtocol;
            snapshots.clearTierIds(e, kept);
        },
        .read_abort => {
            if (h.top_k > 1 or (h.top_k == 1 and ids.len != 8)) return error.NvmeProtocol;
            if (h.top_k == 1) s.side.forgetRead(std.mem.sliceAsBytes(ids)[0..32].*);
            s.side.abortRead();
            if (e.snaps.contains(h.a)) e.dropState(h.a);
        },
        .stop => {
            s.side.stop(h.seed);
            s.slots = @splat(0);
        },
        else => return error.NvmeProtocol,
    }
    return 0;
}

/// IDs are the word-aligned tail after the existing metadata.
fn pageIds(e: *const Engine, data: []const u32, metadata_end: usize, at: u32) ![]const u32 {
    const first = (metadata_end + 3) / 4;
    if (first > data.len) return error.NvmeProtocol;
    const pages = data[first..];
    const page = @import("cuda_state.zig").page_rows;
    if (e.share_pages and (at % page != 0 or pages.len != at / page)) return error.NvmeProtocol;
    if (!e.share_pages and pages.len != 0) return error.NvmeProtocol;
    return pages;
}

fn outcome(e: *Engine, h: follow.Header, ids: []const u32) u64 {
    return apply(e, h, ids) catch |err| {
        std.log.warn("GLM_NVME1107_STEP rank={d} op={d} id={d} error={s}", .{ e.share.rank, h.flags, h.a, @errorName(err) });
        return failed;
    };
}

/// Follower: exactly one status gather per command, including failed local setup and allocation.
pub fn step(e: *Engine, l: *follow.Link, h: follow.Header, ids: []const u32) !void {
    var replies: [8]u64 = undefined;
    if (e.share.world > replies.len) return error.NvmeProtocol;
    const mine = outcome(e, h, ids);
    try ranks.gather(u64, e.d, l.lib, l.comm, l.stream, e.share.rank, mine, replies[0..e.share.world]);
}

/// Leader: command first, then local work and all statuses; transport failure cannot be a rank-local cache miss.
pub fn invoke(e: *Engine, op: Op, options: follow.Header, ids: []const u32, replies: *[8]u64) ![]const u64 {
    if (e.share.rank != 0 or e.share.world > replies.len) return error.NvmeProtocol;
    var h = options;
    h.flags = @backingInt(op);
    const l = if (e.share.world > 1) disk.link orelse return error.NvmeProtocol else null;
    if (l) |link| try (follow.Leader{ .e = e, .link = link }).disk(h, ids);
    const mine = outcome(e, h, ids);
    if (l) |link| try ranks.gather(u64, e.d, link.lib, link.comm, link.stream, 0, mine, replies[0..e.share.world]) else replies[0] = mine;
    return replies[0..e.share.world];
}

pub fn allOk(replies: []const u64) bool {
    for (replies) |v| if (v & failed != 0) return false;
    return true;
}

pub fn open(e: *Engine, root: []const u8, compat: [32]u8, cfg: state.Config) !void {
    const n = @sizeOf(Open) + root.len;
    const words = try e.gpa.alloc(u32, (n + 3) / 4);
    defer e.gpa.free(words);
    const bytes = std.mem.sliceAsBytes(words);
    @memset(bytes, 0);
    const packet: Open = .{ .compat = compat, .cap = cfg.cap, .floor = cfg.floor, .staging = cfg.staging, .index = cfg.index, .min_tokens = cfg.min_tokens, .lookahead = cfg.lookahead, .root_bytes = @intCast(root.len), .reserved = @intFromBool(cfg.serial_mode) };
    @memcpy(bytes[0..@sizeOf(Open)], std.mem.asBytes(&packet));
    @memcpy(bytes[@sizeOf(Open)..][0..root.len], root);
    var replies: [8]u64 = undefined;
    if (!allOk(try invoke(e, .open, .{ .kind = 0 }, words, &replies))) {
        _ = try invoke(e, .close, .{ .kind = 0 }, &.{}, &replies);
        return error.NvmePeerOpen;
    }
}

/// Exchange complete bytes, with explicit padding; no borrowed host buffer survives the command.
pub fn pagePayload(a: std.mem.Allocator, prefix: []const u8, metadata: []const u8, ids: []const u32) ![]u32 {
    const words = (prefix.len + metadata.len + 3) / 4;
    const out = try a.alloc(u32, words + ids.len);
    @memset(out[0..words], 0);
    const bytes = std.mem.sliceAsBytes(out[0..words]);
    @memcpy(bytes[0..prefix.len], prefix);
    @memcpy(bytes[prefix.len..][0..metadata.len], metadata);
    @memcpy(out[words..], ids);
    return out;
}

pub fn payload(a: std.mem.Allocator, prefix: []const u8, metadata: []const u8) ![]u32 {
    const words = try a.alloc(u32, (prefix.len + metadata.len + 3) / 4);
    const bytes = std.mem.sliceAsBytes(words);
    @memset(bytes, 0);
    @memcpy(bytes[0..prefix.len], prefix);
    @memcpy(bytes[prefix.len..][0..metadata.len], metadata);
    return words;
}
