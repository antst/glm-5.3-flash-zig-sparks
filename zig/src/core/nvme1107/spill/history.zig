//! Shared token histories: immutable prefix chunks plus one short tail per checkpoint.
const std = @import("std");
const c = std.c;
const fsio = @import("fsio.zig");
const format = @import("format.zig");
const Allocator = std.mem.Allocator;
pub const Digest = format.Digest;
pub const chunk_tokens: usize = 2048;
pub const chunk_bytes: usize = chunk_tokens * 4;
const node_bytes = 32 + chunk_bytes;
const header_bytes: usize = 80;
const magic = "TFIDS3\x00\x00";
const zero: Digest = @splat(0);

pub const Roots = struct { keyed: Digest = zero, raw: Digest = zero };
pub const Encoded = struct { bytes: []u8, roots: Roots };
const Node = struct { prev: Digest, refs: u64 = 0 };

pub fn fileBytes(at: u32, raw: bool) u64 {
    return header_bytes + @as(u64, at % chunk_tokens) * 4 * (if (raw) @as(u64, 2) else 1);
}

pub fn overhead(at: u32, raw: bool) u64 {
    return header_bytes + @as(u64, at / chunk_tokens) * 32 * (if (raw) @as(u64, 2) else 1);
}

fn name(buf: *[80]u8, digest: Digest) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}.idsb", .{format.hex(&digest)}, 0) catch unreachable;
}

fn digestNode(bytes: []const u8) Digest {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("tensorfold/shared-token-history/v1");
    hash.update(bytes);
    return hash.finalResult();
}

fn eql(a: Digest, b: Digest) bool {
    return std.mem.eql(u8, &a, &b);
}

pub const History = struct {
    gpa: Allocator,
    fd: c.fd_t,
    nodes: std.AutoHashMapUnmanaged(Digest, Node) = .empty,
    cache: std.AutoHashMapUnmanaged(Digest, []u8) = .empty,
    fifo: std.ArrayList(Digest) = .empty,
    hand: usize = 0,
    scanning: bool = true,
    conservative: bool = false,
    bytes: u64 = 0,
    max_nodes: usize,
    const cache_slots = (32 << 20) / node_bytes;

    pub fn open(gpa: Allocator, parent: c.fd_t, max_nodes: usize) !History {
        try fsio.ensureDirAt(parent, "histories");
        try fsio.syncDir(&.{}, parent, "history-parent");
        return .{ .gpa = gpa, .fd = try fsio.openDirAt(parent, "histories"), .max_nodes = max_nodes };
    }

    pub fn deinit(h: *History) void {
        h.dropCache();
        h.nodes.deinit(h.gpa);
        _ = c.close(h.fd);
    }

    pub fn dropCache(h: *History) void {
        var it = h.cache.valueIterator();
        while (it.next()) |buf| h.gpa.free(buf.*);
        h.cache.deinit(h.gpa);
        h.cache = .empty;
        h.fifo.deinit(h.gpa);
        h.fifo = .empty;
    }

    fn add(h: *History, key: Digest, prev: Digest) !void {
        if (h.nodes.get(key)) |old| {
            if (!eql(old.prev, prev)) return error.Corrupt;
            return;
        }
        if (h.nodes.count() >= h.max_nodes) return error.OutOfMemory;
        try h.nodes.put(h.gpa, key, .{ .prev = prev });
    }

    fn cacheNode(h: *History, key: Digest, data: []u8) !void {
        if (h.fifo.items.len == cache_slots) {
            const victim = h.fifo.items[h.hand];
            const old = h.cache.fetchRemove(victim).?;
            h.gpa.free(old.value);
            try h.cache.put(h.gpa, key, data);
            h.fifo.items[h.hand] = key;
            h.hand = (h.hand + 1) % cache_slots;
        } else {
            try h.fifo.ensureUnusedCapacity(h.gpa, 1);
            try h.cache.put(h.gpa, key, data);
            h.fifo.appendAssumeCapacity(key);
        }
    }

    fn readNode(h: *History, key: Digest) ![]const u8 {
        if (h.scanning) if (h.cache.get(key)) |bytes| return bytes;
        var buf: [80]u8 = undefined;
        const bytes = try fsio.readSmall(h.gpa, h.fd, name(&buf, key), node_bytes);
        errdefer h.gpa.free(bytes);
        if (bytes.len != node_bytes or !eql(digestNode(bytes), key)) return error.Corrupt;
        // Recovery walks backwards, so edges are rebuilt once every node has been seen.
        if (h.scanning) try h.add(key, bytes[0..32].*);
        if (h.scanning) try h.cacheNode(key, bytes);
        return bytes;
    }

    fn encodeStream(h: *History, writer: *fsio.Writer, bytes: []const u8) !Digest {
        var prev: Digest = zero;
        const whole = bytes.len / chunk_bytes;
        for (0..whole) |i| {
            var record: [node_bytes]u8 = undefined;
            @memcpy(record[0..32], &prev);
            @memcpy(record[32..], bytes[i * chunk_bytes ..][0..chunk_bytes]);
            const key = digestNode(&record);
            if (!h.nodes.contains(key)) {
                if (h.nodes.count() >= h.max_nodes) return error.OutOfMemory;
                try h.nodes.ensureUnusedCapacity(h.gpa, 1);
                var buf: [80]u8 = undefined;
                const existed = if (fsio.statAt(h.fd, name(&buf, key))) |_| true else |err| switch (err) {
                    error.Missing => false,
                    else => return err,
                };
                _ = try writer.put(h.fd, name(&buf, key), .{ .parts = &.{&record} });
                h.nodes.putAssumeCapacity(key, .{ .prev = prev });
                if (!eql(prev, zero)) h.nodes.getPtr(prev).?.refs += 1;
                if (!existed) h.bytes += node_bytes;
            }
            prev = key;
        }
        return prev;
    }

    pub fn encode(h: *History, writer: *fsio.Writer, ids: []const u8, at: u32, raw: bool) !Encoded {
        std.debug.assert(!h.scanning);
        const span: usize = @as(usize, at) * 4;
        if (ids.len != span * (if (raw) @as(usize, 2) else 1)) return error.Corrupt;
        const out = try h.gpa.alloc(u8, @intCast(fileBytes(at, raw)));
        errdefer h.gpa.free(out);
        @memset(out, 0);
        @memcpy(out[0..8], magic);
        std.mem.writeInt(u32, out[8..12], at, .little);
        out[12] = @intFromBool(raw);
        const roots: Roots = .{ .keyed = try h.encodeStream(writer, ids[0..span]), .raw = if (raw) try h.encodeStream(writer, ids[span..]) else zero };
        @memcpy(out[16..48], &roots.keyed);
        @memcpy(out[48..80], &roots.raw);
        const tail = span % chunk_bytes;
        @memcpy(out[header_bytes..][0..tail], ids[span - tail .. span]);
        if (raw) @memcpy(out[header_bytes + tail ..], ids[2 * span - tail ..]);
        try fsio.syncDir(writer.hooks, h.fd, "histories");
        return .{ .bytes = out, .roots = roots };
    }

    pub fn parse(bytes: []const u8, at: u32, raw: bool) !Roots {
        if (bytes.len != fileBytes(at, raw) or !std.mem.eql(u8, bytes[0..8], magic)) return error.Corrupt;
        if (std.mem.readInt(u32, bytes[8..12], .little) != at or bytes[12] != @intFromBool(raw)) return error.Corrupt;
        if (!std.mem.eql(u8, bytes[13..16], &.{ 0, 0, 0 })) return error.Corrupt;
        const roots: Roots = .{ .keyed = bytes[16..48].*, .raw = bytes[48..80].* };
        if ((!raw and !eql(roots.raw, zero)) or (at < chunk_tokens and !eql(roots.keyed, zero))) return error.Corrupt;
        return roots;
    }

    pub fn decode(h: *History, bytes: []const u8, at: u32, raw: bool) ![]u8 {
        const roots = try parse(bytes, at, raw);
        const span: usize = @as(usize, at) * 4;
        const out = try h.gpa.alloc(u8, span * (if (raw) @as(usize, 2) else 1));
        errdefer h.gpa.free(out);
        const root_list = [_]Digest{ roots.keyed, roots.raw };
        for (root_list[0..if (raw) @as(usize, 2) else 1], 0..) |head, stream| {
            var key = head;
            var n = span / chunk_bytes;
            while (n > 0) {
                n -= 1;
                if (eql(key, zero)) return error.Corrupt;
                const record = try h.readNode(key);
                defer if (!h.scanning) h.gpa.free(record);
                @memcpy(out[stream * span + n * chunk_bytes ..][0..chunk_bytes], record[32..]);
                key = record[0..32].*;
            }
            if (!eql(key, zero)) return error.Corrupt;
            const tail = span % chunk_bytes;
            @memcpy(out[stream * span + span - tail ..][0..tail], bytes[header_bytes + stream * tail ..][0..tail]);
        }
        return out;
    }

    pub fn pin(h: *History, roots: Roots) void {
        for ([_]Digest{ roots.keyed, roots.raw }) |key| {
            if (!eql(key, zero)) h.nodes.getPtr(key).?.refs += 1;
        }
    }

    /// Rebuild DAG edges once after recovery; no file disappears while any scan was uncertain.
    pub fn finishRecovery(h: *History, safe: bool) !void {
        h.dropCache();
        h.scanning = false;
        h.conservative = !safe;
        var it = h.nodes.valueIterator();
        while (it.next()) |node| if (!eql(node.prev, zero)) {
            const p = h.nodes.getPtr(node.prev) orelse continue;
            p.refs += 1;
        };
        // `add` did not count edges during backwards recovery.
        if (safe) try h.collect();
        var names = try fsio.list(h.gpa, h.fd);
        defer fsio.freeList(h.gpa, &names);
        h.bytes = 0;
        for (names.items) |n| {
            const key = if (n.len == 69 and std.mem.endsWith(u8, n, ".idsb")) format.parseHex(n[0..64]) else null;
            if (safe and (key == null or !h.nodes.contains(key.?))) {
                while (c.unlinkat(h.fd, n, 0) != 0) {
                    switch (c.errno(@as(c_int, -1))) {
                        .INTR => continue,
                        .NOENT => break,
                        else => return error.Io,
                    }
                }
                continue;
            }
            h.bytes += (try fsio.statAt(h.fd, n)).size;
        }
        try fsio.syncDir(&.{}, h.fd, "histories");
    }

    pub fn release(h: *History, roots: Roots) !void {
        for ([_]Digest{ roots.keyed, roots.raw }) |key| if (!eql(key, zero)) {
            if (h.nodes.getPtr(key)) |n| {
                if (n.refs == 0) continue;
                n.refs -= 1;
                if (!h.conservative) try h.gcChain(key);
            }
        };
    }

    fn gcChain(h: *History, first: Digest) !void {
        var key = first;
        while (!eql(key, zero)) {
            const n = h.nodes.get(key) orelse return;
            if (n.refs != 0) return;
            var buf: [80]u8 = undefined;
            while (c.unlinkat(h.fd, name(&buf, key), 0) != 0) {
                switch (c.errno(@as(c_int, -1))) {
                    .INTR => continue,
                    .NOENT => break,
                    else => return error.Io,
                }
            }
            _ = h.nodes.remove(key);
            h.bytes -|= node_bytes;
            if (eql(n.prev, zero)) return;
            if (h.nodes.getPtr(n.prev)) |parent| {
                if (parent.refs == 0) return;
                parent.refs -= 1;
            }
            key = n.prev;
        }
    }

    pub fn collect(h: *History) !void {
        if (h.conservative) return;
        var roots: std.ArrayList(Digest) = .empty;
        defer roots.deinit(h.gpa);
        var it = h.nodes.iterator();
        while (it.next()) |node| if (node.value_ptr.refs == 0) try roots.append(h.gpa, node.key_ptr.*);
        for (roots.items) |key| try h.gcChain(key);
    }
};
