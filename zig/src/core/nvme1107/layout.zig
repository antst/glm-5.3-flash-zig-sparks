//! Bounded block and small-state ranges over a family's unchanged logical byte image.
const std = @import("std");
pub const Part = struct { bytes: u64, row: u32 = 0, per: u32 = 1, window: u32 = 0 };
pub const Range = struct { off: u64, len: u64 };

pub const Layout = struct {
    parts: []const Part,
    at: u32,
    block: u32,
    bytes: u64,

    pub fn init(parts: []const Part, at: u32, block: u32) !Layout {
        if (block == 0 or at == 0 or at % block != 0) return error.NvmeLayout;
        var bytes: u64 = 0;
        for (parts) |p| {
            if (p.window != 0) return error.NvmeWindowPart;
            if (p.row > 0) {
                if (p.per == 0 or block % p.per != 0 or p.bytes % p.row != 0) return error.NvmeLayout;
                const n = p.bytes / p.row;
                if (n < at / p.per or n > @as(u64, at / p.per) + 1) return error.NvmeLayout;
            }
            bytes = std.math.add(u64, bytes, p.bytes) catch return error.NvmeLayout;
        }
        return .{ .parts = parts, .at = at, .block = block, .bytes = bytes };
    }

    pub fn blockBytes(l: Layout) u64 {
        var n: u64 = 0;
        for (l.parts) |p| if (p.row > 0) {
            n += @as(u64, l.block / p.per) * p.row;
        };
        return n;
    }

    pub fn total(l: Layout) u64 {
        return l.bytes;
    }

    /// The complete blocks in position order; each packs all growing planes back to back.
    pub fn blockRanges(l: Layout, index: u32, out: *std.ArrayList(Range), a: std.mem.Allocator) !void {
        if (index >= l.at / l.block) return error.NvmeBlockRange;
        var off: u64 = 0;
        for (l.parts) |p| {
            defer off += p.bytes;
            if (p.row == 0) continue;
            const len = @as(u64, l.block / p.per) * p.row;
            try out.append(a, .{ .off = off + @as(u64, index) * len, .len = len });
        }
    }

    pub fn smallRanges(l: Layout, out: *std.ArrayList(Range), a: std.mem.Allocator) !void {
        var off: u64 = 0;
        for (l.parts) |p| {
            defer off += p.bytes;
            const from = if (p.row == 0) 0 else @as(u64, l.at / p.per) * p.row;
            if (from < p.bytes) try out.append(a, .{ .off = off + from, .len = p.bytes - from });
        }
    }

    pub fn smallBytes(l: Layout) u64 {
        return l.total() - @as(u64, l.at / l.block) * l.blockBytes();
    }
};

test "block slots and the small slot cover every image byte once, independent of image size" {
    const a = std.testing.allocator;
    for ([_]u32{ 4, 8, 20, 80 }) |at| {
        const ps = [_]Part{ .{ .bytes = 72 }, .{ .bytes = at * 8, .row = 8 }, .{ .bytes = 17 }, .{ .bytes = at / 4 * 6, .row = 6, .per = 4 }, .{ .bytes = 21 } };
        const l = try Layout.init(&ps, at, 4);
        try std.testing.expectEqual(@as(u64, 110), l.smallBytes());
        var seen = try a.alloc(u8, @intCast(l.total()));
        defer a.free(seen);
        @memset(seen, 0);
        var rs: std.ArrayList(Range) = .empty;
        defer rs.deinit(a);
        for (0..at / l.block) |k| {
            rs.clearRetainingCapacity();
            try l.blockRanges(@intCast(k), &rs, a);
            try std.testing.expectEqual(l.blockBytes(), totalRanges(rs.items));
            for (rs.items) |r| for (seen[@intCast(r.off)..][0..@intCast(r.len)]) |*b| {
                b.* += 1;
            };
        }
        rs.clearRetainingCapacity();
        try l.smallRanges(&rs, a);
        try std.testing.expectEqual(l.smallBytes(), totalRanges(rs.items));
        for (rs.items) |r| for (seen[@intCast(r.off)..][0..@intCast(r.len)]) |*b| {
            b.* += 1;
        };
        for (seen) |v| try std.testing.expectEqual(@as(u8, 1), v);
    }
}

fn totalRanges(rs: []const Range) u64 {
    var n: u64 = 0;
    for (rs) |r| n += r.len;
    return n;
}

test "noncommitted tails and ring layouts are refused before any storage allocation" {
    try std.testing.expectError(error.NvmeLayout, Layout.init(&.{.{ .bytes = 13, .row = 1 }}, 13, 4));
    try std.testing.expectError(error.NvmeLayout, Layout.init(&.{.{ .bytes = 16, .row = 1, .per = 0 }}, 16, 4));
    try std.testing.expectError(error.NvmeWindowPart, Layout.init(&.{.{ .bytes = 16, .row = 1, .window = 16 }}, 16, 4));
}
