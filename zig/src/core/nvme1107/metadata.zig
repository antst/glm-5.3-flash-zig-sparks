//! Canonical GLM checkpoint routes, compared in full before any restored image is exposed.
const std = @import("std");
pub const Span = [2]u32;

pub fn encode(a: std.mem.Allocator, at: u32, grid: u32, lookahead: u32, spans: []const Span, starts: []const u32) ![]u8 {
    if (at == 0 or grid == 0 or lookahead > 1 or spans.len > 4096 or starts.len > 8192) return error.NvmeMetadata;
    var count: usize = 0;
    var last: u32 = 0;
    for (spans, 0..) |s, i| {
        if (s[0] >= s[1] or (i > 0 and s[0] <= last)) return error.NvmeMetadata;
        last = s[1];
        count += @intFromBool(s[0] < at);
    }
    var cuts: std.ArrayList(u32) = .empty;
    defer cuts.deinit(a);
    if (starts.len == 0) {
        var k: u32 = grid;
        while (k < at) : (k = std.math.add(u32, k, grid) catch return error.NvmeMetadata) try cuts.append(a, k);
    } else {
        last = 0;
        for (starts) |k| {
            if (k == 0 or k <= last) return error.NvmeMetadata;
            last = k;
            if (k < at) try cuts.append(a, k);
        }
    }
    if (at % grid != 0 and (starts.len == 0 or std.mem.indexOfScalar(u32, starts, at) == null)) return error.NvmeMetadata;
    const bytes = try a.alloc(u8, 24 + count * 8 + cuts.items.len * 4);
    const header = [6]u32{ 0x314e4c47, 1, at, lookahead, @intCast(count), @intCast(cuts.items.len) };
    @memcpy(bytes[0..24], std.mem.asBytes(&header));
    var off: usize = 24;
    for (spans[0..count]) |s| {
        const clipped = [2]u32{ s[0], @min(s[1], at) };
        @memcpy(bytes[off..][0..8], std.mem.asBytes(&clipped));
        off += 8;
    }
    @memcpy(bytes[off..], std.mem.sliceAsBytes(cuts.items));
    return bytes;
}

test "NVMe route metadata: future intervals do not change a prefix, but consumed row routes do" {
    const a = std.testing.allocator;
    const base = try encode(a, 32, 16, 1, &.{.{ 4, 8 }}, &.{});
    defer a.free(base);
    const same = try encode(a, 32, 16, 1, &.{ .{ 4, 8 }, .{ 32, 40 } }, &.{ 16, 32, 48 });
    defer a.free(same);
    try std.testing.expectEqualSlices(u8, base, same);
    const other = try encode(a, 32, 16, 1, &.{.{ 4, 9 }}, &.{});
    defer a.free(other);
    try std.testing.expect(!std.mem.eql(u8, base, other));
    try std.testing.expectError(error.NvmeMetadata, encode(a, 32, 16, 1, &.{ .{ 4, 8 }, .{ 8, 10 } }, &.{}));
    try std.testing.expectError(error.NvmeMetadata, encode(a, 32, 16, 1, &.{}, &.{ 16, 16 }));
}
