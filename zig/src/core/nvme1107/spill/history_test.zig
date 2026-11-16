//! Shared histories preserve bytes across restarts and keep every referenced prefix alive.
const std = @import("std");
const history = @import("history.zig");
const fsio = @import("fsio.zig");
const format = @import("format.zig");
const gpa = std.testing.allocator;
const t = std.testing;
const c = std.c;

fn tokenBytes(n: usize) ![]u8 {
    const b = try gpa.alloc(u8, n * 4);
    for (0..n) |i| std.mem.writeInt(u32, b[i * 4 ..][0..4], @intCast(i % 120_000), .little);
    return b;
}

const Rig = struct {
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    fd: c.fd_t = -1,
    hooks: fsio.Hooks = .{},
    writer: fsio.Writer = undefined,
    h: ?history.History = null,

    fn init(r: *Rig, tag: []const u8) !void {
        r.root = try fsio.testRoot(&r.path, tag);
        r.fd = try fsio.freshDir(gpa, r.root);
        r.writer = fsio.Writer.init(&r.hooks, false);
        r.h = try history.History.open(gpa, r.fd, 1 << 20);
        try r.h.?.finishRecovery(true);
    }
    fn deinit(r: *Rig) void {
        if (r.h) |*h| h.deinit();
        if (r.fd >= 0) _ = c.close(r.fd);
        fsio.dropDir(gpa, r.root);
    }
};

test "one million token prefixes share all complete chunks; releasing either turn preserves the other" {
    var r: Rig = .{};
    try r.init("history-million");
    defer r.deinit();
    const ids = try tokenBytes(1_048_576 + 300);
    defer gpa.free(ids);
    const h = &r.h.?;
    const a = try h.encode(&r.writer, ids[0 .. 1_048_576 * 4], 1_048_576, false);
    defer gpa.free(a.bytes);
    h.pin(a.roots);
    const written = h.bytes;
    const b = try h.encode(&r.writer, ids, 1_048_576 + 300, false);
    defer gpa.free(b.bytes);
    h.pin(b.roots);
    try t.expectEqual(written, h.bytes);
    try t.expectEqual(@as(usize, 512), h.nodes.count());
    try t.expectEqual(@as(usize, 80 + 300 * 4), b.bytes.len);
    try h.release(a.roots);
    const actual = try h.decode(b.bytes, 1_048_576 + 300, false);
    defer gpa.free(actual);
    try t.expectEqualSlices(u8, ids, actual);
    try h.release(b.roots);
    try t.expectEqual(@as(usize, 0), h.nodes.count());
    try t.expectEqual(@as(u64, 0), h.bytes);
}

test "shared histories reopen byte exact and remove only unreferenced crash debris" {
    var r: Rig = .{};
    try r.init("history-reopen");
    defer r.deinit();
    const ids = try tokenBytes(6_144 + 9);
    defer gpa.free(ids);
    const a = try r.h.?.encode(&r.writer, ids[0 .. 4_096 * 4], 4_096, false);
    defer gpa.free(a.bytes);
    r.h.?.pin(a.roots);
    const abandoned = try r.h.?.encode(&r.writer, ids, 6_144 + 9, false);
    defer gpa.free(abandoned.bytes);
    r.h.?.deinit();
    r.h = null;
    r.h = try history.History.open(gpa, r.fd, 1 << 20);
    const plain = try r.h.?.decode(a.bytes, 4_096, false);
    defer gpa.free(plain);
    r.h.?.pin(a.roots);
    try r.h.?.finishRecovery(true);
    try t.expectEqualSlices(u8, ids[0 .. 4_096 * 4], plain);
    var files = try fsio.list(gpa, r.h.?.fd);
    defer fsio.freeList(gpa, &files);
    try t.expectEqual(@as(usize, 2), files.items.len);
    try t.expectEqual(@as(usize, 2), r.h.?.nodes.count());
}

test "pictured raw ids and content keys share their own prefixes without mixing streams" {
    var r: Rig = .{};
    try r.init("history-raw");
    defer r.deinit();
    const ids = try tokenBytes(2 * (4_096 + 13));
    defer gpa.free(ids);
    const a = try r.h.?.encode(&r.writer, ids, 4_096 + 13, true);
    defer gpa.free(a.bytes);
    r.h.?.pin(a.roots);
    const actual = try r.h.?.decode(a.bytes, 4_096 + 13, true);
    defer gpa.free(actual);
    try t.expectEqualSlices(u8, ids, actual);
    try t.expect(!std.mem.eql(u8, &a.roots.keyed, &a.roots.raw));
}

test "corrupted history data and altered lengths are refused before returning token bytes" {
    var r: Rig = .{};
    try r.init("history-damage");
    defer r.deinit();
    const ids = try tokenBytes(4_096);
    defer gpa.free(ids);
    const a = try r.h.?.encode(&r.writer, ids, 4_096, false);
    defer gpa.free(a.bytes);
    r.h.?.pin(a.roots);
    var buf: [80]u8 = undefined;
    const n = try std.fmt.bufPrintSentinel(&buf, "{s}.idsb", .{format.hex(&a.roots.keyed)}, 0);
    _ = try r.writer.put(r.h.?.fd, n, .{ .parts = &.{"damaged"} });
    try t.expectError(error.Corrupt, r.h.?.decode(a.bytes, 4_096, false));
    try t.expectError(error.Corrupt, history.History.parse(a.bytes, 4_097, false));
}

test "an uncertain scan preserves every history object until a later complete recovery" {
    var r: Rig = .{};
    try r.init("history-uncertain");
    defer r.deinit();
    const ids = try tokenBytes(4_096);
    defer gpa.free(ids);
    const a = try r.h.?.encode(&r.writer, ids, 4_096, false);
    defer gpa.free(a.bytes);
    r.h.?.deinit();
    r.h = null;
    r.h = try history.History.open(gpa, r.fd, 1 << 20);
    try r.h.?.finishRecovery(false);
    var files = try fsio.list(gpa, r.h.?.fd);
    defer fsio.freeList(gpa, &files);
    try t.expectEqual(@as(usize, 2), files.items.len);
    try r.h.?.collect();
    const plain = try r.h.?.decode(a.bytes, 4_096, false);
    defer gpa.free(plain);
    try t.expectEqualSlices(u8, ids, plain);
}
