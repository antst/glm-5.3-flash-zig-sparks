//! Host checks for fsio.
const source_api = @import("fsio.zig");
const std = source_api.std;
const c = source_api.c;
const alignment = source_api.alignment;
const has_direct = source_api.has_direct;
const Event = source_api.Event;
const Hooks = source_api.Hooks;
const fsync = source_api.fsync;
const openAt = source_api.openAt;
const statAt = source_api.statAt;
const Writer = source_api.Writer;
const openDirAt = source_api.openDirAt;
const isSymlinkAt = source_api.isSymlinkAt;
const ensureDirAt = source_api.ensureDirAt;
const removeTree = source_api.removeTree;
const Lock = source_api.Lock;
const testing = source_api.testing;
const testRoot = source_api.testRoot;
const freshDir = source_api.freshDir;
const dropDir = source_api.dropDir;
const Script = source_api.Script;
const readBack = source_api.readBack;
const countTemps = source_api.countTemps;

test "put: short writes and EINTR still write every byte; pages dropped after the fsync, then renamed" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "put");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var script: Script = .{ .steps = &.{ .{ .short = 3 }, .intr, .{ .short = 1 }, .intr, .pass } };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite, .event = Script.event };
    var w = Writer.init(&hooks, true);
    const data = "the quick brown fox jumps over the lazy dog";
    const crc = try w.put(dir, "a.bin", .{ .parts = &.{ data[0..10], data[10..] } });
    try testing.expectEqual(std.hash.Crc32.hash(data), crc);
    const back = try readBack(gpa, dir, "a.bin");
    defer gpa.free(back);
    try testing.expectEqualStrings(data, back);
    try testing.expectEqualSlices(Event, &.{ .write, .fsync, .drop_pages, .rename }, script.events.items);
    try testing.expectEqual(@as(usize, 0), try countTemps(gpa, dir));
}

test "put: a full disk leaves no temporary file and no target" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "nospace");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var script: Script = .{ .steps = &.{ .{ .short = 4 }, .nospace } };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite };
    var w = Writer.init(&hooks, false);
    try testing.expectError(error.NoSpace, w.put(dir, "b.bin", .{ .parts = &.{"0123456789"} }));
    try testing.expectEqual(@as(usize, 0), try countTemps(gpa, dir));
    try testing.expectError(error.Missing, statAt(dir, "b.bin"));
}

test "put: aligned sources, direct where it works, buffered when the file system refuses O_DIRECT" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "aligned");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 3 * alignment);
    defer gpa.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i % 251);
    inline for (.{ false, true }) |refuse| {
        const hooks: Hooks = .{ .no_direct = refuse };
        var w = Writer.init(&hooks, true);
        const name = if (refuse) "buffered.blk" else "direct.blk";
        _ = try w.put(dir, name, .{ .aligned = .{ .buf = buf, .len = 2 * alignment + 100 } });
        const back = try readBack(gpa, dir, name);
        defer gpa.free(back);
        try testing.expectEqualSlices(u8, buf[0 .. 2 * alignment + 100], back);
        if (refuse and has_direct) try testing.expect(!w.direct.load(.monotonic) and w.direct_refused.load(.monotonic) == 1);
    }
}

test "a partial direct write off the 4 KiB grid finishes buffered" {
    if (!has_direct) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "partial");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 2 * alignment);
    defer gpa.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i % 249);
    var script: Script = .{ .steps = &.{.{ .short = 100 }} };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite };
    var w = Writer.init(&hooks, true);
    // (on a file system without O_DIRECT the open falls back first; either way the bytes are whole)
    _ = try w.put(dir, "p.blk", .{ .aligned = .{ .buf = buf, .len = alignment + 7 } });
    const back = try readBack(gpa, dir, "p.blk");
    defer gpa.free(back);
    try testing.expectEqualSlices(u8, buf[0 .. alignment + 7], back);
}

test "owner lock: a second owner is refused while the first holds it" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "lock");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var a = try Lock.take(dir);
    try testing.expectError(error.Busy, Lock.take(dir));
    a.release();
    var b = try Lock.take(dir);
    b.release();
}

test "directories: a symlink is never followed, by an open or a tree removal" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "symlink");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    try ensureDirAt(dir, "target");
    const t = try openDirAt(dir, "target");
    defer _ = c.close(t);
    const f = try openAt(t, "keep.me", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, 0o600);
    _ = c.close(f);
    try ensureDirAt(dir, "tree");
    const tr = try openDirAt(dir, "tree");
    defer _ = c.close(tr);
    try testing.expect(c.symlinkat("../target", tr, "link") == 0);
    try testing.expect(c.symlinkat("target", dir, "top") == 0);
    try testing.expect(isSymlinkAt(dir, "top") and !isSymlinkAt(dir, "target"));
    if (openDirAt(dir, "top")) |fd| {
        _ = c.close(fd);
        return error.TestUnexpectedResult;
    } else |_| {}
    removeTree(gpa, dir, "tree");
    try testing.expect(!isSymlinkAt(tr, "link"));
    _ = try statAt(t, "keep.me"); // the link's target and its contents are untouched
}
