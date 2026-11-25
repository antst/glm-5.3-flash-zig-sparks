//! A new prompt's reservation must settle retired source leases before its backend pass begins.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const tier = @import("prompt_tier.zig");
const a = std.testing.allocator;

const Pool = struct {
    const Image = struct { bytes: u64, pinned: bool = false, retired: bool = false };
    images: std.AutoHashMapUnmanaged(usize, Image) = .empty,
    next: usize = 1,
    used: u64 = 0,
    room_calls: usize = 0,
    phase: enum { mark, boundary, pass } = .mark,

    fn of(p: *anyopaque) *Pool {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return at;
    }
    fn spare(_: *anyopaque) u64 {
        return 0;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f = of(p);
        if (f.used + at > 10) return error.NoSnapshotRoom;
        const id = f.next;
        f.next += 1;
        try f.images.put(a, id, .{ .bytes = at });
        f.used += at;
        return @ptrFromInt(id);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
    fn drop(p: *anyopaque, saved: pc.Saved) void {
        const f = of(p);
        const id = @intFromPtr(saved);
        const v = f.images.getPtr(id).?;
        if (v.pinned) {
            v.retired = true;
        } else f.used -= f.images.fetchRemove(id).?.value.bytes;
    }
    fn queued(p: *anyopaque, e: *const pc.Entry, _: []const u32) void {
        const f = of(p);
        std.debug.assert(f.phase != .boundary);
        f.images.getPtr(@intFromPtr(e.saved)).?.pinned = true;
    }
    fn room(p: *anyopaque, need: u64) void {
        const f = of(p);
        std.debug.assert(f.phase == .boundary);
        if (need <= 10 - f.used) return;
        var it = f.images.iterator();
        while (it.next()) |v| if (v.value_ptr.retired) {
            f.room_calls += 1;
            const id = v.key_ptr.*;
            f.used -= f.images.fetchRemove(id).?.value.bytes;
            return;
        };
    }
    fn recall(_: *anyopaque, _: *pc.Store, _: []const u32, _: u32, _: []const u32, _: []const [2]u32, have: ?*pc.Entry) ?*pc.Entry {
        return have;
    }
    fn pump(_: *anyopaque, _: u64) void {}
    fn pending(_: *anyopaque) bool {
        return false;
    }
    fn stop(p: *anyopaque, _: u64) void {
        var it = of(p).images.valueIterator();
        while (it.next()) |v| v.pinned = false;
    }
    fn destroy(_: *anyopaque) void {}
};

test "bounded tier: reserve retires the old pinned image and prepares physical room before the next save" {
    var f: Pool = .{};
    defer f.images.deinit(a);
    var vtable: tier.Tier.VTable = .{
        .queued = Pool.queued,
        .recall = Pool.recall,
        .pump = Pool.pump,
        .pending = Pool.pending,
        .stop = Pool.stop,
        .destroy = Pool.destroy,
    };
    if (@hasField(tier.Tier.VTable, "room")) vtable.room = Pool.room;
    var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &.{
        .bytes = Pool.bytes,
        .save = Pool.save,
        .restore = Pool.restore,
        .drop = Pool.drop,
        .spare = Pool.spare,
    } }, .{ .min_prompt = 0, .min_gap = 1 }, 10);
    s.tier = .{ .ptr = &f, .vtable = &vtable };
    defer s.deinit();
    try std.testing.expect(s.keep(&.{ 1, 2, 3, 4, 5, 6, 7 }, 6, null, &.{}, &.{}));
    f.phase = .boundary;
    const prompt = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    const plan = try s.begin(a, &prompt, 8, &.{}, &.{}, null, &.{});
    defer a.free(plan.marks);
    try std.testing.expectEqualSlices(u32, &.{8}, plan.marks);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    f.phase = .pass;
    try std.testing.expect(s.keep(&prompt, 8, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), f.room_calls);
    try std.testing.expectEqual(@as(u64, 8), f.used);
}

test "bounded tier: a disk restore settles its evicted source instead of refusing physical room" {
    var f: Pool = .{};
    defer f.images.deinit(a);
    var vtable: tier.Tier.VTable = .{
        .queued = Pool.queued,
        .recall = Pool.recall,
        .pump = Pool.pump,
        .pending = Pool.pending,
        .stop = Pool.stop,
        .destroy = Pool.destroy,
    };
    if (@hasField(tier.Tier.VTable, "room")) vtable.room = Pool.room;
    var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &.{
        .bytes = Pool.bytes,
        .save = Pool.save,
        .restore = Pool.restore,
        .drop = Pool.drop,
        .spare = physicalSpare,
        .reuses = reuses,
    } }, .{ .min_prompt = 0, .min_gap = 1 }, 10);
    s.tier = .{ .ptr = &f, .vtable = &vtable };
    defer s.deinit();
    try std.testing.expect(s.keep(&.{ 1, 2, 3, 4, 5, 6, 7 }, 6, null, &.{}, &.{}));
    f.phase = .boundary;
    const prompt = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    try std.testing.expect(tier.makeRoom(&s, null, &prompt, prompt.len, 8, 8));
    try std.testing.expectEqual(@as(usize, 1), f.room_calls);
    try std.testing.expectEqual(@as(u64, 0), f.used);
}

fn physicalSpare(p: *anyopaque) u64 {
    var used: u64 = 0;
    var it = Pool.of(p).images.valueIterator();
    while (it.next()) |v| if (!v.retired) {
        used += v.bytes;
    };
    return 10 - used;
}

fn reuses(p: *anyopaque, at: u32) bool {
    return Pool.of(p).used + at <= 10;
}

test "bounded tier: optional prompt lookup defers the physical wait until a marked backend boundary" {
    var f: Pool = .{};
    defer f.images.deinit(a);
    const vt: tier.Tier.VTable = .{
        .queued = Pool.queued,
        .recall = Pool.recall,
        .pump = Pool.pump,
        .pending = Pool.pending,
        .room = Pool.room,
        .stop = Pool.stop,
        .destroy = Pool.destroy,
    };
    var s = pc.Store.init(a, .{ .ptr = &f, .vtable = &.{
        .bytes = Pool.bytes,
        .save = Pool.save,
        .restore = Pool.restore,
        .drop = Pool.drop,
        .spare = Pool.spare,
    } }, .{ .min_prompt = 0, .min_gap = 1 }, 10);
    s.tier = .{ .ptr = &f, .vtable = &vt };
    s.room_at_mark = true;
    defer s.deinit();
    try std.testing.expect(s.keep(&.{ 1, 2, 3, 4, 5, 6, 7 }, 6, null, &.{}, &.{}));
    f.phase = .boundary;
    const prompt = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    const plan = try s.lookup(a, &prompt, 8, &.{}, &.{}, &.{});
    defer a.free(plan.marks);
    try std.testing.expectEqualSlices(u32, &.{8}, plan.marks);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(usize, 0), f.room_calls);
    try std.testing.expectEqual(@as(u64, 6), f.used);
    tier.prepareRoom(&s, s.family.vtable.bytes(s.family.ptr, plan.marks[0]));
    f.phase = .pass;
    try std.testing.expect(s.keep(&prompt, 8, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), f.room_calls);
    try std.testing.expectEqual(@as(u64, 8), f.used);
}
