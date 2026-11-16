//! One atomic host-memory budget shared by the index, writer and reader metadata.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

pub const Budget = struct {
    parent: Allocator,
    limit: usize,
    used: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    refused: std.atomic.Value(u64) = .init(0),

    pub fn init(parent: Allocator, limit: usize) Budget {
        return .{ .parent = parent, .limit = limit };
    }
    pub fn allocator(b: *Budget) Allocator {
        return .{ .ptr = b, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn of(p: *anyopaque) *Budget {
        return @ptrCast(@alignCast(p));
    }
    fn take(b: *Budget, len: usize) bool {
        var old = b.used.load(.monotonic);
        while (true) {
            if (len > b.limit -| old) {
                _ = b.refused.fetchAdd(1, .monotonic);
                return false;
            }
            if (b.used.cmpxchgWeak(old, old + len, .monotonic, .monotonic)) |next| old = next else {
                var high = b.peak.load(.monotonic);
                while (high < old + len) {
                    high = b.peak.cmpxchgWeak(high, old + len, .monotonic, .monotonic) orelse break;
                }
                return true;
            }
        }
    }
    fn give(b: *Budget, len: usize) void {
        const old = b.used.fetchSub(len, .monotonic);
        std.debug.assert(old >= len);
    }
    fn alloc(p: *anyopaque, len: usize, alignment: Alignment, ra: usize) ?[*]u8 {
        const b = of(p);
        if (!b.take(len)) return null;
        return b.parent.rawAlloc(len, alignment, ra) orelse {
            b.give(len);
            return null;
        };
    }
    fn resize(p: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) bool {
        const b = of(p);
        const extra = new_len -| memory.len;
        if (!b.take(extra)) return false;
        if (!b.parent.rawResize(memory, alignment, new_len, ra)) {
            b.give(extra);
            return false;
        }
        b.give(memory.len -| new_len);
        return true;
    }
    fn remap(p: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const b = of(p);
        const extra = new_len -| memory.len;
        if (!b.take(extra)) return null;
        const out = b.parent.rawRemap(memory, alignment, new_len, ra) orelse {
            b.give(extra);
            return null;
        };
        b.give(memory.len -| new_len);
        return out;
    }
    fn free(p: *anyopaque, memory: []u8, alignment: Alignment, ra: usize) void {
        const b = of(p);
        b.parent.rawFree(memory, alignment, ra);
        b.give(memory.len);
    }
};

test "metadata allocation refuses at its limit and leaves existing bytes intact" {
    var budget = Budget.init(std.testing.allocator, 96);
    const a = budget.allocator();
    const one = try a.alloc(u8, 64);
    @memset(one, 7);
    try std.testing.expectError(error.OutOfMemory, a.alloc(u8, 33));
    try std.testing.expectEqual(@as(usize, 64), budget.used.load(.monotonic));
    for (one) |v| try std.testing.expectEqual(@as(u8, 7), v);
    const two = try a.alloc(u8, 32);
    try std.testing.expectEqual(@as(usize, 96), budget.peak.load(.monotonic));
    a.free(two);
    a.free(one);
    try std.testing.expectEqual(@as(usize, 0), budget.used.load(.monotonic));
}

test "reader and writer allocations share one hard limit under contention" {
    var budget = Budget.init(std.testing.allocator, 128);
    const Run = struct {
        fn work(b: *Budget) void {
            const a = b.allocator();
            for (0..2000) |_| {
                const buf = a.alloc(u8, 64) catch continue;
                @memset(buf, 1);
                std.atomic.spinLoopHint();
                a.free(buf);
            }
        }
    };
    var workers: [4]std.Thread = undefined;
    for (&workers) |*thread| thread.* = try std.Thread.spawn(.{}, Run.work, .{&budget});
    for (workers) |thread| thread.join();
    try std.testing.expect(budget.peak.load(.monotonic) <= 128);
    try std.testing.expectEqual(@as(usize, 0), budget.used.load(.monotonic));
}
