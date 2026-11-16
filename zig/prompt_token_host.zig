//! The production prompt store's metadata over immutable million-token histories, without a device.
const std = @import("std");
const pc = @import("engine_api").prompt_cache;
const t = std.testing;
const a = t.allocator;
const Family = struct {
    at: u32 = 0,
    live: usize = 0,
    fail: bool = false,
    fn self(p: *anyopaque) *Family {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return at;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) !pc.Saved {
        const f = self(p);
        if (f.fail or f.at != at) return error.BadFixtureSave;
        const n = try a.create(u32);
        n.* = at;
        f.live += 1;
        return n;
    }
    fn restore(p: *anyopaque, _: ?*anyopaque, saved: pc.Saved) !void {
        self(p).at = @as(*u32, @ptrCast(@alignCast(saved))).*;
    }
    fn drop(p: *anyopaque, saved: pc.Saved) void {
        a.destroy(@as(*u32, @ptrCast(@alignCast(saved))));
        self(p).live -= 1;
    }
    fn snapshots(f: *Family) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

test "prompt metadata: checkpoints of one million-token prompt share the full canonical history" {
    var family: Family = .{};
    var store = pc.Store.init(a, family.snapshots(), .{ .min_prompt = 0, .min_gap = 0 }, 1 << 30);
    const prompt = try a.alloc(u32, 1048576);
    defer a.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast(1 + i % 150000);
    for (1..21) |i| {
        family.at = @intCast(i * 8192);
        try t.expect(store.keep(prompt, family.at, null, &.{}, &.{}));
    }
    const old_bytes = 20 * prompt.len * 4;
    try t.expect(store.token_pool.bytes < old_bytes / 10);
    for (store.entries.items) |entry| {
        try t.expect(entry.tokens.ptr == store.entries.items[0].tokens.ptr);
        try t.expectEqualSlices(u32, prompt[0..entry.tokens.len], entry.tokens);
        try t.expectEqualSlices(u32, prompt, entry.last);
    }
    std.debug.print("PROMPT_TOKEN_METADATA independent={d} shared={d}\n", .{ old_bytes, store.token_pool.bytes });
    const next = try a.alloc(u32, prompt.len + 8);
    defer a.free(next);
    @memcpy(next[0..prompt.len], prompt);
    @memset(next[prompt.len..], 11);
    const e = store.entries.items[0];
    store.resumed(e, next, true);
    try t.expectEqualSlices(u32, next, e.last);
    try t.expectEqualSlices(u32, prompt[0..e.tokens.len], e.tokens);
    while (store.entries.items.len > 1) store.remove(1);
    store.remove(0);
    try t.expectEqual(@as(usize, 0), family.live);
    try t.expectEqual(@as(usize, 0), store.token_pool.bytes);
    store.deinit();
}

test "prompt metadata: a failed snapshot and a divergent resume cannot damage earlier histories" {
    var family: Family = .{};
    var store = pc.Store.init(a, family.snapshots(), .{ .min_prompt = 0, .min_gap = 0 }, 1 << 20);
    defer store.deinit();
    var prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    family.at = 4;
    try t.expect(store.keep(&prompt, 4, null, &.{}, &.{}));
    const held = store.token_pool.bytes;
    family.at = 6;
    family.fail = true;
    try t.expect(!store.keep(&prompt, 6, null, &.{}, &.{}));
    try t.expectEqual(held, store.token_pool.bytes);
    const entry = store.entries.items[0];
    prompt[0] = 19;
    store.resumed(entry, &prompt, true);
    try t.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, entry.tokens);
    try t.expectEqualSlices(u32, &prompt, entry.last);
    store.remove(0);
    try t.expectEqual(@as(usize, 0), store.token_pool.bytes);
}

test {
    _ = pc;
}

test "prompt metadata: cold allocation failures publish no lease and later retry succeeds" {
    var induced: usize = 0;
    for (0..8) |fail_at| {
        var family: Family = .{};
        var store = pc.Store.init(a, family.snapshots(), .{}, 1 << 20);
        defer store.deinit();
        var failing = t.FailingAllocator.init(a, .{ .fail_index = fail_at, .resize_fail_index = 0 });
        const got = store.token_pool.take(failing.allocator(), &.{ 1, 2, 3, 4 });
        if (failing.has_induced_failure) {
            induced += 1;
            try t.expectError(error.OutOfMemory, got);
            try t.expectEqual(@as(usize, 0), store.token_pool.bytes);
            try t.expectEqual(@as(usize, 0), store.token_pool.slabs.items.len);
        } else {
            const lease = try got;
            lease.release(a);
            break;
        }
        const retry = try store.token_pool.take(a, &.{ 1, 2, 3, 4 });
        try t.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, retry.words);
        retry.release(a);
        try t.expectEqual(@as(usize, 0), store.token_pool.bytes);
    }
    try t.expectEqual(@as(usize, 3), induced);
}

test "prompt metadata: a budget refusal preserves an immutable borrowed history" {
    var family: Family = .{};
    var store = pc.Store.init(a, family.snapshots(), .{}, 1 << 20);
    defer store.deinit();
    const old = try store.token_pool.take(a, &.{ 1, 2, 3, 4 });
    defer old.release(a);
    store.token_pool.limit = store.token_pool.bytes;
    try t.expectError(error.OutOfMemory, store.token_pool.take(a, &.{ 9, 2, 3, 4 }));
    try t.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, old.words);
    const same = try store.token_pool.take(a, &.{ 1, 2, 3, 4 });
    defer same.release(a);
    try t.expect(same.words.ptr == old.words.ptr);
}
