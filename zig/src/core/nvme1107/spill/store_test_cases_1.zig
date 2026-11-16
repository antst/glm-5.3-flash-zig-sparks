//! Host checks for store_test.
const source_api = @import("store_test.zig");
const std = source_api.std;
const c = source_api.c;
const format = source_api.format;
const fsio = source_api.fsio;
const mutants = source_api.mutants;
const testing = source_api.testing;
const block_tokens = source_api.block_tokens;
const bpt = source_api.bpt;
const max_tokens = source_api.max_tokens;
const CheckFailed = source_api.CheckFailed;
const H = source_api.H;
const settle = source_api.settle;
const write = source_api.write;
const restore = source_api.restore;
const idsOf = source_api.idsOf;
const lin = source_api.lin;
const checks = source_api.checks;
const kills = source_api.kills;
const checkNamed = source_api.checkNamed;

test "recency: ten thousand hits coalesce without disk writes, then persist without KV rewrites" {
    const h = try H.init(testing.allocator, .{ .tag = "recency-batch" });
    defer h.deinit();
    const ids = try idsOf(testing.allocator, 3 * block_tokens, 2900);
    defer testing.allocator.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(90), .source = 90 });
    var nb: [80]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&nb, "{s}.man", .{format.hex(&key)}, 0);
    const before = try fsio.statAt(h.s.points_fd, name);
    const start = h.s.counts();
    for (0..10000) |_| h.s.touch(key);
    try testing.expect(h.s.drain(5 * std.time.ns_per_s).done);
    const dirty = h.s.counts();
    try testing.expectEqual(@as(u64, 1), dirty.recency_pending);
    try testing.expectEqual(@as(u64, 0), dirty.recency_writes);
    try testing.expectEqual(before.mtime_ns, (try fsio.statAt(h.s.points_fd, name)).mtime_ns);
    const used = h.s.points.get(key).?.used;
    try testing.expect(h.s.flushRecency(5 * std.time.ns_per_s));
    const done = h.s.counts();
    try testing.expectEqual(@as(u64, 1), done.recency_writes);
    try testing.expectEqual(@as(u64, 0), done.recency_pending);
    try testing.expectEqual(start.written_bytes, done.written_bytes);
    try testing.expectEqual(start.written_small_bytes, done.written_small_bytes);
    try h.reopen();
    try testing.expectEqual(used, h.s.points.get(key).?.used);
    const got = (try restore(h, key)) orelse return error.CheckFailed;
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u8, h.src(90)[0..got.len], got);
}

test "recency: a failed timestamp write retains dirty priority and can be retried" {
    const h = try H.init(testing.allocator, .{ .tag = "recency-retry" });
    defer h.deinit();
    const ids = try idsOf(testing.allocator, block_tokens, 2901);
    defer testing.allocator.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(91), .source = 91 });
    var nb: [80]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&nb, "{s}.man", .{format.hex(&key)}, 0);
    try testing.expectEqual(@as(c_int, 0), c.renameat(h.s.points_fd, name, h.s.points_fd, "held.man"));
    h.s.touch(key);
    try testing.expect(!h.s.flushRecency(5 * std.time.ns_per_s));
    const failed = h.s.counts();
    try testing.expectEqual(@as(u64, 1), failed.recency_errors);
    try testing.expectEqual(@as(u64, 1), failed.recency_pending);
    try testing.expect(h.s.presence(key));
    try testing.expectEqual(@as(c_int, 0), c.renameat(h.s.points_fd, "held.man", h.s.points_fd, name));
    try testing.expect(h.s.flushRecency(5 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 0), h.s.counts().recency_pending);
    try h.reopen();
    try testing.expect(h.s.presence(key));
}

test "spill store: the GLM 0110 checks and the native port's own, all pass" {
    var failed: usize = 0;
    for (checks) |ch| {
        ch.f(testing.allocator) catch |e| {
            std.debug.print("FAIL {s}: {t}\n", .{ ch.name, e });
            failed += 1;
        };
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

test "spill store mutants: every authored fix switched back off is caught" {
    // a mutant may leak what its broken path never frees: its runs use the page allocator
    const gpa = std.heap.page_allocator;
    source_api.quiet = std.c.getenv("SPILL_VERBOSE") == null;
    defer {
        source_api.quiet = false;
        mutants.current = .none;
    }
    var survived: usize = 0;
    var crashes: usize = 0;
    for (kills) |k| {
        mutants.current = k.m;
        if (!source_api.quiet) std.debug.print("mutant {t}\n", .{k.m});
        var killed = false;
        for (k.checks) |name| {
            // a kill is a failed assertion (CheckFailed) only; any other error is the harness or the check
            if (checkNamed(name)(gpa)) |_| {} else |e| if (e == CheckFailed) {
                killed = true;
            } else {
                std.debug.print("CRASHED {t} in {s}: {t}\n", .{ k.m, name, e });
                crashes += 1;
            }
        }
        mutants.current = .none;
        if (!killed) {
            std.debug.print("SURVIVED {t}\n", .{k.m});
            survived += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), survived);
    try testing.expectEqual(@as(usize, 0), crashes);
}

test "shared token-history files survive store restart and release only the deleted turn" {
    const gpa = testing.allocator;
    const h = try H.init(gpa, .{ .tag = "history-store" });
    defer h.deinit();
    h.closeStore();
    h.cfg.limits.max_tokens = 8192;
    try h.open();
    const device = try gpa.alloc(u8, 8192 * bpt);
    @memset(device, 0x63);
    try h.rows.put(gpa, 1, device);
    const ids = try idsOf(gpa, 4096 + 13, 71);
    defer gpa.free(ids);
    const a = try write(h, ids[0..4096], .{ .lineage = .{ .root = @splat(1) }, .source = 1 });
    const b = try write(h, ids, .{ .lineage = .{ .root = @splat(2) }, .source = 1 });
    try testing.expectEqual(@as(usize, 2), h.s.histories.?.nodes.count());
    try testing.expect(h.s.points.get(a).?.ids_disk_bytes < 128);
    try testing.expect(h.s.points.get(b).?.ids_disk_bytes < 256);
    try h.reopen();
    try testing.expectEqual(@as(u64, 2), h.s.counts().recovered_points);
    const read = (try restore(h, b)) orelse return error.NoRestore;
    defer gpa.free(read);
    try testing.expectEqualSlices(u8, device[0 .. ids.len * bpt], read);
    h.s.forget(a, &.{}, &.{});
    try testing.expect(settle(h).done);
    try h.reopen();
    try testing.expect(!h.s.presence(a));
    try testing.expect(h.s.presence(b));
    try testing.expectEqual(@as(usize, 2), h.s.histories.?.nodes.count());
}
