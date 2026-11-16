//! Real ownership and metadata structures; the device and model are not invoked.
const std = @import("std");
const glm = @import("glm");
const spill = @import("core").nvme1107.spill;
const prefixes = spill.prefixes;
const t = std.testing;
const a = t.allocator;

test "GLM compact: retired-source progress windows scale without a RAM allocation or unbounded wait" {
    const ns = glm.nvme_options.progressNs;
    try t.expectEqual(@as(u64, 2 * std.time.ns_per_s), ns(0));
    try t.expectEqual(@as(u64, 2 * std.time.ns_per_s), ns(64 << 20));
    try t.expectEqual(@as(u64, 4_500_000_000), ns(1100 << 20));
    try t.expectEqual(@as(u64, 10 * std.time.ns_per_s), ns(4 << 30));
    try t.expectEqual(@as(u64, 10 * std.time.ns_per_s), ns(std.math.maxInt(u64)));
    var before: u64 = 0;
    for (0..200) |i| {
        const now = ns(i * 32 * (1 << 20));
        try t.expect(now >= before and now >= 2 * std.time.ns_per_s and now <= 10 * std.time.ns_per_s);
        before = now;
    }
}

test "GLM compact: named tier and limits refuse malformed settings, and account before admission" {
    const opts = glm.nvme_options;
    try t.expect(try opts.tier("native-tiered"));
    try t.expect(try opts.tier("1107"));
    try t.expect(!try opts.tier(null) and !try opts.tier("0"));
    try t.expectEqual(opts.Backend.off, try opts.backend("0"));
    try t.expectEqual(opts.Backend.generic, try opts.backend(null));
    try t.expectEqual(opts.Backend.native_tiered, try opts.backend("native-tiered"));
    try t.expectError(error.BadGlmNvmeTier, opts.tier("1"));
    const serial = try opts.parse(null, null, null, "131072");
    const parallel = try opts.parse("parallel", null, null, null);
    try t.expectEqual(@as(u64, 192 << 20), serial.staging);
    try t.expectEqual(@as(u64, 64 << 20), serial.index);
    try t.expectEqual(@as(u64, 352 << 20), serial.residentBound());
    try t.expectEqual(@as(u64, 512 << 20), parallel.staging);
    try t.expectEqual(@as(u64, 256 << 20), parallel.index);
    try t.expectEqual(@as(u64, (1 << 30) - (352 << 20)), try opts.reserve(serial, 1 << 30, true, true));
    try t.expectEqual(@as(u64, 1 << 30), try opts.reserve(serial, 1 << 30, false, true));
    try t.expectEqual(@as(u64, 1 << 30), try opts.reserve(serial, 1 << 30, true, false));
    try t.expectError(error.NvmeMaintenanceRoom, opts.reserve(serial, 1, true, true));
    try opts.requireCache(serial, 1048576, 2 << 30, 4 << 30);
    try t.expectError(error.NvmeCacheTooSmall, opts.requireCache(serial, 1048576, 2 << 30, 1 << 29));
    try t.expectError(error.NvmeCacheTooSmall, opts.requireCache(serial, 65536, 1, 4 << 30));
    try t.expectError(error.BadGlmNvmeMode, opts.parse("off", null, null, null));
    try t.expectError(error.BadGlmNvmeTier, opts.parse(null, "63", null, null));
    try t.expectError(error.BadGlmNvmeMinimum, opts.parse(null, null, null, "-1"));
}

test "GLM compact: the shared small arena has exactly one owner through repeated write and restore cycles" {
    const slot = spill.slots;
    const b = try a.alignedAlloc(u8, .fromByteUnits(4096), 6 * 4096);
    defer a.free(b);
    var pool = try slot.Slots.init(a, .{ .write = &.{b[0..8192]}, .read = &.{b[8192..16384]}, .small = &.{b[16384..]}, .restore = &.{b[16384..]}, .shared_small_restore = true }, 4096, 8192);
    defer pool.deinit(a);
    for (1..1001) |id| {
        const w = pool.take(.small, id).?;
        @memset(pool.bytes(w), 0xa7);
        try t.expect(pool.take(.restore, id + 1) == null);
        const independent = pool.take(.read, id).?;
        pool.give(independent);
        pool.give(w);
        const r = pool.take(.restore, id + 1).?;
        for (pool.bytes(r)) |v| try t.expectEqual(@as(u8, 0xa7), v);
        try t.expect(pool.take(.small, id) == null);
        pool.give(r);
    }
    try t.expectEqual(@as(u64, 0), pool.heldBytes());
    try t.expect(pool.peakBytes() <= b.len);
    try t.expectError(error.BadSharedStaging, slot.Slots.init(a, .{ .small = &.{b[0..8192]}, .restore = &.{b[8192..16384]}, .shared_small_restore = true }, 4096, 8192));
    try t.expectError(error.BadSharedStaging, slot.Slots.init(a, .{ .write = &.{b[0..4096]}, .small = &.{b[0..8192]}, .restore = &.{b[0..8192]}, .shared_small_restore = true }, 4096, 8192));
    const plan = try slot.serialPlan(192 << 20, 2 << 20, 151 << 20);
    try t.expectEqual(@as(u64, 159 << 20), plan.bytes());
    try t.expectError(error.StagingTooSmall, slot.serialPlan(plan.bytes() - 1, 2 << 20, 151 << 20));
    try t.expectError(error.StagingTooSmall, slot.serialPlan(std.math.maxInt(u64), std.math.maxInt(u64), 1));
}

fn chain(n: usize) ![]spill.Digest {
    const out = try a.alloc(spill.Digest, n);
    for (out, 0..) |*v, i| {
        v.* = @splat(0);
        std.mem.writeInt(u64, v[0..8], i, .little);
    }
    return out;
}

test "GLM compact: twenty million-token append checkpoints share their immutable chain prefixes" {
    var pool: prefixes.Pool = .{ .gpa = a, .io = t.io };
    defer pool.deinit();
    var leases: [20]prefixes.Lease = undefined;
    var baseline: usize = 0;
    for (&leases, 0..) |*lease, i| {
        const n = 4096 + i * 4;
        baseline += n * @sizeOf(spill.Digest);
        lease.* = try pool.take(try chain(n));
    }
    try t.expectEqual(@as(usize, 1), pool.slabs.items.len);
    try t.expect(pool.live_bytes < baseline / 10);
    for (leases) |lease| {
        try t.expect(lease.items.ptr == leases[0].items.ptr);
        const want = try chain(lease.items.len);
        defer a.free(want);
        try t.expectEqualSlices(spill.Digest, want, lease.items);
    }
    const fork = try chain(4100);
    fork[4097][0] = 0xab;
    const separate = try pool.take(fork);
    try t.expect(separate.items.ptr != leases[0].items.ptr);
    separate.release();
    std.debug.print("COMPACT_PREFIX_METADATA independent={d} shared={d}\n", .{ baseline, pool.live_bytes });
    for (leases) |lease| lease.release();
    try t.expectEqual(@as(usize, 0), pool.live_bytes);
}

fn indexInit(s: *spill.Store) void {
    s.* = undefined;
    s.gpa = a;
    s.io = t.io;
    s.have = .empty;
    s.inodes = .empty;
    s.content = .empty;
    s.tasks = .empty;
    s.work = .init;
    s.stored_bytes = 0;
    s.pending_free = 0;
}

test "GLM compact: hardlinked logical blocks share one checked physical record, with atomic refusal" {
    var s: spill.Store = undefined;
    indexInit(&s);
    defer {
        s.have.deinit(a);
        s.inodes.deinit(a);
        s.content.deinit(a);
        s.tasks.deinit(a);
    }
    const first: spill.Digest = @splat(1);
    const second: spill.Digest = @splat(2);
    const sha: spill.Digest = @splat(7);
    try s.addBlock(first, 13, sha, 55, 4096);
    try s.addBlock(second, 13, sha, 55, 4096);
    try t.expectEqual(@as(usize, 1), s.inodes.count());
    try t.expect(s.have.get(first).? == s.have.get(second).?);
    try s.addBlock(first, 13, sha, 55, 4096);
    try t.expectEqual(@as(usize, 2), s.have.get(first).?.names.items.len);
    try t.expectError(error.Corrupt, s.addBlock(@splat(3), 99, sha, 55, 4096));
    try t.expectEqual(@as(usize, 2), s.have.count());
    try t.expectEqual(@as(u64, 4096), s.stored_bytes);
    s.dropBlock(first);
    try t.expect(s.have.contains(second));
    try t.expectEqual(@as(u64, 4096), s.stored_bytes);
    s.dropBlock(second);
    try t.expectEqual(@as(u64, 0), s.stored_bytes);
    try t.expectEqual(@as(usize, 0), s.inodes.count());
}

test "GLM compact: production checkpoint creation shares chain ownership and releases every reference" {
    var s: spill.Store = undefined;
    indexInit(&s);
    s.cfg = .{ .root = "", .rank = 0, .compat = @splat(0), .limits = .{ .block_tokens = 256, .block_bytes = 1, .max_tokens = 1048576, .max_small_bytes = 1 }, .cap_bytes = 1, .staging = .{} };
    s.chain_prefixes = .{ .gpa = a, .io = t.io };
    defer s.chain_prefixes.deinit();
    const ids = try a.alloc(u32, 1048576);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i % 150000);
    var ps: [20]*spill.store.Point = undefined;
    for (&ps, 0..) |*p, i| {
        const n = 1048576 - (19 - i) * 256;
        const ch = try spill.Store.chainOf(a, ids[0..n], 256);
        p.* = try s.makePoint(ids[0..n], ch, &.{}, .{ .root = @splat(1) }, .{ .key = @splat(0), .at = @intCast(n), .shared = false, .group = 0, .created_ns = 0, .has_raw = false, .mutable_tail = false, .origin = null });
        try t.expect(p.*.chain.ptr == ps[0].chain.ptr);
    }
    try t.expectEqual(@as(usize, 1), s.chain_prefixes.slabs.items.len);
    for (ps) |p| p.deinit(a);
    try t.expectEqual(@as(usize, 0), s.chain_prefixes.live_bytes);
}

test "GLM compact: serial capture preserves prior data across ENOSPC, short writes and restart" {
    const f = spill.store_test;
    const h = try f.H.init(a, .{ .tag = "compact-faults", .shared_small_restore = true });
    defer h.deinit();
    const ids = try f.idsOf(a, 4 * f.block_tokens + 3, 2039);
    defer a.free(ids);
    h.hk.short_intr = true;
    const key = try f.write(h, ids, .{ .lineage = f.lin(4), .source = 4 });
    try t.expectEqual(@as(u64, 0), h.s.counts().read_block_calls);
    try t.expectEqual(@as(u64, 0), h.s.counts().read_small_calls);
    h.hk.short_intr = false;
    h.hk.nospace_writes = 1;
    const other = try f.idsOf(a, 3 * f.block_tokens, 2040);
    defer a.free(other);
    try t.expect(f.keepIds(h, other, .{ .lineage = f.lin(5), .source = 5 }) == .queued);
    try t.expect(f.settle(h).done);
    try t.expect(!h.s.presence(spill.format.pointKey(other, f.block_tokens, &.{})));
    try h.reopen();
    const out = (try f.restore(h, key)) orelse return error.NoRestore;
    defer a.free(out);
    try t.expectEqualSlices(u8, h.src(4)[0..out.len], out);
    try t.expectEqual(@as(u64, 1), h.s.counts().loaded);
    try t.expectEqual(@as(u64, 1), h.s.counts().read_small_calls);
    try t.expectEqual(@as(u64, h.s.points.get(key).?.blocks.len), h.s.counts().read_block_calls);
    try t.expectEqual(@as(u64, 0), h.s.counts().staging_bytes);
}

test "GLM compact: cancellation retains the shared arena until capture completion and never exposes stale bytes" {
    const f = spill.store_test;
    const h = try f.H.init(a, .{ .tag = "compact-cancel", .shared_small_restore = true });
    defer h.deinit();
    const ids = try f.idsOf(a, 2 * f.block_tokens, 2041);
    defer a.free(ids);
    const key = try f.write(h, ids, .{ .lineage = f.lin(6), .source = 6 });
    const more = try f.idsOf(a, 3 * f.block_tokens, 2042);
    defer a.free(more);
    const queued = f.keepIds(h, more, .{ .lineage = f.lin(7), .source = 7, .capture = false });
    try t.expect(queued == .queued);
    try t.expectError(error.NoSlot, h.s.beginRead(key));
    try t.expect(h.s.cancel(queued.queued.job));
    try t.expectError(error.NoSlot, h.s.beginRead(key));
    h.s.captured(queued.queued.job, false);
    try t.expect(f.settle(h).done);
    const out = (try f.restore(h, key)) orelse return error.NoRestore;
    defer a.free(out);
    try t.expectEqualSlices(u8, h.src(6)[0..out.len], out);
}

test "GLM compact: serial staging survives every publication crash and isolates damaged checkpoints" {
    const fault = spill.store_test;
    try fault.checkCrashesMode(a, true);
    try fault.checkRecoveryMode(a, true);
}
