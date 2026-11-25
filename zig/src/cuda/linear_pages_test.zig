const std = @import("std");
const pages = @import("linear_pages.zig");
const t = std.testing;

const Recording = struct {
    const bytes = 16;
    const physical = 8;

    data: [physical][bytes]u8 = @splat(@splat(0)),
    allocated: [physical]bool = @splat(false),
    map_ids: [32]?u64 = @splat(null),
    next: usize = bytes,
    creates: usize = 0,
    maps: usize = 0,
    copies: usize = 0,
    fail_map_after: ?usize = null,

    fn backend(r: *Recording) pages.Backend {
        return .{
            .ptr = r,
            .granularity = bytes,
            .reserve = reserve,
            .create = create,
            .release = release,
            .map = map,
            .unmap = unmap,
            .copy = copy,
        };
    }

    fn of(ctx: *anyopaque) *Recording {
        return @ptrCast(@alignCast(ctx));
    }

    fn reserve(ctx: *anyopaque, n: usize) !u64 {
        const r = of(ctx);
        const base = r.next;
        if (n > r.map_ids.len * bytes - base) return error.NoAddress;
        r.next += n;
        return base;
    }

    fn create(ctx: *anyopaque, n: usize) !u64 {
        const r = of(ctx);
        try t.expectEqual(bytes, n);
        for (&r.allocated, 0..) |*used, i| {
            if (used.*) continue;
            used.* = true;
            r.creates += 1;
            return i;
        }
        return error.OutOfBacking;
    }

    fn release(ctx: *anyopaque, h: u64, n: usize) !void {
        try t.expectEqual(bytes, n);
        const r = of(ctx);
        try t.expect(r.allocated[h]);
        try t.expect(std.mem.indexOfScalar(?u64, &r.map_ids, h) == null);
        r.allocated[h] = false;
    }

    fn map(ctx: *anyopaque, address: u64, n: usize, h: u64) !void {
        const r = of(ctx);
        if (r.fail_map_after) |count| {
            r.fail_map_after = if (count == 0) null else count - 1;
            if (count == 0) return error.InjectedMapFailure;
        }
        try t.expectEqual(bytes, n);
        try t.expect(r.allocated[h] and r.map_ids[address / bytes] == null);
        r.map_ids[address / bytes] = h;
        r.maps += 1;
    }

    fn unmap(ctx: *anyopaque, address: u64, n: usize) !void {
        const r = of(ctx);
        try t.expectEqual(bytes, n);
        try t.expect(r.map_ids[address / bytes] != null);
        r.map_ids[address / bytes] = null;
    }

    fn copy(ctx: *anyopaque, dst: u64, src: u64, n: usize) !void {
        const r = of(ctx);
        try t.expectEqual(bytes, n);
        r.data[r.map_ids[dst / bytes].?] = r.data[r.map_ids[src / bytes].?];
        r.copies += 1;
    }

    fn byte(r: *Recording, address: u64) *u8 {
        return &r.data[r.map_ids[address / bytes].?][address % bytes];
    }
};

test "linear cache: snapshots retain physical runs and a branch copies only its shared tail" {
    var sim: Recording = .{};
    var pool = try pages.Pool.init(t.allocator, sim.backend(), 16, 6);
    var a = try pages.Window.init(&pool, 51);
    var b = try pages.Window.init(&pool, 51);
    const base = b.base;

    try a.prepareWrite(0, 29);
    for (0..29) |i| sim.byte(a.base + i).* = @intCast(i + 2);
    var mark = try a.snapshot(29);
    try b.restore(&mark);
    try t.expectEqual(@as(usize, 2), sim.creates);
    try t.expectEqual(@as(usize, 0), sim.copies);
    try t.expectEqual(sim.map_ids[a.base / 16], sim.map_ids[b.base / 16]);

    try b.prepareWrite(29, 30);
    sim.byte(b.base + 29).* = 251;
    try t.expectEqual(base, b.base);
    for (0..29) |i| try t.expectEqual(sim.byte(a.base + i).*, sim.byte(b.base + i).*);
    try t.expectEqual(@as(u8, 0), sim.byte(a.base + 29).*);
    try t.expectEqual(@as(usize, 1), sim.copies);
    const maps = sim.maps;
    try b.prepareWrite(30, 32);
    try t.expectEqual(maps, sim.maps);

    try t.expectError(error.VirtualWindowLimit, a.ensure(52));
    try t.expectError(error.LiveBacking, pool.deinit());
    try a.deinit();
    try b.deinit();
    try t.expectError(error.LiveBacking, pool.deinit());
    mark.deinit();
    try pool.deinit();
    const free: [Recording.physical]bool = @splat(false);
    try t.expectEqual(free, sim.allocated);
}

test "linear cache: failed branch remap and partial restore can be retried" {
    var sim: Recording = .{};
    var pool = try pages.Pool.init(t.allocator, sim.backend(), 16, 6);
    var a = try pages.Window.init(&pool, 64);
    var b = try pages.Window.init(&pool, 64);
    try a.prepareWrite(0, 31);
    sim.byte(a.base + 27).* = 11;
    var mark = try a.snapshot(31);

    sim.fail_map_after = 1;
    try t.expectError(error.InjectedMapFailure, b.restore(&mark));
    try t.expectEqual(@as(usize, 0), b.mapped);
    try b.restore(&mark);
    sim.fail_map_after = 1;
    try t.expectError(error.InjectedMapFailure, b.prepareWrite(30, 31));
    try t.expectEqual(@as(u8, 11), sim.byte(b.base + 27).*);
    try b.prepareWrite(30, 31);
    try t.expectEqual(@as(u8, 11), sim.byte(b.base + 27).*);

    mark.deinit();
    try a.deinit();
    try b.deinit();
    try pool.deinit();
}

test "linear cache: physical capacity is shared and recycled after releasing snapshots" {
    var sim: Recording = .{};
    var pool = try pages.Pool.init(t.allocator, sim.backend(), 16, 2);
    var a = try pages.Window.init(&pool, 64);
    var b = try pages.Window.init(&pool, 64);
    try a.prepareWrite(0, 32);
    var mark = try a.snapshot(32);
    try a.clear();
    try t.expectError(error.PhysicalPageLimit, b.ensure(1));
    mark.deinit();
    try b.ensure(32);
    try t.expectEqual(@as(usize, 2), sim.creates);
    try a.deinit();
    try b.deinit();
    try pool.deinit();
}

test "linear cache: a live fork borrows without copies, then retains bytes across parent release" {
    var sim: Recording = .{};
    var pool = try pages.Pool.init(t.allocator, sim.backend(), 16, 6);
    var parent = try pages.Window.init(&pool, 64);
    var child = try pages.Window.init(&pool, 64);
    try t.expectEqual(@as(usize, 0), sim.creates);

    try parent.prepareWrite(0, 23);
    sim.byte(parent.base + 20).* = 77;
    sim.fail_map_after = 1;
    try t.expectError(error.InjectedMapFailure, child.borrow(&parent, 23));
    try t.expectEqual(@as(usize, 0), child.mapped);
    try child.borrow(&parent, 23);
    try t.expectEqual(@as(usize, 2), sim.creates);
    try t.expectEqual(@as(usize, 0), sim.copies);
    try parent.prepareWrite(23, 30);
    sim.byte(parent.base + 23).* = 99;
    try t.expectEqual(@as(u8, 77), sim.byte(child.base + 20).*);
    try t.expectEqual(@as(u8, 0), sim.byte(child.base + 23).*);
    try t.expectEqual(@as(usize, 1), sim.copies);
    try parent.clear();
    try child.prepareWrite(23, 30);
    try t.expectEqual(@as(usize, 1), sim.copies);
    try parent.deinit();
    try child.deinit();
    try pool.deinit();
}

test "linear cache: sealed writer blocks survive lane remap without a staging mapping" {
    var sim: Recording = .{};
    var pool = try pages.Pool.init(t.allocator, sim.backend(), 16, 4);
    var a = try pages.Window.init(&pool, 48);
    try a.prepareWrite(0, 29);
    sim.byte(a.base + 7).* = 77;
    sim.byte(a.base + 20).* = 22;

    const maps = sim.maps;
    try t.expectError(error.IncompleteBlock, a.pinCompleteBlock(1, 29));
    var sealed = try a.pinCompleteBlock(0, 29);
    var prefix = try a.snapshot(29);
    const first = prefix.block(0).?;
    const last = prefix.block(1).?;
    try t.expectEqual(sealed.block(), first);
    try t.expectEqual(@as(usize, 13), last.len);
    try t.expectEqual(@as(u8, 22), sim.byte(last.ptr + 4).*);
    try t.expect(prefix.block(2) == null);
    try t.expectEqual(maps, sim.maps);

    try a.clear();
    try a.prepareWrite(0, 32);
    sim.byte(a.base + 7).* = 11;
    try t.expectEqual(@as(u8, 77), sim.byte(sealed.block().ptr + 7).*);
    prefix.deinit();
    try t.expectEqual(@as(u8, 77), sim.byte(sealed.block().ptr + 7).*);
    sealed.deinit();
    try a.deinit();
    try pool.deinit();
}
