//! Actual GLM snapshot ownership with compact committed recurrent banks.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;

test "GLM live state: strict cached mode, rank agreement and disk identity separate layouts" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_KV_LIVE_HALF")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.kv_live_half);
    Env.value = "1";
    const on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(on.kv_live_half and !(try glm.ranks.Settings.read(Env.get, &problem)).kv_live_half);
    const FP = glm.ranks.Fingerprint;
    const a = FP.of(4, 0, off, 196608, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 196608, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 86), a.layout);
    try t.expectEqual(FP.Field.kv_live_half, glm.ranks.mismatch(&.{ a, b }).?.field);
    const mode = glm.snapshots.mode;
    const base: [32]u8 = @splat(0x71);
    try t.expectEqual(base, mode.compatibility(base, false));
    try t.expect(!std.mem.eql(u8, &base, &mode.compatibility(base, true)));
    try t.expectEqual(mode.compatibility(base, true), mode.compatibility(base, true));
    try t.expect(!std.mem.eql(u8, &mode.compatibilityFor(base, true, false), &mode.compatibilityFor(base, true, true)));
    var fake: struct { settings: glm.ranks.Settings = .{}, mtp: bool = true, drafter: ?*u8 = null } = .{};
    var marker: u8 = 0;
    fake.drafter = &marker;
    try t.expect(mode.head(&fake));
    fake.settings.kv_live_half = true;
    try t.expect(!mode.head(&fake));
    try t.expectEqual(glm.disk_image.Words.Layout.live_dflash, mode.layoutFor(&fake));
    fake.drafter = null;
    try t.expect(mode.head(&fake));
    for ([_][]const u8{ "yes", "true", "2", "-1" }) |value| {
        Env.value = value;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_KV_LIVE_HALF: 0 or 1", problem);
    }
}

test "GLM live state: actual keep restore pin retire and saved parity preserve every live read-set byte" {
    var d = rec.init();
    for ([_]usize{ 2, 4 }) |world| for ([_]bool{ false, true }) |kv8| {
        var f: Fixture = undefined;
        try f.initFormat(&d, 0, world, true, kv8);
        defer f.deinit();
        const full = f.e.stateBytes(8192);
        f.e.settings.kv_live_half = true;
        const compact = f.e.stateBytes(8192);
        const bank = f.seq.st.kda_layers * f.seq.st.rec_layer;
        try t.expect(compact <= full and full - compact <= bank + glm.snapshots.pages.page);
        const old_plan = try glm.snapshots.layout.Plan.sizeWith(&f.seq.st, &f.e.c, f.e.capacity, 8192, 8192 + glm.forward.window_rows, 0, false);
        const new_plan = try glm.snapshots.layout.Plan.sizeWith(&f.seq.st, &f.e.c, f.e.capacity, 8192, 8192 + glm.forward.window_rows, 0, true);
        try t.expectEqual(bank, old_plan - new_plan);
        try t.expectEqual(full, try glm.snapshots.pages.rounded(old_plan));
        try t.expectEqual(compact, try glm.snapshots.pages.rounded(new_plan));
        try f.e.prepareSnapshots(4 * compact);
        const allocations = rec.allocs;
        for ([_]u1{ 0, 1 }) |parity| {
            try f.mark(8192, 8199, 37 + @as(u8, parity));
            f.seq.st.cur[0] = parity;
            const expected = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
            defer t.allocator.free(expected);
            const id = try f.e.keepState();
            try t.expectEqual(parity, f.e.snaps.get(id).?.cur[0]);
            @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xe9);
            try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{1 - parity}, .{ 0, 0 });
            try f.e.restoreState(id);
            try t.expectEqual(parity, f.seq.st.cur[0]);
            try f.expectPrefix(expected, 8192, 8199);
            for (rec.bytes(f.seq.st.recAt(1 - parity, 0), f.seq.st.rec_layer)) |byte|
                try t.expectEqual(@as(u8, 0xe9), byte);
            try glm.snapshots.pin(&f.e, id);
            f.e.dropState(id);
            try t.expect(f.e.snaps.get(id).?.retired);
            try t.expect(glm.snapshots.retiredBytes(&f.e) > 0);
            glm.snapshots.unpin(&f.e, id);
            try t.expect(!f.e.snaps.contains(id));
            try t.expectEqual(@as(u64, 0), glm.snapshots.retiredBytes(&f.e));
            f.e.dropped.clearRetainingCapacity();
        }
        try t.expectEqual(allocations, rec.allocs);
    };
}

test {
    _ = glm.disk_image;
}

test "GLM live state: a nonempty MTP cache is never discarded under an external drafter" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 4, true);
    defer f.deinit();
    f.e.settings.kv_live_half = true;
    try f.e.prepareSnapshots(2 * f.e.stateBytes(8192));
    try f.mark(2048, 7, 19);
    var dr: glm.dflash.Drafter = undefined;
    f.e.drafter = &dr;
    defer f.e.drafter = null;
    const before = rec.copies;
    const available = f.e.snap_pages.?.spare();
    try t.expectError(error.NotAtMark, f.e.keepState());
    try t.expectEqual(before, rec.copies);
    try t.expectEqual(available, f.e.snap_pages.?.spare());
    try t.expectEqual(@as(u32, 0), f.e.snaps.count());
}

const Sizing = struct {
    var next: u64 = 1 << 32;
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(usize, bytes, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "GLM live state: production TP2 TP4 image and pinned-slot planning shrink without touching live KV" {
    const nvme = @import("core").nvme1107;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    for ([_]usize{ 2, 4 }) |world| {
        var w: glm.weights.Weights = undefined;
        w.config = .{};
        w.share = try glm.shard.Share.of(&w.config, 0, world);
        var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
        w.layers = layers[0..w.config.layers];
        for (w.layers, 0..) |*layer, i| layer.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
        var st = try glm.forward.State.initFormat(t.allocator, &d, &w, glm.native.capacity(196608), true);
        defer st.deinit();
        const live_arena = st.arena.buf.len;
        const bank = try glm.snapshots.layout.recurrentBytes(&st);
        try t.expectEqual(@as(usize, if (world == 4) 35651584 else 71303168), bank);
        var old = try glm.disk_image.Shape.of(&st, &w.config, st.capacity, true);
        var compact = try glm.disk_image.Shape.ofMode(&st, &w.config, st.capacity, false, true);
        old.draft_ring = true;
        compact.draft_ring = true;
        try t.expectEqual(bank, old.prefix - compact.prefix);
        for ([_]u32{ 8192, 137216, 192512 }) |at| {
            var sizes: [2][3]u64 = undefined;
            for ([_]glm.disk_image.Shape{ old, compact }, 0..) |shape, j| {
                var parts: std.ArrayList(glm.disk_image.Part) = .empty;
                defer parts.deinit(t.allocator);
                try glm.disk_image.nvmeParts(shape, at, &parts, t.allocator);
                var mapped: std.ArrayList(nvme.layout.Part) = .empty;
                defer mapped.deinit(t.allocator);
                for (parts.items) |part| try mapped.append(t.allocator, .{ .bytes = part.bytes, .row = part.row, .per = part.per });
                const layout = try nvme.layout.Layout.init(mapped.items, at, 256);
                const plan = try nvme.spill.slots.serialPlan(1 << 30, layout.blockBytes(), layout.smallBytes() + (128 << 10));
                sizes[j] = .{ glm.disk_image.imageBytes(shape, at), layout.blockBytes(), plan.bytes() };
            }
            try t.expect(sizes[1][0] < sizes[0][0]);
            try t.expect(sizes[1][1] < sizes[0][1]);
            try t.expect(sizes[1][2] < sizes[0][2]);
            std.debug.print("LIVE_STATE_PLAN world={d} at={d} recurrent_saved={d} image_before={d} image_after={d} block_before={d} block_after={d} pinned_before={d} pinned_after={d}\n", .{ world, at, bank, sizes[0][0], sizes[1][0], sizes[0][1], sizes[1][1], sizes[0][2], sizes[1][2] });
        }
        try t.expectEqual(live_arena, st.arena.buf.len);
    }
}
