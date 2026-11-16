//! Real GLM snapshot images through the 1107 disk workers, using the recording CUDA device.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const nvme = @import("core").nvme1107;
const spill = nvme.spill;
const t = std.testing;
const a = t.allocator;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;

const HostCopy = struct {
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn take(out: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(out.?))[0..n], rec.bytes(src, n));
        return 0;
    }
    fn put(dst: u64, in: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(rec.bytes(dst, n), @as([*]const u8, @ptrCast(in.?))[0..n]);
        return 0;
    }
    fn putAsync(dst: u64, in: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return put(dst, in, n);
    }
    fn attach(d: *cuda.Driver) void {
        d.api.cuCtxGetDevice = device;
        d.api.cuDeviceGetAttribute = attribute;
        d.api.cuMemcpyDtoH_v2 = take;
        d.api.cuMemcpyHtoD_v2 = put;
        d.api.cuMemcpyHtoDAsync_v2 = putAsync;
    }
};

const Packet = struct {
    var header: glm.follow.Header = undefined;
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, argv: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const args = argv.?;
        const dst = @as(*const u64, @ptrCast(@alignCast(args[0].?))).*;
        const words = @as(*const u32, @ptrCast(@alignCast(args[1].?))).*;
        const src: [*]const u8 = @ptrCast(args[2].?);
        @memcpy(rec.bytes(dst, words * 4), src[0 .. words * 4]);
        return 0;
    }
    fn broadcast(src: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        if (n >= @sizeOf(glm.follow.Header)) header = std.mem.bytesToValue(glm.follow.Header, rec.bytes(src, @sizeOf(glm.follow.Header)));
        return 0;
    }
};

test "GLM NVMe: a disk command preserves its source credit when the follow link binds stream seven" {
    var d = rec.init();
    HostCopy.attach(&d);
    d.api.cuLaunchKernel = Packet.launch;
    var f: Fixture = undefined;
    try f.init(&d, 0, 4, true);
    defer f.deinit();
    f.e.slot = 7;
    f.e.m.glue.put_words = .{ .d = &d, .handle = null };
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclBroadcast = Packet.broadcast;
    var link = try glm.follow.Link.init(a, &d, &lib, null, f.e.stream, 32);
    defer link.deinit();
    link.wait = false;
    const leader: glm.follow.Leader = .{ .e = &f.e, .link = &link };
    try leader.disk(.{ .kind = 0, .flags = @backingInt(glm.nvme_wire.Op.note_metadata), .a = 42, .top_k = 3 }, &.{ 1, 2, 3 });
    try t.expectEqual(@as(u32, 7), Packet.header.slot);
    try t.expectEqual(@as(u32, 1), Packet.header.top_k >> 1);
    try t.expectEqual(@as(u32, 1), Packet.header.top_k & 1);
    try t.expectEqual(@as(u64, 42), Packet.header.a);
    try t.expectEqual(@backingInt(glm.nvme_wire.Op.note_metadata), Packet.header.flags);
}

fn waitWrite(side: *nvme.tier_state.Side) !void {
    const end = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (side.pending()) {
        side.pump(std.time.ns_per_ms);
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > end) return error.WriterTimedOut;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn waitRead(side: *nvme.tier_state.Side) !void {
    const end = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (true) {
        switch (side.pollRead()) {
            .ready => return,
            .failed => return side.read.?.failed.?,
            .pending => {},
        }
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > end) return error.ReaderTimedOut;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

test "GLM NVMe: unequal TP shares and an external drafter restore all state bytes after eviction and restart" {
    for (0..16) |which| {
        const live = which >= 8;
        const serial = which % 8 >= 4;
        const case = which % 4;
        const drafted = case == 3;
        const rank = if (drafted) 0 else case;
        var d = rec.init();
        HostCopy.attach(&d);
        var f: Fixture = undefined;
        try f.init(&d, rank, if (drafted) 4 else 3, true);
        f.e.settings.kv_live_half = live;
        defer f.deinit();
        const at: u32 = 2048;
        const head = if (drafted) 0 else at + 7;
        var dr: glm.dflash.Drafter = undefined;
        dr.gpa = a;
        dr.d = &d;
        dr.slots = .empty;
        defer {
            for (dr.slots.items) |*slot| if (slot.ring) |*ring| ring.free();
            dr.slots.deinit(a);
        }
        var ring_want: []const u8 = &.{};
        defer if (ring_want.len > 0) a.free(ring_want);
        if (drafted) {
            f.e.drafter = &dr;
            const ptr = try dr.ringPtr(0);
            for (rec.bytes(ptr, glm.dflash.ring_bytes), 0..) |*b, i| b.* = @truncate(i *% 23 +% i / 83);
            ring_want = try a.dupe(u8, rec.bytes(ptr, glm.dflash.ring_bytes));
            dr.slots.items[0].first = 0;
            dr.slots.items[0].end = at;
        }
        try f.e.prepareSnapshots(3 * f.e.stateBytes(at));
        try f.mark(at, head, 37);
        const parity: u1 = @truncate(which / 4);
        f.seq.st.cur[0] = parity;
        const want = try a.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer a.free(want);
        const source = try f.e.keepState();
        const image = glm.nvme_image.of(&f.e);
        var ps: std.ArrayList(nvme.layout.Part) = .empty;
        defer ps.deinit(a);
        try image.parts(image.ptr, at, &ps, a);
        const l = try nvme.layout.Layout.init(ps.items, at, 256);
        var root_buf: [256]u8 = undefined;
        const root = try spill.fsio.testRoot(&root_buf, "glm-nvme-image");
        const fd = try spill.fsio.freshDir(a, root);
        _ = std.c.close(fd);
        defer spill.fsio.dropDir(a, root);
        const block_slot = std.mem.alignForward(usize, @intCast(l.blockBytes()), 4096);
        const small_max = std.mem.alignForward(usize, @intCast(l.smallBytes() + 1280), 4096);
        const w = try a.alignedAlloc(u8, .fromByteUnits(4096), 4 * block_slot);
        defer a.free(w);
        const r = try a.alignedAlloc(u8, .fromByteUnits(4096), 4 * block_slot);
        defer a.free(r);
        const small = try a.alignedAlloc(u8, .fromByteUnits(4096), (if (serial) @as(usize, 1) else 2) * small_max);
        defer a.free(small);
        const restore = if (serial) small else try a.alignedAlloc(u8, .fromByteUnits(4096), small_max);
        defer if (!serial) a.free(restore);
        const cfg: spill.Config = .{ .root = root, .rank = @intCast(rank), .compat = glm.snapshots.mode.compatibilityFor(@splat(0x39), live, drafted), .limits = .{ .max_tokens = @intCast(f.e.capacity), .block_tokens = 256, .block_bytes = l.blockBytes(), .max_small_bytes = small_max }, .cap_bytes = 1 << 30, .floor_bytes = 0, .min_tokens = 256, .staging = .{ .write = &.{w}, .read = &.{r}, .small = &.{small}, .restore = &.{restore}, .shared_small_restore = serial }, .direct = false };
        var store = try spill.Store.open(a, t.io, cfg);
        var opened = true;
        defer if (opened) store.close();
        var side = nvme.tier_state.Side.init(a, t.io, store, image, 2, 1024);
        side.serial_read_wait_ns = if (serial) 3 * std.time.ns_per_s else 0;
        side.lookahead = 1;
        defer side.deinit();
        const ids = try spill.format.testIds(a, at + 1, 72);
        defer a.free(ids);
        try side.enqueue(.{ .id = source, .at = at, .tokens = ids, .metadata = "original-prefill-layout", .lineage = .{ .root = @splat(0x71) } });
        f.e.dropState(source);
        try t.expect(f.e.snaps.contains(source));
        try t.expectEqual(@as(u64, f.e.snaps.get(source).?.lease.ids.items.len * glm.snapshots.pages.page), glm.snapshots.retiredBytes(&f.e));
        try waitWrite(&side);
        const written = side.store.counts();
        try t.expect(written.staged_blocks > 0 and written.written_blocks > 0 and written.committed > 0);
        try t.expectEqual(
            written.staged_blocks +% written.written_blocks +% written.committed,
            side.store.writeProgress(),
        );
        try t.expect(!f.e.snaps.contains(source));
        try t.expectEqual(@as(u64, 0), glm.snapshots.retiredBytes(&f.e));
        store.close();
        opened = false;
        store = try spill.Store.open(a, t.io, cfg);
        opened = true;
        side.store = store;
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xea);
        try f.seq.st.restoreHost(f.e.stream, 17, 18, &.{1 - parity}, .{ 0, 0 });
        if (drafted) {
            @memset(rec.bytes(try dr.ringPtr(0), glm.dflash.ring_bytes), 0x44);
            f.e.slot = 1; // the next request resumes on another stream, not its original ring
            @memset(rec.bytes(try dr.ringPtr(1), glm.dflash.ring_bytes), 0xce);
            try dr.restart(1, 9999);
        }
        try side.beginRead(spill.format.pointKey(ids, 256, &.{}), 42, at, "original-prefill-layout");
        try waitRead(&side);
        try side.commitRead();
        try f.e.restoreState(42);
        try f.expectPrefix(want, at, head);
        if (drafted) {
            try t.expectEqual(@as(i64, 0), dr.slots.items[1].from);
            try t.expectEqual(@as(i64, at), dr.slots.items[1].end);
            try t.expectEqualSlices(u8, ring_want, rec.bytes(try dr.ringPtr(1), glm.dflash.ring_bytes));
            try t.expectEqual(@as(u8, 0x44), rec.bytes(try dr.ringPtr(0), 1)[0]);
        }
        try t.expectEqual(head, f.seq.st.mtp_len);
        f.e.dropState(42);
        f.e.dropped.clearRetainingCapacity();
    }
}

test "GLM NVMe: the private image remains bounded at a million rows" {
    const sh0: glm.disk_image.Shape = .{ .prefix = 12 << 20, .tail = 4096, .layers = 12, .latent = 512, .scales = true, .ring_rows = 8192, .ring_row = 64, .pk_rows = 1048608 / 4 + 2, .pooled_row = 128, .capacity = 1048608, .head = true };
    var small_max: u64 = 0;
    for ([_]u32{ 32768, 65536, 524288, 1048576 }) |at| {
        var ps: std.ArrayList(glm.disk_image.Part) = .empty;
        defer ps.deinit(a);
        try glm.disk_image.nvmeParts(sh0, at, &ps, a);
        var mapped: std.ArrayList(nvme.layout.Part) = .empty;
        defer mapped.deinit(a);
        for (ps.items) |p| try mapped.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
        const l = try nvme.layout.Layout.init(mapped.items, at, 256);
        if (small_max == 0) small_max = l.smallBytes();
        try t.expectEqual(small_max, l.smallBytes());
        try t.expectEqual(glm.disk_image.imageBytes(sh0, at), l.total());
        try t.expect(l.smallBytes() < 64 << 20);
    }
}

const Sizing = struct {
    var next: u64 = 1 << 30;
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

test "GLM NVMe: the real default TP4 million-row FP8 layout fits its fixed staging budget" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 4);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*layer, i| {
        layer.* = undefined;
        layer.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    var e: glm.engine.Engine = undefined;
    var seq: @TypeOf(e.cur.*) = undefined;
    seq.st = try glm.forward.State.initFormat(a, &d, &w, glm.native.capacity(1048576), true);
    defer seq.st.deinit();
    e.cur = &seq;
    e.c = w.config;
    e.capacity = seq.st.capacity;
    e.mtp = false;
    e.drafter = @ptrFromInt(4096); // shape's presence only; no drafter address is read
    e.share_pages = false;
    var ps: std.ArrayList(nvme.layout.Part) = .empty;
    defer ps.deinit(a);
    const img = glm.nvme_image.of(&e);
    try img.parts(&e, 1048576, &ps, a);
    const layout = try nvme.layout.Layout.init(ps.items, 1048576, 256);
    const plan = try spill.slots.plan(512 << 20, layout.blockBytes(), layout.smallBytes() + (128 << 10), 2, 1);
    const compact = try spill.slots.serialPlan(192 << 20, layout.blockBytes(), layout.smallBytes() + (128 << 10));
    std.debug.print("GLM NVMe compact TP4 FP8: old_staging={d} shared_staging={d}\n", .{ plan.bytes(), compact.bytes() });
    try t.expect(compact.bytes() <= 192 << 20 and compact.bytes() < plan.bytes() / 2);
    std.debug.print("GLM NVMe default TP4 FP8: window={d} image={d} small={d} block={d} staging={d}\n", .{ seq.st.arena.buf.len, layout.total(), layout.smallBytes(), layout.blockBytes(), plan.bytes() });
    try t.expect(plan.bytes() <= 512 << 20);
}
