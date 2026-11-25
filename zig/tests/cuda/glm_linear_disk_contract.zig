//! Real mapped GLM state and the unchanged native NVMe writer/reader on one device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Engine = glm.engine.Engine;
const Seq = @typeInfo(@FieldType(Engine, "cur")).pointer.child;
const Owner = @typeInfo(@typeInfo(@FieldType(Engine, "linear")).optional.child).pointer.child;
const at: u32 = 8192;
const capacity = 3 * glm.engine.prompt_rows * 2 + glm.forward.window_rows;

pub fn run(gpu: check.Gpu, root: []const u8) !void {
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .kda = undefined };
    layers[1].mixer = .{ .mla = undefined };
    const c: glm.config.Config = .{};
    var w: glm.weights.Weights = undefined;
    w.config = c;
    w.share = try glm.shard.Share.of(&w.config, 0, 4);
    w.layers = &layers;
    const owner = try Owner.init(gpu.gpa, gpu.ctx, s, capacity, 1);
    defer owner.deinit();
    var first: Seq = .{ .st = try glm.forward.State.initLinear(gpu.gpa, gpu.d, &w, capacity, owner) };
    defer first.st.deinit();
    var second: Seq = .{ .st = try glm.forward.State.initLinear(gpu.gpa, gpu.d, &w, capacity, owner) };
    defer second.st.deinit();
    defer cuda.completion.drain(gpu.d, s, "GLM NVMe device fixture teardown");

    var e: Engine = undefined;
    e.gpa = gpu.gpa;
    e.io = gpu.io;
    e.d = gpu.d;
    e.stream = s;
    e.c = c;
    e.share = w.share;
    e.settings = .{ .kv8 = true, .kv_live_half = true, .linear_cache = true, .dflash_only_head = true };
    e.capacity = capacity;
    e.cur = &first;
    e.mtp = false;
    e.drafter = null;
    e.share_pages = false;
    e.verify_graphs = null;
    e.announce = 0;
    e.snaps = .empty;
    e.dropped = .empty;
    e.snap_pages = null;
    try glm.snapshots.prepare(&e, 2 * glm.snapshots.bytes(&e, at));
    defer {
        var it = e.snaps.valueIterator();
        while (it.next()) |snap| glm.snapshots.free(&e, snap);
        e.snaps.deinit(gpu.gpa);
        e.dropped.deinit(gpu.gpa);
        e.snap_pages.?.deinit();
    }
    const parities = [_]u1{1};
    const ring_base = at - first.st.ring_rows;
    try first.st.restoreHost(s, at, 0, &parities, .{ ring_base, 0 });
    const reference = try glm.snapshots.layout.Plan.initWith(&first.st, &c, at, null, 0, &parities, true);
    for (reference.parts[0..reference.count], 0..) |part, i| {
        try gpu.d.check(gpu.d.api.cuMemsetD8Async(part.live, @intCast(11 + i), part.len, s.handle), "cuMemsetD8Async");
    }
    try first.st.setPos(s, at);
    try first.st.setMtpLen(s, 0);
    const first_sha = try digest(gpu, s, &reference);
    try glm.snapshots.keepAs(&e, 1);
    // Vary saved latent values, leaving host-managed position words alone.
    try gpu.d.check(gpu.d.api.cuMemsetD8Async(first.st.caches[0].lc, 90, 512, s.handle), "cuMemsetD8Async");
    const before = try digest(gpu, s, &reference);
    try check.expect(!std.mem.eql(u8, &first_sha, &before), "the two retained fixtures have different native image bytes", .{});
    try glm.snapshots.keepAs(&e, 2);
    const ids = try gpu.gpa.alloc(u32, at + 1);
    defer gpu.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(1000 + i % 1453);
    const next_ids = try gpu.gpa.dupe(u32, ids);
    defer gpu.gpa.free(next_ids);
    next_ids[0] += 1;
    const cfg: glm.nvme_state.Config = .{
        .cap = 1 << 30,
        .floor = 0,
        .staging = 192 << 20,
        .index = 64 << 20,
        .serial_mode = true,
        .lookahead = 1,
    };
    const metadata = "linear-device-roundtrip";
    {
        const state = try glm.nvme_state.State.open(&e, root, @splat(0x5a), cfg);
        defer state.close();
        try state.side.enqueue(.{ .id = 1, .at = at, .tokens = ids, .metadata = metadata, .lineage = .{ .root = @splat(9) } });
        try state.side.enqueue(.{ .id = 2, .at = at, .tokens = next_ids, .metadata = metadata, .lineage = .{ .root = @splat(10) } });
        glm.snapshots.drop(&e, 1);
        glm.snapshots.drop(&e, 2);
        const end = std.Io.Clock.awake.now(gpu.io).toNanoseconds() + 15 * std.time.ns_per_s;
        while (!state.side.completed(1) or !state.side.completed(2)) {
            state.side.pump(std.time.ns_per_ms);
            try pause(gpu.io, end);
        }
        try state.side.release(1);
        try state.side.release(2);
        try check.expect(e.snaps.count() == 0 and state.side.store.counts().committed == 2, "both serial disk writes committed and released retired sources", .{});
    }

    e.cur = &second;
    const reopened = try glm.nvme_state.State.open(&e, root, @splat(0x5a), cfg);
    defer reopened.close();
    const format = @import("core").nvme1107.spill.format;
    const key = format.pointKey(next_ids, glm.nvme_state.block_tokens, &.{});
    try reopened.side.beginRead(key, 2, at, metadata);
    const end = std.Io.Clock.awake.now(gpu.io).toNanoseconds() + 15 * std.time.ns_per_s;
    while (true) {
        switch (reopened.side.pollRead()) {
            .ready => break,
            .failed => return reopened.side.read.?.failed.?,
            .pending => try pause(gpu.io, end),
        }
    }
    try reopened.side.commitRead();
    try glm.snapshots.restore(&e, 2);
    const got = try glm.snapshots.layout.Plan.initWith(&second.st, &c, at, null, 0, &parities, true);
    try check.sameBytes("mapped NVMe reopen image SHA", &(try digest(gpu, s, &got)), &before);
    try check.expect(second.st.pos == at and second.st.base == ring_base and second.st.cur[0] == 1, "mapped NVMe reopen restored host position and KDA parity", .{});
    check.pass("GLM_LINEAR_NVME: both queued checkpoints committed, and the second reopened with {d} exact bytes", .{got.len});
}

fn pause(io: std.Io, end: i128) !void {
    if (std.Io.Clock.awake.now(io).toNanoseconds() > end) return error.Timeout;
    try std.Io.sleep(io, .fromMilliseconds(1), .awake);
}

fn digest(gpu: check.Gpu, s: cuda.Stream, plan: *const glm.snapshots.layout.Plan) ![32]u8 {
    try s.synchronize();
    const buffer = try gpu.gpa.alloc(u8, 64 << 10);
    defer gpu.gpa.free(buffer);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (plan.parts[0..plan.count]) |p| {
        var off: usize = 0;
        while (off < p.len) {
            const n = @min(p.len - off, buffer.len);
            try cuda.memory.download(gpu.d, p.live + off, buffer[0..n]);
            hash.update(buffer[0..n]);
            off += n;
        }
    }
    return hash.finalResult();
}
