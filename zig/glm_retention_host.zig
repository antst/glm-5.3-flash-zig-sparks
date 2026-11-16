//! Actual Engine/State ownership behind host-backed CUDA recording; no device library or model.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const t = std.testing;

const Fake = struct {
    const Kind = enum { sync, free };
    var order: [64]Kind = undefined;
    var n: usize = 0;
    var allocs: usize = 0;
    var frees: usize = 0;
    var pending = false;
    var fail_alloc = false;
    var old_alloc: *const fn (*u64, usize) callconv(.c) c_int = undefined;
    var old_free: *const fn (u64) callconv(.c) c_int = undefined;
    var old_sync: *const fn (cuda.abi.Stream) callconv(.c) c_int = undefined;
    fn driver() cuda.Driver {
        var d = rec.init();
        d.func_name = null;
        old_alloc = d.api.cuMemAlloc_v2;
        old_free = d.api.cuMemFree_v2;
        old_sync = d.api.cuStreamSynchronize;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemsetD8Async = zero;
        n = 0;
        allocs = 0;
        frees = 0;
        pending = false;
        fail_alloc = false;
        return d;
    }
    fn alloc(out: *u64, bytes: usize) callconv(.c) c_int {
        if (fail_alloc) return cuda.abi.error_not_ready;
        const rc = old_alloc(out, bytes);
        if (rc == 0) allocs += 1;
        return rc;
    }
    fn free(ptr: u64) callconv(.c) c_int {
        std.debug.assert(!pending);
        order[n] = .free;
        n += 1;
        frees += 1;
        return old_free(ptr);
    }
    fn sync(s: cuda.abi.Stream) callconv(.c) c_int {
        order[n] = .sync;
        n += 1;
        pending = false;
        return old_sync(s);
    }
    fn zero(ptr: u64, value: u8, bytes: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        @memset(rec.bytes(ptr, bytes), value);
        return 0;
    }
};

const Fixture = struct {
    e: glm.engine.Engine,
    layers: [2]glm.weights.Layer = undefined,
    fn init(f: *Fixture, d: *const cuda.Driver, on: bool) !void {
        f.e = undefined;
        f.layers[0].mixer = .{ .mla = undefined };
        f.layers[1].mixer = .{ .kda = undefined };
        f.e.gpa = t.allocator;
        f.e.d = d;
        f.e.stream = .{ .d = d, .handle = null };
        f.e.settings = .{ .retain_sequence_slots = on, .kv8 = true };
        f.e.sequence_counts = .{};
        f.e.verify_graphs = null;
        f.e.images = null;
        f.e.pages = null;
        f.e.drafter = null;
        f.e.seqs = .empty;
        f.e.capacity = 128;
        f.e.w = undefined;
        f.e.w.config = .{ .layers = 2, .kda_heads = 4, .kv_lora = 512, .i_dim = 8, .hidden = 64 };
        f.e.w.share = try glm.shard.Share.of(&f.e.w.config, 0, 1);
        f.e.share = f.e.w.share;
        f.e.w.layers = &f.layers;
        try f.e.bind(0);
    }
    fn deinit(f: *Fixture) void {
        f.e.freeSequences(); // same unconditional phase used by the real Engine.deinit
    }
};

test "sequence retention: strict cached default-off setting and all-rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_RETAIN_SEQUENCE_SLOTS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.retain_sequence_slots);
    Env.value = "0";
    try t.expect(std.meta.eql(off, try glm.ranks.Settings.read(Env.get, &problem)));
    Env.value = "1";
    const on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(on.retain_sequence_slots);
    const FP = glm.ranks.Fingerprint;
    const f0 = FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const f1 = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 59), f0.layout);
    try t.expectEqual(FP.Field.retain_sequence_slots, glm.ranks.mismatch(&.{ f0, f1 }).?.field);
    const agreed = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expect(glm.ranks.mismatch(&.{ agreed, f1 }) == null);
    for ([_][]const u8{ "", "2", "true", "-1", "1 ", "01" }) |bad| {
        Env.value = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_RETAIN_SEQUENCE_SLOTS: 0 or 1", problem);
    }
}

test "sequence retention: pages and target verify graphs refuse before allocations" {
    for ([_]glm.ranks.Settings{
        .{ .retain_sequence_slots = true, .pages = true },
        .{ .retain_sequence_slots = true, .verify_graphs = true },
    }) |settings| try t.expectError(error.RetainSequenceScope, glm.engine.Engine.init(t.failing_allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = settings, .capacity = 4096, .mtp = true }));
}

test "sequence retention: actual OFF release drains frees and reallocates while slot0 stays" {
    var d = Fake.driver();
    var f: Fixture = undefined;
    try f.init(&d, false);
    defer f.deinit();
    const zero = f.e.cur;
    try f.e.bind(1);
    try t.expectEqual(@as(usize, 2), Fake.allocs);
    Fake.n = 0;
    Fake.pending = true;
    f.e.release(1);
    try t.expectEqualSlices(Fake.Kind, &.{ .sync, .free }, Fake.order[0..Fake.n]);
    try t.expect(f.e.seqs.items[1] == null and f.e.cur == zero and f.e.slot == 0);
    try f.e.bind(1);
    try t.expectEqual(@as(usize, 3), Fake.allocs);
    const before = Fake.n;
    f.e.release(0);
    try t.expectEqual(before, Fake.n);
    try t.expect(f.e.seqs.items[0].? == zero);
    try t.expectEqual(@as(u64, 0), f.e.sequence_counts.created +
        f.e.sequence_counts.retained_releases + f.e.sequence_counts.destroyed);
}

test "sequence retention: actual ON release retains after drain and existing begin resets reused State" {
    var d = Fake.driver();
    var f: Fixture = undefined;
    try f.init(&d, true);
    defer f.deinit();
    try f.e.bind(1);
    const kept = f.e.cur;
    @memset(rec.bytes(kept.st.arena.buf.ptr, kept.st.arena.buf.len), 0xa5);
    @memset(kept.st.cur, 1);
    kept.st.pos = 37;
    kept.st.mtp_len = 41;
    kept.st.base = 32;
    kept.st.mtp_base = 32;
    kept.st.pending = 4;
    kept.in_head = 8;
    Fake.n = 0;
    Fake.pending = true;
    f.e.release(1);
    try t.expectEqualSlices(Fake.Kind, &.{.sync}, Fake.order[0..Fake.n]);
    try t.expect(f.e.seqs.items[1].? == kept and f.e.slot == 0);
    try t.expectEqual(@as(usize, 37), kept.st.pos); // no new eager reset at release
    try f.e.bind(1);
    try t.expect(f.e.cur == kept);
    try t.expectEqual(@as(usize, 2), Fake.allocs);
    try f.e.begin(null);
    try t.expect(f.e.cur.sampling == null and f.e.cur.in_head == 0);
    try t.expect(kept.st.pos == 0 and kept.st.mtp_len == 0 and kept.st.pending == 0);
    try t.expect(kept.st.base == 0 and kept.st.mtp_base == 0);
    for (kept.st.cur) |parity| try t.expectEqual(@as(u1, 0), parity);
    for (rec.bytes(kept.st.conv, kept.st.kda_layers * 3 * kept.st.conv_c * 2)) |b| try t.expectEqual(@as(u8, 0), b);
    for (rec.bytes(kept.st.rec, 2 * kept.st.kda_layers * kept.st.rec_layer)) |b| try t.expectEqual(@as(u8, 0), b);
    try t.expectEqual(@as(u8, 0xa5), rec.bytes(kept.st.caches[0].lc, 1)[0]); // original lazy cache semantics
    try t.expectEqual(@as(u64, 2), f.e.sequence_counts.created);
    try t.expectEqual(@as(u64, 1), f.e.sequence_counts.retained_releases);
}

test "sequence retention: only1to7 retained and actual deinit phase frees every retained slot" {
    var d = Fake.driver();
    var f: Fixture = undefined;
    try f.init(&d, true);
    for (1..9) |slot| {
        try f.e.bind(@intCast(slot));
        Fake.pending = true;
        f.e.release(@intCast(slot));
        try t.expectEqual(slot < 8, f.e.seqs.items[slot] != null);
    }
    try t.expectEqual(@as(u64, 9), f.e.sequence_counts.created);
    try t.expectEqual(@as(u64, 7), f.e.sequence_counts.retained_releases);
    try t.expectEqual(@as(usize, 1), Fake.frees);
    f.deinit();
    try t.expectEqual(@as(usize, 9), Fake.frees);
    try t.expectEqual(f.e.sequence_counts.created, f.e.sequence_counts.destroyed);
}

test "sequence retention: failed actual State creation leaves no retained owner or create count" {
    var d = Fake.driver();
    var f: Fixture = undefined;
    try f.init(&d, true);
    defer f.deinit();
    const zero = f.e.cur;
    Fake.fail_alloc = true;
    try t.expectError(error.NotReady, f.e.bind(2));
    Fake.fail_alloc = false;
    try t.expect(f.e.seqs.items[2] == null and f.e.cur == zero);
    try t.expectEqual(@as(u64, 1), f.e.sequence_counts.created);
    try f.e.bind(2);
    try t.expectEqual(@as(u64, 2), f.e.sequence_counts.created);
}

test "sequence retention: startup reserve materializes only admitted slots and keeps original reset" {
    var d = Fake.driver();
    var f: Fixture = undefined;
    try f.init(&d, true);
    defer f.deinit();
    const e = &f.e;
    const original = e.cur;
    try t.expectError(error.BadSequenceReserve, e.reserveSequenceSlots(4));
    e.settings.prewarm_sequence_slots = true;
    try t.expectError(error.BadSequenceReserve, e.reserveSequenceSlots(0));
    try t.expectError(error.BadSequenceReserve, e.reserveSequenceSlots(9));
    try t.expectEqual(@as(u64, 1), e.sequence_counts.created);
    Fake.fail_alloc = true;
    try t.expectError(error.NotReady, e.reserveSequenceSlots(4));
    Fake.fail_alloc = false;
    try t.expectEqual(original, e.cur);
    try e.reserveSequenceSlots(4);
    try t.expectEqual(@as(u64, 4), e.sequence_counts.created);
    try t.expectEqual(@as(u64, 0), e.sequence_counts.retained_releases);
    try t.expectEqual(@as(u32, 0), e.slot);
    try t.expectEqual(original, e.cur);
    const allocations = Fake.allocs;
    try e.bind(3);
    try e.begin(null);
    try t.expectEqual(allocations, Fake.allocs);
}
