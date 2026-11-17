//! Actual physical State, Forward and shared offset records; no driver library or device arithmetic.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const lanes = @import("lanes");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const t = std.testing;
const fw = glm.forward;
const h: usize = 16;
const n = h * 128;
const layers = 3;

test "verify rows: strict startup width copy coupling cached values and rank agreement" {
    const Env = struct {
        var rows: ?[]const u8 = null;
        var copy: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_VERIFY_ROWS")) return rows;
            return if (std.mem.eql(u8, name, "TF_GLM_FIRST_COPY_TOKENS")) copy else null;
        }
    };
    defer {
        Env.rows = null;
        Env.copy = null;
    }
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(@as(u8, 8), off.verify_rows);
    Env.rows = "8";
    try t.expect(std.meta.eql(off, try glm.ranks.Settings.read(Env.get, &problem)));
    Env.rows = "16";
    Env.copy = "15";
    const on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.rows = "8";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    try t.expectEqual(@as(u8, 16), on.verify_rows);
    try t.expectEqual(@as(u8, 15), on.first_copy_tokens);
    Env.copy = null;
    for ([_][]const u8{ "", "0", "7", "9", "15", "17", "016", "+16", "16 " }) |bad| {
        Env.rows = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    }
    Env.rows = "32";
    Env.copy = "31";
    const wide32 = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(@as(u8, 32), wide32.verify_rows);
    try t.expectEqual(@as(u8, 31), wide32.first_copy_tokens);
    Env.rows = "16";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    Env.rows = "32";
    Env.copy = "15";
    try t.expectEqual(@as(u8, 15), (try glm.ranks.Settings.read(Env.get, &problem)).first_copy_tokens);
    const FP = glm.ranks.Fingerprint;
    const f0 = FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const f1 = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 64), f0.layout);
    try t.expectEqual(FP.Field.verify_rows, glm.ranks.mismatch(&.{ f0, f1 }).?.field);
    const wide_fp = FP.of(4, 1, wide32, 4096, true, @splat(0), @splat(0), @splat(0));
    const old_fp = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.verify_rows, glm.ranks.mismatch(&.{ old_fp, wide_fp }).?.field);
    const agreed = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expect(glm.ranks.mismatch(&.{ agreed, f1 }) == null);
}

test "verify rows: actual Engine startup refuses scope before allocation or config read" {
    const Set = glm.ranks.Settings;
    for ([_]u8{ 16, 32 }) |width| for ([_]Set{
        .{ .verify_rows = width, .pages = true },         .{ .verify_rows = width, .kv_host = 1 },
        .{ .verify_rows = width, .verify_graphs = true },
    }) |s| try t.expectError(error.VerifyRowsScope, glm.engine.Engine.init(t.failing_allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = s, .capacity = 4096, .mtp = true, .drafter = true }));
    try t.expectError(error.VerifyRowsNeedsDFlash, glm.engine.Engine.init(t.failing_allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .verify_rows = 16 }, .capacity = 4096, .mtp = true }));
    try t.expectError(error.OutOfMemory, glm.engine.Engine.init(t.failing_allocator, undefined, undefined, undefined, "", undefined, null, 0, 2, .{ .settings = .{ .verify_rows = 16 }, .capacity = 4096, .mtp = true, .drafter = true }));
    try t.expectError(error.VerifyRowsScope, glm.engine.Engine.init(t.failing_allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .verify_rows = 16 }, .capacity = 4096, .mtp = true, .drafter = true, .vision = .{ .image_tokens = 64, .max_images = 1 } }));
}

test "verify rows: actual claim rejects sampled and grammar drafts before selecting a slot" {
    const Choice = struct {
        var chosen: usize = 0;
        fn choose(_: *glm.lanes.Lanes) u32 {
            chosen += 1;
            return 0;
        }
    };
    var family: glm.lanes.Lanes = .{ .x = .{ .e = undefined, .link = undefined }, .verify_rows = 16 };
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "guard", .prompt = &.{1}, .max_new = 16 });
    defer stream.deinit(t.allocator);
    stream.sampling = .{ .temperature = 1, .seed = 1 };
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, family.claimUsing(&stream, Choice.choose));
    stream.sampling = null;
    stream.grammar = @ptrFromInt(16);
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, family.claimUsing(&stream, Choice.choose));
    stream.grammar = null;
    try t.expectEqual(@as(usize, 0), Choice.chosen);
}

test "verify rows: actual Engine verify refuses active width and nonordinary rows before target setup" {
    var e: glm.engine.Engine = undefined;
    var seq: @typeInfo(@FieldType(glm.engine.Engine, "cur")).pointer.child = undefined;
    seq.sampling = null;
    e.cur = &seq;
    e.settings = .{};
    e.row_counts = .{};
    e.structured = false;
    var ids: [33]u32 = @splat(1);
    var out: [33]u32 = @splat(0xdeadbeef);
    try t.expectError(error.WindowTooWide, e.verify(ids[0..9], out[0..9], 0));
    e.settings.verify_rows = 16;
    try t.expectError(error.WindowTooWide, e.verify(&ids, &out, 0));
    e.settings.verify_rows = 32;
    try t.expectError(error.WindowTooWide, e.verify(&ids, &out, 0));
    seq.sampling = .{ .temperature = 1, .seed = 1 };
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, e.verify(ids[0..16], out[0..16], 0));
    seq.sampling = null;
    e.structured = true;
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, e.verify(ids[0..16], out[0..16], 1));
    for (out) |v| try t.expectEqual(@as(u32, 0xdeadbeef), v);
    try t.expectEqual(@as(usize, 0), e.row_counts.max_part_rows);
}

test "verify rows: ON16/32 success census distinguishes actual parts from neural and copy capacities" {
    var e: glm.engine.Engine = undefined;
    e.settings = .{};
    e.row_counts = .{};
    glm.verify_rows.solo(&e, 16);
    try t.expectEqual(@as(usize, 0), e.row_counts.max_part_rows);
    e.settings.verify_rows = 16;
    e.settings.first_copy_tokens = 15;
    glm.verify_rows.solo(&e, 8);
    try t.expectEqual(@as(u64, 0), e.row_counts.solo_wide_calls);
    glm.verify_rows.solo(&e, 9);
    var ids: [16]u32 = @splat(1);
    const parts = [_]glm.engine.Part{
        .{ .slot = 0, .ids = ids[0..3], .out = &.{} },
        .{ .slot = 1, .ids = &ids, .out = &.{} },
    };
    glm.verify_rows.shared(&e, &parts);
    try t.expectEqual(@as(usize, 16), e.row_counts.max_part_rows);
    try t.expectEqual(@as(u64, 1), e.row_counts.solo_wide_calls);
    try t.expectEqual(@as(u64, 1), e.row_counts.shared_wide_calls);
    e.settings.verify_rows = 32;
    e.settings.first_copy_tokens = 31;
    glm.verify_rows.solo(&e, 32);
    try t.expectEqual(@as(usize, 32), e.row_counts.max_part_rows);
    try glm.verify_rows.request(32, false, true, true);
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, glm.verify_rows.window(32, 17, true, false));
    try glm.verify_rows.request(16, false, true, true);
    try glm.verify_rows.window(16, 1, true, true);
    try t.expectError(error.VerifyRowsNeedsGreedyDrafts, glm.verify_rows.window(16, 2, false, true));
}

test "verify rows: actual facts and copyProposal ask15/31 without increasing neural7 or shared64" {
    const Probe = struct {
        requested: i64 = -1,
        const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
        fn propose(ptr: *anyopaque, _: []const u32, most: i64) ![]const u32 {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.requested = most;
            return ids[0..@intCast(@max(0, @min(most, ids.len)))];
        }
        fn match(_: *anyopaque) i64 {
            return 8;
        }
    };
    for ([_]u8{ 8, 16, 32 }) |width| for ([_]bool{ false, true }) |priority| {
        var target: glm.engine.Engine = undefined;
        target.mtp = true;
        target.drafter = @ptrFromInt(16);
        var family: glm.lanes.Lanes = .{ .x = .{ .e = &target, .link = undefined }, .verify_rows = width, .first_copy_tokens = if (width >= 16) width - 1 else 3, .confidence_policy = .{ .on = true, .floor = 0.7 } };
        const model = family.factsWithCopies(priority);
        var cfg = try lanes.Config.init(t.allocator, model, width, 7);
        defer cfg.deinit(t.allocator);
        try t.expectEqual(@as(u32, width - 1), cfg.max_copy);
        try t.expectEqual(@as(u32, if (width >= 16) width - 1 else 3), cfg.first_copy);
        try t.expectEqual(@as(u32, 7), cfg.most_drafts);
        try t.expectEqual(@as(u32, 64), cfg.batch_rows);
        try t.expect(!cfg.allocate_shared_drafts and cfg.draft_confidence.? == 0.7);
        try t.expectEqual(@as(usize, width), model.window_costs.len);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(t.allocator, &cfg, family.backend(), clock.clock());
        defer core.deinit();
        core.alone = false;
        var probes: [8]Probe = @splat(.{});
        var streams: [8]lanes.Stream = undefined;
        var plans: [8]lanes.windows.Plan = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(t.allocator);
        for (&streams, &probes, &plans) |*s, *probe, *plan| {
            s.* = try lanes.Stream.init(t.allocator, .{ .id = "width", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 128, .proposer = .{ .ptr = probe, .vtable = &.{ .propose = Probe.propose, .last_match = Probe.match } } });
            made += 1;
            s.copy_width = 1;
            const p = try lanes.windows.copyProposal(&core, s);
            try t.expectEqual(@as(i64, if (width >= 16) width - 1 else 3), probe.requested);
            plan.* = try lanes.windows.plan(&core, s, p);
        }
        try lanes.windows.allocate(&core, plans[0..4]);
        var total: usize = 0;
        for (plans[0..4]) |p| total += 1 + p.count();
        try t.expectEqual(@as(usize, if (width >= 16) 64 else 16), total);
        try lanes.windows.allocate(&core, &plans);
        total = 0;
        for (plans) |p| {
            total += 1 + p.count();
            try t.expect(p.count() < width);
        }
        try t.expect(total <= 64);
    };
}

const Fake = struct {
    var states: []fw.State = &.{};
    var shared = false;
    var row0: [2]usize = .{ 0, 0 };
    var copy_count: usize = 0;
    var projected: u64 = 0;
    var li: usize = 0;
    var preps: usize = 0;
    var steps: usize = 0;
    var replays: usize = 0;
    var shifts: usize = 0;
    var keep: usize = 0;
    var commit_st: *fw.State = undefined;
    fn ptr(a: [*]?*anyopaque, i: usize) u64 {
        return @as(*align(1) const u64, @ptrCast(a[i].?)).*;
    }
    fn integer(a: [*]?*anyopaque, i: usize) c_int {
        return @as(*align(1) const c_int, @ptrCast(a[i].?)).*;
    }
    fn wide(a: [*]?*anyopaque, i: usize) usize {
        return @as(*align(1) const usize, @ptrCast(a[i].?)).*;
    }
    fn fun(d: *const cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
    fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (shared) {
            const st = &states[copy_count];
            std.debug.assert(dst == st.proj + li * st.proj_layer * 2);
            std.debug.assert(src == projected + row0[copy_count] * st.proj_row * 2);
            std.debug.assert(bytes == st.pending * st.proj_row * 2);
            copy_count += 1;
        }
        @memcpy(rec.bytes(dst, bytes), rec.bytes(src, bytes));
        return 0;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const a = params.?;
        switch (@intFromPtr(f)) {
            1 => {
                const st = &states[if (shared) preps else 0];
                const kv = li * 32 * n;
                std.debug.assert(gx == st.pending and gy == h and integer(a, 0) == h);
                std.debug.assert(ptr(a, 1) == st.proj + li * st.proj_layer * 2);
                std.debug.assert(ptr(a, 12) == st.save_k + kv * 4);
                std.debug.assert(ptr(a, 13) == st.save_v + kv * 2);
                std.debug.assert(ptr(a, 14) == st.save_g + kv * 4);
                std.debug.assert(ptr(a, 15) == st.save_b + li * 32 * h * 4);
                preps += 1;
            },
            2 => {
                const st = &states[if (shared) steps else 0];
                std.debug.assert(integer(a, 7) == st.pending);
                std.debug.assert(ptr(a, 1) == st.recAt(st.cur[li], li));
                std.debug.assert(ptr(a, 9) == st.recAt(1 - st.cur[li], li));
                steps += 1;
            },
            4 => {
                const st = commit_st;
                std.debug.assert(gx == layers * h and integer(a, 0) == h);
                std.debug.assert(ptr(a, 1) == st.recAt(st.cur[0], 0));
                std.debug.assert(wide(a, 2) == st.rec_layer / 4);
                std.debug.assert(ptr(a, 3) == st.save_k and ptr(a, 4) == st.save_v);
                std.debug.assert(ptr(a, 5) == st.save_g and ptr(a, 6) == st.save_b);
                std.debug.assert(wide(a, 7) == 32 * n and wide(a, 8) == 32 * h);
                std.debug.assert(integer(a, 9) == keep and ptr(a, 10) == st.recAt(1 - st.cur[0], 0));
                replays += 1;
            },
            5 => {
                const st = commit_st;
                std.debug.assert(gx == layers and ptr(a, 0) == st.conv and ptr(a, 1) == st.proj);
                std.debug.assert(wide(a, 2) == keep and wide(a, 4) == st.proj_layer);
                std.debug.assert(wide(a, 5) == st.proj_row and st.proj_layer == 32 * st.proj_row);
                shifts += 1;
            },
            else => {},
        }
        return 0;
    }
};

const Fixture = struct {
    w: glm.weights.Weights,
    layers_: [layers]glm.weights.Layer = undefined,
    modules: fw.Modules = undefined,
    states: [2]fw.State = undefined,
    b: fw.Buffers,
    ranks: [1]fw.Rank = undefined,
    segs: [2]fw.Seg = undefined,
    f: fw.Forward = undefined,
    fn init(fx: *Fixture, d: *const cuda.Driver, shared: bool) !void {
        const cfg: glm.config.Config = .{ .layers = layers, .run = layers, .hidden = 64, .vocab = 256, .experts = 9, .topk = 8, .dense_inter = 256, .moe_inter = 128, .shared_inter = 128 };
        fx.w = .{ .gpa = t.allocator, .config = cfg, .share = try glm.shard.Share.of(&cfg, 0, 4) };
        for (&fx.layers_) |*l| l.mixer = .{ .kda = .{ .proj = .{ .bf16 = 0 }, .fb = 0, .gb = 0, .o = .{ .bf16 = 0 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-5, .lower = -5, .heads = h, .dims = cfg.hidden } };
        fx.w.layers = &fx.layers_;
        for (&fx.states) |*st| st.* = try fw.State.initFormat(t.allocator, d, &fx.w, 256, true);
        fx.b = if (shared) try fw.Buffers.initShared(d, &fx.w, 64, 256) else try fw.Buffers.init(d, &fx.w, 32, false, 256);
        inline for (@typeInfo(@TypeOf(fx.modules.glue)).@"struct".field_names) |field|
            @field(fx.modules.glue, field) = Fake.fun(d, 6);
        inline for (.{ "prep", "step", "out", "replay_layers", "chain", "replay" }, .{ 1, 2, 3, 4, 7, 8 }) |field, id| @field(fx.modules.kda, field) = Fake.fun(d, id);
        fx.modules.glue.conv_shift = Fake.fun(d, 5);
        fx.modules.tiled = false;
        fx.modules.index = null;
        fx.modules.packs = null;
        fx.modules.copy_rows = Fake.fun(d, 9);
        fx.ranks[0] = .{ .w = &fx.w, .st = &fx.states[0], .b = &fx.b };
        fx.f = .{ .d = d, .stream = .{ .d = d, .handle = null }, .m = &fx.modules, .ranks = &fx.ranks, .exchange = .local, .kda_wide = true };
    }
    fn deinit(fx: *Fixture) void {
        fx.b.deinit();
        for (&fx.states) |*st| st.deinit();
    }
};

test "verify rows: actual State and Forward mixer save all three layers with32row strides" {
    var d = rec.init();
    d.func_name = null;
    d.api.cuLaunchKernel = Fake.launch;
    d.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    var fx: Fixture = undefined;
    try fx.init(&d, false);
    defer fx.deinit();
    Fake.states = &fx.states;
    Fake.shared = false;
    for ([_]usize{ 9, 15, 16, 17, 31, 32 }) |rows| for ([_]u1{ 0, 1 }) |parity| {
        const st = &fx.states[0];
        st.pending = rows;
        @memset(st.cur, parity);
        for (0..layers) |li| {
            Fake.li = li;
            Fake.preps = 0;
            Fake.steps = 0;
            try fx.f.mixer(fx.ranks[0], fx.layers_[li], li, 0, rows, 1, null);
            try t.expectEqual(@as(usize, 1), Fake.preps);
            try t.expectEqual(@as(usize, 1), Fake.steps);
        }
        for (1..rows + 1) |keep| {
            st.pending = rows;
            st.pos = 31;
            @memset(st.cur, parity);
            Fake.keep = keep;
            Fake.commit_st = st;
            Fake.replays = 0;
            Fake.shifts = 0;
            try fx.f.commit(keep);
            try t.expectEqual(@as(usize, @intFromBool(keep < rows)), Fake.replays);
            try t.expectEqual(@as(usize, 1), Fake.shifts);
            try t.expectEqual(@as(usize, 0), st.pending);
            try t.expectEqual(31 + keep, st.pos);
            for (st.cur) |p| try t.expectEqual(@as(u1, 1 - parity), p);
        }
    };
    try t.expect(fx.states[0].proj + layers * 32 * 6416 * 2 <= fx.states[0].save_k);
    try t.expect(fx.states[0].save_b + layers * 32 * h * 4 <= fx.states[0].mtp_caches.lc);
    try t.expectEqual(@as(usize, 32), fx.b.part_rows);
}

test "verify rows: actual shared mixer copies each variable part into its own32row layer save region" {
    var d = rec.init();
    d.func_name = null;
    d.api.cuLaunchKernel = Fake.launch;
    d.api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    var fx: Fixture = undefined;
    try fx.init(&d, true);
    defer fx.deinit();
    Fake.states = &fx.states;
    Fake.shared = true;
    defer Fake.shared = false;
    Fake.projected = fx.b.kproj;
    try t.expectEqual(@as(usize, 32), fx.b.part_rows);
    for ([_]usize{ 9, 15, 16, 17, 31, 32 }) |rows| {
        fx.segs = .{ .{ .st = &fx.states[0], .row0 = 0, .rows = rows }, .{ .st = &fx.states[1], .row0 = rows, .rows = 32 } };
        fx.ranks[0].segs = &fx.segs;
        Fake.row0 = .{ 0, rows };
        for (&fx.states, 0..) |*st, i| {
            st.pending = fx.segs[i].rows;
            @memset(st.cur, @intCast(i));
        }
        for (0..layers) |li| {
            Fake.li = li;
            Fake.preps = 0;
            Fake.steps = 0;
            Fake.copy_count = 0;
            try glm.shared.mixer(&fx.f, fx.ranks[0], fx.layers_[li], li, 0, rows + 32, null);
            try t.expectEqual(@as(usize, 2), Fake.copy_count);
            try t.expectEqual(@as(usize, 2), Fake.preps);
            try t.expectEqual(@as(usize, 2), Fake.steps);
        }
    }
}

test "verify rows: shared scratch owns projection even when its total equals physical32" {
    var d = rec.init();
    d.func_name = null;
    var fx: Fixture = undefined;
    try fx.init(&d, false);
    defer fx.deinit();
    var shared = try fw.Buffers.initShared(&d, &fx.w, 32, 256);
    defer shared.deinit();
    try t.expectEqual(@as(usize, 32), fw.window_rows);
    try t.expectEqual(@as(usize, 32), shared.part_rows);
    try t.expect(shared.kda.ka - shared.kproj >= 32 * 6416 * 2);
    try t.expect(fx.b.kda.ka - fx.b.kproj < 32 * 6416 * 2);
    const labels = glm.engine_prof;
    try t.expectEqualStrings("verify 17 rows", labels.verifyLabel(17));
    try t.expectEqualStrings("verify 32 rows", labels.verifyLabel(32));
}
