//! Actual family facts and core allocation, without constructing a driver, kernels or a model.
const std = @import("std");
const lanes = @import("lanes");
const ranks = @import("cuda_ranks.zig");
const policy = @import("cuda_dflash_policy.zig");
const copy = @import("cuda_copy_policy.zig");
const Family = @import("cuda_lanes.zig").Lanes;
const Target = @import("cuda_engine.zig").Engine;
const a = std.testing.allocator;

test "DFlash policy: actual factsWithCopies preserves OFF fields and applies confidence after copies" {
    var target: Target = undefined;
    target.mtp = true;
    target.drafter = @ptrFromInt(16);
    var family: Family = .{ .x = .{ .e = &target, .link = undefined } };
    for ([_]bool{ false, true }) |copies| {
        family.confidence_policy = .{};
        const old = copy.apply(family.facts(), copies);
        const off = family.factsWithCopies(copies);
        try std.testing.expect(std.meta.eql(old, off));
        family.confidence_policy = .{ .on = true, .floor = 0.7 };
        const on = family.factsWithCopies(copies);
        var expected = old;
        expected.draft_confidence = 0.7;
        expected.allocate_shared_drafts = false;
        try std.testing.expect(std.meta.eql(expected, on));
        try std.testing.expectEqual(copies, family.copy_confidence);
        try std.testing.expectEqual(old.root_copies, on.root_copies);
        try std.testing.expectEqual(old.context_copies, on.context_copies);
        try std.testing.expectEqual(old.own_copy_rates, on.own_copy_rates);
        try std.testing.expectEqual(old.copy_match, on.copy_match);
        try std.testing.expectEqual(old.exact_width, on.exact_width);
    }
}

test "DFlash policy: actual family probabilities core budgets and windows obey prefix and32/64caps" {
    for ([_]u32{ 32, 64 }) |cap| for ([_]f64{ 1, 0.9 }) |chance| for ([_]bool{ false, true }) |copies| {
        var target: Target = undefined;
        target.gpa = a;
        target.mtp = true;
        target.drafter = @ptrFromInt(16);
        var family: Family = .{
            .x = .{ .e = &target, .link = undefined },
            .confidence_policy = .{ .on = true, .floor = 0.7 },
        };
        defer family.deinit();
        var model = family.factsWithCopies(copies);
        // The family remains the selected32 source; exercise the same core with either admitted row capacity.
        model.batch_rows = cap;
        var cfg = try lanes.Config.init(a, model, model.exact_width, 7);
        defer cfg.deinit(a);
        try std.testing.expect(!cfg.allocate_shared_drafts);
        try std.testing.expectEqual(@as(?f64, 0.7), cfg.draft_confidence);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, family.backend(), clock.clock());
        defer core.deinit();
        var streams: [8]lanes.Stream = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        var plans: [8]lanes.windows.Plan = undefined;
        var who: [8]lanes.depth.Who = undefined;
        for (&streams, &plans, &who) |*s, *p, *w| {
            s.* = try lanes.Stream.init(a, .{ .id = "policy", .prompt = &.{1}, .max_new = 128 });
            made += 1;
            s.granted = 0; // No reservation slack may hold the greedy proposal at a previous zero grant.
            const held = try family.claim(s);
            held.held_n = 7;
            @memset(&held.chances, chance);
            p.* = .{ .stream = s, .position = 0, .kind = .head, .held = 7 };
            w.* = lanes.windows.who(s);
        }
        core.rule.mtp_step_ms = 1e9; // Explicit confidence must bypass cost-based next-pass reservation.
        var budgets: [8]i64 = undefined;
        try core.rule.budgets(&who, lanes.windows.probe(&core), false, &budgets);
        for (budgets) |b| try std.testing.expectEqual(@as(i64, 7), b);
        try lanes.windows.allocate(&core, &plans);
        var rows: usize = 0;
        for (plans) |p| {
            try std.testing.expect(p.held <= 7);
            rows += 1 + p.count();
            if (chance == 0.9) try std.testing.expectEqual(@as(u32, 3), p.held); //0.9^3>=.7,0.9^4<.7.
        }
        try std.testing.expect(rows <= cap);
        try std.testing.expectEqual(@as(usize, if (chance == 1) cap else 32), rows);
    };
}

test "DFlash policy: sampled drafting refuses before slot selection while plain sampled and OFF remain usable" {
    var target: Target = undefined;
    target.gpa = a;
    var family: Family = .{
        .x = .{ .e = &target, .link = undefined },
        .confidence_policy = .{ .on = true },
    };
    defer family.lanes.deinit(a);
    var s = try lanes.Stream.init(a, .{
        .id = "sampled",
        .prompt = &.{1},
        .max_new = 32,
        .sampling = .{ .seed = 9 },
    });
    defer s.deinit(a);
    const Probe = struct {
        var calls: usize = 0;
        fn choose(_: *const Family) u32 {
            calls += 1;
            return 0;
        }
    };
    Probe.calls = 0;
    try std.testing.expectError(error.GlmConfidenceOnlyNeedsGreedyDrafts, family.claimUsing(&s, Probe.choose));
    try std.testing.expectEqual(@as(usize, 0), Probe.calls);
    try std.testing.expectEqual(@as(usize, 0), family.lanes.count());
    s.drafts = false;
    _ = try family.claimUsing(&s, Probe.choose);
    try std.testing.expectEqual(@as(usize, 1), Probe.calls);
    family.confidence_policy.on = false;
    s.drafts = true;
    _ = try family.claimUsing(&s, Probe.choose);
    try std.testing.expectEqual(@as(usize, 1), Probe.calls); // Existing lane lookup, no new selection.
    try policy.require(.{}, false, false);
    try std.testing.expectError(error.GlmConfidenceOnlyNeedsDFlash, policy.require(.{ .on = true }, false, true));
    try std.testing.expectError(error.GlmConfidenceOnlyNeedsDFlash, policy.require(.{ .on = true }, true, false));
}

test "DFlash policy: strict optin immutable floor and all-rank fingerprint ignore floor only when OFF" {
    const Env = struct {
        var on: ?[]const u8 = null;
        var floor: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_CONFIDENCE_ONLY")) return on;
            return if (std.mem.eql(u8, name, "TF_GLM_DRAFT_CONFIDENCE")) floor else null;
        }
    };
    defer {
        Env.on = null;
        Env.floor = null;
    }
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(!off.dflash_confidence_only);
    Env.floor = "not-a-number";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_confidence_only);
    Env.on = "0";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_confidence_only);
    Env.on = "1";
    Env.floor = null;
    const on = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(on.dflash_confidence_only);
    try std.testing.expectEqual(@as(f64, 0.7), on.dflash_confidence_floor);
    for ([_][]const u8{ "", "NaN", "inf", "-0.1", "1.1", "oops" }) |bad| {
        Env.floor = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    for ([_][]const u8{ "0", "0.3", "0.7", "1" }) |ok| {
        Env.floor = ok;
        const parsed = try ranks.Settings.read(Env.get, &problem);
        try std.testing.expectEqual(try std.fmt.parseFloat(f64, ok), parsed.dflash_confidence_floor);
    }
    Env.floor = "0.3";
    const lower = try ranks.Settings.read(Env.get, &problem);
    const frozen = policy.from(lower);
    Env.floor = "0.7";
    try std.testing.expectEqual(@as(f64, 0.3), frozen.floor);
    Env.on = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    const FP = ranks.Fingerprint;
    const make = struct {
        fn fp(rank: usize, settings: ranks.Settings) FP {
            return FP.of(2, rank, settings, 8192, true, @splat(0), @splat(0), @splat(0));
        }
    }.fp;
    const flag_mismatch = ranks.mismatch(&.{ make(0, off), make(1, on) }).?;
    try std.testing.expectEqual(FP.Field.dflash_confidence_only, flag_mismatch.field);
    const floor_mismatch = ranks.mismatch(&.{ make(0, on), make(1, lower) }).?;
    try std.testing.expectEqual(FP.Field.dflash_confidence_floor, floor_mismatch.field);
    try std.testing.expect(ranks.mismatch(&.{ make(0, lower), make(1, lower) }) == null);
    try std.testing.expectEqual(@as(u32, 62), make(0, on).layout);
}
