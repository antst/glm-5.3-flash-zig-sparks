//! Actual startup parsing, family facts and shared copy planning; no driver or kernels.
const std = @import("std");
const lanes = @import("lanes");
const ranks = @import("cuda_ranks.zig");
const policy = @import("cuda_dflash_policy.zig");
const copy = @import("cuda_copy_policy.zig");
const Family = @import("cuda_lanes.zig").Lanes;
const Target = @import("cuda_engine.zig").Engine;
const a = std.testing.allocator;

test "DFlash policy: first-copy startup accepts only3or7 and freezes all-rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_FIRST_COPY_TOKENS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const legacy = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expectEqual(@as(u8, 3), legacy.first_copy_tokens);
    Env.value = "3";
    const explicit_legacy = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(std.meta.eql(legacy, explicit_legacy));
    Env.value = "7";
    const seven = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expectEqual(@as(u8, 7), seven.first_copy_tokens);
    for ([_][]const u8{ "", "0", "1", "4", "8", "15", "03", "+7", "7 ", "true" }) |bad| {
        Env.value = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_FIRST_COPY_TOKENS: 3/7, 15 requires width>=16, 31 requires width32", problem);
    }
    // The already-read setting is a value, independent of all later environment changes.
    Env.value = "3";
    try std.testing.expectEqual(@as(u8, 7), seven.first_copy_tokens);
    const FP = ranks.Fingerprint;
    const f0 = FP.of(4, 0, legacy, 4096, true, @splat(0), @splat(0), @splat(0));
    const f1 = FP.of(4, 1, seven, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 65), f0.layout);
    try std.testing.expectEqual(FP.Field.first_copy_tokens, ranks.mismatch(&.{ f0, f1 }).?.field);
    const agreed = FP.of(4, 0, seven, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expect(ranks.mismatch(&.{ agreed, f1 }) == null);
}

test "DFlash policy: first-copy3 preserves every legacy fact with either priority-copy mode" {
    for ([_]bool{ false, true }) |mtp| for ([_]bool{ false, true }) |block| {
        var target: Target = undefined;
        target.mtp = mtp;
        target.drafter = if (block) @ptrFromInt(16) else null;
        var family: Family = .{ .x = .{ .e = &target, .link = undefined } };
        for ([_]bool{ false, true }) |priority| for ([_]bool{ false, true }) |confidence| {
            family.confidence_policy = .{ .on = confidence, .floor = 0.7 };
            family.first_copy_tokens = 3;
            const old = policy.apply(family.confidence_policy, copy.apply(family.facts(), priority));
            try std.testing.expect(std.meta.eql(old, family.factsWithCopies(priority)));
            family.first_copy_tokens = 7;
            var expected = old;
            expected.first_copy_rows = 8;
            const seven = family.factsWithCopies(priority);
            try std.testing.expect(std.meta.eql(expected, seven));
            try std.testing.expectEqual(priority and mtp, family.copy_confidence);
            try std.testing.expectEqual(old.draft_confidence, seven.draft_confidence);
            try std.testing.expectEqual(old.allocate_shared_drafts, seven.allocate_shared_drafts);
            try std.testing.expectEqual(old.drafts, seven.drafts);
            try std.testing.expectEqual(old.exact_width, seven.exact_width);
        };
    };
}

test "DFlash policy: actual shared copyProposal asks3or7 and retains64row bound" {
    const Probe = struct {
        requested: i64 = -1,
        const ids = [_]u32{ 11, 12, 13, 14, 15, 16, 17 };
        fn propose(ptr: *anyopaque, _: []const u32, most: i64) ![]const u32 {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.requested = most;
            return ids[0..@intCast(@max(0, @min(most, ids.len)))];
        }
        fn match(_: *anyopaque) i64 {
            return 8;
        }
    };
    for ([_]u8{ 3, 7 }) |tokens| for ([_]bool{ false, true }) |priority| {
        var target: Target = undefined;
        target.mtp = true;
        target.drafter = @ptrFromInt(16);
        var family: Family = .{
            .x = .{ .e = &target, .link = undefined },
            .confidence_policy = .{ .on = true, .floor = 0.7 },
            .first_copy_tokens = tokens,
        };
        const model = family.factsWithCopies(priority);
        var cfg = try lanes.Config.init(a, model, 8, 7);
        defer cfg.deinit(a);
        try std.testing.expectEqual(@as(u32, tokens), cfg.first_copy);
        try std.testing.expectEqual(@as(u32, 7), cfg.max_copy);
        try std.testing.expectEqual(@as(u32, 7), cfg.most_drafts);
        try std.testing.expectEqual(@as(u32, 64), cfg.batch_rows);
        try std.testing.expectEqual(@as(?f64, 0.7), cfg.draft_confidence);
        try std.testing.expect(!cfg.allocate_shared_drafts);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, family.backend(), clock.clock());
        defer core.deinit();
        core.alone = false;
        var probes: [8]Probe = @splat(.{});
        var streams: [8]lanes.Stream = undefined;
        var plans: [8]lanes.windows.Plan = undefined;
        var made: usize = 0;
        defer for (streams[0..made]) |*s| s.deinit(a);
        for (&streams, &probes, &plans) |*s, *probe, *plan| {
            s.* = try lanes.Stream.init(a, .{ .id = "copy-cap", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 128, .proposer = .{ .ptr = probe, .vtable = &.{ .propose = Probe.propose, .last_match = Probe.match } } });
            made += 1;
            // A lone stream's ramp must not influence the shared branch under test.
            s.copy_width = 1;
            const proposal = try lanes.windows.copyProposal(&core, s);
            try std.testing.expectEqual(@as(i64, tokens), probe.requested);
            try std.testing.expectEqual(@as(usize, tokens), proposal.tokens.len);
            plan.* = try lanes.windows.plan(&core, s, proposal);
            try std.testing.expectEqual(lanes.windows.Kind.copy, plan.kind);
        }
        try lanes.windows.allocate(&core, &plans);
        var rows: usize = 0;
        for (plans) |plan| {
            try std.testing.expectEqual(@as(usize, tokens), plan.count());
            rows += 1 + plan.count();
        }
        try std.testing.expectEqual(@as(usize, 8 * (1 + @as(u32, tokens))), rows);
        try std.testing.expect(rows <= cfg.batch_rows);
    };
}
