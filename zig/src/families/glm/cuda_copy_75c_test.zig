//! Frozen Python comparisons plus the native query-tail and request lifetime contracts.
const std = @import("std");
const t = std.testing;
const copies = @import("cuda_copy_75c.zig");
const ranks = @import("cuda_ranks.zig");
const fixtures = @import("cuda_copy_75c_fixtures.zig");
const lanes = @import("lanes");
const api = @import("engine_api");

test "GLM 75c copy: actual pinned Python selector fixtures" {
    for (fixtures.cases) |c| {
        var copy: copies.Copy = .{ .gpa = t.allocator, .prompt = c.prompt };
        defer copy.deinit();
        const p = copy.proposer();
        try t.expectEqualSlices(u32, c.expected, try p.propose(c.context, c.room));
        try t.expect(!p.vtable.priced);
        try t.expect(p.vtable.observe == null);
        p.observe(15, 0);
        try t.expectEqualSlices(u32, c.expected, try p.propose(c.context, c.room));
    }
}

test "GLM 75c copy: strict cached default-off and all-rank fingerprint agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_COPY_POLICY_75C")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try t.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.copy_policy_75c and on.copy_policy_75c);
    var only = off;
    only.copy_policy_75c = true;
    try t.expect(std.meta.eql(only, on));
    Env.value = "0";
    try t.expect(on.copy_policy_75c);
    for ([_][]const u8{ "", "2", "01", "true", "yes", " 1", "1 " }) |v| {
        Env.value = v;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_COPY_POLICY_75C: 0 or 1", problem);
    }
    const FP = ranks.Fingerprint;
    var all = [_]FP{
        FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0)),
        FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0)),
    };
    try t.expectEqual(FP.Field.copy_policy_75c, ranks.mismatch(&all).?.field);
    all[0] = FP.of(4, 0, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expect(ranks.mismatch(&all) == null);
    try t.expectEqual(@as(u32, 63), all[0].layout);
}

test "GLM 75c copy: actual factory owns separate requests and OFF retains null/min_match4" {
    var ctx: u8 = 0;
    try t.expect(copies.factory(false, &ctx) == null);
    var cfg = try lanes.Config.init(t.allocator, .{}, 8, 7);
    defer cfg.deinit(t.allocator);
    var core: lanes.Engine = undefined;
    core.cfg = &cfg;
    const host = api.LaneHost.init(t.allocator, t.io, &core, .{});
    try t.expect(host.proposers == null);
    try t.expectEqual(@as(i64, 4), host.min_match);
    var legacy = try lanes.SuffixLookup.init(t.allocator, .{ .min_match = host.min_match });
    defer legacy.deinit();
    try t.expectEqualSlices(u32, &.{9}, try legacy.propose(&.{ 1, 2, 3, 4, 9, 1, 2, 3, 4 }, 1));
    const f = copies.factory(true, &ctx).?;
    const a = try f.make(f.ctx, t.allocator, &.{ 1, 2 }, &.{99});
    defer f.free(f.ctx, a);
    const b = try f.make(f.ctx, t.allocator, &.{ 1, 2, 3 }, &.{});
    defer f.free(f.ctx, b);
    try t.expect(a.ptr != b.ptr);
    const aa: *copies.Copy = @ptrCast(@alignCast(a.ptr));
    const bb: *copies.Copy = @ptrCast(@alignCast(b.ptr));
    try t.expectEqual(@as(usize, 2), aa.prompt);
    try t.expectEqual(@as(usize, 3), bb.prompt);
    const ctx_ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try a.propose(&ctx_ids, 7);
    try t.expect(aa.indexed > 0);
    try t.expectEqual(@as(usize, 0), bb.indexed);
}

test "GLM 75c copy: query tail is never indexed or returned, including old pending-tail divergence" {
    const history = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 99, 10, 11, 12, 13, 14, 15, 16 };
    var c: copies.Copy = .{ .gpa = t.allocator, .prompt = 1000 };
    defer c.deinit();
    const p = c.proposer();
    const got = try p.proposeAfter(&history, &.{17}, 15);
    try t.expectEqualSlices(u32, &.{ 99, 10, 11, 12, 13, 14, 15, 16 }, got);
    try t.expectEqual(history.len, c.indexed);
    try t.expectEqual(@as(?usize, 8), p.lastEnd());
    try t.expectEqual(@as(i64, 8), p.lastMatch());
    _ = try p.proposeAfter(&history, &.{12345}, 15);
    try t.expectEqual(history.len, c.indexed);
    try t.expectEqual(@as(usize, 0), c.out.items.len);
}

test "GLM 75c copy: incremental append, truncate, boundary replacement and current caller caps" {
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 1, 2, 3, 4, 5, 6, 7, 8 };
    var c: copies.Copy = .{ .gpa = t.allocator, .prompt = 100 };
    defer c.deinit();
    const p = c.proposer();
    _ = try p.propose(ids[0..10], 7);
    try t.expectEqual(@as(usize, 10), c.indexed);
    for ([_]i64{ 1, 3, 7, 15 }) |cap| {
        const got = try p.propose(&ids, cap);
        try t.expect(got.len <= @as(usize, @intCast(cap)));
        try t.expectEqual(ids.len, c.indexed);
    }
    const buckets = c.index.count();
    _ = try p.propose(&ids, 15);
    try t.expectEqual(buckets, c.index.count());
    _ = try p.propose(ids[0..9], 3);
    try t.expectEqual(@as(usize, 9), c.indexed);
    var changed = ids;
    changed[8] = 999;
    _ = try p.propose(changed[0..9], 3);
    try t.expectEqual(@as(usize, 9), c.indexed);
    try t.expectEqual(@as(usize, 0), (try p.propose(&.{}, 0)).len);
    try t.expectEqual(@as(i64, 0), p.lastMatch());
}

fn failing(gpa: std.mem.Allocator) !void {
    var ctx: u8 = 0;
    const f = copies.factory(true, &ctx).?;
    const p = try f.make(f.ctx, gpa, &.{1}, &.{});
    defer f.free(f.ctx, p);
    _ = try p.propose(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8 }, 7);
}

test "GLM 75c copy: every factory/index/output allocation failure releases owned storage" {
    try t.checkAllAllocationFailures(t.allocator, failing, .{});
}
