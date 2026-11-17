//! Opt-in existing KDA batching, with small live groups kept on their original per-sequence kernels.
const std = @import("std");
pub const Options = struct { on: bool = false, min_streams: u32 = 8 };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Options {
    var result: Options = .{};
    result.on = @import("cuda_dense16.zig").flag(get("TF_GLM_KDA_SEGMENTS"), problem, "TF_GLM_KDA_SEGMENTS: 0 or 1") catch return error.BadSetting;
    if (get("TF_GLM_KDA_MIN_STREAMS")) |value| {
        result.min_streams = if (std.mem.eql(u8, value, "2")) 2 else if (std.mem.eql(u8, value, "4"))
            4
        else if (std.mem.eql(u8, value, "8"))
            8
        else {
            problem.* = "TF_GLM_KDA_MIN_STREAMS: 2, 4 or 8";
            return error.BadSetting;
        };
    }
    return result;
}

pub fn require(settings: anytype, world: usize, vision: bool, mtp: bool) !void {
    if (!settings.kda_segments) return;
    if (world != 4 or !mtp or !settings.dense_segments or !settings.kda_wide or settings.pages or
        settings.verify_graphs or settings.kv_host != 0 or vision) return error.KdaBatchScope;
}

pub fn open(e: anytype) !void {
    if (!e.settings.kda_segments) return;
    const buffers = if (e.bs) |*b| b else return error.KdaBatchScope;
    e.kda_segments = try @import("cuda_kda_owner.zig").Owner.init(e.gpa, e.io, e.d, e.stream, &e.w, buffers.rows);
    e.kda_segments.?.min_segments = e.settings.kda_min_streams;
    std.log.info("rank {d}: GLM_KDA_SEGMENTS active=true warps=4 min_streams={d} scratch_and_pinned={d}", .{ e.share.rank, e.settings.kda_min_streams, e.kda_segments.?.bytes() });
}

test "KDA adaptive strict cached mode, scope and rank agreement" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var mode: ?[]const u8 = null;
        var least: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_KDA_SEGMENTS")) return mode;
            return if (std.mem.eql(u8, name, "TF_GLM_KDA_MIN_STREAMS")) least else null;
        }
    };
    defer {
        Env.mode = null;
        Env.least = null;
    }
    var problem: []const u8 = "";
    const legacy = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(!legacy.kda_segments);
    try std.testing.expectEqual(@as(u32, 8), legacy.kda_min_streams);
    Env.mode = "0";
    try std.testing.expect(std.meta.eql(legacy, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "2", "true", "01" }) |bad| {
        Env.mode = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.mode = "1";
    for ([_][]const u8{ "", "0", "1", "3", "9", "04", "+8", "8 " }) |bad| {
        Env.least = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    const FP = ranks.Fingerprint;
    const original = FP.of(4, 0, legacy, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 64), original.layout);
    for ([_][]const u8{ "2", "4", "8" }, [_]u32{ 2, 4, 8 }) |text, least| {
        Env.least = text;
        const settings = try ranks.Settings.read(Env.get, &problem);
        try std.testing.expectEqual(least, settings.kda_min_streams);
        var expected = legacy;
        expected.kda_segments = true;
        expected.kda_min_streams = least;
        try std.testing.expect(std.meta.eql(expected, settings));
        const other = FP.of(4, 1, settings, 4096, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expectEqual(FP.Field.kda_segments, ranks.mismatch(&.{ original, other }).?.field);
        const agreed = FP.of(4, 0, settings, 4096, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expect(ranks.mismatch(&.{ agreed, other }) == null);
        var different = settings;
        different.kda_min_streams = if (least == 8) 4 else 8;
        const mismatch = FP.of(4, 1, different, 4096, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expectEqual(FP.Field.kda_min_streams, ranks.mismatch(&.{ agreed, mismatch }).?.field);
    }
    try require(legacy, 1, true, false);
    const valid: ranks.Settings = .{ .kda_segments = true, .dense_segments = true };
    try require(valid, 4, false, true);
    try std.testing.expectError(error.KdaBatchScope, require(valid, 2, false, true));
    try std.testing.expectError(error.KdaBatchScope, require(valid, 4, true, true));
    try std.testing.expectError(error.KdaBatchScope, require(valid, 4, false, false));
    inline for (.{ "pages", "verify_graphs", "dense_segments", "kda_wide" }) |field| {
        var invalid = valid;
        @field(invalid, field) = !@field(invalid, field);
        try std.testing.expectError(error.KdaBatchScope, require(invalid, 4, false, true));
    }
}
