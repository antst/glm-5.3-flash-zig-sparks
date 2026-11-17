//! Immutable optional Q4 prompt modules; no module or file work when the agreed setting is off.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_affine_prompt.zig");
pub const image_sha = "20972abb31288a7cc4fb4865c7cece11b1d013fcbf82a0b6ecde49b7c93e2d5e";

pub const Owner = struct {
    original: cuda.Module,
    wide: cuda.Module,
    kernels: prompt.Kernels,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !Owner {
        const path = @import("sys").environment.Process.get(.{}, "TF_GLM_Q4_PROMPT_IMAGE") orelse
            return error.Q4PromptImageRequired;
        const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(32 << 20), .@"16", null);
        defer gpa.free(bytes);
        try checkImage(bytes);
        var original = try cuda.Module.load(d, cuda.kernels.qmm_prefill);
        errdefer original.unload();
        var wide = try cuda.Module.load(d, bytes);
        errdefer wide.unload();
        return .{ .original = original, .wide = wide, .kernels = try prompt.Kernels.load(original, wide) };
    }

    pub fn deinit(owner: *Owner) void {
        owner.wide.unload();
        owner.original.unload();
        owner.* = undefined;
    }
};

pub fn checkImage(bytes: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.Q4PromptImageHash;
}

pub fn install(m: anytype, gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, on: bool) !void {
    if (!on) return;
    if (m.mm.affine == null or m.affine_prompt != null) return error.Q4PromptScope;
    const owner = try Owner.init(gpa, io, d);
    m.mm.affine.?.prompt = owner.kernels;
    m.affine_prompt = owner;
    std.log.info("GLM_Q4_PYTHON_PROMPT active=true output=bf16/f32 image={s}", .{image_sha});
}

/// Only the affine target's private kernel context changes; the drafter's m.q4 context remains untouched.
pub fn selectSplits(m: anytype, on: bool) !void {
    if (!on) return;
    const affine = if (m.mm.affine) |*a| a else return error.Q4SplitScope;
    affine.q4.python_shapes = true;
    std.log.info("GLM_Q4_PYTHON_SPLITS active=true schedule=75c target_only=true", .{});
}

/// Prompt state produced by different arithmetic must not share an NVMe compatibility domain.
pub fn compatibility(base: [32]u8, on: bool) [32]u8 {
    if (!on) return base;
    var digest: [32]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&base);
    hash.update("glm-q4-python-prompt-v1");
    hash.final(&digest);
    return digest;
}

pub fn compatibilityWithSplits(base: [32]u8, prompt_on: bool, split_on: bool) [32]u8 {
    const previous = compatibility(base, prompt_on);
    if (!split_on) return previous;
    var digest: [32]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&previous);
    hash.update("glm-q4-python-splits-v1");
    hash.final(&digest);
    return digest;
}

test "affine target: prompt owner OFF is inert and NVMe modes cannot share compatibility" {
    var m: @import("cuda_forward.zig").Modules = undefined;
    try install(&m, undefined, undefined, undefined, false);
    const base: [32]u8 = @splat(9);
    try std.testing.expectEqualSlices(u8, &base, &compatibility(base, false));
    try std.testing.expect(!std.mem.eql(u8, &base, &compatibility(base, true)));
    try std.testing.expectEqualSlices(u8, &compatibility(base, true), &compatibility(base, true));
    try std.testing.expectError(error.Q4PromptImageHash, checkImage(""));
    try std.testing.expectError(error.Q4PromptImageHash, checkImage("not the pinned image"));
}

test "affine target: prompt setting is strict cached scope checked and agreed before weights load" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q4_PYTHON_PROMPT")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "true", "01", "+1", " 1" }) |bad| {
        Env.value = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    var on = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(on.q4_python_prompt);
    try std.testing.expectError(error.Q4PromptScope, @import("cuda_tile_affine.zig").require(on, 4, false));
    on.q4_affine_mse = true;
    on.dense_bits = 4;
    try @import("cuda_tile_affine.zig").require(on, 4, false);
    var legacy = on;
    legacy.q4_python_prompt = false;
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, legacy, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 65), a.layout);
    try std.testing.expectEqual(FP.Field.q4_python_prompt, ranks.mismatch(&.{ a, b }).?.field);
    try std.testing.expectEqualStrings("TF_GLM_Q4_PYTHON_PROMPT", @import("cuda_rank_names.zig").explain(FP.Field.q4_python_prompt));
}

test "affine target: split schedule changes only target context and separates all four NVMe modes" {
    var inert: @import("cuda_forward.zig").Modules = undefined;
    try selectSplits(&inert, false);
    const q4 = @import("cuda_q4.zig");
    const Fake = struct { q4: q4.Kernels, mm: @import("cuda_mm.zig").Kernels };
    var m: Fake = undefined;
    m.q4 = .{ .group = undefined, .pack_dense = undefined, .transpose = undefined, .gb10 = false };
    m.mm.affine = .{ .q4 = m.q4, .sums = undefined };
    try selectSplits(&m, true);
    try std.testing.expect(m.mm.affine.?.q4.python_shapes);
    try std.testing.expect(!m.q4.python_shapes);
    const base: [32]u8 = @splat(9);
    var modes: [4][32]u8 = undefined;
    for (0..4) |i| modes[i] = compatibilityWithSplits(base, i & 1 != 0, i & 2 != 0);
    try std.testing.expectEqualSlices(u8, &base, &modes[0]);
    try std.testing.expectEqualSlices(u8, &compatibility(base, true), &modes[1]);
    for (modes, 0..) |mode, i| for (modes[0..i]) |previous|
        try std.testing.expect(!std.mem.eql(u8, &mode, &previous));
}

test "affine target: split setting strict cached scope and rank agreement" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_Q4_PYTHON_SPLITS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "true", "01", "+1", " 1" }) |bad| {
        Env.value = bad;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    var on = try ranks.Settings.read(Env.get, &problem);
    Env.value = "0";
    try std.testing.expect(on.q4_python_splits);
    try std.testing.expectError(error.Q4SplitScope, @import("cuda_tile_affine.zig").require(on, 4, false));
    on.q4_affine_mse = true;
    on.dense_bits = 4;
    try @import("cuda_tile_affine.zig").require(on, 4, false);
    var legacy = on;
    legacy.q4_python_splits = false;
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, legacy, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 65), a.layout);
    try std.testing.expectEqual(FP.Field.q4_python_splits, ranks.mismatch(&.{ a, b }).?.field);
    try std.testing.expectEqualStrings("TF_GLM_Q4_PYTHON_SPLITS", @import("cuda_rank_names.zig").explain(FP.Field.q4_python_splits));
}
