//! Opt-in historical drafter arithmetic with explicit target BF16, Q8/head16 or Q4/head16 precision.
const std = @import("std");
const cuda = @import("cuda");
const df = @import("dflash.zig");

pub fn scope(s: anytype, world: usize, has_drafter: bool, bits: u8) !void {
    if (!s.dflash_reference) return;
    const heads_ok = (world == 4 and s.dflash_attn_tp4) or (world == 2 and !s.dflash_attn_tp4);
    const pages_ok = !s.pages or (world == 2 and s.kv8 and s.dflash_only_head);
    if (!heads_ok or !has_drafter or bits != 4 or !s.dflash_mlp_wide or
        !s.dflash_confidence_only or !pages_ok or s.kv_host != 0 or s.verify_graphs or
        !@import("cuda_target_precision.zig").supported(s))
        return error.DrafterReferenceScope;
}

fn image(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, want: []const u8) !cuda.Module {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const actual = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &actual, want)) return error.DrafterReferenceImageDigest;
    const aligned = try gpa.alignedAlloc(u8, .@"8", bytes.len);
    defer gpa.free(aligned);
    @memcpy(aligned, bytes);
    return cuda.Module.load(d, aligned);
}

pub fn configure(gpa: std.mem.Allocator, io: std.Io, e: anytype, dr: *df.Drafter) !void {
    if (!e.settings.dflash_reference) return;
    try scope(e.settings, dr.world, true, 4);
    if (dr.compat != 7 or (dr.world == 4 and dr.attention == null) or !dr.split) return error.DrafterReferenceScope;
    var module = try image(gpa, io, e.d, "/opt/glm/dflash_reference.fatbin", "3ffdb516c366c10dced9e36a7abee27bc5484a2a690d992ff9ce36570c7fa9e0");
    errdefer module.unload();
    const kernels = try df.Kernels.load(module);
    if (dr.world == 2) {
        try e.stream.synchronize();
        dr.module.unload();
        dr.module = module;
        dr.k = kernels;
        dr.pdl = false;
        std.log.info("GLM_DFLASH_REFERENCE rank={d} active=true world=2 attention=replicated confidence_beta=2", .{dr.rank});
        return;
    }
    const attention = try @import("dflash_attention.zig").Kernels.load(module);
    var tensor_core = try image(gpa, io, e.d, "/opt/glm/dflash_reference_attention.cubin", "b33690449f594947004f4fd11b2bb08825b56f9a00d779a4d3c8f167cdfc2098");
    errdefer tensor_core.unload();
    const local = try df.PythonAttention.load(tensor_core, module);
    try e.stream.synchronize();
    dr.module.unload();
    dr.module = module;
    dr.k = kernels;
    dr.pdl = false;
    dr.attention.?.kernels = attention;
    dr.python_attention = local;
    if (e.settings.dense_bits == 16) {
        std.log.info("GLM_DFLASH_REFERENCE rank={d} active=true target_bf16=true confidence_beta=2", .{dr.rank});
    } else {
        std.log.info("GLM_DFLASH_REFERENCE rank={d} active=true target_bf16=false confidence_beta=2 target_dense_bits={d} target_head_bits={d}", .{ dr.rank, e.settings.dense_bits, e.settings.head_bits });
    }
}

pub fn eligible(starts: []const i64, froms: []const i64) bool {
    if (starts.len == 0 or starts.len > 8 or starts.len != froms.len) return false;
    for (starts, froms) |start, from| if (start < 0 or from > @max(0, start - 2047)) return false;
    return true;
}

/// Partial/resumed drafter context keeps the existing eager attention route and never replays a full-context graph.
pub fn pass(dr: *df.Drafter, e: anytype, streams: usize, rank0: bool) !void {
    const block = @import("dflash_block.zig");
    if (dr.python_attention != null and !eligible(dr.h_starts[0..streams], dr.h_froms[0..streams])) {
        const saved = dr.python_attention;
        dr.python_attention = null;
        defer dr.python_attention = saved;
        return block.pass(dr, e, streams, rank0);
    }
    if (dr.graphs) |g| try @import("dflash_graph.zig").pass(g, dr, e, streams, rank0) else try block.pass(dr, e, streams, rank0);
}

test "drafter reference: full-context eligibility keeps resumed partial context on eager fallback" {
    try std.testing.expect(eligible(&.{ 0, 4096 }, &.{ 0, 2049 }));
    try std.testing.expect(!eligible(&.{4096}, &.{2050}));
    try std.testing.expect(!eligible(&.{33}, &.{1}));
    try std.testing.expect(!eligible(&.{-1}, &.{0}));
    try std.testing.expect(!eligible(&.{}, &.{}));
}

test "drafter reference: strict default, fingerprint disagreement and initial scope" {
    const ranks = @import("cuda_ranks.zig");
    const Get = struct {
        fn off(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(n: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, n, "TF_GLM_DFLASH_REFERENCE")) "1" else null;
        }
        fn bad(n: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, n, "TF_GLM_DFLASH_REFERENCE")) "true" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(&Get.off, &problem)).dflash_reference);
    try std.testing.expect((try ranks.Settings.read(&Get.on, &problem)).dflash_reference);
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(&Get.bad, &problem));
    const h: [32]u8 = @splat(0);
    var all = [_]ranks.Fingerprint{
        ranks.Fingerprint.of(2, 0, .{}, 4096, true, h, h, h),
        ranks.Fingerprint.of(2, 1, .{}, 4096, true, h, h, h),
    };
    all[1].dflash_reference = 1;
    try std.testing.expectEqual(ranks.Fingerprint.Field.dflash_reference, ranks.mismatch(&all).?.field);
    var s: ranks.Settings = .{ .dflash_reference = true, .dflash_attn_tp4 = true, .dflash_mlp_wide = true, .dflash_confidence_only = true };
    try scope(s, 4, true, 4);
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 2, true, 4));
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 4, false, 4));
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 4, true, 0));
    s.pages = true;
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 4, true, 4));
    s.pages = false;
    s.dense_bits = 4;
    try scope(s, 4, true, 4);
    s.dense_bits = 8;
    try scope(s, 4, true, 4);
    s.dense_bits = 6;
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 4, true, 4));
    s.dflash_attn_tp4 = false;
    s.dense_bits = 8;
    s.pages = true;
    s.kv8 = true;
    try std.testing.expectError(error.DrafterReferenceScope, scope(s, 2, true, 4));
    s.dflash_only_head = true;
    try scope(s, 2, true, 4);
}
