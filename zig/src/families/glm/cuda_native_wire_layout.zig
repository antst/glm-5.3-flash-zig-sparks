//! Fixed borrowed-wire shapes: two ranks on direct f1, four ranks on switched f0.
const std = @import("std");

pub fn supported(world: usize) bool {
    return world == 2 or world == 4;
}

pub const Layout = struct {
    world: usize,
    pub const slots = 4;

    pub fn init(world: usize) !Layout {
        if (!supported(world)) return error.NativeWireWorld;
        return .{ .world = world };
    }

    pub fn createFlags(l: Layout) u32 {
        return if (l.world == 2) 5 else 1; // PDL bit0 and the pinned factory's physical f1 selector bit2.
    }

    pub fn rails(l: Layout) []const u8 {
        return if (l.world == 2) "f1,f1-twin" else "f0,f0-twin";
    }

    pub fn doorbell(l: Layout) usize {
        return slots * l.world;
    }

    pub fn sent(l: Layout) usize {
        return l.doorbell() + 1 + slots;
    }

    pub fn abortWord(l: Layout) usize {
        return l.sent() + slots;
    }

    pub fn metadataBytes(l: Layout) usize {
        return (l.abortWord() + 1) * @sizeOf(u64);
    }
};

test "TP2 direct: pinned factory flags and native metadata do not retain world4 offsets" {
    const two = try Layout.init(2);
    const four = try Layout.init(4);
    try std.testing.expectEqual(@as(u32, 5), two.createFlags());
    try std.testing.expectEqual(@as(u32, 1), four.createFlags());
    try std.testing.expectEqualStrings("f1,f1-twin", two.rails());
    try std.testing.expectEqualStrings("f0,f0-twin", four.rails());
    try std.testing.expectEqual(@as(usize, 8), two.doorbell());
    try std.testing.expectEqual(@as(usize, 13), two.sent());
    try std.testing.expectEqual(@as(usize, 17), two.abortWord());
    try std.testing.expectEqual(@as(usize, 144), two.metadataBytes());
    try std.testing.expectEqual(@as(usize, 16), four.doorbell());
    try std.testing.expectEqual(@as(usize, 21), four.sent());
    try std.testing.expectEqual(@as(usize, 25), four.abortWord());
    try std.testing.expectEqual(@as(usize, 208), four.metadataBytes());
    for ([_]usize{ 0, 1, 3, 5, 8 }) |n| try std.testing.expectError(error.NativeWireWorld, Layout.init(n));
    for ([_]Layout{ two, four }) |l| {
        var occupied: [26]bool = @splat(false);
        for (0..l.world) |rank| for (0..Layout.slots) |slot| {
            const i = slot * l.world + rank;
            try std.testing.expect(!occupied[i]);
            occupied[i] = true;
        };
        for (0..l.doorbell()) |i| try std.testing.expect(occupied[i]);
        try std.testing.expect(!occupied[l.doorbell()]);
    }
}

test "TP2 direct: actual optimized startup scopes admit world2 and keep TP4-only shapes refused" {
    const ranks = @import("cuda_ranks.zig");
    const width = @import("cuda_verify_rows.zig");
    const reference = @import("dflash_reference.zig");
    const graph = @import("dflash_native_graph.zig");
    const head = @import("cuda_head_fp8.zig");
    const t = std.testing;
    var s: ranks.Settings = .{ .verify_rows = 32, .first_copy_tokens = 31, .dflash_reference = true, .dflash_mlp_wide = true, .dflash_confidence_only = true, .ffn_graphs = true, .ffn_warm = true, .exl3_fused_decode = true, .admission_us = 4000 };
    const Opt = struct { settings: ranks.Settings, mtp: bool = true, drafter: bool = true, vision: ?u8 = null };
    for ([_]u8{ 4, 8, 16 }) |bits| {
        s.dense_bits = bits;
        try width.require(s, 2, false, true, true);
        try reference.scope(s, 2, true, 4);
        try @import("cuda_ffn_warm.zig").require(Opt{ .settings = s }, 2);
        try @import("cuda_admission_options.zig").require(Opt{ .settings = s }, 2);
        try t.expectError(error.VerifyRowsScope, width.require(s, 3, false, true, true));
    }
    s.dense_bits = 4;
    s.head_bits = 8;
    s.q4_head_fp8 = true;
    s.q4_affine_mse = true;
    try @import("cuda_tile_affine.zig").require(s, 2, false);
    try reference.scope(s, 2, true, 4);
    const q = try head.Shape.make(77440, 4096);
    try t.expectEqual(@as(usize, 77440 * 4096), q.codes);
    try graph.scope(2, true, false, true);
    try t.expectError(error.DFlashNativeGraphScope, graph.scope(2, true, true, true));
    s.dflash_attn_tp4 = true;
    try t.expectError(error.DrafterReferenceScope, reference.scope(s, 2, true, 4));
    try t.expectError(error.DrafterAttentionNeedsTP4, @import("dflash_attention.zig").Plan.init(0, 2, true));
    const full = try @import("dflash_attention.zig").Plan.init(1, 2, false);
    try t.expectEqual(@as(usize, 6144), full.fusedWidth());
}

test "TP2 direct: dimension-generic EXL3 selectors admit only the two qualified local widths" {
    const common = @import("cuda").exl3_experts;
    const t = std.testing;
    var e: common.Experts = undefined;
    e.decode = null;
    var l = std.mem.zeroes(common.Layer);
    l.dims = 4096;
    l.codebook = .mcg;
    l.uniform_k2 = 8;
    l.width = 1024;
    try t.expect(!e.hasCompactDecode(l, 1));
    e.decode = .{ .module = undefined, .gateup_shared = undefined, .down = undefined };
    for ([_]usize{ 512, 1024 }) |width| {
        l.width = width;
        for (1..65) |rows| try t.expect(e.hasCompactDecode(l, rows));
        try t.expectEqual(@as(u32, 4), try common.plan(4096, width, true));
        try t.expectEqual(@as(u32, 1), try common.plan(width, 4096, false));
    }
    for ([_]usize{ 128, 256, 768, 1536, 2048 }) |width| {
        l.width = width;
        try t.expect(!e.hasCompactDecode(l, 1));
    }
    l.width = 1024;
    try t.expect(!e.hasCompactDecode(l, 0));
    try t.expect(!e.hasCompactDecode(l, 65));
    l.uniform_k2 = 7;
    try t.expect(!e.hasCompactDecode(l, 1));
}
