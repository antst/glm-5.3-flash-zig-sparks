//! TP4 draft-block heads; full replicated context ingest and NVMe ring rows retain their current layout.
const std = @import("std");
const cuda = @import("cuda");

pub const Plan = struct {
    rank: usize,
    q_heads: usize,
    kv_heads: usize,
    q0: usize,
    kv0: usize,

    pub fn init(rank: usize, world: usize, want: bool) !Plan {
        if (world == 0 or rank >= world) return error.BadDrafterAttentionRank;
        if (want and world != 4) return error.DrafterAttentionNeedsTP4;
        return .{
            .rank = rank,
            .q_heads = if (want) 8 else 32,
            .kv_heads = if (want) 2 else 8,
            .q0 = if (want) rank * 1024 else 0,
            .kv0 = if (want) rank * 256 else 0,
        };
    }

    pub fn qWidth(p: Plan) usize {
        return p.q_heads * 128;
    }
    pub fn kvWidth(p: Plan) usize {
        return p.kv_heads * 128;
    }
    pub fn fusedWidth(p: Plan) usize {
        return p.qWidth() + 2 * p.kvWidth();
    }

    pub fn slices(p: Plan) [3][2]usize {
        return .{ .{ p.q0, p.qWidth() }, .{ p.kv0, p.kvWidth() }, .{ p.kv0, p.kvWidth() } };
    }
};

/// These symbols must come from a newly built image; old assets fail lookup rather than running another geometry.
pub const Kernels = struct {
    block_qkv: cuda.Function,
    attn_part: cuda.Function,
    attn_merge: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        return .{
            .block_qkv = try m.function("df_block_qkv_tp4_v1"),
            .attn_part = try m.function("df_attn_part_tp4_v1"),
            .attn_merge = try m.function("df_attn_merge_tp4_v1"),
        };
    }
};

pub const Local = struct { plan: Plan, kernels: Kernels };

test "DFlash attention: TP4 column slices partition Q and KV and keep the original ring stride" {
    var q: [4096]u8 = @splat(0);
    var kv: [1024]u8 = @splat(0);
    for (0..4) |rank| {
        const p = try Plan.init(rank, 4, true);
        try std.testing.expectEqual(@as(usize, 1536), p.fusedWidth());
        for (q[p.q0..][0..p.qWidth()]) |*x| x.* += 1;
        for (kv[p.kv0..][0..p.kvWidth()]) |*x| x.* += 1;
        for (0..2) |kh| for (0..128) |lane| {
            const global = (rank * 2 + kh) * 128 + lane;
            try std.testing.expectEqual(p.kv0 + kh * 128 + lane, global);
            for ([_]usize{ 0, 1, 2047 }) |slot|
                try std.testing.expect(slot * 1024 + global < 2048 * 1024);
        };
    }
    for (q) |x| try std.testing.expectEqual(@as(u8, 1), x);
    for (kv) |x| try std.testing.expectEqual(@as(u8, 1), x);
    try std.testing.expectEqual(@as(usize, 6144), (try Plan.init(0, 4, false)).fusedWidth());
    try std.testing.expectError(error.DrafterAttentionNeedsTP4, Plan.init(0, 2, true));
    try std.testing.expectError(error.BadDrafterAttentionRank, Plan.init(4, 4, true));
}

test "DFlash attention: strict opt-in requires MLP splitting and participates in agreement" {
    const ranks = @import("cuda_ranks.zig");
    const Env = struct {
        var value: ?[]const u8 = null;
        var mlp = false;
        fn get(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_DFLASH_ATTN_TP4")) return value;
            if (std.mem.eql(u8, key, "TF_GLM_DFLASH_MLP_TP4") and mlp) return "1";
            return null;
        }
    };
    defer {
        Env.value = null;
        Env.mlp = false;
    }
    var problem: []const u8 = "";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_attn_tp4);
    Env.value = "0";
    try std.testing.expect(!(try ranks.Settings.read(Env.get, &problem)).dflash_attn_tp4);
    Env.value = "yes";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_ATTN_TP4: 0 or 1", problem);
    Env.value = "1";
    try std.testing.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_ATTN_TP4=1 requires TF_GLM_DFLASH_MLP_TP4=1", problem);
    Env.mlp = true;
    const on = try ranks.Settings.read(Env.get, &problem);
    try std.testing.expect(on.dflash_attn_tp4);
    const a = ranks.Fingerprint.of(4, 0, .{ .dflash_mlp_wide = true }, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(ranks.Fingerprint.Field.dflash_attn_tp4, ranks.mismatch(&.{ a, b }).?.field);
}
