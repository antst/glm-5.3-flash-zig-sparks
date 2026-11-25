const std = @import("std");
const t = std.testing;
const opt = @import("cuda_nvfp4_graph.zig");
const ranks = @import("cuda_ranks.zig");
const graph = @import("cuda_ffn_graph.zig");
const warm = @import("cuda_ffn_warm.zig");

test "FFN graphs: NVFP4 strict default-off and rank mismatch" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_NVFP4_FFN_GRAPHS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const old = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!old.nvfp4_ffn_graphs and !opt.wide(old));
    Env.value = "0";
    try t.expectEqualDeep(old, try ranks.Settings.read(Env.get, &problem));
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(on.nvfp4_ffn_graphs and opt.wide(on));
    const hash: [32]u8 = @splat(0);
    const a = ranks.Fingerprint.of(4, 0, old, 4096, true, hash, hash, hash);
    const b = ranks.Fingerprint.of(4, 1, on, 4096, true, hash, hash, hash);
    try t.expectEqual(ranks.Fingerprint.Field.nvfp4_ffn_graphs, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_NVFP4_FFN_GRAPHS", @import("cuda_rank_names.zig").explain(ranks.Fingerprint.Field.nvfp4_ffn_graphs));
    for ([_][]const u8{ "true", "2", "1 " }) |value| {
        Env.value = value;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
}

test "FFN graphs: NVFP4 startup refuses incompatible configuration before allocation" {
    const Engine = @import("cuda_engine.zig").Engine;
    const Options = @import("cuda_engine.zig").Options;
    var o: Options = .{ .capacity = 4096, .mtp = true, .drafter = true, .settings = .{ .nvfp4_ffn_graphs = true, .ffn_graphs = true, .verify_rows = 16 } };
    try opt.require(o, 4);
    o.settings.verify_rows = 32;
    try opt.require(o, 4);
    o.settings.verify_rows = 16;
    try t.expectError(error.Nvfp4GraphScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 3, o));
    inline for (.{ "ffn_graphs", "tiled" }) |field| {
        var bad = o;
        @field(bad.settings, field) = false;
        try t.expectError(error.Nvfp4GraphScope, opt.require(bad, 4));
    }
    inline for (.{ "exl3_fused_decode", "verify_graphs", "pages" }) |field| {
        var bad = o;
        @field(bad.settings, field) = true;
        try t.expectError(error.Nvfp4GraphScope, opt.require(bad, 4));
    }
    var bad = o;
    bad.settings.kv_host = 1;
    try t.expectError(error.Nvfp4GraphScope, opt.require(bad, 4));
    bad = o;
    bad.settings.verify_rows = 8;
    try t.expectError(error.Nvfp4GraphScope, opt.require(bad, 4));
    o.settings.ffn_warm = true;
    for ([_]u8{ 16, 4 }) |bits| {
        o.settings.dense_bits = bits;
        try warm.require(o, 4);
    }
    try t.expectEqual(@as(u32, 4608), graph.capacityFor(opt.wide(o.settings)));
    try t.expect(graph.capacityFor(opt.wide(o.settings)) >= 45 * (32 + 64));
    o.settings.nvfp4_ffn_graphs = false;
    // The settings are legal for EXL3; checkpoint validation separately refuses NVFP4 without the opt-in.
    try warm.require(o, 4);
}

test "FFN graphs: FP8 pages and the tuned EXL3 decode options reach TP2 and TP4 startup" {
    const scope = @import("cuda_engine_scope.zig");
    const Options = @import("cuda_engine_types.zig").Options;
    for ([_]usize{ 2, 4 }) |world| for ([_][2]u8{ .{ 16, 16 }, .{ 8, 16 }, .{ 4, 16 }, .{ 4, 8 } }) |pair| {
        const o: Options = .{ .capacity = 1 << 20, .mtp = true, .drafter = true, .settings = .{
            .pages = true,
            .kv8 = true,
            .verify_rows = 32,
            .first_copy_tokens = 31,
            .dense_bits = pair[0],
            .head_bits = pair[1],
            .q4_affine_mse = pair[0] == 4,
            .q4_head_fp8 = pair[1] == 8,
            .ffn_graphs = true,
            .ffn_warm = true,
            .dense_segments = true,
            .kda_segments = true,
            .retain_sequence_slots = true,
            .prewarm_sequence_slots = true,
            .exl3_fused_decode = true,
            .admission_us = 4000,
        } };
        _ = try scope.validate(o, world);
    };
}

test "FFN graphs: NVFP4 declaration is required and the legacy nonEXL3 refusal remains" {
    const text = "{\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\",\"config_groups\":{\"g\":{\"weights\":{\"num_bits\":4,\"group_size\":16,\"type\":\"float\"}}}}}";
    var s: ranks.Settings = .{ .ffn_graphs = true };
    try t.expectError(error.FfnGraphScope, opt.checkpoint(t.allocator, text, s));
    s.nvfp4_ffn_graphs = true;
    try opt.checkpoint(t.allocator, text, s);
    try t.expectError(error.NotModelOpt, opt.checkpoint(t.allocator, "{}", s));
    try t.expectError(error.UnsupportedQuant, opt.checkpoint(t.allocator, "{\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"FP8\"}}", s));
    s = .{};
    try opt.checkpoint(t.allocator, "{}", s);
}

test {
    _ = @import("cuda_dense_format.zig");
}
