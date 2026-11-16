//! Optional Q4 keeps the corrected proposer and existing optimized startup modes.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const reference = @import("dflash_reference.zig");
const warm = @import("cuda_ffn_warm.zig");
const admission = @import("cuda_admission_options.zig");
const t = std.testing;
const Options = struct { settings: ranks.Settings, drafter: bool = true, mtp: bool = true, vision: ?u8 = null };

fn settings(bits: u8) ranks.Settings {
    return .{ .dense_bits = bits, .head_bits = 16, .tiled = true, .verify_rows = 16, .ffn_warm = true, .ffn_graphs = true, .exl3_fused_decode = true, .admission_us = 4000, .dflash_reference = true, .dflash_attn_tp4 = true, .dflash_mlp_wide = true, .dflash_confidence_only = true };
}

test "optional Q4 optimized: corrected drafter admits explicit target Q4 without changing the draft mode" {
    try reference.scope(settings(4), 4, true, 4);
    try reference.scope(settings(16), 4, true, 4);
}

test "optional Q4 optimized: capture warmup and bounded admission remain available with Q4 head16" {
    for ([_]u8{ 16, 4 }) |bits| {
        const o: Options = .{ .settings = settings(bits) };
        try warm.require(o, 4);
        try admission.require(o, 4);
    }
}

test "optional Q4 optimized: startup defaults and cached precision disagreement retain existing fields" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DENSE_BITS")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expectEqual(@as(u8, 16), off.dense_bits);
    try t.expectEqual(@as(u8, 16), off.head_bits);
    try t.expect(!off.ffn_graphs and !off.ffn_warm and off.admission_us == 0);
    Env.value = "4";
    const on = try ranks.Settings.read(Env.get, &problem);
    Env.value = "16";
    try t.expectEqual(@as(u8, 4), on.dense_bits);
    const zero: [32]u8 = @splat(0);
    const a = ranks.Fingerprint.of(4, 0, off, 4096, true, zero, zero, zero);
    const b = ranks.Fingerprint.of(4, 1, on, 4096, true, zero, zero, zero);
    try t.expectEqual(ranks.Fingerprint.Field.bits, ranks.mismatch(&.{ a, b }).?.field);
    Env.value = "3";
    try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
}

test "optional Q4 optimized: unsupported quantized combinations refuse the actual startup scopes" {
    for ([_]u8{ 6, 8 }) |bits| {
        const s = settings(bits);
        try t.expectError(error.DrafterReferenceScope, reference.scope(s, 4, true, 4));
        try t.expectError(error.FfnWarmScope, warm.require(Options{ .settings = s }, 4));
        try t.expectError(error.AdmissionWaitScope, admission.require(Options{ .settings = s }, 4));
    }
    var s = settings(4);
    s.head_bits = 8;
    try t.expectError(error.DrafterReferenceScope, reference.scope(s, 4, true, 4));
    try t.expectError(error.FfnWarmScope, warm.require(Options{ .settings = s }, 4));
    try t.expectError(error.AdmissionWaitScope, admission.require(Options{ .settings = s }, 4));
    s = settings(4);
    s.pages = true;
    try t.expectError(error.FfnWarmScope, warm.require(Options{ .settings = s }, 4));
    try t.expectError(error.AdmissionWaitScope, admission.require(Options{ .settings = s }, 4));
}
