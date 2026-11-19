//! CPU recordings of the actual optional tile route, kernels and rank agreement.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const mode = @import("cuda_nvfp4_mode.zig");
const experts = @import("cuda_experts.zig");
const t = std.testing;

test "GLM NVFP4 fact: strict default-off setting and all-rank agreement" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_NVFP4_FACT")) value else null;
        }
    };
    defer Env.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.nvfp4_fact);
    Env.value = "0";
    try t.expect(std.meta.eql(off, try ranks.Settings.read(Env.get, &problem)));
    for ([_][]const u8{ "", "true", "01", "+1", " 1" }) |bad| {
        Env.value = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    try t.expect(on.nvfp4_fact);
    const a = ranks.Fingerprint.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 84), a.layout);
    try t.expectEqual(ranks.Fingerprint.Field.nvfp4_fact, ranks.mismatch(&.{ a, b }).?.field);
    try t.expectEqualStrings("TF_GLM_NVFP4_FACT", @import("cuda_rank_names.zig").explain(ranks.Fingerprint.Field.nvfp4_fact));
    for ([_]usize{ 1, 2, 4, 8 }) |world| {
        try mode.require(false, true, true, world);
        if (world == 4) try mode.require(true, false, false, world) else try t.expectError(error.Nvfp4FactScope, mode.require(true, false, false, world));
    }
    try t.expectError(error.Nvfp4FactScope, mode.require(true, true, false, 4));
    try t.expectError(error.Nvfp4FactScope, mode.require(true, false, true, 4));
}

const Fake = struct {
    var calls: usize = 0;
    var id: usize = 0;
    var tile: c_int = 0;
    var fail: bool = false;
    fn read(comptime T: type, ptr: ?*anyopaque) T {
        return @as(*align(1) const T, @ptrCast(ptr.?)).*;
    }
    fn launch(f: cuda.abi.Function, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        calls += 1;
        id = @intFromPtr(f);
        if (id == 90) tile = read(c_int, args.?[3]);
        return if (fail) cuda.abi.error_not_ready else 0;
    }
};

fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

test "GLM NVFP4 fact: actual router tile and up/down entries through every served row width" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var k: experts.Kernels = undefined;
    k.b32 = null;
    k.fact = false;
    var count: u64 = 0;
    k.fact_calls = &count;
    k.nvfp4.warps = 4;
    k.nvfp4.resident = @splat(100000);
    for (&k.nvfp4.fns, 0..) |*f, i| f.* = function(&d, i + 1);
    for (&k.nvfp4.fact_fns, 0..) |*f, i| f.* = function(&d, i + 20);
    for (&k.nvfp4.prompt_fns, 0..) |*f, i| f.* = function(&d, i + 30);
    k.router = .{ .plan_small = function(&d, 90), .plan_rows = undefined, .plan_rank = undefined, .plan_offsets = undefined, .plan_scatter = undefined, .plan_rank_rows = undefined, .plan_scatter_rows = undefined };
    const l: cuda.experts.Layer = .{ .format = .nvfp4, .up = 100, .down = 200, .up_scale = 300, .down_scale = 400, .width = 512, .dims = 4096, .experts = 512 };
    var p: experts.Plan = .{ .route = .{ .members = 500, .items = 600, .counts = 700, .rank = 800, .hist = 900 }, .slots = 9, .experts = 513, .prompt = false };
    Fake.calls = 0;
    Fake.fail = false;
    for ([_]usize{ 1, 2, 4, 8, 16, 32, 63, 64 }) |rows| {
        for ([_]bool{ false, true }) |on| {
            p.fact = on;
            try experts.route(&k, s, 1000, rows * p.slots, p);
            try t.expectEqual(@as(c_int, if (on) 64 else 16), Fake.tile);
            try experts.run(&k, s, .gate_up, l, p, 1000, 4096, rows, 2000, 512);
            try t.expectEqual(@as(usize, if (on) 20 else 1), Fake.id);
            try experts.run(&k, s, .down_f32, l, p, 2000, 512, rows, 3000, 512);
            try t.expectEqual(@as(usize, 2), Fake.id);
        }
        p.prompt = true;
        try experts.run(&k, s, .gate_up, l, p, 1000, 4096, rows, 2000, 512);
        try t.expectEqual(@as(usize, 31), Fake.id);
        p.prompt = false;
    }
    try t.expectEqual(@as(u64, 8), count);
    Fake.fail = true;
    defer Fake.fail = false;
    const before = Fake.calls;
    try t.expectError(error.NotReady, experts.run(&k, s, .gate_up, l, p, 1000, 4096, 8, 2000, 512));
    try t.expectEqual(before + 1, Fake.calls);
    try t.expectEqual(@as(u64, 8), count);
}

test "GLM NVFP4 fact: loaded-layer validation publishes mode only after every layer passes" {
    const Layer = struct { ffn: union(enum) { dense: void, moe: struct { exl3: ?u8 = null, routed: cuda.experts.Layer } } };
    const Owner = struct {
        settings: ranks.Settings = .{},
        w: struct { layers: []const Layer },
        m: struct { experts: struct { fact: bool = false, fact_calls: ?*u64 = null } = .{} } = .{},
        expert_fact_calls: u64 = 0,
        share: struct { rank: usize = 0 } = .{},
    };
    const valid: Layer = .{ .ffn = .{ .moe = .{ .routed = .{ .format = .nvfp4, .up = 0, .down = 0, .width = 512, .dims = 4096, .experts = 512 } } } };
    var layers = [_]Layer{ valid, valid };
    var owner: Owner = .{ .w = .{ .layers = &layers } };
    try mode.open(&owner);
    try t.expect(!owner.m.experts.fact and owner.m.experts.fact_calls == null);
    owner.settings.nvfp4_fact = true;
    layers[1].ffn.moe.routed.format = .nvfp4_csf0;
    try t.expectError(error.Nvfp4FactScope, mode.open(&owner));
    try t.expect(!owner.m.experts.fact and owner.m.experts.fact_calls == null);
    layers[1] = valid;
    try mode.open(&owner);
    try t.expect(owner.m.experts.fact);
    try t.expect(owner.m.experts.fact_calls == &owner.expert_fact_calls);
}
