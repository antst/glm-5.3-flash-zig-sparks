//! Real field enumeration and packed-format lookup; the values intentionally alias in the role tests.
const std = @import("std");
const cuda = @import("cuda");
const f = @import("cuda_q4_precision_families.zig");
const weights = @import("cuda_weights.zig");
const tile = @import("cuda_tile.zig");
const recurrent = @import("cuda_q4_recurrent.zig");
const ranks = @import("cuda_ranks.zig");
const mm = @import("cuda_mm.zig");
const t = std.testing;

fn fill(w: *weights.Weights) void {
    const c = w.config;
    for (w.layers, 0..) |*l, i| {
        switch (c.kind(@intCast(i))) {
            .kda => l.mixer = .{ .kda = .{ .proj = .{ .bf16 = 17 }, .fb = 17, .gb = 17, .o = .{ .bf16 = 17 }, .conv_w = 17, .a_log = 17, .dt_bias = 17, .norm_w = 17, .eps = c.eps, .lower = -5, .heads = w.share.kda.len(), .dims = c.hidden } },
            .mla => l.mixer = .{ .mla = .{ .proj = .{ .bf16 = 17 }, .q_norm = 17, .kv_norm = 17, .q_b = .{ .bf16 = 17 }, .wk = 17, .wv = 17, .o = .{ .bf16 = 17 }, .index = .{ .kw = .{ .bf16 = 17 }, .qb = .{ .bf16 = 17 }, .gate = 17, .ln_w = 17, .ln_b = 17, .ape = 17, .heads = c.i_heads, .d = c.i_dim }, .heads = w.share.mla.len(), .qk = c.nope, .dv = c.v_dim, .lw = c.kv_lora, .q_lora = c.q_lora, .dims = c.hidden, .eps = c.eps, .dense_limit = c.keyWidth() } },
        }
        if (c.isMoe(@intCast(i))) {
            l.ffn = .{ .moe = undefined };
            l.ffn.moe.shared_gu = .{ .bf16 = 17 };
            l.ffn.moe.shared_down = .{ .bf16 = 17 };
            l.ffn.moe.shared = w.share.shared.len();
            l.ffn.moe.routed.dims = c.hidden;
        } else l.ffn = .{ .dense = .{ .w = .{ .bf16 = .{ .gu = 17, .down = 17 } }, .width = w.share.dense.len(), .dims = c.hidden, .limit = 0 } };
    }
}
fn make(layers: []weights.Layer) !weights.Weights {
    const config: @import("config.zig").Config = .{ .exl3 = true, .head_unit = 128 };
    var w: weights.Weights = .{ .gpa = t.allocator, .config = config, .share = try @import("cuda_shard.zig").Share.of(&config, 0, 2), .layers = layers, .head = .{ .bf16 = 17 } };
    fill(&w);
    return w;
}

test "Q4 precision families: strict startup mask default scope cached value and rank agreement" {
    const E = struct {
        var value: ?[]const u8 = null;
        fn get(n: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, n, "TF_GLM_Q4_BF16_FAMILIES")) value else null;
        }
    };
    defer E.value = null;
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(E.get, &problem);
    try t.expectEqual(@as(u8, 0), off.q4_bf16_families);
    E.value = "0";
    try t.expectEqualDeep(off, try ranks.Settings.read(E.get, &problem));
    for ([_][]const u8{ "8", "01", "0x1", "-1", "true", " 1" }) |bad| {
        E.value = bad;
        try t.expectError(error.BadSetting, ranks.Settings.read(E.get, &problem));
    }
    for (0..8) |mask| {
        const chars = [_]u8{@as(u8, '0') + @as(u8, @intCast(mask))};
        E.value = &chars;
        var on = try ranks.Settings.read(E.get, &problem);
        E.value = "0";
        try t.expectEqual(@as(u8, @intCast(mask)), on.q4_bf16_families);
        if (mask != 0) try t.expectError(error.Q4Bf16FamiliesScope, f.require(on, 2));
        on.q4_affine_mse = true;
        on.q4_activation = true;
        on.dense_bits = 4;
        try f.require(on, 2);
        if (mask != 0) {
            try t.expectError(error.Q4Bf16FamiliesScope, f.require(on, 4));
            var before = on;
            before.q4_bf16_families = 0;
            const a = ranks.Fingerprint.of(2, 0, before, 8192, true, @splat(0), @splat(0), @splat(0));
            const b = ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
            try t.expectEqual(@as(u32, 85), b.layout);
            try t.expectEqual(ranks.Fingerprint.Field.q4_bf16_families, ranks.mismatch(&.{ a, b }).?.field);
        }
    }
}

test "Q4 precision families: actual45 layer linears form complete architectural partition for every mask" {
    var layers: [45]weights.Layer = undefined;
    var w = try make(&layers);
    var list: std.ArrayList(tile.Lin) = .empty;
    defer list.deinit(t.allocator);
    try tile.linears(&w, &list);
    try t.expectEqual(@as(usize, 282), list.items.len);
    const stranger: u64 = 17;
    try t.expectEqual(@as(?f.Family, null), f.which(&w, &stranger));
    try t.expectEqual(@as(?f.Family, null), f.which(&w, &w.head.bf16));
    const expected_counts = [_]usize{ 55, 34, 90 };
    const expected_bytes = [_]usize{ 1352400896, 1140850688, 1509949440 };
    const expected_extra = [_]usize{ 969605120, 819986432, 1085276160 };
    const expected_groups = [_]usize{ 10565632, 8912896, 11796480 };
    for (0..8) |mask| {
        var census: f.Census = .{};
        for (list.items) |l| if (f.selected(&w, l.field, @intCast(mask))) |kind| census.note(kind, l);
        for (0..3) |i| {
            const on = mask & (@as(usize, 1) << @as(u6, @intCast(i))) != 0;
            try t.expectEqual(if (on) expected_counts[i] else 0, census.matrices[i]);
            try t.expectEqual(if (on) expected_bytes[i] else 0, census.bytes[i]);
            try t.expectEqual(if (on) expected_extra[i] else 0, census.extra[i]);
            try t.expectEqual(if (on) expected_groups[i] else 0, census.groups[i]);
        }
        for (0..4) |mode| {
            var selected: usize = 0;
            for (list.items) |l| {
                const a = recurrent.isRetained(&w, l.field, @intCast(mode));
                const b = f.selected(&w, l.field, @intCast(mask)) != null;
                try t.expect(!(a and b));
                selected += @intFromBool(a or b);
            }
            const rec = ([_]usize{ 0, 68, 68, 102 })[mode];
            const fam = census.matrices[0] + census.matrices[1] + census.matrices[2];
            try t.expectEqual(rec + fam, selected);
            if (mode == 3 and mask == 7) try t.expectEqual(@as(usize, 281), selected);
        }
    }
    for (w.layers) |*l| switch (l.mixer) {
        .kda => |*k| for ([_]*const u64{ &k.conv_w, &k.a_log, &k.norm_w }) |p| try t.expectEqual(@as(?f.Family, null), f.which(&w, p)),
        .mla => |*m| for ([_]*const u64{ &m.wk, &m.wv, &m.index.gate, &m.index.ln_w }) |p| try t.expectEqual(@as(?f.Family, null), f.which(&w, p)),
    };
}

test "Q4 precision families: actual packed lookup remains BF16 only for declared roles and head" {
    var layers: [45]weights.Layer = undefined;
    var w = try make(&layers);
    var list: std.ArrayList(tile.Lin) = .empty;
    defer list.deinit(t.allocator);
    try tile.linears(&w, &list);
    for (list.items, 0..) |l, i| l.field.* = 0x100000 + @as(u64, @intCast(i)) * 0x10000000;
    var ptrs: std.ArrayList(u64) = .empty;
    defer ptrs.deinit(t.allocator);
    const bits: [282]u8 = @splat(4);
    for (0..8) |mask| for (0..4) |mode| {
        ptrs.clearRetainingCapacity();
        for (list.items) |l| if (l.field != &w.head.bf16 and !recurrent.isRetained(&w, l.field, @intCast(mode)) and f.selected(&w, l.field, @intCast(mask)) == null)
            try ptrs.append(t.allocator, l.field.*);
        const packed_set: mm.Packed = .{ .ptrs = ptrs.items, .bits = bits[0..ptrs.items.len], .affine = true };
        for (list.items) |l| {
            const retained = l.field == &w.head.bf16 or recurrent.isRetained(&w, l.field, @intCast(mode)) or f.selected(&w, l.field, @intCast(mask)) != null;
            const got = packed_set.tiled(l.field.*, l.n, l.k);
            try t.expectEqual(@as(u8, if (retained) 16 else 4), got.bits);
            try t.expectEqual(retained, got.bias == 0);
        }
    };
}

test "Q4 precision families: actual retained MLA KDAout and FFN tile copy drain payloads" {
    const F = struct {
        const cases = [_][4]usize{ .{ 2048, 4096, 2048, 16777216 }, .{ 4096, 4096, 4096, 33554432 }, .{ 4096, 1024, 1024, 8388608 } };
        var at: usize = 0;
        var step: usize = 0;
        fn launch(_: cuda.abi.Function, gx: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            t.expectEqual(@as(usize, 0), step) catch @panic("family tile order");
            const p = params.?;
            t.expectEqual(@as(u32, @intCast(cases[at][2])), gx) catch @panic("family grid");
            t.expectEqual(@as(i32, @intCast(cases[at][0])), @as(*align(1) const i32, @ptrCast(p[2].?)).*) catch @panic("family N");
            t.expectEqual(@as(i32, @intCast(cases[at][1])), @as(*align(1) const i32, @ptrCast(p[3].?)).*) catch @panic("family K");
            step += 1;
            return 0;
        }
        fn copy(_: u64, _: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            t.expectEqual(@as(usize, 1), step) catch @panic("family copy order");
            t.expectEqual(cases[at][3], n) catch @panic("family bytes");
            step += 1;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            t.expectEqual(@as(usize, 2), step) catch @panic("family drain order");
            step += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = F.launch;
    d.api.cuMemcpyDtoDAsync_v2 = F.copy;
    d.api.cuStreamSynchronize = F.sync;
    var ks: mm.Kernels = undefined;
    ks.tile = .{ .d = &d, .handle = @ptrFromInt(16) };
    var ptr: u64 = 256;
    for (F.cases, 0..) |c, i| {
        F.at = i;
        F.step = 0;
        try recurrent.retain(&ks, .{ .d = &d, .handle = @ptrFromInt(32) }, &d, .{ .field = &ptr, .n = c[0], .k = c[1] }, 2048);
        try t.expectEqual(@as(usize, 3), F.step);
    }
}
