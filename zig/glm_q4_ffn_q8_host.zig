const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const ffn = glm.q4_ffn_q8;
const t = std.testing;

test "Q4 FFN Q8: strict cached choice and all-rank identity" {
    const E = struct {
        var flag: ?[]const u8 = null;
        var digest: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_Q4_FFN_Q8")) return flag;
            if (std.mem.eql(u8, name, "TF_GLM_Q4_FFN_Q8_SHA256")) return digest;
            return null;
        }
    };
    defer {
        E.flag = null;
        E.digest = null;
    }
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(E.get, &problem);
    try t.expect(!off.q4_ffn_q8.on);
    E.digest = "bad";
    try t.expectEqualDeep(off, try glm.ranks.Settings.read(E.get, &problem));
    for ([_][]const u8{ "true", "01", "2", "-1" }) |bad| {
        E.flag = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(E.get, &problem));
    }
    E.flag = "1";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(E.get, &problem));
    E.digest = "9ada369e457fafb9032cdf673d324edbddd23b4e3c0f440f68ea5b760127a065";
    const on = try glm.ranks.Settings.read(E.get, &problem);
    E.flag = "0";
    E.digest = "bad";
    try t.expect(on.q4_ffn_q8.on and on.q4_ffn_q8.sha[0] == 0x9a);
    const a = glm.ranks.Fingerprint.of(2, 0, off, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = glm.ranks.Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 85), b.layout);
    try t.expectEqual(glm.ranks.Fingerprint.Field.q4_ffn_q8, glm.ranks.mismatch(&.{ a, b }).?.field);
    var other = b;
    other.q4_ffn_q8_sha[0] ^= 1;
    var first = b;
    first.rank = 0;
    try t.expectEqual(glm.ranks.Fingerprint.Field.q4_ffn_q8_sha, glm.ranks.mismatch(&.{ first, other }).?.field);
}

test "Q4 FFN Q8: actual startup scopes, cached tiler handoff and disabled zero-loader path" {
    try t.expectEqual(@as(?*ffn.Owner, null), try ffn.Owner.open(t.allocator, undefined, undefined, undefined, .{}, @splat(0), 0));
    var s: glm.ranks.Settings = .{};
    s.q4_ffn_q8 = .{ .on = true, .sha = @splat(0x19) };
    var o = .{ .settings = s, .drafter = true, .mtp = true, .vision = @as(?[]const u8, null) };
    try t.expectError(error.Q4FfnQ8Scope, ffn.require(o, 2));
    o.settings.q4_activation = true;
    o.settings.q4_affine_mse = true;
    o.settings.dense_bits = 4;
    o.settings.head_bits = 16;
    o.settings.tiled = true;
    o.settings.dflash_only_head = true;
    o.settings.q4_recurrent_bf16 = 2;
    try ffn.require(o, 2);
    try t.expectError(error.Q4ActivationScope, ffn.require(o, 4));
    var conflict = o;
    conflict.settings.q4_bf16_families = 4;
    try t.expectError(error.Q4FfnQ8Scope, ffn.require(conflict, 2));
    conflict = o;
    conflict.settings.q4_recurrent_bf16 = 3;
    try t.expectError(error.Q4FfnQ8Scope, ffn.require(conflict, 2));
    var context: glm.tile_affine.Context = undefined;
    context.configure(o.settings);
    try t.expectEqualDeep(o.settings.q4_ffn_q8, context.q4_ffn_q8);
    try t.expectEqual(@as(u8, 2), context.q4_recurrent_bf16);
    try t.expect(context.q4_activation and !context.q8_correlated.on);
}

test "Q4 FFN Q8: mixed pointer sorting and actual format lookup preserve legacy Q4 and Q8" {
    var ptrs = [_]u64{ 0x400000, 0x200000, 0x600000 };
    var bits = [_]u8{ 8, 4, 8 };
    glm.packed_tags.sort(&ptrs, &bits);
    try t.expectEqualDeep([3]u64{ 0x200000, 0x400000, 0x600000 }, ptrs);
    try t.expectEqualDeep([3]u8{ 4, 8, 8 }, bits);
    var p: glm.mm.Packed = .{ .ptrs = &ptrs, .bits = &bits, .affine = true };
    const a = p.tiled(0x200000, 128, 64);
    const b = p.tiled(0x400000, 128, 64);
    try t.expect(a.bits == 4 and a.bias != 0 and a.npad >= 128);
    try t.expect(b.bits == 8 and b.bias == 0 and b.s == 0x400000 + 8192);
    try t.expectEqual(@as(u8, 16), p.tiled(0x800000, 128, 64).bits);
    p.bits = &.{}; // old affine tables without entry tags remain all-Q4
    try t.expectEqual(a, p.tiled(0x200000, 128, 64));
    p.bits = &bits;
    p.affine = false;
    try t.expectEqual(b, p.tiled(0x400000, 128, 64));
}

test "Q4 FFN Q8: architectural selection uses field addresses even when values alias" {
    const cfg: glm.config.Config = .{};
    const share = try glm.shard.Share.of(&cfg, 0, 2);
    var layers: [2]glm.weights.Layer = undefined;
    for (&layers) |*l| l.mixer = .{ .kda = .{ .proj = .{ .bf16 = 17 }, .fb = 17, .gb = 17, .o = .{ .bf16 = 17 }, .conv_w = 17, .a_log = 17, .dt_bias = 17, .norm_w = 17, .eps = 1e-6, .lower = -5, .heads = 32, .dims = 4096 } };
    layers[0].ffn = .{ .dense = .{ .w = .{ .bf16 = .{ .gu = 17, .down = 17 } }, .width = share.dense.len(), .dims = 4096, .limit = 0 } };
    layers[1].ffn = .{ .moe = undefined };
    layers[1].ffn.moe.shared_gu = .{ .bf16 = 17 };
    layers[1].ffn.moe.shared_down = .{ .bf16 = 17 };
    const w: glm.weights.Weights = .{ .gpa = t.allocator, .config = cfg, .share = share, .layers = &layers, .head = .{ .bf16 = 17 } };
    for ([_]*const u64{ &layers[0].ffn.dense.w.bf16.gu, &layers[0].ffn.dense.w.bf16.down, &layers[1].ffn.moe.shared_gu.bf16, &layers[1].ffn.moe.shared_down.bf16 }) |field|
        try t.expect(ffn.selected(&w, field));
    for ([_]*const u64{ &layers[0].mixer.kda.proj.bf16, &layers[0].mixer.kda.o.bf16, &layers[0].mixer.kda.fb, &w.head.bf16 }) |field| try t.expect(!ffn.selected(&w, field));
}

const Fake = struct {
    var next: u64 = 0x100000000;
    fn alloc(p: *u64, n: usize) callconv(.c) cuda.abi.Result {
        p.* = next;
        next += std.mem.alignForward(u64, n, 4096) + 4096;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.func_name = null;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8_v2 = zero;
        return d;
    }
};
test "Q4 FFN Q8: actual shared buffers cover every selected FFN K-slice workspace" {
    var d = Fake.driver();
    const cfg: glm.config.Config = .{};
    const share = try glm.shard.Share.of(&cfg, 0, 2);
    var layers: [1]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .kda = .{ .proj = .{ .bf16 = 17 }, .fb = 17, .gb = 17, .o = .{ .bf16 = 17 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = 32, .dims = 4096 } };
    layers[0].ffn = .{ .dense = .{ .w = .{ .bf16 = .{ .gu = 17, .down = 17 } }, .width = share.dense.len(), .dims = 4096, .limit = 0 } };
    const w: glm.weights.Weights = .{ .gpa = t.allocator, .config = cfg, .share = share, .layers = &layers, .affine_target = true, .mixed_ffn_q8 = true };
    for ([_]usize{ 1, 4, 16, 32, 64 }) |rows| {
        var b = try glm.forward.Buffers.initShared(&d, &w, rows, 8192);
        defer b.deinit();
        for ([_][2]usize{ .{ 2 * share.dense.len(), 4096 }, .{ 4096, share.dense.len() }, .{ 2 * share.shared.len(), 4096 }, .{ 4096, share.shared.len() } }) |shape| {
            const needed = glm.mm.partBytes(rows, .{ .w = 0, .n = shape[0], .k = shape[1], .bits = 8 });
            try t.expect(b.dense.mm_part + needed <= b.dense.tmp);
            try t.expect(b.dense.mm_part >= b.arena.buf.ptr and b.dense.mm_part + needed <= b.arena.buf.ptr + b.arena.buf.len);
        }
    }
}
