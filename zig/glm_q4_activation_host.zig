//! Principal calibrated-affine host checks; no driver library or model is opened.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const fit = glm.affine_activation;
const moments = glm.q8_moments;
const t = std.testing;

test {
    std.testing.refAllDecls(glm.q4_activation_reference);
}

test "Q4 activation: strict cached setting, both head scopes and calibration agreement" {
    const Env = struct {
        var mode: ?[]const u8 = null;
        var digest: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_Q4_ACTIVATION")) return mode;
            if (std.mem.eql(u8, name, "TF_GLM_CALIBRATION_SHA256")) return digest;
            return null;
        }
    };
    defer {
        Env.mode = null;
        Env.digest = null;
    }
    var problem: []const u8 = "";
    Env.digest = "ignored while disabled";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    Env.mode = "0";
    try t.expectEqualDeep(off, try glm.ranks.Settings.read(Env.get, &problem));
    try t.expectEqual(@as(?fit.Owner, null), try fit.load(t.allocator, undefined, undefined, false));
    for ([_][]const u8{ "", "2", "01", "true", " 1" }) |bad| {
        Env.mode = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    }
    Env.mode = "1";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    const hex: [64]u8 = @splat('1');
    Env.digest = &hex;
    var on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.mode = "0";
    try t.expect(on.q4_activation and on.calibration_sha[0] == 0x11);
    on.q4_affine_mse = true;
    on.dense_bits = 4;
    on.dflash_only_head = true;
    var o: glm.engine.Options = .{ .capacity = 8192, .mtp = true, .drafter = true, .settings = on };
    try glm.q4_activation_options.require(o, 2);
    o.settings.head_bits = 8;
    o.settings.q4_head_fp8 = true;
    try glm.q4_activation_options.require(o, 2);
    try t.expectError(error.Q4ActivationScope, glm.q4_activation_options.require(o, 4));
    inline for (.{ "q8_mse", "q8_activation", "pages", "verify_graphs" }) |field| {
        var bad = o;
        @field(bad.settings, field) = true;
        try t.expectError(error.Q4ActivationScope, glm.q4_activation_options.require(bad, 2));
    }
    var no_draft = o;
    no_draft.drafter = false;
    try t.expectError(error.Q4ActivationScope, glm.q4_activation_options.require(no_draft, 2));
    const FP = glm.ranks.Fingerprint;
    var previous = on;
    previous.q4_activation = false;
    const a = FP.of(2, 0, previous, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 78), b.layout);
    try t.expectEqual(FP.Field.q4_activation, glm.ranks.mismatch(&.{ a, b }).?.field);
    var other = on;
    other.calibration_sha[17] ^= 1;
    const c = FP.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0));
    const d = FP.of(2, 1, other, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.calibration_sha, glm.ranks.mismatch(&.{ c, d }).?.field);
}

test "Q4 activation: BF16 and FP8 head maps retain every target ordinal" {
    var entries = [_]moments.Entry{
        .{ .n = 8, .k = 64, .offset = 0 },
        .{ .n = 16, .k = 128, .offset = 64 },
        .{ .n = 32, .k = 64, .offset = 192 },
    };
    var means: [256]f64 = @splat(1);
    const data: moments.Data = .{ .gpa = t.allocator, .entries = &entries, .means = &means, .target_count = 2 };
    const Shape = struct { n: usize, k: usize };
    const list = [_]Shape{ .{ .n = 8, .k = 64 }, .{ .n = 16, .k = 128 }, .{ .n = 32, .k = 64 } };
    try fit.validateMap(&data, &list, true);
    try fit.validateMap(&data, list[0..2], false);
    try t.expectError(error.Q4CalibrationShape, fit.validateMap(&data, list[0..2], true));
    try t.expectError(error.Q4CalibrationShape, fit.validateMap(&data, list[0..0], true));
    var swapped = list;
    std.mem.swap(Shape, &swapped[0], &swapped[1]);
    try t.expectError(error.Q8CalibrationShape, fit.validateMap(&data, &swapped, true));
}

fn buffers(n: usize, k: usize) fit.Buffers {
    return .{
        .source = .{ .ptr = 0x100000000, .len = n * k * 2 },
        .moments = .{ .ptr = 0x200000000, .len = k * 8 },
        .words = .{ .ptr = 0x300000000, .len = n * k / 2 },
        .scales = .{ .ptr = 0x400000000, .len = n * k / 64 * 2 },
        .biases = .{ .ptr = 0x500000000, .len = n * k / 64 * 2 },
        .status = .{ .ptr = 0x600000000, .len = 16 },
    };
}

test "Q4 activation: actual eight-argument launch and pre-enqueue guards" {
    const Fake = struct {
        var calls: usize = 0;
        var groups: i64 = 0;
        var kg: i64 = 0;
        var fail = false;
        fn read(comptime T: type, p: ?*anyopaque) T {
            return @as(*align(1) const T, @ptrCast(p.?)).*;
        }
        fn launch(_: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const a = args.?;
            std.debug.assert(gx == @as(u32, @intCast(@divTrunc(groups + 127, 128))));
            std.debug.assert(gy == 1 and gz == 1 and bx == 128 and by == 1 and bz == 1 and smem == 0);
            std.debug.assert(read(u64, a[0]) == 0x100000000 and read(u64, a[1]) == 0x200000000);
            std.debug.assert(read(i64, a[2]) == groups and read(i64, a[3]) == kg);
            inline for (.{ 4, 5, 6, 7 }) |i|
                std.debug.assert(read(u64, a[i]) == @as(u64, i - 1) * 0x100000000);
            calls += 1;
            return if (fail) cuda.abi.error_not_ready else 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const function: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(1) };
    for ([_]usize{ 1, 3, 127, 128, 129, 4096 }) |n| for ([_]usize{ 64, 128, 4096 }) |k| {
        Fake.groups = @intCast(n * k / 64);
        Fake.kg = @intCast(k / 64);
        try fit.launch(function, stream, n, k, buffers(n, k));
    };
    const before = Fake.calls;
    try t.expectEqual(@as(usize, 18), before);
    try t.expectError(error.Q4ActivationShape, fit.launch(function, stream, 1, 63, buffers(1, 64)));
    var bad = buffers(1, 64);
    bad.moments.len -= 1;
    try t.expectError(error.Q4ActivationCapacity, fit.launch(function, stream, 1, 64, bad));
    bad = buffers(1, 64);
    bad.biases.ptr = bad.scales.ptr;
    try t.expectError(error.Q4ActivationAlias, fit.launch(function, stream, 1, 64, bad));
    try t.expectEqual(before, Fake.calls);
    Fake.fail = true;
    Fake.groups = 1;
    Fake.kg = 1;
    try t.expectError(error.NotReady, fit.launch(function, stream, 1, 64, buffers(1, 64)));
    Fake.fail = false;
}

test "Q4 activation: experimental tier refuses before engine allocations" {
    var e: glm.engine.Engine = undefined;
    e.settings.q8_mse = false;
    e.settings.q4_activation = true;
    try t.expectError(error.Q4ActivationNvmeScope, glm.nvme_state.State.open(&e, "unused", @splat(0), undefined));
}
