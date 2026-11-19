//! Actual host adapter validation and eight-argument recording, without opening CUDA.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const fit = glm.q8_correlated;
const t = std.testing;

fn buffers(n: usize, k: usize, errors: bool) fit.Buffers {
    return .{
        .source = .{ .ptr = 0x100000000, .len = n * k * 2 },
        .covariance = .{ .ptr = 0x200000000, .len = k / 64 * 2080 * 8 },
        .words = .{ .ptr = 0x300000000, .len = n * k },
        .scales = .{ .ptr = 0x400000000, .len = n * k / 64 * 2 },
        .status = .{ .ptr = 0x600000000, .len = 16 },
        .errors = if (errors) .{ .ptr = 0x700000000, .len = n * k / 64 * 16 } else null,
    };
}

test "Q8 correlated: OFF has no loader calls; capacities, overlap, overflow refuse" {
    try t.expectEqual(@as(?fit.Owner, null), try fit.load(t.allocator, undefined, undefined, false));
    try t.expectEqual(@as(usize, 1), try fit.validate(1, 64, buffers(1, 64, false)));
    try t.expectEqual(@as(usize, 3 * 192), try fit.validate(3, 12288, buffers(3, 12288, true)));
    try t.expectError(error.Q8CorrelatedShape, fit.validate(1, 63, buffers(1, 64, false)));
    try t.expectError(error.Q8CorrelatedShape, fit.validate(std.math.maxInt(usize), 64, buffers(1, 64, false)));
    try t.expectError(error.Q8CorrelatedShape, fit.validate(1, std.math.maxInt(usize) & ~@as(usize, 63), buffers(1, 64, false)));
    try t.expectError(error.Overflow, fit.validate(2, std.math.maxInt(usize) & ~@as(usize, 63), buffers(1, 64, false)));
    var bad = buffers(1, 64, true);
    bad.covariance.len -= 1;
    try t.expectError(error.Q8CorrelatedCapacity, fit.validate(1, 64, bad));
    bad = buffers(1, 64, true);
    bad.errors.?.len -= 1;
    try t.expectError(error.Q8CorrelatedCapacity, fit.validate(1, 64, bad));
    bad = buffers(1, 64, true);
    bad.errors.?.ptr = bad.words.ptr;
    try t.expectError(error.Q8CorrelatedAlias, fit.validate(1, 64, bad));
    bad = buffers(1, 64, true);
    bad.covariance.ptr += 1;
    try t.expectError(error.Q8CorrelatedCapacity, fit.validate(1, 64, bad));
}

test "Q8 correlated: actual launch records every argument and null audit output" {
    const Fake = struct {
        var groups: i64 = 0;
        var columns: i64 = 0;
        var audit: u64 = 0;
        var calls: usize = 0;
        var fail = false;
        fn read(comptime T: type, p: ?*anyopaque) T {
            return @as(*align(1) const T, @ptrCast(p.?)).*;
        }
        fn launch(_: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, shared: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const a = args.?;
            std.debug.assert(gx == @as(u32, @intCast(columns)));
            std.debug.assert(gy == @as(u32, @intCast(@divTrunc(@divTrunc(groups, columns) + 3, 4))) and gz == 1 and bx == 128 and by == 1 and bz == 1 and shared == 0);
            std.debug.assert(read(u64, a[0]) == 0x100000000 and read(u64, a[1]) == 0x200000000);
            std.debug.assert(read(i64, a[2]) == groups and read(i64, a[3]) == columns);
            std.debug.assert(read(u64, a[4]) == 0x300000000 and read(u64, a[5]) == 0x400000000 and read(u64, a[6]) == 0x600000000);
            std.debug.assert(read(u64, a[7]) == audit);
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
        Fake.columns = @intCast(k / 64);
        for ([_]bool{ false, true }) |audit| {
            Fake.audit = if (audit) 0x700000000 else 0;
            try fit.launch(function, stream, n, k, buffers(n, k, audit));
        }
    };
    try t.expectEqual(@as(usize, 36), Fake.calls);
    var bad = buffers(1, 64, false);
    bad.status.ptr = bad.scales.ptr;
    try t.expectError(error.Q8CorrelatedAlias, fit.launch(function, stream, 1, 64, bad));
    try t.expectEqual(@as(usize, 36), Fake.calls);
    Fake.fail = true;
    Fake.groups = 1;
    Fake.columns = 1;
    Fake.audit = 0;
    try t.expectError(error.NotReady, fit.launch(function, stream, 1, 64, buffers(1, 64, false)));
    Fake.fail = false;
}

test "Q8 correlated: actual C18 pack is the finite nonzero-Gram fixture seed" {
    const path = std.mem.span(std.c.getenv("GLM_TEST_Q8_CORRELATED_FIXTURE") orelse return error.Q8CorrelatedFixtureRequired);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, path, t.allocator, .limited(64 << 20));
    defer t.allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try t.expectEqualStrings("30a6216a78e0fd8e0a5c4470f00c6b88a35a044e3753a1b7526efa473267c8bf", &std.fmt.bytesToHex(digest, .lower));
    const Panel = struct { name: []const u8, n: usize, k: usize, kind: u32, source: []const u16, covariance_bits: []const u64, words: []const u32, scales: []const u16 };
    const Fixture = struct { version: u32, panels: []const Panel };
    const parsed = try std.json.parseFromSlice(Fixture, t.allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try t.expectEqual(@as(usize, 30), parsed.value.panels.len);
    var tested: usize = 0;
    for (parsed.value.panels) |p| {
        if (p.kind == 1 or p.kind >= 4) continue;
        const importance = try t.allocator.alloc(f64, p.k);
        defer t.allocator.free(importance);
        for (importance, 0..) |*v, i| v.* = @bitCast(p.covariance_bits[i / 64 * 2080 + glm.q8_correlated_reference.tri(i % 64, i % 64)]);
        const words = try t.allocator.alloc(u32, p.words.len);
        defer t.allocator.free(words);
        const scales = try t.allocator.alloc(u16, p.scales.len);
        defer t.allocator.free(scales);
        _ = try glm.q8_activation.quantize(p.source, p.n, p.k, importance, words, scales);
        if (!std.mem.eql(u32, words, p.words) or !std.mem.eql(u16, scales, p.scales)) std.debug.print("seed mismatch {s}\n", .{p.name});
        try t.expectEqualSlices(u32, p.words, words);
        try t.expectEqualSlices(u16, p.scales, scales);
        tested += 1;
    }
    try t.expectEqual(@as(usize, 17), tested);
}

test "Q8 correlated serving: strict cached mode and SHA, scope and rank agreement" {
    const Get = struct {
        var flag: ?[]const u8 = null;
        var sha: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_Q8_CORRELATED")) return flag;
            if (std.mem.eql(u8, name, "TF_GLM_Q8_CORRELATION_SHA256")) return sha;
            return null;
        }
    };
    var problem: []const u8 = "";
    const opts = glm.q8_correlated_options;
    Get.sha = "bad";
    try t.expect(!(try opts.read(Get.get, &problem)).on);
    Get.flag = "0";
    try t.expect(!(try opts.read(Get.get, &problem)).on);
    Get.flag = "true";
    try t.expectError(error.BadSetting, opts.read(Get.get, &problem));
    Get.flag = "1";
    try t.expectError(error.BadSetting, opts.read(Get.get, &problem));
    const hex: [64]u8 = @splat('2');
    Get.sha = &hex;
    const on = try opts.read(Get.get, &problem);
    Get.sha = "bad";
    try t.expect(on.on and std.mem.allEqual(u8, &on.sha, 0x22));
    var settings: glm.ranks.Settings = .{};
    settings.q8_correlated = on;
    settings.q8_activation = true;
    settings.calibration_sha = try glm.q8_moments.digest("3f3566c55dfa5fff0b194fc53f8db0c6bb8a973f705d6e75e4219b6a919b64ce");
    settings.q8_mse = true;
    settings.dense_bits = 8;
    settings.head_bits = 16;
    settings.tiled = true;
    settings.dflash_only_head = true;
    const O = struct { settings: glm.ranks.Settings, drafter: bool = true, mtp: bool = true, vision: ?u8 = null };
    var o: O = .{ .settings = settings };
    try opts.require(o, 2);
    try t.expectError(error.Q8CorrelatedScope, opts.require(o, 4));
    inline for (.{ "q8_activation", "q8_mse", "tiled", "dflash_only_head" }) |field| {
        o = .{ .settings = settings };
        @field(o.settings, field) = false;
        try t.expectError(error.Q8CorrelatedScope, opts.require(o, 2));
    }
    inline for (.{ "pages", "verify_graphs", "q4_affine_mse", "q4_activation" }) |field| {
        o = .{ .settings = settings };
        @field(o.settings, field) = true;
        try t.expectError(error.Q8CorrelatedScope, opts.require(o, 2));
    }
    o = .{ .settings = settings, .vision = 1 };
    try t.expectError(error.Q8CorrelatedScope, opts.require(o, 2));
    o = .{ .settings = settings, .drafter = false };
    try t.expectError(error.Q8CorrelatedScope, opts.require(o, 2));
    o = .{ .settings = settings, .mtp = false };
    try t.expectError(error.Q8CorrelatedScope, opts.require(o, 2));
    o.settings.q8_correlated = .{};
    try opts.require(o, 1);
    try t.expectEqual(@as(?*glm.q8_correlated_owner.Owner, null), try glm.q8_correlated_owner.Owner.open(t.allocator, undefined, undefined, undefined, .{}, @splat(0), 0));
    const FP = glm.ranks.Fingerprint;
    const f = FP.of(2, 0, settings, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 84), f.layout);
    try t.expectEqual(@as(u32, 1), f.q8_correlated);
    try t.expectEqualSlices(u8, &on.sha, &f.correlation_sha);
    var other = f;
    other.rank = 1;
    try t.expectEqual(@as(?glm.ranks.Mismatch, null), glm.ranks.mismatch(&.{ f, other }));
    other.q8_correlated = 0;
    try t.expectEqual(FP.Field.q8_correlated, glm.ranks.mismatch(&.{ f, other }).?.field);
    other.q8_correlated = 1;
    other.correlation_sha[0] ^= 1;
    try t.expectEqual(FP.Field.correlation_sha, glm.ranks.mismatch(&.{ f, other }).?.field);
}

test "Q8 correlated serving: status refuses before accounting, failed owner refuses before GPU" {
    var c: glm.q8_correlated_owner.Counts = .{};
    try c.observe(100, .{ 0, 20, 10, 2 });
    try t.expectEqual(@as(u64, 100), c.groups);
    const old = c;
    for ([_][4]u32{ .{ 1, 0, 0, 0 }, .{ 0, 101, 0, 2 }, .{ 0, 60, 41, 2 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 0, 1 } }) |bad| {
        try t.expectError(error.Q8CorrelatedStatus, c.observe(100, bad));
        try t.expect(std.meta.eql(old, c));
    }
    var owner: glm.q8_correlated_owner.Owner = undefined;
    owner.failed = true;
    owner.prepared = true;
    try t.expectError(error.Q8CorrelatedOwnerState, owner.run(0, 1, 64, 1, 2, 3));
    owner.failed = false;
    owner.prepared = false;
    try t.expectError(error.Q8CorrelatedOwnerState, owner.run(0, 1, 64, 1, 2, 3));
}

test "Q8 correlated serving: both actual frozen captures have original normalized diagonal and ordinal map" {
    const root = std.mem.span(std.c.getenv("GLM_TEST_Q8_CORRELATION_ROOT") orelse return error.CorrelationRootRequired);
    const moment_root = std.mem.span(std.c.getenv("GLM_TEST_CALIBRATION_ROOT") orelse return error.MomentRootRequired);
    const path = try std.fs.path.join(t.allocator, &.{ root, "manifest.json" });
    defer t.allocator.free(path);
    const manifest_text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, t.allocator, .limited(1 << 20));
    defer t.allocator.free(manifest_text);
    const Header = struct { checkpoint: []const u8 };
    const parsed = try std.json.parseFromSlice(Header, t.allocator, manifest_text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const checkpoint = try glm.q8_moments.digest(parsed.value.checkpoint);
    for (0..2) |rank| {
        const data = try glm.q8_correlation_input.Data.open(t.allocator, t.io, root, .{ .manifest = try glm.q8_moments.digest("9ada369e457fafb9032cdf673d324edbddd23b4e3c0f440f68ea5b760127a065"), .checkpoint = checkpoint }, rank);
        defer data.destroy();
        const diagonal = try glm.q8_moments.Data.open(t.allocator, t.io, moment_root, .{ .manifest = try glm.q8_moments.digest("3f3566c55dfa5fff0b194fc53f8db0c6bb8a973f705d6e75e4219b6a919b64ce"), .checkpoint = checkpoint }, rank);
        defer diagonal.destroy();
        const Lin = struct { n: usize, k: usize };
        const list = try t.allocator.alloc(Lin, data.entries.len);
        defer t.allocator.free(list);
        for (list, data.entries) |*l, e| l.* = .{ .n = e.n, .k = e.k };
        try t.expectEqual(@as(usize, 282), list.len);
        try t.expect((try glm.q8_correlated_owner.validateInput(data, diagonal, list, true)) >= 4096);
        const old = diagonal.means[0];
        diagonal.means[0] = old + 1;
        try t.expectError(error.Q8CorrelationDiagonal, glm.q8_correlated_owner.validateInput(data, diagonal, list, true));
        diagonal.means[0] = old;
    }
}
