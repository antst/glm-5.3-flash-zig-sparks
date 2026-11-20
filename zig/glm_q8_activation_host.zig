//! Independent packed-error and identity controls for the optional input-weighted Q8 startup fit.
const std = @import("std");
const glm = @import("glm");
const t = std.testing;
const a = t.allocator;
const fit = glm.q8_activation;
const moments = glm.q8_moments;
const mse = @import("glm").q8_mse;
const hex11: [64]u8 = @splat('1');
const hex22: [64]u8 = @splat('2');

fn digest256(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn errorOf(src: []const u16, k: usize, importance: []const f64, words: []const u32, scales: []const u16) f64 {
    @setFloatMode(.strict);
    var err: f64 = 0;
    for (src, 0..) |h, i| {
        const raw = (words[i / 4] >> @as(u5, @intCast(8 * (i % 4)))) & 255;
        const q: f64 = @floatFromInt(@as(i32, @intCast(raw)) - 128);
        const scale: f64 = @as(f16, @bitCast(scales[i / 64]));
        const value: f64 = @as(f32, @bitCast(@as(u32, h) << 16));
        const delta = q * scale - value;
        err += importance[i % k] * delta * delta;
    }
    return err;
}

test "Q8 activation: actual pack improves independent weighted error and retains zero-weight baseline" {
    const n = 16;
    const k = 256;
    var src: [n * k]u16 = undefined;
    var importance: [k]f64 = undefined;
    for (&src, 0..) |*h, i| {
        const x: f32 = @floatFromInt(@as(i32, @intCast((i * 2341 + i / 7) % 2039)) - 1019);
        h.* = @truncate(@as(u32, @bitCast(x * 0.000071)) >> 16);
    }
    for (&importance, 0..) |*v, i| v.* = if (i % 13 == 0) 100 else if (i % 3 == 0) 0.01 else 1;
    var old: [n * k / 4]u32 = undefined;
    var os: [n * k / 64]u16 = undefined;
    _ = try mse.quantize(&src, n, k, &old, &os);
    var out: @TypeOf(old) = undefined;
    var scales: @TypeOf(os) = undefined;
    const q = try fit.quantize(&src, n, k, &importance, &out, &scales);
    const before = errorOf(&src, k, &importance, &old, &os);
    const after = errorOf(&src, k, &importance, &out, &scales);
    try t.expect(q.improved > 0 and after < before);
    try t.expectApproxEqAbs(before, q.baseline_error, 1e-14);
    try t.expectApproxEqAbs(after, q.selected_error, 1e-14);
    for (0..os.len) |g| {
        const weights = importance[(g % (k / 64)) * 64 ..][0..64];
        const input = src[g * 64 ..][0..64];
        try t.expect(errorOf(input, 64, weights, out[g * 16 ..][0..16], scales[g..][0..1]) <=
            errorOf(input, 64, weights, old[g * 16 ..][0..16], os[g..][0..1]));
    }
    @memset(&importance, 0);
    const zero = try fit.quantize(&src, n, k, &importance, &out, &scales);
    try t.expectEqual(@as(usize, 0), zero.improved);
    try t.expectEqualSlices(u32, &old, &out);
    try t.expectEqualSlices(u16, &os, &scales);
}

test "Q8 activation: invalid moments and shapes refuse before publishing pack bytes" {
    const src: [64]u16 = @splat(0x3f80);
    var weights: [64]f64 = @splat(1);
    var out: [16]u32 = @splat(0xdeadbeef);
    var scales: [1]u16 = .{0x1234};
    for ([_]f64{ -1, std.math.nan(f64), std.math.inf(f64) }) |bad| {
        weights[63] = bad;
        try t.expectError(error.Q8ActivationMoment, fit.quantize(&src, 1, 64, &weights, &out, &scales));
        try t.expectEqual(@as(u32, 0xdeadbeef), out[0]);
        try t.expectEqual(@as(u16, 0x1234), scales[0]);
    }
    try t.expectError(error.Q8ActivationShape, fit.quantize(&src, 1, 64, weights[0..63], &out, &scales));
}

fn put(comptime T: type, b: []u8, at: usize, value: T) void {
    std.mem.writeInt(T, b[at..][0..@sizeOf(T)], value, .little);
}

fn blob(rank: u32) [1200]u8 {
    var b: [1200]u8 = @splat(0);
    @memcpy(b[0..8], "GLMCAL01");
    inline for (.{ .{ 8, 1 }, .{ 16, 2 }, .{ 20, 2 }, .{ 32, 1 }, .{ 36, 32 }, .{ 40, 64 }, .{ 44, 40 } }) |p|
        put(u32, &b, p[0], p[1]);
    put(u32, &b, 12, rank);
    put(u64, &b, 24, 128);
    @memset(b[48..80], 0x11);
    for (0..2) |i| {
        const at = 96 + i * 40;
        put(u32, &b, at + 4, if (i == 0) 0 else 2);
        put(u32, &b, at + 8, @intCast(8 + i * 8));
        put(u32, &b, at + 12, 64);
        put(u64, &b, at + 16, 1);
        put(u64, &b, at + 24, if (i == 0) 2 else 1);
        put(u64, &b, at + 32, i * 512);
    }
    for (0..128) |i| put(u64, &b, 176 + i * 8, @bitCast(@as(f64, if (i < 64) 8 else 2)));
    return b;
}

test "Q8 activation: decoder validates rank/model/map, normalizes counts and rejects corrupt data" {
    const bytes = blob(0);
    const d = try moments.decode(a, &bytes, @splat(0x11), 0);
    defer d.destroy();
    try t.expectEqual(@as(usize, 1), d.target_count);
    try t.expectEqual(@as(f64, 4), (try d.linear(0, 8, 64))[0]);
    try t.expectEqual(@as(f64, 2), (try d.linear(1, 16, 64))[63]);
    try t.expectError(error.Q8CalibrationShape, d.linear(0, 16, 64));
    try t.expectError(error.Q8CalibrationIdentity, moments.decode(a, &bytes, @splat(0x22), 0));
    try t.expectError(error.Q8CalibrationIdentity, moments.decode(a, &bytes, @splat(0x11), 1));
    try t.expectError(error.Q8CalibrationShape, moments.decode(a, bytes[0..1199], @splat(0x11), 0));
    var bad = bytes;
    put(u64, &bad, 176, @bitCast(std.math.nan(f64)));
    try t.expectError(error.Q8CalibrationMoment, moments.decode(a, &bad, @splat(0x11), 0));
    bad = bytes;
    put(u64, &bad, 96 + 24, 0);
    try t.expectError(error.Q8CalibrationShape, moments.decode(a, &bad, @splat(0x11), 0));
    bad = bytes;
    put(u32, &bad, 136 + 4, 0);
    try t.expectError(error.Q8CalibrationShape, moments.decode(a, &bad, @splat(0x11), 0));
}

test "Q8 activation: actual manifest/file identity path and checksum refusal" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const first = blob(0);
    const second = blob(1);
    const hashes = .{
        std.fmt.bytesToHex(digest256(&first), .lower),
        std.fmt.bytesToHex(digest256(&second), .lower),
    };
    inline for (.{ "rank-0.bin", "rank-1.bin" }, .{ &first, &second }) |name, data| {
        const f = try tmp.dir.createFile(t.io, name, .{});
        defer f.close(t.io);
        try f.writeStreamingAll(t.io, data);
    }
    const Row = struct { rank: u32, file: []const u8, bytes: usize, sha256: []const u8 };
    const rows = [_]Row{
        .{ .rank = 0, .file = "rank-0.bin", .bytes = first.len, .sha256 = &hashes[0] },
        .{ .rank = 1, .file = "rank-1.bin", .bytes = second.len, .sha256 = &hashes[1] },
    };
    const manifest = .{ .schema = "glm-input-moments/v1", .world = 2, .checkpoint = @as([]const u8, &hex11), .entries = 2, .target_count = 1, .features = 128, .ranks = rows };
    const text = try std.json.Stringify.valueAlloc(a, manifest, .{});
    defer a.free(text);
    const f = try tmp.dir.createFile(t.io, "manifest.json", .{});
    defer f.close(t.io);
    try f.writeStreamingAll(t.io, text);
    var path: [4096]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &path);
    var id: moments.Identity = .{ .manifest = digest256(text), .checkpoint = @splat(0x11) };
    const d = try moments.Data.open(a, t.io, path[0..len], id, 1);
    defer d.destroy();
    try t.expectEqual(@as(f64, 4), (try d.linear(0, 8, 64))[63]);
    id.manifest[0] ^= 1;
    try t.expectError(error.Q8CalibrationIdentity, moments.Data.open(a, t.io, path[0..len], id, 1));
}

test "Q8 activation: strict cached switch and manifest agreement, unchanged OFF scope" {
    const Env = struct {
        var mode: ?[]const u8 = null;
        var sha: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_Q8_ACTIVATION")) return mode;
            if (std.mem.eql(u8, name, "TF_GLM_CALIBRATION_SHA256")) return sha;
            return null;
        }
    };
    defer {
        Env.mode = null;
        Env.sha = null;
    }
    var problem: []const u8 = "";
    Env.sha = "ignored while OFF";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    Env.mode = "0";
    try t.expectEqualDeep(off, try glm.ranks.Settings.read(Env.get, &problem));
    for ([_][]const u8{ "true", "01", "2" }) |bad| {
        Env.mode = bad;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    }
    Env.mode = "1";
    try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
    Env.sha = &hex11;
    var on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.sha = &hex22;
    try t.expectEqual(@as(u8, 0x11), on.calibration_sha[0]);
    on.q8_mse = true;
    on.dense_bits = 8;
    on.dflash_only_head = true;
    const o: glm.engine.Options = .{ .capacity = 8192, .mtp = true, .drafter = true, .settings = on };
    try glm.q8_activation_options.require(o, 2);
    try t.expectError(error.Q8ActivationScope, glm.q8_activation_options.require(o, 4));
    var old = on;
    old.q8_activation = false;
    const FP = glm.ranks.Fingerprint;
    const f0 = FP.of(2, 0, old, 8192, true, @splat(0), @splat(0), @splat(0));
    const f1 = FP.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.q8_activation, glm.ranks.mismatch(&.{ f0, f1 }).?.field);
    var other = on;
    other.calibration_sha = @splat(0x22);
    const f2 = FP.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0));
    const f3 = FP.of(2, 1, other, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(FP.Field.calibration_sha, glm.ranks.mismatch(&.{ f2, f3 }).?.field);
}

test "Q8 activation: optional real captured package binds both rank maps without CUDA" {
    const raw = std.c.getenv("GLM_TEST_CALIBRATION_ROOT") orelse return error.SkipZigTest;
    const root = std.mem.span(raw);
    const path = try std.fs.path.join(a, &.{ root, "manifest.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(64 << 10));
    defer a.free(text);
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    const checkpoint = try moments.digest(doc.value.object.get("checkpoint").?.string);
    const id: moments.Identity = .{ .manifest = digest256(text), .checkpoint = checkpoint };
    for (0..2) |rank| {
        const d = try moments.Data.open(a, t.io, root, id, rank);
        defer d.destroy();
        try t.expectEqual(@as(usize, 281), d.target_count);
        try t.expectEqual(@as(usize, 282), d.entries.len);
        try t.expectEqual(@as(usize, 751104), d.means.len);
    }
}
