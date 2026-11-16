//! Upstream's multi-CTA decode selection, separately owned so the qualified glue module remains unchanged.
const std = @import("std");
const cuda = @import("cuda");

pub const image_sha = "bdc35fe05bc3ed68b71caca13fd1dd68eea6e44b5677cdf81c1a26db82c2e778";
pub const max_rows = 48;
pub const span = 4096;
pub const candidates = 4096;
pub const topk = 512;

pub fn bitsBytes(rows: usize, np: usize) !usize {
    if (rows == 0 or rows > max_rows or np == 0 or np > std.math.maxInt(i32)) return error.DselShape;
    const words = (try std.math.add(usize, np, 31)) / 32;
    return std.math.mul(usize, try std.math.mul(usize, rows * 2, words), 4);
}

pub fn scanCtas(at: usize, rows: usize, np: usize) usize {
    const last = @min(np, @max(topk, (at + rows) / 4));
    return if (last <= candidates) 0 else (last + span - 1) / span;
}

pub fn require(s: anytype) !void {
    if (s.dsel and s.verify_graphs) return error.DselVerifyGraphScope;
}

pub const Owner = struct {
    module: cuda.Module,
    scan: cuda.Function,
    merge: cuda.Function,
    bits: cuda.DeviceBuffer,
    stream: cuda.Stream,
    np: usize,
    rows: usize,
    rank: usize,
    calls: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, rows: usize, capacity: usize, rank: usize) !Owner {
        if (capacity == 0 or capacity > std.math.maxInt(i32) - max_rows) return error.DselShape;
        const np = @max(1024, try std.math.ceilPowerOfTwo(usize, capacity / 4 + 1));
        const bytes = try bitsBytes(rows, np);
        _ = gpa;
        _ = io;
        const image = cuda.kernels.glm_decode_select;
        if (image.len == 0) return error.DselImageRequired;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(image, &digest, .{});
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.DselImageHash;
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const scan = try module.function("tf_glm_dsel_scan");
        const merge = try module.function("tf_glm_dsel_merge");
        const bits = try cuda.DeviceBuffer.alloc(d, bytes);
        std.log.info("rank {d}: GLM_DSEL active=true sha256={s} rows={d} np={d} bytes={d}", .{ rank, image_sha, rows, np, bytes });
        return .{ .module = module, .scan = scan, .merge = merge, .bits = bits, .stream = stream, .np = np, .rows = rows, .rank = rank };
    }

    /// The ordinary select remains the fallback before either new enqueue; a launch error never falls back.
    pub fn select(o: *Owner, stream: cuda.Stream, scores: u64, pools: u64, position: u64, at: usize, rows: usize, np: usize, prefill: bool) !bool {
        return o.selectPath(stream, scores, pools, position, at, rows, np, prefill, 0);
    }

    /// Optional per-row paths are for the device contract; serving passes zero.
    pub fn selectPath(o: *Owner, stream: cuda.Stream, scores: u64, pools: u64, position: u64, at: usize, rows: usize, np: usize, prefill: bool, path: u64) !bool {
        if (prefill or rows == 0 or rows > max_rows or np < 4096) return false;
        if (at > std.math.maxInt(i32) - rows) return error.DselShape;
        const grid = scanCtas(at, rows, np);
        if (grid == 0) return false;
        if (rows > o.rows or np > o.np or try bitsBytes(rows, np) > o.bits.len) return error.DselCapacity;
        if (stream.handle != o.stream.handle) return error.DselStream;
        if (scores == 0 or scores % 4 != 0 or pools == 0 or position == 0 or position % 4 != 0 or pools % 8 != 0)
            return error.DselAddress;
        const words = (np + 31) / 32;
        var sa: cuda.Args = .{};
        inline for (.{ scores, @as(i64, @intCast(np)), position, @as(c_int, topk), @as(c_int, 2048), o.bits.ptr, @as(i64, @intCast(words)) }) |v| sa.add(v);
        try cuda.launch.launch(o.scan, .{ .grid = .{ .x = @intCast(grid), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &sa);
        var ma: cuda.Args = .{};
        inline for (.{ scores, pools, @as(i64, @intCast(np)), position, @as(c_int, topk), o.bits.ptr, @as(i64, @intCast(words)), path }) |v| ma.add(v);
        try cuda.launch.launch(o.merge, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &ma);
        o.calls += 1;
        return true;
    }

    pub fn deinit(o: *Owner) void {
        cuda.completion.drain(o.stream.d, o.stream, "decode select owner close");
        std.log.info("rank {d}: GLM_DSEL_CLOSE calls={d} bytes={d}", .{ o.rank, o.calls, o.bits.len });
        o.bits.free();
        o.module.unload();
        o.* = undefined;
    }
};

pub fn open(e: anytype, gpa: std.mem.Allocator, io: std.Io) !void {
    if (!e.settings.dsel) return;
    e.decode_select = try Owner.init(gpa, io, e.d, e.stream, e.settings.verify_rows, e.capacity, e.w.share.rank);
}

pub fn close(e: anytype) void {
    if (e.decode_select) |*owner| owner.deinit();
}

test "decode select: bit planes and scan bounds cover rows through48 without wrapping" {
    for (1..49) |rows| for ([_]usize{ 512, 1024, 4095, 4096, 4097, 65536, 524288 }) |np| {
        const bytes = try bitsBytes(rows, np);
        try std.testing.expect(bytes >= rows * np / 4);
        try std.testing.expectEqual(rows * 2 * ((np + 31) / 32) * 4, bytes);
        for ([_]usize{ 0, 16368, 16384, 65535, 1048576 }) |at| {
            const visible = @min(np, @max(topk, (at + rows) / 4));
            const ctas = scanCtas(at, rows, np);
            if (visible <= candidates) try std.testing.expectEqual(@as(usize, 0), ctas) else try std.testing.expect(ctas * span >= visible and (ctas - 1) * span < visible);
        }
    };
    try std.testing.expectError(error.DselShape, bitsBytes(0, 1024));
    try std.testing.expectError(error.DselShape, bitsBytes(49, 1024));
    try std.testing.expectError(error.DselShape, bitsBytes(1, std.math.maxInt(usize)));
}

test "decode select: captured whole-target graphs are refused before allocations" {
    try require(.{ .dsel = false, .verify_graphs = true });
    try require(.{ .dsel = true, .verify_graphs = false });
    try std.testing.expectError(error.DselVerifyGraphScope, require(.{ .dsel = true, .verify_graphs = true }));
}

test {
    _ = @import("cuda_decode_select_test.zig");
}
