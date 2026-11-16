//! Optional dense shared attention; callers own the immutable uploaded row table and completion lifetime.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

pub const max_rows = 64;
pub const max_segments = 8;
pub const Row = extern struct { cache: u64, scales: u64, position: u64, row0: i32, chunks: i32 };
comptime {
    if (@sizeOf(Row) != 32 or @alignOf(Row) != 8) @compileError("Dense row ABI");
}

pub fn enabled(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.DenseSetting;
}

pub const Segment = struct {
    rows: usize,
    host_pos: usize,
    capacity: usize,
    cache: u64,
    scales: u64 = 0,
    position: u64,
};
pub const Scope = struct {
    prefill: bool = false,
    paged: bool = false,
    host_kv: bool = false,
    heads: usize = 16,
    width: usize = 512,
    max_part_rows: usize = 8,
    dense_limit: usize,
    scratch_rows: usize,
    scratch_chunks: usize,
};

pub const Plan = struct {
    table: [max_rows]Row = undefined,
    rows: usize = 0,
    chunks: usize = 0,
    most: usize = 0,
    fp8: bool = false,
    inputs: [max_segments * 3]View = undefined,
    input_count: usize = 0,

    pub fn make(segs: []const Segment, scope: Scope) !Plan {
        if (scope.prefill or scope.paged or scope.host_kv) return error.DenseScope;
        if (scope.max_part_rows != 8 and scope.max_part_rows != 16) return error.DenseRows;
        if (scope.heads != 16 or scope.width != 512) return error.DenseShape;
        if (segs.len == 0 or segs.len > max_segments) return error.DenseSegments;
        var p: Plan = .{ .fp8 = segs[0].scales != 0 };
        for (segs) |s| {
            if (s.rows == 0 or s.rows > scope.max_part_rows or s.rows > max_rows - p.rows) return error.DenseRows;
            if ((s.scales != 0) != p.fp8) return error.DenseMixedKV;
            if (s.cache == 0 or s.cache % 16 != 0 or s.position == 0 or s.position % 4 != 0)
                return error.DenseAddress;
            if (p.fp8 and s.scales % 4 != 0) return error.DenseAddress;
            const end = try std.math.add(usize, s.host_pos, s.rows);
            if (end > s.capacity or end > scope.dense_limit or end > std.math.maxInt(i32))
                return error.DenseScope;
            const nch = (end + 511) / 512;
            if (nch == 0 or nch > scope.scratch_chunks or p.rows + s.rows > scope.scratch_rows)
                return error.DenseScratch;
            try p.input(s.cache, try std.math.mul(usize, s.capacity, if (p.fp8) 512 else 1024));
            if (p.fp8) try p.input(s.scales, try std.math.mul(usize, s.capacity, 4));
            try p.input(s.position, 4);
            for (0..s.rows) |r| p.table[p.rows + r] = .{
                .cache = s.cache,
                .scales = s.scales,
                .position = s.position,
                .row0 = @intCast(p.rows),
                .chunks = @intCast(nch),
            };
            if (scope.max_part_rows > 8) p.most = @max(p.most, s.rows);
            p.rows += s.rows;
            p.chunks = @max(p.chunks, nch);
        }
        return p;
    }

    pub fn bytes(p: *const Plan) []const u8 {
        return std.mem.sliceAsBytes(p.table[0..p.rows]);
    }

    fn input(p: *Plan, ptr: u64, len: usize) !void {
        _ = try std.math.add(u64, ptr, @intCast(len));
        p.inputs[p.input_count] = .{ .ptr = ptr, .len = len };
        p.input_count += 1;
    }
};

pub const View = struct { ptr: u64, len: usize };
pub const Storage = struct { table: View, qa: View, po: View, pm: View, pl: View, out: View };

/// No new symbol lookup or resource mutation when disabled; missing symbols refuse the opt-in.
pub const Kernels = struct {
    chunks: cuda.Function,
    chunks_fp8: cuda.Function,
    merge: cuda.Function,

    pub fn load(module: cuda.Module, on: bool) !?Kernels {
        if (!on) return null;
        const k: Kernels = .{
            .chunks = try module.function("tf_glm_dense_seg_chunks_16_v1"),
            .chunks_fp8 = try module.function("tf_glm_dense_seg_chunks_16_fp8_v1"),
            .merge = try module.function("tf_glm_dense_seg_merge_v1"),
        };
        try k.chunks.allowDynamicShared(glue.chunksSmem(16));
        try k.chunks_fp8.allowDynamicShared(glue.chunksSmem(16));
        return k;
    }

    pub fn launch(k: *const Kernels, stream: cuda.Stream, p: *const Plan, s: Storage, scale: f32) !void {
        try validate(p, s);
        var a: cuda.Args = .{};
        inline for (.{ s.table.ptr, s.qa.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, @as(i64, @intCast(p.rows)), @as(i64, 16), scale }) |v| a.add(v);
        try cuda.launch.launch(if (p.fp8) k.chunks_fp8 else k.chunks, .{
            .grid = .{ .x = @intCast(p.rows), .y = 1, .z = @intCast(p.chunks) },
            .block = .{ .x = 256 },
            .shared = glue.chunksSmem(16),
        }, stream, &a);
        var b: cuda.Args = .{};
        inline for (.{ s.table.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, s.out.ptr, @as(i64, @intCast(p.rows)), @as(i64, 16) }) |v| b.add(v);
        try cuda.launch.launch(k.merge, .{
            .grid = .{ .x = @intCast(p.rows), .y = 16 },
            .block = .{ .x = 128 },
        }, stream, &b);
    }
};

pub fn validate(p: *const Plan, s: Storage) !void {
    if (p.rows == 0 or p.rows > max_rows or p.chunks == 0 or p.input_count > p.inputs.len)
        return error.DenseShape;
    const values = try std.math.mul(usize, p.rows, 16 * 512 * 2);
    const stats = try std.math.mul(usize, try std.math.mul(usize, p.rows, p.chunks), 16 * 4);
    const outputs = [_]View{ try need(s.po, try std.math.mul(usize, stats, 512)), try need(s.pm, stats), try need(s.pl, stats), try need(s.out, values) };
    const inputs = [_]View{ try need(s.table, p.rows * @sizeOf(Row)), try need(s.qa, values) };
    for (outputs, 0..) |a, i| {
        for (outputs[0..i]) |b| if (overlap(a, b)) return error.DenseAlias;
        for (inputs) |b| if (overlap(a, b)) return error.DenseAlias;
        for (p.inputs[0..p.input_count]) |b| if (overlap(a, b)) return error.DenseAlias;
    }
}

fn need(v: View, n: usize) !View {
    if (v.ptr == 0 or v.ptr % 16 != 0 or v.len < n) return error.DenseScratch;
    _ = try std.math.add(u64, v.ptr, @intCast(n));
    return .{ .ptr = v.ptr, .len = n };
}

fn overlap(a: View, b: View) bool {
    return a.ptr < b.ptr + b.len and b.ptr < a.ptr + a.len;
}

test "dense segment plan keeps each row's cache position and chunk count" {
    const p = try Plan.make(&.{
        .{ .rows = 3, .host_pos = 509, .capacity = 4096, .cache = 0x1000, .position = 0x2000 },
        .{ .rows = 6, .host_pos = 511, .capacity = 4096, .cache = 0x3000, .position = 0x4000 },
    }, .{ .dense_limit = 2048, .scratch_rows = 64, .scratch_chunks = 4 });
    try std.testing.expectEqual(@as(usize, 9), p.rows);
    try std.testing.expectEqual(@as(usize, 2), p.chunks);
    try std.testing.expectEqual(@as(i32, 0), p.table[2].row0);
    try std.testing.expectEqual(@as(i32, 3), p.table[3].row0);
    try std.testing.expectEqual(@as(i32, 1), p.table[2].chunks);
    try std.testing.expectEqual(@as(i32, 2), p.table[3].chunks);
    try std.testing.expect((try Kernels.load(undefined, false)) == null);
}

test "dense segment admission refuses sparse crossing and unsupported scope before launch" {
    const t = std.testing;
    const seg = [_]Segment{.{ .rows = 8, .host_pos = 505, .capacity = 4096, .cache = 0x1000, .position = 0x2000 }};
    var scope: Scope = .{ .dense_limit = 512, .scratch_rows = 64, .scratch_chunks = 4 };
    try t.expectError(error.DenseScope, Plan.make(&seg, scope));
    scope.dense_limit = 2048;
    scope.paged = true;
    try t.expectError(error.DenseScope, Plan.make(&seg, scope));
    scope.paged = false;
    scope.prefill = true;
    try t.expectError(error.DenseScope, Plan.make(&seg, scope));
    scope.prefill = false;
    scope.host_kv = true;
    try t.expectError(error.DenseScope, Plan.make(&seg, scope));
    scope.host_kv = false;
    scope.scratch_chunks = 1;
    try t.expectError(error.DenseScratch, Plan.make(&seg, scope));
    try t.expect(!(try enabled(null)));
    try t.expect(!(try enabled("0")));
    try t.expect(try enabled("1"));
    try t.expectError(error.DenseSetting, enabled("yes"));
    try t.expectError(error.DenseSetting, enabled(""));
}

test "dense segment scratch rejects output aliasing immutable row tables and caches" {
    const p = try Plan.make(&.{.{ .rows = 1, .host_pos = 0, .capacity = 1, .cache = 0x1000, .position = 0x2000 }}, .{ .dense_limit = 512, .scratch_rows = 64, .scratch_chunks = 1 });
    var s: Storage = .{
        .table = .{ .ptr = 0x10000, .len = 32 },
        .qa = .{ .ptr = 0x20000, .len = 16384 },
        .po = .{ .ptr = 0x30000, .len = 32768 },
        .pm = .{ .ptr = 0x40000, .len = 64 },
        .pl = .{ .ptr = 0x50000, .len = 64 },
        .out = .{ .ptr = 0x60000, .len = 16384 },
    };
    try validate(&p, s);
    s.out.ptr = s.table.ptr;
    try std.testing.expectError(error.DenseAlias, validate(&p, s));
    s.out.ptr = 0x1000;
    try std.testing.expectError(error.DenseAlias, validate(&p, s));
}
