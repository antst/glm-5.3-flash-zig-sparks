//! Optional shared KDA wide1 wrappers; callers own per-layer table upload, pointers and completion lifetime.
const std = @import("std");
const cuda = @import("cuda");

pub const max_rows = 64;
pub const max_segments = 8;
pub const heads = 16; // canonical TP4 fixture; runtime plans admit16 or32 heads
pub const width = 128;
pub const View = struct { ptr: u64, len: usize };
pub const Row = extern struct {
    p: u64,
    conv: u64,
    state_in: u64,
    state_out: u64,
    k: u64,
    v: u64,
    g: u64,
    b: u64,
    row0: i32,
    rows: i32,
};
comptime {
    if (@sizeOf(Row) != 72 or @alignOf(Row) != 8 or @offsetOf(Row, "row0") != 64 or
        @offsetOf(Row, "rows") != 68) @compileError("KDA segment ABI");
}

pub const Segment = struct {
    row0: usize,
    rows: usize,
    p: View,
    conv: View,
    state_in: View,
    state_out: View,
    k: View,
    v: View,
    g: View,
    b: View,
};
pub const Parameters = struct {
    heads: usize,
    p_stride: usize,
    b_off: usize,
    a_stride: usize,
    g_stride: usize,
    conv_w: View,
    a_log: View,
    dt_bias: View,
    norm_w: View,
    lower: f32,
    eps: f32,
};
pub const Scope = struct { prefill: bool = false, paged: bool = false, wide: bool = true };
/// A/G are the already projected full shared rows. q/y/out must each have disjoint FULL-R storage.
pub const Storage = struct { table: View, a: View, gate: View, q: View, y: View, out: View };

pub const Plan = struct {
    table: [max_segments]Row = undefined,
    count: usize = 0,
    rows: usize = 0,
    most: usize = 0,
    params: Parameters,
    reads: [4 + 3 * max_segments]View = undefined,
    read_count: usize = 0,
    writes: [5 * max_segments]View = undefined,
    write_count: usize = 0,

    pub fn make(segs: []const Segment, p: Parameters, scope: Scope) !Plan {
        if (scope.prefill or scope.paged or !scope.wide) return error.KdaScope;
        if (segs.len == 0 or segs.len > max_segments) return error.KdaSegments;
        if (p.heads != 16 and p.heads != 32) return error.KdaShape;
        const h = p.heads;
        const n = h * width;
        if (p.b_off != 3 * n + 2 * width or p.p_stride < p.b_off + h or
            p.a_stride < n or p.g_stride < n) return error.KdaShape;
        for ([_]usize{ p.p_stride, p.b_off, p.a_stride, p.g_stride }) |v|
            if (v > std.math.maxInt(i32)) return error.KdaShape;
        var plan: Plan = .{ .params = p };
        try plan.read(p.conv_w, 3 * n * 4 * 4);
        try plan.read(p.a_log, h * 4);
        try plan.read(p.dt_bias, n * 4);
        try plan.read(p.norm_w, width * 2);
        for (segs) |s| {
            if (s.row0 != plan.rows or s.rows == 0 or s.rows > 8 or s.rows > max_rows - plan.rows)
                return error.KdaRows;
            try plan.read(s.p, try matrixBytes(s.rows, p.p_stride, p.b_off + h, 2));
            try plan.read(s.conv, 3 * 3 * n * 2); // 3 saved conv rows, each q|k|v
            try plan.read(s.state_in, h * width * width * 4);
            try plan.write(s.state_out, h * width * width * 4);
            try plan.write(s.k, s.rows * n * 4);
            try plan.write(s.v, s.rows * n * 2);
            try plan.write(s.g, s.rows * n * 4);
            try plan.write(s.b, s.rows * h * 4);
            plan.table[plan.count] = .{
                .p = s.p.ptr,
                .conv = s.conv.ptr,
                .state_in = s.state_in.ptr,
                .state_out = s.state_out.ptr,
                .k = s.k.ptr,
                .v = s.v.ptr,
                .g = s.g.ptr,
                .b = s.b.ptr,
                .row0 = @intCast(s.row0),
                .rows = @intCast(s.rows),
            };
            plan.count += 1;
            plan.rows += s.rows;
            plan.most = @max(plan.most, s.rows);
        }
        // Recurrent output and replay saves must not collide with another segment or any input.
        for (plan.writes[0..plan.write_count], 0..) |a, i| {
            for (plan.writes[0..i]) |b| if (overlap(a, b)) return error.KdaAlias;
            for (plan.reads[0..plan.read_count]) |b| if (overlap(a, b)) return error.KdaAlias;
        }
        return plan;
    }

    pub fn bytes(p: *const Plan) []const u8 {
        return std.mem.sliceAsBytes(p.table[0..p.count]);
    }
    fn read(p: *Plan, v: View, n: usize) !void {
        p.reads[p.read_count] = try need(v, n);
        p.read_count += 1;
    }
    fn write(p: *Plan, v: View, n: usize) !void {
        p.writes[p.write_count] = try need(v, n);
        p.write_count += 1;
    }
};

pub const Kernels = struct {
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    /// Disabled means no new symbol lookup, allocation, resource change or launch.
    pub fn load(module: cuda.Module, on: bool) !?Kernels {
        if (!on) return null;
        return .{
            .prep = try module.function("tf_glm_kda_seg_prep_v1"),
            .step = try module.function("tf_glm_kda_seg_step_w4_v1"),
            .out = try module.function("tf_glm_kda_seg_out_v1"),
        };
    }

    pub fn launch(k: *const Kernels, stream: cuda.Stream, p: *const Plan, s: Storage) !void {
        try validate(p, s); // all shape/storage/alias checks before the first new enqueue
        const t = p.params;
        var a: cuda.Args = .{};
        inline for (.{ s.table.ptr, @as(c_int, @intCast(p.count)), @as(c_int, @intCast(t.heads)), @as(c_int,
            @intCast(t.p_stride)), @as(c_int, @intCast(t.b_off)), s.a.ptr, @as(c_int, @intCast(t.a_stride)),
            t.conv_w.ptr, t.a_log.ptr, t.dt_bias.ptr, t.lower, s.q.ptr }) |v|
            a.add(v);
        const grid: cuda.launch.Dim3 = .{
            .x = @intCast(p.most),
            .y = @intCast(t.heads),
            .z = @intCast(p.count),
        };
        try cuda.launch.launch(k.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        inline for (.{ s.table.ptr, @as(c_int, @intCast(p.count)), @as(c_int, @intCast(t.heads)), s.q.ptr, s.y.ptr
            }) |v|
            b.add(v);
        try cuda.launch.launch(k.step, .{
            .grid = .{ .x = @intCast(t.heads), .y = width / 4 / 4, .z = @intCast(p.count) },
            .block = .{ .x = 4 * 32 },
        }, stream, &b);
        var o: cuda.Args = .{};
        inline for (.{ s.table.ptr, @as(c_int, @intCast(p.count)), @as(c_int, @intCast(t.heads)), s.y.ptr,
            s.gate.ptr, @as(c_int, @intCast(t.g_stride)), t.norm_w.ptr, t.eps, s.out.ptr }) |v| o.add(v);
        try cuda.launch.launch(k.out, .{ .grid = grid, .block = .{ .x = width } }, stream, &o);
    }
};

pub fn validate(p: *const Plan, s: Storage) !void {
    if (p.count == 0 or p.count > max_segments or p.rows == 0 or p.rows > max_rows or
        p.most == 0 or p.most > 8 or p.read_count > p.reads.len or p.write_count > p.writes.len)
        return error.KdaShape;
    if (p.params.heads != 16 and p.params.heads != 32) return error.KdaShape;
    const n = p.params.heads * width;
    const reads = [_]View{
        try need(s.table, p.count * @sizeOf(Row)),
        try need(s.a, try matrixBytes(p.rows, p.params.a_stride, n, 2)),
        try need(s.gate, try matrixBytes(p.rows, p.params.g_stride, n, 2)),
    };
    const writes = [_]View{ try need(s.q, p.rows * n * 4), try need(s.y, p.rows * n * 2), try need(s.out, p.rows * n
        * 2) };
    for (reads[1..]) |b| if (overlap(reads[0], b)) return error.KdaAlias;
    for (p.reads[0..p.read_count]) |b| if (overlap(reads[0], b)) return error.KdaAlias;
    for (writes, 0..) |a, i| {
        for (writes[0..i]) |b| if (overlap(a, b)) return error.KdaAlias;
        for (reads) |b| if (overlap(a, b)) return error.KdaAlias;
        for (p.reads[0..p.read_count]) |b| if (overlap(a, b)) return error.KdaAlias;
        for (p.writes[0..p.write_count]) |b| if (overlap(a, b)) return error.KdaAlias;
    }
    for (p.writes[0..p.write_count]) |a| for (reads) |b|
        if (overlap(a, b)) return error.KdaAlias;
}

fn matrixBytes(rows: usize, stride: usize, cols: usize, bytes: usize) !usize {
    return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows - 1, stride), cols), bytes);
}
fn need(v: View, n: usize) !View {
    if (v.ptr == 0 or v.ptr % 16 != 0 or v.len < n) return error.KdaStorage;
    _ = try std.math.add(u64, v.ptr, @intCast(n));
    return .{ .ptr = v.ptr, .len = n };
}
fn overlap(a: View, b: View) bool {
    return a.ptr < b.ptr + b.len and b.ptr < a.ptr + a.len;
}

fn fakeParameters() Parameters {
    return .{ .heads = heads, .p_stride = 6416, .b_off = 6400, .a_stride = 2048, .g_stride = 2048, .conv_w = .{ .ptr
        = 0x70000000, .len = 3 * 2048 * 4 * 4 }, .a_log = .{ .ptr = 0x71000000, .len = heads * 4 }, .dt_bias = .{
        .ptr = 0x72000000, .len = 2048 * 4 }, .norm_w = .{ .ptr = 0x73000000, .len = width * 2 }, .lower = -3.0,
        .eps = 1e-6 };
}
fn fakeSegment(i: usize, row0: usize, rows: usize) Segment {
    const b: u64 = 0x10000000 + i * 0x1000000;
    return .{ .row0 = row0, .rows = rows, .p = .{ .ptr = b, .len = 8 * 6416 * 2 }, .conv = .{ .ptr = b + 0x40000,
        .len = 3 * 3 * 2048 * 2 }, .state_in = .{ .ptr = b + 0x100000, .len = heads * width * width * 4 },
        .state_out = .{ .ptr = b + 0x200000, .len = heads * width * width * 4 }, .k = .{ .ptr = b + 0x300000, .len =
        8 * 2048 * 4 }, .v = .{ .ptr = b + 0x400000, .len = 8 * 2048 * 2 }, .g = .{ .ptr = b + 0x500000, .len = 8 *
        2048 * 4 }, .b = .{ .ptr = b + 0x600000, .len = 8 * heads * 4 } };
}
fn fakeStorage() Storage {
    return .{ .table = .{ .ptr = 0x80000000, .len = max_segments * @sizeOf(Row) }, .a = .{ .ptr = 0x81000000, .len =
        max_rows * 2048 * 2 }, .gate = .{ .ptr = 0x82000000, .len = max_rows * 2048 * 2 }, .q = .{ .ptr =
        0x83000000, .len = max_rows * 2048 * 4 }, .y = .{ .ptr = 0x84000000, .len = max_rows * 2048 * 2 }, .out = .{
        .ptr = 0x85000000, .len = max_rows * 2048 * 2 } };
}

test "KDA shared plan preserves unequal segments and refuses unsupported scope before enqueue" {
    const t = std.testing;
    var parts = [_]Segment{ fakeSegment(0, 0, 3), fakeSegment(1, 3, 6) };
    const p = try Plan.make(&parts, fakeParameters(), .{});
    try t.expectEqual(@as(usize, 9), p.rows);
    try t.expectEqual(@as(usize, 6), p.most);
    try t.expectEqual(@as(usize, 144), p.bytes().len);
    try t.expectEqual(@as(i32, 3), p.table[1].row0);
    try t.expectEqual(parts[1].state_out.ptr, p.table[1].state_out);
    try validate(&p, fakeStorage());
    try t.expect((try Kernels.load(undefined, false)) == null);
    try t.expectError(error.KdaScope, Plan.make(&parts, fakeParameters(), .{ .prefill = true }));
    try t.expectError(error.KdaScope, Plan.make(&parts, fakeParameters(), .{ .paged = true }));
    try t.expectError(error.KdaScope, Plan.make(&parts, fakeParameters(), .{ .wide = false }));
    parts[1].row0 = 4;
    try t.expectError(error.KdaRows, Plan.make(&parts, fakeParameters(), .{}));
    parts[1].row0 = 3;
    parts[1].state_out = parts[0].state_in;
    try t.expectError(error.KdaAlias, Plan.make(&parts, fakeParameters(), .{}));
    var s = fakeStorage();
    s.y.ptr = s.q.ptr;
    try t.expectError(error.KdaAlias, validate(&p, s));
    s = fakeStorage();
    s.q.len = 8 * 2048 * 4;
    try t.expectError(error.KdaStorage, validate(&p, s));
}

test "KDA shared launcher uses four-warp geometry with the original recurrence order and stops after a failed step" {
    const Fake = struct {
        var calls: usize = 0;
        var fail_at: usize = 0;
        fn u(p: ?[*]?*anyopaque, at: usize) u64 {
            return @as(*align(1) const u64, @ptrCast(p.?[at].?)).*;
        }
        fn i(p: ?[*]?*anyopaque, n: usize) i32 {
            return @as(*align(1) const i32, @ptrCast(p.?[n].?)).*;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32, shared: u32, stream:
            cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(by == 1 and bz == 1 and z == 2 and shared == 0 and stream == null);
            std.debug.assert(u(p, 0) == 0x80000000 and i(p, 1) == 2 and i(p, 2) == heads);
            const which = @intFromPtr(f);
            if (which == 1) {
                std.debug.assert(x == 6 and y == heads and bx == 512);
                std.debug.assert(i(p, 3) == 6416 and i(p, 4) == 6400 and u(p, 5) == 0x81000000);
                std.debug.assert(i(p, 6) == 2048 and u(p, 7) == 0x70000000);
                std.debug.assert(u(p, 8) == 0x71000000 and u(p, 9) == 0x72000000 and u(p, 11) == 0x83000000);
            } else if (which == 2) {
                std.debug.assert(x == heads and y == 8 and bx == 128);
                std.debug.assert(u(p, 3) == 0x83000000 and u(p, 4) == 0x84000000);
            } else {
                std.debug.assert(which == 3 and x == 6 and y == heads and bx == width);
                std.debug.assert(u(p, 3) == 0x84000000 and u(p, 4) == 0x82000000);
                std.debug.assert(i(p, 5) == 2048 and u(p, 6) == 0x73000000 and u(p, 8) == 0x85000000);
            }
            calls += 1;
            return if (calls == fail_at) cuda.abi.error_not_ready else 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const k: Kernels = .{ .prep = .{ .d = &d, .handle = @ptrFromInt(1) }, .step = .{ .d = &d, .handle = @ptrFromInt(
        2) }, .out = .{ .d = &d, .handle = @ptrFromInt(3) } };
    const p = try Plan.make(&.{ fakeSegment(0, 0, 3), fakeSegment(1, 3, 6) }, fakeParameters(), .{});
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.calls = 0;
    Fake.fail_at = 0;
    try k.launch(stream, &p, fakeStorage());
    try std.testing.expectEqual(@as(usize, 3), Fake.calls);
    Fake.calls = 0;
    Fake.fail_at = 2;
    try std.testing.expectError(error.NotReady, k.launch(stream, &p, fakeStorage()));
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
}
