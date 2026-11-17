//! Explicit old-wide1 vs separate-module KDA segment byte comparator; no model or default contract changes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const seg = glm.kda_segments;
const Rig = @import("contract_rig.zig").Rig;
const check = @import("check.zig");
const limits = core.contract.limits;
const H = 32;
const PITCH = 3 * N + 256 + H;
const B_OFF = 3 * N + 256;
const D = 128;
const N = H * D;
const state_bytes = H * D * D * 4;
const Save = struct { state: seg.View, k: seg.View, v: seg.View, g: seg.View, b: seg.View };
const Input = struct { p: seg.View, conv: seg.View, state: seg.View };
const List = struct { views: [43]seg.View = undefined, count: usize = 0 };
const Totals = struct { cases: usize = 0, updates: usize = 0, repeats: usize = 0, views: usize = 0, controls: usize
    = 0 };

fn alloc(r: *Rig, len: usize, input: ?[]const u8) !seg.View {
    return .{ .ptr = if (input) |b| try r.input(b) else try r.output(len), .len = len };
}
fn upload(r: *Rig, ptr: u64, data: []const u8) !void {
    try r.s.synchronize();
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = ptr, .len = data.len }).upload(0, data);
}
fn bf(z: i32, denom: f32) u16 {
    return core.contract.spec.toBf16(@as(f32, @floatFromInt(z)) / denom);
}
const Outputs = struct {
    q: seg.View,
    y: seg.View,
    out: seg.View,
    save: [8]Save = undefined,
    fn init(r: *Rig, R: usize, S: usize) !Outputs {
        var o: Outputs = .{ .q = try alloc(r, (R + 1) * N * 4, null), .y = try alloc(r, (R + 1) * N * 2, null), .out
            = try alloc(r, (R + 1) * N * 2, null) };
        for (0..S) |i| o.save[i] = .{ .state = try alloc(r, state_bytes, null), .k = try alloc(r, 9 * N * 4, null),
            .v = try alloc(r, 9 * N * 2, null), .g = try alloc(r, 9 * N * 4, null), .b = try alloc(r, 9 * H * 4,
            null) };
        return o;
    }
    fn list(o: *const Outputs, S: usize) List {
        var a: List = .{};
        for ([_]seg.View{ o.q, o.y, o.out }) |v| {
            a.views[a.count] = v;
            a.count += 1;
        }
        for (o.save[0..S]) |s| for ([_]seg.View{ s.state, s.k, s.v, s.g, s.b }) |v| {
            a.views[a.count] = v;
            a.count += 1;
        };
        return a;
    }
    fn poison(o: *const Outputs, r: *Rig, S: usize) !void {
        const a = o.list(S);
        for (a.views[0..a.count]) |v|
            try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = v.ptr, .len = v.len }).fill8(0xff, r.s.handle);
    }
};

const Fixture = struct {
    rig: Rig,
    R: usize,
    S: usize,
    params: seg.Parameters,
    input: [8]Input = undefined,
    a: seg.View,
    gate: seg.View,
    table: seg.View,
    old: Outputs,
    new: Outputs,

    fn init(gpu: check.Gpu, stream: cuda.Stream, lengths: []const usize, padded: bool, nonfinite: bool) !Fixture {
        const gpa = gpu.gpa;
        var R: usize = 0;
        for (lengths) |n| R += n;
        var f: Fixture = .{ .rig = .{ .gpu = gpu, .s = stream }, .R = R, .S = lengths.len, .params = undefined, .a =
            undefined, .gate = undefined, .table = undefined, .old = undefined, .new = undefined };
        errdefer f.rig.deinit();
        errdefer cuda.completion.drain(gpu.d, stream, "KDA fixture init");
        const cw = try gpa.alloc(f32, 3 * N * 4);
        defer gpa.free(cw);
        for (cw, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 16;
        const logs = try gpa.alloc(f32, H);
        defer gpa.free(logs);
        for (logs, 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i + 1)) / 32;
        const bias = try gpa.alloc(f32, N);
        defer gpa.free(bias);
        for (bias, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 32;
        const norm = try gpa.alloc(u16, D);
        defer gpa.free(norm);
        for (norm, 0..) |*v, i| v.* = bf(@intCast(16 + i % 9), 24);
        const params: seg.Parameters = .{
            .heads = H,
            .p_stride = if (padded) (PITCH + 16) else PITCH,
            .b_off = B_OFF,
            .a_stride = if (padded) N + 32 else N,
            .g_stride = if (padded) N + 16 else N,
            .conv_w = try alloc(&f.rig, cw.len * 4, std.mem.sliceAsBytes(cw)),
            .a_log = try alloc(&f.rig, logs.len * 4, std.mem.sliceAsBytes(logs)),
            .dt_bias = try alloc(&f.rig, bias.len * 4, std.mem.sliceAsBytes(bias)),
            .norm_w = try alloc(&f.rig, norm.len * 2, std.mem.sliceAsBytes(norm)),
            .lower = -3,
            .eps = 1e-6,
        };
        const av = try gpa.alloc(u16, (R + 1) * (N + 32));
        defer gpa.free(av);
        const gv = try gpa.alloc(u16, av.len);
        defer gpa.free(gv);
        @memset(av, 0x7fc1);
        @memset(gv, 0x7fc1);
        for (0..R) |row| for (0..N) |col| {
            av[row * params.a_stride + col] = bf(@as(i32, @intCast((row * 11 + col * 7) % 31)) - 15, 16);
            gv[row * params.g_stride + col] = bf(@as(i32, @intCast((row * 17 + col * 3) % 37)) - 18, 16);
        };
        if (nonfinite) {
            av[7] = 0x7f80;
            gv[9] = 0x7fc1;
        }
        f.params = params;
        f.a = try alloc(&f.rig, av.len * 2, std.mem.sliceAsBytes(av));
        f.gate = try alloc(&f.rig, gv.len * 2, std.mem.sliceAsBytes(gv));
        for (0..f.S) |seq| {
            const pv = try gpa.alloc(u16, 9 * params.p_stride);
            defer gpa.free(pv);
            @memset(pv, 0x7fc1);
            for (0..8) |row| {
                for (0..PITCH) |col|
                    pv[row * params.p_stride + col] = bf(@as(i32, @intCast((row * 19 + col * 3 + seq * 11) % 43)) -
                        21, 32);
            }
            if (nonfinite and seq == 0) pv[3] = 0x7fc1;
            const cv = try gpa.alloc(u16, 3 * 3 * N);
            defer gpa.free(cv);
            for (cv, 0..) |*v, i| v.* = bf(@as(i32, @intCast((i * 7 + seq * 13) % 47)) - 23, 32);
            const state = try gpa.alloc(f32, state_bytes / 4);
            defer gpa.free(state);
            for (state, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3 + seq * 29) % 97)) - 48))
                / 4096;
            if (nonfinite and seq == 0) state[13] = std.math.nan(f32);
            f.input[seq] = .{ .p = try alloc(&f.rig, pv.len * 2, std.mem.sliceAsBytes(pv)), .conv = try alloc(
                &f.rig, cv.len * 2, std.mem.sliceAsBytes(cv)), .state = try alloc(&f.rig, state.len * 4,
                std.mem.sliceAsBytes(state)) };
        }
        f.table = .{ .ptr = try f.rig.scratch((f.S + 1) * @sizeOf(seg.Row)), .len = (f.S + 1) * @sizeOf(seg.Row) };
        f.old = try Outputs.init(&f.rig, R, f.S);
        f.new = try Outputs.init(&f.rig, R, f.S);
        return f;
    }

    fn storage(f: *const Fixture) seg.Storage {
        return .{ .table = f.table, .a = f.a, .gate = f.gate, .q = f.new.q, .y = f.new.y, .out = f.new.out };
    }
    fn plan(f: *const Fixture, lengths: []const usize, update: usize) !seg.Plan {
        var parts: [8]seg.Segment = undefined;
        var row0: usize = 0;
        for (lengths, 0..) |n, i| {
            const slot = (i + update) % f.S;
            const v = f.input[slot];
            const s = f.new.save[slot];
            parts[i] = .{ .row0 = row0, .rows = n, .p = v.p, .conv = v.conv, .state_in = v.state, .state_out =
                s.state, .k = s.k, .v = s.v, .g = s.g, .b = s.b };
            row0 += n;
        }
        return seg.Plan.make(parts[0..f.S], f.params, .{});
    }
};

fn original(f: *Fixture, k: *const glm.kda.Kernels, lengths: []const usize, update: usize) !void {
    try f.old.poison(&f.rig, f.S);
    var row0: usize = 0;
    const p = f.params;
    for (lengths, 0..) |n, i| {
        const slot = (i + update) % f.S;
        const v = f.input[slot];
        const s = f.old.save[slot];
        const c: glm.kda.Chain = .{ .p = v.p.ptr, .p_stride = p.p_stride, .b_off = p.b_off, .a = f.a.ptr + row0 *
            p.a_stride * 2, .a_stride = p.a_stride, .g = f.gate.ptr + row0 * p.g_stride * 2, .g_stride = p.g_stride,
            .conv_state = v.conv.ptr, .conv_w = p.conv_w.ptr, .state_in = v.state.ptr, .a_log = p.a_log.ptr,
            .dt_bias = p.dt_bias.ptr, .norm_w = p.norm_w.ptr, .eps = p.eps, .lower = p.lower, .heads = H };
        try glm.kda.chain(k, f.rig.s, c, n, .{ .out = f.old.out.ptr + row0 * N * 2, .state_out = s.state.ptr, .k =
            s.k.ptr, .v = s.v.ptr, .g = s.g.ptr, .b = s.b.ptr }, .{ .q = f.old.q.ptr + row0 * N * 4, .y =
            f.old.y.ptr + row0 * N * 2 });
        row0 += n;
    }
    try f.rig.s.synchronize();
}

fn poisonTail(r: *Rig, v: seg.View, used: usize) !void {
    for ((try r.bytes(v.ptr, v.len))[used..]) |b|
        try check.expect(b == 0xff, "original KDA unused row padding", .{});
}
fn oldPadding(f: *Fixture, lengths: []const usize, update: usize) !void {
    for ([_]seg.View{ f.old.q, f.old.y, f.old.out }, [_]usize{ 4, 2, 2 }) |v, bytes|
        try poisonTail(&f.rig, v, f.R * N * bytes);
    for (lengths, 0..) |n, i| {
        const s = f.old.save[(i + update) % f.S];
        for ([_]seg.View{ s.k, s.v, s.g, s.b }, [_]usize{ N * 4, N * 2, N * 4, H * 4 }) |v, width|
            try poisonTail(&f.rig, v, n * width);
    }
}
const Snapshot = struct {
    gpa: std.mem.Allocator,
    values: [][]u8,
    fn init(f: *Fixture) !Snapshot {
        const views = f.old.list(f.S);
        const values = try f.rig.gpu.gpa.alloc([]u8, views.count);
        var done: usize = 0;
        errdefer {
            for (values[0..done]) |v| f.rig.gpu.gpa.free(v);
            f.rig.gpu.gpa.free(values);
        }
        for (views.views[0..views.count], 0..) |v, i| {
            values[i] = try f.rig.gpu.gpa.dupe(u8, try f.rig.bytes(v.ptr, v.len));
            done += 1;
        }
        return .{ .gpa = f.rig.gpu.gpa, .values = values };
    }
    fn deinit(s: Snapshot) void {
        for (s.values) |v| s.gpa.free(v);
        s.gpa.free(s.values);
    }
    fn compare(s: Snapshot, f: *Fixture, equal: bool) !void {
        const views = f.new.list(f.S);
        var same = true;
        for (views.views[0..views.count], s.values) |v, want| {
            const got = try f.rig.bytes(v.ptr, v.len);
            if (equal) try check.sameBytes("KDA q/y/out/state/replay/padding", got, want);
            same = same and std.mem.eql(u8, got, want);
        }
        try check.expect(same == equal, "KDA planted control must change compared bytes", .{});
    }
};

fn table(f: *Fixture, p: *const seg.Plan) !void {
    try (cuda.DeviceBuffer{ .d = f.rig.gpu.d, .ptr = f.table.ptr, .len = f.table.len }).fill8(0xa5, f.rig.s.handle);
    try upload(&f.rig, f.table.ptr, p.bytes());
}
fn tableIntact(f: *Fixture, p: *const seg.Plan) !void {
    try check.sameBytes("KDA device table read-only", try f.rig.bytes(f.table.ptr, p.bytes().len), p.bytes());
    for (try f.rig.bytes(f.table.ptr + p.bytes().len, @sizeOf(seg.Row))) |b|
        try check.expect(b == 0xa5, "KDA table padding", .{});
}
fn controls(f: *Fixture, k: *const seg.Kernels, p: *const seg.Plan, want: Snapshot) !usize {
    for (0..3) |fault| {
        var bad = p.*;
        if (fault == 0) {
            bad.table[0].row0 = 6;
            bad.table[1].row0 = 0;
        } // disjoint3+6 row permutation
        if (fault == 1) bad.table[1].state_in = bad.table[0].state_in; // allocated, read-only substitute
        if (fault == 2) bad.params.g_stride += 16; // spare allocated gate rows, no output address change
        try table(f, &bad);
        try f.new.poison(&f.rig, f.S);
        try k.launch(f.rig.s, &bad, f.storage());
        try want.compare(f, false);
        try tableIntact(f, &bad);
        try check.expect(try f.rig.intact(), "KDA control allocation guards", .{});
    }
    return 3;
}

fn finiteWritten(f: *Fixture) !void {
    const words = try f.rig.bytes(f.old.q.ptr, f.R * N * 4);
    for (std.mem.bytesAsSlice(u32, words)) |bits|
        try check.expect(std.math.isFinite(@as(f32, @bitCast(bits))), "KDA finite reference q write", .{});
    for ([_]seg.View{ f.old.y, f.old.out }) |v| {
        const values = try f.rig.bytes(v.ptr, f.R * N * 2);
        for (std.mem.bytesAsSlice(u16, values)) |bits|
            try check.expect(bits & 0x7f80 != 0x7f80, "KDA finite reference BF16 write", .{});
    }
}

fn one(gpu: check.Gpu, stream: cuda.Stream, old: *const glm.kda.Kernels, k: *const seg.Kernels, lengths: []const
    usize, padded: bool, nonfinite: bool, negative: bool, totals: *Totals) !void {
    var f = try Fixture.init(gpu, stream, lengths, padded, nonfinite);
    defer f.rig.deinit();
    defer cuda.completion.drain(gpu.d, stream, "KDA fixture finish");
    const inputs = try gpu.gpa.dupe(u8, try f.rig.inputBytes());
    defer gpu.gpa.free(inputs);
    var lens: [8]usize = undefined;
    @memcpy(lens[0..lengths.len], lengths);
    for (0..2) |update| {
        const p = try f.plan(lens[0..f.S], update);
        try original(&f, old, lens[0..f.S], update);
        try oldPadding(&f, lens[0..f.S], update);
        if (!nonfinite) try finiteWritten(&f);
        const want = try Snapshot.init(&f);
        defer want.deinit();
        try table(&f, &p);
        for (0..5) |_| {
            try f.new.poison(&f.rig, f.S);
            try k.launch(stream, &p, f.storage());
            try want.compare(&f, true);
            try tableIntact(&f, &p);
            totals.repeats += 1;
            totals.views += want.values.len;
        }
        if (negative and update == 0) totals.controls += try controls(&f, k, &p, want);
        try check.sameBytes("KDA immutable P/A/G/conv/state/weights", try f.rig.inputBytes(), inputs);
        try check.expect(try f.rig.intact(), "KDA reference/candidate/input guards", .{});
        std.mem.reverse(usize, lens[0..f.S]);
        totals.updates += 1;
    }
    totals.cases += 1;
    std.debug.print("PASS KDA-SEGMENTED S{d} R{d} padded{} nonfinite{}\n", .{ f.S, f.R, padded, nonfinite });
}

fn resources(gpu: check.Gpu, old: *const glm.kda.Kernels, new: *const seg.Kernels) !void {
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    const fs = [_]cuda.Function{ old.prep, old.step, old.out, new.prep, new.step, new.out };
    const ts = [_]u32{ 512, 256, 128, 512, 128, 128 };
    const names = [_][]const u8{ "tf_glm_kda_prep", "tf_glm_kda_step", "tf_glm_kda_out", "tf_glm_kda_seg_prep_v1",
        "tf_glm_kda_seg_step_w4_v1", "tf_glm_kda_seg_out_v1" };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var errors: usize = 0;
    for (fs, ts, names) |f, threads, name| {
        const u: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(
            .shared_size_bytes)), .local = @intCast(try f.attribute(.local_size_bytes)), .bound = @intCast(try
            f.attribute(.max_threads_per_block)) };
        errors += try limits.cuda(.{ .name = name, .symbol = name, .image = "separate KDA byte comparator", .threads
            = threads, .dynamic = 0 }, u, target, &log, gpu.gpa);
        std.debug.print("RESOURCE KDA-SEGMENTED {s} sm{d} regs{d} static{d} local{d} bound{d}\n", .{ name, cap,
            u.regs, u.static, u.local, u.bound });
    }
    std.debug.print("{s}", .{log.items});
    try check.expect(errors == 0, "KDA live resource limits", .{});
}
fn candidate(gpu: check.Gpu, path: ?[]const u8) !cuda.Module {
    const name = path orelse return cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, name, gpu.gpa, .limited(64 << 20));
    defer gpu.gpa.free(bytes);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", bytes.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, bytes);
    return cuda.Module.load(gpu.d, aligned);
}
pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len > 1) return error.KdaContractArguments;
    const path: ?[]const u8 = if (rest.len == 1) rest[0] else null;
    var reference = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
    defer reference.unload();
    var module = try candidate(gpu, path);
    defer module.unload();
    const k = (try seg.Kernels.load(module, true)).?;
    const old = try glm.kda.Kernels.load(reference);
    try resources(gpu, &old, &k);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    defer cuda.completion.drain(gpu.d, stream, "KDA contract stream");
    std.debug.print("KDA-SEGMENTED reference=embedded-glm_kda candidate={s}\n", .{path orelse
        "embedded; new symbols required"});
    var totals: Totals = .{};
    var lens: [8]usize = undefined;
    for (1..9) |S| for (0..3) |style| {
        for (lens[0..S], 0..) |*n, i| n.* = switch (style) {
            0 => 1,
            1 => 8,
            else => (i * 5 + 3) % 8 + 1,
        };
        try one(gpu, stream, &old, &k, lens[0..S], style == 2, false, false, &totals);
    };
    for ([_][]const usize{ &.{ 7, 8, 8, 8 }, &.{ 1, 8, 8, 8, 8 }, &.{ 7, 8, 8, 8, 8, 8, 8, 8 } }) |tail|
        try one(gpu, stream, &old, &k, tail, true, false, false, &totals);
    try one(gpu, stream, &old, &k, &.{ 3, 6 }, false, false, true, &totals);
    try one(gpu, stream, &old, &k, &.{ 1, 7 }, true, true, false, &totals);
    try check.expect(totals.controls == 3, "all3 host-safe KDA controls caught", .{});
    std.debug.print("KDA-SEGMENTED H32 SUMMARY cases{d} updates{d} repeats{d} controls{d}/3 views{d};" ++
        "device byte replacement only; FP64/stress/allSM/model gates separate\n", .{ totals.cases, totals.updates,
            totals.repeats, totals.controls, totals.views });
}

test "KDA TP2 contract original reference uses actual wide chain and rotated per-state row bindings" {
    const Fake = struct {
        var calls: usize = 0;
        fn fill(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn u(p: ?[*]?*anyopaque, at: usize) u64 {
            return @as(*align(1) const u64, @ptrCast(p.?[at].?)).*;
        }
        fn int(p: ?[*]?*anyopaque, at: usize) i32 {
            return @as(*align(1) const i32, @ptrCast(p.?[at].?)).*;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32, shared: u32, stream:
            cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const part = calls / 3;
            const slot = 1 - part;
            const row0: u64 = if (part == 0) 0 else 3;
            const rows: u32 = if (part == 0) 3 else 6;
            const ib: u64 = 0x10000000 + slot * 0x1000000;
            const ob: u64 = 0x50000000 + slot * 0x1000000;
            std.debug.assert(z == 1 and by == 1 and bz == 1 and shared == 0 and stream == null);
            std.debug.assert(int(p, 0) == H);
            if (calls % 3 == 0) {
                std.debug.assert(@intFromPtr(f) == 1 and x == rows and y == H and bx == 512);
                std.debug.assert(u(p, 1) == ib and int(p, 2) == PITCH and int(p, 3) == B_OFF);
                std.debug.assert(u(p, 4) == 0x30000000 + row0 * N * 2 and int(p, 5) == N);
                std.debug.assert(u(p, 6) == ib + 0x40000 and u(p, 11) == 0x40000000 + row0 * N * 4);
                std.debug.assert(u(p, 12) == ob + 0x200000 and u(p, 13) == ob + 0x300000);
                std.debug.assert(u(p, 14) == ob + 0x400000 and u(p, 15) == ob + 0x500000);
            } else if (calls % 3 == 1) {
                std.debug.assert(@intFromPtr(f) == 2 and x == H and y == 4 and bx == 256);
                std.debug.assert(u(p, 1) == ib + 0x100000 and u(p, 2) == 0x40000000 + row0 * N * 4);
                std.debug.assert(int(p, 7) == @as(i32, @intCast(rows)));
                std.debug.assert(u(p, 8) == 0x41000000 + row0 * N * 2 and u(p, 9) == ob);
            } else {
                std.debug.assert(@intFromPtr(f) == 3 and x == rows and y == H and bx == D);
                std.debug.assert(u(p, 1) == 0x41000000 + row0 * N * 2);
                std.debug.assert(u(p, 2) == 0x31000000 + row0 * N * 2 and int(p, 3) == N);
                std.debug.assert(u(p, 6) == 0x42000000 + row0 * N * 2);
            }
            calls += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemsetD8Async = Fake.fill;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuLaunchKernel = Fake.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = .{
        .rig = .{ .gpu = .{ .d = &d, .ctx = undefined, .gpa = std.testing.allocator, .io = undefined }, .s = stream },
        .R = 9,
        .S = 2,
        .table = .{ .ptr = 0x60000000, .len = 3 * 72 },
        .params = .{ .heads = H, .p_stride = PITCH, .b_off = B_OFF, .a_stride = N, .g_stride = N, .conv_w = .{ .ptr
            = 0x70000000, .len = 3 * N * 4 * 4 }, .a_log = .{ .ptr = 0x71000000, .len = H * 4 }, .dt_bias = .{ .ptr
            = 0x72000000, .len = N * 4 }, .norm_w = .{ .ptr = 0x73000000, .len = D * 2 }, .lower = -3, .eps = 1e-6 },
        .a = .{ .ptr = 0x30000000, .len = 10 * N * 2 },
        .gate = .{ .ptr = 0x31000000, .len = 10 * N * 2 },
        .new = undefined,
        .old = .{ .q = .{ .ptr = 0x40000000, .len = 10 * N * 4 }, .y = .{ .ptr = 0x41000000, .len = 10 * N * 2 },
            .out = .{ .ptr = 0x42000000, .len = 10 * N * 2 } },
    };
    for (0..2) |i| {
        const ib: u64 = 0x10000000 + i * 0x1000000;
        const ob: u64 = 0x50000000 + i * 0x1000000;
        f.input[i] = .{ .p = .{ .ptr = ib, .len = 9 * PITCH * 2 }, .conv = .{ .ptr = ib + 0x40000, .len = 3 * 3 * N
            * 2 }, .state = .{ .ptr = ib + 0x100000, .len = state_bytes } };
        f.old.save[i] = .{ .state = .{ .ptr = ob, .len = state_bytes }, .k = .{ .ptr = ob + 0x200000, .len = 9 * N *
            4 }, .v = .{ .ptr = ob + 0x300000, .len = 9 * N * 2 }, .g = .{ .ptr = ob + 0x400000, .len = 9 * N * 4 },
            .b = .{ .ptr = ob + 0x500000, .len = 9 * H * 4 } };
    }
    f.new = f.old;
    const p = try f.plan(&.{ 3, 6 }, 1);
    try std.testing.expectEqual(f.input[1].p.ptr, p.table[0].p);
    try std.testing.expectEqual(f.new.save[0].state.ptr, p.table[1].state_out);
    const ks: glm.kda.Kernels = .{ .chain = undefined, .replay = undefined, .replay_layers = undefined, .prep = .{
        .d = &d, .handle = @ptrFromInt(1) }, .step = .{ .d = &d, .handle = @ptrFromInt(2) }, .out = .{ .d = &d,
        .handle = @ptrFromInt(3) } };
    Fake.calls = 0;
    try original(&f, &ks, &.{ 3, 6 }, 1);
    try std.testing.expectEqual(@as(usize, 6), Fake.calls);
}
