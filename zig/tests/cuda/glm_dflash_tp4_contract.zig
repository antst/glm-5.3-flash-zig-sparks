//! Explicit new-asset command: local TP4 heads against full-head operators, not a model or graph certificate.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("glm_dflash_tp4_ref.zig");
const c = core.contract.spec;
const limits = core.contract.limits;
const scale: f32 = 1.0 / @sqrt(@as(f32, 128));
const full_ld = 6144;
const local_ld = 1536;

pub const Fns = struct {
    qkv: cuda.Function,
    part: cuda.Function,
    merge: cuda.Function,
    pub fn load(m: cuda.Module, local: bool) !Fns {
        const names: [3][:0]const u8 = if (local) .{ "df_block_qkv_tp4_v1", "df_attn_part_tp4_v1", "df_attn_merge_tp4_v1" } else .{ "df_block_qkv", "df_attn_part", "df_attn_merge" };
        var fs: [3]cuda.Function = undefined;
        for (names, &fs) |name, *f| {
            var handle: cuda.abi.Function = null;
            const res = m.d.api.cuModuleGetFunction(&handle, m.handle, name);
            if (res == cuda.abi.error_not_found) return error.MissingDFlashAttentionKernel;
            try m.d.check(res, "cuModuleGetFunction(dflash own-head)");
            f.* = .{ .d = m.d, .handle = handle };
        }
        return .{ .qkv = fs[0], .part = fs[1], .merge = fs[2] };
    }
};
const Field = enum { q, k, v, pm, pl, acc, out };
const Outputs = struct {
    ptrs: [7]u64,
    lens: [7]usize,
    fn init(rig: *Rig, rows: usize, local: bool) !Outputs {
        const qh: usize = if (local) 8 else 32;
        const kh: usize = if (local) 2 else 8;
        var o: Outputs = .{ .ptrs = undefined, .lens = .{ rows * qh * 256, rows * kh * 256, rows * kh * 256, rows * qh * 64, rows * qh * 64, rows * qh * 8192, rows * qh * 256 } };
        for (&o.ptrs, o.lens) |*p, n| p.* = try rig.output(n);
        return o;
    }
    fn ptr(o: Outputs, field: Field) u64 {
        return o.ptrs[@backingInt(field)];
    }
};
const Inputs = struct { ldq: c_int, qkv: u64, qn: u64, kn: u64, pos: u64, rings: u64, starts: u64, froms: u64 };

fn go(f: cuda.Function, grid: cuda.Dim3, threads: u32, s: cuda.Stream, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}
fn qkv(f: Fns, s: cuda.Stream, i: Inputs, o: Outputs, rows: usize, local: bool, wrong_old_grid: bool) !void {
    var a: cuda.Args = .{};
    a.add(i.qkv);
    a.add(i.ldq);
    a.add(i.qn);
    a.add(i.kn);
    a.add(i.pos);
    a.add(@as(f32, 1e-5));
    a.add(@as(f32, 10000));
    for ([_]u64{ o.ptr(.q), o.ptr(.k), o.ptr(.v) }) |p| a.add(p);
    try go(f.qkv, .{ .x = @intCast(rows), .y = if (local or wrong_old_grid) 10 else 40 }, 64, s, &a);
}
fn part(f: Fns, s: cuda.Stream, i: Inputs, o: Outputs, streams: usize, local: bool, layer: usize, kh0: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ o.ptr(.q), o.ptr(.k), o.ptr(.v), i.rings, i.starts, i.froms }) |p| a.add(p);
    a.add(@as(c_int, @intCast(layer)));
    a.add(scale);
    for ([_]u64{ o.ptr(.pm), o.ptr(.pl), o.ptr(.acc) }) |p| a.add(p);
    if (local) a.add(@as(c_int, @intCast(kh0)));
    try go(f.part, .{ .x = @intCast(streams), .y = if (local) 2 else 8, .z = 16 }, 256, s, &a);
}
fn merge(f: Fns, s: cuda.Stream, o: Outputs, rows: usize, local: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ o.ptr(.pm), o.ptr(.pl), o.ptr(.acc), o.ptr(.out) }) |p| a.add(p);
    try go(f.merge, .{ .x = @intCast(rows), .y = if (local) 8 else 32 }, 128, s, &a);
}
fn pipeline(f: Fns, s: cuda.Stream, i: Inputs, o: Outputs, streams: usize, local: bool, case: ref.Case, rank: usize) !void {
    if (local) try ref.grid(rank < 4, 10, 2, 8);
    try qkv(f, s, i, o, streams * 8, local, false);
    try part(f, s, i, o, streams, local, case.layer, 2 * rank);
    try merge(f, s, o, streams * 8, local);
    try s.synchronize();
}
fn data(a: std.mem.Allocator, n: usize, seed: u64) ![]u16 {
    const x = try a.alloc(u16, n);
    for (x, 0..) |*v, j| {
        const hash = (@as(u64, j) *% 0x9e3779b97f4a7c15) ^ seed;
        const signed: i32 = @as(i32, @intCast((hash >> 24) % 127)) - 63;
        v.* = c.toBf16(@as(f32, @floatFromInt(signed)) / 64);
    }
    return x;
}
fn seen(a: std.mem.Allocator, rig: *Rig, o: Outputs, field: Field) ![]u8 {
    const at = @backingInt(field);
    return a.dupe(u8, try rig.bytes(o.ptrs[at], o.lens[at]));
}
fn words(a: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const out = try a.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}
fn selected(a: std.mem.Allocator, all: []const u8, rows: usize, field: Field, rank: usize) ![]u8 {
    const kv = field == .k or field == .v;
    const heads: usize = if (kv) 8 else 32;
    const unit: usize = switch (field) {
        .pm, .pl => 64,
        .acc => 8192,
        else => 256,
    };
    const out = try a.alloc(u8, rows * heads / 4 * unit);
    errdefer a.free(out);
    try ref.select(out, all, rows, heads, heads / 4, rank, unit);
    return out;
}
fn empties(pm: []const u8, pl: []const u8, acc: []const u8, rows: usize, case: ref.Case) !usize {
    var n: usize = 0;
    for (0..rows) |r| {
        const ct = ref.counts(case.start, case.from, r % 8);
        for (0..8) |h| for (0..16) |ch| if (ch * ct.piece >= ct.total) {
            const at = (r * 8 + h) * 16 + ch;
            try check.expect(std.mem.readInt(u32, pm[at * 4 ..][0..4], .little) == 0xff800000, "empty chunk maximum", .{});
            try check.expect(std.mem.readInt(u32, pl[at * 4 ..][0..4], .little) == 0, "empty chunk denominator", .{});
            for (acc[at * 512 ..][0..512]) |v| try check.expect(v == 0, "empty chunk accumulator", .{});
            n += 1;
        };
    }
    return n;
}

fn one(gpu: check.Gpu, s: cuda.Stream, old: Fns, new: Fns, streams: usize, case: ref.Case, nonfinite: bool, padded: bool) !usize {
    const a = gpu.gpa;
    const rows = streams * 8;
    const full_pitch = full_ld + @as(usize, if (padded) 64 else 0);
    const local_pitch = local_ld + @as(usize, if (padded) 64 else 0);
    var rig: Rig = .{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const raw = try data(a, rows * full_pitch, 0x12);
    defer a.free(raw);
    if (nonfinite) raw[17] = 0x7fc1;
    const norms = try a.alloc(u16, 256);
    defer a.free(norms);
    for (norms, 0..) |*v, j| v.* = c.toBf16(1 + @as(f32, @floatFromInt(j % 9)) / 64);
    const positions = try a.alloc(i32, rows);
    defer a.free(positions);
    for (positions, 0..) |*p, j| p.* = @intCast(case.start + @as(i64, @intCast(j % 8)));
    const ring = try data(a, streams * ref.ring_cells, 0xf00d);
    defer a.free(ring);
    const ring_at = try rig.input(std.mem.sliceAsBytes(ring));
    const addresses = try a.alloc(u64, streams);
    defer a.free(addresses);
    for (addresses, 0..) |*p, j| p.* = ring_at + j * ref.ring_cells * 2;
    const starts = try a.alloc(i64, streams);
    defer a.free(starts);
    const froms = try a.alloc(i64, streams);
    defer a.free(froms);
    @memset(starts, case.start);
    @memset(froms, case.from);
    const input: Inputs = .{
        .ldq = @intCast(full_pitch),
        .qkv = try rig.input(std.mem.sliceAsBytes(raw)),
        .qn = try rig.input(std.mem.sliceAsBytes(norms[0..128])),
        .kn = try rig.input(std.mem.sliceAsBytes(norms[128..])),
        .pos = try rig.input(std.mem.sliceAsBytes(positions)),
        .rings = try rig.input(std.mem.sliceAsBytes(addresses)),
        .starts = try rig.input(std.mem.sliceAsBytes(starts)),
        .froms = try rig.input(std.mem.sliceAsBytes(froms)),
    };
    var local_inputs: [4]Inputs = undefined;
    const compact = try a.alloc(u16, rows * local_pitch);
    defer a.free(compact);
    for (&local_inputs, 0..) |*in, rank| {
        @memset(compact, 0x7fc1);
        try ref.compact(compact, local_pitch, raw, full_pitch, rows, rank);
        in.* = input;
        in.ldq = @intCast(local_pitch);
        in.qkv = try rig.input(std.mem.sliceAsBytes(compact));
    }
    const full = try Outputs.init(&rig, rows, false);
    const local = try Outputs.init(&rig, rows, true);
    const negative = try Outputs.init(&rig, rows, false);
    const inputs_before = try a.dupe(u8, try rig.inputBytes());
    defer a.free(inputs_before);
    try pipeline(old, s, input, full, streams, false, case, 0);
    var baseline: [7][]u8 = undefined;
    var loaded: usize = 0;
    defer for (baseline[0..loaded]) |b| a.free(b);
    for (std.enums.values(Field)) |field| {
        baseline[loaded] = try seen(a, &rig, full, field);
        loaded += 1;
    }
    var empty_count: usize = 0;
    for (local_inputs, 0..) |in, rank| {
        for (0..5) |_| {
            try pipeline(new, s, in, local, streams, true, case, rank);
            for (std.enums.values(Field)) |field| {
                const expected = try selected(a, baseline[@backingInt(field)], rows, field, rank);
                defer a.free(expected);
                try check.sameBytes(@tagName(field), try rig.bytes(local.ptr(field), local.lens[@backingInt(field)]), expected);
            }
        }
        const pm = try seen(a, &rig, local, .pm);
        defer a.free(pm);
        const pl = try seen(a, &rig, local, .pl);
        defer a.free(pl);
        const acc = try seen(a, &rig, local, .acc);
        defer a.free(acc);
        empty_count += try empties(pm, pl, acc, rows, case);
        if (rank > 0 and case.start > case.from and !nonfinite) {
            try part(new, s, in, local, streams, true, case.layer, 0);
            try s.synchronize();
            const expected = try selected(a, baseline[@backingInt(Field.acc)], rows, .acc, rank);
            defer a.free(expected);
            try check.expect(!std.mem.eql(u8, try rig.bytes(local.ptr(.acc), local.lens[@backingInt(Field.acc)]), expected), "kh0=0 control must go red", .{});
        }
        if (streams == 1 and case.start <= 7 and !nonfinite) {
            const q = try words(a, try rig.bytes(local.ptr(.q), local.lens[0]));
            defer a.free(q);
            const k = try words(a, try rig.bytes(local.ptr(.k), local.lens[1]));
            defer a.free(k);
            const v = try words(a, try rig.bytes(local.ptr(.v), local.lens[2]));
            defer a.free(v);
            const o = try words(a, try rig.bytes(local.ptr(.out), local.lens[6]));
            defer a.free(o);
            const d = try ref.attention(q, k, v, ring, o, rank, case, scale);
            std.debug.print("FP64_DIAGNOSTIC start{d} rank{d} outputs{d} maxabs{e} maxrel{e}; NO nonlinear bound certificate\n", .{ case.start, rank, d.outputs, d.max_abs, d.max_rel });
        }
    }
    try qkv(old, s, input, negative, rows, false, true);
    try s.synchronize();
    try check.expect(!std.mem.eql(u8, try rig.bytes(negative.ptr(.k), negative.lens[1]), baseline[1]), "old qkv with local head grid must go red", .{});
    try check.sameBytes("immutable inputs", try rig.inputBytes(), inputs_before);
    try check.expect(try rig.intact(), "all device buffer guards", .{});
    std.debug.print("PASS TP4 own-head S{d} start{d} from{d} layer{d} nonfinite{} padded{};4ranks*5repeats*7byteviews;emptychunks{d}\n", .{ streams, case.start, case.from, case.layer, nonfinite, padded, empty_count });
    return empty_count;
}

fn resources(gpu: check.Gpu, old: Fns, new: Fns) !void {
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var problems: usize = 0;
    for ([_]Fns{ old, new }, 0..) |fs, mode| for ([_]cuda.Function{ fs.qkv, fs.part, fs.merge }, [_]u32{ 64, 256, 128 }, 0..) |f, threads, op| {
        const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)), .static = @intCast(try f.attribute(.shared_size_bytes)), .bound = @intCast(try f.attribute(.max_threads_per_block)), .local = @intCast(try f.attribute(.local_size_bytes)) };
        problems += try limits.cuda(.{ .name = "DFlash own-head comparator", .symbol = "runtime-resolved", .image = "glm_dflash", .threads = threads }, usage, target, &log, gpu.gpa);
        std.debug.print("RESOURCE TP4={d} op{d} sm{d} threads{d} regs{d} static{d} local{d} maxthreads{d}\n", .{ mode, op, cap, threads, usage.regs, usage.static, usage.local, usage.bound });
    };
    std.debug.print("{s}", .{log.items});
    try check.expect(problems == 0, "live resource limits", .{});
}

/// All new symbols must resolve before working-set allocation; this command never falls back to old kernels.
pub fn run(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_dflash);
    defer module.unload();
    const new = try Fns.load(module, true);
    const old = try Fns.load(module, false);
    try resources(gpu, old, new);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var empty: usize = 0;
    for ([_]usize{ 1, 4, 8 }) |streams| for (ref.cases) |case| {
        empty += try one(gpu, s, old, new, streams, case, false, false);
    };
    empty += try one(gpu, s, old, new, 1, ref.cases[1], true, false);
    empty += try one(gpu, s, old, new, 1, ref.cases[1], false, true);
    try check.expect(empty > 0, "empty chunk cases actually exercised", .{});
    std.debug.print("TP4-OWN-HEAD SUMMARY14cases/56rankcases/280newpipelines/1960bytecomparisons;guards+inputs+controls;device bounds/stress/all-arch records/model/graph qualification remain separate\n", .{});
}

test "DFlash TP4 contract: actual pipeline records grids strides and final kh0 argument" {
    const t = std.testing;
    const Fake = struct {
        var local: bool = false;
        var streams: usize = 1;
        var rank: usize = 0;
        var layer: usize = 0;
        var n: usize = 0;
        var syncs: usize = 0;
        fn u(p: ?[*]?*anyopaque, at: usize) u64 {
            return @as(*align(1) const u64, @ptrCast(p.?[at].?)).*;
        }
        fn i(p: ?[*]?*anyopaque, at: usize) c_int {
            return @as(*align(1) const c_int, @ptrCast(p.?[at].?)).*;
        }
        fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32, smem: u32, s: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(s == null and smem == 0 and by == 1 and bz == 1);
            const op = @intFromPtr(f);
            std.debug.assert(op == n + 1);
            switch (op) {
                1 => {
                    std.debug.assert(gx == streams * 8 and gy == (if (local) @as(u32, 10) else 40) and gz == 1 and bx == 64);
                    std.debug.assert(u(p, 0) == 1 and i(p, 1) == @as(c_int, if (local) local_ld else full_ld));
                    std.debug.assert(u(p, 2) == 2 and u(p, 3) == 3 and u(p, 4) == 4);
                    for (0..3) |j| std.debug.assert(u(p, 7 + j) == 10 + j);
                },
                2 => {
                    std.debug.assert(gx == streams and gy == (if (local) @as(u32, 2) else 8) and gz == 16 and bx == 256);
                    for (0..3) |j| std.debug.assert(u(p, j) == 10 + j);
                    for (3..6) |j| std.debug.assert(u(p, j) == j + 2);
                    std.debug.assert(i(p, 6) == layer);
                    for (8..11) |j| std.debug.assert(u(p, j) == j + 5);
                    if (local) std.debug.assert(i(p, 11) == 2 * rank);
                },
                3 => {
                    std.debug.assert(gx == streams * 8 and gy == (if (local) @as(u32, 8) else 32) and gz == 1 and bx == 128);
                    for (0..4) |j| std.debug.assert(u(p, j) == 13 + j);
                },
                else => unreachable,
            }
            n += 1;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            syncs += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    d.api.cuStreamSynchronize = Fake.sync;
    const fs: Fns = .{ .qkv = .{ .d = &d, .handle = @ptrFromInt(1) }, .part = .{ .d = &d, .handle = @ptrFromInt(2) }, .merge = .{ .d = &d, .handle = @ptrFromInt(3) } };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    var input: Inputs = .{ .ldq = full_ld, .qkv = 1, .qn = 2, .kn = 3, .pos = 4, .rings = 5, .starts = 6, .froms = 7 };
    const output: Outputs = .{ .ptrs = .{ 10, 11, 12, 13, 14, 15, 16 }, .lens = @splat(0) };
    for ([_]usize{ 1, 4, 8 }) |streams| for (ref.cases) |case| for (0..4) |rank| for ([_]bool{ false, true }) |local| {
        input.ldq = if (local) local_ld else full_ld;
        Fake.local = local;
        Fake.streams = streams;
        Fake.rank = rank;
        Fake.layer = case.layer;
        Fake.n = 0;
        Fake.syncs = 0;
        try pipeline(fs, s, input, output, streams, local, case, rank);
        try t.expectEqual(@as(usize, 3), Fake.n);
        try t.expectEqual(@as(usize, 1), Fake.syncs);
    };
    Fake.n = 0;
    Fake.syncs = 0;
    try t.expectError(error.Geometry, pipeline(fs, s, input, output, 1, true, ref.cases[0], 4));
    try t.expectEqual(@as(usize, 0), Fake.n);
}
