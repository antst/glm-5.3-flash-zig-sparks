//! Explicit dense-segment byte replacement checks; the default kernel-contract command is unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const seg = glm.dense_segments;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const limits = core.contract.limits;
const H = 16;
const W = 512;
const capacity = 2048;
const max_chunks = 4;
const max_part_rows = 16;
const scale: f32 = 1.0 / 16.0;
const widths = [_]usize{ H * W * 4, H * 4, H * 4, H * W * 2 };
const Fault = enum { none, row0, cache, stride };
const Totals = struct { cases: usize = 0, updates: usize = 0, repeats: usize = 0, controls: usize = 0 };

const Buffers = struct {
    ptrs: [4]u64,
    lens: [4]usize,
    fn init(rig: anytype, rows: usize) !Buffers {
        var b: Buffers = .{ .ptrs = undefined, .lens = undefined };
        for (0..4) |f| {
            b.lens[f] = (rows + 1) * widths[f] * @as(usize, if (f == 3) 1 else max_chunks);
            b.ptrs[f] = try rig.output(b.lens[f]);
        }
        return b;
    }
    fn reference(rig: anytype) !Buffers {
        return init(rig, max_part_rows);
    }
    fn poison(b: Buffers, rig: *Rig) !void {
        for (b.ptrs, b.lens) |ptr, len| {
            try (cuda.DeviceBuffer{ .d = rig.gpu.d, .ptr = ptr, .len = len }).fill8(0xff, rig.s.handle);
        }
    }
};

fn upload(rig: *Rig, ptr: u64, data: []const u8) !void {
    try rig.s.synchronize();
    try (cuda.DeviceBuffer{ .d = rig.gpu.d, .ptr = ptr, .len = data.len }).upload(0, data);
}

const Fixture = struct {
    rig: Rig,
    rows: usize,
    count: usize,
    fp8: bool,
    qa: u64,
    table: u64,
    positions: [8]u64 = undefined,
    caches: [8]u64 = undefined,
    scales: [8]u64 = undefined,
    old: [8]Buffers = undefined,
    new: Buffers,

    fn init(gpu: check.Gpu, s: cuda.Stream, lengths: []const usize, fp8: bool, nonfinite: bool) !Fixture {
        var f: Fixture = .{
            .rig = .{ .gpu = gpu, .s = s },
            .rows = 0,
            .count = lengths.len,
            .fp8 = fp8,
            .qa = 0,
            .table = 0,
            .new = undefined,
        };
        errdefer f.rig.deinit();
        if (lengths.len == 0 or lengths.len > 8) return error.DenseFixtureRows;
        for (lengths) |n| {
            if (n == 0 or n > max_part_rows or n > 64 - f.rows) return error.DenseFixtureRows;
            f.rows += n;
        }
        const a = gpu.gpa;
        const q = try a.alloc(u16, (f.rows + 1) * H * W);
        defer a.free(q);
        @memset(q, 0x7fc1);
        for (q[0 .. f.rows * H * W], 0..) |*v, j| {
            const z: i32 = @as(i32, @intCast((j * 17 + j / 19) % 31)) - 15;
            v.* = c.toBf16(@as(f32, @floatFromInt(z)) / 64);
        }
        if (nonfinite) q[17] = 0x7fc1;
        f.qa = try f.rig.input(std.mem.sliceAsBytes(q));
        for (0..f.count) |seq| {
            if (fp8) {
                const codes = try a.alloc(u8, capacity * W);
                defer a.free(codes);
                for (codes, 0..) |*v, j| v.* = @intCast(0x20 + (j / W * 7 + j % W * 3 + seq * 11) % 32);
                if (nonfinite and seq == 0) codes[W + 37] = 0x7f;
                f.caches[seq] = try f.rig.input(codes);
                const scales = try a.alloc(f32, capacity);
                defer a.free(scales);
                for (scales, 0..) |*v, j| v.* = @as(f32, @floatFromInt(1 + (j + seq) % 4)) / 4;
                f.scales[seq] = try f.rig.input(std.mem.sliceAsBytes(scales));
            } else {
                const cache = try a.alloc(u16, capacity * W);
                defer a.free(cache);
                for (cache, 0..) |*v, j| {
                    const z: i32 = @as(i32, @intCast((j / W * 5 + j % W * 7 + seq * 13) % 61)) - 30;
                    v.* = c.toBf16(@as(f32, @floatFromInt(z)) / 32);
                }
                if (nonfinite and seq == 0) cache[W + 37] = 0x7fc1;
                f.caches[seq] = try f.rig.input(std.mem.sliceAsBytes(cache));
                f.scales[seq] = 0;
            }
            f.positions[seq] = try f.rig.scratch(4);
            f.old[seq] = try Buffers.reference(&f.rig);
        }
        f.table = try f.rig.scratch((f.rows + 1) * @sizeOf(seg.Row));
        f.new = try Buffers.init(&f.rig, f.rows);
        return f;
    }

    fn storage(f: *const Fixture) seg.Storage {
        return .{
            .table = .{ .ptr = f.table, .len = (f.rows + 1) * @sizeOf(seg.Row) },
            .qa = .{ .ptr = f.qa, .len = (f.rows + 1) * H * W * 2 },
            .po = .{ .ptr = f.new.ptrs[0], .len = f.new.lens[0] },
            .pm = .{ .ptr = f.new.ptrs[1], .len = f.new.lens[1] },
            .pl = .{ .ptr = f.new.ptrs[2], .len = f.new.lens[2] },
            .out = .{ .ptr = f.new.ptrs[3], .len = f.new.lens[3] },
        };
    }

    fn plan(f: *Fixture, lengths: []const usize, update: usize) !seg.Plan {
        const at = [_]usize{ 0, 511, 31, 509, 512, 1023, 1024, 1535 };
        var parts: [8]seg.Segment = undefined;
        for (lengths, 0..) |n, i| {
            const slot = (i + update) % f.count;
            const pos: i32 = @intCast(at[(i + 2 * update) % at.len]);
            try upload(&f.rig, f.positions[i], std.mem.asBytes(&pos));
            parts[i] = .{
                .rows = n,
                .host_pos = @intCast(pos),
                .capacity = capacity,
                .cache = f.caches[slot],
                .scales = f.scales[slot],
                .position = f.positions[i],
            };
        }
        return seg.Plan.make(parts[0..lengths.len], .{
            .max_part_rows = max_part_rows,
            .dense_limit = capacity,
            .scratch_rows = seg.max_rows,
            .scratch_chunks = max_chunks,
        });
    }
};

fn original(f: *Fixture, old: glm.glue.Set, p: *const seg.Plan, lengths: []const usize) !void {
    var row0: usize = 0;
    for (lengths, 0..) |n, seq| {
        const d = p.table[row0];
        const b = f.old[seq];
        try b.poison(&f.rig);
        try glm.latent.attentionFormat(old, f.rig.s, f.qa + row0 * H * W * 2, d.cache, d.scales, .{}, d.position, .{ .po = b.ptrs[0], .pm = b.ptrs[1], .pl = b.ptrs[2] }, b.ptrs[3], d.position, n, H, W, @intCast(d.chunks), scale, 16);
        row0 += n;
    }
    try f.rig.s.synchronize();
}

fn expected(f: *Fixture, p: *const seg.Plan, lengths: []const usize) ![4][]u8 {
    const a = f.rig.gpu.gpa;
    var out: [4][]u8 = undefined;
    var made: usize = 0;
    errdefer for (out[0..made]) |v| a.free(v);
    for (0..4) |field| {
        out[field] = try a.alloc(u8, f.new.lens[field]);
        made += 1;
        @memset(out[field], 0xff);
        var row0: usize = 0;
        for (lengths, 0..) |n, seq| {
            const source = try f.rig.bytes(f.old[seq].ptrs[field], f.old[seq].lens[field]);
            const nch: usize = if (field == 3) 1 else @intCast(p.table[row0].chunks);
            for (source[nch * n * widths[field] ..]) |v|
                try check.expect(v == 0xff, "original partial/output padding", .{});
            fieldCopy(field, out[field], source, f.rows, row0, n, nch);
            row0 += n;
        }
    }
    return out;
}

fn fieldCopy(field: usize, dst: []u8, src: []const u8, total: usize, row0: usize, rows: usize, chunks: usize) void {
    for (0..chunks) |ch| for (0..rows) |r| {
        const to = (ch * total + row0 + r) * widths[field];
        const from = (ch * rows + r) * widths[field];
        @memcpy(dst[to..][0..widths[field]], src[from..][0..widths[field]]);
    };
}

fn tableBytes(f: *Fixture, p: *const seg.Plan) !void {
    try (cuda.DeviceBuffer{ .d = f.rig.gpu.d, .ptr = f.table, .len = (f.rows + 1) * @sizeOf(seg.Row) }).fill8(0xa5, f.rig.s.handle);
    try upload(&f.rig, f.table, p.bytes());
}

fn compare(f: *Fixture, want: [4][]u8, equal: bool) !void {
    var same = true;
    for (0..4) |field| {
        const got = try f.rig.bytes(f.new.ptrs[field], f.new.lens[field]);
        if (equal) try check.sameBytes("dense partial/output/padding", got, want[field]);
        same = same and std.mem.eql(u8, got, want[field]);
    }
    try check.expect(same == equal, "planted dense segment control must change compared bytes", .{});
}

// The stride control changes only chunk storage pitch, with an extra allocated row and unchanged live grid.
fn wrongStride(f: *Fixture, k: seg.Kernels, p: *const seg.Plan) !void {
    var a: cuda.Args = .{};
    const s = f.storage();
    inline for (.{ s.table.ptr, s.qa.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, @as(i64, @intCast(p.rows + 1)), @as(i64, H), scale }) |v| a.add(v);
    try cuda.launch.launch(if (p.fp8) k.chunks_fp8 else k.chunks, .{
        .grid = .{ .x = @intCast(p.rows), .y = 1, .z = @intCast(p.chunks) },
        .block = .{ .x = 256 },
        .shared = glm.glue.chunksSmem(16),
    }, f.rig.s, &a);
    var b: cuda.Args = .{};
    inline for (.{ s.table.ptr, s.po.ptr, s.pm.ptr, s.pl.ptr, s.out.ptr, @as(i64, @intCast(p.rows)), @as(i64, H) }) |v| b.add(v);
    try cuda.launch.launch(k.merge, .{
        .grid = .{ .x = @intCast(p.rows), .y = H },
        .block = .{ .x = 128 },
    }, f.rig.s, &b);
}

fn controls(f: *Fixture, k: seg.Kernels, p: *const seg.Plan, want: [4][]u8) !usize {
    var caught: usize = 0;
    for ([_]Fault{ .row0, .cache, .stride }) |fault| {
        var bad = p.*;
        for (bad.table[0..bad.rows]) |*d| {
            if (fault == .row0) d.row0 = 0;
            if (fault == .cache) {
                d.cache = f.caches[0];
                d.scales = f.scales[0];
            }
        }
        try f.new.poison(&f.rig);
        try tableBytes(f, &bad);
        if (fault == .stride) try wrongStride(f, k, &bad) else try k.launch(f.rig.s, &bad, f.storage(), scale);
        try f.rig.s.synchronize();
        try compare(f, want, false);
        try check.sameBytes("control table is read-only", try f.rig.bytes(f.table, bad.bytes().len), bad.bytes());
        const padding = try f.rig.bytes(f.table + bad.bytes().len, @sizeOf(seg.Row));
        for (padding) |v| try check.expect(v == 0xa5, "control descriptor padding", .{});
        try check.expect(try f.rig.intact(), "control stays inside allocated guards", .{});
        caught += 1;
    }
    return caught;
}

fn one(gpu: check.Gpu, s: cuda.Stream, old: glm.glue.Set, k: seg.Kernels, lengths: []const usize, fp8: bool, nonfinite: bool, control_case: bool, total: *Totals) !void {
    var f = try Fixture.init(gpu, s, lengths, fp8, nonfinite);
    defer f.rig.deinit();
    const input_before = try gpu.gpa.dupe(u8, try f.rig.inputBytes());
    defer gpu.gpa.free(input_before);
    var lens: [8]usize = undefined;
    @memcpy(lens[0..lengths.len], lengths);
    for (0..3) |update| {
        const p = try f.plan(lens[0..lengths.len], update);
        try original(&f, old, &p, lens[0..lengths.len]);
        const want = try expected(&f, &p, lens[0..lengths.len]);
        defer for (want) |v| gpu.gpa.free(v);
        try tableBytes(&f, &p);
        for (0..5) |_| {
            try f.new.poison(&f.rig);
            try k.launch(s, &p, f.storage(), scale);
            try s.synchronize();
            try compare(&f, want, true);
            try check.sameBytes("read-only device descriptor rows", try f.rig.bytes(f.table, p.bytes().len), p.bytes());
            const padding = try f.rig.bytes(f.table + p.bytes().len, @sizeOf(seg.Row));
            for (padding) |v| try check.expect(v == 0xa5, "descriptor padding", .{});
            total.repeats += 1;
        }
        if (control_case and update == 0) total.controls += try controls(&f, k, &p, want);
        try check.sameBytes("immutable QA/cache/scales", try f.rig.inputBytes(), input_before);
        for (0..f.count) |i| {
            const at = [_]i32{ 0, 511, 31, 509, 512, 1023, 1024, 1535 };
            try check.sameBytes("position input", try f.rig.bytes(f.positions[i], 4), std.mem.asBytes(&at[(i + 2 * update) % 8]));
        }
        try check.expect(try f.rig.intact(), "all candidate/reference/input guards", .{});
        std.mem.reverse(usize, lens[0..lengths.len]);
        total.updates += 1;
    }
    total.cases += 1;
    std.debug.print("PASS dense-segmented S{d} R{d} fp8{} nonfinite{};" ++
        "3updates*5repeats*4byteviews+padding+inputs\n", .{ lengths.len, f.rows, fp8, nonfinite });
}

fn resources(gpu: check.Gpu, old: *const glm.glue.Kernels, new: seg.Kernels) !void {
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    const fs = [_]cuda.Function{ old.chunks_16, old.chunks_16_fp8, old.merge, new.chunks, new.chunks_fp8, new.merge };
    const names = [_][]const u8{ "tf_glm_chunks_16", "tf_glm_chunks_16_fp8", "tf_glm_merge", "tf_glm_dense_seg_chunks_16_v1", "tf_glm_dense_seg_chunks_16_fp8_v1", "tf_glm_dense_seg_merge_v1" };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var problems: usize = 0;
    for (fs, 0..) |f, i| {
        const merge = i % 3 == 2;
        const usage: limits.Usage = .{
            .regs = @intCast(try f.attribute(.num_regs)),
            .static = @intCast(try f.attribute(.shared_size_bytes)),
            .local = @intCast(try f.attribute(.local_size_bytes)),
            .bound = @intCast(try f.attribute(.max_threads_per_block)),
        };
        problems += try limits.cuda(.{ .name = "dense segmented comparator", .symbol = names[i], .image = if (i < 3) "embedded-glm_glue" else "candidate-dense-segments", .threads = if (merge) 128 else 256, .dynamic = if (merge) 0 else glm.glue.chunksSmem(16), .opt_in = !merge }, usage, target, &log, gpu.gpa);
        std.debug.print("RESOURCE dense-segmented op{d} sm{d} regs{d} static{d} local{d} bound{d}\n", .{ i, cap, usage.regs, usage.static, usage.local, usage.bound });
    }
    std.debug.print("{s}", .{log.items});
    try check.expect(problems == 0, "dense segment live resource limits", .{});
}

fn candidateModule(gpu: check.Gpu, path: ?[]const u8) !cuda.Module {
    const name = path orelse return cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, name, gpu.gpa, .limited(64 << 20));
    defer gpu.gpa.free(bytes);
    const aligned = try gpu.gpa.alignedAlloc(u8, .@"8", bytes.len);
    defer gpu.gpa.free(aligned);
    @memcpy(aligned, bytes);
    return cuda.Module.load(gpu.d, aligned);
}

/// Require all new symbols before any fixture allocation; the reference always uses the embedded original image.
pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len > 1) return error.DenseContractArguments;
    const path: ?[]const u8 = if (rest.len == 1) rest[0] else null;
    var reference = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer reference.unload();
    var candidate = try candidateModule(gpu, path);
    defer candidate.unload();
    const k = (try seg.Kernels.load(candidate, true)).?;
    const old = try glm.glue.Kernels.load(reference);
    std.debug.print("DENSE-SEGMENTED reference=embedded-glm_glue candidate={s}\n", .{path orelse "embedded-glm_glue; rebuilt-image comparison only"});
    try resources(gpu, &old, k);
    const set: glm.glue.Set = .{ .glue = &old };
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var total: Totals = .{};
    var lengths: [8]usize = undefined;
    for ([_]bool{ false, true }) |fp8| {
        for (1..9) |count| for (0..3) |style| {
            for (lengths[0..count], 0..) |*n, i| n.* = switch (style) {
                0 => 1,
                1 => (i * 5 + 3) % 8 + 1,
                else => 8,
            };
            try one(gpu, s, set, k, lengths[0..count], fp8, false, false, &total);
        };
        for ([_]usize{ 2, 3, 4 }) |n| {
            @memset(&lengths, n);
            try one(gpu, s, set, k, &lengths, fp8, false, false, &total);
        }
        for ([_][]const usize{
            &.{ 7, 8 },                   &.{ 1, 8, 8 },                &.{ 7, 8, 8, 8 }, &.{ 1, 8, 8, 8, 8 },
            &.{ 7, 8, 8, 8, 8, 8, 8, 8 }, &.{ 1, 2, 3, 4, 5, 6, 7, 1 },
        }) |tail| try one(gpu, s, set, k, tail, fp8, false, false, &total);
        for ([_][]const usize{ &.{9}, &.{15}, &.{16}, &.{ 9, 15 }, &.{ 16, 16 }, &.{ 16, 15, 9, 8, 7, 5, 3, 1 }, &.{ 1, 16, 1, 16, 1, 16 } }) |wide|
            try one(gpu, s, set, k, wide, fp8, false, false, &total);
        try one(gpu, s, set, k, &.{ 3, 6 }, fp8, false, true, &total);
        try one(gpu, s, set, k, &.{ 1, 7 }, fp8, true, false, &total);
    }
    try check.expect(total.controls == 6, "all6 host-safe planted controls actually caught", .{});
    std.debug.print("DENSE-SEGMENTED SUMMARY cases{d} updates{d} repeats{d} controls{d}/6;" ++
        "partial+outputbytes/padding/inputs;FP64/stress/allSM/model gates remain separate\n", .{ total.cases, total.updates, total.repeats, total.controls });
}

test "dense segmented expected partials scatter sequence-local chunk rows into global rows" {
    const a = std.testing.allocator;
    const unit = widths[1];
    const got = try a.alloc(u8, 3 * 10 * unit);
    defer a.free(got);
    @memset(got, 0xa5);
    for ([_]usize{ 3, 6 }, 0..) |rows, seq| {
        const nch: usize = seq + 1;
        const src = try a.alloc(u8, nch * rows * unit);
        defer a.free(src);
        for (0..nch) |ch| for (0..rows) |r| {
            @memset(src[(ch * rows + r) * unit ..][0..unit], @intCast(1 + seq * 64 + ch * 8 + r));
        };
        fieldCopy(1, got, src, 9, if (seq == 0) 0 else 3, rows, nch);
    }
    for (0..3) |ch| for (0..9) |r| {
        const want: u8 = if (r < 3) (if (ch == 0) @intCast(1 + r) else 0xa5) else (if (ch < 2) @intCast(65 + ch * 8 + r - 3) else 0xa5);
        for (got[(ch * 9 + r) * unit ..][0..unit]) |v| try std.testing.expectEqual(want, v);
    };
    for (got[3 * 9 * unit ..]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
}

test "dense segmented actual launch arguments keep the stride control within padded storage" {
    const Fake = struct {
        var calls: usize = 0;
        var fp8: bool = false;
        var bad: bool = false;
        fn u(p: ?[*]?*anyopaque, i: usize) u64 {
            return @as(*align(1) const u64, @ptrCast(p.?[i].?)).*;
        }
        fn launch(f: cuda.abi.Function, x: u32, y: u32, z: u32, bx: u32, by: u32, bz: u32, smem: u32, stream: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            std.debug.assert(stream == null and by == 1 and bz == 1 and x == 9);
            if (calls == 0) {
                std.debug.assert(@intFromPtr(f) == (if (fp8) @as(usize, 2) else 1));
                std.debug.assert(y == 1 and z == 2 and bx == 256 and smem == glm.glue.chunksSmem(16));
                std.debug.assert(u(p, 0) == 0x60000000 and u(p, 1) == 0x50000000);
                std.debug.assert(u(p, 2) == 0x80000000 and u(p, 3) == 0x90000000);
                std.debug.assert(u(p, 4) == 0xa0000000 and u(p, 5) == (if (bad) @as(u64, 10) else 9));
                std.debug.assert(u(p, 6) == H);
                const sc = @as(*align(1) const f32, @ptrCast(p.?[7].?)).*;
                std.debug.assert(sc == scale);
            } else {
                std.debug.assert(@intFromPtr(f) == 3 and y == H and z == 1 and bx == 128 and smem == 0);
                std.debug.assert(u(p, 0) == 0x60000000 and u(p, 1) == 0x80000000);
                std.debug.assert(u(p, 2) == 0x90000000 and u(p, 3) == 0xa0000000);
                std.debug.assert(u(p, 4) == 0x70000000 and u(p, 5) == 9 and u(p, 6) == H);
            }
            calls += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const k: seg.Kernels = .{
        .chunks = .{ .d = &d, .handle = @ptrFromInt(1) },
        .chunks_fp8 = .{ .d = &d, .handle = @ptrFromInt(2) },
        .merge = .{ .d = &d, .handle = @ptrFromInt(3) },
    };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var f: Fixture = .{
        .rig = .{ .gpu = .{ .d = &d, .ctx = undefined, .gpa = std.testing.allocator, .io = undefined }, .s = stream },
        .rows = 9,
        .count = 2,
        .fp8 = false,
        .qa = 0x50000000,
        .table = 0x60000000,
        .new = .{ .ptrs = .{ 0x80000000, 0x90000000, 0xa0000000, 0x70000000 }, .lens = .{ 4 * 10 * widths[0], 4 * 10 * widths[1], 4 * 10 * widths[2], 10 * widths[3] } },
    };
    for ([_]bool{ false, true }) |fp8| {
        const p = try seg.Plan.make(&.{
            .{ .rows = 3, .host_pos = 0, .capacity = 2048, .cache = 0x10000000, .position = 0x30000000, .scales = if (fp8) 0x31000000 else 0 },
            .{ .rows = 6, .host_pos = 511, .capacity = 2048, .cache = 0x20000000, .position = 0x32000000, .scales = if (fp8) 0x33000000 else 0 },
        }, .{ .dense_limit = 2048, .scratch_rows = 64, .scratch_chunks = 4 });
        Fake.fp8 = fp8;
        Fake.calls = 0;
        Fake.bad = false;
        try k.launch(stream, &p, f.storage(), scale);
        try std.testing.expectEqual(@as(usize, 2), Fake.calls);
        Fake.calls = 0;
        Fake.bad = true;
        try wrongStride(&f, k, &p);
        try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    }
}

test "dense segmented actual reference allocations hold every16-row partial and output plus padding" {
    const Recorder = struct {
        lengths: [4]usize = undefined,
        count: usize = 0,
        fn output(r: *@This(), len: usize) !u64 {
            r.lengths[r.count] = len;
            r.count += 1;
            return r.count * 0x10000000;
        }
    };
    var recorder: Recorder = .{};
    const b = try Buffers.reference(&recorder);
    try std.testing.expectEqual(@as(usize, 4), recorder.count);
    for (1..17) |rows| for (1..5) |chunks| for (0..4) |field| {
        const valid = rows * widths[field] * (if (field == 3) @as(usize, 1) else chunks);
        try std.testing.expect(valid + widths[field] <= recorder.lengths[field]);
        try std.testing.expectEqual(b.lens[field], recorder.lengths[field]);
    };
}
