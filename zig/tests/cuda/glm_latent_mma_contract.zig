//! Explicit optional-module latent projection proof and GPU timing, with original embedded SIMT as a second reference.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const ref = @import("glm_latent_mma_reference.zig");
const c = core.contract.spec;
const mma = glm.latent_mma;

fn digest(bytes: []const u8) [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &result, .{});
    return result;
}

fn resource(gpu: check.Gpu, f: cuda.Function, name: []const u8, dynamic: u32) !void {
    const limits = core.contract.limits;
    const target = limits.target(try gpu.ctx.capability()) orelse return error.UnmeasuredGpu;
    const usage: limits.Usage = .{ .regs = @intCast(try f.attribute(.num_regs)),
        .static = @intCast(try f.attribute(.shared_size_bytes)),
        .local = @intCast(try f.attribute(.local_size_bytes)),
        .bound = @intCast(try f.attribute(.max_threads_per_block)) };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    const problems = try limits.cuda(.{ .name = name, .symbol = name, .image = "pinned latent MMA",
        .threads = 256, .dynamic = dynamic, .opt_in = true }, usage, target, &log, gpu.gpa);
    std.debug.print("{s}LATENT_MMA_RESOURCE {s} regs={d} local={d} dynamic={d}\n",
        .{ log.items, name, usage.regs, usage.local, dynamic });
    try check.expect(problems == 0, "latent MMA live launch limits", .{});
}

fn baseline(set: glm.glue.Set, stream: cuda.Stream, kind: mma.Kind,
    x: u64, w: u64, out: u64, rows: usize, heads: usize) !void
{
    if (kind == .absorb) return glm.latent.absorbQ(set, stream, x, w, out, rows, heads, 256, 512);
    return glm.latent.expandV(set, stream, x, w, out, rows, heads, 256, 512);
}

fn padding(data: []const u8, rows: usize, heads: usize, n: usize, stride: usize) !void {
    for (0..rows) |row| for (data[(row * stride + heads * n) * 2 .. (row + 1) * stride * 2]) |v|
        try check.expect(v == 0xff, "latent output row padding", .{});
    for (data[rows * stride * 2 ..]) |v| try check.expect(v == 0xff, "latent output extra row", .{});
}

fn panel(gpu: check.Gpu, s: cuda.Stream, old: glm.glue.Set, k: *const mma.Kernels,
    kind: mma.Kind, rows: usize, heads: usize, fill: ref.Fill, controls: bool) !void
{
    var rig: Rig = .{ .gpu = gpu, .s = s };
    defer rig.deinit();
    defer cuda.completion.drain(gpu.d, s, "latent MMA panel did not drain");
    const a = gpu.gpa;
    const n: usize = if (kind == .absorb) 512 else 256;
    const cols: usize = if (kind == .absorb) 256 else 512;
    const xr = heads * cols + 8;
    const stride = heads * n + 8;
    const x = try a.alloc(u16, (rows + 1) * xr);
    defer a.free(x);
    const ox = try a.alloc(u16, (rows + 1) * heads * cols);
    defer a.free(ox);
    @memset(x, 0x7fc1);
    for (0..rows + 1) |row| for (0..heads) |h| for (0..cols) |column| {
        const value = ref.input(row, column, fill);
        x[row * xr + h * cols + column] = value;
        ox[(row * heads + h) * cols + column] = value;
    };
    const w = try a.alloc(u16, heads * n * cols + 16);
    defer a.free(w);
    @memset(w, 0x7fc1);
    for (0..heads) |h| for (0..n) |column| for (0..cols) |inner| {
        const at = h * n * cols + if (kind == .absorb) inner * n + column else column * cols + inner;
        w[at] = ref.weight(h, column, inner, fill);
    };
    const dx = try rig.input(std.mem.sliceAsBytes(x));
    const dox = try rig.input(std.mem.sliceAsBytes(ox));
    const dw = try rig.input(std.mem.sliceAsBytes(w));
    const before_inputs = digest(try rig.inputBytes());
    const old_len = (rows + 1) * heads * n * 2;
    const new_len = (rows + 1) * stride * 2;
    const old_out = try rig.output(old_len);
    const new_out = try rig.output(new_len);
    const saved_old = try a.alloc(u8, old_len);
    defer a.free(saved_old);
    var dots: [4][2][8]ref.Dot = undefined;
    for (0..4) |row| for (0..2) |head| for (0..8) |column| {
        dots[row][head][column] = ref.dot(row, head, column, cols, fill);
    };
    var begin_old = try cuda.Event.init(gpu.d, true);
    defer begin_old.deinit();
    var end_old = try cuda.Event.init(gpu.d, true);
    defer end_old.deinit();
    var begin_new = try cuda.Event.init(gpu.d, true);
    defer begin_new.deinit();
    var end_new = try cuda.Event.init(gpu.d, true);
    defer end_new.deinit();
    defer cuda.completion.drain(gpu.d, s, "latent events freed before queued work completed");
    var old_hash: [32]u8 = undefined;
    var new_hash: [32]u8 = undefined;
    var old_ms: [5]f32 = undefined;
    var new_ms: [5]f32 = undefined;
    var old_error: f64 = 0;
    var new_error: f64 = 0;
    var per_column_old: [8]f64 = @splat(0);
    var per_column_new: [8]f64 = @splat(0);
    for (0..5) |repeat| {
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = old_out, .len = old_len }).fill8(0xff, s.handle);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = new_out, .len = new_len }).fill8(0xff, s.handle);
        try begin_old.record(s);
        try baseline(old, s, kind, dox, dw, old_out, rows, heads);
        try end_old.record(s);
        try begin_new.record(s);
        try k.launch(kind, s, dx, xr, dw, new_out, stride, rows, heads);
        try end_new.record(s);
        @memcpy(saved_old, try rig.bytes(old_out, old_len));
        const bytes = try rig.bytes(new_out, new_len);
        const bh = digest(saved_old);
        const nh = digest(bytes);
        if (repeat == 0) {
            old_hash = bh;
            new_hash = nh;
        } else {
            try check.expect(std.mem.eql(u8, &bh, &old_hash), "original SIMT five repeats", .{});
            try check.expect(std.mem.eql(u8, &nh, &new_hash), "MMA five repeats", .{});
        }
        for (0..rows) |row| for (0..heads) |h| for (0..n) |column| {
            const value = c.read(.bf16, bytes, row * stride + h * n + column);
            const original = c.read(.bf16, saved_old, (row * heads + h) * n + column);
            const expected = dots[row % 4][h % 2][column % 8];
            try check.expect(ref.accepts(value, expected, cols), "MMA FP64 bound R{d} row{d} h{d} n{d}",
                .{ rows, row, h, column });
            try check.expect(ref.accepts(original, expected, cols), "original SIMT FP64 bound", .{});
            if (std.math.isFinite(expected.value)) {
                const oe = @abs(original - expected.value);
                const ne = @abs(value - expected.value);
                old_error = @max(old_error, oe);
                new_error = @max(new_error, ne);
                per_column_old[column % 8] = @max(per_column_old[column % 8], oe);
                per_column_new[column % 8] = @max(per_column_new[column % 8], ne);
            }
        };
        try padding(bytes, rows, heads, n, stride);
        try padding(saved_old, rows, heads, n, heads * n);
        old_ms[repeat] = try cuda.Event.elapsedMs(begin_old, end_old);
        new_ms[repeat] = try cuda.Event.elapsedMs(begin_new, end_new);
    }
    if (fill != .nonfinite) {
        try check.expect(new_error <= old_error, "MMA worst FP64 error no worse than original SIMT", .{});
        for (per_column_new, per_column_old) |ne, oe|
            try check.expect(ne <= oe, "MMA worst per-column FP64 error no worse than SIMT", .{});
    }
    if (controls) {
        try k.launch(kind, s, dx + xr * 2, xr, dw, new_out, stride, rows, heads);
        const wrong = try rig.bytes(new_out, new_len);
        var rejected = false;
        for (0..n) |column| if (!ref.accepts(c.read(.bf16, wrong, column), dots[0][0][column % 8], cols)) {
            rejected = true;
            break;
        };
        try check.expect(rejected, "wrong-row allocated red control rejected", .{});
    }
    try check.expect(std.mem.eql(u8, &before_inputs, &digest(try rig.inputBytes())), "all latent inputs immutable", .{});
    try check.expect(try rig.intact() and !rig.overlap(), "all latent input/output guards and distinct allocations", .{});
    std.mem.sort(f32, &old_ms, {}, std.sort.asc(f32));
    std.mem.sort(f32, &new_ms, {}, std.sort.asc(f32));
    std.debug.print("LATENT_MMA_PANEL kind={t} R={d} H={d} fill={t} simt_ms={d} mma_ms={d} old_max={e} new_max={e}\n",
        .{ kind, rows, heads, fill, old_ms[2], new_ms[2], old_error, new_error });
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len != 1) return error.LatentMmaCandidateRequired;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(gpu.io, rest[0], gpu.gpa, .limited(32 << 20), .@"16", null);
    defer gpu.gpa.free(bytes);
    try mma.checkImage(bytes);
    var original = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer original.unload();
    var candidate = try cuda.Module.load(gpu.d, bytes);
    defer candidate.unload();
    const old = try glm.glue.Kernels.load(original);
    const k = try mma.Kernels.load(candidate);
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    defer cuda.completion.drain(gpu.d, s, "latent MMA modules freed before completion");
    try resource(gpu, k.absorb, "tf_glm_latent_mma_kn", mma.shared[0]);
    try resource(gpu, k.expand, "tf_glm_latent_mma_nk", mma.shared[1]);
    var count: usize = 0;
    for ([_]mma.Kind{ .absorb, .expand }) |kind| {
        for ([_]usize{ 1, 63, 64, 65, 127, 128, 129, 256, 2048 }) |rows| {
            try panel(gpu, s, .{ .glue = &old, .prompt = true }, &k, kind, rows, 16, .ordinary, rows == 65);
            count += 1;
        }
        for ([_]usize{ 64, 129, 2048 }) |rows| {
            try panel(gpu, s, .{ .glue = &old, .prompt = true }, &k, kind, rows, 32, .ordinary, false);
            count += 1;
        }
        for ([_]ref.Fill{ .rounding, .stress, .nonfinite }) |fill| {
            try panel(gpu, s, .{ .glue = &old, .prompt = true }, &k, kind, 65, 16, fill, false);
            count += 1;
        }
    }
    check.pass("latent MMA contract panels={d} repeats={d} accum=fp32 image={s}", .{ count, count * 5, mma.image_sha });
}

test {
    _ = ref;
}
