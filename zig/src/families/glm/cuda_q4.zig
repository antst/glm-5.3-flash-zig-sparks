//! GLM-5.3-Flash's affine 4-bit matmul for the draft head (the Python family's qmm.matmul, tile 2) and its tiling.

const std = @import("std");
const cuda = @import("cuda");

/// qmm_group.cu's tile-2 instantiation (LaneTile<64, 16, 64, 1, 4, 8>, bf16 out), as zig/kernels/cuda exports it.
const group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const group_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const rows_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const rows_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const rows_smem: u32 = 43008;
const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

/// A tiled 4-bit matrix [n][k] (shared.Q4): packed words, (k / 64, npad) bf16 scales and biases, npad = n up to 128s.
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

pub const Kernels = struct {
    group: cuda.Function,
    pack_dense: cuda.Function,
    transpose: cuda.Function,
    gb10: bool, // qmm_group_cuda's programmatic dependent launch: on for GB10 (sm_121)
    group_f32: ?cuda.Function = null,
    rows: ?[2]cuda.Function = null,
    python_shapes: bool = false, // explicit diagnostic context; production loaders keep the legacy split policy

    /// From the qmm_group and affine4_pack modules.
    pub fn load(group: cuda.Module, ops: cuda.Module, ctx: *const cuda.Context) !Kernels {
        const k: Kernels = .{
            .group = try group.function(group_symbol),
            .pack_dense = try ops.function("tf_pack_dense"),
            .transpose = try ops.function("tf_transpose_pad16"),
            .gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1,
        };
        try k.group.allowDynamicShared(group_smem);
        return k;
    }
};

/// Explicit opt-in after loading a rebuilt group module; its lifetime must cover all launches and graph owners.
pub fn enableWide(ks: *Kernels, group: cuda.Module) !void {
    if (group.d != ks.group.d) return error.Q4ModuleDriverMismatch;
    var handle: cuda.abi.Function = null;
    const res = group.d.api.cuModuleGetFunction(&handle, group.handle, group_f32_symbol);
    if (res == cuda.abi.error_not_found) return error.MissingQ4F32Kernel;
    try group.d.check(res, "cuModuleGetFunction(q4 fp32)");
    const f: cuda.Function = .{ .d = group.d, .handle = handle };
    try f.allowDynamicShared(group_smem);
    ks.group_f32 = f;
}

/// Optional wider row tile; original entrypoints remain unchanged for rows at most16.
pub fn enableRows(ks: *Kernels, group: cuda.Module) !void {
    if (group.d != ks.group.d) return error.Q4ModuleDriverMismatch;
    var fs: [2]cuda.Function = undefined;
    inline for (.{ rows_symbol, rows_f32_symbol }, 0..) |name, i| {
        var handle: cuda.abi.Function = null;
        const rc = group.d.api.cuModuleGetFunction(&handle, group.handle, name);
        if (rc == cuda.abi.error_not_found) return error.MissingQ4RowsKernel;
        try group.d.check(rc, "cuModuleGetFunction(q4 rows)");
        fs[i] = .{ .d = group.d, .handle = handle };
        try fs[i].allowDynamicShared(rows_smem);
    }
    ks.rows = fs;
}

/// A 64-row tile reuses each weight panel across four original m16 accumulators.
pub fn matmulRows(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize, wide: bool) !void {
    if (rows <= 16) {
        if (wide) return matmulWide(ks, stream, x, ldx, xs, q, out, rows);
        return matmul(ks, stream, x, ldx, xs, q, out, rows);
    }
    const fs = ks.rows orelse return error.Q4RowsNotEnabled;
    return matmulTile(fs[@intFromBool(wide)], ks.gb10, stream, x, ldx, xs, q, out, rows, 64, rows_smem, ks.python_shapes);
}

pub fn npadOf(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// Device bytes of a Q4's words, scales and biases.
pub fn bytes(n: usize, k: usize) [3]usize {
    const p = npadOf(n);
    return .{ p * k / 2, k / 64 * p * 2, k / 64 * p * 2 };
}

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// Exact75c shape overrides, used only by an explicitly selected diagnostic kernel context.
pub fn pythonSplitK(n: usize, k: usize) usize {
    if (n == 12576 and k == 4096) return 4;
    if (n == 4096 and k == 4096) return 2;
    if (n == 2048 and k == 4096) return 4;
    if (n == 8192 and (k == 1536 or k == 512)) return 4;
    if (n == 4096 and k == 8192) return 8;
    return splitK(n, k);
}

/// shared.pack: MLX words i32 [n][k/8] and bf16 scales and biases [n][k/64] -> `q` (caller's buffers, `bytes` long).
pub fn pack(ks: *const Kernels, stream: cuda.Stream, words: u64, scales: u64, biases: u64, q: Q4) !void {
    const kg = q.k / 64;
    const total: u64 = @as(u64, q.npad / 64) * kg * 512;
    var a: cuda.Args = .{};
    a.add(words);
    for ([_]usize{ q.n, q.k / 8, kg }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(q.w);
    a.add(@as(c_longlong, @intCast(total)));
    try cuda.launch.launch(ks.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    for ([_]u64{ scales, biases }, [_]u64{ q.s, q.b }) |src, dst| {
        var t: cuda.Args = .{};
        t.add(src);
        for ([_]usize{ q.n, kg, q.npad }) |v| t.add(@as(c_int, @intCast(v)));
        t.add(dst);
        const cells = @as(u64, kg) * q.npad;
        try cuda.launch.launch(ks.transpose, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, stream, &t);
    }
}

/// qmm.matmul of a Q4 (tile 2, up to 16 rows): x bf16 (`ldx` apart), group sums xs fp32 [rows][k/64] -> out bf16.
pub fn matmul(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize) !void {
    return matmulOut(ks.group, ks.gb10, stream, x, ldx, xs, q, out, rows, ks.python_shapes);
}

/// Same Q4 group and slice math, storing fp32 [rows][q.n]; requires explicit enableWide, never bf16 fallback.
pub fn matmulWide(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize) !void {
    const f = ks.group_f32 orelse return error.Q4F32NotEnabled;
    return matmulOut(f, ks.gb10, stream, x, ldx, xs, q, out, rows, ks.python_shapes);
}

fn matmulOut(f: cuda.Function, gb10: bool, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize, python_shapes: bool) !void {
    return matmulTile(f, gb10, stream, x, ldx, xs, q, out, rows, 16, group_smem, python_shapes);
}

fn matmulTile(f: cuda.Function, gb10: bool, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize, tile_rows: usize, smem: u32, python_shapes: bool) !void {
    if (rows == 0 or rows > tile_rows) return error.WindowTooWide;
    const sk = if (python_shapes) pythonSplitK(q.n, q.k) else splitK(q.n, q.k);
    const tiles = (q.n + 63) / 64;
    var parts: Parts = std.mem.zeroes(Parts);
    parts.count = 1;
    parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = @intCast(q.n), .npad = @intCast(q.npad), .sk = @intCast(sk), .tiles = @intCast(tiles), .first = 0 };
    const rows_t = (rows + tile_rows - 1) / tile_rows;
    const clusters = rows_t * tiles; // one part: a cluster is the sk K slices of one column tile
    var a: cuda.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(parts);
    for ([_]usize{ rows, q.k, if (rows == 1) q.k else ldx, rows_t, sk }) |v| a.add(@as(c_int, @intCast(v))); // M, K, ldx, rows_t, C
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(clusters * sk) },
        .block = .{ .x = 128 },
        .shared = smem,
        .cluster = if (sk > 1) .{ .x = @intCast(sk) } else null,
        .pdl = gb10,
    };
    try cuda.launch.launch(f, cfg, stream, &a);
}

test {
    _ = @import("cuda_q4_wide_test.zig");
}

test "the draft head's shape takes one K slice; npad rounds to 128s" {
    try std.testing.expectEqual(@as(usize, 1), splitK(77440, 4096));
    try std.testing.expectEqual(@as(usize, 77440), npadOf(77440));
    try std.testing.expectEqual(@as(usize, 51712), npadOf(51627));
}
