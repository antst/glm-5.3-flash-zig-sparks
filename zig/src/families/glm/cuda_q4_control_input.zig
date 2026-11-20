//! Preserve the original recurrent-input controls while retaining the full-shape Q4 QKV projection.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const mm = @import("cuda_mm.zig");
const glue = @import("cuda_glue.zig");
const dk = cuda.kda.dk;

pub fn columns(heads: usize) usize {
    return 2 * dk + heads;
}

/// Before the original input matrix is packed: directly tile its original BF16 tail into a weights-owned buffer.
pub fn prepare(w: *weights.Weights, kernels: *const mm.Kernels, stream: cuda.Stream, d: *const cuda.Driver) !void {
    if (w.share.world != 2 or w.mtp != null) return error.Q4ControlInputScope;
    var count: usize = 0;
    var bytes: usize = 0;
    for (w.layers) |*layer| switch (layer.mixer) {
        .kda => |*k| {
            if (k.heads != 32 or k.dims != 4096 or k.proj != .bf16 or k.control_tail != 0 or k.width() != k.matWidth())
                return error.Q4ControlInputScope;
            if (k.proj.bf16 == 0 or k.proj.bf16 % 16 != 0 or
                k.proj.bf16 > std.math.maxInt(u64) - k.width() * k.dims * 2) return error.Q4ControlInputBuffer;
            const n = columns(k.heads);
            const size = n * k.dims * 2;
            var buffer = try cuda.DeviceBuffer.alloc(d, size);
            errdefer buffer.free();
            errdefer @import("cuda_load_owner.zig").drain(stream, d);
            try mm.tile(kernels, stream, k.proj.bf16 + 3 * k.heads * dk * k.dims * 2, buffer.ptr, n, k.dims);
            try stream.synchronize();
            try w.buffers.append(w.gpa, buffer);
            k.control_tail = buffer.ptr;
            count += 1;
            bytes += size;
        },
        .mla => {},
    };
    if (count == 0) return error.Q4ControlInputScope;
    std.log.info("GLM_Q4_CONTROL_INPUT rank={d} active=true matrices={d} bytes={d} original_qkv_shape=true slices=1", .{ w.share.rank, count, bytes });
}

/// The legacy split temporary is unused for the admitted nonsplit input, and can hold contiguous control rows.
pub fn scratchBytes(rows: usize, heads: usize) usize {
    return rows * columns(heads) * 2;
}

fn overlaps(a: u64, an: usize, b: u64, bn: usize) bool {
    return a < b + bn and b < a + an;
}

/// Same stream: the original Q4 projection remains intact; overwrite only f_a/g_a/b before their consumers.
pub fn overwrite(set: glue.Set, stream: cuda.Stream, layer: anytype, scratch: anytype, x: u64, stride: usize, rows: usize) !void {
    const kernels = set.mm orelse return error.Q4ControlInputScope;
    const copy = set.copy_rows orelse return error.FormatNotLoaded;
    const n = columns(layer.heads);
    const width = layer.matWidth();
    if (layer.heads != 32 or layer.dims != 4096 or rows == 0 or rows > 2048 or stride < layer.dims or stride > 65536 or
        layer.proj != .bf16 or layer.control_tail == 0 or width != layer.width() or scratch.tmp_bytes < scratchBytes(rows, layer.heads))
        return error.Q4ControlInputScope;
    for ([_]u64{ x, layer.control_tail, scratch.tmp, scratch.p }) |ptr|
        if (ptr == 0 or ptr % 16 != 0 or ptr > std.math.maxInt(u64) - (256 << 20)) return error.Q4ControlInputBuffer;
    const written = scratchBytes(rows, layer.heads);
    if (overlaps(scratch.tmp, written, scratch.p, rows * width * 2) or overlaps(scratch.tmp, written, x, rows * stride * 2))
        return error.Q4ControlInputBuffer;
    try mm.matmulOneSlice(kernels, stream, x, stride, .{ .w = layer.control_tail, .n = n, .k = layer.dims }, scratch.tmp, scratch.mm_part, rows, false, set.prompt);
    var args: cuda.Args = .{};
    args.add(scratch.tmp);
    args.add(@as(i64, @intCast(n * 2)));
    args.add(scratch.p + 3 * layer.heads * dk * 2);
    args.add(@as(i64, @intCast(width * 2)));
    args.add(@as(c_int, @intCast(rows)));
    args.add(@as(c_int, @intCast(n * 2)));
    try cuda.launch.launch(copy, .{ .grid = .{ .x = @intCast((written + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
}
