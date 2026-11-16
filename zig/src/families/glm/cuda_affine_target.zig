//! Optional affine target Q4: group sums and the existing row-stable affine kernels, on the caller's stream.
const cuda = @import("cuda");
const q4 = @import("cuda_q4.zig");

pub const Prompt = @import("cuda_affine_prompt.zig").Kernels;
pub const Kernels = struct { q4: q4.Kernels, sums: cuda.Function, prompt: ?Prompt = null };

pub fn scratchBytes(rows: usize, k: usize) usize {
    return @min(rows, 64) * (k / 64) * 4;
}

pub fn matmul(ks: Kernels, stream: cuda.Stream, x: u64, ldx: usize, q: q4.Q4, out: u64, scratch: u64, rows: usize, wide: bool) !void {
    if (q.k == 0 or q.k % 64 != 0 or q.n == 0 or q.npad != q4.npadOf(q.n) or ldx < q.k or
        x % 16 != 0 or ldx % 8 != 0 or scratch == 0 or scratch % 16 != 0) return error.BadAffineShape;
    var at: usize = 0;
    while (at < rows) {
        const count = @min(rows - at, 64);
        const src = x + at * ldx * 2;
        var args: cuda.Args = .{};
        args.add(src);
        args.add(scratch);
        args.add(@as(i64, @intCast(ldx)));
        args.add(@as(i64, @intCast(q.k / 64)));
        try cuda.launch.launch(ks.sums, .{ .grid = .{ .x = @intCast(count), .y = @intCast((q.k / 64 + 15) / 16) }, .block = .{ .x = 128 } }, stream, &args);
        try q4.matmulRows(&ks.q4, stream, src, ldx, scratch, q, out + at * q.n * @as(usize, if (wide) 4 else 2), count, wide);
        at += count;
    }
}

test {
    _ = @import("cuda_affine_prompt_test.zig");
    _ = @import("cuda_affine_prompt_owner.zig");
}
