//! Startup-only affine MSE packing; the module is drained and unloaded before prepared weights are published.
const std = @import("std");
const cuda = @import("cuda");
pub const image_sha = "fe539d130e29dcd6f90b53334cf5a931e51d2d9a3273f577109378ced84a9a8d";

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !cuda.Module {
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_Q4_MSE_IMAGE") orelse
        return error.Q4MseImageRequired;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.Q4MseImageHash;
    return cuda.Module.load(d, bytes);
}

pub fn quantize(f: cuda.Function, stream: cuda.Stream, src: u64, n: usize, k: usize, words: u64, scales: u64, biases: u64, status: u64) !void {
    if (n == 0 or k == 0 or k % 64 != 0 or src % 16 != 0 or words % 16 != 0 or
        scales % 16 != 0 or biases % 16 != 0 or status % 4 != 0) return error.Q4MseShape;
    const groups = try std.math.mul(usize, n, k / 64);
    if (groups > @as(usize, std.math.maxInt(u32)) * 4) return error.Q4MseShape;
    var args: cuda.Args = .{};
    inline for (.{ src, @as(i64, @intCast(groups)), words, scales, biases, status }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((groups + 3) / 4) }, .block = .{ .x = 128 } }, stream, &args);
}
