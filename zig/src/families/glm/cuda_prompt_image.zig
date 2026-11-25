//! Hash-pinned CUDA images for the optimized prompt path.
const std = @import("std");
const cuda = @import("cuda");
const shard = @import("cuda_shard.zig");

pub fn optimized(ctx: *const cuda.Context, share: shard.Share) !bool {
    const width_ok = (share.world == 4 and share.kda.len() == 16) or (share.world == 2 and share.kda.len() == 32);
    return width_ok and
        try ctx.attribute(.compute_capability_major) == 12 and
        try ctx.attribute(.compute_capability_minor) == 1;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, sha: []const u8) !cuda.Module {
    const data = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(4 << 20), .@"16", null);
    defer gpa.free(data);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), sha)) return error.PromptKernelImageHash;
    return cuda.Module.load(d, data);
}
