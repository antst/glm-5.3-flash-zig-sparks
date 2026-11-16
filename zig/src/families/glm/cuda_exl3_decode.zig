//! Optional switched-TP4 EXL3 decode kernels.
const std = @import("std");
const cuda = @import("cuda");
const sha = "9d69cdcdd1bb97a7bb7aaf448b8df844227062a0db4ade4f7f6323e35e61f78b";

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, e: *cuda.exl3_experts.Experts) !void {
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), sha)) return error.Exl3DecodeImageHash;
    var module = try cuda.Module.load(d, bytes);
    errdefer module.unload();
    try e.enableDecode(module);
}

pub fn configure(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, e: *cuda.exl3_experts.Experts) !void {
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_EXL3_FUSED_DECODE_IMAGE") orelse
        return error.Exl3DecodeImageRequired;
    try load(gpa, io, d, path, e);
    std.log.info("GLM_EXL3_FUSED_DECODE image={s}", .{sha});
}
