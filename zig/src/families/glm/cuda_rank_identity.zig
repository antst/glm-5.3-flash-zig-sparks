//! Checkpoint byte identities shared by startup agreement.
const std = @import("std");
const hashBytes = @import("cuda_rank_options.zig").hashBytes;

pub fn hashFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![32]u8 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    return hashBytes(bytes);
}

pub fn artifact(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    return @import("core").checkpoint_identity.fullIdentity(gpa, io, dir);
}
