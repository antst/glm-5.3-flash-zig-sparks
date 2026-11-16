//! Optional whole-prompt module; each forward's existing arena owns its scratch.
const std = @import("std");
const cuda = @import("cuda");
const sha = "65c062c3d9f46342c76a29dc844aa57e228dccbc08903c9f877e105b1bc6fc39";

pub fn install(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, e: *cuda.exl3_experts.Experts, on: bool, exl3: bool) !void {
    if (!on) return;
    if (!exl3) return error.Exl3WholeScope;
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_EXL3_PROMPT_WHOLE_IMAGE") orelse
        return error.Exl3WholeImageRequired;
    try load(gpa, io, d, path, e);
    std.log.info("GLM_EXL3_PROMPT_WHOLE image={s} rows=4096 direct_slots=9", .{sha});
}
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, e: *cuda.exl3_experts.Experts) !void {
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), sha)) return error.Exl3WholeImageHash;
    var module = try cuda.Module.load(d, bytes);
    errdefer module.unload();
    try e.enableWhole(module);
}
pub fn attach(e: anytype) void {
    e.w.exl3_prompt_whole = e.settings.exl3_prompt_whole;
    if (e.settings.exl3_prompt_whole) e.m.exl3.whole_calls = &e.expert_prompt_whole_calls;
}
pub fn census(e: anytype) void {
    if (e.settings.exl3_fused_decode)
        std.log.info("rank {d}: GLM_EXL3_FUSED_DECODE_CLOSE calls={d}", .{ e.share.rank, e.expert_fused_calls });
    if (e.settings.exl3_prompt_whole)
        std.log.info("rank {d}: GLM_EXL3_PROMPT_WHOLE_CLOSE calls={d}", .{ e.share.rank, e.expert_prompt_whole_calls });
}
