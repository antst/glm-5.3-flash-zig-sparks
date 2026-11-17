//! Small load-time GLM rank option readers.
const std = @import("std");
const cuda = @import("cuda");

/// Whether TF_GLM_ROCE_MULTI's wire opens at TF_GLM_WORLD: three ranks or more (cuda_roce_multi); TP=2 stays on NCCL.
pub fn wireOpens(get: *const fn ([:0]const u8) ?[]const u8) bool {
    const v = get("TF_GLM_WORLD") orelse return false;
    return (std.fmt.parseInt(usize, v, 10) catch 0) >= 3;
}

pub fn ffnGraphs(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !bool {
    const on = try @import("cuda_dense16.zig").flag(
        get("TF_GLM_FFN_GRAPHS"),
        problem,
        "TF_GLM_FFN_GRAPHS: 0 or 1",
    );
    return on and (cuda.graph_cache.Settings{}).effective(get("TF_GRAPH_CACHE")).on;
}

/// Existing prompt-tail strict reader, shared with the cached rank settings.
pub fn promptTail(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !void {
    _ = try @import("cuda_dense16.zig").flag(get("TF_GLM_PROMPT_TAIL"), problem, "TF_GLM_PROMPT_TAIL: 0 or 1");
}

pub fn kdaWide(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !bool {
    const value = get("TF_GLM_KDA_DECODE_WIDE") orelse return true;
    return @import("cuda_dense16.zig").flag(value, problem, "TF_GLM_KDA_DECODE_WIDE: 0 or 1");
}

pub fn promptK(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !bool {
    const value = get("TF_GLM_PROMPT_K") orelse return false;
    if (std.mem.eql(u8, value, "shape")) return true;
    if (std.mem.eql(u8, value, "one")) return false;
    problem.* = "TF_GLM_PROMPT_K: one or shape";
    return error.BadSetting;
}

pub fn hashBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
