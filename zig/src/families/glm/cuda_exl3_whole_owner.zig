//! Optional whole-prompt module; each forward's existing arena owns its scratch.
const std = @import("std");
const cuda = @import("cuda");
const combine_sha = "0cbf2a88946b3c6e622c315f24adaabb00163932838f315031c134a1a8e19071";
const order_sha = "790f9c700849162f1b990ed3c7658881f04ad2193743ef21a21ad1b960e53d5f";
const sha = "65c062c3d9f46342c76a29dc844aa57e228dccbc08903c9f877e105b1bc6fc39";

pub fn install(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, e: *cuda.exl3_experts.Experts, on: bool, exl3: bool) !void {
    const order_path = @import("sys").environment.Process.get(.{}, "TF_GLM_EXL3_PROMPT_ORDER_IMAGE");
    const combine_path = @import("sys").environment.Process.get(.{}, "TF_GLM_EXL3_PROMPT_DIRECT_SLOT_IMAGE");
    if (!on) {
        if (combine_path != null) return error.Exl3WholeCombineScope;
        if (order_path != null) return error.Exl3WholeOrderScope;
        return;
    }
    if (!exl3) return error.Exl3WholeScope;
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_EXL3_PROMPT_WHOLE_IMAGE") orelse
        return error.Exl3WholeImageRequired;
    try load(gpa, io, d, path, e);
    if (order_path) |file| {
        var order = try loadImage(gpa, io, d, file, order_sha);
        errdefer order.unload();
        try e.enableWholeOrder(order);
        std.log.info("GLM_EXL3_PROMPT_ORDER image={s}", .{order_sha});
    }
    if (combine_path) |file| {
        var combine = try loadImage(gpa, io, d, file, combine_sha);
        errdefer combine.unload();
        try e.enableWholeCombine(combine);
        std.log.info("GLM_EXL3_PROMPT_DIRECT_SLOT image={s}", .{combine_sha});
    }
    std.log.info("GLM_EXL3_PROMPT_WHOLE image={s} rows=4096 direct_slots=9", .{sha});
}
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, e: *cuda.exl3_experts.Experts) !void {
    var module = try loadImage(gpa, io, d, path, sha);
    errdefer module.unload();
    try e.enableWhole(module);
}

fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, expected: []const u8) !cuda.Module {
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), expected)) return error.Exl3WholeImageHash;
    return cuda.Module.load(d, bytes);
}
pub fn attach(e: anytype) void {
    e.w.exl3_prompt_whole = e.settings.exl3_prompt_whole;
    if (e.settings.exl3_prompt_whole) e.m.exl3.whole_calls = &e.expert_prompt_whole_calls;
    if (e.m.exl3.whole) |whole| {
        if (whole.order_module != null) e.m.exl3.whole_order_calls = &e.expert_prompt_order_calls;
        if (whole.combine != null) e.m.exl3.whole_direct_calls = &e.expert_prompt_direct_calls;
    }
}
pub fn census(e: anytype) void {
    if (e.settings.exl3_fused_decode)
        std.log.info("rank {d}: GLM_EXL3_FUSED_DECODE_CLOSE calls={d}", .{ e.share.rank, e.expert_fused_calls });
    if (e.settings.exl3_prompt_whole)
        std.log.info("rank {d}: GLM_EXL3_PROMPT_WHOLE_CLOSE calls={d}", .{ e.share.rank, e.expert_prompt_whole_calls });
    if (e.m.exl3.whole_order_calls) |calls|
        std.log.info("rank {d}: GLM_EXL3_PROMPT_ORDER_CLOSE calls={d}", .{ e.share.rank, calls.* });
    if (e.m.exl3.whole_direct_calls) |calls|
        std.log.info("rank {d}: GLM_EXL3_PROMPT_DIRECT_SLOT_CLOSE calls={d}", .{ e.share.rank, calls.* });
}
