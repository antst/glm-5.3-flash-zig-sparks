//! Explicit NVFP4 scope for the existing compute-only target FFN cache.
const std = @import("std");
const core = @import("core");
const exl3 = @import("cuda_exl3.zig");

pub fn wide(s: anytype) bool {
    return s.exl3_fused_decode or s.nvfp4_ffn_graphs;
}

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (!s.nvfp4_ffn_graphs) return;
    if (world != 4 or !s.ffn_graphs or s.exl3_fused_decode or !s.tiled or
        (s.verify_rows != 16 and s.verify_rows != 32) or s.verify_graphs or s.pages or s.kv_host != 0 or o.vision != null)
        return error.Nvfp4GraphScope;
}

pub fn checkpoint(gpa: std.mem.Allocator, text: []const u8, s: anytype) !void {
    if (!s.ffn_graphs) return;
    if (!s.nvfp4_ffn_graphs) {
        if (!exl3.isExl3(gpa, text)) return error.FfnGraphScope;
        return;
    }
    var why: core.modelopt.Why = .{};
    try core.modelopt.admit(gpa, text, &why);
}
