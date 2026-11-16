//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const p = std.c.getenv("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    const s = std.mem.span(p);
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = comptime labels: {
        var names: [32][]const u8 = undefined;
        for (&names, 1..) |*name, n| name.* = std.fmt.comptimePrint("verify {d} row{s}", .{ n, if (n == 1) "" else "s" });
        break :labels names;
    };
    return l[@min(rows, l.len) - 1];
}
