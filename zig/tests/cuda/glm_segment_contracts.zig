//! Explicit segment comparators; ordinary kernel-contract dispatch remains unchanged.
const std = @import("std");
const check = @import("check.zig");
const dense = @import("glm_dense_segmented_contract.zig");

pub fn run(cmd: []const u8, gpu: check.Gpu, rest: []const [:0]const u8) !bool {
    if (dense.matches(cmd)) {
        try dense.dispatch(cmd, gpu, rest);
        return true;
    }
    if (std.mem.eql(u8, cmd, "glm-kda-segmented-tp2-contract")) {
        try @import("glm_kda_tp2_contract.zig").run(gpu, rest);
        return true;
    }
    return false;
}
