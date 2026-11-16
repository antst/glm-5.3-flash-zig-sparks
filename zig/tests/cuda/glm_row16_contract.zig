//! Explicit width16 qualification; the original shipped kernels and precision are unchanged.
const check = @import("check.zig");

pub fn run(gpu: check.Gpu) !void {
    try @import("glm_row16_state.zig").run(gpu);
    try @import("glm_row16_attn.zig").run(gpu);
    try @import("glm_row16_shared.zig").run(gpu);
    try @import("glm_row16_exl3.zig").run_(gpu);
    check.pass("GLM_ROW16_CONTRACT: State/partial-commit, BF16+FP8 attention and EXL3 decode complete", .{});
}

pub fn command(gpu: check.Gpu, cmd: []const u8) !void {
    if (@import("std").mem.eql(u8, cmd, "glm-row16-contract")) return run(gpu);
    try @import("glm_row16_state.zig").run32(gpu);
    try @import("glm_row16_attn.zig").run32(gpu);
    try @import("glm_row16_shared.zig").run32(gpu);
    try @import("glm_row16_exl3.zig").run32(gpu);
    check.pass("GLM_ROW32_CONTRACT: State/partial-commit, BF16/FP8 attention, shared64 and EXL3 complete", .{});
}
