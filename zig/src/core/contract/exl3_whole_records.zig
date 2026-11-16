//! Exact nvcc13.0.88 records for the separate whole-prompt image, all five supported SMs.
const r = @import("registry.zig");
pub const records = [_]r.CudaRecord{
    .{ .launch = .{ .name = "EXL3 whole rotate", .symbol = "tf_exl3_prompt_rotate", .image = "exl3_prompt_whole", .threads = 32 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 }, .{ .regs = 20, .static = 0, .bound = 0 } }, .later_pending = true },
    .{ .launch = .{ .name = "EXL3 whole gateup", .symbol = "tf_exl3_prompt_gu", .image = "exl3_prompt_whole", .threads = 256 }, .usage = .{ .{ .regs = 126, .static = 43264, .bound = 256 }, .{ .regs = 128, .static = 43264, .bound = 256 }, .{ .regs = 128, .static = 43264, .bound = 256 }, .{ .regs = 124, .static = 43264, .bound = 256 }, .{ .regs = 124, .static = 43264, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "EXL3 whole down", .symbol = "tf_exl3_prompt_down", .image = "exl3_prompt_whole", .threads = 256 }, .usage = .{ .{ .regs = 126, .static = 43264, .bound = 256 }, .{ .regs = 126, .static = 43264, .bound = 256 }, .{ .regs = 126, .static = 43264, .bound = 256 }, .{ .regs = 126, .static = 43264, .bound = 256 }, .{ .regs = 126, .static = 43264, .bound = 256 } }, .later_pending = true },
};
pub const controls = [_]r.CudaRecord{blk: {
    var record = records[0];
    record.launch.name = "control: EXL3 whole above the device thread limit";
    record.launch.threads = 2048;
    break :blk record;
}};
