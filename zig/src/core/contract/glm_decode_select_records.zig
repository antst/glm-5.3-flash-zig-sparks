//! Separately compiled selector usage from pinned nvcc13.0.88; live/device gates are separate.
const registry = @import("registry.zig");
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_decode_select tf_glm_dsel_scan", .symbol = "tf_glm_dsel_scan", .image = "glm_decode_select", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 58, .static = 1032, .bound = 256 }, .{ .regs = 48, .static = 1032, .bound = 256 }, .{ .regs = 48, .static = 1032, .bound = 256 }, .{ .regs = 48, .static = 1032, .bound = 256 }, .{ .regs = 48, .static = 1032, .bound = 256 } } },
    .{ .launch = .{ .name = "glm_decode_select tf_glm_dsel_merge", .symbol = "tf_glm_dsel_merge", .image = "glm_decode_select", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 56, .static = 34056, .bound = 1024 }, .{ .regs = 64, .static = 34056, .bound = 1024 }, .{ .regs = 64, .static = 34056, .bound = 1024 }, .{ .regs = 64, .static = 34056, .bound = 1024 }, .{ .regs = 64, .static = 34056, .bound = 1024 } } },
};
