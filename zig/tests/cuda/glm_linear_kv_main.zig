//! Standalone device test for ordinary and optimized GLM attention on linear page mappings.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const disk = argv.len == 3 and std.mem.eql(u8, argv[1], "--disk");
    const long = argv.len == 5 and std.mem.eql(u8, argv[4], "--long");
    if (argv.len != 4 and !disk and !long) {
        std.debug.print("usage: tf-glm-linear-kv GLM-GLUE PREFILL-H16 PREFILL-H32 [--long] or --disk DIR\n", .{});
        return 2;
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &d, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    if (disk) {
        try @import("glm_linear_disk_contract.zig").run(gpu, argv[2]);
        return 0;
    }
    try @import("glm_linear_kv_contract.zig").testWindows(gpu, .{ argv[1], argv[2], argv[3] }, long);
    return 0;
}
