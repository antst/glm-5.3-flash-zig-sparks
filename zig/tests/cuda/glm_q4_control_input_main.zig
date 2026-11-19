//! Explicit component runner; no model or service entrypoint.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len != 2) return 2;
    const rank = try std.fmt.parseInt(usize, argv[1], 10);
    if (rank > 1) return 2;
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &d, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    @import("glm_q4_control_input_contract.zig").run(gpu, rank) catch |err| {
        std.debug.print("Q4_CONTROL_INPUT_FAIL {t}\n", .{err});
        return 1;
    };
    return 0;
}
