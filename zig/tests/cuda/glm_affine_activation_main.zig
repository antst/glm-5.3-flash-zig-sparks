//! Explicit weighted-quantizer component only; the default CUDA runner is unchanged.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 or !std.mem.eql(u8, args[1], "glm-affine-activation-contract")) return 2;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    @import("glm_affine_activation_contract.zig").run(gpu, args[2]) catch |err| {
        std.debug.print("FAIL glm-affine-activation-contract: {t}\n", .{err});
        return 1;
    };
    return 0;
}
