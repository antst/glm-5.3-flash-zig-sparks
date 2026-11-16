//! Actual local-head asset lookup with an expected-error logger and initialized recording Driver.
const std = @import("std");
const cuda = @import("cuda");
const attn = @import("glm").dflash_attention;
var errors: usize = 0;
pub const std_options: std.Options = .{ .logFn = log };
fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = scope;
    _ = fmt;
    _ = args;
    if (level == .err) errors += 1;
}

pub fn main() !void {
    const Fake = struct {
        var n: usize = 0;
        var missing: usize = 0;
        const names = [_][]const u8{ "df_block_qkv_tp4_v1", "df_attn_part_tp4_v1", "df_attn_merge_tp4_v1" };
        fn errorText(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
            out.* = "recorded absent asset";
            return 0;
        }
        fn get(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            std.debug.assert(n < names.len and std.mem.eql(u8, std.mem.span(name), names[n]));
            const i = n;
            n += 1;
            if (i == missing) return cuda.abi.error_not_found;
            out.* = @ptrFromInt(0x1000 + i * 256);
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Fake.get;
    d.api.cuGetErrorName = Fake.errorText;
    d.api.cuGetErrorString = Fake.errorText;
    errors = 0;
    const m: cuda.Module = .{ .d = &d, .handle = null };
    for (0..3) |which| {
        Fake.n = 0;
        Fake.missing = which;
        try std.testing.expectError(error.NotFound, attn.Kernels.load(m));
        try std.testing.expectEqual(which + 1, Fake.n);
        try std.testing.expectEqual(2 * (which + 1), errors);
    }
    Fake.n = 0;
    Fake.missing = 3;
    const k = try attn.Kernels.load(m);
    try std.testing.expectEqual(@as(usize, 3), Fake.n);
    try std.testing.expect(k.block_qkv.handle != k.attn_part.handle);
}
