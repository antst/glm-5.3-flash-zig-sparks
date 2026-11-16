//! Dense layers retain their stored format; an NVFP4 checkpoint may keep them BF16.
const std = @import("std");
const formats = @import("cuda_formats.zig");

pub fn read(l: anytype, i: u32) !formats.Format {
    var buf: [192]u8 = undefined;
    const module = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.gate_proj", .{i});
    const want: formats.Format = if (l.w.config.exl3) .bf16 else try l.formatOne(l.ck, module, &.{ .bf16, .mxfp8, .nvfp4 });
    inline for (.{ "gate_proj", "up_proj", "down_proj" }) |x|
        try l.format("layers.{d}.mlp." ++ x, .{i}, want);
    return want;
}

const Fake = struct {
    w: struct { config: struct { exl3: bool = false } = .{} } = .{},
    ck: void = {},
    declaration: formats.Declared = .nvfp4,
    stored: [3]formats.Stored,
    seen: usize = 0,

    pub fn formatOne(f: *Fake, _: void, module: []const u8, wants: []const formats.Format) !formats.Format {
        try std.testing.expectEqualStrings("model.language_model.layers.0.mlp.gate_proj", module);
        const got = try formats.of(f.declaration, f.stored[0]);
        for (wants) |want| if (got == want) return got;
        return error.UnreadableFormat;
    }

    pub fn format(f: *Fake, comptime fmt: []const u8, args: anytype, want: formats.Format) !void {
        var buf: [128]u8 = undefined;
        const module = try std.fmt.bufPrint(&buf, fmt, args);
        const index: usize = if (std.mem.endsWith(u8, module, "gate_proj")) 0 else if (std.mem.endsWith(u8, module, "up_proj")) 1 else 2;
        const got = try formats.of(f.declaration, f.stored[index]);
        if (got != want) return error.UnreadableFormat;
        f.seen += 1;
    }
};

const bf16: formats.Stored = .{ .weight = .{ .dtype = .bf16, .shape = &.{ 12288, 4096 } } };
const quantized: formats.Stored = .{
    .weight = .{ .dtype = .u8, .shape = &.{ 12288, 2048 } },
    .scale = .{ .dtype = .f8_e4m3, .shape = &.{ 12288, 256 } },
};

test "FFN graphs: actual dense selection retains BF16 under a single-format NVFP4 declaration" {
    var l: Fake = .{ .stored = .{ bf16, bf16, bf16 } };
    try std.testing.expectEqual(formats.Format.bf16, try read(&l, 0));
    try std.testing.expectEqual(@as(usize, 3), l.seen);
    l.w.config.exl3 = true;
    l.seen = 0;
    try std.testing.expectEqual(formats.Format.bf16, try read(&l, 0));
    try std.testing.expectEqual(@as(usize, 3), l.seen);
}

test "FFN graphs: dense NVFP4 dispatch remains quantized and mixed or malformed tensors refuse" {
    var l: Fake = .{ .stored = .{ quantized, quantized, quantized } };
    try std.testing.expectEqual(formats.Format.nvfp4, try read(&l, 0));
    l.stored[1] = bf16;
    try std.testing.expectError(error.UnreadableFormat, read(&l, 0));
    l.stored = .{ bf16, bf16, bf16 };
    l.stored[0].scale = quantized.scale;
    try std.testing.expectError(error.FormatMismatch, read(&l, 0));
    l.stored = .{ bf16, bf16, bf16 };
    l.declaration = .{ .mixed = .nvfp4 };
    try std.testing.expectError(error.FormatMismatch, read(&l, 0));
}
