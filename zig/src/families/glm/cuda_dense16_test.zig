const std = @import("std");
const dense = @import("cuda_dense_segments.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

test "dense segment rows16 preserves legacy guard and maps wide parts into the same row ABI" {
    const patterns = [_][]const usize{ &.{9}, &.{15}, &.{16}, &.{ 9, 15 }, &.{ 16, 16 }, &.{ 16, 15, 9, 8, 7, 5, 3, 1 } };
    for (patterns) |lengths| for ([_]bool{ false, true }) |fp8| {
        var parts: [8]dense.Segment = undefined;
        for (lengths, 0..) |rows, i| parts[i] = .{
            .rows = rows,
            .host_pos = 509 + i,
            .capacity = 4096,
            .cache = 0x10000000 + i * 0x1000000,
            .scales = if (fp8) 0x20000000 + i * 0x1000000 else 0,
            .position = 0x30000000 + i * 4,
        };
        var scope: dense.Scope = .{ .dense_limit = 2048, .scratch_rows = 64, .scratch_chunks = 6 };
        try t.expectError(error.DenseRows, dense.Plan.make(parts[0..lengths.len], scope));
        scope.max_part_rows = 16;
        const plan = try dense.Plan.make(parts[0..lengths.len], scope);
        var row0: usize = 0;
        for (parts[0..lengths.len]) |part| {
            for (plan.table[row0..][0..part.rows]) |row| {
                try t.expectEqual(part.cache, row.cache);
                try t.expectEqual(part.scales, row.scales);
                try t.expectEqual(part.position, row.position);
                try t.expectEqual(@as(i32, @intCast(row0)), row.row0);
                try t.expectEqual(@as(i32, @intCast((part.host_pos + part.rows + 511) / 512)), row.chunks);
            }
            row0 += part.rows;
        }
        try t.expectEqual(row0, plan.rows);
        try t.expectEqual(row0 * 32, plan.bytes().len);
        try t.expect(plan.most >= 9 and plan.most <= 16);
        parts[0].rows = 17;
        try t.expectError(error.DenseRows, dense.Plan.make(parts[0..lengths.len], scope));
    };
}

test "dense segment rows16 strict cached setting and agreement refuse unsupported settings" {
    const Env = struct {
        var value: ?[]const u8 = null;
        var width: []const u8 = "16";
        var enabled: []const u8 = "1";
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DENSE_ROWS16")) return value;
            if (std.mem.eql(u8, name, "TF_GLM_VERIFY_ROWS")) return width;
            if (std.mem.eql(u8, name, "TF_GLM_DENSE_LATENT_SEGMENTS")) return enabled;
            return null;
        }
    };
    defer {
        Env.value = null;
        Env.width = "16";
        Env.enabled = "1";
    }
    var problem: []const u8 = "";
    const off = try ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.dense_rows16);
    Env.value = "1";
    const on = try ranks.Settings.read(Env.get, &problem);
    const FP = ranks.Fingerprint;
    const a = FP.of(4, 0, off, 4096, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 4096, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 65), a.layout);
    try t.expectEqual(FP.Field.dense_rows16, ranks.mismatch(&.{ a, b }).?.field);
    Env.width = "32";
    try t.expect((try ranks.Settings.read(Env.get, &problem)).dense_rows16);
    Env.width = "8";
    try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    Env.width = "16";
    Env.enabled = "0";
    try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    Env.enabled = "1";
    for ([_][]const u8{ "", "yes", "2" }) |v| {
        Env.value = v;
        try t.expectError(error.BadSetting, ranks.Settings.read(Env.get, &problem));
    }
    const Engine = @import("cuda_engine.zig").Engine;
    try t.expectError(error.DenseRowsScope, Engine.init(t.allocator, undefined, undefined, undefined, "", undefined, null, 0, 4, .{ .settings = .{ .dense_rows16 = true }, .capacity = 4096, .mtp = true }));
}
