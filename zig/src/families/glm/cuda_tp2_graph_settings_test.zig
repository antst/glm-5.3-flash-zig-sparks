const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const native = @import("dflash_native_graph.zig");

const Env = struct {
    fn get(name: [:0]const u8) ?[]const u8 {
        for ([_][]const u8{ "TF_GLM_DFLASH_NATIVE_GRAPH", "TF_GLM_DFLASH_GRAPHS", "TF_GLM_DFLASH_MLP_TP4" }) |key|
            if (std.mem.eql(u8, name, key)) return "1";
        if (std.mem.eql(u8, name, "TF_GLM_DFLASH_ATTN_TP4")) return "0";
        if (std.mem.eql(u8, name, "TF_KV_PAGES")) return "off";
        return null;
    }
};

test "TP2 direct: actual native graph settings parse accepts the qualified replicated attention shape" {
    var problem: []const u8 = "";
    const settings = ranks.Settings.read(Env.get, &problem) catch |err| {
        std.debug.print("TP2 native graph parser rejected: {s}\n", .{problem});
        return err;
    };
    try std.testing.expect(settings.dflash_native_graph and settings.dflash_graphs and settings.dflash_mlp_wide);
    try std.testing.expect(!settings.dflash_attn_tp4);
    try native.scope(2, true, settings.dflash_attn_tp4, settings.dflash_graphs);
    try native.require(2, settings, true);
    try std.testing.expectError(error.DFlashNativeGraphScope, native.require(4, settings, true));
    try std.testing.expectError(error.DFlashNativeGraphScope, native.require(2, settings, false));
    for ([_]usize{ 0, 1, 3, 8 }) |world|
        try std.testing.expectError(error.DFlashNativeGraphScope, native.require(world, settings, true));
    var tp4 = settings;
    tp4.dflash_attn_tp4 = true;
    try native.require(4, tp4, true);
    try std.testing.expectError(error.DFlashNativeGraphScope, native.require(2, tp4, true));
    var off = settings;
    off.dflash_native_graph = false;
    try native.require(1, off, false);
}

test "TP2 direct: native graph parser retains missing dependency cache and strict flag refusals" {
    const Check = struct {
        var missing: []const u8 = "";
        var flag: []const u8 = "1";
        fn get(name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_GLM_DFLASH_NATIVE_GRAPH")) return flag;
            if (std.mem.eql(u8, name, missing)) return "0";
            return Env.get(name);
        }
    };
    defer {
        Check.missing = "";
        Check.flag = "1";
    }
    var problem: []const u8 = "";
    for ([_][]const u8{ "TF_GLM_DFLASH_GRAPHS", "TF_GLM_DFLASH_MLP_TP4", "TF_GRAPH_CACHE" }) |key| {
        Check.missing = key;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Check.get, &problem));
    }
    Check.missing = "";
    for ([_][]const u8{ "", "yes", "2", "01", "+1" }) |value| {
        Check.flag = value;
        try std.testing.expectError(error.BadSetting, ranks.Settings.read(Check.get, &problem));
    }
}
