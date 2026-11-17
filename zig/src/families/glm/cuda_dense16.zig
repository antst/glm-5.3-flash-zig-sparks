//! Startup-only strict flags and the independent sixteen-row dense attention scope.
const dense = @import("cuda_dense_segments.zig");

pub fn flag(value: ?[]const u8, problem: *[]const u8, choices: []const u8) !bool {
    return dense.enabled(value) catch {
        problem.* = choices;
        return error.BadSetting;
    };
}

pub fn check(s: anytype, problem: *[]const u8) !void {
    if (s.dense_rows16 and (!s.dense_segments or s.verify_rows < 16)) {
        problem.* = "TF_GLM_DENSE_ROWS16 requires dense segments and verify width>=16";
        return error.BadSetting;
    }
}

pub fn graph(s: anytype, cache: ?[]const u8, problem: *[]const u8) !void {
    const cuda = @import("cuda");
    if (s.dflash_native_graph and (!s.dflash_graphs or !s.dflash_mlp_wide or
        !(cuda.graph_cache.Settings{}).operator(cache).on))
    {
        problem.* = "TF_GLM_DFLASH_NATIVE_GRAPH requires graphs, MLP split and cache on";
        return error.BadSetting;
    }
}
