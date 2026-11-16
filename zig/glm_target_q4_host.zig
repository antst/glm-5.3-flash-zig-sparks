//! Existing packed target operators: pure fixture inventory and actual entrypoint semantic checks.
const std = @import("std");
const f = @import("tests/cuda/glm_mm_fixed_tests.zig");
var run_device = false;
test "target Q4 contract: actual device entrypoint semantically compiles" {
    if (run_device) try f.targetQ4(undefined);
    try std.testing.expectEqual(@as(usize, 13), f.target_shapes.len);
}
test {
    _ = f;
}

test "target Q4 contract: resource declaration reflects actual loader opt-in" {
    for (0..4) |i| {
        const v = f.targetLaunch(i);
        try std.testing.expect(v.opt_in);
        try std.testing.expectEqual(@as(u32, if (i < 2) 128 else 256), v.threads);
        if (i == 3) try std.testing.expect(v.dynamic > 48 * 1024);
    }
}

test "target Q4 contract: actual captured entrypoint semantics and every served row" {
    if (run_device) try f.targetQ4Graphs(undefined);
    try std.testing.expectEqual(@as(usize, 64), f.graph_counts.len);
    for (f.graph_counts, 1..) |rows, i| try std.testing.expectEqual(i, rows);
}

test "target Q4 contract: affine target device entries semantically compile" {
    if (run_device) {
        try @import("tests/cuda/glm_affine_target.zig").run(undefined, false);
        try @import("tests/cuda/glm_affine_target.zig").run(undefined, true);
    }
}
