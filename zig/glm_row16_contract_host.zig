//! Explicit host semantic and oracle tests, no device execution.
const std = @import("std");
const f = @import("tests/cuda/glm_row16_contract.zig");
var run_device = false;
test "GLM row16 contract: actual device entrypoints semantically compile" {
    if (run_device) {
        try f.run(undefined);
        try f.command(undefined, "glm-row32-contract");
    }
}
test {
    _ = @import("tests/cuda/glm_row16_state.zig");
    _ = @import("tests/cuda/glm_row16_attn.zig");
    _ = @import("tests/cuda/glm_row16_exl3.zig");
}
