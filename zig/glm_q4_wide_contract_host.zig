//! CPU inventory and semantic checks only; device execution requires the explicit tf-cuda-test command.
const std = @import("std");
const core = @import("core");
const fixture = @import("tests/cuda/contract_glm_q4_f32.zig");
var run_device = false;

test "GLM Q4 wide contract: real device entrypoint semantically compiles but never executes here" {
    if (run_device) try fixture.run(undefined);
}

test "GLM Q4 wide contract: fp32 declaration includes served local-O rows split K and tails" {
    for ([_]bool{ false, true }) |strided| {
        const s = fixture.spec(strided);
        try std.testing.expectEqual(core.contract.spec.Elem.f32, s.out);
        try std.testing.expectEqual(core.contract.spec.Accum.f32, s.accum);
        try std.testing.expect(s.replaces == null);
        for ([_]usize{ 1, 2, 4, 6, 8, 16 }) |rows| {
            var found = false;
            for (s.served) |v| if (v.rows == rows and v.n == 4096 and v.k == 1024) {
                found = true;
            };
            try std.testing.expect(found);
        }
    }
}
