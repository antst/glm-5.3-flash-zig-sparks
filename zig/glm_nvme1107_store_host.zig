//! The unmodified 1107 bounded NVMe block store on the current native compiler.
test {
    _ = @import("src/core/nvme1107/spill/root.zig");
    _ = @import("src/core/nvme1107/layout.zig");
    @import("std").testing.refAllDecls(@import("src/core/nvme1107/tier_state.zig"));
    _ = @import("src/core/nvme1107/tier_state_test.zig");
    _ = @import("src/core/nvme1107/metadata.zig");
}
