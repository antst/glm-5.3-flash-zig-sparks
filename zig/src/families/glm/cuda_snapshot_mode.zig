//! Agreed snapshot layout and cache identity, with no per-request environment reads.
const std = @import("std");

pub fn enabled(e: anytype) bool {
    if (@hasField(@TypeOf(e.*), "settings")) {
        if (@hasField(@TypeOf(e.settings), "kv_live_half")) return e.settings.kv_live_half;
    }
    return false;
}

pub fn layout(on: bool) @import("cuda_disk_image.zig").Words.Layout {
    return if (on) .live_kda else .unsplit;
}

pub fn head(e: anytype) bool {
    return e.mtp and !@import("cuda_dflash_head_mode.zig").omitted(e) and (!enabled(e) or e.drafter == null);
}

pub fn layoutFor(e: anytype) @import("cuda_disk_image.zig").Words.Layout {
    return if (enabled(e) and e.drafter != null) .live_dflash else layout(enabled(e));
}

pub fn compatibility(base: [32]u8, on: bool) [32]u8 {
    return compatibilityFor(base, on, false);
}

pub fn compatibilityFor(base: [32]u8, on: bool, external_draft: bool) [32]u8 {
    if (!on) return base;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("glm-native-tiered-live-kda-v1");
    hash.update(&base);
    hash.update(&.{@intFromBool(external_draft)});
    return hash.finalResult();
}
