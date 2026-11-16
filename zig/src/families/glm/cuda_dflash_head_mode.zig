//! Startup policy for omitting the unused target MTP layer beside an external DFlash drafter.
const weights = @import("cuda_weights.zig");

pub fn require(settings: anytype, external: bool, drafts: bool, vision: bool) !void {
    if (settings.dflash_only_head and (!external or !drafts or vision or settings.pages or settings.kv_host != 0))
        return error.DFlashHeadScope;
}

pub fn selection(settings: anytype, drafts: bool, external: bool) weights.Heads {
    return if (!drafts) .none else if (settings.dflash_only_head and external) .draft else .mtp;
}

pub fn omitted(e: anytype) bool {
    if (@hasField(@TypeOf(e.*), "settings")) {
        if (@hasField(@TypeOf(e.settings), "dflash_only_head")) return e.settings.dflash_only_head;
    }
    return false;
}
