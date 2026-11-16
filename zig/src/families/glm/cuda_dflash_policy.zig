//! Explicit native confidence-only policy experiment: no selector or common-core arithmetic changes.
const std = @import("std");
const lanes = @import("lanes");

pub const Options = struct { on: bool = false, floor: f64 = 0.7 };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Options {
    var p: Options = .{};
    if (get("TF_GLM_DFLASH_CONFIDENCE_ONLY")) |v| {
        if (std.mem.eql(u8, v, "1")) p.on = true else if (!std.mem.eql(u8, v, "0")) {
            problem.* = "TF_GLM_DFLASH_CONFIDENCE_ONLY: 0 or 1";
            return error.BadSetting;
        }
    }
    if (!p.on) return p; // OFF retains the old cutoff parser and its per-call behavior.
    if (get("TF_GLM_DRAFT_CONFIDENCE")) |v| {
        p.floor = std.fmt.parseFloat(f64, v) catch {
            problem.* = "TF_GLM_DRAFT_CONFIDENCE: finite0..1 under confidence-only policy";
            return error.BadSetting;
        };
    }
    if (!(p.floor >= 0 and p.floor <= 1)) {
        problem.* = "TF_GLM_DRAFT_CONFIDENCE: finite0..1 under confidence-only policy";
        return error.BadSetting;
    }
    return p;
}

pub fn from(settings: anytype) Options {
    return .{ .on = settings.dflash_confidence_only, .floor = settings.dflash_confidence_floor };
}

/// Called before startup.open: a requested unsupported global mode is refused, never silently activated.
pub fn require(p: Options, has_drafter: bool, drafts: bool) !void {
    if (!p.on) return;
    if (!has_drafter or !drafts) return error.GlmConfidenceOnlyNeedsDFlash;
    std.log.info("GLM_DFLASH_CONFIDENCE_ONLY on=true floor={d} greedy-only", .{p.floor});
}

/// Called after copy_policy.apply, which intentionally nulls confidence in its ordinary cost-priced mode.
pub fn apply(p: Options, base: lanes.Model) lanes.Model {
    if (!p.on) return base;
    var model = base;
    model.draft_confidence = p.floor;
    model.allocate_shared_drafts = false;
    return model;
}

/// Admission guard before a lane slot is selected or a target operation enqueued; plain sampling is unchanged.
pub fn check(drafts: bool, sampled: bool) !void {
    if (drafts and sampled) return error.GlmConfidenceOnlyNeedsGreedyDrafts;
}
