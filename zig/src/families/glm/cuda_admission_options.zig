//! A startup-agreed opt-in for bounded idle whole-prompt collection; default zero keeps admission immediate.
const std = @import("std");

pub fn read(value: ?[]const u8, problem: *[]const u8) !u32 {
    const v = value orelse return 0;
    if (std.mem.eql(u8, v, "0")) return 0;
    if (std.mem.eql(u8, v, "4000")) return 4000;
    problem.* = "TF_GLM_ADMISSION_US: 0 or 4000";
    return error.BadSetting;
}

pub fn require(o: anytype, world: usize) !void {
    const s = o.settings;
    if (s.admission_us == 0) return;
    if (s.admission_us != 4000 or (world != 2 and world != 4) or !o.mtp or !o.drafter or o.vision != null or
        s.pages or s.kv_host != 0 or !@import("cuda_target_precision.zig").supported(s)) return error.AdmissionWaitScope;
}
