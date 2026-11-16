//! Bounded pressure drain uses real work progress; completion credits and failure bits retain their wire meaning.
const std = @import("std");

pub const completion_mask: u64 = 0xffff;
const work_mask: u64 = (@as(u64, 1) << 47) - 1;
pub const hard_ns: u64 = 30 * std.time.ns_per_s;

pub fn packet(completion: u64, work: u64) u64 {
    return (completion & completion_mask) | ((work & work_mask) << 16);
}

pub fn completed(value: u64) u64 {
    return value & completion_mask;
}

pub fn progress(value: u64) u64 {
    return (value >> 16) & work_mask;
}

pub const Watch = struct {
    pub const Reason = enum { running, clock, idle, absolute };
    start: i128,
    last: i128,
    idle_ns: u64,
    absolute_ns: u64,
    seen: [8]u64 = @splat(0),

    pub fn init(start: i128, idle_ns: u64, absolute_ns: u64, baseline: []const u64) Watch {
        std.debug.assert(idle_ns > 0 and absolute_ns >= idle_ns);
        std.debug.assert(baseline.len <= 8);
        var w: Watch = .{ .start = start, .last = start, .idle_ns = idle_ns, .absolute_ns = absolute_ns };
        for (baseline, 0..) |value, rank| w.seen[rank] = progress(value);
        return w;
    }

    pub fn reason(w: *const Watch, now: i128) Reason {
        if (now < w.start or now < w.last) return .clock;
        if (now - w.start >= @as(i128, w.absolute_ns)) return .absolute;
        if (now - w.last >= @as(i128, w.idle_ns)) return .idle;
        return .running;
    }

    pub fn observe(w: *Watch, replies: []const u64, now: i128) bool {
        std.debug.assert(replies.len <= w.seen.len);
        if (now < w.last or now - w.start >= @as(i128, w.absolute_ns)) return false;
        var changed = false;
        for (replies, 0..) |value, rank| {
            const work = progress(value);
            changed = changed or work != w.seen[rank];
            w.seen[rank] = work;
        }
        if (changed) w.last = now;
        return changed;
    }
};
