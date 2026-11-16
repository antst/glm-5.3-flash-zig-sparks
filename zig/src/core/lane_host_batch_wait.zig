//! Idle whole-prompt collection: a lone probe and one fixed hard deadline; live decode never waits here.
const std = @import("std");
const Host = @import("lane_host.zig").LaneHost;
const prefix = @import("lane_host_prefix.zig");
const fetch = @import("lane_host_fetch.zig");
const pressure = @import("lane_host_pressure.zig");

/// False leaves queued jobs for the ordinary cancel/close path; the engine worker owns this call.
pub fn ready(h: *Host) bool {
    if (h.admission_quiet_ns == 0) return true;
    h.lock();
    defer h.unlock();
    if (interrupted(h)) return false;
    if (!collects(h)) return true;
    var end = oldest(h) + h.admission_quiet_ns;
    while (collects(h)) {
        end = @min(end, oldest(h) + h.admission_quiet_ns);
        const bound = if (h.queued.items.len == 1) @min(end, oldest(h) + 1_000_000) else end;
        if (std.Io.Clock.awake.now(h.io).toNanoseconds() >= bound) break;
        const until: std.Io.Timeout = .{ .deadline = .{
            .raw = .{ .nanoseconds = bound },
            .clock = .awake,
        } };
        h.wake.waitTimeout(h.io, &h.mutex, until) catch |err| {
            if (err == error.Timeout) return !interrupted(h);
            return false;
        };
    }
    return !interrupted(h);
}

fn interrupted(h: *const Host) bool {
    return h.closing or h.stale or h.cancels.items.len > 0;
}

fn oldest(h: *const Host) i96 {
    var ns = h.queued.items[0].submitted;
    for (h.queued.items[1..]) |j| ns = @min(ns, j.submitted);
    return ns;
}

/// Only an otherwise idle ordinary queue waits; all existing admission and cache decisions stay downstream.
fn collects(h: *Host) bool {
    if (interrupted(h) or h.queued.items.len == 0 or h.admitted.items.len > 0 or h.filling.items.len > 0 or
        h.core.activeCount() > 0 or h.rounds != null or h.core.fills() or !h.core.batchesPrompts() or
        h.lessons.items.len > 0 or h.lesson_open or h.recall != null or pressure.holds(h)) return false;
    const free = @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes);
    if (free < 2 or h.queued.items.len >= free) return false;
    for (h.queued.items) |j| {
        const r = j.request;
        if (r.background or r.max_tokens == 0 or r.embedding != null or r.readout != null or
            r.images.len > 0 or r.structure != null or r.logprobs != null or r.prompt_logprobs != null or
            r.decode_spans.len > 0 or r.chunks.len > 0 or prefix.waiting(j) or fetch.held(h, j)) return false;
    }
    return true;
}

test "admission wait: OFF returns before reading undefined owner or clock state" {
    var host: Host = undefined;
    host.admission_quiet_ns = 0;
    try std.testing.expect(ready(&host));
}
