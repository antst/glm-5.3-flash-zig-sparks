//! Real progress codec and bounded coordinator policy, without devices or elapsed-time sleeps.
const std = @import("std");
const glm = @import("glm");
const p = glm.nvme_progress;
const t = std.testing;
const second = std.time.ns_per_s;

test "GLM NVMe progress: packets preserve completion credits and never manufacture a protocol failure" {
    try t.expect(@sizeOf(p.Watch) <= 128);
    const work = [_]u64{ 0, 1, 0x7fffffff, (@as(u64, 1) << 47) - 1, std.math.maxInt(u64) };
    for (0..4) |left| {
        for (0..4) |right| {
            for (work) |a| {
                for (work) |b| {
                    const replies = [_]u64{ p.packet(@intCast(left), a), p.packet(@intCast(right), b) };
                    const want: u64 = @intCast(left & right);
                    try t.expect(glm.nvme_wire.allOk(&replies));
                    try t.expectEqual(a & ((@as(u64, 1) << 47) - 1), p.progress(replies[0]));
                    try t.expectEqual(b & ((@as(u64, 1) << 47) - 1), p.progress(replies[1]));
                    try t.expectEqual(want, p.completed(replies[0] & replies[1]));
                    try t.expectEqual(want, p.completed(replies[0]) & p.completed(replies[1]));
                    try t.expect(!glm.nvme_wire.allOk(&.{replies[0] | glm.nvme_wire.failed}));
                }
            }
        }
    }
}

test "GLM NVMe progress: a healthy write continues beyond the former aggregate cutoff" {
    const idle = 4500 * std.time.ns_per_ms;
    var watch = p.Watch.init(0, idle, p.hard_ns, &.{p.packet(0, 1000)});
    for (1..8) |i| {
        const now: i128 = @as(i128, @intCast(i)) * second;
        try t.expectEqual(p.Watch.Reason.running, watch.reason(now));
        try t.expect(watch.observe(&.{p.packet(0, 1000 + i)}, now));
    }
    try t.expect(7 * second > idle);
    try t.expectEqual(p.Watch.Reason.running, watch.reason(7 * second));
}

test "GLM NVMe progress: stale work and completion bits do not renew an idle write" {
    const idle = 4500 * std.time.ns_per_ms;
    var watch = p.Watch.init(0, idle, p.hard_ns, &.{p.packet(0, 123)});
    try t.expect(!watch.observe(&.{p.packet(1, 123)}, 4 * second));
    try t.expectEqual(p.Watch.Reason.running, watch.reason(idle - 1));
    try t.expectEqual(p.Watch.Reason.idle, watch.reason(idle));
}

test "GLM NVMe progress: each rank can demonstrate work after other ranks finish" {
    var replies = [_]u64{ p.packet(1, 100), p.packet(1, 200), p.packet(1, 300), p.packet(0, 400) };
    var watch = p.Watch.init(0, 5 * second, p.hard_ns, &replies);
    replies[3] = p.packet(0, 401);
    try t.expect(watch.observe(&replies, 4 * second));
    try t.expectEqual(p.Watch.Reason.running, watch.reason(8 * second));
    try t.expect(!watch.observe(&replies, 8 * second));
    try t.expectEqual(p.Watch.Reason.idle, watch.reason(9 * second));
}

test "GLM NVMe progress: even endless reported work cannot extend the independent absolute bound" {
    var watch = p.Watch.init(0, 5 * second, p.hard_ns, &.{p.packet(0, 0)});
    for (1..30) |i| {
        const now = @as(i128, @intCast(i)) * second;
        try t.expectEqual(p.Watch.Reason.running, watch.reason(now));
        try t.expect(watch.observe(&.{p.packet(0, i)}, now));
    }
    try t.expect(!watch.observe(&.{p.packet(0, 31)}, p.hard_ns));
    try t.expectEqual(p.Watch.Reason.absolute, watch.reason(p.hard_ns));
}

test "GLM NVMe progress: modular work counters and a backwards clock remain bounded" {
    const last = (@as(u64, 1) << 47) - 1;
    var watch = p.Watch.init(0, 5 * second, p.hard_ns, &.{p.packet(0, last)});
    try t.expect(watch.observe(&.{p.packet(0, last + 1)}, second));
    try t.expect(!watch.observe(&.{p.packet(0, last + 1)}, 2 * second));
    try t.expectEqual(p.Watch.Reason.clock, watch.reason(0));
    try t.expect(!watch.observe(&.{p.packet(0, 123)}, 0));
    try t.expectEqual(p.Watch.Reason.idle, watch.reason(6 * second));
}
