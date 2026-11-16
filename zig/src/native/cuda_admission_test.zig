//! Actual native Host startup propagates the family's optional idle collection limit before worker start.
const std = @import("std");
const lanes = @import("lanes");

pub fn run(comptime Host: type, comptime serve: anytype) !void {
    const t = std.testing;
    const Release = struct {
        fn close(_: *anyopaque) void {}
    };
    for ([_]u64{ 0, 4_000_000 }) |ns| {
        var target: lanes.fake.Fake = .{ .gpa = t.allocator };
        defer target.deinit();
        const h = try t.allocator.create(Host);
        defer t.allocator.destroy(h);
        h.* = .{ .gpa = t.allocator, .gpu = null, .family = &target, .release = Release.close, .inner = target.batched(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .admission_quiet_ns = ns };
        try serve(h, t.io, .{ .exact_width = 8 }, 8, .{ .lanes = 3 }, null);
        defer {
            h.host.stop();
            h.core.deinit();
            h.cfg.deinit(t.allocator);
        }
        try t.expectEqual(ns, h.host.admission_quiet_ns);
        try t.expect(h.host.thread != null);
    }
}
