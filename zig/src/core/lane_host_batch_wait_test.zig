//! The actual lane worker collects to a fixed deadline and wakes for arrivals, cancellation and close.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const Host = @import("lane_host.zig").LaneHost;
const wait = @import("lane_host_batch_wait.zig");
const a = std.testing.allocator;
const request: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 4 };

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: usize = 0,
    ids: [3][4]u32 = @splat(@splat(0)),
    n: [3]usize = @splat(0),
    reasons: [3]?api.Reason = @splat(null),
    fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        const i: usize = @intCast(id - 1);
        if (e.* == .tokens) for (e.tokens) |token| {
            b.ids[i][b.n[i]] = token;
            b.n[i] += 1;
        };
        if (e.* == .finished) {
            b.reasons[i] = e.finished.reason;
            b.done += 1;
        }
    }
    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    fn await(b: *Box, n: usize) !void {
        for (0..1000) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const done = b.done;
            b.mutex.unlock(std.testing.io);
            if (done == n) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.TestTimedOut;
    }
};

const Control = struct {
    const Action = enum { add, add_earlier, add_late, spurious, timeout, cancel, close };
    var table: std.Io.VTable = undefined;
    var ns: std.atomic.Value(i64) = .init(0);
    var start: i64 = 0;
    var host: *Host = undefined;
    var box: *Box = undefined;
    var actions: []const Action = &.{};
    var deadlines: [8]i96 = undefined;
    var expected_deadlines: [8]i96 = undefined;
    var calls: usize = 0;
    var added: u64 = 1;
    var failed: bool = false;
    fn io(script: []const Action) std.Io {
        start = @intCast(std.Io.Clock.awake.now(std.testing.io).toNanoseconds());
        ns.store(start, .monotonic);
        calls = 0;
        added = 1;
        failed = false;
        actions = script;
        table = std.testing.io.vtable.*;
        table.now = now;
        table.futexWait = futex;
        return .{ .userdata = std.testing.io.userdata, .vtable = &table };
    }
    fn now(ctx: ?*anyopaque, clock: std.Io.Clock) std.Io.Timestamp {
        if (clock == .awake) return .{ .nanoseconds = ns.load(.monotonic) };
        return std.testing.io.vtable.now(ctx, clock);
    }
    fn futex(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const end = if (timeout == .deadline) timeout.deadline.raw.nanoseconds else 0;
        if (ptr != &host.wake.epoch.raw or end < start or end > start + 4_000_000)
            return std.testing.io.vtable.futexWait(ctx, ptr, expected, timeout);
        if (calls >= actions.len or calls >= deadlines.len) {
            failed = true;
            ns.store(@intCast(end), .monotonic);
            return;
        }
        deadlines[calls] = end;
        var oldest = host.queued.items[0].submitted;
        for (host.queued.items[1..]) |j| oldest = @min(oldest, j.submitted);
        expected_deadlines[calls] = oldest + @as(i96, if (host.queued.items.len == 1) 1_000_000 else 4_000_000);
        const action = actions[calls];
        calls += 1;
        if (!host.mutex.tryLock()) {
            failed = true;
            return error.Canceled;
        }
        host.mutex.unlock(host.io);
        _ = ns.fetchAdd(500_000, .monotonic);
        switch (action) {
            .add, .add_earlier, .add_late => {
                if (action == .add_late) _ = ns.fetchAdd(1_000_000, .monotonic);
                added += 1;
                host.engine().submit(added, &request, box.sink()) catch {
                    failed = true;
                    return error.Canceled;
                };
                if (action == .add_earlier) {
                    host.lock();
                    host.queued.items[host.queued.items.len - 1].submitted = start - 250_000;
                    host.unlock();
                }
            },
            .spurious => {},
            .timeout => ns.store(@intCast(end), .monotonic),
            .cancel => host.engine().cancel(1),
            .close => {
                host.lock();
                host.closing = true;
                host.wake.signal(host.io);
                host.unlock();
            },
        }
    }
};

fn worker(on: bool, script: []const Control.Action, count: usize, reason: api.Reason, expired: bool, prequeued: usize) ![3][4]u32 {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.batched(), clock.clock());
    defer core.deinit();
    var box: Box = .{};
    const io = Control.io(script);
    var h = Host.init(a, io, &core, .{ .lanes = 3 });
    h.admission_quiet_ns = if (on) 4_000_000 else 0;
    Control.host = &h;
    Control.box = &box;
    try h.engine().submit(1, &request, box.sink());
    const initial = if (on) prequeued else count;
    for (2..initial + 1) |id| try h.engine().submit(id, &request, box.sink());
    Control.added = initial;
    if (expired) Control.ns.store(Control.start + 5_000_000, .monotonic);
    try h.start();
    var stopped = false;
    defer if (!stopped) h.stop();
    try box.await(count);
    h.stop();
    stopped = true;
    try std.testing.expect(!Control.failed);
    for (box.reasons[0..count]) |got| try std.testing.expectEqual(reason, got.?);
    if (reason == .length) {
        try std.testing.expectEqual(@as(usize, if (count > 1) 1 else 0), target.many_passes);
    } else try std.testing.expectEqual(@as(usize, 0), target.many_passes + target.prefill_count);
    if (on) {
        try std.testing.expectEqual(script.len, Control.calls);
        try std.testing.expectEqualSlices(i96, Control.expected_deadlines[0..Control.calls], Control.deadlines[0..Control.calls]);
        for (Control.deadlines[0..Control.calls]) |end|
            try std.testing.expect(end <= @as(i96, Control.start) + 4_000_000);
    } else try std.testing.expectEqual(@as(usize, 0), Control.calls);
    return box.ids;
}

test "admission wait: actual worker collects arrivals, releases mutex and keeps OFF token bytes" {
    const off = try worker(false, &.{}, 3, .length, false, 1);
    const on = try worker(true, &.{ .add, .spurious, .add }, 3, .length, false, 1);
    try std.testing.expectEqualDeep(off, on);
    try std.testing.expect(Control.ns.load(.monotonic) < Control.start + 4_000_000);
}

test "admission wait: solo and spurious wakes keep the oldest submit deadline" {
    _ = try worker(true, &.{ .spurious, .timeout }, 1, .length, false, 1);
    try std.testing.expectEqual(Control.start + 1_000_000, Control.ns.load(.monotonic));
}

test "admission wait: cancellation and close wake the actual worker before any prefill" {
    _ = try worker(true, &.{.cancel}, 1, .cancelled, false, 1);
    _ = try worker(true, &.{.close}, 1, .cancelled, false, 1);
}

test "admission wait: ineligible queues and full capacity read no timer" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8 }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.batched(), clock.clock());
    defer core.deinit();
    var box: Box = .{};
    const io = Control.io(&.{});
    var h = Host.init(a, io, &core, .{ .lanes = 3 });
    h.admission_quiet_ns = 4_000_000;
    Control.host = &h;
    Control.box = &box;
    var r = request;
    try h.engine().submit(1, &r, box.sink());
    const job = h.queued.items[0];
    r.background = true;
    try std.testing.expect(wait.ready(&h));
    r.background = false;
    r.max_tokens = 0;
    try std.testing.expect(wait.ready(&h));
    r.max_tokens = 4;
    r.chunks = &.{1};
    try std.testing.expect(wait.ready(&h));
    r.chunks = &.{};
    r.logprobs = 1;
    try std.testing.expect(wait.ready(&h));
    r.logprobs = null;
    job.prefix.leader = job;
    try std.testing.expect(wait.ready(&h));
    job.prefix.leader = null;
    h.memory_lanes = 1;
    try std.testing.expect(wait.ready(&h));
    h.memory_lanes = null;
    const batch = core.backend;
    core.backend = target.backend();
    try std.testing.expect(wait.ready(&h));
    core.backend = batch;
    try h.engine().submit(2, &request, box.sink());
    try h.engine().submit(3, &request, box.sink());
    try std.testing.expect(wait.ready(&h));
    try std.testing.expectEqual(@as(usize, 0), Control.calls);
    try h.start();
    defer h.stop();
    try box.await(3);
}

test "admission wait: active decode admits without a collection wait" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8 }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.batched(), clock.clock());
    defer core.deinit();
    var box: Box = .{};
    const io = Control.io(&.{});
    var h = Host.init(a, io, &core, .{ .lanes = 3 });
    Control.host = &h;
    Control.box = &box;
    try h.engine().submit(1, &request, box.sink());
    try std.testing.expect(h.admitOne());
    h.admission_quiet_ns = 4_000_000;
    try h.engine().submit(2, &request, box.sink());
    try std.testing.expect(h.admitOne());
    try std.testing.expectEqual(@as(usize, 0), Control.calls);
    try h.start();
    defer h.stop();
    try box.await(2);
}

test {
    _ = @import("lane_host_batch_wait.zig");
}

test "admission wait: an already expired submit deadline adds no wait" {
    const off = try worker(false, &.{}, 1, .length, false, 1);
    const on = try worker(true, &.{}, 1, .length, true, 1);
    try std.testing.expectEqualDeep(off, on);
}

test "admission wait: an already queued pair uses the fixed hard deadline" {
    _ = try worker(true, &.{.timeout}, 2, .length, false, 2);
    try std.testing.expectEqual(@as(i96, Control.start) + 4_000_000, Control.deadlines[0]);
}

test "admission wait: earlier submissions shorten and late wakes never reset the hard deadline" {
    _ = try worker(true, &.{ .add_earlier, .timeout }, 2, .length, false, 1);
    try std.testing.expectEqual(@as(i96, Control.start) + 3_750_000, Control.deadlines[1]);
    _ = try worker(true, &.{ .add_late, .timeout }, 2, .length, false, 1);
    try std.testing.expectEqual(@as(i96, Control.start) + 4_000_000, Control.deadlines[1]);
}

test "admission wait: normal lane trace keeps submit and round events without admission diagnostics" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/normal.jsonl", .{tmp.sub_path});
    defer a.free(path);
    const Trace = @import("lane_trace.zig");
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8 }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.batched(), clock.clock());
    defer core.deinit();
    var box: Box = .{};
    var h = Host.init(a, std.testing.io, &core, .{ .lanes = 3 });
    h.admission_quiet_ns = 4_000_000;
    h.trace = try Trace.Trace.open(a, std.testing.io, path);
    for (1..4) |id| try h.engine().submit(id, &request, box.sink());
    try h.start();
    var stopped = false;
    defer if (!stopped) h.stop();
    try box.await(3);
    h.stop();
    stopped = true;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    defer a.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "\"admit\"") == null);
    var lines = std.mem.tokenizeScalar(u8, bytes, '\n');
    var submits: usize = 0;
    var rounds: usize = 0;
    while (lines.next()) |line| {
        const e = try std.json.parseFromSlice(Trace.Event, a, line, .{});
        defer e.deinit();
        try std.testing.expect(e.value.ok);
        submits += @intFromBool(e.value.kind == .submit);
        rounds += @intFromBool(e.value.kind == .round);
    }
    try std.testing.expectEqual(@as(usize, 3), submits);
    try std.testing.expect(rounds > 0);
    for (box.reasons) |reason| try std.testing.expectEqual(api.Reason.length, reason.?);
}
