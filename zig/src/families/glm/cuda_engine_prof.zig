//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const cuda = @import("cuda");
const Prof = @import("cuda_prof.zig").Prof;
const Engine = @import("cuda_engine.zig").Engine;
/// One shared-forward event owner; its label stays alive through the existing Prof report.
pub const Shared = struct {
    gpa: std.mem.Allocator,
    prof: *Prof,
    serial: u64 = 0,
    label: [192]u8 = undefined,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, io: std.Io, wanted: bool) !?*Shared {
        return initWith(gpa, d, io, wanted, Prof.init);
    }

    pub fn initWith(gpa: std.mem.Allocator, d: *const cuda.Driver, io: std.Io, wanted: bool, comptime create: anytype) !?*Shared {
        if (!wanted) return null;
        const p = try gpa.create(Shared);
        errdefer gpa.destroy(p);
        p.* = .{ .gpa = gpa, .prof = try create(d, io) };
        p.prof.every = 1;
        return p;
    }

    pub fn begin(p: *Shared, stream: cuda.Stream, rank: usize, rows: usize, streams: usize) !void {
        if (rows == 0 or rows > @import("cuda_engine.zig").shared_rows or streams == 0 or
            streams > @import("cuda_engine.zig").max_shared or streams > rows) return error.SharedProfileShape;
        if (p.prof.open) return error.SharedProfileOpen;
        p.serial = try std.math.add(u64, p.serial, 1);
        errdefer p.cancel();
        try p.prof.begin(stream, "verify_shared");
        // Prof.begin and TF_LANE_TRACE use exactly the same awake-clock nanosecond domain.
        p.prof.label = try std.fmt.bufPrint(&p.label, "verify_shared rank={d} R={d} S={d} seq={d} tstart_ns={d}", .{ rank, rows, streams, p.serial, p.prof.t0 });
    }

    /// Called only after the actual target and every part's draw succeeded on the owning stream.
    pub fn finish(p: *Shared, stream: cuda.Stream) void {
        if (!p.prof.open) return;
        const before = p.prof.n;
        p.prof.mark(stream, .draw);
        if (p.prof.n == before) {
            p.cancel(); // a missing terminal event must not publish an apparent target+draw total
            return;
        }
        p.prof.end();
    }

    /// Error unwind publishes no successful call and adds no device wait or event.
    pub fn cancel(p: *Shared) void {
        p.prof.open = false;
        p.prof.n = 0;
        p.prof.sections = @splat(0);
        p.prof.gpu_ms = 0;
        p.prof.wall_ms = 0;
        p.prof.calls = 0;
    }

    pub fn deinit(p: *Shared) void {
        p.prof.deinit();
        const gpa = p.gpa;
        gpa.destroy(p);
    }
};

/// The real verifyShared binds this exact owner after its rows/parts have been validated.
pub fn beginShared(e: *Engine, f: *@import("cuda_forward.zig").Forward, rows: usize, streams: usize) !?*Shared {
    const p = e.shared_prof orelse return null;
    try p.begin(e.stream, e.share.rank, rows, streams);
    f.prof = p.prof;
    return p;
}

pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    if (e.shared_prof) |p| p.deinit();
    e.shared_prof = null;
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const p = std.c.getenv("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    const s = std.mem.span(p);
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = comptime labels: {
        var names: [32][]const u8 = undefined;
        for (&names, 1..) |*name, n| name.* = std.fmt.comptimePrint("verify {d} row{s}", .{ n, if (n == 1) "" else "s" });
        break :labels names;
    };
    return l[@min(rows, l.len) - 1];
}
