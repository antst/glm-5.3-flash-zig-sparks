//! Startup-owned correlation data and qualified fixed-scale GPU rounding operator.
const std = @import("std");
const cuda = @import("cuda");
const fit = @import("cuda_q8_correlated.zig");
const input = @import("cuda_q8_correlation_input.zig");
const moments = @import("cuda_q8_moments.zig");
pub const Counts = struct {
    groups: u64 = 0,
    improved: u64 = 0,
    zero: u64 = 0,

    pub fn observe(self: *Counts, groups: usize, status: [4]u32) !void {
        if (status[0] != 0 or @as(u64, status[1]) + status[2] > groups or
            (status[3] != 0 and status[3] != 2) or (status[1] > 0 and status[3] != 2))
            return error.Q8CorrelatedStatus;
        self.groups += groups;
        self.improved += status[1];
        self.zero += status[2];
    }
};

pub fn validateInput(data: *const input.Data, diagonal: *const moments.Data, list: anytype, head: bool) !usize {
    @setFloatMode(.strict);
    try input.validateMap(data, list, head);
    if (diagonal.entries.len != data.entries.len or diagonal.target_count != data.target_count)
        return error.Q8CorrelationDiagonal;
    var most: usize = 0;
    for (list, 0..) |l, i| {
        const raw = try data.linear(i, l.n, l.k);
        const means = try diagonal.linear(i, l.n, l.k);
        const samples: f64 = @floatFromInt(data.entries[i].samples);
        for (means, 0..) |mean, j| {
            const normalized = raw[j / 64 * 2080 + input.triangle(j % 64, j % 64)] / samples;
            if (@as(u64, @bitCast(mean)) != @as(u64, @bitCast(normalized))) return error.Q8CorrelationDiagonal;
        }
        most = @max(most, l.k);
    }
    return most;
}

pub const Owner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    data: *input.Data,
    operator: fit.Owner,
    rank: usize,
    covariance: ?cuda.DeviceBuffer = null,
    status: ?cuda.DeviceBuffer = null,
    counts: Counts = .{},
    prepared: bool = false,
    failed: bool = false,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, choice: @import("cuda_q8_correlated_options.zig").Choice, checkpoint: [32]u8, rank: usize) !?*Owner {
        if (!choice.on) return null;
        const root = @import("cuda_ranks.zig").environment("TF_GLM_Q8_CORRELATION_ROOT") orelse
            return error.Q8CorrelationIdentity;
        const data = try input.Data.open(gpa, io, root, .{ .manifest = choice.sha, .checkpoint = checkpoint }, rank);
        errdefer data.destroy();
        var operator = (try fit.load(gpa, io, d, true)).?;
        errdefer operator.deinit();
        const self = try gpa.create(Owner);
        self.* = .{ .gpa = gpa, .d = d, .s = s, .data = data, .operator = operator, .rank = rank };
        return self;
    }

    pub fn destroy(self: *Owner) void {
        @import("cuda_load_owner.zig").drain(self.s, self.d);
        if (self.status) |*b| b.free();
        if (self.covariance) |*b| b.free();
        self.operator.deinit();
        self.data.destroy();
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub fn prepare(self: *Owner, diagonal: *const moments.Data, list: anytype, head: bool) !void {
        if (self.failed or self.prepared) return error.Q8CorrelatedOwnerState;
        self.failed = true;
        const k = try validateInput(self.data, diagonal, list, head);
        var covariance = try cuda.DeviceBuffer.alloc(self.d, k / 64 * 2080 * 8);
        errdefer covariance.free();
        const status = try cuda.DeviceBuffer.alloc(self.d, 16);
        self.covariance = covariance;
        self.status = status;
        self.prepared = true;
        self.failed = false;
    }

    pub fn run(self: *Owner, ordinal: usize, n: usize, k: usize, source: u64, words: u64, scales: u64) !void {
        if (self.failed or !self.prepared) return error.Q8CorrelatedOwnerState;
        self.failed = true;
        const values = try self.data.linear(ordinal, n, k);
        const b: fit.Buffers = .{
            .source = .{ .ptr = source, .len = n * k * 2 },
            .covariance = .{ .ptr = self.covariance.?.ptr, .len = values.len * 8 },
            .words = .{ .ptr = words, .len = n * k },
            .scales = .{ .ptr = scales, .len = n * k / 64 * 2 },
            .status = .{ .ptr = self.status.?.ptr, .len = 16 },
        };
        if (b.covariance.len > self.covariance.?.len) return error.Q8CorrelatedCapacity;
        const groups = try fit.validate(n, k, b);
        try self.covariance.?.upload(0, std.mem.sliceAsBytes(values));
        try self.d.check(self.d.api.cuMemsetD8Async(b.status.ptr, 0, 16, self.s.handle), "cuMemsetD8Async");
        try fit.launch(self.operator.function, self.s, n, k, b);
        try self.s.synchronize();
        const at = self.status.?.ptr;
        var counts: [4]u32 = undefined;
        try cuda.memory.download(self.d, at, std.mem.asBytes(&counts));
        try self.counts.observe(groups, counts);
        self.failed = false;
        if (ordinal % 16 == 0) std.log.info("GLM_Q8_CORRELATED rank={d} linear={d} improved={d}", .{ self.rank, ordinal, self.counts.improved });
    }

    pub fn report(self: *const Owner) void {
        std.log.info("GLM_Q8_CORRELATED rank={d} active=true groups={d} improved={d} zero={d} fixed_scales=true image={s} baseline=c18_weighted reported_diagonal_error=c18_incumbent_only", .{ self.rank, self.counts.groups, self.counts.improved, self.counts.zero, fit.image_sha });
    }
};
