//! Optional architectural FFN Q8 inside an affine-Q4 target. Reuses the qualified Q8 seed/operator.
const std = @import("std");
const cuda = @import("cuda");
const tile = @import("cuda_tile.zig");
const mm = @import("cuda_mm.zig");
const moments = @import("cuda_q8_moments.zig");
const q8_owner = @import("cuda_q8_correlated_owner.zig");
pub const Choice = @import("cuda_q8_correlated_options.zig").Choice;

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Choice {
    const on = try @import("cuda_dense16.zig").flag(get("TF_GLM_Q4_FFN_Q8"), problem, "TF_GLM_Q4_FFN_Q8: 0 or 1");
    if (!on) return .{};
    const raw = get("TF_GLM_Q4_FFN_Q8_SHA256") orelse {
        problem.* = "FFN Q8 requires TF_GLM_Q4_FFN_Q8_SHA256";
        return error.BadSetting;
    };
    const digest = moments.digest(raw) catch {
        problem.* = "TF_GLM_Q4_FFN_Q8_SHA256: exactly64 hexadecimal digits";
        return error.BadSetting;
    };
    return .{ .on = true, .sha = digest };
}

pub fn require(o: anytype, world: usize) !void {
    try @import("cuda_q4_activation_options.zig").require(o, world);
    const s = o.settings;
    if (!s.q4_ffn_q8.on) return;
    if (world != 2 or !s.q4_activation or !s.q4_affine_mse or s.dense_bits != 4 or
        s.q8_mse or s.q8_activation or s.q8_correlated.on or s.q4_bf16_families != 0 or s.q4_recurrent_bf16 != 2)
        return error.Q4FfnQ8Scope;
}

pub fn selected(w: *const @import("cuda_weights.zig").Weights, field: *const u64) bool {
    return @import("cuda_q4_precision_families.zig").which(w, field) == .ffn;
}

pub const Owner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    rounding: *q8_owner.Owner,
    host: std.ArrayList(u16) = .empty,
    words: std.ArrayList(u32) = .empty,
    scales: std.ArrayList(u16) = .empty,
    weighted: @import("cuda_q8_activation.zig").Stats = .{},
    matrices: usize = 0,
    bytes: usize = 0,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, choice: Choice, checkpoint: [32]u8, rank: usize) !?*Owner {
        if (!choice.on) return null;
        const rounding = (try q8_owner.Owner.open(gpa, io, d, stream, choice, checkpoint, rank)).?;
        errdefer rounding.destroy();
        const self = try gpa.create(Owner);
        self.* = .{ .gpa = gpa, .d = d, .stream = stream, .rounding = rounding };
        return self;
    }

    pub fn destroy(self: *Owner) void {
        self.rounding.destroy(); // drains before covariance/module/host buffers are released
        self.host.deinit(self.gpa);
        self.words.deinit(self.gpa);
        self.scales.deinit(self.gpa);
        const gpa = self.gpa;
        gpa.destroy(self);
    }

    pub fn prepare(self: *Owner, data: *const moments.Data, list: anytype, head: bool) !void {
        try self.rounding.prepare(data, list, head);
    }

    /// Original-list ordinal is preserved even when other architectural fields stay BF16/Q4.
    pub fn pack(self: *Owner, mk: *const mm.Kernels, data: *const moments.Data, ordinal: usize, l: tile.Lin, tmp: cuda.DeviceBuffer) !cuda.DeviceBuffer {
        const count = try std.math.mul(usize, l.n, l.k);
        _ = try std.math.mul(usize, count, 8); // packedBytes uses a checked bit count at this API boundary
        const sizes = mm.packedBytes(l.n, l.k, 8);
        if (count == 0 or l.k % 64 != 0 or sizes[0] + sizes[1] > tmp.len)
            return error.Q4FfnQ8Capacity;
        try self.host.resize(self.gpa, count);
        try self.words.resize(self.gpa, count / 4);
        try self.scales.resize(self.gpa, count / 64);
        try self.stream.synchronize();
        try cuda.memory.download(self.d, l.field.*, std.mem.sliceAsBytes(self.host.items));
        const q = try @import("cuda_q8_activation.zig").quantize(self.host.items, l.n, l.k, try data.linear(ordinal, l.n, l.k), self.words.items, self.scales.items);
        self.weighted.groups += q.groups;
        self.weighted.improved += q.improved;
        self.weighted.baseline_error += q.baseline_error;
        self.weighted.selected_error += q.selected_error;
        try tmp.upload(0, std.mem.sliceAsBytes(self.words.items));
        try tmp.upload(sizes[0], std.mem.sliceAsBytes(self.scales.items));
        try self.rounding.run(ordinal, l.n, l.k, l.field.*, tmp.ptr, tmp.ptr + sizes[0]);
        var out = try cuda.DeviceBuffer.alloc(self.d, sizes[0] + sizes[1]);
        errdefer out.free();
        errdefer @import("cuda_load_owner.zig").drain(self.stream, self.d);
        try mm.tilePacked(mk, self.stream, 8, tmp.ptr, tmp.ptr + sizes[0], out.ptr, out.ptr + sizes[0], l.n, l.k);
        try self.stream.synchronize();
        self.matrices += 1;
        self.bytes += out.len;
        return out;
    }

    pub fn report(self: *const Owner) void {
        std.log.info("GLM_Q4_FFN_Q8 rank={d} active=true matrices={d} bytes={d} groups={d} weighted_before={d} weighted_after={d} original_q8_fitter=true", .{
            self.rounding.rank,           self.matrices,                self.bytes, self.weighted.groups,
            self.weighted.baseline_error, self.weighted.selected_error,
        });
        self.rounding.report();
    }
};
