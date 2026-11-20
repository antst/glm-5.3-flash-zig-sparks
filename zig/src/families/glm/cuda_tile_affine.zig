//! Optional target affine group64 weights, using the existing tiled affine storage and FP32 kernels.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const tile = @import("cuda_tile.zig");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const quant = @import("cuda_affine_quant.zig");

pub fn require(s: anytype, world: usize, vision: bool) !void {
    if (s.q4_python_prompt and !s.q4_affine_mse) return error.Q4PromptScope;
    if (s.q4_python_splits and !s.q4_affine_mse) return error.Q4SplitScope;
    if (s.q4_head_fp8 and !s.q4_affine_mse) return error.HeadFp8Scope;
    if (!s.q4_affine_mse) return;
    if ((world != 2 and world != 4) or s.dense_bits != 4 or s.head_bits != @as(u8, if (s.q4_head_fp8) 8 else 16) or !s.tiled or s.pages or
        s.kv_host != 0 or s.verify_graphs or vision) return error.AffineTargetScope;
}

pub fn prepare(m: anytype, on: bool) !void {
    if (!on) return;
    try q4.enableWide(&m.q4, m.mods[5]);
    try q4.enableRows(&m.q4, m.mods[5]);
    m.mm.affine = .{ .q4 = m.q4, .sums = m.glue.group_sums };
}

pub fn pack(gpa: std.mem.Allocator, io: std.Io, w: *weights.Weights, m: *const mm.Kernels, qs: *const q4.Kernels, stream: cuda.Stream, d: *const cuda.Driver) !mm.Packed {
    return packWithMoments(gpa, io, w, m, qs, stream, d, null);
}

pub fn packWithMoments(gpa: std.mem.Allocator, io: std.Io, w: *weights.Weights, m: *const mm.Kernels, qs: *const q4.Kernels, stream: cuda.Stream, d: *const cuda.Driver, calibrated: ?*const @import("cuda_q8_moments.zig").Data) !mm.Packed {
    return packWithRecurrent(gpa, io, w, m, qs, stream, d, calibrated, false);
}

pub fn packWithRecurrent(gpa: std.mem.Allocator, io: std.Io, w: *weights.Weights, m: *const mm.Kernels, qs: *const q4.Kernels, stream: cuda.Stream, d: *const cuda.Driver, calibrated: ?*const @import("cuda_q8_moments.zig").Data, keep_recurrent: bool) !mm.Packed {
    return packWithMode(gpa, io, w, m, qs, stream, d, calibrated, @intFromBool(keep_recurrent), 0, null);
}

fn packWithMode(gpa: std.mem.Allocator, io: std.Io, w: *weights.Weights, m: *const mm.Kernels, qs: *const q4.Kernels, stream: cuda.Stream, d: *const cuda.Driver, calibrated: ?*const @import("cuda_q8_moments.zig").Data, mode: u8, family_mask: u8, ffn: ?*@import("cuda_q4_ffn_q8.zig").Owner) !mm.Packed {
    if (mode > 3 or family_mask > 7) return error.Q4RecurrentBf16Scope;
    const keep_recurrent = mode != 0;
    try tile.padKdaInputs(w, d, stream);
    var list: std.ArrayList(tile.Lin) = .empty;
    defer list.deinit(gpa);
    try tile.linears(w, &list);
    if (calibrated) |data| {
        try @import("cuda_affine_activation.zig").validateMap(data, list.items, w.head == .bf16);
    }
    if (ffn) |owner| try owner.prepare(calibrated orelse return error.Q4FfnQ8Scope, list.items, w.head == .bf16);
    var most: usize = 0;
    for (list.items) |l| most = @max(most, l.n * l.k * 2);
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
    var status = try cuda.DeviceBuffer.alloc(d, if (calibrated != null) 16 else 4);
    defer status.free();
    var module = try quant.load(gpa, io, d);
    defer module.unload();
    var refinement = try @import("cuda_affine_activation.zig").load(gpa, io, d, calibrated != null);
    defer if (refinement) |*owner| owner.deinit();
    var device_moments: ?cuda.DeviceBuffer = null;
    defer if (device_moments) |*buffer| buffer.free();
    errdefer @import("cuda_load_owner.zig").drain(stream, d);
    if (calibrated) |data| {
        var buffer = try cuda.DeviceBuffer.alloc(d, data.means.len * 8);
        errdefer buffer.free();
        try buffer.upload(0, std.mem.sliceAsBytes(data.means));
        device_moments = buffer;
    }
    const f = try module.function("tf_glm_q4_mse_v1");
    var ptrs: std.ArrayList(u64) = .empty;
    errdefer ptrs.deinit(gpa);
    var packed_bits: std.ArrayList(u8) = .empty;
    errdefer packed_bits.deinit(gpa);
    try packed_bits.ensureTotalCapacity(gpa, list.items.len);
    var freed: usize = 0;
    var groups: usize = 0;
    var improved: usize = 0;
    var zero_moments: usize = 0;
    var retained: usize = 0;
    var retained_bytes: usize = 0;
    var family_census: @import("cuda_q4_precision_families.zig").Census = .{};
    for (list.items, 0..) |l, i| {
        if (ffn) |owner| if (@import("cuda_q4_ffn_q8.zig").selected(w, l.field)) {
            var packed_buf = try owner.pack(m, calibrated.?, i, l, tmp);
            errdefer packed_buf.free();
            try ptrs.ensureUnusedCapacity(gpa, 1);
            if (tile.bufferAt(w, l.field.*)) |at| {
                freed += w.buffers.items[at].len -| packed_buf.len;
                tile.replace(w, at, packed_buf);
            } else try w.buffers.append(w.gpa, packed_buf);
            l.field.* = packed_buf.ptr;
            ptrs.appendAssumeCapacity(packed_buf.ptr);
            packed_bits.appendAssumeCapacity(8);
            continue;
        };

        const control = @import("cuda_q4_recurrent.zig").isRetained(w, l.field, mode);
        const family = @import("cuda_q4_precision_families.zig").selected(w, l.field, family_mask);
        if ((i + 1 == list.items.len and w.head == .bf16) or control or family != null) {
            if (control) {
                retained += 1;
                retained_bytes += l.n * l.k * 2;
            }
            if (family) |kind| family_census.note(kind, l);
            try @import("cuda_q4_recurrent.zig").retain(m, stream, d, l, tmp.ptr);
            continue;
        }
        const sizes = q4.bytes(l.n, l.k);
        const raw_words = l.n * l.k / 2;
        const raw_scales = l.n * (l.k / 64) * 2;
        if (raw_words + 2 * raw_scales > tmp.len) return error.Q4MseScratch;
        var packed_buf = try cuda.DeviceBuffer.alloc(d, sizes[0] + sizes[1] + sizes[2]);
        errdefer packed_buf.free();
        errdefer @import("cuda_load_owner.zig").drain(stream, d);
        try d.check(d.api.cuMemsetD8Async(status.ptr, 0, if (calibrated != null) 16 else 4, stream.handle), "cuMemsetD8Async");
        try quant.quantize(f, stream, l.field.*, l.n, l.k, tmp.ptr, tmp.ptr + raw_words, tmp.ptr + raw_words + raw_scales, status.ptr);
        if (refinement) |owner| {
            const offset = calibrated.?.entries[i].offset;
            try @import("cuda_affine_activation.zig").launch(owner.function, stream, l.n, l.k, .{
                .source = .{ .ptr = l.field.*, .len = l.n * l.k * 2 },
                .moments = .{ .ptr = device_moments.?.ptr + offset * 8, .len = l.k * 8 },
                .words = .{ .ptr = tmp.ptr, .len = raw_words },
                .scales = .{ .ptr = tmp.ptr + raw_words, .len = raw_scales },
                .biases = .{ .ptr = tmp.ptr + raw_words + raw_scales, .len = raw_scales },
                .status = .{ .ptr = status.ptr, .len = 16 },
            });
            groups += l.n * l.k / 64;
        }
        try q4.pack(qs, stream, tmp.ptr, tmp.ptr + raw_words, tmp.ptr + raw_words + raw_scales, .{ .w = packed_buf.ptr, .s = packed_buf.ptr + sizes[0], .b = packed_buf.ptr + sizes[0] + sizes[1], .n = l.n, .k = l.k, .npad = q4.npadOf(l.n) });
        try stream.synchronize();
        var invalid: u32 = 0;
        try cuda.memory.download(d, status.ptr, std.mem.asBytes(&invalid));
        if (invalid != 0) return error.Q4MseInvalidWeights;
        if (calibrated != null) {
            var counts: [3]u32 = undefined;
            try cuda.memory.download(d, status.ptr + 4, std.mem.asBytes(&counts));
            improved += counts[0];
            zero_moments += counts[1];
            if (counts[2] > 2) return error.Q4ActivationRounds;
        }
        try ptrs.ensureUnusedCapacity(gpa, 1);
        if (tile.bufferAt(w, l.field.*)) |at| {
            freed += w.buffers.items[at].len -| packed_buf.len;
            tile.replace(w, at, packed_buf);
        } else try w.buffers.append(w.gpa, packed_buf);
        l.field.* = packed_buf.ptr;
        ptrs.appendAssumeCapacity(packed_buf.ptr);
        packed_bits.appendAssumeCapacity(4);
    }
    const result = try ptrs.toOwnedSlice(gpa);
    errdefer gpa.free(result);
    const bits = try packed_bits.toOwnedSlice(gpa);
    @import("cuda_packed_tags.zig").sort(result, bits);
    w.affine_target = true;
    w.mixed_ffn_q8 = ffn != null;
    std.log.info("rank {d}: GLM_Q4_AFFINE_MSE active=true group=64 clips=16 head={d} linears={d} freed={d} image={s}", .{ w.share.rank, @as(u8, if (w.head == .fp8g) 8 else 16), result.len, freed, quant.image_sha });
    if (calibrated != null) std.log.info("GLM_Q4_ACTIVATION rank={d} active=true groups={d} improved={d} zero={d} rounds=2", .{
        w.share.rank, groups, improved, zero_moments,
    });
    if (keep_recurrent) std.log.info("GLM_Q4_RECURRENT_BF16 rank={d} active=true matrices={d} bytes={d}", .{ w.share.rank, retained, retained_bytes });
    family_census.report(w.share.rank, family_mask);
    if (ffn) |owner| owner.report();
    return .{ .ptrs = result, .bits = bits, .affine = true };
}

pub const Context = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    affine: bool,
    q8_mse: bool = false,
    q8_correlated: @import("cuda_q8_correlated_options.zig").Choice = .{},
    q4_activation: bool = false,
    q4_recurrent_bf16: u8 = 0,
    q4_bf16_families: u8 = 0,
    q4_ffn_q8: @import("cuda_q4_ffn_q8.zig").Choice = .{},
    calibration: ?@import("cuda_q8_moments.zig").Identity = null,
    head_fp8: bool = false,
    lane: ?*const cuda.qmmf.Lane = null,
    qs: *const @import("cuda_q4.zig").Kernels,
    mk: *const mm.Kernels,
    s: cuda.Stream,
    d: *const cuda.Driver,
    enabled: bool,
    dense: u8,
    head: u8,
    packs: *mm.Packed,
    pub fn configure(t: *@This(), s: anytype) void {
        t.q8_mse = s.q8_mse;
        t.q8_correlated = s.q8_correlated;
        t.q4_activation = s.q4_activation;
        t.q4_recurrent_bf16 = s.q4_recurrent_bf16;
        t.q4_bf16_families = s.q4_bf16_families;
        t.q4_ffn_q8 = s.q4_ffn_q8;
    }
    pub fn apply(t: @This(), w: *weights.Weights) !void {
        if (!t.enabled) return;
        if (t.q4_activation and (!t.affine or t.calibration == null)) return error.Q4ActivationScope;
        var calibrated: ?*@import("cuda_q8_moments.zig").Data = null;
        defer if (calibrated) |data| data.destroy();
        if (t.calibration) |id| {
            if (t.affine) {
                if (!t.q4_activation or t.q8_mse or w.share.world != 2 or w.mtp != null or w.head != .bf16)
                    return error.Q4ActivationScope;
            } else if (!t.q8_mse or t.q4_activation or w.share.world != 2 or w.mtp != null or w.head != .bf16)
                return error.Q8ActivationScope;
            const root = @import("cuda_ranks.zig").environment("TF_GLM_CALIBRATION_ROOT") orelse return error.Q8CalibrationIdentity;
            calibrated = try @import("cuda_q8_moments.zig").Data.open(t.gpa, t.io, root, id, w.share.rank);
        }
        if (t.q8_correlated.on and (t.affine or calibrated == null or !t.q8_mse)) return error.Q8CorrelatedScope;
        const rounding = try @import("cuda_q8_correlated_owner.zig").Owner.open(t.gpa, t.io, t.d, t.s, t.q8_correlated, if (t.calibration) |id| id.checkpoint else @splat(0), w.share.rank);
        defer if (rounding) |owner| owner.destroy();
        if (t.q4_ffn_q8.on and (!t.affine or calibrated == null or !t.q4_activation)) return error.Q4FfnQ8Scope;
        const ffn = try @import("cuda_q4_ffn_q8.zig").Owner.open(t.gpa, t.io, t.d, t.s, t.q4_ffn_q8,
            if (t.calibration) |id| id.checkpoint else @splat(0), w.share.rank);
        defer if (ffn) |owner| owner.destroy();
        if (t.q4_recurrent_bf16 == 2) try @import("cuda_q4_control_input.zig").prepare(w, t.mk, t.s, t.d);
        if (t.head_fp8) try @import("cuda_head_fp8.zig").pack(t.gpa, t.io, w, t.lane orelse return error.HeadFp8Scope, t.s, t.d);
        t.packs.* = if (t.affine) try packWithMode(t.gpa, t.io, w, t.mk, t.qs, t.s, t.d, calibrated, t.q4_recurrent_bf16, t.q4_bf16_families, ffn) else try tile.tileLinearsCorrelated(t.gpa, w, t.mk, t.s, t.d, t.dense, t.head, t.q8_mse, calibrated, rounding);
    }
};
