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
    try tile.padKdaInputs(w, d, stream);
    var list: std.ArrayList(tile.Lin) = .empty;
    defer list.deinit(gpa);
    try tile.linears(w, &list);
    var most: usize = 0;
    for (list.items) |l| most = @max(most, l.n * l.k * 2);
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
    var status = try cuda.DeviceBuffer.alloc(d, 4);
    defer status.free();
    var module = try quant.load(gpa, io, d);
    defer module.unload();
    errdefer @import("cuda_load_owner.zig").drain(stream, d);
    const f = try module.function("tf_glm_q4_mse_v1");
    var ptrs: std.ArrayList(u64) = .empty;
    errdefer ptrs.deinit(gpa);
    var freed: usize = 0;
    for (list.items, 0..) |l, i| {
        if (i + 1 == list.items.len and w.head == .bf16) {
            try mm.tile(m, stream, l.field.*, tmp.ptr, l.n, l.k);
            try d.check(d.api.cuMemcpyDtoDAsync_v2(l.field.*, tmp.ptr, l.n * l.k * 2, stream.handle), "cuMemcpyDtoDAsync");
            try stream.synchronize();
            continue;
        }
        const sizes = q4.bytes(l.n, l.k);
        const raw_words = l.n * l.k / 2;
        const raw_scales = l.n * (l.k / 64) * 2;
        if (raw_words + 2 * raw_scales > tmp.len) return error.Q4MseScratch;
        var packed_buf = try cuda.DeviceBuffer.alloc(d, sizes[0] + sizes[1] + sizes[2]);
        errdefer packed_buf.free();
        errdefer @import("cuda_load_owner.zig").drain(stream, d);
        try d.check(d.api.cuMemsetD8Async(status.ptr, 0, 4, stream.handle), "cuMemsetD8Async");
        try quant.quantize(f, stream, l.field.*, l.n, l.k, tmp.ptr, tmp.ptr + raw_words, tmp.ptr + raw_words + raw_scales, status.ptr);
        try q4.pack(qs, stream, tmp.ptr, tmp.ptr + raw_words, tmp.ptr + raw_words + raw_scales, .{ .w = packed_buf.ptr, .s = packed_buf.ptr + sizes[0], .b = packed_buf.ptr + sizes[0] + sizes[1], .n = l.n, .k = l.k, .npad = q4.npadOf(l.n) });
        try stream.synchronize();
        var invalid: u32 = 0;
        try cuda.memory.download(d, status.ptr, std.mem.asBytes(&invalid));
        if (invalid != 0) return error.Q4MseInvalidWeights;
        try ptrs.ensureUnusedCapacity(gpa, 1);
        if (tile.bufferAt(w, l.field.*)) |at| {
            freed += w.buffers.items[at].len -| packed_buf.len;
            tile.replace(w, at, packed_buf);
        } else try w.buffers.append(w.gpa, packed_buf);
        l.field.* = packed_buf.ptr;
        ptrs.appendAssumeCapacity(packed_buf.ptr);
    }
    const result = try ptrs.toOwnedSlice(gpa);
    errdefer gpa.free(result);
    const bits = try gpa.alloc(u8, result.len);
    @memset(bits, 4);
    std.mem.sort(u64, result, {}, std.sort.asc(u64));
    w.affine_target = true;
    std.log.info("rank {d}: GLM_Q4_AFFINE_MSE active=true group=64 clips=16 head={d} linears={d} freed={d} image={s}", .{ w.share.rank, @as(u8, if (w.head == .fp8g) 8 else 16), result.len, freed, quant.image_sha });
    return .{ .ptrs = result, .bits = bits, .affine = true };
}

pub const Context = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    affine: bool,
    q8_mse: bool = false,
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
    pub fn apply(t: @This(), w: *weights.Weights) !void {
        if (!t.enabled) return;
        var calibrated: ?*@import("cuda_q8_moments.zig").Data = null;
        defer if (calibrated) |data| data.destroy();
        if (t.calibration) |id| {
            if (t.affine or !t.q8_mse or w.share.world != 2 or w.mtp != null or w.head != .bf16) return error.Q8ActivationScope;
            const root = @import("cuda_ranks.zig").environment("TF_GLM_CALIBRATION_ROOT") orelse return error.Q8CalibrationIdentity;
            calibrated = try @import("cuda_q8_moments.zig").Data.open(t.gpa, t.io, root, id, w.share.rank);
        }
        if (t.head_fp8) try @import("cuda_head_fp8.zig").pack(t.gpa, t.io, w, t.lane orelse return error.HeadFp8Scope, t.s, t.d);
        t.packs.* = if (t.affine) try pack(t.gpa, t.io, w, t.mk, t.qs, t.s, t.d) else try tile.tileLinearsFit(t.gpa, w, t.mk, t.s, t.d, t.dense, t.head, t.q8_mse, calibrated);
    }
};
