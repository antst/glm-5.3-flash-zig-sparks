//! Optional shared KDA row tables; state and replay buffers remain owned by the original sequence.
const std = @import("std");
const cuda = @import("cuda");
const seg = @import("cuda_kda_segments.zig");
const weights = @import("cuda_weights.zig");
const block = @import("cuda_kda_block.zig");
const state = @import("cuda_state.zig");
pub const image_sha = "1f2ef5524f67639f681e5a1a3334d0bc89bb4a456d067d7889a8e26be9ded07d";
const slots = 8;
const table_bytes = slots * @sizeOf(seg.Row);

pub const Owner = struct {
    gpa: std.mem.Allocator,
    module: cuda.Module,
    kernels: seg.Kernels,
    stream: cuda.Stream,
    tables: cuda.DeviceBuffer,
    staging: cuda.HostBuffer,
    uploaded: cuda.Event,
    pending: bool = false,
    unusable: bool = false,
    plans: []?seg.Plan,
    rows: usize,
    min_segments: usize = 8,
    calls: u64 = 0,
    fallbacks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, w: *const weights.Weights, rows: usize) !Owner {
        if (rows == 0 or rows > 64) return error.KdaSegmentScope;
        var count: usize = 0;
        for (w.layers) |l| if (l.mixer == .kda) {
            if (l.mixer.kda.heads != 16) return error.KdaSegmentScope;
            count += 1;
        };
        if (count == 0) return error.KdaSegmentScope;
        var module = try loadImage(gpa, io, d);
        errdefer module.unload();
        const kernels = (try seg.Kernels.load(module, true)).?;
        var tables = try cuda.DeviceBuffer.alloc(d, count * table_bytes);
        errdefer tables.free();
        var staging = try cuda.HostBuffer.alloc(d, tables.len);
        errdefer staging.free();
        @memset(staging.bytes, 0);
        var uploaded = try cuda.Event.init(d, false);
        errdefer uploaded.deinit();
        const plans = try gpa.alloc(?seg.Plan, count);
        @memset(plans, null);
        std.log.info("GLM_KDA_SEGMENTS_MODULE sha256={s} layers={d}", .{ image_sha, count });
        return .{ .gpa = gpa, .module = module, .kernels = kernels, .stream = stream, .tables = tables, .staging = staging, .uploaded = uploaded, .plans = plans, .rows = rows };
    }

    pub fn bytes(o: *const Owner) usize {
        return o.tables.len + o.staging.bytes.len;
    }

    pub fn prepare(o: *Owner, r: anytype, rows: usize) !void {
        if (o.unusable) return error.KdaSegmentsUnavailable;
        if (o.pending) try o.uploaded.synchronize();
        o.pending = false;
        @memset(o.plans, null);
        if (r.b.prefill or rows > o.rows or r.segs.len < o.min_segments or r.segs.len > slots or
            r.b.kda.wide.q == 0 or r.b.kda.wide.y == 0) return;
        var any = false;
        var li: usize = 0;
        for (r.w.layers) |l| if (l.mixer == .kda) {
            defer li += 1;
            const k = l.mixer.kda;
            var list: [slots]seg.Segment = undefined;
            for (r.segs, 0..) |g, i| {
                const st = g.st;
                const cur = st.cur[li];
                const kv = li * state.window_rows * st.heads * 128;
                list[i] = .{
                    .row0 = g.row0,
                    .rows = g.rows,
                    .p = .{ .ptr = st.proj + li * st.proj_layer * 2, .len = st.proj_layer * 2 },
                    .conv = .{ .ptr = st.conv + li * 3 * st.conv_c * 2, .len = 3 * st.conv_c * 2 },
                    .state_in = .{ .ptr = st.recAt(cur, li), .len = st.heads * 128 * 128 * 4 },
                    .state_out = .{ .ptr = st.recAt(1 - cur, li), .len = st.heads * 128 * 128 * 4 },
                    .k = .{ .ptr = st.save_k + kv * 4, .len = state.window_rows * st.heads * 128 * 4 },
                    .v = .{ .ptr = st.save_v + kv * 2, .len = state.window_rows * st.heads * 128 * 2 },
                    .g = .{ .ptr = st.save_g + kv * 4, .len = state.window_rows * st.heads * 128 * 4 },
                    .b = .{ .ptr = st.save_b + li * state.window_rows * st.heads * 4, .len = state.window_rows * st.heads * 4 },
                };
            }
            const plan = seg.Plan.make(list[0..r.segs.len], parameters(k), .{}) catch continue;
            if (plan.rows != rows) return error.BadWindow;
            o.plans[li] = plan;
            const bytes_plan = plan.bytes();
            @memcpy(o.staging.bytes[li * table_bytes ..][0..bytes_plan.len], bytes_plan);
            any = true;
        };
        if (!any) return;
        o.unusable = true;
        try o.stream.d.check(o.stream.d.api.cuMemcpyHtoDAsync_v2(o.tables.ptr, o.staging.bytes.ptr, o.staging.bytes.len, o.stream.handle), "cuMemcpyHtoDAsync");
        try o.uploaded.record(o.stream);
        o.pending = true;
        o.unusable = false;
    }

    pub fn run(o: *Owner, f: anytype, r: anytype, k: block.Layer, li: usize, rows: usize) !bool {
        if (li >= o.plans.len) return error.KdaSegmentScope;
        const p = if (o.plans[li]) |*v| v else {
            o.fallbacks += 1;
            return false;
        };
        if (p.rows != rows or !f.kda_wide) return error.KdaSegmentScope;
        const n = k.heads * 128;
        const wide = r.b.kda.wide;
        const s: seg.Storage = .{
            .table = .{ .ptr = o.tables.ptr + li * table_bytes, .len = table_bytes },
            .a = .{ .ptr = r.b.kda.ka, .len = r.b.rows * n * 2 },
            .gate = .{ .ptr = r.b.kda.kg, .len = r.b.rows * n * 2 },
            .q = .{ .ptr = wide.q, .len = r.b.rows * n * 4 },
            .y = .{ .ptr = wide.y, .len = r.b.rows * n * 2 },
            .out = .{ .ptr = r.b.kda.saves.out, .len = r.b.rows * n * 2 },
        };
        try seg.validate(p, s);
        for (r.segs) |g| {
            const dst = g.st.proj + li * g.st.proj_layer * 2;
            try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, r.b.kproj + g.row0 * k.matWidth() * 2, g.rows * k.matWidth() * 2, f.stream.handle), "cuMemcpyDtoDAsync");
        }
        try o.kernels.launch(f.stream, p, s);
        o.calls += 1;
        return true;
    }

    pub fn deinit(o: *Owner) void {
        o.stream.synchronize() catch {
            std.log.err("GLM_KDA_SEGMENTS_RETAINED stream not quiescent", .{});
            return;
        };
        std.log.info("GLM_KDA_SEGMENTS_CLOSE batched_layers={d} fallback_layers={d} min_streams={d}", .{ o.calls, o.fallbacks, o.min_segments });
        o.uploaded.deinit();
        o.staging.free();
        o.tables.free();
        o.module.unload();
        o.gpa.free(o.plans);
        o.* = undefined;
    }
};

fn parameters(k: block.Layer) seg.Parameters {
    const n = k.heads * 128;
    return .{ .heads = k.heads, .p_stride = k.matWidth(), .b_off = 3 * n + 256, .a_stride = n, .g_stride = n, .conv_w = .{ .ptr = k.conv_w, .len = 3 * n * 4 * 4 }, .a_log = .{ .ptr = k.a_log, .len = k.heads * 4 }, .dt_bias = .{ .ptr = k.dt_bias, .len = n * 4 }, .norm_w = .{ .ptr = k.norm_w, .len = 128 * 2 }, .lower = k.lower, .eps = k.eps };
}

fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !cuda.Module {
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_KDA_SEGMENTS_IMAGE") orelse
        return error.KdaSegmentsImageRequired;
    const bytes_image = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes_image);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes_image, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.KdaSegmentsImageHash;
    return cuda.Module.load(d, bytes_image);
}
