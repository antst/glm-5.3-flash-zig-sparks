//! Opt-in dense MLA table/partial owner. All device use and table uploads share the engine's main stream.
const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense_segments.zig");
const dsa = @import("cuda_dsa_block.zig");
const weights = @import("cuda_weights.zig");
pub const image_sha = "8dfd446605497436654d9926153478b93b19959ff79bbe042e2d598f69ae4669";

pub const Owner = struct {
    gpa: std.mem.Allocator,
    kernels: dense.Kernels,
    module: cuda.Module,
    stream: cuda.Stream,
    tables: cuda.DeviceBuffer,
    partials: cuda.DeviceBuffer,
    staging: cuda.HostBuffer,
    uploaded: cuda.Event,
    previous_upload: bool = false,
    unusable: bool = false,
    plans: []?dense.Plan,
    rows: usize,
    chunks: usize,
    stat_bytes: usize,
    max_part_rows: usize = 8,
    wide_calls: u64 = 0,
    calls: u64 = 0,
    fallbacks: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, w: *const weights.Weights, rows: usize, chunks: usize) !Owner {
        if (rows > dense.max_rows or rows == 0 or chunks == 0) return error.DenseShape;
        var layers: usize = 0;
        for (w.layers) |l| if (l.mixer == .mla) {
            if (l.mixer.mla.heads != 16 or l.mixer.mla.lw != 512) return error.DenseShape;
            layers += 1;
        };
        if (layers == 0) return error.DenseShape;
        var module = try loadImage(gpa, io, d);
        errdefer module.unload();
        const kernels = (try dense.Kernels.load(module, true)).?;
        const stat_bytes = try std.math.mul(usize, try std.math.mul(usize, rows, chunks), 16 * 4);
        var partials = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, stat_bytes, 514));
        errdefer partials.free();
        var tables = try cuda.DeviceBuffer.alloc(d, layers * dense.max_rows * @sizeOf(dense.Row));
        errdefer tables.free();
        var staging = try cuda.HostBuffer.alloc(d, tables.len);
        errdefer staging.free();
        @memset(staging.bytes, 0);
        var uploaded = try cuda.Event.init(d, false);
        errdefer uploaded.deinit();
        const plans = try gpa.alloc(?dense.Plan, layers);
        @memset(plans, null);
        std.log.info("GLM_DENSE_SEGMENTS_MODULE sha256={s} layers={d} chunks={d}", .{ image_sha, layers, chunks });
        return .{ .gpa = gpa, .kernels = kernels, .module = module, .stream = stream, .tables = tables, .partials = partials, .staging = staging, .uploaded = uploaded, .plans = plans, .rows = rows, .chunks = chunks, .stat_bytes = stat_bytes };
    }

    pub fn bytes(o: *const Owner) usize {
        return o.partials.len + o.tables.len + o.staging.bytes.len;
    }

    /// Old position/cache setup has already run. No table is uploaded for an unsupported forward.
    pub fn prepare(o: *Owner, r: anytype, rows: usize) !void {
        if (o.unusable) return error.DenseUnavailable;
        if (o.previous_upload) try o.uploaded.synchronize();
        o.previous_upload = false;
        @memset(o.plans, null);
        if (r.b.prefill or r.segs.len == 0 or r.segs.len > dense.max_segments or rows > o.rows) return;
        var any = false;
        var di: usize = 0;
        for (r.w.layers) |l| if (l.mixer == .mla) {
            defer di += 1;
            var segs: [dense.max_segments]dense.Segment = undefined;
            var supported = true;
            var row0: usize = 0;
            for (r.segs, 0..) |g, i| {
                if (g.row0 != row0 or di >= g.st.caches.len) return error.BadWindow;
                const c = g.st.caches[di];
                supported = supported and c.pt == 0 and c.ps == 0 and c.pks == 0;
                segs[i] = .{ .rows = g.rows, .host_pos = g.st.pos, .capacity = g.st.capacity, .cache = c.lc, .scales = c.ls, .position = g.st.pos_dev };
                row0 += g.rows;
            }
            if (row0 != rows) return error.BadWindow;
            if (!supported) continue;
            const p = dense.Plan.make(segs[0..r.segs.len], .{
                .max_part_rows = o.max_part_rows,
                .dense_limit = l.mixer.mla.dense_limit,
                .scratch_rows = o.rows,
                .scratch_chunks = o.chunks,
            }) catch continue; // entire original attend loop remains the fallback, before new enqueues
            o.plans[di] = p;
            const offset = di * dense.max_rows * @sizeOf(dense.Row);
            @memcpy(o.staging.bytes[offset..][0..p.bytes().len], p.bytes());
            any = true;
        };
        if (!any) return;
        o.unusable = true; // An uncertain copy/event failure refuses reuse until teardown drains the stream.
        try o.stream.d.check(o.stream.d.api.cuMemcpyHtoDAsync_v2(o.tables.ptr, o.staging.bytes.ptr, o.staging.bytes.len, o.stream.handle), "cuMemcpyHtoDAsync");
        try o.uploaded.record(o.stream);
        o.previous_upload = true;
        o.unusable = false;
    }

    /// The same cache/index writes, then two table-driven kernels. Never fall back after a new enqueue.
    pub fn attend(o: *Owner, f: anytype, r: anytype, l: dsa.Layer, di: usize, rows: usize) !bool {
        if (di >= o.plans.len) return error.DenseShape;
        const p = if (o.plans[di]) |*plan| plan else {
            o.fallbacks += 1;
            return false;
        };
        if (rows != p.rows) return error.DenseShape;
        const s: dense.Storage = .{
            .table = .{ .ptr = o.tables.ptr + di * dense.max_rows * @sizeOf(dense.Row), .len = dense.max_rows * @sizeOf(dense.Row) },
            .qa = .{ .ptr = r.b.dsa.qa, .len = r.b.rows * 16 * 512 * 2 },
            .po = .{ .ptr = o.partials.ptr, .len = o.stat_bytes * 512 },
            .pm = .{ .ptr = o.partials.ptr + o.stat_bytes * 512, .len = o.stat_bytes },
            .pl = .{ .ptr = o.partials.ptr + o.stat_bytes * 513, .len = o.stat_bytes },
            .out = .{ .ptr = r.b.dsa.ol, .len = r.b.rows * 16 * 512 * 2 },
        };
        try dense.validate(p, s);
        for (r.segs) |g| try dsa.writeSharedKeys(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, l, g.st.caches[di], r.b.dsa, g.row0, g.rows, g.st.pos_dev);
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(l.qk), -0.5));
        try o.kernels.launch(f.stream, p, s, scale);
        o.calls += 1;
        if (p.most > 8) o.wide_calls += 1;
        return true;
    }

    pub fn deinit(o: *Owner) void {
        o.stream.synchronize() catch {
            std.log.err("GLM_DENSE_SEGMENTS_RETAINED stream not quiescent", .{});
            return;
        };
        std.log.info("GLM_DENSE_SEGMENTS_CLOSE batched_layers={d} fallback_layers={d} wide_layers={d}", .{ o.calls, o.fallbacks, o.wide_calls });
        o.uploaded.deinit();
        o.staging.free();
        o.tables.free();
        o.partials.free();
        o.module.unload();
        o.gpa.free(o.plans);
        o.* = undefined;
    }
};

fn loadImage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver) !cuda.Module {
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_DENSE_SEGMENTS_IMAGE") orelse
        return error.DenseImageRequired;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.DenseImageHash;
    return cuda.Module.load(d, bytes);
}
