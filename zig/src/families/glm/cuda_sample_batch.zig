//! Ordinary sampled rows: padded byte gather, then exact copies into the original row-major sampler input.
const std = @import("std");
const cuda = @import("cuda");
const Span = @import("cuda_shard.zig").Span;
const Exchange = @import("cuda_forward.zig").Exchange;
const max_world = @import("cuda_ranks.zig").max_world;
pub const max_rows = @import("cuda_state.zig").window_rows;
pub const Local = struct { rank: usize, logits: u64 };

pub const Plan = struct {
    vocab: usize,
    rows: usize,
    rank: usize,
    world: usize,
    width: usize,
    pitch: usize,
    full_pitch: usize,
    send_bytes: usize,
    recv_bytes: usize,
    spans: [max_world]Span = @splat(.{ .lo = 0, .hi = 0 }),

    pub fn init(vocab: usize, rows: usize, rank: usize, spans: []const Span) !Plan {
        if (vocab == 0 or rows == 0 or rows > max_rows) return error.BadSampleBatchRows;
        if (spans.len < 2 or spans.len > max_world or rank >= spans.len) return error.BadSampleBatchRanks;
        var width: usize = 0;
        var at: usize = 0;
        for (spans) |span| {
            if (span.lo != at or span.hi <= span.lo or span.hi > vocab) return error.BadSampleBatchSpans;
            width = @max(width, span.len());
            at = span.hi;
        }
        if (at != vocab) return error.BadSampleBatchSpans;
        const pitch = try std.math.mul(usize, width, 2);
        const send = try std.math.mul(usize, rows, pitch);
        var p: Plan = .{ .vocab = vocab, .rows = rows, .rank = rank, .world = spans.len, .width = width, .pitch = pitch, .full_pitch = try std.math.mul(usize, vocab, 2), .send_bytes = send, .recv_bytes = try std.math.mul(usize, spans.len, send) };
        @memcpy(p.spans[0..spans.len], spans);
        return p;
    }

    pub fn scratchBytes(p: Plan) !usize {
        return std.math.add(usize, p.send_bytes, p.recv_bytes);
    }
};

pub const Batch = struct {
    plan: Plan,
    pad: cuda.DeviceBuffer,
    got: cuda.DeviceBuffer,

    /// Startup only: allocation failure is fatal, never a rank-local change of collective algorithm.
    pub fn init(d: *const cuda.Driver, stream: cuda.Stream, vocab: usize, rows: usize, rank: usize, spans: []const Span) !Batch {
        const plan = try Plan.init(vocab, rows, rank, spans);
        _ = try plan.scratchBytes();
        var pad = try cuda.DeviceBuffer.alloc(d, plan.send_bytes);
        errdefer pad.free();
        var got = try cuda.DeviceBuffer.alloc(d, plan.recv_bytes);
        errdefer got.free();
        errdefer cuda.completion.drain(d, stream, "a sampled batching startup fill did not drain");
        try pad.fill8(0, stream.handle);
        try stream.synchronize();
        return .{ .plan = plan, .pad = pad, .got = got };
    }

    /// The owning engine drains its stream before releasing the sampler, on every healthy shutdown.
    pub fn deinit(b: *Batch) void {
        b.got.free();
        b.pad.free();
        b.* = undefined;
    }

    /// Pack/gather/unpack on the same stream; no synchronization, readback, conversion or arithmetic here.
    pub fn gather(b: *const Batch, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: Local, rows: usize, full: cuda.DeviceBuffer) !void {
        const p = &b.plan;
        if (rows == 0 or rows > p.rows) return error.BadSampleBatchRows;
        if (mine.rank != p.rank or spans.len != p.world) return error.BadSampleBatchRanks;
        for (spans, p.spans[0..p.world]) |now, old| if (!std.meta.eql(now, old)) return error.BadSampleBatchSpans;
        const x = switch (exchange) {
            .local => return error.BadSampleBatchRanks,
            .nccl => |x| x,
        };
        if (x.group) |g| if (g.world != p.world or g.rank != p.rank) return error.BadSampleBatchRanks;
        const bytes = try std.math.mul(usize, rows, p.pitch);
        const all_bytes = try std.math.mul(usize, p.world, bytes);
        const full_bytes = try std.math.mul(usize, rows, p.full_pitch);
        if (b.pad.len < bytes or b.got.len < all_bytes or full.len < full_bytes) return error.SampleBatchScratchTooSmall;
        const own_pitch = try std.math.mul(usize, spans[mine.rank].len(), 2);
        const input_bytes = try std.math.mul(usize, rows, own_pitch);
        try end(mine.logits, input_bytes);
        try end(b.pad.ptr, bytes);
        try end(b.got.ptr, all_bytes);
        try end(full.ptr, full_bytes);
        try copy(stream, b.pad.ptr, p.pitch, mine.logits, own_pitch, own_pitch, rows);
        if (x.group) |g| try g.allGather(stream, b.pad.ptr, b.got.ptr, bytes) else try x.lib.check(x.lib.api.ncclAllGather(b.pad.ptr, b.got.ptr, bytes, .u8, x.comm, stream.handle), "ncclAllGather");
        for (spans, 0..) |span, q| {
            // ACTUAL rows form each rank's plane, independent of the allocated rows capacity.
            try copy(stream, full.ptr + span.lo * 2, p.full_pitch, b.got.ptr + q * bytes, p.pitch, span.len() * 2, rows);
        }
    }
};

fn end(ptr: u64, bytes: usize) !void {
    if (ptr == 0) return error.BadSampleBatchPointer;
    _ = try std.math.add(u64, ptr, bytes);
}

fn copy(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), "cuMemcpy2DAsync");
}
