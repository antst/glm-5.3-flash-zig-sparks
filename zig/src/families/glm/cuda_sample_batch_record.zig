//! Recording callbacks for the actual sampled host dispatcher; no driver/library is opened.
const std = @import("std");
const cuda = @import("cuda");
const smp = @import("cuda_sampler.zig");
pub const Span = @import("cuda_shard.zig").Span;
pub const Kind = enum { pack, gather, unpack, place, rule, meta, draw, sync, check, download, broadcast };
pub const Fail = enum { none, pack, gather, unpack, place, rule, meta, draw, sync, check, download, fill };
pub const stream_at: cuda.abi.Stream = @ptrFromInt(0x1234);
const draw_at: cuda.abi.Function = @ptrFromInt(0x5678);
const t = std.testing;
const Block = struct { storage: []align(256) u8, host: bool };

pub const Fake = struct {
    var blocks: [32]?Block = @splat(null);
    pub var pending = false;
    pub var allocations: usize = 0;
    pub var frees: usize = 0;
    pub var fail_alloc: usize = 0;
    pub var fail: Fail = .none;
    pub var calls: [1024]Kind = undefined;
    pub var n: usize = 0;
    pub var raw: usize = 0;
    pub var borrowed: usize = 0;
    pub var width: usize = 38784;
    pub var rows: usize = 1;
    pub var rank: usize = 0;
    pub var spans: []const Span = &.{};
    pub var shards: [4][]u16 = undefined;
    pub var expected: []u16 = &.{};
    pub var pad: u64 = 0;
    pub var got: u64 = 0;
    pub var full: u64 = 0;
    pub var stage: u64 = 0;
    pub var rule_at: u64 = 0;
    pub var meta_at: u64 = 0;
    pub var tok: u64 = 0;
    pub var prob: u64 = 0;
    pub var rule: smp.Rule = undefined;
    pub var first: u64 = 1;
    pub var cap: usize = 8;

    pub fn reset() void {
        std.debug.assert(!pending);
        n = 0;
        raw = 0;
        borrowed = 0;
        fail = .none;
        fail_alloc = 0;
    }
    fn note(k: Kind) void {
        calls[n] = k;
        n += 1;
    }
    pub fn bytes(ptr: u64, len: usize) []u8 {
        const p: [*]u8 = @ptrFromInt(ptr);
        return p[0..len];
    }
    fn allocate(out: *u64, len: usize, host: bool) cuda.abi.Result {
        if (!host) allocations += 1;
        if (!host and fail_alloc == allocations) return cuda.abi.error_not_ready;
        for (&blocks) |*entry| if (entry.* == null) {
            const b = t.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(256), len + 512) catch return 2;
            @memset(b, 0xa5);
            entry.* = .{ .storage = b, .host = host };
            out.* = @intFromPtr(b.ptr) + 256;
            return 0;
        };
        return 2;
    }
    fn alloc(out: *u64, len: usize) callconv(.c) cuda.abi.Result {
        return allocate(out, len, false);
    }
    fn free(ptr: u64) callconv(.c) cuda.abi.Result {
        for (&blocks) |*entry| if (entry.*) |b| {
            if (@intFromPtr(b.storage.ptr) + 256 != ptr) continue;
            std.debug.assert(b.host or !pending);
            for (b.storage[0..256]) |v| std.debug.assert(v == 0xa5);
            for (b.storage[b.storage.len - 256 ..]) |v| std.debug.assert(v == 0xa5);
            t.allocator.free(b.storage);
            if (!b.host) frees += 1;
            entry.* = null;
            return 0;
        };
        return 1;
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        var p: u64 = 0;
        const rc = allocate(&p, len, true);
        out.* = @ptrFromInt(p);
        return rc;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) cuda.abi.Result {
        return free(@intFromPtr(p));
    }
    fn fill(dst: u64, v: u8, len: usize, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        std.debug.assert(s == stream_at and v == 0);
        pending = true;
        if (fail == .fill) return cuda.abi.error_not_ready;
        @memset(bytes(dst, len), v);
        return 0;
    }
    fn copy2D(m: *const cuda.abi.Memcpy2D, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const packing = m.dst == pad;
        const destination = spans[spans.len - 1].hi;
        const unpacking = !packing and m.src >= got and m.src < got + spans.len * rows * width * 2;
        const kind: Kind = if (packing) .pack else if (unpacking) .unpack else .place;
        std.debug.assert(s == stream_at and m.src_type == 2 and m.dst_type == 2);
        std.debug.assert(m.src_x == 0 and m.src_y == 0 and m.dst_x == 0 and m.dst_y == 0);
        std.debug.assert(m.src_host == null and m.dst_host == null and m.src_array == null and m.dst_array == null);
        std.debug.assert(m.height == rows);
        if (packing) {
            std.debug.assert(m.src == @intFromPtr(shards[rank].ptr) and m.src_pitch == spans[rank].len() * 2);
            std.debug.assert(m.width == m.src_pitch and m.dst_pitch == width * 2);
        } else if (unpacking) {
            const q = (m.src - got) / (rows * width * 2);
            std.debug.assert(q < spans.len and m.src == got + q * rows * width * 2);
            std.debug.assert(m.dst == full + spans[q].lo * 2 and m.dst_pitch == spans[spans.len - 1].hi * 2);
            std.debug.assert(m.src_pitch == width * 2 and m.width == spans[q].len() * 2);
        } else {
            const lo = (m.dst - full) / 2;
            const q = for (spans, 0..) |span, rank_| {
                if (span.lo == lo) break rank_;
            } else unreachable;
            std.debug.assert(m.src == stage + rows * lo * 2 or m.src == @intFromPtr(shards[q].ptr));
            std.debug.assert(m.src_pitch == spans[q].len() * 2 and m.width == m.src_pitch);
            std.debug.assert(m.dst_pitch == destination * 2);
        }
        note(kind);
        pending = true;
        if ((packing and fail == .pack) or (!packing and fail == .unpack)) return cuda.abi.error_not_ready;
        for (0..m.height) |r| @memcpy(bytes(m.dst + r * m.dst_pitch, m.width), bytes(m.src + r * m.src_pitch, m.width));
        return 0;
    }
    fn allGather(mine: u64, all: u64, count: usize, ty: cuda.nccl.DataType, _: cuda.nccl.Comm, s: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(s == stream_at and ty == .u8 and mine == pad and all == got);
        std.debug.assert(count == rows * width * 2);
        raw += 1;
        note(.gather);
        pending = true;
        if (fail == .gather) return 1;
        for (0..rows) |r| {
            const own = spans[rank].len();
            std.debug.assert(std.mem.eql(u8, bytes(pad + r * width * 2, own * 2), std.mem.sliceAsBytes(shards[rank][r * own ..][0..own])));
            for (bytes(pad + (r * width + own) * 2, (width - own) * 2)) |v| std.debug.assert(v == 0);
        }
        @memset(bytes(all, spans.len * count), 0);
        for (spans, 0..) |span, q| for (0..rows) |r| {
            @memcpy(bytes(all + q * count + r * width * 2, span.len() * 2), std.mem.sliceAsBytes(shards[q][r * span.len() ..][0..span.len()]));
        };
        return 0;
    }
    fn fits(_: *anyopaque, count: usize) bool {
        return count <= 512 * 1024;
    }
    fn grouped(_: *anyopaque, s: cuda.abi.Stream, mine: u64, all: u64, count: usize) !void {
        borrowed += 1;
        const rc = allGather(mine, all, count, .u8, null, s);
        raw -= 1;
        if (rc != 0) return error.NcclFailed;
    }
    fn check(_: *anyopaque) !void {
        note(.check);
        std.debug.assert(!pending);
        if (fail == .check) return error.RecordedWireFailure;
    }
    fn noop(_: *anyopaque) void {}
    fn broadcast(_: u64, dst: u64, count: usize, ty: cuda.nccl.DataType, q_: c_int, _: cuda.nccl.Comm, s: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        const q: usize = @intCast(q_);
        std.debug.assert(s == stream_at and ty == .bf16 and count == rows * spans[q].len());
        std.debug.assert(dst == stage + rows * spans[q].lo * 2);
        @memcpy(bytes(dst, count * 2), std.mem.sliceAsBytes(shards[q][0..count]));
        note(.broadcast);
        pending = true;
        return 0;
    }
    fn groupBound() callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn d2d(dst: u64, src: u64, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        @memcpy(bytes(dst, len), bytes(src, len));
        pending = true;
        return 0;
    }
    fn upload(dst: u64, src: ?*const anyopaque, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const k: Kind = if (dst == rule_at) .rule else .meta;
        note(k);
        pending = true;
        if ((k == .rule and fail == .rule) or (k == .meta and fail == .meta)) return cuda.abi.error_not_ready;
        @memcpy(bytes(dst, len), bytes(@intFromPtr(src), len));
        if (k == .rule) std.debug.assert(std.mem.eql(u8, bytes(dst, len), std.mem.asBytes(&rule))) else {
            const meta: *const [4]i32 = @ptrFromInt(dst);
            std.debug.assert(meta.*[0] == @as(i32, @intCast(first - 1)) and meta.*[1] == 0 and meta.*[2] == 0 and meta.*[3] == 0);
        }
        return 0;
    }
    fn arg(comptime T: type, ps: [*]?*anyopaque, i: usize) T {
        return @as(*align(1) const T, @ptrCast(ps[i].?)).*;
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sh: c_uint, s: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(f == draw_at and gx == rows and gy == 1 and gz == 1);
        std.debug.assert(bx == 1024 and by == 1 and bz == 1 and sh == 0 and s == stream_at);
        const ps = params.?;
        std.debug.assert(arg(u64, ps, 0) == full and arg(u32, ps, 1) == spans[spans.len - 1].hi);
        std.debug.assert(arg(u64, ps, 2) == rule_at and arg(u64, ps, 3) == meta_at and arg(i32, ps, 4) == 0);
        std.debug.assert(arg(u64, ps, 5) == tok and arg(u64, ps, 6) == prob);
        std.debug.assert(std.mem.eql(u8, bytes(full, expected.len * 2), std.mem.sliceAsBytes(expected)));
        note(.draw);
        pending = true;
        if (fail == .draw) return cuda.abi.error_not_ready;
        const out: [*]u32 = @ptrFromInt(tok);
        for (0..rows) |r| out[r] = @intCast(r + 42);
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        note(.sync);
        if (fail == .sync) {
            fail = .none;
            return cuda.abi.error_not_ready;
        }
        pending = false;
        return 0;
    }
    fn ctxSync() callconv(.c) cuda.abi.Result {
        pending = false;
        return 0;
    }
    fn download(dst: ?*anyopaque, src: u64, len: usize) callconv(.c) cuda.abi.Result {
        std.debug.assert(!pending and src == tok and len == rows * 4);
        note(.download);
        if (fail == .download) return cuda.abi.error_not_ready;
        @memcpy(bytes(@intFromPtr(dst), len), bytes(src, len));
        return 0;
    }
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attr(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(0x9876);
        return 0;
    }
    fn eventOk(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        return 0;
    }

    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemcpy2DAsync_v2 = copy2D;
        d.api.cuMemcpyDtoDAsync_v2 = d2d;
        d.api.cuMemcpyHtoDAsync_v2 = upload;
        d.api.cuMemcpyDtoH_v2 = download;
        d.api.cuStreamSynchronize = sync;
        d.api.cuCtxSynchronize = ctxSync;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = event;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventSynchronize = eventOk;
        d.api.cuEventQuery = eventOk;
        d.api.cuEventDestroy_v2 = eventOk;
        d.api.cuLaunchKernel = launch;
        d.api.cuCtxGetDevice = device;
        d.api.cuDeviceGetAttribute = attr;
        d.api.cuModuleUnload = unload;
        return d;
    }
    pub fn library() cuda.nccl.Library {
        var lib: cuda.nccl.Library = undefined;
        lib.api.ncclAllGather = allGather;
        lib.api.ncclBroadcast = broadcast;
        lib.api.ncclGroupStart = groupBound;
        lib.api.ncclGroupEnd = groupBound;
        return lib;
    }
    pub fn group(lib: cuda.nccl.Library) cuda.tp_group.Group {
        return .{ .lib = lib, .comm = null, .world = spans.len, .rank = rank, .multi = .{ .ctx = &n, .fits = fits, .gather = grouped, .check = check, .abort = noop, .deinit = noop } };
    }
    pub fn sampler(d: *const cuda.Driver, vocab: usize) !smp.Sampler {
        var full_ = try cuda.DeviceBuffer.alloc(d, cap * vocab * 2);
        errdefer full_.free();
        var stage_ = try cuda.DeviceBuffer.alloc(d, smp.stagingBytes(vocab, cap));
        errdefer stage_.free();
        var r = try cuda.DeviceBuffer.alloc(d, 32);
        errdefer r.free();
        var m = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer m.free();
        var tok_ = try cuda.DeviceBuffer.alloc(d, cap * 4);
        errdefer tok_.free();
        const prob_ = try cuda.DeviceBuffer.alloc(d, cap * 4);
        full = full_.ptr;
        stage = stage_.ptr;
        got = 0;
        pad = 0;
        rule_at = r.ptr;
        meta_at = m.ptr;
        tok = tok_.ptr;
        prob = prob_.ptr;
        return .{ .d = d, .module = .{ .d = d, .handle = null }, .tf_draw = .{ .d = d, .handle = draw_at }, .tf_logprob_rows = undefined, .tf_grammar_mask = undefined, .vocab = vocab, .rows = cap, .full = full_, .stage = stage_, .rule = r, .meta = m, .tok = tok_, .prob = prob_ };
    }
};
