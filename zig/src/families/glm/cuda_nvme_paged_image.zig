//! Canonical paged NVMe image: a compact state followed by one physical-page-sized run per 512 tokens.
const std = @import("std");
const cuda = @import("cuda");
const tier = @import("core").nvme1107.tier_state;
const dflash = @import("dflash.zig");
const image = @import("cuda_disk_image.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = @import("cuda_engine.zig").Engine;
const page_rows = @import("cuda_state.zig").page_rows;

const Region = enum { words, compact, draft, pages };
const Piece = struct { region: Region, offset: u64, len: usize };

pub const Shape = struct {
    compact: u64,
    draft: u64,
    page: u32,
    count: usize,

    fn size(s: Shape) u64 {
        return image.Words.bytes + s.compact + s.draft + s.page * @as(u64, s.count);
    }

    fn piece(s: Shape, off: u64, len: usize) !Piece {
        if (off > s.size() or len > s.size() - off or len == 0) return error.NvmeImageRange;
        var at = off;
        for ([_]Region{ .words, .compact, .draft, .pages }, [_]u64{
            image.Words.bytes, s.compact, s.draft, s.page * @as(u64, s.count),
        }) |kind, n| {
            if (at < n) return .{ .region = kind, .offset = at, .len = @intCast(@min(len, n - at)) };
            at -= n;
        }
        return error.NvmeImageRange;
    }

    pub fn parts(s: Shape, out: *std.ArrayList(tier.Part), a: std.mem.Allocator) !void {
        try out.appendSlice(a, &.{
            .{ .bytes = image.Words.bytes },
            .{ .bytes = s.compact },
            .{ .bytes = s.draft },
            .{ .bytes = s.page * @as(u64, s.count), .row = s.page, .per = page_rows },
        });
    }
};

pub fn shape(e: *Engine, at: u32) !Shape {
    const p = e.pages orelse return error.NvmePageLayout;
    if (!e.share_pages or p.arena == null or !e.settings.kv8 or snapshots.mode.head(e) or
        at == 0 or at % page_rows != 0) return error.NvmePageLayout;
    const count = at / page_rows;
    const compact = try snapshots.layout.Plan.sizeWith(&e.cur.st, &e.c, e.capacity, at, null, count, snapshots.mode.enabled(e));
    return .{ .compact = compact, .draft = if (e.drafter != null) dflash.ring_bytes else 0, .page = std.math.cast(u32, p.page) orelse return error.NvmePageLayout, .count = count };
}

pub fn copy(e: *Engine, id: u64, off: u64, host: []u8, out: bool) !void {
    const kept = e.snaps.get(id) orelse return error.NoKeptState;
    const s = try shape(e, @intCast(kept.pos));
    if (kept.tier_ids.len != s.count or kept.shared != s.count or kept.len < s.compact or
        off > s.size() or host.len > s.size() - off) return error.NvmeImageRange;
    const p = e.pages.?;
    const dev_base = p.arena.?.ptr;
    var words = (try @import("cuda_disk_ranks.zig").wordsOf(e, id)).encode();
    var done: usize = 0;
    while (done < host.len) {
        const item = try s.piece(off + done, host.len - done);
        const h = host[done..][0..item.len];
        switch (item.region) {
            .words => {
                const w = std.mem.asBytes(&words)[@intCast(item.offset)..][0..item.len];
                if (out) @memcpy(h, w) else @memcpy(w, h);
            },
            .compact => try copyLease(e, &kept.lease, item.offset, h, out),
            .draft => if (kept.ring_from != null)
                try copyLease(e, &kept.lease, s.compact + item.offset, h, out)
            else if (out) @memset(h, 0),
            .pages => {
                var copied: usize = 0;
                while (copied < h.len) {
                    const at = item.offset + copied;
                    const within = at % s.page;
                    const bytes: usize = @intCast(@min(h.len - copied, s.page - within));
                    const page = kept.tier_ids[@intCast(at / s.page)];
                    if (page >= p.pool.total()) return error.NvmePageLayout;
                    try move(e.d, dev_base + @as(u64, page) * s.page + within, h[copied..][0..bytes], out);
                    copied += bytes;
                }
            },
        }
        done += item.len;
    }
}

fn copyLease(e: *Engine, lease: *const snapshots.pages.Lease, off: u64, host: []u8, out: bool) !void {
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const arena = pool.arena orelse return error.NoSnapshotRoom;
    const block = snapshots.pages.page;
    if (off + host.len > lease.ids.items.len * block) return error.SnapshotSize;
    var copied: usize = 0;
    while (copied < host.len) {
        const at = off + copied;
        const within = at % block;
        const bytes: usize = @intCast(@min(host.len - copied, block - within));
        const page = lease.ids.items[@intCast(at / block)];
        try move(e.d, arena.ptr() + @as(u64, page) * block + within, host[copied..][0..bytes], out);
        copied += bytes;
    }
}

fn move(d: *const cuda.Driver, dev: u64, host: []u8, out: bool) !void {
    if (out) try cuda.memory.download(d, dev, host) else try d.check(d.api.cuMemcpyHtoD_v2(dev, host.ptr, host.len), "cuMemcpyHtoD");
}

test "GLM NVMe: the paged image stores one physical run per block and leaves recurrent bytes in the small section" {
    const s: Shape = .{ .compact = 29, .draft = 7, .page = 2048, .count = 3 };
    var parts: std.ArrayList(tier.Part) = .empty;
    defer parts.deinit(std.testing.allocator);
    try s.parts(&parts, std.testing.allocator);
    const x = try @import("core").nvme1107.layout.Layout.init(parts.items, 3 * page_rows, page_rows);
    try std.testing.expectEqual(@as(u64, 2048), x.blockBytes());
    try std.testing.expectEqual(image.Words.bytes + 36, x.smallBytes());
    try std.testing.expectEqual(Piece{ .region = .compact, .offset = 28, .len = 1 }, try s.piece(image.Words.bytes + 28, 2));
    try std.testing.expectEqual(Piece{ .region = .pages, .offset = 2047, .len = 3 }, try s.piece(x.smallBytes() + 2047, 3));
    try std.testing.expectError(error.NvmeImageRange, s.piece(s.size() - 1, 2));
}
