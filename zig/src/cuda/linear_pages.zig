//! Physical backing shared by snapshots and by stable, linear CUDA address windows.

const std = @import("std");

/// One engine thread owns the pool. The caller orders all device uses before remapping or freeing a window.
pub const Backend = struct {
    ptr: *anyopaque,
    granularity: usize,
    reserve: *const fn (*anyopaque, usize) anyerror!u64,
    create: *const fn (*anyopaque, usize) anyerror!u64,
    release: *const fn (*anyopaque, u64, usize) anyerror!void,
    map: *const fn (*anyopaque, u64, usize, u64) anyerror!void,
    unmap: *const fn (*anyopaque, u64, usize) anyerror!void,
    /// Return error.CopyUncertain if asynchronous use may still be outstanding.
    copy: *const fn (*anyopaque, u64, u64, usize) anyerror!void,
};

const Id = u32;
const Page = struct { backing: ?u64 = null, refs: usize = 0, resident_alias: bool = false };

pub const Pool = struct {
    allocator: std.mem.Allocator,
    backend: Backend,
    page_bytes: usize,
    pages: []Page,
    free: []Id,
    nfree: usize,
    io_base: u64,
    windows: usize = 0,
    snapshots: usize = 0,
    healthy: bool = true,

    pub fn init(a: std.mem.Allocator, backend: Backend, page_bytes: usize, limit: usize) !Pool {
        if (backend.granularity == 0 or page_bytes == 0 or page_bytes % backend.granularity != 0 or
            limit == 0 or limit > std.math.maxInt(Id)) return error.PageGeometry;
        const pages = try a.alloc(Page, limit);
        errdefer a.free(pages);
        const free = try a.alloc(Id, limit);
        errdefer a.free(free);
        @memset(pages, .{});
        for (free, 0..) |*id, i| id.* = @intCast(limit - i - 1);
        const io_base = try backend.reserve(backend.ptr, try std.math.mul(usize, page_bytes, limit));
        return .{ .allocator = a, .backend = backend, .page_bytes = page_bytes, .pages = pages, .free = free, .nfree = limit, .io_base = io_base };
    }

    /// The virtual reservation belongs to the backend and must outlive the pool.
    pub fn deinit(p: *Pool) !void {
        if (p.windows != 0 or p.snapshots != 0 or p.nfree != p.pages.len) return error.LiveBacking;
        try p.trim();
        p.allocator.free(p.pages);
        p.allocator.free(p.free);
        p.* = undefined;
    }

    /// Idle physical allocations can be recycled or explicitly returned to the driver under pressure.
    pub fn trim(p: *Pool) !void {
        for (p.free[0..p.nfree]) |id| {
            const page = &p.pages[id];
            if (page.backing) |h| {
                if (page.resident_alias) {
                    try p.unmap(p.residentAddress(id));
                    page.resident_alias = false;
                }
                try p.backend.release(p.backend.ptr, h, p.page_bytes);
                page.backing = null;
            }
        }
    }

    fn acquire(p: *Pool) !Id {
        if (p.nfree == 0) return error.PhysicalPageLimit;
        const id = p.free[p.nfree - 1];
        const page = &p.pages[id];
        if (page.backing == null) page.backing = try p.backend.create(p.backend.ptr, p.page_bytes);
        std.debug.assert(page.refs == 0);
        page.refs = 1;
        p.nfree -= 1;
        errdefer p.drop(id);
        if (!page.resident_alias) {
            try p.map(p.residentAddress(id), id);
            page.resident_alias = true;
        }
        return id;
    }

    fn residentAddress(p: *const Pool, id: Id) u64 {
        return p.io_base + @as(usize, id) * p.page_bytes;
    }

    fn retain(p: *Pool, id: Id) void {
        std.debug.assert(p.pages[id].refs > 0);
        p.pages[id].refs += 1;
    }

    fn drop(p: *Pool, id: Id) void {
        const page = &p.pages[id];
        std.debug.assert(page.refs > 0);
        page.refs -= 1;
        if (page.refs == 0) {
            p.free[p.nfree] = id;
            p.nfree += 1;
        }
    }

    fn map(p: *Pool, at: u64, id: Id) !void {
        p.backend.map(p.backend.ptr, at, p.page_bytes, p.pages[id].backing.?) catch |err| {
            if (err == error.MappingUncertain) {
                p.healthy = false;
                // Keep the backing alive until the owner can tear down the CUDA context.
                p.retain(id);
            }
            return err;
        };
    }

    fn unmap(p: *Pool, at: u64) !void {
        p.backend.unmap(p.backend.ptr, at, p.page_bytes) catch |err| {
            p.healthy = false;
            return err;
        };
    }

    fn pagesFor(p: *const Pool, bytes: usize) usize {
        return bytes / p.page_bytes + @intFromBool(bytes % p.page_bytes != 0);
    }

    fn requireHealthy(p: *const Pool) !void {
        if (!p.healthy) return error.UncertainPageMappings;
    }
};

pub const Block = struct {
    ptr: u64,
    len: usize,
};

/// Hold until the D2H copy completes. The I/O address does not depend on a lane's virtual mappings.
pub const PinnedBlock = struct {
    pool: *Pool,
    id: Id,

    pub fn block(b: *const PinnedBlock) Block {
        return .{ .ptr = b.pool.residentAddress(b.id), .len = b.pool.page_bytes };
    }

    pub fn deinit(b: *PinnedBlock) void {
        b.pool.drop(b.id);
        b.* = undefined;
    }
};

/// References to completed bytes. The last page can have an unwritten tail.
pub const Snapshot = struct {
    pool: *Pool,
    ids: []Id,
    len: usize,

    /// Each block is resident until this snapshot is released. The final block exposes only its valid bytes.
    pub fn block(s: *const Snapshot, i: usize) ?Block {
        if (i >= s.ids.len or !s.pool.healthy) return null;
        return .{ .ptr = s.pool.residentAddress(s.ids[i]), .len = @min(s.pool.page_bytes, s.len - i * s.pool.page_bytes) };
    }

    pub fn deinit(s: *Snapshot) void {
        for (s.ids) |id| s.pool.drop(id);
        s.pool.allocator.free(s.ids);
        s.pool.snapshots -= 1;
        s.* = undefined;
    }
};

pub const Window = struct {
    pool: *Pool,
    base: u64,
    ids: []Id,
    maximum_bytes: usize,
    mapped: usize = 0,

    /// The base address stays fixed across growth, restore, and copy-on-write.
    pub fn init(p: *Pool, maximum_bytes: usize) !Window {
        try p.requireHealthy();
        if (maximum_bytes == 0) return error.PageGeometry;
        const capacity = p.pagesFor(maximum_bytes);
        const span = try std.math.mul(usize, capacity, p.page_bytes);
        const ids = try p.allocator.alloc(Id, capacity);
        errdefer p.allocator.free(ids);
        const base = try p.backend.reserve(p.backend.ptr, span);
        p.windows += 1;
        return .{ .pool = p, .base = base, .ids = ids, .maximum_bytes = maximum_bytes };
    }

    pub fn deinit(w: *Window) !void {
        try w.clear();
        w.pool.allocator.free(w.ids);
        w.pool.windows -= 1;
        w.* = undefined;
    }

    pub fn clear(w: *Window) !void {
        while (w.mapped > 0) {
            const i = w.mapped - 1;
            try w.pool.unmap(w.address(i));
            w.pool.drop(w.ids[i]);
            w.mapped = i;
        }
    }

    /// Map capacity ahead of the kernel. Already mapped ranges make no driver calls.
    pub fn ensure(w: *Window, end: usize) !void {
        const p = w.pool;
        try p.requireHealthy();
        const total = p.pagesFor(end);
        if (end > w.maximum_bytes) return error.VirtualWindowLimit;
        while (w.mapped < total) {
            const id = try p.acquire();
            errdefer p.drop(id);
            try p.map(w.address(w.mapped), id);
            w.ids[w.mapped] = id;
            w.mapped += 1;
        }
    }

    /// The owner must prove that the writer has completed all bytes through ready.
    pub fn pinCompleteBlock(w: *const Window, i: usize, ready: usize) !PinnedBlock {
        const p = w.pool;
        try p.requireHealthy();
        if (ready > w.maximum_bytes or i >= @min(w.mapped, ready / p.page_bytes))
            return error.IncompleteBlock;
        const id = w.ids[i];
        p.retain(id);
        return .{ .pool = p, .id = id };
    }

    /// Capture a prefix once its writes are ordered. Snapshotting retains page IDs without mapping or copying data.
    pub fn snapshot(w: *const Window, end: usize) !Snapshot {
        const p = w.pool;
        try p.requireHealthy();
        const count = p.pagesFor(end);
        if (end > w.maximum_bytes or count > w.mapped) return error.UnmappedPrefix;
        const ids = try p.allocator.dupe(Id, w.ids[0..count]);
        for (ids) |id| p.retain(id);
        p.snapshots += 1;
        return .{ .pool = p, .ids = ids, .len = end };
    }

    /// A rank can restore into an idle window while other windows retain their own aliases.
    pub fn restore(w: *Window, s: *const Snapshot) !void {
        const p = w.pool;
        try p.requireHealthy();
        if (p != s.pool) return error.DifferentPagePool;
        if (w.mapped != 0) return error.WindowNotEmpty;
        if (s.len > w.maximum_bytes) return error.VirtualWindowLimit;
        errdefer w.clear() catch {};
        for (s.ids) |id| {
            try p.map(w.address(w.mapped), id);
            p.retain(id);
            w.ids[w.mapped] = id;
            w.mapped += 1;
        }
    }

    /// Alias a still-live prefix into an empty window without retaining a separate snapshot.
    pub fn borrow(w: *Window, src: *const Window, len: usize) !void {
        const p = w.pool;
        try p.requireHealthy();
        if (src.pool != p) return error.DifferentPagePool;
        if (w == src or w.mapped != 0) return error.WindowNotEmpty;
        const count = p.pagesFor(len);
        if (len > w.maximum_bytes or len > src.maximum_bytes or count > src.mapped) return error.UnmappedPrefix;
        errdefer w.clear() catch {};
        for (src.ids[0..count]) |id| {
            try p.map(w.address(w.mapped), id);
            p.retain(id);
            w.ids[w.mapped] = id;
            w.mapped += 1;
        }
    }

    /// All pages touched by the next write must be private. A copied page retains its whole previous contents.
    pub fn prepareWrite(w: *Window, start: usize, end: usize) !void {
        if (start > end) return error.BadWriteRange;
        if (end > w.maximum_bytes) return error.VirtualWindowLimit;
        if (start == end) return;
        try w.ensure(end);
        const p = w.pool;
        const first = start / p.page_bytes;
        for (first..p.pagesFor(end)) |i| {
            if (p.pages[w.ids[i]].refs > 1) try w.makePrivate(i);
        }
    }

    fn address(w: *const Window, i: usize) u64 {
        return w.base + i * w.pool.page_bytes;
    }

    fn makePrivate(w: *Window, i: usize) !void {
        const p = w.pool;
        const old = w.ids[i];
        const addr = w.address(i);
        const id = try p.acquire();
        errdefer p.drop(id);

        p.backend.copy(p.backend.ptr, p.residentAddress(id), addr, p.page_bytes) catch |err| {
            if (err == error.CopyUncertain) {
                p.healthy = false;
                p.retain(id);
            }
            return err;
        };

        try p.unmap(addr);
        p.map(addr, id) catch |err| {
            // Recover the old view for retry if neither operation left an uncertain mapping.
            if (p.healthy) p.map(addr, old) catch {
                p.healthy = false;
            };
            return err;
        };
        w.ids[i] = id;
        p.drop(old);
    }
};

test {
    _ = @import("linear_pages_test.zig");
}
