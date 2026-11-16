//! Host-thread-owned immutable histories: checkpoints share token prefixes instead of copying whole prompts.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Slab = struct { words: []u32, used: usize, refs: usize };

pub const Lease = struct {
    pool: *Pool,
    slab: *Slab,
    words: []u32,

    pub fn release(l: Lease, a: Allocator) void {
        l.pool.release(a, l.slab);
    }
};

pub const Pool = struct {
    slabs: std.ArrayList(*Slab) = .empty,
    bytes: usize = 0,
    peak: usize = 0,
    limit: usize = 64 << 20,

    pub fn take(p: *Pool, a: Allocator, words: []const u32) !Lease {
        for (p.slabs.items) |s| {
            if (words.len > s.words.len) continue;
            const n = @min(words.len, s.used);
            if (!std.mem.eql(u32, words[0..n], s.words[0..n])) continue;
            if (words.len > s.used) {
                @memcpy(s.words[s.used..words.len], words[s.used..]);
                s.used = words.len;
            }
            s.refs += 1;
            return .{ .pool = p, .slab = s, .words = s.words[0..words.len] };
        }
        const extra = @min(4096, @max(16, words.len / 64));
        const cap = std.math.add(usize, words.len, extra) catch return error.OutOfMemory;
        const bytes = std.math.mul(usize, cap, 4) catch return error.OutOfMemory;
        if (bytes > p.limit -| p.bytes) return error.OutOfMemory;
        try p.slabs.ensureUnusedCapacity(a, 1);
        const s = try a.create(Slab);
        errdefer a.destroy(s);
        const data = try a.alloc(u32, cap);
        @memcpy(data[0..words.len], words);
        s.* = .{ .words = data, .used = words.len, .refs = 1 };
        p.slabs.appendAssumeCapacity(s);
        p.bytes += bytes;
        p.peak = @max(p.peak, p.bytes);
        return .{ .pool = p, .slab = s, .words = data[0..words.len] };
    }

    fn release(p: *Pool, a: Allocator, s: *Slab) void {
        std.debug.assert(s.refs > 0);
        s.refs -= 1;
        if (s.refs != 0) return;
        const i = std.mem.indexOfScalar(*Slab, p.slabs.items, s) orelse unreachable;
        _ = p.slabs.swapRemove(i);
        p.bytes -= s.words.len * 4;
        a.free(s.words);
        a.destroy(s);
    }

    pub fn deinit(p: *Pool, a: Allocator) void {
        std.debug.assert(p.slabs.items.len == 0 and p.bytes == 0);
        p.slabs.deinit(a);
    }
};
