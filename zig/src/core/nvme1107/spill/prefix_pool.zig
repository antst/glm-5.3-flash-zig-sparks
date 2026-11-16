//! Immutable token-chain prefixes share storage across append turns and checkpoints.
const std = @import("std");
const Digest = @import("format.zig").Digest;
const Allocator = std.mem.Allocator;
const Slab = struct { data: []Digest, used: usize, refs: usize };

pub const Lease = struct {
    pool: *Pool,
    slab: *Slab,
    items: []Digest,

    pub fn release(l: Lease) void {
        l.pool.release(l.slab);
    }
};

pub const Pool = struct {
    gpa: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    slabs: std.ArrayList(*Slab) = .empty,
    live_bytes: usize = 0,
    peak_bytes: usize = 0,

    /// Success consumes `owned`; failure leaves it with the caller. Only unused trailing cells may change.
    pub fn take(p: *Pool, owned: []Digest) !Lease {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        for (p.slabs.items) |slab| {
            if (owned.len > slab.data.len) continue;
            const common = @min(owned.len, slab.used);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(owned[0..common]), std.mem.sliceAsBytes(slab.data[0..common]))) continue;
            if (owned.len > slab.used) {
                @memcpy(slab.data[slab.used..owned.len], owned[slab.used..]);
                slab.used = owned.len;
            }
            slab.refs += 1;
            const len = owned.len;
            p.gpa.free(owned);
            return .{ .pool = p, .slab = slab, .items = slab.data[0..len] };
        }
        try p.slabs.ensureUnusedCapacity(p.gpa, 1);
        const slab = try p.gpa.create(Slab);
        errdefer p.gpa.destroy(slab);
        const cap = try std.math.add(usize, owned.len, 256);
        const data = try p.gpa.alloc(Digest, cap);
        @memcpy(data[0..owned.len], owned);
        slab.* = .{ .data = data, .used = owned.len, .refs = 1 };
        p.slabs.appendAssumeCapacity(slab);
        p.live_bytes += data.len * @sizeOf(Digest);
        p.peak_bytes = @max(p.peak_bytes, p.live_bytes);
        const len = owned.len;
        p.gpa.free(owned);
        return .{ .pool = p, .slab = slab, .items = data[0..len] };
    }

    fn release(p: *Pool, slab: *Slab) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        std.debug.assert(slab.refs > 0);
        slab.refs -= 1;
        if (slab.refs != 0) return;
        const i = std.mem.indexOfScalar(*Slab, p.slabs.items, slab) orelse unreachable;
        _ = p.slabs.swapRemove(i);
        p.live_bytes -= slab.data.len * @sizeOf(Digest);
        p.gpa.free(slab.data);
        p.gpa.destroy(slab);
    }

    pub fn deinit(p: *Pool) void {
        std.debug.assert(p.slabs.items.len == 0 and p.live_bytes == 0);
        p.slabs.deinit(p.gpa);
    }
};
