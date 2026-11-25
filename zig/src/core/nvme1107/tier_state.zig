//! One rank's bounded image adapter over the 1107 writer and reader workers.
const std = @import("std");
const spill = @import("spill/root.zig");
const mapping = @import("layout.zig");
const Allocator = std.mem.Allocator;
pub const Range = mapping.Range;
pub const Part = mapping.Part;
pub const Digest = spill.Digest;

/// Calls execute on the engine thread. Copies complete before their staging slot is returned.
pub const Image = struct {
    ptr: *anyopaque,
    parts: *const fn (*anyopaque, u32, *std.ArrayList(Part), Allocator) anyerror!void,
    take: *const fn (*anyopaque, u64, []const Range, []u8) anyerror!void,
    put: *const fn (*anyopaque, u64, []const Range, []const u8) anyerror!void,
    pin: *const fn (*anyopaque, u64) anyerror!void,
    unpin: *const fn (*anyopaque, u64) void,
    blank: *const fn (*anyopaque, u64, u32) anyerror!void,
    /// The first small-image bytes hold the family's commit words; validate before copying the remaining state.
    seal: *const fn (*anyopaque, u64, []const u8) anyerror!void,
    drop: *const fn (*anyopaque, u64) void,
};

pub const Note = struct {
    id: u64,
    at: u32,
    tokens: []const u32,
    metadata: []const u8,
    lineage: spill.Lineage,
    shared: bool = false,
};

const Pending = struct {
    note: Note,
    job: ?spill.JobId = null,
};

pub const Read = struct {
    job: spill.ReadId,
    id: u64,
    at: u32,
    expected: []u8,
    copied: u32 = 0,
    small: bool = false,
    failed: ?anyerror = null,
};

pub const Status = enum(u32) { pending = 0, ready = 1, failed = 2 };
pub const Counts = struct { refused: u64 = 0, captured: u64 = 0, cancelled: u64 = 0, restores: u64 = 0, failures: u64 = 0 };

pub const Side = struct {
    gpa: Allocator,
    io: std.Io,
    store: *spill.Store,
    image: Image,
    max_pending: usize,
    max_metadata: usize,
    notes: std.ArrayList(Pending) = .empty,
    read: ?Read = null,
    counts: Counts = .{},
    stopped: bool = false,
    lookahead: u1 = 0, // an extra key token is a sentinel tail, never a KV row
    serial_read_wait_ns: u64 = 0,
    serial_writes: bool = false, // allow queued sources without another pinned staging slot
    defer_releases: bool = false, // TP: source pages return only after every rank reports this point complete
    held: [4]u64 = @splat(0),

    pub fn init(gpa: Allocator, io: std.Io, store: *spill.Store, image: Image, max_pending: usize, max_metadata: usize) Side {
        return .{ .gpa = gpa, .io = io, .store = store, .image = image, .max_pending = max_pending, .max_metadata = max_metadata };
    }

    /// No copies, I/O or peer calls here: a mark hook only pins immutable source state and owns its metadata.
    pub fn enqueue(s: *Side, note: Note) !void {
        const held = std.mem.count(u64, &s.held, &.{0});
        if (s.stopped or s.notes.items.len + (s.held.len - held) >= s.max_pending or note.metadata.len > s.max_metadata) return error.NvmeBacklog;
        if (note.at == 0 or note.tokens.len != @as(usize, note.at) + s.lookahead or note.at % s.store.cfg.limits.block_tokens != 0) return error.NvmeMark;
        for (s.notes.items) |p| if (p.note.id == note.id) return error.NvmeDuplicate;
        if (s.completed(note.id)) return error.NvmeDuplicate;
        const toks = try s.gpa.dupe(u32, note.tokens);
        errdefer s.gpa.free(toks);
        const meta = try s.gpa.dupe(u8, note.metadata);
        errdefer s.gpa.free(meta);
        var lineage = try note.lineage.clone(s.gpa);
        errdefer lineage.deinit(s.gpa);
        try s.notes.ensureUnusedCapacity(s.gpa, 1);
        try s.image.pin(s.image.ptr, note.id);
        var own = note;
        own.tokens = toks;
        own.metadata = meta;
        own.lineage = lineage;
        s.notes.appendAssumeCapacity(.{ .note = own });
    }

    fn freeNote(s: *Side, p: *Pending) void {
        if (s.defer_releases) {
            const i = std.mem.indexOfScalar(u64, &s.held, 0) orelse unreachable;
            s.held[i] = p.note.id;
        } else s.image.unpin(s.image.ptr, p.note.id);
        s.gpa.free(p.note.tokens);
        s.gpa.free(p.note.metadata);
        p.note.lineage.deinit(s.gpa);
    }

    fn erase(s: *Side, index: usize) void {
        var p = s.notes.orderedRemove(index);
        s.freeNote(&p);
    }

    fn layout(s: *Side, at: u32, ps: *std.ArrayList(Part)) !mapping.Layout {
        try s.image.parts(s.image.ptr, at, ps, s.gpa);
        return mapping.Layout.init(ps.items, at, s.store.cfg.limits.block_tokens);
    }

    fn start(s: *Side, p: *Pending) !bool {
        var ps: std.ArrayList(Part) = .empty;
        defer ps.deinit(s.gpa);
        const l = try s.layout(p.note.at, &ps);
        if (l.blockBytes() != s.store.cfg.limits.block_bytes) return error.NvmeBlockShape;
        const k = s.store.keep(.{ .keyed = p.note.tokens, .lineage = p.note.lineage, .shared = p.note.shared, .sections = &.{ .{ .name = "glm.metadata", .len = p.note.metadata.len }, .{ .name = "glm.small", .len = l.smallBytes() } }, .tail_bytes = s.lookahead, .source = .{ .id = p.note.id, .generation = p.note.id } });
        if (k != .queued) {
            if (k != .stored and k != .short) s.counts.refused += 1;
            return false;
        }
        const q = k.queued;
        p.job = q.job;
        if (q.tail) |tail| {
            if (tail.len != 1 or s.lookahead != 1) return error.NvmeKeyTail;
            tail[0] = 0x6c;
        }
        var rs: std.ArrayList(Range) = .empty;
        defer rs.deinit(s.gpa);
        errdefer s.store.captured(q.job, false);
        try l.smallRanges(&rs, s.gpa);
        for (q.sections) |sec| {
            const dst = q.small[@intCast(sec.offset)..][0..@intCast(sec.len)];
            if (std.mem.eql(u8, sec.name.slice(), "glm.metadata")) @memcpy(dst, p.note.metadata) else if (std.mem.eql(u8, sec.name.slice(), "glm.small")) try s.image.take(s.image.ptr, p.note.id, rs.items, dst) else return error.NvmeSections;
        }
        s.store.captured(q.job, true);
        s.counts.captured += 1;
        return true;
    }

    /// Bounded copies only; worker threads own the bytes after staged/captured returns successfully.
    pub fn pump(s: *Side, budget_ns: u64) void {
        s.pumpMode(budget_ns, false);
    }

    /// Pressure and shutdown reuse returned staging slots within one command; ordinary rounds never wait for them.
    pub fn pumpWait(s: *Side, budget_ns: u64) void {
        s.pumpMode(budget_ns, true);
    }

    fn pumpMode(s: *Side, budget_ns: u64, wait_slots: bool) void {
        const until = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
        s.reap();
        var i: usize = 0;
        while (i < s.notes.items.len) {
            if (s.serial_writes and i != 0) break;
            if (s.notes.items[i].job == null) {
                const started = s.start(&s.notes.items[i]) catch {
                    s.counts.failures += 1;
                    if (s.notes.items[i].job) |j| _ = s.store.cancel(j);
                    s.erase(i);
                    continue;
                };
                if (!started) {
                    s.erase(i);
                    continue;
                }
            }
            i += 1;
            if (std.Io.Clock.awake.now(s.io).toNanoseconds() >= until) break;
        }
        while (std.Io.Clock.awake.now(s.io).toNanoseconds() < until) {
            var reqs: [4]spill.StageReq = undefined;
            const n = s.store.stage(&reqs);
            if (n == 0) {
                if (!wait_slots) break;
                s.reap();
                if (!s.pending()) break;
                std.Io.sleep(s.io, .fromMicroseconds(100), .awake) catch {};
                continue;
            }
            for (reqs[0..n]) |r| {
                var ps: std.ArrayList(Part) = .empty;
                defer ps.deinit(s.gpa);
                var rs: std.ArrayList(Range) = .empty;
                defer rs.deinit(s.gpa);
                const p = for (s.notes.items) |*p| {
                    if (p.note.id == r.source.id and r.source.generation == p.note.id) break p;
                } else null;
                const ok = if (p) |source| blk: {
                    const l = s.layout(source.note.at, &ps) catch break :blk false;
                    l.blockRanges(r.first / l.block, &rs, s.gpa) catch break :blk false;
                    if (r.tokens != l.block or mappingBytes(rs.items) != r.dst.len) break :blk false;
                    s.image.take(s.image.ptr, source.note.id, rs.items, r.dst) catch break :blk false;
                    break :blk true;
                } else false;
                s.store.staged(r.ticket, ok);
            }
        }
        s.reap();
    }

    fn reap(s: *Side) void {
        var done: [8]spill.Completion = undefined;
        while (true) {
            const n = s.store.poll(&done);
            if (n == 0) break;
            for (done[0..n]) |d| {
                const job = switch (d) {
                    inline else => |v| v.job,
                };
                for (s.notes.items, 0..) |p, i| if (p.job == job) {
                    if (d != .published) s.counts.failures += 1;
                    s.erase(i);
                    break;
                };
            }
        }
    }

    pub fn pending(s: *const Side) bool {
        return s.notes.items.len > 0;
    }

    pub fn completed(s: *const Side, id: u64) bool {
        return id != 0 and std.mem.indexOfScalar(u64, &s.held, id) != null;
    }

    /// The coordinator's all-rank credit. A duplicate release is refused, never a second unpin.
    pub fn release(s: *Side, id: u64) !void {
        const i = std.mem.indexOfScalar(u64, &s.held, id) orelse return error.NvmeNotReleased;
        if (id == 0) return error.NvmeNotReleased;
        s.held[i] = 0;
        s.image.unpin(s.image.ptr, id);
    }

    pub fn cancelNote(s: *Side, id: u64) void {
        for (s.notes.items, 0..) |p, i| if (p.note.id == id) {
            if (p.job) |j| _ = s.store.cancel(j);
            s.erase(i);
            return;
        };
    }

    pub fn beginRead(s: *Side, key: Digest, id: u64, at: u32, metadata: []const u8) !void {
        if (s.read != null or metadata.len > s.max_metadata) return error.NvmeReadBusy;
        const expected = try s.gpa.dupe(u8, metadata);
        errdefer s.gpa.free(expected);
        const until = std.Io.Clock.awake.now(s.io).toNanoseconds() + s.serial_read_wait_ns;
        const job = while (true) {
            const j = s.store.beginRead(key) catch |err| {
                if (s.serial_read_wait_ns == 0 or (err != error.NoSlot and err != error.TooMany) or
                    std.Io.Clock.awake.now(s.io).toNanoseconds() >= until) return err;
                s.pump(std.time.ns_per_ms);
                std.Io.sleep(s.io, .fromMicroseconds(100), .awake) catch {};
                continue;
            };
            break j;
        };
        errdefer {
            _ = s.store.endRead(job, .abort) catch {};
        }
        try s.image.blank(s.image.ptr, id, at);
        s.read = .{ .job = job, .id = id, .at = at, .expected = expected };
    }

    fn smallRead(s: *Side, r: *Read, ps: []const Part) !bool {
        const v = s.store.readSmallView(r.job) orelse return false;
        var meta: ?[]const u8 = null;
        var small: ?[]const u8 = null;
        for (v.sections) |sec| {
            if (sec.offset > v.bytes.len or sec.len > v.bytes.len - sec.offset) return error.NvmeSections;
            const bytes = v.bytes[@intCast(sec.offset)..][0..@intCast(sec.len)];
            if (std.mem.eql(u8, sec.name.slice(), "glm.metadata") and meta == null) meta = bytes else if (std.mem.eql(u8, sec.name.slice(), "glm.small") and small == null) small = bytes else return error.NvmeSections;
        }
        if (!std.mem.eql(u8, meta orelse return error.NvmeSections, r.expected)) return error.NvmeArithmetic;
        const payload = small orelse return error.NvmeSections;
        const l = try mapping.Layout.init(ps, r.at, s.store.cfg.limits.block_tokens);
        if (payload.len != l.smallBytes()) return error.NvmeSections;
        try s.image.seal(s.image.ptr, r.id, payload);
        var rs: std.ArrayList(Range) = .empty;
        defer rs.deinit(s.gpa);
        try l.smallRanges(&rs, s.gpa);
        try s.image.put(s.image.ptr, r.id, rs.items, payload);
        return true;
    }

    /// Read-ahead workers validate files; only this engine-thread step copies their checked bytes into a blank state.
    pub fn pollRead(s: *Side) Status {
        const r = if (s.read) |*r| r else return .failed;
        if (r.failed != null) return .failed;
        if (s.store.readFailure(r.job).err != null) {
            r.failed = error.NvmeRead;
            return .failed;
        }
        var ps: std.ArrayList(Part) = .empty;
        defer ps.deinit(s.gpa);
        const l = s.layout(r.at, &ps) catch |err| {
            r.failed = err;
            return .failed;
        };
        if (!r.small) {
            r.small = s.smallRead(r, ps.items) catch |err| {
                r.failed = err;
                return .failed;
            };
            if (!r.small) return .pending;
        }
        var ready: [4]spill.ReadyBlock = undefined;
        const got = s.store.readReady(r.job, null, &ready);
        for (ready[0..got.n]) |b| {
            defer s.store.readRelease(b.slot);
            if (r.failed != null) continue;
            if (s.lookahead == 1 and b.index == r.at / l.block) {
                if (b.index != r.copied or b.first != r.at or b.tokens != 1 or b.data.len != 1 or b.data[0] != 0x6c) r.failed = error.NvmeKeyTail else r.copied += 1;
                continue;
            }
            var rs: std.ArrayList(Range) = .empty;
            defer rs.deinit(s.gpa);
            if (b.index != r.copied or b.first != r.copied * l.block or b.tokens != l.block) {
                r.failed = error.NvmeReadOrder;
                continue;
            }
            l.blockRanges(b.index, &rs, s.gpa) catch |err| {
                r.failed = err;
                continue;
            };
            if (b.data.len != mappingBytes(rs.items)) {
                r.failed = error.NvmeBlockShape;
                continue;
            }
            s.image.put(s.image.ptr, r.id, rs.items, b.data) catch |err| {
                r.failed = err;
                continue;
            };
            r.copied += 1;
        }
        if (got.status == .failed and r.failed == null) r.failed = error.NvmeRead;
        if (r.failed != null) return .failed;
        return if (r.copied == r.at / l.block + s.lookahead) .ready else .pending;
    }

    /// Called only after the coordinator has seen ready on every rank.
    pub fn commitRead(s: *Side) !void {
        const r = s.read orelse return error.NvmeNoRead;
        if (r.failed != null or !r.small or r.copied != r.at / s.store.cfg.limits.block_tokens + s.lookahead) return error.NvmeNotReady;
        var done = (try s.store.endRead(r.job, .commit)) orelse return error.NvmeNotReady;
        done.lineage.deinit(s.store.gpa);
        s.gpa.free(r.expected);
        s.read = null;
        s.counts.restores += 1;
    }

    pub fn abortRead(s: *Side) void {
        const r = s.read orelse return;
        _ = s.store.endRead(r.job, .abort) catch {};
        s.image.drop(s.image.ptr, r.id);
        s.gpa.free(r.expected);
        s.read = null;
    }

    /// Before abort frees the read's failure details, unlist its key and every local alias of corrupt storage.
    /// A different numerical route is a cache miss and never calls this.
    pub fn forgetRead(s: *Side, key: Digest) void {
        if (s.read) |r| {
            const f = s.store.readFailure(r.job);
            s.store.forget(key, f.bad, f.lost);
        } else s.store.forget(key, &.{}, &.{});
    }

    pub fn stop(s: *Side, budget_ns: u64) void {
        if (s.stopped) return;
        s.stopped = true;
        s.abortRead();
        const until = std.Io.Clock.awake.now(s.io).toNanoseconds() + budget_ns;
        while (s.pending() and std.Io.Clock.awake.now(s.io).toNanoseconds() < until) {
            s.pump(@min(budget_ns, std.time.ns_per_ms));
            std.Io.sleep(s.io, .fromMicroseconds(100), .awake) catch {};
        }
        while (s.notes.items.len > 0) {
            if (s.notes.items[0].job) |j| _ = s.store.cancel(j);
            s.counts.cancelled += 1;
            s.erase(0);
        }
        for (s.held) |id| if (id != 0) {
            s.release(id) catch unreachable;
        };
    }

    pub fn deinit(s: *Side) void {
        s.stop(0);
        s.notes.deinit(s.gpa);
    }
};

fn mappingBytes(rs: []const Range) u64 {
    var n: u64 = 0;
    for (rs) |r| n += r.len;
    return n;
}
