//! Port of wojo's GLM 0110 `tools/spill_blocks_check.py` and `tools/spill_blocks_mutants.sh` (PR #124) as Zig tests,
pub const std = @import("std");
pub const c = std.c;
pub const format = @import("format.zig");
pub const fsio = @import("fsio.zig");
pub const slots = @import("slots.zig");
pub const store = @import("store.zig");
pub const mutants = @import("mutants.zig");
pub const Allocator = std.mem.Allocator;
pub const Store = store.Store;
pub const Digest = format.Digest;
pub const Lineage = format.Lineage;
pub const testing = std.testing;
pub const io = std.testing.io;

pub const block_tokens = 16;
pub const bpt = 256; // bytes a token on this rank
pub const block_bytes = block_tokens * bpt;
pub const max_tokens = 1024;
pub const small_max = 8192;
pub const secs = [_]store.SectionSpec{ .{ .name = "ring", .len = 300 }, .{ .name = "raw", .len = 200 } };

pub var quiet = false; // a mutant run: failures are expected, not printed
pub var serial: u32 = 0;

pub const CheckFailed = error.CheckFailed;

pub const ensure = @import("store_test_support_1.zig").ensure;

pub const Opts = struct {
    tag: []const u8,
    write_slots: usize = 8,
    read_slots: usize = 8,
    small_slots: usize = 4,
    restore_slots: usize = 2,
    cap: u64 = 64 << 20,
    floor: u64 = 0,
    turns: u32 = 4,
    quota: f64 = 0,
    max_reads: u32 = 2,
    keep_builds: u32 = 1,
    backlog: u64 = 1 << 30,
    free: ?u64 = null,
    no_direct: bool = false,
    shared_small_restore: bool = false,
};

pub const Ev = struct { ev: fsio.Event, name: [96]u8, len: usize };

/// The test's side of the hooks: event log, an action at a step, a crash point, a pwrite script, a disk-free value.
pub const Hk = struct {
    mutex: std.Io.Mutex = .init,
    events: std.ArrayList(Ev) = .empty,
    gpa: Allocator,
    action_step: ?fsio.Step = null,
    action_suffix: []const u8 = "",
    action: ?*const fn (h: *H) void = null,
    crash_step: ?fsio.Step = null,
    crash_suffix: []const u8 = "",
    crash_skip: u32 = 0,
    nospace_writes: u32 = 0, // the next pwrites failing with ENOSPC
    short_intr: bool = false, // every pwrite short (at most 1000 bytes), and every other one EINTR first
    flip: bool = false,
    free: ?u64 = null,
    h: *H = undefined,

    pub fn crash(ctx: ?*anyopaque, step: fsio.Step, name: []const u8) bool {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (std.c.getenv("SPILL_DEBUG") != null) std.debug.print("hook {t} {s}\n", .{ step, name });
        if (k.action) |f| if (k.action_step == step and std.mem.endsWith(u8, name, k.action_suffix)) {
            k.action = null;
            f(k.h);
        };
        if (k.crash_step) |cs| if (cs == step and std.mem.endsWith(u8, name, k.crash_suffix)) {
            if (k.crash_skip > 0) {
                k.crash_skip -= 1;
                return false;
            }
            k.crash_step = null;
            return true;
        };
        return false;
    }

    pub fn event(ctx: ?*anyopaque, ev: fsio.Event, name: []const u8) void {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        var e: Ev = .{ .ev = ev, .name = undefined, .len = @min(name.len, 96) };
        @memcpy(e.name[0..e.len], name[0..e.len]);
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.append(k.gpa, e) catch {};
    }

    pub fn pwrite(ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) fsio.Error!usize {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (k.nospace_writes > 0) {
            k.nospace_writes -= 1;
            return error.NoSpace;
        }
        var want = buf.len;
        if (k.short_intr) {
            k.flip = !k.flip;
            if (k.flip) return error.Interrupted;
            if (off % fsio.alignment == 0 and want > 1000) want = 1000;
        }
        const n = c.pwrite(fd, buf.ptr, want, @intCast(off));
        if (n < 0) return switch (c.errno(n)) {
            .INTR => error.Interrupted,
            .NOSPC, .DQUOT => error.NoSpace,
            .INVAL, .OPNOTSUPP => error.Unsupported,
            else => error.Io,
        };
        return @intCast(n);
    }

    pub fn freeBytes(ctx: ?*anyopaque) u64 {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        return k.free orelse (1 << 40);
    }

    pub fn clear(k: *Hk) void {
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.clearRetainingCapacity();
    }
};

/// A store over a fresh scratch directory and the test's fake device.
pub const H = struct {
    gpa: Allocator,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    bufs: [4][]align(slots.alignment) u8 = @splat(&.{}),
    lists: [4][1]slots.Buffer = undefined,
    cfg: store.Config = undefined,
    s: *Store = undefined,
    rows: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    hk: Hk,
    opened: bool = false,

    pub fn init(gpa: Allocator, o: Opts) !*H {
        const h = try gpa.create(H);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .hk = .{ .gpa = gpa, .free = o.free } };
        h.hk.h = h;
        serial += 1;
        var tb: [64]u8 = undefined;
        const tag = try std.fmt.bufPrint(&tb, "{s}-{d}", .{ o.tag, serial });
        const r = try fsio.testRoot(&h.path, tag);
        h.root = r;
        const dfd = try fsio.freshDir(gpa, r);
        _ = c.close(dfd);
        const counts = [4]usize{ o.write_slots, o.read_slots, if (o.shared_small_restore) 1 else o.small_slots, if (o.shared_small_restore) 0 else o.restore_slots };
        const per = [4]usize{ block_bytes, block_bytes, small_max, small_max };
        for (0..4) |i| {
            h.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), @max(counts[i] * per[i], 0));
            h.lists[i] = .{h.bufs[i]};
        }
        if (o.shared_small_restore) h.lists[3] = h.lists[2];
        h.cfg = .{
            .root = h.root,
            .rank = 0,
            .compat = @splat(0x5a),
            .limits = .{ .max_tokens = max_tokens, .block_tokens = block_tokens, .block_bytes = block_bytes, .max_small_bytes = small_max },
            .cap_bytes = o.cap,
            .floor_bytes = o.floor,
            .quota = o.quota,
            .turns = o.turns,
            .min_tokens = 16,
            .backlog_bytes = o.backlog,
            .staging = .{ .write = &h.lists[0], .read = &h.lists[1], .small = &h.lists[2], .restore = &h.lists[3], .shared_small_restore = o.shared_small_restore },
            .readers = 3,
            .max_reads = if (o.shared_small_restore) 1 else o.max_reads,
            .keep_builds = o.keep_builds,
            .hooks = .{ .ctx = &h.hk, .crash = Hk.crash, .event = Hk.event, .pwrite = Hk.pwrite, .free = Hk.freeBytes, .no_direct = o.no_direct },
        };
        try h.open();
        return h;
    }

    pub fn open(h: *H) !void {
        h.s = try Store.open(h.gpa, io, h.cfg);
        h.opened = true;
    }

    pub fn closeStore(h: *H) void {
        if (h.opened) h.s.close();
        h.opened = false;
    }

    pub fn reopen(h: *H) !void {
        h.closeStore();
        try h.open();
    }

    pub fn deinit(h: *H) void {
        h.closeStore();
        fsio.dropDir(h.gpa, h.root);
        for (h.bufs) |b| h.gpa.free(b);
        var it = h.rows.valueIterator();
        while (it.next()) |r| h.gpa.free(r.*);
        h.rows.deinit(h.gpa);
        h.hk.events.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    /// Source `id`'s rows on the fake device (deterministic bytes, made on first use).
    pub fn src(h: *H, id: u64) []u8 {
        const gop = h.rows.getOrPut(h.gpa, id) catch @panic("oom");
        if (!gop.found_existing) {
            const r = h.gpa.alloc(u8, max_tokens * bpt) catch @panic("oom");
            var prng = std.Random.DefaultPrng.init(id *% 0x9e3779b97f4a7c15 +% 1);
            prng.random().bytes(r);
            gop.value_ptr.* = r;
        }
        return gop.value_ptr.*;
    }

    pub fn dir(h: *H, which: enum { blocks, points }) c.fd_t {
        return if (which == .blocks) h.s.blocks_fd else h.s.points_fd;
    }
};

pub const smallByte = @import("store_test_support_1.zig").smallByte;

pub const KO = struct {
    lineage: Lineage,
    source: u64,
    gen: u64 = 1,
    shared: bool = false,
    group: u64 = 0,
    media: []const format.Media = &.{},
    raw: ?[]const u32 = null, // with media: the raw ids (the same length as the keyed ones)
    capture: bool = true,
};

/// The engine at a keep: `keep`, then the small state and tail captured from the device, reported.
pub const keepIds = @import("store_test_support_1.zig").keepIds;

/// The engine between rounds: every requested block packed from its source and reported.
pub const pump = @import("store_test_support_1.zig").pump;

/// The engine until nothing waits on it: blocks staged as write slots come back, then a drain.
pub const settle = @import("store_test_support_1.zig").settle;

/// Keep and write `ids` at once; the point's key once it is published.
pub const write = @import("store_test_support_1.zig").write;

/// A restore of `key` through the engine's side: rows (at * bpt bytes, gpa-owned) or null when it fails.
pub const restore = @import("store_test_support_1.zig").restore;

/// A restore that fails, its failure forgotten the way every rank forgets it: true when it failed.
pub const restoreFails = @import("store_test_support_1.zig").restoreFails;

pub const idsOf = @import("store_test_support_1.zig").idsOf;

pub const cat = @import("store_test_support_1.zig").cat;

pub const lin = @import("store_test_support_1.zig").lin;

pub const manifestShared = @import("store_test_support_1.zig").manifestShared;

pub const fileExists = @import("store_test_support_1.zig").fileExists;

pub const blockFile = @import("store_test_support_1.zig").blockFile;

pub const pointFile = @import("store_test_support_1.zig").pointFile;

pub const countFiles = @import("store_test_support_1.zig").countFiles;

pub const waitUntil = @import("store_test_support_1.zig").waitUntil;

// -- the checks ------------------------------------------------------------------------------------------------------

/// GLM review 1: the block a kept state's end cuts holds the rows of the keep, not what the stream writes after
pub const checkTail = @import("store_test_support_1.zig").checkTail;

/// The reader reads; the engine copies, only as far as it is told (LOADPART's boundary), the rest after.
pub const checkUpto = @import("store_test_support_1.zig").checkUpto;

/// Last use survives a restart (the manifest's time): the least recently used conversation is still evicted first.
pub const checkLastUse = @import("store_test_support_1.zig").checkLastUse;

/// Keeps the store refuses say why and hold nothing: too short, invalid, out of staging, past the backlog.
pub const checkRefusals = @import("store_test_support_1.zig").checkRefusals;

/// Shared flag 3: the same ids kept again as a shared prefix make the stored point shared, its manifest rewritten
pub const checkSharedUpgrade = @import("store_test_support_1.zig").checkSharedUpgrade;

/// ... also when they were still waiting to be written.
pub const checkSharedWaiting = @import("store_test_support_1.zig").checkSharedWaiting;

pub const markWhileWriting = @import("store_test_support_1.zig").markWhileWriting;

/// ... and when kept again as shared while its own files were being written: its manifest is rewritten shared.
pub const checkSharedDuringWrite = @import("store_test_support_1.zig").checkSharedDuringWrite;

pub var drop_key: Digest = undefined;
pub const dropDuring = @import("store_test_support_1.zig").dropDuring;

/// A point dropped while its manifest is rewritten does not get the manifest back (no resurrection).
pub const checkDropDuringRewrite = @import("store_test_support_1.zig").checkDropDuringRewrite;

pub const slowRewrite = @import("store_test_support_1.zig").slowRewrite;

/// A drain (a clean stop) waits for a queued metadata rewrite.
pub const checkDrainWaitsRewrite = @import("store_test_support_1.zig").checkDrainWaitsRewrite;

/// A (2 blocks), B (other ids, the same rows: its blocks links to A's) and C (A's ids and a block more, A's lineage:
pub const linked = @import("store_test_support_1.zig").linked;

/// GLM review 4: a read failing on a missing block file drops that name and the points naming it, not the points
pub const checkForgetLost = @import("store_test_support_1.zig").checkForgetLost;

/// ... one failing on damaged bytes drops every name of them (A, B and C).
pub const checkForgetDamaged = @import("store_test_support_1.zig").checkForgetDamaged;

/// A point's small files (.ids, .st, .man) each have their pages dropped after their fsync.
pub const checkPages = @import("store_test_support_1.zig").checkPages;

/// Publication order: every block renamed and its folder fsynced before the point's files; the ids and small state
pub const checkPublishOrder = @import("store_test_support_1.zig").checkPublishOrder;

pub const Pending = struct {
    h: *H,
    held: std.ArrayList(slots.Slot) = .empty,

    pub fn recycle(ctx: ?*anyopaque) void {
        const p: *Pending = @ptrCast(@alignCast(ctx.?));
        for (p.held.items) |s| p.h.s.readRelease(s);
        p.held.clearRetainingCapacity();
    }
};

/// PR 124's hang: a follower whose peer failed early (LOADED with nothing copied before) and more blocks left than
pub const checkHang = @import("store_test_support_2.zig").checkHang;

/// Read-ahead accounting: preallocated buffers are kept and only the shortfall is charged.
pub const checkCarve = @import("store_test_support_2.zig").checkCarve;

pub const Slow = struct {
    h: *H,
    calls: std.ArrayList(store.ReadId) = .empty,

    pub fn copy(ctx: ?*anyopaque, id: store.ReadId, b: store.ReadyBlock) bool {
        const s: *Slow = @ptrCast(@alignCast(ctx.?));
        s.calls.append(s.h.gpa, id) catch {};
        std.Io.sleep(io, .fromMilliseconds(30), .awake) catch {};
        s.h.s.readRelease(b.slot);
        return true;
    }
};

/// GLM review (wojo#2): the restores in flight share one copy budget a pause, each pause starting with the next.
pub const checkBudget = @import("store_test_support_2.zig").checkBudget;

pub const CrashMode = enum {
    publish, // a new point's publication
    link, // ... whose first blocks equal a stored point's bytes (written as hard links)
    rewrite, // a published point's manifest rewritten with the shared flag
};

pub const crash_points = [_]struct { step: fsio.Step, suffix: []const u8, skip: u32 = 0, mode: CrashMode = .publish }{
    .{ .step = .written, .suffix = ".blk" },
    .{ .step = .synced, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk", .skip = 1 },
    .{ .step = .dir_synced, .suffix = "blocks" },
    .{ .step = .written, .suffix = ".ids" },
    .{ .step = .renamed, .suffix = ".ids" },
    .{ .step = .written, .suffix = ".st" },
    .{ .step = .renamed, .suffix = ".st" },
    .{ .step = .dir_synced, .suffix = "points" },
    .{ .step = .written, .suffix = ".man" },
    .{ .step = .synced, .suffix = ".man" },
    .{ .step = .renamed, .suffix = ".man" },
    .{ .step = .published, .suffix = "points" },
    // PORT-SPILL-V2 stage 5: while a block is linked, and while shared metadata is rewritten
    .{ .step = .linked, .suffix = ".blk", .mode = .link },
    .{ .step = .linked, .suffix = ".blk", .skip = 1, .mode = .link },
    .{ .step = .written, .suffix = ".man", .mode = .rewrite },
    .{ .step = .synced, .suffix = ".man", .mode = .rewrite },
    .{ .step = .renamed, .suffix = ".man", .mode = .rewrite },
    .{ .step = .rewritten, .suffix = ".man", .mode = .rewrite },
};

pub const crashed = @import("store_test_support_2.zig").crashed;

/// The durability fault campaign: a kill after every write, fsync and rename of a point's publication, after a
pub const checkCrashes = @import("store_test_support_2.zig").checkCrashes;
pub const checkCrashesMode = @import("store_test_support_2.zig").checkCrashesMode;
pub const checkRecoveryMode = @import("store_test_support_2.zig").checkRecoveryMode;

/// Recovery: a truncated block, a missing small state and a damaged manifest each drop only their point; a stale
pub const checkRecovery = @import("store_test_support_2.zig").checkRecovery;

/// CRC-32 forging: the 4 bytes that make `data[0 .. len - 4] ++ them` hash to `target` (reflected, ISO-HDLC).
pub const forgeCrc = @import("store_test_support_2.zig").forgeCrc;

/// Dedup links only equal bytes: a block with the same CRC-32 and length but other bytes is written, not linked.
pub const checkDedupSha = @import("store_test_support_2.zig").checkDedupSha;

/// A guard before source rows change: the point's blocks still to stage are staged now (the engine queues those
pub const checkGuard = @import("store_test_support_2.zig").checkGuard;

/// ... past the room it has, or for an older generation of the source, the unpublished point is cancelled, its
pub const checkGuardCancel = @import("store_test_support_2.zig").checkGuardCancel;

/// The staging cap holds for a point larger than all of staging (the GPU's 250k-token point is ~220 MB against a
pub const checkStagingCap = @import("store_test_support_2.zig").checkStagingCap;

/// 0110's eviction order: old turns, abandoned forks, whole conversations least recently used first (longest state
pub const checkEvictionOrder = @import("store_test_support_2.zig").checkEvictionOrder;

/// Old turns past retention on the active path go as soon as a newer turn is written; their blocks stay (the
pub const checkTurns = @import("store_test_support_2.zig").checkTurns;

/// A resume forks the lineage at its cut: the same names before it (nothing restaged), its own after.
pub const checkFork = @import("store_test_support_2.zig").checkFork;

/// 0111 groups: branches of one conversation that share only a shared prefix count as one conversation.
pub const checkGroups = @import("store_test_support_2.zig").checkGroups;

/// A conversation over its quota trims its own old turns and forks first, not another conversation's states.
pub const checkQuota = @import("store_test_support_2.zig").checkQuota;

/// The free-disk floor: no write leaves less free; room is made in eviction order, else the write is refused.
pub const checkFloor = @import("store_test_support_2.zig").checkFloor;

/// The cap: a write that would pass it evicts the least recently used conversation, not a newer one.
pub const checkCap = @import("store_test_support_2.zig").checkCap;

/// A second owner of a live store is refused before it scans or deletes anything.
pub const checkOwner = @import("store_test_support_2.zig").checkOwner;

/// Other builds' namespaces: pruned unless live (locked).
pub const checkPrune = @import("store_test_support_2.zig").checkPrune;

/// Lookups: the longest complete point the prompt extends within its bounds; media must match.
pub const checkLookup = @import("store_test_support_3.zig").checkLookup;

/// Replace a point file's bytes (same name, truncated first).
pub const rewritePointFile = @import("store_test_support_3.zig").rewritePointFile;

/// A picture point's raw ids altered on disk with every CRC made good (the .ids file's in the manifest, the
pub const checkPictureRecovery = @import("store_test_support_3.zig").checkPictureRecovery;

/// Disk faults on the write path: ENOSPC fails the point and leaves no temporary file or staging behind; short
pub const checkWriteFaults = @import("store_test_support_3.zig").checkWriteFaults;

/// A restore cancelled while its reads are in flight gives every slot back, and the point back to eviction.
pub const checkCancelRead = @import("store_test_support_3.zig").checkCancelRead;

/// Two restores share the read slots: each gets blocks in turn.
pub const checkFairReads = @import("store_test_support_3.zig").checkFairReads;

/// Replicated and owner-sharded points never read as each other: a store of another layout drops them as foreign.
pub const checkLayout = @import("store_test_support_3.zig").checkLayout;

/// 0118's owner-sharded store: a rank keeps a point of its own part with full blocks of every plane and a point of
pub const checkOwnerParts = @import("store_test_support_3.zig").checkOwnerParts;

/// GLM 0110's pin (review of the native port): a resumed conversation that is the least recently used one, kept
pub const checkKeepReuse = @import("store_test_support_3.zig").checkKeepReuse;

/// A cancelled write's unstaged blocks go to another waiting write that names them, and so does their
pub const checkHandOver = @import("store_test_support_3.zig").checkHandOver;

/// A quota that is not a fraction in [0, 1] is refused at open.
pub const checkBadConfig = @import("store_test_support_3.zig").checkBadConfig;

pub const chmodAt = @import("store_test_support_3.zig").chmodAt;

/// Recovery deletes only damage: a point it cannot read for another reason (here: its ids, or one of its blocks,
pub const checkRecoverTransient = @import("store_test_support_3.zig").checkRecoverTransient;

// -- the tests -------------------------------------------------------------------------------------------------------
pub const Check = struct { name: []const u8, f: *const fn (Allocator) anyerror!void };
pub const checks = [_]Check{
    .{ .name = "tail", .f = checkTail },
    .{ .name = "upto", .f = checkUpto },
    .{ .name = "shared upgrade", .f = checkSharedUpgrade },
    .{ .name = "shared waiting", .f = checkSharedWaiting },
    .{ .name = "shared during write", .f = checkSharedDuringWrite },
    .{ .name = "drop during rewrite", .f = checkDropDuringRewrite },
    .{ .name = "drain waits for rewrites", .f = checkDrainWaitsRewrite },
    .{ .name = "forget lost", .f = checkForgetLost },
    .{ .name = "forget damaged", .f = checkForgetDamaged },
    .{ .name = "pages", .f = checkPages },
    .{ .name = "publish order", .f = checkPublishOrder },
    .{ .name = "hang", .f = checkHang },
    .{ .name = "carve", .f = checkCarve },
    .{ .name = "budget", .f = checkBudget },
    .{ .name = "crashes", .f = checkCrashes },
    .{ .name = "recovery", .f = checkRecovery },
    .{ .name = "dedup sha", .f = checkDedupSha },
    .{ .name = "guard", .f = checkGuard },
    .{ .name = "guard cancel", .f = checkGuardCancel },
    .{ .name = "eviction order", .f = checkEvictionOrder },
    .{ .name = "turns", .f = checkTurns },
    .{ .name = "fork", .f = checkFork },
    .{ .name = "groups", .f = checkGroups },
    .{ .name = "quota", .f = checkQuota },
    .{ .name = "floor", .f = checkFloor },
    .{ .name = "cap", .f = checkCap },
    .{ .name = "owner", .f = checkOwner },
    .{ .name = "prune", .f = checkPrune },
    .{ .name = "lookup", .f = checkLookup },
    .{ .name = "picture recovery", .f = checkPictureRecovery },
    .{ .name = "write faults", .f = checkWriteFaults },
    .{ .name = "cancel read", .f = checkCancelRead },
    .{ .name = "fair reads", .f = checkFairReads },
    .{ .name = "last use", .f = checkLastUse },
    .{ .name = "refusals", .f = checkRefusals },
    .{ .name = "layout", .f = checkLayout },
    .{ .name = "keep reuse", .f = checkKeepReuse },
    .{ .name = "recover transient", .f = checkRecoverTransient },
    .{ .name = "hand over", .f = checkHandOver },
    .{ .name = "bad config", .f = checkBadConfig },
    .{ .name = "owner parts", .f = checkOwnerParts },
    .{ .name = "staging cap", .f = checkStagingCap },
};

/// Which checks each mutant must fail (the GLM mutants that apply to a single-rank store, and the port's own).
pub const kills = [_]struct { m: mutants.Mutant, checks: []const []const u8 }{
    .{ .m = .tail_staged_later, .checks = &.{"tail"} },
    .{ .m = .loadpart_past_upto, .checks = &.{"upto"} },
    .{ .m = .no_upgrade, .checks = &.{ "shared upgrade", "shared waiting", "shared during write" } },
    .{ .m = .upgrade_not_durable, .checks = &.{"shared upgrade"} },
    .{ .m = .waiting_not_flagged, .checks = &.{"shared waiting"} },
    .{ .m = .written_no_rewrite, .checks = &.{"shared during write"} },
    .{ .m = .rewrite_not_put, .checks = &.{"shared upgrade"} },
    .{ .m = .rewrite_no_dir_fsync, .checks = &.{"shared upgrade"} },
    .{ .m = .drain_skips_rewrites, .checks = &.{"drain waits for rewrites"} },
    .{ .m = .lost_is_damage, .checks = &.{"forget lost"} },
    .{ .m = .lost_names_kept, .checks = &.{"forget lost"} },
    .{ .m = .damage_not_widened, .checks = &.{"forget damaged"} },
    .{ .m = .small_files_keep_pages, .checks = &.{"pages"} },
    .{ .m = .pages_dropped_before_fsync, .checks = &.{"pages"} },
    .{ .m = .hang_no_recycle_in_wait, .checks = &.{"hang"} },
    .{ .m = .budget_each_read, .checks = &.{"budget"} },
    .{ .m = .budget_same_first, .checks = &.{"budget"} },
    .{ .m = .read_ahead_charges_preallocate, .checks = &.{"carve"} },
    .{ .m = .manifest_not_last, .checks = &.{"publish order"} },
    .{ .m = .scan_keeps_temps, .checks = &.{ "crashes", "recovery" } },
    .{ .m = .recover_no_size_check, .checks = &.{"recovery"} },
    .{ .m = .dedup_without_sha, .checks = &.{"dedup sha"} },
    .{ .m = .guard_cancels_only, .checks = &.{"guard"} },
    .{ .m = .evict_shared_first, .checks = &.{"eviction order"} },
    .{ .m = .floor_ignored, .checks = &.{"floor"} },
    .{ .m = .owner_lock_ignored, .checks = &.{"owner"} },
    .{ .m = .keep_unpinned, .checks = &.{"keep reuse"} },
    .{ .m = .forget_skips_reading, .checks = &.{"forget damaged"} },
    .{ .m = .recover_drops_on_io, .checks = &.{"recover transient"} },
    .{ .m = .handover_unreserved, .checks = &.{"hand over"} },
};

pub const checkNamed = @import("store_test_support_3.zig").checkNamed;

test {
    _ = @import("store_test_cases_1.zig");
}
