//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block layout, lineage, write-at-keep staging ring,
pub const std = @import("std");
pub const builtin = @import("builtin");
pub const c = std.c;
pub const format = @import("format.zig");
pub const fsio = @import("fsio.zig");
pub const slots_mod = @import("slots.zig");
pub const direct_io = @import("../direct_io.zig");
pub const mutants = @import("mutants.zig");
pub const mut = mutants.on;
pub const history = @import("history.zig");
pub const Budget = @import("budget.zig").Budget;
pub const Allocator = std.mem.Allocator;

pub const Digest = format.Digest;
pub const Lineage = format.Lineage;
pub const Media = format.Media;
pub const Slot = slots_mod.Slot;
pub const Staging = slots_mod.Staging;
pub const Hooks = fsio.Hooks;

/// The logical rows a write reads from: the family's extent and its generation. A later occupant of the same base
pub const Source = struct {
    id: u64,
    generation: u64,

    pub fn eql(a: Source, b: Source) bool {
        return a.id == b.id and a.generation == b.generation;
    }
};

pub const Config = struct {
    /// The spill directory (a local NVMe per rank; never the shared model/Engram mount). Must exist.
    root: [:0]const u8,
    rank: u32,
    /// The agreed compatibility digest (`format.compatDigest` over the common digest every rank exchanged).
    compat: Digest,
    /// Written into every manifest; a point of another layout is foreign here (owner_sharded with TF_DS_KV_SHARD).
    owner_layout: format.OwnerLayout = .replicated,
    limits: format.Limits,
    /// Bytes this rank's store may hold.
    cap_bytes: u64,
    /// No write leaves less than this free on the store's file system.
    floor_bytes: u64 = 50 << 30,
    /// A conversation holding more than this fraction of the cap trims its own old turns and forks first (0: off).
    quota: f64 = 0,
    /// Newest states kept on a conversation's active path (0: every turn until room is needed).
    turns: u32 = 4,
    /// Shorter states are not written.
    min_tokens: u32 = 8192,
    /// Bytes of unpublished writes (rows still in device memory, queued staging) past which a keep is refused.
    backlog_bytes: u64 = 16 << 30,
    /// The caller's staging buffers (block slots of `limits.block_bytes`, small slots of `limits.max_small_bytes`).
    staging: Staging,
    /// Host allocations for indexes, pending metadata and reader/writer bookkeeping.
    index_bytes: usize = 512 << 20,
    readers: u32 = 4,
    worker_stack_bytes: usize = 1 << 20,
    /// Restores at once (each holds a restore slot for its small state).
    max_reads: u32 = 4,
    direct: bool = true,
    /// Other compat namespaces kept at open, newest first (a live one, still locked, is never touched).
    keep_builds: u32 = 1,
    /// RAM priority changes immediately; changed disk timestamps are batched every two hours.
    recency_interval_ns: u64 = 7200 * std.time.ns_per_s,
    hooks: Hooks = .{},
};

pub const SectionSpec = struct { name: []const u8, len: u64 };

/// What `keep` stores.
pub const Keep = struct {
    /// Canonical key ids of [0, at) (the prompt cache's keyed ids, picture keys included).
    keyed: []const u32,
    /// The raw ids when they differ from `keyed` (pictures), else null.
    raw: ?[]const u32 = null,
    media: []const Media = &.{},
    lineage: Lineage,
    origin: ?format.Origin = null,
    shared: bool = false,
    group: u64 = 0,
    sections: []const SectionSpec,
    /// Bytes of the last block when `keyed.len` is not a whole number of blocks (captured now: a mutable tail).
    tail_bytes: u64 = 0,
    source: Source,
};

pub const JobId = u64;
pub const ReadId = u64;

pub const Queued = struct {
    job: JobId,
    /// The small state's bytes: the caller fills each section at its offset (asynchronously), then `captured`.
    small: []u8,
    sections: []const format.Section,
    /// The tail block's bytes when the point has a tail (filled with the small state, before `captured`).
    tail: ?[]u8 = null,
};

pub const KeepResult = union(enum) {
    queued: Queued,
    /// Stored or on its way already (a shared request made it shared).
    stored,
    short,
    no_space,
    no_staging,
    metadata,
    backlog,
    invalid,
};

/// One block to pack into `dst`: the source's tokens [first, first + tokens), `bytes` long.
pub const StageReq = struct {
    ticket: Ticket,
    source: Source,
    first: u32,
    tokens: u32,
    dst: []u8,
};

pub const Ticket = struct { job: JobId, index: u32, slot: Slot };

pub const WriteFailure = enum { io, no_space, capture, stage, missing_block, crashed, invalid };
pub const CancelReason = enum { guard, source_gone, explicit, shutdown };

pub const Completion = union(enum) {
    published: struct { job: JobId, key: Digest, at: u32 },
    failed: struct { job: JobId, key: Digest, why: WriteFailure },
    cancelled: struct { job: JobId, key: Digest, why: CancelReason },
};

pub const GuardResult = struct { staged: usize = 0, cancelled: usize = 0 };

pub const Query = struct {
    keyed: []const u32,
    media: []const Media = &.{},
    /// Only points longer than this.
    longer_than: u32 = 0,
    /// The longest usable point (`Lanes.usable`: at < prompt.len, at <= prompt.len - window, ...).
    max_at: u32,
    /// Points on this grid only (0: any).
    grid: u32 = 0,
};

/// A lookup's point: `part` is its lineage's part salt (owner-sharded stores: the part its rows were written from,
pub const Hit = struct { key: Digest, at: u32, shared: bool, group: u64, part: ?u32 = null };

pub const ReadyBlock = struct { index: u32, first: u32, tokens: u32, data: []const u8, slot: Slot };

pub const ReadStatus = enum { more, waiting, done, failed };

pub const ReadError = enum { missing, damaged, io, small, cancelled };

pub const ReadDone = struct {
    key: Digest,
    at: u32,
    shared: bool,
    group: u64,
    has_raw: bool,
    lineage: Lineage, // gpa-owned
    origin: ?format.Origin,
};

pub const SmallView = struct { bytes: []const u8, sections: []const format.Section };

pub const Counters = struct {
    keeps: u64 = 0,
    queued: u64 = 0,
    stored_skips: u64 = 0,
    short: u64 = 0,
    no_space: u64 = 0,
    no_staging: u64 = 0,
    metadata_refused: u64 = 0,
    index_bytes: u64 = 0,
    index_peak: u64 = 0,
    index_limit: u64 = 0,
    backlog_refused: u64 = 0,
    invalid: u64 = 0,
    committed: u64 = 0,
    write_failed: u64 = 0,
    cancelled: u64 = 0,
    cancelled_guard: u64 = 0,
    cancelled_source: u64 = 0,
    staged_blocks: u64 = 0,
    guard_staged: u64 = 0,
    written_blocks: u64 = 0,
    written_bytes: u64 = 0,
    written_small_bytes: u64 = 0,
    dedup_blocks: u64 = 0,
    dedup_bytes: u64 = 0,
    shared_marked: u64 = 0,
    shared_rewrites: u64 = 0,
    pruned_turns: u64 = 0,
    pruned_forks: u64 = 0,
    pruned_conversations: u64 = 0,
    pruned_shared: u64 = 0,
    trimmed_turns: u64 = 0,
    quota_trims: u64 = 0,
    dropped: u64 = 0,
    disk_lookups: u64 = 0,
    disk_hits: u64 = 0,
    read_deferred: u64 = 0,
    loaded: u64 = 0,
    loaded_tokens: u64 = 0,
    loaded_bytes: u64 = 0,
    read_block_calls: u64 = 0,
    read_small_calls: u64 = 0,
    load_failed: u64 = 0,
    crc_errors: u64 = 0,
    size_errors: u64 = 0,
    missing_errors: u64 = 0,
    recovered_points: u64 = 0,
    recovered_dropped: u64 = 0,
    /// Points recovery could not read for a reason other than damage (an I/O error, EMFILE, a busy file): left on
    recover_deferred: u64 = 0,
    temps_removed: u64 = 0,
    orphans_removed: u64 = 0,
    builds_pruned: u64 = 0,
    direct_refused: u64 = 0,
    recency_writes: u64 = 0,
    recency_errors: u64 = 0,
    recency_pending: u64 = 0,
    // gauges
    points: u64 = 0,
    blocks: u64 = 0,
    stored_bytes: u64 = 0,
    queued_bytes: u64 = 0,
    pending_jobs: u64 = 0,
    reads: u64 = 0,
    staging_bytes: u64 = 0,
    staging_peak_bytes: u64 = 0,
    disk_free_bytes: u64 = 0,
};

pub const BlockInfo = struct { crc: u32, sha: Digest, ino: u64, bytes: u64 };
pub const Inode = struct {
    crc: u32,
    sha: Digest,
    ino: u64,
    bytes: u64,
    names: std.ArrayList(Digest) = .empty,
};

pub const Point = struct {
    key: Digest,
    at: u32,
    shared: bool,
    group: u64,
    used: i96,
    created_ns: i64,
    has_raw: bool,
    mutable_tail: bool,
    lineage: Lineage,
    origin: ?format.Origin,
    chain: []Digest,
    chain_lease: ?@import("prefix_pool.zig").Lease = null,
    tail: []u32, // ids after the last whole block
    media: []Media,
    blocks: []format.BlockRef,
    sections: []format.Section,
    history_roots: history.Roots = .{},
    ids_disk_bytes: u64 = 0,
    ids_bytes: u64,
    ids_crc: u32,
    small_bytes: u64,
    small_crc: u32,
    files: u64 = 0, // bytes of its .ids, .st and .man
    pins: u32 = 0, // restores holding it (freed only once none does)
    removed: bool = false,
    recency_dirty: bool = false,
    recency_epoch: u64 = 0,

    pub fn deinit(p: *Point, gpa: Allocator) void {
        p.lineage.deinit(gpa);
        if (p.chain_lease) |lease| lease.release() else gpa.free(p.chain);
        gpa.free(p.tail);
        gpa.free(p.media);
        gpa.free(p.blocks);
        gpa.free(p.sections);
        gpa.destroy(p);
    }

    pub fn whole(p: *const Point) usize {
        return p.chain.len;
    }
};

pub const WriteJob = struct {
    id: JobId,
    point: *Point,
    source: Source,
    new: std.ArrayList(u32) = .empty, // block indices this job stages (no tail)
    next: usize = 0,
    tail: ?u32 = null,
    tail_slot: ?Slot = null,
    outstanding: u32 = 0, // staged blocks not yet written (or given back)
    engine: u32 = 0, // of those, slots the engine holds (its copy not yet reported by `staged`)
    small_slot: Slot,
    capture: enum { pending, ok, failed } = .pending,
    state: enum { open, publishing, failed, cancelled } = .open,
    fail: WriteFailure = .io,
    cancel: CancelReason = .explicit,
    size: u64,
    ids: []u8,
    listed: bool = true, // in the staging FIFO
    parked: bool = false,
    shared_at_write: bool = false,
};

pub const ReadJob = struct {
    id: ReadId,
    point: *Point,
    total: u32,
    state: []BState,
    slot: []Slot,
    next_assign: u32 = 0,
    next_hand: u32 = 0,
    small_slot: Slot,
    small: enum { queued, ready, failed } = .queued,
    err: ?ReadError = null,
    bad: std.ArrayList(Digest) = .empty,
    lost: std.ArrayList(Digest) = .empty,
    disk: u32 = 0, // tasks queued or running on readers
    held: u32 = 0, // read slots it holds (reading, read, or handed and not released)
    ended: bool = false,
    t0: i96,

    pub const BState = enum(u8) { idle, reading, ready, handed, failed };
};

pub const Task = union(enum) {
    block: struct { job: *WriteJob, index: u32, slot: Slot },
    point: *WriteJob,
    shared: Digest,
    touch: struct { key: Digest, ns: i96 },
    unlink_block: struct { name: Digest, bytes: u64 },
    unlink_point: struct { key: Digest, bytes: u64, roots: history.Roots = .{} },
    sync: u64,
    recency_sync: u64,
};

pub const RTask = struct { job: *ReadJob, index: ?u32, slot: Slot };
