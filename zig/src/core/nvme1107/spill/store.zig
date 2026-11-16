//! Bounded per-rank spill store ownership and workers.
const types = @import("store_types.zig");
pub const std = types.std;
pub const builtin = types.builtin;
pub const c = types.c;
pub const format = types.format;
pub const fsio = types.fsio;
pub const slots_mod = types.slots_mod;
pub const direct_io = types.direct_io;
pub const mutants = types.mutants;
pub const mut = types.mut;
pub const history = types.history;
pub const Budget = types.Budget;
pub const Allocator = types.Allocator;
pub const Digest = types.Digest;
pub const Lineage = types.Lineage;
pub const Media = types.Media;
pub const Slot = types.Slot;
pub const Staging = types.Staging;
pub const Hooks = types.Hooks;
pub const Source = types.Source;
pub const Config = types.Config;
pub const SectionSpec = types.SectionSpec;
pub const Keep = types.Keep;
pub const JobId = types.JobId;
pub const ReadId = types.ReadId;
pub const Queued = types.Queued;
pub const KeepResult = types.KeepResult;
pub const StageReq = types.StageReq;
pub const Ticket = types.Ticket;
pub const WriteFailure = types.WriteFailure;
pub const CancelReason = types.CancelReason;
pub const Completion = types.Completion;
pub const GuardResult = types.GuardResult;
pub const Query = types.Query;
pub const Hit = types.Hit;
pub const ReadyBlock = types.ReadyBlock;
pub const ReadStatus = types.ReadStatus;
pub const ReadError = types.ReadError;
pub const ReadDone = types.ReadDone;
pub const SmallView = types.SmallView;
pub const Counters = types.Counters;
pub const BlockInfo = types.BlockInfo;
pub const Inode = types.Inode;
pub const Point = types.Point;
pub const WriteJob = types.WriteJob;
pub const ReadJob = types.ReadJob;
pub const Task = types.Task;
pub const RTask = types.RTask;

pub const Store = struct {
    metadata: Budget,
    histories: ?history.History = null,
    history_bytes: u64 = 0,
    gpa: Allocator,
    io: std.Io,
    cfg: Config,
    hooks: Hooks,
    root_fd: c.fd_t = -1,
    top_fd: c.fd_t = -1,
    rank_fd: c.fd_t = -1,
    blocks_fd: c.fd_t = -1,
    points_fd: c.fd_t = -1,
    lock: fsio.Lock = .{ .fd = -1 },
    writer: fsio.Writer = undefined,
    block_slot: usize,
    small_slot: usize,

    mutex: std.Io.Mutex = .init,
    work: std.Io.Condition = .init,
    rwork: std.Io.Condition = .init,
    done: std.Io.Condition = .init,

    points: std.AutoHashMapUnmanaged(Digest, *Point) = .empty,
    have: std.AutoHashMapUnmanaged(Digest, *Inode) = .empty,
    content: std.AutoHashMapUnmanaged(Digest, Digest) = .empty,
    inodes: std.AutoHashMapUnmanaged(u64, *Inode) = .empty,
    chain_prefixes: @import("prefix_pool.zig").Pool = undefined,
    refs: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
    inflight: std.AutoHashMapUnmanaged(Digest, JobId) = .empty,
    parent: std.AutoHashMapUnmanaged(Digest, ?Digest) = .empty,
    reading: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
    waiting: std.AutoHashMapUnmanaged(Digest, *WriteJob) = .empty,
    jobs_by_id: std.AutoHashMapUnmanaged(JobId, *WriteJob) = .empty,
    fifo: std.ArrayList(*WriteJob) = .empty, // jobs with blocks left to stage, oldest first
    parked: std.ArrayList(*WriteJob) = .empty,
    tasks: std.ArrayList(Task) = .empty,
    task_head: usize = 0,
    rtasks: std.ArrayList(RTask) = .empty,
    rtask_head: usize = 0,
    reads: std.ArrayList(*ReadJob) = .empty,
    read_turn: usize = 0,
    completions: std.ArrayList(Completion) = .empty,
    slots: slots_mod.Slots = .{},
    counters: Counters = .{},

    next_id: u64 = 1,
    clock: i96 = 0,
    stored_bytes: u64 = 0,
    queued_bytes: u64 = 0,
    disk_free: u64 = 0,
    pending_free: u64 = 0, // bytes removed from the index whose files the writer has not unlinked yet
    sync_sent: u64 = 0,
    sync_done: u64 = 0,
    recency_next: i96 = 0,
    recency_epoch: u64 = 0,
    recency_sync_sent: u64 = 0,
    recency_sync_done: u64 = 0,
    recency_sync_failed: u64 = 0,
    busy: u32 = 0, // writer tasks running
    stopping: bool = false,
    crashed: bool = false,
    writer_thread: ?std.Thread = null,
    reader_threads: std.ArrayList(std.Thread) = .empty,

    // -- open / close --------------------------------------------------------------------------------------------

    pub const open = @import("store_lifecycle.zig").open;
    pub const stopThreads = @import("store_lifecycle.zig").stopThreads;
    pub const close = @import("store_lifecycle.zig").close;
    pub const destroy = @import("store_lifecycle.zig").destroy;
    pub const freeJob = @import("store_lifecycle.zig").freeJob;
    pub const freeRead = @import("store_lifecycle.zig").freeRead;
    pub const namespaceName = @import("store_lifecycle.zig").namespaceName;
    pub const lockS = @import("store_lifecycle.zig").lockS;
    pub const unlockS = @import("store_lifecycle.zig").unlockS;
    pub const now = @import("store_lifecycle.zig").now;
    pub const pruneBuilds = @import("store_lifecycle.zig").pruneBuilds;
    pub const pointName = @import("store_lifecycle.zig").pointName;
    pub const blockName = @import("store_lifecycle.zig").blockName;
    pub const scan = @import("store_lifecycle.zig").scan;
    pub const scanned = @import("store_lifecycle.zig").scanned;

    pub const recover = @import("store_index.zig").recover;
    pub const PointInit = @import("store_index.zig").PointInit;
    pub const chainOf = @import("store_index.zig").chainOf;
    pub const makePoint = @import("store_index.zig").makePoint;
    pub const addBlock = @import("store_index.zig").addBlock;
    pub const dropBlock = @import("store_index.zig").dropBlock;
    pub const ref = @import("store_index.zig").ref;
    pub const gc = @import("store_index.zig").gc;
    pub const queue = @import("store_index.zig").queue;
    pub const complete = @import("store_index.zig").complete;
    pub const remove = @import("store_index.zig").remove;
    pub const extends = @import("store_index.zig").extends;
    pub const parentOf = @import("store_index.zig").parentOf;
    pub const adopt = @import("store_index.zig").adopt;
    pub const Tree = @import("store_index.zig").Tree;
    pub const find = @import("store_index.zig").find;
    pub const tree = @import("store_index.zig").tree;
    pub const Why = @import("store_index.zig").Why;
    pub const Victim = @import("store_index.zig").Victim;
    pub const evictionOrder = @import("store_index.zig").evictionOrder;

    pub const victims = @import("store_eviction.zig").victims;
    pub const removeWhy = @import("store_eviction.zig").removeWhy;
    pub const makeRoom = @import("store_eviction.zig").makeRoom;
    pub const trimPath = @import("store_eviction.zig").trimPath;
    pub const convBytes = @import("store_eviction.zig").convBytes;
    pub const trimOwn = @import("store_eviction.zig").trimOwn;

    pub const keep = @import("store_admission.zig").keep;
    pub const refuseOom = @import("store_admission.zig").refuseOom;
    pub const refuse = @import("store_admission.zig").refuse;
    pub const alreadyKept = @import("store_admission.zig").alreadyKept;
    pub const validKeep = @import("store_admission.zig").validKeep;
    pub const floorShort = @import("store_admission.zig").floorShort;
    pub const registerJob = @import("store_admission.zig").registerJob;
    pub const captured = @import("store_admission.zig").captured;
    pub const stage = @import("store_admission.zig").stage;
    pub const req = @import("store_admission.zig").req;
    pub const staged = @import("store_admission.zig").staged;
    pub const stillNamed = @import("store_admission.zig").stillNamed;
    pub const dropInflight = @import("store_admission.zig").dropInflight;

    pub const guard = @import("store_staging.zig").guard;
    pub const cancel = @import("store_staging.zig").cancel;
    pub const cancelJob = @import("store_staging.zig").cancelJob;
    pub const release = @import("store_staging.zig").release;
    pub const handOver = @import("store_staging.zig").handOver;
    pub const settle = @import("store_staging.zig").settle;
    pub const endJob = @import("store_staging.zig").endJob;
    pub const recheckParked = @import("store_staging.zig").recheckParked;
    pub const blocksSettled = @import("store_staging.zig").blocksSettled;
    pub const poll = @import("store_staging.zig").poll;
    pub const pendingStage = @import("store_staging.zig").pendingStage;
    pub const Drain = @import("store_staging.zig").Drain;
    pub const waitsOnEngine = @import("store_staging.zig").waitsOnEngine;
    pub const drain = @import("store_staging.zig").drain;
    pub const flushRecency = @import("store_staging.zig").flushRecency;
    pub const dirtyRecency = @import("store_staging.zig").dirtyRecency;
    pub const flushRecencyLocked = @import("store_staging.zig").flushRecencyLocked;
    pub const pins = @import("store_staging.zig").pins;
    pub const pinsAny = @import("store_staging.zig").pinsAny;

    pub const writerLoop = @import("store_writer.zig").writerLoop;
    pub const dropTask = @import("store_writer.zig").dropTask;
    pub const runTask = @import("store_writer.zig").runTask;
    pub const refreshHistoryBytes = @import("store_writer.zig").refreshHistoryBytes;
    pub const unlinkPointFiles = @import("store_writer.zig").unlinkPointFiles;
    pub const crash = @import("store_writer.zig").crash;
    pub const writeBlock = @import("store_writer.zig").writeBlock;
    pub const writePoint = @import("store_writer.zig").writePoint;
    pub const failPoint = @import("store_writer.zig").failPoint;

    pub const manifestOf = @import("store_reader.zig").manifestOf;
    pub const rewriteShared = @import("store_reader.zig").rewriteShared;
    pub const lookup = @import("store_reader.zig").lookup;
    pub const eligible = @import("store_reader.zig").eligible;
    pub const mediaMatch = @import("store_reader.zig").mediaMatch;
    pub const complete_ = @import("store_reader.zig").complete_;
    pub const presence = @import("store_reader.zig").presence;
    pub const partOf = @import("store_reader.zig").partOf;
    pub const touch = @import("store_reader.zig").touch;
    pub const markShared = @import("store_reader.zig").markShared;
    pub const BeginError = @import("store_reader.zig").BeginError;
    pub const beginRead = @import("store_reader.zig").beginRead;
    pub const readJob = @import("store_reader.zig").readJob;
    pub const readsFull = @import("store_reader.zig").readsFull;
    pub const liveReads = @import("store_reader.zig").liveReads;
    pub const assign = @import("store_reader.zig").assign;
    pub const giveRead = @import("store_reader.zig").giveRead;
    pub const readerLoop = @import("store_reader.zig").readerLoop;
    pub const BlockReadError = @import("store_reader.zig").BlockReadError;
    pub const readBlock = @import("store_reader.zig").readBlock;
    pub const readSmall = @import("store_reader.zig").readSmall;

    pub const readFailed = @import("store_results.zig").readFailed;
    pub const readReady = @import("store_results.zig").readReady;
    pub const readRelease = @import("store_results.zig").readRelease;
    pub const Recycler = @import("store_results.zig").Recycler;
    pub const readAwait = @import("store_results.zig").readAwait;
    pub const readSmallView = @import("store_results.zig").readSmallView;
    pub const Failure = @import("store_results.zig").Failure;
    pub const readFailure = @import("store_results.zig").readFailure;
    pub const EndError = @import("store_results.zig").EndError;
    pub const Outcome = @import("store_results.zig").Outcome;
    pub const endRead = @import("store_results.zig").endRead;
    pub const maybeFreeRead = @import("store_results.zig").maybeFreeRead;
    pub const forget = @import("store_results.zig").forget;
    pub const stepReads = @import("store_results.zig").stepReads;
    pub const counts = @import("store_results.zig").counts;
    pub const refreshDiskFree = @import("store_results.zig").refreshDiskFree;
    pub const isCrashed = @import("store_results.zig").isCrashed;
    pub const Summary = @import("store_results.zig").Summary;
    pub const summary = @import("store_results.zig").summary;
};
