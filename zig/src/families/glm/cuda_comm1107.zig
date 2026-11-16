//! Private comparison adapter: pinned 1107 native Ring, lent through Group.Multi.
//! One TP4 context, one ordered main stream, ordinary rank-major byte gathers.
//! Source-only handoff: no GLM build, hardware, fault or healthy-close certificate.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const tp = @import("tp");
const Group = cuda.tp_group.Group;
const H = ?*anyopaque;
const cap = 1024 * 1024;
const bridge_sha = "1bf5042123f471a1c2c90ea67d262c2986e00a7eb34c98724a14f22945eba3b0";
const kernels_sha = "23affd2303d7da3077ef6afc97f09e1c607f1325aacbb9a24842388389c3bd80";

const Api = struct {
    lib: std.DynLib,
    tf_create: *const fn (u32, u32, [*]align(8) const u8, usize, usize, u32, u32) callconv(.c) H,
    tf_info_size: *const fn () callconv(.c) usize,
    tf_info: *const fn (H, u32, [*]u8) callconv(.c) void,
    tf_connect: *const fn (H, [*]const u8) callconv(.c) c_int,
    tf_start: *const fn (H, u32) callconv(.c) c_int,
    tf_enqueue: *const fn (H, cuda.abi.Stream, u64, u64, u32) callconv(.c) c_int,
    tf_tune: *const fn (H, u32, u32, u32, u32, u32) callconv(.c) c_int,
    tf_failure: *const fn (H) callconv(.c) ?[*:0]const u8,
    tf_abort: *const fn (H) callconv(.c) void,
    tf_close: *const fn (H) callconv(.c) void,
    tf_host_ranges: *const fn (H, [*]u64) callconv(.c) void,

    fn open(path: []const u8) !Api {
        var lib = try std.DynLib.open(path);
        errdefer lib.close();
        var a: Api = undefined;
        a.lib = lib;
        const ti = @typeInfo(Api).@"struct";
        inline for (ti.field_names, ti.field_types) |name, T| {
            if (comptime !std.mem.eql(u8, name, "lib"))
                @field(a, name) = lib.lookup(T, name) orelse return error.Native1107Symbol;
        }
        return a;
    }
};

// The exact bridge's native Info ABI. It contains no capacity or peer-MTU field.
const Info = extern struct {
    devices: u32,
    qpn: [4]u32,
    psn: [4]u32,
    rkey: [4]u32,
    gid: [4][16]u8,
    recv: u64,
};

const Settings = struct {
    library: []const u8,
    kernels: []const u8,
    cpu: u32,
    master: []const u8,
    port: u16,
    timeout_s: u32,

    fn read() !Settings {
        // These are fixed capabilities of this pinned factory, not working override knobs.
        if (try number("TF_GLM_1107_MAX_KB", 1024) != 1024 or
            try number("TF_GLM_1107_GID_INDEX", 3) != 3 or
            try number("TF_GLM_1107_TC", 104) != 104 or
            try number("SPARKCOMM_TF_SLOTS", 4) != 4 or
            try number("TF_DS_RDMA_PREFIX", 24) != 24 or
            !eq("SC_TF_ENABLE", "0") or !eq("TF_COMPARE_NATIVE_CONTROL", "1") or
            !eq("TF_ROCE", "0") or !eq("TF_GLM_ROCE_MULTI", "0")) return error.Native1107Settings;
        const cpu = try std.fmt.parseInt(u32, env("TF_GLM_1107_CPU") orelse return error.Native1107Settings, 10);
        if (cpu >= 1024) return error.Native1107Settings;
        // Exactly one lease in the bridge's checked native comparison lease pool.
        const lease = try std.fmt.parseInt(u32, env("SC_POSTER_CPUS") orelse return error.Native1107Settings, 10);
        if (lease != cpu) return error.Native1107Settings;
        const port = try number("TF_TP_PLAN_PORT", 0);
        const timeout_s = try number("TF_TP_PLAN_TIMEOUT", 30);
        if (port == 0 or port > 65535 or timeout_s == 0 or timeout_s > 300) return error.Native1107Settings;
        const master = env("TF_TP_MASTER") orelse return error.Native1107Settings;
        if (master.len == 0) return error.Native1107Settings;
        return .{
            .library = env("TF_GLM_1107_LIBRARY") orelse return error.Native1107Settings,
            .kernels = env("TF_GLM_1107_KERNELS") orelse return error.Native1107Settings,
            .cpu = cpu,
            .master = master,
            .port = @intCast(port),
            .timeout_s = timeout_s,
        };
    }
};

// Agreed before opening any backend. Local CPU/path strings may differ; payload,
// protocol, control endpoint and fixed binary hashes must not differ.
const Choice = extern struct {
    version: u32,
    enabled: u32,
    valid: u32,
    world: u32,
    payload: u32,
    slots: u32,
    gid: u32,
    tc: u32,
    stage: u32,
    collect: u32,
    grain: u32,
    pdl: u32,
    control_port: u32,
    timeout_s: u32,
    master_sha: [32]u8,
};

/// Called on every rank before Group.openRoce/cuda_roce_multi.open.
/// true means this adapter owns Group.multi; false leaves both existing selectors alone.
pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, g: *Group, main: cuda.Stream, stock_multi: bool) !bool {
    if (g.fast != null or g.multi != null) return error.Native1107AlreadyOpen;
    const requested = env("TF_GLM_COMM_1107") orelse "0";
    const on = std.mem.eql(u8, requested, "1");
    if (g.world < 2) {
        if (!std.mem.eql(u8, requested, "0")) return error.Native1107Settings;
        return false;
    }
    const parsed: ?Settings = if (on) Settings.read() catch null else null;
    var mine = std.mem.zeroes(Choice);
    mine.version = 1;
    mine.enabled = @intFromBool(on);
    mine.world = @intCast(g.world);
    mine.valid = @intFromBool(std.mem.eql(u8, requested, "0") or (on and parsed != null and g.world == 4 and !stock_multi and builtin.os.tag == .linux));
    if (parsed) |s| {
        mine.payload = cap;
        mine.slots = 4;
        mine.gid = 3;
        mine.tc = 104;
        mine.stage = 48;
        mine.collect = 48;
        mine.grain = 2;
        mine.pdl = 1;
        mine.control_port = s.port;
        mine.timeout_s = s.timeout_s;
        std.crypto.hash.sha2.Sha256.hash(s.master, &mine.master_sha, .{});
    }
    const choices = try gpa.alloc(Choice, g.world);
    defer gpa.free(choices);
    // No wire is lent yet: this is the group's original NCCL, even for 16-byte records.
    try g.gather(Choice, d, main, mine, choices);
    for (choices) |*c| {
        if (c.valid != 1 or !std.mem.eql(u8, std.mem.asBytes(c), std.mem.asBytes(&mine)))
            return error.Native1107Agreement;
    }
    if (!on) return false;
    const s = parsed.?;
    var b = try tp.Bootstrap.open(.{
        .rank = @intCast(g.rank),
        .world = @intCast(g.world),
        .master = s.master,
        .link_port = s.port,
        .connect_timeout_ns = @as(u64, s.timeout_s) * std.time.ns_per_s,
    });
    defer b.close(); // startup control only; this does not supply process liveness later.
    const local: ?*Wire = create(gpa, io, d, g.rank, main, s) catch |err| blk: {
        std.log.err("rank {d}: native1107 prepare: {s}", .{ g.rank, @errorName(err) });
        break :blk null;
    };
    var accepted = false;
    defer if (!accepted) {
        if (local) |w| dispose(w);
    };
    if (!try every(&b, local != null)) return error.Native1107Prepare;
    const w = local.?;
    const connected = connect(w, &b) catch false;
    if (!try every(&b, connected)) return error.Native1107Connect;
    const started = w.api.tf_start(w.h, s.cpu) == 0;
    if (!try every(&b, started)) return error.Native1107Start;
    try probe(w, g, &b);
    g.multi = .{ .ctx = w, .fits = fits, .gather = gather, .check = check, .abort = abort, .deinit = deinit };
    accepted = true;
    std.log.info("rank {d}: GLM_NATIVE1107_ACTIVE bridge={s} kernels={s} cpu={d} world=4 cap=1048576 slots=4 stage=48 collect=48 grain=2 PDL=1 GID=3 TC=104 rails=f0,f0-twin SC=OFF idle_gate=unwired", .{ g.rank, bridge_sha, kernels_sha, s.cpu });
    return true;
}

/// Only this exact factory may lend the native graph path; no stock or fallback wire is admitted.
pub fn graphReady(g: *const Group, s: cuda.Stream, bytes: usize) bool {
    if (g.world != 4 or g.fast != null) return false;
    const m = g.multi orelse return false;
    if (m.gather != gather or m.check != check) return false;
    const w = of(m.ctx);
    return s.handle == w.main.handle and !w.aborted.load(.acquire) and bytes > 0 and bytes <= cap and bytes % 16 == 0;
}

const Wire = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    api: Api,
    h: H,
    rank: usize,
    main: cuda.Stream,
    host_calls: std.atomic.Value(u64) = .init(0),
    host_bytes: std.atomic.Value(u64) = .init(0),
    rejected_counts: std.atomic.Value(u64) = .init(0),
    aborted: std.atomic.Value(bool) = .init(false),
};

fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, rank: usize, main: cuda.Stream, s: Settings) !*Wire {
    // Load-time file pins require immutable parent-owned mounts to avoid path replacement races.
    const binary = try std.Io.Dir.cwd().readFileAlloc(io, s.library, gpa, .limited(64 << 20));
    defer gpa.free(binary);
    try digest(binary, bridge_sha);
    const image = try std.Io.Dir.cwd().readFileAllocOptions(io, s.kernels, gpa, .limited(64 << 20), .@"16", null);
    defer gpa.free(image);
    try digest(image, kernels_sha);
    var api = try Api.open(s.library);
    errdefer api.lib.close();
    if (api.tf_info_size() != @sizeOf(Info) or @sizeOf(Info) != 128) return error.Native1107InfoAbi;
    var before: cuda.abi.Context = null;
    try d.check(d.api.cuCtxGetCurrent(&before), "native1107 context before");
    if (before == null) return error.Native1107Context;
    // The pinned factory opens a driver table and module in the caller's context; it must not create a context.
    const h = api.tf_create(@intCast(rank), 4, image.ptr, image.len, cap, 1, 2) orelse return error.Native1107Create;
    var after: cuda.abi.Context = null;
    const got_context = d.api.cuCtxGetCurrent(&after) == cuda.abi.success;
    if (!got_context or after != before) {
        // No enqueues/proxy yet. Destroy in the factory's current context, restore the caller, refuse.
        api.tf_close(h);
        _ = d.api.cuCtxSetCurrent(before);
        return error.Native1107Context;
    }
    errdefer api.tf_close(h);
    if (api.tf_tune(h, 48, 48, 2, 1, 1) != 0) return error.Native1107Tune;
    const w = try gpa.create(Wire);
    w.* = .{ .gpa = gpa, .d = d, .api = api, .h = h, .rank = rank, .main = main };
    return w;
}

fn connect(w: *Wire, b: *tp.Bootstrap) !bool {
    var mine: [4 * @sizeOf(Info)]u8 = undefined;
    for (0..4) |p| w.api.tf_info(w.h, @intCast(p), mine[p * @sizeOf(Info) ..].ptr);
    var storage: [4 * @sizeOf(@TypeOf(mine))]u8 = undefined;
    var blobs: [4][]const u8 = undefined;
    const all = try b.allGather(&mine, &storage, &blobs);
    var remote: [4]Info = undefined;
    const own = std.mem.bytesToValue(Info, mine[0..@sizeOf(Info)]);
    for (all, 0..) |blob, p| {
        if (blob.len != mine.len) return false;
        remote[p] = std.mem.bytesToValue(Info, blob[w.rank * @sizeOf(Info) ..][0..@sizeOf(Info)]);
        if (remote[p].devices != 2) return false;
        for (0..2) |h| {
            const gid = remote[p].gid[h];
            // Match the native route's IPv4 /24 on BOTH PCIe functions. No relay/one-rail arm.
            if (!std.mem.eql(u8, gid[0..12], &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255 }) or
                !std.mem.eql(u8, own.gid[h][12..15], gid[12..15])) return false;
        }
    }
    return w.api.tf_connect(w.h, std.mem.sliceAsBytes(&remote).ptr) == 0;
}

// In-place, main-stream 4 KiB native gather versus DIRECT NCCL, with host readiness
// votes out of band. A failed enqueue aborts every local wait before any sync/free.
fn probe(w: *Wire, g: *const Group, b: *tp.Bootstrap) !void {
    const n = 4096;
    var buf: ?cuda.DeviceBuffer = cuda.DeviceBuffer.alloc(w.d, 8 * n) catch null;
    var done = false;
    defer if (buf) |*v| {
        if (!done) w.api.tf_abort(w.h);
        // Do not free a device allocation when the context cannot prove quiescence.
        if (w.d.api.cuCtxSynchronize() == cuda.abi.success) v.free();
    };
    if (!try every(b, buf != null)) return error.Native1107ProbeAlloc;
    var row: [n]u8 = undefined;
    for (&row, 0..) |*v, i| v.* = @truncate(i * 17 + w.rank * 53);
    const uploaded = blk: {
        buf.?.uploadAsync(w.rank * n, &row, w.main.handle) catch break :blk false;
        buf.?.uploadAsync((4 + w.rank) * n, &row, w.main.handle) catch break :blk false;
        w.main.synchronize() catch break :blk false;
        break :blk true;
    };
    if (!try every(b, uploaded)) return error.Native1107ProbeUpload;
    const enqueued = w.api.tf_enqueue(w.h, w.main.handle, buf.?.ptr + w.rank * n, buf.?.ptr, n) == 0;
    if (!try every(b, enqueued)) return error.Native1107ProbeEnqueue;
    const reference_queued = g.lib.api.ncclAllGather(buf.?.ptr + (4 + w.rank) * n, buf.?.ptr + 4 * n, n, .u8, g.comm, w.main.handle) == 0;
    if (!try every(b, reference_queued)) return error.Native1107ProbeNccl;
    const waited = if (w.main.synchronize()) |_| true else |_| blk: {
        w.api.tf_abort(w.h);
        break :blk false;
    };
    if (!try every(b, waited)) return error.Native1107ProbeWait;
    const checked = if (check(w)) |_| true else |_| false;
    if (!try every(b, checked)) return error.Native1107ProbeFailure;
    var got: [8 * n]u8 = undefined;
    const downloaded = if (buf.?.download(0, &got)) |_| true else |_| false;
    if (!try every(b, downloaded)) return error.Native1107ProbeDownload;
    if (!try every(b, std.mem.eql(u8, got[0 .. 4 * n], got[4 * n ..]))) return error.Native1107ProbeMismatch;
    done = true;
}

fn fits(ctx: *anyopaque, bytes: usize) bool {
    const w = of(ctx);
    const yes = bytes > 0 and bytes <= cap and bytes % 16 == 0;
    if (!yes) _ = w.rejected_counts.fetchAdd(1, .monotonic);
    return yes;
}

fn gather(ctx: *anyopaque, s: cuda.abi.Stream, mine: u64, all: u64, bytes: usize) anyerror!void {
    const w = of(ctx);
    // Cross-stream use of a shared epoch is refused, not serialized silently.
    if (s != w.main.handle or mine % 16 != 0 or all % 16 != 0 or !fits(w, bytes)) {
        abort(w);
        return error.Native1107ShapeOrStream;
    }
    try check(w);
    if (w.api.tf_enqueue(w.h, s, mine, all, @intCast(bytes)) != 0) {
        abort(w);
        return error.Native1107Enqueue;
    }
    _ = w.host_calls.fetchAdd(1, .monotonic);
    _ = w.host_bytes.fetchAdd(bytes, .monotonic);
}

fn check(ctx: *anyopaque) anyerror!void {
    const w = of(ctx);
    if (w.aborted.load(.acquire)) return error.Native1107Aborted;
    if (w.api.tf_failure(w.h)) |why| {
        std.log.err("rank {d}: native1107: {s}", .{ w.rank, std.mem.span(why) });
        return error.Native1107Failed;
    }
}

fn abort(ctx: *anyopaque) void {
    const w = of(ctx);
    w.aborted.store(true, .release);
    w.api.tf_abort(w.h);
}

fn deinit(ctx: *anyopaque) void {
    dispose(of(ctx));
}

fn dispose(w: *Wire) void {
    // Outer Engine.deinit must also drain BEFORE freeing its own producer/consumer
    // buffers. Retaining this wire cannot repair an earlier outer lifetime failure.
    if (w.d.api.cuCtxSynchronize() != cuda.abi.success) {
        std.log.err("rank {d}: GLM_NATIVE1107_RETAINED context not quiescent; module/registrations/lease kept to process exit", .{w.rank});
        return;
    }
    var ranges: [7]u64 = undefined;
    w.api.tf_host_ranges(w.h, &ranges);
    const meta: [*]const u64 = @ptrFromInt(ranges[4]);
    const published = @atomicLoad(u64, &meta[16], .acquire); // slots4 * world4
    var sent_max: u64 = 0;
    for (21..25) |i| sent_max = @max(sent_max, @atomicLoad(u64, &meta[i], .acquire));
    std.log.info("rank {d}: GLM_NATIVE1107_CLOSE host_enqueues={d} host_bytes={d} rejected_count_checks={d} GPU_published_including_probe={d} local_TX_sent_max_before_close={d} aborted={any}", .{ w.rank, w.host_calls.load(.monotonic), w.host_bytes.load(.monotonic), w.rejected_counts.load(.monotonic), published, sent_max, w.aborted.load(.acquire) });
    // Native close has a void ABI: this line is an invocation, not a close-success
    // certificate or proof that the last local TX CQE was reaped before destroy.
    w.api.tf_close(w.h);
    w.api.lib.close();
    w.gpa.destroy(w);
}

fn every(b: *tp.Bootstrap, yes: bool) !bool {
    var out: [4]u8 = undefined;
    var blobs: [4][]const u8 = undefined;
    const all = try b.allGather(&.{@intFromBool(yes)}, &out, &blobs);
    for (all) |v| if (v.len != 1 or v[0] != 1) return false;
    return true;
}

fn digest(bytes: []const u8, expected: []const u8) !void {
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    var hex: [64]u8 = undefined;
    const chars = "0123456789abcdef";
    for (actual, 0..) |v, i| {
        hex[2 * i] = chars[v >> 4];
        hex[2 * i + 1] = chars[v & 15];
    }
    if (!std.mem.eql(u8, &hex, expected)) return error.Native1107Pin;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |s| std.mem.span(s) else null;
}
fn eq(name: [*:0]const u8, value: []const u8) bool {
    return if (env(name)) |v| std.mem.eql(u8, v, value) else false;
}
fn number(name: [*:0]const u8, fallback: u32) !u32 {
    return if (env(name)) |v| std.fmt.parseInt(u32, v, 10) else fallback;
}
fn of(ctx: *anyopaque) *Wire {
    return @ptrCast(@alignCast(ctx));
}
