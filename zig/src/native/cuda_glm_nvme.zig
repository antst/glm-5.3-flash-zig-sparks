//! GLM conversation retention on every rank's local NVMe, using the 1107 bounded block tier.
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const nvme = @import("core").nvme1107;
const pc = api.prompt_cache;
const radix = api.prompt_radix;
const page_adapter = @import("cuda_prompt_pages.zig");
const wire = glm.nvme_wire;
const rank_state = glm.nvme_state;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Header = glm.follow.Header;

pub fn wanted() !bool {
    return glm.nvme_options.wanted();
}

fn gib(value: f64) !u64 {
    if (!std.math.isFinite(value) or value < 0 or value > 65536) return error.BadGlmNvmeTier;
    return @intFromFloat(value * (1 << 30));
}

fn identity(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, e: *Engine, o: api.Open) ![32]u8 {
    const exe = try std.process.executablePathAlloc(io, a);
    defer a.free(exe);
    const code = try @import("sys").file_identity.hash(io, exe, 1 << 31);
    const f = glm.ranks.Fingerprint.of(e.share.world, 0, e.settings, e.capacity, e.mtp, e.checkpoint_digest, code, e.checkpoint_digest);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(if (e.share_pages) "glm-native-nvme-page-image-v1" else "glm-native-nvme1107-image-v1/ordinary-snapshot-lease");
    const physical_rows: u64 = glm.forward.window_rows;
    h.update(std.mem.asBytes(&physical_rows));
    inline for (@typeInfo(@TypeOf(f)).@"struct".field_names) |name| h.update(std.mem.asBytes(&@field(f, name)));
    const gpu = [3]u64{ try ctx.capability(), @intCast(try ctx.d.version()), @intFromBool(e.mtp) };
    h.update(std.mem.asBytes(&gpu));
    const draft_digest: ?[32]u8 = if (e.drafter != null)
        try glm.ranks.artifact(a, io, o.drafter orelse return error.NvmeDrafterIdentityScope)
    else
        null;
    return @import("cuda_glm_cache_identity.zig").withDrafter(h.finalResult(), draft_digest, o.drafter_bits);
}

const Slot = struct { id: u64 = 0, sent: bool = false };
const Owner = struct {
    e: *Engine,
    ctx: *const cuda.Context,
    store: ?*pc.Store = null,
    lookahead: u32,
    slots: [rank_state.max_pending]Slot = @splat(.{}),
    replies: [8]u64 = undefined,
    pump_status: [8]u64 = @splat(0),
    boot: [16]u8,
    serial: u64 = 0,
    stopped: bool = false,
    queued: u64 = 0,
    refused: u64 = 0,
    restored: u64 = 0,
    failed_reads: u64 = 0,
    room_waits: u64 = 0,
    room_ns: u64 = 0,
    room_cancelled: u64 = 0,
    room_commands: u64 = 0,
    room_max_budget_ns: u64 = 0,
    room_progress_changes: u64 = 0,
    room_hard_stops: u64 = 0,

    fn of(p: *anyopaque) *Owner {
        return @ptrCast(@alignCast(p));
    }
    fn local(o: *Owner) *rank_state.State {
        const s = rank_state.local orelse unreachable;
        std.debug.assert(s.e == o.e);
        return s;
    }

    fn command(o: *Owner, op: wire.Op, h: Header, ids: []const u32) []const u64 {
        // A restarted cache may be the lane thread's first backend work; no CUDA context is inherited by that thread.
        o.ctx.makeCurrent() catch |err| {
            std.log.err("GLM_NVME1107_CONTEXT failed ({s}); every rank must restart", .{@errorName(err)});
            std.c._exit(75);
        };
        return wire.invoke(o.e, op, h, ids, &o.replies) catch |err| {
            std.log.err("GLM_NVME1107_PROTOCOL failed ({s}); every rank must restart", .{@errorName(err)});
            std.c._exit(75);
        };
    }

    fn required(o: *Owner, op: wire.Op, h: Header, ids: []const u32) []const u64 {
        const r = o.command(op, h, ids);
        if (!wire.allOk(r)) {
            std.log.err("GLM_NVME1107_PROTOCOL op={d} was refused; every rank must restart", .{@backingInt(op)});
            std.c._exit(75);
        }
        return r;
    }

    /// The mark hook pins an immutable state. I/O and rank commands run later on the engine thread.
    fn queue(p: *anyopaque, entry: *const pc.Entry, starts: []const u32) void {
        const o = of(p);
        if (entry.image_frame != 0 or entry.images.len != 0) return;
        const saved: *const Kept = @ptrCast(@alignCast(entry.saved));
        o.queueImage(saved.id, entry.at, entry.tokens, entry.decode_spans, starts, entry.shared, &.{});
    }

    fn queuePaged(p: *anyopaque, request: *anyopaque, kept: radix.Kept, q: radix.Query, at: u32, ids: []const u32) void {
        const req: *const glm.share.Owner = @ptrCast(@alignCast(request));
        if (!req.persist or !persistable(q)) return;
        const prompt = q.key;
        const o = of(p);
        const saved: *const Kept = @ptrCast(@alignCast(kept.saved orelse return));
        const n = @as(usize, at) + o.lookahead;
        if (n > (q.prefilled orelse q.prompt_len) or saved.below != 0) return;
        o.queueImage(saved.id, at, prompt[0..n], &.{}, &.{}, kept.shared_cut, ids);
    }

    fn queueImage(o: *Owner, id: u64, at: u32, tokens: []const u32, spans: []const [2]u32, starts: []const u32, shared: bool, ids: []const u32) void {
        if (o.stopped or at < o.local().side.store.cfg.min_tokens) return;
        o.enqueueImage(id, at, tokens, spans, starts, shared, ids) catch |err| {
            o.refused += 1;
            if (o.refused <= 4) std.log.warn("GLM NVMe checkpoint at {d} was refused: {s}", .{ at, @errorName(err) });
            return;
        };
        o.queued += 1;
    }

    fn enqueueImage(o: *Owner, id: u64, at: u32, tokens: []const u32, spans: []const [2]u32, starts: []const u32, shared: bool, ids: []const u32) !void {
        const slot = for (o.slots, 0..) |s, i| {
            if (s.id == 0) break i;
        } else return error.NvmeSourceRoom;
        const meta = try nvme.metadata.encode(o.e.gpa, at, glm.native.snap_grid, o.lookahead, spans, starts);
        defer o.e.gpa.free(meta);
        o.serial = try std.math.add(u64, o.serial, 1);
        if (o.e.share_pages) try glm.snapshots.attachTierIds(o.e, id, ids);
        errdefer if (o.e.share_pages) {
            if (o.e.snaps.getPtr(id)) |kept| glm.snapshots.clearTierIds(o.e, kept);
        };
        try o.local().side.enqueue(.{ .id = id, .at = at, .tokens = tokens, .metadata = meta, .lineage = nvme.spill.Lineage.fresh(o.boot, o.serial), .shared = shared });
        o.slots[slot] = .{ .id = id };
    }

    fn cancel(o: *Owner, i: usize) void {
        _ = o.required(.cancel, .{ .kind = 0, .a = o.slots[i].id, .top_k = @as(u32, @intCast(i)) << 1 }, &.{});
        o.slots[i] = .{};
        o.refused += 1;
    }

    fn dispatch(o: *Owner, i: usize) void {
        const source = for (o.local().side.notes.items) |p| {
            if (p.note.id == o.slots[i].id) break p.note;
        } else {
            o.cancel(i);
            return;
        };
        var h: Header = .{ .kind = 0, .a = source.id, .seed = source.at, .top_k = @as(u32, @intCast(i)) << 1 };
        if (!wire.allOk(o.command(.note_tokens, h, source.tokens))) {
            o.cancel(i);
            return;
        }
        const ids = if (o.e.share_pages)
            (o.e.snaps.get(source.id) orelse {
                o.cancel(i);
                return;
            }).tier_ids
        else
            &.{};
        const packet = wire.pagePayload(o.e.gpa, &source.lineage.root, source.metadata, ids) catch {
            o.cancel(i);
            return;
        };
        defer o.e.gpa.free(packet);
        h.depth = @intCast(source.metadata.len);
        h.top_k |= @intFromBool(source.shared);
        if (glm.disk_ranks.link) |l| if (packet.len > l.ids.len) {
            o.cancel(i);
            return;
        };
        if (!wire.allOk(o.command(.note_metadata, h, packet))) {
            o.cancel(i);
            return;
        }
        o.slots[i].sent = true;
    }

    fn pump(p: *anyopaque, budget_ns: u64) void {
        pumpMode(p, budget_ns, false);
    }

    fn pumpMode(p: *anyopaque, budget_ns: u64, wait_slots: bool) void {
        const o = of(p);
        if (o.stopped or !pending(o)) return;
        for (o.slots, 0..) |s, i| if (s.id != 0 and !s.sent) o.dispatch(i);
        if (!pending(o)) return;
        const replies = o.required(.pump, .{ .kind = 0, .seed = @min(budget_ns, 10 * std.time.ns_per_ms), .depth = @intFromBool(wait_slots) }, &.{});
        var complete: u64 = std.math.maxInt(u64);
        for (replies, 0..) |v, rank| {
            o.pump_status[rank] = v;
            complete &= glm.nvme_progress.completed(v);
        }
        for (o.slots, 0..) |s, i| if (s.id != 0 and complete & (@as(u64, 1) << @intCast(i)) != 0) {
            _ = o.required(.release, .{ .kind = 0, .a = s.id, .top_k = @as(u32, @intCast(i)) << 1 }, &.{});
            o.slots[i] = .{};
        };
    }

    fn pending(p: *anyopaque) bool {
        const o = of(p);
        if (o.stopped) return false;
        for (o.slots) |s| if (s.id != 0) return true;
        return false;
    }

    /// A mark cannot run peer work; finish its retired sources before the next backend pass begins instead.
    fn room(p: *anyopaque, bytes: u64) void {
        const o = of(p);
        if (o.stopped or bytes == 0 or bytes > (o.store orelse return).budget or !pending(o)) return;
        const retired = glm.snapshots.retiredBytes(o.e);
        if (retired == 0) return;
        const pool = if (o.e.snap_pages) |*v| v else return;
        if (pool.spare() >= bytes) return;
        o.room_waits += 1;
        const start = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
        defer o.room_ns += @intCast(@max(0, std.Io.Clock.awake.now(o.e.io).toNanoseconds() - start));
        const budget = glm.nvme_options.progressNs(retired);
        o.room_max_budget_ns = @max(o.room_max_budget_ns, budget);
        var watch = glm.nvme_progress.Watch.init(
            start,
            budget,
            glm.nvme_progress.hard_ns,
            o.pump_status[0..o.e.share.world],
        );
        while (pending(o) and pool.spare() < bytes and
            watch.reason(std.Io.Clock.awake.now(o.e.io).toNanoseconds()) == .running)
        {
            pumpMode(o, 10 * std.time.ns_per_ms, true);
            o.room_commands += 1;
            const now = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
            if (watch.observe(o.pump_status[0..o.e.share.world], now)) o.room_progress_changes += 1;
            if (pool.spare() < bytes) std.Io.sleep(o.e.io, .fromMicroseconds(100), .awake) catch {};
        }
        if (pool.spare() >= bytes) return;
        if (watch.reason(std.Io.Clock.awake.now(o.e.io).toNanoseconds()) == .absolute) o.room_hard_stops += 1;
        // Staging copies are synchronous; cancellation returns only all-rank source credit, never live DMA pages.
        for (o.slots, 0..) |s, i| if (s.id != 0) {
            const source = o.e.snaps.get(s.id) orelse continue;
            if (source.retired) {
                o.cancel(i);
                o.room_cancelled += 1;
            }
        };
    }

    fn stop(p: *anyopaque, budget_ns: u64) void {
        const o = of(p);
        if (o.stopped) return;
        const end = std.Io.Clock.awake.now(o.e.io).toNanoseconds() + budget_ns;
        while (pending(o) and std.Io.Clock.awake.now(o.e.io).toNanoseconds() < end) {
            pumpMode(o, 10 * std.time.ns_per_ms, true);
            std.Io.sleep(o.e.io, .fromMicroseconds(100), .awake) catch {};
        }
        _ = o.required(.stop, .{ .kind = 0, .seed = 0 }, &.{});
        o.slots = @splat(.{});
        o.stopped = true;
    }

    fn destroy(p: *anyopaque) void {
        const o = of(p);
        stop(o, 0);
        std.log.info("GLM_NVME1107_CLOSE queued={d} refused={d} restored={d} failed_reads={d}", .{ o.queued, o.refused, o.restored, o.failed_reads });
        std.log.info("NATIVE_TIERED_PRESSURE waits={d} wait_ns={d} cancelled={d}", .{ o.room_waits, o.room_ns, o.room_cancelled });
        std.log.info("NATIVE_TIERED_PROGRESS commands={d} max_budget_ns={d}", .{ o.room_commands, o.room_max_budget_ns });
        std.log.info("NATIVE_TIERED_WATCH progress_changes={d} hard_stops={d} hard_ns={d}", .{
            o.room_progress_changes, o.room_hard_stops, glm.nvme_progress.hard_ns,
        });
        if (o.store) |s| std.log.info("NATIVE_TIERED_PROMPT_METADATA peak={d} limit={d}", .{ s.token_pool.peak, s.token_pool.limit });
        _ = o.required(.close, .{ .kind = 0 }, &.{});
        o.e.gpa.destroy(o);
    }

    fn abortRead(o: *Owner, id: u64, damaged: ?*const [32]u8) void {
        var words: [8]u32 = undefined;
        if (damaged) |key| @memcpy(std.mem.asBytes(&words), key);
        _ = o.required(.read_abort, .{ .kind = 0, .a = id, .top_k = @intFromBool(damaged != null) }, if (damaged != null) &words else &.{});
        o.failed_reads += 1;
    }

    fn recall(p: *anyopaque, store: *pc.Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const [2]u32, have: ?*pc.Entry) ?*pc.Entry {
        const o = of(p);
        if (o.stopped or store.image_frame != 0 or prompt.len < 2) return have;
        if (store.image_query) |images| if (images.len != 0) return have;
        pump(o, std.time.ns_per_ms);
        const ahead = store.rules.lookahead;
        var max_at: u32 = @intCast(@min(prompt.len - @intFromBool(ahead == 0), std.math.maxInt(u32)));
        for (0..2) |_| {
            const hit = o.local().side.store.lookup(.{ .keyed = prompt, .max_at = max_at, .longer_than = if (have) |v| v.at + ahead else 0 }) orelse return have;
            max_at = hit.at -| 1;
            if (hit.at <= ahead) continue;
            const at = hit.at - ahead;
            if (!store.usable(at, starts) or at >= prompt.len) continue;
            const bytes = store.family.vtable.bytes(store.family.ptr, at);
            if (!pc.tier.makeRoom(store, have, prompt, born, at, bytes)) return have;
            const meta = nvme.metadata.encode(o.e.gpa, at, glm.native.snap_grid, ahead, spans, starts) catch return have;
            defer o.e.gpa.free(meta);
            const id = o.read(hit.key, at, meta, &.{}, null) orelse continue;
            const k = o.e.gpa.create(Kept) catch {
                o.abortRead(id, null);
                return have;
            };
            k.* = .{ .id = id, .at = at };
            const entry = pc.tier.adopt(store, k, prompt, born, at, spans, hit.shared) catch {
                glm.native.snap_drop(o.e, k);
                return have;
            };
            o.restored += 1;
            std.log.info("GLM_NVME1107_RESTORED at={d} bytes={d} all_ranks={d}", .{ at, entry.bytes, o.e.share.world });
            return entry;
        }
        return have;
    }

    /// Stage a checked image directly into the new pages, while the owner withholds publication.
    fn read(o: *Owner, key: [32]u8, at: u32, meta: []const u8, ids: []const u32, check: ?@import("lanes").stream.CancelCheck) ?u64 {
        const packet = wire.pagePayload(o.e.gpa, &key, meta, ids) catch return null;
        defer o.e.gpa.free(packet);
        if (glm.disk_ranks.link) |l| if (packet.len > l.ids.len) return null;
        if (o.e.next_snap >= std.math.maxInt(u32)) return null;
        const id = o.e.next_snap;
        o.e.next_snap += 1;
        const h: Header = .{ .kind = 0, .a = id, .seed = at, .depth = @intCast(meta.len) };
        if (!wire.allOk(o.command(.read_begin, h, packet))) {
            o.abortRead(id, null);
            return null;
        }
        var ready = false;
        var failed = false;
        var damaged = false;
        const end = std.Io.Clock.awake.now(o.e.io).toNanoseconds() + 60 * std.time.ns_per_s;
        while (!cancelled(check) and std.Io.Clock.awake.now(o.e.io).toNanoseconds() < end) {
            const r = o.required(.read_poll, .{ .kind = 0 }, &.{});
            ready = true;
            for (r) |v| {
                ready = ready and v == 1;
                failed = failed or v >= 2;
                damaged = damaged or v == 3;
            }
            if (ready or failed) break;
            std.Io.sleep(o.e.io, .fromMicroseconds(100), .awake) catch {};
        }
        if (!ready or cancelled(check) or !wire.allOk(o.command(.read_commit, .{ .kind = 0 }, &.{}))) {
            o.abortRead(id, if (damaged) &key else null);
            return null;
        }
        return id;
    }

    fn cancelled(check: ?@import("lanes").stream.CancelCheck) bool {
        return if (check) |c| c.check(c.ptr) else false;
    }

    fn persistable(q: radix.Query) bool {
        // The disk prefix key represents text rows with their default chunk schedule.
        return q.images.len == 0 and q.namespace == 0 and q.frame == 0 and q.starts.len == 0 and
            q.key.len == q.prompt_len;
    }

    fn recallPaged(p: *anyopaque, store: *radix.Store, request: *anyopaque, q: radix.Query) void {
        if (!persistable(q)) return;
        const prompt = q.key;
        const req: *const glm.share.Owner = @ptrCast(@alignCast(request));
        const o = of(p);
        if (o.stopped or !req.persist or prompt.len < 2 or cancelled(req.cancel_check)) return;
        pump(o, std.time.ns_per_ms);
        const ahead = o.lookahead;
        const have = if (store.find(q)) |n| n.at() + ahead else 0;
        const max_at: u32 = @intCast(@min(prompt.len - @intFromBool(ahead == 0), std.math.maxInt(u32)));
        const hit = o.local().side.store.lookup(.{ .keyed = prompt, .max_at = max_at, .longer_than = have }) orelse return;
        const at = hit.at -| ahead;
        if (at == 0 or at >= prompt.len or at % glm.native.snap_grid != 0 or store.tree.snaps >= store.limits.snaps) return;
        const pool = glm.native.share_pool(o.e) orelse return;
        const pages = at / store.rules.page;
        const compact = o.e.stateBytes(at);
        if (pool.available() < pages or (o.e.snap_pages orelse return).spare() < compact) return;
        const ids = o.e.gpa.alloc(u32, pages) catch return;
        defer o.e.gpa.free(ids);
        if (!pool.hold(ids)) return;
        defer for (ids) |id| pool.release(id);
        const meta = nvme.metadata.encode(o.e.gpa, at, glm.native.snap_grid, ahead, &.{}, &.{}) catch return;
        defer o.e.gpa.free(meta);
        const k = o.e.gpa.create(Kept) catch return;
        const id = o.read(hit.key, at, meta, ids, req.cancel_check) orelse {
            o.e.gpa.destroy(k);
            return;
        };
        k.* = .{ .id = id, .at = at };
        store.adoptPages(k, q, at, ids, hit.shared) catch {
            o.abortRead(id, null);
            o.e.gpa.destroy(k);
            return;
        };
        _ = o.required(.read_publish, .{ .kind = 0, .a = id }, &.{});
        o.restored += 1;
        std.log.info("GLM_NVME_PAGED_RESTORED at={d} pages={d} compact_bytes={d} all_ranks={d}", .{ at, pages, compact, o.e.share.world });
    }

    const vtable: pc.tier.Tier.VTable = .{ .queued = queue, .recall = recall, .pump = pump, .pending = pending, .room = room, .stop = stop, .destroy = destroy };
};

pub fn attach(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, e: *Engine, store: *pc.Store) !void {
    if (e.share.rank != 0 or store.disk != null or store.tier != null or e.share_pages) return error.NvmeTierConflict;
    if (store.rules.lookahead > 1) return error.NvmeLookahead;
    const room = @import("sys").environment.Process.get(.{}, "TF_GLM_NVME_MARK_ROOM") orelse "0";
    const room_at_mark = std.mem.eql(u8, room, "1");
    if (!room_at_mark and !std.mem.eql(u8, room, "0")) return error.BadGlmNvmeMarkRoom;
    const limits = try glm.nvme_options.read();
    const minimum = if (limits.min_cache_tokens == 0) 0 else e.stateBytes(limits.min_cache_tokens);
    glm.nvme_options.requireCache(limits, e.capacity - @import("glm_cuda").forward.window_rows, minimum, store.budget) catch |err| {
        std.log.err("Native Tiered KV Cache: {d} tokens need {d} snapshot bytes; admitted cache has {d}; adjust --prompt-cache-gib, --context or --parallel", .{ limits.min_cache_tokens, minimum, store.budget });
        return err;
    };
    const cfg: rank_state.Config = .{ .cap = try gib(o.learn_gib), .floor = try gib(o.learn_min_free_gib), .staging = limits.staging, .index = limits.index, .serial_mode = limits.mode == .serial, .lookahead = @intCast(store.rules.lookahead) };
    try wire.open(e, root, try identity(a, io, ctx, e, o), cfg);
    errdefer {
        var reply: [8]u64 = undefined;
        _ = wire.invoke(e, .close, .{ .kind = 0 }, &.{}, &reply) catch {};
    }
    const owner = try e.gpa.create(Owner);
    owner.* = .{ .e = e, .ctx = ctx, .store = store, .lookahead = store.rules.lookahead, .boot = undefined };
    io.random(&owner.boot);
    store.tier = .{ .ptr = owner, .vtable = &Owner.vtable };
    store.room_at_mark = room_at_mark;
    if (room_at_mark) std.log.info("GLM NVMe snapshot room is prepared at the prompt mark", .{});
    std.log.info("GLM_NVME1107_ACTIVE ranks={d} root={s} format=3 staging_mib={d} index_mib={d} min_tokens=8192 shared_history=1", .{ e.share.world, root, cfg.staging >> 20, cfg.index >> 20 });
}

pub fn attachPaged(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, e: *Engine, adapter: *page_adapter.Adapter) !void {
    if (e.share.rank != 0 or !e.share_pages or adapter.tier != null or !adapter.store.on()) return error.NvmeTierConflict;
    const ahead = adapter.store.rules.lookahead;
    if (ahead > 1) return error.NvmeLookahead;
    const limits = try glm.nvme_options.read();
    const minimum = if (limits.min_cache_tokens == 0) 0 else e.stateBytes(limits.min_cache_tokens / glm.native.snap_grid * glm.native.snap_grid);
    const pool = if (e.snap_pages) |*v| v else return error.NoSnapshotCache;
    try glm.nvme_options.requireCache(limits, e.capacity - glm.forward.window_rows, minimum, pool.spare());
    const cfg: rank_state.Config = .{ .cap = try gib(o.learn_gib), .floor = try gib(o.learn_min_free_gib), .staging = limits.staging, .index = limits.index, .serial_mode = limits.mode == .serial, .lookahead = @intCast(ahead) };
    try wire.open(e, root, try identity(a, io, ctx, e, o), cfg);
    errdefer {
        var reply: [8]u64 = undefined;
        _ = wire.invoke(e, .close, .{ .kind = 0 }, &.{}, &reply) catch {};
    }
    const owner = try e.gpa.create(Owner);
    owner.* = .{ .e = e, .ctx = ctx, .lookahead = ahead, .boot = undefined };
    io.random(&owner.boot);
    adapter.tier = .{ .ptr = owner, .queue = Owner.queuePaged, .recall = Owner.recallPaged, .destroy = Owner.destroy, .work = .{ .ptr = owner, .pending = Owner.pending, .pump = Owner.pump, .stop = Owner.stop } };
    std.log.info("GLM_NVME_PAGED_ACTIVE ranks={d} root={s} staging_mib={d} index_mib={d}", .{ e.share.world, root, cfg.staging >> 20, cfg.index >> 20 });
}
