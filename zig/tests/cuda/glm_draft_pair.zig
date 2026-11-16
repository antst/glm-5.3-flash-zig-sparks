//! Explicit same-token-prefix DFlash snapshot; no serving path imports this diagnostic.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");

const Case = struct { name: []const u8, prefix_ids: []const u32, anchor: u32, streams: usize, continuation: usize = 7 };
const Fixture = struct { cases: []const Case, target_kv: enum { bf16, fp8 } = .bf16 };
const Record = struct {
    name: []const u8,
    prefix_ids: []const u32,
    anchor: u32,
    streams: usize,
    target_firsts: []const u32,
    tap_hashes: []const [64]u8,
    proposals: []const [7]u32,
    probabilities: []const [7]f64,
    candidate_words: []const i32,
    projection_bits: []const u32,
    target_greedy_after_anchor: [7]u32,
    target_continuation_ids: []const u32,
};

fn dump(gpu: check.Gpu, a: std.mem.Allocator, base: []const u8, name: []const u8, src: u64, size: usize) !void {
    const bytes = try a.alloc(u8, size);
    try cuda.memory.download(gpu.d, src, bytes);
    const path = try std.fmt.allocPrint(a, "{s}.{s}.bin", .{ base, name });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = bytes });
}

fn headPack(gpu: check.Gpu, a: std.mem.Allocator, base: []const u8, head: anytype) !void {
    const common: usize = 38720;
    if (head.n < common or head.k != 4096 or head.npad < head.n) return error.BadHeadTrace;
    var hashes: [3][64]u8 = undefined;
    for ([_]u64{ head.w, head.s, head.b }, 0..) |ptr, i| {
        const size = if (i == 0) common * head.k / 2 else head.npad * (head.k / 64) * 2;
        const bytes = try a.alloc(u8, size);
        try cuda.memory.download(gpu.d, ptr, bytes);
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        if (i == 0) h.update(bytes) else for (0..head.k / 64) |g| h.update(bytes[g * head.npad * 2 ..][0 .. common * 2]);
        var digest: [32]u8 = undefined;
        h.final(&digest);
        hashes[i] = std.fmt.bytesToHex(digest, .lower);
    }
    const data = try std.json.Stringify.valueAlloc(a, .{
        .n = head.n,
        .k = head.k,
        .npad = head.npad,
        .common_rows = common,
        .word_sha256 = hashes[0],
        .scale_sha256 = hashes[1],
        .bias_sha256 = hashes[2],
    }, .{ .emit_strings_as_arrays = false });
    const path = try std.fmt.allocPrint(a, "{s}.head-pack.json", .{base});
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = data });
}

/// MODEL DRAFTER RANK FIXTURE OUTPUT MODE; all ranks agree on the fixture and diagnostic mode.
pub fn run(gpu: check.Gpu, rest: []const []const u8) !void {
    if (rest.len < 6 or rest.len > 8) return error.MissingArgument;
    const compat = try std.fmt.parseInt(u8, rest[5], 10);
    if (compat != 0 and compat != 6 and compat != 7) return error.BadDrafterCompatibility;
    const rank = try std.fmt.parseInt(usize, rest[2], 10);
    if (rank >= 4) return error.BadWorld;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[3], a, .limited(1 << 20));
    const fixture = try std.json.parseFromSliceLeaky(Fixture, a, bytes, .{ .ignore_unknown_fields = true });
    if (fixture.cases.len == 0 or fixture.cases.len > 16) return error.BadFixture;
    for (fixture.cases) |c| {
        if (c.prefix_ids.len == 0 or c.prefix_ids.len > 512 or (c.streams != 1 and c.streams != 8)) return error.BadFixture;
        if (c.continuation < 7 or c.continuation > 400 or (c.continuation > 7 and c.streams != 1)) return error.BadFixture;
        for (c.prefix_ids) |id| if (id >= 154880) return error.BadFixture;
        if (c.anchor >= 154880) return error.BadFixture;
    }
    var problem: []const u8 = "";
    var settings = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    settings.kv8 = fixture.target_kv == .fp8;
    if (settings.verify_graphs or settings.pages or settings.kv_host != 0) return error.DiagnosticScope;
    var group = try cuda.tp_group.Group.open(gpu.d, rank, 4);
    defer group.close();
    var handshake = try cuda.Stream.init(gpu.d, false);
    defer handshake.deinit();
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    var schedule: [32]u8 = undefined;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&digest);
    hasher.update(&.{compat});
    var replacement: ?[]align(8) u8 = null;
    if (rest.len >= 7) {
        const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[6], a, .limited(64 << 20));
        replacement = try a.alignedAlloc(u8, .@"8", raw.len);
        @memcpy(replacement.?, raw);
        hasher.update(raw);
    }
    var tensor_core: ?[]align(8) u8 = null;
    if (rest.len == 8) {
        const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[7], a, .limited(64 << 20));
        tensor_core = try a.alignedAlloc(u8, .@"8", raw.len);
        @memcpy(tensor_core.?, raw);
        hasher.update(raw);
    }
    hasher.final(&schedule);
    var agreed: [4][32]u8 = undefined;
    try group.gather([32]u8, gpu.d, handshake, schedule, &agreed);
    for (agreed) |d| if (!std.mem.eql(u8, &d, &schedule)) return error.FixtureDisagreement;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, rest[0], &group.lib, group.comm, rank, 4, .{
        .capacity = 4096,
        .mtp = true,
        .settings = settings,
        .drafter = true,
    });
    defer e.deinit();
    const borrowed = try glm.comm1107.open(gpu.gpa, gpu.io, gpu.d, &group, e.stream, settings.roce_multi);
    if (!borrowed) return error.BorrowedWireRequired;
    e.group = &group;
    e.drafter = try glm.dflash_load.loadMode(gpu.gpa, gpu.io, e, rest[1], @intCast(rank), 4, 4, compat);
    const dr = e.drafter.?;
    defer e.stream.synchronize() catch {};
    if (replacement) |data| {
        try e.stream.synchronize();
        var module = try cuda.Module.load(gpu.d, data);
        errdefer module.unload();
        const kernels = try glm.dflash.Kernels.load(module);
        const attention: ?glm.dflash_attention.Kernels = if (dr.attention != null)
            try glm.dflash_attention.Kernels.load(module)
        else
            null;
        dr.module.unload();
        dr.module = module;
        dr.k = kernels;
        if (attention) |k| dr.attention.?.kernels = k;
    }
    if (tensor_core) |data| {
        var module = try cuda.Module.load(gpu.d, data);
        errdefer module.unload();
        dr.python_attention = try glm.dflash.PythonAttention.load(module, dr.module);
    }
    if (rank == 0) try headPack(gpu, a, rest[4], e.w.mtp.?.draft);
    var records: std.ArrayList(Record) = .empty;
    for (fixture.cases) |c| {
        const firsts = try a.alloc(u32, c.streams);
        const hashes = try a.alloc([64]u8, c.streams);
        const taps = try a.alloc(u8, c.prefix_ids.len * glm.dflash.TAP_ROW);
        for (0..c.streams) |slot| {
            try e.bind(@intCast(slot));
            try e.begin(null);
            firsts[slot] = try e.prefill(c.prefix_ids, null, null, null);
            try e.stream.synchronize();
            try cuda.memory.download(gpu.d, dr.taps_prefill, taps);
            var h: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(taps, &h, .{});
            hashes[slot] = std.fmt.bytesToHex(h, .lower);
            if (slot == 0) {
                const path = try std.fmt.allocPrint(a, "{s}.{s}.taps.bf16", .{ rest[4], c.name });
                try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = taps });
            }
        }
        var ids: [8][7]u32 = undefined;
        var probs: [8][7]f64 = undefined;
        var reqs: [8]glm.dflash.Req = undefined;
        for (0..c.streams) |slot| reqs[slot] = .{
            .slot = @intCast(slot),
            .pos = @intCast(c.prefix_ids.len),
            .n = 0,
            .anchor = c.anchor,
            .out = if (rank == 0) &ids[slot] else null,
            .probs = if (rank == 0) &probs[slot] else null,
        };
        try e.draftBlock(reqs[0..c.streams]);
        try e.stream.synchronize();
        try group.check();
        const M = c.streams * glm.dflash.BLOCK;
        if (rank == 0 and c.streams == 1) {
            const base = try std.fmt.allocPrint(a, "{s}.{s}", .{ rest[4], c.name });
            try dump(gpu, a, base, "fc", dr.c1, c.prefix_ids.len * 4096 * 2);
            try dump(gpu, a, base, "context", dr.ctx, c.prefix_ids.len * 4096 * 2);
            try dump(gpu, a, base, "hidden", dr.x, M * 4096 * 2);
            try dump(gpu, a, base, "normed", dr.hn, M * 4096 * 2);
            try dump(gpu, a, base, "sums", dr.xs, M * 64 * 4);
            try dump(gpu, a, base, "logits", dr.logits, M * e.w.mtp.?.draft.n * 2);
        }
        // One fixed BF16 target history is the oracle for both runtimes' raw drafts.
        try e.bind(0);
        const continuation = try a.alloc(u32, c.continuation);
        var reference: [7]u32 = undefined;
        var pending = c.anchor;
        for (continuation) |*token| {
            var next: [1]u32 = undefined;
            try e.verify(&.{pending}, &next, 0);
            try e.keep(1);
            token.* = next[0];
            pending = next[0];
        }
        @memcpy(&reference, continuation[0..7]);
        try records.append(a, .{
            .name = c.name,
            .target_greedy_after_anchor = reference,
            .target_continuation_ids = continuation,
            .prefix_ids = c.prefix_ids,
            .anchor = c.anchor,
            .streams = c.streams,
            .target_firsts = firsts,
            .tap_hashes = hashes,
            .proposals = if (rank == 0) try a.dupe([7]u32, ids[0..c.streams]) else &.{},
            .probabilities = if (rank == 0) try a.dupe([7]f64, probs[0..c.streams]) else &.{},
            .candidate_words = if (rank == 0) try a.dupe(i32, dr.cand[0 .. 4 * M * glm.dflash.CW]) else &.{},
            .projection_bits = if (rank == 0) try a.dupe(u32, @as([]const u32, @ptrCast(dr.gh[0 .. M * glm.dflash.RANK]))) else &.{},
        });
    }
    const text = try std.json.Stringify.valueAlloc(a, .{
        .schema = "glm-same-prefix-draft-v1",
        .runtime = "zig",
        .target_kv = @tagName(fixture.target_kv),
        .rank = rank,
        .world = 4,
        .fixture_sha256 = std.fmt.bytesToHex(digest, .lower),
        .block_rows = glm.dflash.BLOCK,
        .selector_rank = glm.dflash.RANK,
        .candidate_words_per_row = glm.dflash.CW,
        .selector_edge06 = dr.edge_06,
        .selector_temperature = dr.temp,
        .borrowed_wire = borrowed,
        .diagnostic_compat_mask = compat,
        .diagnostic_module = if (rest.len >= 7) rest[6] else "embedded",
        .diagnostic_tensor_core = tensor_core != null,
        .scope = "Fixed prompt IDs and forced anchor; raw DFlash before cutoff, allocator and copies",
        .cases = records.items,
    }, .{ .emit_strings_as_arrays = false });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = rest[4], .data = text });
    check.pass("same-prefix draft snapshot: rank {d}, {d} cases", .{ rank, records.items.len });
}
