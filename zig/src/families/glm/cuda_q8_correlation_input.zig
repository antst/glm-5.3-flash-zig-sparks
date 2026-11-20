//! One startup-owned compact correlation array, bound to checkpoint and prepared linear ordinals.
const std = @import("std");
const moments = @import("cuda_q8_moments.zig");
pub const Identity = moments.Identity;
pub const Entry = struct { n: usize, k: usize, offset: usize, samples: u64 };
const Rank = struct {
    rank: u32,
    covariance: []const u8,
    covariance_sha256: []const u8,
    diagonal: []const u8,
    diagonal_sha256: []const u8,
};
const Manifest = struct { version: u32, world: u32, checkpoint: []const u8, ranks: []const Rank };

pub const Data = struct {
    gpa: std.mem.Allocator,
    raw: []align(16) u8,
    entries: []Entry,
    values: []const f64,
    target_count: usize,

    pub fn destroy(self: *Data) void {
        const gpa = self.gpa;
        gpa.free(self.entries);
        gpa.free(self.raw);
        gpa.destroy(self);
    }

    pub fn linear(self: *const Data, ordinal: usize, n: usize, k: usize) ![]const f64 {
        if (ordinal >= self.entries.len) return error.Q8CorrelationShape;
        const e = self.entries[ordinal];
        if (e.n != n or e.k != k) return error.Q8CorrelationShape;
        return self.values[e.offset..][0 .. e.k / 64 * 2080];
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, root: []const u8, id: Identity, rank: usize) !*Data {
        if (rank >= 2 or root.len == 0 or root.len > 4096 or !std.fs.path.isAbsolute(root))
            return error.Q8CorrelationIdentity;
        const path = try std.fs.path.join(gpa, &.{ root, "manifest.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        if (!std.mem.eql(u8, &hash(text), &id.manifest)) return error.Q8CorrelationIdentity;
        const parsed = try std.json.parseFromSlice(Manifest, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const m = parsed.value;
        if (m.version != 1 or m.world != 2 or m.ranks.len != 2 or
            !std.mem.eql(u8, &(try digest(m.checkpoint)), &id.checkpoint)) return error.Q8CorrelationIdentity;
        for (m.ranks, 0..) |r, i| {
            if (r.rank != i or !std.mem.eql(u8, r.covariance, if (i == 0) "rank-0.covariance.bin" else "rank-1.covariance.bin") or
                !std.mem.eql(u8, r.diagonal, if (i == 0) "rank-0.diagonal.bin" else "rank-1.diagonal.bin"))
                return error.Q8CorrelationIdentity;
        }
        const file = try std.fs.path.join(gpa, &.{ root, m.ranks[rank].covariance });
        defer gpa.free(file);
        const raw = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(256 << 20), .@"16", null);
        errdefer gpa.free(raw);
        if (!std.mem.eql(u8, &hash(raw), &(try digest(m.ranks[rank].covariance_sha256))))
            return error.Q8CorrelationIdentity;
        const companion = try std.fs.path.join(gpa, &.{ root, m.ranks[rank].diagonal });
        defer gpa.free(companion);
        const diagonal = try std.Io.Dir.cwd().readFileAlloc(io, companion, gpa, .limited(40 << 20));
        defer gpa.free(diagonal);
        if (!std.mem.eql(u8, &hash(diagonal), &(try digest(m.ranks[rank].diagonal_sha256))))
            return error.Q8CorrelationIdentity;
        return adopt(gpa, raw, diagonal, id.checkpoint, rank);
    }
};

pub fn validateMap(data: *const Data, list: anytype, head_bf16: bool) !void {
    const head: usize = @intFromBool(head_bf16);
    if (list.len < head) return error.Q8CorrelationShape;
    const targets = list.len - head;
    if (data.target_count != targets or data.entries.len != targets + 1) return error.Q8CorrelationShape;
    for (list, 0..) |linear, i| _ = try data.linear(i, linear.n, linear.k);
}

pub fn adopt(gpa: std.mem.Allocator, raw: []align(16) u8, diagonal: []const u8, checkpoint: [32]u8, rank: usize) !*Data {
    if (@import("builtin").cpu.arch.endian() != .little) return error.Q8CorrelationIdentity;
    if (raw.len < 96 or raw.len > 256 << 20 or diagonal.len < 96 or diagonal.len > 40 << 20 or
        !std.mem.eql(u8, raw[0..8], "GLMCOV01") or !std.mem.eql(u8, diagonal[0..8], "GLMCAL01") or
        read(u32, raw, 8) != 1 or rank >= 2 or read(u32, raw, 12) != rank or read(u32, raw, 16) != 2 or
        !std.mem.eql(u8, raw[48..80], &checkpoint) or !std.mem.eql(u8, raw[8..24], diagonal[8..24]) or
        !std.mem.eql(u8, raw[32..80], diagonal[32..80]) or !std.mem.allEqual(u8, diagonal[80..96], 0))
        return error.Q8CorrelationIdentity;
    const count: usize = read(u32, raw, 20);
    const target: usize = read(u32, raw, 32);
    const max_k: usize = read(u32, raw, 40);
    const features = std.math.cast(usize, read(u64, raw, 80)) orelse return error.Q8CorrelationShape;
    const values_count = std.math.cast(usize, read(u64, raw, 24)) orelse return error.Q8CorrelationShape;
    if (count == 0 or count > 2048 or target != count - 1 or features == 0 or features > 4 << 20 or
        features % 64 != 0 or values_count != features / 64 * 2080 or
        read(u64, diagonal, 24) != features or read(u32, raw, 36) != 32 or read(u32, raw, 44) != 40 or
        read(u32, raw, 88) != 64 or read(u32, raw, 92) != 2080 or max_k == 0 or max_k > 524288)
        return error.Q8CorrelationShape;
    const payload = 96 + count * 40;
    if (raw.len != payload + values_count * 8 or diagonal.len != payload + features * 8)
        return error.Q8CorrelationShape;
    const entries = try gpa.alloc(Entry, count);
    errdefer gpa.free(entries);
    const values = std.mem.bytesAsSlice(f64, @as([]align(8) const u8, @alignCast(raw[payload..])));
    var offset: usize = 0;
    var seen_max: usize = 0;
    for (entries, 0..) |*e, i| {
        const at = 96 + i * 40;
        const n: usize = read(u32, raw, at + 8);
        const k: usize = read(u32, raw, at + 12);
        const calls = read(u64, raw, at + 16);
        const samples = read(u64, raw, at + 24);
        const limit = std.math.mul(u64, calls, 32) catch return error.Q8CorrelationShape;
        if (read(u32, raw, at) != (if (i < target) i else 0) or
            read(u32, raw, at + 4) != @as(u32, if (i < target) 0 else 2) or
            n == 0 or k == 0 or k % 64 != 0 or k > max_k or k > features -| offset or
            calls == 0 or samples < calls or samples > limit or
            !std.mem.eql(u8, raw[at..][0..32], diagonal[at..][0..32]) or
            read(u64, raw, at + 32) != offset / 64 * 2080 * 8 or read(u64, diagonal, at + 32) != offset * 8)
            return error.Q8CorrelationShape;
        e.* = .{ .n = n, .k = k, .offset = offset / 64 * 2080, .samples = samples };
        for (0..k) |j| {
            const g = (offset + j) / 64 * 2080 + triangle(j % 64, j % 64);
            if (@as(u64, @bitCast(values[g])) != read(u64, diagonal, payload + (offset + j) * 8))
                return error.Q8CorrelationDiagonal;
        }
        offset += k;
        seen_max = @max(seen_max, k);
    }
    if (offset != features or seen_max != max_k) return error.Q8CorrelationShape;
    for (0..features / 64) |group| {
        const block = values[group * 2080 ..][0..2080];
        for (0..64) |i| {
            const a = block[triangle(i, i)];
            if (!std.math.isFinite(a) or a < 0) return error.Q8CorrelationMoment;
            for (i..64) |j| {
                const b = block[triangle(j, j)];
                const v = block[triangle(i, j)];
                if (!std.math.isFinite(b) or b < 0 or !std.math.isFinite(v) or
                    @abs(v) > @sqrt(a) * @sqrt(b) * (1 + 3e-12)) return error.Q8CorrelationMoment;
            }
        }
    }
    const data = try gpa.create(Data);
    data.* = .{ .gpa = gpa, .raw = raw, .entries = entries, .values = values, .target_count = target };
    return data;
}

pub fn triangle(i: usize, j: usize) usize {
    std.debug.assert(i <= j and j < 64);
    return i * (129 - i) / 2 + j - i;
}

fn digest(text: []const u8) ![32]u8 {
    return moments.digest(text) catch return error.Q8CorrelationIdentity;
}

fn read(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

fn hash(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
