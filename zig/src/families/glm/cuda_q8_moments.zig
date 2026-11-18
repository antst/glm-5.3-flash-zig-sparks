//! Bounded startup input statistics, bound to checkpoint, world, rank and prepared-linear geometry.
const std = @import("std");

pub const Entry = struct { n: usize, k: usize, offset: usize };
pub const Identity = struct { manifest: [32]u8, checkpoint: [32]u8 };
const Rank = struct { rank: u32, file: []const u8, bytes: usize, sha256: []const u8 };
const Manifest = struct {
    schema: []const u8,
    world: u32,
    checkpoint: []const u8,
    entries: usize,
    target_count: usize,
    features: usize,
    ranks: []const Rank,
};

pub const Data = struct {
    gpa: std.mem.Allocator,
    entries: []Entry,
    means: []f64,
    target_count: usize,

    pub fn destroy(self: *Data) void {
        const gpa = self.gpa;
        gpa.free(self.entries);
        gpa.free(self.means);
        gpa.destroy(self);
    }

    pub fn linear(self: *const Data, ordinal: usize, n: usize, k: usize) ![]const f64 {
        if (ordinal >= self.entries.len) return error.Q8CalibrationShape;
        const e = self.entries[ordinal];
        if (e.n != n or e.k != k) return error.Q8CalibrationShape;
        return self.means[e.offset..][0..e.k];
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, root: []const u8, id: Identity, rank: usize) !*Data {
        if (rank >= 2 or root.len == 0 or root.len > 4096 or !std.fs.path.isAbsolute(root))
            return error.Q8CalibrationIdentity;
        const path = try std.fs.path.join(gpa, &.{ root, "manifest.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 10));
        defer gpa.free(text);
        if (!std.mem.eql(u8, &hash(text), &id.manifest)) return error.Q8CalibrationIdentity;
        const parsed = try std.json.parseFromSlice(Manifest, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const m = parsed.value;
        if (!std.mem.eql(u8, m.schema, "glm-input-moments/v1") or m.world != 2 or m.ranks.len != 2 or
            m.entries == 0 or m.entries > 2048 or m.target_count != m.entries - 1 or
            m.features == 0 or m.features > 4 << 20 or
            !std.mem.eql(u8, &(try digest(m.checkpoint)), &id.checkpoint)) return error.Q8CalibrationIdentity;
        for (m.ranks, 0..) |r, i| {
            if (r.rank != i or r.bytes < 96 or r.bytes > 40 << 20 or
                !std.mem.eql(u8, r.file, if (i == 0) "rank-0.bin" else "rank-1.bin"))
                return error.Q8CalibrationIdentity;
        }
        const file = try std.fs.path.join(gpa, &.{ root, m.ranks[rank].file });
        defer gpa.free(file);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, gpa, .limited(40 << 20));
        defer gpa.free(bytes);
        if (bytes.len != m.ranks[rank].bytes or
            !std.mem.eql(u8, &hash(bytes), &(try digest(m.ranks[rank].sha256)))) return error.Q8CalibrationIdentity;
        const data = try decode(gpa, bytes, id.checkpoint, rank);
        errdefer data.destroy();
        if (data.entries.len != m.entries or data.target_count != m.target_count or data.means.len != m.features)
            return error.Q8CalibrationShape;
        return data;
    }
};

pub fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.Q8CalibrationIdentity;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.Q8CalibrationIdentity;
    return out;
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, checkpoint: [32]u8, rank: usize) !*Data {
    if (bytes.len < 96 or bytes.len > 40 << 20 or !std.mem.eql(u8, bytes[0..8], "GLMCAL01") or
        read(u32, bytes, 8) != 1 or read(u32, bytes, 12) != rank or rank >= 2 or read(u32, bytes, 16) != 2 or
        !std.mem.eql(u8, bytes[48..80], &checkpoint) or !std.mem.allEqual(u8, bytes[80..96], 0))
        return error.Q8CalibrationIdentity;
    const count: usize = read(u32, bytes, 20);
    const features: usize = std.math.cast(usize, read(u64, bytes, 24)) orelse return error.Q8CalibrationShape;
    const target: usize = read(u32, bytes, 32);
    const max_k: usize = read(u32, bytes, 40);
    if (count == 0 or count > 2048 or target != count - 1 or features == 0 or features > 4 << 20 or
        read(u32, bytes, 36) != 32 or max_k == 0 or max_k > 524288 or read(u32, bytes, 44) != 40)
        return error.Q8CalibrationShape;
    const payload = 96 + count * 40;
    if (bytes.len != payload + features * 8) return error.Q8CalibrationShape;
    const entries = try gpa.alloc(Entry, count);
    errdefer gpa.free(entries);
    const means = try gpa.alloc(f64, features);
    errdefer gpa.free(means);
    var offset: usize = 0;
    for (entries, 0..) |*e, i| {
        const at = 96 + i * 40;
        const ordinal = read(u32, bytes, at);
        const role = read(u32, bytes, at + 4);
        const n: usize = read(u32, bytes, at + 8);
        const k: usize = read(u32, bytes, at + 12);
        const calls = read(u64, bytes, at + 16);
        const samples = read(u64, bytes, at + 24);
        const limit = std.math.mul(u64, calls, 32) catch return error.Q8CalibrationShape;
        if (role != @as(u32, if (i < target) 0 else 2) or ordinal != (if (i < target) i else 0) or
            n == 0 or k == 0 or k % 64 != 0 or k > max_k or k > features -| offset or
            calls == 0 or samples < calls or samples > limit or read(u64, bytes, at + 32) != offset * 8)
            return error.Q8CalibrationShape;
        for (means[offset..][0..k], 0..) |*v, j| {
            const sum: f64 = @bitCast(read(u64, bytes, payload + (offset + j) * 8));
            if (!std.math.isFinite(sum) or sum < 0) return error.Q8CalibrationMoment;
            v.* = sum / @as(f64, @floatFromInt(samples));
        }
        e.* = .{ .n = n, .k = k, .offset = offset };
        offset += k;
    }
    if (offset != features) return error.Q8CalibrationShape;
    const data = try gpa.create(Data);
    data.* = .{ .gpa = gpa, .entries = entries, .means = means, .target_count = target };
    return data;
}

fn read(comptime T: type, bytes: []const u8, at: usize) T {
    return std.mem.readInt(T, bytes[at..][0..@sizeOf(T)], .little);
}

fn hash(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}
