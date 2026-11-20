//! Optional fixed-scale Q8 startup rounding; inference consumes the unchanged signed-byte layout.
const std = @import("std");
const cuda = @import("cuda");

pub const image_sha = "767bcdc9b11a921ca23be9d6e99cd53c6718c919e3595a77842e84591c30829b";
pub const View = struct { ptr: u64, len: usize };
pub const Buffers = struct {
    source: View,
    covariance: View,
    words: View,
    scales: View,
    status: View,
    errors: ?View = null,
};
pub const Owner = struct {
    module: cuda.Module,
    function: cuda.Function,

    pub fn deinit(self: *Owner) void {
        self.module.unload();
        self.* = undefined;
    }
};

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, on: bool) !?Owner {
    if (!on) return null;
    const path = @import("sys").environment.Process.get(.{}, "TF_GLM_Q8_CORRELATED_IMAGE") orelse
        return error.Q8CorrelatedImageRequired;
    return try loadPath(gpa, io, d, path);
}

pub fn loadPath(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8) !Owner {
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(16 << 20), .@"16", null);
    defer gpa.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), image_sha)) return error.Q8CorrelatedImageHash;
    var module = try cuda.Module.load(d, bytes);
    errdefer module.unload();
    return .{ .module = module, .function = try module.function("tf_glm_q8_correlated_v1") };
}

fn fits(v: View, bytes: usize, alignment: u64) bool {
    return v.ptr != 0 and v.ptr % alignment == 0 and v.len >= bytes and
        bytes <= std.math.maxInt(u64) - v.ptr;
}

fn overlap(a: View, b: View) bool {
    return if (a.ptr <= b.ptr) b.ptr - a.ptr < a.len else a.ptr - b.ptr < b.len;
}

pub fn validate(n: usize, k: usize, b: Buffers) !usize {
    if (n == 0 or n > 65535 * 4 or k == 0 or k % 64 != 0) return error.Q8CorrelatedShape;
    const count = try std.math.mul(usize, n, k);
    const groups = count / 64;
    if (groups > std.math.maxInt(u32)) return error.Q8CorrelatedShape;
    const covariance_bytes = try std.math.mul(usize, k / 64, 2080 * 8);
    if (covariance_bytes > 256 << 20) return error.Q8CorrelatedCapacity;
    const sizes = [_]usize{
        try std.math.mul(usize, count, 2),
        covariance_bytes,
        count,
        try std.math.mul(usize, groups, 2),
        16,
        if (b.errors != null) try std.math.mul(usize, groups, 16) else 0,
    };
    const views = [_]View{ b.source, b.covariance, b.words, b.scales, b.status, b.errors orelse .{ .ptr = 0, .len = 0 } };
    const alignments = [_]u64{ 2, 8, 4, 2, 4, 8 };
    for (views, sizes, alignments) |v, bytes, alignment| {
        if (bytes != 0 and !fits(v, bytes, alignment)) return error.Q8CorrelatedCapacity;
    }
    for (views, sizes, 0..) |v, bytes, i| {
        if (bytes == 0) continue;
        for (views[i + 1 ..], sizes[i + 1 ..]) |other, other_bytes| {
            if (other_bytes != 0 and overlap(.{ .ptr = v.ptr, .len = bytes }, .{ .ptr = other.ptr, .len = other_bytes }))
                return error.Q8CorrelatedAlias;
        }
    }
    return groups;
}

pub fn launch(f: cuda.Function, stream: cuda.Stream, n: usize, k: usize, b: Buffers) !void {
    const groups = try validate(n, k, b);
    var args: cuda.Args = .{};
    inline for (.{ b.source.ptr, b.covariance.ptr, @as(i64, @intCast(groups)), @as(i64, @intCast(k / 64)), b.words.ptr, b.scales.ptr, b.status.ptr, if (b.errors) |v| v.ptr else @as(u64, 0) }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(k / 64), .y = @intCast((n + 3) / 4) }, .block = .{ .x = 128 } }, stream, &args);
}

pub fn refine(f: cuda.Function, stream: cuda.Stream, n: usize, k: usize, data: anytype, ordinal: usize, buffer: cuda.DeviceBuffer, outputs: Buffers) !void {
    const values = try data.linear(ordinal, n, k);
    var b = outputs;
    b.covariance = .{ .ptr = buffer.ptr, .len = values.len * 8 };
    if (b.covariance.len > buffer.len) return error.Q8CorrelatedCapacity;
    _ = try validate(n, k, b);
    try buffer.upload(0, std.mem.sliceAsBytes(values));
    try launch(f, stream, n, k, b);
}
