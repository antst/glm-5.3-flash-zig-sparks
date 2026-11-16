//! GLM's local-rank image operations for the bounded 1107 store; no peer calls occur in these callbacks.
const std = @import("std");
const tier = @import("core").nvme1107.tier_state;
const image = @import("cuda_disk_image.zig");
const ranks = @import("cuda_disk_ranks.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = @import("cuda_engine.zig").Engine;

fn engine(p: *anyopaque) *Engine {
    return @ptrCast(@alignCast(p));
}

pub fn of(e: *Engine) tier.Image {
    return .{ .ptr = e, .parts = parts, .take = take, .put = put, .pin = pin, .unpin = unpin, .blank = blank, .seal = seal, .drop = drop };
}

fn parts(p: *anyopaque, at: u32, out: *std.ArrayList(tier.Part), a: std.mem.Allocator) !void {
    const e = engine(p);
    if (e.share_pages) return error.NvmeSharedPages;
    var ps: std.ArrayList(image.Part) = .empty;
    defer ps.deinit(a);
    try image.nvmeParts(try ranks.shape(e), at, &ps, a);
    for (ps.items) |v| try out.append(a, .{ .bytes = v.bytes, .row = v.row, .per = v.per });
}

fn take(p: *anyopaque, id: u64, ranges: []const tier.Range, out: []u8) !void {
    var done: usize = 0;
    for (ranges) |r| {
        if (r.len > out.len - done) return error.NvmeImageRange;
        try ranks.copy(engine(p), id, r.off, out[done..][0..@intCast(r.len)], true);
        done += @intCast(r.len);
    }
    if (done != out.len) return error.NvmeImageRange;
}

fn put(p: *anyopaque, id: u64, ranges: []const tier.Range, in: []const u8) !void {
    var done: usize = 0;
    for (ranges) |r| {
        if (r.len > in.len - done) return error.NvmeImageRange;
        try ranks.copy(engine(p), id, r.off, @constCast(in[done..][0..@intCast(r.len)]), false);
        done += @intCast(r.len);
    }
    if (done != in.len) return error.NvmeImageRange;
}

fn pin(p: *anyopaque, id: u64) !void {
    try snapshots.pin(engine(p), id);
}
fn unpin(p: *anyopaque, id: u64) void {
    snapshots.unpin(engine(p), id);
}
fn blank(p: *anyopaque, id: u64, at: u32) !void {
    try snapshots.blankAs(engine(p), id, at);
}
fn drop(p: *anyopaque, id: u64) void {
    snapshots.drop(engine(p), id);
}

fn seal(p: *anyopaque, id: u64, small: []const u8) !void {
    const e = engine(p);
    const s = e.snaps.get(id) orelse return error.NoKeptState;
    if (small.len < image.Words.bytes) return error.DiskWords;
    var words: [image.Words.count]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&words), small[0..image.Words.bytes]);
    const w = try ranks.decode(e, words, @intCast(s.pos));
    try snapshots.seal(e, id, w.mtp_len, .{ w.base, w.mtp_base }, w.cur);
    try snapshots.sealRing(e, id, w.ring_from);
}
