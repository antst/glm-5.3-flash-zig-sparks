//! GLM request-local copy selection; the native planner still caps, prices and verifies every proposal.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const Proposer = lanes.proposer.Proposer;

pub fn factory(on: bool, ctx: *anyopaque) @FieldType(api.LaneHost, "proposers") {
    if (!on) return null;
    std.log.info("GLM_COPY_POLICY_75C match=8 reply_match=16 miss_max=0 padding=false priced=false", .{});
    return .{ .ctx = ctx, .make = make, .free = free };
}

fn make(_: *anyopaque, gpa: std.mem.Allocator, prompt: []const u32, _: []const u32) !Proposer {
    const p = try gpa.create(Copy);
    p.* = .{ .gpa = gpa, .prompt = prompt.len };
    return p.proposer();
}
fn free(_: *anyopaque, p: Proposer) void {
    const c: *Copy = @ptrCast(@alignCast(p.ptr));
    c.deinit();
    c.gpa.destroy(c);
}

pub const Copy = struct {
    gpa: std.mem.Allocator,
    prompt: usize,
    out: std.ArrayList(u32) = .empty,
    index: std.AutoHashMapUnmanaged([8]u32, std.ArrayList(u32)) = .empty,
    indexed: usize = 0,
    last_key: [8]u32 = undefined,
    last_match: i64 = 0,
    last_end: usize = 0,

    pub fn deinit(c: *Copy) void {
        c.clearIndex();
        c.index.deinit(c.gpa);
        c.out.deinit(c.gpa);
    }
    pub fn proposer(c: *Copy) Proposer {
        return .{ .ptr = c, .vtable = &table };
    }
    const table: Proposer.VTable = .{
        .propose = propose,
        .propose_after = after,
        .last_match = match,
        .last_end = end,
    };
    fn match(p: *anyopaque) i64 {
        return (@as(*Copy, @ptrCast(@alignCast(p)))).last_match;
    }
    fn end(p: *anyopaque) usize {
        return (@as(*Copy, @ptrCast(@alignCast(p)))).last_end;
    }
    fn propose(p: *anyopaque, history: []const u32, room: i64) ![]const u32 {
        return after(p, history, &.{}, room);
    }
    fn at(history: []const u32, tail: []const u32, i: usize) u32 {
        return if (i < history.len) history[i] else tail[i - history.len];
    }
    fn key(history: []const u32, tail: []const u32, end_: usize) [8]u32 {
        var out: [8]u32 = undefined;
        for (&out, end_ - 8..) |*v, i| v.* = at(history, tail, i);
        return out;
    }
    fn clearIndex(c: *Copy) void {
        var it = c.index.valueIterator();
        while (it.next()) |hits| hits.deinit(c.gpa);
        c.index.clearRetainingCapacity();
        c.indexed = 0;
    }
    fn extend(c: *Copy, history: []const u32) !void {
        if (c.indexed > history.len or
            (c.indexed >= 8 and !std.mem.eql(u32, &key(history, &.{}, c.indexed), &c.last_key))) c.clearIndex();
        var end_ = @max(c.indexed + 1, 8);
        while (end_ <= history.len) : (end_ += 1) {
            const got = try c.index.getOrPut(c.gpa, key(history, &.{}, end_));
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(c.gpa, @intCast(end_ - 8));
            c.indexed = end_;
            c.last_key = key(history, &.{}, end_);
        }
    }
    fn after(p: *anyopaque, history: []const u32, tail: []const u32, room: i64) ![]const u32 {
        const c: *Copy = @ptrCast(@alignCast(p));
        c.out.clearRetainingCapacity();
        c.last_match = 0;
        c.last_end = 0;
        if (room <= 0) return c.out.items;
        const len = try std.math.add(usize, history.len, tail.len);
        if (len < 8 or history.len < 9) return c.out.items;
        if (history.len > std.math.maxInt(u32)) return error.ContextTooLong;
        try c.extend(history);
        const hits = (c.index.get(key(history, tail, len)) orelse return c.out.items).items;
        const k: usize = @intCast(room);
        var first: ?usize = null;
        var full: ?usize = null;
        for (hits) |hit| {
            const start: usize = hit;
            if (start + 8 >= history.len) continue;
            if (start >= c.prompt) {
                if (start < 8 or len < 16) continue;
                var yes = true;
                for (1..9) |i| if (history[start - i] != at(history, tail, len - 8 - i)) {
                    yes = false;
                    break;
                };
                if (!yes) continue;
            }
            if (first == null) first = start;
            if (k <= history.len - 8 - start) full = start;
        }
        const start = full orelse first orelse return c.out.items;
        c.last_end = start + 8;
        c.last_match = if (start >= c.prompt) 16 else 8;
        const count = @min(k, history.len - c.last_end);
        try c.out.appendSlice(c.gpa, history[c.last_end..][0..count]);
        return c.out.items;
    }
};
