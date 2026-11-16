//! Spill worker operations, preserving the Store lock and publication ordering.
const Store = @import("store.zig").Store;
const std = @import("store_types.zig").std;
const mut = @import("store_types.zig").mut;
const Digest = @import("store_types.zig").Digest;
const Point = @import("store_types.zig").Point;
const remove = Store.remove;
const parentOf = Store.parentOf;
const tree = Store.tree;
const Why = Store.Why;
const Victim = Store.Victim;

pub fn victims(s: *Store, only: ?usize) !std.ArrayList(Victim) {
    const gpa = s.gpa;
    var t = try s.tree();
    defer t.deinit(gpa);
    const n = t.keys.len;
    var out: std.ArrayList(Victim) = .empty;
    errdefer out.deinit(gpa);
    const used = try gpa.alloc(i96, n);
    defer gpa.free(used);
    const at = try gpa.alloc(u32, n);
    defer gpa.free(at);
    const par = try gpa.alloc(?usize, n);
    defer gpa.free(par);
    const has_kid = try gpa.alloc(bool, n);
    defer gpa.free(has_kid);
    @memset(has_kid, false);
    for (t.keys, 0..) |k, j| {
        const p = s.points.get(k).?;
        used[j] = p.used;
        at[j] = p.at;
        par[j] = if (s.parent.get(k) orelse null) |pk| t.index.get(pk) else null;
    }
    for (0..n) |j| if (par[j]) |pj| {
        if (t.conv[j] != null and t.conv[pj] == t.conv[j]) has_kid[pj] = true;
    };
    const kind = try gpa.alloc(u8, n); // 0 keep, 1 turn, 2 fork
    defer gpa.free(kind);
    @memset(kind, 0);
    const Whole = struct { conv: usize, used: i96 };
    var wholes: std.ArrayList(Whole) = .empty;
    defer wholes.deinit(gpa);
    var seen: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer seen.deinit(gpa);
    const on_path = try gpa.alloc(bool, n);
    defer gpa.free(on_path);
    for (0..n) |j0| {
        const cv = t.conv[j0] orelse continue;
        if ((try seen.getOrPut(gpa, cv)).found_existing) continue;
        if (only != null and only.? != cv) continue;
        var newest: ?usize = null;
        var newest_any: ?usize = null;
        var last: i96 = std.math.minInt(i96);
        for (0..n) |j| {
            if (t.conv[j] != cv) continue;
            last = @max(last, used[j]);
            if (newest_any == null or used[j] > used[newest_any.?]) newest_any = j;
            if (!has_kid[j] and (newest == null or used[j] > used[newest.?])) newest = j;
        }
        @memset(on_path, false);
        var path_len: usize = 0;
        var cur: ?usize = newest orelse newest_any;
        const keep_n = @max(1, s.cfg.turns);
        while (cur) |j| {
            if (t.conv[j] != cv) break;
            on_path[j] = true;
            if (path_len >= keep_n) kind[j] = 1;
            path_len += 1;
            cur = par[j];
        }
        for (0..n) |j| {
            if (t.conv[j] != cv or on_path[j]) continue;
            kind[j] = if (has_kid[j]) 1 else 2;
        }
        try wholes.append(gpa, .{ .conv = cv, .used = last });
    }
    const Ent = struct { j: usize, used: i96, at: u32 };
    var turns: std.ArrayList(Ent) = .empty;
    defer turns.deinit(gpa);
    var forks: std.ArrayList(Ent) = .empty;
    defer forks.deinit(gpa);
    for (0..n) |j| switch (kind[j]) {
        1 => try turns.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
        2 => try forks.append(gpa, .{ .j = j, .used = used[j], .at = at[j] }),
        else => {},
    };
    const byUsed = struct {
        pub fn lt(_: void, a: Ent, b: Ent) bool {
            return a.used < b.used or (a.used == b.used and a.at > b.at);
        }
    }.lt;
    std.mem.sort(Ent, turns.items, {}, byUsed);
    std.mem.sort(Ent, forks.items, {}, byUsed);
    for (turns.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_turns });
    for (forks.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_forks });
    if (only != null) return out;
    if (mut(.evict_shared_first)) {
        var sh: std.ArrayList(Ent) = .empty;
        defer sh.deinit(gpa);
        for (0..n) |j| if (t.conv[j] == null) try sh.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
        std.mem.sort(Ent, sh.items, {}, byUsed);
        for (sh.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
    }
    std.mem.sort(Whole, wholes.items, {}, struct {
        pub fn lt(_: void, a: Whole, b: Whole) bool {
            return a.used < b.used;
        }
    }.lt);
    var rest: std.ArrayList(Ent) = .empty;
    defer rest.deinit(gpa);
    for (wholes.items) |w| {
        rest.clearRetainingCapacity();
        for (0..n) |j| if (t.conv[j] == w.conv and kind[j] == 0) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
        std.mem.sort(Ent, rest.items, {}, struct {
            pub fn longer(_: void, a: Ent, b: Ent) bool {
                return a.at > b.at;
            }
        }.longer);
        for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_conversations });
    }
    rest.clearRetainingCapacity();
    if (!mut(.evict_shared_first)) for (0..n) |j| if (t.conv[j] == null) try rest.append(gpa, .{ .j = j, .used = used[j], .at = at[j] });
    std.mem.sort(Ent, rest.items, {}, byUsed);
    for (rest.items) |e| try out.append(gpa, .{ .key = t.keys[e.j], .why = .pruned_shared });
    return out;
}

pub fn removeWhy(s: *Store, key: Digest, why: Why) void {
    switch (why) {
        inline else => |w| s.remove(key, @tagName(w)),
    }
}

pub fn makeRoom(s: *Store, nbytes: u64, protect: ?Digest) void {
    var v = s.victims(null) catch return;
    defer v.deinit(s.gpa);
    const target = s.stored_bytes -| nbytes;
    for (v.items) |x| {
        if (s.stored_bytes <= target) return;
        if (protect != null and std.mem.eql(u8, &protect.?, &x.key)) continue;
        if (!s.points.contains(x.key) or s.reading.contains(x.key)) continue;
        s.removeWhy(x.key, x.why);
    }
}

pub fn trimPath(s: *Store, p: *Point) void {
    if (s.cfg.turns == 0 or p.shared) return;
    var path: std.ArrayList(Digest) = .empty;
    defer path.deinit(s.gpa);
    var cur: ?Digest = p.key;
    while (cur) |k| {
        const q = s.points.get(k) orelse break;
        if (q.shared) break;
        path.append(s.gpa, k) catch return;
        cur = s.parent.get(k) orelse null;
    }
    if (path.items.len <= s.cfg.turns) return;
    for (path.items[s.cfg.turns..]) |k| {
        if (!s.reading.contains(k)) s.remove(k, "trimmed_turns");
    }
}

pub fn convBytes(s: *Store, keys: []const Digest) u64 {
    const gpa = s.gpa;
    var mine: std.AutoHashMapUnmanaged(Digest, void) = .empty;
    defer mine.deinit(gpa);
    for (keys) |k| mine.put(gpa, k, {}) catch return 0;
    var others: std.AutoHashMapUnmanaged(Digest, void) = .empty;
    defer others.deinit(gpa);
    var it = s.points.iterator();
    while (it.next()) |e| {
        if (mine.contains(e.key_ptr.*)) continue;
        for (e.value_ptr.*.blocks) |b| others.put(gpa, b.name, {}) catch return 0;
    }
    var inos: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer inos.deinit(gpa);
    var n: u64 = 0;
    for (keys) |k| {
        const p = s.points.get(k).?;
        n += p.files;
        for (p.blocks) |b| {
            if (others.contains(b.name)) continue;
            const info = s.have.get(b.name) orelse continue;
            const ino = s.inodes.get(info.ino) orelse continue;
            var all_mine = true;
            for (ino.names.items) |nm| if (others.contains(nm)) {
                all_mine = false;
            };
            if (!all_mine) continue;
            if ((inos.getOrPut(gpa, info.ino) catch return n).found_existing) continue;
            n += ino.bytes;
        }
    }
    return n;
}

pub fn trimOwn(s: *Store, p: *Point, need: u64) void {
    if (s.cfg.quota <= 0 or p.shared or s.points.count() == 0) return;
    const gpa = s.gpa;
    var t = s.tree() catch return;
    defer t.deinit(gpa);
    // the conversation `p` would join: its parent's (below a shared point), or its group's
    var cv: ?usize = null;
    if (s.parentOf(p)) |up| {
        const q = s.points.get(up).?;
        if (!q.shared) cv = t.conv[t.index.get(up).?];
    }
    if (cv == null and p.group != 0) {
        for (t.keys, 0..) |k, j| if (s.points.get(k).?.group == p.group) {
            cv = t.conv[j];
            break;
        };
    }
    const conv = cv orelse return;
    var keys: std.ArrayList(Digest) = .empty;
    defer keys.deinit(gpa);
    for (t.keys, 0..) |k, j| if (t.conv[j] == conv) keys.append(gpa, k) catch return;
    const allowed: u64 = @intFromFloat(s.cfg.quota * @as(f64, @floatFromInt(s.cfg.cap_bytes)));
    const held = s.convBytes(keys.items) + need;
    if (held <= allowed) return;
    s.counters.quota_trims += 1;
    const target = s.stored_bytes -| (held - allowed);
    var v = s.victims(conv) catch return;
    defer v.deinit(gpa);
    for (v.items) |x| {
        if (s.stored_bytes <= target) break;
        if (s.reading.contains(x.key) or !s.points.contains(x.key)) continue;
        s.removeWhy(x.key, x.why);
    }
}
