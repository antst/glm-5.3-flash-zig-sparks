//! DFlash-only loading keeps the legacy path and borrows one owned head, without the unused target layer.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const mode = glm.dflash_head_mode;
const t = std.testing;

test {
    _ = glm.mtp_load;
}

test "GLM DFlash head: strict cached flag, fingerprint and scope precede allocations" {
    const Env = struct {
        var raw: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_DFLASH_ONLY_HEAD")) raw else null;
        }
    };
    defer Env.raw = null;
    var problem: []const u8 = "";
    const off = try glm.ranks.Settings.read(Env.get, &problem);
    try t.expect(!off.dflash_only_head);
    Env.raw = "1";
    const on = try glm.ranks.Settings.read(Env.get, &problem);
    Env.raw = "0";
    try t.expect(on.dflash_only_head and !(try glm.ranks.Settings.read(Env.get, &problem)).dflash_only_head);
    try mode.require(on, true, true, false);
    try t.expectError(error.DFlashHeadScope, mode.require(on, false, true, false));
    try t.expectError(error.DFlashHeadScope, mode.require(on, true, false, false));
    try t.expectError(error.DFlashHeadScope, mode.require(on, true, true, true));
    var unsupported = on;
    unsupported.pages = true;
    try t.expectError(error.DFlashHeadScope, mode.require(unsupported, true, true, false));
    unsupported = on;
    unsupported.kv_host = 1;
    try t.expectError(error.DFlashHeadScope, mode.require(unsupported, true, true, false));
    try mode.require(off, false, false, true);
    try t.expectEqual(glm.weights.Heads.none, mode.selection(off, false, false));
    try t.expectEqual(glm.weights.Heads.mtp, mode.selection(off, true, true));
    try t.expectEqual(glm.weights.Heads.draft, mode.selection(on, true, true));
    const FP = glm.ranks.Fingerprint;
    const a = FP.of(4, 0, off, 196608, true, @splat(0), @splat(0), @splat(0));
    const b = FP.of(4, 1, on, 196608, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(@as(u32, 64), a.layout);
    try t.expectEqual(FP.Field.dflash_only_head, glm.ranks.mismatch(&.{ a, b }).?.field);
    var fake: struct { settings: glm.ranks.Settings, mtp: bool = true, drafter: ?*u8 = null } = .{ .settings = on };
    try t.expect(!glm.snapshots.mode.head(&fake));
    for ([_][]const u8{ "", "2", "-1", "yes", "true" }) |raw| {
        Env.raw = raw;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Env.get, &problem));
        try t.expectEqualStrings("TF_GLM_DFLASH_ONLY_HEAD: 0 or 1", problem);
    }
}

test "GLM DFlash head: explicit policy selects the owned head and never falls back to the other owner" {
    var w: glm.weights.Weights = .{ .gpa = t.allocator, .config = .{}, .share = undefined };
    const owned: glm.q4.Q4 = .{ .w = 256, .s = 512, .b = 768, .n = 128, .k = 4096, .npad = 128 };
    const legacy: glm.q4.Q4 = .{ .w = 1024, .s = 1280, .b = 1536, .n = 128, .k = 4096, .npad = 128 };
    try t.expect(w.draftHead(false) == null and w.draftHead(true) == null);
    w.draft_head = owned;
    try t.expectEqualDeep(owned, w.draftHead(true).?);
    try t.expect(w.draftHead(false) == null);
    w.mtp = @as(glm.weights.Mtp, undefined);
    w.mtp.?.draft = legacy;
    try t.expectEqualDeep(legacy, w.draftHead(false).?);
    try t.expectEqualDeep(owned, w.draftHead(true).?);
    w.draft_head = null;
    try t.expect(w.draftHead(true) == null);
}

const Sizing = struct {
    var next: u64 = 1 << 30;
    fn alloc(out: *u64, bytes: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(usize, bytes, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "GLM DFlash head: actual State omits only the MTP cache and keeps the canonical no-head snapshot" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*layer, i| {
        layer.* = undefined;
        layer.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
    }
    for ([_]usize{ 2, 4 }) |world| for ([_]usize{ 8192, 137216, 196608, 1048576 }) |cap| {
        w.share = try glm.shard.Share.of(&w.config, world - 1, world);
        for ([_]bool{ false, true }) |kv8| {
            var old = try glm.forward.State.initFormat(t.allocator, &d, &w, cap, kv8);
            defer old.deinit();
            var keep = try glm.forward.State.initHead(t.allocator, &d, &w, cap, kv8, true);
            defer keep.deinit();
            var omit = try glm.forward.State.initHead(t.allocator, &d, &w, cap, kv8, false);
            defer omit.deinit();
            try t.expectEqual(old.arena.buf.len, keep.arena.buf.len);
            const c = &w.config;
            const latent = std.mem.alignForward(usize, try glm.latent.cacheBytes(cap, c.kv_lora, kv8), 256);
            const ring = std.mem.alignForward(usize, keep.ring_rows * c.i_dim * 2, 256);
            const pooled = std.mem.alignForward(usize, (cap / glm.sparse.pool + 2) * c.i_dim * 2, 256);
            const removed = latent + 2 * ring + pooled;
            try t.expectEqual(removed, keep.arena.buf.len - omit.arena.buf.len);
            try t.expectEqual(removed, keep.prompt_hidden - keep.arena.buf.ptr - (omit.prompt_hidden - omit.arena.buf.ptr));
            try t.expectEqual(keep.rec_layer, omit.rec_layer);
            try t.expectEqual(keep.proj_layer, omit.proj_layer);
            for (keep.caches, omit.caches) |yes, no| {
                try t.expectEqual(yes.lc - keep.arena.buf.ptr, no.lc - omit.arena.buf.ptr);
                try t.expectEqual(yes.ik - keep.arena.buf.ptr, no.ik - omit.arena.buf.ptr);
                try t.expectEqual(yes.pk - keep.arena.buf.ptr, no.pk - omit.arena.buf.ptr);
            }
            for ([_]bool{ false, true }) |live| {
                const before = try glm.snapshots.layout.Plan.sizeWith(&keep, c, cap, 2048, null, 0, live);
                const after = try glm.snapshots.layout.Plan.sizeWith(&omit, c, cap, 2048, null, 0, live);
                try t.expectEqual(before, after);
            }
        }
    };
}
