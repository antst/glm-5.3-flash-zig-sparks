//! The three whole-prompt stages under the shared gates, against unchanged shipped stage symbols.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const ref = @import("contract_exl3_whole_ref.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Case = struct {
    rig: rig_.Rig,
    kind: ref.Kind,
    shared: bool = true,
    short: bool = false,
    candidate: cuda.Function,
    original: cuda.Function,
    fixture: ?ref.Fixture = null,
    x: u64 = 0,
    tables: [2]u64 = @splat(0),
    scales: [3]u64 = @splat(0),
    items: u64 = 0,
    counts: [2]u64 = @splat(0),
    members: u64 = 0,
    used: usize = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Case) void {
        self.drop();
        self.rig.deinit();
    }
    fn drop(self: *Case) void {
        if (self.fixture) |*f| f.deinit();
        self.fixture = null;
        self.rig.reset();
    }
    pub fn kernel(self: *Case) g.Kernel {
        var s = ref.spec(self.kind, self.shared);
        if (self.short) s.name = "control: EXL3 whole gateup omits its last routed item";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.short) .shapes else null };
    }
    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.Fixture.valid(self.kind, shape)) return false;
        self.drop();
        self.fixture = try ref.Fixture.init(self.rig.gpu.gpa, self.kind, shape, fill, seed);
        const f = &self.fixture.?;
        const a = self.rig.gpu.gpa;
        const copies: usize = if (self.kind == .down or (self.kind == .gateup and !self.shared)) ref.slots else 1;
        const xs = try a.alloc(u16, f.x.len * copies);
        defer a.free(xs);
        for (0..shape.rows) |r| for (0..copies) |p| @memcpy(xs[(r * copies + p) * shape.k ..][0..shape.k], f.x[r * shape.k ..][0..shape.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(xs));
        for (f.scales, &self.scales) |sc, *dst| {
            const expanded = try a.alloc(f16, sc.len * ref.experts);
            defer a.free(expanded);
            for (0..ref.experts) |e| @memcpy(expanded[e * sc.len ..][0..sc.len], sc);
            dst.* = try self.rig.input(std.mem.sliceAsBytes(expanded));
        }
        if (self.kind != .rotate) {
            for (f.trellis, &self.tables) |tr, *dst| {
                const data = try self.rig.input(std.mem.sliceAsBytes(tr));
                const ptrs: [ref.experts]u64 = @splat(data);
                dst.* = try self.rig.input(std.mem.sliceAsBytes(&ptrs));
            }
            var items: std.ArrayList(i32) = .empty;
            defer items.deinit(a);
            var members: std.ArrayList(i32) = .empty;
            defer members.deinit(a);
            var routed: usize = 0;
            for (0..ref.slots) |e| {
                var row: usize = 0;
                while (row < shape.rows) : (row += 64) {
                    const first = members.items.len;
                    const n = @min(@as(usize, 64), shape.rows - row);
                    for (row..row + n) |r| try members.append(a, @intCast(r * ref.slots + e));
                    try items.appendSlice(a, &.{ @intCast(e), @intCast(first), @intCast(n) });
                }
                if (e + 1 == ref.experts) routed = items.items.len / 3;
            }
            self.used = items.items.len / 3;
            self.items = try self.rig.input(std.mem.sliceAsBytes(items.items));
            self.members = try self.rig.input(std.mem.sliceAsBytes(members.items));
            const own = [1]i32{@intCast(if (self.short) routed - 1 else self.used)};
            const old = [1]i32{@intCast(routed)};
            self.counts = .{ try self.rig.input(std.mem.sliceAsBytes(&own)), try self.rig.input(std.mem.sliceAsBytes(&old)) };
        }
        const bytes = self.count() * c.size(ref.spec(self.kind, self.shared).out);
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        return true;
    }

    fn launch(self: *Case, old: bool) !void {
        const f = &self.fixture.?;
        const shape = f.shape;
        const output = if (old) self.base else self.out;
        const elem = ref.spec(self.kind, self.shared).out;
        const buf: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = output, .len = self.count() * c.size(elem) };
        try buf.fill32(if (elem == .f32) 0x3f800000 else 0x3c003c00, self.rig.s.handle);
        var args: cuda.Args = .{};
        var grid: cuda.launch.Dim3 = .{};
        switch (self.kind) {
            .rotate => {
                inline for (.{ self.x, @as(i32, @intCast(shape.k)), self.scales[0], output, @as(i32, @intCast(shape.k)) }) |v| args.add(v);
                grid = .{ .x = @intCast(shape.rows), .y = @intCast(shape.k / 128) };
            },
            .gateup => {
                inline for (.{ self.x, self.x, self.tables[0], self.tables[1], self.items, self.counts[@intFromBool(old)], self.members, self.scales[0], self.scales[1], self.scales[2], output }) |v| args.add(v);
                for ([_]usize{ shape.k, shape.n, if (self.shared) ref.slots else 1 }) |v| args.add(@as(i32, @intCast(v)));
                args.add(@as(f32, 7));
                args.add(@as(i32, 0));
                if (!old) args.add(@as(i32, ref.experts));
                grid = .{ .x = @intCast(shape.n / 128), .y = @intCast(self.used) };
            },
            .down => {
                inline for (.{ self.x, self.tables[0], self.items, self.counts[@intFromBool(old)], self.members, self.scales[0], output }) |v| args.add(v);
                for ([_]usize{ shape.k, shape.n, if (old) ref.slots else ref.experts }) |v| args.add(@as(i32, @intCast(v)));
                grid = .{ .x = @intCast(shape.n / 256), .y = @intCast(self.used) };
            },
        }
        try cuda.launch.launch(if (old) self.original else self.candidate, .{ .grid = grid, .block = .{ .x = if (self.kind == .rotate) 32 else 256 } }, self.rig.s, &args);
    }
    pub fn run(self: *Case) !void {
        try self.launch(false);
        try self.rig.s.synchronize();
    }
    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(true);
        return self.rig.bytes(self.base, self.count() * c.size(ref.spec(self.kind, self.shared).out));
    }
    pub fn count(self: *Case) usize {
        return self.fixture.?.count();
    }
    pub fn reference(self: *Case, out: []f64, mag: []f64) !void {
        return self.fixture.?.reference(out, mag);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.fixture.?.evaluate(out, acc, order);
    }
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const out = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(out);
        const old = try self.replaced();
        if (std.mem.indexOfDiff(u8, out, old)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from shipped stage", .{at});
        for (0..self.count()) |i| if (c.classOf(c.read(ref.spec(self.kind, self.shared).out, out, i)) != .finite) return null;
        return "nonfinite plants did not reach an output";
    }
};

pub fn isolated(gpu: check.Gpu) !void {
    const bytes align(16) = @embedFile("fatbin_exl3_prompt_whole").*;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&bytes, &digest, .{});
    try check.expect(std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), "65c062c3d9f46342c76a29dc844aa57e228dccbc08903c9f877e105b1bc6fc39"), "whole prompt exact candidate asset", .{});
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var candidate = try cuda.Module.load(gpu.d, &bytes);
    defer candidate.unload();
    var original = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
    defer original.unload();
    const kinds = [_]ref.Kind{ .rotate, .gateup, .gateup, .down, .gateup };
    const symbols = [_][:0]const u8{ "tf_exl3_prompt_rotate", "tf_exl3_prompt_gu", "tf_exl3_prompt_gu", "tf_exl3_prompt_down", "tf_exl3_prompt_gu" };
    var cases: [kinds.len]Case = undefined;
    var initialized: usize = 0;
    defer for (cases[0..initialized]) |*item| item.deinit();
    defer cuda.completion.drain(gpu.d, stream, "whole prompt standard contract did not drain");
    var kernels: [kinds.len]g.Kernel = undefined;
    for (&cases, &kernels, kinds, symbols, 0..) |*item, *kernel, kind, symbol, i| {
        item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .kind = kind, .shared = i != 2, .short = i == 4, .candidate = try candidate.function(symbol), .original = try original.function(symbol) };
        initialized += 1;
        kernel.* = item.kernel();
    }
    var tally: g.Tally = .{};
    const start = check.now(gpu.io);
    try g.runAll(gpu.gpa, &kernels, 0x6b63, &tally);
    try check.expect(tally.kernels == 4 and tally.passed == 4 and tally.controls == tally.caught and tally.caught == 1, "whole prompt standard gate census", .{});
    try @import("contract_finish.zig").run(gpu, &.{"EXL3 whole"}, tally, start);
}
