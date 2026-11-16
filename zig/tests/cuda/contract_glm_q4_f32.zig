//! Explicit opt-in contract for the NEW Q4 fp32 symbol; the ordinary kernel-contract command is unchanged.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const q4 = @import("glm_cuda").q4;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Affine = core.contract.affine.Affine;
const limits = core.contract.limits;

pub const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 4096, .k = 1024 }, .{ .rows = 2, .n = 4096, .k = 1024 },
    .{ .rows = 4, .n = 4096, .k = 1024 }, .{ .rows = 6, .n = 4096, .k = 1024 },
    .{ .rows = 8, .n = 4096, .k = 1024 }, .{ .rows = 16, .n = 4096, .k = 1024 },
    .{ .rows = 7, .n = 65, .k = 512 },    .{ .rows = 15, .n = 129, .k = 2048 },
    .{ .rows = 16, .n = 64, .k = 4096 },  .{ .rows = 3, .n = 4096, .k = 4096 },
};
fn depth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64) + 8;
}
pub fn spec(strided: bool) c.Spec {
    return .{
        .name = if (strided) "cuda GLM Q4 fp32 output strided" else "cuda GLM Q4 fp32 output",
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .tiles = .{ .{ .size = 16 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } },
        .served = &shapes,
    };
}

const Product = struct {
    rig: Rig,
    ks: *const q4.Kernels,
    strided: bool = false,
    short: bool = false,
    f: ?Affine = null,
    x: u64 = 0,
    xs: u64 = 0,
    out: u64 = 0,
    ldx: usize = 0,
    q: q4.Q4 = undefined,

    fn drop(self: *Product) void {
        if (self.f) |f| f.deinit();
        self.f = null;
        self.rig.reset();
    }
    fn deinit(self: *Product) void {
        self.drop();
        self.rig.deinit();
    }
    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const f = try Affine.init(a, shape, 4, 64);
        self.f = f;
        try f.fill(fill, seed);
        self.ldx = shape.k + @as(usize, if (self.strided) 64 else 0);
        const x = try a.alloc(u16, shape.rows * self.ldx);
        defer a.free(x);
        @memset(x, 0x7fc1);
        for (0..shape.rows) |r| @memcpy(x[r * self.ldx ..][0..shape.k], f.x[r * shape.k ..][0..shape.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        const xs = try a.alloc(f32, shape.rows * (shape.k / 64));
        defer a.free(xs);
        for (xs, 0..) |*v, i| {
            var sum: f32 = 0;
            for (f.x[i * 64 ..][0..64]) |h| sum += c.fromBf16(h);
            v.* = sum;
        }
        self.xs = try self.rig.input(std.mem.sliceAsBytes(xs));
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        f.words(words);
        const src_w = try self.rig.input(std.mem.sliceAsBytes(words));
        const src_s = try self.rig.input(std.mem.sliceAsBytes(f.scales));
        const src_b = try self.rig.input(std.mem.sliceAsBytes(f.biases));
        const sizes = q4.bytes(shape.n, shape.k);
        const zeros = try a.alloc(u8, @max(sizes[0], sizes[1]));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.q = .{
            .w = try self.rig.input(zeros[0..sizes[0]]),
            .s = try self.rig.input(zeros[0..sizes[1]]),
            .b = try self.rig.input(zeros[0..sizes[2]]),
            .n = shape.n,
            .k = shape.k,
            .npad = q4.npadOf(shape.n),
        };
        try q4.pack(self.ks, self.rig.s, src_w, src_s, src_b, self.q);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.count() * 4);
        return true;
    }
    pub fn run(self: *Product) !void {
        const rows = self.f.?.shape.rows;
        const launched = rows - @as(usize, @intFromBool(self.short and rows > 1));
        try q4.matmulWide(self.ks, self.rig.s, self.x, self.ldx, self.xs, self.q, self.out, launched);
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Product) usize {
        return self.f.?.outputs();
    }
    pub fn reference(self: *Product, out: []f64, mag: []f64) !void {
        return self.f.?.reference(out, mag);
    }
    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.f.?.simulate(acc, order, out);
    }
};

/// Gates 1-4 plus live function resource limits on this SM, after explicitly requiring the rebuilt symbol.
pub fn run(gpu: check.Gpu) !void {
    const cap = try gpu.ctx.capability();
    const target = limits.target(cap) orelse return error.UnmeasuredGpu;
    if (cap < 90) return error.UnsupportedGpu;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var group = try cuda.Module.load(gpu.d, cuda.kernels.qmm_group);
    defer group.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer ops.unload();
    var ks = try q4.Kernels.load(group, ops, gpu.ctx);
    try q4.enableWide(&ks, group);
    var contiguous: Product = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &ks };
    defer contiguous.deinit();
    var strided: Product = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &ks, .strided = true };
    defer strided.deinit();
    var truncated: Product = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = &ks, .short = true };
    defer truncated.deinit();
    var control = spec(false);
    control.name = "control: GLM Q4 fp32 omits the final row";
    const kernels = [_]g.Kernel{
        .{ .spec = spec(false), .ctx = &contiguous, .vt = &g.hooks(Product) },
        .{ .spec = spec(true), .ctx = &strided, .vt = &g.hooks(Product) },
        .{ .spec = control, .ctx = &truncated, .vt = &g.hooks(Product), .control = .shapes },
    };
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, &kernels, 0x71346633, &tally);
    const f = ks.group_f32.?;
    const usage: limits.Usage = .{
        .regs = @intCast(try f.attribute(.num_regs)),
        .static = @intCast(try f.attribute(.shared_size_bytes)),
        .bound = @intCast(try f.attribute(.max_threads_per_block)),
        .local = @intCast(try f.attribute(.local_size_bytes)),
    };
    var log: std.ArrayList(u8) = .empty;
    defer log.deinit(gpu.gpa);
    var problems: usize = 0;
    for ([_]u32{ 1, 2, 4, 8 }) |cluster| problems += try limits.cuda(.{
        .name = spec(false).name,
        .symbol = "qmm_group tile 2 F32=true",
        .image = "qmm_group",
        .threads = 128,
        .dynamic = 35328,
        .cluster = cluster,
        .opt_in = true,
        .min_sm = 90,
    }, usage, target, &log, gpu.gpa);
    std.debug.print("{s}Q4-F32 resource sm{d}: regs{d} static{d} maxthreads{d} local{d}; dynamic35328\n", .{ log.items, cap, usage.regs, usage.static, usage.bound, usage.local });
    std.debug.print("Q4-F32 SUMMARY {d} kernels: {d} pass gates1-4, {d} fail; {d}/{d} controls caught; {d} live limit problems\n", .{ tally.kernels, tally.passed, tally.failed, tally.caught, tally.controls, problems });
    try check.expect(tally.problems() + problems == 0, "Q4-F32 contract failures", .{});
}
