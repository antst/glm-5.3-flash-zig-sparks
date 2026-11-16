//! Actual State strides and Forward.commit at TP4/H16, against original fused-prefix KDA and CPU conv windows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Rig = qr.RigT;
const H = 16;
const N = H * 128;
const L = 3;
const W = glm.forward.window_rows;

fn upload(r: *Rig, ptr: u64, bytes: []const u8) !void {
    try r.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
}

fn floats(r: *Rig, count: usize, scale: f32) ![]f32 {
    const out = try r.a.alloc(f32, count);
    for (out) |*v| v.* = (r.rng.float(f32) * 2 - 1) * scale;
    return out;
}

fn same(r: *Rig, what: []const u8, ptr: u64, want: []const u8) !void {
    try check.sameBytes(what, try r.back(ptr, want.len), want);
}

fn sentinel(r: *Rig, what: []const u8, ptr: u64, bytes: usize, value: u8) !void {
    if (bytes == 0) return;
    for (try r.back(ptr, bytes)) |v| try check.expect(v == value, "{s}: untouched padding", .{what});
}

fn finiteBf16(raw: []const u8) bool {
    var i: usize = 0;
    while (i < raw.len) : (i += 2) {
        const b = std.mem.readInt(u16, raw[i..][0..2], .little);
        if (!std.math.isFinite(@as(f32, @bitCast(@as(u32, b) << 16)))) return false;
    }
    return true;
}

test "row16 State oracle refuses an unwritten wide/fused output" {
    try std.testing.expect(!finiteBf16(&.{ 0xff, 0xff, 0xff, 0xff }));
    try std.testing.expect(finiteBf16(&.{ 0, 0, 0x80, 0x3f }));
}

fn fixture(gpu: check.Gpu, m: *const glm.forward.Modules, stream: cuda.Stream, rows: usize, parity: u1, nonfinite: bool) ![32]u8 {
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(1600 + rows * 2 + parity);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var layers: [L]glm.weights.Layer = undefined;
    for (&layers) |*l| l.mixer = .{ .kda = undefined };
    var w: glm.weights.Weights = .{ .gpa = r.a, .config = .{ .layers = L }, .share = undefined };
    w.share = try glm.shard.Share.of(&w.config, 0, 4);
    w.layers = &layers;
    try check.expect(w.share.kda.len() == H, "row16: actual TP4 heads16", .{});
    var st = try glm.forward.State.initFormat(r.a, gpu.d, &w, 128, true);
    defer st.deinit();
    defer cuda.completion.drain(gpu.d, stream, "row16 State fixture teardown");
    try check.expect(glm.forward.window_rows == W and st.proj_layer == W * st.proj_row, "row16: State projection layer physical stride", .{});
    const kw = st.proj_row;
    const projs = try r.bfs(L * W * kw, 0.5);
    if (nonfinite) projs[(rows - 1) * kw + 19] = 0x7fc0;
    const convs = try r.bfs(L * 3 * 3 * N, 0.5);
    const recs = try floats(&r, L * H * 128 * 128, 0.02);
    const a = try r.bfs(L * W * N, 1);
    const gate = try r.bfs(L * W * N, 1);
    const cw = try floats(&r, L * 3 * N * 4, 0.2);
    const al = try floats(&r, L * H, 1);
    const dt = try floats(&r, L * N, 0.2);
    const norm = try r.bfs(128, 1);
    const da = try r.dev(std.mem.sliceAsBytes(a));
    const dg = try r.dev(std.mem.sliceAsBytes(gate));
    const dcw = try r.dev(std.mem.sliceAsBytes(cw));
    const dal = try r.dev(std.mem.sliceAsBytes(al));
    const ddt = try r.dev(std.mem.sliceAsBytes(dt));
    const dn = try r.dev(std.mem.sliceAsBytes(norm));
    const out = try r.zeros(L * W * N * 2);
    const q = try r.zeros(L * W * N * 4);
    const y = try r.zeros(L * W * N * 2);
    const expected = try r.zeros(L * st.rec_layer);
    const eo = try r.zeros(L * W * N * 2);
    const ek = try r.zeros(L * W * N * 4);
    const ev = try r.zeros(L * W * N * 2);
    const eg = try r.zeros(L * W * N * 4);
    const eb = try r.zeros(L * W * H * 4);
    var b: glm.forward.Buffers = undefined;
    b.prefill = false;
    const ranks = [_]glm.forward.Rank{.{ .w = &w, .st = &st, .b = &b }};
    const f: glm.forward.Forward = .{ .d = gpu.d, .m = m, .stream = stream, .ranks = &ranks, .exchange = .local };
    const conv_want = try r.a.alloc(u16, convs.len);
    for (1..rows + 1) |keep| {
        try stream.synchronize();
        try st.arena.buf.fill8(0xa5, stream.handle);
        try upload(&r, st.proj, std.mem.sliceAsBytes(projs));
        try upload(&r, st.conv, std.mem.sliceAsBytes(convs));
        try upload(&r, st.recAt(parity, 0), std.mem.sliceAsBytes(recs));
        @memset(st.cur, parity);
        try st.setPos(stream, 17);
        try st.setMtpLen(stream, 0);
        st.pending = rows;
        st.prompt = false;
        for (0..L) |li| {
            const ch: glm.kda.Chain = .{
                .p = st.proj + li * st.proj_layer * 2,
                .p_stride = kw,
                .b_off = 3 * N + 2 * 128,
                .a = da + li * W * N * 2,
                .a_stride = N,
                .g = dg + li * W * N * 2,
                .g_stride = N,
                .conv_state = st.conv + li * 3 * 3 * N * 2,
                .conv_w = dcw + li * 3 * N * 4 * 4,
                .state_in = st.recAt(parity, li),
                .a_log = dal + li * H * 4,
                .dt_bias = ddt + li * N * 4,
                .norm_w = dn,
                .eps = 1e-6,
                .lower = -5,
                .heads = H,
            };
            const full: glm.kda.Saves = .{ .out = out + li * W * N * 2, .state_out = st.recAt(1 - parity, li), .k = st.save_k + li * W * N * 4, .v = st.save_v + li * W * N * 2, .g = st.save_g + li * W * N * 4, .b = st.save_b + li * W * H * 4 };
            try glm.kda.chain(&m.kda, stream, ch, rows, full, .{ .q = q + li * W * N * 4, .y = y + li * W * N * 2 });
            try glm.kda.chain(&m.kda, stream, ch, keep, .{ .out = eo + li * W * N * 2, .state_out = expected + li * st.rec_layer, .k = ek + li * W * N * 4, .v = ev + li * W * N * 2, .g = eg + li * W * N * 4, .b = eb + li * W * H * 4 }, null);
            if (!nonfinite or keep < rows) try check.expect(finiteBf16(try r.back(full.out, keep * N * 2)), "row16 State: finite positive wide writes", .{});
            try check.sameBytes("row16: wide/fused kept output", try r.back(full.out, keep * N * 2), try r.back(eo + li * W * N * 2, keep * N * 2));
            for (0..3) |tap| for (0..3 * N) |col| {
                const src = keep + tap;
                conv_want[(li * 3 + tap) * 3 * N + col] = if (src < 3)
                    convs[(li * 3 + src) * 3 * N + col]
                else
                    projs[(li * W + src - 3) * kw + col];
            };
        }
        try f.commit(keep);
        try check.expect(st.pending == 0 and st.pos == 17 + keep, "row16: actual commit position/pending", .{});
        for (st.cur) |cur| try check.expect(cur == 1 - parity, "row16: all layer parities", .{});
        try check.sameBytes("row16: actual replay_layers vs fused prefix", try r.back(st.recAt(1 - parity, 0), L * st.rec_layer), try r.back(expected, L * st.rec_layer));
        try same(&r, "row16: committed convolution CPU reference", st.conv, std.mem.sliceAsBytes(conv_want));
        try same(&r, "row16: input recurrent states immutable", st.recAt(parity, 0), std.mem.sliceAsBytes(recs));
        try same(&r, "row16: projected input rows immutable", st.proj, std.mem.sliceAsBytes(projs));
        for (0..L) |li| {
            try sentinel(&r, "row16: save_k", st.save_k + (li * W + rows) * N * 4, (W - rows) * N * 4, 0xa5);
            try sentinel(&r, "row16: save_v", st.save_v + (li * W + rows) * N * 2, (W - rows) * N * 2, 0xa5);
            try sentinel(&r, "row16: save_g", st.save_g + (li * W + rows) * N * 4, (W - rows) * N * 4, 0xa5);
            try sentinel(&r, "row16: save_b", st.save_b + (li * W + rows) * H * 4, (W - rows) * H * 4, 0xa5);
        }
        digest.update(try r.back(out, L * W * N * 2));
        digest.update(try r.back(st.recAt(1 - parity, 0), L * st.rec_layer));
        digest.update(try r.back(st.conv, convs.len * 2));
        for ([_]u64{ st.save_k, st.save_g }) |ptr| digest.update(try r.back(ptr, L * W * N * 4));
        digest.update(try r.back(st.save_v, L * W * N * 2));
        digest.update(try r.back(st.save_b, L * W * H * 4));
        const device_pos = try r.back(st.pos_dev, 4);
        try check.expect(std.mem.bytesToValue(i32, device_pos) == @as(i32, @intCast(17 + keep)), "row16: GPU position", .{});
        if (keep == rows) {
            const snapshot = try r.back(st.arena.buf.ptr, st.arena.buf.len);
            const saved_cur = try r.a.dupe(u1, st.cur);
            try st.reset(stream);
            try upload(&r, st.arena.buf.ptr, snapshot);
            try st.restoreHost(stream, 17 + keep, 0, saved_cur, .{ 0, 0 });
            try same(&r, "row16: actual reset and restoreHost with full State image", st.arena.buf.ptr, snapshot);
        }
        if ((rows == W or rows == 16) and keep == 1) {
            try upload(&r, st.conv, std.mem.sliceAsBytes(convs));
            @memset(st.cur, parity);
            st.pending = rows;
            var wrong = st;
            wrong.proj_layer = 8 * kw;
            const bad_ranks = [_]glm.forward.Rank{.{ .w = &w, .st = &wrong, .b = &b }};
            var bad = f;
            bad.ranks = &bad_ranks;
            try bad.commit(keep);
            try check.expect(!std.mem.eql(u8, try r.back(st.conv, convs.len * 2), std.mem.sliceAsBytes(conv_want)), "row16: allocated legacy8 projection-stride control must differ", .{});
        }
        try st.reset(stream);
        try sentinel(&r, "row16: reset recurrent buffers", st.rec, 2 * L * st.rec_layer, 0);
        try sentinel(&r, "row16: reset convolution", st.conv, L * 3 * st.conv_c * 2, 0);
    }
    try stream.synchronize();
    return digest.finalResult();
}

fn qualify(gpu: check.Gpu, widths: []const usize) !void {
    var m = try glm.forward.Modules.load(gpu.d, gpu.ctx, false);
    defer m.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    defer cuda.completion.drain(gpu.d, s, "row16 State modules teardown");
    for (widths) |rows| for ([_]u1{ 0, 1 }) |parity| {
        const first = try fixture(gpu, &m, s, rows, parity, false);
        for (0..4) |_| try check.sameBytes("row16 State: five-repeat output/save/commit SHA", &first, &(try fixture(gpu, &m, s, rows, parity, false)));
    };
    const first = try fixture(gpu, &m, s, widths[widths.len - 1], 0, true);
    for (0..4) |_| try check.sameBytes("row16 State: five-repeat nonfinite SHA", &first, &(try fixture(gpu, &m, s, widths[widths.len - 1], 0, true)));
}

pub fn run(gpu: check.Gpu) !void {
    try qualify(gpu, &.{ 9, 15, 16 });
    check.pass("GLM_ROW16_STATE: TP4H16/3layers, every keep, both parities, five repeats, NaN/reset/restore", .{});
}

pub fn run32(gpu: check.Gpu) !void {
    try check.expect(W == 32, "row32 State: physical32 required", .{});
    try qualify(gpu, &.{ 17, 31, 32 });
    check.pass("GLM_ROW32_STATE: fixtures35 commits960 TP4H16/3layers, both parities, five repeats, stride control", .{});
}
