//! The actual sampled draw/gather against recording CUDA and byte movement, not sampler arithmetic emulation.
const std = @import("std");
const cuda = @import("cuda");
const smp = @import("cuda_sampler.zig");
const batch = @import("cuda_sample_batch.zig");
const ranks = @import("cuda_ranks.zig");
const rec = @import("cuda_sample_batch_record.zig");
const F = rec.Fake;
const t = std.testing;
const spans4 = [_]rec.Span{ .{ .lo = 0, .hi = 38784 }, .{ .lo = 38784, .hi = 77568 }, .{ .lo = 77568, .hi = 116224 }, .{ .lo = 116224, .hi = 154880 } };

fn inputs(spans: []const rec.Span, rows: usize) !void {
    for (spans, 0..) |span, q| {
        F.shards[q] = try t.allocator.alloc(u16, rows * span.len());
        for (F.shards[q], 0..) |*v, i| v.* = @truncate(i + 7919 * q);
    }
}
fn freeInputs(spans: []const rec.Span) void {
    for (0..spans.len) |q| t.allocator.free(F.shards[q]);
}
fn unchanged(spans: []const rec.Span) !void {
    for (0..spans.len) |q| for (F.shards[q], 0..) |v, i| try t.expectEqual(@as(u16, @truncate(i + 7919 * q)), v);
}
fn bind(s: *const smp.Sampler) void {
    F.pad = s.batch.?.pad.ptr;
    F.got = s.batch.?.got.ptr;
}
fn reference(s: *smp.Sampler, stream: cuda.Stream, ex: @import("cuda_forward.zig").Exchange, rows: usize, host: []u16) !void {
    F.rows = rows;
    F.reset();
    try s.gather(stream, ex, F.spans, &.{.{ .rank = F.rank, .logits = @intFromPtr(F.shards[F.rank].ptr) }}, rows, s.stage.ptr, s.full.ptr);
    try t.expectEqual(F.spans.len * 2, F.n);
    try stream.synchronize();
    @memcpy(host, @as([*]const u16, @ptrFromInt(s.full.ptr))[0..host.len]);
    F.expected = host;
    F.reset();
}

test "sample batching: actual draw retains BF16 bits, actual-row planes, Rule/meta and one checked readback" {
    var d = F.driver();
    const lib = F.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    F.spans = &spans4;
    F.width = 38784;
    F.cap = 32;
    F.reset();
    try inputs(&spans4, 32);
    defer freeInputs(&spans4);
    defer cuda.download_stage.forget();
    const host = try t.allocator.alloc(u16, 32 * 154880);
    defer t.allocator.free(host);
    F.rule = smp.rule(.{ .seed = 0x123456789abcdef, .temperature = 0.73, .top_k = 17, .top_p = 0.91 });
    for (0..4) |q| {
        F.rank = q;
        var s = try F.sampler(&d, 154880);
        defer s.deinit();
        try s.enableBatch(stream, &spans4, q);
        bind(&s);
        try t.expectEqual(@as(usize, 12410880), try s.batch.?.plan.scratchBytes());
        var g = F.group(lib);
        for ([_]bool{ false, true }) |borrowed| for ([_]usize{ 32, 17, 31, 16, 1, 15, 2, 9, 3, 8, 4, 7, 5, 6 }) |rows| {
            const ex: @import("cuda_forward.zig").Exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .group = if (borrowed) &g else null } };
            try reference(&s, stream, ex, rows, host[0 .. rows * 154880]);
            F.first = 123456 + rows;
            var out: [34]u32 = @splat(0xdeadbeef);
            try s.draw(stream, ex, &spans4, &.{.{ .rank = q, .logits = @intFromPtr(F.shards[q].ptr) }}, rows, F.first, F.rule, out[1 .. 1 + rows]);
            const tail: []const rec.Kind = if (borrowed) &.{ .rule, .meta, .draw, .sync, .check, .download } else &.{ .rule, .meta, .draw, .sync, .download };
            try t.expectEqual(rec.Kind.pack, F.calls[0]);
            try t.expectEqual(rec.Kind.gather, F.calls[1]);
            for (F.calls[2..6]) |kind| try t.expectEqual(rec.Kind.unpack, kind);
            try t.expectEqualSlices(rec.Kind, tail, F.calls[6..F.n]);
            try t.expectEqual(@as(usize, @intFromBool(borrowed and rows <= 6)), F.borrowed);
            try t.expectEqual(@as(usize, @intFromBool(!borrowed or rows > 6)), F.raw);
            for (0..rows) |r| try t.expectEqual(@as(u32, @intCast(42 + r)), out[1 + r]);
            try t.expectEqual(@as(u32, 0xdeadbeef), out[0]);
            try t.expectEqual(@as(u32, 0xdeadbeef), out[1 + rows]);
            try unchanged(&spans4);
        };
    }
}

test "sample batching: pack gather unpack upload draw wait check and readback errors publish no tokens" {
    var d = F.driver();
    const lib = F.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    F.reset();
    F.spans = &spans4;
    F.width = 38784;
    F.rank = 2;
    F.cap = 8;
    try inputs(&spans4, 8);
    defer freeInputs(&spans4);
    defer cuda.download_stage.forget();
    const host = try t.allocator.alloc(u16, 3 * 154880);
    defer t.allocator.free(host);
    var s = try F.sampler(&d, 154880);
    defer s.deinit();
    try s.enableBatch(stream, &spans4, F.rank);
    bind(&s);
    var g = F.group(lib);
    const ex: @import("cuda_forward.zig").Exchange = .{ .nccl = .{ .lib = &lib, .comm = null, .group = &g } };
    try reference(&s, stream, ex, 3, host);
    F.first = 999;
    F.rule = smp.rule(.{ .seed = 87, .temperature = 1, .top_p = 0.8 });
    for ([_]rec.Fail{ .pack, .gather, .unpack, .rule, .meta, .draw, .sync, .check, .download }) |fail| {
        F.reset();
        F.fail = fail;
        var out: [3]u32 = @splat(0xdeadbeef);
        const expected = switch (fail) {
            .gather => error.NcclFailed,
            .check => error.RecordedWireFailure,
            else => error.NotReady,
        };
        try t.expectError(expected, s.draw(stream, ex, &spans4, &.{.{ .rank = F.rank, .logits = @intFromPtr(F.shards[F.rank].ptr) }}, 3, F.first, F.rule, &out));
        try t.expectEqualSlices(u32, &@as([3]u32, @splat(0xdeadbeef)), &out);
        if (fail == .gather) try t.expectEqual(@as(usize, 0), F.raw); // no raw retry after admitted failure
        F.fail = .none;
        try stream.synchronize(); // the engine still owns scratch on every error path
    }
}

test "sample batching: partition rows rank pointer capacity and count overflow fail before enqueues" {
    var d = F.driver();
    const lib = F.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    F.reset();
    F.spans = &spans4;
    F.rank = 0;
    var s = try F.sampler(&d, 154880);
    defer s.deinit();
    try s.enableBatch(stream, &spans4, 0);
    bind(&s);
    F.reset();
    const ex: @import("cuda_forward.zig").Exchange = .{ .nccl = .{ .lib = &lib, .comm = null } };
    const mine: smp.Local = .{ .rank = 0, .logits = 0x10000 };
    var out: [8]u32 = undefined;
    for ([_]usize{ 0, 9 }) |rows| try t.expectError(error.BadSampleBatchRows, s.draw(stream, ex, &spans4, &.{mine}, rows, 1, undefined, &out));
    try t.expectError(error.BadSampleBatchRows, s.draw(stream, ex, &spans4, &.{mine}, 2, 1, undefined, out[0..1]));
    try t.expectError(error.BadSampleBatchRows, s.draw(stream, ex, &spans4, &.{mine}, 1, 0, undefined, &out));
    try t.expectError(error.BadSampleBatchRows, s.draw(stream, ex, &spans4, &.{mine}, 1, 0xffffffff, undefined, &out));
    try t.expectError(error.BadSampleBatchRanks, s.batch.?.gather(stream, ex, &spans4, .{ .rank = 1, .logits = 0x10000 }, 1, s.full));
    var bad = spans4;
    bad[2].lo += 1;
    try t.expectError(error.BadSampleBatchSpans, s.batch.?.gather(stream, ex, &bad, mine, 1, s.full));
    try t.expectError(error.Overflow, s.batch.?.gather(stream, ex, &spans4, .{ .rank = 0, .logits = std.math.maxInt(u64) - 3 }, 1, s.full));
    var short = s.full;
    short.len = 1;
    try t.expectError(error.SampleBatchScratchTooSmall, s.batch.?.gather(stream, ex, &spans4, mine, 1, short));
    const wide = [_]rec.Span{ .{ .lo = 0, .hi = std.math.maxInt(usize) - 1 }, .{ .lo = std.math.maxInt(usize) - 1, .hi = std.math.maxInt(usize) } };
    try t.expectError(error.Overflow, batch.Plan.init(std.math.maxInt(usize), 8, 0, &wide));
    try t.expectError(error.BadSampleBatchSpans, batch.Plan.init(154880, 8, 0, &bad));
    try t.expectEqual(@as(usize, 0), F.n);
}

test "sample batching: allocation fill and startup-wait failures drain before freeing owned scratch" {
    var d = F.driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    for (0..4) |which| {
        F.reset();
        const start = cuda.memory.usage(false).device;
        const frees_before = F.frees;
        if (which < 2) F.fail_alloc = F.allocations + which + 1 else F.fail = if (which == 2) .fill else .sync;
        try t.expectError(error.NotReady, batch.Batch.init(&d, stream, 154880, 8, 0, &spans4));
        try t.expect(!F.pending);
        try t.expectEqual(start, cuda.memory.usage(false).device);
        try t.expectEqual(frees_before + @as(usize, if (which < 2) which else 2), F.frees);
    }
    F.reset();
    var b = try batch.Batch.init(&d, stream, 154880, 8, 0, &spans4);
    b.deinit();
}

test "sample batching: old gather still accepts128 external rows even when sampled scratch is enabled" {
    const spans = [_]rec.Span{ .{ .lo = 0, .hi = 4 }, .{ .lo = 4, .hi = 8 }, .{ .lo = 8, .hi = 11 }, .{ .lo = 11, .hi = 14 } };
    var d = F.driver();
    const lib = F.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    F.reset();
    F.spans = &spans;
    F.rank = 1;
    F.cap = 8;
    F.width = 4;
    try inputs(&spans, 128);
    defer freeInputs(&spans);
    var s = try F.sampler(&d, 14);
    defer s.deinit();
    try s.enableBatch(stream, &spans, F.rank);
    var external = try cuda.DeviceBuffer.alloc(&d, 128 * 14 * 2);
    defer external.free();
    var stage = try cuda.DeviceBuffer.alloc(&d, 128 * 14 * 2);
    defer stage.free();
    F.full = external.ptr;
    F.stage = stage.ptr;
    F.rows = 128;
    F.reset();
    const ex: @import("cuda_forward.zig").Exchange = .{ .nccl = .{ .lib = &lib, .comm = null } };
    try s.gather(stream, ex, &spans, &.{.{ .rank = F.rank, .logits = @intFromPtr(F.shards[F.rank].ptr) }}, 128, stage.ptr, external.ptr);
    try t.expectEqualSlices(rec.Kind, &.{ .broadcast, .broadcast, .broadcast, .broadcast, .place, .place, .place, .place }, F.calls[0..F.n]);
    try stream.synchronize();
    for (spans, 0..) |span, q| for (0..128) |row| {
        const full: [*]const u16 = @ptrFromInt(external.ptr);
        try t.expectEqualSlices(u16, F.shards[q][row * span.len() ..][0..span.len()], full[row * 14 + span.lo ..][0..span.len()]);
    };
}

test "sample batching: disabled sampled draw and enabled local exchange retain the old paths" {
    const spans = [_]rec.Span{ .{ .lo = 0, .hi = 4 }, .{ .lo = 4, .hi = 8 }, .{ .lo = 8, .hi = 11 }, .{ .lo = 11, .hi = 14 } };
    var d = F.driver();
    const lib = F.library();
    const stream: cuda.Stream = .{ .d = &d, .handle = rec.stream_at };
    F.reset();
    F.spans = &spans;
    F.rank = 1;
    F.cap = 8;
    F.width = 4;
    try inputs(&spans, 8);
    defer freeInputs(&spans);
    defer cuda.download_stage.forget();
    var s = try F.sampler(&d, 14);
    defer s.deinit();
    var host: [3 * 14]u16 = undefined;
    const ex: @import("cuda_forward.zig").Exchange = .{ .nccl = .{ .lib = &lib, .comm = null } };
    try reference(&s, stream, ex, 3, &host);
    F.rule = smp.rule(.{ .seed = 19, .temperature = 0.5 });
    F.first = 123;
    var out: [3]u32 = undefined;
    try s.draw(stream, ex, &spans, &.{.{ .rank = F.rank, .logits = @intFromPtr(F.shards[F.rank].ptr) }}, 3, F.first, F.rule, &out);
    try t.expectEqual(@as(usize, 13), F.n);
    for (F.calls[0..4]) |kind| try t.expectEqual(rec.Kind.broadcast, kind);
    for (F.calls[4..8]) |kind| try t.expectEqual(rec.Kind.place, kind);
    try t.expectEqual(@as(usize, 0), F.raw + F.borrowed);
    try s.enableBatch(stream, &spans, F.rank);
    F.reset();
    var mine: [4]smp.Local = undefined;
    for (&mine, 0..) |*l, q| l.* = .{ .rank = q, .logits = @intFromPtr(F.shards[q].ptr) };
    try s.draw(stream, .local, &spans, &mine, 3, F.first, F.rule, &out);
    try t.expectEqualSlices(rec.Kind, &.{ .place, .place, .place, .place, .rule, .meta, .draw, .sync, .download }, F.calls[0..F.n]);
    try t.expectEqual(@as(usize, 0), F.raw + F.borrowed);
}

test "sample batching: strict default-OFF setting is fingerprinted before scratch allocation" {
    const Get = struct {
        var value: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_SAMPLE_BATCH")) value else null;
        }
    };
    defer Get.value = null;
    var problem: []const u8 = "";
    try t.expect(!(try ranks.Settings.read(Get.get, &problem)).sample_batch);
    Get.value = "0";
    try t.expect(!(try ranks.Settings.read(Get.get, &problem)).sample_batch);
    Get.value = "1";
    const on = try ranks.Settings.read(Get.get, &problem);
    try t.expect(on.sample_batch);
    Get.value = "yes";
    try t.expectError(error.BadSetting, ranks.Settings.read(Get.get, &problem));
    try t.expectEqualStrings("TF_GLM_SAMPLE_BATCH: 0 or 1", problem);
    const a = ranks.Fingerprint.of(4, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = ranks.Fingerprint.of(4, 1, on, 8192, true, @splat(0), @splat(0), @splat(0));
    try t.expectEqual(ranks.Fingerprint.Field.sample_batch, ranks.mismatch(&.{ a, b }).?.field);
}
