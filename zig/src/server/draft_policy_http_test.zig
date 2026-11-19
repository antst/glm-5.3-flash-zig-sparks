//! Actual HTTP request preparation preserves sampling/grammar and reports the effective draft choice.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const pre = @import("stream_preflight_test.zig");
const Conn = @import("http_conn.zig").Conn;

const Fake = struct {
    restricted: bool = true,
    submits: usize = 0,
    drafts: bool = false,
    sampled: bool = false,
    structured: bool = false,

    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .structures = true, .greedy_unconstrained_drafts = f.restricted };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.submits += 1;
        f.drafts = request.drafts;
        f.sampled = request.sampling != null;
        f.structured = request.structure != null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{'x'} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
};

// Only grammar compilation is faked: the real server owns the returned compile through request teardown.
const Compile = struct {
    fn compiler(c: *Compile) api.grammar.Compiler {
        var out: api.grammar.Compiler = undefined;
        out.h = c;
        out.words = 4;
        out.f.compile = compile;
        out.f.free = free;
        out.f.close = free;
        return out;
    }
    fn compile(ptr: *anyopaque, _: c_int, _: [*]const u8, _: usize, _: [*]u8, _: usize) callconv(.c) ?*anyopaque { return ptr; }
    fn free(_: *anyopaque) callconv(.c) void {}
};

const Got = struct {
    opened: bool = false,
    status: ?u16 = null,
    fn out(g: *Got) openai.Out { return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } }; }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.opened = true;
    }
    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}
    fn reply(ctx: *anyopaque, status: u16, _: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status;
    }
};

const Case = struct {
    extra: []const u8,
    restricted: bool = true,
    refused: bool = false,
    drafts: bool = false,
    sampled: bool = false,
    structured: bool = false,
};

fn check(c: Case, stream: bool) !void {
    var fake: Fake = .{ .restricted = c.restricted };
    var text: pre.TestText = .{};
    var compiler_owner: Compile = .{};
    var compiler = compiler_owner.compiler();
    const srv = try server.Server.init(std.testing.allocator, std.testing.io, fake.engine(), text.text(),
        .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = true }, null);
    defer srv.deinit();
    srv.grammars = &compiler;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = try std.fmt.allocPrint(a,
        "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":8,\"stream\":{}{s}}}",
        .{ stream, c.extra });
    const value = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var got: Got = .{};
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, value);
    if (c.refused) {
        try std.testing.expectEqual(@as(?u16, 400), got.status);
        try std.testing.expect(!got.opened);
        try std.testing.expectEqual(@as(usize, 0), fake.submits);
    } else {
        try std.testing.expect(got.status == 200 or (stream and got.opened));
        try std.testing.expectEqual(@as(usize, 1), fake.submits);
        try std.testing.expectEqual(c.drafts, fake.drafts);
        try std.testing.expectEqual(c.sampled, fake.sampled);
        try std.testing.expectEqual(c.structured, fake.structured);
    }
}

test "draft policy HTTP: greedy defaults and explicit off retain their existing target path" {
    for ([_]bool{ false, true }) |stream| {
        try check(.{ .extra = ",\"temperature\":0", .drafts = true }, stream);
        try check(.{ .extra = ",\"temperature\":0,\"draft\":false" }, stream);
        try check(.{ .extra = ",\"temperature\":0,\"draft\":true", .drafts = true }, stream);
    }
}

test "draft policy HTTP: sampled and constrained defaults preserve those features using plain decoding" {
    for ([_]bool{ false, true }) |stream| {
        try check(.{ .extra = ",\"temperature\":0.7", .sampled = true }, stream);
        try check(.{ .extra = ",\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}", .structured = true }, stream);
        try check(.{ .extra = ",\"temperature\":0.7,\"draft\":false", .sampled = true }, stream);
    }
}

test "draft policy HTTP: explicit incompatible drafting refuses before HTTP stream or engine submit" {
    for ([_]bool{ false, true }) |stream| {
        try check(.{ .extra = ",\"temperature\":0.7,\"draft\":true", .refused = true }, stream);
        try check(.{ .extra = ",\"temperature\":0,\"draft\":true,\"response_format\":{\"type\":\"json_object\"}", .refused = true }, stream);
    }
}

test "draft policy HTTP: engines without the constraint retain sampled drafting" {
    for ([_]bool{ false, true }) |stream|
        try check(.{ .extra = ",\"temperature\":0.7", .restricted = false, .drafts = true, .sampled = true }, stream);
}
