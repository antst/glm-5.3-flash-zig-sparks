//! Mia 0103 quoted-opener and nested-value parser regressions.
const std = @import("std");
const json = @import("json");
const parse = @import("tool_parse.zig").parse;
const closeCall = @import("tool_parse.zig").closeCall;
const Timed = @import("tool_parse_test.zig").Timed;

test "prose that quotes the opener ahead of a call does not swallow the call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"},\"q\":{\"type\":\"string\"}}}}}]")).ok.array;
    // the model names the labs' formats in its prose, then calls in its own (GLM's, Qwen's JSON, XML)
    const prose = "Each lab's format differs (<minimax:tool_call>, GLM's xml-ish, Qwen's <tool_call>{})...";
    const calls = [_][]const u8{
        "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":7}}</tool_call>",
        "<tool_call>\n<function=lookup>\n<parameter=n>\n7\n</parameter>\n</function>\n</tool_call>",
    };
    for (calls) |call| {
        const text = try std.mem.concat(a, u8, &.{ prose, call });
        for ([_]?usize{ null, 1 }) |max_calls| {
            const r = try parse(a, text, tools, max_calls);
            try std.testing.expectEqual(@as(usize, 0), r.rejected_envelopes);
            try std.testing.expectEqual(@as(usize, 1), r.calls.?.len);
            try std.testing.expectEqualStrings("lookup", r.calls.?[0].get("function").?.get("name").?.string);
            try std.testing.expectEqualStrings("{\"n\":7}", r.calls.?[0].get("function").?.get("arguments").?.string);
            try std.testing.expectEqualStrings(prose, r.content);
        }
    }
    // a value that quotes the opener parses whole (no retry); quoted ahead of such a call, the retry goes one opener back
    const quoting = "<tool_call>lookup<arg_key>q</arg_key><arg_value>grep -c '<tool_call>' log</arg_value></tool_call>";
    for ([_][]const u8{ quoting, try std.mem.concat(a, u8, &.{ prose, quoting }) }) |text| {
        const r = try parse(a, text, tools, null);
        try std.testing.expectEqual(@as(usize, 0), r.rejected_envelopes);
        try std.testing.expectEqualStrings("{\"q\":\"grep -c '<tool_call>' log\"}", r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    // the call found that way still has to name an offered tool, and (one call asked) parse whole, as before
    const kept = [_]struct { []const u8, ?usize }{
        .{ "Qwen's <tool_call>{}... <tool_call>launch<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", null },
        .{ "Qwen's <tool_call>{}... <tool_call>launch<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", 1 },
        .{ "Qwen's <tool_call>{}... <tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value> and then</tool_call>", 1 },
        .{ "Qwen's <tool_call>{}... <tool_call></tool_call>", null },
    };
    for (kept) |c| {
        const r = try parse(a, c[0], tools, c[1]);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqual(@as(usize, 1), r.rejected_envelopes);
        try std.testing.expectEqualStrings(c[0], r.content);
    }
    // an earlier opener no longer takes the closer that closes the call the end token left open
    const tags = "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}";
    try std.testing.expectEqualStrings("</tool_call>", try closeCall(a, tags, tools));
    const tagged = try parse(a, try std.mem.concat(a, u8, &.{ tags, "</tool_call>" }), tools, null);
    try std.testing.expectEqualStrings("lookup", tagged.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expectEqualStrings("Trying <tool_call> tags.", tagged.content);
    // a call after the quoted opener that the end token left open closes as before, and parses once closed
    const open = "Qwen's <tool_call>{}... <tool_call>lookup<arg_key>n</arg_key><arg_value>7";
    const close = try closeCall(a, open, tools);
    try std.testing.expectEqualStrings("</arg_value></tool_call>", close);
    const closed = try parse(a, try std.mem.concat(a, u8, &.{ open, close }), tools, null);
    try std.testing.expectEqualStrings("{\"n\":7}", closed.calls.?[0].get("function").?.get("arguments").?.string);
    try std.testing.expectEqualStrings("Qwen's <tool_call>{}...", closed.content);
    // an unoffered call stays open, as before
    try std.testing.expectEqualStrings("", try closeCall(a, "Qwen's <tool_call>{}... <tool_call>launch<arg_key>n</arg_key><arg_value>7", tools));
}

test "an envelope of many quoted openers is rejected in linear time" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    var t: Timed = .{ .replies = undefined, .tools = tools, .want = null };
    for (&t.replies, [_]usize{ 800, 6400 }) |*reply, n| {
        var r: std.ArrayList(u8) = .empty;
        for (0..n) |_| try r.appendSlice(a, "<tool_call>{} ");
        try r.appendSlice(a, "</tool_call>");
        reply.* = r.items;
    }
    try t.linear(); // 8x the openers: near 8x the time, not 64x
}

test "a cut call whose value quotes a whole call stays content: the retry is for prose ahead of a call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"},\"q\":{\"type\":\"string\"}}}}}]")).ok.array;
    // the first </tool_call> cuts the outer call inside its value; retrying from the inner opener would call lookup(n=7).
    // The JSON and XML forms stay content (rejected, as before the retry); GLM's lenient form takes the cut outer call
    // with its q truncated at the inner closer, as it did before - never the quoted inner call.
    const cut = [_][]const u8{
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>\"}}</tool_call>",
        "<tool_call>lookup<arg_key>q</arg_key><arg_value><tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call></arg_value></tool_call>",
        "<tool_call>\n<function=lookup>\n<parameter=q>\n<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>\n</parameter>\n</function>\n</tool_call>",
        "Note: <tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>\"}}</tool_call>",
    };
    for (cut) |text| {
        for ([_]?usize{ null, 1 }) |max_calls| {
            const r = try parse(a, text, tools, max_calls);
            if (r.calls) |c| {
                const args = c[0].get("function").?.get("arguments").?.string;
                try std.testing.expect(!std.mem.eql(u8, args, "{\"n\":7}")); // the quoted inner call must never be the call
            } else {
                try std.testing.expectEqual(@as(usize, 1), r.rejected_envelopes);
            }
        }
    }
}
