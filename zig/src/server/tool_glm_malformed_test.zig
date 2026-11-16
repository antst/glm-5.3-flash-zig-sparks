//! Malformed GLM call values retain text instead of consuming another argument.
const std = @import("std");
const json = @import("json");
const parse = @import("tool_parse.zig").parse;

test "a GLM value that runs into the next pair is text, not a call" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"session_search\",\"parameters\":{\"properties\":{\"query\":{\"type\":\"string\"},\"arguments\":{\"type\":\"string\"}}}}}]")).ok.array;
    // #2801: a missing </arg_value> read the next pair's markup, and a last value may never close at all
    const cases = [_][2][]const u8{ // the reply, and the arguments of the call it makes ("" keeps its text)
        .{ "<tool_call>session_search<arg_key>query</arg_key><arg_value>decided<arg_key>arguments</arg_key><arg_value>{\"query\": \"decided\", \"limit\": 10}</arg_value></tool_call>", "" },
        .{ "<tool_call>session_search<arg_key>query</arg_key><arg_value>decided</arg_value><arg_key>arguments</arg_key><arg_value>{\"limit\": 10}</tool_call>", "" },
        .{ "<tool_call>session_search<arg_key>query</arg_key><arg_value>2 < 3</arg_value></tool_call>", "{\"query\":\"2 < 3\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        if (c[1].len == 0) {
            try std.testing.expect(r.calls == null);
            try std.testing.expectEqualStrings(c[0], r.content);
        } else try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
}
