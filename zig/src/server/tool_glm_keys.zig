//! Restore missing GLM argument key openers before decoding the tool call.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;
const space = reply_text.isSpaceByte;

const GlmTag = enum { key_open, key_close, value_open, value_close };
const glm_tags = [_]struct { GlmTag, []const u8 }{ .{ .key_open, "<arg_key>" }, .{ .key_close, "</arg_key>" }, .{ .value_open, "<arg_value>" }, .{ .value_close, "</arg_value>" } };

/// The next GLM argument tag at or after `from`: which, and where it starts and ends.
fn nextGlmTag(block: []const u8, from: usize) ?struct { tag: GlmTag, start: usize, end: usize } {
    var i = std.mem.indexOfScalarPos(u8, block, from, '<') orelse return null;
    while (true) : (i = std.mem.indexOfScalarPos(u8, block, i + 1, '<') orelse return null) {
        for (glm_tags) |t| if (std.mem.startsWith(u8, block[i..], t[1])) return .{ .tag = t[0], .start = i, .end = i + t[1].len };
    }
}

/// A GLM call whose ``<arg_key>`` the model wrote as whitespace gets it back before the word ahead of each lone tag.
pub fn repairGlmKeys(a: Allocator, block: []const u8) Allocator.Error![]const u8 {
    if (std.mem.count(u8, block, "</arg_key>") <= std.mem.count(u8, block, "<arg_key>")) return block;
    var out: std.ArrayList(u8) = .empty;
    var copied: usize = 0;
    var prev: ?GlmTag = null;
    var prev_end: usize = 0;
    var pos: usize = 0;
    while (nextGlmTag(block, pos)) |t| : (pos = t.end) {
        if (t.tag == .key_close and (prev == null or prev.? == .value_close)) {
            const seg = prev_end; // 0 before the first tag: the call's name lives there
            var e = t.start;
            while (e > seg and space(block[e - 1])) e -= 1;
            var w = e;
            while (w > seg and !space(block[w - 1])) w -= 1;
            // no word, or (before any tag) no name left ahead of the word: not a key we can put back
            if (w == e or (prev == null and reply_text.pyStrip(block[seg..w]).len == 0)) return block;
            try out.appendSlice(a, block[copied..w]);
            try out.appendSlice(a, "<arg_key>");
            copied = w;
        }
        prev = t.tag;
        prev_end = t.end;
    }
    try out.appendSlice(a, block[copied..]);
    return out.items;
}

/// The first ``</arg_key>`` at or after `from` that an ``<arg_value>`` follows, and where that value starts.
pub fn valueKey(rest: []const u8, from: usize) ?[2]usize {
    if (from > rest.len) return null;
    var at = from;
    while (std.mem.indexOfPos(u8, rest, at, "</arg_key>")) |ke| : (at = ke + 1) {
        var j = ke + 10;
        while (j < rest.len and space(rest[j])) j += 1;
        if (std.mem.startsWith(u8, rest[j..], "<arg_value>")) return .{ ke, j + 11 };
    }
    return null;
}
