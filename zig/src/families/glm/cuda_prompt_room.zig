//! Settle retired snapshot sources at a prompt boundary, before sending a collective command.
const std = @import("std");
const Stream = @import("lanes").Stream;
const Marks = @import("cuda_engine_types.zig").Marks;

pub fn before(s: *Stream, marks_: ?Marks, start: ?usize) ?Marks {
    const marks = marks_ orelse return null;
    const hook = s.reuse.hook orelse return marks;
    const prepare = hook.prepare orelse return marks;
    if (start) |at| {
        if (at != 0 and std.mem.indexOfScalar(u32, marks.at, @intCast(at)) != null) prepare(hook.ptr, @intCast(at));
    } else if (marks.at.len > 0) prepare(hook.ptr, null);
    return marks;
}

pub fn interrupted(s: *Stream, marks: ?Marks, at: usize) ?Marks {
    if (s.reuse.hook) |hook| if (hook.prepare) |prepare| prepare(hook.ptr, @intCast(at));
    return marks;
}
