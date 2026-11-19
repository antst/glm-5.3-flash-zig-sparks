//! Keep each packed pointer's format attached while sorting the startup table.
const std = @import("std");

pub fn sort(ptrs: []u64, bits: []u8) void {
    std.debug.assert(ptrs.len == bits.len);
    const Context = struct {
        p: []u64,
        b: []u8,
        pub fn lessThan(c: @This(), a: usize, b: usize) bool {
            return c.p[a] < c.p[b];
        }
        pub fn swap(c: @This(), a: usize, b: usize) void {
            std.mem.swap(u64, &c.p[a], &c.p[b]);
            std.mem.swap(u8, &c.b[a], &c.b[b]);
        }
    };
    std.sort.pdqContext(0, ptrs.len, Context{ .p = ptrs, .b = bits });
}
