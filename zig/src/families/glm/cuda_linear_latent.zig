//! Linear-address FP8 latent planes backed by a shared physical-page pool.
const std = @import("std");
const pages = @import("cuda").linear_pages;

pub const row_bytes = 512;
pub const scale_bytes = @sizeOf(f32);

pub const Prefix = struct {
    codes: pages.Snapshot,
    scales: pages.Snapshot,
    rows: usize,

    pub fn deinit(p: *Prefix) void {
        p.scales.deinit();
        p.codes.deinit();
        p.* = undefined;
    }
};

pub const View = struct {
    codes: pages.Window,
    scales: pages.Window,
    capacity: usize,

    pub fn init(pool: *pages.Pool, capacity: usize) !View {
        var codes = try pages.Window.init(pool, try std.math.mul(usize, capacity, row_bytes));
        errdefer codes.deinit() catch {};
        const scales = try pages.Window.init(pool, try std.math.mul(usize, capacity, scale_bytes));
        return .{ .codes = codes, .scales = scales, .capacity = capacity };
    }

    pub fn deinit(v: *View) !void {
        try v.clear();
        try v.scales.deinit();
        try v.codes.deinit();
        v.* = undefined;
    }

    pub fn clear(v: *View) !void {
        try v.scales.clear();
        try v.codes.clear();
    }

    pub fn prepareWrite(v: *View, start: usize, end: usize) !void {
        if (start > end or end > v.capacity) return error.LatentRows;
        try v.codes.prepareWrite(start * row_bytes, end * row_bytes);
        try v.scales.prepareWrite(start * scale_bytes, end * scale_bytes);
    }

    pub fn borrow(v: *View, from: *const View, rows: usize) !void {
        if (rows > v.capacity or rows > from.capacity) return error.LatentRows;
        try v.codes.borrow(&from.codes, rows * row_bytes);
        errdefer v.codes.clear() catch {};
        try v.scales.borrow(&from.scales, rows * scale_bytes);
    }

    pub fn snapshot(v: *const View, rows: usize) !Prefix {
        if (rows > v.capacity) return error.LatentRows;
        var codes = try v.codes.snapshot(rows * row_bytes);
        errdefer codes.deinit();
        const scales = try v.scales.snapshot(rows * scale_bytes);
        return .{ .codes = codes, .scales = scales, .rows = rows };
    }

    pub fn restore(v: *View, prefix: *const Prefix) !void {
        if (prefix.rows > v.capacity) return error.LatentRows;
        if (v.codes.mapped != 0 or v.scales.mapped != 0) return error.WindowNotEmpty;
        try v.codes.restore(&prefix.codes);
        errdefer v.codes.clear() catch {};
        try v.scales.restore(&prefix.scales);
    }
};
