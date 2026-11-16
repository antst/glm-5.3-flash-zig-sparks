//! Native tier maintenance limits, separate from the budget that holds useful prompt states.
const std = @import("std");

pub const Mode = enum { serial, parallel };
pub const Backend = enum { off, generic, native_tiered };

/// A source-drain window scales with retired bytes, bounded independently of the useful cache's RAM budget.
pub fn progressNs(bytes: u64) u64 {
    const quantum: u64 = 64 << 20;
    const units: u64 = @min(@as(u64, 40), (bytes +| (quantum - 1)) / quantum);
    return @max(2 * std.time.ns_per_s, units * 250 * std.time.ns_per_ms);
}
pub const Limits = struct {
    mode: Mode,
    staging: u64,
    index: u64,
    min_cache_tokens: u32 = 0,

    pub fn residentBound(l: Limits) u64 {
        return l.staging + l.index + (96 << 20);
    }
};

pub fn tier(raw: ?[]const u8) !bool {
    return try backend(raw) == .native_tiered;
}

pub fn backend(raw: ?[]const u8) !Backend {
    const v = raw orelse return .generic;
    if (std.mem.eql(u8, v, "native-tiered") or std.mem.eql(u8, v, "1107")) return .native_tiered;
    if (std.mem.eql(u8, v, "0")) return .off;
    return error.BadGlmNvmeTier;
}

pub fn mode(raw: ?[]const u8) !Mode {
    const v = raw orelse return .serial;
    if (std.mem.eql(u8, v, "serial")) return .serial;
    if (std.mem.eql(u8, v, "parallel")) return .parallel;
    return error.BadGlmNvmeMode;
}

fn mib(raw: ?[]const u8, fallback: u64, least: u64) !u64 {
    const n = if (raw) |v| std.fmt.parseInt(u64, v, 10) catch return error.BadGlmNvmeTier else fallback;
    if (n < least or n > 2048) return error.BadGlmNvmeTier;
    return n << 20;
}

pub fn parse(raw_mode: ?[]const u8, staging: ?[]const u8, index: ?[]const u8, minimum: ?[]const u8) !Limits {
    const m = try mode(raw_mode);
    const n = if (minimum) |v| std.fmt.parseInt(u32, v, 10) catch return error.BadGlmNvmeMinimum else 0;
    return .{
        .mode = m,
        .staging = try mib(staging, if (m == .serial) 192 else 512, 64),
        .index = try mib(index, if (m == .serial) 64 else 256, 16),
        .min_cache_tokens = n,
    };
}

fn env(name: [:0]const u8) ?[]const u8 {
    return @import("sys").environment.Process.get(.{}, name);
}

pub fn wanted() !bool {
    return tier(env("TF_GLM_NVME_TIER"));
}

pub fn disabled() !bool {
    return try backend(env("TF_GLM_NVME_TIER")) == .off;
}

pub fn read() !Limits {
    return parse(env("TF_GLM_NVME_MODE"), env("TF_GLM_NVME_STAGING_MIB"), env("TF_GLM_NVME_INDEX_MIB"), env("TF_GLM_NVME_MIN_CACHE_TOKENS"));
}

pub fn requireCache(l: Limits, context: usize, bytes: u64, budget: u64) !void {
    if (l.min_cache_tokens == 0) return;
    if (l.min_cache_tokens > context or bytes == std.math.maxInt(u64) or bytes > budget)
        return error.NvmeCacheTooSmall;
}

pub fn reserve(l: Limits, room: u64, unified: bool, enabled: bool) !u64 {
    if (!enabled or !unified) return room;
    const need = l.residentBound();
    if (need > room) return error.NvmeMaintenanceRoom;
    return room - need;
}
