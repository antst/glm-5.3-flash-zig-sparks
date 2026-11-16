//! Agreed active verify width over one physical32-row allocation baseline.
const std = @import("std");

pub const Choice = struct { rows: u8 = 8, copy: u8 = 3 };
pub const Counts = struct { max_part_rows: usize = 0, solo_wide_calls: u64 = 0, shared_wide_calls: u64 = 0 };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !Choice {
    var c: Choice = .{};
    if (get("TF_GLM_VERIFY_ROWS")) |v| {
        if (std.mem.eql(u8, v, "32")) c.rows = 32 else if (std.mem.eql(u8, v, "16")) c.rows = 16 else if (!std.mem.eql(u8, v, "8")) {
            problem.* = "TF_GLM_VERIFY_ROWS: 8, 16 or 32";
            return error.BadSetting;
        }
    }
    if (get("TF_GLM_FIRST_COPY_TOKENS")) |v| {
        if (std.mem.eql(u8, v, "7")) c.copy = 7 else if (std.mem.eql(u8, v, "15") and c.rows >= 16)
            c.copy = 15
        else if (std.mem.eql(u8, v, "31") and c.rows == 32) c.copy = 31 else if (!std.mem.eql(u8, v, "3")) {
            problem.* = "TF_GLM_FIRST_COPY_TOKENS: 3/7, 15 requires width>=16, 31 requires width32";
            return error.BadSetting;
        }
    }
    return c;
}

/// Before allocation; the physical32 baseline is shared by all active widths.
pub fn require(s: anytype, world: usize, vision: bool, drafter: bool, drafts: bool) !void {
    if ((s.verify_rows != 8 and s.verify_rows != 16 and s.verify_rows != 32) or
        (s.first_copy_tokens != 3 and s.first_copy_tokens != 7 and
            !(s.first_copy_tokens == 15 and s.verify_rows >= 16) and
            !(s.first_copy_tokens == 31 and s.verify_rows == 32))) return error.BadVerifyRows;
    if (s.verify_rows == 8) return;
    if ((world != 2 and world != 4) or s.pages or s.kv_host != 0 or vision or s.verify_graphs)
        return error.VerifyRowsScope;
    if (!drafter or !drafts) return error.VerifyRowsNeedsDFlash;
}

/// Rank zero checks this before selecting a slot; plain single-row sampling keeps its old path.
pub fn request(rows: u8, drafts: bool, sampled: bool, grammar: bool) !void {
    if (rows >= 16 and drafts and (sampled or grammar)) return error.VerifyRowsNeedsGreedyDrafts;
}

/// Direct Engine callers must also refuse before the target is enqueued.
pub fn window(limit: u8, rows: usize, sampled: bool, grammar: bool) !void {
    if (rows == 0 or rows > limit) return error.WindowTooWide;
    try request(limit, rows > 1, sampled, grammar);
}

pub fn startup(s: anytype, rank: usize) void {
    std.log.info("rank {d}: GLM_VERIFY_ROWS active={d} physical=32 new_memory_baseline=true copy_tokens={d}", .{ rank, s.verify_rows, s.first_copy_tokens });
}

/// Called only after the real target and draw returned successfully; no per-call logging.
pub fn solo(e: anytype, rows: usize) void {
    if (e.settings.verify_rows < 16) return;
    e.row_counts.max_part_rows = @max(e.row_counts.max_part_rows, rows);
    if (rows > 8) e.row_counts.solo_wide_calls += 1;
}

pub fn shared(e: anytype, parts: anytype) void {
    if (e.settings.verify_rows < 16) return;
    var most: usize = 0;
    for (parts) |p| most = @max(most, p.ids.len);
    e.row_counts.max_part_rows = @max(e.row_counts.max_part_rows, most);
    if (most > 8) e.row_counts.shared_wide_calls += 1;
}

pub fn close(e: anytype) void {
    if (e.settings.verify_rows < 16) return;
    std.log.info("rank {d}: GLM_VERIFY_ROWS_CLOSE max_part_rows={d} solo_wide_calls={d} shared_wide_calls={d} neural_max=7 copy_tokens={d}", .{ e.share.rank, e.row_counts.max_part_rows, e.row_counts.solo_wide_calls, e.row_counts.shared_wide_calls, e.settings.first_copy_tokens });
}
