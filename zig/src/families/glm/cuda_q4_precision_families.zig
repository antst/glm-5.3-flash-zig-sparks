//! Architectural weight precision choices, frozen before loading; no per-request selector.
const std = @import("std");
const weights = @import("cuda_weights.zig");
const projection = @import("cuda_proj.zig");
const tile = @import("cuda_tile.zig");
const q4 = @import("cuda_q4.zig");

pub const Family = enum(u2) { mla, kda_o, ffn };

pub fn read(get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !u8 {
    const raw = get("TF_GLM_Q4_BF16_FAMILIES") orelse return 0;
    if (raw.len == 1 and raw[0] >= '0' and raw[0] <= '7') return raw[0] - '0';
    problem.* = "TF_GLM_Q4_BF16_FAMILIES: 0..7 architectural mask";
    return error.BadSetting;
}

pub fn require(settings: anytype, world: usize) !void {
    if (settings.q4_bf16_families == 0) return;
    if (settings.q4_bf16_families > 7 or world != 2 or !settings.q4_affine_mse or
        !settings.q4_activation or settings.dense_bits != 4 or !settings.tiled)
        return error.Q4Bf16FamiliesScope;
}

fn is(p: *const projection.Proj, field: *const u64) bool {
    return p.* == .bf16 and field == &p.bf16;
}

/// Actual field addresses distinguish roles, including when stored values happen to alias.
pub fn which(w: *const weights.Weights, field: *const u64) ?Family {
    for (w.layers) |*l| {
        switch (l.mixer) {
            .mla => |*x| {
                if (is(&x.proj, field) or is(&x.q_b, field) or is(&x.o, field) or
                    is(&x.index.kw, field) or is(&x.index.qb, field)) return .mla;
            },
            .kda => |*x| if (is(&x.o, field)) return .kda_o,
        }
        switch (l.ffn) {
            .dense => |*x| switch (x.w) {
                .bf16 => |*b| if (field == &b.gu or field == &b.down) return .ffn,
                else => {},
            },
            .moe => |*x| if (is(&x.shared_gu, field) or is(&x.shared_down, field)) return .ffn,
        }
    }
    return null;
}

pub fn selected(w: *const weights.Weights, field: *const u64, mask: u8) ?Family {
    if (mask == 0) return null;
    const f = which(w, field) orelse return null;
    return if (mask & (@as(u8, 1) << @backingInt(f)) != 0) f else null;
}

/// Startup-only counters; no model-sized metadata or extra forward scratch.
pub const Census = struct {
    matrices: [3]usize = @splat(0),
    bytes: [3]usize = @splat(0),
    extra: [3]usize = @splat(0),
    groups: [3]usize = @splat(0),

    pub fn note(c: *Census, f: Family, l: tile.Lin) void {
        const i = @backingInt(f);
        const sizes = q4.bytes(l.n, l.k);
        c.matrices[i] += 1;
        c.bytes[i] += l.n * l.k * 2;
        c.extra[i] += (l.n * l.k * 2) -| (sizes[0] + sizes[1] + sizes[2]);
        c.groups[i] += l.n * l.k / 64;
    }

    pub fn report(c: Census, rank: usize, mask: u8) void {
        if (mask == 0) return;
        std.log.info("GLM_Q4_BF16_FAMILIES rank={d} mask={d} matrices={d},{d},{d} bytes={d},{d},{d} extra={d},{d},{d}", .{
            rank,       mask,       c.matrices[0], c.matrices[1], c.matrices[2], c.bytes[0], c.bytes[1], c.bytes[2],
            c.extra[0], c.extra[1], c.extra[2],
        });
    }
};
