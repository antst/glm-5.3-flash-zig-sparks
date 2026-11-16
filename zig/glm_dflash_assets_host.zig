//! Inspect actual cubin records on the CPU; no Driver, module load or device execution.
const std = @import("std");
const contract = @import("core").contract;
const limits = contract.limits;
const wide = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.NeedFatbinDirectory;
    var cases: usize = 0;
    for ([_][]const u8{ "glm_dflash", "qmm_group" }) |unit| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ args[1], unit });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 26));
        const imgs = try contract.cubin.images(a, bytes);
        var seen = std.StaticBitSet(256).empty;
        for (imgs) |im| {
            if (im.sm >= 256 or seen.isSet(im.sm)) return error.DuplicateOrUnknownSm;
            seen.set(im.sm);
            const target = limits.target(im.sm) orelse return error.UnknownSm;
            const kernels = try contract.cubin.kernels(a, im.elf, im.sm);
            const want: []const []const u8 = if (std.mem.eql(u8, unit, "glm_dflash"))
                &.{ "df_block_qkv_tp4_v1", "df_attn_part_tp4_v1", "df_attn_merge_tp4_v1" }
            else
                &.{wide};
            for (want, 0..) |symbol, i| {
                const k = for (kernels) |k| {
                    if (std.mem.eql(u8, k.name, symbol)) break k;
                } else return error.MissingNewSymbol;
                const q4 = std.mem.eql(u8, unit, "qmm_group");
                const threads: u32 = if (q4) 128 else if (i == 0) 64 else if (i == 1) 256 else 128;
                const clusters: []const u32 = if (q4 and target.clusters) &.{ 1, 2, 4, 8 } else &.{1};
                for (clusters) |cluster| {
                    var out: std.ArrayList(u8) = .empty;
                    if (try limits.cuda(.{
                        .name = symbol,
                        .symbol = symbol,
                        .image = unit,
                        .threads = threads,
                        .dynamic = if (q4) 35328 else 0,
                        .cluster = cluster,
                        .opt_in = q4,
                    }, k.usage, target, &out, a) != 0) {
                        std.debug.print("{s}", .{out.items});
                        return error.CompiledLaunchDoesNotFit;
                    }
                    cases += 1;
                }
                std.debug.print("{{\"image\":\"{s}\",\"sm\":{d},\"symbol\":\"{s}\",\"regs\":{d},\"static\":{d},\"bound\":{d},\"threads\":{d}}}\n", .{ unit, im.sm, symbol, k.usage.regs, k.usage.static, k.usage.bound, threads });
            }
        }
        for ([_]usize{ 80, 86, 89, 120, 121 }) |sm| if (!seen.isSet(sm)) return error.MissingSm;
        if (seen.count() != 5) return error.UnexpectedSm;
    }
    std.debug.print("{{\"passed\":true,\"compiled_launch_cases\":{d},\"device_executed\":false}}\n", .{cases});
}
