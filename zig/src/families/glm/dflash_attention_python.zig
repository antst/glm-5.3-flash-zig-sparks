//! Explicit SM121 diagnostic bridge to the exact frozen DFlash tensor-core attention asset.
const cuda = @import("cuda");
const df = @import("dflash.zig");
pub const cap = 2176;
pub const shared = 77824;

pub const Local = struct {
    module: cuda.Module,
    attention: cuda.Function,
    prepare: cuda.Function,

    pub fn load(module: cuda.Module, glue: cuda.Module) !Local {
        const attention = try module.function("_dattn_seg_kernel");
        try attention.allowDynamicShared(shared);
        return .{ .module = module, .attention = attention, .prepare = try glue.function("df_py_attention_prepare") };
    }

    pub fn run(self: Local, dr: anytype, e: anytype, streams: usize, layer: usize) !void {
        if (streams == 0 or streams > 8 or dr.world != 4 or dr.attention == null) return error.PythonAttentionScope;
        for (dr.h_starts[0..streams], dr.h_froms[0..streams]) |start, from| {
            if (start < 0 or from > @max(0, start - 2047)) return error.PythonAttentionPartialContext;
        }
        const rows = streams * 8;
        const q = dr.cin;
        const k = q + df.MAX_M * 8 * 128 * 2;
        const v = k + df.MAX_S * cap * 2 * 128 * 2;
        const end = v + df.MAX_S * cap * 2 * 128 * 2;
        if (end > dr.cin + df.MAX_CTX * df.TAP_ROW) return error.PythonAttentionScratch;
        var a: cuda.Args = .{};
        for ([_]u64{ dr.q, dr.kb, dr.vb, dr.rings, dr.starts }) |p| a.add(p);
        a.add(@as(i32, @intCast(layer)));
        a.add(@as(i32, @intCast(dr.rank * 2)));
        a.add(@as(i32, @intCast(streams)));
        for ([_]u64{ q, k, v, dr.ids }) |p| a.add(p);
        try cuda.launch.launch(self.prepare, .{
            .grid = .{ .x = cap, .y = @intCast(streams) },
            .block = .{ .x = 256 },
        }, e.stream, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ q, k, v, dr.att, dr.starts, dr.ids }) |p| b.add(p);
        b.add(@as(i32, 2047));
        b.add(df.SCALE);
        b.add(@as(i32, @intCast(rows)));
        b.add(@as(i32, @intCast(streams * cap)));
        b.add(@as(u64, 0)); // Triton global scratch, unused by this pinned asset.
        b.add(@as(u64, 0)); // Triton profile scratch, unused by this pinned asset.
        try cuda.launch.launch(self.attention, .{
            .grid = .{ .x = 2, .y = @intCast(streams) },
            .block = .{ .x = 128 },
            .shared = shared,
        }, e.stream, &b);
    }
};
