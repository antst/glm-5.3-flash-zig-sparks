//! Exact stored-weight FP64 oracle; finite fixtures repeat over217 columns and5 input rows.
const std = @import("std");
const core = @import("core");
const glm = @import("glm_cuda");
const c = core.contract.spec;
pub const K: usize = 4096;
pub const period: usize = 217;
pub const xr: usize = 5;

pub fn weight(row: usize, col: usize) u16 {
    const tag = row % period;
    const q = @as(i32, @intCast((tag * 13 + col * 7 + (col / 128) * 3) % 31)) - 15;
    const divisor: f32 = @floatFromInt(@as(u32, 1) << @as(u5, @intCast(4 + (tag + col / 128) % 7)));
    return c.toBf16(@as(f32, @floatFromInt(q)) / divisor);
}

pub fn input(row: usize, col: usize) u16 {
    const q = @as(i32, @intCast(((row % xr) * 11 + col * 7 + col / 128) % 31)) - 15;
    return c.toBf16(@as(f32, @floatFromInt(q)) / 32);
}

fn depth(_: c.Shape) usize {
    return 64 + K / 64;
}

pub fn spec(out: c.Elem) c.Spec {
    return .{ .name = "group64-scaled FP8 head through FP8G64", .accum = .f32, .depth = depth, .out = out, .served = &.{.{ .rows = 64, .n = 38784, .k = K }} };
}

pub const Oracle = struct {
    a: std.mem.Allocator,
    columns: usize,
    codes: []u8,
    scales: []f32,
    sums: []f64,
    mag: []f64,

    pub fn init(a: std.mem.Allocator, n: usize) !Oracle {
        const cols: usize = @min(n, period);
        const codes = try a.alloc(u8, cols * K);
        errdefer a.free(codes);
        const scales = try a.alloc(f32, cols * (K / 64));
        errdefer a.free(scales);
        const sums = try a.alloc(f64, xr * cols);
        errdefer a.free(sums);
        const mag = try a.alloc(f64, xr * cols);
        errdefer a.free(mag);
        const result: Oracle = .{ .a = a, .columns = cols, .codes = codes, .scales = scales, .sums = sums, .mag = mag };
        for (0..cols) |j| for (0..K / 64) |g| {
            var src: [64]u16 = undefined;
            for (&src, 0..) |*v, i| v.* = weight(j, g * 64 + i);
            const q = try glm.head_fp8_reference.quantize64(src);
            @memcpy(codes[(j * (K / 64) + g) * 64 ..][0..64], &q.codes);
            scales[j * (K / 64) + g] = q.scale;
        };
        for (0..xr) |r| for (0..cols) |j| {
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (0..K) |k| {
                const product = @as(f64, c.fromBf16(input(r, k))) * result.at(j, k);
                sum += product;
                magnitude += @abs(product);
            }
            sums[r * cols + j] = sum;
            mag[r * cols + j] = magnitude;
        };
        return result;
    }

    pub fn deinit(o: *Oracle) void {
        inline for (.{ o.codes, o.scales, o.sums, o.mag }) |v| o.a.free(v);
    }

    pub fn at(o: Oracle, j: usize, k: usize) f64 {
        const tag = j % period;
        return core.quant.fp8.e4m3(o.codes[tag * K + k]) * o.scales[tag * (K / 64) + k / 64];
    }

    pub fn want(o: Oracle, r: usize, j: usize, nonfinite: bool) f64 {
        if (nonfinite and r < 3) {
            const w = o.at(j, 17);
            if (r == 0 or w == 0) return std.math.nan(f64);
            return std.math.copysign(std.math.inf(f64), if (r == 1) w else -w);
        }
        return o.sums[(r % xr) * o.columns + j % period];
    }

    pub fn bound(o: Oracle, r: usize, j: usize, out: c.Elem) f64 {
        const index = (r % xr) * o.columns + j % period;
        return c.bound(spec(out), .{ .rows = 64, .n = 38784, .k = K }, o.sums[index], o.mag[index]);
    }
};
