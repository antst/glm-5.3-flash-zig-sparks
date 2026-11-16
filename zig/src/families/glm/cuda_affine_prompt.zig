//! Optional Python-style Q4 prompt arithmetic; functions borrow modules retained by the caller.
const std = @import("std");
const cuda = @import("cuda");
const q4 = @import("cuda_q4.zig");

pub const shared_bytes = 62976;
pub const bf16_symbol = cuda.qlinear.symbols.prefill;
pub const f32_symbol = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";

pub const Kernels = struct {
    functions: [2]cuda.Function,

    /// BF16 comes from the shipped module; the new module is used only for its explicit FP32 instance.
    pub fn load(original: cuda.Module, wide: cuda.Module) !Kernels {
        if (original.d != wide.d) return error.PromptModuleDriverMismatch;
        const functions = [2]cuda.Function{ try original.function(bf16_symbol), try wide.function(f32_symbol) };
        for (functions) |f| try f.allowDynamicShared(shared_bytes);
        return .{ .functions = functions };
    }

    pub fn matmul(ks: Kernels, stream: cuda.Stream, x: u64, ldx: usize, q: q4.Q4, out: u64, rows: usize, wide: bool) !void {
        if (rows == 0 or rows > 2048 or q.k == 0 or q.k % 64 != 0 or q.n == 0 or q.n % 8 != 0 or
            q.npad != q4.npadOf(q.n) or ldx < q.k or ldx % 8 != 0) return error.BadAffinePromptShape;
        for ([_]u64{ x, q.w, q.s, q.b, out }) |ptr| if (ptr == 0 or ptr % 16 != 0) return error.BadAffinePromptShape;
        var args: cuda.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |ptr| args.add(ptr);
        for ([_]usize{ rows, q.n, q.k, q.npad, ldx, 8 }) |v| {
            args.add(std.math.cast(c_int, v) orelse return error.BadAffinePromptShape);
        }
        const blocks = ((rows + 127) / 128) * ((q.n + 127) / 128);
        return cuda.launch.launch(ks.functions[@intFromBool(wide)], .{
            .grid = .{ .x = std.math.cast(u32, blocks) orelse return error.BadAffinePromptShape },
            .block = .{ .x = 256 },
            .shared = shared_bytes,
        }, stream, &args);
    }
};
