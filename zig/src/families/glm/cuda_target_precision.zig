//! Explicit target precision pairs admitted by corrected drafting and optimized startup scopes.
pub fn supported(s: anytype) bool {
    const fp8_head = if (@hasField(@TypeOf(s), "q4_head_fp8")) s.q4_head_fp8 else false;
    if (fp8_head) return s.dense_bits == 4 and s.head_bits == 8 and s.tiled and s.q4_affine_mse;
    if (s.dense_bits == 16) return s.head_bits == 16;
    if (s.dense_bits == 8) {
        if (@hasField(@TypeOf(s), "q4_affine_mse")) if (s.q4_affine_mse) return false;
        if (@hasField(@TypeOf(s), "q4_python_prompt")) if (s.q4_python_prompt) return false;
        return s.head_bits == 16 and s.tiled;
    }
    return s.dense_bits == 4 and s.head_bits == 16 and s.tiled;
}
