//! Explicit target precision pairs admitted by corrected drafting and optimized startup scopes.
pub fn supported(s: anytype) bool {
    const fp8_head = if (@hasField(@TypeOf(s), "q4_head_fp8")) s.q4_head_fp8 else false;
    if (fp8_head) return s.dense_bits == 4 and s.head_bits == 8 and s.tiled and s.q4_affine_mse;
    if (s.dense_bits == 16) return s.head_bits == 16;
    return s.dense_bits == 4 and s.head_bits == 16 and s.tiled;
}
