// Contiguous FP8 sparse attention for prompt rows. This compilation unit
// preserves the selected compact mode's math and keeps it out of the decode image.

#include "glm_glue_router.cuh"
#include <cuda_fp8.h>

namespace glm_prefill_attn {

constexpr int LW = 512, KT = 32, CH = 512, KPAD = LW + 8;

// p fp32 summed as Triton's tile does: a thread's two columns, the quad (xor 2, 1), then warps (w0 + w2) + (w1 + w3).
__device__ __forceinline__ float quad_sum(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  tf_common::cp16z(dst, src, ok);
}
__device__ __forceinline__ void commit() { tf_common::cp_commit(); }
template <int N>
__device__ __forceinline__ void wait() { tf_common::cp_wait<N>(); }
template <bool TRANS>
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  const unsigned a = tf_common::smem(p);
  if (TRANS) qmm_frag::ldmatrix4t(r, a);
  else qmm_frag::ldmatrix4(r, a);
}

// A 512-wide bf16 row's 16-byte unit u in shared memory, swizzled by the row so ldmatrix's eight rows hit eight banks.
__device__ __forceinline__ int wide(int row, int u) { return row * LW + ((u ^ (row & 7)) << 3); }
// p's [32][32] rows (64 bytes): unit u of row r.
__device__ __forceinline__ int narrow(int row, int u) { return row * KT + ((u ^ ((row >> 1) & 3)) << 3); }

__device__ __forceinline__ bool fp8_nan4(uint32_t codes) {
  const uint32_t x = (codes & 0x7f7f7f7fu) ^ 0x7f7f7f7fu;
  return ((x - 0x01010101u) & ~x & 0x80808080u) != 0;
}

// Normal powers of two commute with the FP32 multiplication and BF16 rounding.
// Each warp reads a single key, with the rounded scale times the 16 mantissas in lanes 0..15.
__device__ __forceinline__ uint32_t scaled_code(uint32_t code, uint32_t table) {
  const int e = (code >> 3) & 15;
  const int m = (code & 7) | (e ? 8 : 0);
  const int base = __shfl_sync(0xffffffffu, table, m);
  const int value = m ? base + (e ? e - 10 : -9) * 128 : 0;
  return static_cast<uint32_t>(value) | ((code & 128) << 8);
}

// latent._dense_chunks/_sparse_chunks (sparse 0/1), block (row, HB heads, chunk), 8 warps: online softmax by 32 keys.
template <int HB>
constexpr int wide_smem() {
  return (HB * LW + 2 * KT * LW + HB * KT) * 2 + (HB * 4 + 2 * HB) * 4;
}

struct RecentWindow {
  const __nv_bfloat16* cache;
  int64_t begin;
  int64_t end;
};

__device__ __forceinline__ uint32_t decode_fp8_pair(uint16_t word, float scale) {
  const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(word, __NV_E4M3)));
  const __nv_bfloat162_raw bf = __floats2bfloat162_rn(__fmul_rn(v.x, scale), __fmul_rn(v.y, scale));
  return static_cast<uint32_t>(bf.x) | (static_cast<uint32_t>(bf.y) << 16);
}

__device__ __forceinline__ uint4 decode_eight_fp8(uint32_t lo, uint32_t hi, float scale) {
  return make_uint4(decode_fp8_pair(static_cast<uint16_t>(lo), scale),
                    decode_fp8_pair(static_cast<uint16_t>(lo >> 16), scale),
                    decode_fp8_pair(static_cast<uint16_t>(hi), scale),
                    decode_fp8_pair(static_cast<uint16_t>(hi >> 16), scale));
}

template <int HB, bool FP8 = false, bool FUSED = false, bool FOUR_WARPS = false,
          bool DIRECT_REDUCTIONS = false, bool COMPACT = false, bool PACKED32 = false,
          bool SCALE_LUT = false, bool KEEP_QUERY_ROW = false, bool RECENT_FP8 = false,
          bool WARP_KEY = false, bool KEY_PREFETCH = false, bool PAGED8 = false, bool PREFETCH8 = false>
__device__ void chunks_wide(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                            const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                            float scale, int sparse, const int* pt, int64_t pstep, const float* scales = nullptr,
                            __nv_bfloat16* out = nullptr, int nch = 0, RecentWindow recent = {}) {
  static_assert(!FUSED || HB == 16);
  static_assert(!FOUR_WARPS || FUSED);
  static_assert(!COMPACT || (FUSED && DIRECT_REDUCTIONS));
  static_assert(!PACKED32 || (FP8 && COMPACT));
  static_assert(!SCALE_LUT || (PACKED32 && !FOUR_WARPS));
  static_assert(!KEEP_QUERY_ROW || (FP8 && FUSED && COMPACT));
  static_assert(!RECENT_FP8 || (FP8 && FUSED && COMPACT && !KEEP_QUERY_ROW));
  static_assert(!WARP_KEY || (FP8 && FUSED && COMPACT && !FOUR_WARPS));
  static_assert(!KEY_PREFETCH || (WARP_KEY && !RECENT_FP8));
  static_assert(!PAGED8 || (FP8 && !RECENT_FP8));
  static_assert(!PREFETCH8 || (FP8 && !FOUR_WARPS && !COMPACT));
  constexpr int MT = HB / 16;
  constexpr int value_tiles = FOUR_WARPS ? 16 : 8;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * LW;
  // In compact mode the probability tile borrows head zero's query row after QK.
  __nv_bfloat16* ps = COMPACT ? qs : kvs + 2 * KT * LW;
  float* red = reinterpret_cast<float*>(COMPACT ? kvs + KT * LW : ps + HB * KT);
  float* stat = red + HB * 4;
  float* sum_red = DIRECT_REDUCTIONS ? stat : red;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  auto valueColumn = [warp](int tile) {
    if constexpr (FOUR_WARPS) return (warp + (tile / 8) * 4) * 64 + (tile % 8) * 8;
    return warp * 64 + tile * 8;
  };
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * HB;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r;
  if constexpr (FUSED) if (n == 0) return;
  uint4 saved_query = {};
  if constexpr (KEEP_QUERY_ROW) {
    // The probability tile reuses query head zero's shared memory.
    // Keep that head in the first two warps and restore it after each tile.
    if (t < LW / 8)
      saved_query = reinterpret_cast<const uint4*>(qa + (r * heads + h0) * LW)[t];
  }
  float merged_max[2] = {-INFINITY, -INFINITY}, merged_sum[2] = {0.0f, 0.0f};
  float merged_out[value_tiles][4] = {};
  // Codes and scales have different row widths and the same page byte stride.
  auto pageOffset = [&](int64_t logical) -> int64_t {
    if constexpr (PAGED8) return (int64_t)pt[logical / CH] * pstep;
    return 0;
  };
  auto withinPage = [](int64_t logical) -> int64_t {
    if constexpr (PAGED8) return logical % CH;
    return logical;
  };
  const int64_t first_chunk = FUSED ? 0 : blockIdx.z;
  const int64_t last_chunk = FUSED ? nch : first_chunk + 1;
  for (int64_t c = first_chunk; c < last_chunk; ++c) {
    const int64_t start = c * CH;
    const bool live = sparse ? start < n : start <= limit;
    // a dense chunk's 512 keys are one page, its rows from that page's; a sparse key's row comes as the token list has it
    const __nv_bfloat16* keys_at = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * LW) : lc;
    const int64_t dense_page = PAGED8 && !sparse && live ? pageOffset(start) : 0;
    auto offset = [&](int64_t key, bool ok) -> int64_t {
      if constexpr (PAGED8) return sparse ? (ok ? pageOffset(key) : 0) : dense_page;
      return 0;
    };
    auto load_kv = [&](int tile) {
      __nv_bfloat16* dst = kvs + (COMPACT ? 0 : (tile & 1) * KT * LW);
      const int64_t first = start + tile * KT;
      if constexpr (WARP_KEY) {
        // One warp fetches a full key. Each lane decodes two eight-value
        // sections, then places them in the original shared-memory layout.
        for (int key = warp; key < KT; key += 8) {
          const int64_t ki = first + key;
          const bool ok = ki < n;
          const int64_t src = ok ? static_cast<int64_t>(tok[r * w + ki]) : 0;
          if constexpr (RECENT_FP8) {
            if (ok && src >= recent.begin && src < recent.end) {
              const __nv_bfloat16* values = recent.cache + (src - recent.begin) * LW + lane * 16;
              cp16(dst + wide(key, 2 * lane), values, true);
              cp16(dst + wide(key, 2 * lane + 1), values + 8, true);
              continue;
            }
          }
          const int64_t page = offset(src, ok);
          const int64_t row = withinPage(src);
          const auto* codes = reinterpret_cast<const uint4*>(reinterpret_cast<const unsigned char*>(lc) + page);
          const auto* row_scales = reinterpret_cast<const float*>(
              reinterpret_cast<const unsigned char*>(scales) + page);
          const float row_scale = ok ? row_scales[row] : 1.0f;
          const uint4 packed = ok ? codes[row * (LW / 16) + lane] : uint4{};
          *reinterpret_cast<uint4*>(dst + wide(key, 2 * lane)) = decode_eight_fp8(packed.x, packed.y, row_scale);
          *reinterpret_cast<uint4*>(dst + wide(key, 2 * lane + 1)) = decode_eight_fp8(packed.z, packed.w, row_scale);
        }
        return;
      }
      for (int idx = t; idx < KT * (LW / 8); idx += blockDim.x) {
        const int key = idx / (LW / 8), u = idx % (LW / 8);
        const int64_t ki = first + key;
        const bool ok = sparse ? ki < n : ki <= limit;
        const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
        if constexpr (RECENT_FP8) {
          if (ok && src >= recent.begin && src < recent.end) {
            cp16(dst + wide(key, u), recent.cache + (src - recent.begin) * LW + u * 8, true);
            continue;
          }
        }
        if constexpr (FP8) {
          const int64_t page = offset(src, ok);
          const int64_t row = withinPage(src);
          const unsigned char* codes = reinterpret_cast<const unsigned char*>(lc) + page;
          const auto* row_scales = reinterpret_cast<const float*>(
              reinterpret_cast<const unsigned char*>(scales) + page);
          const float row_scale = ok ? row_scales[row] : 1.0f;
          if constexpr (PACKED32) {
            // The cache stride and each eight-column group are four-byte aligned.
            const auto* words = reinterpret_cast<const uint32_t*>(codes + row * LW + u * 8);
            if constexpr (SCALE_LUT) {
              const uint32_t low = ok ? words[0] : 0;
              const uint32_t high = ok ? words[1] : 0;
              if (row_scale >= 0x1p-110f && row_scale <= 0x1p110f &&
                  !__any_sync(0xffffffffu, fp8_nan4(low) || fp8_nan4(high))) {
                const uint32_t table = __bfloat16_as_ushort(__float2bfloat16_rn(
                    __fmul_rn(static_cast<float>(lane & 15), row_scale)));
                uint32_t* pairs = reinterpret_cast<uint32_t*>(dst + wide(key, u));
#pragma unroll
                for (int word = 0; word < 2; ++word) {
                  const uint32_t packed = word == 0 ? low : high;
                  const uint32_t a = scaled_code(packed & 255, table);
                  const uint32_t b = scaled_code((packed >> 8) & 255, table);
                  const uint32_t c = scaled_code((packed >> 16) & 255, table);
                  const uint32_t d = scaled_code(packed >> 24, table);
                  pairs[word * 2] = a | (b << 16);
                  pairs[word * 2 + 1] = c | (d << 16);
                }
                continue;
              }
            }
#pragma unroll
            for (int word = 0; word < 2; ++word) {
              const uint32_t packed = ok ? words[word] : 0;
#pragma unroll
              for (int pair = 0; pair < 2; ++pair) {
                const auto bits = static_cast<uint16_t>(packed >> (16 * pair));
                const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
                const __nv_bfloat162 value = __floats2bfloat162_rn(__fmul_rn(v.x, row_scale), __fmul_rn(v.y, row_scale));
                *reinterpret_cast<__nv_bfloat162*>(dst + wide(key, u) + word * 4 + pair * 2) = value;
              }
            }
          } else {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
              const uint16_t bits = ok ? (uint16_t)codes[row * LW + u * 8 + j] |
                                            ((uint16_t)codes[row * LW + u * 8 + j + 1] << 8) : 0;
              const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
              dst[wide(key, u) + j] = __float2bfloat16_rn(__fmul_rn(v.x, row_scale));
              dst[wide(key, u) + j + 1] = __float2bfloat16_rn(__fmul_rn(v.y, row_scale));
            }
          }
        } else {
          cp16(dst + wide(key, u), keys_at + src * LW + u * 8, ok);
        }
      }
    };
    // TensorFold PR #3454: tile + 2's codes load 8 bytes a unit into registers before tile's math, decode after it, 16-byte stores.
    constexpr int FU = PREFETCH8 ? KT * (LW / 8) / 256 : 1;  // units a thread: every launcher runs 256 threads
    uint2 fc[FU];
    float fs[FU];
    auto fetch_fp8 = [&](int tile) {
      const int64_t first = start + tile * KT;
#pragma unroll
      for (int i = 0; i < FU; ++i) {
        const int idx = t + i * 256, key = idx / (LW / 8), u = idx % (LW / 8);
        const int64_t ki = first + key;
        const bool ok = sparse ? ki < n : ki <= limit;
        const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
        const int64_t page = offset(src, ok);
        const int64_t row = withinPage(src);
        const unsigned char* codes = reinterpret_cast<const unsigned char*>(lc) + page + row * LW + u * 8;
        const auto* row_scales = reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(scales) + page);
        fs[i] = ok ? row_scales[row] : 1.0f;
        fc[i] = ok ? *reinterpret_cast<const uint2*>(codes) : make_uint2(0u, 0u);
      }
    };
    auto store_fp8 = [&](int tile) {
      __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
#pragma unroll
      for (int i = 0; i < FU; ++i) {
        const int idx = t + i * 256, key = idx / (LW / 8), u = idx % (LW / 8);
        uint32_t out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint16_t bits = (uint16_t)((j < 2 ? fc[i].x : fc[i].y) >> ((j & 1) * 16));
          const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
          const __nv_bfloat162 b = __floats2bfloat162_rn(__fmul_rn(v.x, fs[i]), __fmul_rn(v.y, fs[i]));
          out[j] = *reinterpret_cast<const uint32_t*>(&b);
        }
        *reinterpret_cast<uint4*>(dst + wide(key, u)) = make_uint4(out[0], out[1], out[2], out[3]);
      }
    };
    float m[MT][2], l[MT][2], o[MT][value_tiles][4];
    for (int i = 0; i < MT; ++i)
      for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < value_tiles; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
    constexpr int chunk_tiles = CH / KT;
    int tiles = chunk_tiles;
    if constexpr (FUSED) {
      if (live) {
        const int64_t valid = n - start;
        const int64_t occupied = (valid + KT - 1) / KT;
        // One empty tile retains the original zero-FMA transition before the merge.
        if (occupied + 1 < chunk_tiles) tiles = static_cast<int>(occupied + 1);
      }
    }
    auto load_q = [&](int head_count) {
      for (int idx = t; idx < head_count * (LW / 8); idx += blockDim.x) {
        const int hd = idx / (LW / 8), u = idx % (LW / 8);
        const bool ok = h0 + hd < heads;
        cp16(qs + wide(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * LW + u * 8, ok);
      }
    };
    if constexpr (!COMPACT) {
      if (live) {
        if (!FUSED || c == 0) load_q(HB);
        if constexpr (PREFETCH8) {
          fetch_fp8(0);
          store_fp8(0);
          fetch_fp8(1);
          store_fp8(1);
          commit();
        } else {
          load_kv(0);
          commit();
          load_kv(1);
          commit();
        }
      }
    }
    for (int tile = 0; live && tile < tiles; ++tile) {
      if constexpr (KEEP_QUERY_ROW) {
        const bool first = c == 0 && tile == 0;
        if (first) load_q(HB);
        else if (t < LW / 8)
          *reinterpret_cast<uint4*>(qs + wide(0, t)) = saved_query;
        load_kv(tile);
        if (first) {
          commit();
          wait<0>();
        }
      } else if constexpr (COMPACT) {
        load_q(c == 0 && tile == 0 ? HB : 1);
        load_kv(tile);
        commit();
        wait<0>();
      } else if constexpr (FP8) {
        if constexpr (PREFETCH8) if (tile + 2 < tiles) fetch_fp8(tile + 2);
        wait<0>();
      } else {
        if (tile + 1 < tiles)
          wait<1>();
        else
          wait<0>();
      }
      __syncthreads();
      const __nv_bfloat16* kv = kvs + (COMPACT ? 0 : (tile & 1) * KT * LW);
      const int64_t first = start + tile * KT;
      if constexpr (KEY_PREFETCH) {
        // Each lane in warp zero hints the next selected key. QK/PV on this tile
        // gives the cache time to fetch the next key's FP8 row.
        if (t < KT) {
          const int64_t next = first + KT + t;
          if (next < n) {
            const int64_t key = tok[r * w + next];
            const auto* payload = reinterpret_cast<const unsigned char*>(lc) + pageOffset(key) + withinPage(key) * LW;
#pragma unroll
            for (int line = 0; line < LW; line += 128)
              asm volatile("prefetch.global.L2 [%0];" :: "l"(payload + line) : "memory");
          }
        }
      }
      float sc[4];
      if (hh < MT) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int kk = 0; kk < LW; kk += 32) {
          uint32_t a0[4], a1[4], b[4];
          ldm4<false>(a0, qs + wide(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
          ldm4<false>(a1, qs + wide(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
          ldm4<false>(b, kv + wide(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
          glm_router::mma16816(acc, a0, b);
          glm_router::mma16816(acc, a1, b + 2);
        }
        for (int e = 0; e < 2; ++e) {
          const int64_t ki = first + kg * 8 + q * 2 + e;
          const bool ok = sparse ? ki < n : ki <= limit;
          sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
          sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
        }
        for (int hf = 0; hf < 2; ++hf) {
          const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
          if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
        }
      }
      __syncthreads();
      if constexpr (!DIRECT_REDUCTIONS) {
        if (t < HB * 4) {
          const float v = quad_max(red[t]);
          if ((t & 3) == 0) stat[t / 4] = v;
        }
        __syncthreads();
      }
      float alpha[MT][2];
      for (int i = 0; i < MT; ++i)
        for (int hf = 0; hf < 2; ++hf) {
          const int hr = i * 16 + hf * 8 + g;
          float tm;
          if constexpr (DIRECT_REDUCTIONS) {
            // Every output warp needs these two heads. Reduce across their four key groups here.
            const float v = quad_max(red[hr * 4 + q]);
            tm = __shfl_sync(0xffffffffu, v, lane & ~3);
          } else {
            tm = stat[hr];
          }
          const bool active = tm != -INFINITY;
          const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
          alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
          if (i == hh) {
            float p[2];
            for (int e = 0; e < 2; ++e) {
              const int64_t ki = first + kg * 8 + q * 2 + e;
              const bool ok = sparse ? ki < n : ki <= limit;
              p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
            }
            const int col = kg * 8 + q * 2;
            *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
            const float s = quad_sum(p[0] + p[1]);
            if (q == 0) sum_red[hr * 4 + kg] = s;
          }
          m[i][hf] = nm;
        }
      __syncthreads();
      if constexpr (!DIRECT_REDUCTIONS) {
        if (t < HB * 4) {
          const float v = quad_sum(sum_red[t]);
          if ((t & 3) == 0) stat[HB + t / 4] = v;
        }
      }
      for (int i = 0; i < MT; ++i)
        for (int j = 0; j < value_tiles; ++j)
          for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
      for (int k16 = 0; k16 < KT; k16 += 16) {
        uint32_t a[MT][4];
        for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
        for (int j = 0; j < value_tiles; j += 2) {
          uint32_t b[4];
          ldm4<true>(b, kv + wide(k16 + (lane & 15), ((valueColumn(j)) >> 3) + (lane >> 4)));
          for (int i = 0; i < MT; ++i) {
            glm_router::mma16816(o[i][j], a[i], b);
            glm_router::mma16816(o[i][j + 1], a[i], b + 2);
          }
        }
      }
      for (int i = 0; i < MT; ++i)
        for (int hf = 0; hf < 2; ++hf) {
          if constexpr (DIRECT_REDUCTIONS) {
            const int hr = i * 16 + hf * 8 + g;
            const float v = quad_sum(sum_red[hr * 4 + q]);
            const float ready = __shfl_sync(0xffffffffu, v, lane & ~3);
            l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], ready);
          }
        }
      __syncthreads();
      if constexpr (!DIRECT_REDUCTIONS) {
        for (int i = 0; i < MT; ++i)
          for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
      }
      if constexpr (!COMPACT) {
        if (tile + 2 < tiles) {
          if constexpr (PREFETCH8) store_fp8(tile + 2);
          else load_kv(tile + 2);
        }
        commit();
      }
    }
    if constexpr (FUSED) {
      // Keep the same FP32 merge operations and 512-key boundaries as tf_glm_merge.
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const float cm = m[0][hf], cl = l[0][hf];
        const bool active = cl > 0.0f;
        const float previous = merged_max[hf], nm = active ? fmaxf(previous, cm) : previous;
        const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
        const float a = active ? (previous == -INFINITY ? 0.0f : glm_triton::exp_approx(previous - nm)) : 1.0f;
#pragma unroll
        for (int j = 0; j < value_tiles; ++j) {
#pragma unroll
          for (int e = hf * 2; e < hf * 2 + 2; ++e) {
            const float part = o[0][j][e];
            merged_out[j][e] = c == 0 ? __fmaf_rn(b, part, 0.0f)
                                       : __fmaf_rn(merged_out[j][e], a, __fmul_rn(b, part));
          }
        }
        merged_sum[hf] = c == 0 ? __fmaf_rn(b, cl, 0.0f) : __fmaf_rn(merged_sum[hf], a, __fmul_rn(b, cl));
        merged_max[hf] = nm;
      }
      __syncthreads();
    } else {
      for (int i = 0; i < MT; ++i)
        for (int e = 0; e < 4; ++e) {
          const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
          if (hd >= heads) continue;
          const int64_t base = (c * rows + r) * heads + hd;
          for (int j = 0; j < value_tiles; ++j) po[base * LW + valueColumn(j) + q * 2 + e % 2] = o[i][j][e];
          if (warp == 0 && q == 0 && e % 2 == 0) {
            pm[base] = m[i][e / 2];
            pl[base] = l[i][e / 2];
          }
        }
    }
  }
  if constexpr (FUSED) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int64_t head = h0 + g + (e / 2) * 8;
#pragma unroll
      for (int j = 0; j < value_tiles; ++j) {
        const auto col = valueColumn(j) + q * 2 + e % 2;
        out[(r * heads + head) * LW + col] = __float2bfloat16_rn(
            glm_triton::div_full(merged_out[j][e], merged_sum[e / 2]));
      }
    }
  }
}

}  // namespace glm_prefill_attn

// Two independent blocks cover the two 16-head tiles of a TP2 rank.
extern "C" __global__ void __launch_bounds__(256, 2)
    tf_glm_sparse_prompt_fp8_warp_key_prefetch_lines_h32(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales,
                                     const int* tok, const int* cnt, __nv_bfloat16* out, int64_t rows,
                                     int64_t w, float scale, int nch) {
  glm_prefill_attn::chunks_wide<16, true, true, false, true, true, false, false, false, false, true, true>(
      qa, reinterpret_cast<const __nv_bfloat16*>(lc), nullptr, tok, cnt, nullptr, nullptr, nullptr,
      rows, 32, w, scale, 1, nullptr, 0, scales, out, nch);
}
