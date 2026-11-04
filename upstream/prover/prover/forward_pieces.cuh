// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Small kernels of the forward pass. prover/prover/forward.cuh includes this file once the names
// used here (ResidualInt, GRID_LOOP, Pos2, E_XS ...) are declared. That file's own byte transpose transpose_i8() takes
// no ldb and its own k_epi_residual no `add`; the two here overload them. ForwardPass::shared_down() ends a
// mixture-of-experts layer that groups tokens by expert: k_unpermute brings the routed experts'
// sums to token order; k_epi_residual adds them to the shared expert's before rounding.
// Public source: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops" (NVIDIA Technical
// Blog): a grid-stride loop (GRID_LOOP). See also:
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// Public source: M. Harris, "An Efficient Matrix Transpose in CUDA C/C++" (NVIDIA Technical
// Blog, 2013): a padded tile in shared memory (k_transpose_i8).
// See also: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
#pragma once
#include "common/index_guard.cuh"
// out[j][t] = in[j][rank_of[t]], j < d, t < T. rank_of[t] (k_moe_prep, forward_moe.cuh) is token
// t's rank in expert order; outside [0, T) it sets the request's failure bit INDEX and counts as 0.
__global__ void k_unpermute(const ResidualInt *in, const int *rank_of, int T, int d, ResidualInt *out) {
  GRID_LOOP(i, (size_t)T * d) {
    const int t = i % T;
    const size_t j = i / T;
    out[i] = in[j * T + index_below(rank_of[t], T, INDEX_MOE_TABLE)];
  }
}
static inline void unpermute(const ResidualInt *in, const int *rank_of, int T, int d, ResidualInt *out, cudaStream_t s) {
  k_unpermute<<<grid_blocks((size_t)T * d), 256, 0, s>>>(in, rank_of, T, d, out);
  CK(cudaGetLastError());
}
// Transposes signed bytes: b[c * ldb + r] = a[r * Cn + c] for r < R, c < Cn. A launch must have
// blockDim (32, 8) and one block for each square of 32 by 32 bytes of `a`, as in transpose_i8(). ldb,
// the pitch of b's rows in bytes, must be at least R: not checked. Each row of `s` has a 33rd byte
// to spread the second loop's reads, which go down a column, over the banks of shared memory.
// The idea is public: a tiled transpose through padded shared memory: Harris, "An Efficient Matrix
// Transpose in CUDA C/C++", cited at the top of this file. There the elements are floats.
__global__ void k_transpose_i8(const i8 *a, i8 *b, int R, int Cn, int ldb) {
  __shared__ i8 s[32][33];
  int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
  for (int y = threadIdx.y; y < 32; y += 8)
    if (r0 + y < R && c0 + threadIdx.x < Cn) s[y][threadIdx.x] = a[(size_t)(r0 + y) * Cn + c0 + threadIdx.x];
  __syncthreads();
  for (int y = threadIdx.y; y < 32; y += 8)
    if (c0 + y < Cn && r0 + threadIdx.x < R) b[(size_t)(c0 + y) * ldb + r0 + threadIdx.x] = s[threadIdx.x][y];
}
// The same transpose, reading `a` and writing `b` in 32-bit words. It must be launched with blocks
// of 256 threads, one for each tile of 64 x 64 bytes, and checks no bounds: R and Cn must be
// multiples of 64, ldb a multiple of 4, and `a` and `b` 4-byte aligned. transpose_i8() launches it only
// if all this holds, else k_transpose_i8. A row of `s` has 68 bytes: padded, yet 4-byte aligned.
// Public source: a padded tile in shared memory: Harris's post, cited above; there the elements are
// floats. k_transpose_bytes (prover/weights_blocks/decode.cuh) moves words in the same way.
__global__ void __launch_bounds__(256) k_transpose_i8_words(const i8 *a, i8 *b, int Cn, int ldb) {
  __shared__ __align__(4) i8 s[64][68];
  const int c0 = blockIdx.x * 64, r0 = blockIdx.y * 64;
  for (int i = threadIdx.x; i < 64 * 16; i += 256) {
    const int r = i >> 4, w = i & 15;
    const u32 v = *reinterpret_cast<const u32 *>(a + (size_t)(r0 + r) * Cn + c0 + 4 * w);
#pragma unroll
    for (int k = 0; k < 4; k++) s[4 * w + k][r] = (i8)(v >> (8 * k));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 64 * 16; i += 256)
    *reinterpret_cast<u32 *>(b + (size_t)(c0 + (i >> 4)) * ldb + r0 + 4 * (i & 15)) =
        *reinterpret_cast<const u32 *>(&s[i >> 4][4 * (i & 15)]);
}
static inline void transpose_i8(const i8 *a, i8 *b, int R, int Cn, cudaStream_t s, int ldb) {
  if (R % 64 == 0 && Cn % 64 == 0 && ldb % 4 == 0 && (uintptr_t)a % 4 == 0 && (uintptr_t)b % 4 == 0)
    k_transpose_i8_words<<<dim3(Cn / 64, R / 64), 256, 0, s>>>(a, b, Cn, ldb);
  else
    k_transpose_i8<<<dim3((Cn + 31) / 32, (R + 31) / 32), dim3(32, 8), 0, s>>>(a, b, R, Cn, ldb);
  CK(cudaGetLastError());
}
// Updates the residual stream: xnew = xold + floor((Acc + add + 2^(k-1)) / 2^k) (rq, defined in
// docs/model.md, "1 Notation"), Acc = sum over s of 256^s P_s (P3 and P4 only if nP > 3), all
// arrays [j < d][t]. P0 .. P4 are a matrix product's partial sums: P_s adds up the products of an
// activation byte and a weight byte whose indices sum to s. `add` may be null or xnew: a thread
// reads only what it then overwrites. A result outside [`least`, `most`] sets the range flag E_XS
// and becomes 0. If `proof`, xnew's limbs go to xc, the remainder's cells and lookup keys to rh,
// kLo, kHi (write_digits()). A thread takes W adjacent tokens (LAUNCH_BY_FOUR, forward_by_four.cuh).
// lie::at() alters a value only in a test build that is not part of this repository.
template <int W = 1>
__global__ void k_epi_residual(const int *P0, const int *P1, const int *P2, const int *P3, const int *P4, int nP, int T,
                          int d, int k, int form, int xl, const ResidualInt *xold, ResidualInt *xnew, i8 *xc /* xl limbs */, size_t stX,
                          i8 *rh, size_t stR, u16 *kLo, u16 *kHi, Pos2 pLo, Pos2 pHi, size_t bLo, size_t bHi, u32 *err,
                          int proof, const ResidualInt *add) {
  const i64 least = xl == 5 ? -547599908735ll : -2147483647ll, most = xl == 5 ? 547599908735ll : 2139062143ll;
  GRID_LOOP(z, (size_t)T * d / W) {
    const size_t i = W * z;
    const int t = i % T, j = i / T;
    i64 a[W], old[W], xn[W], left[W];
    {
      int p0[W], p1[W], p2[W];
      ld_w<W>(P0 + i, p0), ld_w<W>(P1 + i, p1), ld_w<W>(P2 + i, p2);
#pragma unroll
      for (int e = 0; e < W; e++) a[e] = (i64)p0[e] + ((i64)p1[e] << 8) + ((i64)p2[e] << 16);
    }
    if (nP > 3) {
      int p3[W], p4[W];
      ld_w<W>(P3 + i, p3), ld_w<W>(P4 + i, p4);
#pragma unroll
      for (int e = 0; e < W; e++) a[e] += ((i64)p3[e] << 24) + ((i64)p4[e] << 32);
    }
    if (add) {
      i64 more[W];
      ld_w<W>(add + i, more);
#pragma unroll
      for (int e = 0; e < W; e++) a[e] += more[e];
    }
    ld_w<W>(xold + i, old);
    bool outside = false;
#pragma unroll
    for (int e = 0; e < W; e++) {
      a[e] = lie::at(lie::ACCUMULATOR, j, t + e, a[e]) + (1ll << (k - 1));
      const i64 q = lie::at(lie::ROUNDED, j, t + e, a[e] >> k);
      xn[e] = lie::at(lie::STREAM, j, t + e, old[e] + q), left[e] = a[e] - (q << k);
      if (xn[e] < least || xn[e] > most) outside = true, xn[e] = 0;
    }
    if (outside) atomicOr(err, (u32)E_XS);
    st_w<W>(xnew + i, xn);
    if (!proof) continue;
    write_limbs_wide<W>(xn, xl, xc + i, stX);
    write_digits_wide<W>(form, k, left, rh + i, stR, kLo + bLo + (size_t)pLo.at(j) * T + t, kHi + bHi + (size_t)pHi.at(j) * T + t);
  }
}
