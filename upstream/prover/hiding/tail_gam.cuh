// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU kernels for a sub-proof's approximate range check (`tail` in identifiers). They compute each
// matrix column's coefficient after the check's TAU random combinations of columns are batched, and
// tables of powers of the batching challenge theta. Only prover/common/common.h and
// prover/common/slot_flags.h are included, so prover/tests/tail_gam_test.cu can test the kernels
// alone, against plain arithmetic.
#pragma once
#include "common/common.h"
#include "common/slot_flags.h"

/* g[c] = sum_(u < TAU) v[u] Cf[c][u] for the k columns c. Cf[c][u] is column c's coefficient in
 * combination u; v[u] is combination u's batching factor. Only the sign of Cf[c][u] is read, so it
 * must be -1, 0 or 1 (not checked). Every partial sum is reduced modulo the prime
 * p = 2^64 - 2^32 + 1, by `gf_add` and `gf_sub`: a plain sum of two coordinates
 * can overflow 64 bits.
 * The verifier (cp_verify) accepts no proof with a zero g[c]: the batched equations then do not
 * test column c. For an honest prover this happens if all TAU coefficients of a column are 0
 * (probability 2^-TAU per column) or if non-zero ones cancel. With v[u] = theta^u, which
 * `make_powers` in tail_f.cuh passes, they cancel for at most TAU - 1 of the p^EDEG values of
 * theta. Both events depend only on the challenges, and so on hidden values and the prover's private randomness
 * (prover/common/element_from_hash.h). A zero sum sets the failure flag ZERO_ELEMENT: no part of the proof is released.
 * One warp per column. Launch with blocks of 256 threads; a block size that is not a multiple of 32
 * gives wrong sums (not checked). */
// The compiler hints are public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; #pragma unroll: NVIDIA,
// "CUDA Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// unrolling to keep arrays in registers: M. Milakov,
// "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int TAU>
__global__ void __launch_bounds__(256) kh_gam(const signed char *Cf, size_t k, const Ext *v, Ext *g) {
  constexpr int PER = TAU / 32;
  static_assert(TAU % 32 == 0 && PER >= 1, "kh_gam requires that the number of combinations is a multiple of 32");
  static_assert(EDEG == 3, "the code below assumes three coordinates per element (EDEG == 3)");
  const int lane = threadIdx.x & 31;
  const size_t w0 = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5, nw = ((size_t)gridDim.x * blockDim.x) >> 5;
  Ext a[PER];
  for (int i = 0; i < PER; i++) a[i] = v[lane + 32 * i];
  for (size_t c = w0; c < k; c += nw) {
    u64 s[EDEG] = {0, 0, 0};
#pragma unroll
    for (int i = 0; i < PER; i++) {
      const int x = Cf[c * TAU + lane + 32 * i];
#pragma unroll
      for (int e = 0; e < EDEG; e++) {
        if (x > 0) s[e] = gf_add(s[e], coord(a[i], e));
        if (x < 0) s[e] = gf_sub(s[e], coord(a[i], e));
      }
    }
#pragma unroll
    for (int e = 0; e < EDEG; e++)
      for (int o = 16; o; o >>= 1) s[e] = gf_add(s[e], __shfl_xor_sync(~0u, s[e], o));
    if (lane == 0) {
      g[c] = ext_from(s);
      if (!(s[0] | s[1] | s[2])) raise_flags(ZERO_ELEMENT);
    }
  }
}

// base^e by squaring and multiplying.
// Public source: A. Menezes, P. van Oorschot, S. Vanstone, Handbook of Applied Cryptography, CRC
// Press, 1996, chapter 14, section 14.6.1, Algorithm 14.76 "Right-to-left binary exponentiation",
// https://cacr.uwaterloo.ca/hac/about/chap14.pdf; used in kf_powers() and kf_tie_powers().
__host__ __device__ static inline Ext tail_power(Ext base, unsigned long long e) {
  Ext power = ext_base(1);
  for (; e; e >>= 1, base = k3_mul(base, base))
    if (e & 1) power = k3_mul(power, base);
  return power;
}
// o[i] = (*base)^e[i] for i < n, or (*base)^i if e is null.
// The loop form is public: the grid-stride loop:
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void kf_powers(const Ext *base, const unsigned long long *e, size_t n, Ext *o) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) o[i] = tail_power(*base, e ? e[i] : i);
}
// Er[x] = (*step)^x for x < n; r[j] = (*step)^(2^(nb - 1 - j)) for j < nb. So Er[x], x < 2^nb, is
// the product of the r[j] at x's set bits, r[0] at the highest of the nb bits.
// Requires nb <= n and nb <= 64 (not checked). The loop form: cited at kf_powers().
__global__ void kf_tie_powers(const Ext *step, int nb, unsigned n, Ext *r, Ext *Er) {
  for (unsigned x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += gridDim.x * blockDim.x) {
    Er[x] = tail_power(*step, x);
    if (x < (unsigned)nb) r[nb - 1 - x] = tail_power(*step, 1ull << x);
  }
}
