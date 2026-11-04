// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Device functions for the Reed-Solomon encoders' fast transform (NTT, number-theoretic transform)
// over the Goldilocks field, p = 2^64 - 2^32 + 1. group() runs up to four butterfly layers on
// sixteen words in registers: one product by tau^j, a power of one twiddle factor, for a group's
// j-th word, then layers whose factors are powers of 2 (shifts). Four layers on sixteen words cost
// 15 general products and 17 shifts in place of 32 general products (see group()).
// Public sources: A. Schoenhage, V. Strassen, "Schnelle Multiplikation grosser Zahlen",
// Computing 7 (1971) (transforms whose roots of unity are powers of two); J. W. Cooley,
// J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series", Mathematics of
// Computation 19 (1965) (the step from a layer's factors to tau^j times a transform with fixed
// factors). We have not searched for public code that does this for this prime.
//
// Use: the threads of a thread block load the twiddle factors into a heap in shared memory (layout
// at powers()), barrier, all call powers(), barrier; then each thread may call group(). So do the
// kernels in prover/engine/encode_kernels.cuh (through `PassFactors`) and
// prover/hash_commitment/hf_code_groups.cuh. Results need not be canonical (below p): apply
// blk::GL64::canonical() after the last layer.
// Three thread blocks of 256 threads per multiprocessor (80 registers) were fastest on an H100.
// Correct source does not guarantee a correct build (see powers()): test every instantiation where
// the factors are not 1 (on a coset, or in a block other than a layer's first).
#pragma once
#include <utility>
#include "field/field_lazy.cuh"
namespace blk {
namespace lazy {
namespace groups {
// A value inside a group: the integer l + 2^32 h + 2^64 k, with a small signed k, in the value's
// residue class modulo p. plus() and minus() do not reduce. Outside a group a value is any 64-bit
// word of its class. Inside a group the values stay unreduced, after D. Harvey, "Faster arithmetic
// for number-theoretic transforms", arXiv:1205.2926.
// Below, e = 2^32 - 1 (`GE`), which is 2^64 modulo p.
struct W3 {
  u32 l, h, k; // k in two's complement
};
// in() converts a group's word 0, which takes no factor, and biases its k:
// x = (x - BIAS e) + 2^64 BIAS modulo p, so k = BIAS or BIAS - 1. Word 0 enters every result of the
// group with a plus sign and is never part of an operand of shifted(). The rest of a result is a
// signed sum of at most fifteen values of times() or shifted(), each with |k| <= 1, so it is within
// (-30, 30) x 2^64: every result has 2 <= k <= 63, and out() handles no negative k.
constexpr u32 BIAS = 33;
__device__ __forceinline__ W3 in(u64 x) {
  W3 r;
  asm("sub.cc.u32 %0, %3, %5;\n\t"
      "subc.cc.u32 %1, %4, %6;\n\t"
      "subc.u32 %2, %7, 0;"
      : "=&r"(r.l), "=&r"(r.h), "=&r"(r.k)
      : "r"((u32)x), "r"((u32)(x >> 32)), "n"((u32)((u64)BIAS * 0xFFFFFFFFull)), "n"((u32)(((u64)BIAS * 0xFFFFFFFFull) >> 32)), "n"(BIAS));
  return r;
}
__device__ __forceinline__ W3 plus(W3 a, W3 b) {
  W3 r;
  asm("add.cc.u32 %0, %3, %6;\n\t"
      "addc.cc.u32 %1, %4, %7;\n\t"
      "addc.u32 %2, %5, %8;"
      : "=&r"(r.l), "=&r"(r.h), "=&r"(r.k) : "r"(a.l), "r"(a.h), "r"(a.k), "r"(b.l), "r"(b.h), "r"(b.k));
  return r;
}
__device__ __forceinline__ W3 minus(W3 a, W3 b) {
  W3 r;
  asm("sub.cc.u32 %0, %3, %6;\n\t"
      "subc.cc.u32 %1, %4, %7;\n\t"
      "subc.u32 %2, %5, %8;"
      : "=&r"(r.l), "=&r"(r.h), "=&r"(r.k) : "r"(a.l), "r"(a.h), "r"(a.k), "r"(b.l), "r"(b.h), "r"(b.k));
  return r;
}
// t x for any two 64-bit words, with k = -1, 0 or 1: the product's four 32-bit words e0 .. e3,
// reduced to (e0 + 2^32 e1) + e e2 - e3 (2^96 = -1 modulo p).
__device__ __forceinline__ W3 times(u64 t, u64 x) {
  const u32 a0 = (u32)t, a1 = (u32)(t >> 32), b0 = (u32)x, b1 = (u32)(x >> 32);
  u32 e0, e1, e2, e3, o0, o1, o2;
  // Part of this statement takes the same steps as the product in class gl64_t of sppark
  // (Supranational LLC; ff/gl64_t.cuh; Apache-2.0). See THIRD_PARTY.md, "sppark".
  asm("mul.lo.u32 %0, %7, %9;\n\t"
      "mul.hi.u32 %1, %7, %9;\n\t"
      "mul.lo.u32 %2, %8, %10;\n\t"
      "mul.hi.u32 %3, %8, %10;\n\t"
      "mul.lo.u32 %4, %7, %10;\n\t"
      "mul.hi.u32 %5, %7, %10;\n\t"
      "mad.lo.cc.u32 %4, %8, %9, %4;\n\t"
      "madc.hi.cc.u32 %5, %8, %9, %5;\n\t"
      "addc.u32 %6, 0, 0;\n\t"
      "add.cc.u32 %1, %1, %4;\n\t"
      "addc.cc.u32 %2, %2, %5;\n\t"
      "addc.u32 %3, %3, %6;"
      : "=&r"(e0), "=&r"(e1), "=&r"(e2), "=&r"(e3), "=&r"(o0), "=&r"(o1), "=&r"(o2)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
  W3 r;
  asm("mad.lo.cc.u32 %0, %5, 0xFFFFFFFF, %3;\n\t"
      "madc.hi.cc.u32 %1, %5, 0xFFFFFFFF, %4;\n\t"
      "addc.u32 %2, 0, 0;\n\t"
      "sub.cc.u32 %0, %0, %6;\n\t"
      "subc.cc.u32 %1, %1, 0;\n\t"
      "subc.u32 %2, %2, 0;"
      : "=&r"(r.l), "=&r"(r.h), "=&r"(r.k) : "r"(e0), "r"(e1), "r"(e2), "r"(e3));
  return r;
}
// 2^S x for S = 12, 24, .., 84 and |k| <= 32; the result's k is -1, 0 or 1.
// With v = l + 2^32 h cut as v 2^S = (y0 + 2^32 y1 + 2^64 y2) 2^(32 q), S = 32 q + r, and modulo p
// 2^64 = 2^32 - 1, 2^96 = -1, 2^128 = -2^32:
//   q = 0: v 2^S = (y0 - y2) + 2^32 (y1 + y2),   2^(64 + S) k = (2^32 - 1) (k << S);
//   q = 1: v 2^S = (-y1 - y2) + 2^32 (y0 + y1),  2^(64 + S) k = -(k << (S - 32));
//   q = 2: v 2^S = (-y0 - y1) + 2^32 (y0 - y2),  2^(64 + S) k = -2^32 (k << (S - 64)).
// The two sums L and H are signed 64-bit numbers, |L| < 2^34 + 2^33, and L's upper half is carried
// into H.
template <int S> __device__ __forceinline__ W3 shifted(W3 x) {
  constexpr int q = S >> 5, r = S & 31;
  const int k = (int)x.k;
  const u32 y0 = x.l << r, y1 = __funnelshift_l(x.l, x.h, r), y2 = x.h >> (32 - r);
  long long L, H;
  if constexpr (q == 0) {
    const int t = (int)y2 + k * (1 << (S & 31)); // y2 < 2^24, |k| 2^S <= 2^29
    L = (long long)y0 - t, H = (long long)y1 + t;
  } else if constexpr (q == 1) {
    L = -(long long)y1 - (long long)y2 - (long long)k * (1 << (S & 31)), H = (long long)y0 + (long long)y1;
  } else {
    const int t = (int)y2 + k * (1 << (S & 31)); // y2 < 2^20, |k| 2^(S - 64) <= 2^25
    L = -(long long)y0 - (long long)y1, H = (long long)y0 - t;
  }
  H += L >> 32;
  return W3{(u32)L, (u32)H, (u32)(H >> 32)};
}
// One 64-bit word of z's class, for 0 <= k < 2^20: (l + 2^32 h) + k e by a multiply-add. After a
// carry the word is below 2^52, so adding e once more cannot carry.
__device__ __forceinline__ u64 out(W3 z) {
  u32 l, h, c;
  // The instructions of this statement are, step for step and in order, those of the reduction in
  // class gl64_t of sppark (Supranational LLC; ff/gl64_t.cuh; Apache-2.0).
  // See THIRD_PARTY.md, "sppark".
  asm("mad.lo.cc.u32 %0, %5, 0xFFFFFFFF, %3;\n\t"
      "madc.hi.cc.u32 %1, %5, 0xFFFFFFFF, %4;\n\t"
      "addc.u32 %2, 0, 0;\n\t"
      "mad.lo.cc.u32 %0, %2, 0xFFFFFFFF, %0;\n\t"
      "madc.hi.u32 %1, %2, 0xFFFFFFFF, %1;"
      : "=&r"(l), "=&r"(h), "=&r"(c) : "r"(z.l), "r"(z.h), "r"(z.k));
  return l | ((u64)h << 32);
}
__host__ __device__ constexpr int reversed_bits(int m, int d) {
  int r = 0;
  for (int i = 0; i < d; i++) r |= ((m >> i) & 1) << (d - 1 - i);
  return r;
}
// One butterfly of layer d, if j is the lower index of its pair. The shift S is computed from
// template arguments because, in unrolled loops, the compiler kept it in a register.
template <int NL, int SK, int d, int j> __device__ __forceinline__ void one_pair(W3 (&v)[16]) {
  constexpr int bit = 3 - SK - d; // the bit of j that the layer pairs
  if constexpr (!(j >> bit & 1)) {
    constexpr int m = (j >> (bit + 1)) & ((1 << d) - 1), S = 96 * reversed_bits(m, d) >> d, k = j | (1 << bit);
    const W3 a = v[j];
    W3 b = v[k];
    if constexpr (S != 0) b = shifted<S>(b);
    v[j] = plus(a, b), v[k] = minus(a, b);
  }
}
template <int NL, int SK, int d, int... J> __device__ __forceinline__ void one_layer(W3 (&v)[16], std::integer_sequence<int, J...>) {
  (one_pair<NL, SK, d, J>(v), ...);
}
template <int NL, int SK, int... Dd> __device__ __forceinline__ void all_layers(W3 (&v)[16], std::integer_sequence<int, Dd...>) {
  (one_layer<NL, SK, Dd>(v, std::make_integer_sequence<int, 16>{}), ...);
}
// Runs NL layers on a thread's sixteen words x[j]. A layer pairs the words whose j differ in one of
// the top NL bits, highest bit first. The widest SK layers only copy (their second operands are
// zero: a message shorter than the codeword) and are skipped: the caller loads the copies. The
// other LG = NL - SK layers run on groups of 2^LG words. 0 <= SK <= NL <= 4 is not checked.
// `pw`: powers()'s table, from the first entry of the thread's layer block n0. pw[idx], idx = the
// top NL bits of j, is tau^jj for the group that idx's top SK bits pick and jj = idx's other bits.
//
// Derivation. A layer maps each pair (a, b) to (a + t b, a - t b). A layer at depth D has 2^D
// layer blocks (the widest layer has the fewest), and its block h has the twiddle factor
// T_D[h] = s_D W[h]. W[h] is the product of r_(i+2) over the set bits i of h, where r_n has order
// 2^n and r_n = r_(n+1)^2; so W[h] is a power of one root, with h's reversed bits as the exponent.
// s_D is a scalar of the layer (1 on the subgroup, a power of the shift on a coset) with
// s_D = s_(D+1)^2. The factors' form is not checked.
// Public source for these layers: decimation in time, factors in bit-reversed order: P. Longa,
// M. Naehrig, "Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based
// Cryptography", IACR ePrint 2016/504, section 2.2 and Algorithm 1 (there for the negacyclic
// transform), https://eprint.iacr.org/2016/504.
// A group's layers d = 0 .. LG - 1 are at depths D + d, within block H of layer 0. Block
// (H << d) + m, m < 2^d, of layer d has the factor T_(D+d)[H << d] W[m], and
// T_(D+d)[H << d] = tau^(2^(LG - 1 - d)) with tau = T_(D+LG-1)[H << (LG - 1)], the narrowest
// layer's factor. So the group multiplies its word jj by tau^jj (no product for jj = 0) and runs
// the same layers with the factors W[m], m < 8, which are 16th roots of unity. 2 has order 192
// modulo p, and blocks/code/cosets.h takes a root of order 2^32 whose power of order 64 is 8, so
// W[m] = 2^(96 reversed_bits(m, d) / 2^d): shifts by multiples of 12 bits below 96.
//
// Range of k, without word 0's bias. After the products |k| <= 1 (0 or -1 for jj = 0). A layer
// gives k_a +- k_b with a carry or borrow, so |k| <= 2 |k|_before + 1: 3, 7, 15, 31 after one to
// four layers, within shifted()'s bound.
template <int NL, int SK> __device__ __forceinline__ void group(u64 (&x)[16], const u64 *pw) {
  constexpr int LG = NL - SK;
  if constexpr (LG > 0) {
    W3 v[16];
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const int idx = j >> (4 - NL), jj = idx & ((1 << LG) - 1);
      v[j] = jj ? times(pw[idx], x[j]) : in(x[j]);
    }
    all_layers<NL, SK>(v, std::make_integer_sequence<int, LG>{});
#pragma unroll
    for (int j = 0; j < 16; j++) x[j] = out(v[j]);
  }
}
// Fills group<NL, SK>()'s table for NL layers that start at depth d0 of a tile (the words that a
// thread block transforms together): pw[n0 2^NL + idx] = tau^jj for each layer block n0 < 2^d0.
// `heap`: the tile's factors. Index 1 is the factor of its widest layer, indices 2 y and 2 y + 1
// are the layer blocks below index y; block n0 has index 2^d0 + n0. The skipped layers' factors are
// not read.
// `pitch`: if not 0, the words from one n0's powers to the next's, against bank conflicts. It must
// be at least 2^NL, which is not checked.
//
// NL and SK are template arguments, and the host picks the instantiation, because of a compiler
// fault. With the number of skipped layers as a run-time argument, clamped here to 0 .. NL, and a
// loop of the run-time length LG, CUDA 13.0's ptxas (-O1 and -O3; correct at -O0) made a kernel for
// NL = 3 that left the loop out whenever that argument was 0, 1 or 2 (it ran only for a negative
// one, in the later passes): all powers were 1, and every symbol whose factor is not 1 was wrong.
// In the machine code the loop was unrolled by four, and its test for remaining steps was taken
// from the predicate output of the clamping instruction.
// Such a fault does not show where tau is really 1: on the subgroup, and there only in a layer's
// first block.
template <int NL, int SK> __device__ __forceinline__ void powers(u64 *pw, const u64 *heap, int d0, int pitch = 0) {
  constexpr int LG = NL - SK;
  for (int e = threadIdx.x; e < (1 << (d0 + NL)); e += (int)blockDim.x) {
    const u32 n0 = e >> NL, idx = e & ((1 << NL) - 1), blk = idx >> LG, jj = idx & ((1u << LG) - 1);
    const u32 node = (((1u << d0) | n0) << SK) + blk;
    u64 v = 1;
#pragma unroll
    for (int b = 0; b < LG; b++)
      if (jj >> b & 1) v = canonical(mul(Loose{v}, Loose{heap[node << (LG - 1 - b)]})).v;
    pw[pitch ? n0 * pitch + idx : e] = v;
  }
}
} // namespace groups
} // namespace lazy
} // namespace blk
