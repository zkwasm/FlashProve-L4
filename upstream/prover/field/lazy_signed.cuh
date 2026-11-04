// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.
//
// Sums of products e k, for elements e of the cubic extension field E and small integers k of
// either sign, such as signed bytes of the weights, reduced modulo the prime p only once, at the
// end (delayed reduction: source at ired()). smac() takes unsigned k only.
// Include field_lazy.cuh, which provides smac() and the names used here, not this file.
//
// iprep() splits each coordinate of e, which must be canonical (less than p; not checked), into
// two signed 32-bit limbs, so that a term costs one signed 32 x 32 -> 64-bit multiply-add per
// limb, in imac(). ired() gives the canonical element.
// The caller must keep within the capacity stated at IMAC_ROOM_BITS: imac() does not check it.
#pragma once
#include <cstdint>
namespace blk {
namespace lazy {
using i32 = std::int32_t;
// Capacity: (calls of imac() between izero() and ired()) x (largest |k|) <= 2^IMAC_ROOM_BITS.
// A limb is an i32, at most 2^31 in absolute value, so each sum stays within
// 2^(31 + IMAC_ROOM_BITS) = 2^60, below the 2^JOIN_BITS that join_digit() takes.
constexpr int IMAC_ROOM_BITS = 29;
// An element of E: coordinate c is l[c] + 2^ILIMB_BITS h[c] modulo p. It may be kept in shared or
// global memory.
struct IPrep {
  i32 l[3], h[3];
};
// A sum of terms e k: l[c] adds up k times the limb l[c] of e, and h[c] likewise. A caller may form
// or add such sums itself and pass them to ired(); their terms then count together against the
// capacity.
struct IAcc {
  i64 l[3], h[3];
};
// p = 2^64 - 2^32 + 1. Two signed 32-bit limbs, as lo + 2^32 hi, represent exactly the integers
// from -2^63 - 2^31 to 2^63 - 2^31 - 1. So a canonical word x is centred:
// xc = x if x < 2^63 - 2^31, otherwise xc = x - p, which is between -2^63 + 2^31 - 1 and -1.
// The code gets x - p as x + GE read as signed: GE is 2^64 - p, and the sum does not wrap.
// lo is the low half of xc read as signed. xc - lo = 2^32 hi is at least -2^63: no overflow.
// The right shift must be arithmetic, which C++ guarantees only from C++20.
constexpr int ILIMB_BITS = 32;
LAZY_HD void ilimbs(u64 x, i32 &lo, i32 &hi) {
  const i64 xc = (i64)(x < 0x7FFFFFFF80000000ull ? x : x + GE);
  lo = (i32)(u32)(u64)xc;
  hi = (i32)((xc - lo) >> 32);
}
LAZY_HD IPrep iprep(const E &e) {
  IPrep r;
#pragma unroll
  for (int c = 0; c < 3; c++) ilimbs(e.c[c].v, r.l[c], r.h[c]);
  return r;
}
LAZY_HD IAcc izero() {
  return IAcc{{0, 0, 0}, {0, 0, 0}};
}
// Adds without a reduction: see ired().
// The compiler hint, also in iprep(), is public: #pragma unroll: NVIDIA, "CUDA
// Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
LAZY_HD void imac(IAcc &a, const IPrep &e, i32 k) {
#pragma unroll
  for (int c = 0; c < 3; c++) a.l[c] += (i64)e.l[c] * k, a.h[c] += (i64)e.h[c] * k;
}
// join_digit(v, s) gives v 2^s modulo p, so each of the six sums is reduced once. This is delayed
// reduction (delayed division) in modular arithmetic, a public idea: J.-G. Dumas, P. Giorgi,
// C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK packages",
// ACM TOMS 35(3), 2008, arXiv:cs/0601133, section 2.1.2, https://arxiv.org/pdf/cs/0601133.
LAZY_HD E ired(const IAcc &a) {
  return E{{join_digit(a.l[0], 0) + join_digit(a.h[0], ILIMB_BITS), join_digit(a.l[1], 0) + join_digit(a.h[1], ILIMB_BITS),
            join_digit(a.l[2], 0) + join_digit(a.h[2], ILIMB_BITS)}};
}
static_assert(31 + IMAC_ROOM_BITS < JOIN_BITS, "a sum of limbs times integers stays within what join_digit takes");
} // namespace lazy
} // namespace blk
