// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// FK2 extends the traits struct FK (prover/hiding/field.h) of the cubic extension field
// K3 = F_p[x] / (x^3 - 2), p = 2^64 - 2^32 + 1, for loops that reduce a sum of products once. It
// forwards products in K3 and reductions to prover/field/field_lazy.cuh. What such a loop delays is
// a reduction modulo p, which floating-point and machine-integer arithmetic do not have. K3
// operands must be canonical (reduced to 0 .. p - 1), which is not checked; results are canonical.
// L3 and M3L store each 64-bit word as two 32-bit halves. A caller should use the functions of
// prover/field/field_lazy.cuh rather than compute with the halves itself: a sum of products of two
// halves can overflow 64 bits, and a half can turn negative in a cast to int.
// The compiler hint is public: #pragma unroll on loops over small arrays: M. Milakov, "Fast Dynamic
// Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, part "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
#pragma once
#include "hiding/field.h"
#include "field/field_lazy.cuh"
// K3, the prover's element type, and blk::E (blocks/field/field.h) have the same three coordinates
// in the same order.
HD blk::E e_of(const K3 &x) {
  return blk::E{{blk::F{x.k0}, blk::F{x.k1}, blk::F{x.k2}}};
}
HD K3 k3_of(const blk::E &e) {
  return {e.c[0].v, e.c[1].v, e.c[2].v};
}
// 32 x 32-bit multiply-add: returns x*y + c modulo 2^64.
HD u64 wm(u32 x, u32 y, u64 c) {
  return (u64)x * y + c;
}
// An element's coordinates in 32-bit halves: coordinate i = l[i] + 2^32 h[i].
struct L3 {
  u32 l[3], h[3];
};
HD L3 l3(K3 x) {
  return {{(u32)x.k0, (u32)x.k1, (u32)x.k2}, {(u32)(x.k0 >> 32), (u32)(x.k1 >> 32), (u32)(x.k2 >> 32)}};
}
// The left operand of a product, prepared once for many products: k0, k1, k2, 2 k1, 2 k2. The
// doubled coordinates serve the terms in x^3 and x^4, which wrap around by x^3 = 2. These are the
// five words of blk::lazy::Prepared, each in its two 32-bit halves.
// This is a public idea that is specific to modular arithmetic, here in K3: precomputed multiples:
// D. J. Bernstein and P. Schwabe, "NEON Crypto", CHES 2012, LNCS 7428, pp. 320-339, section 4,
// https://link.springer.com/content/pdf/10.1007/978-3-642-33027-8_19.pdf;
// this is a public idea that is specific to modular arithmetic: one factor's pieces stored
// premultiplied: D. J. Bernstein, "The Poly1305-AES message-authentication code", document dated
// 2005.03.29, section 4, https://cr.yp.to/mac/poly1305-20050329.pdf.
// There the 5 comes from the integer modulus 2^130 - 5, here the 2 from x^3 = 2.
struct alignas(8) M3L {
  u32 l[5], h[5];
};
HD M3L m3l(K3 x) {
  const blk::lazy::Prepared p = blk::lazy::prepare(e_of(x));
  M3L r;
#pragma unroll
  for (int i = 0; i < 5; i++) r.l[i] = (u32)p.a[i], r.h[i] = (u32)(p.a[i] >> 32);
  return r;
}
HD blk::lazy::Prepared prepared_of(const M3L &x) {
  blk::lazy::Prepared p;
#pragma unroll
  for (int i = 0; i < 5; i++) p.a[i] = x.l[i] | ((u64)x.h[i] << 32);
  return p;
}
struct FK2 : FK { // same field and storage as FK; products and sums by field/field_lazy.cuh
  static constexpr const char *name = "K3f";
  // mulp(prep(x), y) = mul(x, y) = x*y. The three products of a coordinate are reduced as one sum.
  // The idea is public: lazy reduction in an extension field (also in mac()): P. Longa, "Efficient
  // Algorithms for Large Prime Characteristic Fields and Their Application to Bilinear Pairings",
  // IACR Transactions on Cryptographic Hardware and Embedded Systems, Vol. 2023, No. 3,
  // pp. 445-472, section 1, https://tches.iacr.org/index.php/TCHES/article/download/10970/10277,
  // which names earlier works for it.
  // The idea is public: one reduction for each coordinate: the same section of Longa, and
  // D. F. Aranha, K. Karabina, P. Longa, C. H. Gebotys, J. Lopez, "Faster Explicit Formulas for
  // Computing Pairings over Ordinary Curves", EUROCRYPT 2011, IACR ePrint 2010/526, section 3.1,
  // https://eprint.iacr.org/2010/526, which gives this count of reductions as a result of Lim and
  // Hwang (its reference [24]: C. H. Lim and H. S. Hwang, "Fast Implementation of Elliptic Curve
  // Arithmetic in GF(p^n)", PKC 2000).
  // Both sections state it for a product in F_p[x] / (x^k - w): k reductions modulo p suffice,
  // where a conventional product would require k^2; here k = 3.
  typedef M3L P;
  static HD P prep(T x) { return m3l(x); }
  static HD T mulp(const P &x, T y) { return k3_of(blk::lazy::mulp(prepared_of(x), e_of(y))); }
  static HD T mul(T x, T y) { return k3_of(blk::lazy::mul(e_of(x), e_of(y))); }
  // Acc is a sum of products: mac() adds x*y unreduced, and ared() reduces the sum. At most
  // blk::lazy::MAC_ROOM = 2^29 calls of mac() between azero() and ared(), which is not checked.
  // The idea is public: delayed reduction of a sum of products: J.-G. Dumas, P. Giorgi, C. Pernet,
  // "Dense Linear Algebra over Word-Size Prime Fields: the FFLAS and FFPACK Packages",
  // ACM TOMS 35(3), 2008, arXiv:cs/0601133, section 2.1.2, https://arxiv.org/pdf/cs/0601133
  // (there the sum is reduced just before an overflow or after a fixed number of accumulations).
  // This is a public idea that is specific to modular arithmetic: a sum of products reduced once:
  // J.-G. Dumas, "Efficient dot product over word-size finite fields", section 3.1,
  // https://arxiv.org/pdf/cs/0404008.
  typedef blk::lazy::AccE Acc;
  static HD Acc azero() { return blk::lazy::zero_e(); }
  static HD void mac(Acc &a, const P &x, T y) { blk::lazy::mac(a, prepared_of(x), e_of(y)); }
  static HD T ared(const Acc &a) { return k3_of(blk::lazy::reduce(a)); }
  // SAcc is a sum of terms k*e, for an integer k and an element e that sprep() has split into
  // 32-bit halves (SP): smac() adds one term unreduced, and sred() reduces the sum. Not checked:
  // (number of terms) x (largest k) <= 2^32, for example 2^8 terms with k < 2^24. Within this bound
  // no 64-bit sum overflows, because a half is below 2^32.
  // The idea is public: delayed reduction of a sum of products: Dumas, Giorgi, Pernet, "Dense
  // Linear Algebra over Word-Size Prime Fields", section 2.1.2 (cited at mac()).
  typedef L3 SP;
  struct SAcc {
    u64 l[3], h[3];
  };
  static HD SP sprep(T x) { return l3(x); }
  static HD SAcc szero() { return {{0, 0, 0}, {0, 0, 0}}; }
  static HD void smac(SAcc &a, const SP &e, u32 k) {
#pragma unroll
    for (int c = 0; c < 3; c++) {
      a.l[c] = wm(e.l[c], k, a.l[c]);
      a.h[c] = wm(e.h[c], k, a.h[c]);
    }
  }
  static HD T sred(const SAcc &a) {
    return {blk::lazy::sjoin(a.l[0], a.h[0]).v, blk::lazy::sjoin(a.l[1], a.h[1]).v, blk::lazy::sjoin(a.l[2], a.h[2]).v};
  }
};
// FE is the traits struct of Ext (prover/common/common.h) that the prover's kernels use.
typedef FK2 FE;
static_assert(sizeof(M3L) == 40 && sizeof(L3) == 24 && sizeof(FE::SAcc) == 48, "kernels lay these out in shared memory by their sizes");
static_assert(sizeof(FE::T) == sizeof(Ext) && sizeof(Ext) == 8 * EDEG, "FE must be the trait of Ext");
