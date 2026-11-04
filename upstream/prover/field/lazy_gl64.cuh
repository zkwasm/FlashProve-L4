// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Field arithmetic for the inner loops of GPU kernels, with delayed reduction (a sum of products is
// reduced modulo p once, at its end), for p = 2^64 - 2^32 + 1: the bodies of the functions that
// prover/field/field_lazy.cuh specifies. Include that file, not this one; that file also defines
// LAZY_HD (`__host__ __device__ __forceinline__` under nvcc, `inline` otherwise).
// F and E (blocks/field/field.h) are elements of the field and of its cubic extension, in canonical
// words (below p). The bounds stated at the functions are the caller's duty: nothing here checks
// them at run time.
// With e = 2^32 - 1: 2^64 = e, 2^96 = -1, 2^128 = -2^32, 2^192 = 1 (mod p). The residues
// x = 0 .. e - 1 have two 64-bit words, x and x + p; the others have one. There is no spare bit, so
// a carry out of a word is repaired by adding e, a borrow by subtracting e, and a sum of products
// keeps a fifth 32-bit word.
// Two rules for carry chains (asm instructions linked by the carry flag); breaking either gave
// wrong values on a GPU. An operand that is read and written is marked "+&r": with "+r" the
// compiler may give it the register of an input of equal value, which the chain then overwrites
// while it is still needed. subc never continues a chain of additions, nor addc one of
// subtractions: the machine keeps a borrow as the complement of a carry.
//
// Public source for the reduction: the plonky2 library, field/src/goldilocks_field.rs, function
// reduce128 (see blocks/field/gl64.h).
// Public source for the carry chains: NVIDIA, "Parallel Thread Execution ISA", chapter "Instruction
// Set", part "Extended-Precision Integer Arithmetic Instructions" (add.cc, addc, sub.cc, subc,
// mad.cc, madc).
// Public source for 128-bit integers in device code (reduce_loose(), shifted(), and
// TransformStep::out(), which also uses the signed type): NVIDIA, CUDA C++ Programming Guide,
// "Host Compiler Extensions". See also: NVIDIA, CUDA Programming Guide, "C/C++ Language
// Extensions", "Host Compiler Type Extensions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// The other public sources are cited below, where they are used.
#pragma once
namespace blk {
namespace lazy {
// GP is p, GE is e.
constexpr u64 GP = GL64::P, GE = GL64::EPS;

// ---- five words to one ------------------------------------------------------------------------------------------------------------
// w0 + 2^32 w1 + 2^64 w2 + 2^96 w3 + 2^128 w4 modulo p, not necessarily canonical, for w4 < 2^31.
// The value is congruent to V = L + 2^32 H' with L = w0 - w2 - w3 and H' = w1 + w2 - w4, taken as
// signed 64-bit numbers. L's upper half is carried into H, so that V = (low half of L) + 2^32 H,
// and H's upper half k is what V has above 64 bits: V = base + 2^64 k. The result is base + k e. It
// does not wrap: -2^31 - 1 <= H <= 2^33 - 2, so k is -1, 0 or 1; for k = 1,
// base <= 2^64 - 2^32 - 1; for k = -1, base >= (2^31 - 1) 2^32 > e.
// L >> 32 and H >> 32, like the signed right shifts in sub() and TransformStep::out(), must round
// down for negative numbers. Before C++20 the language leaves this to the compiler.
LAZY_HD u64 reduce160(u32 w0, u32 w1, u32 w2, u32 w3, u32 w4) {
  const long long L = (long long)w0 - (long long)w2 - (long long)w3;
  const long long H = (long long)w1 + (long long)w2 - (long long)w4 + (L >> 32);
  const long long k = H >> 32;
  const u64 base = (u32)L | ((u64)(u32)H << 32);
  return base + ((u64)k << 32) - (u64)k;
}

// ---- Loose ------------------------------------------------------------------------------------------------------------------------
// An element of F as a word of its residue class modulo p; canonical() gives the word below p.
struct Loose {
  u64 w; // any 64-bit word
};
LAZY_HD Loose loosen(F a) {
  return Loose{a.v};
}
LAZY_HD F canonical(Loose x) {
  return F{GL64::canonical(x.w)};
}
// After a carry the word is at most p - 2 (b is canonical), so adding e cannot carry again.
LAZY_HD Loose add(Loose a, F b) {
  const u64 lo = (u64)(u32)a.w + (u32)b.v, hi = (a.w >> 32) + (b.v >> 32) + (lo >> 32), k = hi >> 32;
  return Loose{((u32)lo | (hi << 32)) + (k << 32) - k};
}
// After a borrow the word is at least 2^32 > e (b is canonical).
LAZY_HD Loose sub(Loose a, F b) {
  const long long lo = (long long)(u32)a.w - (long long)(u32)b.v, hi = (long long)(a.w >> 32) - (long long)(b.v >> 32) + (lo >> 32),
                  k = hi >> 32;
  return Loose{((u32)lo | ((u64)hi << 32)) + ((u64)k << 32) - (u64)k};
}
// a b + c: the 128-bit number (it cannot overflow) and one reduction.
// Keeping the even and the odd words in two accumulators and joining them at the end is a known
// method: the wide product in sppark's ff/mont_t.cuh (Supranational LLC; Apache-2.0) has it too.
// See THIRD_PARTY.md, "sppark".
LAZY_HD Loose mad(Loose a, Loose b, Loose c) {
#ifdef __CUDA_ARCH__
  const u32 a0 = (u32)a.w, a1 = (u32)(a.w >> 32), b0 = (u32)b.w, b1 = (u32)(b.w >> 32), c0 = (u32)c.w, c1 = (u32)(c.w >> 32);
  u32 e0, e1, e2, e3, o0, o1, o2;
  asm("mad.lo.cc.u32 %0, %7, %9, %11;\n\t"
      "madc.hi.cc.u32 %1, %7, %9, %12;\n\t"
      "madc.lo.cc.u32 %2, %8, %10, 0;\n\t"
      "madc.hi.u32 %3, %8, %10, 0;\n\t"
      "mul.lo.u32 %4, %7, %10;\n\t"
      "mul.hi.u32 %5, %7, %10;\n\t"
      "mad.lo.cc.u32 %4, %8, %9, %4;\n\t"
      "madc.hi.cc.u32 %5, %8, %9, %5;\n\t"
      "addc.u32 %6, 0, 0;\n\t"
      "add.cc.u32 %1, %1, %4;\n\t"
      "addc.cc.u32 %2, %2, %5;\n\t"
      "addc.u32 %3, %3, %6;"
      : "=&r"(e0), "=&r"(e1), "=&r"(e2), "=&r"(e3), "=&r"(o0), "=&r"(o1), "=&r"(o2)
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
  return Loose{reduce160(e0, e1, e2, e3, 0)};
#else
  const u128 t = (u128)a.w * b.w + c.w;
  return Loose{reduce160((u32)t, (u32)(t >> 32), (u32)(t >> 64), (u32)(t >> 96), 0)};
#endif
}
LAZY_HD Loose mul(Loose a, Loose b) {
  return mad(a, b, Loose{0});
}

// ---- sums of products -----------------------------------------------------------------------------------------------------------------
// The sum is (e[0] + 2^32 e[1] + .. + 2^128 e[4]) + 2^32 (o[0] + 2^32 o[1] + 2^64 o[2]): two
// numbers, so that every 32 x 32-bit product is added at an even word (a pair of registers). A
// product of two 64-bit words takes six machine instructions in compiled code (mac_words() has
// eleven PTX instructions).
// MAC_ROOM counts calls: a call of mac() on an AccE adds up to three products to each coordinate.
// 3 * 2^29 products, each below 2^128, sum to less than 2^159, so word 4 of the joined sum stays
// below 2^31, as reduce160() requires. Their 6 * 2^29 mixed terms (a0 b1, a1 b0) sum to less than
// 2^96 and fit in o.
constexpr u64 MAC_ROOM = u64(1) << 29;
static_assert(3 * MAC_ROOM <= (u64(1) << 31), "3 MAC_ROOM products below 2^128 stay below 2^159: reduce160's fifth word is below 2^31");
// The caller must make no more than MAC_ROOM calls on one sum; no function in this file checks
// that.
// Public source for sums of products whose reduction is delayed (AccF, AccE, SAcc, FourKeys::get();
// IAcc in prover/field/lazy_signed.cuh): J.-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra
// over Word-Size Prime Fields: the FFLAS and FFPACK Packages", ACM TOMS 35(3), 2008,
// arXiv:cs/0601133, section 2.1.2, https://arxiv.org/pdf/cs/0601133.
// The idea is public: a sum of products that is reduced once: J.-G. Dumas, "Efficient dot product
// over word-size finite fields", section 3.1, https://arxiv.org/pdf/cs/0404008.
struct AccF {
  u32 e[5], o[3];
};
LAZY_HD AccF zero_f() {
  return AccF{{0, 0, 0, 0, 0}, {0, 0, 0}};
}
LAZY_HD void mac_words(AccF &s, u64 a, u64 b) { // any two words
  const u32 a0 = (u32)a, a1 = (u32)(a >> 32), b0 = (u32)b, b1 = (u32)(b >> 32);
#ifdef __CUDA_ARCH__
  asm("mad.lo.cc.u32 %0, %8, %10, %0;\n\t"
      "madc.hi.cc.u32 %1, %8, %10, %1;\n\t"
      "madc.lo.cc.u32 %2, %9, %11, %2;\n\t"
      "madc.hi.cc.u32 %3, %9, %11, %3;\n\t"
      "addc.u32 %4, %4, 0;\n\t"
      "mad.lo.cc.u32 %5, %8, %11, %5;\n\t"
      "madc.hi.cc.u32 %6, %8, %11, %6;\n\t"
      "addc.u32 %7, %7, 0;\n\t"
      "mad.lo.cc.u32 %5, %9, %10, %5;\n\t"
      "madc.hi.cc.u32 %6, %9, %10, %6;\n\t"
      "addc.u32 %7, %7, 0;"
      : "+&r"(s.e[0]), "+&r"(s.e[1]), "+&r"(s.e[2]), "+&r"(s.e[3]), "+&r"(s.e[4]), "+&r"(s.o[0]), "+&r"(s.o[1]), "+&r"(s.o[2])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
  const u128 t = (u128)((u64)a0 * b0) + ((u128)((u64)a1 * b1) << 64);
  const u128 lo = ((u128)s.e[3] << 96) | ((u128)s.e[2] << 64) | ((u128)s.e[1] << 32) | s.e[0], r = lo + t;
  s.e[4] += r < t;
  s.e[0] = (u32)r, s.e[1] = (u32)(r >> 32), s.e[2] = (u32)(r >> 64), s.e[3] = (u32)(r >> 96);
  const u128 q = (((u128)s.o[2] << 64) | ((u128)s.o[1] << 32) | s.o[0]) + (u64)a0 * b1 + (u64)a1 * b0;
  s.o[0] = (u32)q, s.o[1] = (u32)(q >> 32), s.o[2] = (u32)(q >> 64);
#endif
}
LAZY_HD void mac(AccF &s, F a, F b) {
  mac_words(s, a.v, b.v);
}
LAZY_HD void mac(AccF &s, Loose a, Loose b) {
  mac_words(s, a.w, b.w);
}
LAZY_HD void add(AccF &s, F a) {
  mac_words(s, a.v, 1);
}
LAZY_HD u64 reduce_loose(const AccF &s) {
  const u128 q = (((u128)s.e[4] << 96) | ((u128)s.e[3] << 64) | ((u128)s.e[2] << 32) | s.e[1]) +
                 (((u128)s.o[2] << 64) | ((u128)s.o[1] << 32) | s.o[0]);
  return reduce160(s.e[0], (u32)q, (u32)(q >> 32), (u32)(q >> 64), (u32)(q >> 96));
}
LAZY_HD F reduce(const AccF &s) {
  return F{GL64::canonical(reduce_loose(s))};
}

// ---- E ------------------------------------------------------------------------------------------------------------------------------------
// In E = F[X] / (X^3 - 2):
// (a b)_0 = a0 b0 + 2 (a1 b2 + a2 b1); (a b)_1 = a0 b1 + a1 b0 + 2 a2 b2;
// (a b)_2 = a0 b2 + a1 b1 + a2 b0.
struct Prepared {
  u64 a[5]; // a0, a1, a2, 2 a1, 2 a2, canonical
};
// The multiples 2 a1 and 2 a2, which a product's terms of degree 3 and 4 need (X^3 = 2), are
// computed beforehand for one factor: once for many products (mulp() and mac() on a Prepared) or
// in each call (mul() and mac() on two elements of E).
// The idea is public: multiples that are computed beforehand: D. J. Bernstein, P. Schwabe, "NEON
// Crypto", CHES 2012, LNCS 7428, pp. 320-339, section 4 and the paragraph "Multiplication mod
// 2^130 - 5 on NEON", https://link.springer.com/content/pdf/10.1007/978-3-642-33027-8_19.pdf
// The idea is public: the same, with multiplied copies of one factor's pieces: D. J. Bernstein,
// "The Poly1305-AES message-authentication code", document dated 2005.03.29, section 4,
// https://cr.yp.to/mac/poly1305-20050329.pdf.
// There the pieces are an integer's limbs and a factor 5 comes from the integer modulus
// 2^130 - 5; here they are an element's coordinates and the 2 comes from X^3 = 2. Here the
// multiples are reduced, since a word of this field has no spare bit.
LAZY_HD Prepared prepare(const E &x) {
  return Prepared{{x.c[0].v, x.c[1].v, x.c[2].v, GL64::add(x.c[1].v, x.c[1].v), GL64::add(x.c[2].v, x.c[2].v)}};
}
struct AccE {
  AccF r[3];
};
LAZY_HD AccE zero_e() {
  return AccE{{zero_f(), zero_f(), zero_f()}};
}
// Delayed reduction in an extension field, where the literature calls it lazy reduction: three
// products are added unreduced to each coordinate, and reduce() reduces each coordinate once, so
// mulp() reduces three sums. Here many products in E may be added before that reduction (MAC_ROOM).
// Public source for one reduction per coordinate of a product in F[X] / (X^k - w): D. F. Aranha,
// K. Karabina, P. Longa, C. H. Gebotys, J. Lopez, "Faster Explicit Formulas for Computing Pairings
// over Ordinary Curves", EUROCRYPT 2011, IACR ePrint 2010/526, section 3.1 (lazy reduction; it
// gives the count of k reductions as a result of Lim and Hwang), https://eprint.iacr.org/2010/526.
// The idea is public: lazy reduction: P. Longa, "Efficient Algorithms for Large Prime
// Characteristic Fields and Their Application to Bilinear Pairings", IACR Transactions on
// Cryptographic Hardware and Embedded Systems, Vol. 2023, No. 3, pp. 445-472, section 1 (Longa's
// account of it; the method that Longa's paper itself proposes departs from the algorithms that use
// it), https://tches.iacr.org/index.php/TCHES/article/download/10970/10277.
LAZY_HD void mac(AccE &s, const Prepared &a, const E &b) {
  mac_words(s.r[0], a.a[0], b.c[0].v), mac_words(s.r[0], a.a[3], b.c[2].v), mac_words(s.r[0], a.a[4], b.c[1].v);
  mac_words(s.r[1], a.a[0], b.c[1].v), mac_words(s.r[1], a.a[1], b.c[0].v), mac_words(s.r[1], a.a[4], b.c[2].v);
  mac_words(s.r[2], a.a[0], b.c[2].v), mac_words(s.r[2], a.a[1], b.c[1].v), mac_words(s.r[2], a.a[2], b.c[0].v);
}
LAZY_HD void mac(AccE &s, const E &a, const E &b) { // counts as one call against MAC_ROOM
  mac(s, prepare(a), b);
}
LAZY_HD void mac(AccE &s, const E &a, F b) {
  mac_words(s.r[0], a.c[0].v, b.v), mac_words(s.r[1], a.c[1].v, b.v), mac_words(s.r[2], a.c[2].v, b.v);
}
LAZY_HD void add(AccE &s, const E &a) {
  add(s.r[0], a.c[0]), add(s.r[1], a.c[1]), add(s.r[2], a.c[2]);
}
LAZY_HD E reduce(const AccE &s) {
  return E{{reduce(s.r[0]), reduce(s.r[1]), reduce(s.r[2])}};
}
LAZY_HD E mulp(const Prepared &a, const E &b) {
  AccE s = zero_e();
  mac(s, a, b);
  return reduce(s);
}
LAZY_HD E mul(const E &a, const E &b) {
  return mulp(prepare(a), b);
}

// ---- sums of (element of E) x (small integer) ------------------------------------------------------------------------------------------
// Each 32-bit half of a coordinate (l: low, h: high) times an integer k < 2^SMAC_BITS is below
// 2^56, so SMAC_TERMS such products fit a 64-bit word. Sums joined by sadd() count their terms
// together.
constexpr int SMAC_BITS = 24, SMAC_TERMS = 256;
struct SPrep {
  u32 l[3], h[3];
};
struct SAcc {
  u64 l[3], h[3];
};
LAZY_HD SPrep sprep(const E &e) {
  return SPrep{{(u32)e.c[0].v, (u32)e.c[1].v, (u32)e.c[2].v}, {(u32)(e.c[0].v >> 32), (u32)(e.c[1].v >> 32), (u32)(e.c[2].v >> 32)}};
}
LAZY_HD SAcc szero() {
  return SAcc{{0, 0, 0}, {0, 0, 0}};
}
LAZY_HD void sadd(SAcc &a, const SAcc &b) { // the six numbers are plain integers
#pragma unroll
  for (int c = 0; c < 3; c++) a.l[c] += b.l[c], a.h[c] += b.h[c];
}
LAZY_HD void smac(SAcc &a, const SPrep &e, u32 k) {
#pragma unroll
  for (int c = 0; c < 3; c++) a.l[c] = (u64)e.l[c] * k + a.l[c], a.h[c] = (u64)e.h[c] * k + a.h[c];
}
LAZY_HD F sjoin(u64 l, u64 h) { // l + 2^32 h, any two words
  const u64 mid = (l >> 32) + (u32)h, top = (h >> 32) + (mid >> 32); // top <= 2^32: its bit 32 goes to word 3
  return F{GL64::canonical(reduce160((u32)l, (u32)mid, (u32)top, (u32)(top >> 32), 0))};
}
LAZY_HD E sred(const SAcc &a) {
  return E{{sjoin(a.l[0], a.h[0]), sjoin(a.l[1], a.h[1]), sjoin(a.l[2], a.h[2])}};
}
// x 2^S.
template <int S> LAZY_HD F shifted(F x) {
  static_assert(S > 0 && S < 64, "a shift within a word");
  const u128 t = (u128)x.v << S;
  return F{GL64::canonical(reduce160((u32)t, (u32)(t >> 32), (u32)(t >> 64), (u32)(t >> 96), 0))};
}
template <int S> LAZY_HD E shifted(const E &a) {
  return E{{shifted<S>(a.c[0]), shifted<S>(a.c[1]), shifted<S>(a.c[2])}};
}

// ---- byte digits --------------------------------------------------------------------------------------------------------------------------
// signed_digits() returns DIGITS signed bytes d_i, d_0 the lowest: sum of d_i 256^i = x modulo p.
// Eight signed bytes represent the numbers from -128 (2^64 - 1) / 255 to MOST =
// 127 (2^64 - 1) / 255 = 0x7F7F7F7F7F7F7F7F. For a number in that range, adding 128 at every byte
// and then taking 128 off every byte (one xor) gives all digits at once. A canonical x above MOST
// is decomposed as x - p: its word modulo 2^64 is x + e (no wrap, since x < p), and the carry out
// of the eighth digit, 2^64, is dropped.
// A loop that carries from digit to digit must also start from x + e: started from x, its digits
// stand for x - 2^64, which is wrong by e, and nothing notices.
// Public source for all signed digits by one addition: the TFHE library's gadget decomposition
// (see prover/hiding/gram_products.cuh).
constexpr int DIGITS = 8;
// The same holds for a word x that is not canonical: x + e wraps to x - p, the canonical word.
LAZY_HD u64 signed_digits(Loose x) {
  constexpr u64 H = 0x8080808080808080ull, MOST = 0x7F7F7F7F7F7F7F7Full;
  u64 w = x.w;
  if (w > MOST) w += GE;
  return (w + H) ^ H;
}
LAZY_HD u64 signed_digits(F x) {
  return signed_digits(Loose{x.v});
}
// x 2^s, or -x 2^s if `minus`, for any 64-bit word x and 0 <= s < 192. The first line uses
// 2^96 = -1. Then x 2^s fills five words; the fifth has at most the top 31 bits of x, as
// reduce160() requires.
LAZY_HD F word_times_2_to(u64 x, int s, bool minus) {
  if (s >= 96) s -= 96, minus = !minus;
  const int q = s >> 5, r = s & 31;
  // the two shifts by 32 - r are shifts of 64-bit words, so r = 0 is defined and gives the words
  // themselves
  const u32 y0 = (u32)x << r, y1 = (u32)(x >> (32 - r)), y2 = (u32)((x >> 32) >> (32 - r));
  u32 w[5] = {0, 0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const u32 y = i == 0 ? y0 : i == 1 ? y1 : y2;
#pragma unroll
    for (int j = 0; j < 5; j++)
      if (j == q + i) w[j] = y;
  }
  const u64 z = GL64::canonical(reduce160(w[0], w[1], w[2], w[3], w[4]));
  return F{minus ? GL64::neg(z) : z};
}
// field_lazy.cuh specifies join_digit() for |v| < 2^JOIN_BITS only; the bound is not checked.
constexpr int JOIN_BITS = 61;
LAZY_HD F join_digit(i64 v, int s) {
  return word_times_2_to(v < 0 ? 0 - (u64)v : (u64)v, s, v < 0);
}
LAZY_HD F times_2_to(F x, int s) {
  return word_times_2_to(x.v, s, false);
}
LAZY_HD E times_2_to(const E &a, int s) {
  return E{{times_2_to(a.c[0], s), times_2_to(a.c[1], s), times_2_to(a.c[2], s)}};
}

// ---- four keys ----------------------------------------------------------------------------------------------------------------------------
// The product of four factors alpha - v_i of a lookup argument, from its integer keys v_i.
// prod (alpha - v_i) = alpha^4 - e1 alpha^3 + e2 alpha^2 - e3 alpha + e4 with the elementary
// symmetric polynomials of the v_i (18, 35, 50 and 64 bits for v_i < 2^16, not checked): one sum
// of three products a coordinate, plus alpha^4 and, at c = 0, e4.
// FourKeys::prepare() stores n3 = -alpha^3, p2 = alpha^2, n1 = -alpha and a4 = alpha^4.
struct FourKeys {
  u64 n3[3], p2[3], n1[3], a4[3];
  LAZY_HD void prepare(const E &alpha) {
    const E x2 = mul(alpha, alpha), x3 = mul(x2, alpha), x4 = mul(x2, x2);
#pragma unroll
    for (int c = 0; c < 3; c++) n3[c] = GL64::neg(x3.c[c].v), p2[c] = x2.c[c].v, n1[c] = GL64::neg(alpha.c[c].v), a4[c] = x4.c[c].v;
  }
  // Public sources for this function's techniques. Delayed reduction, a public idea: Dumas, Giorgi,
  // Pernet, section 2.1.2 (cited at AccF). The compiler hint is public: #pragma unroll: NVIDIA,
  // "CUDA Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
  // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
  // The idea is public: r.c, three entries, can stay in registers, as unrolling puts 0, 1, 2 for c:
  // M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
  // "Static indexing",
  // https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
  LAZY_HD E get(u32 v0, u32 v1, u32 v2, u32 v3) const {
    const u32 a = v0 * v1, b = v2 * v3, s = v0 + v1, t = v2 + v3, e1 = s + t;
    const u64 e2 = (u64)a + b + (u64)s * t, e3 = (u64)a * t + (u64)b * s, e4 = (u64)a * b;
    E r;
#pragma unroll
    for (int c = 0; c < 3; c++) {
      AccF z = zero_f();
      z.e[0] = (u32)a4[c], z.e[1] = (u32)(a4[c] >> 32);
      mac_words(z, e1, n3[c]), mac_words(z, e2, p2[c]), mac_words(z, e3, n1[c]);
      if (c == 0) mac_words(z, e4, 1);
      r.c[c] = reduce(z);
    }
    return r;
  }
};

// ---- small things -------------------------------------------------------------------------------------------------------------------------
LAZY_HD F half(F x) { // x / 2 = floor(x / 2) + (x odd ? (p + 1) / 2 : 0); the sum is below p
  return F{(x.v >> 1) + ((x.v & 1) ? (GP >> 1) + 1 : 0)};
}
// NR x, where X^3 = NR in E: GL64::NR is 2.
LAZY_HD F times_nr(F x) {
  return F{GL64::add(x.v, x.v)};
}

// ---- the transform's step -----------------------------------------------------------------------------------------------------------------
// butterfly() is one step of a layer of a number-theoretic transform: it maps (a, b) to
// (a + t b, a - t b) for a factor t from the transform's table.
// Between a transform's layers values stay unreduced, after D. Harvey, "Faster arithmetic for
// number-theoretic transforms", arXiv:1205.2926. Here such a value is
// v + 2^64 (k - BIAS): a 64-bit word v (halves l and h) and a small number k, stored with a bias.
// A sum or a difference is one carry chain of three instructions and needs no repair.
// The product takes b's k along:
// t (v + 2^64 (k - BIAS)) = t v + 2^64 k t + c, where c = -BIAS 2^64 t (mod p) is stored with t and
// is the product's addend. Its five words are reduced to rl + 2^32 rh + 2^64 rk, rk = -1, 0 or 1.
// A sum changes k by rk + carry, from -1 to 2, a difference by -rk - borrow, from -2 to 1. Over at
// most sixteen layers k stays within 0 .. 64: never negative, which is what limits the layers, and
// within the byte that put() stores in `more`. MORE = true tells a kernel to keep that array of
// bytes beside its words.
// A value must become canonical only after a transform's last layer: canonical() does that; out()
// gives a word that need not be canonical.
#ifdef __CUDACC__
struct TransformStep {
  static constexpr bool MORE = true;
  static constexpr u32 BIAS = 32;
  struct Word {
    u32 l, h, k;
  };
  typedef ulonglong2 Tw; // x: t, y: c
  static __device__ __forceinline__ Word in(u64 x) { return Word{(u32)x, (u32)(x >> 32), BIAS}; }
  // v + (k - BIAS) e as one word. S = v + k e is below 2^64 + 2^38; BIAS e is taken off; what is
  // then above or below the word, c = -1, 0 or 1, is repaired by c e, which cannot wrap: after a
  // carry the word is below 2^38, after a borrow above 2^64 - 2^38.
  static __device__ __forceinline__ u64 out(Word z) {
    const u64 v = z.l | ((u64)z.h << 32);
    const __int128 T = (__int128)((u128)v + (u64)z.k * GE) - (__int128)((u64)BIAS * GE);
    const long long c = (long long)(T >> 64);
    return (u64)T + ((u64)c << 32) - (u64)c;
  }
  static __device__ __forceinline__ u64 canonical(Word z) { return GL64::canonical(out(z)); }
  static __device__ __forceinline__ void put(u64 *words, unsigned char *more, u32 at, Word z) {
    words[at] = z.l | ((u64)z.h << 32), more[at] = (unsigned char)z.k;
  }
  static __device__ __forceinline__ Word get(const u64 *words, const unsigned char *more, u32 at) {
    const u64 v = words[at];
    return Word{(u32)v, (u32)(v >> 32), more[at]};
  }
  static __device__ __forceinline__ Tw prepare(u64 t) { // t canonical
    return Tw{t, GL64::neg(lazy::canonical(mul(Loose{t}, Loose{(u64)BIAS * GE})).v)};
  }
  // Public source: the butterfly is that of Patrick Longa and Michael Naehrig, ePrint 2016/504,
  // Algorithm 1 (Function NTT based on the Cooley-Tukey butterfly): "Speeding up the Number
  // Theoretic Transform for Faster Ideal Lattice-Based Cryptography",
  // https://eprint.iacr.org/2016/504.
  static __device__ __forceinline__ void butterfly(Word &a, Word &b, Tw tw) {
    const u32 t0 = (u32)tw.x, t1 = (u32)(tw.x >> 32), c0 = (u32)tw.y, c1 = (u32)(tw.y >> 32);
    u32 e0, e1, e2, e3, e4, o0, o1, o2, o3;
    asm("mad.lo.cc.u32 %0, %9, %11, %14;\n\t" // t0 bl + c at word 0, t1 bh at word 2: no carry leaves word 3
        "madc.hi.cc.u32 %1, %9, %11, %15;\n\t"
        "madc.lo.cc.u32 %2, %10, %12, 0;\n\t"
        "madc.hi.u32 %3, %10, %12, 0;\n\t"
        "mad.lo.cc.u32 %2, %13, %9, %2;\n\t" // k t0 at word 2
        "madc.hi.cc.u32 %3, %13, %9, %3;\n\t"
        "addc.u32 %4, 0, 0;\n\t"
        "mul.lo.u32 %5, %9, %12;\n\t" // t0 bh and t1 bl at word 1, k t1 at word 3
        "mul.hi.u32 %6, %9, %12;\n\t"
        "mad.lo.cc.u32 %5, %10, %11, %5;\n\t"
        "madc.hi.cc.u32 %6, %10, %11, %6;\n\t"
        "madc.lo.cc.u32 %7, %13, %10, 0;\n\t"
        "madc.hi.u32 %8, %13, %10, 0;\n\t"
        "add.cc.u32 %1, %1, %5;\n\t"
        "addc.cc.u32 %2, %2, %6;\n\t"
        "addc.cc.u32 %3, %3, %7;\n\t"
        "addc.u32 %4, %4, %8;"
        : "=&r"(e0), "=&r"(e1), "=&r"(e2), "=&r"(e3), "=&r"(e4), "=&r"(o0), "=&r"(o1), "=&r"(o2), "=&r"(o3)
        : "r"(t0), "r"(t1), "r"(b.l), "r"(b.h), "r"(b.k), "r"(c0), "r"(c1));
    u32 rl, rh, rk; // (e0 + 2^32 e1) + e e2 - (e3 + 2^32 e4)
    asm("mad.lo.cc.u32 %0, %5, 0xFFFFFFFF, %3;\n\t"
        "madc.hi.cc.u32 %1, %5, 0xFFFFFFFF, %4;\n\t"
        "addc.u32 %2, 0, 0;\n\t"
        "sub.cc.u32 %0, %0, %6;\n\t"
        "subc.cc.u32 %1, %1, %7;\n\t"
        "subc.u32 %2, %2, 0;"
        : "=&r"(rl), "=&r"(rh), "=&r"(rk)
        : "r"(e0), "r"(e1), "r"(e2), "r"(e3), "r"(e4));
    Word x, y;
    asm("add.cc.u32 %0, %3, %6;\n\t"
        "addc.cc.u32 %1, %4, %7;\n\t"
        "addc.u32 %2, %5, %8;"
        : "=&r"(x.l), "=&r"(x.h), "=&r"(x.k)
        : "r"(a.l), "r"(a.h), "r"(a.k), "r"(rl), "r"(rh), "r"(rk));
    asm("sub.cc.u32 %0, %3, %6;\n\t"
        "subc.cc.u32 %1, %4, %7;\n\t"
        "subc.u32 %2, %5, %8;"
        : "=&r"(y.l), "=&r"(y.h), "=&r"(y.k)
        : "r"(a.l), "r"(a.h), "r"(a.k), "r"(rl), "r"(rh), "r"(rk));
    a = x, b = y;
  }
};
#endif
} // namespace lazy
} // namespace blk
