// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Field arithmetic for the inner loops of GPU kernels (namespace blk::lazy).
//
// blocks/field/field.h keeps every element of the base field F and of its cubic extension E
// canonical: each word is the least residue. That is too slow for loops that add up millions of
// products. The types here let such a loop put off the reduction, and the loop still gets the same
// field elements as with blk's operators. Every function except those of TransformStep also
// compiles for the host.
// The function bodies are in prover/field/lazy_gl64.cuh, for the prime p = 2^64 - 2^32 + 1.
// A caller that keeps to the table below need not know the prime.
// The public sources that the bodies follow are cited in the comment at the top of that file, at
// its functions, in prover/field/lazy_signed.cuh and in prover/field/transform_groups_gl64.cuh.
//
// The table states the form of each function's operands and result (canonical; loose, that is, not
// necessarily the least residue; or an integer within a bound) and the capacities: how many terms a
// sum, or layers a transform, can take before it must be reduced. Nothing checks these at run time,
// and beyond them a result can be silently wrong: a caller must derive each bound from its own
// sizes. To make an element of F from a signed integer, use blk::from_i64(); from a random 64-bit
// word, blk::F_from_word(), which may reject the word.
#pragma once
#include "base.h"
#include "field/field.h"

#if defined(__CUDACC__)
#define LAZY_HD __host__ __device__ __forceinline__
#else
#define LAZY_HD inline
#endif

#include "field/lazy_gl64.cuh"

namespace blk {
namespace lazy {
// ---- what the bodies define, with the meaning that a caller may rely on -------------------------
//
// struct Loose { u64 w; }
//   A word of the element's residue class, not necessarily the least: any 64-bit word is allowed.
//   Loose loosen(F a)                    no cost.
//   F     canonical(Loose x)             the element.
//   Loose add(Loose a, F b), sub(Loose a, F b)      b must be canonical, hence an F.
//   Loose mul(Loose a, Loose b)          the product.
//   Loose mad(Loose a, Loose b, Loose c) a b + c, at the cost of mul().
//
// E mul(const E &a, const E &b)          the product in E, canonical operands and result: three
//                                        sums of three products, each reduced once. Faster than
//                                        blk's operator* on the GPU.
// struct Prepared; Prepared prepare(const E &a)
//                                        a as the left factor of many products; 40 bytes, which
//                                        may be stored in shared or global memory.
// E mulp(const Prepared &a, const E &b)  a b.
//
// struct AccF                            a sum of products of two words, 32 bytes of registers.
//   AccF zero_f(); void mac(AccF &s, F a, F b); void mac(AccF &s, Loose a, Loose b);
//   void add(AccF &s, F a); F reduce(const AccF &s)
//                                        mac(): s += a b. add(): s += a. reduce(): the element.
// struct AccE                            three AccF: a sum of products in E, 96 bytes of registers.
//   AccE zero_e(); void mac(AccE &s, const E &a, const E &b);
//   void mac(AccE &s, const Prepared &a, const E &b); void mac(AccE &s, const E &a, F b);
//   void add(AccE &s, const E &a); E reduce(const AccE &s)
//   constexpr u64 MAC_ROOM               2^29. Capacity: at most MAC_ROOM calls of mac() or add()
//                                        on one AccF or AccE between zero_f() or zero_e() and
//                                        reduce().
//
// struct SPrep, SAcc                     sums of e k for elements e of E and integers k,
//                                        0 <= k < 2^SMAC_BITS (24).
//   SPrep sprep(const E &e); SAcc szero(); void smac(SAcc &s, const SPrep &e, u32 k);
//   E sred(const SAcc &s)                smac(): s += e k. sred(): the element.
//                                        Capacity: at most SMAC_TERMS (2^8) calls of smac() between
//                                        szero() and sred(); more generally (calls) x (largest k)
//                                        <= 2^32. Six multiply-adds a call, no carries.
//   void sadd(SAcc &s, const SAcc &t)    s += t. The six numbers of an SAcc are plain integers, so
//                                        two sums may also be added number by number or through a
//                                        warp shuffle. The calls of all sums that are added count
//                                        together against the capacity.
//   template <int S> F shifted(F x), E shifted(const E &a)
//                                        2^S times the element, for 0 < S < 64 known at compile
//                                        time; used to weight the pieces of a longer integer.
//                                        No test runs S = 61, 62 or 63.
// struct IPrep, IAcc                     sums of e k for integers k of either sign, with another
//                                        capacity: see
//                                        lazy_signed.cuh.
//
// u64 signed_digits(F x), (Loose x)      DIGITS = 8 signed bytes d_0 .. d_7, d_0 in the lowest
//                                        byte, -128 <= d_i <= 127, with sum of d_i 256^i = x modulo
//                                        p (as an integer the sum is x or x - p).
// F join_digit(i64 v, int s)             v 2^s modulo p, for |v| < 2^JOIN_BITS (61), 0 <= s < 192.
//                                        For example, v is a sum of products of two digits and 2^s
//                                        the weight of their positions m and m': s = 8 (m + m') <=
//                                        112. Digits spaced more widely than bytes have larger
//                                        weights. Never pass an element's word as v: it is not a
//                                        small integer (use times_2_to()).
// F times_2_to(F x, int s), E times_2_to(const E &a, int s)
//                                        2^s x for 0 <= s < 192, s a run-time number (shifted<S>()
//                                        is cheaper).
//
// struct FourKeys { void prepare(const E &alpha); E get(u32 v0, u32 v1, u32 v2, u32 v3) const; }
//                                        the product of the four alpha - v_i, for integers
//                                        v_i < 2^16 (keys of a lookup argument). 96 bytes of
//                                        registers.
//
// F half(F x)                            x / 2.        F times_nr(F x)   NR x (X^3 = NR in E).
//
// struct TransformStep                   GPU only. One step (a, b) -> (a + t b, a - t b) of a layer
//                                        of the fast transform that computes codewords, on values
//                                        that are not reduced.
//   Word          a value between two layers, in registers
//   Tw            a twiddle factor t from the transform's table, in the form that a kernel keeps
//   in(), out()   from and to a 64-bit word in GPU memory (for in(): canonical, or made by out())
//   put(), get()  to and from a tile in shared memory: for each value a 64-bit word in `words` and
//                 a byte in `more` (MORE = true says that the byte is needed)
//   prepare()     makes a Tw from the table's canonical word
//   butterfly()   the step
//   canonical()   the canonical word; call it once, after the last layer
//   Capacity: at most sixteen layers between in() and out() or canonical().
//   transform_groups_gl64.cuh takes up to four layers at a time and is about a fifth faster
//   (measured for one shape on one H100).
static_assert(MAC_ROOM >= (u64(1) << 29) && SMAC_BITS == 24 && SMAC_TERMS == 256 && DIGITS == 8 && JOIN_BITS == 61,
              "both bodies promise the same capacities");
} // namespace lazy
} // namespace blk
#include "field/lazy_signed.cuh"
