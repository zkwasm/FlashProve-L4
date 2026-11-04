// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU implementation of LkpProver (blocks/iop/lkp.h) for lookup keys that are too many for GPU
// memory: the product trees of the lookup argument LKP (every key occurs in a public table) and the
// sumchecks over them, without masks. An LkpKeySource writes some keys at a time, as often as it
// is asked. The table must be one plain range table 0 .. 2^d - 1, d <= 16, without a shift
// (LkpPublic::shifted). One object serves one run of LKP. The comment at LkpPiece describes the
// method.
// Hidden values: the keys, the stored levels, the folded arrays, the matrices, the sums, the roots
// and the end values; on the CPU also the multiplicities, the counts and the table side's levels.
// Every GPU buffer of an LkpPieces that has held one is zeroed before it is released (the
// destructor calls wipe()); for the CPU see wipe(). Where a failed check (BLK_REQUIRE, ENGINE_CK)
// ends the process, this file zeroes nothing. What key_roots(), plane_roots(), round() and
// end_values() return is the caller's to zero.
// Public source for the matrix passes (comment at LkpPiece), that is, for rounds cut into stages,
// each served by a table that one pass over the input fills: A. Chiesa, E. Fedele, G. Fenzi,
// A. Zitek-Estrada, "A Time-Space Tradeoff for the Sumcheck Prover", Cryptology ePrint Archive
// 2024/524 (version of August 15, 2024), section 4. That note sums one multilinear polynomial; here
// two are multiplied, so the table has an entry for each pair of values.
// Public sources for the byte digits (at LKPP_ROWS), cited in full in
// prover/hiding/gram_products.cuh: S. Fan et al., "TensorFHE", arXiv:2212.14191, section IV-C;
// H. Ootomo, K. Ozaki, R. Yokota, "DGEMM on Integer Matrix Multiplication Unit",
// arXiv:2306.11975, 4.1.
#pragma once
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/blas.h"
#include "engine/device.cuh"
#include "engine/left_unzeroed.h"
#include "engine/sumcheck_cpu.h"
#include "field/field.h"
#include "field/field_lazy.cuh"
#include "iop/lkp.h"

namespace blk {

typedef unsigned short Key16;  // a key: a row's number in a table of at most 2^16 rows

// How a part of the keys (LkpPublic::parts) is proved. A part has 2^v keys, 2^k_col of them (the
// columns) at each place: key number = column + 2^k_col * place. (x | y) joins bit fields, the
// highest first.
//
// The product tree, for LKP's challenge alpha: KL_v[i] = alpha - key[i];
// KL_k[i] = KL_(k+1)[i] * KL_(k+1)[i + 2^k]; KL_0 is the root. The sumcheck of level k has k
// variables and sums w(x) L[x] R[x], where L and R are the halves of KL_(k+1) and w(x) = eq(z, x),
// the equality weights of the level's point z. Messages, masks and the hash chain (Fiat-Shamir) are
// the caller's: a round returns three coefficients and takes one challenge. Level k has k rounds
// and k runs from 0 to v - 1 (LkpPieces::level_begin() and level_ends() require both), so a part
// has v (v - 1) / 2 rounds: 630 at 2^36 keys, by this formula, not counted in a run. So the round
// trips cost little compared with the passes described below.
//
// A small part (v <= LkpCut::base) has all levels in memory; its rounds may bind the variables in
// any order.
//
// A large part is cut into pieces. A level multiplies entries that differ in the top bit of the
// next level's index, and a sumcheck binds the top bit first. So a piece is closed under the top G
// bits of the place: piece q has the 2^Lp places (g | q | p), all g < 2^G and p < 2^(Lp - G). Its
// 2^np keys, np = Lp + k_col, are in GPU memory in the order (g | p | column).
//   - An entry of the base level, l0 = v - G, is a product of leaves over all g, within one piece.
//     Its index is (q | p | column). Levels l0 to 0 are stored: 2^(l0 + 1) elements.
//   - Levels l0 + 1 to v are not stored. The sumcheck that reads level l0 + t first binds the
//     t - 1 bits of g in the index of L and R, so these folds stay within a piece. Then L and R
//     have 2^l0 entries each and are stored over the levels, which are not needed any more; the
//     other rounds run in memory.
//   - The sumcheck of the last level, v - 1, instead first binds bits 2, 1, 0, 5, 4, 3, 8, 7, 6 of
//     the column, also within a piece; then L and R are stored. The caller must set this order in
//     LkpPublic::last_key_level_order.
//
// A round's polynomial is a sum over all pieces, and its challenge comes after it. So the rounds
// before L and R are stored run in passes over all pieces; each pass has the keys written again:
//   a matrix pass   serves a few rounds (LkpCut::rounds).
//                   M[a][b] = sum over the rest of the index of weight(rest) * L[a, rest] *
//                   R[b, rest]; a and b run over the values of these rounds' bits. The CPU forms
//                   each round's polynomial from M, then folds M by the challenge.
//   a storing pass  folds L and R by all challenges so far and stores them.

// 2^log_places places of part `part`: local place (g | p), where g has `closed` bits, is the part's
// place (g | number | p).
struct LkpPiece {
  unsigned part = 0, closed = 0, log_places = 0;
  u64 number = 0;
};
class LkpKeySource {
public:
  virtual ~LkpKeySource() = default;
  // Writes the piece's keys to keys[column + 2^k_col * (local place)] in GPU memory, as work on the
  // default stream. A place of padding gets its public row number (LkpPadding::row). Not checked:
  // every call for one piece must write the same keys.
  virtual void lay(const LkpPiece &piece, Key16 *keys) = 0;
};
// How a large part is cut and how much memory LkpPieces may use. No setting changes what is sent.
// With `plain`, lkpp_k_from_top() and lkpp_k_from_leaves() use the field's ordinary arithmetic, and
// the byte digits and lkpp_k_leaf_matrix() are not used. Without it they use the faster forms of
// prover/field/field_lazy.cuh (namespace lazy), which give the same elements; several other kernels
// use these with either setting. lazy::SPrep, lazy::SAcc: a prepared element, and a sum of
// (element) x (small integer). prover/engine/tests/wf_heavy_test.cu runs both settings, in WfGpu,
// against WfCpu (blocks/iop/wf.h), at sizes too small for the byte digits: see the test's head.
struct LkpCut {
  unsigned piece_places = 14;  // Lp: a piece has 2^Lp places, or the whole part's if that is fewer
  unsigned base = 29;          // a part of up to 2^base keys is small
  unsigned room = 26;          // the two arrays of a matrix pass have at most 2^room entries each
  unsigned rounds = 3;         // most rounds of one matrix pass, 1 to 5; the last level: always 3
  bool plain = false;          // described in the comment above
  unsigned digit_columns = 20; // 15 to 24: a digit matrix has <= 2^digit_columns 192-byte columns
};

// The shape of a part's pieces: sets only `closed` (G) and `log_places` (Lp). A small part is one
// piece. G is at most 10, even if the base level then exceeds 2^cut.base entries: L and R of the
// last level have 2^(v - 10) entries each when they are stored, in space for 2^l0 each. Not checked
// here: v >= k_col (the constructor of LkpPieces checks it).
inline LkpPiece lkp_piece_of(unsigned v, unsigned k_col, const LkpCut &cut) {
  LkpPiece p;
  p.log_places = v <= cut.base ? v - k_col : std::min(v - k_col, cut.piece_places);
  p.closed = v <= cut.base ? 0 : std::min(10u, v - cut.base);
  return p;
}

#define LKPP_EACH(i, n) for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < (u64)(n); i += (u64)gridDim.x * blockDim.x)

// The sum of v over a warp, in lane 0.
// __shfl_down_sync: NVIDIA Technical Blog, "Using CUDA Warp-Level Primitives".
__device__ __forceinline__ E lkpp_warp_sum(E v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    E w;
#pragma unroll
    for (int c = 0; c < 3; c++) w.c[c].v = __shfl_down_sync(0xFFFFFFFFu, v.c[c].v, o);
    v += w;
  }
  return v;
}
// The sum of v over a block of a multiple of 32 threads, in thread 0. room: 32 elements of shared
// memory, as raw words. The form is public:
// M. Harris, "Optimizing Parallel Reduction in CUDA";
// J. Luitjens, "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014.
__device__ __forceinline__ E lkpp_block_sum(E v, u64 (*room)[3]) {
  v = lkpp_warp_sum(v);
  __syncthreads();  // an earlier sum's readers are through
  if (!(threadIdx.x & 31))
    for (int c = 0; c < 3; c++) room[threadIdx.x >> 5][c] = v.c[c].v;
  __syncthreads();
  v = E::zero();
  if (threadIdx.x < (blockDim.x >> 5))
    for (int c = 0; c < 3; c++) v.c[c].v = room[threadIdx.x][c];
  return threadIdx.x < 32 ? lkpp_warp_sum(v) : v;
}

// Equality weights: out[x] = the product over i < bits of (z[i] if bit i of x is 1, else 1 - z[i]).
__global__ void lkpp_k_weights(const E *z, unsigned bits, E *out) {
  LKPP_EACH(x, u64(1) << bits) {
    E v = E::one();
    for (unsigned i = 0; i < bits; i++) v *= ((x >> i) & 1) ? z[i] : E::one() - z[i];
    out[x] = v;
  }
}
// The weight of index x is low[x mod 2^bits] * high[x div 2^bits].
struct LkppWeights {
  const E *low, *high;
  unsigned bits;
  __device__ __forceinline__ E at(u64 x) const { return lazy::mul(low[x & ((u64(1) << bits) - 1)], high[x >> bits]); }
};

// ---- in memory ----
__global__ void lkpp_k_leaves(const Key16 *keys, u64 n, E alpha, E *leaf) {
  LKPP_EACH(x, n) leaf[x] = alpha - from_u64(keys[x]);
}
__global__ void lkpp_k_level(const E *child, u64 half, E *parent) {
  LKPP_EACH(x, half) parent[x] = lazy::mul(child[x], child[x + half]);
}
// One round on L and R of 2 * half entries, which binds bit b of their index. With i0 = x with a 0
// inserted at bit b, i1 = i0 + 2^b: each block's share of the three coefficients of
//   sum over x < half of w(x) (L[i0] + X (L[i1] - L[i0])) (R[i0] + X (R[i1] - R[i0])).
__global__ void __launch_bounds__(256) lkpp_k_round(const E *L, const E *R, u64 half, unsigned b, const LkppWeights w, E *part) {
  __shared__ u64 room[32][3];
  lazy::AccE c0 = lazy::zero_e(), c1 = lazy::zero_e(), c2 = lazy::zero_e();  // a thread adds fewer than 2^27 terms to each (MAC_ROOM)
  LKPP_EACH(x, half) {
    const u64 i0 = ((x >> b) << (b + 1)) | (x & ((u64(1) << b) - 1)), i1 = i0 | (u64(1) << b);
    const E l0 = L[i0], r0 = R[i0], dl = L[i1] - l0, dr = R[i1] - r0;
    const lazy::Prepared wx = lazy::prepare(w.at(x)), wl = lazy::prepare(lazy::mulp(wx, l0)), wd = lazy::prepare(lazy::mulp(wx, dl));
    lazy::mac(c0, wl, r0), lazy::mac(c1, wl, dr), lazy::mac(c1, wd, r0), lazy::mac(c2, wd, dr);
  }
  const E a = lkpp_block_sum(lazy::reduce(c0), room), m = lkpp_block_sum(lazy::reduce(c1), room), z = lkpp_block_sum(lazy::reduce(c2), room);
  if (!threadIdx.x) part[3 * blockIdx.x] = a, part[3 * blockIdx.x + 1] = m, part[3 * blockIdx.x + 2] = z;
}
// Binds bit b at the challenge g. Only where b is the top bit may toL be L and toR be R.
__global__ void lkpp_k_bound(const E *L, const E *R, u64 half, unsigned b, const E g, E *toL, E *toR) {
  const lazy::Prepared gp = lazy::prepare(g);
  LKPP_EACH(x, half) {
    const u64 i0 = ((x >> b) << (b + 1)) | (x & ((u64(1) << b) - 1)), i1 = i0 | (u64(1) << b);
    const E l0 = L[i0], l1 = L[i1], r0 = R[i0], r1 = R[i1];
    toL[x] = l0 + lazy::mulp(gp, l1 - l0);
    toR[x] = r0 + lazy::mulp(gp, r1 - r0);
  }
}

// ---- from a piece's keys: the entries of a level ----
/* `lkpp_k_from_top`. The fields of a key's index in its piece, from the top: (g | h | A | e).
     g   `product` bits: an entry is the product of the leaves alpha - key over all g;
     h   one bit if `sides`: 0 for L, 1 for R. Without it there is one array, L;
     A   `folded` bits, which earlier rounds have bound: an entry is the sum over A of fold[A] times
         the product (fold: the equality weights of these rounds' challenges);
     e   `entry` bits, e = (a | l), where a has `group` bits.
   The kernel computes the entries with l in first .. first + count - 1, for all a, and writes entry
   (a, l) to index a * count + (l - first) of L and R. A thread computes four neighbouring entries
   and loads four keys at once. Not checked: `first` and `count` are multiples of 4, entry - group
   is at least 2, and `keys` is 8-byte aligned.
   With DIGITS (group = 3) the entries are not stored: their byte digits go into the two digit
   matrices described below (`LkppDigitRooms`: A left, B right), column l - first of slab a.
   FAST (also in `lkpp_k_from_leaves`): `LkpCut::plain` is false. */
struct LkppDigitRooms {
  const E *low;
  signed char *A, *B;
  u64 ld;
};
struct LkppFromTop {
  unsigned product, sides, folded, entry, group;
};
/* ---- a matrix of three rounds from byte digits ----
   Each of an entry's three coordinates is cut into eight signed bytes, a form that the tensor cores
   take. cuBLAS adds up all products of bytes exactly in 32 bits, and `lkpp_k_join` joins the sums
   modulo the field's prime.
   A digit matrix has LKPP_ROWS = 192 rows, row ((a * 3 + coordinate) * 8 + digit) for a < 8, and
   one column for each l; rows are ld bytes apart. The 24 rows of one a are its slab. The left
   matrix has the digits of low[l mod 2^15] * L[a, l] (`low` as in `LkppWeights`), the right one
   those of R[b, l]. Over a chunk of 2^15 columns a sum of products of two signed bytes is at most
   2^29 in size. */
constexpr unsigned LKPP_ROWS = 192, LKPP_CHUNK_BITS = 15;
// Four neighbouring columns of slab a: the digits of e[0 .. 3]. A row's four bytes are written as
// one 32-bit word, so 4 must divide the address `at` and the row distance ld (not checked).
__device__ __forceinline__ void lkpp_put_digits(signed char *at, u64 ld, unsigned a, const E (&e)[4]) {
#pragma unroll
  for (int c = 0; c < 3; c++) {
    u64 d[4];
#pragma unroll
    for (int i = 0; i < 4; i++) d[i] = lazy::signed_digits(e[i].c[c]);
#pragma unroll
    for (int m = 0; m < 8; m++) {
      const u32 word = (u32)((d[0] >> (8 * m)) & 0xFF) | (u32)((d[1] >> (8 * m)) & 0xFF) << 8 | (u32)((d[2] >> (8 * m)) & 0xFF) << 16 | (u32)((d[3] >> (8 * m)) & 0xFF) << 24;
      *(u32 *)(at + ((u64)(a * 3 + c) * 8 + m) * ld) = word;
    }
  }
}
// (alpha - k1) (alpha - k2) = alpha^2 - (k1 + k2) alpha + k1 k2, for two keys below 2^16: no
// product in the field. square: alpha^2.
__device__ __forceinline__ E lkpp_two_leaves(const E &square, const lazy::SPrep &alpha, u32 k1, u32 k2) {
  lazy::SAcc sum = lazy::szero();
  lazy::smac(sum, alpha, k1 + k2);
  return square - lazy::sred(sum) + from_u64((u64)k1 * k2);
}
template <bool FAST, bool DIGITS>
__global__ void __launch_bounds__(256) lkpp_k_from_top(const Key16 *__restrict__ keys, const E alpha, const LkppFromTop s, const E *__restrict__ fold,
                                                       u64 first, u64 count, E *__restrict__ L, E *__restrict__ R, const LkppDigitRooms rooms) {
  const u64 fours = count / 4;
  lazy::FourKeys four_leaves;
  lazy::SPrep alpha_small;
  E square;
  if (FAST) four_leaves.prepare(alpha), alpha_small = lazy::sprep(alpha), square = lazy::mul(alpha, alpha);
  LKPP_EACH(id, fours << s.group) {
    const u64 a = id / fours, l = first + 4 * (id % fours), e = (a << (s.entry - s.group)) + l;
    for (unsigned h = 0; h <= s.sides; h++) {
      E sum[4];
#pragma unroll
      for (int i = 0; i < 4; i++) sum[i] = E::zero();
      for (u64 A = 0; A < u64(1) << s.folded; A++) {
        E entry[4];
        auto four = [&](u64 g) { return *(const ushort4 *)(keys + (((((g << s.sides) | (u64)h) << s.folded | A) << s.entry) + e)); };
        if (FAST && s.product >= 2) {
          for (u64 g = 0; g < u64(1) << s.product; g += 4) {
            const ushort4 k0 = four(g), k1 = four(g + 1), k2 = four(g + 2), k3 = four(g + 3);
            const E p[4] = {four_leaves.get(k0.x, k1.x, k2.x, k3.x), four_leaves.get(k0.y, k1.y, k2.y, k3.y), four_leaves.get(k0.z, k1.z, k2.z, k3.z),
                            four_leaves.get(k0.w, k1.w, k2.w, k3.w)};
#pragma unroll
            for (int i = 0; i < 4; i++) entry[i] = g ? lazy::mul(entry[i], p[i]) : p[i];
          }
        } else if (FAST && s.product == 1) {
          const ushort4 k0 = four(0), k1 = four(1);
          entry[0] = lkpp_two_leaves(square, alpha_small, k0.x, k1.x), entry[1] = lkpp_two_leaves(square, alpha_small, k0.y, k1.y);
          entry[2] = lkpp_two_leaves(square, alpha_small, k0.z, k1.z), entry[3] = lkpp_two_leaves(square, alpha_small, k0.w, k1.w);
        } else {
          for (u64 g = 0; g < u64(1) << s.product; g++) {
            const ushort4 k = four(g);
            const u32 key[4] = {k.x, k.y, k.z, k.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
              const E leaf = alpha - from_u64(key[i]);
              entry[i] = g ? entry[i] * leaf : leaf;
            }
          }
        }
#pragma unroll
        for (int i = 0; i < 4; i++) sum[i] = !s.folded ? entry[i] : sum[i] + (FAST ? lazy::mul(fold[A], entry[i]) : fold[A] * entry[i]);
      }
      if (DIGITS) {
        if (!h)
#pragma unroll
          for (int i = 0; i < 4; i++) sum[i] = lazy::mul(rooms.low[(l + i) & ((u64(1) << LKPP_CHUNK_BITS) - 1)], sum[i]);
        lkpp_put_digits((h ? rooms.B : rooms.A) + (l - first), rooms.ld, (unsigned)a, sum);
        continue;
      }
      E *to = (h ? R : L) + a * count + (l - first);
#pragma unroll
      for (int i = 0; i < 4; i++) to[i] = sum[i];
    }
  }
}
/* `lkpp_k_from_leaves`: the last level, whose L and R are leaves. A key's index in its piece is
   (h | l | a | A), with the letters of `lkpp_k_from_top`. An entry is alpha - sum over A of
   fold[A] * key, because the fold's weights add up to 1. Entry (a, l) goes where `lkpp_k_from_top`
   puts it or, where `wide` is set (group = 0), to its index in the whole part's array: the bits of
   l from `low` on move up by `gap` bits, and `piece` fills the gap. */
struct LkppFromLeaves {
  unsigned folded, group, low, gap;
  u64 piece;
  bool wide;
};
template <bool FAST>
__global__ void __launch_bounds__(256) lkpp_k_from_leaves(const Key16 *__restrict__ keys, u64 half, const E alpha, const LkppFromLeaves s,
                                                          const E *__restrict__ fold, u64 first, u64 count, E *__restrict__ L, E *__restrict__ R) {
  extern __shared__ u64 lkpp_shared[];  // FAST: the fold's weights, prepared: 2^folded of them
  lazy::SPrep *weight = (lazy::SPrep *)lkpp_shared;
  if (FAST) {
    for (u64 A = threadIdx.x; s.folded && A < u64(1) << s.folded; A += blockDim.x) weight[A] = lazy::sprep(fold[A]);
    __syncthreads();
  }
  LKPP_EACH(x, count) {
    const u64 l = first + x;
    const u64 place = s.wide ? ((((l >> s.low) << s.gap) | s.piece) << s.low) | (l & ((u64(1) << s.low) - 1)) : x;
    for (int h = 0; h < 2; h++) {
      const Key16 *run = keys + (h ? half : 0) + (l << (s.group + s.folded));
      E *to = h ? R : L;
      for (u64 a = 0; a < u64(1) << s.group; a++) {
        if (!s.folded) {
          to[a * count + place] = alpha - from_u64(run[a]);
          continue;
        }
        if (FAST) {  // at most 2^9 keys below 2^16; SPrep allows (terms) x (greatest key) <= 2^32
          lazy::SAcc sum = lazy::szero();
          for (u64 A = 0; A < u64(1) << s.folded; A++) lazy::smac(sum, weight[A], run[(a << s.folded) + A]);
          to[a * count + place] = alpha - lazy::sred(sum);
          continue;
        }
        E sum = E::zero();
        for (u64 A = 0; A < u64(1) << s.folded; A++) sum += fold[A] * from_u64(run[(a << s.folded) + A]);
        to[a * count + place] = alpha - sum;
      }
    }
  }
}
// The matrix of a pass: block x's share of
//   M[a][b] = sum over l < count of w(first + l) L[a * count + l] R[b * count + l]
// goes to part[(x * J + a) * J + b]; the matrix has J rows. A thread holds a tile of S x S sums;
// blockIdx.y and .z number the tiles.
template <int S>
__global__ void __launch_bounds__(128) lkpp_k_matrix(const E *__restrict__ L, const E *__restrict__ R, u64 count, u64 first, const LkppWeights w, int J,
                                                     E *__restrict__ part) {
  __shared__ u64 room[32][3];
  const int a0 = blockIdx.y * S, b0 = blockIdx.z * S;
  E sum[S][S];
#pragma unroll
  for (int a = 0; a < S; a++)
#pragma unroll
    for (int b = 0; b < S; b++) sum[a][b] = E::zero();
  LKPP_EACH(l, count) {
    const lazy::Prepared wl = lazy::prepare(w.at(first + l));
    lazy::Prepared left[S];
#pragma unroll
    for (int a = 0; a < S; a++) left[a] = lazy::prepare(lazy::mulp(wl, L[(a0 + a) * count + l]));
#pragma unroll
    for (int b = 0; b < S; b++) {
      const E right = R[(b0 + b) * count + l];
#pragma unroll
      for (int a = 0; a < S; a++) sum[a][b] += lazy::mulp(left[a], right);
    }
  }
#pragma unroll
  for (int a = 0; a < S; a++)
#pragma unroll
    for (int b = 0; b < S; b++) {
      const E all = lkpp_block_sum(sum[a][b], room);
      if (!threadIdx.x) part[((u64)blockIdx.x * J + a0 + a) * J + b0 + b] = all;
    }
}
/* `lkpp_k_leaf_matrix`: the first matrix pass of the last level, from the keys as integers, without
   a product in the field for each key. An entry is alpha - key, so
       M[a][b] = alpha^2 N[8][8] - alpha (N[a][8] + N[8][b]) + N[a][b],
       N[a][b] = sum over l of w(l) x_a(l) y_b(l),
   where x_a(l), a < 8, is key (0 | l | a), y_b(l) is key (1 | l | b), x_8 = y_8 = 1, and w is as in
   `LkppWeights`. Not checked: the keys are below 2^13 (`leaves_matrix_pass` uses this kernel only
   if d <= 13). A product of two is then below 2^26, so 64 terms are within SPrep's capacity,
   (terms) x (greatest integer) <= 2^32. A block has one of the nine tiles of 3 x 3 sums, tile
   blockIdx.x mod 9: the nine blocks that read the same keys have consecutive numbers, so they are
   likely to run at the same time (CUDA does not promise it). The kernel writes N, not M: block x's
   share of N[a][b] goes to part[81 (x div 9) + 9 a + b], and `leaves_matrix_pass` forms M on the
   CPU. Also not checked: the grid is a multiple of 9 blocks, `keys` is 16-byte aligned and `half`
   is a multiple of 8. */
__device__ __forceinline__ u32 lkpp_key_or_one(const uint4 &v, int i) {
  const u32 word = i < 4 ? (i < 2 ? v.x : v.y) : (i < 6 ? v.z : v.w);
  return i == 8 ? 1u : (word >> (16 * (i & 1))) & 0xFFFF;
}
__global__ void __launch_bounds__(128) lkpp_k_leaf_matrix(const Key16 *__restrict__ keys, u64 half, const E *__restrict__ low, unsigned bits, const E *__restrict__ high,
                                                          u64 highs, E *__restrict__ part) {
  __shared__ u64 room[32][3];
  const int a0 = 3 * ((blockIdx.x % 9) / 3), b0 = 3 * (blockIdx.x % 3);
  const uint4 *left = (const uint4 *)keys, *right = (const uint4 *)(keys + half);
  E sum[3][3];
#pragma unroll
  for (int a = 0; a < 3; a++)
#pragma unroll
    for (int b = 0; b < 3; b++) sum[a][b] = E::zero();
  for (u64 hi = blockIdx.x / 9; hi < highs; hi += gridDim.x / 9) {
    E under[3][3];
    lazy::SAcc acc[3][3];
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
      for (int b = 0; b < 3; b++) under[a][b] = E::zero(), acc[a][b] = lazy::szero();
    int terms = 0;
    for (u64 lo = threadIdx.x; lo < u64(1) << bits; lo += blockDim.x) {
      const uint4 x = left[(hi << bits) + lo], y = right[(hi << bits) + lo];
      const lazy::SPrep w = lazy::sprep(low[lo]);
      u32 ka[3], kb[3];
#pragma unroll
      for (int i = 0; i < 3; i++) ka[i] = lkpp_key_or_one(x, a0 + i), kb[i] = lkpp_key_or_one(y, b0 + i);
#pragma unroll
      for (int a = 0; a < 3; a++)
#pragma unroll
        for (int b = 0; b < 3; b++) lazy::smac(acc[a][b], w, ka[a] * kb[b]);
      if (++terms == 64) {
        terms = 0;
#pragma unroll
        for (int a = 0; a < 3; a++)
#pragma unroll
          for (int b = 0; b < 3; b++) under[a][b] += lazy::sred(acc[a][b]), acc[a][b] = lazy::szero();
      }
    }
    const lazy::Prepared top = lazy::prepare(high[hi]);
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
      for (int b = 0; b < 3; b++) sum[a][b] += lazy::mulp(top, under[a][b] + lazy::sred(acc[a][b]));
  }
#pragma unroll
  for (int a = 0; a < 3; a++)
#pragma unroll
    for (int b = 0; b < 3; b++) {
      const E all = lkpp_block_sum(sum[a][b], room);
      if (!threadIdx.x) part[81 * (u64)(blockIdx.x / 9) + 9 * (a0 + a) + b0 + b] = all;
    }
}
// The byte digits of the entries in L and R: columns from .. from + columns - 1 go to columns
// 0 .. columns - 1 of A and B. Not checked: `columns` is a multiple of 4.
__global__ void __launch_bounds__(256) lkpp_k_digits(const E *__restrict__ L, const E *__restrict__ R, u64 count, u64 from, u64 columns, const E *__restrict__ low,
                                                     signed char *__restrict__ A, signed char *__restrict__ B, u64 ld) {
  LKPP_EACH(id, 2 * columns) {  // 8 slabs x columns / 4
    const unsigned a = (unsigned)(id / (columns / 4));
    const u64 l = 4 * (id % (columns / 4));
    E left[4], right[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      left[i] = lazy::mul(low[(from + l + i) & ((u64(1) << LKPP_CHUNK_BITS) - 1)], L[a * count + from + l + i]);
      right[i] = R[a * count + from + l + i];
    }
    lkpp_put_digits(A + l, ld, a, left), lkpp_put_digits(B + l, ld, a, right);
  }
}
// C: for each chunk the 192 x 192 sums, C[row of B * 192 + row of A].
// part[64 * chunk + 8 * a + b] = high[chunk] * sum over the chunk's l of (low * L)[a, l] * R[b, l].
// Launch: one block per chunk, 64 threads.
__global__ void __launch_bounds__(64) lkpp_k_join(const int *__restrict__ C, const E *__restrict__ high, E *__restrict__ part) {
  const int *sums = C + (u64)blockIdx.x * LKPP_ROWS * LKPP_ROWS;
  const unsigned a = threadIdx.x >> 3, b = threadIdx.x & 7;
  F xy[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) {
      i64 by_weight[15];  // [t]: the sum over the digit pairs m + n = t; at most 8 x 2^29 in size
#pragma unroll
      for (int t = 0; t < 15; t++) by_weight[t] = 0;
#pragma unroll
      for (int m = 0; m < 8; m++)
#pragma unroll
        for (int n = 0; n < 8; n++) by_weight[m + n] += sums[(u64)((b * 3 + j) * 8 + n) * LKPP_ROWS + (a * 3 + i) * 8 + m];
      F v = F::zero();
#pragma unroll
      for (int t = 0; t < 15; t++) v += lazy::join_digit(by_weight[t], 8 * t);
      xy[i][j] = v;
    }
  // (x0 + x1 X + x2 X^2) (y0 + y1 X + y2 X^2) with X^3 = NR
  const E product = E::from(xy[0][0] + lazy::times_nr(xy[1][2] + xy[2][1]), xy[0][1] + xy[1][0] + lazy::times_nr(xy[2][2]), xy[0][2] + xy[1][1] + xy[2][0]);
  part[64 * (u64)blockIdx.x + threadIdx.x] = lazy::mul(high[blockIdx.x], product);
}
// M[i] += factor * (the sum over the blocks b of part[b * entries + i]).
__global__ void lkpp_k_matrix_added(const E *part, unsigned blocks, unsigned entries, const E factor, E *M) {
  LKPP_EACH(i, entries) {
    E sum = E::zero();
    for (unsigned b = 0; b < blocks; b++) sum += part[(u64)b * entries + i];
    M[i] += factor * sum;
  }
}

// The GPU memory (`Card` in identifiers: prover/engine/device.cuh) that an LkpPieces allocates: the
// entries of each buffer, and bytes() in all. The constructor allocates by lkp_card_needs(), so
// code that shares the GPU can call it beforehand. Not counted: what cuBLAS allocates when
// blas::blas() creates a thread's handle.
struct LkpCardNeeds {
  static constexpr unsigned LOW = 15, HIGH = 20, BLOCKS = 1024, MATRIX_BLOCKS = 264;
  u64 store = 0, keys = 0, room = 0, digit_columns = 0;
  u64 small() const { return (u64(1) << LOW) + (u64(1) << HIGH) + 64 + 1024 + 1024 + std::max<u64>(3 * BLOCKS, u64(MATRIX_BLOCKS) << 10); }  // weights, point, fold, matrix, shares
  u64 digit_sums() const { return (digit_columns >> LKPP_CHUNK_BITS) * LKPP_ROWS * LKPP_ROWS; }
  u64 bytes() const {
    return (store + 2 * room + small()) * sizeof(E) + keys * sizeof(Key16) + (digit_columns ? 2 * LKPP_ROWS * digit_columns + digit_sums() * sizeof(int) + blas::BLAS_WORKSPACE : 0);
  }
};
inline LkpCardNeeds lkp_card_needs(const std::vector<unsigned> &part_bits, unsigned k_col, const LkpCut &cut) {
  LkpCardNeeds n;
  for (unsigned v : part_bits) {
    const LkpPiece p = lkp_piece_of(v, k_col, cut);
    const unsigned np = p.log_places + k_col;
    n.store = std::max(n.store, u64(2) << (v - p.closed)), n.keys = std::max(n.keys, u64(1) << np);
    if (v > cut.base) n.room = std::max(n.room, u64(1) << std::min(cut.room, np));
  }
  if (n.room && !cut.plain && (n.room >> 3) >= u64(1) << LKPP_CHUNK_BITS) n.digit_columns = std::min<u64>(n.room >> 3, u64(1) << cut.digit_columns);
  return n;
}

__global__ void lkpp_k_any(const u64 *words, u64 n, unsigned *any) {
  LKPP_EACH(i, n) if (words[i]) atomicOr(any, 1u);
}

// The key side runs on the GPU: kernels and copies on the default stream, cuBLAS on
// cudaStreamPerThread (blas::blas()). No event orders the two: under nvcc's --default-stream
// per-thread (prover/build.sh) they are one stream, and in a build without that option CUDA itself
// runs the work of the two in the order of the calls. The table side (2^(mu + d) entries, at most
// 2^22) runs on the CPU, as in LkpCpu (SumcheckCpu, blocks/engine/sumcheck_cpu.h).
class LkpPieces : public LkpProver {
public:
  // `bits`: the multiplicities, as LkpCpu takes them: bits[j + 2^mu t] is bit j of the multiplicity
  // of row t, and 0 for j >= fixed.planes(). `source` must outlive this object.
  LkpPieces(const LkpPublic &fixed, unsigned k_col, LkpKeySource &source, std::vector<E> bits, const LkpCut &cut = LkpCut())
      : fixed_(fixed), k_col_(k_col), source_(source), bits_(std::move(bits)), shape_(cut) {
    BLK_REQUIRE(!fixed.shifted && fixed.g == 0 && fixed.h == 0 && fixed.d <= 16 && fixed.tables.size() == 1 && fixed.tables[0].rows.empty(),
                "look-up in pieces: one plain range table of at most 2^16 rows, no shift, no trees' numbers");
    BLK_REQUIRE(bits_.size() == size_t(1) << (fixed.mu + fixed.d), "look-up in pieces: the multiplicities have not the announced number of bits");
    BLK_REQUIRE(cut.rounds >= 1 && cut.rounds <= 5 && cut.room >= 10 && cut.room <= 34 && cut.piece_places >= 1 && cut.base >= 1 && cut.base <= 34 && cut.digit_columns >= LKPP_CHUNK_BITS && cut.digit_columns <= 24,
                "look-up in pieces: the cut is out of range");
    for (unsigned v : fixed.part_bits()) cuts_.push_back(cut_of(v));
    const LkpCardNeeds n = lkp_card_needs(fixed.part_bits(), k_col, cut);
    store_.resize(n.store), keys_.resize(n.keys), roomL_.resize(n.room), roomR_.resize(n.room);
    low_.resize(size_t(1) << LOW), high_.resize(size_t(1) << HIGH), point_.resize(64), fold_.resize(1024), matrix_.resize(1024);
    part_.resize(std::max<size_t>(3 * BLOCKS, size_t(MATRIX_BLOCKS) << 10));
    // wipe() does not zero these four (see held()); each call states why.
    left_unzeroed(point_, "coordinates of a level's point: of the claim's equality weights, which the verifier holds too");
    left_unzeroed(low_, "the equality weights of that point's low coordinates: a function of the verifier's challenges alone");
    left_unzeroed(high_, "the equality weights of that point's high coordinates: a function of the verifier's challenges alone");
    left_unzeroed(fold_, "the equality weights of the challenges that the level's rounds have bound: a function of the verifier's challenges alone");
    if ((digit_columns_ = n.digit_columns)) {
      digitsA_.resize(LKPP_ROWS * digit_columns_), digitsB_.resize(LKPP_ROWS * digit_columns_);
      digit_sums_.resize(n.digit_sums()), blas_work_.resize(blas::BLAS_WORKSPACE);
      blas::blas(nullptr, nullptr, 0);  // the thread's first call may allocate
    }
    u64 taken = keys_.size() * sizeof(Key16) + digitsA_.size() + digitsB_.size() + digit_sums_.size() * sizeof(int) + blas_work_.size();
    for (const Card<E> *c : {&store_, &roomL_, &roomR_, &low_, &high_, &point_, &fold_, &matrix_, &part_}) taken += c->size() * sizeof(E);
    BLK_REQUIRE(taken == n.bytes(), "look-up in pieces: the engine takes another number of bytes than lkp_card_needs() says");
  }
  ~LkpPieces() override { wipe(); }
  LkpPieces(const LkpPieces &) = delete;
  LkpPieces &operator=(const LkpPieces &) = delete;

  // Optional, before begin(). count[part][t]: how often row t occurs among all keys of the part,
  // those at places of padding too. The caller asks for all roots (key_roots()) before any part's
  // sumchecks. With the counts, begin() computes a root as the product over t of
  // (alpha - t)^count[part][t], and the process ends if the part's tree later has another root.
  // Without them, begin() builds each part's stored levels for the root only, last part first, and
  // all but the first part's are built a second time.
  void counts_by_part(std::vector<std::vector<u64>> count) {
    BLK_REQUIRE(count.size() == cuts_.size(), "look-up in pieces: not one list of counts for each part");
    for (size_t part = 0; part < count.size(); part++) {
      u64 all = 0;
      BLK_REQUIRE(count[part].size() == size_t(1) << fixed_.d, "look-up in pieces: not one count for each row of the table");
      for (u64 c : count[part]) all += c;
      BLK_REQUIRE(all == u64(1) << cuts_[part].v, "look-up in pieces: a part's counts do not add up to its keys");
    }
    counts_ = std::move(count);
  }
  // `routine` is called from end_values() of a part's last level: it can have the part's keys
  // written once more while the key source still has what it keeps for the part. by_variable: that
  // level's v - 1 challenges. `room`: this object's buffer for a piece's keys, free to use.
  void at_part_end(std::function<void(unsigned part, const std::vector<E> &by_variable, Key16 *room)> routine) { at_part_end_ = std::move(routine); }

  void begin(const E &theta, const E &alpha) override {
    BLK_REQUIRE(theta == E::zero(), "look-up in pieces: no shift");
    alpha_ = alpha;
    roots_.assign(cuts_.size(), E::zero());
    for (size_t part = cuts_.size(); part-- > 0;) {
      if (!counts_.empty()) {
        E root = E::one();
        for (u64 t = 0; t < counts_[part].size(); t++)
          if (counts_[part][t]) root *= pow(alpha - from_u64(t), counts_[part][t]);
        roots_[part] = root;
        continue;
      }
      open(part);
      to_host(&roots_[part], store_.get() + 1, 1);
    }
    const std::vector<E> table = lkp_rows(fixed_, theta);
    const size_t w = size_t(1) << fixed_.mu;
    std::vector<E> leaves(bits_.size());
    for (size_t i = 0; i < bits_.size(); i++) leaves[i] = E::one() + bits_[i] * (alpha - E::one() - table[i / w]);
    table_levels_.clear();
    table_levels_.push_back(std::move(leaves));
    while (table_levels_.back().size() > w) {
      const std::vector<E> &below = table_levels_.back();
      std::vector<E> level(below.size() / 2);
      for (size_t i = 0; i < level.size(); i++) level[i] = below[i] * below[i + level.size()];
      table_levels_.push_back(std::move(level));
    }
    std::reverse(table_levels_.begin(), table_levels_.end());
  }
  std::vector<E> key_roots() override { return roots_; }
  std::vector<E> plane_roots() override { return table_levels_[0]; }
  std::unique_ptr<SumcheckProver<E>> sumcheck(LkpStep step, unsigned part, unsigned level, const Summand<E> &summand, const std::vector<Weight<E>> &vars,
                                              const std::vector<unsigned> &order) override {
    if (step == LkpStep::KEYS) {
      level_begin(part, level, vars, order);
      return std::unique_ptr<SumcheckProver<E>>(new Level(*this));
    }
    std::vector<std::vector<E>> maps;
    if (step == LkpStep::TABLE) {
      const std::vector<E> &below = table_levels_[level + 1];
      const size_t half = below.size() / 2;
      maps.push_back(lkp_in_order(std::vector<E>(below.begin(), below.begin() + half), order));
      maps.push_back(lkp_in_order(std::vector<E>(below.begin() + half, below.end()), order));
    } else {
      maps.push_back(lkp_in_order(bits_, order));
    }
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(summand, vars, std::move(maps)));
  }

  // Zeroes the GPU buffers of held() and the CPU members that have held a hidden value. (Temporary
  // CPU vectors of sums and matrices are zeroed where they are dropped.) Never zeroed: the copies
  // of the table side's levels and of the multiplicities that sumcheck() makes for SumcheckCpu, and
  // single elements in local variables, such as a root in begin() and open(). Nor does this file
  // zero what its kernels leave in registers, shared memory or local memory. Do not use the object
  // afterwards.
  void wipe() {
    ENGINE_CK(cudaDeviceSynchronize());
    for (const Buffer &h : held())
      if (h.bytes) ENGINE_CK(cudaMemset(h.at, 0, h.bytes));
    if (digitsA_.size()) blas::blas(nullptr, nullptr, 0);  // unsets the cuBLAS handle's work space
    ENGINE_CK(cudaDeviceSynchronize());
    erase(roots_), erase(mat_.M), erase(bits_), erase(coins_), erase(by_position_);
    for (std::vector<u64> &c : counts_) {
      volatile u64 *p = c.data();
      for (size_t i = 0; i < c.size(); i++) p[i] = 0;
    }
    for (std::vector<E> &l : table_levels_) erase(l);
  }
  // For a test: whether any GPU buffer of held() has a word that is not 0. It uses wipe()'s list,
  // so it cannot see a buffer that is missing from it. LeftBehind
  // (prover/engine/tests/left_behind.cuh) needs no list: it reads the memory released to its own
  // arena. `any` is zeroed too before it is released.
  bool anything_left() {
    Card<unsigned> any(1);
    ENGINE_CK(cudaMemset(any.get(), 0, sizeof(unsigned)));
    for (const Buffer &h : held())
      if (h.bytes) lkpp_k_any<<<1024, 256>>>((const u64 *)h.at, h.bytes / 8, any.get());
    unsigned word = 1;
    ENGINE_CK(cudaMemcpy(&word, any.get(), sizeof word, cudaMemcpyDeviceToHost));
    ENGINE_CK(cudaMemset(any.get(), 0, sizeof(unsigned)));  // one bit about hidden buffers
    return word != 0;
  }

  // For a report: passes over a part's pieces, and seconds by kind of work.
  size_t passes = 0, digit_products = 0, integer_passes = 0;  // the last two: batched products of byte matrices; matrix passes from the keys' integers
  double s_base = 0, s_matrix = 0, s_store = 0, s_memory = 0;

private:
  static constexpr unsigned LOW = LkpCardNeeds::LOW, HIGH = LkpCardNeeds::HIGH, BLOCKS = LkpCardNeeds::BLOCKS, MATRIX_BLOCKS = LkpCardNeeds::MATRIX_BLOCKS;
  static_assert(LOW == LKPP_CHUNK_BITS, "a chunk of the digit matrices has one high weight, and the low weights go by the place in it");
  static constexpr u64 DIGIT_COLUMNS_LEAST = u64(1) << LKPP_CHUNK_BITS;
  // v, Lp, G, np, l0: see the comment at LkpPiece. The part has 2^lwr places and 2^Qb pieces; a
  // piece gives 2^s0 entries of the base level. whole: a small part (G = 0, l0 = v).
  struct Cut {
    unsigned v = 0, lwr = 0, Lp = 0, G = 0, np = 0, Qb = 0, s0 = 0, l0 = 0;
    bool whole = true;
    u64 pieces() const { return u64(1) << Qb; }
  };
  Cut cut_of(unsigned v) const {
    BLK_REQUIRE(v >= k_col_ && v <= 40, "look-up in pieces: a part has fewer keys than a place, or more than 2^40");
    Cut c;
    c.v = v, c.lwr = v - k_col_;
    if (v <= shape_.base) return c.Lp = c.lwr, c.np = c.s0 = c.l0 = v, c;
    c.whole = false;
    c.Lp = lkp_piece_of(v, k_col_, shape_).log_places, c.G = lkp_piece_of(v, k_col_, shape_).closed;
    c.np = c.Lp + k_col_, c.Qb = c.lwr - c.Lp, c.s0 = c.np - c.G, c.l0 = v - c.G;
    BLK_REQUIRE(k_col_ >= 9, "look-up in pieces: fewer than 2^9 columns: the last level's first nine rounds would leave a piece");
    BLK_REQUIRE(c.G <= c.Lp, "look-up in pieces: the pieces are too small: a piece must hold all values of the top bits");
    BLK_REQUIRE(c.np >= c.G + 11, "look-up in pieces: a piece's share of the base is too small");
    return c;
  }

  // The state is in the LkpPieces: a Level must not be used once sumcheck() has begun another level
  // (not checked).
  class Level : public SumcheckProver<E> {
  public:
    explicit Level(LkpPieces &of) : of_(of) {}
    std::vector<E> round() override { return of_.level_round(); }
    void bind(const E &g) override { of_.level_bind(g); }
    std::vector<E> end_values() override { return of_.level_ends(); }

  private:
    LkpPieces &of_;
  };

  // ---- small tools ----
  typedef std::chrono::steady_clock::time_point Time;
  static Time now() { return std::chrono::steady_clock::now(); }
  static double since(Time then) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count(); }
  static unsigned grid(u64 n, unsigned threads = 256, u64 most = 1 << 16) { return (unsigned)std::max<u64>(1, std::min(most, (n + threads - 1) / threads)); }
  static void to_host(E *to, const E *from, size_t count) { ENGINE_CK(cudaMemcpy(to, from, count * sizeof(E), cudaMemcpyDeviceToHost)); }
  static void erase(std::vector<E> &v) {
    volatile u64 *p = (volatile u64 *)v.data();
    for (size_t i = 0; i < v.size() * (sizeof(E) / 8); i++) p[i] = 0;
  }
  // Equality weights on the CPU: eq_low_first() as lkpp_k_weights(); eq_top_first() with z[t] for
  // bit n - 1 - t of the index.
  static std::vector<E> eq_top_first(const E *z, size_t n) {
    std::vector<E> t(1, E::one());
    for (size_t v = 0; v < n; v++) {
      std::vector<E> next(2 * t.size());
      for (size_t x = 0; x < t.size(); x++) next[2 * x + 1] = t[x] * z[v], next[2 * x] = t[x] - next[2 * x + 1];
      t.swap(next);
    }
    return t;
  }
  static std::vector<E> eq_low_first(const std::vector<E> &z) {
    std::vector<E> t(1, E::one());
    for (size_t v = 0; v < z.size(); v++) {
      const size_t s = t.size();
      t.resize(2 * s);
      for (size_t x = 0; x < s; x++) t[s + x] = t[x] * z[v], t[x] = t[x] - t[s + x];
    }
    return t;
  }
  // The equality weights of z, z[i] for bit i of the index, in `low_` and `high_`: the next call
  // overwrites both. Card::spent(): the old contents will not be read again; a no-op outside tests.
  LkppWeights weights(const std::vector<E> &z) {
    const unsigned bits = (unsigned)z.size(), low = std::min(bits, LOW);
    BLK_REQUIRE(bits <= LOW + HIGH && bits <= 64, "look-up in pieces: an index has more bits than the tables of weights hold");
    point_.spent(), low_.spent(), high_.spent();
    if (bits) ENGINE_CK(cudaMemcpy(point_.get(), z.data(), z.size() * sizeof(E), cudaMemcpyHostToDevice));
    lkpp_k_weights<<<grid(u64(1) << low), 256>>>(point_.get(), low, low_.get());
    lkpp_k_weights<<<grid(u64(1) << (bits - low)), 256>>>(point_.get() + low, bits - low, high_.get());
    ENGINE_CK(cudaGetLastError());
    return {low_.get(), high_.get(), low};
  }
  void fold_table(const std::vector<E> &table) {
    BLK_REQUIRE(table.size() <= 1024, "look-up in pieces: more than ten bits are bound before a level is stored");
    fold_.spent();
    ENGINE_CK(cudaMemcpy(fold_.get(), table.data(), table.size() * sizeof(E), cudaMemcpyHostToDevice));
  }
  void from_top(unsigned blocks, const Key16 *keys, const E &alpha, const LkppFromTop &s, const E *fold, u64 first, u64 count, E *L, E *R) {
    if (shape_.plain)
      lkpp_k_from_top<false, false><<<blocks, 256>>>(keys, alpha, s, fold, first, count, L, R, LkppDigitRooms());
    else
      lkpp_k_from_top<true, false><<<blocks, 256>>>(keys, alpha, s, fold, first, count, L, R, LkppDigitRooms());
  }
  void from_leaves(unsigned blocks, const Key16 *keys, u64 half, const E &alpha, const LkppFromLeaves &s, const E *fold, u64 first, u64 count, E *L, E *R) {
    if (shape_.plain)
      lkpp_k_from_leaves<false><<<blocks, 256>>>(keys, half, alpha, s, fold, first, count, L, R);
    else
      lkpp_k_from_leaves<true><<<blocks, 256, sizeof(lazy::SPrep) << s.folded>>>(keys, half, alpha, s, fold, first, count, L, R);
  }
  void lay(u64 q) {
    keys_.spent();
    source_.lay(LkpPiece{(unsigned)open_, c_.G, c_.Lp, q}, keys_.get());
  }
  E at(unsigned variable) const { return z_[variable]; }
  static E piece_weight(const std::vector<E> &z, u64 q) {
    E w = E::one();
    for (size_t i = 0; i < z.size(); i++) w *= ((q >> i) & 1) ? z[i] : E::one() - z[i];
    return w;
  }
  // held(): the GPU buffers that hold hidden values. `point_`, `low_`, `high_` and `fold_` contain
  // functions of the verifier's messages only (weights(), fold_table()) and are not listed. Every
  // size is a multiple of 8 bytes (that of `keys_` from four keys on), which anything_left() relies
  // on.
  struct Buffer {
    void *at;
    size_t bytes;
  };
  std::vector<Buffer> held() {
    return {{store_.get(), store_.size() * sizeof(E)},     {roomL_.get(), roomL_.size() * sizeof(E)},   {roomR_.get(), roomR_.size() * sizeof(E)},
            {part_.get(), part_.size() * sizeof(E)},       {matrix_.get(), matrix_.size() * sizeof(E)}, {keys_.get(), keys_.size() * sizeof(Key16)},
            {digitsA_.get(), digitsA_.size()},             {digitsB_.get(), digitsB_.size()},           {digit_sums_.get(), digit_sums_.size() * sizeof(int)},
            {blas_work_.get(), blas_work_.size()}};
  }

  // ---- a part's stored levels: level l at store_ + 2^l ----
  void open(size_t part) {
    if (open_ == part && fresh_) return;
    const Time then = now();
    open_ = part, c_ = cuts_[part], fresh_ = true;
    store_.spent();
    E *store = store_.get();
    if (c_.whole) {
      lay(0);
      lkpp_k_leaves<<<grid(u64(1) << c_.v), 256>>>(keys_.get(), u64(1) << c_.v, alpha_, store + (u64(1) << c_.v));
    } else {
      const LkppFromTop s{c_.G, 0, 0, c_.s0, 0};
      const u64 share = u64(1) << c_.s0;
      for (u64 q = 0; q < c_.pieces(); q++) {
        lay(q);
        from_top(grid(share / 4), keys_.get(), alpha_, s, nullptr, 0, share, store + (u64(1) << c_.l0) + q * share, nullptr);
        ENGINE_CK(cudaGetLastError());
      }
    }
    passes++;
    for (unsigned l = c_.l0; l > 0; l--) lkpp_k_level<<<grid(u64(1) << (l - 1)), 256>>>(store + (u64(1) << l), u64(1) << (l - 1), store + (u64(1) << (l - 1)));
    ENGINE_CK(cudaGetLastError());
    ENGINE_CK(cudaDeviceSynchronize());
    if (!counts_.empty()) {
      E root;
      to_host(&root, store + 1, 1);
      BLK_REQUIRE(root == roots_[part], "look-up in pieces: the root of a part's tree is not the one that the part's counts give");
    }
    s_base += since(then);
  }

  // ---- the rounds of a level whose two arrays are in GPU memory ----
  struct ResidentLevel {
    E *L = nullptr, *R = nullptr, *otherL = nullptr, *otherR = nullptr;  // other: where a fold that does not bind the top bit writes
    std::vector<unsigned> variable;                                      // of each bit of the arrays' index
  };
  ResidentLevel mem_;
  unsigned bit_of(unsigned variable) const {
    for (unsigned b = 0; b < mem_.variable.size(); b++)
      if (mem_.variable[b] == variable) return b;
    BLK_REQUIRE(false, "look-up in pieces: a round binds a variable that is bound already, or that the level has not");
    return 0;
  }
  std::vector<E> memory_round(unsigned variable) {
    const unsigned bits = (unsigned)mem_.variable.size(), b = bit_of(variable);
    const u64 half = u64(1) << (bits - 1);
    std::vector<E> z;
    for (unsigned i = 0; i < bits; i++)
      if (i != b) z.push_back(at(mem_.variable[i]));
    const LkppWeights w = weights(z);
    const unsigned blocks = grid(half, 256, BLOCKS);
    part_.spent();
    lkpp_k_round<<<blocks, 256>>>(mem_.L, mem_.R, half, b, w, part_.get());
    ENGINE_CK(cudaGetLastError());
    std::vector<E> part(3 * size_t(blocks)), c(3, E::zero());
    to_host(part.data(), part_.get(), part.size());
    for (size_t i = 0; i < part.size(); i++) c[i % 3] += part[i];
    erase(part);
    return c;
  }
  void memory_bind(unsigned variable, const E &g) {
    const unsigned bits = (unsigned)mem_.variable.size(), b = bit_of(variable);
    const u64 half = u64(1) << (bits - 1);
    if (b + 1 == bits) {
      lkpp_k_bound<<<grid(half), 256>>>(mem_.L, mem_.R, half, b, g, mem_.L, mem_.R);
    } else {
      BLK_REQUIRE(mem_.otherL, "look-up in pieces: a large part's rounds in memory bind the top bit");
      lkpp_k_bound<<<grid(half), 256>>>(mem_.L, mem_.R, half, b, g, mem_.otherL, mem_.otherR);
      std::swap(mem_.L, mem_.otherL), std::swap(mem_.R, mem_.otherR);
    }
    ENGINE_CK(cudaGetLastError());
    mem_.variable.erase(mem_.variable.begin() + b);
  }

  // ---- the rounds that a matrix serves ----
  // M[a * J + b], J = 2^(bits - done): the matrix of a matrix pass, on the CPU, when `done` of its
  // `bits` rounds are bound. The top bit of a and of b is the next round's. z: the level's point
  // (`z_`) at these rounds' variables, in round order.
  struct Matrix {
    std::vector<E> M, z;
    unsigned bits = 0, done = 0;
  };
  Matrix mat_;
  std::vector<E> matrix_round() const {
    const unsigned rest = mat_.bits - mat_.done - 1;
    const size_t J = size_t(2) << rest, H = J / 2;
    const std::vector<E> w = eq_top_first(mat_.z.data() + mat_.done + 1, rest);
    std::vector<E> c(3, E::zero());
    for (size_t y = 0; y < H; y++) {
      const E m00 = mat_.M[y * J + y], m01 = mat_.M[y * J + H + y], m10 = mat_.M[(H + y) * J + y], m11 = mat_.M[(H + y) * J + H + y];
      c[0] += w[y] * m00;
      c[1] += w[y] * (m01 + m10 - m00 - m00);
      c[2] += w[y] * (m11 - m10 - m01 + m00);
    }
    return c;
  }
  void matrix_bind(const E &r) {
    const unsigned rest = mat_.bits - mat_.done - 1;
    const size_t J = size_t(2) << rest, H = J / 2;
    const E s = E::one() - r;
    std::vector<E> next(H * H);
    for (size_t a = 0; a < H; a++)
      for (size_t b = 0; b < H; b++)
        next[a * H + b] = s * (s * mat_.M[a * J + b] + r * mat_.M[a * J + H + b]) + r * (s * mat_.M[(H + a) * J + b] + r * mat_.M[(H + a) * J + H + b]);
    erase(mat_.M);
    mat_.M.swap(next);
    if (++mat_.done == mat_.bits) erase(mat_.M);  // its last round is served
  }
  // Adds to `matrix_` `factor` times what the entries in `roomL_` and `roomR_` contribute: `count`
  // values of l from `first` on. w must be the latest result of weights(): the branch with byte
  // digits reads `low_` and `high_`, not w.
  void matrix_of_rooms(unsigned group, u64 first, u64 count, const LkppWeights &w, const E &factor) {
    const int J = 1 << group;
    if (digit_columns_ && group == 3 && count >= DIGIT_COLUMNS_LEAST) {
      for (u64 from = 0; from < count; from += digit_columns_) {
        const u64 columns = std::min(digit_columns_, count - from);
        digitsA_.spent(), digitsB_.spent();
        lkpp_k_digits<<<grid(2 * columns), 256>>>(roomL_.get(), roomR_.get(), count, from, columns, low_.get(), digitsA_.get(), digitsB_.get(), digit_columns_);
        ENGINE_CK(cudaGetLastError());
        matrix_of_digits(first + from, columns, factor);
      }
      return;
    }
    const unsigned blocks = grid(count, 128, MATRIX_BLOCKS);
    part_.spent();
    if (group == 1)
      lkpp_k_matrix<2><<<dim3(blocks, 1, 1), 128>>>(roomL_.get(), roomR_.get(), count, first, w, J, part_.get());
    else
      lkpp_k_matrix<4><<<dim3(blocks, J / 4, J / 4), 128>>>(roomL_.get(), roomR_.get(), count, first, w, J, part_.get());
    lkpp_k_matrix_added<<<grid(u64(J) * J), 256>>>(part_.get(), blocks, unsigned(J * J), factor, matrix_.get());
    ENGINE_CK(cudaGetLastError());
  }
  // The same from the first `columns` columns of the two digit matrices, the values of l from
  // `first` on. Both numbers must be multiples of a chunk, 2^15 (not checked).
  void matrix_of_digits(u64 first, u64 columns, const E &factor) {
    const u64 chunk = u64(1) << LKPP_CHUNK_BITS;
    const int chunks = (int)(columns >> LKPP_CHUNK_BITS), one = 1, zero = 0;
    digit_sums_.spent(), blas_work_.spent(), part_.spent();
    // Chunk q: both matrices from column q 2^15 on. NVIDIA, cuBLAS documentation,
    // "cublasGemmStridedBatchedEx()", with the operand types and conditions that
    // prover/common/blas.h lists.
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        blas::blas(nullptr, blas_work_.get(), blas_work_.size()), CUBLAS_OP_T, CUBLAS_OP_N, (int)LKPP_ROWS, (int)LKPP_ROWS, (int)chunk, &one, digitsA_.get(), CUDA_R_8I,
        (int)digit_columns_, (long long)chunk, digitsB_.get(), CUDA_R_8I, (int)digit_columns_, (long long)chunk, &zero, digit_sums_.get(), CUDA_R_32I, (int)LKPP_ROWS,
        (long long)(LKPP_ROWS * LKPP_ROWS), chunks, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    BLK_REQUIRE(status == CUBLAS_STATUS_SUCCESS, "look-up in pieces: cuBLAS's batched product of byte matrices failed");
    digit_products++;
    lkpp_k_join<<<chunks, 64>>>(digit_sums_.get(), high_.get() + (first >> LKPP_CHUNK_BITS), part_.get());
    lkpp_k_matrix_added<<<1, 64>>>(part_.get(), (unsigned)chunks, 64, factor, matrix_.get());
    ENGINE_CK(cudaGetLastError());
  }
  void matrix_begin(unsigned group, const std::vector<E> &z) {
    matrix_.spent();
    ENGINE_CK(cudaMemset(matrix_.get(), 0, sizeof(E) << (2 * group)));
    mat_.bits = group, mat_.done = 0, mat_.z = z;
  }
  void matrix_end() {
    mat_.M.assign(size_t(1) << (2 * mat_.bits), E::zero());
    to_host(mat_.M.data(), matrix_.get(), mat_.M.size());
  }

  // ---- the passes ----
  /* A matrix pass in the sumcheck of level l0 + t - 1, 2 <= t < G. z, zq, zg: the level's point at
     the bits of l (`LkppFromTop`), of the piece's number and of this pass's `group` rounds. */
  void top_matrix_pass(unsigned t, unsigned group) {
    const unsigned folded = (unsigned)coins_.size(), left = t - 1 - folded - group, bits = left + c_.s0, mu = c_.l0 + t - 1;
    const Time then = now();
    fold_table(eq_top_first(coins_.data(), folded));
    std::vector<E> z(bits), zq(c_.Qb), zg(group);
    for (unsigned i = 0; i < bits; i++) z[i] = at(i < c_.s0 ? i : i + c_.Qb);
    for (unsigned i = 0; i < c_.Qb; i++) zq[i] = at(c_.s0 + i);
    for (unsigned i = 0; i < group; i++) zg[i] = at(mu - 1 - folded - i);
    const LkppWeights w = weights(z);
    matrix_begin(group, zg);
    const LkppFromTop s{c_.G - t, 1, folded, group + bits, group};
    const u64 all = u64(1) << bits;
    const bool digits = digit_columns_ && group == 3 && all >= DIGIT_COLUMNS_LEAST;  // the entries' digits go straight to the digit matrices
    const u64 count = digits ? std::min(all, digit_columns_) : std::min(all, (u64)roomL_.size() >> group);
    for (u64 q = 0; q < c_.pieces(); q++) {
      lay(q);
      const E wq = piece_weight(zq, q);
      for (u64 first = 0; first < all; first += count) {
        if (digits) {
          digitsA_.spent(), digitsB_.spent();
          lkpp_k_from_top<true, true><<<grid((count / 4) << group), 256>>>(keys_.get(), alpha_, s, fold_.get(), first, count, nullptr, nullptr,
                                                                           LkppDigitRooms{low_.get(), digitsA_.get(), digitsB_.get(), digit_columns_});
          ENGINE_CK(cudaGetLastError());
          matrix_of_digits(first, count, wq);
          continue;
        }
        roomL_.spent(), roomR_.spent();
        from_top(grid((count / 4) << group), keys_.get(), alpha_, s, fold_.get(), first, count, roomL_.get(), roomR_.get());
        matrix_of_rooms(group, first, count, w, wq);
      }
    }
    matrix_end();
    passes++;
    s_matrix += since(then);
  }
  void top_storing_pass(unsigned t) {
    const Time then = now();
    store_.spent();
    fold_table(eq_top_first(coins_.data(), coins_.size()));
    const LkppFromTop s{c_.G - t, 1, t - 1, c_.s0, 0};
    const u64 share = u64(1) << c_.s0;
    for (u64 q = 0; q < c_.pieces(); q++) {
      lay(q);
      from_top(grid(share / 4), keys_.get(), alpha_, s, fold_.get(), 0, share, store_.get() + q * share,
                                                store_.get() + (u64(1) << c_.l0) + q * share);
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaDeviceSynchronize());
    passes++;
    s_store += since(then);
  }
  // A matrix pass for the last level. by_position_[i]: the challenge of bit i of the column.
  void leaves_matrix_pass() {
    const unsigned folded = (unsigned)by_position_.size(), group = 3, bits = c_.np - 1 - folded - group;
    const Time then = now();
    fold_table(eq_low_first(by_position_));
    std::vector<E> z(bits), zq(c_.Qb), zg(group);
    for (unsigned i = 0; i < bits; i++) {
      const unsigned in_piece = i + folded + group;
      z[i] = at(in_piece < c_.s0 ? in_piece : in_piece + c_.Qb);
    }
    for (unsigned i = 0; i < c_.Qb; i++) zq[i] = at(c_.s0 + i);
    for (unsigned i = 0; i < group; i++) zg[i] = at(folded + group - 1 - i);
    const LkppWeights w = weights(z);
    matrix_begin(group, zg);
    const LkppFromLeaves s{folded, group, 0, 0, 0, false};
    const u64 all = u64(1) << bits, count = std::min(all, (u64)roomL_.size() >> group);
    const bool integers = !shape_.plain && !folded && fixed_.d <= 13;
    if (integers) ENGINE_CK(cudaMemset(matrix_.get(), 0, sizeof(E) * 81));
    for (u64 q = 0; q < c_.pieces(); q++) {
      lay(q);
      const E wq = piece_weight(zq, q);
      if (integers) {
        const u64 highs = all >> w.bits;
        const unsigned blocks = (unsigned)std::min<u64>(highs, 132 * 4);
        part_.spent();
        lkpp_k_leaf_matrix<<<9 * blocks, 128>>>(keys_.get(), u64(1) << (c_.np - 1), low_.get(), w.bits, high_.get(), highs, part_.get());
        lkpp_k_matrix_added<<<1, 96>>>(part_.get(), blocks, 81, wq, matrix_.get());
        ENGINE_CK(cudaGetLastError());
        continue;
      }
      for (u64 first = 0; first < all; first += count) {
        roomL_.spent(), roomR_.spent();
        from_leaves(grid(count), keys_.get(), u64(1) << (c_.np - 1), alpha_, s, fold_.get(), first, count, roomL_.get(), roomR_.get());
        matrix_of_rooms(group, first, count, w, wq);
      }
    }
    if (integers) {
      std::vector<E> N(81);
      to_host(N.data(), matrix_.get(), 81);
      const E square = alpha_ * alpha_;
      mat_.M.assign(64, E::zero());
      for (int a = 0; a < 8; a++)
        for (int b = 0; b < 8; b++) mat_.M[8 * a + b] = square * N[80] - alpha_ * (N[9 * a + 8] + N[72 + b]) + N[9 * a + b];
      erase(N);
      integer_passes++;
    } else {
      matrix_end();
    }
    passes++;
    s_matrix += since(then);
  }
  void leaves_storing_pass() {
    const unsigned folded = (unsigned)by_position_.size(), bits = c_.np - 1 - folded;
    const Time then = now();
    store_.spent();
    fold_table(eq_low_first(by_position_));
    for (u64 q = 0; q < c_.pieces(); q++) {
      lay(q);
      const LkppFromLeaves s{folded, 0, c_.s0 - folded, c_.Qb, q, true};
      from_leaves(grid(u64(1) << bits), keys_.get(), u64(1) << (c_.np - 1), alpha_, s, fold_.get(), 0, u64(1) << bits, store_.get(),
                                                        store_.get() + (u64(1) << c_.l0));
      ENGINE_CK(cudaGetLastError());
    }
    ENGINE_CK(cudaDeviceSynchronize());
    passes++;
    s_store += since(then);
  }

  // ---- one level's sumcheck ----
  void level_begin(size_t part, unsigned level, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) {
    BLK_REQUIRE(part < cuts_.size() && level < cuts_[part].v && vars.size() == level && order.size() == level, "look-up in pieces: no such level, or not one weight and one place in the order for each of its variables");
    if (level == 0) open(part);
    BLK_REQUIRE(open_ == part && level == next_level_of(part), "look-up in pieces: the levels of a part are proved in turn, from level 0 on, part after part");
    fresh_ = false, level_ = level, round_ = 0, order_ = order;
    z_.assign(level, E::zero());
    for (unsigned j = 0; j < level; j++) {
      BLK_REQUIRE(order[j] < level && vars[j].moment[0] == E::one(), "look-up in pieces: a key level runs under equality weights");
      z_[order[j]] = vars[j].moment[1];
    }
    coins_.clear(), by_position_.clear(), bound_.clear(), mat_.bits = mat_.done = 0;
    E *store = store_.get();
    if (c_.whole || level < c_.l0) {
      early_ = 0, last_ = false;
      mem_.L = store + (u64(2) << level), mem_.R = mem_.L + (u64(1) << level);
      mem_.otherL = store, mem_.otherR = store + (u64(1) << level) / 2;
      if (!c_.whole)
        for (unsigned j = 0; j < level; j++) BLK_REQUIRE(order[j] == level - 1 - j, "look-up in pieces: a large part's levels bind their variables from the highest down");
      mem_.variable.resize(level);
      for (unsigned i = 0; i < level; i++) mem_.variable[i] = i;
      return;
    }
    last_ = level == c_.v - 1, t_ = level - c_.l0 + 1, early_ = last_ ? 9 : t_ - 1;
    static const unsigned nine[9] = {2, 1, 0, 5, 4, 3, 8, 7, 6};
    for (unsigned j = 0; j < level; j++)
      BLK_REQUIRE(order[j] == (!last_ ? level - 1 - j : j < 9 ? nine[j] : level - 1 - (j - 9)),
                  "look-up in pieces: a large part's levels bind their variables from the highest down, the last level after the bits 2, 1, 0, 5, 4, 3, 8, 7, 6");
    if (!early_) stored();
  }
  unsigned next_level_of(size_t) const { return fresh_ ? 0 : level_ + 1; }
  void stored() {
    mem_.variable.clear();
    if (last_) {
      leaves_storing_pass();
      for (unsigned v = 9; v < level_; v++) mem_.variable.push_back(v);
    } else {
      top_storing_pass(t_);
      for (unsigned v = 0; v < c_.l0; v++) mem_.variable.push_back(v);
    }
    mem_.L = store_.get(), mem_.R = store_.get() + (u64(1) << c_.l0), mem_.otherL = mem_.otherR = nullptr;
  }
  std::vector<E> level_round() {
    BLK_REQUIRE(round_ < level_ && !asked_, "look-up in pieces: no variable is left, or the last round was not bound");
    asked_ = true;
    if (coins_.size() >= early_) {
      const Time then = now();
      std::vector<E> c = memory_round(order_[round_]);
      s_memory += since(then);
      return c;
    }
    if (mat_.done == mat_.bits) {
      if (last_)
        leaves_matrix_pass();
      else
        top_matrix_pass(t_, std::min<unsigned>(shape_.rounds, early_ - (unsigned)coins_.size()));
    }
    return matrix_round();
  }
  void level_bind(const E &g) {
    BLK_REQUIRE(asked_, "look-up in pieces: a challenge without its round");
    asked_ = false;
    bound_.push_back(g);
    if (coins_.size() >= early_) {
      const Time then = now();
      memory_bind(order_[round_++], g);
      s_memory += since(then);
      return;
    }
    round_++;
    matrix_bind(g);
    coins_.push_back(g);
    if (last_ && coins_.size() % 3 == 0) {  // the rounds bind the bits 2, 1, 0, then 5, 4, 3, then 8, 7, 6
      const size_t s = coins_.size();
      by_position_.insert(by_position_.end(), {coins_[s - 1], coins_[s - 2], coins_[s - 3]});
    }
    if (coins_.size() == early_) stored();
  }
  std::vector<E> level_ends() {
    BLK_REQUIRE(round_ == level_ && !asked_ && mem_.variable.empty(), "look-up in pieces: variables are left");
    std::vector<E> v(2);
    to_host(&v[0], mem_.L, 1), to_host(&v[1], mem_.R, 1);
    if (level_ + 1 == c_.v && at_part_end_) {
      std::vector<E> by_variable(level_);
      for (unsigned j = 0; j < level_; j++) by_variable[order_[j]] = bound_[j];
      keys_.spent(); // the callback may overwrite it
      at_part_end_((unsigned)open_, by_variable, keys_.get());
    }
    return v;
  }

  // k_col_: a place has 2^k_col_ keys, one for each column (comment at LkpPiece). WfGpu::prover()
  // (prover/engine/wf_heavy.cuh) passes WfMatrix::k_col: the committed matrix has 2^k_col columns.
  const LkpPublic fixed_;
  const unsigned k_col_;
  LkpKeySource &source_;
  std::vector<E> bits_;
  const LkpCut shape_;
  std::vector<Cut> cuts_;
  Card<E> store_, roomL_, roomR_, low_, high_, point_, part_, matrix_, fold_;
  Card<Key16> keys_;
  Card<signed char> digitsA_, digitsB_;  // the two digit matrices, of digit_columns_ columns; 0: none
  Card<int> digit_sums_;
  Card<unsigned char> blas_work_;
  u64 digit_columns_ = 0;
  E alpha_ = E::zero();
  std::vector<E> roots_;
  std::vector<std::vector<E>> table_levels_;
  // the open part and its current level
  size_t open_ = ~size_t(0);
  bool fresh_ = false;  // its stored levels are intact: no sumcheck has run on them
  Cut c_;
  // t_: t in the comment at LkpPiece. early_: how many of the level's first rounds matrix passes
  // serve.
  unsigned level_ = 0, round_ = 0, t_ = 0, early_ = 0;
  bool last_ = false, asked_ = false;
  std::vector<unsigned> order_;
  // z_[i]: the level's point at variable i. coins_: the challenges of the first early_ rounds, in
  // round order.
  std::vector<E> z_, coins_, by_position_, bound_;  // bound_: the level's challenges, in the order of the rounds
  std::vector<std::vector<u64>> counts_;
  std::function<void(unsigned, const std::vector<E> &, Key16 *)> at_part_end_;
};

}  // namespace blk
