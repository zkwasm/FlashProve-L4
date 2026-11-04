// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The table side of the lookup forest (printed "look-up forest"), which proves a group of lookup
// calls as one lookup. The argument (blocks/iop/lkp.h) compares a product over the keys with the
// product of (alpha - row value)^multiplicity over the table rows; a multiplicity counts the
// non-padding keys that look up its row. For the second product this file has 32 product trees, one
// per bit position of a multiplicity, over the rows and then the calls, and the sumcheck layers
// from their roots down to those bits. The engine, struct Forest (prover/hiding/forest.cuh), calls
// it and provides the large scratch memory. Include after prover/hiding/product_test.cuh.
//
// An index is (t | y | j), top bits first: table row t (kT bits), call number y (rK bits), bit
// position j of a multiplicity (5 bits; `plane` in identifiers). beta is bit j of the multiplicity
// of row t in call y, and x = 32 y + j is its column. A layer is a sumcheck over the indices c,
// with the weights eq(rho, c) at its point rho; its rounds bind the top index bit first.
//   rK + kT tree layers: summand L R, the two halves of a level of the product tree whose leaves
//                        are 1 + beta (alpha - 1 - tauP(t)); struct Trees defines alpha and tauP
//   leaf layer:          summand 1 + L R, L = beta, R = alpha - 1 - tauP(t)
//   bit layer:           summand L R, L = beta, R = beta - 1
// The ideas are public: the sources are those of prover/hiding/table_side.cuh; those for the mask
// bytes are cited at k_mask8().
#pragma once
#include "hiding/forest_plan.h"
namespace ft {
// tside's arithmetic on T: its add(), sub() and mul() call F's.
using tside::PF;
// "Tree y" means the 32 columns of call number y, each a product tree over the table rows. The
// trees into which a call's keys are cut (prover/hiding/forest_plan.h) are called "trees of keys".
// There are NT = 2^rK call numbers for the group's NC calls; this rK is the group's cK
// (ForestPlan::of_the_table_side()).
// Number y has the table of call c = min(y, NC - 1): its 2^k_c row values tau_c, shifted by c theta
// and repeated periodically: tauP(t) = c theta + tau_c(t mod 2^k_c), t < 2^kT. alpha and theta are
// the lookup's challenges. A key is counted at the first copy of its row, in its call's one
// histogram, so the multiplicities at t >= 2^k_c and in the numbers y >= NC are 0.
// `held`: how many numbers have stored columns; nlo = 32 `held` columns in all.
// ForestTable::init() requires NT = 2^rK, so `held` = NT, and the kernels' branches for an unused
// number, y >= NT, are never taken.
struct Trees {
  const T *lut[16] = {}; // for each call c: lut[t] = alpha - c theta - tau_c(t), 2^k_c entries
  u32 last[16] = {};     // for each call: 2^k - 1
  u8 call[256] = {};     // the call of tree y
  int NT = 0, held = 0, kT = 0; // a launch copies every member, so each has an initial value
  // Returns alpha - tauP(t) for tree y < NT.
  __device__ __forceinline__ T row(int y, size_t t) const {
    const int c = call[y];
    return ld16(lut[c] + ((u32)t & last[c]));
  }
};
// Returns a if bit is 1, b if bit is 0. The multiplicities are hidden values. Wherever one of their
// bits decides what enters a sum or a product, this file's kernels compute both values and select
// with a bit mask: in these kernels no branch, no load and no skipped multiplication depends on a
// multiplicity. Two limits:
// (1) That describes the source only. The field arithmetic on the selected values reduces by
// conditional expressions (for example GL64::add() in blocks/field/gl64.h), which the compiler is
// free to turn into branches.
// (2) Kernels of other files that ForestTable launches have addresses or branches that depend on
// multiplicity bits or on the counted keys. At least these do. In counts(), h_idk and
// k_count_keys16 call atomicAdd() in shared memory at an address computed from each key, the second
// under a branch on the key's top bit. In bit_layer(), tside::k_mask (for RD = 1 and 2) and
// k_foldmaskN load shared-memory tables at indices made of multiplicity bits.
__device__ __forceinline__ T pick(u32 bit, const T &a, const T &b) {
  const u64 m = 0 - (u64)bit;
  return T{(a.k0 & m) | (b.k0 & ~m), (a.k1 & m) | (b.k1 & ~m), (a.k2 & m) | (b.k2 & ~m)};
}
// A mask byte packs 8 multiplicity bits. The bottom layer (the last tree layer, which reads the
// leaves) and the bit layer compute their first three rounds from mask bytes, as
// prover/hiding/table_side.cuh describes. cnt[y 2^kT + t] is the multiplicity of row t in tree y.
// The bit layer's bytes: bit a of M[tl nlo + x] is bit j of the multiplicity of row
// a 2^(kT - 3) + tl in tree y.
// The idea is public: Y. Jeon, B. Park, S. J. Kwon, B. Kim, J. Yun, D. Lee, "BiQGEMM: Matrix
// Multiplication with Lookup Table For Binary-Coding-based Quantized DNNs", arXiv:2005.09904
// (2020), section III-B: the one-bit weights that meet the same inputs of a matrix product are
// grouped into one integer, which is used as the index into a table of their possible partial sums.
// The idea is public: J. Wei, S. Cao, T. Cao, L. Ma, L. Wang, Y. Zhang, M. Yang, "T-MAC: CPU
// Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge", arXiv:2407.00088 (2024),
// section 3.1: the same grouping, used as a table index, for each bit plane of low-bit weights.
// Here the group is the eight entries that one 8-to-1 fold combines, and the sums are of field
// elements; the table is in k_foldmaskN (prover/hiding/product_test.cuh).
__global__ void k_mask8(const u32 *__restrict__ cnt, int kT, int NT, int held, u8 *__restrict__ M) {
  const u32 nlo = 32u * held;
  const size_t n = (size_t)nlo << (kT - 3);
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < n; c += (size_t)gridDim.x * blockDim.x) {
    const u32 tl = (u32)(c / nlo), x = (u32)(c - (size_t)tl * nlo), y = x >> 5, j = x & 31;
    u32 m = 0;
    if (y < (u32)NT) {
      const u32 *p = cnt + ((size_t)y << kT) + tl;
#pragma unroll
      for (int a = 0; a < 8; a++) m |= ((p[(u32)a << (kT - 3)] >> j) & 1u) << a;
    }
    M[c] = (u8)m;
  }
}
// The bottom layer's bytes, for its tables L (ML) and R (MR), indexed in the same way: bit a of ML
// belongs to row a 2^(kT - 4) + tl, bit a of MR to row (8 + a) 2^(kT - 4) + tl.
__global__ void k_mask16(const u32 *__restrict__ cnt, int kT, int NT, int held, u8 *__restrict__ ML,
                         u8 *__restrict__ MR) {
  const u32 nlo = 32u * held;
  const size_t n = (size_t)nlo << (kT - 4);
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < n; c += (size_t)gridDim.x * blockDim.x) {
    const u32 tl = (u32)(c / nlo), x = (u32)(c - (size_t)tl * nlo), y = x >> 5, j = x & 31;
    u32 l = 0, r = 0;
    if (y < (u32)NT) {
      const u32 *p = cnt + ((size_t)y << kT) + tl;
#pragma unroll
      for (int a = 0; a < 8; a++) {
        l |= ((p[(u32)a << (kT - 4)] >> j) & 1u) << a;
        r |= ((p[(8u + a) << (kT - 4)] >> j) & 1u) << a;
      }
    }
    ML[c] = (u8)l;
    MR[c] = (u8)r;
  }
}
// Writes one byte for each multiplicity bit of the trees y < NT. order 0: bit (t, y, j) goes to
// (y 2^kT + t) 32 + j. order 1: to (t 2^rK + y) 32 + j, which requires NT = 2^rK.
__global__ void k_bits(const u32 *__restrict__ cnt, int kT, int rK, int NT, int order, u8 *__restrict__ to) {
  const size_t n = (size_t)NT << (kT + 5);
  for (size_t X = blockIdx.x * (size_t)blockDim.x + threadIdx.x; X < n; X += (size_t)gridDim.x * blockDim.x) {
    const size_t h = X >> 5, at = order ? ((h & (((size_t)1 << rK) - 1)) << kT) + (h >> rK) : h;
    to[X] = (u8)((cnt[at] >> (X & 31)) & 1u);
  }
}
// Level l of the trees over the rows has 2^l entries per column: level 0 is their roots, level kT
// the leaves. Returns entry c = t nlo + x of level kT - 1: the product of column x's leaves at rows
// t and t + 2^(kT - 1).
__device__ __forceinline__ T pair_at(const u32 *__restrict__ cnt, const Trees &tr, u32 nlo, size_t c) {
  const u32 half = 1u << (tr.kT - 1), t = (u32)(c / nlo), x = (u32)(c - (size_t)t * nlo), y = x >> 5, j = x & 31;
  if (y >= (u32)tr.NT) return F::zero();
  const u32 *p = cnt + ((size_t)y << tr.kT) + t;
  const T one = F::one();
  return F::mul(pick((p[0] >> j) & 1, tr.row(y, t), one), pick((p[half] >> j) & 1, tr.row(y, t + half), one));
}
// Writes the levels kT - 1 (o0) to kT - 4 (o3) in one pass, as k_up3
// (prover/hiding/pt_transcript.h) does, but from entries that pair_at() computes.
__global__ void __launch_bounds__(256) k_pairs(const u32 *__restrict__ cnt, const Trees tr, T *__restrict__ o0,
                                               T *__restrict__ o1, T *__restrict__ o2, T *__restrict__ o3) {
  const u32 nlo = 32u * tr.held;
  const size_t n8 = (size_t)nlo << (tr.kT - 4);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
    T a[4], b[2];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const T lo = pair_at(cnt, tr, nlo, i + k * n8), hi = pair_at(cnt, tr, nlo, i + (k + 4) * n8);
      st16(o0 + i + k * n8, lo);
      st16(o0 + i + (k + 4) * n8, hi);
      a[k] = F::mul(lo, hi);
      st16(o1 + i + k * n8, a[k]);
    }
#pragma unroll
    for (int k = 0; k < 2; k++) {
      b[k] = F::mul(a[k], a[k + 2]);
      st16(o2 + i + k * n8, b[k]);
    }
    st16(o3 + i, F::mul(b[0], b[1]));
  }
}
// The roots, computed without the tree: Y_(y,j) is the product of alpha - tauP(t) over the rows
// t < 2^k of tree y's table whose multiplicity has bit j set. Block (b, y) writes a partial product
// to part[(b NT + y) 32 + j]. Blocks of 256 threads. A warp takes one row at a time; its lane j
// handles bit position j.
// The idea is public: a reduction in two passes, by block and then over the blocks: M. Harris,
// "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology,
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. There the operation is a
// sum, here a product.
// Public source: the same two passes: NVIDIA, the CUDA samples, "reduction".
// Public source: a block's result from its warps' results in shared memory: J. Luitjens,
// "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014,
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/. No warp shuffle is used
// here: lane j keeps column j's product to the end.
// The compiler hint is public: __launch_bounds__: NVIDIA, "CUDA C++ Best Practices Guide",
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(256) k_root_parts(const u32 *__restrict__ cnt, const Trees tr, T *part) {
  __shared__ T sh[8][32];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5, y = blockIdx.y;
  const u32 rows = tr.last[tr.call[y]] + 1;
  const u32 *p = cnt + ((size_t)y << tr.kT);
  const T one = F::one();
  T v = one;
  for (u32 t = blockIdx.x * 8 + wb; t < rows; t += gridDim.x * 8) v = F::mul(v, pick((p[t] >> lane) & 1, tr.row(y, t), one));
  sh[wb][lane] = v;
  __syncthreads();
  if (wb == 0) {
#pragma unroll
    for (int w = 1; w < 8; w++) v = F::mul(v, sh[w][lane]);
    part[((size_t)blockIdx.x * tr.NT + y) * 32 + lane] = v;
  }
}
// Multiplies each column's nb partial products into Y[32 y + j]. One block of 256 threads
// per tree y. The reduction's second pass, with the same step through shared memory and the same
// compiler hint (cited at k_root_parts()).
__global__ void __launch_bounds__(256) k_root_join(const T *part, int nb, int NT, T *Y) {
  __shared__ T sh[8][32];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5, y = blockIdx.x;
  T v = F::one();
  for (int b = wb; b < nb; b += 8) v = F::mul(v, part[((size_t)b * NT + y) * 32 + lane]);
  sh[wb][lane] = v;
  __syncthreads();
  if (wb == 0) {
#pragma unroll
    for (int w = 1; w < 8; w++) v = F::mul(v, sh[w][lane]);
    Y[32 * y + lane] = v;
  }
}
// Round RD (0, 1 or 2) of the bottom layer. Let c = tl nlo + x with tl < 2^lsl, and let bL_a(c) and
// bR_a(c) be bit a of ML[c] and MR[c]. The layer's tables are
//   L(a, c) = 1 + bL_a(c) (alpha - tauP(a 2^lsl + tl) - 1)
//   R(a, c) = 1 + bR_a(c) (alpha - tauP((8 + a) 2^lsl + tl) - 1)
// and index c has the weight w = eqT[tl] eqL[x]. rs->E[h] = eq(the earlier rounds' challenges, h).
// part gets each block's partial sums, two for each value of the 2 - RD bits of a below this
// round's bit, which tside::k_end (prover/hiding/table_side_common.inc) adds. Blocks of tside::TB
// threads. A warp takes one pair (tl, y) at a time; its lane j handles column x = 32 y + j.
template <int RD>
__global__ void __launch_bounds__(tside::TB)
    k_rounds(const Trees tr, const u8 *__restrict__ ML, const u8 *__restrict__ MR, const T *__restrict__ eqT,
             const T *__restrict__ eqL, int lsl, const tside::Rs *rs, T *part) {
  constexpr int NS = 8 >> RD, NV = 4 >> RD, NH = 1 << RD, WPB = tside::WPB;
  // tv[.][side][a] = E[h(a)] (alpha - tauP((8 side + a) 2^lsl + tl) - 1), h(a) = a >> (3 - RD).
  // For RD = 0, E = 1.
  __shared__ T tv[WPB][2][8];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
  const size_t nlo = 32 * (size_t)tr.held, all = (size_t)tr.NT << lsl;
  const T one = F::one();
  T v[2 * NV];
#pragma unroll
  for (int q = 0; q < 2 * NV; q++) v[q] = tside::zero();
  for (size_t u = blockIdx.x * (size_t)WPB + wb; u < all; u += (size_t)gridDim.x * WPB) {
    const size_t tl = u / tr.NT;
    const int y = (int)(u - tl * tr.NT);
    __syncwarp();
    if (lane < 16) {
      const T g = PF::sub(tr.row(y, ((size_t)lane << lsl) + tl), one);
      tv[wb][lane >> 3][lane & 7] = RD ? PF::mul(rs->E[(lane & 7) >> (3 - RD)], g) : g;
    }
    __syncwarp();
    const size_t x = 32 * (size_t)y + lane;
    const u32 bl = ML[tl * nlo + x], br = MR[tl * nlo + x];
    const T w = PF::mul(eqT[tl], eqL[x]);
#pragma unroll
    for (int t = 0; t < NV; t++) {
      T l[2], r[2]; // the low and the high half
#pragma unroll
      for (int hl = 0; hl < 2; hl++) {
        l[hl] = r[hl] = one;
#pragma unroll
        for (int h = 0; h < NH; h++) {
          const int a = h * NS + hl * NV + t;
          l[hl] = PF::add(l[hl], pick((bl >> a) & 1, tv[wb][0][a], tside::zero()));
          r[hl] = PF::add(r[hl], pick((br >> a) & 1, tv[wb][1][a], tside::zero()));
        }
      }
      v[2 * t] = PF::add(v[2 * t], PF::mul(w, PF::mul(l[0], r[0])));
      v[2 * t + 1] = PF::add(v[2 * t + 1], PF::mul(w, PF::mul(PF::sub(l[1], l[0]), PF::sub(r[1], r[0]))));
    }
  }
  tside::block_sums<2 * NV>(v, part);
}
// The 8-to-1 fold of the bottom layer: out[c] = sum over a of k8[a] L(a, c), and out[n + c] is the
// same for R, with c, L and R as in k_rounds (lsl = kT - 4), n = 2^(kT - 4) nlo and k8[a] =
// eq(the three rounds' challenges, a). The k8[a] sum to 1, so the 1 in L and R is added once.
// Blocks of 256 threads, with warps and lanes assigned as in k_rounds.
__global__ void __launch_bounds__(256) k_fold_bottom(const Trees tr, const u8 *__restrict__ ML,
                                                     const u8 *__restrict__ MR, const T *k8, T *__restrict__ out) {
  __shared__ T kx[8][16];
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5, lsl = tr.kT - 4;
  const size_t nlo = 32 * (size_t)tr.held, all = (size_t)tr.held << lsl, n = nlo << lsl;
  const T one = F::one();
  for (size_t u = blockIdx.x * (size_t)8 + wb; u < all; u += (size_t)gridDim.x * 8) {
    const size_t tl = u / tr.held, c = tl * nlo + 32 * (u - tl * tr.held) + lane;
    const int y = (int)(u - tl * tr.held);
    T l = F::zero(), r = l;
    if (y < tr.NT) {
      __syncwarp();
      if (lane < 16) kx[wb][lane] = F::mul(k8[lane & 7], F::sub(tr.row(y, ((size_t)lane << lsl) + tl), one));
      __syncwarp();
      const u32 bl = ML[c], br = MR[c];
      l = r = one;
#pragma unroll
      for (int a = 0; a < 8; a++) {
        l = F::add(l, pick((bl >> a) & 1, kx[wb][a], F::zero()));
        r = F::add(r, pick((br >> a) & 1, kx[wb][8 + a], F::zero()));
      }
    }
    st16(out + c, l);
    st16(out + n + c, r);
  }
}
// The leaf layer's two tables over (t | y), 2^(kT + rK) entries each: Lam = the sum of eqJ[j]
// over the bits j set in the multiplicity of row t in tree y (Lambda in leaf_layer());
// R = alpha - tauP(t) - 1.
__global__ void __launch_bounds__(256) k_leaf_tables(const u32 *__restrict__ cnt, const Trees tr, int rK, const T *eqJ,
                                                     T *__restrict__ Lam, T *__restrict__ R) {
  __shared__ T e[32];
  if (threadIdx.x < 32) e[threadIdx.x] = eqJ[threadIdx.x];
  __syncthreads();
  const size_t n = (size_t)1 << (tr.kT + rK);
  for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < n; c += (size_t)gridDim.x * blockDim.x) {
    const size_t t = c >> rK, y = c & (((size_t)1 << rK) - 1);
    T l = F::zero(), r = l;
    if (y < (size_t)tr.NT) {
      const u32 m = cnt[(y << tr.kT) + t];
      for (int j = 0; j < 32; j++) l = F::add(l, pick((m >> j) & 1, e[j], F::zero()));
      r = F::sub(tr.row((int)y, t), F::one());
    }
    st16(Lam + c, l);
    st16(R + c, r);
  }
}
// o[h] = eq(the layer's first n round challenges, h), h < 2^n; the first challenge belongs to the
// top bit of h.
__global__ void k_eq_coins(const Chal<F> *ch, int n, T *o) {
  __shared__ T tb[7][32];
  if (blockIdx.x * (size_t)blockDim.x >= ((size_t)1 << n)) return;
  const int ng = eq_groups(ch->r, n, tb);
  __syncthreads();
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < ((size_t)1 << n);
       x += (size_t)gridDim.x * blockDim.x)
    o[x] = eq_at(tb, n, ng, x);
}
// The leaf layer's tables over (t | y | j), once the top nb bits of t are bound. For e = (tlow | y)
// of ls bits:
//   X[32 e + j] = the sum over h < 2^nb of eqH[h] times bit j of the multiplicity of row
//                 (h | tlow) in tree y;
//   X[2^(ls + 5) + 32 e + j] = R[e], the folded table over (t | y).
// Blocks of 256 threads. A warp takes one e at a time; its lane j handles bit position j.
__global__ void __launch_bounds__(256) k_leaf_planes(const u32 *__restrict__ cnt, const Trees tr, int rK, int ls, int nb,
                                                     const T *__restrict__ eqH, const T *__restrict__ R,
                                                     T *__restrict__ X) {
  const int lane = threadIdx.x & 31;
  const size_t n = (size_t)1 << ls;
  for (size_t e = blockIdx.x * (size_t)8 + (threadIdx.x >> 5); e < n; e += (size_t)gridDim.x * 8) {
    const size_t y = e & (((size_t)1 << rK) - 1);
    T l = F::zero();
    if (y < (size_t)tr.NT) {
      const u32 *p = cnt + (y << tr.kT) + (e >> rK);
      for (size_t h = 0; h < (size_t)1 << nb; h++) l = F::add(l, pick((p[h << (ls - rK)] >> lane) & 1, ld16(eqH + h), F::zero()));
    }
    st16(X + 32 * e + lane, l);
    st16(X + 32 * n + 32 * e + lane, ld16(R + e));
  }
}
// Takes a call's padding keys (public filler, all equal to one row) out of its histogram. at: the
// multiplicity of that row. pub: the number of padding keys in each of the call's I trees of keys;
// copied to pubo. The subtraction is unsigned and unchecked: the caller must have counted these
// keys in this row. One thread.
__global__ void k_pub_call(u32 *at, const u32 *pub, u32 *pubo, int I) {
  u32 all = 0;
  for (int b = 0; b < I; b++) pubo[b] = pub[b], all += pub[b];
  *at -= all;
}
// Draws the bit layer's point, n <= MAXV coordinates in one message. The first one drawn is the
// lowest variable's; rho[0] is the top variable's. Sets the layer's claimed sum to 0. One thread.
__global__ void k_bits_point(St *st, Chal<F> *ch, int n, int id) {
  T drawn[MAXV];
  tr_point(st->tr, Round::ZERO, drawn, n);
  for (int i = 0; i < n; i++) ch->rho[n - 1 - i] = drawn[i];
  st->start[id] = F::zero();
}
// Runs the first three rounds of the bottom layer, as tside::mask_rounds()
// (prover/hiding/table_side_kernels.inc) does where all columns share one table. `room`:
// tside::ctx_elements() elements.
static inline void rounds(const Trees &tr, const u8 *ML, const u8 *MR, const T *eqT, const T *eqL, int lsl, T *room,
                          St *st, Chal<F> *cd, T *out, T *k8, cudaStream_t stream) {
  const tside::Ctx x = tside::ctx_at(room);
  const size_t warps = (size_t)tr.NT << lsl;
  const int nb = (int)std::min<size_t>((warps + tside::WPB - 1) / tside::WPB, (size_t)std::min(tside::MAXB, g_sms * 4));
  k_rounds<0><<<nb, tside::TB, 0, stream>>>(tr, ML, MR, eqT, eqL, lsl, x.rs, x.part);
  CK(cudaGetLastError());
  tside::end_round(x, nb, 0, st, cd, 0, 1, 2, out, k8, stream);
  k_rounds<1><<<nb, tside::TB, 0, stream>>>(tr, ML, MR, eqT, eqL, lsl, x.rs, x.part);
  CK(cudaGetLastError());
  tside::end_round(x, nb, 1, st, cd, 0, 1, 2, out, k8, stream);
  k_rounds<2><<<nb, tside::TB, 0, stream>>>(tr, ML, MR, eqT, eqL, lsl, x.rs, x.part);
  CK(cudaGetLastError());
  tside::end_round(x, nb, 2, st, cd, 0, 1, 2, out, k8, stream);
}
} // namespace ft
// Call init(), counts() for every call and planes(); then, once alpha is drawn and the engine has
// filled every tabs[c].lut, roots() and prove(). Eng (prover/hiding/product_test.cuh) runs a
// layer's stored tables through Gram passes (three sumcheck rounds each,
// prover/hiding/gram_products.cuh) until they have at most 2^lmid entries. The tail kernel
// (k_tails, prover/hiding/tails.cuh) runs the remaining rounds.
struct ForestTable {
  ForestPlan plan; // of the table side, for which `v`, `key_slots` and `at_...` are meaningless
  ForestPlan keys; // the group's own plan: how a call's keys are cut into trees of keys
  std::vector<ForestCall> calls;
  std::vector<fk::Table> tabs; // one per call; the engine fills each `lut` once alpha is drawn
  Eng et;                      // runs the layers; et.st has their transcript (`tr`), alpha and err
  Chal<F> *chd = nullptr;      // one entry (point `rho`, challenges `r`) per layer; used here:
                               // 40 - rK + l for the rK tree layers over the calls, l < rK;
                               // 40 + l for the tree layer that reads level l + 1, l < kT;
                               // 40 + kT for the leaf layer; 80 for the bit layer
  // Test only, and set by no file of this tree: if false at init(), the columns of the unused
  // numbers are stored too. See `packed`.
  bool may_pack = true;
  // Test hook, which no file of this tree sets: called on the host once each phase of prove() has
  // been launched, with the phase's name.
  void (*phase_ends)(const char *) = nullptr;
  // If true, only the columns of the trees y < NT are stored. Always false: it needs NT < 2^rK,
  // which init() rejects. Its other conditions: lk = rK + 5 <= 10 is a limit of k_eqjobs, and
  // Eng::gram_rest() runs Gram passes on such tables only while their size with all columns is at
  // least 2^(lk + 7) entries, so smaller tables must fit the tail kernel.
  bool packed = false;
  int kT = 0, lk = 0, lmid = 0, lx = 0; // lx: the levels 1 .. lx have a copy with all columns, for the tail kernel
  size_t nlo = 0;                       // the stored columns: 32 NT or 32 2^rK
  ft::Trees trees;
  // cnt[y 2^kT + t]: the multiplicities.
  u32 *cnt = nullptr, *pubo = nullptr;
  u8 *M8 = nullptr, *ML = nullptr, *MR = nullptr; // the mask bytes of k_mask8 and k_mask16
  T *eqT = nullptr, *eqL = nullptr, *eqJ = nullptr, *eqH = nullptr, *tside_room = nullptr, *root_part = nullptr;
  int root_blocks = 0; // the blocks that share a tree's rows in roots()
  std::vector<T *> Tc, Tx; // level l as stored (in room); level l with all columns
  T *top = nullptr;
  std::vector<T *> Top;    // Top[u], u <= rK: level u of the trees over the calls, 32 2^u entries
  // keys_ must come from forest_plan() and plan_ from keys_.of_the_table_side(): forest_plan()'s
  // limits, such as NC <= 16 (the length of Trees::lut), are not checked again. Exit codes: 1 if NT
  // is not 2^rK, 2 if kT < 7.
  // Allocates through PA and PAN and frees nothing; pt_alloc (prover/hiding/product_test.cuh)
  // defines the kind numbers (0: for the object's lifetime; 1: from counts() to prove();
  // 2: scratch). The kind-2 buffers are small, and Forest::init() installs a hook that places them
  // in its one kind-2 allocation. The large scratch (the tree's levels, the fold buffers Fm and
  // et.S, the bytes of bits()) is not allocated here: the caller passes it as `room` or `to`.
  // prove() launches et's tail kernels through `solo`, which must then be valid (not checked).
  void init(const ForestPlan &plan_, const ForestPlan &keys_, const std::vector<ForestCall> &calls_,
            const std::vector<fk::Table> &tabs_, int lmid_, Chal<F> *chd_, Sched *solo) {
    if (!g_sms) dev_info();
    g_grid = g_sms * 8;
    plan = plan_, keys = keys_, calls = calls_, tabs = tabs_, chd = chd_;
    kT = plan.kT, lk = plan.rK + 5;
    // The argument needs leaves of 1 at a call number without multiplicities, but the kernels'
    // branches for an unused number give 0. So no number may be unused.
    if (plan.NT != 1 << plan.rK) {
      printf("internal error: look-up forest: the table side is set up with %d numbers of calls, no power of two (ForestPlan::of_the_table_side())\n", plan.NT);
      fflush(stdout);
      exit(1);
    }
    // lmid_ is ignored: tables of up to 2^pt_tail_cap = 2^16 entries always go to the tail
    // kernel, the cap for an Eng that runs by itself. The packed layout and the leaf layer rely on
    // this. No byte of the proof depends on lmid.
    (void)lmid_;
    lmid = pt_tail_cap;
    if (kT < 7) {
      printf("look-up forest: rejected: the table side wants a table of 2^7 rows or more in the group (kT = %d)\n", kT);
      fflush(stdout);
      exit(2);
    }
    packed = may_pack && plan.NT < (1 << plan.rK) && lk <= 10 && lmid >= lk + 6;
    trees.NT = plan.NT, trees.held = packed ? plan.NT : 1 << plan.rK, trees.kT = kT;
    nlo = 32 * (size_t)trees.held;
    for (int c = 0; c < plan.NC; c++) trees.lut[c] = tabs[c].lut, trees.last[c] = (1u << calls[c].k) - 1;
    for (int y = 0; y < plan.NT; y++) trees.call[y] = (u8)plan.call_of[y];
    PAN(pubo, 4 * (size_t)keys.NR, 0);
    PAN(top, sizeof(T) * 64 * (size_t)plan.NT, 0);
    Top.assign(plan.rK + 1, nullptr);
    {
      T *p = top;
      for (int u = plan.rK; u >= 0; u--) Top[u] = p, p += (size_t)32 << u;
    }
    PAN(cnt, (size_t)4 * plan.NT << kT, 1);
    PAN(M8, nlo << (kT - 3), 1);
    PAN(ML, nlo << (kT - 4), 1);
    PAN(MR, nlo << (kT - 4), 1);
    PA(eqT, 16ull << (kT - 3), 2);
    PA(eqL, 16 * nlo, 2);
    PA(eqJ, 16 * 32, 2);
    PA(eqH, 16ull << std::max(0, kT + plan.rK - 9), 2); // the leaf layer's Gram passes leave 2^9 entries or more
    PA(tside_room, 16 * tside::ctx_elements(), 2);
    root_blocks = std::max(1, std::min(1 << (kT - 5), g_grid / plan.NT));
    PA(root_part, 16 * 32 * (size_t)root_blocks * plan.NT, 2);
    Tc.assign(kT, nullptr), Tx.assign(kT, nullptr);
    lx = packed ? std::min(kT - 1, lmid + 1 - lk) : 0;
    for (int l = 1; l <= lx; l++) PA(Tx[l], 16ull << (l + lk), 2);
    // et is set up for a small layer; s_at() later points et.S[0] and et.S[1] into `room`. Gcb, a
    // Gram pass's sums per chunk, is sized for the largest stored layer: a pass sums over
    // nlo 2^(kT - 5) indices, in one chunk or in chunks of more than 2^14.
    et.init((size_t)1 << 16, lmid, chd);
    PA(et.Gcb, (((nlo << (kT - 5)) >> 14) + 2) * 1024, 2);
    PA(et.X, 32ull << lmid, 2);
    et.solo = solo;
#ifndef FLASHPROVE_L4
    CK_ATTRIBUTE(h_idk<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
#endif
  }
  // prove() needs this many elements of T at `room`, 256-byte aligned. Level l starts at element
  // nlo (2^l - 1). Nothing there is needed between two calls.
  size_t room_elements() const { return nlo << kT; }
  // Counts the keys of call c. from: GPU memory, 16-byte aligned, the call's 2^n_c row numbers
  // (n_c: `ForestCall::n`); the counting kernels mask each to k bits. pub: GPU memory, the number
  // of padding keys in each of its 2^iota_c trees of keys, of 2^dT keys each (iota_c = n_c - dT:
  // `ForestPlan::iota`; see k_pub_call), or null for none. The call's trees of keys are contiguous
  // in memory, so the counting kernels take all its keys as one array, into one histogram.
  void counts(int c, const uint16_t *from, const u32 *pub) {
    const int I = 1 << keys.iota[c], k = calls[c].k;
    const size_t words = (((size_t)1 << keys.dT) / 8) << keys.iota[c];
    const uint4 *kk = (const uint4 *)from;
    u32 *to = cnt + ((size_t)c << kT);
    CK(cudaMemsetAsync(to, 0, (size_t)4 << kT, g_st));
#ifdef FLASHPROVE_L4
    count_keys_l4(kk, words, g_sms, 1, k, to, g_st);
#else
    if (k == 16)
      count_keys16(kk, words, g_sms, 1, to, g_st);
    else if (k <= 10)
      h_idk<8><<<g_sms * 2, 512, (size_t)32 << k, g_st>>>(kk, words, g_sms * 2, k, to);
    else
      h_idk<1><<<g_sms * 2, 512, (size_t)4 << k, g_st>>>(kk, words, g_sms * 2, k, to);
#endif
    CK(cudaGetLastError());
    if (pub)
      ft::k_pub_call<<<1, 1, 0, g_st>>>(to + calls[c].t0, pub, pubo + keys.first[c], I);
    else
      CK(cudaMemsetAsync(pubo + keys.first[c], 0, 4 * (size_t)I, g_st));
    CK(cudaGetLastError());
  }
  // Zeroes the multiplicities of the numbers from NC on, writes the mask bytes and sets et.st->err
  // to 0. No overflow is tested: a group has fewer than 2^32 keys (forest_plan()), so every
  // multiplicity fits 32 bits.
  void planes() {
    k_err0<<<1, 1, 0, g_st>>>(et.st);
    if (plan.NT > plan.NC) CK(cudaMemsetAsync(cnt + ((size_t)plan.NC << kT), 0, (size_t)4 * (plan.NT - plan.NC) << kT, g_st));
    ft::k_mask8<<<g_grid, 256, 0, g_st>>>(cnt, kT, plan.NT, trees.held, M8);
    ft::k_mask16<<<g_grid, 256, 0, g_st>>>(cnt, kT, plan.NT, trees.held, ML, MR);
    CK(cudaGetLastError());
  }
  // Runs ft::k_bits. `to`: NT 2^(kT + 5) bytes. Call after planes().
  void bits(int order, u8 *to) {
    if (order && plan.NT != 1 << plan.rK) {
      printf("internal error: look-up forest: the bits in the table side's own order exist only if no number of a call is empty\n");
      fflush(stdout);
      exit(2);
    }
    ft::k_bits<<<g_grid, 256, 0, g_st>>>(cnt, kT, plan.rK, plan.NT, order, to);
    CK(cudaGetLastError());
  }
  const u32 *pub_dev() const { return pubo; } // the numbers of padding keys of the calls' trees of keys, keys.NR of them, kind 0
  // Writes the roots of the trees over the rows, Top[rK][32 y + j], without building a tree, and
  // the levels above them.
  void roots() {
    ft::k_root_parts<<<dim3(root_blocks, plan.NT), 256, 0, g_st>>>(cnt, trees, root_part);
    ft::k_root_join<<<plan.NT, 256, 0, g_st>>>(root_part, root_blocks, plan.NT, Top[plan.rK]);
    for (int u = plan.rK; u >= 1; u--) k_up1<F><<<1, 256, 0, g_st>>>(Top[u], Top[u - 1], (size_t)32 << (u - 1));
    CK(cudaGetLastError());
  }
  const T *plane_roots() const { return Top[0]; } // 32 entries; valid after roots()
  // Proves the table side on the transcript et.st->tr: the rK + kT tree layers from the 32 roots
  // down, the leaf layer, a new point, the bit layer. The caller must have set the first layer's
  // point (chd[40 - rK].rho, 5 coordinates) and claimed sum (et.st->start[40 - rK]). Builds the
  // trees over the rows in `room`. Launches on g_st. Writes plan.table_slots() message slots (one
  // for each scalar that is sent), starting at out. Each slot's mask entry, the random element
  // added to its value, must be PADOFF elements after it (PADOFF: a device variable that the
  // program sets once), which nothing checks (prover/hiding/sumcheck_door.cuh).
  void prove(T *room, T *out) {
    build(room);
    for (int l = 1; l <= lx; l++) { // level l as the pair of tables that layer l - 1 reads
      const size_t nh = (size_t)1 << (l - 1);
      k_expand<<<(int)std::min<size_t>(((nh << (lk + 1)) + 255) / 256, 1056), 256, 0, g_st>>>(Tc[l], Tx[l], nh,
                                                                                               (u32)nlo, lk, 4);
    }
    CK(cudaGetLastError());
    mark("the tree");
    et.sch = nullptr;
    et.out = out;
    for (int u = 1; u <= plan.rK; u++) et.stored(Top[u], 5 + u - 1, 40 - plan.rK + u - 1, 40 - plan.rK + u);
    // Layer l reads level l + 1. The levels up to l are free by then: the layers that read them are
    // launched, on this stream, before layer l writes there (for a layer whose tables have more
    // than 2^lmid entries et.stored() first launches the tail list, the layers that await the tail
    // kernel).
    for (int l = 0; l < kT - 1; l++) {
      const bool big = lk + l > lmid;
      if (big) s_at(room, nlo << l >> 1);
      et.stored(big || !packed ? Tc[l + 1] : Tx[l + 1], lk + l, 40 + l, 40 + l + 1, big ? cmp(4) : Cmp());
    }
    et.flush();
    mark("stored layers");
    // From here on no level is needed. The folded tables reuse level kT - 1, and et's fold buffers
    // the other levels.
    Fm = Tc[kT - 1];
    s_at(room, nlo << (kT - 3));
    {
      const int mu = lk + kT - 1, id = 40 + kT - 1;
      T *o = et.take(mu);
      eq_tables(chd + id, kT - 4, kT - 1);
      pt_hi(k_layer_begin, 1, 32, 0, g_st, et.st, (const Chal<F> *)(chd + id), mu, id, 0, o, 0);
      ft::rounds(trees, ML, MR, eqT, eqL, kT - 4, tside_room, et.st, chd + id, o, et.k8, g_st);
      ft::k_fold_bottom<<<g_grid, 256, 0, g_st>>>(trees, ML, MR, et.k8, Fm);
      CK(cudaGetLastError());
      et.gram_rest(Fm, mu, 3, id, id + 1, o + 6, nullptr, 0, cmp(4));
      et.flush();
    }
    mark("bottom layer");
    leaf_layer(40 + kT);
    mark("leaf layer");
    ft::k_bits_point<<<1, 1, 0, g_st>>>(et.st, chd + 80, lk + kT, 80);
    bit_layer(80);
    mark("bit layer");
  }

private:
  T *Fm = nullptr;
  void mark(const char *phase) {
    CK(cudaGetLastError());
    if (phase_ends) phase_ends(phase);
  }
  // mode, for k_expand: 4 for a tree layer, 2 for the bit layer. Without effect unless `packed`.
  Cmp cmp(int mode) const {
    Cmp c;
    if (packed) c.ka = (int)nlo, c.lk = lk, c.mode = mode;
    return c;
  }
  // Places et's two fold buffers, m elements each (a multiple of 32), at the start of `room`.
  void s_at(T *room, size_t m) {
    et.S[0] = room, et.S[1] = room + m;
    et.s_bytes = m * sizeof(T);
  }
  // eqT[tl] = eq(rho[3 .. 3 + nb), tl); eqL[x] = eq(rho[low .. low + lk), (y | j)) for the stored
  // columns x = 32 y + j.
  void eq_tables(const Chal<F> *cd, int nb, int low) {
    EqJobs J;
    J.n = 0;
    J.add(3, nb, eqT);
    if (packed)
      J.add(low, 0, eqL, (int)nlo, lk);
    else
      J.add(low, lk, eqL);
    k_eqjobs<<<dim3(32, 2), 256, 0, g_st>>>(cd, J);
    CK(cudaGetLastError());
  }
  // Builds the levels kT - 1 .. 0 in `room` and sets Tc[]. The leaves are not stored.
  void build(T *room) {
    for (int l = 0; l < kT; l++) Tc[l] = room + nlo * (((size_t)1 << l) - 1);
    ft::k_pairs<<<g_grid * 2, 256, 0, g_st>>>(cnt, trees, Tc[kT - 1], Tc[kT - 2], Tc[kT - 3], Tc[kT - 4]);
    int l = kT - 4;
    for (; l >= 3; l -= 3) {
      const size_t n8 = nlo << (l - 3);
      k_up3<F, true, true><<<(int)std::min<size_t>((n8 + 255) / 256, g_grid * 4), 256, 0, g_st>>>(
          Tc[l], Tc[l - 1], Tc[l - 2], Tc[l - 3], n8);
    }
    for (; l >= 1; l--) {
      const size_t n = nlo << (l - 1);
      k_up1<F><<<(int)std::min<size_t>((n + 255) / 256, g_grid), 256, 0, g_st>>>(Tc[l], Tc[l - 1], n);
    }
    CK(cudaGetLastError());
  }
  // The bit layer. See pick() for what depends on multiplicity bits here.
  void bit_layer(int id) {
    et.flush();
    const int mu = lk + kT;
    T *o = et.out;
    et.out += sc_door::slots(sc_door::Form::EQ2, mu, 1, 2);
    eq_tables(chd + id, kT - 3, kT);
    pt_hi(k_layer_begin, 1, 32, 0, g_st, et.st, (const Chal<F> *)(chd + id), mu, id, 0, o, 0);
    tside::mask_door(1, M8, nullptr, nullptr, &et.st->alpha, kT - 3, (int)nlo, eqT, eqL, tside_room, et.st, chd + id, o,
                     et.k8, g_st);
    k_foldmaskN<1><<<g_grid, 256, 0, g_st>>>(M8, (u32)nlo, kT, et.k8, et.st, Fm);
    CK(cudaGetLastError());
    et.gram_rest(Fm, mu, 3, id, -1, o + 6, nullptr, 0, cmp(2), false, TAIL_BITS);
    et.flush();
  }
  // The leaf layer's R does not depend on j, which is bound last. Until then L contributes to a
  // round's sum only through
  //   Lambda(t, y) = sum over j of eq(the point's coordinates for j, j) beta(t | y | j).
  // So the Gram passes, which bind top bits of t, run on Lambda and R over (t | y), 32 times
  // smaller. Once these have at most 2^(lmid - 5) = 2^11 entries, k_leaf_planes computes the folded
  // L over (t | y | j) from the multiplicities: it then fits the tail kernel. mr counts the
  // variables of (t | y).
  void leaf_layer(int id) {
    et.flush();
    const int mu = lk + kT, mr = kT + plan.rK;
    Chal<F> *cd = chd + id;
    T *o = et.out;
    et.out += 1 + 2 * (size_t)mr + 5; // degree 2 in (t | y)'s variables, 1 in j's; no end value
    pt_hi(k_leaves_begin, 1, 32, 0, g_st, et.st, (const Chal<F> *)cd, mu, id, o);
    k_eqtabT<<<1, 256, 0, g_st>>>(cd, mr, 5, eqJ);
    ft::k_leaf_tables<<<g_grid, 256, 0, g_st>>>(cnt, trees, plan.rK, eqJ, Fm, Fm + ((size_t)1 << mr));
    CK(cudaGetLastError());
    // A Gram pass starts above 2^11 entries and leaves 2^9 or more, which is more than 2^rK, so all
    // bound variables are bits of t.
    const T *L = Fm, *R = Fm + ((size_t)1 << mr);
    const int bound = gram_passes(L, R, mr, lmid - 5, cd, o), ls = mr - bound;
    ft::k_eq_coins<<<32, 256, 0, g_st>>>(cd, bound, eqH);
    ft::k_leaf_planes<<<g_grid, 256, 0, g_st>>>(cnt, trees, plan.rK, ls, bound, eqH, R, et.X);
    CK(cudaGetLastError());
    const int i = et.ll.n++;
    et.ll.src[i] = et.X, et.ll.out[i] = o + 2 * bound;
    et.ll.mu[i] = mu, et.ll.j0[i] = bound, et.ll.id[i] = id, et.ll.idn[i] = -1;
    et.ll.begins[i] = 0, et.ll.kind[i] = TAIL_LEAVES;
    et.flush();
  }
  // Runs Gram passes on the stored tables L and R of 2^mr entries, with the weights
  // eq(cd->rho[0 .. mr), .), while they have more than 2^stop entries and at least 2^11: the loop
  // of Eng::gram_rest() for tables with all their columns, without its hand-over to the tail
  // kernel. Each pass fills the next 6 message slots of the sumcheck that k_leaves_begin began. The
  // kernels ignore the argument o: the running sumcheck's state (`Tr::p`) has the address of its
  // next slot. Leaves the folded tables in L and R and returns the number of rounds.
  int gram_passes(const T *&L, const T *&R, int mr, int stop, Chal<F> *cd, T *o) {
    auto more = [&](int l) {
      return l > stop && l - 3 >= 8;
    };
    auto low = [](int l) { // log2 of a chunk's entries
      return std::min(l - 3, 15);
    };
    const M3L *lo[8];
    const T *hi[8];
    EqJobs J;
    J.n = 0;
    size_t olo = 0, ohi = 0;
    for (int l = mr, p = 0; more(l); l -= 3, p++) {
      const int x = low(l), nh = l - 3 - x;
      if (p && x == low(l + 3))
        lo[p] = lo[p - 1];
      else {
        J.add(mr - x, x, nullptr, 0, 0, et.eqprep + olo);
        lo[p] = et.eqprep + olo, olo += (size_t)1 << x;
      }
      J.add(mr - l + 3, nh, et.eqhi + ohi);
      hi[p] = et.eqhi + ohi, ohi += (size_t)1 << nh;
    }
    if (J.n) k_eqjobs<<<dim3(128, J.n), 256, 0, g_st>>>(cd, J);
    int j = 0;
    for (int p = 0; more(mr - j); p++, j += 3) {
      const size_t SC = (size_t)1 << (mr - j - 3);
      et.gram_slab_in(p ? p & 1 : 2); // pass p writes its folded tables to S[p & 1]; the first pass's tables lie outside S
      et.lib_pass(L, R, SC, (size_t)1 << low(mr - j), lo[p], hi[p], cd, j, o + 2 * j, 0, 0, mr, g_st);
      T *to = et.S[p & 1];
      fold8_k3(L, R, SC, SC, et.k8, to, to + SC, g_st);
      L = to, R = to + SC;
    }
    return j;
  }
};
