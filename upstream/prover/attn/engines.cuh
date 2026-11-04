// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU kernels and host functions of the prover's attention sub-proof (a sub-proof is `fork` in
// identifiers), the argument for the attention part of the forward pass: SCD, a sumcheck whose
// degree varies by round (a_scd(), a_scd_end(); round kernels: prover/attn/engines_sums.cuh);
// helpers of SC3, a sumcheck of degree 3 (the rest: prover/attn/engines_s3.cuh); and tables, an
// int8 matrix times vectors over E (a_tables_n()).
// Preconditions are not checked in this file. T = Ext: an element of the extension field E,
// EDEG = 3 coordinates; F = FE: its arithmetic (prover/hiding/pt_field.h). Elements stay canonical,
// so a sum is the same in any order. Tr: the Fiat-Shamir transcript, a hash chain
// (prover/hiding/transcript.cuh). Pointers are device pointers unless noted. A point's coordinate 0
// belongs to the index's top bit.
#pragma once
#include "common/element_from_hash.h"
#include "hiding/product_test.cuh"
#include "hiding/place.cuh"
#include "common/row_pass_core.cuh"
#include "field/field_lazy.cuh"
namespace ga { // kernels and helpers in engines*.cuh; the host functions `a_*` are outside it
// NTH: the default number of threads per block. MAXBLK: eblk()'s largest grid. MAXV: see Eng::part.
constexpr int NTH = 256, MAXBLK = 1056, MAXV = 20;
// T and blk::E, the type of the library in blocks/ (blocks/field/field.h), have the same three
// coordinates.
HD blk::E as_e(const T &v) {
  return blk::E{{blk::F{coord(v, 0)}, blk::F{coord(v, 1)}, blk::F{coord(v, 2)}}};
}
HD T as_t(const blk::E &e) {
  T v;
  for (int i = 0; i < EDEG; i++) set_coord(v, i, e.c[i].v);
  return v;
}
// Exact sums of (element of E) x (small integer of either sign), from prover/field/lazy_signed.cuh.
// Limbs: an element cut into signed limbs, 24 bytes; it may be kept in shared memory. No code in
// this file reads the limbs. I4: such a sum, not yet reduced.
// Capacity, not checked: (calls of i4mac() between i4zero() and i4red()) x (largest |v|) <= 2^29.
// One reduction for a whole sum is delayed reduction: a public idea ([FFLAS]) that is specific to
// modular arithmetic: J.-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime
// Fields: the FFLAS and FFPACK packages", section 2.1.2 (in full at k_table()).
typedef blk::lazy::IPrep Limbs;
typedef blk::lazy::IAcc I4;
HD Limbs limbs_of(const T &x) {
  return blk::lazy::iprep(as_e(x));
}
HD I4 i4zero() {
  return blk::lazy::izero();
}
HD void i4mac(I4 &s, const Limbs &l, int v) {
  blk::lazy::imac(s, l, v);
}
HD T i4red(const I4 &s) {
  return as_t(blk::lazy::ired(s));
}
// Device scratch memory of this file's functions; the struct is in host memory. The functions that
// use it reserve it, except a_scd_end(), which uses what a_scd() reserved. Concurrent streams need
// one Eng each. A reserve() of more than cap bytes allocates a new buffer; the old one is neither
// copied nor freed. hz::FORK (prover/hiding/place.cuh): scratch memory of one run of a sub-proof.
// part: MAXBLK * MAXV elements: the blocks' partial sums, at most 18 per block (block_sums()), and
// in the last MAXV elements a round's totals (totals()).
struct Eng {
  T *part = nullptr;
  char *buf = nullptr;
  size_t cap = 0;
  void base() {
    if (!part) {
      hz::alloc(part, sizeof(T) * MAXBLK * MAXV, hz::FORK);
    }
  }
  void reserve(size_t bytes) {
    base();
    if (bytes > cap) {
      hz::alloc(buf, bytes, hz::FORK);
      cap = bytes;
    }
  }
};
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
__device__ __forceinline__ T wsum(T v) { // a warp's sum by shuffles, by the method of Luitjens's article (cited below)
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    T u;
#pragma unroll
    for (int i = 0; i < EDEG; i++) set_coord(u, i, __shfl_down_sync(~0u, coord(v, i), o));
    v = F::add(v, u);
  }
  return v;
}
// Sums v over the block; only thread 0 gets the total. Requires 32 | blockDim.x, and
// blockDim.x / 32 elements in sh.
__device__ __forceinline__ T bsum(T v, T *sh) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  v = wsum(v); // by the method of J. Luitjens, "Faster Parallel Reductions on Kepler" (NVIDIA Technical Blog, 2014)
  __syncthreads();
  if (lane == 0) sh[w] = v;
  __syncthreads();
  if (w == 0) {
    v = lane < nw ? sh[lane] : F::zero();
    v = wsum(v);
  }
  return v;
}
// Blocks for `items` items: at least 1, at most MAXBLK, so a kernel must loop.
static inline int eblk(size_t items, int th = NTH) {
  size_t b = (items + th - 1) / th;
  return (int)(b < 1 ? 1 : b > MAXBLK ? MAXBLK : b);
}
// Returns x0 + r (x1 - x0); r is passed as F::prep(r).
__device__ __forceinline__ T fold1(T x0, T x1, const F::P &r) {
  return F::add(x0, F::mulp(r, F::sub(x1, x0)));
}
// The kernel of a_coins(). Requires a launch of one block.
// The idea is public: one thread computes a value for the whole block, in shared memory: M. Harris,
// "Optimizing CUDA", NVIDIA Developer Technology, slides headed "S05: High Performance Computing
// with CUDA", slide "Take Advantage of Shared Memory" (the address runs over two lines),
// https://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/Cours09/CUDA/
// SC07_CUDA_5_Optimization_Harris.pdf.
// The loop form is public: the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming
// Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void k_coins(Tr *tr, int n, T *om) {
  __shared__ T drawn;
  if (threadIdx.x == 0) {
    Tr t = *tr;
    tr_point(t, Round::BAT_SIGMA, &drawn, 1);
    *tr = t;
  }
  __syncthreads();
  for (int x = threadIdx.x; x < n; x += blockDim.x) {
    T power = F::one(), square = drawn;
    for (int e = x; e; e >>= 1, square = F::mul(square, square))
      if (e & 1) power = F::mul(power, square);
    om[x] = power;
  }
}
// t[x] = eq(z, x) for every n-bit index x: the product over v < n of z[v] if bit n - 1 - v of x
// is 1, and of 1 - z[v] otherwise. Requires blocks of NTH threads.
// The compiler hint and the loop form are public: __launch_bounds__: NVIDIA, "CUDA C++ Best
// Practices Guide", "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; the
// block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels", cited at k_coins().
template <int LLO> __global__ void __launch_bounds__(NTH) k_eq(const T *z, int n, T *t) {
  __shared__ T lo[1 << LLO];
  const int nl = n < LLO ? n : LLO, nh = n - nl; // lo is the eq table of the last nl coordinates of z
  if (threadIdx.x == 0) lo[0] = F::one();
  __syncthreads();
  for (int v = 0; v < nl; v++) {
    const int s = 1 << v;
    const T zz = z[nh + v];
    T h[(1 << LLO) / NTH], c[(1 << LLO) / NTH];
    int q = 0;
    for (int x = threadIdx.x; x < s; x += NTH, q++) {
      c[q] = lo[x];
      h[q] = F::mul(c[q], zz);
    }
    __syncthreads();
    q = 0;
    for (int x = threadIdx.x; x < s; x += NTH, q++) {
      lo[2 * x + 1] = h[q];
      lo[2 * x] = F::sub(c[q], h[q]);
    }
    __syncthreads();
  }
  for (size_t hi = blockIdx.x; hi < ((size_t)1 << nh); hi += gridDim.x) {
    T p = F::one();
    for (int v = 0; v < nh; v++) {
      const T zz = z[v];
      p = F::mul(p, ((hi >> (nh - 1 - v)) & 1) ? zz : F::sub(F::one(), zz));
    }
    const F::P pp = F::prep(p);
    for (int x = threadIdx.x; x < (1 << nl); x += NTH) st16(t + (hi << nl) + x, F::mulp(pp, lo[x]));
  }
}
// k_table()'s pairs (phi, out), at most nine.
struct PT9 {
  const T *phi[9];
  T *out[9];
};
// out[e] = sum_y cells[e * L + y] * phi[y] for e < rows; blockIdx.y selects phi and out. A thread's
// sum has at most ceil(L / NTH) terms with |cell| <= 2^7: within I4's capacity if L / NTH <= 2^22.
// Requires blocks of NTH threads.
// The idea is public: thread coarsening (here R rows at a time): Wen-mei W. Hwu, David Kirk,
// "Advanced Algorithmic Techniques for GPUs, Lecture 4: Thread Coarsening and more on
// Tiling/Blocking", PASI Summer School, Chile, January 5-7, 2011, slide "Outline of Technique",
// https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf.
// Delayed reduction. This is a public idea that is specific to modular arithmetic: delayed
// division: J.-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields:
// the FFLAS and FFPACK packages", ACM TOMS 35(3), 2008, arXiv:cs/0601133, section 2.1.2,
// https://arxiv.org/pdf/cs/0601133.
// The compiler hints and the loop form are public: __launch_bounds__: Best Practices Guide,
// "Occupancy", cited at k_eq(); the block-stride loop: CUDA Programming Guide, "Writing SIMT
// Kernels", cited at k_coins(); #pragma unroll: CUDA Programming Guide, "C/C++ Language
// Extensions", cited at wsum().
template <int R>
__global__ void __launch_bounds__(NTH) k_table(const int8_t *__restrict__ cells, size_t rows, size_t L, PT9 p) {
  __shared__ T sh[32];
  const T *__restrict__ phi = p.phi[blockIdx.y];
  T *out = p.out[blockIdx.y];
  for (size_t e0 = (size_t)blockIdx.x * R; e0 < rows; e0 += (size_t)gridDim.x * R) {
    I4 a[R];
#pragma unroll
    for (int q = 0; q < R; q++) a[q] = i4zero();
    for (size_t y = threadIdx.x; y < L; y += NTH) {
      const Limbs l = limbs_of(ld16(phi + y));
#pragma unroll
      for (int q = 0; q < R; q++)
        if (e0 + q < rows) i4mac(a[q], l, cells[(e0 + q) * L + y]);
    }
#pragma unroll
    for (int q = 0; q < R; q++) {
      const T s = bsum(i4red(a[q]), sh);
      if (threadIdx.x == 0 && e0 + q < rows) out[e0 + q] = s;
    }
  }
}
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_addto(T *o, const T *x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    o[i] = F::add(o[i], x[i]);
}
} // namespace ga
// Gets one batching challenge c (`coins` in the name) from the transcript *tr and writes its
// powers: om[x] = c^x, x < n.
static inline void a_coins(Tr *tr, int n, T *om, cudaStream_t st = 0) {
  ga::k_coins<<<1, 128, 0, st>>>(tr, n, om);
}
// t[x] = eq(z, x), x < 2^n.
static inline void a_eq(const T *z, int n, T *t, cudaStream_t st = 0) {
  const int nh = n > 10 ? n - 10 : 0;
  ga::k_eq<10><<<(int)std::min<size_t>((size_t)1 << nh, 1056), ga::NTH, 0, st>>>(z, n, t);
}
// Bytes of scratch memory that a_tables_x<nphi>() reserves where it uses row_core_i8().
static inline size_t a_table_scratch(int nphi, size_t rows, size_t L) {
  const size_t K = L < 65536 ? L : 65536;
  return row_core_scratch(nphi, (int)K, (int)rows) + 4096 + (L > 65536 ? (size_t)nphi * rows * sizeof(T) : 0);
}
#include "attn/engines_sums.cuh"
namespace ga {
// SCD indices: layer lam < Lam of the layer group (the Lam layers that one attention sub-proof
// covers), key-value head g < 2^lgk, its query head a < HA (h = g HA + a), tokens i, j < 2^lt.
// Arrays: see a_scd(). P[q] = om4[lam HA + a] eri[i] erg[g] for row
// q = (lam HA 2^lgk + h) 2^lt + i, the row order of ix. Copies KC to Kw for k_scd_kfold().
// The loop form is public: the grid-stride loop (cited at k_addto()).
__global__ void k_scd_init(int lt, int lgk, int HA, int Lam, const T *KC, const T *eri, const T *erg, const T *om4,
                           T *P, T *Kw) {
  const size_t Tn = (size_t)1 << lt, R = ((size_t)Lam * HA << lgk) << lt, NK = ((size_t)Lam << lgk) << lt;
  for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x; q < R; q += (size_t)gridDim.x * blockDim.x) {
    const size_t i = q & (Tn - 1), hh = q >> lt;
    const int h = (int)(hh % ((size_t)HA << lgk)), lam = (int)(hh / ((size_t)HA << lgk)), g = h / HA, a = h % HA;
    P[q] = F::mul(om4[lam * HA + a], F::mul(eri[i], erg[g]));
    if (q < NK) Kw[q] = KC[q];
  }
}
// Folds Kw in place over bit kap1 of j by that bit's challenge *rp. Kw has nlg = Lam 2^lgk blocks
// (lam, g), which stay 2^lt elements apart.
// The loop form is public: the grid-stride loop (cited at k_addto()).
__global__ void k_scd_kfold(int lt, int kap1, size_t nlg, T *Kw, const T *rp) {
  const F::P rr = F::prep(*rp);
  const size_t h = (size_t)1 << kap1, n = nlg << kap1;
  for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x; q < n; q += (size_t)gridDim.x * blockDim.x) {
    const size_t o = ((q >> kap1) << lt) + (q & (h - 1));
    Kw[o] = fold1(Kw[o], Kw[o + h], rr);
  }
}
// For each pair `in` = lam HA + a, nf = lt + 2 factor tables: f at FT[(in nf + f) GT + g 2^lt + i].
// f = 0: om4[in] eri[i] erg[g]. f = 1: Kw[(lam 2^lgk + g) 2^lt], which is block (lam, g) of KC
// evaluated at pj. f = 2 + k: eq1(bit k of the row's ix, pj[lt - 1 - k]), where
// eq1(b, t) = b ? t : 1 - t, as in k_scd_j().
// Public source: Q. Dao, J. Thaler, "More Optimizations to Sum-Check Proving", ePrint 2024/1210,
// section 2.1 "Recap: Linear-Time Prover for Sum-check", https://eprint.iacr.org/2024/1210 (one
// array of full size for each factor, all folded alike).
// Public source: NumPy documentation, numpy.meshgrid,
// https://numpy.org/doc/2.0/reference/generated/numpy.meshgrid.html (a factor that depends on part
// of the index is repeated).
// Here there is one such set of arrays for each pair (lam, a).
// The loop form is public: the grid-stride loop (cited at k_addto()).
__global__ void k_scd_tabs(int lt, int lgk, int HA, int Lam, const int *__restrict__ ix, const T *Kw, const T *eri,
                           const T *erg, const T *om4, const T *pj, T *FT) {
  const size_t GT = (size_t)1 << (lt + lgk), n = (size_t)Lam * HA * GT;
  const int nf = lt + 2;
  for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x; q < n; q += (size_t)gridDim.x * blockDim.x) {
    const size_t X = q & (GT - 1);
    const int in = (int)(q >> (lt + lgk)), lam = in / HA, a = in % HA, g = (int)(X >> lt),
              i = (int)(X & (((size_t)1 << lt) - 1));
    const int x = ix[((((size_t)lam * (HA << lgk)) + g * HA + a) << lt) + i];
    T *o = FT + (size_t)in * nf * GT + X;
    o[0] = F::mul(om4[in], F::mul(eri[i], erg[g]));
    o[GT] = Kw[(((size_t)lam << lgk) + g) << lt];
    for (int kp = 0; kp < lt; kp++) {
      const T y = pj[lt - 1 - kp];
      o[(size_t)(2 + kp) * GT] = ((x >> kp) & 1) ? y : F::sub(F::one(), y);
    }
  }
}
// KCs[lam]: factor table f = 1 of the pair (lam, a = 0), folded by the last challenge *rp.
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels",
// cited at k_coins().
__global__ void k_scd_end(int Lam, int HA, int nf, size_t GT, const T *FT, const T *rp, T *KCs) {
  const F::P rr = F::prep(*rp);
  for (int lam = threadIdx.x; lam < Lam; lam += blockDim.x) {
    const T *o = FT + ((size_t)lam * HA * nf + 1) * GT;
    KCs[lam] = fold1(o[0], o[1], rr);
  }
}
// The degree of SCD's round number q: 2 in a bit of j, lt + 2 in a bit of g, lt + 1 in a bit of i.
HD constexpr int scd_degree(int lt, int lgk, int q) {
  return q < lt ? 2 : q < lt + lgk ? lt + 2 : lt + 1;
}
// Begins SCD: sends the sum of the hiding polynomial (a random polynomial added to the summand) and
// gets the challenge that joins it to the claim, sum_b om4[b] Bs[b]. SCD sends everything through
// sc_door (prover/hiding/sumcheck_door.cuh), which does the masking. slots: SCD's first slot in the
// sub-proof's message array; a slot holds one sent element. lt + 1: the summand's degree in Y (see
// k_scd_in_y()). Form::NODES: a round's polynomial g, of degree d, is given by g(0), g(2), ..,
// g(d); the running claim is g(0) + g(1). No equality point is used, so the function that would
// give its coordinates returns 0. Launch: one thread.
__global__ void k_scd_begin(int lt, int lgk, int nbt, const T *om4, const T *Bs, Tr *tr, T *slots) {
  T claim = F::zero();
  for (int b = 0; b < nbt; b++) claim = F::add(claim, F::mul(om4[b], Bs[b]));
  Tr t = *tr;
  (void)sc_door::begin<sc_door::Form::NODES>(
      sc_door::Site::ATTENTION_SCD, t, slots, 2 * lt + lgk, [](int) { return F::zero(); }, claim, lt + 1,
      [=](int q) { return scd_degree(lt, lgk, q); });
  *tr = t;
}
// With y = pj[lt - 1 - k]: cy[k] = 2 y - 1 and cy[lt + k] = 1 - y for k < lt, so that
// eq1(u, y) = cy[k] u + cy[lt + k]; cy[2 lt] = eq(ri, pi) eq(rg, pg). Launch: one thread.
__global__ void k_scd_chain(const T *pj, int lt, const T *ri, const T *pi, const T *rg, const T *pg, int lg, T *cy) {
  for (int k = 0; k < lt; k++) {
    const T y = pj[lt - 1 - k];
    cy[k] = F::sub(F::add(y, y), F::one());
    cy[lt + k] = F::sub(F::one(), y);
  }
  T e = F::one();
  auto q = [&](T a, T b) {
    const T ab = F::mul(a, b);
    return F::sub(F::add(F::add(ab, ab), F::one()), F::add(a, b));
  };
  for (int i = 0; i < lt; i++) e = F::mul(e, q(ri[i], pi[i]));
  for (int i = 0; i < lg; i++) e = F::mul(e, q(rg[i], pg[i]));
  cy[2 * lt] = e;
}
// The largest lt that SCD's kernels support: a round has degree up to lt + 2, and the round in Y
// has lt + 1 scalars, which pass through totals(), the last MAXV elements of Eng::part.
constexpr int SCD_MOST_LT = 16;
static_assert(SCD_MOST_LT + 2 <= psc::MAX_DEGREE && SCD_MOST_LT + 1 <= MAXV, "the door's and the totals' room");
// SCD's round in Y, the further variable defined in blocks/iop/psc.h: end value number n (a table's
// value at the challenge point) enters as value + c_n Y, c_n being its inner mask, the mask that
// sc_door::inner_mask() returns for it. With lam = b / HA and m = b lt + k, the summand is
//   cy[2 lt] sum_b om4[b] (KCs[lam] + c_lam Y) prod_k (cy[k] (ibs[m] + c_(Lam + m) Y) + cy[lt + k])
// at the challenge point, a polynomial of degree lt + 1 in Y. in_y[t - 1] gets its coefficient of
// Y^t, t = 1 .. lt + 1. Requires one block of 64 threads and lt <= SCD_MOST_LT.
// The block's sums are formed by bsum() (above); its public source is named there.
// The loop form is public: the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels",
// cited at k_coins().
__global__ void __launch_bounds__(64)
    k_scd_in_y(int lt, int HA, int Lam, int nbt, const T *KCs, const T *ibs, const T *om4, const T *cy, const Tr *tr,
               T *in_y) {
  __shared__ T sh[32];
  T sum[SCD_MOST_LT + 2];
  for (int t = 0; t <= lt + 1; t++) sum[t] = F::zero();
  for (int b = threadIdx.x; b < nbt; b += 64) {
    T p[SCD_MOST_LT + 2]; // the product so far, by powers of Y
    p[0] = F::mul(om4[b], KCs[b / HA]);
    p[1] = F::mul(om4[b], sc_door::inner_mask(*tr, b / HA));
    for (int k = 0; k < lt; k++) {
      const T e0 = F::add(F::mul(cy[k], ibs[(size_t)b * lt + k]), cy[lt + k]),
              e1 = F::mul(cy[k], sc_door::inner_mask(*tr, Lam + b * lt + k));
      p[k + 2] = F::mul(p[k + 1], e1);
      for (int t = k + 1; t >= 1; t--) p[t] = F::add(F::mul(p[t], e0), F::mul(p[t - 1], e1));
      p[0] = F::mul(p[0], e0);
    }
    for (int t = 1; t <= lt + 1; t++) sum[t] = F::add(sum[t], p[t]);
  }
  for (int t = 1; t <= lt + 1; t++) {
    const T s = bsum(sum[t], sh);
    if (threadIdx.x == 0) in_y[t - 1] = F::mul(cy[2 * lt], s);
    __syncthreads();
  }
}
// Sends the round in Y and gets its challenge, which is kept in *tr. Launch: one thread.
__global__ void k_scd_further(const T *in_y, Tr *tr) {
  Tr t = *tr;
  (void)sc_door::further(sc_door::Site::ATTENTION_SCD, t, in_y);
  *tr = t;
}
// Masks the end values and stores them in the message array: KCs[lam] as number lam, ibs[x] as
// number Lam + x, x < n. k_scd_end_send() absorbs them.
// The loop form is public: the grid-stride loop (cited at k_addto()).
__global__ void k_scd_ends(int Lam, size_t n, const T *KCs, const T *ibs, const Tr *tr) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < Lam + n; x += (size_t)gridDim.x * blockDim.x)
    sc_door::end(sc_door::Site::ATTENTION_SCD, *tr, x < (size_t)Lam ? KCs[x] : ibs[x - Lam], (int)x);
}
// Absorbs the n end values as one message. Launch: one thread.
__global__ void k_scd_end_send(Tr *tr, int n) {
  Tr t = *tr;
  sc_door::ends_sent(t, n);
  *tr = t;
}
} // namespace ga
// Number of SCD's slots in the message array, in sending order: 1 for the hiding polynomial's sum;
// 2 per round over j, lt + 2 per round over g, lt + 1 per round over i; lt + 1 for the round in Y;
// Lam + nbt lt for the end values (see a_scd_end()). Requires 2 lt + lgk <= 64.
static inline size_t a_scd_nout(int lt, int lgk, int Lam, int nbt) {
  int degree[2 * ga::SCD_MOST_LT + 32];
  for (int q = 0; q < 2 * lt + lgk; q++) degree[q] = ga::scd_degree(lt, lgk, q);
  return sc_door::slots_degrees(degree, 2 * lt + lgk, (size_t)Lam + (size_t)nbt * lt, lt + 1);
}
// Bytes of scratch memory that a_scd() reserves.
// Public sources for the factor tables (the term with lt + 2): see k_scd_tabs().
static inline size_t a_scd_scratch(int lt, int lgk, int HA, int Lam) {
  const size_t GT = (size_t)1 << (lt + lgk);
  return sizeof(T) * ((size_t)Lam * HA * GT + (size_t)Lam * GT + (size_t)Lam * HA * (lt + 2) * GT) + 1024;
}
// Runs SCD's rounds. SCD sums, over j, g, i and pairs (lam, a), om4[lam HA + a] eri[i] erg[g]
// KC[lam][g][j] eq(j, ix[lam][g HA + a][i]). ix: each row's first token of largest score. KC: the
// keys, their channels combined. eri, erg: eq tables (see a_scd_end()). om4: from a_coins(). Bs:
// the preceding SC3's end values B. out: a_scd_nout() slots. rp: 2 lt + lgk challenges,
// (pj | pg | pi), each top bit first. KCs[lam]: KC[lam] at (pg, pj). a_scd_end() must follow.
// Requires lt <= SCD_MOST_LT = 16: a round's degree, at most lt + 2, must not exceed 18, the
// largest degree bound (template parameter MD) for which scd_gi() (engines_sums.cuh) is
// instantiated below.
static inline void a_scd(int lt, int lgk, int HA, int Lam, const int *ix, const T *KC, const T *eri, const T *erg,
                         const T *om4, const T *Bs, Tr *tr, T *out, T *rp, T *KCs, cudaStream_t st, ga::Eng *e) {
  using namespace ga;
  e->reserve(a_scd_scratch(lt, lgk, HA, Lam));
  const size_t GT = (size_t)1 << (lt + lgk), R = (size_t)Lam * HA * GT;
  T *P = (T *)e->buf, *Kw = P + R, *FT = Kw + (size_t)Lam * GT;
  const int nbt = Lam * HA, nf = lt + 2;
  k_scd_begin<<<1, 1, 0, st>>>(lt, lgk, nbt, om4, Bs, tr, out);
  k_scd_init<<<eblk(R), NTH, 0, st>>>(lt, lgk, HA, Lam, KC, eri, erg, om4, P, Kw);
  for (int s = 0; s < lt; s++) {
    const int kap = lt - 1 - s;
    if (s == 0)
      k_scd_j<false><<<eblk(R), NTH, 0, st>>>(lt, lgk, HA, R, kap, ix, P, Kw, nullptr, e->part);
    else {
      k_scd_kfold<<<eblk(((size_t)Lam << lgk) << (kap + 1)), NTH, 0, st>>>(lt, kap + 1, (size_t)Lam << lgk, Kw,
                                                                           rp + s - 1);
      k_scd_j<true><<<eblk(R), NTH, 0, st>>>(lt, lgk, HA, R, kap, ix, P, Kw, rp + s - 1, e->part);
    }
    sum_parts(e, dim3(eblk(R)), 2, 2, st);
    k_send_chal<<<1, 1, 0, st>>>(totals(e), 2, tr, rp + s);
  }
  k_scd_kfold<<<eblk((size_t)Lam << lgk), NTH, 0, st>>>(lt, 0, (size_t)Lam << lgk, Kw, rp + lt - 1);
  k_scd_tabs<<<eblk(R), NTH, 0, st>>>(lt, lgk, HA, Lam, ix, Kw, eri, erg, om4, rp, FT);
  for (int s = 0; s < lgk + lt; s++) {
    const int lh = lgk + lt - 1 - s, d = s < lgk ? lt + 2 : lt + 1, g = eblk((size_t)nbt << lh, 128);
    const T *rq = rp + lt + s - 1;
    T *rn = rp + lt + s;
    if (d <= 8)
      scd_gi<8>(s > 0, g, lh, nbt, nf, d, GT, FT, rq, e, tr, rn, st);
    else if (d <= 13)
      scd_gi<13>(s > 0, g, lh, nbt, nf, d, GT, FT, rq, e, tr, rn, st);
    else
      scd_gi<18>(s > 0, g, lh, nbt, nf, d, GT, FT, rq, e, tr, rn, st);
  }
  k_scd_end<<<1, 64, 0, st>>>(Lam, HA, nf, GT, FT, rp + 2 * lt + lgk - 1, KCs);
}
// Ends SCD: the round in Y, then the end values KCs[lam] and ibs[b lt + k], which is bit k of pair
// b's ix at (pg, pi). Nothing may use the transcript between a_scd() and this call. ri, rg: the
// points of a_scd()'s eq tables eri, erg. cy: 2 lt + 1 elements of scratch. rp, e: as in a_scd().
static inline void a_scd_end(int lt, int lgk, int HA, int Lam, const T *KCs, const T *ibs, const T *om4, const T *ri,
                             const T *rg, const T *rp, T *cy, Tr *tr, cudaStream_t st, ga::Eng *e) {
  using namespace ga;
  const int nbt = Lam * HA;
  const size_t n = (size_t)nbt * lt;
  k_scd_chain<<<1, 1, 0, st>>>(rp, lt, ri, rp + lt + lgk, rg, rp + lt, lgk, cy);
  k_scd_in_y<<<1, 64, 0, st>>>(lt, HA, Lam, nbt, KCs, ibs, om4, cy, tr, totals(e));
  k_scd_further<<<1, 1, 0, st>>>(totals(e), tr);
  k_scd_ends<<<eblk(Lam + n), NTH, 0, st>>>(Lam, n, KCs, ibs, tr);
  k_scd_end_send<<<1, 1, 0, st>>>(tr, (int)(Lam + n));
}
// Whether a_tables_x() uses row_core_i8() (prover/common/row_pass_core.cuh) rather than k_table().
static inline bool a_table_rowpass_ok(const int8_t *cells, size_t rows, size_t L) {
  return L % 1024 == 0 && (L <= 65536 || L % 65536 == 0) && ((size_t)cells & 15) == 0 && rows < ((size_t)1 << 31);
}
// Computes NPHI <= 9 tables over the same cells: out[v][e] = sum_y phi[v][y] cells[e L + y],
// e < ncells / L. phi and out are host arrays of NPHI device pointers. row_core_i8() takes at most
// 65,536 columns, so a longer row goes in slices, whose results k_addto() adds.
template <int NPHI>
static inline void a_tables_x(const int8_t *cells, size_t ncells, const T *const *phi, size_t L, T *const *out,
                              cudaStream_t st, ga::Eng *e) {
  using namespace ga;
  const size_t rows = ncells / L;
  if (!a_table_rowpass_ok(cells, rows, L)) {
    PT9 p;
    for (int v = 0; v < 9; v++) {
      p.phi[v] = v < NPHI ? phi[v] : nullptr;
      p.out[v] = v < NPHI ? out[v] : nullptr;
    }
    k_table<4><<<dim3((unsigned)std::min<size_t>((rows + 3) / 4, 1 << 16), NPHI), NTH, 0, st>>>(cells, rows, L, p);
    return;
  }
  e->reserve(a_table_scratch(NPHI, rows, L));
  const size_t K = L < 65536 ? L : 65536;
  T *tmp = (T *)e->buf;
  char *rs = e->buf + (L > 65536 ? ((size_t)NPHI * rows * sizeof(T) + 255) / 256 * 256 : 0);
  for (size_t s = 0; s * K < L; s++) {
    const T *ph[NPHI];
    T *o[NPHI];
    for (int v = 0; v < NPHI; v++) {
      ph[v] = phi[v] + s * K;
      o[v] = s ? tmp + v * rows : out[v];
    }
    row_core_i8<NPHI>(cells + s * K, L, (int)rows, (int)K, ph, o, rs, st);
    if (s)
      for (int v = 0; v < NPHI; v++) k_addto<<<eblk(rows), NTH, 0, st>>>(out[v], o[v], rows);
  }
}
// Computes n tables over the same cells, at most nine per call of a_tables_x(); phi, out: as there.
static inline void a_tables_n(int n, const int8_t *cells, size_t ncells, const T *const *phi, size_t L, T *const *out,
                              cudaStream_t st, ga::Eng *e) {
  e->base();
  while (n > 0) {
    int c = n >= 9 ? 9 : n;
    switch (c) {
    case 9:
      a_tables_x<9>(cells, ncells, phi, L, out, st, e);
      break;
    case 8:
      a_tables_x<8>(cells, ncells, phi, L, out, st, e);
      break;
    case 7:
      a_tables_x<7>(cells, ncells, phi, L, out, st, e);
      break;
    case 6:
      a_tables_x<6>(cells, ncells, phi, L, out, st, e);
      break;
    case 5:
      a_tables_x<5>(cells, ncells, phi, L, out, st, e);
      break;
    case 4:
      a_tables_x<4>(cells, ncells, phi, L, out, st, e);
      break;
    case 3:
      a_tables_x<3>(cells, ncells, phi, L, out, st, e);
      break;
    case 2:
      a_tables_x<2>(cells, ncells, phi, L, out, st, e);
      break;
    default:
      a_tables_x<1>(cells, ncells, phi, L, out, st, e);
    }
    phi += c;
    out += c;
    n -= c;
  }
}
namespace ga {
// SC3 sums W_b(x) A_b(x) B_b(x) over b < nb, x < 2^nv. W_b(x) = wb[b] W0[x >> nlow], so all b share
// W0 (2^(nv - nlow) entries). A functor Src (prover/attn/close.cuh) reads A_b, B_b: T A(b, x),
// T B(b, x), or int Ai(b, x), Bi(b, x) if Src::INT. If also Src::ROW, Ar(b, xt), Br(b, xt) point to
// the int16 row x >> nlow == xt.
// s3_w(): round j's weights w0, w1 (variable = 0, 1) at entry xt; nvW = nv - nlow. If 0 < j <= nvW,
// it folds Wp, round j - 1's table, by rr = F::prep(that round's challenge); if j > nvW, it copies
// Wp[0]. With wr set (by one thread per entry) and j > 0, it writes Wn, round j's table.
__device__ __forceinline__ void s3_w(int j, int nvW, const T *Wp, T *Wn, size_t xt, const F::P &rr, bool wr, T &w0,
                                     T &w1) {
  if (j == 0) {
    if (nvW == 0)
      w0 = w1 = Wp[0];
    else {
      w0 = ld16(Wp + xt);
      w1 = ld16(Wp + xt + ((size_t)1 << (nvW - 1)));
    }
  } else if (j < nvW) {
    const size_t Ht = (size_t)1 << (nvW - 1 - j);
    w0 = fold1(ld16(Wp + xt), ld16(Wp + xt + 2 * Ht), rr);
    w1 = fold1(ld16(Wp + xt + Ht), ld16(Wp + xt + 3 * Ht), rr);
    if (wr) {
      st16(Wn + xt, w0);
      st16(Wn + xt + Ht, w1);
    }
  } else if (j == nvW) {
    w0 = w1 = fold1(Wp[0], Wp[1], rr);
    if (wr) Wn[0] = w0;
  } else {
    w0 = w1 = Wp[0];
    if (wr) Wn[0] = w0;
  }
}
// Returns x0 + r (x1 - x0) for integer endpoints x0, x1.
__device__ __forceinline__ T foldi(long long x0, long long x1, T r) {
  T y = F::mulb(r, F::fromi(x1 - x0));
  set_coord(y, 0, gf_add(coord(y, 0), F::fromi(x0)));
  return y;
}
// Adds to s0 and sm the values at 0 and at -1 of the product of the linear functions a and b, and
// to sd the product's quadratic coefficient. Here a(0) = a0 and a(1) = a1, and likewise for b.
__device__ __forceinline__ void s3_acc(T a0, T a1, T b0, T b1, T &s0, T &sm, T &sd) {
  const T da = F::sub(a1, a0), db = F::sub(b1, b0);
  s0 = F::add(s0, F::mul(a0, b0));
  sm = F::add(sm, F::mul(F::sub(a0, da), F::sub(b0, db)));
  sd = F::add(sd, F::mul(da, db));
}
// log2 of the number of index values that a thread takes at a time in SC3's round j: 0, 1 or 2.
static inline int s3_le(int j, int nv, int nlow) {
  const int lh = nv - 1 - j;
  int le = j < nv - nlow ? nlow - 5 : lh - 5;
  return le < 0 ? 0 : le > 2 ? 2 : le;
}
// An SC3 round kernel's grid, y = b: at most MAXBLK blocks if nb <= MAXBLK.
static inline dim3 s3_grid(int lh, int le, int nb) {
  size_t g = (((size_t)1 << lh >> le) + NTH - 1) / NTH;
  const size_t cap = (size_t)(MAXBLK / nb) < 1 ? 1 : MAXBLK / nb;
  return dim3((unsigned)(g < 1 ? 1 : g > cap ? cap : g), nb);
}
} // namespace ga
// SC3's device scratch memory, which the caller of a_s3_run() (engines_s3.cuh) allocates.
struct S3Work {
  // A, B: nb * 2^(nv-1) elements each.
  // W1, W2: W0's folded tables (s3_w()'s Wp, Wn): round 1's in W1, round 2's in W2, then in turn;
  // 2^(nv-nlow-1) and 2^(nv-nlow-2) elements, at least 2 each.
  T *A, *B, *W1, *W2;
};
#include "attn/engines_s3.cuh"
