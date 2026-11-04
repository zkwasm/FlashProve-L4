// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU helpers and kernels for the grand-product check (product test) of the lookup argument LKP,
// which is to show that every key occurs in a public table; blocks/iop/lkp.h states it. LKP
// compares products, which product trees compute: entry c of a level of n entries is the product
// of entries c and c + n of the level below. On the keys' side a leaf is alpha minus a key's value,
// for a challenge alpha.
// No function here checks its arguments at run time.
// F is the field's traits struct (FE in pt_field.h), and TT is its element type.
//
// A product layer is a sumcheck for sum_c eq(rho, c) L(c) R(c) over two arrays, L stored before R,
// where eq(rho, c) = prod_i (rho_i c_i + (1 - rho_i)(1 - c_i)) is the eq weight of c. rho[0] goes
// with the top bit of c. build3() is for rounds that bind the top bit first.
//
// Source for committing to nothing further for each key: Qu, Guo, Xuan, Liu, Zhang, "Celer: A
// Lookup Argument for Large-Scale Queries", CRYPTO 2026, Cryptology ePrint Archive 2026/1453,
// abstract (the number of the prover's commitments is sublinear in the number of keys). How Celer
// reaches this has not been compared with this argument. So "Source" stands here for that aim
// alone; for the way to it, Celer is cited only as related work.
// Source for keys that are not committed but defined as functions of committed data: Diamond,
// Posen, "Succinct Arguments over Towers of Binary Fields", Cryptology ePrint Archive 2023/1784,
// section 4.1 (virtual polynomials).
#pragma once
#include "hiding/pt_field.h"
#define TT typename F::T
// TPB: threads in a block of k_tails() (tails.cuh). MAXV: the largest number of coordinates of a
// layer's point. tail_layer() relies on TPB = 2^LT and on build3()'s first pyramid having LT = 8
// variables.
constexpr int TPB = 256, LT = 8, MAXV = 32; // LT: eq(rho, c) is the weight of c >> LT times that of c's low LT bits
// One layer's point rho and its round challenges r.
template <class F> struct Chal {
  TT rho[MAXV], r[MAXV];
};
// Binds a round's variable to its challenge r.
template <class F> HD TT fold1(const typename F::P &r, TT lo, TT hi) {
  return F::add(lo, F::mulp(r, F::sub(hi, lo)));
}
// The length of tail_layer()'s array e (tails.cuh): enough for nv <= 21, which build3() does not
// check.
// Public sources for the split tables of eq: see tails.cuh, tail_layer().
constexpr int E3S = 1088;
// Fills e with three eq pyramids over the last nv of the mu variables of rho: e[0..512) covers the
// last <= 8, e[512..1024) the <= 8 before those, e[1024..3072) the <= 10 before those. In a pyramid
// that starts at base, base[(1<<t)+x] is the eq weight of x over the pyramid's last t variables.
// A weight over several pyramids' variables is the product of one entry from each. This is specific
// to the sumcheck's weights eq(rho, c), which are products over the bits of c.
// All threads of the block must call build3() together.
// The loop form is public: the block-stride loop, shown in R. Van der Wijngaart and F. Oh,
// "Boosting Application Performance with GPU Memory Prefetching", NVIDIA Technical Blog, 2022,
// https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/.
template <class F> __device__ void build3(const TT *rho, int mu, int nv, TT *e) {
  const int tid = threadIdx.x, nth = blockDim.x;
  if (tid < 3) e[tid * 512 + 1] = F::one();
  __syncthreads();
  for (int t = 0; t < 10; t++) {
    for (int i = tid; i < (3 << t); i += nth) {
      int g = i >> t, x = i & ((1 << t) - 1), len = nv - 8 * g;
      if (g < 2 && len > 8) len = 8;
      if (t < len) {
        TT *b = e + g * 512;
        TT v = b[(1 << t) + x], hh = F::mul(v, rho[mu - 1 - 8 * g - t]);
        b[(2 << t) + (1 << t) + x] = hh;
        b[(2 << t) + x] = F::sub(v, hh);
      }
    }
    __syncthreads();
  }
}
// The eq weight of t over the nbh variables before the last 8, from build3()'s second and third
// pyramids. This is specific to the sumcheck's weights eq(rho, c).
template <class F> __device__ __forceinline__ TT wtile(const TT *e, size_t t, int nbh) {
  if (nbh <= 8) return e[512 + ((size_t)1 << nbh) + t];
  return F::mul(e[1024 + ((size_t)1 << (nbh - 8)) + (t >> 8)], e[512 + 256 + (t & 255)]);
}
// All 32 lanes of the warp must call shfl_down().
// An element wider than the shuffle's word is passed word by word. The idea is public: J. Luitjens,
// "Faster Parallel Reductions on Kepler", NVIDIA Technical Blog, 2014, "Shuffle On Down",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// the aim here, that the compiler can hold w, a few words for each thread, in registers:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <class F> __device__ __forceinline__ TT shfl_down(TT v, int o) {
  u64 w[sizeof(TT) / 8];
  memcpy(w, &v, sizeof(TT));
#pragma unroll
  for (int i = 0; i < (int)(sizeof(TT) / 8); i++) w[i] = __shfl_down_sync(~0u, w[i], o);
  TT r;
  memcpy(&r, w, sizeof(TT));
  return r;
}
// Block reduction in two levels: in each warp, then in the first warp. The idea is public:
// Luitjens, "Faster Parallel Reductions on Kepler" (cited at shfl_down()), "Block Reduce".
// Several sums go through one call. The idea is public: interleaved reductions: the same post,
// "Interleaving Multiple Reductions". Here an unrolled loop takes the sums in turn, and the
// interleaving is left to the compiler.
// NQ is a template parameter, so that the loops over it unroll. The idea is public, with the block
// size as the parameter: M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA Developer
// Technology, slides "Complete Unrolling", "Unrolling with Templates" and "Reduction #6: Completely
// Unrolled", https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// The compiler hint is public: #pragma unroll: Milakov (cited at shfl_down()).
// All threads of the block must call blk_red() together, and blockDim.x must be a multiple of 32.
template <class F, int NQ>
__device__ void blk_red(TT (&v)[NQ], TT *sh) { // v becomes the block's sums, valid in thread 0. sh has NQ * 32 entries
  int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    for (int o = 16; o; o >>= 1) v[q] = F::add(v[q], shfl_down<F>(v[q], o));
    if (l == 0) sh[q * 32 + w] = v[q];
  }
  __syncthreads();
  if (w == 0) {
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      TT x = l < nw ? sh[q * 32 + l] : F::zero();
      for (int o = 16; o; o >>= 1) x = F::add(x, shfl_down<F>(x, o));
      v[q] = x;
    }
  }
}
// Accumulates x * f[1] * ... * f[NT-1] into a.
template <class F, int NT> __device__ __forceinline__ void chain(typename F::Acc &a, TT x, const TT (&f)[NT]) {
#pragma unroll
  for (int k = 1; k < NT - 1; k++) x = F::mul(x, f[k]);
  F::mac(a, F::prep(x), f[NT - 1]);
}
// Loads data that another block may have written during the same launch.
template <class T> __device__ __forceinline__ T ldcg16(const T *p) {
  static_assert(sizeof(T) == 16, "an element of another size needs its own ldcg16");
  uint4 v = __ldcg(reinterpret_cast<const uint4 *>(p));
  T r;
  memcpy(&r, &v, 16);
  return r;
}
// The element is loaded with __ldcg(). The idea is public: ld.cg, a load that goes past the L1 cache,
// because L1 may still hold an older value of data in global memory, and that L2 alone caches:
// NVIDIA's PTX manual, "Cache Operators", https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
template <> __device__ __forceinline__ K3 ldcg16<K3>(const K3 *p) {
  const u64 *q = reinterpret_cast<const u64 *>(p);
  return {__ldcg(q), __ldcg(q + 1), __ldcg(q + 2)};
}
// In tails.cuh a round is narrow if each of its two arrays has at most 2^NL entries, and wide
// otherwise. Only a layer with a wide round has barriers over all blocks (tail_barrier()).
constexpr int NL = 9;
// One level of the product tree.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
template <class F> __global__ void k_up1(const TT *__restrict__ in, TT *__restrict__ o, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    st16(o + i, F::mul(ld16(in + i), ld16(in + i + n)));
}
// Three levels of the product tree in one launch: o1 (if W1), o2 (if W2) and o3.
// The loop form and the compiler hint are public: the grid-stride loop (cited at k_up1());
// #pragma unroll: NVIDIA, "CUDA Programming Guide", and Milakov (cited at shfl_down()).
template <class F, bool W1, bool W2>
__global__ void k_up3(const TT *__restrict__ in, TT *__restrict__ o1, TT *__restrict__ o2, TT *__restrict__ o3,
                      size_t n8) { // in has 8 * n8 entries
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8; i += (size_t)gridDim.x * blockDim.x) {
    TT a[4], b[2];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      a[k] = F::mul(ld16(in + i + k * n8), ld16(in + i + (k + 4) * n8));
      if (W1) st16(o1 + i + k * n8, a[k]);
    }
#pragma unroll
    for (int k = 0; k < 2; k++) {
      b[k] = F::mul(a[k], a[k + 2]);
      if (W2) st16(o2 + i + k * n8, b[k]);
    }
    st16(o3 + i, F::mul(b[0], b[1]));
  }
}
// U4 is for a range table, in which row k has the value k. A key is the number of its row, so that
// a leaf is alpha - key (Form<RangeSrc> in prover/hiding/forest_keys.cuh).
// U4::get() returns the product of (alpha - v) over four integers v < 2^16. It expands the product
// as alpha^4 - e1 alpha^3 + e2 alpha^2 - e3 alpha + e4, where e_d is the elementary symmetric
// polynomial of degree d in the v. e1 to e4 have 18, 35, 50, 64 bits.
// U4<FE> forwards to blk::lazy::FourKeys (prover/field/field_lazy.cuh), which takes one sum of
// products, reduced once, for each coordinate. The public source for such a sum is cited at the
// body of FourKeys: prover/field/lazy_gl64.cuh, "four keys".
template <class F> struct U4;
template <> struct U4<FE> {
  blk::lazy::FourKeys four;
  __device__ __forceinline__ void prep(Ext al) { four.prepare(e_of(al)); }
  __device__ __forceinline__ Ext get(u32 v0, u32 v1, u32 v2, u32 v3) const { return k3_of(four.get(v0, v1, v2, v3)); }
};
