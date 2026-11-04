// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The last few of the token sumcheck's lt rounds over t, run by a single kernel launch; each
// earlier round takes three. The sumcheck, over token positions t < 2^lt, is for
//   Cl = sum_t eq(P, t) sum_m co_m F_m[t] G_m[t].
// prover/hiding/token_sumcheck.cuh defines it, a round's polynomial q and the fold, and includes
// this file. Each round computes its sums over all terms m, sends its message and draws its
// challenge from the transcript (`Tr`, prover/hiding/transcript.cuh). The next round's tsc_pairs()
// folds Fv and Gv (all F_m, all G_m) with that challenge (`pend`); the kernel does the last fold.
// Call tsc_resident(); its requirements, listed at k_tsc_resident, are not checked.
//
// Public source: NVIDIA, CUDA C++ Programming Guide, "Cooperative Groups", "Grid Synchronization"
// (this_grid().sync() under a cooperative launch); "Warp Shuffle Functions" (sums within a warp).
// See also: NVIDIA, CUDA Programming Guide, "Device-Callable APIs and Intrinsics", "Cooperative
// Groups API", "Grid Synchronization",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html;
// "C/C++ Language Extensions", "Warp Shuffle Functions" (between threads that have not exited),
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
#pragma once
#include <cooperative_groups.h>
// Replaces each of v[0 .. NQ) by its sum over the calling block; the sums are valid in thread 0
// only. All threads of the block must call it, and the block must consist of whole warps (not
// checked). sh is shared memory of NQ * 32 elements, and the kernel must zero them at its start: a
// block with fewer than 32 warps leaves some unwritten.
// Public sources for the techniques of this routine.
// The idea is public: a block reduction by warp shuffles: J. Luitjens, "Faster Parallel Reductions
// on Kepler", NVIDIA Technical Blog, 2014, "Block Reduce",
// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// The idea is public: a count as a template parameter: NVIDIA, CUTLASS documentation, "Programming
// Guidelines", "Templates",
// https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/programming_guidelines.html.
// The compiler hint is public: #pragma unroll: NVIDIA, "CUDA Programming Guide", page "C/C++
// Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
template <int NQ> __device__ void tsc_blk_red(T (&v)[NQ], T *sh) {
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
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
      T x = sh[q * 32 + l];
      for (int o = 16; o; o >>= 1) x = F::add(x, shfl_down<F>(x, o));
      v[q] = x;
    }
  }
}
// Computes a round's NQ sums over all terms into v: q(0), qinf (q's X^2 coefficient) and, if
// NQ == 3, q(1). NQ is 3 in round 0 and 2 otherwise. Only thread 0 of block 0 gets the sums.
// The calling block's terms are m0 .. m1 - 1. All threads of all blocks must call it (block and
// grid barriers).
// Public sources for the techniques of this routine.
// A count as a template parameter: see tsc_blk_red.
// The loop form and the compiler hint are public: the block-stride loop: shown in a listing in
// NVIDIA, "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html;
// #pragma unroll: the page "C/C++ Language Extensions", cited at tsc_blk_red.
template <int NQ>
__device__ void tsc_sums(TscSh &S, T *Fv, T *Gv, const T *co, size_t stride, int m0, int m1, int nb, bool pend,
                               T *slots, T (&v)[NQ]) {
  const int u = threadIdx.x, bx = blockIdx.x, nbk = gridDim.x;
#pragma unroll
  for (int q = 0; q < NQ; q++) v[q] = F::zero();
  tsc_pairs<NQ>(u, TSC_LTPB, Fv, Gv, co, stride, m0, m1, nb, pend, S.r, S.e + ((size_t)1 << nb), v);
  tsc_blk_red<NQ>(v, S.red);
  if (nbk == 1) return;
  if (u == 0) {
#pragma unroll
    for (int q = 0; q < NQ; q++) slots[bx * 3 + q] = v[q];
  }
  cooperative_groups::this_grid().sync(); // every block's sums are in slots before block 0 reads them
  if (bx != 0) return;
#pragma unroll
  for (int q = 0; q < NQ; q++) v[q] = F::zero();
  for (int b = u; b < nbk; b += TSC_TPB) {
#pragma unroll
    for (int q = 0; q < NQ; q++) v[q] = F::add(v[q], ldcg16(slots + b * 3 + q));
  }
  tsc_blk_red<NQ>(v, S.red);
}
// k_tsc_resident runs rounds j0 .. lt - 1 (at most TSC_NV of them) and then the final fold. Round j's challenge is
// written to o->r[j]. The final fold leaves F_m(r) and G_m(r), r = o->r[0 .. lt), in
// Fv[m * stride] and Gv[m * stride]. Before round 0's message, Cl is stored in o->Cl and sent, and
// the sumcheck begun (msg, shape: see tsc_claim()). The transcript object, with the state of the
// running sumcheck in it, is read from *tr and written back at the end. Each block takes up to
// `per` of the NT terms. `slots` is working memory (tsc_resident_elements()).
// Required, not checked: 1 <= lt <= 32 (the size of TscSh::P and TscOut::r); at most TSC_NV rounds
// (TscSh::e has 2^TSC_NV elements); Fv and Gv folded with all earlier rounds' challenges; TSC_TPB
// threads per block; gridDim.x * per >= NT; a cooperative launch if gridDim.x > 1.
// The compiler hint and the loop form are public: __launch_bounds__: NVIDIA, "CUDA C++ Best
// Practices Guide", "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; the
// block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels", cited at tsc_sums.
__global__ void __launch_bounds__(TSC_TPB)
    k_tsc_resident(T *Fv, T *Gv, const T *co, int NT, int per, size_t stride, const T *P, int lt, int j0,
                   TscShape shape, Tr *tr, T *msg, TscOut *o, T *slots) {
  __shared__ TscSh S;
  const int u = threadIdx.x, m0 = blockIdx.x * per, m1 = min(NT, m0 + per);
  const bool sends = blockIdx.x == 0 && u == 0; // this thread alone holds the transcript object
  Tr t;
  if (sends) t = *tr;
  if (u < lt) S.P[u] = ldcg16(P + u);
  if (u == 0) {
    S.e[1] = F::one();
    S.r = F::zero();
  }
  if (u < 3 * 32) S.red[u] = F::zero(); // tsc_blk_red() requires these zeros
  __syncthreads();
  for (int nb = 1; nb < lt - j0; nb++) {
    for (int x = u; x < (1 << (nb - 1)); x += TSC_TPB) tsc_eq_step(S.e, nb, x, S.P, lt);
    __syncthreads();
  }
  for (int j = j0; j < lt; j++) {
    T q[3];
    if (j == 0)
      tsc_sums<3>(S, Fv, Gv, co, stride, m0, m1, lt - 1, false, slots, q);
    else
      tsc_sums<2>(S, Fv, Gv, co, stride, m0, m1, lt - j - 1, j > j0, slots, reinterpret_cast<T(&)[2]>(q));
    if (blockIdx.x == 0 && u == 0) {
      if (j == 0) {
        const T Cl = F::add(F::mul(F::sub(F::one(), S.P[0]), q[0]), F::mul(S.P[0], q[2]));
        o->Cl = Cl;
        tsc_claim(t, Cl, S.P, lt, shape, msg);
      }
      const T r = tsc_round(t, j, q);
      S.r = r;
      o->r[j] = r;
    }
    if (gridDim.x > 1) {
      cooperative_groups::this_grid().sync(); // block 0 has written o->r[j] before the other blocks read it
      if (blockIdx.x != 0 && u == 0) S.r = ldcg16(o->r + j);
    }
    __syncthreads();
  }
  const F::P rp = F::prep(S.r);
  for (int m = m0 + u; m < m1; m += TSC_TPB) {
    T *f = Fv + m * stride, *g = Gv + m * stride;
    f[0] = F::add(f[0], F::mulp(rp, F::sub(f[1], f[0])));
    g[0] = F::add(g[0], F::mulp(rp, F::sub(g[1], g[0])));
  }
  if (sends) *tr = t;
}
// Elements of `slots` needed by nbk > 1 blocks: each leaves up to three sums there for block 0.
static inline size_t tsc_resident_elements(int nbk) {
  return 3 * (size_t)nbk;
}
// Queues k_tsc_resident on `stream`. The caller must allocate `slots` and meet the kernel's
// requirements on lt, rounds, folds and `per`. If nbk > 1, the launch fails unless all blocks can
// be resident at once. On a failed launch, CK() (prover/common/common.h) calls exit(1).
static inline void tsc_resident(int nbk, T *Fv, T *Gv, const T *co, int NT, int per, size_t stride, const T *P_dev,
                                int lt, int j0, TscShape shape, Tr *tr, T *msg, TscOut *out, T *slots,
                                cudaStream_t stream) {
  if (nbk == 1) {
    k_tsc_resident<<<1, TSC_TPB, 0, stream>>>(Fv, Gv, co, NT, per, stride, P_dev, lt, j0, shape, tr, msg, out, slots);
  } else {
    void *a[] = {&Fv, &Gv, &co, &NT, &per, &stride, &P_dev, &lt, &j0, &shape, &tr, &msg, &out, &slots};
    CK(cudaLaunchCooperativeKernel((void *)k_tsc_resident, dim3(nbk), dim3(TSC_TPB), a, 0, stream));
  }
  CK(cudaGetLastError());
}
