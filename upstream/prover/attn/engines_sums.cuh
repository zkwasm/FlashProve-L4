// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Code that the attention sub-proof's sumchecks share to add up a round's values, and k_scd_j and
// k_scd_gi, two round kernels of SCD, the sumcheck over a token j, a key-value head g and a token
// i (a_scd() in prover/attn/engines.cuh, which includes this file after its helpers: wsum(),
// bsum(), fold1(), `Eng`). A sub-proof is the argument for one part of the forward pass. A row of
// SCD is a triple (layer lam, query head, token i); ix[row] is its token j. T is an extension-field
// element; F does T's arithmetic; `Tr` (prover/hiding/transcript.cuh) is the transcript.
// A round's sums take three kernel launches, in order on one stream: the round's kernel, whose last
// step, block_sums(), writes its blocks' sums to the scratch array `Eng::part`; k_sum_parts, which
// adds them; and a one-thread kernel that reads the totals (k_send_chal for SCD). The last kernel,
// k_wsum, belongs to the kernels of prover/attn/close.cuh (namespace gc) and sums the same way.
// Public source: partial sums per block, then a second launch that adds them: M. Harris,
// "Optimizing Parallel Reduction in CUDA" (NVIDIA),
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// See also the CUDA samples' "reduction". Public source: a warp's sum by shuffles, a block's from
// its warps' sums in shared memory: J. Luitjens, "Faster Parallel Reductions on Kepler" (NVIDIA
// Technical Blog, 2014), https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/.
// Public source: a warp's sum, in wsum(): NVIDIA, CUDA C++ Programming Guide, "Warp Shuffle
// Functions". See also: __shfl_down_sync() copies from a higher lane: NVIDIA, CUDA Programming
// Guide, "C/C++ Language Extensions", "Warp Shuffle Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// Further public works, cited below for ideas that are public:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
// NVIDIA, "CUDA C++ Best Practices Guide", "Occupancy" (__launch_bounds__),
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
// NVIDIA, "CUDA Programming Guide", page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// NVIDIA, "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#pragma once
namespace ga {
// Sums v[k] over the block's threads into part[bid N + k], k < n; bid numbers the blocks of a 1- or
// 2-D grid. Needs blockDim.x % 32 == 0, n <= N <= blockDim.x. All threads call it, once per kernel:
// no barrier follows the last read of the shared array sh. Nothing checks these conditions.
// Field elements stay canonical, so the sums are the same in any order.
// The idea is public: a block reduction through shared memory: Luitjens's post, "Block Reduce" (at
// the top of this file). There the first warp reduces the warps' sums again; here a loop adds them.
// The idea is public: the template's parameter N is the length of v, each thread's own few values:
// Milakov, "Static indexing" (at the top). The compiler hint is public: #pragma unroll: CUDA
// Programming Guide, "C/C++ Language Extensions", "#pragma unroll" (at the top).
template <int N> __device__ __forceinline__ void block_sums(T (&v)[N], int n, T *part) {
  __shared__ T sh[32 * N];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
  const u32 bid = blockIdx.y * gridDim.x + blockIdx.x;
#pragma unroll
  for (int k = 0; k < N; k++)
    if (k < n) v[k] = wsum(v[k]);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < N; k++)
      if (k < n) sh[w * N + k] = v[k];
  }
  __syncthreads();
  if (tid < n) {
    T tot = sh[tid];
    for (int x = 1; x < nw; x++) tot = F::add(tot, sh[x * N + tid]);
    part[(size_t)bid * N + tid] = tot;
  }
}
// The second launch: tot[k] = sum over b < nb of part[b N + k], k < n. Needs one block of 256
// threads. bsum() (prover/attn/engines.cuh) forms the block's sum; its public source is named
// there. The compiler hint and the loop form are public: __launch_bounds__: Best Practices Guide,
// "Occupancy"; the block-stride loop: CUDA Programming Guide, "Writing SIMT Kernels" (at the top).
__global__ void __launch_bounds__(256) k_sum_parts(const T *__restrict__ part, int nb, int N, int n, T *tot) {
  __shared__ T sh[32];
  for (int k = 0; k < n; k++) {
    T s = F::zero();
    for (int b = threadIdx.x; b < nb; b += 256) s = F::add(s, part[(size_t)b * N + k]);
    s = bsum(s, sh);
    if (threadIdx.x == 0) tot[k] = s;
    __syncthreads();
  }
}
// A round's totals are the last MAXV of the MAXBLK * MAXV elements of `Eng::part`. The blocks' sums
// end before them if the kernel has at most MAXBLK blocks (see eblk(), s3_grid()) and calls
// block_sums() with N <= 18. This file checks neither condition.
// Public source: blocks' partial sums and the total in one array: the sum example in NVIDIA's CUDA
// C++ Programming Guide ("Memory Fence Functions"). There the last block writes the total over
// element 0; here a second launch writes the totals to the array's last MAXV elements.
// See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions", "Memory Fence Functions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
static_assert(MAXV >= 19, "MAXBLK blocks of 18 sums and MAXV totals must fit in MAXBLK * MAXV elements");
static inline T *totals(Eng *e) {
  return e->part + (size_t)MAXBLK * MAXV - MAXV;
}
// Adds the sums that the blocks of grid g wrote: N elements per block, the first n of them in use.
static inline void sum_parts(Eng *e, dim3 g, int N, int n, cudaStream_t st) {
  k_sum_parts<<<1, 256, 0, st>>>(e->part, (int)(g.x * g.y), N, n, totals(e));
}
// The third launch, one thread. The round's polynomial h has degree n; tot is h(0), h(2), .., h(n).
// sc_door::round() (prover/hiding/sumcheck_door.cuh) sends the round and returns the challenge.
__global__ void k_send_chal(const T *tot, int n, Tr *tr, T *r) {
  Tr t = *tr;
  *r = sc_door::round<sc_door::Form::NODES, sc_door::Draw::CHAL>(sc_door::Site::ATTENTION_SCD, -1, t, tot, n);
  *tr = t;
}
// SCD round for bit kap of j: h(t) = sum_rows P eq1(b, t) (K0 + t (K1 - K0)). Here b is bit kap of
// x = ix[row] and eq1(b, t) = b ? t : 1 - t; K0, K1 are entries m, m + 2^kap of the row's Kw block,
// m = x mod 2^kap. The partial sums for h(0) and h(2) go to part. With UPD, P is first multiplied
// by eq1(bit kap + 1 of x, *rp). *rp is the previous challenge; Kw must already be folded by it
// (k_scd_kfold in prover/attn/engines.cuh; not checked). k_scd_init there fills P and Kw. Row
// (lam HA 2^lgk + h) 2^lt + i has query head h, g = h / HA and Kw block Kw + (lam 2^lgk + g) 2^lt.
// The idea is public: a flag that is the same for a whole launch as a template parameter: Harris,
// slides "Unrolling with Templates", "Reduction #6: Completely Unrolled" and "Invoking Template
// Kernels" (at the top). In the slides the parameter is the block size; here it is a flag.
// The compiler hint and the loop form are public: __launch_bounds__: Best Practices Guide,
// "Occupancy"; the grid-stride loop: the post on grid-stride loops (both at the top).
template <bool UPD>
__global__ void __launch_bounds__(NTH)
    k_scd_j(int lt, int lgk, int HA, size_t R, int kap, const int *__restrict__ ix, T *__restrict__ P,
            const T *__restrict__ Kw, const T *rp, T *part) {
  T r1, r0;
  if (UPD) {
    r1 = *rp;
    r0 = F::sub(F::one(), r1);
  }
  T v[2] = {F::zero(), F::zero()};
  const int HG = HA << lgk;
  for (size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x; q < R; q += (size_t)gridDim.x * blockDim.x) {
    const int x = ix[q];
    T p = ld16(P + q);
    if (UPD) {
      p = F::mul(p, ((x >> (kap + 1)) & 1) ? r1 : r0);
      st16(P + q, p);
    }
    const size_t hh = q >> lt;
    const int h = (int)(hh % HG), lam = (int)(hh / HG), g = h / HA;
    const T *K = Kw + ((((size_t)lam << lgk) + g) << lt) + (x & ((1 << kap) - 1));
    const T K0 = ld16(K), K1 = ld16(K + ((size_t)1 << kap)), K2 = F::sub(F::add(K1, K1), K0);
    if ((x >> kap) & 1) {
      const T m = F::mul(p, K2);
      v[1] = F::add(v[1], F::add(m, m));
    } else {
      v[0] = F::add(v[0], F::mul(p, K0));
      v[1] = F::sub(v[1], F::mul(p, K2));
    }
  }
  block_sums<2>(v, 2, part);
}
// SCD round over a bit of g or i, on the factor tables that k_scd_tabs (prover/attn/engines.cuh)
// writes: h(t) = sum_in sum_X prod_f (c0 + t (c1 - c0)), in < nbt, X < 2^lh, f < nf; c0, c1 are
// entries X, X + 2^lh of table (in, f), which starts at FT[(in nf + f) GT]. The caller gives h's
// degree d <= MD (not checked). The partial sums for h(0) and h(2) to h(d) go to part (block_sums()
// with N = MD). With FOLD, each table is first folded in place by the previous challenge *rp.
// The idea is public: kernel fusion in a sumcheck, the previous round's fold with this round's
// sums: "Hardware-friendliness of HyperPlonk, Part2", a note on HackMD under the account Ingonyama,
// "Sumcheck in ICICLE", https://hackmd.io/@Ingonyama/Hardware-Friendliness-HyperPlonk-Part2.
// The compiler hints and the loop form are public: __launch_bounds__: Best Practices Guide,
// "Occupancy"; #pragma unroll: CUDA Programming Guide, "C/C++ Language Extensions", "#pragma
// unroll"; the grid-stride loop: the post on grid-stride loops (all at the top).
template <int MD, bool FOLD>
__global__ void __launch_bounds__(128) k_scd_gi(int lh, int nbt, int nf, int d, size_t GT, T *__restrict__ FT,
                                                const T *rp, T *part) {
  const size_t H = (size_t)1 << lh, n = (size_t)nbt << lh;
  F::P rr;
  if (FOLD) rr = F::prep(*rp);
  T acc[MD];
#pragma unroll
  for (int k = 0; k < MD; k++) acc[k] = F::zero();
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n; idx += (size_t)gridDim.x * blockDim.x) {
    T *o = FT + (idx >> lh) * nf * GT + (idx & (H - 1));
    T pr[MD];
    for (int f = 0; f < nf; f++, o += GT) {
      T c0, c1;
      if (FOLD) {
        c0 = fold1(ld16(o), ld16(o + 2 * H), rr);
        c1 = fold1(ld16(o + H), ld16(o + 3 * H), rr);
        st16(o, c0);
        st16(o + H, c1);
      } else {
        c0 = ld16(o);
        c1 = ld16(o + H);
      }
      const T dl = F::sub(c1, c0);
      T val = c1;
      pr[0] = f ? F::mul(pr[0], c0) : c0;
#pragma unroll
      for (int k = 1; k < MD; k++)
        if (k < d) {
          val = F::add(val, dl);
          pr[k] = f ? F::mul(pr[k], val) : val;
        }
    }
#pragma unroll
    for (int k = 0; k < MD; k++)
      if (k < d) acc[k] = F::add(acc[k], pr[k]);
  }
  block_sums<MD>(acc, d, part);
}
// Runs one such round, all three launches; the challenge goes to *r. g is the number of blocks.
template <int MD>
static inline void scd_gi(bool fold, int g, int lh, int nbt, int nf, int d, size_t GT, T *FT, const T *rp, Eng *e,
                          Tr *tr, T *r, cudaStream_t st) {
  if (fold)
    k_scd_gi<MD, true><<<g, 128, 0, st>>>(lh, nbt, nf, d, GT, FT, rp, e->part);
  else
    k_scd_gi<MD, false><<<g, 128, 0, st>>>(lh, nbt, nf, d, GT, FT, rp, e->part);
  sum_parts(e, dim3(g), MD, d, st);
  k_send_chal<<<1, 1, 0, st>>>(totals(e), d, tr, r);
}
} // namespace ga
namespace gc {
// Writes partial sums of sum_b wb[b] sum_x W0[x] X[b N + x], where N = 2^lN and b = blockIdx.y, to
// part, one per block; the caller adds them with ga::k_sum_parts. Requires 256 threads per block.
// The compiler hint and the loop form are public: __launch_bounds__: Best Practices Guide,
// "Occupancy"; the grid-stride loop: the post on grid-stride loops (both at the top).
__global__ void __launch_bounds__(256) k_wsum(const T *wb, const T *W0, const T *X, int lN, T *part) {
  const int b = blockIdx.y;
  const size_t N = (size_t)1 << lN;
  T v[1] = {F::zero()};
  for (size_t x = blockIdx.x * (size_t)256 + threadIdx.x; x < N; x += (size_t)gridDim.x * 256)
    v[0] = F::add(v[0], F::mul(ld16(W0 + x), ld16(X + ((size_t)b << lN) + x)));
  v[0] = F::mul(v[0], wb[b]);
  ga::block_sums<1>(v, 1, part);
}
} // namespace gc
