// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Two steps around a Gram pass of a product layer (both terms: prover/hiding/gram_products.cuh).
// A product layer of a lookup argument runs a sumcheck on the sum over x of eq(rho, x) L(x) R(x),
// where L and R are two tables of field elements, eq is the multilinear equality polynomial and
// rho is the layer's point.
// One Gram pass over L and R yields the sums from which the messages of three sumcheck rounds are
// computed. The three index bits that these rounds bind divide each table into eight slabs.
// k_prepeq() runs before a Gram pass, on a table of eq weights (factors of eq(rho, x)); fold8_k3()
// runs after one, on tables stored in GPU memory. (Some passes have no stored tables and compute
// their entries from a lookup call's keys: prover/hiding/gram_products.cuh.) An entry of
// a table is an element of K3 = F_p[X] / (X^3 - w), with p = 2^64 - 2^32 + 1 and w = 2: three
// 64-bit coordinates. fold8_k3() moves entries in pairs, as three 128-bit words, so its tables must
// be 16-byte aligned and its `stride` and SC even; it checks neither.
//
// Public sources.
// Two small tables in place of the whole table of eq weights; k_prepeq() converts one of the two:
// Dao and Thaler, "More Optimizations to Sum-Check Proving", ePrint 2024/1210, section 1.1.
// Lazy reduction: a product in F_p[X] / (X^k - w) takes k reductions modulo p; here the eight
// products of one output are added before each of the three coordinates is reduced (FK2::mac,
// FK2::ared in prover/hiding/pt_field.h): Aranha et al., "Faster Explicit Formulas for Computing
// Pairings over Ordinary Curves", EUROCRYPT 2011, section 3.1, where they state it as a result of
// Lim and Hwang; ePrint 2010/526, https://eprint.iacr.org/2010/526.
// Loads and stores of 128 bits; the post widens accesses to an array of integers, and here two
// neighbouring entries are moved as three 128-bit words: "CUDA Pro Tip: Increase Performance with
// Vectorized Memory Access", NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The grid-stride loop of k_fold8(), whose unit is a pair of entries: M. Harris, "CUDA Pro Tip:
// Write Flexible Kernels with Grid-Stride Loops", NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
#pragma once
#include "hiding/pt_field.h"
#include <algorithm>
namespace kg3 {
static_assert(sizeof(K3) == 24 && sizeof(M3L) == 40, "K3 must have 24 bytes and M3L 40 bytes");
// Prepares a table of eq weights once, for the many products with them: prepared[i] = table[i] as
// an M3L, for i < entries. An M3L (prover/hiding/pt_field.h) contains the three coordinates and w
// times the second and the third, each as two u32 limbs, so that a product with it need not
// multiply by w. Launch at least `entries` threads.
__global__ void k_prepeq(const K3 *table, int entries, M3L *prepared) {
  const int i = (int)(threadIdx.x + blockDim.x * blockIdx.x);
  if (i >= entries) return;
  prepared[i] = FK2::prep(table[i]);
}
// Two neighbouring entries: 48 bytes, three words of 16.
struct Two {
  K3 first, second;
};
// In load_two() and store_two(), `at` must be 16-byte aligned (not checked). load_two() reads with
// __ldg(), so the entries must not change while the kernel runs.
// Public sources for the techniques of this routine.
// The idea is public: vectorized loads of aligned data through a recast pointer: "CUDA Pro Tip:
// Increase Performance with Vectorized Memory Access", cited at the top of this file.
// The idea is public: read-only loads with __ldg(): NVIDIA, Kepler Tuning Guide, "Memory
// Throughput", part "Read-Only Data Cache", https://docs.nvidia.com/cuda/kepler-tuning-guide/.
__device__ __forceinline__ Two load_two(const K3 *at) {
  const ulonglong2 *word = reinterpret_cast<const ulonglong2 *>(at);
  const ulonglong2 a = __ldg(word), b = __ldg(word + 1), c = __ldg(word + 2);
  return {{a.x, a.y, b.x}, {b.y, c.x, c.y}};
}
__device__ __forceinline__ void store_two(K3 *at, const K3 &first, const K3 &second) {
  ulonglong2 *word = reinterpret_cast<ulonglong2 *>(at);
  word[0] = make_ulonglong2(first.k0, first.k1);
  word[1] = make_ulonglong2(first.k2, second.k0);
  word[2] = make_ulonglong2(second.k1, second.k2);
}
constexpr int FOLD8_THREADS = 256;
// The kernel of fold8_k3(). Launch blocks of FOLD8_THREADS threads, the size that the loop's index
// and step assume, in a grid of (any number, 2). The grid's second dimension is the table: 0 folds
// L into Lo, 1 folds R into Ro. A thread computes two neighbouring outputs at a time.
// Public sources for the techniques of this routine.
// The idea is public: several outputs per thread (thread coarsening), which there saves redundant
// work and uses more registers: Wen-mei W. Hwu and David Kirk, "Lecture 3: Thread Coarsening and
// More on Tiling/Blocking", ICCS School "Advanced GPU Programming for Science", Beijing,
// March 26-27, 2012, slide "Outline of Technique",
// https://silkroad.zah.uni-heidelberg.de/silkroad-save/lecture3%20coarsening%20tiling%20Beijing%202012.pdf.
// The idea is public: delayed reduction (there: just before an overflow or after a fixed number of
// accumulations): J.-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime
// Fields: the FFLAS and FFPACK packages", section 2.1.2, https://arxiv.org/pdf/cs/0601133. Here the
// fixed number is eight, the products of one output (FK2::mac, FK2::ared).
__global__ void __launch_bounds__(FOLD8_THREADS)
    k_fold8(const K3 *__restrict__ L, const K3 *__restrict__ R, size_t stride, size_t SC, const K3 *__restrict__ k,
            K3 *__restrict__ Lo, K3 *__restrict__ Ro) {
  __shared__ M3L weight[8];
  if (threadIdx.x < 8) weight[threadIdx.x] = FK2::prep(k[threadIdx.x]);
  __syncthreads();
  const K3 *from = blockIdx.y ? R : L;
  K3 *to = blockIdx.y ? Ro : Lo;
  const size_t step = 2 * (size_t)gridDim.x * FOLD8_THREADS;
  for (size_t c = 2 * (blockIdx.x * (size_t)FOLD8_THREADS + threadIdx.x); c < SC; c += step) {
    FK2::Acc even = FK2::azero(), odd = FK2::azero();
#pragma unroll 2
    for (int slab = 0; slab < 8; slab++) {
      const Two entries = load_two(from + slab * stride + c);
      FK2::mac(even, weight[slab], entries.first);
      FK2::mac(odd, weight[slab], entries.second);
    }
    store_two(to + c, FK2::ared(even), FK2::ared(odd));
  }
}
} // namespace kg3
// Folds each table's eight slabs of SC entries into one, which binds the three index bits: for
// every c < SC, Lo[c] = the sum over a < 8 of k[a] * L[a * stride + c], and Ro[c] is the same sum
// over R. Slab a starts at entry a * stride. The callers pass k[a] = eq(the three rounds'
// challenges, a). k is in GPU memory. Not checked: `stride` and SC must be even, the four tables
// must be 16-byte aligned, and Lo and Ro must not overlap L, R, k or each other. The results are
// canonical: each coordinate is below p. The grid is empty if SC = 0, or if dev_info() has not yet
// set g_sms, the GPU's multiprocessor count; the launch then fails and CK() exits with code 1
// (these three names: prover/common/common.h).
static inline void fold8_k3(const K3 *L, const K3 *R, size_t stride, size_t SC, const K3 *k, K3 *Lo, K3 *Ro,
                            cudaStream_t st = 0) {
  const size_t wanted = (SC / 2 + kg3::FOLD8_THREADS - 1) / kg3::FOLD8_THREADS;
  const dim3 grid((unsigned)std::min<size_t>(wanted, (size_t)g_sms * 4), 2);
  kg3::k_fold8<<<grid, kg3::FOLD8_THREADS, 0, st>>>(L, R, stride, SC, k, Lo, Ro);
  CK(cudaGetLastError());
}
