// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Adds the per-chunk sums of a lookup argument's product layer (a level of a product tree) whose
// entries are computed from the lookup keys, not stored. A chunk is a range of the layer's indices
// (defined in gram_products.cuh). trees_pass() in forest.cuh also runs addchunks() on
// ff::first_range()'s sums (forest_fused_first.cuh), one set per thread block.
//
// Public source: a block's tree sum: M. Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA
// Developer Technology), the part on sequential addressing, and, for several adds per
// thread first, "Multiple Adds / Thread",
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// Differences: the slides add integers, not field elements, and place the barrier after each
// step, not before it.
#pragma once
#include "hiding/pt_field.h"
namespace t3k {
constexpr int ADD_THREADS = 128;
// G[p] = sum over ch < nchunks of Gc[ch * nv + p], for p < nv, in K3, the cubic extension field
// (prover/common/common.h). The terms must be canonical (each coordinate below the modulus;
// unchecked); then G[p] is too.
// Launch nv blocks of exactly ADD_THREADS threads (unchecked). ADD_THREADS must be a power of two.
// Public source: see the top of this file.
// The idea is public: several adds per thread first: the same slides, "Algorithm Cascading" and
// "Reduction #7: Multiple Adds / Thread" (there the stride is that of the grid).
// The loop form is public: the block-stride loop, shown in a listing in NVIDIA,
// "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
__global__ void addchunks(const K3 *Gc, K3 *G, int nchunks, int nv) {
  __shared__ K3 part[ADD_THREADS];
  K3 mine = FK2::zero();
  for (int ch = threadIdx.x; ch < nchunks; ch += ADD_THREADS) mine = k3_add(mine, Gc[(size_t)ch * nv + blockIdx.x]);
  part[threadIdx.x] = mine;
  for (int half = ADD_THREADS / 2; half > 0; half /= 2) {
    __syncthreads(); // after this barrier both parts that a thread adds are in shared memory
    if (threadIdx.x < half) part[threadIdx.x] = k3_add(part[threadIdx.x], part[threadIdx.x + half]);
  }
  if (threadIdx.x == 0) G[blockIdx.x] = part[0];
}
} // namespace t3k
