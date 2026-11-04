// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Counts 16-bit lookup keys, for a lookup argument's multiplicities (how often each table row is
// used). A key is a row number in a lookup table of 2^16 rows. An identity is a part of a lookup's
// keys that is counted separately. Include prover/common/common.h (u32, CK, CK_ATTRIBUTE) first.
//
// Public sources: N. Sakharnykh, "GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell"
// (NVIDIA Technical Blog, 2015): a shared-memory histogram per thread block, by atomics; a second
// kernel adds these histograms. V. Podlozhnyuk, "Histogram calculation in OpenCL" (NVIDIA, 2009):
// sub-histograms in local memory. AMD, HIP documentation, "Atomic operations: Histogram tutorial"
// (https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/programming-patterns/atomic_operations_histogram.html):
// "Use fast, low-latency shared memory to aggregate partial results within a block before issuing a
// single atomic update to global memory."; here each block adds its counts that way, in the same
// kernel. See also, for Sakharnykh's post:
// https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/.
#pragma once
#ifdef FLASHPROVE_L4
// Small-model path: direct global atomics avoid Hopper's 128 KiB shared histogram.
// The output layout and every counted low-k-bit key are unchanged.
__global__ void k_count_keys_l4(const uint4 *keys, size_t nvb, int bpi, int k, u32 *outAll) {
  const int b = blockIdx.x / bpi, w = blockIdx.x % bpi;
  const uint4 *kb = keys + (size_t)b * nvb;
  u32 *out = outAll + ((size_t)b << k);
  const u32 mask = (1u << k) - 1;
  for (size_t i = w * (size_t)blockDim.x + threadIdx.x; i < nvb; i += (size_t)bpi * blockDim.x) {
    const uint4 v = kb[i];
    const u32 x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int a = 0; a < 8; a++) atomicAdd(out + ((x[a >> 1] >> (16 * (a & 1))) & mask), 1u);
  }
}
static inline void count_keys_l4(const uint4 *keys, size_t nvb, int bpi, int identities, int k, u32 *out, cudaStream_t stream) {
  k_count_keys_l4<<<identities * bpi, 256, 0, stream>>>(keys, nvb, bpi, k, out);
  CK(cudaGetLastError());
}
#endif
// outAll[(b << 16) + key] += the number of times that key occurs among identity b's keys.
// keys: for each identity in turn, nvb uint4 values of eight keys. bpi: thread blocks per identity.
// ForestTable::counts() (prover/hiding/forest_table.cuh) uses one identity: all the keys that it is
// given go into one histogram.
// 65536 u32 counters exceed a block's shared memory, so a block reads its keys twice, once for each
// half of the key range. Launch the kernel through count_keys16(), which allows it 131072 bytes.
// Not checked: `keys` must be 16-byte aligned (the caller casts it from uint16_t *); a count wraps
// at 2^32. h_idk() (prover/hiding/product_test.cuh) counts keys of lookup tables with fewer rows.
// Public sources for the techniques of this routine (for the histogram see also the file's head).
// The idea is public: a shared-memory histogram per thread block, merged by global atomics: NVIDIA,
// "CUDA Programming Guide", page "Writing SIMT Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
// That the counters in shared memory are updated by atomic operations is public too: Sakharnykh's
// post, cited at the file's head.
// The idea is public: several passes over the input, each of which updates only one range of the
// bins: T. Henriksen, S. Hellfritzsch, P. Sadayappan, C. Oancea, "Compiling Generalized Histograms
// for GPU", SC20, 2020, sections I and III, https://futhark-lang.org/publications/sc20.pdf.
// The idea is public: vectorized memory access, one load for several narrow values: "CUDA Pro Tip:
// Increase Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/;
// NVIDIA, CUTLASS documentation, "Fundamental Types", "AlignedArray",
// https://docs.nvidia.com/cutlass/media/docs/cpp/fundamental_types.html.
// The compiler hint and the loop form are public: #pragma unroll: NVIDIA, "CUDA Programming Guide",
// page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// the block-stride loop: shown in a listing on the page "Writing SIMT Kernels", cited above.
__global__ void k_count_keys16(const uint4 *keys, size_t nvb, int bpi, u32 *outAll) {
  extern __shared__ u32 bins[];
  const int b = blockIdx.x / bpi, w = blockIdx.x % bpi;
  const uint4 *kb = keys + (size_t)b * nvb;
  u32 *out = outAll + ((size_t)b << 16);
  for (u32 half = 0; half < 2; half++) {
    for (int i = threadIdx.x; i < 32768; i += blockDim.x) bins[i] = 0;
    __syncthreads();
    for (size_t i = w * (size_t)blockDim.x + threadIdx.x; i < nvb; i += (size_t)bpi * blockDim.x) {
      uint4 v = kb[i];
      u32 x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int a = 0; a < 8; a++) {
        const u32 k = (x[a >> 1] >> (16 * (a & 1))) & 65535u;
        if (k >> 15 == half) atomicAdd(&bins[k & 32767u], 1u);
      }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < 32768; c += blockDim.x) atomicAdd(&out[half * 32768u + c], bins[c]);
    __syncthreads();
  }
}
// Launches k_count_keys16(); exits if the launch is rejected. `out`: the kernel's outAll; zero it
// first. The 131072-byte limit is set at the first call only, for the device then current.
static inline void count_keys16(const uint4 *keys, size_t nvb, int bpi, int identities, u32 *out, cudaStream_t stream) {
#ifdef FLASHPROVE_L4
  count_keys_l4(keys, nvb, bpi, identities, 16, out, stream);
#else
  static const bool once = [] {
    CK_ATTRIBUTE(k_count_keys16, cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
    return true;
  }();
  (void)once;
  k_count_keys16<<<identities * bpi, 1024, 131072, stream>>>(keys, nvb, bpi, out);
  CK(cudaGetLastError());
#endif
}
