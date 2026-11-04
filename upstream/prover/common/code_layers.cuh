// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// The layers of the transform that encodes a row of coefficients into its codeword: radix-2
// butterflies mod p = 2^64 - 2^32 + 1. blk::lazy::TransformStep
// (prover/field/field_lazy.cuh) supplies the butterfly and the form of a value between two layers,
// so no code here depends on p. A row of 2^ln 64-bit words has layers ln - 1 down to 0, run in
// that order: upper_layers(), then last_layers(). Between launches a word need not be the smallest
// of its residue class; last_layers() writes canonical words (below p).
// Both functions enqueue kernels on stream st and do not wait; a launch error ends the process with
// exit code 1 (CK in prover/common/common.h). No argument is checked. The caller must ensure that
// no launch needs over 2^31 - 1 thread blocks, one for each tile (at most 2^TILE positions) of each
// row: a count of 2^32 or more is cut to 32 bits, and the launch can then succeed and skip tiles.
//
// Public source: D. Harvey, "Faster arithmetic for number-theoretic transforms", arXiv:1205.2926:
// fewer reductions mod p by a redundant representation of the residues.
// Public source: radix-2 layers in registers and shared memory, a long transform cut into launches:
// S. Kim, W. Jung, J. Park, J. H. Ahn, "Accelerating Number Theoretic Transformations for
// Bootstrappable Homomorphic Encryption on GPUs", arXiv:2012.01968.
// Public source: a number-theoretic transform on a GPU: O. Ozerk, C. Elgezen, A. C. Mert,
// E. Ozturk, E. Savas, "Efficient Number Theoretic Transform Implementation on GPU for Homomorphic
// Encryption", ePrint 2021/124.
// Public source: unused words that pad a shared array, as in padded(): NVIDIA, CUDA C++ Best Practices Guide,
// "Shared Memory in Matrix Multiplication (C=AAT)",
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
#pragma once
#include "common/common.h"
#include "field/field_lazy.cuh"
#include <algorithm>
namespace code_layers {
typedef blk::lazy::TransformStep Step;
// Returns x mod p, for any 64-bit word x.
__device__ __forceinline__ u64 canonical(u64 x) {
  return field_from_u64(x);
}
// A table of the butterflies' factors (twiddles; tw below) has two forms; every function with a
// parameter tw takes either. As u64 it contains canonical words, and each butterfly converts its
// factor by Step::prepare(), which costs a multiplication mod p. As `Prepared` it contains
// converted factors (16 bytes each): for tables read many times.
struct Prepared {
  Step::Tw w;
};
__device__ __forceinline__ Step::Tw factor(const u64 *__restrict__ tw, size_t i) {
  return Step::prepare(tw[i]);
}
__device__ __forceinline__ Step::Tw factor(const Prepared *__restrict__ tw, size_t i) {
  return tw[i].w;
}
__device__ __forceinline__ void put_factor(u64 *tw, size_t i, u64 canonical_word) {
  tw[i] = canonical_word;
}
__device__ __forceinline__ void put_factor(Prepared *tw, size_t i, u64 canonical_word) {
  tw[i].w = Step::prepare(canonical_word);
}
// A tile (see k_layers_by_trips) has at most 2^TILE positions.
constexpr int TILE = 12;
// TransformStep's capacity: the largest number of layers that a value may pass between Step::in()
// (from a 64-bit word) and Step::out() or Step::canonical() (back to one). A launch has at most
// TILE layers, so none converts midway. TransformStep has no constant for this capacity, so no
// code ties the 16 here to it.
constexpr int LAYERS_IN_A_ROW = 16;
static_assert(TILE <= LAYERS_IN_A_ROW, "a launch's layers are within the step's capacity");
// The index of a tile's e-th word in the shared buffer of k_layers_by_trips: one unused word
// follows every 16.
// The idea is public: padding a shared array against bank conflicts, index i to i + i / B (there
// B = 16 banks): M. Harris, S. Sengupta, J. D. Owens, GPU Gems 3, chapter 39, at Figure 39-5,
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda. Here
// the divisor is the number of elements that span all banks.
__device__ __forceinline__ u32 padded(u32 e) {
  return e + (e >> 4);
}
// One trip of k_layers_by_trips: applies layers lo + low + c - 1 down to lo + low to the whole
// tile (s, `more`). A thread takes 2^c words at a time through the c layers. tl is hi - lo, lw is
// lt - tl, and `up` is (any position of the tile) >> hi; the other names are the kernel's. The
// caller synchronizes the thread block before and after a trip.
// Public source: A. S. Ozcan, E. Savas, "Two Algorithms for Fast GPU Implementation of NTT",
// ePrint 2023/1410, Algorithm 5: in a kernel that runs several layers in shared memory, the twiddle
// index is a power of two plus the block's and the thread's indices, there summed and divided by a
// power of two. Here each part is shifted by itself and ORed in (the parts do not overlap), and k,
// the butterfly within the group, is one part more.
// The compiler hint and the loop form are public: #pragma unroll: NVIDIA, "CUDA Programming Guide",
// page "C/C++ Language Extensions", "#pragma unroll",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// registers can hold v, a thread's few (2^c) words, as each subscript is fixed by the unrolling:
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015,
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/;
// the block-stride loop: shown in a listing in NVIDIA, "CUDA Programming Guide", page "Writing SIMT
// Kernels", "Distributed Shared Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
template <int c, class Table>
__device__ __forceinline__ void trip(u64 *s, unsigned char *more, const Table *__restrict__ tw, int ln, int lo, int lt, int tl,
                                     int lw, int low, size_t up) {
  const int dl = low + lw;
  for (u32 g = threadIdx.x; g < (1u << (lt - c)); g += blockDim.x) {
    const u32 gh = g >> dl, e0 = (gh << (dl + c)) | (g & ((1u << dl) - 1));
    Step::Word v[1 << c];
#pragma unroll
    for (int k = 0; k < (1 << c); k++) v[k] = Step::get(s, more, padded(e0 + ((u32)k << dl)));
#pragma unroll
    for (int j = c - 1; j >= 0; j--) {
      const int layer = low + j;
      const size_t first =
          ((size_t)1 << (ln - 1 - lo - layer)) | (up << (tl - 1 - layer)) | ((size_t)gh << (c - 1 - j));
#pragma unroll
      for (int k = 0; k < (1 << c); k++)
        if (!(k & (1 << j))) Step::butterfly(v[k], v[k | (1 << j)], factor(tw, first | (size_t)(k >> (j + 1))));
    }
#pragma unroll
    for (int k = 0; k < (1 << c); k++) Step::put(s, more, padded(e0 + ((u32)k << dl)), v[k]);
  }
}
// Applies layers hi - 1 down to lo, in place, to `rows` contiguous rows of n = 2^ln positions.
// Layer L replaces the words a at position i and b at i + 2^L, for every i whose bit L is 0, by
// (a + w b, a - w b) mod p, with w = tw[2^(ln - 1 - L) + (i >> (L + 1))]; row_tree::tables() in
// hash_tree.cuh makes such a table. On entry a word is canonical or as an earlier launch left it.
// `finish`: write canonical words. A tile is a set of 2^lt positions of one row that these layers
// map to itself: 2^(hi - lo) positions 2^lo apart, times 2^(lt - hi + lo) adjacent positions.
// A thread block handles one tile in shared memory, in trips of up to three layers. Requires
// hi - lo <= lt <= min(TILE, hi). Of each row's 2^(ln - lt) tiles, T * rows thread blocks handle
// tiles 0 to T - 1 if `some` is null, otherwise tiles some[x] >> some_shift for x < T.
// Public sources for the techniques of this routine.
// The idea is public: coalesced access to strided data, by combining thread blocks so that the
// data of a memory transaction is not wasted: Kim and others (cited at the top of this file),
// section VI.C, "Avoiding uncoalesced memory accesses in Kernel-1", which attributes the combining
// of thread blocks to prior work on the DFT (N. K. Govindaraju, B. Lloyd, Y. Dotsenko, B. Smith,
// J. Manferdelli, "High Performance Discrete Fourier Transforms on Graphics Processors",
// Proceedings of the 2008 ACM/IEEE Conference on Supercomputing, 2008,
// https://csd.uwo.ca/~mmorenom/CS433-CS9624/Resources/DiscreteFourierTransformsonGPUs.pdf).
// The idea is public: the largest radix as often as it fits, and a smaller one last: Kim and
// others, section VI.C; Govindaraju and others (both named in the paragraph above).
// The padded buffer: see padded().
// The compiler hint and the loop form are public: __launch_bounds__: the Best Practices Guide
// (cited at the top of this file), "Occupancy"; the block-stride loop: CUDA Programming Guide,
// "Writing SIMT Kernels" (cited at trip()).
template <class Index, class Table>
__global__ void __launch_bounds__(256)
    k_layers_by_trips(u64 *__restrict__ cw, const Table *__restrict__ tw, int ln, int hi, int lo, int lt, u32 rows,
                      const Index *__restrict__ some, int some_shift, bool finish) {
  __shared__ u64 s[(1 << TILE) + (1 << (TILE - 4))];
  __shared__ unsigned char more[Step::MORE ? (1 << TILE) + (1 << (TILE - 4)) : 1]; // what a value has beyond its word, if anything
  const int tl = hi - lo, lw = lt - tl;
  const size_t n = (size_t)1 << ln, row = blockIdx.x % rows, x = blockIdx.x / rows,
               tile = some ? (size_t)some[x] >> some_shift : x;
  const size_t lot = tile & (((size_t)1 << (lo - lw)) - 1), up = tile >> (lo - lw);
  u64 *o = cw + row * n + ((up << hi) | (lot << lw));
  const u32 run = (1u << lw) - 1;
  for (u32 e = threadIdx.x; e < (1u << lt); e += blockDim.x)
    Step::put(s, more, padded(e), Step::in(o[((size_t)(e >> lw) << lo) + (e & run)]));
  __syncthreads();
  for (int left = tl; left > 0;) {
    const int c = left >= 3 ? 3 : left;
    if (c == 3)
      trip<3>(s, more, tw, ln, lo, lt, tl, lw, left - 3, up);
    else if (c == 2)
      trip<2>(s, more, tw, ln, lo, lt, tl, lw, left - 2, up);
    else
      trip<1>(s, more, tw, ln, lo, lt, tl, lw, left - 1, up);
    left -= c;
    __syncthreads();
  }
  for (u32 e = threadIdx.x; e < (1u << lt); e += blockDim.x) {
    const Step::Word z = Step::get(s, more, padded(e));
    o[((size_t)(e >> lw) << lo) + (e & run)] = finish ? Step::canonical(z) : Step::out(z);
  }
}
// Applies layers hi - 1 down to `end` to all rows, at most eight layers a launch: a tile of 2^TILE
// positions then consists of runs of at least 16 adjacent words. Does nothing if hi <= end. The
// words that it writes need not be canonical.
template <class Table>
static inline void upper_layers(u64 *cw, const Table *tw, int ln, int hi, int end, int rows, cudaStream_t st) {
  while (hi > end) {
    const int up = hi - end, launches = (up + 7) / 8, tl = (up + launches - 1) / launches, lt = std::min(TILE, hi);
    const unsigned blocks = (unsigned)((((size_t)1 << ln) >> lt) * rows);
    k_layers_by_trips<u64, Table><<<blocks, 256, 0, st>>>(cw, tw, ln, hi, hi - tl, lt, (u32)rows, nullptr, 0, false);
    CK(cudaGetLastError());
    hi -= tl;
  }
}
// Applies layers last - 1 down to 0 (last <= TILE) and writes canonical words. Tile t starts at
// t * 2^last. `some`: null for all tiles, or n_some tiles as in the kernel, which must be distinct
// because a tile is transformed in place. No call in this repository passes a non-null `some`, so
// no program or test runs that path.
template <class Index, class Table>
static inline void last_layers(u64 *cw, const Table *tw, int ln, int last, int rows, const Index *some, int n_some,
                               int some_shift, cudaStream_t st) {
  const size_t tiles = some ? (size_t)n_some : ((size_t)1 << ln) >> last;
  k_layers_by_trips<Index, Table>
      <<<(unsigned)(tiles * rows), 256, 0, st>>>(cw, tw, ln, last, 0, last, (u32)rows, some, some_shift, true);
  CK(cudaGetLastError());
}
} // namespace code_layers
