// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU kernels of the row encoder, and the SlabEncoder functions that encode.cuh only declares. A
// slab is up to sixteen rows that are encoded together. Layer L of its transform combines values
// 2^L positions apart. `real` is the number of layers that the slab's longest row needs (top_of());
// the layers from `real` up only copy. A pass is one kernel launch that runs 4 to 8 layers: strided
// passes run the layers from 8 up, the last pass layers 7 to 0. A tile is what one thread block
// works on at a time: 4,096 words of one row in a strided pass, 256 positions of all the slab's
// rows in the last pass.
// An includer must define ENCODER_FN: encode.cuh, the one includer in this tree, defines it as
// `inline`.
//
// The public works in full (a function that cites one does so in short):
// Public source: S. Kim, W. Jung, J. Park, J. H. Ahn, "Accelerating Number Theoretic
// Transformations for Bootstrappable Homomorphic Encryption on GPUs", arXiv 2012.01968 (layers
// grouped into passes over GPU memory: sections V and VI; several layers on values held in
// registers; the runs of layers of a launch joined through shared memory with a barrier between).
// Public source: A. S. Ozcan, E. Savas, "Two Algorithms for Fast GPU Implementation of NTT", ePrint
// 2023/1410, section V (layers grouped into passes over GPU memory).
// Public source: O. Ozerk, C. Elgezen, A. C. Mert, E. Ozturk, E. Savas, "Efficient Number Theoretic
// Transform Implementation on GPU for Homomorphic Encryption", ePrint 2021/124 (a number-theoretic
// transform on a GPU).
// Public source: Patrick Longa and Michael Naehrig, "Speeding up the Number Theoretic Transform for
// Faster Ideal Lattice-Based Cryptography", ePrint 2016/504, Algorithm 1 (Function NTT based on the
// Cooley-Tukey butterfly), https://eprint.iacr.org/2016/504 (the order of the factors: factor_at();
// the step of a layer, from (a, b) to (a + t b, a - t b), is the butterfly of that algorithm).
// Public source: Jongmin Kim, Wonseok Choi and Jung Ho Ahn, "Cheddar: A Swift Fully Homomorphic
// Encryption Library for CUDA GPUs", arXiv 2407.13055 (2024), section 4.1, "Number-theoretic
// transform (NTT)", https://arxiv.org/abs/2407.13055 (factor_at()).
// Public source: Triton documentation (triton-lang.org), tutorial "Matrix Multiplication", part
// "Compute Kernel",
// https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
// (factor_at()).
// Public source: D. Harvey, "Faster arithmetic for number-theoretic transforms", arXiv 1205.2926
// (words unreduced between the layers). After that paper, a value stays unreduced between two runs
// of layers (any 64-bit word of its residue class); it is made canonical once, after the last
// layer, in last_pass_symbols().
// Public source: David Harvey, "A cache-friendly truncated FFT", arXiv:0810.3203 (2008), section 2
// "Notation and setup" and section 3 "A cache-friendly TFT" (Algorithm 1 and the proof of
// Theorem 1), https://arxiv.org/abs/0810.3203 (k_strided_pass()).
// Public source: FFTW, "Pruned FFTs", https://www.fftw.org/pruned.html (layers and columns that are
// left out: top_of(), k_last_pass_gather()).
// Public source: T. Lu, Y. Chen, Z. Wang, X. Wang, W. Chen, J. Zhang, "BatchZK: A Fully Pipelined
// GPU-Accelerated System for Batch Generation of Zero-Knowledge Proofs", IACR ePrint 2024/1862 (a
// hash tree on a GPU; section 3.1: many independent calls of a hash on a GPU: k_last_pass()).
// Public source: Trevor Gale, Deepak Narayanan, Cliff Young, Matei Zaharia, "MegaBlocks: Efficient
// Sparse Training with Mixture-of-Experts", 2022, Section 5.1.3, https://arxiv.org/abs/2211.15841
// (the list `at`: k_last_pass_gather()).
// Public source: Ahmad Abdelfattah, Azzam Haidar, Stanimire Tomov, Jack Dongarra, "Performance,
// Design, and Autotuning of Batched GEMM for GPUs", Section 3,
// https://library.eecs.utk.edu/files/ut-eecs-16-739.pdf (the list `at`: k_last_pass_gather()).
// Public source: NVIDIA, CUDA C++ Programming Guide, release 12.8: the sections are named where
// they are cited. See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html:
// the parts are named where they are cited ("Launch Bounds", "#pragma unroll", "Warp
// Synchronization Function").
// Public source: NVIDIA, CUDA C++ Best Practices Guide,
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (its headings are named where they are
// cited).
// Further public works, cited below for ideas that are public:
// M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology, slides 24 to 27,
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf (template parameters and
// the switches over their instances);
// M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, part
// "Static indexing", https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/
// (#pragma unroll on the loops over a thread's small array);
// "CUDA Pro Tip: Increase Performance with Vectorized Memory Access", a post on NVIDIA's developer
// blog (a version ran Dec. 4, 2013),
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
// (k_last_pass()).
#pragma once
#include <atomic>

#include "engine/encode.cuh"
#include "field/field_lazy.cuh"
#include "field/transform_groups_gl64.cuh"

namespace blk {
#ifdef ENGINE_ENCODER_OBJECT
inline namespace encoder_object {
#else
inline namespace encoder_in_this_program {
#endif

// Threads per block: the kernels rely on it and do not check blockDim.x.
constexpr int ENCODE_THREADS = 256;
static_assert(ENCODE_THREADS == 16 * SLAB_ROWS, "in the last pass sixteen threads share a row");

// Reads twiddle factor x >= 1 (not checked) of a tile's heap into t. A block of layer L is
// 2^(L + 1) neighbouring positions, which share one factor. x = 1 is the factor of block `top` of
// layer `widest`; x = 2^d to 2^(d + 1) - 1 are the factors of the 2^d blocks of layer widest - d
// under it. Returns false for layers from log_n up, which have none.
// In `table`, block i of a layer with m blocks has its factor at index m + i. Here i is taken
// modulo m: a code shorter than a tile is processed as if its positions repeated periodically.
// Public source: Triton's matrix-multiplication tutorial takes a tile's row and column indices
// modulo the matrix's size and masks the store. Here the same is done for the positions of a
// butterfly network: the kernels read at x modulo n (from a row's own cells: modulo 2^real), and
// those that go over all positions store only where x < n.
// Public source: the heap has the order that Longa and Naehrig (ePrint 2016/504, Algorithm 1) give
// the table of a whole number-theoretic transform.
// Public source: Kim, Choi and Ahn (arXiv 2407.13055, section 4.1) number the factors of successive
// stages i, then 2i and 2i + 1.
// What differs here: the copy in shared memory contains one tile's factors only.
// __clz: CUDA Math API, "Integer Intrinsics".
__device__ __forceinline__ bool factor_at(const u64 *__restrict__ table, int log_n, int widest, u32 top, int x, u64 &t) {
  const int d = 31 - __clz(x), lm = log_n - 1 - (widest - d);
  if (lm < 0) return false;
  const u32 m = 1u << lm;
  t = table[m + (((top << d) + (x - (1 << d))) & (m - 1))];
  return true;
}

// A pass's factors in shared memory, and its layers on one thread's sixteen words. first() runs the
// FIRST widest layers, from layer `widest` (as given to load()) down; second() runs the four below
// them, in the tile's block `root` (below 2^FIRST) of layer widest - FIRST. The caller must put a
// barrier after load(), before first() or second(). All threads of the thread block must call
// load(): it has a barrier inside. lazy::groups runs the layers
// (prover/field/transform_groups_gl64.cuh), and load() also computes from the factors the tables of
// powers that group() reads; PITCH: words from one root's powers to the next's.
// SKIP: how many of a pass's widest layers are at or above `real` and so only copy:
// widest - real + 1 clamped to 0 .. FIRST + 4. It is a template argument, and the host picks the
// kernel (with_skip()), because a run-time count was miscompiled: see powers() in
// prover/field/transform_groups_gl64.cuh. The unnamed parameters, for which callers pass the widest
// layer and `real`, are not used.
// SKIP_IS_COMPILED is always true: the host's branches for false are never taken.
// ENCODE_BLOCKS: thread blocks per multiprocessor that the kernels are compiled for.
// Public source: Kim, Jung, Park and Ahn (arXiv:2012.01968, section VI.C) preload a thread block's
// twiddle factors into shared memory, in one of their two kernels (Kernel-1); load() does the same
// for the layers of one pass, in every pass.
// The compiler resolves the tests on the template parameters. The idea is public: Harris,
// "Optimizing Parallel Reduction in CUDA", slides 25 and 26.
constexpr int ENCODE_BLOCKS = 3;
constexpr bool SKIP_IS_COMPILED = true;
template <int FIRST, int PITCH, int SKIP> struct PassFactors {
  static_assert(SKIP >= 0 && SKIP <= FIRST + 4, "a pass has FIRST + 4 layers");
  static constexpr int SKIP_FIRST = SKIP < FIRST ? SKIP : FIRST, SKIP_SECOND = SKIP - SKIP_FIRST;
  u64 powers_first[16], powers_second[PITCH << FIRST], heap[16 << FIRST];
  __device__ __forceinline__ void load(const u64 *__restrict__ table, int log_n, int widest, u32 top, int) {
    u64 t;
    for (int x = threadIdx.x; x < (16 << FIRST); x += ENCODE_THREADS)
      if (x && factor_at(table, log_n, widest, top, x, t)) heap[x] = t;
    __syncthreads();
    if constexpr (FIRST > 0) lazy::groups::powers<FIRST, SKIP_FIRST>(powers_first, heap, 0);
    lazy::groups::powers<4, SKIP_SECOND>(powers_second, heap, FIRST, PITCH == 16 ? 0 : PITCH);
  }
  __device__ __forceinline__ void first(u64 (&v)[16], int, int) const {
    if constexpr (FIRST > 0) lazy::groups::group<FIRST, SKIP_FIRST>(v, powers_first);
  }
  __device__ __forceinline__ void second(u64 (&v)[16], u32 root, int, int) const {
    lazy::groups::group<4, SKIP_SECOND>(v, powers_second + root * PITCH);
  }
  static __device__ __forceinline__ u64 canonical(u64 x) { return GL64::canonical(x); }
};

// Runs layers lo + B1 + 3 down to lo of `rows` rows, in place in `work`. If `sources` is not null,
// this is the slab's first pass that runs, and it reads coefficient x mod 2^real of each row from
// there instead. A tile takes W = 2^(8 - B1) neighbouring positions at each of its strided offsets.
// lo >= 8 - B1 is not checked. A tile that reaches past the positions of a short code: see
// factor_at().
// Public source: Harvey's cache-friendly truncated FFT (arXiv:0810.3203, sections 2 and 3) works in
// place on an array of which only the first cells hold input and the others stand for zeros, on
// strided columns and then on rows; it is stated for one processor's cache.
// Coalesced access to strided data. The idea is public: Kim, Jung, Park and Ahn (arXiv:2012.01968),
// section VI.C, "Avoiding uncoalesced memory accesses in Kernel-1", which attributes the combining
// of thread blocks to prior work on the DFT (N. K. Govindaraju, B. Lloyd, Y. Dotsenko, B. Smith, J.
// Manferdelli, "High Performance Discrete Fourier Transforms on Graphics Processors", Proceedings
// of the 2008 ACM/IEEE Conference on Supercomputing, 2008,
// https://csd.uwo.ca/~mmorenom/CS433-CS9624/Resources/DiscreteFourierTransformsonGPUs.pdf).
// Template parameters, and on the host a switch over the instances of B1
// (SlabEncoder::upper_layers()) and a chain of tests over those of SKIP (pick()). The idea is
// public: Harris, "Optimizing Parallel Reduction in CUDA", slides 24 to 27. There the parameter is
// the block size.
// Occupancy tuning through the launch bounds. The idea is public: CUDA Programming Guide, "C/C++
// Language Extensions", "Launch Bounds".
// The compiler hints are public: __launch_bounds__: Best Practices Guide, "Occupancy"; the
// directive #pragma unroll: CUDA Programming Guide, "C/C++ Language Extensions", "#pragma unroll";
// on the loops over a thread's array, to make every index a constant: Milakov, "Static indexing".
template <int B1, int SKIP>
__global__ __launch_bounds__(ENCODE_THREADS, ENCODE_BLOCKS) void k_strided_pass(u64 *work, const RowSource *sources,
                                                                    const u64 *__restrict__ table, int log_n, int lo, int real,
                                                                    int rows) {
  constexpr int BITS = B1 + 4, LW = 8 - B1, W = 1 << LW;
  __shared__ u64 tile[4096];
  __shared__ PassFactors<B1, 16, SKIP> factors;
  const u32 tid = threadIdx.x, n = 1u << log_n, mask = (u32)(((u64)1 << real) - 1);
  const u32 group = blockIdx.x & ((1u << (lo - LW)) - 1), top = blockIdx.x >> (lo - LW);
  const u32 origin = (top << (lo + BITS)) | (group << LW);
  factors.load(table, log_n, lo + BITS - 1, top, real);
  const u32 g = tid >> 4, kl = tid & 15;
  const u32 jt2 = tid >> LW, kk2 = tid & (W - 1);
  u64 v[16];
  for (int r = blockIdx.y; r < rows; r += gridDim.y) {
    u64 *row = work + ((size_t)r << log_n);
    RowSource source;
    if (sources) source = sources[r];
    auto read = [&](u32 x) { return sources ? coefficient(source, x & mask).v : row[x & (n - 1)]; };
    if (B1 > 0) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        const u32 e = (j >> (4 - B1)) * 16 + g, kk = (j & ((1 << (4 - B1)) - 1)) * 16 + kl;
        v[j] = read(origin | (e << lo) | kk);
      }
      __syncthreads();
      factors.first(v, lo + BITS - 1, real);
#pragma unroll
      for (int j = 0; j < 16; j++) {
        const u32 e = (j >> (4 - B1)) * 16 + g, kk = (j & ((1 << (4 - B1)) - 1)) * 16 + kl;
        tile[e * W + kk] = v[j];
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 16; j++) v[j] = tile[(jt2 * 16 + j) * W + kk2];
    } else {
#pragma unroll
      for (int j = 0; j < 16; j++) v[j] = read(origin | (j << lo) | kk2);
      __syncthreads();
    }
    factors.second(v, jt2, lo + BITS - 1, real);
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const u32 x = origin | ((jt2 * 16 + j) << lo) | kk2;
      if (x < n) row[x] = v[j];
    }
  }
}

// Word e of a row's 256 is stored at index padded(e) = e + (e >> 4) of the row's ROW_ROOM words
// (similar to M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog,
// 2013, https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).
// Padding of a shared array against bank conflicts. The idea is public: M. Harris, S. Sengupta, J.
// D. Owens, GPU Gems 3, chapter 39, at Figure 39-5,
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda:
// every shared-array index i grows by i over the bank count (16 in that chapter). Here the divisor
// is the number of 8-byte elements that span all banks, which is also 16.
// Public source: Best Practices Guide, "Shared Memory in Matrix Multiplication (C=AAT)": padding
// that makes each row of a shared array one element longer. Here each root's sixteen powers, too,
// have a pitch of 17 words (LastFactors).
constexpr int ROW_ROOM = 256 + 16; // a row's 256 symbols and sixteen unused words
__device__ __forceinline__ u32 padded(u32 e) { return e + (e >> 4); }

// Runs layers 7 to 0 of rows 0 .. live - 1 at the 256 positions p = tile_number * 256 + i. On
// return symbols[row * ROW_ROOM + padded(i)] is the row's canonical symbol (below the modulus) at
// p, or 0 if row >= live. All threads of the thread block must call it: it has barriers inside. A
// row's sixteen threads are in one warp.
// __syncwarp: CUDA C++ Programming Guide, "Synchronization Functions". See also: NVIDIA, CUDA
// Programming Guide, "C/C++ Language Extensions", "Warp Synchronization Function",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
template <int SKIP> using LastFactors = PassFactors<4, 17, SKIP>;
template <int SKIP>
__device__ __forceinline__ void last_pass_symbols(u64 *symbols, LastFactors<SKIP> &factors, const u64 *work,
                                                  const RowSource *sources, const u64 *__restrict__ table, int log_n, int real,
                                                  int live, u32 tile_number) {
  const u32 tid = threadIdx.x, g = tid & 15, r = tid >> 4, origin = tile_number << 8;
  const u32 n_mask = (u32)(((u64)1 << log_n) - 1), mask = (u32)(((u64)1 << real) - 1);
  factors.load(table, log_n, 7, tile_number, real);
  const u64 *row = work + ((size_t)r << log_n);
  u64 *mine = symbols + r * ROW_ROOM;
  u64 v[16];
  if (r < live) {
    RowSource source;
    if (sources) source = sources[r];
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const u32 x = origin | (j * 16 + g);
      v[j] = sources ? coefficient(source, x & mask).v : row[x & n_mask];
    }
  }
  __syncthreads();
  if (r < live) {
    factors.first(v, 7, real);
#pragma unroll
    for (int j = 0; j < 16; j++) mine[padded(j * 16 + g)] = v[j];
  }
  __syncwarp();
  if (r < live) {
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = mine[padded(g * 16 + j)];
    factors.second(v, g, 7, real);
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 16; j++) mine[padded(g * 16 + j)] = r < live ? LastFactors<SKIP>::canonical(v[j]) : 0;
  __syncthreads();
}

// The last pass, then the hashing of words 16 slab .. 16 slab + 15 of each position's tree leaf.
// The hashed string is the label (blocks/hash/hash.h: the tag's word, the index), the column's
// symbols (one from each row), the copied symbols (of shorter codewords), zeros up to
// chain.salt_at, the salt, zeros up to a whole hash block. Row i of a launch is word 16 slab + i,
// but word LABEL_WORDS + i in launch 0, which begins with the label. chain.cv carries the chaining
// values between a tree's launches, which must run in the order of `slab`; its alignment to 16
// bytes is not checked.
// Public source: many independent calls of a hash on a GPU: Lu et al., "BatchZK", section 3.1.
// Template instances. The idea is public: Harris's slides (cited at k_strided_pass()).
// The compiler hints are public: __launch_bounds__ with two arguments: Best Practices Guide and
// CUDA Programming Guide; the directive #pragma unroll on the loops over a thread's arrays: CUDA
// Programming Guide and Milakov (all cited at k_strided_pass()).
// Loads and stores through a pointer cast to a wider type, here ulonglong2. The idea is public:
// "CUDA Pro Tip: Increase Performance with Vectorized Memory Access".
// The tile's padded index: see padded().
template <class Fn, int SKIP>
__global__ __launch_bounds__(ENCODE_THREADS, ENCODE_BLOCKS) void k_last_pass(const u64 *work, const RowSource *sources,
                                                                 const u64 *__restrict__ table, int log_n, int real, int live,
                                                                 u32 slab, LeafChainT<Fn> chain) {
  constexpr u32 S = Fn::BLKW / 2, CV = Fn::CVW / 2;
  static_assert(SLAB_ROWS % S == 0, "a slab is a whole number of the hash's blocks");
  __shared__ u64 symbols[SLAB_ROWS * ROW_ROOM];
  __shared__ LastFactors<SKIP> factors;
  last_pass_symbols<SKIP>(symbols, factors, work, sources, table, log_n, real, live, blockIdx.x);
  const u64 n = (u64)1 << log_n, x = ((u64)blockIdx.x << 8) + threadIdx.x;
  if (x >= n) return;
  const u32 words = chain.words(), calls = (words + S - 1) / S;
  const u64 index = n + x; // tree_index(log_n, x) of tree/tree.h
  const u64 short_x = short_position((unsigned)log_n, chain.log_short, x); // code/short_position.h
  constexpr u32 LEAD = (u32)LABEL_WORDS;
  const u32 ahead = slab ? 0 : LEAD; // a word's number within this launch minus its row's
  u64 cv[CV], block[S];
  u64 *kept = chain.cv + x * CV;
  if (slab == 0)
    leaf_begin<Fn>(cv, chain.tag);
  else {
#pragma unroll
    for (u32 i = 0; i < CV; i += 2) {
      const ulonglong2 two = *reinterpret_cast<const ulonglong2 *>(kept + i);
      cv[i] = two.x, cv[i + 1] = two.y;
    }
  }
#pragma unroll
  for (u32 h = 0; h < SLAB_ROWS / S; h++) {
    const u32 call = slab * (SLAB_ROWS / S) + h;
    if (call >= calls) return;
#pragma unroll
    for (u32 i = 0; i < S; i++) {
      const u32 word = call * S + i;
      const u32 symbol = word - LEAD; // wraps for the label's words, which the first test catches
      block[i] = word < LEAD                                  ? (word == 0 ? (u64)chain.tag.word : index)
                 : symbol < chain.width                       ? symbols[(h * S + i - ahead) * ROW_ROOM + padded(threadIdx.x)]
                 : symbol - chain.width < chain.copied_width ? __ldg(chain.copied + ((u64)(symbol - chain.width) << chain.log_short) + short_x)
                 : word - chain.salt_at < chain.salt_words   ? __ldg(chain.salt + x * chain.salt_words + (word - chain.salt_at))
                                                             : 0;
    }
    if (call + 1 == calls) {
      chain.leaves[x] = leaf_last<Fn>(cv, block, 8 * words);
      return;
    }
    leaf_step<Fn>(cv, block, call);
  }
#pragma unroll
  for (u32 i = 0; i < CV; i += 2) *reinterpret_cast<ulonglong2 *>(kept + i) = ulonglong2{cv[i], cv[i + 1]};
}

// The last pass with nothing hashed: the slab's symbols at all positions go to
// out[(first_row + row) * n + x].
// Template instances and __launch_bounds__ with two arguments: the ideas are public, see
// k_strided_pass(). The tile's padded index: see padded().
template <int SKIP>
__global__ __launch_bounds__(ENCODE_THREADS, ENCODE_BLOCKS) void k_last_pass_store(const u64 *work, const RowSource *sources,
                                                                              const u64 *__restrict__ table, int log_n, int real,
                                                                              int live, u64 *out, u64 first_row) {
  __shared__ u64 symbols[SLAB_ROWS * ROW_ROOM];
  __shared__ LastFactors<SKIP> factors;
  last_pass_symbols<SKIP>(symbols, factors, work, sources, table, log_n, real, live, blockIdx.x);
  const u64 n = (u64)1 << log_n, x = ((u64)blockIdx.x << 8) + threadIdx.x;
  if (x >= n) return;
  for (int row = 0; row < live; row++) out[((first_row + row) << log_n) + x] = symbols[row * ROW_ROOM + padded(threadIdx.x)];
}

// The last pass on the tiles of the listed positions only: thread block i writes the slab's symbols
// at position at[i] to out[i * width + first_word + row]. Nothing is hashed. at[i] < n is not
// checked.
// Public source: MegaBlocks (Gale, Narayanan, Young, Zaharia, arXiv:2211.15841, Section 5.1.3)
// stores a list in which thread blocks look up which block they compute (there a sparse matrix's
// nonzero blocks, here positions).
// Public source: Abdelfattah, Haidar, Tomov and Dongarra (Section 3) give each product of a batch
// an equal part of the grid.
// Template instances and __launch_bounds__ with two arguments: the ideas are public, see
// k_strided_pass(). The tile's padded index: see padded().
template <int SKIP>
__global__ __launch_bounds__(ENCODE_THREADS, ENCODE_BLOCKS) void k_last_pass_gather(const u64 *work, const RowSource *sources,
                                                                               const u64 *__restrict__ table, int log_n, int real,
                                                                               int live, const u64 *at, u64 *out, u64 width,
                                                                               u64 first_word) {
  __shared__ u64 symbols[SLAB_ROWS * ROW_ROOM];
  __shared__ LastFactors<SKIP> factors;
  const u64 x = at[blockIdx.x];
  last_pass_symbols<SKIP>(symbols, factors, work, sources, table, log_n, real, live, (u32)(x >> 8));
  if ((int)threadIdx.x < live) out[blockIdx.x * width + first_word + threadIdx.x] = symbols[threadIdx.x * ROW_ROOM + padded((u32)x & 255)];
}

// ---- the host's side ----
// One bit per kernel, set when the host picks it. `kind`: 0 is a strided pass, 1 to 3 the last pass
// of leaves(), store(), gather(). A strided pass of B1 + 4 layers (`beyond` = B1) has B1 + 4
// kernels, one per SKIP; each kind of last pass has nine.
inline std::atomic<u64> &picked_bits() {
  static std::atomic<u64> bits{0};
  return bits;
}
constexpr int STRIDED_KERNELS = 4 + 5 + 6 + 7 + 8, LAST_KERNELS = 9;
constexpr int kernel_bit(int kind, int beyond, int skip) {
  return kind ? STRIDED_KERNELS + (kind - 1) * LAST_KERNELS + skip : beyond * 4 + beyond * (beyond - 1) / 2 + skip;
}
static_assert(kernel_bit(0, 4, 7) == STRIDED_KERNELS - 1 && kernel_bit(0, 1, 0) == 4 && kernel_bit(3, 0, 8) == STRIDED_KERNELS + 3 * LAST_KERNELS - 1 &&
                  STRIDED_KERNELS + 3 * LAST_KERNELS <= 64,
              "every kernel has a bit of its own");
inline void note_kernel(int kind, int beyond, int skip) { picked_bits().fetch_or((u64)1 << kernel_bit(kind, beyond, skip), std::memory_order_relaxed); }
ENCODER_FN size_t SlabEncoder::kernels(int kind) { return !SKIP_IS_COMPILED ? (kind ? 1 : 5) : kind ? LAST_KERNELS : STRIDED_KERNELS; }
ENCODER_FN size_t SlabEncoder::picked(int kind) {
  const int from = kernel_bit(kind, 0, 0), count = kind ? LAST_KERNELS : STRIDED_KERNELS;
  return (size_t)__builtin_popcountll((picked_bits().load(std::memory_order_relaxed) >> from) & (((u64)1 << count) - 1));
}
ENCODER_FN const char *SlabEncoder::made() {
#ifdef ENGINE_ENCODER_OBJECT
  return "one object that several programs link";
#else
  return "compiled with this program";
#endif
}

inline unsigned tiles_of(unsigned log_n) { return log_n > 8 ? 1u << (log_n - 8) : 1; }
inline int live_of(size_t rows, u64 from) { return from >= rows ? 0 : rows - from < SLAB_ROWS ? (int)(rows - from) : SLAB_ROWS; }
// A slab's number of real layers: the smallest `top` with every row's coefficients at indices below
// 2^top. Layers from `top` up only copy and are not computed. A shorter row is zero above its own
// coefficients. The kernels rely on top <= log_n, which nothing in this file checks.
inline unsigned top_of(const std::vector<RowSource> &rows, u64 from, int live) {
  unsigned top = 0;
  for (int i = 0; i < live; i++) {
    const RowSource &s = rows[from + i];
    const u64 end_m = s.ell ? s.message_at + s.ell : 0, end_t = s.r ? s.tail_at + s.r : 0;
    const unsigned need = ceil_log2(end_m > end_t ? end_m : end_t);
    if (need > top) top = need;
  }
  return top;
}
template <int S, int MOST, class F> void pick(int skip, F f) {
  if constexpr (S == MOST)
    f(std::integral_constant<int, S>{});
  else if (skip <= S)
    f(std::integral_constant<int, S>{});
  else
    pick<S + 1, MOST>(skip, f);
}
// Calls f(std::integral_constant<int, s>{}): s = skip clamped to 0 .. MOST.
template <int MOST, class F> void with_skip(int skip, F f) {
  if constexpr (SKIP_IS_COMPILED)
    pick<0, MOST>(skip, f);
  else
    f(std::integral_constant<int, 0>{});
}
// Launches one strided pass, layers lo to lo + B1 + 3. Requires lo < real, which is not checked:
// SKIP is at most B1 + 3, so layer lo is always computed.
template <int B1>
void strided(u64 *work, dim3 grid, const RowSource *from, const u64 *table, int log_n, int lo, int real, int live, cudaStream_t stream) {
  with_skip<B1 + 3>(lo + B1 + 4 - real, [&](auto skip) {
    note_kernel(0, B1, decltype(skip)::value);
    k_strided_pass<B1, decltype(skip)::value><<<grid, ENCODE_THREADS, 0, stream>>>(work, from, table, log_n, lo, real, live);
  });
}

// leaves(), store() and gather() rely on `stream` keeping order with stream 0, on which Card
// (device.cuh) fills their buffers and, without an arena, allocates and frees them. They do not
// check it; EngineGpu's constructor (engine_gpu.cuh) does. A failed BLK_REQUIRE or ENGINE_CK ends
// the process.
ENCODER_FN void SlabEncoder::leaves(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, LeafChain chain, cudaStream_t stream) {
  BLK_REQUIRE(chain.width == rows.size() && log_n <= 30, "encoder: the column's width is not the number of rows, or too long a code");
  const u64 *table = tables.of(log_n);
  rows_.from_host(rows.data(), rows.size());
  BLK_REQUIRE(chain.salt_at >= LABEL_WORDS + chain.width + chain.copied_width, "encoder: a salt that begins among the symbols");
  BLK_REQUIRE(!chain.copied_width || (chain.copied && chain.log_short <= log_n), "encoder: copied symbols without their words, or of a longer code");
  const u64 slabs = ((u64)chain.words() + SLAB_ROWS - 1) / SLAB_ROWS;
  if (slabs > 1) cv_.resize((size_t)(HashFn::CVW / 2) << log_n);
  chain.cv = cv_.get();
  for (u64 k = 0; k < slabs; k++) {
    // Launch 0 begins with the label (see k_last_pass()), so it takes LABEL_WORDS rows fewer, and
    // each later launch begins that many rows earlier: rows[j] is word LABEL_WORDS + j.
    const u64 from = k ? k * SLAB_ROWS - LABEL_WORDS : 0;
    const int all = live_of(rows.size(), from), live = !k && all > SLAB_ROWS - (int)LABEL_WORDS ? SLAB_ROWS - (int)LABEL_WORDS : all;
    const int real = (int)top_of(rows, from, live);
    const RowSource *sources = rows_.get() + from;
    const bool passed = upper_layers(sources, live, (int)log_n, real, table, stream);
    with_skip<8>(8 - real, [&](auto skip) {
      note_kernel(1, 0, decltype(skip)::value);
      k_last_pass<HashFn, decltype(skip)::value><<<tiles_of(log_n), ENCODE_THREADS, 0, stream>>>(
          work_.get(), passed ? nullptr : sources, table, (int)log_n, real, live, (u32)k, chain);
    });
    ENGINE_CK(cudaGetLastError());
  }
}


ENCODER_FN void SlabEncoder::store(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, u64 *out_on_card, cudaStream_t stream) {
  BLK_REQUIRE(log_n <= 30, "encoder: too long a code");
  const u64 *table = tables.of(log_n);
  rows_.from_host(rows.data(), rows.size());
  for (u64 from = 0; from < rows.size(); from += SLAB_ROWS) {
    const int live = live_of(rows.size(), from), real = (int)top_of(rows, from, live);
    const bool passed = upper_layers(rows_.get() + from, live, (int)log_n, real, table, stream);
    with_skip<8>(8 - real, [&](auto skip) {
      note_kernel(2, 0, decltype(skip)::value);
      k_last_pass_store<decltype(skip)::value><<<tiles_of(log_n), ENCODE_THREADS, 0, stream>>>(
          work_.get(), passed ? nullptr : rows_.get() + from, table, (int)log_n, real, live, out_on_card, from);
    });
    ENGINE_CK(cudaGetLastError());
  }
}


ENCODER_FN void SlabEncoder::gather(const std::vector<RowSource> &rows, unsigned log_n, CardTable &tables, const u64 *at_on_card, size_t listed,
                                    u64 *out_on_card, cudaStream_t stream) {
  BLK_REQUIRE(log_n <= 30, "encoder: too long a code");
  const u64 *table = tables.of(log_n);
  rows_.from_host(rows.data(), rows.size());
  for (u64 from = 0; from < rows.size(); from += SLAB_ROWS) {
    const int live = live_of(rows.size(), from), real = (int)top_of(rows, from, live);
    const bool passed = upper_layers(rows_.get() + from, live, (int)log_n, real, table, stream);
    with_skip<8>(8 - real, [&](auto skip) {
      note_kernel(3, 0, decltype(skip)::value);
      k_last_pass_gather<decltype(skip)::value><<<(unsigned)listed, ENCODE_THREADS, 0, stream>>>(
          work_.get(), passed ? nullptr : rows_.get() + from, table, (int)log_n, real, live, at_on_card, out_on_card, rows.size(), from);
    });
    ENGINE_CK(cudaGetLastError());
  }
}


// Runs a slab's strided passes: afterwards work_ contains its rows after all layers from 8 up.
// Layers 8 to log_n - 1 are split into the fewest passes of 4 to 8 layers, of nearly equal size. If
// log_n < 12, the one pass has layers 8 to 11, of which those from log_n up only copy. Returns
// whether a pass ran; if none did (real <= 8, or no row), the last pass reads `sources`.
// The switch over the instances of strided(): the idea is public, see k_strided_pass(). A pass
// whose layers all only copy is left out; the next one then reads `sources`. Public source: see
// k_strided_pass().
ENCODER_FN bool SlabEncoder::upper_layers(const RowSource *sources, int live, int log_n, int real, const u64 *table, cudaStream_t stream) {
  if (real <= 8 || !live) return false;
  work_.resize((size_t)SLAB_ROWS << log_n);
  const int count = (log_n - 8 + 7) / 8;
  bool first = true;
  for (int k = 0, top = log_n; k < count; k++) {
    int bits = (top - 8 + (count - k - 1)) / (count - k);
    if (bits < 4) bits = 4;
    top = top - bits > 8 ? top - bits : 8;
    if (top >= real) continue; // the pass would only copy
    const u32 tiles = log_n > 12 ? 1u << (log_n - 12) : 1;
    const dim3 grid(tiles, tiles >= 2048 ? 1 : 2048 / tiles < (u32)live ? 2048 / tiles : live);
    const RowSource *from = first ? sources : nullptr;
    switch (bits) {
    case 4: strided<0>(work_.get(), grid, from, table, log_n, top, real, live, stream); break;
    case 5: strided<1>(work_.get(), grid, from, table, log_n, top, real, live, stream); break;
    case 6: strided<2>(work_.get(), grid, from, table, log_n, top, real, live, stream); break;
    case 7: strided<3>(work_.get(), grid, from, table, log_n, top, real, live, stream); break;
    default: strided<4>(work_.get(), grid, from, table, log_n, top, real, live, stream); break;
    }
    ENGINE_CK(cudaGetLastError());
    first = false;
  }
  return !first;
}

} // the inline namespace
} // namespace blk
