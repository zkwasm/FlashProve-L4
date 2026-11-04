// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The last m layers of the transform that encodes a weight table's rows, on one run of height m (an
// aligned block of 2^m codeword positions), in place: a second implementation of hf::run_encode(),
// written for speed, for m = 12 to 30.
// prover/hash_commitment/hf_code.cuh defines runs, folds, layers and factors (twiddle factors);
// include it first. prover/hash_commitment/tests/hf_commit_table_test.cu holds the two
// implementations' results against each other through the tree's digests (by a reading of the
// code, at seven heights between 12 and 20); no test of this repository compares them word for
// word.
// This one needs no table of factors for each run (see PartFactors) and takes up to four
// layers as one transform of up to sixteen points (prover/field/transform_groups_gl64.cuh).
// That file, at powers(), describes a compiler fault: reading the source cannot show that a build
// of these kernels is correct, and every instantiation must be compared with hf::run_encode() on a
// run other than the first. (The fault does not show where a factor is 1; on the cosets used here
// no factor is 1, in any run.)
// Use: small_table() once, part_factors() for each run, then code_groups::run_encode(). A rejected
// argument ends the process (hf::code_refuse(): exit code 4).
//
// The passes over GPU memory (see `Passes`) are those of k_strided_pass and k_last_pass in
// prover/engine/encode_kernels.cuh. The techniques' public sources: below, at each routine, and in
// full at that file's head; those of the transform of sixteen points:
// prover/field/transform_groups_gl64.cuh.
// Below, the titles, headings and addresses of these works are unchecked: NVIDIA's documents,
// except the slides cited at k_strided_pass and the last two web pages cited at `PartFactors`;
// GPU Gems 3; and the last two papers cited at `Passes`.
//
// Hidden values: the rows. No address or loop count here depends on a row's words, and no branch,
// unless the compiler makes one of the conditional expression in blk::GL64::canonical().
#pragma once
#include "field/transform_groups_gl64.cuh"
namespace hf {
namespace code_groups {
namespace gr = blk::lazy::groups;
constexpr int THREADS = 256, SLAB_ROWS = 16;
constexpr int LEAST_HEIGHT = 12, MOST_HEIGHT = 30; // a tile has 2^12 words; positions are 32-bit numbers
// A run's factors (`Part`: the name for a run in the committer program cp_commit).
// Layer L < m replaces each pair (a, b) of words 2^L apart by (a + t b, a - t b). It has 2^d
// blocks, d = m - 1 - L (its depth). Block y = (position within the run) >> (L + 1) has the factor
// t = z^(2^L), where z is the point of position I = h 2^m + y 2^(L + 1) and h is the run's number.
// That point is a power of 7 that depends only on the coset's number, I >> s, times the product of
// root[i] over the set bits i < s of I (s and root are members of `Domain`, hf_code.cuh), and
// root[i]^(2^L) = root[i - L]. In I the bits of y are below those of h and, since m <= s, below
// bit s. So
//     t = scalar[d] Omega[y],   Omega[y] = the product of root[k + 1] over the set bits k of y,
// where scalar[d] = run_table_entry(D, m, h, 2^d) is the factor of the layer's first block. Omega
// does not depend on the run, the layer or the code's length, and its first 2^d entries serve the
// layer of depth d. So one table of 2^(m - 1) words serves all runs, and a run adds m scalars, in
// place of the 2^m entries that hf::run_table() computes for each run.
// A layer's butterfly is that of Longa and Naehrig, ePrint 2016/504, Algorithm 1 (cited at
// k_small_table).
// k_strided_pass and k_last_pass take a PartFactors (248 bytes) as a __grid_constant__ parameter,
// so that no thread makes a copy of it when load_heap() indexes it by a number known only at run
// time.
// Public source: __grid_constant__ on a kernel's parameter: NVIDIA, CUDA C++ Programming Guide,
// release 12.8, "__grid_constant__", "Function Parameters".
// See also: NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions",
// "__grid_constant__ Parameters",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html;
// "C++ Language Support", "__global__ Function Parameters",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html;
// "Writing SIMT Kernels", "Local Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html.
struct PartFactors {
  const u64 *omega;        // small_table()'s, GPU memory
  u64 scalar[MOST_HEIGHT]; // [d], d < m
};
inline size_t small_table_words(int m) {
  return (size_t)1 << (m - 1);
}
// Computes Omega, the powers of a root of unity in bit-reversed order, of which every layer reads a
// prefix: for y < 2^(m - 1), Omega[y] = root[m - 1]^(the m - 1 bits of y, reversed).
// Launch, not checked: at least count / 256 blocks of 256 threads.
// Public source: the decimation-in-time transform with factors in bit-reversed order: Patrick Longa
// and Michael Naehrig, "Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based
// Cryptography", IACR ePrint 2016/504, section 2.2 and Algorithm 1 (Function NTT based on the
// Cooley-Tukey butterfly; there for the negacyclic transform), https://eprint.iacr.org/2016/504.
// The compiler hint is public: __launch_bounds__: NVIDIA, CUDA C++ Best Practices Guide,
// "Occupancy", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/.
__global__ void __launch_bounds__(256) k_small_table(const Domain D, u64 count, u64 *__restrict__ omega) {
  const u64 y = (u64)blockIdx.x * 256 + threadIdx.x;
  if (y >= count) return;
  u64 v = 1;
  for (int k = 0; (y >> k) != 0; k++)
    if ((y >> k) & 1) v = code_mul(v, D.root[k + 1]);
  omega[y] = v;
}
// Fills omega[y] = Omega[y] for y < 2^(m - 1). The table serves runs of any height up to m, in any
// code whose cosets have at least 2^m points.
static inline void small_table(const Domain &D, int m, u64 *omega, cudaStream_t st) {
  if (m < LEAST_HEIGHT || m > MOST_HEIGHT || m > most_height(D)) code_refuse("small_table: a run's height must be from 12 to 30, and within one coset");
  const u64 count = small_table_words(m);
  k_small_table<<<(unsigned)((count + 255) / 256), 256, 0, st>>>(D, count, omega);
  CK(cudaGetLastError());
}
// Returns the factors of run h of height m. On the host, it also compares scalar[d] Omega[y] with
// run_table_entry() at up to four blocks y of every layer and exits with an error if they differ:
// the formula depends on the form of the code's points. It computes Omega[y] from D.root and does
// not read `omega`. Not checked: D.root[3] = 2^12, which the shifts in gr::group() assume and
// domain() gives.
static inline PartFactors part_factors(const Domain &D, int m, u64 h, const u64 *omega) {
  if (m < LEAST_HEIGHT || m > MOST_HEIGHT || m > most_height(D)) code_refuse("part_factors: a run's height must be from 12 to 30, and within one coset");
  if (m < D.ln ? (h >> (D.ln - m)) != 0 : h != 0) code_refuse("part_factors: the run's number must be below 2^(ln - m)");
  PartFactors f{};
  f.omega = omega;
  for (int d = 0; d < m; d++) {
    f.scalar[d] = run_table_entry(D, m, h, (u64)1 << d);
    for (u64 y : {(u64)1, ((u64)1 << d) - 1, ((u64)1 << d) / 3, ((u64)1 << d) / 2}) {
      if (y >> d) continue;
      u64 o = 1;
      for (int k = 0; (y >> k) != 0; k++)
        if ((y >> k) & 1) o = code_mul(o, D.root[k + 1]);
      if (run_table_entry(D, m, h, ((u64)1 << d) + y) != code_mul(f.scalar[d], o))
        code_refuse("part_factors: a factor of the run is not its layer's scalar times the small table's entry");
    }
  }
  return f;
}
// Fills heap[1 .. count - 1] (shared memory) with the factors of a tile (the positions that one
// thread block takes):
//     heap[2^e + x] = the factor of block (top << e) + x of layer `widest` - e, for x < 2^e,
// each computed as scalar[d] Omega[y] and canonical (below the field's modulus).
// Public source: a thread block's twiddle factors preloaded into shared memory: S. Kim, W. Jung,
// J. Park, J. H. Ahn, "Accelerating Number Theoretic Transformations for Bootstrappable Homomorphic
// Encryption on GPUs", arXiv 2012.01968, Section VI-C, in one of their two kernels (Kernel-1).
// load_heap() does the same in every pass, and computes each factor on loading.
// Public source: the order of `heap`: Longa and Naehrig, ePrint 2016/504 (cited at k_small_table),
// Algorithm 1.
// Public source: the order of `heap`: Jongmin Kim, Wonseok Choi and Jung Ho Ahn, "Cheddar: A Swift
// Fully Homomorphic Encryption Library for CUDA GPUs", arXiv 2407.13055 (2024), section 4.1
// "Number-theoretic transform (NTT)", https://arxiv.org/abs/2407.13055.
// What differs here: the heap contains one tile's factors only.
// __clz: NVIDIA, CUDA Math API, "Integer Intrinsics".
__device__ __forceinline__ void load_heap(u64 *heap, const PartFactors &f, int m, int widest, u32 top, int count) {
  for (int x = threadIdx.x; x < count; x += THREADS)
    if (x) {
      const int e = 31 - __clz(x), d = m - 1 - (widest - e);
      using namespace blk::lazy;
      heap[x] = canonical(mul(Loose{f.scalar[d]}, Loose{__ldg(f.omega + ((top << e) + (x - (1u << e))))})).v;
    }
}
// A strided pass: layers lo + B1 + 3 .. lo of all rows, in place; lo >= 8. A thread block takes a
// tile of 4,096 positions: 2^(B1 + 4) offsets e at the stride 2^lo, times W = 2^(8 - B1)
// neighbouring positions kk. The first __syncthreads() in the row loop orders powers() before
// group(), and one row's reads of `tile` before the next row's writes.
// Launch, not checked: 2^(m - 12) by any number of blocks of THREADS threads; lo + B1 + 4 <= m.
// Public source: strided columns in place: David Harvey, "A cache-friendly truncated FFT",
// arXiv:0810.3203 (2008), section 2 "Notation and setup" and section 3 "A cache-friendly TFT"
// (Algorithm 1 and the proof of Theorem 1), https://arxiv.org/abs/0810.3203. There only a prefix of
// the array is input, and the method is stated for one processor's cache; here every word of a row
// is input.
// The idea is public: a template parameter (here B1) and a switch on the host over its instances
// (run_encode()): M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology,
// slides 24 to 27, https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// There the parameter is the block size.
// The idea is public: occupancy tuning through the launch bounds (minBlocksPerMultiprocessor is 3
// here): CUDA Programming Guide (cited at `PartFactors`), "C/C++ Language Extensions",
// "Launch Bounds".
// The compiler hint is public: __launch_bounds__: Best Practices Guide (cited at k_small_table),
// "Occupancy".
// The compiler hint is public: #pragma unroll: CUDA Programming Guide, "C/C++ Language Extensions",
// "#pragma unroll".
// The compiler hint is public: #pragma unroll on the loops over a thread's array, to make every
// index a constant: M. Milakov, "Fast Dynamic Indexing of Private Arrays in CUDA", NVIDIA Technical
// Blog, 2015, part "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
template <int B1>
__global__ void __launch_bounds__(THREADS, 3) k_strided_pass(u64 *cw, const __grid_constant__ PartFactors f, int m, int lo, int rows) {
  constexpr int BITS = B1 + 4, LW = 8 - B1, W = 1 << LW;
  __shared__ u64 tile[4096];
  __shared__ u64 pw1[16], pw2[16 << B1], heap[16 << B1];
  const u32 tid = threadIdx.x;
  const u32 grp = blockIdx.x & ((1u << (lo - LW)) - 1), top = blockIdx.x >> (lo - LW);
  const u32 origin = (top << (lo + BITS)) | (grp << LW);
  load_heap(heap, f, m, lo + BITS - 1, top, 16 << B1);
  __syncthreads();
  if constexpr (B1 > 0) gr::powers<B1, 0>(pw1, heap, 0);
  gr::powers<4, 0>(pw2, heap, B1);
  const u32 g = tid >> 4, kl = tid & 15;
  const u32 jt2 = tid >> LW, kk2 = tid & (W - 1);
  u64 v[16];
  for (int r = blockIdx.y; r < rows; r += gridDim.y) {
    u64 *row = cw + ((size_t)r << m);
    if constexpr (B1 > 0) {
#pragma unroll
      for (int j = 0; j < 16; j++) {
        const u32 e = (j >> (4 - B1)) * 16 + g, kk = (j & ((1 << (4 - B1)) - 1)) * 16 + kl;
        v[j] = row[origin | (e << lo) | kk];
      }
      __syncthreads();
      gr::group<B1, 0>(v, pw1);
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
      for (int j = 0; j < 16; j++) v[j] = row[origin | (j << lo) | kk2];
      __syncthreads();
    }
    gr::group<4, 0>(v, pw2 + jt2 * 16);
#pragma unroll
    for (int j = 0; j < 16; j++) row[origin | ((jt2 * 16 + j) << lo) | kk2] = v[j];
  }
}
// In shared memory a row's 256 words of a tile have one word of padding after every sixteen,
// against bank conflicts: word i is kept at index padded(i) of the row's ROW_ROOM words (similar to
// M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical Blog, 2013).
// The idea is public: padded() adds i / 16 to index i, and there the 16 counts the banks: M. Harris,
// S. Sengupta, J. D. Owens, GPU Gems 3, chapter 39, at Figure 39-5,
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda.
// Here the divisor is the number of 64-bit words that span all banks, which is also 16.
__device__ __forceinline__ u32 padded(u32 i) {
  return i + (i >> 4);
}
constexpr int ROW_ROOM = 256 + 16;
// The last pass: layers 7 .. 0 of the first `live` <= SLAB_ROWS rows at cw, in place; it writes
// canonical words. A thread block takes 256 neighbouring positions of all these rows; the sixteen
// threads that share a row are in one warp. Launch, not checked: 2^(m - 8) blocks of THREADS
// threads.
// __syncwarp: CUDA C++ Programming Guide (cited at `PartFactors`), "Synchronization Functions".
// See also: CUDA Programming Guide (cited at `PartFactors`), "C/C++ Language Extensions",
// "Warp Synchronization Function".
// pw2's rows of sixteen words are 17 words apart, against bank conflicts. Public source: a shared
// array whose rows each have room for one entry more than they hold: Best Practices Guide (cited at
// k_small_table), "Shared Memory in Matrix Multiplication (C=AAT)".
// The compiler hints are public: __launch_bounds__ with two arguments (Best Practices Guide, CUDA
// Programming Guide) and #pragma unroll on the loops over a thread's array (CUDA Programming Guide,
// Milakov), all cited at k_strided_pass. The tile's padded index: see padded().
__global__ void __launch_bounds__(THREADS, 3) k_last_pass(u64 *cw, const __grid_constant__ PartFactors f, int m, int live) {
  __shared__ u64 symbols[SLAB_ROWS * ROW_ROOM];
  __shared__ u64 pw1[16], pw2[16 * 17], heap[256];
  const u32 tid = threadIdx.x, g = tid & 15, r = tid >> 4, top = blockIdx.x, origin = top << 8;
  load_heap(heap, f, m, 7, top, 256);
  __syncthreads();
  gr::powers<4, 0>(pw1, heap, 0);
  gr::powers<4, 0>(pw2, heap, 4, 17);
  const u64 *row = cw + ((size_t)r << m);
  u64 *mine = symbols + r * ROW_ROOM;
  u64 v[16];
  if ((int)r < live) {
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = row[origin | (j * 16 + g)];
  }
  __syncthreads();
  if ((int)r < live) {
    gr::group<4, 0>(v, pw1);
#pragma unroll
    for (int j = 0; j < 16; j++) mine[padded(j * 16 + g)] = v[j];
  }
  __syncwarp();
  if ((int)r < live) {
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = mine[padded(g * 16 + j)];
    gr::group<4, 0>(v, pw2 + g * 17);
  }
  __syncwarp();
  if ((int)r < live) {
#pragma unroll
    for (int j = 0; j < 16; j++) mine[padded(g * 16 + j)] = blk::GL64::canonical(v[j]);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 16; i++)
    if (i < live) cw[((size_t)i << m) + origin + tid] = symbols[i * ROW_ROOM + padded(tid)];
}
// The passes over GPU memory for a height m: strided pass k < `count` runs layers
// lo[k] + bits[k] - 1 .. lo[k], 4 to 8 of them; the last pass then runs layers 7 .. 0 (four passes
// at m = 27). A pass runs at most two groups of layers (calls of gr::group()), with one exchange
// through shared memory between them; hf::run_encode()'s kernels go through shared memory after at
// most three layers. Between two groups and between two passes a value is one 64-bit word that need
// not be canonical.
// passes_of() expects LEAST_HEIGHT <= m <= MOST_HEIGHT and does not check it.
// Public source: layers grouped into passes over GPU memory: Kim, Jung, Park and Ahn, arXiv
// 2012.01968 (cited at load_heap()), sections V and VI.
// Public source: layers grouped into passes over GPU memory: A. Ozcan, E. Savas, "Two Algorithms
// for Fast GPU Implementation of NTT", ePrint 2023/1410, section V.
// Public source: a number-theoretic transform on a GPU: O. Ozerk, C. Elgezen, A. C. Mert,
// E. Ozturk, E. Savas, "Efficient Number Theoretic Transform Implementation on GPU for Homomorphic
// Encryption", ePrint 2021/124.
// Public source: D. Harvey, "Faster arithmetic for number-theoretic transforms", arXiv 1205.2926:
// words unreduced between the layers, after that paper.
struct Passes {
  int count;
  int lo[3], bits[3];
};
inline Passes passes_of(int m) {
  Passes p{};
  p.count = (m - 8 + 7) / 8;
  for (int k = 0, top = m; k < p.count; k++) {
    p.bits[k] = (top - 8 + (p.count - k - 1)) / (p.count - k);
    top -= p.bits[k];
    p.lo[k] = top;
  }
  return p;
}
// The same result as hf::run_encode(cw, tw, m, rows, st) with tw the table of the run of f. cw
// contains `rows` rows of 2^m words: on entry each row's fold, on return its code symbols on the
// run. Input words need not be canonical; the output is. Not checked: that f comes from
// part_factors() with this m, and that f.omega contains small_table()'s words, for a height of at
// least m, by the time the kernels run.
// The idea is public: the switch over k_strided_pass's instances: Harris (cited at k_strided_pass).
static inline void run_encode(u64 *cw, const PartFactors &f, int m, int rows, cudaStream_t st) {
  if (rows <= 0) return;
  if (m < LEAST_HEIGHT || m > MOST_HEIGHT) code_refuse("code_groups::run_encode: a run's height must be from 12 to 30");
  const Passes p = passes_of(m);
  const u32 tiles = 1u << (m - 12);
  const dim3 grid(tiles, tiles >= 2048 ? 1 : std::min<u32>(2048 / tiles, (u32)rows));
  for (int k = 0; k < p.count; k++) {
    switch (p.bits[k]) {
    case 4: k_strided_pass<0><<<grid, THREADS, 0, st>>>(cw, f, m, p.lo[k], rows); break;
    case 5: k_strided_pass<1><<<grid, THREADS, 0, st>>>(cw, f, m, p.lo[k], rows); break;
    case 6: k_strided_pass<2><<<grid, THREADS, 0, st>>>(cw, f, m, p.lo[k], rows); break;
    case 7: k_strided_pass<3><<<grid, THREADS, 0, st>>>(cw, f, m, p.lo[k], rows); break;
    case 8: k_strided_pass<4><<<grid, THREADS, 0, st>>>(cw, f, m, p.lo[k], rows); break;
    default: code_refuse("code_groups::run_encode: internal error: a pass of fewer than 4 or more than 8 layers");
    }
    CK(cudaGetLastError());
  }
  for (int r0 = 0; r0 < rows; r0 += SLAB_ROWS) {
    k_last_pass<<<1u << (m - 8), THREADS, 0, st>>>(cw + ((size_t)r0 << m), f, m, std::min(SLAB_ROWS, rows - r0));
    CK(cudaGetLastError());
  }
}
} // namespace code_groups
} // namespace hf
