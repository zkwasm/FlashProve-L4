// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Parts of this file are adapted from ggml (llama.cpp), which is under the MIT License. That project's notice, which
// applies to those parts:
// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Decodes Q4_K and Q5_K blocks (super-blocks of 256 weights in GGUF model files) on the GPU.
//   decode_planes()     decodes a window of a tensor's blocks into two planes of signed bytes
//   blocks_to_planes()  decodes an entire matrix of blocks in the same way
//   blocks_to_cells()   writes the same bytes transposed
//   transpose_bytes()   transposes a matrix of bytes
// prover/weights_blocks/fold_rule.h defines a weight's committed integer w, which the committer
// program cp_commit commits to and the prover program cp_prove computes with. This file does the
// same binary32 operations, each rounded to nearest (RN): v = RN(RN(RN(d * sc) * q) - RN(dmin * m))
// and RN(G * v), rounded to the nearest integer (ties to even) to give w. d and dmin are the
// block's two factors, sc and m the sub-block's scale and minimum, q the weight's 4- or 5-bit code.
// G = 2^e * gamma_j, with the tensor's exponent e and column j's multiplier gamma_j. The planes
// hold lo and hi of w = 256 * hi + lo, -128 <= lo <= 127. No function here checks what its comment
// requires, or that w fits in 16 bits (see weight()). Includers must first define the macro CK().
// The file also decodes the wider types Q8_0, F16 and BF16, in runs of 256 weights: for them decode_planes(),
// blocks_to_planes() and blocks_to_cells() launch k_decode_planes_wide or k_cells_wide, where v = RN(d q) or the stored
// value itself. An int 'type' is a wb::BlockType (weights_blocks/blocks.h): 0 Q4_K, 1 Q5_K, 4 Q8_0, 5 F16, 6 BF16.
//
// Public source: the decoding in this file is adapted from ggml's get_scale_min_k4 and
// dequantize_block_q4_K, _q5_K (ggml/src/ggml-cuda/convert.cu). ggml is the tensor library of the
// llama.cpp project (github.com/ggml-org/llama.cpp;
// MIT licence, "Copyright (c) 2023-2024 The ggml authors"; the notice is in NOTICE and THIRD_PARTY.md).
// The byte layout of the block formats Q4_K and Q5_K, and the rule that decodes them, are ggml's:
// ggml/src/ggml-common.h (block_q4_K, block_q5_K) and ggml/src/ggml-quants.c (dequantize_row_q4_K,
// _q5_K); the 12 scale bytes are drawn as a diagram on the project's wiki page "Tensor Encoding
// Schemes".
// Public source: Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf), "Quantization Types":
// Q4_K, Q5_K: super-blocks of 8 blocks of 32 weights with 6-bit scales and minima.
// Public source: NVIDIA, CUDA Math API Reference Manual (https://docs.nvidia.com/cuda/cuda-math-api/):
// __half2float; __fmul_rn, __fsub_rn (each gives the nearest float, the even one at a tie, and is
// always rounded by itself, as no part of a fused multiply-add); __float2int_rn.
// Public source: M. Harris, "An Efficient Matrix Transpose in CUDA C/C++" (NVIDIA Technical Blog,
// 2013): the shared-memory tile is wider than its data (k_cells: PITCH; k_transpose_bytes: 68).
// See also: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// Public source: aligned loads of 128 and 32 bits (k_decode_planes): "CUDA Pro Tip: Increase
// Performance with Vectorized Memory Access", NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// (The post joins words into wider ones; here four single bytes also become one word.)
// Public source: few divisions, unrolled loops: NVIDIA, CUDA C++ Best Practices Guide,
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/: "Division Modulo Operations" (in
// k_decode_planes two divisors for a warp's task, none for a row), "Control Flow Instructions"
// (#pragma unroll). See also, in the same guide, "Occupancy" (__launch_bounds__ with one argument).
// Public source: task loops striding by warps or thread blocks (the post: by threads): M. Harris,
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// Public source: code moved out of the loop over rows, by hand: LLVM Project, "LLVM's Analysis and
// Transform Passes", "licm: Loop Invariant Code Motion", https://llvm.org/docs/Passes.html.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace qblocks {
template <int Q5> struct Blk {
  static constexpr int BB = Q5 ? 176 : 144, QS = Q5 ? 48 : 16; // BB: bytes per block; QS: offset of its 128 code bytes
};
// Blk<Q5> (above): a block's layout constants. In every template here Q5 is 1 for Q5_K, 0 for Q4_K.
// A block's header (`head`) is its first 16 bytes, loaded as one 128-bit word: d and dmin
// (binary16) in head.x, then the twelve bytes that pack the 6-bit scales and minima of the eight
// sub-blocks of 32 weights, in head.y, head.z and head.w, the lowest byte first.
// Extracts the scale sc and minimum m of sub-block j (0 to 7) from a header. The overload further
// down shows the rule byte by byte; here (x >> 2) & 48 replaces (x >> 6) << 4. This one gives
// the same result as fr::sub_scale (fold_rule.h); the rule is that of ggml's get_scale_min_k4.
__device__ __forceinline__ void scale_min(int j, const uint4 &head, int &sc, int &m) {
  const int shift = 8 * (j & 3);
  const uint32_t a = head.y >> shift, b = head.z >> shift, c = head.w >> shift;
  sc = (int)(j < 4 ? a & 63 : (c & 15) | ((a >> 2) & 48));
  m = (int)(j < 4 ? b & 63 : ((c >> 4) & 15) | ((b >> 2) & 48));
}
// The two numbers that all weights of a sub-block share: ds = RN(d * sc) and dm = RN(dmin * m).
struct Sub {
  float ds, dm;
};
// Computes ds and dm of sub-block j of the block with this header.
// Public sources for the techniques of this routine.
// Each product is the intrinsic __fmul_rn, so that it is rounded by itself. The idea is public:
// intrinsics that name their rounding mode are not fused into multiply-adds: N. Whitehead and A.
// Fit-Florea, "Floating Point and IEEE 754 Compliance for NVIDIA GPUs", NVIDIA white paper, 2011,
// 4.3 "Controlling Fused Multiply-add", https://docs.nvidia.com/cuda/floating-point/index.html.
// NVIDIA's CUDA Math API reference states the same of __fmul_rn (top of this file).
__device__ __forceinline__ Sub sub_of(const uint4 &head, int j) {
  const float d = __half2float(__ushort_as_half((unsigned short)(head.x & 0xFFFF))),
              dmin = __half2float(__ushort_as_half((unsigned short)(head.x >> 16)));
  int s, m;
  scale_min(j, head, s, m);
  return {__fmul_rn(d, (float)s), __fmul_rn(dmin, (float)m)};
}
// Returns the committed integer w of code q (formula: top of this file). The kernels store only its
// low 16 bits. If |w| > 4095, BlockWeights::fill() (prover/weights_blocks/block_weights.cuh) exits
// and cp_commit writes no commitment. If RN(G * v) is not finite or |RN(G * v)| >= 2147483000,
// fr::fold() returns INT_MIN; weight() has no such case.
// Public sources for the techniques of this routine.
// Each arithmetic operation is an intrinsic, to be rounded by itself. The idea is public: the white
// paper of Whitehead and Fit-Florea, cited at sub_of(const uint4 &, int). NVIDIA's CUDA Math API
// reference (top of this file) states that __fsub_rn is not fused and how __float2int_rn rounds.
__device__ __forceinline__ int weight(float ds, float dm, int q, float G) {
  return __float2int_rn(__fmul_rn(G, __fsub_rn(__fmul_rn(ds, (float)q), dm)));
}
// Splits w into lo + 256 * hi with lo in [-128, 127].
__device__ __forceinline__ void limbs(int w, int &lo, int &hi) {
  lo = (int)(int8_t)w;
  hi = (w - lo) >> 8;
}
// What decode_planes() reads. F, 16-byte aligned, is a tensor [expert][Lfull rows][nsb blocks].
// The window is blocks sb0 to sb0 + nsbw - 1 of each row; window block S is block sb0 + S % nsbw
// of a row of expert S / nsbw. The call's block s is window block S0 + s; its row x is row x0 + x.
struct Where {
  const uint8_t *F;
  int nsb, nsbw, sb0, S0, x0;
  size_t Lfull;
};
// Rows per warp task in k_decode_planes. More rows spread the task's divisions and its eight
// products with sc over more weights; fewer rows give more warps to fill the GPU.
constexpr int ROWS_A_TASK = 8;
constexpr int WARPS = 8; // the warps of a thread block in k_decode_planes
// Writes lo and hi of weight n of block s in row x at index x * ld + 256 * s + n, for x < L and
// s < nS (see `Where`). G[j] = gamma_j for tensor column j < 256 * w.nsb; a null G means all 1.
// sc = 2^e, not a sub-block's scale. A warp's task is block s in up to ROWS_A_TASK adjacent rows.
// In each row a lane decodes eight weights: four adjacent ones of sub-block 2 * il
// (il = lane / 8) and the four 32 further on, in sub-block 2 * il + 1. Requires
// nS * ceil(L / ROWS_A_TASK) + WARPS * gridDim.x <= 2^32 (decode_planes() meets this up to 2 TB of
// output), blockDim.x == 32 * WARPS, ld % 4 == 0, and 4-byte aligned lo and hi.
//
// Public sources for the techniques of this routine.
// The compiler hint is public: a launch bound with one argument: NVIDIA, CUDA C++ Best Practices
// Guide, "Occupancy" (top of this file).
// The idea is public: unrolling lets small arrays stay in registers: M. Milakov, "Fast Dynamic
// Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, "Static indexing",
// https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
// The idea is public: thread coarsening (a warp takes several rows): Wen-mei W. Hwu and David Kirk,
// "Advanced Algorithmic Techniques for GPUs", Lecture 4, "Thread Coarsening and more on
// Tiling/Blocking", PASI Summer School, Chile, January 5-7, 2011, slide "Outline of Technique",
// https://www.bu.edu/pasi/files/2011/01/WenMeiHwu4-06-1600.pdf.
// The idea is public: a template argument chosen on the host: M. Harris, "Optimizing Parallel
// Reduction in CUDA", NVIDIA Developer Technology, slides "Unrolling with Templates" to "Invoking
// Template Kernels", https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf.
// Here the parameter is the format of the data.
// The idea is public: loop-invariant code motion, here by hand: A. Platzer, "Lecture Notes on
// Loop-Invariant Code Motion", Lecture 17, section 2,
// https://symbolaris.com/course/Compilers/17-loopinv.pdf (also LLVM's "licm": top of this file).
// The idea is public: an addition replaces a loop's multiplication, here by hand: Wikipedia,
// "Strength reduction", https://en.wikipedia.org/wiki/Strength_reduction.
// The products with sc are written as __fmul_rn like the products in weight(): see there.
template <int Q5>
__global__ void __launch_bounds__(32 * WARPS)
    k_decode_planes(const Where w, int L, int nS, const float *__restrict__ G, float sc, int8_t *__restrict__ lo,
                    int8_t *__restrict__ hi, size_t ld) {
  constexpr int BB = Blk<Q5>::BB, QS = Blk<Q5>::QS;
  const int lane = threadIdx.x & 31, il = lane >> 3, ir = lane & 7;
  const uint32_t tasks = (uint32_t)nS * ((L + ROWS_A_TASK - 1) / ROWS_A_TASK);
  for (uint32_t task = blockIdx.x * WARPS + (threadIdx.x >> 5); task < tasks; task += gridDim.x * WARPS) {
    const int s = (int)(task % nS), from = (int)(task / nS) * ROWS_A_TASK, to = min(L, from + ROWS_A_TASK);
    const int S = w.S0 + s, sb = w.sb0 + S % w.nsbw; // sb: the block's number in its row
    const uint8_t *b = w.F + (((size_t)(S / w.nsbw) * w.Lfull + w.x0 + from) * w.nsb + sb) * BB;
    float g0[4], g1[4]; // G for the lane's eight columns
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float *g = G ? G + (size_t)sb * 256 + 64 * il + 4 * ir + i : nullptr;
      g0[i] = g ? __fmul_rn(g[0], sc) : sc;
      g1[i] = g ? __fmul_rn(g[32], sc) : sc;
    }
    size_t o = (size_t)from * ld + (size_t)s * 256 + 64 * il + 4 * ir;
    for (int x = from; x < to; x++, b += (size_t)w.nsb * BB, o += ld) {
      const uint4 head = *(const uint4 *)b;
      const uint32_t codes = *(const uint32_t *)(b + QS + 32 * il + 4 * ir);
      const uint32_t fifth = Q5 ? *(const uint32_t *)(b + 16 + 4 * ir) : 0; // bit j of a byte: sub-block j's fifth bit
      const Sub A = sub_of(head, 2 * il), B = sub_of(head, 2 * il + 1);
      uint32_t l0 = 0, h0 = 0, l1 = 0, h1 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const uint32_t code = (codes >> (8 * i)) & 255, bits = (fifth >> (8 * i + 2 * il)) & 3;
        const int q0 = (int)((code & 15) | ((bits & 1) << 4)), q1 = (int)((code >> 4) | ((bits & 2) << 3));
        int l, h;
        limbs(weight(A.ds, A.dm, q0, g0[i]), l, h);
        l0 |= (uint32_t)(l & 255) << (8 * i);
        h0 |= (uint32_t)(h & 255) << (8 * i);
        limbs(weight(B.ds, B.dm, q1, g1[i]), l, h);
        l1 |= (uint32_t)(l & 255) << (8 * i);
        h1 |= (uint32_t)(h & 255) << (8 * i);
      }
      *(uint32_t *)(lo + o) = l0;
      *(uint32_t *)(hi + o) = h0;
      *(uint32_t *)(lo + o + 32) = l1;
      *(uint32_t *)(hi + o + 32) = h1;
    }
  }
}
// Returns tasks, clamped to 1 .. 2^20, as a launch's grid size. The kernels loop over their tasks,
// so every grid size up to 2^20 gives the same bytes.
static inline unsigned grid_for(size_t tasks) {
  return (unsigned)(tasks < ((size_t)1 << 20) ? (tasks ? tasks : 1) : (size_t)1 << 20);
}
// The wider types. TY: 4 Q8_0, 5 F16, 6 BF16. BB: the bytes of a block of 256 weights.
template <int TY> struct Wide {
  static_assert(TY == 4 || TY == 5 || TY == 6, "the wider types are 4 (Q8_0), 5 (F16) and 6 (BF16)");
  static constexpr int BB = TY == 4 ? 272 : 512;
};
// Ends the program: 'type' is not a type that this file decodes.
[[noreturn]] static inline void no_such_type(int type) {
  fprintf(stderr,
          "internal error in the decoding of weight blocks: type %d is not one of 0 (Q4_K), 1 (Q5_K), 4 (Q8_0), 5 (F16) "
          "and 6 (BF16)\n",
          type);
  exit(4);
}
// Returns the bytes of a block of 256 weights of a type that this file decodes. Any other number ends the program.
static inline int block_bytes(int type) {
  if (type != 0 && type != 1 && type != 4 && type != 5 && type != 6) no_such_type(type);
  return type == 0 ? 144 : type == 1 ? 176 : type == 4 ? 272 : 512;
}
// Sets v[0 .. N - 1] to the decoded values of elements c .. c + N - 1 of the block at b, as fr::value8 and fr::value16
// (fold_rule.h) define them. N is 4 or 8, c a multiple of N, and b a multiple of 16. A Q8_0 block of 34 bytes begins at
// an even address only, so its scale and its codes are loaded as words of 16 bits. The 16-bit types are loaded as one
// word of 64 or 128 bits (vector loads: this file's head cites the public source). Public source for the layouts, and
// where the decoding comes from: see fold_rule.h, value8() and value16(). Public source for the intrinsics: NVIDIA,
// CUDA Toolkit 13.0, the comments at their declarations: include/cuda_fp16.h (__half2float gives a half's value as a
// float; __ushort_as_half is a cast from a 16-bit unsigned integer to half that leaves the bits as they are) and
// include/crt/device_functions.h (__uint_as_float is a cast from a 32-bit unsigned integer to float that leaves the
// bits as they are; __fmul_rn multiplies two floats and rounds the result to nearest, ties to even). A product written
// as __fmul_rn is rounded on its own and never fused with an addition that follows: see weight() above.
template <int TY, int N> __device__ __forceinline__ void values(const uint8_t *b, int c, float *v) {
  static_assert(N == 4 || N == 8, "four or eight values at a time");
  if constexpr (TY == 4) {
    const uint8_t *s = b + 34 * (c >> 5);
    const float d = __half2float(__ushort_as_half(*(const unsigned short *)s));
    const unsigned short *q = (const unsigned short *)(s + 2 + (c & 31));
#pragma unroll
    for (int i = 0; i < N; i += 2) {
      const unsigned short two = q[i / 2]; // two codes
      v[i] = __fmul_rn(d, (float)(int)(int8_t)(two & 255));
      v[i + 1] = __fmul_rn(d, (float)(int)(int8_t)(two >> 8));
    }
  } else {
    uint32_t word[N / 2]; // two values in each
    if constexpr (N == 4) {
      const uint2 two = *(const uint2 *)(b + 2 * c);
      word[0] = two.x, word[1] = two.y;
    } else {
      const uint4 four = *(const uint4 *)(b + 2 * c);
      word[0] = four.x, word[1] = four.y, word[2] = four.z, word[3] = four.w;
    }
#pragma unroll
    for (int i = 0; i < N; i++) {
      const uint32_t h = (word[i / 2] >> (16 * (i % 2))) & 0xFFFF;
      v[i] = TY == 5 ? __half2float(__ushort_as_half((unsigned short)h)) : __uint_as_float(h << 16);
    }
  }
}
// Returns the committed integer of the decoded value v: RN(G v), rounded to the nearest integer, ties to even. These
// are the last two steps of weight() above, by the same intrinsics; their public sources are cited there. It tests
// nothing, and a value that is not finite or too large would give a wrong integer without any error. So the order
// matters: every value must have passed BlockWeights::fill() in cp_prove, or the checking pass in cp_commit
// (fold::integers()), before it comes here.
__device__ __forceinline__ int wide_weight(float v, float G) {
  return __float2int_rn(__fmul_rn(G, v));
}
// k_decode_planes for the wider types, with the same arguments, requirements and tasks. Of each block a lane decodes
// eight weights: elements 4 lane .. 4 lane + 3, and the four that lie 128 places further. So the lanes of a warp store
// neighbouring words.
// Public sources for the techniques of this routine.
// The routine follows k_decode_planes above. Its loop over the tasks and its way from a task to the block's address and
// to the place in the output are that kernel's. What differs: which elements a lane takes, and how they are decoded.
// It shares these techniques with that kernel, whose comment cites a public source for each: the launch bound with one
// argument; short loops unrolled completely, so that the compiler keeps a lane's small arrays, here g and v, in
// registers; thread coarsening (a lane's eight products with sc are made once and serve all rows of the task); a
// template parameter whose instance the host code chooses, here the storage type; loop-invariant code motion and
// strength reduction, written by hand; and products written as __fmul_rn.
// The three types' layouts and the rule that decodes them: values() above, and value8() and value16() in fold_rule.h,
// which cite their public sources.
template <int TY>
__global__ void __launch_bounds__(32 * WARPS)
    k_decode_planes_wide(const Where w, int L, int nS, const float *__restrict__ G, float sc, int8_t *__restrict__ lo,
                         int8_t *__restrict__ hi, size_t ld) {
  constexpr int BB = Wide<TY>::BB;
  const int lane = threadIdx.x & 31;
  const uint32_t tasks = (uint32_t)nS * ((L + ROWS_A_TASK - 1) / ROWS_A_TASK);
  for (uint32_t task = blockIdx.x * WARPS + (threadIdx.x >> 5); task < tasks; task += gridDim.x * WARPS) {
    const int s = (int)(task % nS), from = (int)(task / nS) * ROWS_A_TASK, to = min(L, from + ROWS_A_TASK);
    const int S = w.S0 + s, sb = w.sb0 + S % w.nsbw; // sb: the block's number in its row
    const uint8_t *b = w.F + (((size_t)(S / w.nsbw) * w.Lfull + w.x0 + from) * w.nsb + sb) * BB;
    float g[2][4]; // G for the lane's eight columns
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int i = 0; i < 4; i++) g[half][i] = G ? __fmul_rn(G[(size_t)sb * 256 + 128 * half + 4 * lane + i], sc) : sc;
    size_t o = (size_t)from * ld + (size_t)s * 256 + 4 * lane;
    for (int x = from; x < to; x++, b += (size_t)w.nsb * BB, o += ld) {
#pragma unroll
      for (int half = 0; half < 2; half++) {
        float v[4];
        values<TY, 4>(b, 128 * half + 4 * lane, v);
        uint32_t low = 0, high = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          int l, h;
          limbs(wide_weight(v[i], g[half][i]), l, h);
          low |= (uint32_t)(l & 255) << (8 * i);
          high |= (uint32_t)(h & 255) << (8 * i);
        }
        *(uint32_t *)(lo + o + 128 * half) = low;
        *(uint32_t *)(hi + o + 128 * half) = high;
      }
    }
  }
}
// Launches k_decode_planes or k_decode_planes_wide on stream st, for blocks of the given type.
// Public sources for the techniques of this routine.
// The choice between the kernel's instances: see M. Harris,
// "Optimizing Parallel Reduction in CUDA", cited at k_decode_planes.
static inline void decode_planes(const Where &w, int type, int L, int nS, const float *G, float sc, int8_t *lo,
                                 int8_t *hi, size_t ld, cudaStream_t st) {
  const size_t thread_blocks = ((size_t)nS * ((L + ROWS_A_TASK - 1) / ROWS_A_TASK) + WARPS - 1) / WARPS;
  const unsigned grid = grid_for(thread_blocks);
  if (type == 0)
    k_decode_planes<0><<<grid, 32 * WARPS, 0, st>>>(w, L, nS, G, sc, lo, hi, ld);
  else if (type == 1)
    k_decode_planes<1><<<grid, 32 * WARPS, 0, st>>>(w, L, nS, G, sc, lo, hi, ld);
  else if (type == 4)
    k_decode_planes_wide<4><<<grid, 32 * WARPS, 0, st>>>(w, L, nS, G, sc, lo, hi, ld);
  else if (type == 5)
    k_decode_planes_wide<5><<<grid, 32 * WARPS, 0, st>>>(w, L, nS, G, sc, lo, hi, ld);
  else if (type == 6)
    k_decode_planes_wide<6><<<grid, 32 * WARPS, 0, st>>>(w, L, nS, G, sc, lo, hi, ld);
  else
    no_such_type(type);
  CK(cudaGetLastError());
}
// Extracts sc and m of sub-block j from the twelve packed bytes at q, with the same result as
// fr::sub_scale() (fold_rule.h). Adapted from ggml's get_scale_min_k4 (see the top of this file).
__device__ __forceinline__ void scale_min(int j, const uint8_t *q, int &sc, int &m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}
// Computes ds and dm of sub-block j of the block at blk, which must be 2-byte aligned.
// Public sources for the techniques of this routine.
// Each product is the intrinsic __fmul_rn, so that it is rounded by itself. The idea is public: see
// the white paper of Whitehead and Fit-Florea, cited at sub_of(const uint4 &, int).
__device__ __forceinline__ Sub sub_of(const uint8_t *blk, int j) {
  const float d = __half2float(*(const __half *)blk), dmin = __half2float(*(const __half *)(blk + 2));
  int s, m;
  scale_min(j, blk + 4, s, m);
  return {__fmul_rn(d, (float)s), __fmul_rn(dmin, (float)m)};
}
// For row < rows and k < 256 * nsb, writes lo and hi of weight (row, k) of the rows x nsb blocks at
// F at index k * ldc + row. rows need not be a multiple of 4. G and sc are as in k_decode_planes.
// Requires blockDim.x == 256, ldc % 4 == 0, 4-byte aligned lo and hi, a 2-byte aligned F, and
// `tiles` + gridDim.x <= 2^31 (a tile: TR rows by the 64 columns of two sub-blocks).
// Public sources for the techniques of this routine.
// The idea is public: a shared-memory buffer lets accesses to global memory coalesce; results may
// leave the same way: M. Harris, "Optimizing CUDA", NVIDIA Developer Technology, slides headed
// "S05: High Performance Computing with CUDA", slides 8 and 19 (the address runs over two lines):
// https://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/Cours09/CUDA/
// SC07_CUDA_5_Optimization_Harris.pdf.
// The idea is public: thread coarsening: Hwu and Kirk, Lecture 4, cited at k_decode_planes.
// The compiler hint is public: a launch bound with one argument: NVIDIA, CUDA C++ Best Practices
// Guide, "Occupancy" (top of this file).
// The products with sc are written as __fmul_rn like the products in weight(): see there.
template <int Q5>
__global__ void __launch_bounds__(256)
    k_cells(const uint8_t *__restrict__ F, int rows, int nsb, int8_t *__restrict__ lo, int8_t *__restrict__ hi,
            size_t ldc, const float *__restrict__ G, float sc) {
  constexpr int TR = 128, PITCH = TR + 4;
  __shared__ __align__(4) int8_t sl[64 * PITCH], sh[64 * PITCH];
  const int ntx = (rows + TR - 1) / TR, tiles = ntx * nsb * 4;
  const int r = threadIdx.x >> 1, half = threadIdx.x & 1;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int il = t & 3, sb = (t >> 2) % nsb, x0 = ((t >> 2) / nsb) * TR;
    if (x0 + r < rows) {
      const uint8_t *x = F + ((size_t)(x0 + r) * nsb + sb) * Blk<Q5>::BB,
                    *q = x + Blk<Q5>::QS + 32 * il + 16 * half, *qh = x + 16 + 16 * half;
      const Sub A = sub_of(x, 2 * il), B = sub_of(x, 2 * il + 1);
      const float *g = G ? G + (size_t)sb * 256 + 64 * il + 16 * half : nullptr;
      for (int l = 0; l < 16; l++) {
        int qa = q[l] & 0xF, qb = q[l] >> 4;
        if (Q5) {
          qa += ((qh[l] >> (2 * il)) & 1) << 4;
          qb += ((qh[l] >> (2 * il + 1)) & 1) << 4;
        }
        const int wa = weight(A.ds, A.dm, qa, g ? __fmul_rn(g[l], sc) : sc),
                  wb = weight(B.ds, B.dm, qb, g ? __fmul_rn(g[l + 32], sc) : sc);
        const int a0 = ((wa + 128) & 255) - 128, b0 = ((wb + 128) & 255) - 128, e = 16 * half + l;
        sl[e * PITCH + r] = (int8_t)a0;
        sh[e * PITCH + r] = (int8_t)((wa - a0) >> 8);
        sl[(e + 32) * PITCH + r] = (int8_t)b0;
        sh[(e + 32) * PITCH + r] = (int8_t)((wb - b0) >> 8);
      }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, x = x0 + 4 * lane;
    for (int e = warp; e < 64; e += 8) {
      const size_t o = ((size_t)sb * 256 + 64 * il + e) * ldc + x;
      if (x + 4 <= rows) {
        *(uint32_t *)(lo + o) = *(const uint32_t *)(sl + e * PITCH + 4 * lane);
        *(uint32_t *)(hi + o) = *(const uint32_t *)(sh + e * PITCH + 4 * lane);
      } else
        for (int b = 0; x + b < rows; b++) {
          lo[o + b] = sl[e * PITCH + 4 * lane + b];
          hi[o + b] = sh[e * PITCH + 4 * lane + b];
        }
    }
    __syncthreads();
  }
}
// k_cells for the wider types, with the same arguments, requirements, tiles and stores: a tile is 128 rows of a quarter
// il of a block column sb, that is 64 columns. It also requires an address F that is a multiple of 16 (values() loads
// words of 128 bits). The two threads of a row decode its 64 columns eight at a time, in turn: thread 'half' takes the
// columns 16 part + 8 half to 16 part + 8 half + 7 of the tile, for part = 0 to 3.
// Public sources for the techniques of this routine.
// The routine follows k_cells above. The tile in shared memory, its pitch, the loop over the tiles and the stores from
// the tile to the output are that kernel's. What differs: which columns a thread takes, and how they are decoded.
// It shares these techniques with that kernel, whose comment cites a public source for each: the output staged in a
// tile in shared memory; thread coarsening; the launch bound with one argument; products written as __fmul_rn. Two
// more are cited at k_decode_planes: short loops unrolled completely, and a template parameter whose instance the host
// code chooses. The tile's padded pitch is cited at this file's head.
// The three types' layouts and the rule that decodes them: values() above, and value8() and value16() in fold_rule.h,
// which cite their public sources.
template <int TY>
__global__ void __launch_bounds__(256)
    k_cells_wide(const uint8_t *__restrict__ F, int rows, int nsb, int8_t *__restrict__ lo, int8_t *__restrict__ hi,
                 size_t ldc, const float *__restrict__ G, float sc) {
  constexpr int TR = 128, PITCH = TR + 4;
  __shared__ __align__(4) int8_t sl[64 * PITCH], sh[64 * PITCH];
  const int ntx = (rows + TR - 1) / TR, tiles = ntx * nsb * 4;
  const int r = threadIdx.x >> 1, half = threadIdx.x & 1;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int il = t & 3, sb = (t >> 2) % nsb, x0 = ((t >> 2) / nsb) * TR;
    if (x0 + r < rows) {
      const uint8_t *x = F + ((size_t)(x0 + r) * nsb + sb) * Wide<TY>::BB;
#pragma unroll
      for (int part = 0; part < 4; part++) {
        const int e = 16 * part + 8 * half, c = 64 * il + e; // the first of eight columns: in the tile, in the block
        float v[8];
        values<TY, 8>(x, c, v);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int l, h;
          limbs(wide_weight(v[i], G ? __fmul_rn(G[(size_t)sb * 256 + c + i], sc) : sc), l, h);
          sl[(e + i) * PITCH + r] = (int8_t)l;
          sh[(e + i) * PITCH + r] = (int8_t)h;
        }
      }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, x = x0 + 4 * lane;
    for (int e = warp; e < 64; e += 8) {
      const size_t o = ((size_t)sb * 256 + 64 * il + e) * ldc + x;
      if (x + 4 <= rows) {
        *(uint32_t *)(lo + o) = *(const uint32_t *)(sl + e * PITCH + 4 * lane);
        *(uint32_t *)(hi + o) = *(const uint32_t *)(sh + e * PITCH + 4 * lane);
      } else
        for (int b = 0; x + b < rows; b++) {
          lo[o + b] = sl[e * PITCH + 4 * lane + b];
          hi[o + b] = sh[e * PITCH + 4 * lane + b];
        }
    }
    __syncthreads();
  }
}
// For y < m, sets out[y * ldo + c] to in[c * ldi + y] if c < k and to 0 if k <= c < kp. Requires
// blockDim.x == 256, k <= kp, kp % 64 == 0, ldi % 4 == 0, ldo % 4 == 0, 4-byte aligned in and out,
// and (kp / 64) * ceil(m / 64) + gridDim.x <= 2^31.
// Public sources for the techniques of this routine.
// The idea is public: output written from a shared-memory tile in consecutive runs: Harris's post
// on the matrix transpose (top of this file). A thread reads what another wrote, hence the barrier.
// The idea is public: a further barrier before the tile's reuse: documentation of cub::BlockStore,
// https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockStore.html.
// The compiler hint and the loop form are public: __launch_bounds__: the Best Practices Guide,
// "Occupancy"; a loop covers a tile in several passes: Harris's post on the matrix transpose.
__global__ void __launch_bounds__(256) k_transpose_bytes(const int8_t *__restrict__ in, size_t ldi, int k, int kp,
                                                         int m, int8_t *__restrict__ out, size_t ldo) {
  __shared__ __align__(4) int8_t t[64 * 68];
  const int ncx = kp / 64, nyx = (m + 63) / 64;
  for (int tile = blockIdx.x; tile < ncx * nyx; tile += gridDim.x) {
    const int c0 = tile % ncx * 64, y0 = tile / ncx * 64;
    for (int i = threadIdx.x; i < 64 * 16; i += 256) {
      const int c = i >> 4, j = i & 15;
      uint32_t v = 0;
      if (c0 + c < k) {
        const int8_t *p = in + (size_t)(c0 + c) * ldi + y0 + 4 * j;
        if (y0 + 4 * j + 3 < m)
          v = *(const uint32_t *)p;
        else
          for (int b = 0; b < 4 && y0 + 4 * j + b < m; b++) v |= (uint32_t)(uint8_t)p[b] << (8 * b);
      }
      for (int b = 0; b < 4; b++) t[(4 * j + b) * 68 + c] = (int8_t)(v >> (8 * b));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 16; i += 256) {
      const int y = i >> 4, j = i & 15;
      if (y0 + y < m)
        *(uint32_t *)(out + (size_t)(y0 + y) * ldo + c0 + 4 * j) = *(const uint32_t *)(t + y * 68 + 4 * j);
    }
    __syncthreads();
  }
}
// Launches k_transpose_bytes on stream st. Requires (kp / 64) * ceil(m / 64) <= 2^31 - 2^20.
static inline void transpose_bytes(const int8_t *in, size_t ldi, int k, int kp, int m, int8_t *out, size_t ldo,
                                   cudaStream_t st) {
  k_transpose_bytes<<<grid_for((size_t)(kp / 64) * ((m + 63) / 64)), 256, 0, st>>>(in, ldi, k, kp, m, out, ldo);
  CK(cudaGetLastError());
}
} // namespace qblocks
// Decodes the rows x nsb blocks at F, of the given type (see the file's head), into the planes w0
// (lo) and w1 (hi) at index row * ld + k, on stream st, for a product that reads a weight matrix by
// rows. G and sc are as in k_decode_planes. Requires ld % 4 == 0, 4-byte aligned w0 and w1, and a
// 16-byte aligned F.
static inline void blocks_to_planes(const void *F, int rows, int nsb, int8_t *w0, int8_t *w1, size_t ld,
                                    cudaStream_t st = 0, const float *G = nullptr, float sc = 4096.f, int type = 0) {
  const qblocks::Where whole{(const uint8_t *)F, nsb, nsb, 0, 0, 0, (size_t)rows};
  qblocks::decode_planes(whole, type, rows, nsb, G, sc, w0, w1, ld, st);
}
// Decodes the same blocks into lo and hi at index k * ldc + row. Requires rows > 0, nsb > 0 and
// `tiles` <= 2^30 (the grid size is not clamped), and what k_cells or, for the wider types,
// k_cells_wide requires of ldc, lo, hi and F.
static inline void blocks_to_cells(const void *F, int rows, int nsb, int8_t *lo, int8_t *hi, size_t ldc,
                                   cudaStream_t st = 0, const float *G = nullptr, float sc = 4096.f, int type = 0) {
  const int tiles = (rows + 127) / 128 * nsb * 4;
  const uint8_t *bytes = (const uint8_t *)F;
  if (type == 0)
    qblocks::k_cells<0><<<tiles, 256, 0, st>>>(bytes, rows, nsb, lo, hi, ldc, G, sc);
  else if (type == 1)
    qblocks::k_cells<1><<<tiles, 256, 0, st>>>(bytes, rows, nsb, lo, hi, ldc, G, sc);
  else if (type == 4)
    qblocks::k_cells_wide<4><<<tiles, 256, 0, st>>>(bytes, rows, nsb, lo, hi, ldc, G, sc);
  else if (type == 5)
    qblocks::k_cells_wide<5><<<tiles, 256, 0, st>>>(bytes, rows, nsb, lo, hi, ldc, G, sc);
  else if (type == 6)
    qblocks::k_cells_wide<6><<<tiles, 256, 0, st>>>(bytes, rows, nsb, lo, hi, ldc, G, sc);
  else
    qblocks::no_such_type(type);
  CK(cudaGetLastError());
}
