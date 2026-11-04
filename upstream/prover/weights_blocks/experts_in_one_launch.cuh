// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Four expressions of two_scales() in this file are those of ggml (llama.cpp), which is under the MIT License.
// That project's notice:
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

// One launch for all routed experts of a tensor. weights_blocks/experts.cuh includes this file if its line
// EXPERTS_IN_ONE_LAUNCH says 1, and grouped_blocks() there then hands a tensor whose shape fits (pk_fits()) to
// pk_grouped_blocks(); for any other shape it goes on with its own body.
//
// The routed experts' matrix products, read straight from the model file's 4-bit and 5-bit blocks.
//
// A routed tensor lies on the GPU as the file has it: rows cut into blocks of 256 weights (weights_blocks/blocks.h).
// Each weight stands for a 16-bit integer w, fixed by the rule of weights_blocks/fold_rule.h, and w = 256 b1 + b0 with
// two signed bytes. An activation is an integer of NL signed bytes in the same way. So a product of a weight matrix
// with the activations is a sum of products of signed-byte matrices, which the tensor cores compute exactly into 32-bit
// sums:
//   pk_grouped_blocks  acc[row][rank] = sum over k of W_e[row][k] x[slot][k], for every expert e and each of its
//                      tokens.
// The tokens have been sorted by expert before (MoeTables, prover/prover/forward_moe.cuh): a token has a slot (each
// expert's run of slots is filled up to a multiple of 64) and a rank (no filling).
//
// Where the techniques of this file are public.
// The idea is public: llama.cpp's CUDA back end (MIT licence) multiplies block-quantized expert weights batched through
// its MMQ kernel, which is told which columns belong to which expert and skips the other tiles; the helper that
// prepares this runs on the GPU, so no synchronization with the host is needed (pull requests #13199 and #15525,
// JohannesGaessler, 2025); vLLM's fused_moe kernel likewise reads its padded token count from GPU memory.
// Here one kernel takes every expert's tiles, and it reads each expert's number of tokens and first slot from tables in
// GPU memory, so that the launch needs nothing of them on the host. Of this file's own: a thread block belongs to one
// expert and loops over that expert's tokens (for an expert without tokens the loop is empty); the codes are decoded to
// the two signed bytes of the integer that weights_blocks/fold_rule.h fixes; and the products are exact, in integers. Each tile stacks the weights' two bytes as rows and the activations' bytes as columns of one product. No source was found in public papers for stacking bytes of both sides; the search was made for gram_products.cuh.
// The idea is public: a multistage shared-memory pipeline with asynchronous copies, one wait and one __syncthreads()
// per tile, and register prefetch of the next fragment is the method of the main loop of CUTLASS's example for
// sm80 as explained in the appendix of Colfax Research's 'CUTLASS Tutorial: Efficient GEMM kernel designs with
// Pipelining'; NVIDIA's
// 'CUTLASS: Fast Linear Algebra in CUDA C++' notes that double buffering needs no second __syncthreads(); the
// __pipeline_* primitives are shown in NVIDIA's CUDA Programming Guide, 'Asynchronous Data Copies'.
// Here the activations go to shared memory by such copies, with one wait and one __syncthreads() a stage, and the next
// stage's weight bytes are read into registers meanwhile. What differs: the weights reach shared memory by ordinary
// stores, decoded from those registers, and the two operands have unequal numbers of buffers, two and three.
// The means that this file uses are public.
// - Tensor cores: NVIDIA's C++ API nvcuda::wmma, with fragments of signed char, 16 x 16 x 16, and accumulators of int.
//   See NVIDIA, CUDA Programming Guide, "Warp Matrix Functions",
//   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html: "Element Types and
//   Matrix Sizes" lists signed char for both factors with int for the accumulator, at 16x16x16.
// - Copies from global to shared memory that do not wait. See the same guide, "Pipeline Primitives Interface",
//   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html:
//   __pipeline_memcpy_async: size_and_align takes only 4, 8 or 16; dst_shared and src_global must
//   be aligned to it; __pipeline_commit: the pipeline's current batch is now the submitted copies;
//   __pipeline_wait_prior(N): waits until at least every batch but the newest N is complete.
// - Each rounding by itself. See NVIDIA, CUDA Math API Reference Manual,
//   https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html: __fmul_rn:
//   rounds to nearest even the product of two floats, __fsub_rn likewise, and under both: no
//   multiply-add instruction ever absorbs the call. weights_blocks/decode.cuh uses the same
//   intrinsics.
// - Bytes picked out of two words. See the same manual,
//   https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html: __byte_perm:
//   takes two unsigned 32-bit integers and gives back bytes picked out of them.
// - More than 48 KB of shared memory for a block. See NVIDIA, Hopper Tuning Guide,
//   https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html: 227 KB is the most that one block can have (Tile4
//   asserts that it fits); to take over 48 KB dynamically a kernel must first be allowed, explicitly (pk_blocks_ready).
// - The blocks. Their byte layout and the rule that decodes them are ggml's (the llama.cpp project; MIT licence):
//   weights_blocks/decode.cuh and weights_blocks/fold_rule.h give the credit, and NOTICE and THIRD_PARTY.md the notice.
//   The three decoding helpers of this file follow fold_rule.h's rule. See also Hugging Face, "GGUF",
//   https://huggingface.co/docs/hub/gguf: Q4_K and Q5_K: "Super-blocks with 8 blocks, each block has 32 weights. Weight
//   formula: w = q * block_scale(6-bit) + block_min(6-bit)" (the page adds the minimum; the rule subtracts it), and the
//   project's wiki, "Tensor Encoding Schemes", https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes: "The
//   12 bytes in Q4_K .scales are packed a bit like this".
#pragma once
#include "weights_blocks/blocks.h"
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace pk {
using namespace nvcuda;
typedef unsigned char u8p;

// Four codes, one in each byte of 'codes', to the two bytes of their weights: w = round(g (ds q - dm)), without g if GN
// is 0; lo gets the four b0, hi the four b1, of w = 256 b1 + b0 with b0 from -128 to 127. Each product and the
// difference is rounded once, to nearest even. No test that w fits 16 bits: of a larger w the upper bits are lost.
// The idea is public: NVIDIA's CUTLASS packs byte k of four 32-bit integers into one word with three byte permutes
// (selectors 0x0040, 0x0040, 0x5410 for byte 0) in its in-thread 4x4 int8 transpose
// (cutlass::transform::thread::Transpose, BSD-3-Clause).
// four_codes() makes the same __byte_perm calls, with the same selectors, as NVIDIA CUTLASS,
// include/cutlass/transform/thread/transpose.h (int8_t 4x4 transpose), BSD-3-Clause: of that routine's four groups of
// three calls, the first (0x0040, 0x0040, 0x5410) and the second (0x0051, 0x0051, 0x5410). That file's notice follows,
// as in a copy with the years 2017 - 2022.
// Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// (That file's notice ends here.)
template <int GN>
__device__ __forceinline__ void four_codes(unsigned codes, float ds, float dm, const float4 g, unsigned &lo,
                                           unsigned &hi) {
  const float gg[4] = {g.x, g.y, g.z, g.w};
  int t[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float v = __fsub_rn(__fmul_rn(ds, (float)((codes >> (8 * k)) & 255u)), dm); // scale times code, less the minimum
    if (GN) v = __fmul_rn(gg[k], v);
    t[k] = __float2int_rn(v) + 128; // to the nearest integer, ties to even; w + 128 = 256 b1 + (b0 + 128)
  }
  // the four words' lowest bytes (b0 + 128: the xor takes the 128 off) and their second bytes
  lo = __byte_perm(__byte_perm(t[0], t[1], 0x0040), __byte_perm(t[2], t[3], 0x0040), 0x5410) ^ 0x80808080u;
  hi = __byte_perm(__byte_perm(t[0], t[1], 0x0051), __byte_perm(t[2], t[3], 0x0051), 0x5410);
}
// sc times the scale (ds) and the minimum (dm) of the sub-blocks 2 c and 2 c + 1, c from 0 to 3. hd: a block's first 16
// bytes: d and dmin as binary16, then the twelve bytes that pack eight 6-bit scales and eight 6-bit minima, as
// fr::sub_scale (weights_blocks/fold_rule.h) unpacks them. sc is a power of two: multiplying by it commutes with
// rounding unless a value underflows or overflows, so the integers are those of fold_rule.h, which scales last.
__device__ __forceinline__ void two_scales(const uint4 hd, int c, float sc, float (&ds)[2], float (&dm)[2]) {
  const __half2 dd = *(const __half2 *)&hd.x;
  const float d = __low2float(dd), dmin = __high2float(dd);
#pragma unroll
  for (int u = 0; u < 2; u++) {
    const int sh = 16 * (c & 1) + 8 * u;
    const unsigned y = (hd.y >> sh) & 255u, z = (hd.z >> sh) & 255u, w = (hd.w >> sh) & 255u;
    const unsigned s = c < 2 ? (y & 63u) : ((w & 15u) | ((y >> 6) << 4)),
                   m = c < 2 ? (z & 63u) : ((w >> 4) | ((z >> 6) << 4));
    ds[u] = __fmul_rn(__fmul_rn(d, (float)s), sc);
    dm[u] = __fmul_rn(__fmul_rn(dmin, (float)m), sc);
  }
}
// Decodes sixteen bytes of codes, qs[32 c + 16 j ..] of a block: their low nibbles are the elements 64 c + 16 j .. + 15
// (sub-block 2 c), their high nibbles the elements 32 further on (sub-block 2 c + 1). qh: with Q5, the bytes
// qh[16 j ..], whose bits 2 c and 2 c + 1 are the codes' fifth bits. g: with GN, the gains of the block's 256 columns,
// 16-byte aligned. Out: the sixteen b0 (l1) and b1 (h1) of the first run, l2 and h2 of the second.
template <int Q5, int GN>
__device__ __forceinline__ void sixteen_bytes(const uint4 qs, const uint4 qh, int c, int j, const float (&ds)[2],
                                              const float (&dm)[2], const float *g, uint4 &l1, uint4 &h1, uint4 &l2,
                                              uint4 &h2) {
  const unsigned qw[4] = {qs.x, qs.y, qs.z, qs.w}, hw[4] = {qh.x, qh.y, qh.z, qh.w};
  unsigned a[4], b[4], cc[4], dd[4];
#pragma unroll
  for (int w = 0; w < 4; w++) {
    unsigned c1 = qw[w] & 0x0F0F0F0Fu, c2 = (qw[w] >> 4) & 0x0F0F0F0Fu;
    if (Q5) { // the fifth bits
      c1 |= ((hw[w] >> (2 * c)) & 0x01010101u) << 4;
      c2 |= ((hw[w] >> (2 * c + 1)) & 0x01010101u) << 4;
    }
    float4 g1 = make_float4(1.f, 1.f, 1.f, 1.f), g2 = g1;
    if (GN) {
      const float *gp = g + 64 * c + 16 * j + 4 * w;
      g1 = __ldg((const float4 *)gp), g2 = __ldg((const float4 *)(gp + 32));
    }
    four_codes<GN>(c1, ds[0], dm[0], g1, a[w], b[w]);
    four_codes<GN>(c2, ds[1], dm[1], g2, cc[w], dd[w]);
  }
  l1 = make_uint4(a[0], a[1], a[2], a[3]), h1 = make_uint4(b[0], b[1], b[2], b[3]);
  l2 = make_uint4(cc[0], cc[1], cc[2], cc[3]), h2 = make_uint4(dd[0], dd[1], dd[2], dd[3]);
}

// A block of threads makes one tile at a time: BM rows of one expert's matrix times NS of that expert's slots, over all
// of K, a stage of BK4 columns (one block of weights a row) at a time. In shared memory a stage lies in strips: strip k
// holds bytes 16 k to 16 k + 15 of every row, one row after the other, so that a fragment of 16 rows is 256 bytes on
// end. Every fragment here begins on a multiple of 32 bytes and has the leading dimension 16. A strip is SKEW bytes
// longer than its data, so that neighbouring strips begin in different banks.
// - A row of shared memory made longer than its data. See NVIDIA, CUDA C++ Best Practices Guide, "Shared Memory and
//   Memory Banks", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html: a bank
//   serves in turn the addresses that one request sends it, and M. Harris, "An Efficient Matrix
//   Transpose in CUDA C/C++", https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/:
//   a tile declared wider than its data avoids such conflicts. There a tile's rows get one element
//   more; here a strip gets 32 bytes more.
// - Where a fragment may begin. See the guide's "Warp Matrix Functions" (this file's head), of load_matrix_sync and of
//   store_matrix_sync: mptr points at the matrix's first element and must be aligned to 256 bits.
//   Of the leading dimension the page says that it has to be a whole number of 16-byte units for
//   __half (8 elements a unit) and for float (4), and names no figure for signed char or int;
//   here it is 16 bytes for the signed bytes and 128 bytes or more for the sums.
constexpr int BM = 128, BK4 = 256, SKEW = 32;
// The tile as a product of signed bytes: 2 BM rows (b0's, then b1's) by NC = NS NL columns (limb by limb). Shared
// memory: two buffers of weights and three of activations; the 2 BM x NC sums are laid over them at the end. Eight
// warps in WROWS x WCOLS, each with WR x WC fragments of 16 x 16.
template <int NL, int NS> struct Tile4 {
  static constexpr int NC = NS * NL;
  static constexpr int A_STRIP = 2 * BM * 16 + SKEW, B_STRIP = NC * 16 + SKEW;
  static constexpr int A_BYTES = BK4 / 16 * A_STRIP, B_BYTES = BK4 / 16 * B_STRIP;
  static constexpr int SMEM = 2 * A_BYTES + 3 * B_BYTES; // more than the 2 BM x NC ints of the sums take
  static constexpr int WCOLS = NC >= 64 ? 2 : 1, WROWS = 8 / WCOLS;
  static constexpr int WR = 2 * BM / WROWS / 16, WC = NC / WCOLS / 16;
  static_assert(SMEM <= 227 * 1024, "a thread block has at most 227 KB of shared memory (compute capability 9.0)");
  static_assert(A_STRIP % 32 == 0 && B_STRIP % 32 == 0,
                "every strip begins a multiple of 32 bytes from the buffer's start");
};
template <int NL> struct Args4 {
  MoeTables m;
  int rows, K;
  const u8p *F;        // [expert][rows][K / 256 blocks]
  const float *G;      // a gain a column, or null
  float sc;            // a power of two: the weights' common scale (wb::Blocks::scale)
  const int8_t *x[NL]; // [slot][ldx]
  size_t ldx;
  long long *acc; // [row][rank < T]
};
// acc[row][rank] = sum over k < K of W_e[row][k] x[slot][k], exactly, for the BM rows from BM blockIdx.x on of expert
// e = blockIdx.y and all of e's tokens, NS at a time. Q5: the blocks are Q5_K, else Q4_K. GN: a.G holds a gain a
// column. While a stage is multiplied, the next stage's weight bytes are read into registers and its activations are
// copied into shared memory. The loop's length, and so the running time, depends on how many tokens the router sent to
// e. Needs 256 threads, K a multiple of 256, rows a multiple of BM, Tile4::SMEM bytes of dynamic shared memory.
template <int NL, int NS, int Q5, int GN>
__global__ void __launch_bounds__(256, 1) k_grouped_blocks(const Args4<NL> a) {
  typedef Tile4<NL, NS> Tiling;
  constexpr int BB = Q5 ? 176 : 144, QS = Q5 ? 48 : 16;
  // 16 and not more: with 128 here, the static shared memory of every kernel of the program came out rounded up to a
  // multiple of 128 bytes; without this file the figures are multiples of 16. The tiles need a start at a multiple of
  // 32 bytes: that is tested here. It hangs on where the runtime puts the buffer and on nothing of a request. After a
  // trap the next CUDA call fails, and the process ends.
  extern __shared__ __align__(16) unsigned char sm[];
  if (__cvta_generic_to_shared(sm) % 32) __trap();
  unsigned char *const Bs = sm + 2 * Tiling::A_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, wr = warp / Tiling::WCOLS, wc = warp % Tiling::WCOLS;
  const int row = tid >> 1, h = tid & 1; // two threads a row: thread h takes the nibble groups c = h and c = h + 2
  const int nk = a.K / BK4, tm = blockIdx.x, e = blockIdx.y;
  int rank0 = a.m.rbase[e];
  const int count = indexes_below(rank0, a.m.cnt[e], a.m.T, a.m.T, INDEX_MOE_TABLE);
  const int slot0 = index_below(a.m.sbase[e], a.m.S - (count + 63) / 64 * 64 + 1, INDEX_MOE_TABLE);
  for (int s0 = 0; s0 < count; s0 += NS) {
    const int nreal = min(NS, count - s0), rk0 = rank0 + s0, sl0 = slot0 + s0;
    const u8p *blocks = a.F + ((size_t)e * a.rows + (size_t)tm * BM + row) * nk * BB;
    uint4 hd, qs[4], qh[2];
    auto fetch = [&](int kt) {
      const u8p *p = blocks + (size_t)kt * BB;
      hd = *(const uint4 *)p;
#pragma unroll
      for (int z = 0; z < 2; z++)
#pragma unroll
        for (int j = 0; j < 2; j++) qs[2 * z + j] = *(const uint4 *)(p + QS + 32 * (h + 2 * z) + 16 * j);
      if (Q5) qh[0] = *(const uint4 *)(p + 16), qh[1] = *(const uint4 *)(p + 32);
    };
    auto decode = [&](int buf, int kt) {
      unsigned char *A0 = sm + buf * Tiling::A_BYTES + row * 16, *A1 = A0 + BM * 16;
#pragma unroll
      for (int z = 0; z < 2; z++) {
        const int c = h + 2 * z;
        float ds[2], dm[2];
        two_scales(hd, c, a.sc, ds, dm);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          uint4 l1, h1, l2, h2;
          sixteen_bytes<Q5, GN>(qs[2 * z + j], qh[j], c, j, ds, dm, GN ? a.G + kt * BK4 : nullptr, l1, h1, l2, h2);
          const int o = (4 * c + j) * Tiling::A_STRIP; // the columns 64 c + 16 j .. + 15 are strip 4 c + j
          *(uint4 *)(A0 + o) = l1;
          *(uint4 *)(A1 + o) = h1;
          *(uint4 *)(A0 + o + 2 * Tiling::A_STRIP) = l2;
          *(uint4 *)(A1 + o + 2 * Tiling::A_STRIP) = h2;
        }
      }
    };
    auto loadB = [&](int buf, int kt) { // 16 bytes a copy, not waited for here
      unsigned char *B = Bs + buf * Tiling::B_BYTES;
#pragma unroll
      for (int i = tid; i < Tiling::NC * (BK4 / 16); i += 256) {
        const int r = i / (BK4 / 16), c = i % (BK4 / 16);
        __pipeline_memcpy_async(B + c * Tiling::B_STRIP + r * 16,
                                a.x[r / NS] + (size_t)(sl0 + r % NS) * a.ldx + kt * BK4 + c * 16, 16);
      }
    };
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[Tiling::WR][Tiling::WC];
#pragma unroll
    for (int i = 0; i < Tiling::WR; i++)
#pragma unroll
      for (int j = 0; j < Tiling::WC; j++) wmma::fill_fragment(acc[i][j], 0);
    fetch(0);
    loadB(0, 0);
    __pipeline_commit();
    for (int kt = 0; kt < nk; kt++) {
      decode(kt & 1, kt); // this buffer was last read two stages ago, ahead of the barrier of the stage before
      if (kt + 1 < nk) {
        fetch(kt + 1);
        loadB((kt + 1) % 3, kt + 1); // three buffers: this one was last read two stages ago
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      __syncthreads();
      const signed char *A = (const signed char *)(sm + (kt & 1) * Tiling::A_BYTES),
                        *B = (const signed char *)(Bs + (kt % 3) * Tiling::B_BYTES);
#pragma unroll
      for (int k = 0; k < BK4 / 16; k++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> fa[Tiling::WR];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> fb[Tiling::WC];
#pragma unroll
        for (int i = 0; i < Tiling::WR; i++)
          wmma::load_matrix_sync(fa[i], A + k * Tiling::A_STRIP + (wr * Tiling::WR + i) * 16 * 16, 16);
#pragma unroll
        for (int j = 0; j < Tiling::WC; j++)
          wmma::load_matrix_sync(fb[j], B + k * Tiling::B_STRIP + (wc * Tiling::WC + j) * 16 * 16, 16);
#pragma unroll
        for (int i = 0; i < Tiling::WR; i++)
#pragma unroll
          for (int j = 0; j < Tiling::WC; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
      }
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    int *C = (int *)sm;
#pragma unroll
    for (int i = 0; i < Tiling::WR; i++)
#pragma unroll
      for (int j = 0; j < Tiling::WC; j++)
        wmma::store_matrix_sync(C + (wr * Tiling::WR + i) * 16 * Tiling::NC + (wc * Tiling::WC + j) * 16, acc[i][j],
                                Tiling::NC, wmma::mem_row_major);
    __syncthreads();
    // The limbs' sums joined in base 256; only real tokens are stored, in rank order.
    for (int i = tid; i < BM * NS; i += 256) {
      const int r = i / NS, tl = i % NS;
      long long v = 0;
#pragma unroll
      for (int l = NL - 1; l >= 0; l--)
        v = (v << 8) + (long long)C[r * Tiling::NC + l * NS + tl] +
            ((long long)C[(BM + r) * Tiling::NC + l * NS + tl] << 8);
      if (tl < nreal) a.acc[(size_t)(tm * BM + r) * a.m.T + rk0 + tl] = v;
    }
    __syncthreads();
  }
}

} // namespace pk

// The slots in a tile, one width for a whole launch: 16 where an expert meets at most 16 tokens on average, else 32. It
// depends only on the numbers of tokens and experts.
static inline int pk_width(const MoeTables &m) {
  return m.T <= 16 * m.NE ? 16 : 32;
}
// Allows the instance its dynamic shared memory, once in the life of the process. The macro reads the calling thread's
// CUDA status before it sets the attribute (common/common.h, CK_ATTRIBUTE).
template <int NL, int NS, int Q5, int GN> static inline void pk_blocks_ready() {
  static const bool once = [] {
    CK_ATTRIBUTE(pk::k_grouped_blocks<NL, NS, Q5, GN>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                 pk::Tile4<NL, NS>::SMEM);
    return true;
  }();
  (void)once;
}
// One launch; the status is read directly after it, and a launch that the driver rejects ends the process there.
template <int NL, int NS, int Q5, int GN> static inline void pk_blocks_launch(const pk::Args4<NL> &a, cudaStream_t st) {
  pk_blocks_ready<NL, NS, Q5, GN>();
  pk::k_grouped_blocks<NL, NS, Q5, GN><<<dim3(a.rows / pk::BM, a.m.NE), 256, pk::Tile4<NL, NS>::SMEM, st>>>(a);
  CK(cudaGetLastError());
}
template <int NL, int NS> static inline void pk_blocks_pick(const pk::Args4<NL> &a, int q5, cudaStream_t st) {
  if (q5)
    a.G ? pk_blocks_launch<NL, NS, 1, 1>(a, st) : pk_blocks_launch<NL, NS, 1, 0>(a, st);
  else
    a.G ? pk_blocks_launch<NL, NS, 0, 1>(a, st) : pk_blocks_launch<NL, NS, 0, 0>(a, st);
}
// Whether pk_grouped_blocks() can take the tensor: by its shape alone.
static inline bool pk_fits(const wb::Blocks &b) {
  return b.rows % pk::BM == 0;
}
// acc[row][rank] = sum over k of W_e[row][k] x[slot][k] for all experts of the routed tensor b, in one launch on st. x:
// NL planes of signed bytes, [slot][ldx], 16-byte aligned rows. acc: [b.rows][m.T]. b.gain: null or 16-byte aligned.
// Requires pk_fits(b) and b.inputs % 256 == 0; grouped_blocks() tests both before it calls this routine.
template <int NL>
static inline void pk_grouped_blocks(const MoeTables &m, const wb::Blocks &b, const int8_t *const *x, size_t ldx,
                                     long long *acc, cudaStream_t st) {
  pk::Args4<NL> a;
  a.m = m, a.rows = (int)b.rows, a.K = b.inputs, a.F = b.bytes, a.G = b.gain, a.sc = b.scale, a.ldx = ldx, a.acc = acc;
  for (int l = 0; l < NL; l++) a.x[l] = x[l];
  if (pk_width(m) == 16)
    pk_blocks_pick<NL, 16>(a, b.type == wb::Q5_K, st);
  else
    pk_blocks_pick<NL, 32>(a, b.type == wb::Q5_K, st);
}
// Makes the eight instances for activations of NL bytes ready. For set-up (grouped_blocks_warm_up()), so that no
// request meets the call that sets a kernel's attribute.
template <int NL> static inline void pk_grouped_blocks_ready() {
  pk_blocks_ready<NL, 16, 0, 0>(), pk_blocks_ready<NL, 16, 0, 1>(), pk_blocks_ready<NL, 16, 1, 0>();
  pk_blocks_ready<NL, 16, 1, 1>(), pk_blocks_ready<NL, 32, 0, 0>(), pk_blocks_ready<NL, 32, 0, 1>();
  pk_blocks_ready<NL, 32, 1, 0>(), pk_blocks_ready<NL, 32, 1, 1>();
}
