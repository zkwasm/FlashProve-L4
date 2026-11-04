// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// The second form of rowpass1_q4k()'s work. weights_blocks/row_pass_blocks.cuh includes this file if its line
// ROW_PASS_IN_ONE_KERNEL says 1.
//
// The row pass over Q4_K or Q5_K weight blocks in one kernel, hand-written: row_pass_fused(). For every column c of the
// blocks that a call names it computes the sums over the rows x < L of phi[x] times the low byte, and times the high
// byte, of the weight (x, c). The weights' bytes are to stay in registers: the kernel must be compiled without spills,
// that is, cuobjdump's resource listing must give it no stack frame and no local memory (nothing in the build checks
// that). A warp decodes them there, in the order in which the tensor cores' instruction wants its left operand, and
// multiplies them at once with the byte digits of phi (row_core_digits(), common/row_pass_core.cuh). The 32-bit sums are
// exact integers, so the outputs do not depend on the order of the additions.
//
// Who does what. A thread block takes 64 neighbouring columns through all L rows: quarter il (0 to 3) of the 256
// columns of one block in the row, which is the sub-blocks 2 il and 2 il + 1. Their codes are the 32 bytes at 32 il of
// a block's 128 code bytes: byte j holds column j of the first sub-block in its low four bits and column j of the
// second in its high four bits. The thread block's warps share out the rows in steps of 32. In a step, lane 4 g + t of
// a warp (g = 0 .. 7, t = 0 .. 3) reads the four code bytes 4 g .. 4 g + 3 of the eight rows 8 t .. 8 t + 7: 64
// weights, eight rows of eight columns.
//
// The instruction is mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32. Which lane holds which element: with groupID =
// lane >> 2 (our g) and threadID_in_group = lane % 4 (our t), A (16 x 32) lies in four registers of four bytes;
// registers 0 and 2 hold row g, registers 1 and 3 row g + 8; registers 0 and 1 hold columns 4 t .. 4 t + 3, registers 2
// and 3 columns 16 + 4 t .. 16 + 4 t + 3. B (32 x 8) lies in two registers: rows 4 t .. 4 t + 3 and rows 16 + 4 t .. 16
// + 4 t + 3 of column g. C (16 x 8) lies in four registers: row g in 0 and 1, row g + 8 in 2 and 3; columns 2 t and 2 t
// + 1.
//
// The file's layout of the weights stays; the freedom used is that a sum does not depend on the order of its terms (see
// k_row_pass). The next step's loads are issued before this step is decoded. The lanes of a warp exchange values
// through shared memory between two calls of __syncwarp(). The warps' sums are added in shared memory by atomicAdd();
// integer addition is associative, so the result does not depend on the order in which the warps arrive. That array is
// padded so that the lanes of a warp fall into different banks (RP_PITCH). A head is loaded as 16 bytes, eight digits
// as 8, four code bytes as 4; all are aligned. A step that has all its rows is read by a path without tests, and only
// the last, shorter step by a guarded one (read_step).
//
// Credits.
// - The idea is public: E. Frantar, R. L. Castro, J. Chen, T. Hoefler, D. Alistarh, "MARLIN: Mixed-Precision
//   Auto-Regressive Parallel Inference on Large Language Models" (arXiv:2408.11743,
//   https://arxiv.org/html/2408.11743v1), 3.4 "Kernel Design": "we want to dequantize directly into the right register
//   layout for subsequent Tensor Core calls"; "we fix the sub-tile width of each warp to 64 and further split the
//   computation across" K_sm; the warps' partial results "must then eventually be reduced in shared memory before the
//   final write-out". There the weights are laid out anew ahead of time; here the file's layout stays.
// - The idea is public: Y. Lin, H. Tang, S. Yang et al., "QServe: W4A8KV4 Quantization and System Co-design for
//   Efficient LLM Serving" (arXiv:2405.04532, https://arxiv.org/pdf/2405.04532), 5.2.2: "the efficient conversion from
//   ZINT4 (i.e., unsigned 4-bit integers with zero points) to SINT8 within the main loop"; 3.2: "all computations are
//   performed on INT8 tensor cores" (4-bit codes become signed bytes inside the product's loop).
// - The idea is public: NVIDIA, CUTLASS documentation, "Efficient GEMM in CUDA", "Pipelining"
//   (https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html): occupancy is rather low,
//   since as a rule no less than half of all the registers that a thread may use hold accumulators.
//   A thread's memory accesses are under way while that thread computes something else: by this
//   pipelining in software CUTLASS makes memory latency cost less (the next step's loads).
// - The idea is public: "Using CUDA Warp-Level Primitives" (NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), "Warp Synchronization":
//   at __syncwarp() the lanes of one warp wait for each other (the exchange through shared memory).
// - The idea is public: NVIDIA, CUDA Programming Guide, 5.4 "C/C++ Language Extensions", "Legacy Atomic Functions"
//   (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html): atomicAdd()
//   works on shared memory as well as on global memory: it loads the word at the address given and
//   stores it there again with a number added (the warps' sums).
// - The idea is public: M. Harris, "An Efficient Matrix Transpose in CUDA C/C++" (NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/), "Shared Memory Bank Conflicts": declare
//   the array in shared memory with rows longer than their data (RP_PITCH).
// - The idea is public: "CUDA Pro Tip: Increase Performance with Vectorized Memory Access" (NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/): the data
//   must be aligned for a vector load; such loads use the bandwidth better, cut latency and take
//   fewer instructions in all (the loads of 16 and 8 bytes; four code bytes read as one word).
// - The idea is public: J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, T. Dao, "FlashAttention-3: Fast and
//   Accurate Attention with Asynchrony and Low-precision" (arXiv:2407.08608, https://arxiv.org/pdf/2407.08608), 3.3: a
//   permutation of registers "permutes its columns", and the other operand gets "a matching row permutation": the
//   summed index is permuted alike in both operands at run time (the order of the summed places in k_row_pass).
// - The idea is public: NVIDIA's CuTe documentation ("CuTe's support for Matrix Multiply-Accumulate instructions",
//   TiledMMAs; https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0t_mma_atom.html) has each thread access
//   neighbouring m-coordinates by permuting a tiled product's M-mode, in the same way for A and for C; the
//   N- and the K-mode may be permuted too, it says. Zhou et al., "Linear Layouts" (arXiv:2505.23819, 5.2 and
//   footnote 3), pre-shuffle the wider operand so that the 4-bit operand can be loaded with wider vector instructions,
//   and note that with mma similar optimizations need no pre-shuffle. Neither treats packed codes in a file's own
//   layout, or a load from global memory without shared memory. (the order of output rows and of the summed places)
// - The idea is public: QServe (arXiv:2405.04532, 5.2.3) applies a group's integer scale and zero point to each 4-bit
//   weight in the main loop, before the product on the INT8 tensor cores; Ootomo, Ozaki and Yokota (arXiv:2306.11975)
//   cut wider numbers into 8-bit slices and multiply the slices on integer matrix units. In QServe the result fits one
//   byte; here it is cut into two. (scale and minimum before the product; two signed bytes)
// - The idea is public: tinygemm (Elhoushi and Johnson, "any4", arXiv:2507.04610, 4.2) keeps the weights on the left
//   for up to eight activation rows, "flipping the order of matrices presented to mma". (the weights as the left
//   operand)
// - The idea is public: ggml's CUDA loader for Q4_K (llama.cpp, load_tiles_q4_K) works out each row's (scale, minimum)
//   pairs once and stores them in the tile from which the threads that need them read. There for a whole thread block;
//   here inside a warp. (how a row's scale pairs go round)
// - The idea is public: tinygemm (arXiv:2507.04610, 4.2) lets matrix data "pass directly from global memory to
//   registers", as "there is little to no weight reuse opportunity for small batch sizes, and loads into registers can
//   be asynchronous". There the weights are first laid out in the format that mma expects. (no shared memory for the
//   codes; the next loads issued into registers)
// The sources of the decoding rule and of the digits and their join are cited in decode.cuh and row_pass_core*.cuh.
#pragma once
#include "common/row_pass_core.cuh"
#include "weights_blocks/decode.cuh"
namespace kq4 {
constexpr int RP_WARPS = 4; // the number of warps in a thread block
// 32-bit words from one digit's 64 sums to the next digit's in shared memory. Lane 4 g + t adds at word 2 t * 68 + g +
// a constant, which is 8 t + g modulo 32: 32 different banks.
constexpr int RP_PITCH = 68;
static_assert(EDEG == 3, "k_row_pass is written for EDEG == 3: three tiles of eight digits");
// One warp-wide matrix product with accumulation on signed bytes, sums = A * B + sums, by the PTX instruction
// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32: A is 16 x 32, row-major; B is 32 x 8, column-major; sums is 16 x 8.
// Each lane hands in its fragments: four .s32 registers of sums (read as C, written as D), four .b32 registers of A and two
// of B, each holding four .s8 elements. Without .satfinite the sums wrap. All 32 lanes of a warp call it. Public source for
// the form of the asm statement: Inline PTX Assembly in CUDA, https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html.
// The asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that
// project's notice is in NOTICE. A later comparison found a like form in Marlin's kernel, flashinfer/mma.cuh, SageAttention.
__device__ __forceinline__ void mma_bytes(int (&sums)[4], const uint32_t (&a_regs)[4], const uint2& b_regs) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(sums[0]), "+r"(sums[1]), "+r"(sums[2]), "+r"(sums[3])
        : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), "r"(b_regs.x), "r"(b_regs.y));
}
// What a lane reads from GPU memory for one step of 32 rows.
template <int Q5> struct Step {
  uint4 head;                 // the head of the block in row 'lane' of the step
  uint32_t codes[8];          // code bytes 4 g .. 4 g + 3 of the quarter, rows 8 t .. 8 t + 7
  uint32_t fifth[Q5 ? 8 : 1]; // Q5_K: the bytes that hold these columns' fifth bits, same rows
  uint2 digits[3];            // rows 8 t .. 8 t + 7 of the digit vectors g, 8 + g and 16 + g
};
// Where a lane reads: the addresses for the step at hand. They advance by RP_WARPS steps at a time.
struct Reader {
  const uint8_t *head;  // the block in row 'lane' of the step
  const uint8_t *codes; // code byte 4 g of the quarter in row 8 t of the step
  const uint8_t *fifth; // Q5_K: the byte with the fifth bits of the same four columns, same row
  const int8_t *digits; // row 8 t of the step in digit vector g
};
// Reads the step that begins at row x0. pitch: bytes from a row to the next. A step that has all its 32 rows is read
// without tests. In the last, shorter step the rows from L on read as zero bytes, which decode to the weight 0, and the
// digits from byte K on read as 0. If x0 >= L it leaves s as it is; the caller's loop then ends; s enters no product.
template <int Q5>
__device__ __forceinline__ void read_step(Step<Q5> &s, const Reader &at, uint32_t pitch, int x0, int L, int K) {
  const int lane = threadIdx.x & 31, t = lane & 3;
  if (x0 + 32 <= L) {
    s.head = *(const uint4 *)at.head;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      s.codes[i] = *(const uint32_t *)(at.codes + i * pitch);
      if (Q5) s.fifth[i] = *(const uint32_t *)(at.fifth + i * pitch);
    }
#pragma unroll
    for (int n = 0; n < 3; n++) s.digits[n] = *(const uint2 *)(at.digits + (size_t)8 * n * K);
  } else if (x0 < L) {
    s.head = x0 + lane < L ? *(const uint4 *)at.head : uint4{0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const bool in = x0 + 8 * t + i < L;
      s.codes[i] = in ? *(const uint32_t *)(at.codes + i * pitch) : 0;
      if (Q5) s.fifth[i] = in ? *(const uint32_t *)(at.fifth + i * pitch) : 0;
    }
#pragma unroll
    for (int n = 0; n < 3; n++)
      s.digits[n] = x0 + 8 * t < K ? *(const uint2 *)(at.digits + (size_t)8 * n * K) : uint2{0, 0};
  }
}
// The lowest bytes of four numbers in one word: the first number's in the word's lowest byte.
// The idea is public: __byte_perm(), NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics"
// (https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html): out of two
// unsigned integers of 32 bits each, it gives back the bytes that are chosen.
// The three __byte_perm calls and their selectors are the same as those for the first row in NVIDIA CUTLASS,
// include/cutlass/transform/thread/transpose.h (int8_t 4x4 transpose), BSD-3-Clause; that file's notice follows, as in
// a copy with the years 2017 - 2022.
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
__device__ __forceinline__ uint32_t lowest_bytes(const int (&v)[4]) {
  return __byte_perm(__byte_perm(v[0], v[1], 0x0040), __byte_perm(v[2], v[3], 0x0040), 0x5410);
}
// The bytes of four weights of one column, in four neighbouring rows: the low bytes in lo, the high bytes in hi, the
// first row's in the lowest byte. ds, dm: the four rows' pairs (qblocks::Sub); the rows' codes are byte p of codes[0 ..
// 3]; G as in qblocks::weight(). The rule itself is decode.cuh's: weight() and limbs().
__device__ __forceinline__ void four_rows(const float (&ds)[4], const float (&dm)[4], const uint32_t (&codes)[4], int p,
                                          float G, uint32_t &lo, uint32_t &hi) {
  int l[4], h[4];
#pragma unroll
  for (int r = 0; r < 4; r++)
    qblocks::limbs(qblocks::weight(ds[r], dm[r], (int)__byte_perm(codes[r], 0, 0x4440 + p), G), l[r], h[r]);
  lo = lowest_bytes(l), hi = lowest_bytes(h);
}
// For the blocks S0 + blockIdx.x / 4 (block S is block S % nsb in the rows of expert S / nsb; an expert's blocks begin
// expert_bytes apart) and their quarters blockIdx.x % 4: writes outLo[c ostr] and outHi[c ostr] for the quarter's 64
// columns, c = 64 blockIdx.x .. 64 blockIdx.x + 63. digits: 24 vectors of K bytes, as row_core_digits<1>() writes
// them, each zero from byte L on, at a multiple of 8; K is a multiple of 16, L <= K, L <= 33,025 (so that the 32-bit
// sums are exact). F (a multiple of 16), G, sc as in rowpass1_q4k(). It requires blockDim.x == 32 RP_WARPS.
//
// The product. In the instruction's terms A's 16 rows are 16 columns of the weight matrix, the 32 summed places are the
// step's 32 rows, and B's eight columns are the eight digits of one coordinate of phi. A sum does not depend on the
// order of its terms, so the 32 rows may be given to the 32 places in any order that A and B share: place 4 t + j is
// row 8 t + j and place 16 + 4 t + j is row 8 t + 4 + j (j < 4), so that a lane's eight rows are neighbours. Likewise
// the instruction's row g may be any column: in product p (0 to 3), row g is the column in the low bits of the lane's
// code byte p, column 4 g + p of the quarter, and row g + 8 the one in its high bits, column 32 + 4 g + p. So a lane's
// 64 weights are exactly its parts of the four products' A, for the low bytes and again for the high bytes.
//
// The sub-blocks' pairs (ds, dm): a lane needs those of its eight rows, and the eight lanes with the same t need the
// same ones. Each lane computes the two pairs of one row, and the warp passes them round through its own piece of
// shared memory.
// The public sources for this routine's techniques are named at this file's head.
template <int Q5>
__global__ void __launch_bounds__(32 * RP_WARPS)
    k_row_pass(const uint8_t *__restrict__ F, int L, int nsb, size_t expert_bytes, int S0,
               const int8_t *__restrict__ digits, int K, const float *__restrict__ G, float sc, Ext *__restrict__ outLo,
               Ext *__restrict__ outHi, int ostr) {
  __shared__ int total[2 * 24 * RP_PITCH]; // [low or high byte][digit][place of the column]
  __shared__ float4 pairs[RP_WARPS][32];   // a warp's own: (ds, dm) of the two sub-blocks, for the step's 32 rows
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int S = S0 + blockIdx.x / 4, il = blockIdx.x & 3, sb = S % nsb;
  const uint8_t *block = F + (size_t)(S / nsb) * expert_bytes + (size_t)sb * qblocks::Blk<Q5>::BB;
  const uint32_t pitch = (uint32_t)nsb * qblocks::Blk<Q5>::BB;
  for (int i = threadIdx.x; i < 2 * 24 * RP_PITCH; i += 32 * RP_WARPS) total[i] = 0;
  __syncthreads();
  float g0[4], g1[4]; // G for the lane's eight columns
#pragma unroll
  for (int p = 0; p < 4; p++) {
    const float *gp = G ? G + (size_t)sb * 256 + 64 * il + 4 * g + p : nullptr;
    g0[p] = gp ? __fmul_rn(gp[0], sc) : sc;
    g1[p] = gp ? __fmul_rn(gp[32], sc) : sc;
  }
  int sums[4][2][3][4] = {}; // [product][low or high byte][coordinate][the lane's four places of C]
  Step<Q5> now, next;
  Reader at;
  at.head = block + (size_t)(32 * warp + lane) * pitch;
  at.fifth = block + (size_t)(32 * warp + 8 * t) * pitch + 16 + 4 * g;
  at.codes = at.fifth - 16 + qblocks::Blk<Q5>::QS + 32 * il;
  at.digits = digits + (size_t)g * K + 32 * warp + 8 * t;
  read_step<Q5>(now, at, pitch, 32 * warp, L, K);
  for (int x0 = 32 * warp; x0 < L; x0 += 32 * RP_WARPS, now = next) {
    at.head += (size_t)32 * RP_WARPS * pitch, at.fifth += (size_t)32 * RP_WARPS * pitch;
    at.codes += (size_t)32 * RP_WARPS * pitch, at.digits += 32 * RP_WARPS;
    read_step<Q5>(next, at, pitch, x0 + 32 * RP_WARPS, L, K);
    const qblocks::Sub A = qblocks::sub_of(now.head, 2 * il), B = qblocks::sub_of(now.head, 2 * il + 1);
    __syncwarp(); // every lane has taken the last step's pairs out of 'pairs'
    // Row 8 t' + i' lies at 4 i' + t', so that the four entries which a warp reads at once (one i, t = 0 .. 3) are
    // neighbours.
    pairs[warp][4 * (lane & 7) + (lane >> 3)] = make_float4(A.ds, A.dm, B.ds, B.dm);
    __syncwarp();
    float ds0[2][4], dm0[2][4], ds1[2][4], dm1[2][4]; // [rows 0 to 3 or 4 to 7 of the lane's eight][row]
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 pair = pairs[warp][4 * i + t];
      ds0[i >> 2][i & 3] = pair.x, dm0[i >> 2][i & 3] = pair.y, ds1[i >> 2][i & 3] = pair.z,
                      dm1[i >> 2][i & 3] = pair.w;
    }
    uint32_t codes0[2][4], codes1[2][4]; // the codes of the first and of the second sub-block, a byte for each column
#pragma unroll
    for (int i = 0; i < 8; i++) {
      codes0[i >> 2][i & 3] = now.codes[i] & 0x0F0F0F0Fu;
      codes1[i >> 2][i & 3] = (now.codes[i] >> 4) & 0x0F0F0F0Fu;
      if (Q5) {
        codes0[i >> 2][i & 3] |= ((now.fifth[i] >> (2 * il)) & 0x01010101u) << 4;
        codes1[i >> 2][i & 3] |= ((now.fifth[i] >> (2 * il + 1)) & 0x01010101u) << 4;
      }
    }
#pragma unroll
    for (int p = 0; p < 4; p++) {
      uint32_t lo[4], hi[4];
#pragma unroll
      for (int half = 0; half < 2; half++) {
        four_rows(ds0[half], dm0[half], codes0[half], p, g0[p], lo[2 * half], hi[2 * half]);
        four_rows(ds1[half], dm1[half], codes1[half], p, g1[p], lo[2 * half + 1], hi[2 * half + 1]);
      }
#pragma unroll
      for (int n = 0; n < 3; n++) {
        mma_bytes(sums[p][0][n], lo, now.digits[n]);
        mma_bytes(sums[p][1][n], hi, now.digits[n]);
      }
    }
  }
  // The place of a column in 'total': 16 p + the instruction's row.
#pragma unroll
  for (int p = 0; p < 4; p++)
#pragma unroll
    for (int limb = 0; limb < 2; limb++)
#pragma unroll
      for (int n = 0; n < 3; n++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          atomicAdd(&total[(24 * limb + 8 * n + 2 * t + (j & 1)) * RP_PITCH + 16 * p + g + 8 * (j >> 1)],
                    sums[p][limb][n][j]);
  __syncthreads();
  // The join, as in row_pass_core::k_join: a coordinate is the sum over b of 2^(8 b) times digit b's sum, modulo p.
  for (int o = threadIdx.x; o < 2 * EDEG * 64; o += 32 * RP_WARPS) {
    const int place = o & 63, e = (o >> 6) % EDEG, limb = (o >> 6) / EDEG;
    const int *sum = total + (24 * limb + 8 * e) * RP_PITCH + place; // digit b's sum at sum[b RP_PITCH]
    const u64 s = row_pass_core::join8(sum[0], sum[RP_PITCH], sum[2 * RP_PITCH], sum[3 * RP_PITCH], sum[4 * RP_PITCH], sum[5 * RP_PITCH],
                              sum[6 * RP_PITCH], sum[7 * RP_PITCH]);
    const int column = 4 * (place & 7) + (place >> 4) + 32 * ((place >> 3) & 1);
    set_coord((limb ? outHi : outLo)[((size_t)64 * blockIdx.x + column) * ostr], e, s);
  }
}
// Launches k_row_pass on stream st for the blocks S0 .. S0 + nS - 1: Q5_K if q5 is not 0, otherwise Q4_K. The grid
// follows nS alone.
static inline void row_pass_fused(const void *F, int L, int nsb, size_t expert_bytes, int S0, int nS,
                                  const int8_t *digits, int K, const float *G, float sc, int q5, Ext *outLo, Ext *outHi,
                                  int ostr, cudaStream_t st) {
  if (q5)
    k_row_pass<1><<<4 * (unsigned)nS, 32 * RP_WARPS, 0, st>>>((const uint8_t *)F, L, nsb, expert_bytes, S0, digits, K,
                                                              G, sc, outLo, outHi, ostr);
  else
    k_row_pass<0><<<4 * (unsigned)nS, 32 * RP_WARPS, 0, st>>>((const uint8_t *)F, L, nsb, expert_bytes, S0, digits, K,
                                                              G, sc, outLo, outHi, ostr);
  CK(cudaGetLastError());
}
} // namespace kq4
