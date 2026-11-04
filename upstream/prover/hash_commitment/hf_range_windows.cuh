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

// Computes, for WfWindows (prover/hash_commitment/hf_wf_keys.cuh), the lookup keys of WF
// (blocks/iop/wf.h), the range proof that the committer program `cp_commit` makes on the committed
// weight matrix W. WF's two lookups are to show that each cell of W is in -128 .. 127, and that
// each weight stored as two cells, its low and high "digits", is in -4,096 .. 4,095. A plan
// (prover/hash_commitment/hf_range_plan.h) cuts the keys into "calls": a call is a part of one
// lookup, or a piece of a part. Call f has 2^lwr places with one 16-bit key for each of W's 2^lk
// columns, together its "window" of 2^n keys. key[pl 2^lk + c] is
//     in a call B (bytes):    W[y][c] + 128                    for the one row y at place pl,
//     in a call W (weights):  W[y0][c] + 256 W[y1][c] + 4,096  for its two rows, a weight's digits,
//     0, the padding key, at a place that has no rows.
// off_f is the constant added, and the call's table has 2^(k_f) rows (off_f, k_f and, below, ob_f:
// the plan's `off`, `k`, `ob`). Which rows are at a place is read only from HfRangeView::rows_at().
//
//   RangeWindows win;                 // call setup() before code that takes all free GPU memory:
//   win.setup(committed, plan);       // it allocates all the GPU memory that win owns
//   win.keep(pool, bytes, from, to);  // optional: for calls whose keys are written many times
//   win.lay(f, keys, stream);         // writes the keys of call f
//   win.end();                        // committed and plan must stay valid until then
// A unit (`UnitW`, prover/common/weight_layout.h; kind: `UnitW::X`) is the rows of W for one tensor
// (kind 0: two). Digit l of the weights in its weight row r is W's row UnitW::off + r + l
// weight_rows. The bits of a tensor's indices are split between weight row and column. setup()
// joins a call's places into runs (`Run`) of consecutive weight rows of one unit. The host holds
// each tensor once, as the model's file stores it. A run's keys are written in one of two ways.
// 1. Straight from the blocks, for a tensor in Q4_K or Q5_K blocks (GGUF's super-blocks of 256
//    weights) whose input index gives its low 8 bits or more to the lowest column bits; unit_of()
//    has the other conditions. At full size: kind 1 to 10, 96 % of W and more. A block is 256
//    consecutive inputs of a row of the model's file ("file row"), so 256 consecutive columns of
//    one weight row: a warp decodes it and writes its keys (k_block_keys); no integers are stored
//    or transposed. Of a tensor above `whole_most` bytes, only file rows of a run's output rows are
//    copied. A run is cut into jobs (`Job`) whose packed bytes fit a staging buffer ("slot" in
//    identifiers). Copies run on a separate stream, up to `slots` jobs ahead of the kernels:
//    without keep(), those for call f + 1 start by the end of lay(f), during the caller's work.
//    A tensor of a wider type that prover/weights_blocks/decode.cuh reads (Q8_0, F16, BF16) goes the
//    same way, with a run of 256 inputs as its block and k_block_keys_wide as its kernel.
// 2. Through a pass over the weights (WeightWalker::walk_units(), implemented by PinnedWeights),
//    for every other unit, such as kind 0, the embedding, the output head, a tensor of another type
//    (Q6_K, F32), and a unit in which a test's fault-injection hook has changed an integer.
//    walk_units() delivers a unit in chunks ("piece" in identifiers), as integers or byte planes,
//    which k_keys reads by the index rules of prover/hash_commitment/hf_range_run_rules.h and
//    prover/hash_commitment/hf_range_fold_rules.h (the `fold_` functions and `Fold` types).
// A cell of W is a signed byte by its type, and fold_cells() and k_block_keys yield every digit as
// one, so in an honest build a key of a call B is in its table; it is not checked. A key of a call
// W outside its table becomes 0 and sets the failure bit UNFIT, which WfWindows::all_fit() reads.
//
// The counts. Only tests call count_all(): the range proof counts its keys itself
// (prover/engine/wf_heavy.cuh). counters[32 ob_f + (b << k_f) + t] is the number of keys of call f,
// padding keys excluded, with the value t and the "identity" b, which is the column modulo 2^iota.
//
// GPU memory, all from the driver, in setup(): at full size, the staging buffers (10 x 256 MiB) and
// the WeightWalker's buffers for chunks of `piece_bytes` (0.6 GB); the plan's tables (25 MB).
// Secrets: the first two hold weights, and end() zeroes them (an exit() before end() zeroes
// neither: see end()). The caller must zero the buffer given to lend(), the pool given to keep(),
// the keys and the counters: they hold weights or values computed from them. The norms' gains
// (weights too), which the WeightWalker copies to the GPU at its first use, stay there after end():
// only PinnedWeights::end() zeroes and frees them, and cp_commit_main() calls it once, after its
// last GPU use of the weights. After an earlier return or exit() the clean-up removes files and
// zeroes host keys only: at worst all gains stay in the clear in GPU memory on the committer's
// machine (no client reads it) at least until the process ends.
//
// The decoding of a block is that of weights_blocks/decode.cuh, by its own routines.
// k_block_keys repeats statements of that file's k_decode_planes, which are adapted from ggml's
// dequantize_block_q4_K, _q5_K (ggml/src/ggml-cuda/convert.cu; MIT
// licence, "Copyright (c) 2023-2024 The ggml authors"; the notice is in NOTICE and THIRD_PARTY.md).
// For the wider types the routines are that file's values() and wide_weight(), used as in its
// k_decode_planes_wide.
//
// Include this file after prover/weights_commitment/lent_memory.h,
// prover/weights_commitment/common.cuh, prover/weights_commitment/arguments.h,
// prover/hash_commitment/hf_shape.cuh and prover/hash_commitment/hf_commit_weights.cuh.
#pragma once
#include "hash_commitment/hf_commit_handed.h" // hf::Committed; hf_commit_walk.h: Piece, PackedBlocks, WeightWalker
#include "hash_commitment/hf_range_run_rules.h"
#include "hash_commitment/commit_lie.cuh" // a test's malicious committer: empty in this repository
#include "hash_commitment/hf_range_plan.h"
#include "weights_blocks/decode.cuh"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <vector>
namespace rf {
namespace range_windows {
constexpr u32 NO_PLACE = ~(u32)0;

// Consecutive weight rows r0 .. r0 + n - 1 of one unit, at consecutive places of one call.
struct Run {
  u32 r0 = 0, n = 0;
  // A call B: digit l of weight row r0 + i is at place at[l] + i, unless at[l] is NO_PLACE.
  u32 at[::hfw::FOLD_DIGITS_MOST] = {NO_PLACE, NO_PLACE, NO_PLACE, NO_PLACE};
  u32 sum = NO_PLACE; // a call W: the pair of weight row r0 + i is at place sum + i
  u32 off = 0;        // off_f
  u32 second = 0;     // a call W: the coefficient of the pair's second row
  u32 rows = 0;       // 2^(k_f) - fewer_rows: a key of a call W must be below it
};

template <int N> __device__ __forceinline__ void store_keys(u16 *to, const int (&key)[N]) {
  if constexpr (N == 8) {
    *(uint4 *)to = make_uint4((u32)key[0] | (u32)key[1] << 16, (u32)key[2] | (u32)key[3] << 16,
                              (u32)key[4] | (u32)key[5] << 16, (u32)key[6] | (u32)key[7] << 16);
  } else if constexpr (N == 4) {
    *(uint2 *)to = make_uint2((u32)key[0] | (u32)key[1] << 16, (u32)key[2] | (u32)key[3] << 16);
  } else {
#pragma unroll
    for (int q = 0; q < N; q++) to[q] = (u16)key[q];
  }
}
// Writes the keys of weight row r0 + i of run s in the N columns from c, a multiple of N, on.
// as_read[l][q]: digit l in column c + q. Returns whether a key of a call W was outside its table.
template <int NL, int N>
__device__ __forceinline__ int lay_digits(const Run &s, int digits, u32 i, int lk, u64 c, const int (&as_read)[NL][N],
                                          u16 *__restrict__ keys) {
  const int (&d)[NL][N] = as_read;
  int unfit = 0, key[N];
#pragma unroll
  for (int l = 0; l < NL; l++) {
    if (l >= digits || s.at[l] == NO_PLACE) continue;
#pragma unroll
    for (int q = 0; q < N; q++) key[q] = d[l][q] + (int)s.off;
    store_keys<N>(keys + (((size_t)s.at[l] + i) << lk) + c, key);
  }
  if (s.sum != NO_PLACE) {
#pragma unroll
    for (int q = 0; q < N; q++) {
      key[q] = d[0][q] + (int)s.second * d[1][q] + (int)s.off;
      const bool outside = (u32)key[q] >= s.rows;
      unfit |= outside;
      if (outside) key[q] = 0;
    }
    store_keys<N>(keys + (((size_t)s.sum + i) << lk) + c, key);
  }
  return unfit;
}

// ---- 2. Through a pass over the weights ----

// Writes the keys of run s whose weights the chunk holds. u: the unit, restricted to the chunk
// (run_launches()). S = run_shape(lk, lk). A thread loads W consecutive columns at once; W > 1
// requires fold_wide(). Grid: s.n * ceil(2^lk / (256 W)) thread blocks of 256 threads, each for
// 256 W columns of one weight row. raise_flags() is called even if no key was outside its table.
template <class Form, int W>
__global__ void __launch_bounds__(256) k_keys(const Run s, const ::hfw::RunShape S, const ::hfw::FoldUnit u,
                                              const ::hfw::FoldSource src, u16 *__restrict__ keys) {
  const u32 tiles = (u32)((S.width + (u64)(256 * W) - 1) / (u64)(256 * W));
  // With W = 1, neighbours in the source are mostly consecutive rows, so i varies fastest.
  const u32 i = W == 1 ? blockIdx.x % s.n : blockIdx.x / tiles, tile = W == 1 ? blockIdx.x / s.n : blockIdx.x % tiles;
  const ::hfw::RunRow row = ::hfw::run_row(u, (u64)s.r0 + i);
  const u64 c = ((u64)tile * 256 + threadIdx.x) * W;
  ::hfw::RunHalf column;
  if (row.in && c < S.width) column = ::hfw::run_half(S, u, c, false);
  int unfit = 0;
  if (column.in) {
    int d[Form::digits][W];
    ::hfw::fold_cells<W>(Form(), src, row.at + column.at, d);
    unfit = lay_digits<Form::digits, W>(s, u.digits, i, S.lk, c, d, keys);
  }
  const int any = __syncthreads_or(unfit);
  if (!threadIdx.x) raise_flags(any ? (unsigned)UNFIT : 0u);
}

// keys[0 .. n - 1] = value. n must be a multiple of 8, and keys aligned to 16 bytes.
__global__ void k_fill(u16 *keys, size_t n, u32 value) {
  const u32 two = value | value << 16;
  GSL(i, n / 8) {
    ((uint4 *)keys)[i] = make_uint4(two, two, two, two);
  }
}

// ---- 1. Straight from the blocks ----

// Locates a weight's block among a job's staged file rows. Tensor index g is the OR of bits from
// the weight row and from the column. Index 0 is the input; its low 8 bits count within the block.
struct BlockMap {
  int groups = 0;
  int lcb = 0; // the unit has cells in 2^lcb blocks of 256 columns
  u32 left[3] = {1, 1, 1};
  int cls[3] = {'S', 'S', 'S'}, z[3] = {0, 0, 0}, take[3] = {0, 0, 0}, cb0[3] = {0, 0, 0};
  // `left` to `cb0`: as in hfw::FoldGroup. The steps: by how many staged file rows the block moves
  // when index g's bits from the row increase by 1, and when its field of the column number does.
  u64 row_step[3] = {0, 0, 0}, column_step[3] = {0, 0, 0};
  i64 bias = 0;
  u32 blocks_a_row = 0;
  const uint8_t *staged = nullptr; // aligned to 16 bytes
  const float *gains = nullptr;
  float scale = 1.f;
};
// Writes the keys of run s. A warp's task is one weight row in up to 8 consecutive blocks. A lane
// decodes eight weights of a block, as in qblocks::k_decode_planes and with its routines: four
// consecutive ones of sub-block 2 il and the four 32 further on. blockDim.x = 256, any grid.
template <int Q5>
__global__ void __launch_bounds__(256)
    k_block_keys(const Run s, const BlockMap m, const int lk, u16 *__restrict__ keys) {
  constexpr int BB = ::qblocks::Blk<Q5>::BB, QS = ::qblocks::Blk<Q5>::QS;
  const int lane = threadIdx.x & 31, il = lane >> 3, ir = lane & 7;
  const int together = m.lcb < 3 ? m.lcb : 3;
  const u64 tasks = (u64)s.n << (m.lcb - together);
  int unfit = 0;
  for (u64 task = (u64)blockIdx.x * 8 + (threadIdx.x >> 5); task < tasks; task += (u64)gridDim.x * 8) {
    const u32 i = (u32)(task >> (m.lcb - together));
    const u32 first = (u32)(task & (((u64)1 << (m.lcb - together)) - 1)) << together;
    u32 r = s.r0 + i, input_of_row = 0;
    i64 of_row = m.bias;
    for (int g = 0; g < m.groups; g++) {
      const u32 lf = r % m.left[g];
      r /= m.left[g];
      ::hfw::FoldGroup group;
      group.cls = m.cls[g], group.z = m.z[g], group.take = m.take[g];
      const u64 bits = ::hfw::fold_row_bits(group, lf);
      if (g == 0)
        input_of_row = (u32)bits;
      else
        of_row += (i64)(bits * m.row_step[g]);
    }
    for (u32 cb = first; cb < first + (1u << together); cb++) {
      const u32 c = cb << 8;
      const u32 sb = (input_of_row | (c & ((1u << m.take[0]) - 1))) >> 8; // the block's number in its file row
      i64 file_row = of_row;
      for (int g = 1; g < m.groups; g++)
        file_row += (i64)((u64)((c >> m.cb0[g]) & ((1u << m.take[g]) - 1)) * m.column_step[g]);
      const uint8_t *b = m.staged + ((size_t)file_row * m.blocks_a_row + sb) * BB;
      const uint4 head = *(const uint4 *)b;
      const uint32_t codes = *(const uint32_t *)(b + QS + 32 * il + 4 * ir);
      const uint32_t fifth = Q5 ? *(const uint32_t *)(b + 16 + 4 * ir) : 0;
      const ::qblocks::Sub A = ::qblocks::sub_of(head, 2 * il), B = ::qblocks::sub_of(head, 2 * il + 1);
      int d0[2][4], d1[2][4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float *g = m.gains ? m.gains + (size_t)sb * 256 + 64 * il + 4 * ir + e : nullptr;
        const float g0 = g ? __fmul_rn(g[0], m.scale) : m.scale, g1 = g ? __fmul_rn(g[32], m.scale) : m.scale;
        const uint32_t code = (codes >> (8 * e)) & 255, bits = (fifth >> (8 * e + 2 * il)) & 3;
        const int q0 = (int)((code & 15) | ((bits & 1) << 4)), q1 = (int)((code >> 4) | ((bits & 2) << 3));
        int lo, hi; // a cell is a signed byte: the second digit is cut to one, as decode_planes() stores it
        ::qblocks::limbs(::qblocks::weight(A.ds, A.dm, q0, g0), lo, hi);
        d0[0][e] = lo, d0[1][e] = (int)(int8_t)hi;
        ::qblocks::limbs(::qblocks::weight(B.ds, B.dm, q1, g1), lo, hi);
        d1[0][e] = lo, d1[1][e] = (int)(int8_t)hi;
      }
      const u64 column = (u64)c + 64 * il + 4 * ir;
      unfit |= lay_digits<2, 4>(s, 2, i, lk, column, d0, keys);
      unfit |= lay_digits<2, 4>(s, 2, i, lk, column + 32, d1, keys);
    }
  }
  const int any = __syncthreads_or(unfit);
  if (!threadIdx.x) raise_flags(any ? (unsigned)UNFIT : 0u);
}
// k_block_keys for the wider types (TY: 4 Q8_0, 5 F16, 6 BF16), with the same arguments, tasks and grids. Of a block a
// lane decodes eight weights, as in qblocks::k_decode_planes_wide and with its routines: elements 4 lane .. 4 lane + 3,
// and the four that lie 128 places further.
// Public sources for the techniques of this routine.
// The routine follows k_block_keys above. Its loop over the tasks, its way from a weight row and a block of columns to
// the block's address, the laying of the digits (lay_digits()) and the raising of the flag are that kernel's. What
// differs is the decoding of a lane's eight weights, which is that of qblocks::k_decode_planes_wide
// (weights_blocks/decode.cuh). The comment there names the techniques of the decoding and says where their public
// sources are cited; of them this routine has the launch bound with one argument, short loops unrolled completely, a
// template parameter whose instance the host code chooses, and products written as __fmul_rn. The three types' layouts
// and the rule that decodes them: value8() and value16() in weights_blocks/fold_rule.h, which cite public sources.
template <int TY>
__global__ void __launch_bounds__(256)
    k_block_keys_wide(const Run s, const BlockMap m, const int lk, u16 *__restrict__ keys) {
  constexpr int BB = ::qblocks::Wide<TY>::BB;
  const int lane = threadIdx.x & 31;
  const int together = m.lcb < 3 ? m.lcb : 3;
  const u64 tasks = (u64)s.n << (m.lcb - together);
  int unfit = 0;
  for (u64 task = (u64)blockIdx.x * 8 + (threadIdx.x >> 5); task < tasks; task += (u64)gridDim.x * 8) {
    const u32 i = (u32)(task >> (m.lcb - together));
    const u32 first = (u32)(task & (((u64)1 << (m.lcb - together)) - 1)) << together;
    u32 r = s.r0 + i, input_of_row = 0;
    i64 of_row = m.bias;
    for (int g = 0; g < m.groups; g++) {
      const u32 lf = r % m.left[g];
      r /= m.left[g];
      ::hfw::FoldGroup group;
      group.cls = m.cls[g], group.z = m.z[g], group.take = m.take[g];
      const u64 bits = ::hfw::fold_row_bits(group, lf);
      if (g == 0)
        input_of_row = (u32)bits;
      else
        of_row += (i64)(bits * m.row_step[g]);
    }
    for (u32 cb = first; cb < first + (1u << together); cb++) {
      const u32 c = cb << 8;
      const u32 sb = (input_of_row | (c & ((1u << m.take[0]) - 1))) >> 8; // the block's number in its file row
      i64 file_row = of_row;
      for (int g = 1; g < m.groups; g++)
        file_row += (i64)((u64)((c >> m.cb0[g]) & ((1u << m.take[g]) - 1)) * m.column_step[g]);
      const uint8_t *b = m.staged + ((size_t)file_row * m.blocks_a_row + sb) * BB;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int at = 128 * half + 4 * lane; // the first of the lane's four elements in the block
        float v[4];
        ::qblocks::values<TY, 4>(b, at, v);
        int d[2][4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float g = m.gains ? __fmul_rn(m.gains[(size_t)sb * 256 + at + e], m.scale) : m.scale;
          int lo, hi; // a cell is a signed byte: the second digit is cut to one, as decode_planes() stores it
          ::qblocks::limbs(::qblocks::wide_weight(v[e], g), lo, hi);
          d[0][e] = lo, d[1][e] = (int)(int8_t)hi;
        }
        unfit |= lay_digits<2, 4>(s, 2, i, lk, (u64)c + at, d, keys);
      }
    }
  }
  const int any = __syncthreads_or(unfit);
  if (!threadIdx.x) raise_flags(any ? (unsigned)UNFIT : 0u);
}

// ---- The counts ----

// Key e < 8 of eight. The counting kernels' threads take their eight keys in different orders, so
// that fewer of a warp's lanes add to one word at a time.
__device__ __forceinline__ u32 key_of(const uint4 &v, int e) {
  const u32 word = e < 4 ? (e < 2 ? v.x : v.y) : (e < 6 ? v.z : v.w);
  return (word >> (16 * (e & 1))) & 0xFFFF;
}
// Adds to counters[(b << k) + t] the number of the 8 n8 keys with the value t and an index equal to
// b modulo 2^iota. Thread blocks with blockIdx.y = y count identities y 2^lib to (y+1) 2^lib - 1,
// in 2^(lib + k) words of shared memory. Keys of 2^k or more are not counted. gridDim.x is free.
__global__ void __launch_bounds__(512)
    k_count(const u16 *__restrict__ keys, size_t n8, int iota, int k, int lib, u32 *__restrict__ counters) {
  extern __shared__ u32 bins[];
  const u32 words = 1u << (lib + k), mask = (1u << iota) - 1, from = blockIdx.y << lib, to = from + (1u << lib);
  for (u32 x = threadIdx.x; x < words; x += blockDim.x) bins[x] = 0;
  __syncthreads();
  GSL(q, n8) {
    const u32 b0 = (u32)(8 * q) & mask; // with iota >= 3 the eight keys have the identities b0 .. b0 + 7
    if (iota >= 3 && (b0 + 8 <= from || b0 >= to)) continue;
    const uint4 v = __ldcs((const uint4 *)keys + q);
#pragma unroll
    for (int x = 0; x < 8; x++) {
      const int e = (x + threadIdx.x) & 7;
      const u32 b = (b0 + e) & mask, t = key_of(v, e);
      if (b >= from && b < to && !(t >> k)) atomicAdd(&bins[((b - from) << k) + t], 1u);
    }
  }
  __syncthreads();
  // Every bin is added, an empty one too: a count is a hidden value, and no work is skipped because
  // one is 0.
  for (u32 x = threadIdx.x; x < words; x += blockDim.x) atomicAdd(&counters[((size_t)from << k) + x], bins[x]);
}
// Subtracts the padding keys, pub in each identity, from the count of the value 0.
__global__ void k_less_padding(u32 *counters, int iota, int k, u32 pub) {
  GSL(b, (size_t)1 << iota) counters[b << k] -= pub;
}
// call_of_row[y]: the call B that has W's row y, or NO_CALL; IN_A_CALL_W is set if a call W has it.
constexpr u32 IN_A_CALL_W = 1u << 31, NO_CALL = ~IN_A_CALL_W;
// Counts for the calls B, place by place, from the keys of call f. first[g] is the first counter of
// call g; kb and off_b are the calls B's k and offset. Not checked: 128 <= off_b <= 2^kb - 128.
//   DIGITS: f is a call W. A key minus off is a weight, lo + 256 hi; lo + off_b and hi + off_b are
//           the keys of its two digits in the calls B that have the pair's rows.
//   else:   f is a call B. Only the places whose row is in no call W are counted: the others are
//           counted from that call W.
// Needs 2^(iota + kb) words of shared memory, twice that with DIGITS; a thread block adds them to
// the counters when the rows' calls change and at its end. Any gridDim.x is valid.
// The ideas are public. A histogram in shared memory, here and in k_count: N. Sakharnykh,
//   "GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell", NVIDIA Technical Blog, 2015.
template <bool DIGITS>
__global__ void __launch_bounds__(512)
    k_count_places(const HfRangeView v, u32 f, u32 off, int kb, u32 off_b, const u32 *__restrict__ call_of_row,
                   const u64 *__restrict__ first, const u16 *__restrict__ keys, u32 *__restrict__ counters) {
  extern __shared__ u32 bins[];
  const u32 words = 1u << (v.iota + kb), mask = (1u << v.iota) - 1, all = DIGITS ? 2 * words : words;
  const u32 share = (v.places() + gridDim.x - 1) / gridDim.x, begin = blockIdx.x * share;
  const u32 end = begin + share < v.places() ? begin + share : v.places();
  u32 now[2] = {NO_CALL, NO_CALL};
  auto flush = [&]() { // adds the bins to the counters of the calls 'now', and zeroes them
    __syncthreads();
    for (u32 x = threadIdx.x; x < all; x += blockDim.x) {
      // Whether a row is in a call is public (the plan). Empty bins are added too, as in k_count.
      if (now[x / words] != NO_CALL) atomicAdd(&counters[first[now[x / words]] + x % words], bins[x]);
      bins[x] = 0;
    }
    __syncthreads();
  };
  flush();
  for (u32 pl = begin; pl < end; pl++) {
    u32 row[2], wt[2];
    const int rows = v.rows_at(f, pl, row, wt); // the same for all threads of the block
    if (rows != (DIGITS ? 2 : 1)) continue;
    const u32 c0 = call_of_row[row[0]], c1 = DIGITS ? call_of_row[row[1]] : NO_CALL;
    if (!DIGITS && (c0 & IN_A_CALL_W)) continue;
    if ((c0 & NO_CALL) != now[0] || (c1 & NO_CALL) != now[1]) {
      flush();
      now[0] = c0 & NO_CALL, now[1] = c1 & NO_CALL;
    }
    const uint4 *place = (const uint4 *)(keys + ((size_t)pl << v.lk));
    for (u32 q = threadIdx.x; q < 1u << (v.lk - 3); q += blockDim.x) {
      const uint4 eight = __ldcs(place + q);
#pragma unroll
      for (int x = 0; x < 8; x++) {
        const int e = (x + threadIdx.x) & 7;
        const u32 b = (8 * q + e) & mask;
        if (DIGITS) {
          const int weight = (int)key_of(eight, e) - (int)off, lo = (int)(int8_t)weight, hi = (weight - lo) >> 8;
          const u32 t0 = (u32)(lo + (int)off_b), t1 = (u32)(hi + (int)off_b);
          atomicAdd(&bins[(b << kb) + t0], 1u);
          if (!(t1 >> kb)) atomicAdd(&bins[words + (b << kb) + t1], 1u);
        } else {
          const u32 t = key_of(eight, e);
          if (!(t >> kb)) atomicAdd(&bins[(b << kb) + t], 1u);
        }
      }
    }
  }
  flush();
}
} // namespace range_windows

struct RangeWindows {
  typedef range_windows::Run Run;
  typedef range_windows::BlockMap BlockMap;
  // Test settings, read only during setup(). Only `fewer_rows` is meant to change a key or counter.
  // In identifiers, `walk` is a pass over the weights; a `walked` run's keys are written by one.
  // A test that is not part of this repository compares the keys with and without `plain`,
  // that is, the two ways of writing a run's keys, on synthetic weights.
  bool plain = false;                         // every unit goes through a pass over the weights
  int slots = 10;                             // the number of staging buffers
  size_t slot_bytes_most = (size_t)256 << 20; // limit on the bytes of a large tensor's job
  size_t whole_most = (size_t)64 << 20;       // a tensor of at most this many packed bytes is staged whole
  size_t piece_bytes = (size_t)128 << 20;     // `walk_units`: the target size of a chunk's integers
  bool count_every_call = false;              // `count_all` counts each call's own keys: two passes
  u32 fewer_rows =
      0; // a call W's keys are checked against a table with this many fewer rows, to test UNFIT

  RangeWindows() = default;
  RangeWindows(const RangeWindows &) = delete;
  RangeWindows &operator=(const RangeWindows &) = delete;
  ~RangeWindows() { end(); }

  // Returns 0, or 4 after a line on stderr for a plan it cannot handle. Allocates from the driver:
  // (cudaMalloc) in parentheses is not the macro of prover/weights_commitment/lent_memory.h.
  int setup(const ::hf::Committed &committed, const ::hf::HfRangePlan &plan) {
    end();
    c_ = &committed, plan_ = &plan;
    slots_ = (size_t)std::max(1, slots), piece_bytes_ = piece_bytes;
    const LayW &Y = committed.shape.Y;
    heads_ = (u64)committed.PO["H"] + (u64)committed.PO["G"], query_heads_ = (u64)committed.PO["H"];
    shape_ = ::hfw::run_shape(plan.lk, plan.lk);
    if (plan.lk != Y.lk || plan.m0 != Y.m0 || plan.lk < 3) return fault("the plan is not this layout's");
    units_.clear();
    for (size_t x = 0; x < Y.U.size(); x++) units_.push_back(unit_of(x, Y.U[x]));
    calls_.assign(plan.NF, Call());
    jobs_.clear(), kept_.clear(), kept_there_.clear(), cache_ = nullptr;
    slot_bytes_ = 0;
    call_of_row_.assign(plan.m0, range_windows::NO_CALL);
    for (u32 f = 0; f < plan.NF; f++)
      if (int rc = read_call(f)) return rc;
    // The one-pass count (k_count_places) requires: the second digit's coefficient is 256; the
    // calls B share one table, small enough for the kernel; each row of a call W is in a call B.
    one_walk_ = !count_every_call && plan.NB > 0 && plan.iota + plan.c[0].k + 1 <= COUNT_BITS;
    for (u32 f = 0; f < plan.NF; f++) {
      const bool bytes = plan.c[f].kind == HF_RANGE_B;
      if (bytes ? plan.c[f].k != plan.c[0].k || plan.c[f].off != plan.c[0].off : calls_[f].second != 256)
        one_walk_ = false;
    }
    for (u32 x : call_of_row_)
      if (x == (range_windows::NO_CALL | range_windows::IN_A_CALL_W)) one_walk_ = false;
    for (u32 f = 0; f < plan.NB; f++) calls_[f].rest = false;
    for (u32 x : call_of_row_)
      if (x != range_windows::NO_CALL && !(x & range_windows::IN_A_CALL_W)) calls_[x].rest = true;
    committed.weights.reserve(piece_bytes_);
    CK(cudaStreamCreateWithFlags(&copies_, cudaStreamNonBlocking));
    loaded_.assign(slots_, nullptr), decoded_.assign(slots_, nullptr), slot_.assign(slots_, nullptr);
    slot_bytes_ = (slot_bytes_ + 255) & ~(size_t)255;
    for (size_t s = 0; s < slots_; s++) {
      CK(cudaEventCreateWithFlags(&loaded_[s], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&decoded_[s], cudaEventDisableTiming));
      if (!slot_bytes_) continue;
      CK((cudaMalloc)((void **)&slot_[s], slot_bytes_));
      CK(::lent::poison(slot_[s], slot_bytes_)); // only if the test variable LENT_POISON is set
    }
    std::vector<u64> first(plan.NF);
    for (u32 f = 0; f < plan.NF; f++) first[f] = 32 * plan.c[f].ob;
    CK((cudaMalloc)((void **)&map_, plan.map.size() * 4));
    CK((cudaMalloc)((void **)&call_of_row_gpu_, call_of_row_.size() * 4));
    CK((cudaMalloc)((void **)&first_, first.size() * 8));
    CK(cudaMemcpy(map_, plan.map.data(), plan.map.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(call_of_row_gpu_, call_of_row_.data(), call_of_row_.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(first_, first.data(), first.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize()); // copies from pageable memory: the other streams do not wait for these calls' stream
    int device = 0;
    CK(cudaGetDevice(&device));
    CK(cudaDeviceGetAttribute(&processors_, cudaDevAttrMultiProcessorCount, device));
    CK_ATTRIBUTE(range_windows::k_count, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 << COUNT_BITS);
    CK_ATTRIBUTE(range_windows::k_count_places<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 << COUNT_BITS);
    CK_ATTRIBUTE(range_windows::k_count_places<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 << COUNT_BITS);
    next_copy_ = next_use_ = 0, ahead_end_ = jobs_.size();
    up_ = true;
    return 0;
  }
  // The bytes of the staging buffers only.
  size_t held_bytes() const { return slot_bytes_ * slots_; }
  // The size of the GPU buffer to lend() for count_all(): one call's keys.
  size_t room_bytes() const { return (size_t)2 << plan_->n; }
  // GPU memory that only count_all() uses. It then holds keys: its owner must zero it.
  void lend(void *room, size_t bytes) { room_ = (u16 *)room, lent_bytes_ = bytes; }

  // counters (formula at the top): 32 plan.bwK GPU words, zeroed by the caller. Blocks until done.
  void count_all(u32 *counters) {
    if (!room_ || lent_bytes_ < room_bytes()) stop("count_all() has no room for a call's keys");
    const cudaStream_t st = cudaStreamPerThread;
    const ::hf::HfRangePlan &plan = *plan_;
    const HfRangeView view = plan.view_with(map_);
    const int kb = plan.c[0].k;
    const size_t shared = (size_t)4 << (plan.iota + kb);
    const unsigned grid = (unsigned)std::min<u32>(plan.places(), (u32)processors_ * 2);
    for (u32 f = one_walk_ ? plan.NB : 0; f < plan.NF; f++) {
      lay((int)f, room_, st);
      count(f, room_, counters + 32 * plan.c[f].ob, st);
      if (!one_walk_) continue;
      range_windows::k_count_places<true><<<grid, 512, 2 * shared, st>>>(view, f, plan.c[f].off, kb, plan.c[0].off,
                                                                         call_of_row_gpu_, first_, room_, counters);
      CK(cudaGetLastError());
    }
    for (u32 f = 0; f < plan.NB && one_walk_; f++) {
      if (!calls_[f].rest) continue;
      lay((int)f, room_, st);
      range_windows::k_count_places<false><<<grid, 512, shared, st>>>(view, f, plan.c[f].off, kb, plan.c[0].off,
                                                                      call_of_row_gpu_, first_, room_, counters);
      CK(cudaGetLastError());
    }
    CK(cudaStreamSynchronize(st));
    next_use_ = next_copy_ = calls_[0].job_from; // call 0 is expected next: its copies start now
    copy_ahead();
  }
  // Adds call f's counts, padding keys excluded, to its 2^(k_f + iota) counters. keys: as in lay().
  void count(u32 f, const u16 *keys, u32 *counters_of_f, cudaStream_t st) const {
    const int iota = plan_->iota, k = plan_->c[f].k, lib = std::min(iota, std::max(0, COUNT_BITS - k));
    if (k > COUNT_BITS) stop("a table has more rows than the counting kernel's shared memory holds");
    const size_t shared = (size_t)4 << (lib + k);
    const dim3 grid((unsigned)processors_ * (shared <= (48 << 10) ? 4 : 1), 1u << (iota - lib));
    range_windows::k_count<<<grid, 512, shared, st>>>(keys, (size_t)1 << (plan_->n - 3), iota, k, lib, counters_of_f);
    CK(cudaGetLastError());
    range_windows::k_less_padding<<<1, 256, 0, st>>>(counters_of_f, iota, k, plan_->c[f].pub);
    CK(cudaGetLastError());
  }

  // Writes all 2^n keys of call f. On return the work is queued on st; a pass over the weights, if
  // needed, has finished. Fastest in rising order of f: the next call's copies are then under way.
  // keys: aligned to 16 bytes. The caller must order two lay() that share keys or keep()'s pool.
  void lay(int f, u16 *keys, cudaStream_t st) {
    if (!up_ || f < 0 || (u32)f >= plan_->NF)
      stop("lay() is called outside setup() and end(), or for no call of the plan");
    const Call &call = calls_[f];
    const int lk = plan_->lk;
    for (const Fill &x : call.fills) {
      const size_t n = (size_t)x.n << lk;
      range_windows::k_fill<<<gsl_blocks_of(n / 8, 256, 1 << 16), 256, 0, st>>>(keys + ((size_t)x.place << lk), n, x.value);
      CK(cudaGetLastError());
    }
    if (cache_ && (u32)f >= cache_from_ &&
        (u32)f < cache_to_) { // after `keep`: the `walked` runs' keys are copied from the cache
      fill_cache(st);
      const std::vector<Run> &there = cached_[(u32)f - cache_from_];
      for (size_t x = 0; x < call.walked.size(); x++) {
        const Run &run = call.walked[x].second;
        auto rows = [&](u32 place, u32 cached) {
          if (place == range_windows::NO_PLACE) return;
          CK(cudaMemcpyAsync(keys + ((size_t)place << lk), cache_ + ((size_t)cached << lk), (size_t)run.n << (lk + 1),
                             cudaMemcpyDeviceToDevice, st));
        };
        for (int l = 0; l < ::hfw::FOLD_DIGITS_MOST; l++) rows(run.at[l], there[x].at[l]);
        rows(run.sum, there[x].sum);
      }
    } else
      walk(call.walked, keys, st);
    // keep()'s calls are written repeatedly, so nothing is copied ahead for the calls after them.
    const bool held = !kept_.empty() && (u32)f >= cache_from_ && (u32)f < cache_to_;
    ahead_end_ = held ? keep_job_to_ : jobs_.size();
    if (next_use_ != call.job_from)
      next_use_ = next_copy_ = call.job_from; // out of order: what was copied ahead is discarded
    for (size_t j = call.job_from; j < call.job_to; j++) {
      copy_ahead();
      const Job &job = jobs_[j];
      const size_t s = j % slots_;
      BlockMap m = job.map;
      if (uint8_t *kept = kept_at(j)) { // copied once, on this stream, and then read from there
        if (kept_there_.insert(kept).second) copy(job, kept, st);
        m.staged = kept;
      } else {
        CK(cudaStreamWaitEvent(st, loaded_[s], 0));
        m.staged = slot_[s];
      }
      const u64 tasks = (u64)job.run.n << (m.lcb - std::min(m.lcb, 3));
      const unsigned grid = (unsigned)std::min<u64>((tasks + 7) / 8, (u64)processors_ * 64);
      switch (units_[job.unit].blocks.type) {
      case 0: range_windows::k_block_keys<0><<<grid, 256, 0, st>>>(job.run, m, lk, keys); break;
      case 1: range_windows::k_block_keys<1><<<grid, 256, 0, st>>>(job.run, m, lk, keys); break;
      case 4: range_windows::k_block_keys_wide<4><<<grid, 256, 0, st>>>(job.run, m, lk, keys); break;
      case 5: range_windows::k_block_keys_wide<5><<<grid, 256, 0, st>>>(job.run, m, lk, keys); break;
      case 6: range_windows::k_block_keys_wide<6><<<grid, 256, 0, st>>>(job.run, m, lk, keys); break;
      default: stop("a tensor's blocks have a type that no kernel of the windows reads");
      }
      CK(cudaGetLastError());
      if (!kept_at(j)) CK(cudaEventRecord(decoded_[s], st));
      next_use_ = j + 1;
    }
    copy_ahead();
  }

  /* For calls from .. to - 1 that are written many times (`HfRangePlan::pieces_of`: the pieces of
   * one part). `pool`, GPU memory, keeps the keys of their `walked` runs (the cache), if all fit: a
   * pass reads entire units, and a piece has few rows of many units. It then keeps each job's
   * packed bytes that fit, from their first copy on; a tensor staged whole, once. Not checked: the
   * calls are of one kind; from < to; pool is 16-byte aligned. Replaces an earlier `keep`. The pool
   * holds weights: its owner zeroes it. Returns the bytes used; *all: the bytes these jobs copy. */
  size_t keep(void *pool, size_t bytes, u32 from, u32 to, size_t *all = nullptr) {
    kept_.assign(jobs_.size(), nullptr);
    kept_there_.clear();
    keep_job_to_ = calls_[to - 1].job_to;
    std::map<size_t, uint8_t *> whole;
    size_t used = 0, sum = 0;
    cache_ = nullptr, cache_filled_ = false, cache_from_ = from, cache_to_ = to, cache_rows_ = 0;
    cached_.assign(to - from, {});
    for (u32 f = from; f < to; f++)
      for (const auto &entry : calls_[f].walked) {
        Run there = entry.second;
        auto rows = [&](u32 &place) {
          if (place == range_windows::NO_PLACE) return;
          place = cache_rows_, cache_rows_ += there.n;
        };
        for (u32 &place : there.at) rows(place);
        rows(there.sum);
        cached_[f - from].push_back(there);
      }
    if (const size_t need = ((size_t)cache_rows_ << (plan_->lk + 1)); pool && need && need <= bytes)
      cache_ = (u16 *)pool, used = (need + 255) & ~(size_t)255;
    for (size_t j = calls_[from].job_from; j < calls_[to - 1].job_to; j++) {
      const Job &job = jobs_[j];
      const size_t need = (job.width * job.height + 255) & ~(size_t)255;
      sum += job.width * job.height;
      if (units_[job.unit].whole && whole.count(job.unit)) {
        kept_[j] = whole[job.unit];
        continue;
      }
      if (!pool || used + need > bytes) continue;
      kept_[j] = (uint8_t *)pool + used;
      if (units_[job.unit].whole) whole[job.unit] = kept_[j];
      used += need;
    }
    if (all) *all = sum;
    return used;
  }

  // Zeroes the staging buffers, frees what setup() allocated, and calls PinnedWeights::release(),
  // which zeroes and frees the WeightWalker's buffers for chunks but not the norms' gains. May be
  // called twice. An exit() before it, as by stop() or a failed CK, zeroes none of these buffers.
  void end() {
    if (!up_) return;
    up_ = false;
    CK(cudaDeviceSynchronize());
    for (size_t s = 0; s < slots_; s++) {
      if (slot_[s]) CK(cudaMemset(slot_[s], 0, slot_bytes_));
      CK((cudaFree)(slot_[s]));
      CK(cudaEventDestroy(loaded_[s]));
      CK(cudaEventDestroy(decoded_[s]));
    }
    CK(cudaStreamDestroy(copies_));
    CK((cudaFree)(map_));
    CK((cudaFree)(call_of_row_gpu_));
    CK((cudaFree)(first_));
    map_ = call_of_row_gpu_ = nullptr, first_ = nullptr;
    slot_.clear(), loaded_.clear(), decoded_.clear();
    c_->weights.release();
    room_ = nullptr, lent_bytes_ = 0;
  }

  // For a report.
  size_t jobs() const { return jobs_.size(); }

private:
#ifdef FLASHPROVE_L4
  static constexpr int COUNT_BITS = 14; // 64 KiB; more passes, unchanged multiplicities.
#else
  static constexpr int COUNT_BITS = 15; // the counting kernel's bins: 2^15 words, 128 KiB of shared memory
#endif
  static constexpr size_t NO_UNIT = ~(size_t)0, NOT_COVERED = ~(size_t)0 - 1;
  struct Unit {
    ::hfw::FoldUnit u;  // index rules; each `step` is for the model file's order
    bool full = false; // every cell of the unit's rows, in all 2^lk columns, holds a weight's digit
    bool fast = false; // written straight from the blocks
    ::hf::PackedBlocks blocks;
    bool whole = true;   // the tensor is staged whole
    size_t period = 0;   // else: a job stages some consecutive rows of every `period` file rows
    u32 rows_a_step = 1; // weight rows for one step of the output index's row part
  };
  struct Fill {
    u32 place, n, value;
  };
  struct Job {
    size_t unit;
    Run run;
    BlockMap map;                     // except `staged`
    const uint8_t *from;              // in page-locked host memory
    size_t width, height, from_pitch; // the copy: height runs of width bytes, from_pitch apart
  };
  struct Call {
    std::vector<Fill> fills;
    std::vector<std::pair<size_t, Run>> walked; // (unit, run), in rising order of the unit
    size_t job_from = 0, job_to = 0;
    u32 second = 0;    // as Run::second
    bool rest = false; // a call B: some of its rows are in no call W
  };
  const ::hf::Committed *c_ = nullptr;
  const ::hf::HfRangePlan *plan_ = nullptr;
  ::hfw::RunShape shape_;
  u64 heads_ = 0, query_heads_ = 0;
  std::vector<Unit> units_;
  std::vector<Call> calls_;
  std::vector<Job> jobs_;
  size_t slots_ = 1, piece_bytes_ = 0; // the settings, as setup() found them
  size_t slot_bytes_ = 0, next_copy_ = 0, next_use_ = 0, lent_bytes_ = 0;
  std::vector<uint8_t *> slot_;
  std::vector<uint8_t *> kept_;    // `keep`: where each job's bytes stay, or null; empty: no `keep`
  size_t keep_job_to_ = 0;          // `keep`: the first job after its calls
  size_t ahead_end_ = 0;            // copy_ahead() stops before this job
  std::set<uint8_t *> kept_there_; // the addresses in kept_ that have been filled
  u16 *cache_ = nullptr;           // `keep`: the `walked` runs' keys, row after row; null: none
  u32 cache_from_ = 0, cache_to_ = 0, cache_rows_ = 0;
  bool cache_filled_ = false;
  std::vector<std::vector<Run>>
      cached_; // per call f of `keep`, per `walked` run: the run, its places changed to cache rows
  std::vector<cudaEvent_t> loaded_, decoded_;
  cudaStream_t copies_ = nullptr;
  u16 *room_ = nullptr;
  std::vector<u32> call_of_row_;                    // for each row of W; see IN_A_CALL_W
  u32 *map_ = nullptr, *call_of_row_gpu_ = nullptr; // GPU copies of plan_->map and call_of_row_
  u64 *first_ = nullptr;                            // each call's first counter
  bool one_walk_ = false;
  int processors_ = 1;
  bool up_ = false;
  // Prints a line on stderr and returns 4, which is stop()'s exit code. A failed CUDA call: 5 (CK).
  static int fault(const char *what) {
    fprintf(stderr, "the range file's windows: internal error: %s\n", what);
    return 4;
  }
  [[noreturn]] static void stop(const char *what) { exit(fault(what)); }
  // `cls`: columns get the lowest ('S') or highest ('P') of the z low index bits; 'A': all or none.
  Unit unit_of(size_t x, const UnitW &w) {
    Unit o;
    o.u = ::hfw::fold_unit_of(w, heads_);
    o.full = w.ncb == plan_->lk;
    for (int g = 0; g < o.u.groups; g++) {
      const ::hfw::FoldGroup &group = o.u.g[g];
      if (group.cls == 'A' && group.extent != (group.take ? (u64)1 << group.take : group.left)) o.full = false;
    }
    if (plain || !c_->weights.blocks(x, o.blocks)) return o;
    (void)o.blocks.block_bytes(); // ends the program before the windows copy a block, on a type that no kernel below reads
    // Straight from the blocks: two digits; the input index first, its low 8 bits or more in the
    // lowest column bits; no index of class 'A'; every other index moves by entire file rows.
    const ::hfw::FoldGroup &input = o.u.g[0];
    if (o.u.digits != 2 || o.u.groups < 2 || input.cls != 'S' || input.cb0 != 0 || input.step != 1 || input.take < 8 ||
        input.extent != o.blocks.inputs || w.ncb < 8)
      return o;
    for (int g = 1; g < o.u.groups; g++)
      if (o.u.g[g].cls == 'A' || o.u.g[g].step % (i64)o.blocks.inputs) return o;
    const size_t bytes = o.blocks.file_rows * o.blocks.row_bytes();
    if (bytes <= whole_most) {
      o.fast = true;
      return o;
    }
    // A large tensor: the last index is the output row and moves by one file row. Its row part must
    // be its low part, below `period`; the middle indices move by multiples of `period` file rows.
    const ::hfw::FoldGroup &output = o.u.g[o.u.groups - 1];
    if (output.cls != 'P' || output.step != (i64)o.blocks.inputs) return o;
    if (output.take && output.extent != (u64)1 << output.z) return o;
    o.period = output.take ? (size_t)1 << (output.z - output.take) : (size_t)output.extent;
    if (output.left != o.period || o.blocks.file_rows % o.period) return o;
    for (int g = 1; g + 1 < o.u.groups; g++) {
      if (o.u.g[g].cls != 'S' || (size_t)(o.u.g[g].step / (i64)o.blocks.inputs) % o.period) return o;
      o.rows_a_step *= (u32)o.u.g[g].left;
    }
    o.rows_a_step *= (u32)input.left;
    if (o.blocks.file_rows / o.period * o.blocks.row_bytes() > slot_bytes_most)
      return o; // one step does not fit a staging buffer
    o.whole = false, o.fast = true;
    return o;
  }
  // The unit of W's row y, the row's digit and its weight row. NO_UNIT: in no unit; W is 0 there.
  size_t locate(u32 y, int &digit, u32 &r) const {
    const std::vector<UnitW> &U = c_->shape.Y.U;
    size_t a = 0, b = U.size(); // the last unit that begins at or before y
    while (b - a > 1) {
      const size_t mid = (a + b) / 2;
      (U[mid].off <= y ? a : b) = mid;
    }
    digit = 0, r = 0;
    if (U.empty() || y < U[a].off || y >= U[a].off + U[a].rows) return NO_UNIT;
    const u64 weight_rows = units_[a].u.weight_rows;
    digit = (int)((y - U[a].off) / weight_rows), r = (u32)((y - U[a].off) % weight_rows);
    return a;
  }
  // Groups call f's places into fills (one key repeated), `walked` runs (for a pass) and jobs.
  int read_call(u32 f) {
    const ::hf::HfRangePlan &plan = *plan_;
    const ::hf::HfRangePlan::One &one = plan.c[f];
    Call &call = calls_[f];
    struct Stretch { // consecutive places with consecutive weight rows of one digit of one unit
      size_t unit;
      int digit;
      u32 r0, place, n;
    };
    std::vector<Stretch> stretches;
    u32 second = 0;
    for (u32 pl = 0; pl < plan.places(); pl++) {
      u32 row[2] = {0, 0}, wt[2] = {0, 0};
      const int rows = plan.rows_at(f, pl, row, wt);
      Stretch x{NOT_COVERED, 0, 0, pl, 1};
      if (rows) {
        if (wt[0] != 1 || rows != (one.kind == HF_RANGE_B ? 1 : 2))
          return fault("a place holds another number of rows, or another weight, than its call's kind has");
        x.unit = locate(row[0], x.digit, x.r0);
        for (int i = 0; i < rows; i++) {
          u32 &of_row = call_of_row_[row[i]];
          of_row = rows == 1 ? (of_row & range_windows::IN_A_CALL_W) | f : of_row | range_windows::IN_A_CALL_W;
        }
      }
      if (rows == 2) {
        int digit;
        u32 r;
        if (x.unit == NO_UNIT || locate(row[1], digit, r) != x.unit || x.digit != 0 || digit != 1 || r != x.r0 ||
            units_[x.unit].u.digits != 2 || (second && second != wt[1]))
          return fault("a place of a call on weights does not hold the two digits of one weight");
        second = wt[1];
      }
      if (!stretches.empty()) {
        Stretch &last = stretches.back();
        const bool rows_go_on =
            x.unit == NOT_COVERED || x.unit == NO_UNIT || (x.digit == last.digit && x.r0 == last.r0 + last.n);
        if (x.unit == last.unit && rows_go_on) {
          last.n++;
          continue;
        }
      }
      stretches.push_back(x);
    }
    call.second = second;
    std::map<std::tuple<size_t, u32, u32>, Run> runs; // by (unit, r0, n): the digits of the same weights go together
    for (const Stretch &x : stretches) {
      if (x.unit == NOT_COVERED || x.unit == NO_UNIT) { // the padding key; a cell 0, whose key is off_f
        call.fills.push_back({x.place, x.n, x.unit == NO_UNIT ? one.off : 0});
        continue;
      }
      if (!units_[x.unit].full) call.fills.push_back({x.place, x.n, one.off});
      Run &run = runs[{x.unit, x.r0, x.n}];
      run.r0 = x.r0, run.n = x.n, run.off = one.off, run.second = second;
      run.rows = ((u32)1 << one.k) - fewer_rows;
      (one.kind == HF_RANGE_B ? run.at[x.digit] : run.sum) = x.place;
    }
    call.job_from = jobs_.size();
    for (const auto &entry : runs) {
      const size_t unit = std::get<0>(entry.first);
      if (units_[unit].fast)
        cut_into_jobs(unit, entry.second);
      else
        call.walked.push_back({unit, entry.second});
    }
    call.job_to = jobs_.size();
    return 0;
  }
  void cut_into_jobs(size_t unit, const Run &run) {
    const Unit &o = units_[unit];
    const size_t row_bytes = o.blocks.row_bytes(), inputs = o.blocks.inputs;
    Job job;
    job.unit = unit;
    BlockMap &m = job.map;
    m.groups = o.u.groups, m.lcb = o.u.ncb - 8, m.blocks_a_row = (u32)(inputs / 256);
    m.gains = o.blocks.gains, m.scale = o.blocks.scale;
    for (int g = 0; g < o.u.groups; g++) {
      const ::hfw::FoldGroup &group = o.u.g[g];
      m.left[g] = (u32)group.left, m.cls[g] = group.cls, m.z[g] = group.z, m.take[g] = group.take, m.cb0[g] = group.cb0;
    }
    if (o.whole) {
      for (int g = 1; g < o.u.groups; g++) {
        const ::hfw::FoldGroup &group = o.u.g[g];
        m.row_step[g] = (u64)group.step / inputs;
        m.column_step[g] = m.row_step[g] << (group.cls == 'P' ? group.z - group.take : 0);
      }
      job.run = run, job.from = o.blocks.host;
      job.width = o.blocks.file_rows * row_bytes, job.height = 1, job.from_pitch = job.width;
      slot_bytes_ = std::max(slot_bytes_, job.width);
      jobs_.push_back(job);
      return;
    }
    const size_t periods = o.blocks.file_rows / o.period, most = slot_bytes_most / (periods * row_bytes);
    const int last = o.u.groups - 1;
    for (u32 done = 0; done < run.n;) {
      const size_t r = (size_t)run.r0 + done, end = (size_t)run.r0 + run.n;
      const size_t from = r / o.rows_a_step, to = std::min(from + most, (end - 1) / o.rows_a_step + 1),
                   steps = to - from;
      const u32 n = (u32)(std::min(end, to * o.rows_a_step) - r);
      job.run = run;
      job.run.r0 = (u32)r, job.run.n = n;
      for (u32 &place : job.run.at)
        if (place != range_windows::NO_PLACE) place += done;
      if (job.run.sum != range_windows::NO_PLACE) job.run.sum += done;
      for (int g = 1; g < last; g++) m.row_step[g] = m.column_step[g] = (u64)o.u.g[g].step / inputs / o.period * steps;
      m.row_step[last] = 1, m.column_step[last] = steps, m.bias = -(i64)from;
      job.from = o.blocks.host + from * row_bytes;
      job.width = steps * row_bytes, job.height = periods, job.from_pitch = o.period * row_bytes;
      slot_bytes_ = std::max(slot_bytes_, job.width * job.height);
      jobs_.push_back(job);
      done += n;
    }
  }
  // Starts the next jobs' copies while staging buffers are free. A buffer is free once the kernel
  // of the job that was in it is queued: the copy waits for that kernel through a `decoded_` event.
  // The ideas are public. Copies that overlap kernels, ordered by events:
  //   NVIDIA, "CUDA C++ Best Practices Guide", "Asynchronous and Overlapping Transfers with
  //   Computation" (that section has no word on events);
  //   M. Harris, "How to Overlap Data Transfers in CUDA C/C++", NVIDIA Technical Blog, 2012.
  // cudaMemcpy2DAsync, cudaStreamWaitEvent (a wait for an event that was never recorded is no
  //   wait): NVIDIA, CUDA Runtime API.
  void copy_ahead() {
    for (; next_copy_ < ahead_end_ && next_copy_ < next_use_ + slots_; next_copy_++) {
      if (kept_at(next_copy_)) continue;
      const size_t s = next_copy_ % slots_;
      CK(cudaStreamWaitEvent(copies_, decoded_[s], 0));
      copy(jobs_[next_copy_], slot_[s], copies_);
      CK(cudaEventRecord(loaded_[s], copies_));
    }
  }
  // Writes runs' keys in passes over the weights. walked: (unit, run), in rising order of the unit.
  void walk(const std::vector<std::pair<size_t, Run>> &walked, u16 *keys, cudaStream_t st) {
    for (size_t a = 0; a < walked.size();) { // the runs of consecutive units share one pass
      size_t b = a + 1;
      while (b < walked.size() && walked[b].first <= walked[b - 1].first + 1) b++;
      c_->weights.walk_units(st, piece_bytes_, walked[a].first, walked[b - 1].first + 1, true,
                             [&](const ::hf::Piece &piece, cudaStream_t stream) {
                               for (size_t x = a; x < b; x++)
                                 if (walked[x].first == piece.unit) lay_piece(piece, walked[x].second, keys, stream);
                             });
      a = b;
    }
  }
  // One walk() for all calls given to keep(): every run's keys go to its rows of the cache. A cell
  // without a weight's digit is 0: the cache is first filled with the key of 0, the calls' offset.
  void fill_cache(cudaStream_t st) {
    if (cache_filled_) return;
    cache_filled_ = true;
    const size_t n = (size_t)cache_rows_ << plan_->lk;
    range_windows::k_fill<<<gsl_blocks_of(n / 8, 256, 1 << 16), 256, 0, st>>>(cache_, n, plan_->c[cache_from_].off);
    CK(cudaGetLastError());
    std::vector<std::pair<size_t, Run>> all;
    for (u32 f = cache_from_; f < cache_to_; f++)
      for (size_t x = 0; x < calls_[f].walked.size(); x++)
        all.push_back({calls_[f].walked[x].first, cached_[f - cache_from_][x]});
    std::stable_sort(all.begin(), all.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    walk(all, cache_, st);
  }
  static void copy(const Job &job, uint8_t *to, cudaStream_t st) {
    if (job.height == 1)
      CK(cudaMemcpyAsync(to, job.from, job.width, cudaMemcpyHostToDevice, st));
    else
      CK(cudaMemcpy2DAsync(to, job.width, job.from, job.from_pitch, job.width, job.height, cudaMemcpyHostToDevice, st));
  }
  uint8_t *kept_at(size_t job) const { return kept_.empty() ? nullptr : kept_[job]; }
  // Launches k_keys for the part of a run whose weights one chunk of a pass (`Piece`) holds.
  template <class Form>
  void lay_source(const ::hfw::FoldUnit &whole, int pg, const ::hf::Piece &piece, const ::hfw::FoldSource &src,
                  int planes, const Run &run, u16 *keys, cudaStream_t st) const {
    constexpr int W = Form::cell_bytes <= 2 ? 8 : 16 / Form::cell_bytes;
    std::vector<::hfw::RunLaunch> launches;
    if (!::hfw::run_launches(shape_, whole, pg, piece.first, piece.count, src, planes, Form::cell_bytes, W, launches))
      stop("a piece holds a range of an index of class P");
    for (const ::hfw::RunLaunch &x : launches) {
      const u64 per = 256 * (u64)(x.wide ? W : 1), blocks = (shape_.width + per - 1) / per * run.n;
      if (blocks > 0x7FFFFFFFull) stop("a launch would have more than 2^31 - 1 blocks");
      if (x.wide)
        range_windows::k_keys<Form, W><<<(unsigned)blocks, 256, 0, st>>>(run, shape_, x.u, src, keys);
      else
        range_windows::k_keys<Form, 1><<<(unsigned)blocks, 256, 0, st>>>(run, shape_, x.u, src, keys);
      CK(cudaGetLastError());
    }
  }
  // Byte planes: one array for each digit (`lo`, `hi`), holding that digit of the chunk's weights.
  void lay_piece(const ::hf::Piece &piece, const Run &run, u16 *keys, cudaStream_t st) const {
    const UnitW &w = *piece.u;
    ::hfw::FoldUnit whole = units_[piece.unit].u;
    ::hfw::FoldSource src;
    if (piece.lo) { // byte planes (hf_commit_walk.h, `Piece`); kind 7, 8, 10: the routed experts
      const bool routed = w.X == 7 || w.X == 8 || w.X == 10;
      whole.g[0].step = 1;
      whole.g[1].step = routed ? (i64)whole.g[0].extent : (i64)piece.ld;
      if (routed) whole.g[2].step = (i64)piece.ld;
      src.p[0] = piece.lo, src.p[1] = piece.hi;
      lay_source<::hfw::FoldPlanes<2>>(whole, routed ? 1 : -1, piece, src, 2, run, keys, st);
      return;
    }
    src.p[0] = piece.w0;
    if (w.X == 0 &&
        piece.w1 != (const void *)((const short *)piece.w0 + (size_t)query_heads_ * (size_t)whole.g[0].step))
      stop("the key matrix's rows must follow the query matrix's rows in one buffer");
    const int pg = ::hfw::fold_piece_group(w.X);
    if (w.X == 12)
      lay_source<::hfw::FoldIntegers<int, 4>>(whole, pg, piece, src, 1, run, keys, st);
    else
      lay_source<::hfw::FoldIntegers<short, 2>>(whole, pg, piece, src, 1, run, keys, st);
  }
};
} // namespace rf
