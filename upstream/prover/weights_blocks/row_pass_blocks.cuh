// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// The row pass over a weight tensor stored as Q4_K or Q5_K blocks (GGUF's super-blocks of 256
// weights). A weight is 256 * hi + lo with signed bytes hi and lo. For a vector phi of field
// elements (`Ext`, prover/common/common.h), rowpass1_q4k() computes for each requested column the
// sums over the rows x of phi[x] * lo and of phi[x] * hi. All experts of a mixture-of-experts
// tensor share phi. Once per call, each coordinate of each phi[x] is split into eight signed bytes,
// its digits. Called by BlockWeights::row_pass() (prover/weights_blocks/block_weights.cuh).
// Blocks of the wider types that decode.cuh reads (Q8_0, F16, BF16) go the same way.
//
// Public source for writing the digits before the loop over the experts: LLVM Project, "LLVM's
// Analysis and Transform Passes" (https://llvm.org/docs/Passes.html), "licm: Loop Invariant Code
// Motion"; there a compiler's pass over instructions, here done by hand with kernel launches.
// Public source for one launch over several experts' sums: NVIDIA, "GPU Performance Background
// User's Guide" (several times more thread blocks than multiprocessors),
// https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html.
// Public source: a grid-stride loop (the two kernels below): M. Harris, "CUDA Pro Tip: Write
// Flexible Kernels with Grid-Stride Loops" (NVIDIA Technical Blog),
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// The decoding's sources are cited in prover/weights_blocks/decode.cuh, those of row_core_digits()
// to row_core_join() in prover/common/row_pass_core.cuh and prover/common/row_pass_core_kernels.cuh.
#pragma once
#include "common/row_pass_core.cuh"
#include "weights_blocks/decode.cuh"
#include <algorithm>
// Selects one of two variants of rowpass1_q4k() for Q4_K and Q5_K blocks. Both multiply the digits with the
// weights' bytes into exact 32-bit sums and join each eight sums into one coordinate of a result,
// so both are meant to write the same bytes. 0: a loop here: for each expert, blocks_to_cells()
// decodes the blocks into bytes and one cuBLAS call multiplies them. 1: one kernel
// (prover/weights_blocks/row_pass_fused.cuh) decodes, multiplies on the tensor cores and joins, in
// one launch for all experts of a call. Arguments, checks and scratch size are the same. That kernel
// reads Q4_K and Q5_K blocks and no other type. So with the 1 below a build contains both variants,
// and at run time the blocks' type alone chooses: Q4_K and Q5_K take variant 1, the wider types
// variant 0. A type that decode.cuh does not read ends the program before either
// (qblocks::block_bytes()). Two tests that are not part of this repository check only the variant
// chosen for Q4_K and Q5_K, against CPU sums and expected digests.
#define ROW_PASS_IN_ONE_KERNEL 1
#if ROW_PASS_IN_ONE_KERNEL
#include "weights_blocks/row_pass_fused.cuh"
#endif
namespace kq4 {
// out[x] = phi[x] for x < L, zero for L <= x < K.
// The loop form is public: a grid-stride loop (the post cited at the top of this file).
static __global__ void k_phi_padded(const Ext *__restrict__ phi, int L, int K, Ext *__restrict__ out) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < K; x += gridDim.x * blockDim.x)
    out[x] = x < L ? phi[x] : Ext{};
}
// Used by variant 0 only.
// `sums` contains, for each expert of a group (see EXPERTS_A_JOIN), first the low bytes' sums of
// its `inputs` columns, then the high bytes'. Column u of the group is column u % inputs of its
// expert u / inputs. Column u0 + j goes to olo[j * ostr] and ohi[j * ostr], for j < n.
// The loop form is public: a grid-stride loop (the post cited at the top of this file).
static __global__ void k_spread(const Ext *__restrict__ sums, int inputs, size_t u0, size_t n, int ostr,
                                Ext *__restrict__ olo, Ext *__restrict__ ohi) {
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n; j += (size_t)gridDim.x * blockDim.x) {
    const size_t u = u0 + j, at = u / inputs * 2 * inputs + u % inputs;
    olo[j * ostr] = sums[at];
    ohi[j * ostr] = sums[at + inputs];
  }
}
static inline size_t up256(size_t n) {
  return (n + 255) / 256 * 256;
}
} // namespace kq4
// The largest number of experts in a group, whose sums variant 0 joins and stores together.
constexpr int EXPERTS_A_JOIN = 16;
// The bytes of scratch that rowpass1_q4k() needs for L rows of nsb blocks in a call that touches
// at most `experts` experts. In order (K: L rounded up to a multiple of 16): one expert's decoded
// bytes, phi zero-padded to K elements, a group's joined sums, and row_core_scratch() for a group.
static inline size_t rowpass1_q4k_scratch(int L, int nsb, int experts) {
  const size_t K = (size_t)(L + 15) / 16 * 16, inputs = (size_t)256 * nsb,
               group = (size_t)std::min(std::max(experts, 1), EXPERTS_A_JOIN);
  return kq4::up256(2 * inputs * K) + kq4::up256(K * sizeof(Ext)) + kq4::up256(group * 2 * inputs * sizeof(Ext)) +
         row_core_scratch(1, (int)K, (int)(group * 2 * inputs)) + 256;
}
// Writes column c's two sums over rows x < L to outLo[c * ostr] and outHi[c * ostr]. L <= 33,025
// keeps the 32-bit sums exact. Exits with code 1 on a bad L, nS, Lexp or sc. Not checked: nsb, S0,
// the alignments, scratch's size, sc = 2^e, hi fits a byte. All work is queued on st.
// F: blocks, 16-byte aligned, nsb per row, Lexp rows per expert (0: L), the first L summed. Blocks
// S0 .. S0 + nS - 1 (S = nsb * expert + block in row) give the columns c < 256 * nS.
// G: a multiplier for each of the row's 256 * nsb columns (the same for every expert), or null.
// sc = 2^e with an integer e, 0 <= e <= 24. G, sc and type: as in decode.cuh.
// scratch: rowpass1_q4k_scratch(L, nsb, experts) bytes, 256-byte aligned, for one stream at a time.
static inline void rowpass1_q4k(const void *F, int L, int nsb, int S0, int nS, const Ext *phi, Ext *outLo, Ext *outHi,
                                void *scratch, cudaStream_t st = 0, int type = 0, int Lexp = 0, int ostr = 1,
                                const float *G = nullptr, float sc = 4096.f) {
  if (!(sc >= 1.f && sc <= 16777216.f)) {
    printf("internal error in the row pass over blocks: the tensor's scale is %g; it must be from 1 to 2^24\n", sc);
    exit(1);
  }
  if (L <= 0 || L > 33025 || nS <= 0 || (Lexp && Lexp < L)) {
    printf("internal error in the row pass over blocks: %d rows (allowed: 1 to 33025), %d blocks (at least 1), %d rows "
           "per expert (0, or at least the number of rows)\n",
           L, nS, Lexp);
    exit(1);
  }
  if (!Lexp) Lexp = L;
  const int K = (L + 15) / 16 * 16, inputs = 256 * nsb;
  const int e_first = S0 / nsb, e_last = (S0 + nS - 1) / nsb, group = std::min(e_last - e_first + 1, EXPERTS_A_JOIN);
  int8_t *cells = (int8_t *)scratch;
  Ext *padded = (Ext *)(cells + kq4::up256((size_t)2 * inputs * K));
  Ext *sums = (Ext *)((char *)padded + kq4::up256((size_t)K * sizeof(Ext)));
  void *below = (char *)sums + kq4::up256((size_t)group * 2 * inputs * sizeof(Ext));
  row_core_check(K, K);
  kq4::k_phi_padded<<<(K + 255) / 256, 256, 0, st>>>(phi, L, K, padded);
  CK(cudaGetLastError());
  const Ext *in[1] = {padded};
  row_core_digits<1>(K, in, below, st);
  const size_t expert_bytes = (size_t)Lexp * nsb * qblocks::block_bytes(type);
#if ROW_PASS_IN_ONE_KERNEL
  if (type == 0 || type == 1) { // Q4_K, Q5_K
    kq4::row_pass_fused(F, L, nsb, expert_bytes, S0, nS, (const int8_t *)below, K, G, sc, type, outLo, outHi, ostr, st);
    CK(cudaGetLastError());
    return;
  }
#endif
  Ext *out[1] = {sums};
  for (int e0 = e_first; e0 <= e_last; e0 += group) {
    const int n = std::min(group, e_last - e0 + 1); // the experts e0 .. e0 + n - 1
    for (int i = 0; i < n; i++) {
      // If K != L, clears the buffer so that rows L to K - 1 are zero, like phi's entries there.
      if (K != L) CK(cudaMemsetAsync(cells, 0, (size_t)2 * inputs * K, st));
      blocks_to_cells((const uint8_t *)F + (e0 + i) * expert_bytes, L, nsb, cells, cells + (size_t)inputs * K, K, st, G,
                      sc, type);
      row_core_product<1>(cells, K, i * 2 * inputs, 2 * inputs, group * 2 * inputs, K, below, st);
    }
    row_core_join<1>(n * 2 * inputs, K, out, below, st);
    // The call needs these experts' blocks `first` to `behind` - 1; its columns start at block S0.
    const size_t first = std::max(S0, e0 * nsb), behind = std::min(S0 + nS, (e0 + n) * nsb),
                 columns = (behind - first) * 256, c0 = (first - S0) * 256 * ostr;
    kq4::k_spread<<<(unsigned)((columns + 255) / 256), 256, 0, st>>>(sums, inputs, (first - (size_t)e0 * nsb) * 256,
                                                                    columns, ostr, outLo + c0, outHi + c0);
    CK(cudaGetLastError());
  }
  CK(cudaGetLastError());
}
