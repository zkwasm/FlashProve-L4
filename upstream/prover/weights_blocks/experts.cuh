// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The matrix products of a mixture-of-experts layer's routed experts, in exact integers, and the
// element-wise step between them. A routed tensor is stored on the GPU in the model file's format:
// rows of blocks of 256 weights of 4 to 16 bits (prover/weights_blocks/blocks.h). A weight's code
// represents a 16-bit integer w, by the rule of prover/weights_blocks/fold_rule.h, and
// w = 256 b1 + b0 with signed bytes b1 and b0. An activation is likewise NL signed bytes (limbs),
// limb l scaled by 256^l. So a product of weights and activations is a sum of products of
// signed-byte matrices. A plane is the array of one such byte for a whole matrix. Entry points:
//   grouped_blocks(): multiplies each expert's weight matrix with the activations of its tokens;
//   gate_up(): rounds the sums of the gating and the up projection, reads SiLU of the gating value
//     from a table, multiplies the two, and stores the product as the down projection's input.
// The caller's routing tables, MoeTables (prover/prover/forward_moe.cuh), give the tokens sorted by
// expert. A token's rank is its position in that order; its slot is its position once each expert's
// run is padded to a multiple of 64. prover/prover/forward.cuh includes this file after
// prover/prover/forward_moe.cuh and defines gate_key(), grid_blocks() and the range flags E_GI and E_U.
// grouped_blocks() stacks the bytes of both sides in one product. No source for this was found in
// public papers; the search was made for prover/hiding/gram_products.cuh.
// Public source: signed-byte operands with 32-bit integer sums: NVIDIA, cuBLAS documentation
// (https://docs.nvidia.com/cuda/cublas/), "cublasGemmEx()" (prover/common/blas.h has the details).
// Public source: the grid-stride loop (k_expert_rows, k_join_limbs): M. Harris, "CUDA Pro Tip:
// Write Flexible Kernels with Grid-Stride Loops" (NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
// ExpertsWork::fetch() copies the routing tables with cudaMemcpyAsync() and cudaStreamSynchronize()
// of NVIDIA's CUDA Runtime API, https://docs.nvidia.com/cuda/cuda-runtime-api/.
// Public source: padded rows in a shared-memory tile: M. Harris, "An Efficient Matrix Transpose in
// CUDA C/C++" (NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/).
// Public source: bank conflicts in shared memory: NVIDIA, CUDA C++ Best Practices Guide,
// "Shared Memory and Memory Banks". Both are for k_gate_up.
#pragma once
#include "common/blas.h"
#include "common/lie.cuh"
#include "weights_blocks/blocks.h"
#include "weights_blocks/decode.cuh"
#include <algorithm>
#include <cstdlib>
#include <string.h> // explicit_bzero
#include <vector>
// This value selects how grouped_blocks() computes; no build option changes it. 0: always its host
// loop, which calls cuBLAS. 1: one launch for all experts, pk_grouped_blocks()
// (prover/weights_blocks/experts_in_one_launch.cuh), if the tensor's type is Q4_K or Q5_K and
// pk_fits() accepts its shape, and otherwise the host loop. ffn_moe() (prover/prover/forward.cuh)
// passes only shapes that fit, so at 1 the host loop runs for tensors of type Q8_0, F16 and BF16
// alone. Both forms are meant to give the same sums.
// Three tests that are not part of this repository compare the sums with a plain kernel's: two
// test the selected form, and one tests both forms (it builds only at 1).
#ifdef FLASHPROVE_L4
#define EXPERTS_IN_ONE_LAUNCH 0
#else
#define EXPERTS_IN_ONE_LAUNCH 1
#endif
#if EXPERTS_IN_ONE_LAUNCH
#include "weights_blocks/experts_in_one_launch.cuh"
#endif

namespace expert_tiles {
template <int NL> struct Limbs {
  const int8_t *x[NL];
};
// Copies the activation rows of n consecutive slots next to each other, limb by limb: for l < NL,
// j < n and k < K, out[(l n + j) K + k] = p.x[l][(slot0 + j) ldx + k], where p.x[l] is limb l's
// plane. Requires K % 16 == 0 and 16-byte aligned rows of p.x and out.
//
// Public sources for the techniques of this routine.
// The idea is public: vectorized memory access in a copy kernel: "CUDA Pro Tip: Increase
// Performance with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The loop form is public: the grid-stride loop: the post cited at the top of this file.
template <int NL> __global__ void k_expert_rows(Limbs<NL> p, size_t ldx, int slot0, int n, int K, int8_t *out) {
  const size_t words = (size_t)NL * n * (K / 16);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
    const int c = (int)(i % (K / 16)), j = (int)(i / (K / 16) % n), l = (int)(i / (K / 16) / n);
    ((uint4 *)out)[i] = *(const uint4 *)(p.x[l] + (size_t)(slot0 + j) * ldx + 16 * c);
  }
}
// Adds up the byte products, scaled by powers of 256. With c = C + (l n + j) 2 rows, for row < rows
// and j < count: acc[row T + rank0 + j] = sum over l < NL of 256^l (c[row] + 256 c[rows + row]).
// The loop form is public: the grid-stride loop: the post cited at the top of this file.
template <int NL>
__global__ void k_join_limbs(const int *C, int rows, int n, int count, int rank0, int T, long long *acc) {
  const size_t all = (size_t)rows * count;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < all; i += (size_t)gridDim.x * blockDim.x) {
    const int row = (int)(i % rows), j = (int)(i / rows);
    long long v = 0;
    for (int l = NL - 1; l >= 0; l--) {
      const int *c = C + (size_t)(l * n + j) * 2 * rows;
      v = v * 256 + (long long)c[row] + 256 * (long long)c[rows + row];
    }
    acc[(size_t)row * T + rank0 + j] = v;
  }
}

// ag, au: the gating and the up projection's sums. kg, ku: the number of bits rounded off ag, au.
// gcl: if not 0, gate_key() clamps the key. Ts: the table of SiLU, read at key + 32768. h: four
// planes for the product's bytes. tq, t1, t2, th: see k_gate_up. err: E_GI (4) is set if gate_key()
// returns false, E_U (8) if an up value is outside -32768 .. 32639. Not checked: 1 <= kg, ku <= 32.
struct EpiArgs {
  MoeTables m;
  int fe, kg, ku, gcl;
  const long long *ag, *au; // [j < fe][rank < T]
  const short *Ts;
  int8_t *h[4]; // [slot][ldH]
  size_t ldH;
  u32 *tq, *t1, *t2; // [j < fe][rank < T]
  uint16_t *th;      // indexed as tq
  u32 *err;
};
// For each slot with a token and each feature j < fe: ag + 2^(kg-1) = 2^kg q1 + r1 with
// 0 <= r1 < 2^kg, likewise q2 and r2 from au and ku, and h = Ts[key(q1)] q2. With PROOF it stores
// the key and q2 (tq), r1 (t1), r2 (t2) and, under gcl, q1's high part (th). h's four signed bytes
// go to a.h[0 .. 3][slot][j]. MoeTables's tables give each group of 32 slots one expert or none. With
// an expert, a slot with no token gets zeros. Without one, the group is left unchanged, and no
// product reads it. A value out of range counts as 0 and sets err. Launch it through gate_up().
// Public sources for the techniques of this routine. The padded tile: see the top of this file.
// The activation and the product as one kernel: see k_epi_gate_up (prover/prover/forward.cuh).
// h's signed bytes. The idea is public: signed digits with carries: drouyang, "Signed Bucket
// Indexes for Multi-Scalar Multiplication (MSM)" (https://hackmd.io/@drouyang/signed-bucket-index).
// The page finds each carry by comparison. Here a byte s_i >= 128 becomes s_i - 256 and carries 1,
// and one addition makes all carries: byte i is (h + c) >> 8 i, with 128 in each byte of c below i.
template <int PROOF> __global__ void __launch_bounds__(256) k_gate_up(const EpiArgs a) {
  __shared__ unsigned char H[4][32][128 + 4];
  const int tn = blockIdx.y, j0 = blockIdx.x * 128, tid = threadIdx.x;
  if (a.m.expert_of(tn) < 0) return;
  int rk0 = a.m.tr[tn];
  const int nreal = indexes_below(rk0, a.m.tc[tn], 32, a.m.T, INDEX_MOE_TABLE);
  const int kg = a.kg, ku = a.ku;
  const long long hg = 1ll << (kg - 1), hu = 1ll << (ku - 1);
  const u32 mg = kg >= 32 ? ~0u : (1u << kg) - 1, mu = ku >= 32 ? ~0u : (1u << ku) - 1;
  u32 bad = 0;
  for (int i = tid; i < 128 * 32; i += 256) {
    const int jl = i >> 5, tl = i & 31;
    int h = 0;
    if (tl < nreal) {
      const size_t o = (size_t)(j0 + jl) * a.m.T + rk0 + tl;
      // lie::at() is a test hook: the identity. row: fe + j; the shared expert has rows < fe.
      const long long row = a.fe + j0 + jl, token = a.m.tok_of[tn * 32 + tl];
      const long long ag = lie::at(lie::ACCUMULATOR, row, token, a.ag[o]) + hg, au = lie::at(lie::ACCUMULATOR, row, token, a.au[o], 1) + hu;
      long long q1 = ag >> kg, q2 = au >> ku;
      u32 r1 = (u32)ag & mg, r2 = (u32)au & mu;
      q1 = lie::at(lie::GATE, row, token, q1);
      int key, gh;
      if (!gate_key(q1, a.gcl, key, gh)) {
        bad |= 4u;
        key = gh = 0;
        r1 = (u32)ag;
      }
      if (q2 < -32768 || q2 > 32639) {
        bad |= 8u;
        q2 = 0;
        r2 = (u32)au;
      }
      q2 = lie::at(lie::UP, row, token, q2);
      h = lie::at(lie::HIDDEN, row, token, lie::at(lie::SILU, row, token, (int)a.Ts[key + 32768]) * (int)q2);
      if (PROOF) {
        a.tq[o] = ((u32)key & 65535u) | ((u32)q2 << 16);
        a.t1[o] = r1;
        a.t2[o] = r2;
        if (a.gcl) a.th[o] = (uint16_t)gh;
      }
    }
    H[0][tl][jl] = (unsigned char)h;
    H[1][tl][jl] = (unsigned char)((h + 128) >> 8);
    H[2][tl][jl] = (unsigned char)((h + 32896) >> 16);
    H[3][tl][jl] = (unsigned char)((h + 8421504) >> 24);
  }
  if (bad) atomicOr(a.err, bad);
  __syncthreads();
  for (int i = tid; i < 4 * 32 * 32; i += 256) {
    const int l = i >> 10, tl = (i >> 5) & 31, w = i & 31;
    *(unsigned *)(a.h[l] + (size_t)(tn * 32 + tl) * a.ldH + j0 + 4 * w) = *(const unsigned *)&H[l][tl][4 * w];
  }
}
// Sets the index guard's flag (prover/common/index_guard.cuh) if ExpertsWork::fetch() found a table
// entry out of range. fetch() launches it in either case: its launches do not depend on the tables.
__global__ void k_table_outside(bool outside) {
  if (outside) index_is_outside(INDEX_MOE_TABLE);
}
} // namespace expert_tiles

// The work buffers of grouped_blocks()'s host loop and the host's copy of the routing tables. The
// caller owns the buffers, which are scratch. pk_grouped_blocks() uses neither them nor the copy.
struct ExpertsWork {
  cublasHandle_t blas = nullptr; // already set to the stream of the calls
  int8_t *planes = nullptr;      // one expert's tensor as two planes of bytes, at least 2 * rows * inputs bytes
  int8_t *rows = nullptr;        // the copied activation rows
  int *sums = nullptr;           // the library's 32-bit sums
  size_t planes_bytes = 0, rows_bytes = 0, sums_ints = 0;
  std::vector<int> count, slot0, rank0; // for each expert: its number of tokens, its first slot, its first rank
  // Copies the three tables to the host and waits for st. The host loop computes indexes from them,
  // so an expert with an entry out of range counts as one without tokens. pk_grouped_blocks() reads
  // the GPU's tables and takes an entry out of range as 0. In either case the index guard's flag is
  // set. Public source for the extra rows, up to a multiple of 4 tokens: see grouped_blocks().
  void fetch(const MoeTables &m, cudaStream_t st) {
    count.resize(m.NE), slot0.resize(m.NE), rank0.resize(m.NE);
    CK(cudaMemcpyAsync(count.data(), m.cnt, sizeof(int) * m.NE, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(slot0.data(), m.sbase, sizeof(int) * m.NE, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(rank0.data(), m.rbase, sizeof(int) * m.NE, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    bool outside = false;
    for (int e = 0; e < m.NE; e++) {
      const long long n = count[e];
      if (n >= 0 && n <= m.T && slot0[e] >= 0 && slot0[e] + (n + 3) / 4 * 4 <= m.S && rank0[e] >= 0 &&
          rank0[e] + n <= m.T)
        continue;
      count[e] = 0, outside = true;
    }
    expert_tiles::k_table_outside<<<1, 1, 0, st>>>(outside);
  }
  // Zeroes the host's copy of the tables, which depend on the request's routing. Call it once the
  // last product that reads them (a host loop) has been queued.
  void forget() {
    for (std::vector<int> *table : {&count, &slot0, &rank0}) explicit_bzero(table->data(), table->size() * sizeof(int));
  }
};
// Computes acc[row][rank] = sum over k of W_e[row][k] x[slot][k] for every expert e of the routed
// tensor b (W_e: the weights of e as integers) and each token of e, on the stream st. w.fetch()
// must have been called for m. x has NL planes of signed bytes, [slot][ldx], with 16-byte aligned
// rows. acc is [b.rows][m.T]. Not checked: b.inputs < 2^17, so that a sum of byte products, each at
// most 2^14, fits in 32 bits. Exits if the check below fails, even where pk_grouped_blocks() would
// compute. The host loop, for each expert with tokens: blocks_to_planes() decodes its weights into
// two planes, k_expert_rows copies its tokens' rows, one cuBLAS call multiplies the two planes with
// all NL planes of the rows, and k_join_limbs adds up the byte products; an expert with more tokens
// than the buffers hold takes several calls. The number and the sizes of the loop's launches depend
// on the request's routing. In pk_grouped_blocks() the launch does not, but the running time does
// (docs/limits-all-rows.tsv, row H14.2).
// Public source: extra rows, computed and discarded: JAX's Pallas documentation ("Grids and
// BlockSpecs", https://docs.jax.dev/en/latest/pallas/grid_blockspec.html), there when a "block shape
// does not divide" the array; here the extra rows come from the padding of the expert's own run of
// slots, each output column depends only on its own row, and only real tokens' columns are stored.
template <int NL>
static inline void grouped_blocks(const MoeTables &m, const wb::Blocks &b, const int8_t *const *x, size_t ldx,
                                  long long *acc, const ExpertsWork &w, cudaStream_t st) {
  const int rows = (int)b.rows, K = b.inputs;
  const size_t most = std::min(w.rows_bytes / ((size_t)NL * K), w.sums_ints / ((size_t)2 * rows * NL)) / 4 * 4;
  if (K % 256 || most < 4 || w.planes_bytes < (size_t)2 * rows * K) {
    printf("internal error in the routed experts' product: inputs must be a multiple of 256, and the work buffers must "
           "hold one expert's tensor and four tokens\n");
    exit(1);
  }
#if EXPERTS_IN_ONE_LAUNCH
  if ((b.type == wb::Q4_K || b.type == wb::Q5_K) && pk_fits(b)) { // that kernel reads these two types and no other
    pk_grouped_blocks<NL>(m, b, x, ldx, acc, st);
    return;
  }
#endif
  expert_tiles::Limbs<NL> p;
  for (int l = 0; l < NL; l++) p.x[l] = x[l];
  for (int e = 0; e < m.NE; e++) {
    if (!w.count[e]) continue;
    blocks_to_planes(b.bytes + e * b.expert_bytes(), rows, K / 256, w.planes, w.planes + (size_t)rows * K, K, st,
                     b.gain, b.scale, b.type);
    for (int j0 = 0; j0 < w.count[e]; j0 += (int)most) {
      const int count = std::min(w.count[e] - j0, (int)most);
      const int n = (count + 3) / 4 * 4; // rounded up to a multiple of 4; the slots' padding provides the extra rows
      expert_tiles::k_expert_rows<NL>
          <<<grid_blocks((size_t)NL * n * (K / 16)), 256, 0, st>>>(p, ldx, w.slot0[e] + j0, n, K, w.rows);
      blas::gemm_tn(w.blas, 2 * rows, NL * n, K, w.planes, K, w.rows, K, w.sums, (size_t)2 * rows, false);
      expert_tiles::k_join_limbs<NL>
          <<<grid_blocks((size_t)rows * count), 256, 0, st>>>(w.sums, rows, n, count, w.rank0[e] + j0, m.T, acc);
    }
  }
  CK(cudaGetLastError());
}
// Makes, once, every cuBLAS call that grouped_blocks<NL>() can make for a tensor of rows x K and at
// most T tokens, on whatever the work buffers contain. Call it at set-up, with the buffer sizes of
// the later calls (not checked). Which kernel cuBLAS launches depends on the sizes, and a kernel's
// code is loaded at its first use; after this call no request's routing leads to a size not yet
// used. At EXPERTS_IN_ONE_LAUNCH 1 it also sets pk_grouped_blocks()'s kernels' shared-memory limit.
// The cuBLAS calls are queued on w.blas's stream and not waited for.
//
// Public source: kernels are loaded at their first use: NVIDIA, CUDA C++ Programming Guide
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/), "Lazy Loading".
template <int NL> static inline void grouped_blocks_warm_up(int rows, int K, int T, const ExpertsWork &w) {
#if EXPERTS_IN_ONE_LAUNCH
  pk_grouped_blocks_ready<NL>();
#endif
  const size_t most = std::min(w.rows_bytes / ((size_t)NL * K), w.sums_ints / ((size_t)2 * rows * NL)) / 4 * 4;
  for (int n = 4; n <= (int)std::min(most, (size_t)(T + 3) / 4 * 4); n += 4)
    blas::gemm_tn(w.blas, 2 * rows, NL * n, K, w.planes, K, w.rows, K, w.sums, (size_t)2 * rows, false);
}
// Launches k_gate_up on st. Not checked: fe % 128 == 0, m.S % 32 == 0, 4-byte aligned rows of h.
static inline void gate_up(const MoeTables &m, const long long *ag, const long long *au, int fe, int kg, int ku,
                           const short *Ts, int8_t *const h[4], size_t ldH, u32 *tq, u32 *t1, u32 *t2, u32 *err,
                           int proof, cudaStream_t st, int gcl, uint16_t *th) {
  expert_tiles::EpiArgs a{m, fe, kg, ku, gcl, ag, au, Ts, {h[0], h[1], h[2], h[3]}, ldH, tq, t1, t2, th, err};
  const dim3 grid(fe / 128, m.S / 32);
  (proof ? expert_tiles::k_gate_up<1> : expert_tiles::k_gate_up<0>)<<<grid, 256, 0, st>>>(a);
  CK(cudaGetLastError());
}
