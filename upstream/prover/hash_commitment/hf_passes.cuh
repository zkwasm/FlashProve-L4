// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU code for an opening of the weight table (a proof's test on it): the two types of pass over
// the weights, and the masks' shares of their results. rf::HfFold folds the table's rows into one
// row, u. rf::HfColumns computes the sums g that give the table's columns at drawn positions. Both
// compute from the weights, which the caller feeds unit by unit; neither reads an encoded row.
// u, omega and g are exact linear functions of the weights, hidden as W is: the caller must zero
// them. An object's wipe() zeroes its own copies. A host function that finds a precondition broken
// exits with code 4 (hcp::internal_error()); the kernels check none of theirs.
//
// W is the committed matrix: m0 rows and k = 2^lk columns of signed bytes (cells). The weight table
// encodes a matrix M of R = 2^b rows; a row of M is a message, made of cells of W, and a tail of nm
// masks. With lo = lk - b (at full size, which is Llama 4 Maverick's shape, b = 11 and lo = 7):
//     message[r][y 2^lo + cl] = W[y][r 2^lo + cl],  cl < 2^lo;    tail[r][j] = mask[r][j],  j < nm.
// This file counts a message's and a tail's entries each from 0. The caller places the two among a
// row's coefficients (`HfSideRow`, prover/hash_commitment/hf_open_side.cuh), as blocks/code/code.h
// defines. A unit (`UnitW`) is a range of whole rows of W, given as cells[c ld + row] or, for a
// routed unit (a tensor of the routed experts), as the model file's blocks (decode.cuh's types).
// (1) rf::HfFold computes u and omega in the extension field E from the caller's eg[r] and el[cl]:
//         u[y 2^lo + cl] = sum over r < R of eg[r] W[y][r 2^lo + cl]
//         omega[y]       = sum over cl < 2^lo of el[cl] u[y 2^lo + cl]
//     u is the message part of the sum over r of eg[r] M[r]; (3) computes its tail part. If eg[r] =
//     eq(rw_hi, r) and el[cl] = eq(rw_lo, cl), then omega = W eq(rw) for rw = (rw_hi || rw_lo);
//     coordinate 0 belongs to the top bit of the column index. Row y's entries of u are final once
//     the unit with row y is added, and are then copied to the caller's u (9.45 GB at full size).
// (2) rf::HfColumns computes the sums of t fibres, in the base field F:
//         g[a][c] = sum over y < m0 of hi_a(y) W[y][c],    a < t, c < k,
//     where hi_a(idx), hfp::hi_at(), is the product of hv[a][j] over the set bits j of idx. A fibre
//     is an aligned range of 2^lo codeword positions (the caller's `run`), and hv[a][j] is the
//     caller's factor for bit j of y on fibre a. From g[a][r 2^lo + cl], cl < 2^lo, a transform of
//     size 2^lo, hf::short_runs_encode(), gives the caller the message's share of row r's symbols
//     in fibre a, up to message_scale() of blocks/code/code.h: one product serves all R rows.
//     After lay_values() the coefficients are not such products but are given one by one:
//         g[a][c] = sum over y < m0 of values[a][y] W[y][c].
// (3) hf_mask_fold() computes u's tail part, um, and hf_mask_columns() adds the masks' share to g.
//     Neither reads a weight: a table's making key (`key`: a ChaCha20 key derived from the
//     commitment's seed; defined in prover/hash_commitment/hf_seed.h) generates its masks.
// An element of F is a canonical u64 in every buffer that passes between files; an element of E is
// ::Ext. F is the field of p = 2^64 - 2^32 + 1 (prover/field/field_lazy.cuh); no function here
// depends on the prime's form. gf_mul() (prover/common/common.h)
// uses this p.
// HfColumns and hf_mask_columns() write elements of F as eight signed bytes (digits; hcp::limbs()),
// which cuBLAS multiplies into 32-bit sums. A sum takes at most hcp::CUT = 2^16 products, each at
// most 2^14 in magnitude, so it stays within 2^30. k_join_g and k_mask_join join the sums modulo p.
// Include first prover/common/common.h and, inside namespace rf, prover/common/weight_layout.h
// (UnitW, LayW), as prover/weights_commitment/common.cuh does. cudaMalloc and cudaFree may be the
// macros of prover/weights_commitment/lent_memory.h. A host thread must create its cuBLAS handle,
// blas::blas(stream, nullptr, 0), before it feeds an HfColumns: creating one allocates.
// Slow reference versions for the host are in prover/hash_commitment/hf_passes_ref.h;
// a test that is not part of this repository compares this file's results with theirs.
// Public sources.
// cuBLAS: NVIDIA, cuBLAS documentation (https://docs.nvidia.com/cuda/cublas/), "cublasGemmEx()":
// CUDA_R_8I operands, a CUDA_R_32I result, CUBLAS_COMPUTE_32I; requirements: prover/common/blas.h.
// Limbs, their products on tensor cores and the join: prover/hash_commitment/hc_product_ref.h.
// Delayed reduction in a matrix product modulo a prime, once per block of accumulated products:
// J. Berthomieu, S. Graillat, D. Lesnoff, T. Mary, "Multiword matrix multiplication over large
// finite fields in floating-point arithmetic", arXiv:2601.07508, Algorithm 2.3 and section 3.
// There the exact sums are kept in floating-point numbers, here in integers.
// The kernels' loop form, the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with
// Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// Wide loads and stores: "CUDA Pro Tip: Increase Performance with Vectorized Memory Access",
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
#pragma once
#include "common/blas.h"
#include "hash_commitment/hc_product_ref.h" // hcp::limbs, join, add, routed_fits, routed_calls
#include "hash_commitment/hf_mask.cuh"      // hf_mask8, hf_key32, hf::key_for_a_launch
#include "hash_commitment/hf_passes_ref.h"
#include "weights_blocks/decode.cuh" // qblocks::decode_planes, transpose_bytes, grid_for
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace hcp {
static_assert(sizeof(Ext) == 8 * EDEG && EDEG == 3 && sizeof(Ext) == sizeof(blk::E), "an element of E is three words, here and in blk");
static_assert(P == FIELD_P, "p here and in field/field_names.h");
// By cuBLAS: C[j ldc + i] = sum over c < K of A[i lda + c] B[j ldb + c], i < m, j < n, added to C
// if add_to. In every call here m, K, lda, ldb are multiples of 4; A, B, C are 16-byte aligned.
static inline void gemm(cudaStream_t st, void *work, int m, int n, int K, const int8_t *A, size_t lda, const int8_t *B,
                        size_t ldb, int *C, size_t ldc, bool add_to) {
  blas::gemm_tn(blas::blas(st, work), m, n, K, A, lda, B, ldb, C, ldc, add_to);
}
static inline size_t up512(size_t n) {
  return (n + 511) / 512 * 512;
}
[[noreturn]] static inline void internal_error(const char *what) {
  fprintf(stderr, "internal error in the hash commitment's product: %s\n", what);
  exit(4);
}
} // namespace hcp

namespace hfp {
// An Ext as blk's E and back: the same three canonical words in the same order.
__host__ __device__ __forceinline__ blk::E element(const Ext &x) {
  return blk::E{{blk::F{coord(x, 0)}, blk::F{coord(x, 1)}, blk::F{coord(x, 2)}}};
}
__host__ __device__ __forceinline__ Ext ext_of(const blk::E &x) {
  Ext z{};
  set_coord(z, 0, x.c[0].v), set_coord(z, 1, x.c[1].v), set_coord(z, 2, x.c[2].v);
  return z;
}
// A sum of signed bytes w times elements of E is a blk::lazy::IAcc: two signed 64-bit sums per
// coordinate, of w times the coordinate's two signed limbs (prover/field/lazy_signed.cuh). An IAcc
// takes at most 2^29 / 128 = 2^FOLD_TERMS_BITS = 2^22 terms. A sum of the fold has at most R = 2^b
// terms; HfFold::lay() rejects a larger b.
constexpr int FOLD_TERMS_BITS = blk::lazy::IMAC_ROOM_BITS - 7;
static_assert(FOLD_TERMS_BITS == 22 && blk::lazy::MAC_ROOM == (uint64_t)1 << 29, "the numbers 22 and 29 stand in this file's error messages");
__device__ __forceinline__ u64 hi_at(const u64 *__restrict__ hv, int nb, int a, u64 idx) {
  u64 v = 1;
  for (int j = 0; j < nb; j++)
    if ((idx >> j) & 1) v = gf_mul(v, hv[(size_t)a * nb + j]);
  return v;
}
// Row i of a panel, a matrix of cells, is W's row y0 + i ys for i < half, else y1 + (i - half) ys:
// the weights' low, then their high bytes in a rectangle of a routed tensor (hcp::RoutedCall).
struct Rows {
  size_t y0, y1, ys;
  int half;
  __host__ __device__ size_t of(int i) const { return i < half ? y0 + (size_t)i * ys : y1 + (size_t)(i - half) * ys; }
};

// Adds to acc, which holds u from W's row `base` on, the share of W's columns c0 <= c < c0 + nc.
// bytes[i ld + (c - c0)] is the cell in panel row i < n, column c. For each i and each cl < 2^lo,
//     acc[(rows.of(i) - base) 2^lo + cl] += sum of eg[r] bytes[i ld + (c - c0)]
// over the r whose c = r 2^lo + cl is among them; nc >= 1. The loop form is public: the grid-stride
// loop (cited at the top). For 2 <= lo <= 8 HfFold::blocks() runs k_fold_window4 instead.
__global__ void k_fold_window(const int8_t *__restrict__ bytes, size_t ld, int n, Rows rows, size_t base, size_t c0,
                              size_t nc, int lo, const Ext *__restrict__ eg, Ext *__restrict__ acc) {
  const size_t C = (size_t)1 << lo;
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < (size_t)n << lo;
       x += (size_t)gridDim.x * blockDim.x) {
    const size_t cl = x & (C - 1);
    const int i = (int)(x >> lo);
    blk::lazy::IAcc s = blk::lazy::izero();
    for (size_t r = c0 >> lo; r <= (c0 + nc - 1) >> lo; r++) {
      const size_t c = (r << lo) + cl;
      if (c < c0 || c >= c0 + nc) continue;
      blk::lazy::imac(s, blk::lazy::iprep(element(eg[r])), bytes[(size_t)i * ld + (c - c0)]);
    }
    Ext *o = acc + ((rows.of(i) - base) << lo) + cl;
    *o = ext_of(element(*o) + blk::lazy::ired(s));
  }
}
// The same fold for n rows given as cells[c ld + row], with all k = R 2^lo columns: it overwrites
// acc[row 2^lo + cl]. A form with four rows per thread was slower.
__global__ void k_fold_cells(const int8_t *__restrict__ cells, size_t ld, size_t n, int lo, size_t R,
                             const Ext *__restrict__ eg, Ext *__restrict__ acc) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < n << lo; x += (size_t)gridDim.x * blockDim.x) {
    const size_t row = x % n, cl = x / n;
    blk::lazy::IAcc s = blk::lazy::izero();
    for (size_t r = 0; r < R; r++) blk::lazy::imac(s, blk::lazy::iprep(element(eg[r])), cells[((r << lo) + cl) * ld + row]);
    acc[(row << lo) + cl] = ext_of(blk::lazy::ired(s));
  }
}
// k_fold_window4 gives k_fold_window's results. The ideas are public: wide loads of four cells (the
// post on vectorized memory access, cited at the top); the limbs of eg[r], which all threads of a
// block read, are staged in shared memory, SPAN at a time (NVIDIA, CUDA C++ Best Practices Guide,
// "Shared Memory", https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/; see PUBLISHED-WORKS.md).
constexpr int SPAN = 512;
// Requires lo >= 2, c0 and nc multiples of 2^lo, ld % 4 == 0 and `bytes` at a multiple of 4.
__global__ void __launch_bounds__(256) k_fold_window4(const int8_t *__restrict__ bytes, size_t ld, int n, Rows rows,
                                                      size_t base, size_t c0, size_t nc, int lo,
                                                      const Ext *__restrict__ eg, Ext *__restrict__ acc) {
  __shared__ blk::lazy::IPrep h[SPAN]; // the signed limbs of eg[r], 24 bytes each
  const size_t quads = (size_t)1 << (lo - 2), all = (size_t)n << (lo - 2), nr = nc >> lo;
  for (size_t r0 = 0; r0 < nr; r0 += SPAN) {
    const int span = (int)(nr - r0 < SPAN ? nr - r0 : SPAN);
    __syncthreads();
    for (int i = threadIdx.x; i < span; i += blockDim.x) h[i] = blk::lazy::iprep(element(eg[(c0 >> lo) + r0 + i]));
    __syncthreads();
    for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < all; x += (size_t)gridDim.x * blockDim.x) {
      const size_t cl = (x & (quads - 1)) * 4;
      const int i = (int)(x >> (lo - 2));
      const int8_t *from = bytes + (size_t)i * ld + (r0 << lo) + cl;
      blk::lazy::IAcc s[4] = {blk::lazy::izero(), blk::lazy::izero(), blk::lazy::izero(), blk::lazy::izero()};
      for (int r = 0; r < span; r++) { // at most SPAN = 2^9 terms a sum
        const uint32_t word = *(const uint32_t *)(from + ((size_t)r << lo));
        const blk::lazy::IPrep g = h[r];
#pragma unroll
        for (int j = 0; j < 4; j++) blk::lazy::imac(s[j], g, (int)(int8_t)(word >> (8 * j)));
      }
      Ext *o = acc + ((rows.of(i) - base) << lo) + cl;
#pragma unroll
      for (int j = 0; j < 4; j++) o[j] = ext_of(element(o[j]) + blk::lazy::ired(s[j]));
    }
  }
}
// omega[row] = sum over cl of el[cl] acc[row 2^lo + cl].
__global__ void k_omega(const Ext *__restrict__ acc, size_t n, int lo, const Ext *__restrict__ el,
                        Ext *__restrict__ omega) {
  for (size_t row = blockIdx.x * (size_t)blockDim.x + threadIdx.x; row < n; row += (size_t)gridDim.x * blockDim.x) {
    blk::lazy::AccE s = blk::lazy::zero_e(); // 2^lo products: `HfFold::lay` rejects 2^lo > MAC_ROOM
    for (size_t cl = 0; cl < (size_t)1 << lo; cl++) blk::lazy::mac(s, element(el[cl]), element(acc[(row << lo) + cl]));
    omega[row] = ext_of(blk::lazy::reduce(s));
  }
}
// D[(8 a + d) K + i] = digit d (hcp::limbs()) of hi_a(rows.of(i)) for i < n, and 0 for n <= i < K.
__global__ void k_hi_digits(const u64 *__restrict__ hv, int nb, int t, Rows rows, int n, int K,
                            int8_t *__restrict__ D) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < (size_t)t * K;
       x += (size_t)gridDim.x * blockDim.x) {
    const int i = (int)(x % K), a = (int)(x / K);
    int8_t d[hcp::LIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (i < n) hcp::limbs(hi_at(hv, nb, a, rows.of(i)), d);
    for (int b = 0; b < hcp::LIMBS; b++) D[((size_t)a * hcp::LIMBS + b) * K + i] = d[b];
  }
}
// The same for coefficients given one by one: D[(8 a + d) K + i] = digit d of
// values[a m0 + rows.of(i)] for i < n, and 0 for n <= i < K. values: [t][m0], canonical.
__global__ void k_value_digits(const u64 *__restrict__ values, size_t m0, int t, Rows rows, int n, int K,
                               int8_t *__restrict__ D) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < (size_t)t * K;
       x += (size_t)gridDim.x * blockDim.x) {
    const int i = (int)(x % K), a = (int)(x / K);
    int8_t d[hcp::LIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (i < n) hcp::limbs(values[(size_t)a * m0 + rows.of(i)], d);
    for (int b = 0; b < hcp::LIMBS; b++) D[((size_t)a * hcp::LIMBS + b) * K + i] = d[b];
  }
}
// g[a k + c0 + j] += hcp::join() of the eight sums at S[(j t + a) 8], for a < t, j < nc.
// Neighbouring threads take neighbouring a, so that S, the larger array, is read in order.
__global__ void k_join_g(const int *__restrict__ S, size_t nc, int t, u64 *__restrict__ g, size_t k, size_t c0) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < nc * t; x += (size_t)gridDim.x * blockDim.x) {
    const size_t a = x % t, j = x / t;
    const int4 *s = (const int4 *)(S + x * hcp::LIMBS);
    const int4 p = s[0], q = s[1];
    u64 *o = g + a * k + c0 + j;
    *o = hcp::add(*o, hcp::join(p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w));
  }
}
// The eight balanced digits of a canonical v (hcp::limbs()) in one word; above 0x7F7F7F7F7F7F7F7F,
// those of v - p. signed_digits() adds 128 to each byte, with carries, and subtracts it by an xor.
// A related form is public: H. S. Warren, "Hacker's Delight", 2nd ed., section 2-6, "Sign
// Extension": ((x + 0x00000080) & 0x000000FF) - 0x00000080, by the same adding and removing of 128.
// There the byte's carry is masked off; here it enters the next byte. The form ((x + 0x80) ^ 0x80)
// is not in that section. The idea is public: making all signed digits at once by one addition of
// half the base at each digit position is the TFHE library's gadget decomposition for base 2^k
// (src/libtfhe/tgsw-functions.cpp, tgsw.cpp; Apache 2.0; there for the top l digits, modulo 2^32),
// and Knuth, The Art of Computer Programming, vol. 2, section 4.1, for balanced ternary. This is
// its base-256 case; the subtraction is one xor (Wikipedia, "Offset binary", for a single byte).
__host__ __device__ static inline u64 balanced(u64 v) {
  return blk::lazy::signed_digits(blk::F{v});
}
// o[l] = byte l of x0, x1, x2, x3: a 4 x 4 transpose of bytes by the byte-permute instruction
// (NVIDIA, CUDA Math API, __byte_perm).
// The same eight __byte_perm calls, with the same operands and selectors, are in MegEngine,
// dnn/src/cuda/memory_utils.cuh, transpose_int8_4x4_impl (Copyright (c) 2014-2021 Megvii Inc.; Apache License 2.0; see
// NOTICE); there the second and the third call stand in the other order.
__device__ __forceinline__ void turn4(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t (&o)[4]) {
  const uint32_t t0 = __byte_perm(x0, x1, 0x5140), t1 = __byte_perm(x2, x3, 0x5140);
  const uint32_t t2 = __byte_perm(x0, x1, 0x7362), t3 = __byte_perm(x2, x3, 0x7362);
  o[0] = __byte_perm(t0, t1, 0x5410), o[1] = __byte_perm(t0, t1, 0x7632);
  o[2] = __byte_perm(t2, t3, 0x5410), o[3] = __byte_perm(t2, t3, 0x7632);
}
// A[((rr 2^lo + cl) 8 + d) K + q] = digit d of m[rr nj + q 2^lo + cl] if q < nq, 0 if nq <= q < K.
// Requires K % 16 == 0 and A at a multiple of 16 bytes. A thread takes 16 neighbouring q of one
// (rr, cl); neighbouring threads take neighbouring cl, so that the masks are read in order.
__global__ void k_mask_limbs(const u64 *__restrict__ m, uint32_t nr, size_t nj, int lo, size_t nq, size_t K,
                             int8_t *__restrict__ A) {
  const size_t C = (size_t)1 << lo, groups = K / 16;
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < ((size_t)nr * groups) << lo;
       x += (size_t)gridDim.x * blockDim.x) {
    const size_t cl = x & (C - 1), qg = (x >> lo) % groups, rr = (x >> lo) / groups;
    uint32_t low[4][4], high[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      u64 v[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const size_t q = qg * 16 + j * 4 + i;
        v[i] = q < nq ? balanced(m[rr * nj + (q << lo) + cl]) : 0;
      }
      turn4((uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3], low[j]);
      turn4((uint32_t)(v[0] >> 32), (uint32_t)(v[1] >> 32), (uint32_t)(v[2] >> 32), (uint32_t)(v[3] >> 32), high[j]);
    }
    int8_t *o = A + ((rr << lo) + cl) * 8 * K + qg * 16;
#pragma unroll
    for (int d = 0; d < 4; d++) {
      *(uint4 *)(o + (size_t)d * K) = make_uint4(low[0][d], low[1][d], low[2][d], low[3][d]);
      *(uint4 *)(o + (size_t)(d + 4) * K) = make_uint4(high[0][d], high[1][d], high[2][d], high[3][d]);
    }
  }
}
// Joins modulo p the 64 sums for each entry of g (d: the mask's digit; e: hi_a's digit):
//     g[a k + r0 2^lo + rc] += sum over d, e < 8 of 256^(d + e) S[((8 rc + d) t + a) 8 + e],
// rc = rr 2^lo + cl. Each S[..] is within 2^30, so T[s], their sum over d + e = s, is within 2^33.
// Four neighbours are added as integers, T[s] + 2^8 T[s + 1] + 2^16 T[s + 2] + 2^24 T[s + 3], below
// 2^33 (1 + 2^8 + 2^16 + 2^24) < 2^58, within join_digit()'s bound of 2^JOIN_BITS (JOIN_BITS = 61);
// the four groups' weights are 2^0, 2^32, 2^64 and 2^96 (join_digit() takes 2^s for s < 192).
// Neighbouring threads take neighbouring a.
__global__ void k_mask_join(const int *__restrict__ S, uint32_t nr, int lo, int t, uint32_t r0, u64 *__restrict__ g,
                            size_t k) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < ((size_t)nr * t) << lo;
       x += (size_t)gridDim.x * blockDim.x) {
    const size_t a = x % t, rc = x / t; // rc = rr 2^lo + cl
    long long T[16] = {}; // T[15] stays 0
#pragma unroll
    for (int d = 0; d < 8; d++) {
      const int4 *p = (const int4 *)(S + ((rc * 8 + d) * t + a) * 8);
      const int4 u = p[0], v = p[1];
      T[d] += u.x, T[d + 1] += u.y, T[d + 2] += u.z, T[d + 3] += u.w;
      T[d + 4] += v.x, T[d + 5] += v.y, T[d + 6] += v.z, T[d + 7] += v.w;
    }
    u64 *o = g + a * k + ((size_t)r0 << lo) + rc;
    blk::F sum{*o};
#pragma unroll
    for (int s = 0; s < 16; s += 4)
      sum += blk::lazy::join_digit(T[s] + T[s + 1] * 256 + T[s + 2] * 65536 + T[s + 3] * 16777216, 8 * s);
    *o = sum.v;
  }
}
// The masks' fold with the mask generator (hf_mask8(), hf_mask.cuh) inside: the kernel stores no mask
// in any buffer. Requires blocks of 128 threads. Thread x < ng takes the generator's block g0 + x:
//     out[8 x + sigma] = sum over r < R of eg[r] mask[r][8 (g0 + x) + sigma],  sigma < 8.
// Each of the 24 sums is one word (blk::lazy::Loose), reduced at each term by mad(): any number of
// terms. Seen at the prime 2^61 - 1, not at this p: eight sums with a delayed reduction (AccE, 96
// bytes each) do not fit, since the kernel then needs all 255 registers and 832 bytes of stack; the
// reductions cost time, 18.8 to 19.0 ms for 2^31 masks on one H100 against 16.5 to 16.7 ms for a
// form for that prime alone, which adds 32 products, each below 2^122, in 128 bits before reducing.
__global__ void __launch_bounds__(128)
    k_mask_fold_made(const __grid_constant__ hf_key32 key, u64 limit, uint32_t R, uint32_t g0, size_t ng,
                     const Ext *__restrict__ eg, Ext *__restrict__ out) {
  const size_t x = blockIdx.x * (size_t)128 + threadIdx.x;
  if (x >= ng) return;
  blk::lazy::Loose s[8][EDEG];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int e = 0; e < EDEG; e++) s[i][e] = blk::lazy::loosen(blk::F{0});
  for (uint32_t r = 0; r < R; r++) {
    u64 m[8];
    hf_mask8(key, r, g0 + (uint32_t)x, limit, m);
    const blk::E g = element(eg[r]);
#pragma unroll
    for (int e = 0; e < EDEG; e++)
#pragma unroll
      for (int i = 0; i < 8; i++) s[i][e] = blk::lazy::mad(blk::lazy::loosen(g.c[e]), blk::lazy::loosen(blk::F{m[i]}), s[i][e]);
  }
#pragma unroll
  for (int i = 0; i < 8; i++)
    out[8 * x + i] = ext_of(blk::E{{blk::lazy::canonical(s[i][0]), blk::lazy::canonical(s[i][1]), blk::lazy::canonical(s[i][2])}});
}
// k_mask_limbs with the mask generator inside, for lo >= 3:
//     A[((rr 2^lo + cl) 8 + d) K + q] = digit d of mask[r0 + rr][(q0 + q) 2^lo + cl]
// for q < nq and a mask index below nm, and 0 for every other q < K. Requires K % 32 == 0, A at a
// multiple of 16 bytes, blocks of 256 threads and the grid (K / 32, ceil(2^lo / 128), nr). A thread
// block transposes its balanced() words through shared memory. Requires nm <= 2^35 (a generator
// block's number has 32 bits); not checked. The idea is public: a transpose through a tile in
// shared memory (Harris's post on the matrix transpose, cited in prover/weights_blocks/decode.cuh).
__global__ void __launch_bounds__(256)
    k_mask_limbs_made(const __grid_constant__ hf_key32 key, u64 limit, uint32_t r0, u64 q0, size_t nq, u64 nm, int lo,
                      size_t K, int8_t *__restrict__ A) {
  __shared__ __align__(16) u64 tile[32][128 + 1]; // the pad shifts each row by two banks
  const int CC = lo >= 7 ? 128 : 1 << lo;
  const size_t qt = (size_t)blockIdx.x * 32, cl0 = (size_t)blockIdx.y * 128;
  const uint32_t rr = blockIdx.z;
  for (int x = threadIdx.x; x < 32 * (CC / 8); x += 256) {
    const int q = x / (CC / 8), c8 = x % (CC / 8);
    const u64 j = ((q0 + qt + q) << lo) + cl0 + 8 * c8;
    u64 m[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (qt + q < nq && j < nm) hf_mask8(key, r0 + rr, (uint32_t)(j / 8), limit, m);
#pragma unroll
    for (int i = 0; i < 8; i++) tile[q][8 * c8 + i] = j + i < nm ? balanced(m[i]) : 0;
  }
  __syncthreads();
  for (int x = threadIdx.x; x < CC * 8; x += 256) {
    const int cl = x / 8, d = x % 8;
    uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t v = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) v |= (uint32_t)(uint8_t)(tile[4 * i + j][cl] >> (8 * d)) << (8 * j);
      w[i] = v;
    }
    uint4 *o = (uint4 *)(A + ((((size_t)rr << lo) + cl0 + cl) * 8 + d) * K + qt);
    o[0] = make_uint4(w[0], w[1], w[2], w[3]);
    o[1] = make_uint4(w[4], w[5], w[6], w[7]);
  }
}
static inline unsigned grid(size_t threads) {
  return qblocks::grid_for((threads + 255) / 256);
}
} // namespace hfp

namespace rf {
// A caller feeds W by cells() or blocks(), in whole rows, then calls finish(). A row fed twice ends
// the process in HfFold::finish(); HfColumns does not check. Results build up only in buffers given
// to the derived struct's lay(); the scratch buffers (`rooms`) carry nothing between calls and may
// be released and taken again then. A caller with several streams must order the calls' work.
struct HfPass {
  virtual ~HfPass() {}
  virtual void take_rooms() = 0;
  virtual void give_rooms() = 0; // the caller must first call `wipe` and wait for its streams
  // Rows first_row .. first_row + rows - 1 of W as cells[c ld + row], c < k. Requires ld % 16 == 0,
  // `cells` 16-byte aligned, rows <= ld. Row indices up to the next multiple of 16 may be read.
  virtual void cells(const int8_t *cells, size_t ld, size_t rows, size_t first_row, cudaStream_t st) = 0;
  // A routed unit from its quantized blocks. `blocks` is the tensor as in the model's file,
  // [expert][output index][super-block of 256 inputs], at a multiple of 16 bytes. `type` is the
  // blocks' type as in prover/weights_blocks/decode.cuh (0 Q4_K, 1 Q5_K, 4 Q8_0, 5 F16, 6 BF16); any
  // other number ends the program there. `gains` is null or one multiplier per input column (gamma_j
  // in prover/weights_blocks/fold_rule.h). two_e = 2^e for the tensor's exponent e. Requires fits(u).
  virtual void blocks(const UnitW &u, const void *blocks, int type, const float *gains, float two_e,
                      cudaStream_t st) = 0;
  // After the last unit, with the rooms still taken: completes the pass's results.
  virtual void finish(cudaStream_t st) = 0;
  virtual void wipe(cudaStream_t st) = 0; // zeroes the rooms that depend on W, if they are taken
  virtual void end() = 0;                 // `give_rooms` (same duties), then resets the object
  static bool fits(const UnitW &u) { return hcp::routed_fits(u); }
  // PLANES: bytes for decoded cells. PANEL: the most tensor rows in a panel (2 PANEL rows of W).
  static constexpr size_t PLANES = (size_t)2 * 20 * 256 * 8192; // 2 x 20 blocks of 256 x 8,192 rows
  static constexpr int PANEL = 8192;
};

// (1) The folded row u, and omega. The set-up, lay(), requires 1 <= b <= lk, b <= 22, lk - b <= 29.
struct HfFold : HfPass {
  int b = 0, lo = 0;
  size_t k = 0, m0 = 0, R = 0, acc_rows = 0;
  const Ext *eg = nullptr, *el = nullptr;
  Ext *u = nullptr, *omega = nullptr, *acc = nullptr; // acc = slot[turn], the accumulator in use
  Ext *slot[2] = {nullptr, nullptr};
  cudaStream_t copier = nullptr;
  cudaEvent_t filled[2] = {nullptr, nullptr}, sent[2] = {nullptr, nullptr};
  int turn = 0;
  int8_t *planes = nullptr;
  std::vector<std::pair<size_t, size_t>> done; // runs of rows added so far: (first, number)

  // An accumulator's size in rows of W. blocks() needs space for a whole routed unit; cells() works
  // in steps of this many rows.
  static size_t rows_at_a_time(const LayW &Y) {
    size_t n = 2 * (size_t)PANEL;
    for (const UnitW &x : Y.U)
      if (fits(x)) n = std::max(n, x.rows);
    return std::min(n, std::max<size_t>(Y.m0, 16));
  }
  // Bytes that take_rooms() allocates, buffers rounded up to 512. with_copier: lay() gets a copier.
  static size_t room_bytes(const LayW &Y, int b, bool with_copier = false) {
    return hcp::up512(PLANES) + (with_copier ? 2 : 1) * hcp::up512((rows_at_a_time(Y) << (Y.lk - b)) * sizeof(Ext));
  }
  // eg: GPU, 2^b elements. el: GPU, 2^(lk - b) elements. omega: GPU, m0 elements.
  // u: m0 2^(lk - b) elements of E, GPU or pinned host memory. eg
  // and el are read until the last unit's work has run. Once finish()'s work has run, every entry
  // of u and omega is written: the rows that are in no unit get 0. lay() allocates nothing.
  // copier: null, or a stream of the caller's that nothing else uses until end(). u is then copied
  // out on it, from two accumulators (`slot`) in turn, while the next unit is processed. finish()
  // and wipe() make their stream wait for these copies; give_rooms() blocks until they end.
  void lay(const LayW &Y, int b_, const Ext *eg_, const Ext *el_, Ext *u_, Ext *omega_, cudaStream_t copier_ = nullptr) {
    if (b_ < 1 || b_ > Y.lk || b_ > hfp::FOLD_TERMS_BITS || ((uint64_t)1 << (Y.lk - b_)) > blk::lazy::MAC_ROOM || u || !eg_ || !u_ ||
        !el_ || !omega_)
      hcp::internal_error("the fold's lay() needs 1 <= b <= lk, b <= 22, lk - b <= 29, an object that is not in use, eg, u, and el "
                          "with omega");
    b = b_, lo = Y.lk - b_, k = Y.k, m0 = Y.m0, R = (size_t)1 << b_, acc_rows = rows_at_a_time(Y);
    eg = eg_, el = el_, u = u_, omega = omega_, copier = copier_;
    done.clear();
  }
  void take_rooms() override {
    if (!u || planes) hcp::internal_error("take_rooms() comes after lay(), and not twice without give_rooms()");
    CK(cudaMalloc(&planes, PLANES));
    for (int i = 0; i < (copier ? 2 : 1); i++) {
      CK(cudaMalloc(&slot[i], (acc_rows << lo) * sizeof(Ext)));
      if (!copier) break;
      CK(cudaEventCreateWithFlags(&filled[i], cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&sent[i], cudaEventDisableTiming));
    }
    turn = 0, acc = slot[0];
  }
  void give_rooms() override {
    if (copier && planes) CK(cudaStreamSynchronize(copier));
    cudaFree(planes);
    for (int i = 0; i < 2; i++) {
      cudaFree(slot[i]);
      if (filled[i]) cudaEventDestroy(filled[i]), cudaEventDestroy(sent[i]);
      slot[i] = nullptr, filled[i] = sent[i] = nullptr;
    }
    planes = nullptr, acc = nullptr;
  }
  void begin(const LayW &Y, int b_, const Ext *eg_, const Ext *el_, Ext *u_, Ext *omega_,
             cudaStream_t copier_ = nullptr) {
    lay(Y, b_, eg_, el_, u_, omega_, copier_);
    take_rooms();
  }
  // Makes st wait for the last copy out of acc, which st will overwrite.
  void await_slot(cudaStream_t st) {
    if (copier) CK(cudaStreamWaitEvent(st, sent[turn], 0));
  }
  // Copies acc's first n rows to rows y .. of u; with a copier, switches to the other accumulator.
  void send(size_t y, size_t n, cudaStream_t st) {
    if (copier) {
      CK(cudaEventRecord(filled[turn], st));
      CK(cudaStreamWaitEvent(copier, filled[turn], 0));
    }
    CK(cudaMemcpyAsync(u + (y << lo), acc, (n << lo) * sizeof(Ext), cudaMemcpyDefault, copier ? copier : st));
    if (copier) {
      CK(cudaEventRecord(sent[turn], copier));
      turn ^= 1, acc = slot[turn];
    }
  }
  void await_all(cudaStream_t st) {
    if (copier)
      for (int i = 0; i < 2; i++) CK(cudaStreamWaitEvent(st, sent[i], 0));
  }
  // Called when acc's first n rows (rows y .. of W) are complete: omega, then send().
  void out(size_t y, size_t n, cudaStream_t st) {
    hfp::k_omega<<<hfp::grid(n), 256, 0, st>>>(acc, n, lo, el, omega + y);
    CK(cudaGetLastError());
    send(y, n, st);
    done.push_back({y, n});
  }
  void cells(const int8_t *cells, size_t ld, size_t rows, size_t first_row, cudaStream_t st) override {
    if (!planes) hcp::internal_error("cells() needs the rooms");
    if (ld % 16 || (uintptr_t)cells % 16 || rows > ld || first_row + rows > m0)
      hcp::internal_error("cells() needs ld % 16 == 0, cells at a multiple of 16 bytes, rows <= ld and rows within W");
    for (size_t r0 = 0; r0 < rows; r0 += acc_rows) {
      const size_t n = std::min(acc_rows, rows - r0);
      await_slot(st);
      hfp::k_fold_cells<<<hfp::grid(n << lo), 256, 0, st>>>(cells + r0, ld, n, lo, R, eg, acc);
      CK(cudaGetLastError());
      out(first_row + r0, n, st);
    }
  }
  void blocks(const UnitW &x, const void *blocks, int type, const float *gains, float two_e, cudaStream_t st) override {
    if (!planes || !fits(x) || (uintptr_t)blocks % 16 || x.off + x.rows > m0 || x.rows > acc_rows)
      hcp::internal_error("blocks() needs the rooms, a unit that fits and blocks at a multiple of 16 bytes");
    await_slot(st);
    CK(cudaMemsetAsync(acc, 0, (x.rows << lo) * sizeof(Ext), st));
    hcp::routed_calls(x, [&](const hcp::RoutedCall &c) {
      const uint8_t *from = (const uint8_t *)blocks + c.expert0 * c.outputs * c.per_row * qblocks::block_bytes(type);
      const int nS = c.experts * c.in_window, rows_most = (int)std::min<size_t>(PANEL, PLANES / 512);
      for (int i0 = 0; i0 < c.L; i0 += rows_most) {
        const int L = std::min(rows_most, c.L - i0), most = (int)std::min<size_t>(nS, PLANES / ((size_t)512 * L));
        for (int s0 = 0; s0 < nS; s0 += most) {
          const int n = std::min(most, nS - s0);
          const size_t ld = (size_t)n * 256;
          qblocks::decode_planes(qblocks::Where{from, c.per_row, c.in_window, c.sb0, s0, c.x0 + i0, (size_t)c.outputs},
                                 type, L, n, gains, two_e, planes, planes + (size_t)L * ld, ld, st);
          const hfp::Rows rows{c.y_lo + (size_t)i0 * c.ys, c.y_hi + (size_t)i0 * c.ys, (size_t)c.ys, L};
          const size_t c0 = c.c0 + (size_t)s0 * 256;
          if (lo >= 2 && lo <= 8) // a window starts and ends at a multiple of 256, so of 2^lo
            hfp::k_fold_window4<<<hfp::grid((size_t)(2 * L) << (lo - 2)), 256, 0, st>>>(planes, ld, 2 * L, rows, x.off, c0,
                                                                                       ld, lo, eg, acc);
          else
            hfp::k_fold_window<<<hfp::grid((size_t)(2 * L) << lo), 256, 0, st>>>(planes, ld, 2 * L, rows, x.off, c0, ld,
                                                                                 lo, eg, acc);
          CK(cudaGetLastError());
        }
      }
    });
    out(x.off, x.rows, st);
  }
  void finish(cudaStream_t st) override {
    if (!planes) hcp::internal_error("finish() needs the rooms");
    std::sort(done.begin(), done.end());
    await_all(st);
    CK(cudaMemsetAsync(acc, 0, (acc_rows << lo) * sizeof(Ext), st));
    auto zeros = [&](size_t y, size_t end) {
      for (; y < end; y += acc_rows) {
        const size_t n = std::min(acc_rows, end - y);
        CK(cudaMemcpyAsync(u + (y << lo), acc, (n << lo) * sizeof(Ext), cudaMemcpyDefault, st));
        CK(cudaMemsetAsync(omega + y, 0, n * sizeof(Ext), st));
      }
    };
    size_t y = 0;
    for (const auto &run : done) {
      if (run.first < y) hcp::internal_error("a row of W has been added twice to the fold");
      zeros(y, run.first);
      y = run.first + run.second;
    }
    zeros(y, m0);
    done.clear();
  }
  void wipe(cudaStream_t st) override {
    if (!planes) return;
    await_all(st);
    CK(cudaMemsetAsync(planes, 0, PLANES, st));
    for (Ext *a : slot)
      if (a) CK(cudaMemsetAsync(a, 0, (acc_rows << lo) * sizeof(Ext), st));
  }
  void end() override {
    give_rooms();
    b = lo = 0, k = m0 = R = acc_rows = 0, eg = el = nullptr, u = omega = nullptr, copier = nullptr;
    done.clear();
  }
  // For a program's start-up: launches this struct's kernels once, on empty inputs. The driver
  // loads a kernel's code at its first launch; without this call it allocates the memory for that
  // inside a batch (the requests proved together), when little is free. Not covered: the block
  // decoder, qblocks::decode_planes(), which needs a tensor.
  static void warm_up() {
    const cudaStream_t st = cudaStreamPerThread;
    hfp::k_fold_window<<<1, 256, 0, st>>>(nullptr, 0, 0, hfp::Rows{0, 0, 1, 0}, 0, 0, 1, 0, nullptr, nullptr);
    hfp::k_fold_window4<<<1, 256, 0, st>>>(nullptr, 0, 0, hfp::Rows{0, 0, 1, 0}, 0, 0, 0, 2, nullptr, nullptr);
    hfp::k_fold_cells<<<1, 256, 0, st>>>(nullptr, 0, 0, 0, 0, nullptr, nullptr);
    hfp::k_omega<<<1, 256, 0, st>>>(nullptr, 0, 0, nullptr, nullptr);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(st));
  }
};

// (2) The fibres' sums. lay() requires 2^nb >= m0: hv[a] has nb factors, one for each bit of y.
struct HfColumns : HfPass {
  // The maximum number of columns of W, and of summed rows of W, in one product.
  static constexpr size_t WINDOW = (size_t)1 << 15;
  static constexpr int TERMS = 2 * PANEL;
  static_assert(TERMS <= hcp::CUT && TERMS % 64 == 0, "a product's 32-bit sums; the transpose's pitch");
  int t = 0, width = 0, nb = 0;
  size_t k = 0, m0 = 0;
  const u64 *hv = nullptr;
  const u64 *values = nullptr; // set only by `lay_values`: coefficients one by one; hv is then null
  u64 *g = nullptr;
  int8_t *planes = nullptr, *turned = nullptr, *digits = nullptr;
  int *sums = nullptr;
  void *work = nullptr;

  // The widest product's columns. room_bytes(): as HfFold's, the bytes that take_rooms() allocates.
  static size_t sum_columns() { return WINDOW; }
  static size_t room_bytes(int t) {
    return 2 * hcp::up512(PLANES) + hcp::up512((size_t)t * hcp::LIMBS * TERMS) +
           hcp::up512(sum_columns() * t * hcp::LIMBS * 4) + hcp::up512(blas::BLAS_WORKSPACE);
  }
  // hv: GPU, [t][nb], canonical, read until the last unit's work has run. g: GPU, [t][k], canonical
  // after every call. Allocates nothing. Zeroes g on cudaStreamPerThread; other streams must wait.
  void lay(const LayW &Y, int t_, const u64 *hv_, int nb_, u64 *g_) {
    if (t_ < 1 || g || !hv_ || !g_ || nb_ < 1 || nb_ > 63 || ((Y.m0 - 1) >> nb_)) {
      char what[320]; // which condition failed: public numbers, and whether a pointer is null
      snprintf(what, sizeof what,
               "the columns' lay() needs t >= 1, an object that is not in use, hv, g and 2^nb >= m0; it was given t = %d, an object that is %s, hv %s, "
               "g %s, nb = %d, m0 = %zu",
               t_, g ? "IN USE" : "free", hv_ ? "set" : "NULL", g_ ? "set" : "NULL", nb_, (size_t)Y.m0);
      hcp::internal_error(what);
    }
    t = t_, width = t_ * hcp::LIMBS, nb = nb_, k = Y.k, m0 = Y.m0, hv = hv_, g = g_;
    CK(cudaMemsetAsync(g, 0, (size_t)t * k * 8, cudaStreamPerThread));
  }
  // As lay(), for coefficients given one by one: g[a] += values_[a m0 + y] times row y of W.
  // values_: GPU, [t][m0], canonical, read until the last unit's work has run. g: as in lay().
  void lay_values(const LayW &Y, int t_, const u64 *values_, u64 *g_) {
    if (t_ < 1 || g || !values_ || !g_) hcp::internal_error("the columns' lay_values() needs t >= 1, an object that is not in use, the values and g");
    t = t_, width = t_ * hcp::LIMBS, nb = 0, k = Y.k, m0 = Y.m0, hv = nullptr, values = values_, g = g_;
    CK(cudaMemsetAsync(g, 0, (size_t)t * k * 8, cudaStreamPerThread));
  }
  void take_rooms() override {
    if (!g || planes) hcp::internal_error("take_rooms() comes after lay(), and not twice without give_rooms()");
    CK(cudaMalloc(&planes, PLANES));
    CK(cudaMalloc(&turned, PLANES));
    CK(cudaMalloc(&digits, (size_t)width * TERMS));
    CK(cudaMalloc(&sums, sum_columns() * width * 4));
    CK(cudaMalloc(&work, blas::BLAS_WORKSPACE));
  }
  void give_rooms() override {
    for (void *p : {(void *)planes, (void *)turned, (void *)digits, (void *)sums, work}) cudaFree(p);
    planes = turned = digits = nullptr, sums = nullptr, work = nullptr;
  }
  void begin(const LayW &Y, int t_, const u64 *hv_, int nb_, u64 *g_) {
    lay(Y, t_, hv_, nb_, g_);
    take_rooms();
  }
  void begin_values(const LayW &Y, int t_, const u64 *values_, u64 *g_) {
    lay_values(Y, t_, values_, g_);
    take_rooms();
  }
  // g[a][c0 + j] += sum over i < K of (the element at (a, i) of `digits`) bytes[j ld + i], j < nc.
  void product(const int8_t *bytes, size_t ld, int K, size_t c0, size_t nc, cudaStream_t st) {
    hcp::gemm(st, work, width, (int)nc, K, digits, K, bytes, ld, sums, width, false);
    hfp::k_join_g<<<hfp::grid(nc * t), 256, 0, st>>>(sums, nc, t, g, k, c0);
    CK(cudaGetLastError());
  }
  void lay_digits(const hfp::Rows &rows, int n, int K, cudaStream_t st) {
    if (values)
      hfp::k_value_digits<<<hfp::grid((size_t)t * K), 256, 0, st>>>(values, m0, t, rows, n, K, digits);
    else
      hfp::k_hi_digits<<<hfp::grid((size_t)t * K), 256, 0, st>>>(hv, nb, t, rows, n, K, digits);
    CK(cudaGetLastError());
  }
  void cells(const int8_t *cells, size_t ld, size_t rows, size_t first_row, cudaStream_t st) override {
    if (!planes) hcp::internal_error("cells() needs the rooms");
    if (ld % 16 || (uintptr_t)cells % 16 || rows > ld || first_row + rows > m0)
      hcp::internal_error("cells() needs ld % 16 == 0, cells at a multiple of 16 bytes, rows <= ld and rows within W");
    for (size_t r0 = 0; r0 < rows; r0 += TERMS) {
      const int n = (int)std::min<size_t>(TERMS, rows - r0), K = (n + 15) / 16 * 16;
      lay_digits(hfp::Rows{first_row + r0, 0, 1, n}, n, K, st);
      for (size_t c0 = 0; c0 < k; c0 += WINDOW)
        product(cells + c0 * ld + r0, ld, K, c0, std::min(WINDOW, k - c0), st);
    }
  }
  void blocks(const UnitW &x, const void *blocks, int type, const float *gains, float two_e, cudaStream_t st) override {
    if (!planes || !fits(x) || (uintptr_t)blocks % 16 || x.off + x.rows > m0)
      hcp::internal_error("blocks() needs the rooms, a unit that fits and blocks at a multiple of 16 bytes");
    hcp::routed_calls(x, [&](const hcp::RoutedCall &c) {
      const uint8_t *from = (const uint8_t *)blocks + c.expert0 * c.outputs * c.per_row * qblocks::block_bytes(type);
      const int nS = c.experts * c.in_window, rows_most = (int)std::min<size_t>(PANEL, PLANES / 512);
      for (int i0 = 0; i0 < c.L; i0 += rows_most) {
        const int L = std::min(rows_most, c.L - i0), K = (2 * L + 63) / 64 * 64;
        const int most = (int)std::min<size_t>(std::min<size_t>(nS, WINDOW / 256), PLANES / ((size_t)256 * K));
        lay_digits(hfp::Rows{c.y_lo + (size_t)i0 * c.ys, c.y_hi + (size_t)i0 * c.ys, (size_t)c.ys, L}, 2 * L, K, st);
        for (int s0 = 0; s0 < nS; s0 += most) {
          const int n = std::min(most, nS - s0);
          const size_t ld = (size_t)n * 256;
          qblocks::decode_planes(qblocks::Where{from, c.per_row, c.in_window, c.sb0, s0, c.x0 + i0, (size_t)c.outputs},
                                 type, L, n, gains, two_e, planes, planes + (size_t)L * ld, ld, st);
          // turned[j K + i] = planes[i ld + j] for i < 2 L, else 0: the summed index i is fastest.
          qblocks::transpose_bytes(planes, ld, 2 * L, K, (int)ld, turned, K, st);
          product(turned, K, K, c.c0 + (size_t)s0 * 256, ld, st);
        }
      }
    });
  }
  void finish(cudaStream_t) override {}
  void wipe(cudaStream_t st) override {
    if (!planes) return;
    CK(cudaMemsetAsync(planes, 0, PLANES, st));
    CK(cudaMemsetAsync(turned, 0, PLANES, st));
    CK(cudaMemsetAsync(sums, 0, sum_columns() * width * 4, st));
    CK(cudaMemsetAsync(work, 0, blas::BLAS_WORKSPACE, st));
  }
  void end() override {
    give_rooms();
    t = width = nb = 0, k = m0 = 0, hv = values = nullptr, g = nullptr;
  }
  // As HfFold::warm_up(); again the block decoder is not covered. It also runs, on zeros, one
  // product of each size that a pass with t fibres computes if every unit arrives in one call, by
  // blocks() if it fits(): cuBLAS's kernels too are loaded at first use, and it picks them by size
  // (prover/weights_blocks/experts.cuh, grouped_blocks_warm_up(), has the public source). If the
  // sizes below drift from the rules of cells() and blocks(), a pass loads code after all, but no
  // value changes. It allocates 0.9 GB at full size with, say, t = 48 (computed, not measured)
  // and frees it before it returns. A thread without a cuBLAS handle gets one here, which stays.
  static void warm_up(const LayW &Y, int t) {
    const cudaStream_t st = cudaStreamPerThread;
    int nb = 1;
    while ((Y.m0 - 1) >> nb) nb++;
    const size_t window = std::min(WINDOW, Y.k);
    u64 *hv, *g;
    int8_t *panel;
    CK(cudaMalloc(&hv, (size_t)t * nb * 8));
    CK(cudaMalloc(&g, (size_t)t * Y.k * 8));
    CK(cudaMalloc(&panel, (size_t)TERMS * window));
    CK(cudaMemsetAsync(hv, 0, (size_t)t * nb * 8, st));
    CK(cudaMemsetAsync(panel, 0, (size_t)TERMS * window, st));
    HfColumns p;
    p.begin(Y, t, hv, nb, g);
    p.wipe(st); // new allocations are not cleared: zeroes planes, turned, sums and work
    std::vector<std::pair<int, size_t>> seen; // (summed terms, columns) of the products made so far
    auto once = [&](const int8_t *bytes, int K, size_t nc) {
      if (!K || !nc || std::find(seen.begin(), seen.end(), std::make_pair(K, nc)) != seen.end()) return;
      seen.push_back({K, nc});
      p.lay_digits(hfp::Rows{0, 0, 1, 0}, 0, K, st);
      p.product(bytes, K, K, 0, nc, st);
    };
    for (const UnitW &x : Y.U) {
      if (!fits(x)) {
        for (size_t rows : {std::min<size_t>(TERMS, x.rows), x.rows % TERMS})
          for (size_t nc : {window, Y.k % window}) once(panel, (int)((rows + 15) / 16 * 16), nc);
        continue;
      }
      bool first = true; // all rectangles (`RoutedCall`) of a unit have one shape
      hcp::routed_calls(x, [&](const hcp::RoutedCall &c) {
        if (!first) return;
        first = false;
        const int nS = c.experts * c.in_window, rows_most = (int)std::min<size_t>(PANEL, PLANES / 512);
        for (int L : {std::min(rows_most, c.L), c.L % rows_most}) {
          if (!L) continue;
          const int K = (2 * L + 63) / 64 * 64;
          const int most = (int)std::min<size_t>(std::min<size_t>(nS, WINDOW / 256), PLANES / ((size_t)256 * K));
          qblocks::transpose_bytes(p.planes, (size_t)most * 256, 2 * L, K, most * 256, p.turned, K, st);
          for (int n : {most, nS % most}) once(p.turned, K, (size_t)n * 256);
        }
      });
    }
    CK(cudaStreamSynchronize(st));
    p.end();
    for (void *q : {(void *)hv, (void *)g, (void *)panel}) cudaFree(q);
  }
};

// (3) The masks' shares. fill(r0, nr, j0, nj, out, st) writes mask[r][j] for r0 <= r < r0 + nr, j0 <= j < j0 + nj to
// out[(r - r0) nj + (j - j0)] (GPU, canonical) on stream st, and 0 where j >= nm: hf::masks_of_rows() (hf_mask.cuh)
// in a wrapper that supplies the key. In every call the first j and nj are multiples of 8, the size
// of a generator block.
// room: GPU scratch at a multiple of 512 bytes; each function zeroes it, on st, as its last step.
// Outside room, the key and the masks exist here in kernel arguments, local variables and the
// shared memory of k_mask_limbs_made; none of these is erased.
typedef std::function<void(uint32_t r0, uint32_t nr, uint64_t j0, uint64_t nj, u64 *out, cudaStream_t st)> HfMaskFill;
// The portion of hf_mask_fold(), whose kernel generates the masks: no buffer holds them.
// Its key: host; key_for_a_launch() ends the run if it is null or all zero.
// limit: hf::mask_limit(key), the mask generator's threshold: a block with a value at or above it
// is retried (hf_mask8(), hf_mask.cuh). room: at least 1024 + 8 sizeof(Ext) bytes, or the process
// exits; 32 MB is plenty.
// Requires nm <= 2^35 (a generator block's number has 32 bits); not checked.
static inline size_t hf_mask_fold_made_portion(size_t room_bytes) {
  if (room_bytes < 1024 + 8 * sizeof(Ext)) hcp::internal_error("the masks' fold needs a larger room");
  return (room_bytes - 1024) / sizeof(Ext) / 8 * 8;
}
// um[j] = sum over r < R of eg[r] mask[r][j], j < nm. eg: GPU. um: nm elements, GPU or pinned host.
static inline void hf_mask_fold(const hf_key32 *key, uint64_t limit, uint32_t R, uint64_t nm, const Ext *eg, Ext *um,
                                void *room, size_t room_bytes, cudaStream_t st) {
  const size_t most = hf_mask_fold_made_portion(room_bytes);
  Ext *out = (Ext *)room;
  for (uint64_t j0 = 0; j0 < nm; j0 += most) {
    const size_t n = (size_t)std::min<uint64_t>(most, nm - j0), ng = (n + 7) / 8;
    hfp::k_mask_fold_made<<<(unsigned)((ng + 127) / 128), 128, 0, st>>>(hf::key_for_a_launch(key), limit, R,
                                                                        (uint32_t)(j0 / 8), ng, eg, out);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(um + j0, out, n * sizeof(Ext), cudaMemcpyDefault, st));
  }
  CK(cudaMemsetAsync(room, 0, room_bytes, st));
}
// The smallest room_bytes, with a null `key`, for hf_mask_columns(), which computes its sums
// by cuBLAS: masks and hi_a are both written as eight balanced digits, so
// 64 sums are joined per entry of g. At full size (lo = 7) with, say, t = 48, 1 GiB holds 7 rows of
// 65,536 x 128 masks, or 14 with `key`. No source was found in public papers for stacking byte
// digits of both sides; the search was made for prover/hiding/gram_products.cuh.
static inline size_t hf_mask_columns_room_least(int lo, int t) {
  return 4096 + blas::BLAS_WORKSPACE + (size_t)t * 8 * 32 + ((size_t)32 * 16 << lo) + ((size_t)t * 8 * 8 * 4 << lo);
}
// g[a][r 2^lo + cl] += sum over q, q 2^lo + cl < nm, of hi_a((i0 >> lo) + q) mask[r][q 2^lo + cl].
// i0: the first mask's coefficient index; in the programs 0, since the masks have an array of
// their own. It must be a multiple of 2^lo: only then does coefficient
// i0 + q 2^lo + cl have the low bits cl, like the message entries summed in the same entry of g.
// hv: GPU, [t][nb], 2^nb > (i0 + nm - 1) >> lo. g: GPU, [t][R 2^lo], canonical before and after.
// key: if null, the masks come from fill, through memory. Otherwise lo >= 3 is required, fill is
// not called, and k_mask_limbs_made generates the masks, under key and limit = hf::mask_limit(key).
// handle: if null, the calling thread's cuBLAS handle (blas::blas()), which the thread should have
// created beforehand. A thread that lives for one opening, such as the one that
// Prover::fold_begin() starts, should pass the handle of a longer-lived thread, which must be idle
// meanwhile: no handle is then looked up or created inside a batch (prover/common/blas.h). The
// owner's next blas::blas() call sets stream and workspace again.
static inline void hf_mask_columns(const HfMaskFill &fill, uint32_t R, int lo, uint64_t nm, const u64 *hv, int nb,
                                   uint64_t i0, int t, u64 *g, void *room, size_t room_bytes, cudaStream_t st,
                                   const hf_key32 *key = nullptr, uint64_t limit = 0, cublasHandle_t handle = nullptr) {
  if ((i0 & (((uint64_t)1 << lo) - 1)) || !nm || ((i0 + nm - 1) >> lo >> nb) || (key && lo < 3))
    hcp::internal_error("the masks' share of the columns needs i0 at a multiple of 2^lo and 2^nb above the last index");
  const uint64_t base = i0 >> lo, all = (nm + ((uint64_t)1 << lo) - 1) >> lo;
  const size_t width = (size_t)t * hcp::LIMBS, k = (size_t)R << lo;
  // K: the values of q in a product, a multiple of 32. nr: the rows at a time.
  size_t K = (size_t)std::min<uint64_t>(hcp::CUT, (all + 31) / 32 * 32), nr = 0;
  for (;; K = (K / 2 + 31) / 32 * 32) {
    const size_t fixed = 4096 + hcp::up512(blas::BLAS_WORKSPACE) + width * K,
                 a_row = (K * (key ? 8 : 16) + 8 * width * 4) << lo;
    nr = room_bytes > fixed ? std::min<size_t>((room_bytes - fixed) / a_row, R) : 0;
    if (nr || K == 32) break;
  }
  if (!nr) hcp::internal_error("the masks' share of the columns needs a larger room");
  nr = std::min(nr, (((size_t)1 << 31) - 1) / (width * 8 << lo)); // cuBLAS's 32-bit interface
  nr = std::min<size_t>(nr, 65535);                               // a grid's third dimension
  char *at = (char *)room;
  auto take = [&](size_t bytes) {
    char *p = at;
    at += hcp::up512(bytes);
    return p;
  };
  void *work = take(blas::BLAS_WORKSPACE);
  int8_t *D = (int8_t *)take(width * K), *A = (int8_t *)take((nr * K * 8) << lo);
  u64 *m = key ? nullptr : (u64 *)take((nr * K * 8) << lo);
  int *S = (int *)take((nr * 8 * width * 4) << lo);
  if (at > (char *)room + room_bytes) hcp::internal_error("the masks' share of the columns: the room's reckoning");
  for (uint64_t q0 = 0; q0 < all; q0 += K) {
    const size_t nq = (size_t)std::min<uint64_t>(K, all - q0), Kn = (nq + 31) / 32 * 32, nj = ((nq << lo) + 7) / 8 * 8;
    hfp::k_hi_digits<<<hfp::grid((size_t)t * Kn), 256, 0, st>>>(hv, nb, t, hfp::Rows{base + q0, 0, 1, (int)nq}, (int)nq,
                                                                 (int)Kn, D);
    CK(cudaGetLastError());
    for (uint32_t r0 = 0; r0 < R; r0 += (uint32_t)nr) {
      const uint32_t n = (uint32_t)std::min<size_t>(nr, R - r0);
      if (key)
        hfp::k_mask_limbs_made<<<dim3((unsigned)(Kn / 32), (unsigned)((((size_t)1 << lo) + 127) / 128), n), 256, 0, st>>>(
            hf::key_for_a_launch(key), limit, r0, q0, nq, nm, lo, Kn, A);
      else {
        fill(r0, n, q0 << lo, nj, m, st);
        hfp::k_mask_limbs<<<hfp::grid(((size_t)n * (Kn / 16)) << lo), 256, 0, st>>>(m, n, nj, lo, nq, Kn, A);
      }
      CK(cudaGetLastError());
      if (handle) {
        if (cublasSetStream(handle, st) != CUBLAS_STATUS_SUCCESS ||
            cublasSetWorkspace(handle, work, blas::BLAS_WORKSPACE) != CUBLAS_STATUS_SUCCESS)
          hcp::internal_error("cuBLAS refused the stream or the work space of the masks' share of the columns");
        blas::gemm_tn(handle, (int)width, (int)(((size_t)n * 8) << lo), (int)Kn, D, Kn, A, Kn, S, width, false);
      } else
        hcp::gemm(st, work, (int)width, (int)(((size_t)n * 8) << lo), (int)Kn, D, Kn, A, Kn, S, width, false);
      hfp::k_mask_join<<<hfp::grid(((size_t)n * t) << lo), 256, 0, st>>>(S, n, lo, t, r0, g, k);
      CK(cudaGetLastError());
    }
  }
  CK(cudaMemsetAsync(room, 0, room_bytes, st));
}
// The indices x at which the caller's self-check of um ([own-test-fold], the name that a failure
// prints) compares it with a sum computed on the host: the first and last x of every portion of
// hf_mask_fold() (portion: hf_mask_fold_made_portion() of its room),
// 8 * 65,536 if it is below nm, and the caller's `drawn` (must be below nm). Ascending, no repeats.
static inline std::vector<uint64_t> hf_mask_fold_places(uint64_t portion, uint64_t nm, std::vector<uint64_t> drawn) {
  for (uint64_t j0 = 0; j0 < nm; j0 += portion) drawn.push_back(j0), drawn.push_back(std::min(j0 + portion, nm) - 1);
  if ((uint64_t)8 * 65536 < nm) drawn.push_back((uint64_t)8 * 65536);
  std::sort(drawn.begin(), drawn.end());
  drawn.erase(std::unique(drawn.begin(), drawn.end()), drawn.end());
  return drawn;
}
} // namespace rf

namespace hf {
typedef rf::HfPass HfPass;
// A function that adds every unit of W to `pass` and returns when the last unit's work has run.
// Nothing uses this type: the opening uses rf::HfWalk (hf_walk.h).
typedef std::function<void(HfPass &pass)> HfWalk;
} // namespace hf
