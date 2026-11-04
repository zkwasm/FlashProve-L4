// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// An exact matrix product, computed by cuBLAS. row_core_i8() multiplies a matrix X of signed bytes
// by elements of the extension field Ext, each of the EDEG coordinates modulo the field's prime
// p = 2^64 - 2^32 + 1: it splits each coordinate into
// eight signed byte digits, multiplies the digits by X in int32, and joins the sums modulo p.
// zfold_kp() and zfold_prep() serve zfold_turned_i8() (prover/common/zfold_turned.cuh).
// Every error path prints a message and ends the program. A rejected argument or a failed cuBLAS
// call leads to exit(1). A failed CUDA call goes through the includer's CK(): exit(1) as
// prover/common/common.h defines it, exit(5) as prover/weights_commitment/common.cuh redefines it.
// Include prover/common/common.h first: it provides u64, Ext, EDEG, coord() and CK().
// Where the rows of X are several byte limbs of the caller's values (two in
// prover/head/hidden_fork.cuh), both sides of the product carry stacked byte digits, and no source
// was found in public papers for that; the search was made for
// prover/hiding/gram_products.cuh.
#pragma once
#include "field/field_names.h"
#include "field/field_lazy.cuh" // blk::lazy::signed_digits, join_digit
#include <cuda_runtime.h>
namespace row_pass_core {
typedef int8_t i8_;
// Returns the sum over b < 8 of 2^(8 b) s_b modulo p, canonical (below p), for any eight 32-bit
// sums. join_digit(v, s) (prover/field/field_lazy.cuh) returns v 2^s modulo p if |v| < 2^JOIN_BITS,
// where JOIN_BITS is 61. The four low and the four high terms are first added as integers, whose
// absolute values are at most 2^31 (1 + 2^8 + 2^16 + 2^24) < 2^56: two calls instead of eight.
__device__ __forceinline__ u64 join8(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
  const long long lo = (long long)s0 + (long long)s1 * 256 + (long long)s2 * 65536 + (long long)s3 * 16777216;
  const long long hi = (long long)s4 + (long long)s5 * 256 + (long long)s6 * 65536 + (long long)s7 * 16777216;
  return (blk::lazy::join_digit(lo, 0) + blk::lazy::join_digit(hi, 32)).v;
}
// CfT[u * kp + c] = Cf[c * TW + u] for c < k, zero for k <= c < kp. See zfold_prep().
// Each thread writes at most one byte of CfT, so a launch needs at least TW kp threads.
// Public source: a thread's index, tested against the size: NVIDIA, CUDA C++ Programming Guide,
// "Thread Hierarchy".
__global__ void k_cft(const i8_ *Cf, int k, int kp, int TW, i8_ *CfT) {
  const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (x < (size_t)TW * kp) {
    const size_t c = x % kp, u = x / kp;
    CfT[x] = c < (size_t)k ? Cf[c * TW + u] : (i8_)0;
  }
}
// The kernels' pointers to row_core_i8()'s vectors phi and out. NPHI, a template parameter, is the
// number of vectors phi, and of vectors out, in one call. This file does not check NPHI <= 9.
struct P9 {
  const Ext *phi[9];
  Ext *out[9];
};
} // namespace row_pass_core
#include "common/row_pass_core_kernels.cuh"
// cuBLAS's workspace for row_core_i8() is the end of the call's scratch buffer, so zeroing the
// buffer zeroes it, and calls with separate buffers have separate workspaces. NVIDIA, cuBLAS
// documentation (https://docs.nvidia.com/cuda/cublas/), "cublasSetWorkspace()": a workspace of at
// least 16 KiB rules out the error CUBLAS_STATUS_ALLOC_FAILED.
constexpr size_t ROW_CORE_WORKSPACE = (size_t)4 << 20;
static inline int zfold_kp(int k) {
  return (k + 127) / 128 * 128;
}
// Writes CfT, which is Cf transposed and zero-padded to zfold_kp(k) columns (k_cft's formula):
// TW * zfold_kp(k) bytes, which zfold_turned_i8() (prover/common/zfold_turned.cuh)
// reads. Cf[c * TW + u] is the coefficient of a matrix's column c in random combination u of the
// approximate range check on committed cells, and TW is the number of combinations. A request's
// sub-proofs (each is the argument for one part of the forward pass) use TW = ETAU
// (prover/common/setting.h). The launch is not checked.
template <int TW> static inline void zfold_prep(const int8_t *Cf, int k, int8_t *CfT, cudaStream_t st = 0) {
  const int kp = zfold_kp(k);
  // The + 1 covers a remainder and keeps the grid from being empty when k is 0.
  row_pass_core::k_cft<<<(unsigned)((size_t)TW * kp / 256 + 1), 256, 0, st>>>(Cf, k, kp, TW, CfT);
}
// Bytes of row_core_i8()'s scratch buffer that precede cuBLAS's workspace: the digits of phi
// (k_digits splits each coordinate into eight signed bytes), rounded up to a multiple of 256, then
// 8 * EDEG * nphi 32-bit sums for each row.
static inline size_t row_core_ahead(int nphi, int K, int rows) {
  return ((size_t)nphi * EDEG * 8 * K + 255) / 256 * 256 + (size_t)rows * nphi * EDEG * 32;
}
// Bytes of row_core_i8()'s scratch buffer. The 256 let row_core_product() align the workspace.
static inline size_t row_core_scratch(int nphi, int K, int rows) {
  return row_core_ahead(nphi, K, rows) + 256 + ROW_CORE_WORKSPACE;
}
// row_core_i8() in three steps, for a caller that multiplies several matrices X by the same
// vectors phi (prover/weights_blocks/row_pass_blocks.cuh, if ROW_PASS_IN_ONE_KERNEL is 0 there; if
// it is 1, that file calls step 1 but not steps 2 and 3).
// The steps share a scratch buffer of row_core_scratch(NPHI, K, rows) bytes, which only one stream
// may use at a time. Between row_core_digits() and the last step that follows it, nothing else may
// use the buffer, and NPHI, K and `rows` must stay the same. The steps do not check their
// arguments: the caller must call row_core_check() before step 1.

// Exits unless K and ld meet the conditions in its message. K <= 65,536 keeps the 32-bit sums
// exact: |digit * byte| <= 2^14, so |sum| <= 2^30.
static inline void row_core_check(int K, size_t ld) {
  if (K % 16 || K > 65536 || K < 16 || ld % 16) {
    printf("internal error: row pass over bytes: K must be a multiple of 16 from 16 to 65,536 and ld a multiple of 16 "
           "(K %d, ld %zu)\n", K, ld);
    exit(1);
  }
}
// Step 1: writes the digits of the vectors phi, K elements each, to the start of scratch.
template <int NPHI>
static inline void row_core_digits(int K, const Ext *const *phi, void *scratch, cudaStream_t st = 0) {
  row_pass_core::P9 p{};
  for (int v = 0; v < NPHI; v++) p.phi[v] = phi[v];
  row_pass_core::k_digits<<<(K + 255) / 256, 256, 0, st>>>(p, NPHI, K, (int8_t *)scratch);
  CK(cudaGetLastError());
}
// Step 2: multiplies the digits by the n rows of X (ld bytes apart) and writes the 32-bit sums to
// rows first .. first + n - 1 of the buffer's `rows` rows. Requires first + n <= rows. The
// workspace follows all `rows` rows, so a product does not overwrite earlier products' sums.
template <int NPHI>
static inline void row_core_product(const int8_t *X, size_t ld, int first, int n, int rows, int K, void *scratch,
                                     cudaStream_t st = 0) {
  constexpr int NT = EDEG * NPHI;
  int8_t *base = (int8_t *)scratch;
  const int8_t *digits = base;
  int *sums = (int *)(base + row_core_ahead(NPHI, K, 0)) + (size_t)first * NT * 8;
  void *work = (void *)(((uintptr_t)scratch + row_core_ahead(NPHI, K, rows) + 255) & ~(uintptr_t)255);
  blas::gemm_tn(blas::blas(st, work, ROW_CORE_WORKSPACE), NT * 8, n, K, digits, K, X, ld, sums, NT * 8, false);
}
// Step 3: joins each coordinate's eight sums (k_join) into out[v][r], for v < NPHI and r < n.
template <int NPHI>
static inline void row_core_join(int n, int K, Ext *const *out, void *scratch, cudaStream_t st = 0) {
  constexpr int NT = EDEG * NPHI;
  row_pass_core::P9 p{};
  for (int v = 0; v < NPHI; v++) p.out[v] = out[v];
  const int *sums = (const int *)((const int8_t *)scratch + row_core_ahead(NPHI, K, 0));
  row_pass_core::k_join<<<(unsigned)(((size_t)n * NT + 255) / 256), 256, 0, st>>>(sums, n, NT, p);
  CK(cudaGetLastError());
}
// out[v][r] = sum over j < K of phi[v][j] * X[r * ld + j] for v < NPHI and r < rows, each of the
// EDEG coordinates modulo p. phi and out are host arrays of pointers to GPU memory. Exits unless K
// and ld pass row_core_check(), whose message calls this product the "row pass over bytes". Not
// checked: scratch must have row_core_scratch(NPHI, K, rows) bytes, and X and scratch must be
// aligned to 4 bytes, as cuBLAS's documentation requires of byte operands (cited in
// prover/common/blas.h). Calls on streams that run concurrently need separate scratch buffers. No
// function in this file zeroes scratch, whose contents depend on X.
template <int NPHI>
static inline void row_core_i8(const int8_t *X, size_t ld, int rows, int K, const Ext *const *phi, Ext *const *out,
                                void *scratch, cudaStream_t st = 0) {
  row_core_check(K, ld);
  row_core_digits<NPHI>(K, phi, scratch, st);
  row_core_product<NPHI>(X, ld, 0, rows, rows, K, scratch, st);
  row_core_join<NPHI>(rows, K, out, scratch, st);
}
