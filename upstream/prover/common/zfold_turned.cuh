// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// zfold_turned_i8<TW>() computes TW random combinations of a matrix M's columns for a sub-proof's
// approximate range check (every entry of Z must lie in an interval: prover/hiding/tail.cuh):
//   Z[y * TW + u] = sum_{c < k} M[c * ldm + y] * Cf[c][u], in int32, for y < m and u < TW.
// A sub-proof is the argument for one part of the forward pass. The sums ("fold" in the names) are
// exact in any order if the caller keeps all partial sums below 2^31 in absolute value, which is
// not checked. It transposes M first ("turned" in the names). A test that is not part of this
// repository keeps a second way, which computes the same sums from M as it is stored: compared
// with that, zfold_turned_i8() is written to be faster on large matrices and needs a larger work
// buffer. The idea is public: the 4 x 4 byte transpose by eight __byte_perm calls
// (selectors 0x5140, 0x7362, then 0x5410, 0x7632) is in MegEngine, dnn/src/cuda/memory_utils.cuh,
// transpose_int8_4x4_impl (Copyright (c) 2014-2021 Megvii Inc.; Apache License 2.0 by the file's
// header and the repository's README). k_zfold_turn makes the same eight calls, with the same
// operands and selectors (see NOTICE).
#pragma once
#include "common/blas.h"
#include "common/row_pass_core.cuh" // row_pass_core::i8_, zfold_kp(), zfold_prep()
#include "common/local_memory.cuh"
#include <algorithm>
namespace row_pass_core {
// Transposes a piece of M: T[y * kc + c] = M[c * ldm + y] for y < rows, c < k; 0 for k <= c < kc.
// Requires, unchecked: rows % 256 == 0, kc % 16 == 0, k <= kc, ldm % 16 == 0, and M and T aligned
// to 16 bytes. Launch: 256 threads in a block, rows / 256 blocks in x, (kc + 255) / 256 in y.
// A thread transposes 16 x 16 bytes in registers: each of the sixteen 4 x 4 squares takes two
// levels of four __byte_perm calls, and the squares change places by the choice of registers. In a
// warp 4 lanes are adjacent in y and 8 in c, so its 16-byte reads touch runs of 64 bytes and its
// writes runs of 128. In the source no address, branch or number of steps depends on a byte of M.
// Tested via zfold_turned_i8(): a test that is not part of this repository checks it and a second
// way against the host's sums (and can time both).
// Ours, after k_cft in prover/common/row_pass_core.cuh, which transposes coefficients byte by byte.
// The idea is public: a square transposed on the chip (there through shared memory, here in a
// thread's registers): M. Harris, "An Efficient Matrix Transpose in CUDA C/C++", NVIDIA Technical
// Blog, https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/.
// The intrinsic is public: __byte_perm(x, y, s) returns four of the eight bytes of x and y, chosen
// by s: NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics" (as of 2025-11-04),
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html.
// The idea is public: a 4 x 4 byte transpose in eight __byte_perm calls appears in MegEngine's
// transpose_int8_4x4_impl (dnn/src/cuda/memory_utils.cuh, Apache-2.0), and building a 16 x 16 byte
// transpose in registers from 4 x 4 squares is described for SSE in the blog post 'How to transpose
// a 16x16 byte matrix using SSE' (pzemtsov.github.io, 2014).
// The compiler hint is public: __launch_bounds__ with minBlocksPerMultiprocessor: NVIDIA, CUDA
// Programming Guide, "C/C++ Language Extensions", "Launch Bounds" (as of 2026-03-12),
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// Two blocks for each multiprocessor leave a thread 128 registers. Without the hint the compiler
// used 80 and spilled three pairs of registers, entries of M among them, to local memory, which
// nothing erases. zfold_turned_i8() makes the program exit if the kernel has local memory.
__global__ void __launch_bounds__(256, 2)
    k_zfold_turn(const i8_ *__restrict__ M, size_t ldm, int k, int kc, int rows, i8_ *__restrict__ T) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int y = blockIdx.x * 256 + (warp & 3) * 64 + (lane >> 3) * 16;
  const int c = blockIdx.y * 256 + (warp >> 2) * 128 + (lane & 7) * 16;
  if (c >= kc || y >= rows) return;
  unsigned in[16][4]; // in[i][w]: the bytes at c + i and y + 4 w .. y + 4 w + 3
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint4 x = make_uint4(0, 0, 0, 0);
    if (c + i < k) x = *(const uint4 *)(M + (size_t)(c + i) * ldm + y);
    in[i][0] = x.x, in[i][1] = x.y, in[i][2] = x.z, in[i][3] = x.w;
  }
#pragma unroll
  for (int w = 0; w < 4; w++) {
    unsigned out[4][4]; // out[t][v]: the bytes at y + 4 w + t and c + 4 v .. c + 4 v + 3
#pragma unroll
    for (int v = 0; v < 4; v++) {
      const unsigned a0 = in[4 * v][w], a1 = in[4 * v + 1][w], a2 = in[4 * v + 2][w], a3 = in[4 * v + 3][w];
      const unsigned low01 = __byte_perm(a0, a1, 0x5140), low23 = __byte_perm(a2, a3, 0x5140);
      const unsigned high01 = __byte_perm(a0, a1, 0x7362), high23 = __byte_perm(a2, a3, 0x7362);
      out[0][v] = __byte_perm(low01, low23, 0x5410);
      out[1][v] = __byte_perm(low01, low23, 0x7632);
      out[2][v] = __byte_perm(high01, high23, 0x5410);
      out[3][v] = __byte_perm(high01, high23, 0x7632);
    }
#pragma unroll
    for (int t = 0; t < 4; t++)
      *(uint4 *)(T + (size_t)(y + 4 * w + t) * kc + c) = make_uint4(out[t][0], out[t][1], out[t][2], out[t][3]);
  }
}
} // namespace row_pass_core
// zfold_turned_i8()'s work buffer: cuBLAS's workspace, then the space for a transposed piece of M.
constexpr size_t ZFOLD_TURNED_BLAS = (size_t)4 << 20, ZFOLD_TURNED_PIECE = (size_t)32 << 20;
constexpr size_t ZFOLD_TURNED_WORKSPACE = ZFOLD_TURNED_BLAS + ZFOLD_TURNED_PIECE;
// cuBLAS's fast kernels for signed bytes need the summed index to vary fastest in both matrices
// (blas::gemm_tn(), prover/common/blas.h). In M it varies slowest, so M is transposed by pieces
// into the work buffer (k_zfold_turn) and each piece is multiplied by CfT. If 256 rows of all
// columns (zero-padded to a multiple of 16) fit in `piece` bytes, M is cut by rows into the fewest
// pieces that fit, of the same multiple of 256 rows each (the last perhaps fewer). If not, a piece
// has 256 rows of as many columns as fit, and the products of pieces with the same rows are added.
// The program exits if a size or address fails the check below; the prover fixes them at set-up.
// CfT: as zfold_prep<TW>() writes it, kp = zfold_kp(k): CfT[u * kp + c] = Cf[c][u], 0 for c >= k.
// `work`: ZFOLD_TURNED_BLAS + `piece` bytes, for one stream at a time. After the call it is still
// the thread's cuBLAS workspace and keeps a copy of entries of M; the prover zeroes it only when it
// zeroes the sub-proofs' shared scratch memory, the slab (defined in prover/prover/fork_memory.h).
// `piece`: a piece's bytes. The prover uses the default; no program of this repository passes less.
template <int TW>
static inline void zfold_turned_i8(const int8_t *M, size_t ldm, int m, int k, const int8_t *CfT, int kp, int *Z,
                                   void *work, cudaStream_t st = 0, size_t piece = ZFOLD_TURNED_PIECE) {
  if (m < 0 || k < 0 || m % 256 || ldm % 16 || ldm < (size_t)m || ((uintptr_t)M & 15) || ((uintptr_t)CfT & 15) ||
      kp % 16 || kp < k || !piece || piece % 4096 || ((uintptr_t)work & 255)) {
    printf("internal error in the sign fold by pieces: %d rows (a multiple of 256), %d columns, %d columns of "
           "coefficients (a multiple of 16, at least the columns), pitch %zu (a multiple of 16, at least the rows), a "
           "matrix not at a multiple of 16 bytes, a room of %zu bytes (a multiple of 4,096), or a work buffer not at a "
           "multiple of 256\n",
           m, k, kp, ldm, piece);
    exit(1);
  }
  if (!k) CK(cudaMemsetAsync(Z, 0, (size_t)m * TW * 4, st));
  if (!k || !m) return;
  const size_t k16 = ((size_t)k + 15) / 16 * 16, columns = std::min(k16, piece / 256);
  const size_t most_rows = piece / columns / 256 * 256, pieces = ((size_t)m + most_rows - 1) / most_rows;
  const size_t rows = ((size_t)m + pieces * 256 - 1) / (pieces * 256) * 256;
  static const bool looked_at = (local_memory::refuse_local_memory(row_pass_core::k_zfold_turn, "row_pass_core::k_zfold_turn"), true);
  (void)looked_at;
  int8_t *turned = (int8_t *)work + ZFOLD_TURNED_BLAS;
  const cublasHandle_t handle = blas::blas(st, work, ZFOLD_TURNED_BLAS);
  for (size_t y = 0; y < (size_t)m; y += rows)
    for (size_t c = 0; c < k16; c += columns) {
      const int ny = (int)std::min(rows, m - y), nc = (int)std::min(columns, k16 - c);
      const int real = (int)std::min((size_t)nc, k - c); // the other columns of the piece are zeros
      row_pass_core::k_zfold_turn<<<dim3(ny / 256, (nc + 255) / 256), 256, 0, st>>>(M + c * ldm + y, ldm, real, nc, ny, turned);
      CK(cudaGetLastError());
      blas::gemm_tn(handle, TW, ny, nc, CfT + c, kp, turned, nc, Z + y * TW, TW, c > 0);
    }
}
