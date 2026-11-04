// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// GPU kernels for the exact matrix product row_core_i8() in
// prover/common/row_pass_core.cuh, which includes this file after declaring i8_, join8() and P9.
// row_core_i8() computes out[v][r] = sum over j < K of phi[v][j] * X[r * ld + j], where X is
// signed bytes and phi and out are extension-field elements (Ext, prover/common/common.h): EDEG
// coordinates modulo the field's prime p, summed separately. k_digits() splits phi into signed-byte
// digits (blk::lazy::signed_digits(), prover/field/field_lazy.cuh). cuBLAS multiplies the digits by
// X into exact 32-bit sums (prover/common/blas.h). k_join() combines every eight sums with join8().
// The kernels check no argument, and no launcher checks nphi <= 9, the length of P9's arrays.
// Public source: a radix-L digit s_i >= L/2 becomes s_i - L, with a carry: drouyang, "Signed Bucket
// Indexes for Multi-Scalar Multiplication (MSM)", https://hackmd.io/@drouyang/signed-bucket-index.
// Public source: S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, M. Zhang, "TensorFHE: Achieving Practical Computation on
// Encrypted Data Using GPGPU" (arXiv:2212.14191): "We use a segment-fusion scheme to fulfill the GEMM without precision
// loss"; "For each 32-bit integer element m_{i,j} in the matrix M, we segment every consecutive 8-bits into an 8-bit
// integer": residues cut into bytes, multiplied on tensor cores into 32-bit sums, and fused again.
#pragma once
#include "common/blas.h"
#include <cstdio>
#include <cstdlib>
namespace row_pass_core {
// Writes each coordinate x of phi[v][j] as digits from -128 to 127: x = sum over b < 8 of 256^b *
// digit_b modulo p. Row (v * EDEG + e) * 8 + b of E, of length K, stores digit b of coordinate e.
// A byte-to-byte carry loop would be wrong for p near 2^64: its digits for large x sum to x - 2^64.
__global__ void k_digits(P9 p, int nphi, int K, i8_ *E) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < K; j += gridDim.x * blockDim.x)
    for (int v = 0; v < nphi; v++) {
      const Ext w = p.phi[v][j];
      for (int e = 0; e < EDEG; e++) {
        const u64 x = blk::lazy::signed_digits(blk::F{coord(w, e)});
        for (int b = 0; b < 8; b++) E[(size_t)((v * EDEG + e) * 8 + b) * K + j] = (i8_)(x >> (8 * b));
      }
    }
}
// Coordinate c % EDEG of out[c / EDEG][r] = sum over b < 8 of 256^b * raw[(r * nc + c) * 8 + b]
// modulo p, for r < rows and c < nc. nc = EDEG times the number of out vectors. raw: cuBLAS's
// sums of K terms (|digit * byte| <= 2^14), exact for K <= 65,536, which row_core_check() tests.
// The cast to u64 * relies on an Ext being EDEG consecutive 8-byte coordinates.
// Public sources for the techniques of this routine. The loop form is public: a grid-stride loop:
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_join(const int *raw, int rows, int nc, P9 p) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rows * nc;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / nc;
    const int c = (int)(i % nc);
    const int *s = raw + i * 8;
    ((u64 *)p.out[c / EDEG])[r * EDEG + c % EDEG] = join8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
  }
}
} // namespace row_pass_core
