// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Set-up kernels that convert a tensor's stored blocks F (type ty: 0 Q4_K, 1 Q5_K, 2 Q6_K, 3 F32,
// 4 Q8_0, 5 F16, 6 BF16) to its committed integers w (`fold` in identifiers) as fold_rule.h defines:
// w is the weight times 2^e G[j], rounded; a null G means 1. Row o (of `rows`) is an output, element
// j (of n0) an input.
#pragma once
#include "weights_blocks/fold_rule.h"
namespace fs_ {
// Kernel of fold_i16(). F has nblk blocks, nb0 = n0 / 256 per row. Not checked: ty, n0 % 256 == 0.
// The loop form is public: the grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post
// on NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_int(int ty, const uint8_t *F, size_t nblk, int nb0, const float *G, int e, int16_t *out,
                      unsigned *over, int lim) {
  const int BB = fr::block_bytes(ty);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk * 256; i += (size_t)gridDim.x * blockDim.x) {
    const size_t blk = i >> 8;
    const int el = (int)(i & 255);
    const int w = fr::weight(ty, F + blk * BB, el, G ? G[(blk % nb0) * 256 + el] : 1.f, e);
    if (w > lim || w < -lim) atomicOr(over, 1u);
    out[i] = (int16_t)w;
  }
}
// Kernel of fold_range(). The loop form is public: the grid-stride loop (cited at k_int()).
__global__ void k_range(int ty, const uint8_t *F, size_t nblk, int nb0, const float *G, int e, unsigned *over) {
  const int BB = fr::block_bytes(ty);
  bool out = false;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk * 256; i += (size_t)gridDim.x * blockDim.x) {
    const size_t blk = i >> 8;
    const int el = (int)(i & 255);
    const int w = fr::weight(ty, F + blk * BB, el, G ? G[(blk % nb0) * 256 + el] : 1.f, e);
    out = out || w > 4095 || w < -4095;
  }
  if (out) atomicOr(over, 1u);
}
// Kernel of fold_planes(). The loop form is public: the grid-stride loop (cited at k_int()).
__global__ void k_pl(int ty, const uint8_t *F, size_t nblk, int nb0, const float *G, int e, int8_t *w0, int8_t *w1,
                     size_t ld) {
  const int BB = fr::block_bytes(ty);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk * 256; i += (size_t)gridDim.x * blockDim.x) {
    const size_t blk = i >> 8;
    const int el = (int)(i & 255);
    const size_t j = (blk % nb0) * 256 + el;
    int b0, b1;
    fr::limbs(fr::weight(ty, F + blk * BB, el, G ? G[j] : 1.f, e), &b0, &b1);
    const size_t o = (blk / nb0) * ld + j;
    w0[o] = (int8_t)b0;
    w1[o] = (int8_t)b1;
  }
}
// Kernel of fold_cells(). The loop form is public: the grid-stride loop (cited at k_int()).
__global__ void k_ce(int ty, const uint8_t *F, size_t nblk, int nb0, const float *G, int e, int8_t *lo, int8_t *hi,
                     size_t ldc) {
  const int BB = fr::block_bytes(ty);
  const size_t rows = nblk / nb0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nblk * 256; i += (size_t)gridDim.x * blockDim.x) {
    const size_t o = i % rows, j = i / rows; // o: the output (the row); j: the input (the element of the row)
    int b0, b1;
    fr::limbs(fr::weight(ty, F + (o * nb0 + (j >> 8)) * BB, (int)(j & 255), G ? G[j] : 1.f, e), &b0, &b1);
    lo[j * ldc + o] = (int8_t)b0;
    hi[j * ldc + o] = (int8_t)b1;
  }
}
// Grid size for a launch: nblk, clamped to 1 .. 2^20. The kernels work with any grid size.
static inline unsigned grid(size_t nblk) {
  return (unsigned)(nblk < ((size_t)1 << 20) ? (nblk ? nblk : 1) : (size_t)1 << 20);
}
} // namespace fs_
// Writes the committed integers w as int16 to out[o * n0 + j]; sets *over as fold_range() does.
static inline void fold_i16(int ty, const void *F, size_t rows, int n0, const float *G, int e, int16_t *out,
                            unsigned *over, cudaStream_t st = 0) {
  fs_::k_int<<<fs_::grid(rows * (n0 / 256)), 256, 0, st>>>(ty, (const uint8_t *)F, rows * (n0 / 256), n0 / 256, G, e,
                                                           out, over, 4095);
}
// Sets bit 0 of *over if a weight is not finite or its w has |w| > 4095; never clears it.
static inline void fold_range(int ty, const void *F, size_t rows, int n0, const float *G, int e, unsigned *over,
                              cudaStream_t st = 0) {
  fs_::k_range<<<fs_::grid(rows * (n0 / 256)), 256, 0, st>>>(ty, (const uint8_t *)F, rows * (n0 / 256), n0 / 256, G, e,
                                                             over);
}
// Writes w's limbs (fr::limbs()): low to w0[o * ld + j], high to w1[o * ld + j]. No check of w.
static inline void fold_planes(int ty, const void *F, size_t rows, int n0, const float *G, int e, int8_t *w0,
                               int8_t *w1, size_t ld, cudaStream_t st = 0) {
  fs_::k_pl<<<fs_::grid(rows * (n0 / 256)), 256, 0, st>>>(ty, (const uint8_t *)F, rows * (n0 / 256), n0 / 256, G, e, w0,
                                                          w1, ld);
}
// As fold_planes(), but transposed: to lo[j * ldc + o] and hi[j * ldc + o].
static inline void fold_cells(int ty, const void *F, size_t rows, int n0, const float *G, int e, int8_t *lo, int8_t *hi,
                              size_t ldc, cudaStream_t st = 0) {
  fs_::k_ce<<<fs_::grid(rows * (n0 / 256)), 256, 0, st>>>(ty, (const uint8_t *)F, rows * (n0 / 256), n0 / 256, G, e, lo,
                                                          hi, ldc);
}
