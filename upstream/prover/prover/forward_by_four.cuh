// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Loads and stores for the forward pass's element-wise kernels, which let a thread take W adjacent
// tokens, W = 1 or 4. The arrays concerned are indexed [row][t], with T tokens in a row, so with
// W = 4 a thread moves four elements in one wide access (two for 64-bit elements). Such an access
// needs an aligned address (see fits_four()), so W = 4 needs a T that is a multiple of 4.
// LAUNCH_BY_FOUR chooses W for a launch.
// A test that is not part of this repository checks that the kernels write the same bytes with
// either W, at up to 100 tokens and 36 rows.
// This header has no #include: prover/prover/forward.cuh includes it after the integer types, ResidualInt
// (a 64-bit integer), grid_blocks(), write_limbs() and write_digits() are declared.
// Public source: wide loads and stores through vector types: "CUDA Pro Tip: Increase Performance
// with Vectorized Memory Access", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/, as cited at
// ld16() (prover/hiding/field.h). In the post the elements are ints; packing four bytes or four
// 16-bit lookup keys into one or two 32-bit words first is ours.
// The idea is public: a compile-time constant, here W, as a template parameter: NVIDIA, CUTLASS
// documentation, "Programming Guidelines", "Templates",
// https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/programming_guidelines.html.
#pragma once
template <int W> __device__ __forceinline__ void ld_w(const int *p, int (&v)[W]) {
  static_assert(W == 1 || W == 4, "one token or four");
  if constexpr (W == 4) {
    const int4 x = *reinterpret_cast<const int4 *>(p);
    v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
  } else
    v[0] = *p;
}
// X must be a 64-bit integer type; only its size is checked.
template <int W, class X> __device__ __forceinline__ void ld_w(const X *p, i64 (&v)[W]) {
  static_assert(sizeof(X) == 8 && (W == 1 || W == 4), "elements of 64 bits; one token or four");
  if constexpr (W == 4) {
    const longlong2 x = *reinterpret_cast<const longlong2 *>(p), y = *reinterpret_cast<const longlong2 *>(p + 2);
    v[0] = x.x, v[1] = x.y, v[2] = y.x, v[3] = y.y;
  } else
    v[0] = (i64)*p;
}
// The st_w() overloads do not check W: with any W other than 4 they store only v[0].
template <int W> __device__ __forceinline__ void st_w(ResidualInt *p, const i64 (&v)[W]) {
  if constexpr (W == 4) {
    *reinterpret_cast<longlong2 *>(p) = make_longlong2(v[0], v[1]);
    *reinterpret_cast<longlong2 *>(p + 2) = make_longlong2(v[2], v[3]);
  } else
    *p = v[0];
}
template <int W> __device__ __forceinline__ void st_w(i8 *p, const i8 (&v)[W]) {
  if constexpr (W == 4)
    *reinterpret_cast<u32 *>(p) =
        (u32)(uint8_t)v[0] | (u32)(uint8_t)v[1] << 8 | (u32)(uint8_t)v[2] << 16 | (u32)(uint8_t)v[3] << 24;
  else
    *p = v[0];
}
template <int W> __device__ __forceinline__ void st_w(u16 *p, const u16 (&v)[W]) {
  if constexpr (W == 4)
    *reinterpret_cast<uint2 *>(p) = make_uint2((u32)v[0] | (u32)v[1] << 16, (u32)v[2] | (u32)v[3] << 16);
  else
    *p = v[0];
}
// write_limbs() for W adjacent values: limb l < nl of v[e], a base-256 digit from -128 to 127, goes to
// dst[l * stride + e].
template <int W> __device__ __forceinline__ void write_limbs_wide(const i64 (&v)[W], int nl, i8 *dst, size_t stride) {
  i64 w[W];
#pragma unroll
  for (int e = 0; e < W; e++) w[e] = v[e];
  for (int l = 0; l < nl; l++) {
    i8 b[W];
#pragma unroll
    for (int e = 0; e < W; e++) {
      const int x = (int)(((w[e] + 128) & 255) - 128);
      b[e] = (i8)x, w[e] = (w[e] - x) >> 8;
    }
    st_w<W>(dst + l * stride, b);
  }
}
// write_digits() for the k-bit rounding remainders of W adjacent tokens: what write_digits() writes to `cell`,
// `lo` and `hi` for r[e] goes to cell + e, lo + e and hi + e. This holds for `form` 0, 1 and 2;
// `form` is not checked.
template <int W>
__device__ __forceinline__ void write_digits_wide(int form, int k, const i64 (&r)[W], i8 *cell, size_t st, u16 *lo, u16 *hi) {
  u16 l[W], h[W];
  i64 c[W];
#pragma unroll
  for (int e = 0; e < W; e++) {
    if (k <= 16) {
      l[e] = (u16)(r[e] << (16 - k)), h[e] = 0, c[e] = form == 1 ? -128 : -32896;
    } else {
      const i64 x = r[e] >> 16;
      l[e] = (u16)(r[e] & 65535), h[e] = (u16)(x << ((form == 1 ? 24 : 32) - k)), c[e] = x - (form == 1 ? 128 : 32896);
    }
  }
  st_w<W>(lo, l);
  if (k <= 16 && form == 0) return;
  write_limbs_wide<W>(c, form == 1 ? 1 : 2, cell, st);
  st_w<W>(hi, h);
}
// Returns whether p is aligned for ld_w() or st_w() with W = 4: to 4 bytes for 1-byte elements
// (limbs), 8 for 2-byte elements (lookup keys) and 16 for larger ones. A null pointer fits.
template <class X> static inline bool fits_four(X *p) {
  return (uintptr_t)p % (sizeof(X) >= 4 ? 16 : 4 * sizeof(X)) == 0;
}
template <class X> static inline bool fits_four(const X &) {
  return true;
}
// Returns whether a launch over rows of T tokens with these arguments can use W = 4. A pointer
// inside a struct is tested only if fits_four() has an overload for that struct;
// prover/prover/forward.cuh adds one for GateClamp. Strides and offsets are not tested: at every
// call they are multiples of T elements, which keeps the addresses aligned. A misaligned access is
// undefined behaviour in CUDA: in practice the kernel fails with cudaErrorMisalignedAddress, but
// that is not guaranteed. A test that is not part of this repository checks only the choice of W:
// its kernel makes no wide access.
template <class... A> static inline bool by_four(int T, const A &...arguments) {
  return T % 4 == 0 && (fits_four(arguments) && ...);
}
// Launches the kernel template `kernel` on stream s, with W = 4 if by_four() allows it and W = 1
// otherwise. `items` is the kernel's loop count with W = 1. grid_blocks() caps the number of blocks, so
// the kernel must use a grid-stride loop. The kernel's arguments are evaluated twice, so they must
// have no side effects. The macro does not check the launch's result.
// The idea is public: deciding at launch, from the alignment of every input and output pointer and
// from the length's divisibility by the vector width, between a wide-load kernel, a wide-load
// kernel that guards its edges, and a single-element kernel is done in Apache MXNet's
// cuda_vectorization.cuh (CheckAlignment, VectorizedKernelLauncher; Apache-2.0, 2020); a 4 / 2 / 1
// ladder, the minimum over the output pointer and every input pointer, is PyTorch's
// can_vectorize_up_to (2020), which tests alignment only. Here no kernel guards its edges, and a
// width of two is not used.
#define LAUNCH_BY_FOUR(kernel, T, items, s, ...)                                                                       \
  do {                                                                                                                 \
    if (by_four(T, __VA_ARGS__))                                                                                       \
      kernel<4><<<grid_blocks((items) / 4), 256, 0, s>>>(__VA_ARGS__);                                                        \
    else                                                                                                               \
      kernel<1><<<grid_blocks(items), 256, 0, s>>>(__VA_ARGS__);                                                              \
  } while (0)
