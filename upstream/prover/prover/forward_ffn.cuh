// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// `Dig` gives the lookup keys and cells (committed signed bytes) that write_digits()
// (prover/prover/forward.cuh) writes for a rounding's remainder. It serves k_moe_gu_out()
// (prover/prover/forward_moe.cuh), which has only the remainder's low 32 bits, r. They determine
// every byte written, even where an out-of-range quotient is set to 0 and the remainder, then the
// entire dividend, exceeds 32 bits.
// Include first: prover/common/field_host.h (u32).
#pragma once
#include "common/index_guard.cuh"
namespace ffn_moe_kernels {
typedef int8_t i8_;
typedef uint16_t u16_;
typedef long long i64_;
// k is the rounding's right shift, `form` its digit form (0, 1 or 2: see write_digits()). Not checked
// here: k <= 32 (24 in form 1). lo() and hi() are the 16-bit keys, c0() and c1() the cells as
// values 0 .. 255. write_digits() writes hi() only if hasHi(), one cell if one(), two if two().
struct Dig {
  int k, form;
  __device__ __forceinline__ bool two() const { return k > 16 ? form != 1 : form == 2; }
  __device__ __forceinline__ bool one() const { return form == 1; }
  __device__ __forceinline__ bool hasHi() const { return k > 16 || form == 1 || form == 2; }
  __device__ __forceinline__ u32 lo(u32 r) const { return k <= 16 ? (r << (16 - k)) & 65535u : r & 65535u; }
  __device__ __forceinline__ u32 hi(u32 r) const {
    if (k <= 16) return 0;
    const u32 h = r >> 16;
    return (form == 1 ? h << (24 - k) : h << (32 - k)) & 65535u;
  }
  __device__ __forceinline__ u32 c0(u32 r) const {
    if (k <= 16) return 0x80u;
    const u32 h = r >> 16;
    return (form == 1 ? h - 128u : h - 32896u) & 255u;
  }
  __device__ __forceinline__ u32 c1(u32 r) const {
    if (k <= 16) return 0x80u;
    const u32 v = (r >> 16) - 32896u;
    return ((v + 128u) >> 8) & 255u;
  }
};
} // namespace ffn_moe_kernels
