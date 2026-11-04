// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Adds to prover/common/common.h: arithmetic mod p = 2^64 - 2^32 + 1 (FIELD_P) and in
// the cubic extension K3; loads and stores. An element's u64s must be below p (not checked here).
#pragma once
#include "common/common.h"
#include <cstring>
#define HD __host__ __device__ __forceinline__ // a function for host and GPU code, forced inline
HD u64 gf_pow(u64 a, u64 e) {
  u64 r = 1;
  while (e) {
    if (e & 1) r = gf_mul(r, a);
    a = gf_mul(a, a);
    e >>= 1;
  }
  return r;
}
// Returns y mod p in [0, p), for any y.
HD u64 gf_from_signed(long long y) {
  return field_from_signed(y);
}
struct FK { // traits of the field K3
  typedef K3 T;
  static constexpr const char *name = "K3";
  static HD T zero() { return {0, 0, 0}; }
  static HD T one() { return {1, 0, 0}; }
  static HD T add(T a, T b) { return k3_add(a, b); }
  static HD T sub(T a, T b) { return {gf_sub(a.k0, b.k0), gf_sub(a.k1, b.k1), gf_sub(a.k2, b.k2)}; }
  static HD T mul(T a, T b) { return k3_mul(a, b); }
  static HD T small(u64 v) { return {field_from_u64(v), 0, 0}; }
  static HD T muls(T a, u32 s) { return {gf_mul(a.k0, s), gf_mul(a.k1, s), gf_mul(a.k2, s)}; }
  typedef u64 B;
  static HD B fromi(long long y) { return gf_from_signed(y); }
  static HD T mulb(T a, B s) { return {gf_mul(a.k0, s), gf_mul(a.k1, s), gf_mul(a.k2, s)}; }
  static HD T emb(B b) { return {b, 0, 0}; }
  static HD bool eq(T a, T b) { return a.k0 == b.k0 && a.k1 == b.k1 && a.k2 == b.k2; }
  // 1/v = (A + Bq x + Cq x^2) / N, where x^3 = FIELD_W. With M the matrix of multiplication by v,
  // (A, Bq, Cq) is the first column of M's adjugate and N is det M (v's norm). 0 maps to 0.
  static HD T inv(T v) {
    u64 A = gf_sub(gf_mul(v.k0, v.k0), gf_times_w(gf_mul(v.k1, v.k2)));
    u64 Bq = gf_sub(gf_times_w(gf_mul(v.k2, v.k2)), gf_mul(v.k0, v.k1));
    u64 Cq = gf_sub(gf_mul(v.k1, v.k1), gf_mul(v.k0, v.k2));
    u64 N = gf_add(gf_mul(v.k0, A), gf_times_w(gf_add(gf_mul(v.k1, Cq), gf_mul(v.k2, Bq))));
    u64 ni = gf_pow(N, FIELD_P - 2);
    return {gf_mul(A, ni), gf_mul(Bq, ni), gf_mul(Cq, ni)};
  }
};
// ld16() reads and st16() writes one element. In GPU code a 16-byte element moves as one uint4, so
// its address must be a multiple of 16, which is not checked. Host code, and the specializations
// for K3 (24 bytes) in host and GPU code, access *p directly.
// Public sources for the techniques of this routine.
// Vectorized memory access. The idea is public: a 128-bit load or store through a vector type:
// "CUDA Pro Tip: Increase Performance with Vectorized Memory Access", NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
template <class T> HD T ld16(const T *p) {
  static_assert(sizeof(T) == 16, "an element of another size needs its own ld16");
#ifdef __CUDA_ARCH__
  uint4 v = *reinterpret_cast<const uint4 *>(p);
  T r;
  memcpy(&r, &v, 16);
  return r;
#else
  return *p;
#endif
}
// Public sources for the techniques of this routine.
// Vectorized memory access. The idea is public (cited at ld16()).
template <class T> HD void st16(T *p, T v) {
  static_assert(sizeof(T) == 16, "an element of another size needs its own st16");
#ifdef __CUDA_ARCH__
  uint4 w;
  memcpy(&w, &v, 16);
  *reinterpret_cast<uint4 *>(p) = w;
#else
  *p = v;
#endif
}
template <> HD K3 ld16<K3>(const K3 *p) {
  return *p;
}
template <> HD void st16<K3>(K3 *p, K3 v) {
  *p = v;
}
