// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Helpers shared by the GPU programs: CK(x), which if x is a CUDA error prints it and calls exit(1)
// (prover/weights_commitment/common.cuh redefines CK); device information; a 64-bit mixer;
// arithmetic in the base field F of p = 2^64 - 2^32 + 1 elements and in its cubic extension E.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "common/end_name.h"
#include "common/setting.h"
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
typedef unsigned __int128 u128;
#define CK(x)                                                                                                          \
  do {                                                                                                                 \
    cudaError_t e_ = (x);                                                                                              \
    if (e_ != cudaSuccess) {                                                                                           \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                                  \
      name_the_end("cuda-call");                                                                                       \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)
// One SplitMix64 step (increment, then output mix): a pseudo-random word from z. Not cryptographic.
__host__ __device__ static inline u64 mix64(u64 z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static int g_sms = 0;
// Prints one line about device 0 and sets g_sms, which is 0 until then, to that device's SM count.
static void dev_info() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  g_sms = p.multiProcessorCount;
  int clk = 0;
  cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
  printf("# device %s, %d SMs, max SM clock %.0f MHz, %.1f GB\n", p.name, g_sms, clk / 1000.0, p.totalGlobalMem / 1e9);
}
// cp_prove and cp_commit each need a GPU of their own. Each compares the GPU memory in use with
// this limit, in bytes, once, at its start: Prover::read_inputs() (prover/prover/inputs.cuh) and
// cp_commit_main() (prover/hash_commitment/hf_commit.cu). Above it another process is assumed to
// use the GPU, and the run is rejected unless it is a test run (docs/limits-all-rows.tsv, row
// P3.4.1). The limit is generous, because the process's own CUDA context counts too and may be
// larger on another driver or GPU. By then no GPU memory is sized by the model: cp_prove has only
// ChainHost::init()'s ROOTW words, cp_commit none.
constexpr size_t GPU_IN_USE_AT_THE_START = (size_t)2 << 30;
// The gf_ functions work modulo p (FIELD_P): they wrap the field_ functions of
// prover/field/field_names.h. gf_times_nonresidue() multiplies by FIELD_W = 2.

__host__ __device__ static inline u64 gf_add(u64 a, u64 b) {
  return field_add(a, b);
}
__host__ __device__ static inline u64 gf_sub(u64 a, u64 b) {
  return field_sub(a, b);
}
// Every gf_ function in this file needs operands below p; none checks this.
__host__ __device__ static inline u64 gf_mul(u64 a, u64 b) {
  return field_mul(a, b);
}
// A K3 is an element k0 + k1 X + k2 X^2 of E = F[X]/(X^3 - FIELD_W); Ext, below, is the same type.
// FIELD_W = 2 is not a cube in F, so E is a field; it has about 2^192 elements.
// Because of the three-argument constructor, K3{a} and K3{a, b} do not compile. K3{} is zero;
// `K3 k;` leaves the coordinates uninitialised.
struct K3 {
  u64 k0, k1, k2;
  K3() = default;
  __host__ __device__ constexpr K3(u64 c0, u64 c1, u64 c2) : k0(c0), k1(c1), k2(c2) {}
};
__host__ __device__ static inline u64 gf_times_w(u64 v) { // FIELD_W v
  return field_times_w(v);
}
__host__ __device__ static inline u64 gf_times_nonresidue(u64 v) { // alias of `gf_times_w`: FIELD_W v
  return field_times_w(v);
}
__host__ __device__ static inline K3 k3_add(K3 x, K3 y) {
  return {gf_add(x.k0, y.k0), gf_add(x.k1, y.k1), gf_add(x.k2, y.k2)};
}
// Schoolbook product modulo X^3 - FIELD_W.
__host__ __device__ static inline K3 k3_mul(K3 x, K3 y) {
  u64 r0 = gf_add(gf_mul(x.k0, y.k0), gf_times_w(gf_add(gf_mul(x.k1, y.k2), gf_mul(x.k2, y.k1))));
  u64 r1 = gf_add(gf_add(gf_mul(x.k0, y.k1), gf_mul(x.k1, y.k0)), gf_times_w(gf_mul(x.k2, y.k2)));
  u64 r2 = gf_add(gf_add(gf_mul(x.k0, y.k2), gf_mul(x.k2, y.k0)), gf_mul(x.k1, y.k1));
  return {r0, r1, r2};
}
// Ext is the type for elements of E, such as challenges and the values computed from them: EDEG = 3
// coordinates (prover/common/setting.h) of 8 bytes, each below p, in the order k0, k1, k2. A proof
// stores an element in the same layout, so sizeof(Ext) is also its size in a proof. Code that does
// not implement field arithmetic must use coord() and set_coord(), not the members' names.
typedef K3 Ext;
__host__ __device__ static inline u64 coord(const Ext &x, int i) {
  return i == 0 ? x.k0 : i == 1 ? x.k1 : x.k2;
}
__host__ __device__ static inline void set_coord(Ext &x, int i, u64 v) {
  if (i == 0)
    x.k0 = v;
  else if (i == 1)
    x.k1 = v;
  else
    x.k2 = v;
}
// The element with coordinates c[0], .., c[EDEG - 1], which must be below p (not checked).
__host__ __device__ static inline Ext ext_from(const u64 *c) {
  Ext x{};
  for (int i = 0; i < EDEG; i++) set_coord(x, i, c[i]);
  return x;
}
// v in F (below p) as an element of E: (v, 0, 0).
__host__ __device__ static inline Ext ext_base(u64 v) {
  Ext x{};
  set_coord(x, 0, v);
  return x;
}
// Use CK_ATTRIBUTE instead of cudaFuncSetAttribute(), with the same arguments. It checks the
// calling thread's CUDA status first: on the one CUDA version that was tried (13.0), a successful
// cudaFuncSetAttribute() resets that status, which would hide an earlier launch that the driver
// rejected (described in prover/docs/cuda-status.md, "A call that resets the status"). So the error
// that it prints may belong to an earlier launch or call. Both checks use the CK defined where
// CK_ATTRIBUTE is used, and print that file and line. Elsewhere in `cp_prove` and `cp_commit` only
// its counterpart ENGINE_CK_ATTRIBUTE (prover/engine/device.cuh) names cudaFuncSetAttribute(); a
// test outside this repository checks this, but not in verifier/src/cp_verify.cpp, part of both.
#define CK_ATTRIBUTE(...)                                                                                              \
  do {                                                                                                                 \
    CK(cudaGetLastError());                                                                                            \
    CK(cudaFuncSetAttribute(__VA_ARGS__));                                                                             \
  } while (0)
