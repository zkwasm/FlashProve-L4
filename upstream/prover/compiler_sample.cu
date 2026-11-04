// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Checks the compiler that builds the GPU programs for one known bug: runs a small kernel on one
// plain input and 64 pseudo-random ones, compares the values with the CPU's, and prints one line,
// on standard error unless the exit code is 0. Before every target that runs nvcc, prover/build.sh
// compiles prover/compiler_sample.cu with its `nvcc_base` options and runs it: a wrong value stops
// the build; with no usable GPU it only warns. SUITE.md's test `gpu.compiler_sample` runs it too.
//
// The bug: k_small() has two local objects of 40 bytes, a P and a Q, in use at once. Their arrays
// are indexed by numbers read at run time, so the compiler keeps the objects in local memory, where
// each needs its own place. If both get the same place, the kernel silently computes wrong values.
// Compiled by nvcc 13.0.88 with the `nvcc_base` options, it has 40 bytes of local memory per thread
// and gave wrong values on an H100; by nvcc 13.2.86, it has 80 bytes, and all values were right
// (the size is printed but decides nothing). We found no mention of this bug in NVIDIA's release
// notes, so it is not known whether 13.2.86 repairs it or only compiles this kernel differently.
//
// Exit code 0: all values are equal. 1: a value differs or the kernel failed. 77: no GPU was found.
// 2: the GPU could not be used (e.g. no memory or no code for it), so not all values were compared.
//
// Exit code 0 shows nothing about other bugs or other kernels: no kernel of the prover runs here.

#include <cstdio>
#include <cuda_runtime.h>
// The exact source that gave the wrong values, up to "clang-format on": do not edit or reformat it.
// clang-format off
typedef unsigned long long u64; typedef unsigned u32;
struct alignas(8) P { u32 l[5], h[5]; };
#define FI __host__ __device__ __forceinline__
FI P prep(u64 x) { P p; 
#pragma unroll
  for (int i = 0; i < 5; i++) { p.l[i] = (u32)(x * (2 * i + 1)); p.h[i] = (u32)(x >> (i + 7)); } return p; }
FI u64 mulp(const P &x, u64 y, const int *ix) { u64 r = 0;
  for (int k = 0; k < 3; k++) r += (u64)x.l[ix[k]] * (u32)y + (u64)x.h[ix[k + 3]] * (u32)(y >> 32);
  return r; }
struct alignas(8) Q { u32 l[5], h[5]; };
FI Q prepq(u64 x) { Q p; 
#pragma unroll
  for (int i = 0; i < 5; i++) { p.l[i] = (u32)(x * (2 * i + 1)); p.h[i] = (u32)(x >> (i + 7)); } return p; }
FI u64 mulq(const Q &x, u64 y, const int *ix) { u64 r = 0;
  for (int k = 0; k < 3; k++) r += (u64)x.l[ix[k]] * (u32)y + (u64)x.h[ix[k + 3]] * (u32)(y >> 32);
  return r; }
FI u64 mul(u64 x, u64 y, const int *ix) { return mulq(prepq(x), y, ix); }
FI void chain(u64 &a, u64 x, u64 y, const int *ix) { a += mulp(prep(x), y, ix); }
__global__ void k_small(const u64 *s, const u64 *tab, u64 *out, const int *ix, int m, int n) {
  for (int j = 0; j < m; j++) {
    u64 acc = 0;
    for (int t = 0; t < n; t++) chain(acc, s[t], s[n + t], ix);
    out[2 + j] = mul(acc, s[j], ix);
  }
  const P w = prep(tab[4]);
  out[0] = mul(mulp(w, tab[0], ix), tab[1], ix);
  out[1] = mul(mulp(w, tab[2], ix), tab[3], ix);
}
// clang-format on
// The kernel's body once more, for the CPU, with the same functions.
static void small_on_the_cpu(const u64 *s, const u64 *tab, u64 *out, const int *ix, int m, int n) {
  for (int j = 0; j < m; j++) {
    u64 acc = 0;
    for (int t = 0; t < n; t++) chain(acc, s[t], s[n + t], ix);
    out[2 + j] = mul(acc, s[j], ix);
  }
  const P w = prep(tab[4]);
  out[0] = mul(mulp(w, tab[0], ix), tab[1], ix);
  out[1] = mul(mulp(w, tab[2], ix), tab[3], ix);
}

// One step of SplitMix64, from a fixed start (THIRD_PARTY.md, "SplitMix64"): the drawn inputs.
static u64 drawn() {
  static u64 state = 0x243f6a8885a308d3;
  u64 z = (state += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

int main() {
  constexpr int M = 3, N = 4, INPUTS = 1 + 64; // the largest m and n; the plain input has m = 2 and n = 3
  int gpus = 0;
  if (cudaGetDeviceCount(&gpus) != cudaSuccess || gpus == 0) {
    fprintf(stderr, "compiler_sample: no GPU was found, so nothing was compared\n");
    return 77;
  }
  u64 *d_s, *d_tab, *d_out;
  int *d_ix;
  cudaFuncAttributes kernel;
  cudaError_t error = cudaSuccess; // of the first call that failed
  auto good = [&error](cudaError_t e) { return (error = e) == cudaSuccess; };
  bool usable = good(cudaMalloc(&d_s, 2 * N * sizeof(u64))) && good(cudaMalloc(&d_tab, 5 * sizeof(u64))) && good(cudaMalloc(&d_out, (2 + M) * sizeof(u64))) &&
                good(cudaMalloc(&d_ix, 6 * sizeof(int))) && good(cudaFuncGetAttributes(&kernel, k_small));
  int values = 0, wrong = 0, first_input = 0, first_value = 0;
  u64 first_gpu = 0, first_cpu = 0;
  for (int k = 0; usable && k < INPUTS; k++) {
    u64 s[2 * N], tab[5];
    int ix[6] = {0, 2, 4, 1, 3, 0};
    const int m = k == 0 ? 2 : M, n = k == 0 ? 3 : N;
    if (k == 0) { // the plain input: each number is one byte, 1, 2, 3 and so on, eight times over
      u64 byte = 0;
      for (u64 &x : s) x = ++byte * 0x0101010101010101;
      for (u64 &x : tab) x = ++byte * 0x0101010101010101;
    } else {
      for (u64 &x : s) x = drawn();
      for (u64 &x : tab) x = drawn();
      for (int &x : ix) x = (int)(drawn() % 5); // an index into l[5] or h[5]
    }
    u64 cpu[2 + M], gpu[2 + M];
    small_on_the_cpu(s, tab, cpu, ix, m, n);
    usable = good(cudaMemcpy(d_s, s, sizeof s, cudaMemcpyHostToDevice)) && good(cudaMemcpy(d_tab, tab, sizeof tab, cudaMemcpyHostToDevice)) &&
             good(cudaMemcpy(d_ix, ix, sizeof ix, cudaMemcpyHostToDevice)) && good(cudaMemset(d_out, 0, sizeof gpu));
    if (!usable) break;
    k_small<<<1, 1>>>(d_s, d_tab, d_out, d_ix, m, n);
    usable = good(cudaGetLastError()); // an error of the launch itself
    if (!usable) break;
    if (!good(cudaMemcpy(gpu, d_out, sizeof gpu, cudaMemcpyDeviceToHost))) {
      fprintf(stderr, "compiler_sample: the kernel failed at input %d: %s\n", k, cudaGetErrorString(error));
      return 1;
    }
    for (int i = 0; i < 2 + m; i++, values++)
      if (gpu[i] != cpu[i] && wrong++ == 0) first_input = k, first_value = i, first_gpu = gpu[i], first_cpu = cpu[i];
  }
  if (wrong) { // reported first: what was compared before a call failed counts
    fprintf(stderr, "compiler_sample: WRONG VALUES: %d of %d differ; the first: input %d, value %d is %016llx on the GPU and %016llx on the CPU; the kernel has %zu bytes of local memory per thread\n",
            wrong, values, first_input, first_value, first_gpu, first_cpu, kernel.localSizeBytes);
    return 1;
  }
  if (!usable) {
    fprintf(stderr, "compiler_sample: the GPU could not be used (%s), so not all values were compared\n", cudaGetErrorString(error));
    return 2;
  }
  printf("compiler_sample: %d values are the same on the GPU and on the CPU; the kernel has %zu bytes of local memory per thread\n", values, kernel.localSizeBytes);
  return 0;
}
