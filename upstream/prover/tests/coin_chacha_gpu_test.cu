// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks coin_output() (prover/common/coin_chacha.h), a generator of private randomness (`coin` in
// identifiers), in a kernel against the eight test vectors (key, label, output) of
// prover/tests/coin_chacha_vectors.h. A script that is not part of this repository prints that
// file: it implements the generator in Python from RFC 8439's text. coin_context_key() derives
// the kernel's key on the host, as in the programs.
// prover/tests/coin_fill_gpu_test.cu covers hz::coin_words() of prover/hiding/trees.cuh, which
// wraps coin_output(). That test reaches it via hz::fill(), which uses output words 0 to 7 only.
// No GPU: "not run" and exit code 2, read by prover/tests/unit.sh.
//   coin_chacha_gpu_test          one GPU; run by prover/tests/coin_gpu_tests.sh
#include "common/common.h"
#include "common/coin_chacha.h"
#include "tests/coin_chacha_vectors.h"
#include <cstdio>
#include <cstring>
struct ContextKey {
  u32 key[COINKEYW];
};
// `block` is coin_output()'s `call`, `label` its `retry`.
__global__ void k_plain(const __grid_constant__ ContextKey k, u32 stream, u64 block, u32 label, u32 *out) {
  u32 o[COINOUTW];
  coin_output(o, k.key, stream, block, label);
  for (int i = 0; i < COINOUTW; i++) out[i] = o[i];
}
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coin_chacha_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  u32 *gpu;
  CK(cudaMalloc(&gpu, 4 * COINOUTW));
  int differ = 0;
  for (const CoinChachaVector &v : COIN_CHACHA_VECTORS) {
    const u32 *m = v.label;
    u32 out[COINOUTW];
    ContextKey k;
    coin_context_key(k.key, v.key, m[4], m[5], m[10], m[11], m + 6, m[12]);
    k_plain<<<1, 1>>>(k, m[1], m[0] | (u64)m[3] << 32, m[2], gpu);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, gpu, sizeof out, cudaMemcpyDeviceToHost));
    differ += memcmp(out, v.out, sizeof out) != 0;
  }
  printf(differ ? "coin_chacha_gpu_test: %d of 8 outputs DIFFER from tests/coin_chacha_vectors.h\n"
                : "coin_chacha_gpu_test: all 8 outputs match tests/coin_chacha_vectors.h\n",
         differ);
  return differ ? 1 : 0;
}
