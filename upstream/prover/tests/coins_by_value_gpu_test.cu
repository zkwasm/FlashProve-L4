// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compares the private random values (`coins` in identifiers) that hz::fill() generates on the GPU
// (prover/hiding/trees.cuh) with the host's recomputation (prover/hiding/coins_by_value.cuh).
// A cell is one such value, a field element, in a numbered stream. The key is the public test key;
// hz::set_attempt()'s arguments are arbitrary.
//   1. Host and GPU agree in every cell of a range (printed: stretch). Ranges start and end at each
//      offset within a generator call (coin_output(): four cells); only the longest is split among
//      host threads.
//   2. Negative control: one changed cell (first, middle, last, around each boundary between
//      threads' shares) counts as exactly one difference. hz::coins_own_test(), a self-test of the
//      programs cp_commit and cp_prove, uses rf::hf_cells_differ(). No other test of this
//      repository gives that function a changed cell.
// Not covered: errors in hz::coink() (the key), coin_output() or field_from_word(), used by both
// sides; a cell drawn again because its first candidate is not below the prime
// p = 2^64 - 2^32 + 1 (probability about 2^-32; tested by prover/tests/coins_field_rule_test.cu).
// Needs a GPU; built and run by prover/tests/coin_gpu_tests.sh.
#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "hiding/coins_by_value.cuh"
#include <cstdio>
#include <vector>
static int failed = 0;
static void line(bool good, const char *what, size_t a = 0, size_t b = 0) {
  printf("%s (%zu, %zu): %s\n", what, a, b, good ? "ok" : "FAILS");
  failed += !good;
}
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coins_by_value_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  hz::key_test();
  hz::set_attempt(5, 6, 2);
  const u32 stream = hz::ATTEMPT_STREAMS_FROM + 40;
  hz::CoinK key = hz::coink(stream);
  u64 next = 0; // no cell is drawn twice
  // Parts 1 and 2 (hz::fill() exits if an attempt draws a cell twice)
  for (size_t n : {(size_t)1, (size_t)3, (size_t)4, (size_t)5, (size_t)1023, (size_t)4097, (size_t)300001}) {
    for (u64 shift = 0; shift < 4; shift++) {
      const u64 first = (next + 3) / 4 * 4 + shift;
      next = first + n;
      u64 *gpu;
      CK(cudaMalloc(&gpu, 8 * n));
      hz::fill(gpu, stream, first, n);
      std::vector<u64> got(n);
      CK(cudaMemcpy(got.data(), gpu, 8 * n, cudaMemcpyDeviceToHost));
      bool good = !rf::hf_cells_differ(key, stream, first, got.data(), n) &&
                  !rf::hf_cells_differ_on_threads(key, stream, first, got.data(), n);
      for (size_t i = 0; i < n; i++) good = good && rf::hf_coin_on_the_host(key, stream, first + i) == got[i];
      line(good, "a drawn stretch differs nowhere: first cell, cells", (size_t)first, n);
      std::vector<size_t> places = {0, n - 1, n / 2};
      // the cells before, at and after each share boundary (a copy of hf_sum_on_threads()'s
      // formula, unit 4)
      const size_t threads = std::max<size_t>(1, std::min(rf::HF_COINS_THREADS, n / rf::HF_COINS_A_THREADS_LEAST));
      const size_t share = ((n + threads - 1) / threads + 3) / 4 * 4;
      for (size_t t = 1; t < threads; t++)
        for (size_t beside = 0; beside < 3; beside++) places.push_back(std::min(n - 1, t * share + beside - 1));
      size_t missed = 0;
      for (size_t at : places) {
        got[at] ^= 1;
        missed += rf::hf_cells_differ(key, stream, first, got.data(), n) != 1;
        missed += rf::hf_cells_differ_on_threads(key, stream, first, got.data(), n) != 1;
        got[at] ^= 1;
      }
      line(!missed, "one changed cell counts as one: places tried, counts that were not 1", places.size(), missed);
      CK(cudaFree(gpu));
    }
  }
  hz::forget(key);
  hz::end_attempt();
  hz::key32_wipe();
  printf(failed ? "coins_by_value_gpu_test: %d lines FAIL\n" : "coins_by_value_gpu_test: all lines ok\n", failed);
  return failed ? 1 : 0;
}
