// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks private random values (`coin` in identifiers) by value, which no verifier can do: the host
// derives them again from the same ChaCha20 key and compares them with the caller's copy of what
// the GPU wrote. A cell is one value of a numbered stream: a field element. Host and GPU share the
// key, the stream number and the source of coin_output() (prover/common/coin_chacha.h) and
// field_from_word() (prover/field/field_names.h): the comparison cannot notice an error in any of
// these. The programs cp_commit and cp_prove call only hf_cells_differ(), from their start-up
// self-test (hz::coins_own_test(), prover/hiding/coins_own.cuh), under public test keys: no cell
// that a proof or commitment uses is compared here. prover/tests/coins_by_value_gpu_test.cu tests
// all its functions. Host code only. Include this file after prover/hiding/trees.cuh.
#pragma once
#include "common/coin_chacha.h"
#include <algorithm>
#include <system_error>
#include <thread>
#include <vector>
namespace rf {
// Returns cell i of a stream under the key of a blinding context (say, one request's proof): the
// host's version of hz::coin4() and hz::coin_again() (prover/hiding/trees.cuh). The rule, rejection
// sampling, is defined in docs/format/COIN-KEYS.md, "2a. From a key to a random value".
// prover/tests/coins_field_rule_test.cu tests the retry.
static u64 hf_coin_on_the_host(const ::hz::CoinK &key, u32 stream, u64 i) {
  const u32 q = (u32)(i & 3);
  for (u32 attempt = 0;; attempt++) {
    u32 block[COINOUTW];
    coin_output(block, key.k, stream, i >> 2, attempt ? RETRYQ * attempt + q : 0u);
    const u32 at = attempt ? 0 : 2 * q;
    u64 cell;
    const bool taken = field_from_word(((u64)block[at + 1] << 32) | block[at], cell);
    explicit_bzero(block, sizeof block);
    if (taken) return cell;
  }
}
// Returns how many of came[0] to came[n - 1] differ from the stream's cells `first` to
// first + n - 1. It reads only its arguments.
static size_t hf_cells_differ(const ::hz::CoinK &key, u32 stream, u64 first, const u64 *came, size_t n) {
  size_t differ = 0;
  u32 block[COINOUTW];
  for (u64 i = first; i - first < n;) {
    coin_output(block, key.k, stream, i >> 2, 0);
    for (u32 q = (u32)(i & 3); q < 4 && i - first < n; q++, i++) {
      u64 cell;
      if (!field_from_word(((u64)block[2 * q + 1] << 32) | block[2 * q], cell))
        cell = hf_coin_on_the_host(key, stream, i); // drawn again; the chance is about 2^-32
      differ += cell != came[i - first];
    }
  }
  explicit_bzero(block, sizeof block);
  return differ;
}
constexpr size_t HF_COINS_THREADS = 8;               // the host threads that compare a long range
constexpr size_t HF_COINS_A_THREADS_LEAST = 1 << 14; // a shorter share is not worth a thread
// Returns the sum of part(from, to) over shares [from, to) that partition [0, n) and may run
// concurrently on host threads (joined before it returns). A share begins at a multiple of `unit`
// or is empty. `unit` must not be 0; part() must be thread-safe, must not throw and must not call
// the CUDA runtime.
template <class Part> static size_t hf_sum_on_threads(size_t n, size_t unit, Part part) {
  const size_t threads = std::max<size_t>(1, std::min(HF_COINS_THREADS, n / HF_COINS_A_THREADS_LEAST));
  const size_t share = ((n + threads - 1) / threads + unit - 1) / unit * unit;
  std::vector<size_t> sums(threads, 0);
  std::vector<std::thread> others;
  for (size_t t = 1; t < threads; t++) {
    auto a_share = [&, t] { sums[t] = part(std::min(n, t * share), std::min(n, (t + 1) * share)); };
    try {
      others.emplace_back(a_share);
    } catch (const std::system_error &) { // thread creation failed: run the share here
      a_share();
    }
  }
  sums[0] = part(0, std::min(n, share));
  for (std::thread &other : others) other.join();
  size_t sum = 0;
  for (size_t s : sums) sum += s;
  return sum;
}

static size_t hf_cells_differ_on_threads(const ::hz::CoinK &key, u32 stream, u64 first, const u64 *came, size_t n) {
  return hf_sum_on_threads(n, 4, [&](size_t from, size_t to) {
    return hf_cells_differ(key, stream, first + from, came + from, to - from);
  });
}
} // namespace rf
