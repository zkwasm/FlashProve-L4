// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs, outside the programs, two parts of [own-test-coins]: the self-test against known answers
// that the programs cp_prove and cp_commit run at start-up on the code that makes their private
// random values (`coin` in identifiers). cp::keys_own_test() covers the keys derived from a seed;
// hz::coins_own_test() covers the values generated under a key. The third part,
// blk::coins_own_test_card() (prover/engine/stream_coins.cuh), is not run here. This test also
// prints which answers differ, which the programs do not, and each part's time and GPU memory.
// The control at the end of main() shows that the digest comparison of coins_own_test() can fail,
// but not that a part reports a failure when an answer differs, nor that a program acts on the
// report. For these two parts, no test in this repository shows either. A test that is not part of
// this repository shows only the first, only for coins_own_test() and only for one fault: it
// zeroes a drawn value and expects coins_own_test() to report it.
// Exit codes. 0: both runs of each part and the control pass. 1: any of them fails; the rest still
// runs. Also 1, after "CUDA error .." and with no line "coins_own_gpu_test: ..": a CUDA call fails
// (CK(), prover/common/common.h). 2, after "coins_own_gpu_test: not run: ..": the machine has no
// GPU, and prover/tests/unit.sh reports 77 (cannot run). Also 2, after "coins: ..": a check inside
// a part fails and ends the process (key_refuse(), replay_refuse() and others in
// prover/hiding/trees.cuh).

#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "hiding/coins_long.cuh"
#include "common/cp_keys_own.h"
#include "hiding/coins_own.cuh"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// GPU memory in use by all processes: it describes this program only if nothing else uses the GPU.
static size_t gpu_bytes_in_use() {
  size_t free_bytes, all;
  CK(cudaMemGetInfo(&free_bytes, &all));
  return all - free_bytes;
}
// Runs part() twice and prints two lines; the second lists the answers that differ. part() returns
// them as bits: bit names.size() * x + a, below 60, is names[a] of test context x (a context: the
// numbers that select a derived key); bit 60 + a is last[a]. The second run is the one timed, so
// that loading the kernels is not counted.
template <class Part>
static bool play(const char *what, Part part, std::vector<const char *> names, std::vector<const char *> last) {
  const size_t before = gpu_bytes_in_use();
  size_t most = before;
  std::atomic<bool> done{false};
  std::thread sampler([&] { // first run only; sleeps 0.5 ms between samples: can miss a short peak
    while (!done) {
      most = std::max(most, gpu_bytes_in_use());
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  });
  u64 missed = part();
  done = true;
  sampler.join();
  const auto from = std::chrono::steady_clock::now();
  missed |= part();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
  printf("%s: %s; %.1f ms; at most %zu bytes of GPU memory more than before it\n", what, missed ? "FAILS" : "ok", ms, most - before);
  printf("missed, %s:", what);
  for (size_t a = 0; a < names.size(); a++) {
    std::string contexts;
    for (size_t x = 0; names.size() * x + a < 60; x++)
      if (missed >> (names.size() * x + a) & 1) contexts += (contexts.empty() ? "" : ",") + std::to_string(x);
    if (!contexts.empty()) printf(" %s:%s", names[a], contexts.c_str());
  }
  for (size_t a = 0; a < last.size(); a++)
    if (missed >> (60 + a) & 1) printf(" %s", last[a]);
  printf(missed ? "\n" : " nothing\n");
  return !missed;
}
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coins_own_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  CK(cudaFree(nullptr)); // creates the CUDA context, so that its memory is not counted below
  const size_t before = gpu_bytes_in_use();
  bool good = play("the keys from a seed", [] { return (u64)!cp::keys_own_test() << 60; }, {}, {"keys"});
  // The answers of coins_own_test(), in the order of its bits. For each of twelve test contexts:
  // its key; digests of cells (64-bit values) from Tree::slot_pads(), of streams' first cells and
  // of two runs from later cells; the CPU's recomputation of these cells, rf::hf_cells_differ().
  // Once: key32_word_of() of two keys, one long run's digest (drawn in context 6, whose "host"
  // also covers the run's first 16,384 cells), the generator's state afterwards, six RFC 8439
  // vectors. Not checked here: that the first list has hz::COINS_OWN_ANSWERS names.
  good &= play("the streams numbered from a base", [] { return hz::coins_own_test(); }, {"key", "pads", "starts", "runs", "host"},
               {"words", "stream", "state", "rfc8439"});
  // Printed always. SUITE.md's test of this program looks for this line and for "all hold" below.
  printf("the long rows' streams: not played: no file that includes prover/hiding/long_rows_words.h builds the row commitment that it defines\n");
  printf("GPU memory in use: %zu bytes before, %zu after\n", before, gpu_bytes_in_use());
  // The control. coins_known_digest_is() is the digest comparison of coins_own_test(); the cells
  // and the digest are this file's own, so the table of known answers is not involved.
  const u64 cells[3] = {1, 2, 3}, changed[3] = {1, 2, 3 ^ ((u64)1 << 60)};
  // BLAKE2b-512 of the 24 bytes of cells[], by python's hashlib: its first 32 bytes as eight words.
  const u32 want[8] = {0x2949baa7u, 0xa1ab18f2u, 0x20abb1f1u, 0x6957f03eu, 0x8bdc034bu, 0x63636057u, 0xc0934391u, 0x2b805a53u};
  const bool control = hz::coins_known_digest_is(cells, 3, want) && !hz::coins_known_digest_is(changed, 3, want);
  printf("control: the digest of three cells is hashlib's, and one changed bit is seen: %s\n", control ? "ok" : "FAILS");
  good &= control;
  printf("coins_own_gpu_test: %s\n", good ? "all hold" : "FAILS");
  return good ? 0 : 1;
}
