// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests hz::key32_given() and hz::key32_renew() (prover/hiding/trees.cuh), which set and replace
// the process key: 32 bytes from which private random values ("coins") are derived. One GPU;
// without one, exit code 2.
//   coin_renew_gpu_test         Values drawn on the GPU equal the second host implementation's
//    (prover/tests/longrow/made_key.h). The same key and set_attempt() counter give equal values,
//    another key gives others. A renewal resets the counters, which otherwise must rise. 32 keys
//    with one non-zero byte, one per position, and a key whose bytes XOR to zero are accepted.
//   coin_renew_gpu_test <case>  A misuse: must exit with code 2 after one line "coins: .."
//    (prover/tests/coin_gpu_tests.sh checks both, not which message): renew-while-set,
//    renew-while-parked (an attempt is current or suspended), counter-not-above, given-twice,
//    given-zero, renew-zero, low-stream, coin-after-wipe (set_attempt() exits: the draw's check
//    (coink()) is not reached).
#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "tests/longrow/made_key.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static int failed = 0;
static void expect(bool held, const char *what) {
  printf("%s: %s\n", what, held ? "ok" : "FAILS");
  failed += !held;
}
static void bytes_of(uint8_t key[32], int number) {
  for (int i = 0; i < 32; i++) key[i] = (uint8_t)(mix64(1000 * number + i) >> 24);
}
static std::vector<u64> cells(size_t count) {
  u64 *gpu;
  CK(cudaMalloc(&gpu, 8 * count));
  hz::fill(gpu, hz::ATTEMPT_STREAMS_FROM + 2, 0, count);
  std::vector<u64> got(count);
  CK(cudaMemcpy(got.data(), gpu, 8 * count, cudaMemcpyDeviceToHost));
  CK(cudaFree(gpu));
  return got;
}
// Values (`cells`) 0 to 100 of one stream in the attempt (counter, object kind 6, batch slot 1).
static std::vector<u64> context(u64 counter) {
  hz::set_attempt(counter, 6, 1);
  std::vector<u64> got = cells(101);
  hz::end_attempt();
  return got;
}
static std::vector<u64> by_the_second(const uint8_t key[32], u64 counter) {
  lr::Key k; // its nonce is zero
  for (int i = 0; i < COINKEYW; i++) memcpy(&k.key[i], key + 4 * i, 4); // assumes little-endian
  k.ctr = counter, k.obj = 6, k.mem = 1;
  lr::Params p;
  p.working_word = hz::COINS_WORD;
  u32 context_key[8];
  second::context_key(context_key, k, p);
  std::vector<u64> want(101);
  for (size_t i = 0; i < want.size(); i++) want[i] = second::cell(context_key, hz::ATTEMPT_STREAMS_FROM + 2, i);
  return want;
}
int main(int argc, char **argv) {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coin_renew_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  uint8_t a[32], b[32];
  bytes_of(a, 1), bytes_of(b, 2);
  if (argc > 1) {
    const std::string scene = argv[1];
    const uint8_t zeros[32] = {0};
    // given-zero sets no other key first: a second key32_given() exits whatever key it gets.
    hz::key32_given(scene == "given-zero" ? zeros : a);
    if (scene == "renew-zero") hz::key32_renew(zeros);
    if (scene == "given-twice") hz::key32_given(b);
    if (scene == "renew-while-set") hz::set_attempt(1, 6, 1), hz::key32_renew(b);
    if (scene == "renew-while-parked") hz::set_attempt(1, 6, 1), hz::park_attempt(), hz::key32_renew(b);
    if (scene == "counter-not-above") context(1), context(1);
    if (scene == "coin-after-wipe") hz::key32_wipe(), context(1);
    if (scene == "low-stream") {
      u64 *gpu;
      CK(cudaMalloc(&gpu, 8 * 4));
      hz::set_attempt(1, 6, 1);
      hz::fill(gpu, hz::ATTEMPT_STREAMS_FROM - 1, 0, 4);
    }
    printf("coin_renew_gpu_test %s: FAILS: the program went on\n", scene.c_str());
    return 1;
  }
  hz::key32_given(a);
  expect(hz::key_state == hz::KEY_GIVEN, "after key32_given() the state is KEY_GIVEN");
  const std::vector<u64> a1 = context(1), a2 = context(2);
  expect(a1 == by_the_second(a, 1) && a2 == by_the_second(a, 2), "key a, counters 1 and 2: the second implementation's cells");
  hz::key32_renew(b);
  const std::vector<u64> b0 = context(0), b1 = context(1);
  expect(b0 == by_the_second(b, 0) && b1 == by_the_second(b, 1), "renewed to key b, counters 0 and 1: the same");
  expect(b1 != a1, "keys a and b differ at counter 1");
  hz::key32_wipe();
  hz::key32_renew(a);
  expect(context(1) == a1, "wiped, renewed to key a: counter 1 gives the first run's cells");
  hz::key32_wipe();
  int taken = 0; // `key32_renew` exits with code 2 on a key that it rejects
  for (int i = 0; i < 32; i++) {
    uint8_t one[32] = {0};
    one[i] = (uint8_t)(1 << i % 8);
    hz::key32_renew(one);
    taken += hz::key_state == hz::KEY_GIVEN;
  }
  expect(taken == 32, "32 keys with one non-zero byte (one bit), one for each of the 32 places, are taken");
  uint8_t pair[32] = {0};
  pair[0] = pair[31] = 0x80;
  hz::key32_renew(pair);
  expect(hz::key_state == hz::KEY_GIVEN, "a key whose bytes XOR to zero and add up to zero modulo 256 is taken");
  hz::key32_wipe();
  printf(failed ? "coin_renew_gpu_test: %d lines FAIL\n" : "coin_renew_gpu_test: all lines ok\n", failed);
  return failed ? 1 : 0;
}
