// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Tests hz::fill() of prover/hiding/trees.cuh, which fills a GPU buffer with private random values
// (`coin` in identifiers; cell i is value i of a stream), against a second host implementation of
// the generator: namespace `second` of prover/tests/longrow/made_key.h, whose ChaCha20 block
// function (RFC 8439, 2.3) is adapted from D. J. Bernstein's chacha-merged.c, as
// coin_chacha.h's is. The comparison covers keys, labels (the function's other input words),
// the choice of output words and retries, not a fault that both block functions share. The
// RFC's vectors test the programs' chacha20_block() in prover/tests/coins_rfc8439_test.cpp (host)
// and in coins_own_test() of prover/hiding/coins_own.cuh (host and kernel). Under the public test
// key the test draws in two contexts (`attempt` in identifiers; nine label words each) and in
// "a committer's stream", whose nine words are 0. That covers coink(), which derives a context
// key on the host, k_fill(), draws from inside a block (one generator call: four cells) and the
// version word hz::COINS_WORD. Not covered: a nonzero nonce: coink() leaves it 0.
//   coin_fill_gpu_test                  one GPU
//   coin_fill_gpu_test refused-launch   the driver rejects a launch inside hz::fill(): the call
//     must end the program with exit code 1, from CK() of prover/common/common.h, and one line that
//     names `hiding/trees.cuh` (prover/tests/coin_gpu_tests.sh checks both); a return gives 4.
#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "tests/longrow/made_key.h"
#include <cstdio>
#include <cstring>
#include <vector>
static int failed = 0;
// Draws cells first .. first + count - 1 of a stream with hz::fill() and compares them with
// second::cell(). Not checked: k and p must hold hz's key and the nine words of hz's current
// context: counter (2 words), nonce (4), version word, object, member (all 0 for a stream below
// hz::ATTEMPT_STREAMS_FROM).
// `redrawn`: how many of these cells have, by the host code, a first try that is not below the prime
// p, so that coin4() must take them from hz::coin_again(). Any other count fails the test, so a
// change of the test key or of a label cannot silently remove that case.
static void play(const char *what, const lr::Key &k, const lr::Params &p, u32 stream, u64 first, size_t count,
                 size_t redrawn = 0) {
  u64 *gpu;
  CK(cudaMalloc(&gpu, 8 * count));
  hz::fill(gpu, stream, first, count);
  std::vector<u64> got(count);
  CK(cudaMemcpy(got.data(), gpu, 8 * count, cudaMemcpyDeviceToHost));
  CK(cudaFree(gpu));
  u32 context_key[8];
  second::context_key(context_key, k, p);
  size_t off = 0, not_below_p = 0;
  for (size_t i = 0; i < count; i++) off += got[i] != second::cell(context_key, stream, first + i);
  for (size_t i = 0; i < count; i++) not_below_p += second::a_try(context_key, stream, first + i, 0) >= lr::P;
  const bool bad = off || not_below_p != redrawn;
  printf("%s: %zu cells from cell %llu of stream %u, compared with the host code of tests/longrow/made_key.h: "
         "%zu differ; cells whose first try is not below p: %zu (expected: %zu): %s\n",
         what, count, (unsigned long long)first, stream, off, not_below_p, redrawn, bad ? "FAILS" : "ok");
  failed += bad;
}
// out[x] = what hz::coin_again() gives for cell x % 4 of generator call bk + x / 4 of the stream.
__global__ void k_again(const __grid_constant__ hz::CoinK ck, u32 stream, u64 bk, u64 *out) {
  out[threadIdx.x] = hz::coin_again(ck, stream, bk + threadIdx.x / 4, (int)(threadIdx.x & 3));
}
// Tests hz::coin_again(), which redraws a cell whose first try is not below the prime p. Draws
// rarely reach it (about one in 2^32 for p = 2^64 - 2^32 + 1), so k_again() calls it directly, for
// 64 cells from generator call bk on. Each must equal the host's first try from 1 on that is below
// p, and none may equal try 0's value. A retry is rejected just as rarely, so main() also passes a
// call with a cell whose try 1 is not below p: there a coin_again() that always returned try 1, or
// that rejected the word p only, fails. `rejected`: how many tries from 1 on the host code must skip
// in the 64 cells; any other count fails the test. Does not test when coin4() calls coin_again():
// `play` with a nonzero `redrawn` does.
static void again(const char *what, const lr::Key &k, const lr::Params &p, u32 stream, u64 bk, int rejected = 0) {
  const int cells = 64;
  hz::CoinK ck;
  second::context_key(ck.k, k, p);
  u64 *gpu, got[cells];
  CK(cudaMalloc(&gpu, sizeof got));
  k_again<<<1, cells>>>(ck, stream, bk, gpu);
  CK(cudaGetLastError());
  CK(cudaMemcpy(got, gpu, sizeof got, cudaMemcpyDeviceToHost));
  CK(cudaFree(gpu));
  int off = 0, as_the_first_try = 0, skipped = 0;
  for (int x = 0; x < cells; x++) {
    u32 tr = 1;
    while (second::a_try(ck.k, stream, 4 * bk + x, tr) >= lr::P) tr++;
    skipped += (int)tr - 1;
    off += got[x] != second::a_try(ck.k, stream, 4 * bk + x, tr);
    as_the_first_try += got[x] == second::a_try(ck.k, stream, 4 * bk + x, 0);
  }
  const bool bad = off || as_the_first_try || skipped != rejected;
  printf("%s: coin_again() for %d cells from call %llu of stream %u, compared with the host code's next try: %d differ; %d equal "
         "the first try's value; tries from 1 on that are not below p: %d (expected: %d): %s\n",
         what, cells, (unsigned long long)bk, stream, off, as_the_first_try, skipped, rejected, bad ? "FAILS" : "ok");
  failed += bad;
}
int main(int argc, char **argv) {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coin_fill_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  hz::key_test();
  if (argc == 2 && !strcmp(argv[1], "refused-launch")) {
    hz::set_attempt(1, 6, 0);
    // 2^42 cells need 2^32 thread blocks, which is 0 in 32 bits. The driver rejects such a grid.
    // If hz::fill() went on silently, the buffer would stay as it was, in place of random values.
    hz::fill(nullptr, hz::ATTEMPT_STREAMS_FROM, 0, (size_t)1 << 42);
    printf("coin_fill_gpu_test: a draw of coins whose launch the driver rejected has returned: FAILS\n");
    return 4;
  }
  lr::Key k; // the public test key; lr::Key's nonce is zero
  for (int i = 0; i < COINKEYW; i++) k.key[i] = (u32)(mix64(hz::TEST_KEY + i) >> 16);
  lr::Params p;
  // second::cell() takes five cells from each block of a stream whose bits 14 and 15 hold 2, and
  // hz::fill() always takes four, so the streams here have other values in those bits.
  // Stream 5 is below hz::ATTEMPT_STREAMS_FROM: coink() allows that under the test key only.
  k.ctr = 0, k.obj = 0, k.mem = 0, p.working_word = 0;
  play("a committer's stream", k, p, 5, 0, 1000);
  p.working_word = hz::COINS_WORD;
  k.ctr = 3, k.obj = 6, k.mem = 0;
  hz::set_attempt(k.ctr, k.obj, k.mem);
  play("counter 3, object 6", k, p, hz::ATTEMPT_STREAMS_FROM, 0, 4097);
  play("the same context, from inside a block", k, p, hz::ATTEMPT_STREAMS_FROM + 15360, 6, 333);
  hz::end_attempt();
  k.ctr = ((u64)1 << 33) + 9, k.obj = 7, k.mem = 23;
  hz::set_attempt(k.ctr, k.obj, k.mem);
  play("counter 2^33 + 9, object 7, member 23", k, p, hz::ATTEMPT_STREAMS_FROM, 0, 4097);
  play("the same context, cells above 2^34", k, p, hz::ATTEMPT_STREAMS_FROM + 1, ((u64)1 << 34) + 2, 64);
  // The two cell numbers below depend on the test key and on this context's labels. If one of these
  // changes, find new cells by a search with second::a_try() (about 2^32 cells for each).
  // The first try of cell 1383388835 is 0xfffffffffedfb580, which is not below p; its try 1 is.
  play("the same context, a cell whose first try is not below p", k, p, hz::ATTEMPT_STREAMS_FROM, 1383388800, 64, 1);
  hz::end_attempt();
  again("counter 2^33 + 9, object 7, member 23", k, p, hz::ATTEMPT_STREAMS_FROM, 0);
  again("the same context, calls above 2^32", k, p, hz::ATTEMPT_STREAMS_FROM + 1, ((u64)1 << 32) + 5);
  // Try 1 of cell 6400819091, the last cell of call 1600204772, is 0xffffffff8753c954, which is not
  // below p; its try 2 is.
  again("the same context, a cell whose try 1 is not below p", k, p, hz::ATTEMPT_STREAMS_FROM, 1600204772, 1);
  hz::key32_wipe();
  printf(failed ? "coin_fill_gpu_test: %d lines FAIL\n" : "coin_fill_gpu_test: all lines ok\n", failed);
  return failed ? 1 : 0;
}
