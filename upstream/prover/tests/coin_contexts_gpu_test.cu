// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Terms: docs/glossary.md.

// Checks on a GPU that hz::fill() and hz::fill_again() (prover/hiding/trees.cuh) derive private
// random values (`coin` in names) under the key of the current context while two contexts take
// turns. A context (`attempt` in identifiers) is nine 32-bit words, listed at `Context`.
// hz::coink() derives the context's key, K_ctx in prover/common/coin_chacha.h, from the process key
// (32 bytes, synthetic here) and these words, on the host, at every call. A stale K_ctx, or one
// shared by two contexts, would repeat one-time values in proofs that still verify, so no verifier
// would detect it.
//
// The values are cells: field elements, numbered within a stream. Every cell drawn is compared with
// a second host implementation of the ChaCha20-based generator (namespace `second` in
// prover/tests/longrow/made_key.h). Faults that the test catches: a K_ctx kept across set_attempt()
// or resume_attempt(); a context that resumes with a counter, object or member other than it began
// with. Other functions that call hz::coink() are not run here.
// Takes no arguments; needs one GPU. Exit code 0: all lines ok; 1: a line FAILS or a CUDA call
// failed; 2: no GPU, or a check inside an hz:: function called exit().
// prover/tests/coin_gpu_tests.sh builds and runs it.
#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "hiding/coins_long.cuh"
#include "tests/longrow/made_key.h"
#include <cstdio>
#include <vector>
static int failed = 0;
static void line(bool good, const char *context, const char *what) {
  printf("context %s: %s: %s\n", context, what, good ? "ok" : "FAILS");
  failed += !good;
}
// Set by begin(), in the host reference's types (namespace `lr` in prover/longrow/). k: process
// key, counter (2 words), nonce (4), kind of object, `member` (a request's batch slot).
// p.working_word: the format's version word.
struct Context {
  const char *name;
  lr::Key k;
  lr::Params p;
};
// Begins a context and records its words. The nonce stays zero, lr::Key's default, as always in
// hz::coink().
static Context begin(const char *name, const lr::Key &process_key, u64 counter, u32 object, u32 member) {
  hz::set_attempt(counter, object, member);
  Context c{name, process_key, {}};
  c.k.ctr = counter, c.k.obj = object, c.k.mem = member, c.p.working_word = hz::COINS_WORD;
  return c;
}
enum Road { FILL, FILL_AGAIN };
// Draws cells first to first + count - 1 of a stream in the current context, which should be c,
// with the function that `road` names, compares them with the second implementation's under c's
// words, and returns them.
static std::vector<u64> draw(const Context &c, Road road, u32 stream, u64 first, size_t count) {
  u64 *gpu;
  CK(cudaMalloc(&gpu, 8 * count));
  CK(cudaMemset(gpu, 0xFF, 8 * count)); // no cell has this value, so what an earlier draw left here cannot pass
  if (road == FILL) {
    hz::fill(gpu, stream, first, count);
  } else {
    hz::fill_again(gpu, stream, first, count);
  }
  std::vector<u64> got(count);
  CK(cudaMemcpy(got.data(), gpu, 8 * count, cudaMemcpyDeviceToHost));
  CK(cudaFree(gpu));
  u32 context_key[8];
  second::context_key(context_key, c.k, c.p);
  size_t off = 0;
  for (size_t i = 0; i < count; i++) off += got[i] != second::cell(context_key, stream, first + i);
  printf("context %s, %s: %zu cells from cell %llu of stream %u equal the second implementation's (%zu differ): %s\n",
         c.name,
         road == FILL ? "hz::fill()" : "hz::fill_again()", count, (unsigned long long)first, stream, off, off ? "FAILS" : "ok");
  failed += off != 0;
  return got;
}
// second::cell() takes five cells from a block of generator output if bits 14 and 15 of the stream
// number are 2, hz::fill() always four, so here they are 1. Draws from INSIDE start mid-block.
static const u32 PLAIN = hz::ATTEMPT_STREAMS_FROM, INSIDE = hz::ATTEMPT_STREAMS_FROM + 15360;
struct Cells {
  std::vector<u64> plain, inside;
};
// A context's first turn: the same streams and cells in A and in B.
static Cells first_turn(const Context &c) {
  return {draw(c, FILL, PLAIN, 0, 1000), draw(c, FILL, INSIDE, 6, 333)};
}
// A context's second turn, after hz::resume_attempt(): new cells with hz::fill(), which exits if a
// cell was drawn before in this context, and the first turn's cells again with hz::fill_again().
static void second_turn(const Context &c, const Cells &before) {
  draw(c, FILL, PLAIN, 1000, 500);
  draw(c, FILL, INSIDE, 339, 101);
  const Cells again{draw(c, FILL_AGAIN, PLAIN, 0, 1000), draw(c, FILL_AGAIN, INSIDE, 6, 333)};
  line(again.plain == before.plain && again.inside == before.inside, c.name,
       "the cells derived again equal those of its first turn");
}
// b must be at least as long as a; this is not checked.
static size_t equal_cells(const std::vector<u64> &a, const std::vector<u64> &b) {
  size_t equal = 0;
  for (size_t i = 0; i < a.size(); i++) equal += a[i] == b[i];
  return equal;
}
// The turns go A, B, A, B; hz::park_attempt() suspends the current context.
// At the end A is resumed once more only to end it: hz::end_attempt() acts on the current context.
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("coin_contexts_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  lr::Key key; // eight synthetic words
  uint8_t bytes[4 * COINKEYW];
  for (int i = 0; i < COINKEYW; i++) {
    key.key[i] = (u32)mix64(20261012 + i);
    for (int b = 0; b < 4; b++) bytes[4 * i + b] = (uint8_t)(key.key[i] >> 8 * b);
  }
  hz::key32_given(bytes);

  const size_t apart = second::differs_from_the_reference();
  printf("the second implementation equals the host reference's cells and context keys (%zu differ): %s\n", apart,
         apart ? "FAILS" : "ok");
  failed += apart != 0;

  const Context A = begin("A", key, 1, hz::lng::REQUEST, 5);
  const Cells a = first_turn(A);
  hz::park_attempt();
  const Context B = begin("B", key, 2, hz::lng::FILLER, 6);
  const Cells b = first_turn(B);
  line(!equal_cells(a.plain, b.plain) && !equal_cells(a.inside, b.inside), "B",
       "none of its cells equals A's cell of the same stream and number");
  hz::park_attempt();
  hz::resume_attempt(1);
  second_turn(A, a);
  hz::park_attempt();
  hz::resume_attempt(2);
  second_turn(B, b);
  hz::end_attempt();
  hz::resume_attempt(1);
  hz::end_attempt();
  hz::key32_wipe();
  printf(failed ? "coin_contexts_gpu_test: %d lines FAIL\n" : "coin_contexts_gpu_test: all lines ok\n", failed);
  return failed ? 1 : 0;
}
