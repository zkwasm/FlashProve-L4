// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Unit test of hz::Tree::slot_pads_agree() (prover/hiding/trees.cuh), the prover's self-check of a
// sub-proof's slot pads: a one-time pad, an element of the extension field E, for each scalar
// ("slot") that the sub-proof sends masked. A pad's hz::CE (three) components are 64-bit words
// ("cells") of one stream of the generator of private randomness (`coin` in identifiers);
// slot_pads() draws each component into its own array ("plane"). slot_pads_agree() derives the
// cells again and adds, to a counter in GPU memory (`word`), how many cells of a copy of the pads
// match. Run by prover/tests/coin_gpu_tests.sh on one GPU. The test uses the public test key
// (hz::key_test()) and one blinding context (hz::set_attempt()). Not covered: the sub-proofs'
// kernels that copy the pads; finish_request() (prover/prover/request.cuh), which compares the
// count; how the counter is kept with a request's state.

#include "hiding/pt_field.h"
#include "common/hash.h"
#include "common/hash_tree.cuh"
#include "hiding/trees.cuh"
#include "tests/longrow/made_key.h"
#include <cstdio>
#include <vector>
static int failed = 0;
static unsigned long long *word = nullptr; // on the GPU
static void expect(bool good, size_t ns, const char *what, unsigned long long counted, unsigned long long wanted) {
  if (!good)
    printf("%zu slots: %s: %llu cells are counted, %llu are wanted: FAILS\n", ns, what, counted, wanted);
  failed += !good;
}
// Returns what slot_pads_agree() adds to the counter for `laid`: ns pads of hz::CE words each.
static unsigned long long counted(const hz::Tree &S, const std::vector<u64> &laid, size_t ns, hz::T *gpu) {
  unsigned long long before = 0, after = 0;
  CK(cudaMemcpy(&before, word, 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gpu, laid.data(), 8 * laid.size(), cudaMemcpyHostToDevice));
  S.slot_pads_agree(gpu, ns);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&after, word, 8, cudaMemcpyDeviceToHost));
  return after - before;
}
// A tree with only what slot_pads() and slot_pads_agree() use: the name, the stream, zeroed planes.
static hz::Tree pads_alone(const char *name, u32 stream, size_t ns) {
  hz::Tree S;
  S.nm = name, S.stream = stream, S.sh.Kc = ns + 5; // a plane is longer than the slots, as in the sub-proofs
  CK(cudaMalloc(&S.ext, 8 * hz::CE * S.sh.Kc));
  CK(cudaMemset(S.ext, 0, 8 * hz::CE * S.sh.Kc));
  return S;
}
// The pads of S as ns elements of E, gathered from its planes.
static std::vector<u64> as_elements(const hz::Tree &S, size_t ns) {
  std::vector<u64> planes(hz::CE * S.sh.Kc), laid(hz::CE * ns);
  CK(cudaMemcpy(planes.data(), S.ext, 8 * planes.size(), cudaMemcpyDeviceToHost));
  for (size_t j = 0; j < ns; j++)
    for (int kappa = 0; kappa < hz::CE; kappa++) laid[hz::CE * j + kappa] = planes[kappa * S.sh.Kc + j];
  return laid;
}
// The cases for ns slots (ns >= 1: not checked), with the counts required:
//   - pads as drawn: every cell. Each must equal second::cell() of prover/tests/longrow/made_key.h,
//     a second host implementation of the generator; both ChaCha20 block functions are adapted
//     from the public-domain file chacha-merged.c. That checks keys, streams and cell numbers, not
//     a fault that both block functions share: prover/tests/coins_rfc8439_test.cpp compares, on the
//     host, the generator's chacha20_block() (prover/common/coin_chacha.h) with RFC 8439's vectors;
//   - one bit changed in one cell: one cell fewer; one component zero: a third fewer;
//   - two components exchanged in every pad: a third; all pads zero, pads rotated by one slot (if
//     there are several), another tree's pads: none. In these four, two more may match by chance;
//   - another tree's pads, checked against that tree: every cell.
static void play(const u32 context_key[8], u32 stream, size_t ns) {
  const unsigned long long all = (unsigned long long)hz::CE * ns;
  hz::Tree S = pads_alone("S", stream, ns), other = pads_alone("other", stream + 4, ns);
  S.slot_pads(ns);
  other.slot_pads(ns);
  const std::vector<u64> drawn = as_elements(S, ns);
  hz::T *gpu;
  CK(cudaMalloc(&gpu, 8 * drawn.size()));
  size_t off = 0;
  for (size_t j = 0; j < ns; j++)
    for (int kappa = 0; kappa < hz::CE; kappa++)
      off += drawn[hz::CE * j + kappa] != second::cell(context_key, stream + 1, (u64)kappa * ns + j);
  expect(!off, ns, "the pads as drawn, compared with the host code of tests/longrow/made_key.h", all - off, all);
  unsigned long long n = counted(S, drawn, ns, gpu);
  expect(n == all, ns, "the pads as drawn", n, all);
  for (size_t j : {(size_t)0, ns / 2, ns - 1})
    for (int kappa = 0; kappa < hz::CE; kappa++) {
      std::vector<u64> laid = drawn;
      laid[hz::CE * j + kappa] ^= 1;
      n = counted(S, laid, ns, gpu);
      expect(n == all - 1, ns, "one cell changed by a bit", n, all - 1);
    }
  for (int kappa = 0; kappa < hz::CE; kappa++) {
    std::vector<u64> laid = drawn;
    for (size_t j = 0; j < ns; j++) laid[hz::CE * j + kappa] = 0;
    n = counted(S, laid, ns, gpu);
    expect(n == all - ns, ns, "one component zero", n, all - ns);
  }
  n = counted(S, std::vector<u64>(drawn.size(), 0), ns, gpu);
  expect(n <= 2, ns, "all pads zero", n, 0);
  {
    std::vector<u64> laid(drawn.begin() + hz::CE, drawn.end());
    laid.insert(laid.end(), drawn.begin(), drawn.begin() + hz::CE);
    n = counted(S, laid, ns, gpu);
    expect(ns == 1 ? n == all : n <= 2, ns, "the pads moved on by one slot", n, ns == 1 ? all : 0);
  }
  {
    std::vector<u64> laid = drawn;
    for (size_t j = 0; j < ns; j++) std::swap(laid[hz::CE * j], laid[hz::CE * j + 1]);
    n = counted(S, laid, ns, gpu);
    expect(n >= ns && n <= ns + 2, ns, "two components exchanged", n, ns);
  }
  // Control for the next case, which would pass too if the other tree's pads were all zero.
  n = counted(other, as_elements(other, ns), ns, gpu);
  expect(n == all, ns, "another tree's pads, checked against that tree", n, all);
  n = counted(S, as_elements(other, ns), ns, gpu);
  expect(n <= 2, ns, "another tree's pads", n, 0);
  CK(cudaFree(gpu));
  CK(cudaFree(S.ext));
  CK(cudaFree(other.ext));
}
int main() {
  int count = 0;
  const cudaError_t count_error = cudaGetDeviceCount(&count);
  // cudaErrorInsufficientDriver is also what a machine without any NVIDIA driver returns.
  if (count_error == cudaErrorNoDevice || count_error == cudaErrorInsufficientDriver ||
      (count_error == cudaSuccess && !count)) {
    printf("slot_pads_agree_gpu_test: not run: this machine has no GPU\n");
    return 2;
  }
  CK(count_error); // any other error is a failure, not a missing GPU
  hz::key_test();
  lr::Key k; // the public test key; lr::Key's nonce is zero
  for (int i = 0; i < COINKEYW; i++) k.key[i] = (u32)(mix64(hz::TEST_KEY + i) >> 16);
  lr::Params p;
  p.working_word = hz::COINS_WORD;
  k.ctr = 3, k.obj = 6, k.mem = 0;
  hz::set_attempt(k.ctr, k.obj, k.mem);
  u32 context_key[8];
  second::context_key(context_key, k, p);
  CK(cudaMalloc(&word, 8));
  const unsigned long long held = 1000; // the count is added to what the word holds
  CK(cudaMemcpy(word, &held, 8, cudaMemcpyHostToDevice));
  hz::slot_pads_agreeing = word;
  u32 stream = hz::ATTEMPT_STREAMS_FROM;
  // Unless four divides ns, a component begins inside a four-cell block of the generator.
  const size_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 255, 256, 257, 1021, 1022, 1023, 1024, 1025, 4099, 70001};
  for (size_t ns : sizes) {
    play(context_key, stream, ns);
    stream += 8;
  }
  hz::end_attempt();
  hz::key32_wipe();
  if (failed)
    printf("slot_pads_agree_gpu_test: %d lines FAIL\n", failed);
  else
    printf("slot_pads_agree_gpu_test: all hold at %zu numbers of slots\n", sizeof sizes / sizeof sizes[0]);
  return failed ? 1 : 0;
}
