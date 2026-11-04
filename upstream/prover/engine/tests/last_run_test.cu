// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compares the GPU and CPU implementations of `Engine` (a prover's heavy steps:
// blocks/engine/engine.h) on whole toy proofs over synthetic (printed: `made-up`) data. The cases
// of blocks/iop/tests/iopp_cases.h send a group's oracle messages (rows under Merkle roots; SEND),
// make three linear claims on them and run the proximity test (batching, folds, code switches,
// base case). Those of blocks/iop/tests/req_cases.h run a request's segments (`link` in
// identifiers), some with kept rows (read by later segments), close each, then run that test.
// For every case and seed the two proofs must be equal byte for byte, and the cases' verifier must
// accept the GPU engine's and reject (printed: refused) its proof with the first claim's `target`
// off by 1; no other false proof is tried.
// Build it outside the source tree with prover/tests/unit.sh, which runs nvcc with -DBLK_FIELD=2
// (Goldilocks; no other value compiles). SUITE.md's tests run only the comparison.
// Usage: last_run_test      the comparison (any argument count but 3)
//        last_run_test <log2 of the message length> <main rows> <draws>      (not validated)
//            proves one case on the GPU engine only and prints the seconds
// Exit code 0: all is as expected (timed case: the verifier accepts); 1: not; 3: no GPU.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // allows plans written by hand (blocks/numbers/plan.h)
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "engine/engine_cpu.h"
#include "engine/engine_gpu.cuh"
#include "iop/tests/iopp_cases.h"
#include "iop/tests/req_cases.h"
using namespace blk;

static int compare_engines() {
  int runs = 0, differ = 0, refused_true = 0, accepted_false = 0;
  for (const IoppCase &t : iopp_toy_cases(6))
    for (u64 seed : {11, 12, 13}) {
      EngineCpu cpu;
      EngineGpu gpu;
      const IoppProved a = iopp_toy_prove(t, seed, E::zero(), cpu), b = iopp_toy_prove(t, seed, E::zero(), gpu);
      runs++;
      if (a.proof != b.proof) differ++, std::printf("the two engines' proofs differ: %s, seed %llu\n", t.name, (unsigned long long)seed);
      if (!iopp_toy_verify(t, seed, b.proof)) refused_true++, std::printf("a true proof is refused: %s\n", t.name);
      EngineGpu again;
      const IoppProved f = iopp_toy_prove(t, seed, E::one(), again);
      if (iopp_toy_verify(t, seed, f.proof)) accepted_false++, std::printf("A FALSE TARGET IS ACCEPTED: %s\n", t.name);
    }
  for (const ReqCase &t : req_toy_cases())
    for (u64 seed : {21, 22}) {
      const ReqWorld w = req_toy_world(t);
      EngineCpu cpu;
      EngineGpu gpu, again;
      const ReqProved a = req_toy_prove(w, seed, ~0u, cpu), b = req_toy_prove(w, seed, ~0u, gpu);
      runs++;
      if (a.proof != b.proof) differ++, std::printf("the two engines' proofs differ: a request, %s, seed %llu\n", t.name, (unsigned long long)seed);
      if (!req_toy_verify(w, seed, b.proof)) refused_true++, std::printf("a true proof is refused: a request, %s\n", t.name);
      if (req_toy_verify(w, seed, req_toy_prove(w, seed, 0, again).proof)) accepted_false++, std::printf("A FALSE TARGET IS ACCEPTED: a request, %s\n", t.name);
    }
  const bool good = !differ && !refused_true && !accepted_false;
  std::printf("last_run_test: field %d, hash %d: %d runs: %d proofs differ between the engines, %d true proofs refused, %d false "
              "targets accepted: %s\n", BLK_FIELD, BLK_HASH, runs, differ, refused_true, accepted_false,
              good ? "all as expected" : "NOT as expected");
  return good ? 0 : 1;
}

static int timed_case(unsigned log_ell, size_t rows, unsigned draws) {
  // Stages: the rows are folded into one, which is cut into 2^4 rows of a sixteenth of the length,
  // until the length is 2^12 or less.
  IoppCase t{"timed", {{rows, 1, 1}}, (size_t)1 << log_ell, 64, {}, draws, 1, false};
  for (unsigned l = log_ell; l > 12; l -= 4) t.later.push_back(Later{4, 0});
  t.later.push_back(Later{0, 0});
  EngineGpu gpu;
  const auto from = std::chrono::steady_clock::now();
  const IoppProved p = iopp_toy_prove(t, 1, E::zero(), gpu);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
  const bool ok = iopp_toy_verify(t, 1, p.proof);
  std::printf("last_run_test: field %d, hash %d, made-up data, %zu rows of 2^%u, %u draws a stage, %zu stages: proved in %.3f s "
              "(data made and sent included), %zu bytes, %zu trees, the verifier says %s\n", BLK_FIELD, BLK_HASH, rows, log_ell, draws,
              t.later.size() + 1, seconds, p.proof.size(), p.trees, ok ? "yes" : "NO");
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
    std::printf("no GPU: nothing was run\n");
    return 3;
  }
  if (argc == 4) return timed_case((unsigned)atoi(argv[1]), (size_t)atoll(argv[2]), (unsigned)atoi(argv[3]));
  return compare_engines();
}
