// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Main function of the toy programs for blocks/toy/segments.h, for any `Engine`
// (blocks/engine/engine.h): proves and verifies three segments (`link` in identifiers and output)
// and the final proximity test, on a random witness (2^11-entry vectors), at the default
// `Setting::target` (128 bits). Prints the error accounting and the proof's size, digest and
// verdict.
//   <program> [work bits [key]]
// work bits: proof of work on the rounds that draw positions, counted toward the target; default 0.
// key: a number that fixes the witness and the private randomness (`Coins`, from the library's
// generator, blocks/coins/run_coins.h): nothing is hidden then. Default: fresh system bytes.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define BLK_REPEATABLE_COINS_HIDE_NOTHING 1 // for runs with a key
#include "coins/tests/repeatable_coins.h"
#include "numbers/accounting.h"
#include "toy/segments.h"
#include "toy/toy_coins.h"

namespace blk {

template <class EngineType, class CoinsType = StreamCoins> int segments_main(int argc, char **argv) {
  coins_own_test();
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  if (argc > 1 && (std::atoi(argv[1]) < 0 || std::atoi(argv[1]) > 30)) {
    std::printf("work bits (the first word) must be 0 to 30: each further bit doubles the prover's search\n");
    return 2;
  }
  if (argc > 1) set.set_work(ErrorKind::QUERY, Scope::REQUEST, (unsigned)std::atoi(argv[1]));
  const SegmentsRun run = segments_size(set, 3, 12, 15, 11, {StageChoice{1, 1, 14, 12}, StageChoice{0, 0, 14, 14}});
  if (run.plan->refused()) {
    std::printf("no such parameters: %s\n", run.plan->why_not().c_str());
    return 2;
  }
  print_accounting(account(run.plan), "three links");

  ToyCoins made_up = argc > 2 ? ToyCoins((u64)std::atoll(argv[2])) : ToyCoins::fresh(); // the witness, and nothing else
  std::vector<F> x(size_t(1) << run.vars());
  std::vector<std::vector<F>> b(run.links(), x);
  made_up.fill(x.data(), x.size());
  for (std::vector<F> &v : b) made_up.fill(v.data(), v.size());
  CoinInputs all;
  all.add("the public file", run.text).add("x", x.data(), x.size());
  for (const std::vector<F> &v : b) all.add("b", v.data(), v.size());
  const Digest inputs = all.finish();
  const std::unique_ptr<CoinsType> own = argc > 2 ? run_coins_that_hide_nothing<CoinsType>((u64)std::atoll(argv[2]), inputs) : run_coins_without_a_seed<CoinsType>(inputs);
  Coins &coins = *own;

  EngineType engine;
  Writer proof;
  const auto t0 = std::chrono::steady_clock::now();
  segments_prove(run, x, b, engine, coins, proof);
  const auto t1 = std::chrono::steady_clock::now();
  const char *why = nullptr;
  const bool ok = segments_verify_file(run.text, proof.data(), proof.size(), &why); // sizes and plans again, from the public file only
  const auto t2 = std::chrono::steady_clock::now();
  const PartSize planned = run.plan->size_of(0);
  std::printf("proof: %zu bytes (planned: at most %llu, expected %.0f); prover %.3f s, verifier %.3f s\n", proof.size(),
              (unsigned long long)planned.bytes, planned.expected, std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count());
  Hasher h(Tag(Domain::STATEMENT), 2); // only for the printed digest; index 2 is this function's
  h.push(proof.data(), proof.size());
  u8 digest[DIGEST_BYTES];
  digest_to_bytes(h.finish(), digest);
  std::printf("proof digest: ");
  for (u8 byte : digest) std::printf("%02x", byte);
  std::printf("\nverdict: %s%s%s\n", ok ? "ACCEPT" : "REJECT", ok ? "" : ": ", ok ? "" : why);
  return ok ? 0 : 1;
}

} // namespace blk
