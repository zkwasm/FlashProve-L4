// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A toy program's main function, for any `Engine` (blocks/engine/engine.h: the prover's heavy
// steps): makes one toy proof (blocks/toy/product.h: vectors with a[i] * b[i] = c[i]) at the
// default target (`Setting::target`, 128 bits), verifies it, and prints the error accounting (the
// rounds' error bounds) and the proof's size, digest and verdict.
//   <program> [work bits [key]]
// work bits: proof of work on the rounds that draw positions, counted toward the target; default 0.
// key: a number that fixes the witness and the private randomness (`Coins`): nothing is hidden
// then, and two engines that request randomness in the same order make the same proof.
// The generator is the library's (blocks/coins/run_coins.h), keyed with the key or with fresh
// system bytes (two draws of 32), and with a digest of the run's inputs.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define BLK_REPEATABLE_COINS_HIDE_NOTHING 1 // for the run with a key on the command line
#include "coins/tests/repeatable_coins.h"
#include "numbers/accounting.h"
#include "toy/product.h"
#include "toy/toy_coins.h"

namespace blk {

// `CoinsType`: `StreamCoins`, or an engine's generator that also writes into GPU memory.
// Exit code: 0 accepted, 1 rejected, 2 invalid work bits or proof parameters.
template <class EngineType, class CoinsType = StreamCoins> int toy_main(int argc, char **argv) {
  coins_own_test();
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  if (argc > 1 && (std::atoi(argv[1]) < 0 || std::atoi(argv[1]) > 30)) {
    std::printf("work bits (the first word) must be 0 to 30: each further bit doubles the prover's search\n");
    return 2;
  }
  if (argc > 1) set.set_work(ErrorKind::QUERY, Scope::REQUEST, (unsigned)std::atoi(argv[1]));
  // Vectors of 2^12 entries. In the proximity test, a stage's mask rows carry the previous stage's
  // masks and encoding randomness, which do not shrink with the message, so the message is halved
  // only once.
  const ToyRun run = toy_size(set, 12, 15, 11, {StageChoice{1, 1, 14, 12}, StageChoice{0, 0, 14, 14}});
  if (run.plan->refused()) {
    std::printf("no such parameters: %s\n", run.plan->why_not().c_str());
    return 2;
  }
  print_accounting(account(run.plan), "toy run");

  ToyCoins made_up = argc > 2 ? ToyCoins((u64)std::atoll(argv[2])) : ToyCoins::fresh(); // the witness, and nothing else
  Table abc(1, 3, run.shape.codes.base[0].ell);
  made_up.fill(abc.a.data(), 2 * abc.len);
  for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
  const Digest inputs = CoinInputs().add("the public file", run.shape.text).add("the witness", abc.a.data(), abc.a.size()).finish();
  const std::unique_ptr<CoinsType> own = argc > 2 ? run_coins_that_hide_nothing<CoinsType>((u64)std::atoll(argv[2]), inputs) : run_coins_without_a_seed<CoinsType>(inputs);
  Coins &coins = *own;

  EngineType engine;
  Writer proof;
  const auto t0 = std::chrono::steady_clock::now();
  toy_prove(run, abc, engine, coins, proof);
  const auto t1 = std::chrono::steady_clock::now();
  const char *why = nullptr;
  const bool ok = toy_verify_file(run.shape.text, proof.data(), proof.size(), &why); // derives sizes and plan from the public file only
  const auto t2 = std::chrono::steady_clock::now();
  const PartSize planned = run.plan->size_of(0);
  std::printf("proof: %zu bytes (planned: at most %llu, expected %.0f); prover %.3f s, verifier %.3f s\n", proof.size(),
              (unsigned long long)planned.bytes, planned.expected, std::chrono::duration<double>(t1 - t0).count(),
              std::chrono::duration<double>(t2 - t1).count());
  // Outside tests, each user of this tag takes an index (the second argument) of its own.
  Hasher h(Tag(Domain::STATEMENT), 1); // only for the printed digest; index 1 is this function's
  h.push(proof.data(), proof.size());
  u8 digest[DIGEST_BYTES];
  digest_to_bytes(h.finish(), digest);
  std::printf("proof digest: ");
  for (u8 b : digest) std::printf("%02x", b);
  std::printf("\nverdict: %s%s%s\n", ok ? "ACCEPT" : "REJECT", ok ? "" : ": ", ok ? "" : why);
  return ok ? 0 : 1;
}

} // namespace blk
