// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes the generator (`EngineCoins`, prover/hiding/engine_coins.cuh) of private randomness, such
// as a proof's masks and salts, for one run of the committer cp_commit. A run creates the key file
// (`ivk/key.bin`) or one leaf file (`ivk/leaf/<k>.bin`). Its 32-byte key (the process key) is read
// from the static state of prover/hiding/trees.cuh (`KEYW`).
//
// The prover cp_prove must not include this file or name `KEYW` outside
// prover/hiding/trees.cuh; the compiler does not enforce this. cp_prove passes the process key to
// engine_coins() as an argument, because with overlapping batches (`-overlap`) the key in the state
// need not belong to the batch that a function works for.
//
// cp_commit reads the state's copy of the key because it does one run at a time and its self-check
// [own-test-key] (`RunKeys`, prover/hash_commitment/hf_commit_keys.h) is on that copy.
#pragma once
#include "hiding/engine_coins.cuh"

namespace hz {
namespace lng {
// Copies the process key out of the state, for engine_coins_once() only, which wipes the copy once
// the generator has its own key. Exits with code 2 if the key is missing or wiped, and in a child
// of fork(), which would repeat its parent's random values.
// cp_commit's self-check compares a 32-bit word derived from each key. The key file's run compares
// the state's copy with the key derived from the seed file, read again, before the generator is
// made and after the key file is complete. A leaf file's run compares it only once, when the key is
// stored, with the key just derived.
static void engine_key(u32 out[COINKEYW]) {
  if (key_state != KEY_TEST && key_state != KEY_GIVEN) key_refuse("engine_key() needs a key");
  if (!*own_process_()) key_refuse("the key was asked for by a fork's child of the process that chose it");
  for (int k = 0; k < COINKEYW; k++) out[k] = KEYW[k];
}
// The generator for a whole run: its context (`blk::CoinContext`, blocks/coins/coins.h: the label
// words that, with the process key, fix its key) has counter 0, member 0 and `object`:
// `ENGINE_KEY_FILE` or (leaf file) `ENGINE_ENTRY`. Call it once per process key (not checked at run
// time; prover/tests/coin_contexts_listed.py checks only its test runs).
// A second generator with the same `object` would give the same random values to other messages.
static inline std::unique_ptr<EngineCoins> engine_coins_once(Object object) {
  blk::CoinKey process;
  engine_key(process.w); // ends the program if no key is there
  auto coins = engine_coins(process, 0, object, 0);
  blk::wipe(process);
  return coins;
}
} // namespace lng
} // namespace hz
