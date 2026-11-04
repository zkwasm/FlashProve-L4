// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Private randomness (`coins` in identifiers) from a seed that anybody knows: it hides nothing. For
// tests and toy runs (blocks/toy/) that compare two runs, or two engines (the CPU and GPU
// implementations of blocks/engine/engine.h), byte for byte. So the file is under tests/, and
// includers must first #define BLK_REPEATABLE_COINS_HIDE_NOTHING 1. Under SERVING_BUILD, which
// production builds define, it does not compile.
#pragma once
#ifndef BLK_REPEATABLE_COINS_HIDE_NOTHING
#error "repeatable_coins.h gives coins under a seed that anybody knows: nothing is hidden under them. Only a test or a toy run includes it (see its head)."
#endif
#ifdef SERVING_BUILD // even if the macro above is defined too late for blocks/fence.h to see it
#error "coins under a seed that anybody knows in the build of a program that serves"
#endif
#include "coins/run_coins.h"

BLK_MARK_NOT_FOR_SERVING(); // blocks/fence.h: embeds a not-for-serving marker string in the program

namespace blk {

// Derives the generator as run_coins() (blocks/coins/run_coins.h) does with `kind` 0, except that
// the seed comes from `public_number` and zeros replace the 32 fresh bytes of the operating system.
// `inputs`, `context`: as for run_coins(). The same arguments and order of draws give the same
// values in every run.
template <class Generator = StreamCoins>
inline std::unique_ptr<Generator> run_coins_that_hide_nothing(u64 public_number, const Digest &inputs, const CoinContext &context = CoinContext{}) {
  CoinKey seed{};
  seed.w[0] = (u32)public_number, seed.w[1] = (u32)(public_number >> 32), seed.w[7] = 0x54534554u; // "TEST"
  const u8 no_fresh_bytes[32] = {};
  return std::make_unique<Generator>(process_key(seed_child(seed, 0, 0), no_fresh_bytes, inputs), context);
}

} // namespace blk
