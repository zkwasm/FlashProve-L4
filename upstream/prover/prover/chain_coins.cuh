// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The generators of the private random values (among them random tails, mask rows, pads, salts;
// `coins` in identifiers and messages) that cp_prove, the prover, draws through the library
// blocks/. Each request of a batch has one generator, and the batch has one for what its requests
// share.
//
// Prover::begin_the_batch_on_a_leaf() (prover/prover/isk.cuh) makes all generators of a batch where
// it derives the batch's process key (their common key), then wipes that key object; copies of
// its bytes stay in the state of prover/hiding/trees.cuh and in `slot_pads_second`, a self-check of
// the mask entries of the sub-proofs (each argues one part of the forward pass). No other code of
// cp_prove may make a generator, so that none is made under the key of an overlapping batch. The
// compiler does not enforce this; a test that is not part of this repository counts, in the
// sources, the calls of engine_coins() and of its two callers below.
#pragma once
#include <memory>

#include "common/setting.h" // TESTENV
#include "hiding/engine_coins.cuh"
#include "prover/slot.h" // Slot::coins, Batch::coins: std::unique_ptr<blk::Coins>

// A generator (hz::lng::EngineCoins, prover/hiding/engine_coins.cuh) derives its values by
// ChaCha20's block function from the process key and a context of four fields: counter, version
// word, object, member (blocks/coins/coins.h). It holds its own context key, so it still works
// after hz::key32_wipe() (prover/hiding/trees.cuh); its destructor wipes that key.
//
// The sub-proofs' kernels draw from prover/hiding/trees.cuh: the same derivation under the same key
// and, for a request, the same counter and member. Both use stream numbers from 16,384 up, and
// neither knows which streams the other has used. Only the object field keeps their values apart:
// ENGINE or ENGINE_BATCH here, REQUEST there (prover/hiding/long_rows_words.h has the
// static_assert).
//
// Under one process key a context must get only one generator: a second one would repeat the
// first one's values in other messages. Nothing checks this at run time;
// prover/tests/coin_contexts_listed.py checks it on listings of test runs. In a production build
// every batch's process key depends on, among other inputs, its requests and 32 fresh bytes of the
// operating system.
//
// The batch's generator. end_the_batch() (prover/prover/service.cuh) destroys it;
// the_batch_gets_no_proof() does not: it stays, with its context key, until a later batch's
// generator replaces it or the process ends. The caller wipes `process`.
static inline std::unique_ptr<blk::Coins> chain_coins_of_the_batch(const blk::CoinKey &process) { return hz::lng::engine_coins(process, 0, hz::lng::ENGINE_BATCH, 0); }
// The generator of a slot (a request's place in its batch) that is not empty. `s.attempt`, the
// context's counter, must be set to a number that no other slot of the batch has.
// hz::set_attempt(), called by begin_request() (prover/prover/request.cuh), ends the program unless
// that number exceeds every earlier counter under the key.
static inline std::unique_ptr<blk::Coins> chain_coins_of_a_seat(const blk::CoinKey &process, const Slot &s) { return hz::lng::engine_coins(process, s.attempt, hz::lng::ENGINE, (u32)s.j); }

// [own-test-engine-coins] Compares up to four values of each draw of `coins` since the last call,
// read back from host or GPU memory, with a second implementation of the derivation
// (EngineCoins::hold(); `second writing` in messages). Both implementations' ChaCha20 block
// functions follow the same public source. So the comparison checks what goes into that function
// (context key, stream, position) and what is made of its output, but cannot show a fault that
// both have from the source. Test vectors of RFC 8439 check the function itself: the first
// implementation's in blk::coins_own_test_card(), when cp_prove starts (one vector, section 2.3.2),
// and a copy of the second's in prover/tests/coins_rfc8439_test.cpp (a test that is not part of
// this repository compares the copy with the original byte for byte).
// EngineCoins::hold() also reads the counts of [own-test-coins-bound]
// (prover/engine/stream_coins.cuh), which compares every main row's random tail and every leaf's
// salt in GPU memory, once their hash tree is made, with the first implementation's values, but no
// value drawn on the CPU. If a check fails, no value was sampled, or `coins` is not an
// EngineCoins, cp_prove exits with code 9 because the end's name begins "own-test-", unless another
// ending takes precedence (stop(), prover/prover/ends.h). Code that lets bytes of a proof leave the
// process must call this first.
static inline void coins_hold(blk::Coins *coins, const char *whose) {
  if (auto *own = dynamic_cast<hz::lng::EngineCoins *>(coins)) {
    const size_t compared = own->hold();
    if (TESTENV("CHAIN_TIMES")) printf("# coins: %s: %zu values held against the second writing\n", whose, compared);
    if (!compared) { // each call should follow a draw: at 0, nothing was compared
      printf("internal error: %s coins: no value was sampled since the last look: the coins' own test by value has seen nothing\n", whose);
      name_the_end("own-test-engine-coins-unseen");
      exit(4);
    }
    return;
  }
  printf("internal error: %s coins are not this program's own generator: their values cannot be held against the second writing\n", whose);
  name_the_end("own-test-foreign-coins");
  exit(4);
}

// The generator of the request whose slot is entered. It stays until end_the_batch() or
// the_batch_gets_no_proof() clears the slot: the request's second hash chain draws from it, and the
// GPU engine regenerates the salts of opened tree leaves from it.
inline blk::Coins &Prover::chain_coins() {
  if (!g_slot_entered) {
    printf("internal error: the chain: a request's coins are asked for while no slot is entered\n");
    exit(4);
  }
  Slot &s = *g_slot_entered;
  if (!s.coins) { // `begin_the_batch_on_a_leaf` makes one for every slot that is not empty
    printf("internal error: the chain: the entered slot has no generator of coins\n");
    exit(4);
  }
  return *s.coins;
}
