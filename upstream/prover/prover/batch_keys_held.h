// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// [own-test-batch-keys] Self-check in the prover program cp_prove. A second implementation of the
// key schedule, prover/second_writing/schedule_keys.h (namespace `coins`), derives a batch's first
// key and process key again. Its ChaCha20 and BLAKE2b follow the same public sources as cp_prove's.
// Prover::begin_the_batch_on_a_leaf() (prover/prover/isk.cuh) compares each key before first use.
//
// prover/common/coin_keys.h derives the keys from the provider's seed, 32 bytes from the operating
// system and the batch's inputs (`facts` in the message). The batch's private randomness (`coins`
// in identifiers and messages) is generated under the process key. A proof verifies under any key,
// so no verifier notices a key that omits an input, such as the requests or the name of the leaf
// table (a committed copy of the weights; `copies` counts them). If the system's bytes ever
// repeated, such a key would give the same randomness to different messages.
// prover/tests/coin_keys_test.cpp compares the two implementations but starts no program, so it
// cannot see which function cp_prove calls, or with which inputs.
// Not detected: a wrong seed, system bytes or input, because both derivations get the same ones; a
// fault that both implementations share; a change to the key after the comparison. Host only.
#pragma once
#include "common/coin_keys.h"
#include "common/end_name.h"
#include "common/stack_scrub.h"
#include "second_writing/schedule_keys.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace batch_keys {
inline coins::Key words_of(const uint8_t bytes[32]) {
  uint8_t b[32];
  memcpy(b, bytes, 32);
  const coins::Key k = coins::key_from_bytes(b);
  explicit_bzero(b, sizeof b);
  return k;
}
// Never inlined, so that scrub_the_stack_below() zeroes what f and its callees leave on the stack
// (limits: prover/common/stack_scrub.h).
template <class F> __attribute__((noinline)) static void a_frame_of_its_own(F &f) { f(); }
// Compares `in_hand`, cp_prove's key, with derived(), the second implementation's. If they differ,
// prints which key, but no value, and calls exit(4). In cp_prove, stop() (prover/prover/ends.h)
// exits with code 9 instead, because the name given to name_the_end() begins "own-test-", unless
// another ending takes precedence.
// No test of this repository makes the comparison fail.
// A match shows that the two implementations agree on the schedule: which bytes go into which
// ChaCha20 or BLAKE2b call, in which order. It cannot show a fault that both ChaCha20 block
// functions, or both texts of BLAKE2b, took from their common source. The RFCs' test
// vectors check the block and the compression functions alone: RFC 8439's in blocks/coins/tests/coins_test.cpp (cp_prove's) and
// prover/tests/coins_rfc8439_test.cpp (a copy of the second implementation's); RFC 7693's in
// blocks/hash/tests/functions_test.cpp (cp_prove's only).
template <class Derived> inline void held(const uint8_t in_hand[32], const char *which, Derived derived) {
  coins::Word difference = 0;
  auto compare = [&] {
    coins::Key theirs = words_of(in_hand), mine = derived();
    for (int i = 0; i < 8; i++) difference |= theirs[i] ^ mine[i];
    coins::wipe(theirs), coins::wipe(mine);
  };
  a_frame_of_its_own(compare);
  scrub_the_stack_below();
  if (!difference) return;
  printf("internal error (own-test-batch-keys): a self-check failed: the batch's %s is not what a second, separately written form of the coins' schedule (prover/second_writing/) derives "
         "from the same seed, bytes and facts. This is a fault of the program or of the machine, not of a request. No coin was drawn under the key and no proof of this batch is made. "
         "Report this line\n", which);
  name_the_end("own-test-batch-keys");
  exit(4);
}
// Checks the first key. The other arguments must be those given to coin_keys::batch_first_key().
inline void the_first_key_is(const uint8_t in_hand[32], const uint8_t seed[32], const uint8_t of_the_system[32], const blk::Digest &configuration, uint32_t copies, uint32_t first_live,
                             const std::vector<coin_keys::Slot> &slots) {
  held(in_hand, "first key", [&] {
    uint8_t c[32];
    blk::digest_to_bytes(configuration, c);
    std::vector<coins::schedule::SlotWords> list;
    for (const coin_keys::Slot &s : slots) list.push_back(coins::schedule::SlotWords{s.is_request, s.words});
    coins::Key root = words_of(seed), f = words_of(of_the_system);
    const coins::Key k = coins::schedule::batch_first(root, f, coins::schedule::Octets(c, c + 32), copies, first_live, list);
    coins::wipe(root), coins::wipe(f);
    for (coins::schedule::SlotWords &s : list) explicit_bzero(s.words.data(), 4 * s.words.size()); // clients' data
    return k;
  });
}
// Checks the process key. The other arguments must be those given to
// coin_keys::batch_process_key(), the first key as 32 bytes.
inline void the_process_key_is(const uint8_t in_hand[32], const uint8_t first_key[32], const coin_keys::Bytes &name, const coin_keys::Bytes &answers) {
  held(in_hand, "process key", [&] {
    coins::Key first = words_of(first_key);
    const coins::Key k = coins::schedule::batch_process(first, name, answers);
    coins::wipe(first);
    return k;
  });
}
} // namespace batch_keys
