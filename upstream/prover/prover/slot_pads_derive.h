// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives a sub-proof's slot pads, which mask what it sends, with the second implementation of the
// private-randomness derivation (prover/second_writing/), for the self-check [own-test-slot-pads]
// in prover/prover/slot_pads_second.cuh. Free of CUDA, so that a C++ test can compile this function
// and read the stack after it; such a test is not part of this repository.
#pragma once
#include "common/stack_scrub.h"
#include "second_writing/request_coins.h"
#include <cstdint>
#include <cstring>

// Fills want[kappa * slots + y] with coordinate kappa < 3 of the pad of message slot y < `slots` of
// sub-proof gid. key_words: the batch's process key, eight 32-bit words. `version`: the format's
// version word. `slot`: the request's index in its batch.
// Returns false if the second implementation (`coins::`) throws: it rejects gid's stream number or
// runs out of retries.
// Call scrub_the_stack_below() (prover/common/stack_scrub.h) right after the return, whatever the
// result: the second implementation leaves ChaCha20 states on the stack, unwiped, each with the key
// that it ran under. All calls of the cipher after the third run under the key of
// request_proof_context(version, slot), which determines every private random value of that
// context. Those are the slot pads of all the request's sub-proofs: cp_prove draws nothing
// else under that key. The first three calls derive that key. The first runs under the process key,
// which determines every private random value of the batch: those of all its requests and of what
// they share. The next two run under keys derived from it. These three keys can remain on the stack
// too, unless a later call overwrites them.
// Never inlined, so the states are below the caller's frame.
__attribute__((noinline)) static bool slot_pads_derive(const void *key_words, uint32_t version, uint32_t slot, uint32_t gid,
                                                       size_t slots, uint64_t *want) {
  try {
    coins::Key key;
    memcpy(key.data(), key_words, sizeof key);
    const coins::RequestCoins request(key, version, slot);
    coins::wipe(key);
    request.slot_pads(gid, slots, want);
    return true;
  } catch (...) {
    return false;
  }
}
