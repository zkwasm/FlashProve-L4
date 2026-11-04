// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Sets and reads the prover's failure bits, which decide whether a request's files are released
// (prover/common/slot_flags.h defines, keeps and clears them). Setting a bit stops nothing. So before a
// program releases a file, it must read the bits of every translation unit in its process: they are
// kept in static words, one set per unit. The prover program cp_prove also links compiled copies of
// prover/hash_commitment/hf_keeper.cu, each with its own set, read by take_flags(). Identifiers say
// "flags" for the bits, and "draw" for the derivation of a field element from hash output. This file
// derives none: a proof's hash chains (its Fiat-Shamir transcripts) derive the verifier's challenges
// (blocks/chain/chain.h). test_fault() is a stub of a test hook and does nothing.
#pragma once
#include "common/common.h"
#include "common/refusal_test.h"
#include "common/slot_flags.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ORs `flags`, bits of SlotFlag (prover/common/slot_flags.h), into the live word of the calling
// side, host or GPU: that side's failure bits for the work in progress. This file itself sets one
// bit, NO_ELEMENT, for a test switch and k_keys() (below).
//
// ZERO_ELEMENT: a value that the verifier requires to be non-zero is zero: kh_gam()
// (prover/hiding/tail_gam.cuh) raises it; the comment there gives the probabilities. Whether that
// occurs depends on the challenges, so on the Fiat-Shamir transcript, and so on hidden values and on
// the prover's private randomness, with a probability that does not depend on them if the hash is
// modelled as a random oracle.
__host__ __device__ inline void raise_draw_flags(unsigned flags) {
  raise_flags(flags);
}
// Called by spells_p() below and, when a key does not fit in 16 bits, by a kernel
// that nothing launches (k_keys(), prover/weights_commitment/kernels.cuh): no element is derived
// there, but the failure is reported as NO_ELEMENT.
__host__ __device__ inline void raise_the_refused_draw_flag() {
  raise_draw_flags(NO_ELEMENT);
}

// Returns whether any failure bit, not only NO_ELEMENT, is set on the host or on the GPU; the
// caller must then release nothing of the work in progress. It reads only this translation unit's
// words (see the top of this file). It waits for the whole GPU first. A word that cannot be read
// counts as set.
//
// Test switch: the environment variable COINS_REFUSED_DRAW=attempt:<k> (builds with the macro
// CHAIN_HOST_ENGINE) or try:<k> (all other builds). The k-th call of this function in the
// translation unit sets NO_ELEMENT on the host, so the switch can only make a run withhold what it
// would have released. Each kind of build ignores the other's form. At any other value the first
// call ends in exit(2). (refusal_test::knows(), which is asked first, is a stub that returns false:
// prover/common/refusal_test.h.) A production
// build (SERVING_BUILD) does not read the variable (TESTENV, prover/common/setting.h). While it is
// set, cp_commit and a test build's cp_prove exit unless they run on synthetic weights with a test
// option: they call test_word_in_the_environment() (prover/common/test_words.h). A production
// cp_prove, and a test build that sets
// CLOSED_KEEPS_THE_TEST_WORD, first restarts itself without the variable in its environment
// (ClosedFront::enter(), prover/prover/closed_front.h).
static bool spells_p() {
  static const long strike = [] {
    const char *const word = TESTENV("COINS_REFUSED_DRAW");
    if (!word || refusal_test::knows(word)) return 0L;
    const bool attempt = !strncmp(word, "attempt:", 8), a_try = !strncmp(word, "try:", 4);
    const char *const digits = attempt ? word + 8 : a_try ? word + 4 : "";
    if (!*digits || strlen(digits) > 9 || strspn(digits, "0123456789") != strlen(digits) || !atol(digits)) {
      fprintf(stderr, "COINS_REFUSED_DRAW must be attempt:<k> or try:<k>, with k a whole number from 1 to 999999999\n");
      exit(2);
    }
#if defined(CHAIN_HOST_ENGINE) && CHAIN_HOST_ENGINE
    return attempt ? atol(digits) : 0L;
#else
    return a_try ? atol(digits) : 0L;
#endif
  }();
  if (strike > 0) {
    static long askings = 0;
    if (++askings == strike) raise_the_refused_draw_flag();
  }
  return live_flags() != 0;
}
// A stub of a test hook: does nothing and returns 0 in every build from these sources. Its
// callers return the result at once if it is not 0, which cannot happen.
static inline int test_fault(const char *where) {
  (void)where;
  return 0;
}
