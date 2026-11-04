// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Finds test-only environment variables. They switch on a self-test, tamper with a value, print
// internal values or change a tuning setting. They are meant only for test runs on synthetic
// weights (generated, not a model's): some write out hidden values, and some change what is sent.
#pragma once
#include <cstring>
extern char **environ;
// Returns the first test variable in `environ`, as "NAME=value", or null. A variable whose name
// matches no string below is not found. If one is found, the committer program cp_commit and the
// prover program cp_prove exit with an error, unless the run is on synthetic weights with the test
// build's option -test-seed (cp_commit) or -test-coins (cp_prove). A production build
// (`SERVING_BUILD` defined) of cp_prove finds none: it first restarts itself with such variables
// removed (ClosedFront::enter(), prover/prover/closed_front.h).
// Three names are never returned (`served`), so no caller exits because of them: CHAIN_HOST_ENGINE
// and CHAIN_STAND_IN name compile-time macros, and neither program reads CHAIN_KEEP_NODES.
static inline const char *test_word_in_the_environment() {
  for (char **e = environ; e && *e; e++) {
    bool ours = false;
    // A string ending in "=" is a whole name; the others are prefixes. Five of the first six name
    // families of variables that the sources read, mostly through TESTENV()
    // (prover/common/setting.h), which gives a null pointer in a production build. Three examples:
    // ROWS_LOSE_COINS zeroes random values after they are drawn
    // (prover/hiding/lost_coins_door.cuh). LENT_POISON fills each newly allocated GPU buffer with a
    // given byte (prover/weights_commitment/lent_memory.h). FORMF_STREAMS_REACH prints two numbers
    // that depend on the weights (prover/prover/layer_prove.cuh). No source in this repository
    // reads a name that begins with DEAD_REQUEST.
    // The generator of the masks (random values encoded with the weights) draws a block of eight
    // values again if any of them is at or above a threshold. In a test program built with
    // `HC_MASK_LOW_LIMIT` nonzero, H_HC_MASK_LIMIT=k (k from 1 to 7) lowers that threshold to k
    // eighths of its value, to make retries common. Other test builds exit if the variable is set,
    // and a production build does not read it (prover/hash_commitment/hf_mask.cuh).
    // WEIGHT_FEED tunes weight loading (prover/prover/weight_stream.cuh); it should change no
    // output file. The last seven are for tests with malicious provers or injected faults; this
    // function reports them in every build. Beyond that report no source of cp_commit or cp_prove
    // in this repository reads one of them. The test builds that read the first four, through the
    // test hooks in prover/common/lie.cuh and prover/hiding/sumcheck_door.cuh (which do nothing in
    // this repository), are not part of this repository.
    for (const char *head : {"CHAIN_", "FORMF_", "ROWS_", "COINS_", "DEAD_REQUEST", "LENT_", "H_HC_", "WEIGHT_FEED=", "LIE=", "LIE_SEAT=", "PLANT=", "DOOR_CHEAT=", "LAYER_LIE=", "COMMIT_LIE=", "COMMIT_LIE_"})
      ours = ours || !strncmp(*e, head, strlen(head));
    if (!ours) continue;
    bool served = false;
    for (const char *name : {"CHAIN_KEEP_NODES=", "CHAIN_HOST_ENGINE=", "CHAIN_STAND_IN="})
      served = served || !strncmp(*e, name, strlen(name));
    if (!served) return *e;
  }
  return nullptr;
}
