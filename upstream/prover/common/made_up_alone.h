// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tells the test hooks of cp_commit (committer) and cp_prove (prover) whether this run
// uses synthetic weights (`made_up` in identifiers) and a test seed (read from a file, for
// repeatable runs). Only then may a hook write or print values of the forward pass or of the
// committed arrays, weights among them, or remove a weight table's masks or change its salts. The
// aim: no hook produces a file or line derived from a model's weights, or with private randomness
// from any but a test seed. This file only answers: it cannot make a hook ask.
#pragma once
namespace made_up_alone {
// False until say() is called. Static, so each translation unit has its own copy; nothing sets the
// copy in prover/hash_commitment/hf_keeper.cu (compiled separately), which stays false.
static bool answer = false;
// Called in cp_prove only by Prover::read_isk() (prover/prover/isk.cuh), with arguments from
// `isk.txt` (lines `weights`, "seed test"); in cp_commit by cp_commit_main()
// (prover/hash_commitment/hf_commit.cu), from -weights and -test-seed.
// weights_are_made_up: true only for the weights `generated` and `made-up-blocks`.
static inline void say(bool weights_are_made_up, bool seed_is_a_tests) { answer = weights_are_made_up && seed_is_a_tests; }
} // namespace made_up_alone
// Among the hooks that ask: cp_commit's printed lines on integers out of range (`Scan::say` in
// prover/weights_commitment/check_commitment.cuh) and these test variables, read with TESTENV()
// (prover/common/setting.h), so never in a production build:
//   CHAIN_PUT_VECTORS (prover/hiding/door_chain.cuh) writes a proof's coefficient vectors, their
//     values on the committed arrays, and mask rows;
//   CHAIN_ROWS_OUT (prover/head/forward.cuh) writes each position's top choice, logit gap and cost;
//   FORMF_STREAMS_REACH (prover/prover/layer_prove.cuh) prints a layer group's largest values;
//   COINS_CONTEXTS (prover/hiding/engine_coins.cuh) lists what was drawn at random, not the values;
//   CHAIN_HELD_STEM (prover/hash_commitment/hf_commit.cu, only with HELD_LOOK defined) writes a key
//     file made with the verifier's messages fixed;
//   ROWS_TABLE_FAULT (prover/hash_commitment/hf_commit_table.cuh) writes no value: it zeroes masks
//     or hashes other salts.
// If the answer is false, Prover::read_isk() also rejects, outside a production build, a run with
// one of the first three set. A test build that is not part of this repository (malicious provers,
// injected faults), which prints the values that it alters, rejects every run with a false answer.
// A tool that is not part of this repository does not ask: it writes the same records as CHAIN_ROWS_OUT
// under any seed. It cannot be built as a production build, and every other build rejects a model's
// files (BUILT_FOR_TESTS_ALONE in prover/common/setting.h).
static inline bool made_up_under_a_tests_seed() { return made_up_alone::answer; }
