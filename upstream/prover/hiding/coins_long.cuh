// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The numbers of a blinding context's `object`: word 11 of the label of each private random value
// ("coin" in identifiers and messages). cp_commit and cp_prove use the enum Object. Beside it the
// file has two static_asserts: on that list, and on the retry word of a label.
// set_attempt(number, object, member) (prover/hiding/trees.cuh) begins a context ("attempt" in
// identifiers), whose key depends on all three arguments.
// Nothing makes host code that launches a kernel outside trees.cuh record the cells (values) that
// the kernel generates: coink() returns a context's key to any caller, and the ledger of draws holds
// only what callers enter (enter_draw(), trees.cuh).
// Include after prover/common/common.h (u32) and prover/common/coin_chacha.h (COINOUTW).
#pragma once
#include "hiding/long_rows_words.h"
namespace hz {
// lng ("long", as in "long trees"): in this file the enum Object and two static_asserts.
// prover/hiding/engine_coins.cuh adds the generators for the ENGINE objects to this namespace.
namespace lng {
// A context's `object`: word 11 of the sixteen-word label of each of its random values
// (prover/common/coin_chacha.h). Word 12, `member`, is the request's slot in its batch (0 if the
// context is no request's). REQUEST is for values drawn through prover/hiding/trees.cuh; the four
// ENGINE constants are for the generators of prover/hiding/engine_coins.cuh. The two sides keep
// separate records of draws, so under one process key only this word keeps their values apart;
// set_attempt() rejects no ENGINE constant. Only tests use FILLER.
enum Object : u32 { REQUEST = 1, ENGINE = 2, FILLER = 3, ENGINE_BATCH = 4, ENGINE_KEY_FILE = 5, ENGINE_ENTRY = 10 };
static_assert((u32)REQUEST == (u32)lrw::REQUEST && (u32)FILLER == (u32)lrw::FILLER && (u32)ENGINE == (u32)lrw::ENGINE &&
                  (u32)ENGINE_BATCH == (u32)lrw::ENGINE_BATCH && (u32)ENGINE_KEY_FILE == (u32)lrw::ENGINE_KEY_FILE && (u32)ENGINE_ENTRY == (u32)lrw::ENGINE_ENTRY,
              "this list is hiding/long_rows_words.h's, which is held against the opening's numbers there");
// A block has 4 cells, or SALTW / 2 in a SALT stream (lrw::cells_a_block()). Try number tr >= 1 of
// cell q of a block has the retry word RETRYQ * tr + q (coin_again(), which does not check q, and
// coin4(): prover/hiding/trees.cuh). q is below RETRYQ, so no two cells share a retry word.
static_assert(SALTW / 2 >= 4 && SALTW <= COINOUTW && SALTW / 2 <= RETRYQ,
              "a block's cells: room, words, retry words apart");
} // namespace lng
} // namespace hz
