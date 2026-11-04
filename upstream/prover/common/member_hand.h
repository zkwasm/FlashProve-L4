// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// What a request ("member" in identifiers) would pass to the opening, a batch's test on the weight
// table. Unused: `cp_prove` declares one, Slot::hand (prover/prover/slot.h), but fills and uses no
// field.
#pragma once
#include <cstddef>
#include <cstdint>

// The arrays must be in host memory, never in a file. The weight matrix has 2^lkW columns (`lkW`:
// the last number of the parameter file's line `table`).
// Code that fills pad_tree_rows must pass a request's rows only once, ever, for one slot of one
// batch, and then zero them; nothing in `cp_prove` checks this. In the scheme of
// prover/common/pad_tree_rows.h a request's pads are random values added to the values of its
// claims on the weights, and its pad tree (`Prover::CP`, never filled) commits to them.
// A row's random tail (random coefficients) is sized for that file's CP_LISTS = 2 lists of
// positions: one for the request's own proof, one for a single second message of the opening.
// Passed again, in the same batch or another, the rows would be opened under a second challenge
// and at a further list, which reveals the request's pads and with them the values of its claims
// on the weights.
struct MemberHand {
  const uint32_t *record = nullptr; // the request's record for the opening: its digest, its pad tree's root, its claims
  size_t record_words = 0;
  const uint64_t *fold_tables = nullptr; // one table of 2^lkW field elements for each of the request's sub-proofs
  size_t fold_table_words = 0;           // that involve weights; pinned memory
  const uint64_t *pad_tree_rows = nullptr; // the rows and salts of the request's pad tree; pinned memory
  size_t pad_tree_row_words = 0;
};
