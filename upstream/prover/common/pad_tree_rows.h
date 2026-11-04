// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Row numbers and counts of Prover::CP (prover/prover/prover.h), a request's "pad tree"
// (crossing-pad commitment): three rows over the extension field, and each column ends in
// SALTW / 2 (five) salt symbols (SALTW: prover/common/setting.h). The only user,
// Prover::plan_memory() (prover/prover/inputs.cuh), sets up CP from the two counts. No code draws,
// commits to or opens CP's rows; code that does must keep to the scheme below. The masks of a
// request's claims on the weights come instead from its index-mask message
// (ChainRun::index_mask_entry(), prover/prover/chain_run.cuh).
#pragma once
// In the scheme, row CP_ROW_PADS, hC, would hold the pads of these claims: claim f's value fv_f
// would be sent as fv_f + hC[f]. hC would be read twice, each time in a combination with a
// challenge and under a masking row that is used for this reading only:
//   the request's own proof would send the entire row C0 = rhoC * hC + rC0 for a challenge rhoC;
//     rC0 is row CP_ROW_CHAIN;
//   the second message of the request's part of the opening of the weight table would use rC, row
//     CP_ROW_SECOND_MESSAGE.
// The scheme follows Dalal, Hemo, Rabinovich, Rothblum, "VEIL: Lightweight Zero-Knowledge for
// Hash-Based Multilinear Proof Systems", Cryptology ePrint Archive 2026/683 (version of
// April 13, 2026): Figure 7 for the pads, Figures 1 and 2 for the masking rows and the readings.
// For a committed vector VEIL commits to one masking vector; CP has two masking rows for hC.
// The two masking rows must differ. A second message masked by rC0 would be C0 plus a multiple of
// hC. A verifier that read the same row would accept it, and no test on honest or altered proofs
// would detect the mistake. But the argument that the pads stay hidden needs a different uniform
// row for each reading.
//
// Each reading would open CP's columns at one list of t positions, and a row's random tail has
// only (CP_LISTS * t + 5) & ~1 coefficients (`t` and `mk` in pad_tree_shape(), prover/hiding/trees.cuh):
// with these counts there must be no third reading and no second message generated twice.
constexpr int CP_ROW_PADS = 0, CP_ROW_CHAIN = 1, CP_ROW_SECOND_MESSAGE = 2;
constexpr int CP_LISTS = 2;            // the lists of positions at which the tree is ever opened: one for each reading
constexpr int CP_ROWS = 1 + CP_LISTS;  // the pads' row, and a masking row for each reading
static_assert(CP_ROW_CHAIN != CP_ROW_SECOND_MESSAGE, "the two readings of the pad tree have masking rows of their own");
static_assert(CP_ROW_PADS == 0 && CP_ROW_CHAIN >= 1 && CP_ROW_CHAIN < CP_ROWS && CP_ROW_SECOND_MESSAGE >= 1 &&
                  CP_ROW_SECOND_MESSAGE < CP_ROWS,
              "the pads lie in row 0, and both masking rows are rows of the tree behind it");
static_assert(CP_ROWS == 3, "the format's pad tree has three rows");
