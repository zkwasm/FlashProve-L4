// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The type of a weight table's making key (`key` or `mask_seed` in identifiers; elsewhere also mask
// seed, kM): the 32 bytes from which the masks of the table's rows, the salts of its Merkle tree's
// leaves and its rows' pads are generated. A weight table holds all the
// weights; each row is encoded with its masks, extra coefficients meant to hide it.
// prover/common/coin_keys.h derives the key from the commitment's seed (the provider's 32 secret
// bytes): first_table_key() for the anchor (the first weight table); copy_table_key() for a leaf
// table (a further table of the same weights), with 32 bytes that the committer program `cp_commit`
// draws each time it generates one (zeros under `-test-seed` and under `-public-weights`).
// Namespace hf re-exports seed types and functions of prover/hash_commitment/hc_seed.h.
//
// A making key must serve one table only: two tables with the same making key differ by an unmasked
// encoding of the difference of their data. Keyed with a making key, the generator f (the ChaCha20
// block function) must run only on the inputs that prover/hash_commitment/hf_mask.cuh and
// hf_pads.cuh define, whose fourth word is "MASK", "SALT" or "PADS". A making key must not be
// stored in a table or enter a hash input.
#pragma once
#include "hash_commitment/hc_chacha.cuh" // hc_seed32, the generator f
#include "hash_commitment/hc_seed.h"     // how a seed is drawn, read, compared and wiped

// A struct, not an alias of hc_seed32, so that other headers can forward-declare it. Like hc_seed32
// it is a dedicated type, so that no array of words or unrelated key type compiles where a making
// key is expected. `cp_commit` also stores the commitment's seed in it (`Seed` in hf_commit.cu).
struct hf_key32 : hc_seed32 {};
static_assert(sizeof(hf_key32) == 32 && std::is_trivially_copyable<hf_key32>::value,
              "the key is its 32 bytes, and can be a kernel's argument");

namespace hf {
using hc::SeedEnd;
using hc::SeedSource;
using hc::seed_draw;
using hc::seed_from_file;
using hc::seed_is_of_the_tests;
using hc::seed_is_zero;
using hc::seed_machine_source;
using hc::seed_says;
using hc::seed_stop;
using hc::seed_wipe;
} // namespace hf
