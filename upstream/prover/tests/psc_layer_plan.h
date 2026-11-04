// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The round plan (`blk::Plan`) under which prover/tests/psc_layer_test.cu proves two product layers
// in a row and verifier/tests/psc_layer_check.cpp checks the proofs. Plan, label and configuration
// enter the hash chain's first query, so both programs must agree on them. The last round batches
// three linear claims: one from each layer's sumcheck, and one that the second layer passes on.
// mu: the first layer's variable count; the second has one more.
#pragma once
#include "hash/hash.h"
#include "prover/plan_rounds.h"

// `blk::Binder` needs a label (naming a protocol) and a configuration digest, both synthetic here.
constexpr blk::u64 PSC_LAYER_TEST_LABEL = 6;
inline blk::Digest psc_layer_test_configuration() {
  blk::Digest configuration{};
  configuration.w[0] = 0x70736c79u;
  return configuration;
}
inline void plan_psc_layer_test(blk::Plan &p, unsigned mu) {
  blk::plan_zero(p, "first layer's point", mu);
  blk::plan_product_layer(p, "first layer", mu);
  blk::plan_product_layer(p, "second layer", mu + 1);
  blk::plan_bat_sigma(p, "claims", 3);
}
