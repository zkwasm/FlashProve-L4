// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The token sumcheck's rounds (prover/hiding/token_sumcheck.cuh, verifier/src/vf_sumcheck_token.h) as entries
// of `Plan`, the list of a proof's rounds. Prover and verifier compile this file. Needs -I blocks.
#pragma once
#include <string>

#include "numbers/blocks.h"

namespace blk {

// Adds the claimed sum, a hidden value sent masked, then its masked sumcheck (PSC) under eq
// weights: lt rounds over 2^lt token positions, then `more` rounds over an index with equal
// weights. A round polynomial omits its eq factor: degree below 3, two scalars if
// `Setting::saved_coefficient`. If shown > 0, a round in a further variable Y and `shown` end
// values follow, then no challenge. degree_in_y: the summand's degree in the maps whose end values
// are sent.
inline void plan_token_sum(Plan &p, const std::string &where, unsigned lt, unsigned shown, unsigned degree_in_y,
                           unsigned more = 0) {
  plan_scalars(p, "token sum", where, 1);
  PscSize x;
  x.degree.assign(lt + more, 3), x.degree_Y = degree_in_y, x.values = shown, x.with_Y = shown > 0;
  plan_psc(p, where, x);
}

} // namespace blk
