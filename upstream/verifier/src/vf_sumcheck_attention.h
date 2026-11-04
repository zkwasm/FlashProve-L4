// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of SC3 (rounds of degree 3) and SCD (a degree per round), two types of
   sumcheck in the attention sub-proof (`Fork` in identifiers; verifier/src/vf_attention_check.h). Both run
   PSC, the sumcheck with a hiding polynomial (verifier/src/vf_sumcheck_psc.h). Before `sc3In` or
   `scdIn` the caller must have received every message and sampled every challenge on which the
   summand or `start`, the affine form that the sum must equal, depends (checked only in part, by
   `pscBegin`). Then it must add `rowU`, the only test of the sum, with `HFork::lin`, and tie
   each end value (a map's value at the challenges) to its map, by a claim or in a later `start`.
   prover/attn/plan_rounds.h lists the rounds; the prover's side is prover/attn/engines*.cuh. Kept
   separate so that a test that is not part of this repository can use it without the sub-proof. */
#pragma once
#include "vf_subproof_engine.h"

// SC3 proves start = sum_b sum_x W_b(x) A_b(x) B_b(x), b < nb, x in {0,1}^nv, with public W_b.
// Message slots, one per scalar sent, in order: the hiding polynomial's sum; 3 coefficients per
// round; 2 for the last round, in a further variable Y; from sEnd on, the end values A_0(r),
// B_0(r), .. at the challenges r.
struct Degree3SumcheckState {
  vector<Ex> r;
  u32 sEnd = 0;
  size_t nb = 0;
  Psc p;
  Aff rowU; // the one claim that the sumcheck leaves (sc3Row)
};
static u64 sc3Slots(int nv, size_t nb) {
  return sumcheckSlots(RoundForm::SUM3, nv, 2 * (u64)nb, 2);
}
// Moves `slot` past the sumcheck's slots.
static Degree3SumcheckState sc3In(HFork &F, Tr &tr, u32 &slot, int nv, size_t nb, const Aff &start) {
  Degree3SumcheckState o;
  o.r.resize(nv);
  o.nb = nb;
  o.p = pscBegin(F, tr, slot, start);
  for (int j = 0; j < nv; j++) o.r[j] = roundIn(F, tr, slot, RoundForm::SUM3, o.p);
  o.sEnd = pscEnds(F, tr, slot, o.p, 2, (u32)(2 * nb));
  return o;
}
// The hidden end values A_b(r) and B_b(r): what was sent, without the masks.
static Aff sc3A(const Degree3SumcheckState &o, size_t b) {
  return affSlot(o.sEnd + (u32)(2 * b));
}
static Aff sc3B(const Degree3SumcheckState &o, size_t b) {
  return affSlot(o.sEnd + (u32)(2 * b + 1));
}
// Sets o.rowU. W[b] is W_b(r). The summand is evaluated on the scalars sent for the end values
// (F.shown()), not on the hidden values.
static void sc3Row(HFork &F, Degree3SumcheckState &o, const vector<Ex> &W, const string &nm) {
  if (W.size() != o.nb) refuse("internal-sc3-" + nm);
  Ex s0 = eZ();
  for (size_t b = 0; b < o.nb; b++)
    s0 = s0 + W[b] * F.shown(o.sEnd + (u32)(2 * b)) * F.shown(o.sEnd + (u32)(2 * b + 1));
  o.rowU = pscRow(F, o.p, s0);
}
// SCD: round j has degree d[j], 1 to 64. Slots as for SC3, with d[j] coefficients for round j,
// degreeInY (the summand's degree in the maps with end values) for the round in Y, and n end
// values. The caller evaluates the summand and sets rowU with pscRow().
struct VariedDegreeSumcheckState {
  vector<Ex> r;
  u32 sEnd = 0;
  Psc p;
  Aff rowU;
};
static VariedDegreeSumcheckState scdIn(HFork &F, Tr &tr, u32 &slot, const vector<int> &d, int degreeInY, size_t n,
                       const Aff &start) {
  VariedDegreeSumcheckState o;
  o.r.resize(d.size());
  o.p = pscBegin(F, tr, slot, start);
  for (size_t j = 0; j < d.size(); j++) o.r[j] = roundIn(F, tr, slot, RoundForm::NODES, o.p, eZ(), d[j]);
  o.sEnd = pscEnds(F, tr, slot, o.p, degreeInY, (u32)n);
  return o;
}
