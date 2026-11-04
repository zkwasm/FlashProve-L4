// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of the token sumcheck (TSC); the prover's is
   prover/hiding/token_sumcheck.cuh.

   The claim is Cl = sum_t eq(P, t) s(t) over the token positions t < 2^lt, where the summand s is
   a sum of products of functions of t ("maps"). P must have lt coordinates: not checked. The
   prover's first message is Cl, masked; the masked sumcheck (PSC) of verifier/src/vf_sumcheck_psc.h starts
   from it.
     tscRounds   Cl, the hiding polynomial's sum, lt rounds of two scalars. The caller runs
                   further rounds and the end, and sets `sEnd`.
     tscIn       the same, and the end (`pscEnds`): one more round, in a further variable Y
                   (degreeInY scalars), then `shown` masked end values (maps' values at the
                   challenges). Y is there for the masks: a map f whose end value is sent becomes
                   f + c Y, c its mask entry.
   The caller then calls `pscRow` with S0 and `kept`. S0 is the summand at the challenges,
   computed on the end values as sent, without any terms in maps whose values are not sent. `kept`
   is those terms, as an affine form (`Aff`). `pscRow` returns the one linear claim that the
   sumcheck leaves; that claim is the sumcheck's only test. The caller adds it (`HFork::lin`).
   For its second token sumcheck, verifier/src/vf_layers_check.inc instead solves the claim for `kept`,
   which enters a later sumcheck's start. Nothing here checks what Cl and the end values are:
   further claims of the caller must.
   A message slot (`slot`) is a sent scalar's number within the sub-proof (`fork` in identifiers).
   verifier/src/vf_subproof_engine.h includes this file, after verifier/src/vf_sumcheck_receive.h. */
#pragma once
// From prover/, shared with the prover: the output head's rounds and a token sumcheck's for the
// proof's plan (its list of rounds). verifier/src/vf_output_head_plan.inc uses the head's; it is included
// inside a function, so the header is included here instead.
#include "head/plan_rounds.h"

struct TokenSumcheckState {
  u32 sCl = 0;  // the slot of Cl
  u32 sEnd = 0; // the slot of the first end value
  vector<Ex> r; // the challenges, coordinate 0 first
  Psc p;
  Aff rowU; // the claim from `pscRow`; empty unless the caller stores it
};
static TokenSumcheckState tscRounds(HFork &F, Tr &tr, u32 &slot, const Ex *P, int lt) {
  TokenSumcheckState o;
  o.sCl = F.claimsIn(tr, slot, 1);
  o.p = pscBegin(F, tr, slot, affSlot(o.sCl));
  o.r.resize((size_t)lt);
  for (int j = 0; j < lt; j++) o.r[(size_t)j] = roundIn(F, tr, slot, RoundForm::EQ2, o.p, P[j]);
  return o;
}
static TokenSumcheckState tscIn(HFork &F, Tr &tr, u32 &slot, const Ex *P, int lt, u32 shown, int degreeInY) {
  TokenSumcheckState o = tscRounds(F, tr, slot, P, lt);
  o.sEnd = pscEnds(F, tr, slot, o.p, degreeInY, shown);
  return o;
}
// A token sumcheck's number of slots. `more`: the caller's further rounds, of two scalars each.
static constexpr u64 tscSlots(int lt, u64 shown, int degreeInY, int more = 0) {
  return 1 + sumcheckSlots(RoundForm::EQ2, lt + more, shown, degreeInY);
}
