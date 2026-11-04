// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of the rounds of the sumchecks of a sub-proof (`fork` in identifiers).
   These are PSC, the sumcheck with a hiding polynomial, implemented in verifier/src/vf_sumcheck_psc.h,
   which verifier/src/vf_subproof_engine.h includes before this file. The prover's side is
   prover/hiding/sumcheck_door.cuh. The prover sends the hiding polynomial's sum, the rounds and, if
   end values (the summand's maps at the challenges) are sent, a round in a further variable Y, then
   those values. `pscEnds` reads the round in Y; every other round must go through `roundIn`. A
   round checks no equation: the equation for the running claim supplies the coefficient that is not
   sent. A sumcheck's only test is the linear claim from `pscRow`. A message slot (`slot`) is the
   index of a sent scalar in the sub-proof. */
#pragma once

/* How a round sums its variable, and the degree of its polynomial g. c is the running claim.
     EQ2    degree 2; c = (1 - z) g(0) + z g(1): g omits the summand's factor eq(p, x), and z is the
            coordinate of p for this round. Product layers, token sumchecks, a lookup's bit test.
     SUM2   degree 2; c = g(0) + g(1). `stackRounds`, and the merge of weight claims.
     SUM3   degree 3; c = g(0) + g(1). Attention's `sc3In`.
     NODES  degree `degree`; c = g(0) + g(1). Attention's `scdIn`.
     EQ     degree `degree`; c as for EQ2. The leaves of a lookup's table side: degree 2 or 1.
   The prover sends g's coefficients, lowest first, except the one that the equation for c fixes:
   the constant for EQ2 and EQ, else the linear one. The names follow the prover's `psc::Form`
   (prover/hiding/psc_device.cuh).
   Source: a sumcheck round whose polynomial g omits the equality factor, with the relation
   c = (1 - z) g(0) + z g(1): Gruen, "Some Improvements for the PIOP for ZeroCheck", Cryptology
   ePrint Archive 2024/108, sections 3.2 and 3.3. There the claimed sum is 0, and the prover sends
   values of g, not coefficients. */
enum class RoundForm { EQ2, SUM2, SUM3, NODES, EQ };
static constexpr u32 roundValues(RoundForm form, int degree = 0) {
  return form == RoundForm::SUM3 ? 3u : form == RoundForm::NODES || form == RoundForm::EQ ? (u32)degree : 2u;
}
/* The slots of a sumcheck with `rounds` rounds of one form and `ends` end values. The round in Y
   takes degreeInY, the summand's degree in the maps with these values: 0 exactly if ends is 0
   (not checked). */
static constexpr u64 sumcheckSlots(RoundForm form, int rounds, u64 ends, int degreeInY, int degree = 0) {
  return 1 + (u64)rounds * roundValues(form, degree) + (u64)degreeInY + ends;
}
// The same for rounds of the form NODES or EQ whose degrees differ.
static u64 sumcheckSlotsDegrees(const vector<int> &degree, u64 ends, int degreeInY) {
  u64 n = 1 + (u64)degreeInY + ends;
  for (int d : degree) n += (u64)d;
  return n;
}
// The slots of a product layer with mu variables: two end values, degree 2 in Y.
static constexpr u64 layerSlots(int mu) {
  return sumcheckSlots(RoundForm::EQ2, mu, 2, 2);
}
// Returns the challenge; advances `slot`. Only EQ2 and EQ read z, only NODES and EQ read degree
// (1 to 64). Not to be called after pscEnds() on p.
static Ex roundIn(HFork &F, Tr &tr, u32 &slot, RoundForm form, Psc &p, Ex z = eZ(), int degree = 0) {
  if (p.ended) refuse(F.G_ + "internal-psc-order");
  const bool equality = form == RoundForm::EQ2 || form == RoundForm::EQ;
  if ((form == RoundForm::NODES || form == RoundForm::EQ) && (degree < 1 || degree > 64)) refuse(F.G_ + "internal-psc-degree");
  const unsigned d = roundValues(form, degree) + 1;
  return pscRound(tr, slot, p, equality ? blk::Weight<ExFieldAdapter>::equality({z}, d) : blk::Weight<ExFieldAdapter>::plain(d),
                    [&] { return trAnswer(tr, RoundName::PSC_ROUND); });
}
