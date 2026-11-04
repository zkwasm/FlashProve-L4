// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of PSC, the masked sumcheck defined in blocks/iop/psc.h, for the
   sub-proofs (`fork` in identifiers; each argues one part of the forward pass). The prover's side
   is prover/hiding/psc_device.cuh. The weights of a variable (how it is summed) and the hiding
   polynomial's linear form are taken from blocks/iop/hp.h. verifier/src/vf_subproof_engine.h includes this
   file after its `HFork` and `Aff`.

   A sumcheck proves that a sum equals a hidden value, given as an affine form C (`Aff`). It runs
   on a hiding polynomial U plus eps, a challenge, times the summand, so U's sum and the rounds'
   coefficients are sent without masks. A round tests no equation: its check fills in the
   coefficient that is not sent. Calls, in order:
     pscBegin   reads U's sum, draws eps
     roundIn    (verifier/src/vf_sumcheck_receive.h) a round; it calls pscRound
     pscEnds    if end values (the summand's maps at the challenges) are shown: the last round,
                  in a further variable Y, then the end values under masks
     pscRow     the one linear claim that is left, and the only test */
#pragma once
#include "iop/hp.h"

// Ex (extension-field element) as the field type that blocks/iop/hp.h requires.
struct ExFieldAdapter {
  Ex v = eZ();
  static ExFieldAdapter zero() { return {eZ()}; }
  static ExFieldAdapter one() { return {eO()}; }
  friend ExFieldAdapter operator+(ExFieldAdapter a, ExFieldAdapter b) { return {a.v + b.v}; }
  friend ExFieldAdapter operator-(ExFieldAdapter a, ExFieldAdapter b) { return {a.v - b.v}; }
  friend ExFieldAdapter operator*(ExFieldAdapter a, ExFieldAdapter b) { return {a.v * b.v}; }
  friend bool operator==(ExFieldAdapter a, ExFieldAdapter b) { return a.v == b.v; }
};
static ExFieldAdapter inv(ExFieldAdapter a) {
  return {eInv(a.v)};
}

// One sumcheck's state.
// Message slots: a sub-proof's scalars are numbered in file order. A sumcheck uses the entries of
// the committed mask row that have its slots' numbers: first U's coefficients, in the order of
// blk::hp_place() (by variable, then by degree; only the last variable has a constant term), then
// one for each end value, so that an end value and its mask entry have the same number.
// Hidden values: affSlot(j) is shown[j] - factor[j] * mask[j]. An end value has the factor beta; a
// coefficient of U has shown 0 and factor -1.
struct Psc {
  Aff start;          // what the sum must equal
  Ex target = eZ();     // start, each hidden value replaced by what was shown, `Aff::e` dropped
  Ex eps = eZ();
  Ex claim = eZ();      // the running claim of the masked sumcheck
  u32 first = 0;        // the number of its first slot and first mask entry
  vector<blk::Weight<ExFieldAdapter>> vars;  // the variables so far, in the order of the rounds
  vector<ExFieldAdapter> r;                  // their challenges
  bool ended = false;   // the round in the further variable has been run
  bool rowMade = false;
  bool started = false; // start is set
};

// eps must come after everything that C, the summand and the variables' weights depend on: that is
// the caller's duty. Of that, pscBegin() checks only that each slot that occurs in C already has
// its hidden value (`internal-slot-order`).
static Psc pscBegin(Tr &tr, u32 &slot, Ex target);
static Psc pscBegin(HFork &F, Tr &tr, u32 &slot, const Aff &C) {
  Psc p = pscBegin(tr, slot, F.publicPart(C));
  p.start = C;
  p.started = true;
  return p;
}
// Any `target` that is fixed before eps is sound, since the claim of pscRow() has the term
// P (target - start). The overload above sets `start`; without it pscRow() ends the run
// (`internal-psc-order`: a fault of this program, which cp_verify reports as NOT CHECKED).
static Psc pscBegin(Tr &tr, u32 &slot, Ex target) {
  Psc p;
  p.target = target;
  p.first = slot;
  const Ex mu = trSum(tr);
  p.eps = trAnswer(tr, RoundName::PSC_EPSILON);
  p.claim = mu + p.eps * p.target;
  slot += 1;
  return p;
}
/* One round in a variable with the weight w. The message: the coefficients of the round polynomial
   q, lowest degree first, except degree `w.fixed`, which the check w(q) = claim fills in. */
template <class Draw> static Ex pscRound(Tr &tr, u32 &slot, Psc &p, const blk::Weight<ExFieldAdapter> &w, Draw draw) {
  const vector<Ex> sent = trRead(tr, w.d - 1);
  vector<ExFieldAdapter> q(w.d);
  for (unsigned t = 0, i = 0; t < w.d; t++)
    if (t != w.fixed) q[t] = {sent[i++]};
  w.complete(q, {p.claim});
  const Ex r = draw();
  p.claim = blk::poly_at(q, ExFieldAdapter{r}).v;
  p.vars.push_back(w);
  p.r.push_back({r});
  slot += w.d - 1;
  return r;
}
// degreeInY: the summand's degree in the maps whose end values are shown.
// Messages: the round in Y (degreeInY scalars), beta, then the n end values as one message.
// Returns their first slot e: F.shown(e + j) is what was sent, affSlot(e + j) the hidden end value.
static u32 pscEnds(HFork &F, Tr &tr, u32 &slot, Psc &p, int degreeInY, u32 n) {
  if (p.ended || degreeInY < 1 || n < 1) refuse(F.G_ + "internal-psc-order");
  // With beta = 0 the end values would be sent without their masks.
  const Ex beta = pscRound(tr, slot, p, blk::Weight<ExFieldAdapter>::at_zero((unsigned)degreeInY + 1),
                             [&] { return trAnswerNotZero(tr, RoundName::PSC_Y); });
  p.ended = true;
  const u32 e = slot;
  const vector<Ex> v = trRead(tr, n);
  trEmpty(tr, RoundName::PSC_VALUES);
  for (u32 j = 0; j < n; j++) F.hide(e + j, v[j], beta);
  slot += n;
  return e;
}
/* The claim that a sumcheck leaves, as a form that must be 0:
     <form(challenges), U's coefficients> + P (target - start) + eps kept - (claim - eps S0).
   P: eps times the product of the variables' blk::Weight::psi at the challenges. S0: the summand at
   the challenge point on the shown end values, without its terms in maps whose values are not
   shown. kept: those terms, sum_i A_i * (the map's value as a form), A_i public.
   The caller adds the form with `HFork::lin` or builds it into another claim; nothing here checks
   that. `doorH` (verifier/src/vf_subproof_segment.h) passes the claims to the request's segment chain,
   which tests them at its end. */
static Aff pscRow(HFork &F, Psc &p, Ex S0, const Aff &kept = Aff()) {
  if (p.rowMade || !p.started || p.vars.empty()) refuse(F.G_ + "internal-psc-order");
  p.rowMade = true;
  const vector<ExFieldAdapter> form = blk::hp_form(p.vars, p.r);
  Aff u = affC(eZ() - (p.claim - p.eps * S0));
  for (size_t i = 0; i < form.size(); i++) {
    F.hide(p.first + (u32)i, eZ(), eZ() - eO());
    affAdd(u, affSlot(p.first + (u32)i), form[i].v);
  }
  Ex P = p.eps;
  for (size_t t = 0; t < p.vars.size(); t++) P = P * blk::poly_at(p.vars[t].psi, p.r[t]).v;
  affAdd(u, affC(p.target) - p.start, P);
  affAdd(u, kept, p.eps);
  return u;
}
