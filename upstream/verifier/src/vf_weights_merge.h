// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The merge reduces the claims of a layer or output-head sub-proof (`fork` in identifiers) on the
// weight table to one. The claim with index a says that a linear form of the weights equals v_a.
// The prover has sent v_a earlier, masked, either as an end value of a sumcheck that reads the
// weights or separately.
//
// Messages (wsForkMerge()): one challenge, whose powers rho_a batch the claims; then the masked
// sumcheck PSC (verifier/src/vf_sumcheck_psc.h) of sum_a rho_a v_a, with a round for each of the table's lk
// column bits, whose challenges form the point rm. In PSC a summand is a polynomial in maps
// (functions of the summed variables); the maps' values at the challenges are the end values. Here
// the whole summand counts as one map (hence the arguments 1, 1 below: degree 1, one end value).
// Its value fv at rm is sent masked: fv' = fv + beta c (beta: PSC's last challenge, after rm).
//
// "fv + beta c = fv'" leaves the sub-proof as a pending claim on the weight table, for LIN, the
// test on that table. c is an entry of the request's index-mask message (its masks for such
// claims), not of the sub-proof's mask row (its committed masks, one for each of its sent scalars).
// This file does not build that claim: it returns fvp, beta, rm, and wsPending()
// (verifier/src/vf_weights_pending_claim.h) builds all of it but c.
// Include verifier/src/vf_subproof_engine.h first.
// Prover's side: prover/head/claims_lib.cuh, CEnd::send_merge().
#pragma once

struct WsMerged {
  Ex fvp = eZ();  // fv', as it was sent
  Ex beta = eZ(); // the factor of its mask c
  vector<Ex> rm;  // the merge's point, the highest column bit first
  Psc p;
  u32 sFv = 0; // the slot of fv': its number among the sub-proof's sent scalars. No claim of the
               // sub-proof may name it: the mask row's entry with that number masks nothing.
               // `wsForkLin`'s callers test this (`internal-g-free-slot`, `internal-h-free-slot`)
};
// sv[a]: the slot of v_a, already received (else pscBegin() rejects with `internal-slot-order`).
// Output: rho[a] = rho_a.
static WsMerged wsForkMerge(HFork &F, Tr &tr, u32 &slot, int lk, const vector<u32> &sv,
                              vector<Ex> &rho) {
  WsMerged o;
  rho = trBatch(tr, sv.size());
  Aff start;
  for (size_t a = 0; a < sv.size(); a++) affAdd(start, affSlot(sv[a]), rho[a]);
  o.p = pscBegin(F, tr, slot, start);
  o.rm.resize(lk);
  for (int i = 0; i < lk; i++) o.rm[i] = roundIn(F, tr, slot, RoundForm::SUM2, o.p);
  o.sFv = pscEnds(F, tr, slot, o.p, 1, 1);
  o.fvp = F.shown(o.sFv);
  o.beta = o.p.r.back().v;
  return o;
}
// Adds PSC's one remaining claim, the sub-proof's last, on the sub-proof's mask entries only.
static void wsForkLin(HFork &F, WsMerged &o) {
  F.lin(pscRow(F, o.p, o.fvp));
}
// How many slots wsForkMerge() takes (one for fv', none for the v_a).
static size_t wsForkSlots(int lk) {
  return (size_t)sumcheckSlots(RoundForm::SUM2, lk, 1, 1);
}
