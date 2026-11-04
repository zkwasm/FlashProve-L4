// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The rounds of a layer sub-proof (one for each group of layers) after its lookup and before the
// root of its next oracle message, the reads (weighted sums of committed cells: blocks/iop/rng.h).
// plan_layer_own() enters them into the `Plan` of the request's first chain: the list of that hash
// chain's rounds and their errors (blocks/numbers/plan.h). The order must match the messages in
// verifier/src/vf_layers_check.inc and the steps in prover/prover/layer_prove.cuh. The prover sends only
// scalars and the verifier elements of the field E or nothing, so a GPU kernel can run these
// rounds on a copy of the chain (ProverRounds::lend(), blocks/chain/rounds.h). Prover and verifier
// compile this file: no CUDA.
#pragma once
#include <string>

#include "numbers/blocks.h"

namespace blk {

// prover/prover/layer_setup.cuh and verifier/src/vf_layers_plan.inc fill these sizes and must agree.
// "Bits of n": log2 of n. A sumcheck (`psc` in identifiers) "shows" the values that it sends,
// masked, at its last point.
// TSC(P_V), TSC(zt): sumchecks over the token bits, at the norms' token point and at zt (below).
// R: a norm's scaled reciprocal. sc: a routed layer's score. Selector: one entry per expert.
struct LayerRounds {
  unsigned lt = 0;            // token bits
  unsigned ls = 0;            // bits of a block of features in a long read
  unsigned lNI = 0;           // bits of the norm instances' number
  unsigned lPd = 0;           // bits of the padded model width
  unsigned ll = 0;            // bits of the layers' number in the group
  unsigned lPf = 0;           // bits of the padded hidden width
  unsigned lfe = 0;           // bits of a routed expert's hidden width
  unsigned lamS = 0;          // rounds of the batched inner-product sumcheck
  unsigned lkW = 0;           // rounds of the merge of weight claims into one
  u64 NI = 0;                 // norm instances
  u64 NS = 0, NP = 0, NO = 0; // for each instance: sums of limb products, row products, row values that TSC(P_V) shows
  bool with_terms = true;     // the call list has terms for the second token sumcheck, TSC(zt)
  u64 shown_by_the_terms = 0; // R per instance; with routed experts sc and the selector's entries
  u64 never_sent = 0;         // values of never-sent vectors that the batched sumcheck shows: 1 + the routed layers
  u64 weight_claims = 0;    // weight claims, without the embedding's
  bool first_group = false; // the group that also proves the embedding's claim
  bool routed = false;      // the group has layers with routed experts
};

inline void plan_product_layer(Plan &p, const std::string &where, unsigned variables) {
  PscSize x;
  x.degree.assign(variables, 3), x.degree_Y = 2, x.values = 2;
  plan_psc(p, where, x);
  plan_line(p, where);
}
inline void plan_token_sumcheck(Plan &p, const std::string &where, unsigned lt, unsigned degree_Y, u64 shown) {
  plan_scalars(p, "claimed sum", where, 1);
  PscSize x;
  x.degree.assign(lt, 3), x.degree_Y = degree_Y, x.values = (unsigned)shown;
  plan_psc(p, where, x);
}
// An error event without a round of its own: for each layer with routed experts, the identity
// "the selector's entries sum to 1 at every token" is tested at zt, the last lt coordinates of the
// lookup's key point. The rounds that draw these coordinates carry its error: the sumcheck rounds
// of the lookup's last key level (error numerator 2, over |E|) and that level's plan_line() round
// (numerator 1), if its challenge is one of the coordinates.
// The rule, round by round: a malicious prover has lost while some false relation stays false, so
// a challenge rescues it only if it rescues every false relation at once. Across these relations a
// round's error is the largest of their errors, not their sum.
// Its limit: within one relation's chain of reductions the errors add wherever one verifier
// message gives it several ways out: n for a point of n coordinates, the number of claims less one
// for batching by powers, both terms for positions that all miss or a batching challenge that
// cancels.
// Three conditions must hold; each is argued below, not proved. No code checks them: anyone who
// moves the selector into a later message, or tests the identity at a point with a fixed or
// repeated coordinate, must check them again.
// (1) The multilinear polynomial sum_e sel_e(X) - 1 is fixed before the first of these coordinates
//     is drawn. The selector's cells are the one small unpacking of packed entries of the
//     sub-proof's first prover message, whose out-of-domain sample is answered before the lookup's
//     first draw. That message denotes the one member of its decoding list with the answered
//     sample: the round of the sample's seed, and the list's factor where the rows are batched,
//     account for that. If its entries have no small unpacking, the approximate range check's
//     relation is the false one.
// (2) Each coordinate is the challenge of a planned round whose numerator is at least 1, the
//     polynomial's degree in it.
// (3) A false identity at zt leads to a rejection by planned rounds. It is one of the sub-proof's
//     linear claims, not on the cells but on the reads, which are sent after zt is drawn. Their
//     sample is answered before the range check's matrix is drawn, and that matrix's round catches
//     reads that are not the cells'. Powers of one element batch the check's equations and the
//     claims, and that element's round counts them all.
inline void plan_layer_own(Plan &p, const std::string &where, const LayerRounds &x) {
  plan_zero(p, where + ", the norms' token point", x.lt);
  if (x.first_group) {
    plan_zero(p, where + ", the embedding's point", x.ls + x.lt);
    plan_zero(p, where + ", the embedding's feature point", x.lPd - x.ls);
  }
  plan_scalars(p, "sums of limb products", where, x.NS * x.NI);
  plan_bat_sigma(p, where + ", limbs", x.NS);
  if (x.lNI) plan_join(p, where + ", norm instances", x.lNI);
  plan_product_layer(p, where + ", sums of squares", x.lNI + x.lPd + x.lt);
  plan_scalars(p, "row products", where, x.NP * x.NI);
  plan_bat_sigma(p, where + ", row products", x.NP * x.NI);
  plan_token_sumcheck(p, where + ", row products", x.lt, 2, x.NO * x.NI);
  if (x.with_terms) {
    plan_token_sumcheck(p, where + ", terms", x.lt, 1, x.shown_by_the_terms);
    plan_bat_sigma(p, where + ", start of the inner products", 2);
  }
  {
    PscSize s;
    s.degree.assign(x.lamS, 3), s.degree_Y = 2, s.values = (unsigned)(2 + x.never_sent + x.weight_claims);
    plan_psc(p, where + ", inner products", s);
  }
  plan_product_layer(p, where + ", gated products", x.ll + x.lPf + x.lt);
  if (x.routed) {
    if (x.ll > 1) plan_join(p, where + ", layers with routed experts", x.ll - 1);
    plan_product_layer(p, where + ", routed gated products", x.ll - 1 + x.lfe + x.lt);
  }
  if (x.first_group) plan_scalars(p, "the embedding's value", where, 1);
  plan_bat_sigma(p, where + ", weight claims", x.weight_claims + (x.first_group ? 1 : 0));
  {
    PscSize m; // the merge's whole summand counts as one map, which is shown; the claim on that
               // value leaves the sub-proof (verifier/src/vf_weights_merge.h)
    m.degree.assign(x.lkW, 3), m.degree_Y = 1, m.values = 1;
    plan_psc(p, where + ", weight claims", m);
  }
}

} // namespace blk
