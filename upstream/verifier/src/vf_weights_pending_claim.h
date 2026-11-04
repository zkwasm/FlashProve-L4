// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Builds the pending claim of a sub-proof (`Fork` in identifiers) that has claims on the weights.
   The weight table holds the committed weight matrix W, and LIN (blocks/iop/lin.h) tests linear
   claims on it. The sub-proof leaves LIN one such claim:
       < Gamma (x) eq(rm, .), W > + beta c = fv'.
   Gamma, one weight for each row of W, is the sum of the row weights of the claims in
   WsForkOut::rec. c: the sub-proof's mask entry, one random element of the request's index-mask
   message (an oracle message of mask rows only; blocks/iop/member.h).
   `wsForkMerge` (verifier/src/vf_weights_merge.h) gives rm (a point on W's columns), the
   challenge beta and fv'. Its sumcheck's end value is fv, the claim's first term, which the
   prover sends masked: fv' = fv + beta c. */
#pragma once
#include "vf_weights_layout.h"
#include "vf_subproof_segment.h" // toBlkE
#include "iop/member.h"

// The claim without c, as arguments for blk::member_pending_row(). The caller takes c's
// `blk::MaskSlot` from the index-mask message's `blk::MaskLedger`. target: fv'.
struct WsPending {
  std::vector<blk::Tensor> g; // the row weights are their sum
  std::vector<blk::E> psi;    // the point on the columns, the coordinate of column bit 0 first
  blk::E beta, target;
};
// g stores Gamma on the rows from the claims' lowest unit (`WsUnit`: a run of W's rows) through
// their highest, gaps between units included; elsewhere Gamma is 0.
// claims: the size that o.rec must have: the number of weight claims in the sub-proof's plan
// (field NCL), which follows from the parameter file (policy.txt).
// LIN ignores row weights from row S.matrix_rows on, so the layout `lay` must have exactly that
// many rows.
static WsPending wsPending(const WsLay &lay, const WsForkOut &o, size_t claims, const blk::LinShape &S) {
  const unsigned rowBits = S.k_row();
  if (lay.m0 != S.matrix_rows) refuse("internal-gamma-rows");
  if (o.rec.empty() || o.rec.size() != claims || o.rm.size() != (size_t)lay.lk) refuse("link-claim-count");
  u64 first = ~0ull, last = 0;
  for (auto &c : o.rec) {
    if (c.ui >= lay.U.size()) refuse("internal-gamma-rows");
    first = std::min<u64>(first, lay.U[c.ui].off);
    last = std::max<u64>(last, lay.U[c.ui].off + lay.U[c.ui].rows);
  }
  if (first >= last || last > lay.m0 || rowBits > 62 || lay.m0 > (1ull << rowBits)) refuse("internal-gamma-rows");
  vector<Ex> Gam(last, eZ());
  for (auto &c : o.rec) wsGammaAdd(Gam, lay, c, eO());
  auto values = std::make_shared<vector<blk::E>>(last - first);
  for (u64 y = first; y < last; y++) (*values)[y - first] = toBlkE(Gam[y]);
  WsPending p;
  p.g = blk::member_weights_on_rows(S, first, std::move(values));
  p.psi.resize(o.rm.size()); // rm has the highest column bit first
  for (size_t i = 0; i < p.psi.size(); i++) p.psi[i] = toBlkE(o.rm[p.psi.size() - 1 - i]);
  p.beta = toBlkE(o.beta), p.target = toBlkE(o.fvp);
  return p;
}
