// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// BATT batches J linear claims (`TensorRow`; "row" in identifiers and messages) on the committed
// matrix M, each at its own column point, into one claim at a fresh column point. M is not sent in
// a proof, and nothing here reads it: it is in the committed table. LIN (blocks/iop/lin.h), the
// protocol for claims on M, takes one per request, so blocks/iop/member.h runs BATT if J > 1.
// In:  J >= 1 claims; g_j, psi_j, tau_j public; (x): tensor product:
//        < g_j (x) eq(psi_j, .), M >  +  (mask entries)  =  tau_j.
//      g_j weights M's rows, eq(psi_j, .) its 2^k_col columns; k_col >= 1. Mask entries: random
//      elements of the challenge field E in an oracle message's mask rows (blocks/iop/mask.h).
// Out: one claim of the same form at a point gamma, with g = sum_j a_j g_j for public a_j.
//
// The caller's duty, not checked here (the rule of blocks/chain/public.h): the batching challenge x
// comes after everything that enters a claim. g_j, psi_j, tau_j and the choice of mask entries must
// follow from the first query of the hash chain (the Fiat-Shamir transcript), from what the prover
// has sent on the chain, or from the verifier's challenges on it. A prover that can choose a tau_j
// after x can make two false claims cancel.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "field/mle.h"
#include "iop/mask.h"
#include "iop/psc.h"
#include "iop/summand.h"
#include "iop/tensor.h"
#include "numbers/blocks.h"
#include "numbers/round_names.h"

namespace blk {

// < g (x) eq(psi, .), M > + <masks> = target.
struct TensorRow {
  std::vector<Tensor> g;  // the row weights are their sum; all have the same number of variables
  std::vector<E> psi;     // one coordinate for each column bit
  std::vector<MaskTerm<E>> masks;
  E target = E::zero();
};

// Number of scalars that one BATT sends. It takes as many mask entries: they are the coefficients
// of the hiding polynomial of PSC (psc.h), the sumcheck that BATT runs.
inline size_t batt_masks(unsigned k_col) { return 1 + 2 * size_t(k_col); }

// What both sides compute from the claims and x: PSC's input. With chi_j = x^j and W_j = g_j^T M:
//   sum_c sum_j chi_j eq(psi_j, c) W_j(c)  +  sum_j chi_j (masks of claim j)  =  sum_j chi_j tau_j,
// c over the columns. The summand is affine in the maps W_j, so all are kept maps (summand.h): PSC
// sends no value W_j(gamma) and has no round for its variable Y, added only for sent values.
// batt_claim() moves the input claims' masks into `claim`.
struct BattClaim {
  std::vector<E> chi;
  std::vector<std::vector<E>> psi;
  SumClaim<E> claim;
  Summand<E> summand;
};
inline unsigned batt_k_col(const std::vector<TensorRow> &rows) {
  BLK_REQUIRE(!rows.empty() && !rows[0].psi.empty(), "BATT: no row, or a matrix of one column");
  return unsigned(rows[0].psi.size());
}
inline BattClaim batt_claim(std::vector<TensorRow> &rows, const E &x) {
  const unsigned k_col = batt_k_col(rows);
  BattClaim b;
  E power = E::one();
  b.claim.target = E::zero();
  for (TensorRow &r : rows) {
    BLK_REQUIRE(r.psi.size() == k_col, "BATT: the rows' points differ in length");
    BLK_REQUIRE(!r.g.empty(), "BATT: a row without weights");
    for (const Tensor &t : r.g) BLK_REQUIRE(t.vars() == rows[0].g[0].vars(), "BATT: not all weights have the same number of row variables");
    b.chi.push_back(power);
    b.claim.target += power * r.target;
    for (MaskTerm<E> &m : r.masks) b.claim.masks.push_back({std::move(m.slot), power * m.coeff});
    r.masks.clear();
    b.psi.push_back(r.psi);
    power = power * x;
  }
  b.claim.vars.assign(k_col, Weight<E>::plain(3));
  b.summand.shown = 0, b.summand.kept = unsigned(rows.size()), b.summand.degree = 0;
  b.summand.at = [chi = b.chi, psi = b.psi, k_col](const std::vector<E> &point, const std::vector<E> &) {
    std::vector<E> a{E::zero()};
    for (size_t j = 0; j < chi.size(); j++) a.push_back(chi[j] * eq_eval(psi[j].data(), point.data(), k_col));
    return a;
  };
  return b;
}
// PSC's output claim, (U) in psc.h, as a `TensorRow`: its terms a_j W_j(gamma), with
// a_j = eps chi_j eq(psi_j, gamma) and eps PSC's first challenge, are
// a_j < g_j (x) eq(gamma, .), M >. Its masks include the input claims'. It is computed from those
// claims, challenges and scalars sent through the channel, so it meets the duty at the top of this
// file if they do.
inline TensorRow batt_row(const std::vector<TensorRow> &rows, PscOutput<E> &&out) {
  TensorRow r;
  r.psi = std::move(out.point);
  for (const KeptTerm<E> &t : out.row_maps)
    for (Tensor g : rows[t.map].g) {
      g.scalar = g.scalar * t.coeff;
      r.g.push_back(std::move(g));
    }
  r.masks = std::move(out.row_masks);
  r.target = out.row_target;
  return r;
}

// The prover's costly part, which the caller supplies: the sumcheck without masks
// (`SumcheckProver`, summand.h) of BattClaim::claim over the kept maps W_j, in the claims' order.
// The caller also computes the W_j. chi and psi give the summand,
// sum_j chi[j] eq(psi[j], X) W_j(X), to an implementation that does not call Summand::at.
using BattSumcheck = std::function<std::unique_ptr<SumcheckProver<E>>(const Summand<E> &, const std::vector<Weight<E>> &, const std::vector<E> &chi,
                                                                      const std::vector<std::vector<E>> &psi)>;

// Draws x (round `BAT-sigma`). The round's planned error counts the claims: unless it has rejected
// the proof, the channel ends the process if the `Plan` (blocks/numbers/plan.h) has another number
// than `rows`.
template <class Ch>
E batt_draw(Ch &ch, size_t rows) {
  auto from = ch.close_algebraic(R_BAT_SIGMA);
  from.require_rows(rows);
  return from.draw();
}

// Messages. V: x. Then PSC: P: the hiding polynomial's sum; V: eps; k_col times, P: two
// coefficients of a polynomial of degree 2, V: a coordinate of gamma.
// Round errors: (J - 1) / |E| for x, 1 / |E| for eps, 2 / |E| for each of the k_col rounds, all by
// counting a polynomial's zeros; plan_batt() in blocks/numbers/blocks.h plans them. Unless it has
// rejected the proof, the channel ends the process at a round that is not the next planned one, for
// example under a `Plan` for another k_col.
//
// Ch: the channel (blocks/chain/rounds.h). ledger: gives out the batt_masks(k_col) mask entries,
// from one oracle message (for a request: its index-mask message); the process ends if it has
// fewer. masks: the prover's copy of all mask entries. a: if not null, gets the a_j, so that a
// prover that has the W_j can form (sum_j a_j g_j)^T M = sum_j a_j W_j without reading the matrix
// again.
template <class Ch>
TensorRow batt_prove(Ch &ch, std::vector<TensorRow> rows, MaskLedger &ledger, const MaskValues<E> &masks, const BattSumcheck &heavy,
                     std::vector<E> *a = nullptr) {
  const unsigned k_col = batt_k_col(rows);
  BattClaim b = batt_claim(rows, batt_draw(ch, rows.size()));
  const std::unique_ptr<SumcheckProver<E>> inner = heavy(b.summand, b.claim.vars, b.chi, b.psi);
  PscOutput<E> out = psc_prove<E>(ch, std::move(b.claim), b.summand, *inner, ledger.take(batt_masks(k_col)), masks);
  if (a) {
    a->assign(rows.size(), E::zero());
    for (const KeptTerm<E> &t : out.row_maps) (*a)[t.map] = t.coeff;
  }
  return batt_row(rows, std::move(out));
}
// Returns false only if the channel has rejected the proof; `out` then means nothing. PSC's checks
// only fill in the coefficient that is not sent, so a false input claim, or a sumcheck on other
// vectors than the W_j, shows only in `out`: it is then false, except with the probability of the
// round errors. The caller must still have `out` proved.
// Tests: blocks/iop/tests/batt_test.cpp (its messages call `out` "the row that comes out"), with
// honest provers only: no run there gives BATT a false claim. Four malicious provers are in test
// builds that are not part of this repository (with one of them, a control verifier that leaves
// the tau_j out of its chain is required to accept the false claims);
// blocks/iop/tests/batt_hiding_test.cpp (toy size).
template <class Ch>
bool batt_verify(Ch &ch, std::vector<TensorRow> rows, MaskLedger &ledger, TensorRow &out) {
  const unsigned k_col = batt_k_col(rows);
  BattClaim b = batt_claim(rows, batt_draw(ch, rows.size()));
  PscOutput<E> o;
  if (!psc_verify<E>(ch, std::move(b.claim), b.summand, ledger.take(batt_masks(k_col)), o)) return false;
  out = batt_row(rows, std::move(o));
  return ch.ok();
}

}  // namespace blk
