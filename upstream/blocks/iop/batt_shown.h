// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// BATT with its end value, the summed map at the sumcheck's last point, sent under a mask (`shown`
// in names). BATT (blocks/iop/batt.h) batches J linear claims (`row` in names) on the committed
// matrix M into one; this form also takes them out of a sub-proof's segment, the sub-proof's part
// of a request's first hash chain (`link` in names: `ReqLink`, blocks/iop/req.h).
//
// A segment's claims may name only oracle messages of the segment's own group (the messages that
// one proximity test reads). Claims on M are tested at the request's end, in a group that contains
// the request's index-mask message (mask rows only; blocks/iop/member.h) and no message of a
// segment. But a sub-proof's sumchecks (PSC, blocks/iop/psc.h: with a hiding polynomial) leave
// claims on M masked by entries of the segment's mask row. This block replaces them by two claims
// (`BattShown`), neither of which names mask entries of both messages.
//
// The programs `cp_prove` (prover) and `cp_verify` (verifier) do not include this header but run
// its messages in their own code (see batt_shown_prove()).
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/batt.h"

namespace blk {

// The block puts out two claims:
//   (E) one claim on M at a fresh column point gamma, whose only mask entry, c, is in the
//       index-mask message. It goes on to BATT or LIN, the test on the committed table for a batch
//       of requests (member.h), not to a further sumcheck of the segment.
//   (U) one claim on entries of the segment's mask row only, with a public value.
// The block checks no claim, so the caller must have both tested. Up to the round errors, a false
// incoming claim makes (U) false; a sumcheck on other vectors, or another end value, makes (E) or
// (U) false.
//
// The claims could also be batched as in batt.h, with no end value sent and all masks in the
// index-mask message. This form sends two scalars more and takes one entry of that message for
// each sub-proof. req.h gives a segment one group and never splits its claims between groups,
// hence this form.
struct BattShown {
  TensorRow row;                       // (E)
  std::vector<MaskTerm<E>> own_masks;  // (U): <own_masks> = own_target
  E own_target = E::zero();
};

inline size_t batt_shown_own_masks(unsigned k_col) { return 2 + 2 * size_t(k_col); }  // entries of the segment's mask row; one more, c, of the index-mask message

// The rounds. Incoming claim j (`TensorRow`): < g_j (x) eq(psi_j, .), M > + (masks) = tau_j.
// Its mask entries must be in the segment's mask row: not checked here.
// V: one element x; chi_j = x^j. Then PSC on
//     sum over M's 2^k_col columns of F  +  sum_j chi_j (masks of claim j)  =  sum_j chi_j tau_j,
//     F(X) = sum_j chi_j eq(psi_j, X) W_j(X),   W_j = g_j^T M.
// P: the hiding polynomial's sum. V: eps. k_col times: P: two coefficients; V: a challenge
// (together gamma). P: one coefficient of the round in PSC's further variable Y. V: beta != 0.
// P: v = F(gamma) + beta c.
// F(gamma) = < g (x) eq(gamma, .), M >, g = sum_j a_j g_j (a_j: batt_shown_scalars()), is linear in
// M, so PSC's claim on v is a claim on M: (E).
// Round errors (formulas: blocks/numbers/blocks.h). x: (J - 1) / |E|; eps: 1 / |E|; each of the
// k_col rounds: 2 / |E|; Y: 1 / (|E| - 1). `rows` is J: if a run batches another number of claims,
// batt_draw() ends the process, unless the channel has already rejected the proof.
inline void plan_batt_shown(Plan &p, const std::string &where, u64 rows, unsigned k_col) {
  plan_bat_sigma(p, where, rows);
  PscSize x;
  x.degree.assign(k_col, 3), x.degree_Y = 1, x.values = 1, x.with_Y = true;
  plan_psc(p, where, x);
}

// PSC's summand S(X, y) = y: one map, F, whose end value is sent (`Summand::shown`).
inline Summand<E> batt_shown_summand() {
  Summand<E> s;
  s.shown = 1, s.kept = 0, s.degree = 1;
  s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0]}; };
  return s;
}
// a_j = chi_j eq(psi_j, gamma)
inline std::vector<E> batt_shown_scalars(const BattClaim &b, const std::vector<E> &gamma) {
  std::vector<E> a = b.summand.at(gamma, {});
  a.erase(a.begin());
  return a;
}
// PSC's mask entries, in its order: the hiding polynomial's from `own` (the segment's mask row),
// then c, the mask of F's end value, from `index_masks`.
//
// Hiding. 3 + 2 k_col scalars are sent, as many as entries are taken. If the segment's 2 + 2 k_col
// entries are uniform, fresh and enter nothing else that is shown, then the hiding polynomial's
// sum, the coefficients of the k_col rounds and the coefficient of the round in Y are uniform for
// every fixed c, hence independent of c: the entries map onto them one to one. If c too is uniform
// and fresh, v is then uniform because of c, since beta != 0; v is the only scalar that c hides.
// c also enters the coefficient of the round in Y (under an entry of the segment that occurs in no
// other scalar), the round polynomials of BATT at the request's end and the answer to the
// out-of-domain sample of the index-mask message; each of those is hidden by another mask entry or
// by the sample's pad, if that entry or pad too is uniform, fresh and enters nothing else that is
// shown.
// Each c serves one run of one sub-proof: with one c for two runs, beta_1 v_0 - beta_0 v_1 reveals
// a functional of the weights. So a prover that proves a segment again must first draw and send a
// new index-mask message: a new run of a request has all its private randomness fresh. Nothing here
// prevents reuse across runs, and batt_shown_verify() accepts such a run.
// (Tested for the sent scalars only, at toy size, with the verifier's messages fixed:
// blocks/iop/tests/batt_hiding_test.cpp.)
// This argument is about the code here, which `cp_prove` and `cp_verify` do not run; alone it says
// nothing about their counterparts of it (see batt_shown_prove()).
// What is and is not established about hiding for a whole run: docs/limits.md#weight-hiding.
// c is in one oracle message only, the index-mask message. A program that numbers a segment's sent
// scalars and masks them with the entries of its mask row that have the same numbers
// (`slot` in verifier/src/vf_weights_merge.h) keeps c outside that numbering: the segment's own entry with
// the number of v stays as committed, masks nothing, and no claim may name it. `cp_verify` tests
// its own claims for this (WsMerged::sFv in that file).
inline std::vector<MaskSlot> batt_shown_slots(unsigned k_col, MaskLedger &own, MaskLedger &index_masks) {
  std::vector<MaskSlot> fresh = own.take(batt_shown_own_masks(k_col));
  fresh.push_back(index_masks.take());
  return fresh;
}
inline BattShown batt_shown_rows(const std::vector<TensorRow> &rows, const BattClaim &b, PscOutput<E> &&out) {
  BLK_REQUIRE(out.ends.size() == 1 && out.row_maps.empty(), "BATT, shown: one end value and no kept map");
  BattShown r;
  const std::vector<E> a = batt_shown_scalars(b, out.point);
  for (size_t j = 0; j < rows.size(); j++)
    for (Tensor g : rows[j].g) {
      g.scalar = g.scalar * a[j];
      r.row.g.push_back(std::move(g));
    }
  r.row.psi = std::move(out.ends[0].point), r.row.masks = std::move(out.ends[0].masks), r.row.target = out.ends[0].value;
  r.own_masks = std::move(out.row_masks), r.own_target = out.row_target;
  return r;
}

// BATT's sumcheck over the maps W_j, presented as the sumcheck of the one map F: the same
// rounds; the end value is F(gamma). `b` must outlive the object.
class BattShownSumcheck : public SumcheckProver<E> {
 public:
  BattShownSumcheck(std::unique_ptr<SumcheckProver<E>> over_w, const BattClaim &b) : over_w_(std::move(over_w)), b_(b) {}
  std::vector<E> round() override { return over_w_->round(); }
  void bind(const E &g) override { gamma_.push_back(g), over_w_->bind(g); }
  std::vector<E> end_values() override {
    const std::vector<E> a = batt_shown_scalars(b_, gamma_), w = over_w_->end_values();
    BLK_REQUIRE(w.size() == a.size(), "BATT, shown: one end value for each row");
    E f = E::zero();
    for (size_t j = 0; j < a.size(); j++) f += a[j] * w[j];
    return {f};
  }

 private:
  std::unique_ptr<SumcheckProver<E>> over_w_;
  const BattClaim &b_;
  std::vector<E> gamma_;
};

// `own`, `index_masks`: see batt_shown_slots(). `heavy`, `masks`, `a`: as in batt_prove(), but a_j
// has no factor eps.
// x must follow all that enters the incoming claims (the caller's duty, as in blocks/iop/batt.h).
// Soundness also needs the index-mask message (its root and its sample's answer) to be sent before
// beta: a prover that can choose c after beta fits v to (U), then c to (E), and proves a false
// claim. When a program takes c from its `MaskLedger` does not matter: take() gives out the entries
// in a fixed order. Both messages with mask entries are sampled before eps, as PSC needs
// (blocks/iop/mask.h).
// Tests: blocks/iop/tests/batt_shown_test.cpp and, in whole proof files against a committed
// table, blocks/frame/tests/seam_test.cpp. False proof files are
// stored, with a test that checks that they are refused, in examples/protocol-toy-false-files/
// (LIST.tsv there names each file and the words that its checker must print).
// The programs run counterparts of this source, CEnd::send_merge() (prover/head/claims_lib.cuh) and
// wsForkMerge() (verifier/src/vf_weights_merge.h), and build claim (E) with member_pending_row()
// (blocks/iop/member.h). These tests alone say nothing about that code, which whole runs exercise
// (verifier/tests/).
template <class Ch>
BattShown batt_shown_prove(Ch &ch, std::vector<TensorRow> rows, MaskLedger &own, MaskLedger &index_masks, const MaskValues<E> &masks, const BattSumcheck &heavy,
                           std::vector<E> *a = nullptr) {
  const unsigned k_col = batt_k_col(rows);
  BattClaim b = batt_claim(rows, batt_draw(ch, rows.size()));
  BattShownSumcheck inner(heavy(b.summand, b.claim.vars, b.chi, b.psi), b);
  PscOutput<E> out = psc_prove<E>(ch, std::move(b.claim), batt_shown_summand(), inner, batt_shown_slots(k_col, own, index_masks), masks);
  if (a) *a = batt_shown_scalars(b, out.point);
  return batt_shown_rows(rows, b, std::move(out));
}
template <class Ch>
[[nodiscard]] bool batt_shown_verify(Ch &ch, std::vector<TensorRow> rows, MaskLedger &own, MaskLedger &index_masks, BattShown &out) {
  const unsigned k_col = batt_k_col(rows);
  BattClaim b = batt_claim(rows, batt_draw(ch, rows.size()));
  PscOutput<E> o;
  if (!psc_verify<E>(ch, std::move(b.claim), batt_shown_summand(), batt_shown_slots(k_col, own, index_masks), o)) return false;
  out = batt_shown_rows(rows, b, std::move(o));
  return ch.ok();
}

}  // namespace blk
