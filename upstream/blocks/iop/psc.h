// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// PSC: the masked sumcheck for a polynomial summand. It reduces a sum claim
//     (phi_1 x .. x phi_k)(S(X, F(X))) + <masks> = target          (blocks/iop/summand.h)
// to linear claims with public targets (`row` in identifiers); if the mask entries (committed
// random elements: blocks/iop/mask.h) are uniform and each is used once, its messages reveal
// nothing about the maps F (witness polynomials). phi_j sums over variable j (`Weight`,
// blocks/iop/hp.h); <masks> is a public linear form in mask entries.
// The sumcheck runs on H = U + eps * (S' + T):
//   * U is a hiding polynomial (blocks/iop/hp.h) whose coefficients w are mask entries. The prover
//     sends its sum mu' before the verifier draws eps. The round polynomials of H are uniform among
//     those that pass the checks.
//   * S' is S with each shown map F_j(X) (its end value is sent, masked) replaced by
//     F_j(X) + c_j Y. c_j is a mask entry. Y is a new last variable, summed by g -> g(0). Its
//     challenge is beta != 0, so the end value sent, F_j(gamma) + beta c_j, is uniform
//     (gamma: challenges of X).
//   * T = sigma * prod_t psi_t(X_t), where sigma is the value of the mask form <masks>. T sums to
//     sigma, because phi_t(psi_t) = 1 (`Weight`).
// What this achieves for a whole run: docs/limits.md#weight-hiding.
// The caller must keep the order stated at psc_verify() and have every output claim checked against
// the committed vectors: psc_verify() itself rejects malformed input only.
//
// The hiding polynomial's sum, eps and the masked sumcheck follow the zero-knowledge sumcheck IOR,
// and the masks on end values follow the step that reduces R1CS in zero knowledge, of
// the paper that this protocol builds on (see THIRD_PARTY.md). IOR: interactive oracle reduction,
// the paper's word for a protocol step that reduces one claim to another.
#pragma once
#include <utility>
#include <vector>

#include "base.h"
#include "iop/hp.h"
#include "iop/mask.h"
#include "iop/summand.h"
#include "numbers/blocks.h"
#include "numbers/round_names.h"

namespace blk {

template <class K>
struct KeptTerm {
  unsigned map;  // number among the kept maps
  K coeff;
};

// The linear claims that PSC leaves; psc_rows() states them in full. A kept map's end value is not
// sent and stays as a term of (U) (`Summand`, summand.h).
template <class K>
struct PscOutput {
  std::vector<K> point;  // gamma: the challenges of X_1..X_k
  // (U): <row_masks> + sum_i row_maps[i].coeff * (kept map)(point) = row_target
  std::vector<MaskTerm<K>> row_masks;
  std::vector<KeptTerm<K>> row_maps;
  K row_target;
  // (E_j), one for each shown map, all at `point`
  std::vector<PointClaim<K>> ends;
};

// The variables of the masked sumcheck: those of the claim and, if a map is shown, Y. In the
// rounds of X, Y is read at 0, so S' gives the round polynomials of S.
template <class K>
std::vector<Weight<K>> psc_vars(const SumClaim<K>& claim, const Summand<K>& S) {
  BLK_REQUIRE((S.shown == 0) == (S.degree == 0), "psc: degree > 0 exactly if a map is shown");
  BLK_REQUIRE(S.shown > 0 || !claim.vars.empty(), "psc: nothing to sum over and nothing to show");
  std::vector<Weight<K>> vars = claim.vars;
  if (S.shown) vars.push_back(Weight<K>::at_zero(S.degree + 1));
  return vars;
}

// Mask entries that one PSC uses = scalars that it sends = 1 + sum_j (d_j - 1) + S.degree + S.shown
// (d_j: Weight::d of variable j of X). Their order in the argument `fresh` of psc_prove() and
// psc_verify(): w as in hp_place(), then the c_j.
template <class K>
size_t psc_masks(const SumClaim<K>& claim, const Summand<K>& S) {
  return hp_entries(psc_vars(claim, S)) + S.shown;
}

// The sizes that plan_psc() (blocks/numbers/blocks.h) needs for a plan (the list of all rounds,
// with sizes and errors). They are read from the claim and summand that are run, so that the
// degrees and the number of shown values are written once. The points of the weights phi_j, the
// mask form and the target do not enter: a caller that plans before these exist passes in a claim
// with any points. `PscSize`, and so the plan, does not record the type of each weight, the number
// of kept maps or the summand's formula.
template <class K>
PscSize psc_size(const SumClaim<K>& claim, const Summand<K>& S) {
  (void)psc_vars(claim, S);  // only for its BLK_REQUIRE checks
  PscSize x;
  for (const Weight<K>& v : claim.vars) x.degree.push_back(v.d);
  x.with_Y = S.shown > 0, x.degree_Y = S.degree, x.values = S.shown;
  return x;
}

// The output claims, from what both sides know at the end:
//   (U)    <hp_form(gamma, beta), w> + eps * prod_t psi_t(gamma_t) * <masks>
//              + eps * sum_i A_i * (kept map i)(gamma) = `last` - eps * A_0,
//          where (A_0, A_1, ..) = Summand::at(gamma, end values);
//   (E_j)  (shown map j)(gamma) + beta * c_j = end value j.
// `challenges` has one entry for each variable of psc_vars(); `last` is the last round polynomial
// at the last challenge.
template <class K>
PscOutput<K> psc_rows(SumClaim<K>&& claim, const Summand<K>& S, const std::vector<Weight<K>>& vars,
                      std::vector<MaskSlot>&& fresh, const K& eps, const std::vector<K>& challenges, const K& last,
                      const std::vector<K>& end_values) {
  PscOutput<K> out;
  out.point.assign(challenges.begin(), challenges.begin() + claim.vars.size());
  const std::vector<K> a = S.at(out.point, end_values);
  BLK_REQUIRE(a.size() == 1 + size_t(S.kept), "psc: Summand::at gives one coefficient for each kept map");
  const std::vector<K> form = hp_form(vars, challenges);
  for (size_t i = 0; i < form.size(); ++i) out.row_masks.push_back({std::move(fresh[i]), form[i]});
  K psis = eps;  // eps * prod_t psi_t(gamma_t)
  for (size_t t = 0; t < vars.size(); ++t) psis = psis * poly_at(vars[t].psi, challenges[t]);
  for (MaskTerm<K>& m : claim.masks) out.row_masks.push_back({std::move(m.slot), psis * m.coeff});
  for (unsigned i = 0; i < S.kept; ++i) out.row_maps.push_back({i, eps * a[1 + i]});
  out.row_target = last - eps * a[0];
  for (unsigned j = 0; j < S.shown; ++j) {
    PointClaim<K> e;
    e.point = out.point;
    e.masks.push_back({std::move(fresh[form.size() + j]), challenges.back()});
    e.value = end_values[j];
    out.ends.push_back(std::move(e));
  }
  return out;
}

// ---- The verifier ----
// Messages, through the channel `ch`. P: mu'. V: eps. For each variable of X, then for Y: P: the
// round polynomial q_j without the coefficient that the check fixes; V: a challenge. P: the end
// values of the shown maps; V: empty. Without shown maps there is no round of Y and no end value.
//
// Order. eps must come after
//   * mu';
//   * the answer to the out-of-domain sample of every oracle message that contains a mask entry
//     used here (oracle message: encoded rows sent as Merkle roots, blocks/iop/relation.h;
//     out-of-domain sample: a random linear form of the whole message, blocks/iop/send.h);
//   * everything that enters the weights phi_j, the summand or the target.
// (A prover that knows eps when it chooses mu' proves any target: it sets mu' so that the first
// check fits and is honest from there on; no later check reads mu'.) So PSC takes the complete
// claim, sends mu' and closes that round itself, and no caller can pass in an eps. The types of
// blocks/iop/mask.h keep a caller from creating a `MaskSlot` by mistake before the sample of its
// oracle message is answered.
// What PSC cannot see is the caller's duty: that the weights, the summand and the target were fixed
// by messages that the channel received earlier.
//
// Checks. phi_1(q_1) = mu' + eps * target and phi_{j+1}(q_{j+1}) = q_j(gamma_j) are used to fill in
// the coefficient that is not sent, so they hold by construction. A false claim or a wrong message
// instead makes an output claim false, up to the round errors below.
// Round errors (K: the challenge field): 1/|K| for eps; (d_j - 1)/|K| for variable j;
// S.degree/(|K| - 1) for Y; none for the end values.
//
// Ch: the channel's type, such as `VerifierRounds` (blocks/chain/rounds.h). It needs bool recv(K&),
// bool recv_hiding_sum(K&), K close_epsilon(name), close_algebraic(name) -> draw(), draw_nonzero(),
// close_empty(name), bool ok().
// fresh: psc_masks(claim, S) mask entries, each newly taken from a `MaskLedger`. Hiding needs each
// to be used nowhere else; a second use makes no claim false, so no verifier notices it.
// Returns false if the channel has rejected the proof (an item missing or malformed, a round
// without its proof of work); `out` is then worth nothing. A caller's mistake ends the process
// (BLK_REQUIRE, blocks/base.h).
template <class K, class Ch>
bool psc_verify(Ch& ch, SumClaim<K>&& claim, const Summand<K>& S, std::vector<MaskSlot>&& fresh, PscOutput<K>& out) {
  const std::vector<Weight<K>> vars = psc_vars(claim, S);
  BLK_REQUIRE(fresh.size() == hp_entries(vars) + S.shown, "psc: wrong number of fresh mask entries");
  const size_t k = claim.vars.size();

  K mu;
  if (!ch.recv_hiding_sum(mu)) return false;
  const K eps = ch.close_epsilon(R_PSC_EPSILON);

  K running = mu + eps * claim.target;  // what the next round polynomial must sum to
  std::vector<K> challenges;
  for (size_t j = 0; j < vars.size(); ++j) {
    std::vector<K> q(vars[j].d, K::zero());
    for (unsigned t = 0; t < vars[j].d; ++t)
      if (t != vars[j].fixed && !ch.recv(q[t])) return false;
    vars[j].complete(q, running);  // the check phi_j(q) = running, solved for the coefficient that is not sent
    auto draw = ch.close_algebraic(j < k ? R_PSC_ROUND : R_PSC_Y);
    challenges.push_back(j < k ? draw.draw() : draw.draw_nonzero());
    running = poly_at(q, challenges.back());
  }
  std::vector<K> end_values(S.shown);
  for (K& v : end_values)
    if (!ch.recv(v)) return false;
  if (S.shown) ch.close_empty(R_PSC_VALUES);

  out = psc_rows(std::move(claim), S, vars, std::move(fresh), eps, challenges, running, end_values);
  return ch.ok();  // a close can reject too, and gives fixed values from then on
}

// ---- The prover ----
// Ch: such as `ProverRounds`. It needs send(K), send_hiding_sum(K), K close_epsilon(name),
// close_algebraic(name) -> draw(), draw_nonzero(), close_empty(name).
// fresh: as for psc_verify().
// inner: the sumcheck of the same claim without masks, which hides nothing; PSC wraps it, and it
// computes exactly what it would compute alone. masks: the prover's copy of the mask entries.
// blocks/iop/tests/psc_hiding_test.cpp runs psc_prove() for every value of the mask entries and
// counts each tuple of sent scalars, over fields of 3 and 5 elements; it shows nothing about other
// fields, shapes or witnesses.
template <class K, class Ch>
PscOutput<K> psc_prove(Ch& ch, SumClaim<K>&& claim, const Summand<K>& S, SumcheckProver<K>& inner,
                       std::vector<MaskSlot>&& fresh, const MaskValues<K>& masks) {
  const std::vector<Weight<K>> vars = psc_vars(claim, S);
  const size_t entries = hp_entries(vars);
  BLK_REQUIRE(fresh.size() == entries + S.shown, "psc: wrong number of fresh mask entries");
  const size_t k = claim.vars.size();

  std::vector<K> w(entries), c(S.shown);
  for (size_t i = 0; i < entries; ++i) w[i] = masks(fresh[i].oracle(), fresh[i].index());
  for (size_t j = 0; j < c.size(); ++j) c[j] = masks(fresh[entries + j].oracle(), fresh[entries + j].index());
  K sigma = K::zero();  // the value of the claim's mask form
  for (const MaskTerm<K>& m : claim.masks) sigma = sigma + m.coeff * masks(m.slot.oracle(), m.slot.index());

  HidingPolynomial<K> U(vars, w);
  K mu = U.sum();
  ch.send_hiding_sum(mu);
  const K eps = ch.close_epsilon(R_PSC_EPSILON);

  K running = mu + eps * claim.target;
  K psis = K::one();  // prod_{t<j} psi_t(gamma_t)
  std::vector<K> challenges, ends;
  for (size_t j = 0; j < vars.size(); ++j) {
    std::vector<K> s;  // the summand's own round polynomial
    if (j < k) {
      s = inner.round();
    } else {
      // Round of Y: S(gamma, F_j(gamma) + c_j Y, kept values) as a polynomial in Y, from its values
      // at Y = 0 .. S.degree.
      ends = inner.end_values();
      BLK_REQUIRE(ends.size() == size_t(S.shown) + S.kept, "psc: one end value for each map");
      const std::vector<K> u(ends.begin() + S.shown, ends.end());
      std::vector<K> values, y(S.shown);
      for (unsigned n = 0; n <= S.degree; ++n) {
        for (unsigned i = 0; i < S.shown; ++i) y[i] = ends[i] + c[i] * small_int<K>(n);
        values.push_back(S.at_values(challenges, y, u));
      }
      s = poly_from_values_at_0_1_2(std::move(values));
    }
    BLK_REQUIRE(s.size() <= vars[j].d, "psc: the summand's round polynomial has too many coefficients");
    std::vector<K> q = U.round();
    for (size_t t = 0; t < s.size(); ++t) q[t] = q[t] + eps * s[t];
    for (size_t t = 0; t < vars[j].psi.size(); ++t) q[t] = q[t] + eps * sigma * psis * vars[j].psi[t];
    // The coefficient that is not sent is the right one only if the polynomial sums to the running
    // claim. If not, the claim is false or `inner` computes something other than what summand.h
    // asks for.
    BLK_REQUIRE(vars[j].apply(q) == running, "psc: the round polynomial does not sum to the running claim");
    for (size_t t = 0; t < q.size(); ++t)
      if (t != vars[j].fixed) ch.send(q[t]);
    auto draw = ch.close_algebraic(j < k ? R_PSC_ROUND : R_PSC_Y);
    challenges.push_back(j < k ? draw.draw() : draw.draw_nonzero());
    const K& g = challenges.back();
    running = poly_at(q, g);
    U.bind(g);
    psis = psis * poly_at(vars[j].psi, g);
    if (j < k) inner.bind(g);
  }
  std::vector<K> end_values(S.shown);
  for (unsigned i = 0; i < S.shown; ++i) end_values[i] = ends[i] + challenges.back() * c[i];
  for (const K& v : end_values) ch.send(v);
  if (S.shown) ch.close_empty(R_PSC_VALUES);

  return psc_rows(std::move(claim), S, vars, std::move(fresh), eps, challenges, running, end_values);
}

}  // namespace blk
