// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The statement of a sumcheck, shared by prover and verifier (`SumClaim`, `Summand`), and the
// interface to the sumcheck without masks, the prover's heavy computation (`SumcheckProver`).
//
// The claim has the form
//
//     (phi_1 x .. x phi_k)( S(X, F_1(X), .., F_a(X)) )  +  <masks>  =  target.
//
// X = (X_1..X_k). phi_j is the linear functional that sums over variable j, in which polynomials
// have d_j coefficients (`Weight` in hp.h). The maps F_1..F_a are polynomials in X that belong to
// the witness: the extension of a vector that the prover has sent, or of one that is never sent but
// is a function of sent ones (a level of a product tree, a matrix times a vector). S is a public
// polynomial in X and in the maps' values. <masks> is a public linear form in mask entries (random
// elements of oracle messages' mask rows, mask.h). It is empty for a claim from a public statement,
// but not for one on an end value of an earlier sumcheck: a map's value at the challenge point,
// sent with a mask on it. The target is public. K is the challenge field; hp.h lists what it needs.
#pragma once
#include <functional>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/hp.h"
#include "iop/mask.h"

namespace blk {

// "V(point) + <masks> = value" for a map V: what a sumcheck leaves for each map whose end value it
// sends. If V is the extension of a sent vector, this is a linear claim on that vector and the mask
// rows. If V is never sent, it is the next sumcheck's claim (SumClaim::at_point()) or, if V is
// linear in a committed matrix, a linear claim on that matrix (batt_shown_rows(), batt_shown.h).
template <class K>
struct PointClaim {
  std::vector<K> point;
  std::vector<MaskTerm<K>> masks;
  K value;
};

// The claim in the comment at the top, without S and the maps. psc_prove() and psc_verify() in
// psc.h (PSC, the sumcheck with a hiding polynomial) take it together with a `Summand`. If `vars`
// is empty, `Summand::shown` must not be 0.
template <class K>
struct SumClaim {
  std::vector<Weight<K>> vars;      // one for each variable; may be empty (a claim on values, nothing to sum over)
  std::vector<MaskTerm<K>> masks;
  K target;

  // V(point) = sum_b eq(point, b) V(b) for a multilinear V: the claim on V, as a sum under equality
  // weights. The summand must give V on {0,1}^k, with at most d[j] >= 2 coefficients in variable j.
  // If V has a variable over another set of points, build the claim by hand: Weight::lagrange().
  static SumClaim at_point(PointClaim<K>&& c, const std::vector<unsigned>& d) {
    BLK_REQUIRE(d.size() == c.point.size(), "SumClaim::at_point: one degree bound for each variable");
    SumClaim s;
    for (size_t j = 0; j < d.size(); ++j) s.vars.push_back(Weight<K>::equality(c.point[j], d[j]));
    s.masks = std::move(c.masks);
    s.target = c.value;
    return s;
  }
};

// The public polynomial S. Each of its maps is of one of two types, counted by `shown` and `kept`.
//   shown: the prover sends the map's end value, under a mask of its own.
//   kept:  S is affine in the kept maps jointly (once X and the shown maps are fixed). Their values
//          are not sent; they stay as terms of the linear claim that the sumcheck outputs. A
//          summand that is affine in all its maps jointly needs no shown map.
template <class K>
struct Summand {
  unsigned shown = 0;
  unsigned kept = 0;
  // At least the total degree of S in its shown maps; 0 if and only if there are none, which
  // psc_vars() (psc.h) checks. The prover does not notice a number that is too low: the linear
  // claim in its output is then false.
  unsigned degree = 0;
  // S at a point, as an affine form in the kept maps: returns (A_0, A_1, .., A_kept) with
  // S(point, y, u) = A_0 + sum_i A_i u_i for shown values y and kept values u. The verifier calls
  // it with values y read from the proof: it must cope with every y, without aborting.
  std::function<std::vector<K>(const std::vector<K>& point, const std::vector<K>& y)> at;

  // S(point, y, u), computed from at().
  K at_values(const std::vector<K>& point, const std::vector<K>& y, const std::vector<K>& u) const {
    const std::vector<K> a = at(point, y);
    BLK_REQUIRE(a.size() == 1 + size_t(kept) && u.size() == kept, "Summand: one coefficient for each kept map");
    K r = a[0];
    for (unsigned i = 0; i < kept; ++i) r = r + a[1 + i] * u[i];
    return r;
  }
};

// The prover's sumcheck without masks: the same computation as in a protocol that hides nothing.
// psc_prove() (psc.h) adds the masks, arithmetic on a few scalars. It calls round() and bind() once
// per variable, in order, then end_values() if a map is shown. An implementation may run on a GPU.
template <class K>
class SumcheckProver {
 public:
  virtual ~SumcheckProver() = default;
  // For the next variable j: the coefficients, lowest degree first and at most d_j of them, of
  //   s_j(X) = (phi_{j+1} x .. x phi_k)( S(g_1, .., g_{j-1}, X, . , F(g_1, .., g_{j-1}, X, .)) ),
  // where g_1, .., g_{j-1} are the challenges given to bind() so far. Under equality weights the
  // equality factor is part of the phi's, not of S: s_j has none for a bound variable or for X.
  // poly_from_values_at_0_1_2() (hp.h) makes coefficients from values.
  virtual std::vector<K> round() = 0;
  virtual void bind(const K& challenge) = 0;
  // After the last bind(): the maps' values at the challenge point, without masks, the shown maps
  // first, then the kept ones.
  virtual std::vector<K> end_values() = 0;
};

// The prover's copy of the mask entries: value(oracle, index), numbered as in `MaskSlot` (mask.h).
template <class K>
using MaskValues = std::function<K(u32 oracle, u32 index)>;

}  // namespace blk
