// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Does what the prover of PSC sends depend on its witness? PSC (blocks/iop/psc.h) is the sumcheck
// with a hiding polynomial (`hp` in identifiers); blocks/tests/hiding/exact.h explains the method.
// psc_prove() runs unchanged on the CPU, with `SumcheckCpu` as the inner sumcheck. A test channel
// (`ToyProverChannel`) gives it fixed challenges, and no verifier runs. A pass says nothing about
// the GPU code with which cp_prove makes its proofs. The prover's private randomness is the mask
// row: 3 `coins` (in the field F) for each entry (in E, its cubic extension). The view is the
// scalars sent: there must be as many as mask entries, and the map from coins to view must have
// full rank. Three witnesses must give the view the same distribution ("law"), and each mask entry
// must be needed for that (exception: see test()). Negative controls must tell two witnesses apart:
// faults in the mask row, and the challenge 0 for the further variable that PSC adds (`Y` in
// blocks/iop/psc.h). A pass covers only the witnesses and challenges that were run, all from fixed
// seeds. blocks/iop/tests/psc_hiding_test.cpp tries all mask rows over fields of 3 and 5 elements;
// this test runs in the large field, also on larger shapes.
#include <cstdio>

#include "engine/sumcheck_cpu.h"
#include "field/mle.h"
#include "iop/tests/toy_support.h"
#include "tests/hiding/exact.h"

using namespace blk;

namespace {

// One case: k variables under equality weights (Weight::equality(), blocks/iop/hp.h), with round
// polynomials of d coefficients. Two summands:
//   product   a b - c, as in blocks/toy/product.h: a and b are shown (each end value, at the
//             challenge point, is sent under an inner mask), c is kept (not sent). The two inner
//             masks multiply each other, so the view is not affine ("curved") in their 6 coins;
//   linear    one shown map times the public polynomial 1 + X_1 + .. + X_k: affine in all coins.
struct Shape {
  const char *name;
  unsigned k, d;
  bool product;
  bool zero_challenge = false; // negative control: the further variable's challenge is 0
  Summand<E> summand() const {
    Summand<E> s;
    if (product) {
      s.shown = 2, s.kept = 1, s.degree = 2;
      s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] * y[1], -E::one()}; };
    } else {
      s.shown = 1, s.kept = 0, s.degree = 1;
      s.at = [](const std::vector<E> &x, const std::vector<E> &y) {
        E g = E::one();
        for (const E &c : x) g = g + c;
        return std::vector<E>{y[0] * g};
      };
    }
    return s;
  }
};

// The public values and the verifier's challenges, the same in all runs of a shape. The protocol's
// channel (blocks/chain/rounds.h) never draws 0 for `Y`; listed_coins()
// (blocks/iop/tests/toy_support.h) allows it.
struct Fixed {
  std::vector<E> zeta, said; // equality weights' point; challenges: eps, one per variable, Y's
  E target;
  explicit Fixed(const Shape &s) {
    ToyCoins from(7);
    for (unsigned j = 0; j < s.k; j++) zeta.push_back(from.e());
    for (unsigned j = 0; j < s.k + 2; j++) said.push_back(from.e());
    BLK_REQUIRE(said.back() != E::zero(), "the further variable's challenge is not 0");
    if (s.zero_challenge) said.back() = E::zero();
    target = s.product ? E::zero() : from.e();
  }
};

// Witness number `witness`: pseudo-random maps that make the claim of claim_of() true.
std::vector<std::vector<E>> witness_maps(const Shape &s, const Fixed &fixed, unsigned witness) {
  ToyCoins from(100 + witness);
  const size_t n = size_t(1) << s.k;
  std::vector<std::vector<E>> maps(s.product ? 3 : 1, std::vector<E>(n));
  for (size_t i = 0; i < n; i++) maps[0][i] = from.e();
  if (s.product) {
    for (size_t i = 0; i < n; i++) maps[1][i] = from.e(), maps[2][i] = maps[0][i] * maps[1][i];
  } else { // maps[0][0], where the public polynomial is 1, makes the sum fixed.target
    std::vector<E> eq(n);
    eq_table(fixed.zeta.data(), s.k, eq.data());
    E rest = E::zero();
    for (size_t i = 1; i < n; i++) rest = rest + eq[i] * maps[0][i] * small_int<E>(1 + (unsigned)__builtin_popcountll(i));
    maps[0][0] = (fixed.target - rest) * inv(eq[0]);
  }
  return maps;
}

SumClaim<E> claim_of(const Shape &s, const Fixed &fixed) {
  SumClaim<E> claim;
  for (const E &z : fixed.zeta) claim.vars.push_back(Weight<E>::equality(z, s.d));
  claim.target = fixed.target;
  return claim;
}

// The prover under test, as a `World` (blocks/tests/hiding/exact.h).
World world_of(const Shape &s) {
  const Fixed fixed(s);
  return [s, fixed](unsigned witness, Coins &coins) {
    const Summand<E> summand = s.summand();
    SumClaim<E> claim = claim_of(s, fixed);
    std::vector<E> row(psc_masks(claim, summand));
    for (E &x : row) x = coins.e();
    SumcheckCpu<E> inner(summand, claim.vars, witness_maps(s, fixed, witness));
    MaskLedger ledger(SendAccess::make(0, 0, u32(row.size())));
    ToyProverChannel<E> ch(listed_coins(fixed.said), false);
    psc_prove<E>(ch, std::move(claim), summand, inner, ledger.take(row.size()), mask_row(row));
    Look l;
    for (size_t i = 0; i < ch.tape.size(); i++) l.push(ch.tape[i], i);
    return l;
  };
}

int failures = 0;
unsigned controls = 0;
unsigned long long runs = 0;
void expect(bool ok, const Shape &s, const char *what) {
  if (ok) return;
  failures++;
  std::printf("psc_exact_test: %s: %s\n", s.name, what);
}

// Overwrites mask entry `entry`, from first_coordinate on, with entry `from`, or 0 if from < 0.
// Negative control: passes only if compare() tells witnesses 0 and 1 apart and reports an entry of
// the view (`Verdict::entry`).
void control(const Shape &s, const char *what, size_t entry, long from, unsigned first_coordinate = 0) {
  Exact x(world_of(s), 1);
  std::vector<size_t> held;
  for (unsigned c = first_coordinate; c < 3; c++) held.push_back(3 * entry + c);
  x.spoil([=](std::vector<F> &list) {
    for (unsigned c = first_coordinate; c < 3; c++) list[3 * entry + c] = from < 0 ? F::zero() : list[3 * (size_t)from + c];
  }, held);
  const Exact::Verdict v = x.compare(1);
  controls++, runs += x.runs();
  if (v.same || v.entry < 0) {
    failures++;
    std::printf("psc_exact_test: %s: CONTROL NOT NOTICED: %s (%s)\n", s.name, what, v.same ? "same law" : v.why.c_str());
  }
}

void test(const Shape &s) {
  const Summand<E> summand = s.summand();
  const SumClaim<E> claim = claim_of(s, Fixed(s));
  const size_t masks = psc_masks(claim, summand);
  expect(masks == 1 + size_t(s.k) * (s.d - 1) + summand.degree + summand.shown, s, "another number of mask entries than 1 + sum (d - 1) + degree + shown");
  const std::vector<Weight<E>> vars = psc_vars(claim, summand);
  const size_t constant = hp_place(vars, unsigned(vars.size()) - 1, 0);
  Exact x(world_of(s), 1);
  expect(x.coins() == 3 * masks && x.entries() == 3 * masks, s, "the scalars sent are not as many as the mask entries");
  for (unsigned second = 1; second <= 2; second++) {
    const Exact::Verdict v = x.compare(second);
    if (!v.same) std::printf("psc_exact_test: %s: TOLD APART: %s; %s\n", s.name, v.why.c_str(), x.where(v.entry).c_str());
    expect(v.same, s, "two witnesses do not have the same law");
    expect(v.rank == x.entries(), s, "the map from coins to view has not full rank");
    expect(v.curved == (s.product ? 6u : 0u) && v.idle == 0, s, "other coins are curved or idle than foretold");
    // The exception: with nothing to sum over, the hiding polynomial's constant is its sum and
    // enters no other scalar. Without it that scalar is a constant, the same for every witness.
    bool all = v.same && v.draw_matters.size() == masks;
    for (size_t i = 0; i < v.draw_matters.size(); i++) all = all && (v.draw_matters[i] || (vars.size() == 1 && i == constant));
    expect(all, s, "a mask entry could be taken away without a trace");
  }
  runs += x.runs();

  // The mask row: the hiding polynomial's coefficients (hp_place()), then the inner masks.
  const size_t w = hp_entries(vars);
  control(s, "the hiding polynomial's coefficient of the further variable = 0", hp_place(vars, unsigned(vars.size()) - 1, 1), -1);
  if (vars.size() > 1) control(s, "the hiding polynomial's constant = 0", constant, -1);
  control(s, "a linear coefficient of the hiding polynomial = 0", hp_place(vars, 0, 1), -1);
  control(s, "no inner mask on the first end value", w, -1);
  control(s, "an inner mask that lies in F: one coordinate drawn, two zero", w, -1, 1);
  control(s, "an inner mask that is also an entry of the hiding polynomial", w, 0);
  if (s.product) control(s, "one inner mask for two end values", w + 1, (long)w);
  { // the challenge that multiplies the inner masks is 0: the end values are sent unmasked
    Shape zero = s;
    zero.zero_challenge = true;
    Exact z(world_of(zero), 1);
    const Exact::Verdict v = z.compare(1);
    controls++, runs += z.runs();
    if (v.same || v.entry < 0) failures++, std::printf("psc_exact_test: %s: CONTROL NOT NOTICED: the further variable's challenge is 0 (%s)\n", s.name, v.same ? "same law" : v.why.c_str());
  }
}

} // namespace

int main() {
  const Shape shapes[] = {{"product, 3 variables", 3, 3, true, false}, {"product, no variable", 0, 3, true, false}, {"linear, 4 variables", 4, 3, false, false}};
  for (const Shape &s : shapes) test(s);
  if (failures)
    std::printf("psc_exact_test: FAILED (%d)\n", failures);
  else
    std::printf("psc_exact_test: ok (3 shapes: full rank, the same law for three random witnesses, every mask entry needed but the one that is foretold; all %u controls told witnesses apart; %llu runs of the prover)\n",
                controls, runs);
  return failures ? 1 : 0;
}
