// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks the exact test of hiding (exact.h), which other tests in this directory rely on. That test
// asks whether what a prover sends (its view) has the same distribution (printed: "law") for two
// witnesses. On tiny provers with known laws it must pass equal laws, and must not pass where they
// differ, where it cannot decide, or where nothing was tested. A form of exact.h that is not in
// this repository wrongly passed one of these provers. w(n, i) is entry i of witness n; a, b, ...
// are private random values ("coins"). A "curved" coin: one that occurs in a product of coins.
#include <cstdio>
#include <string>
#include <vector>

#include "tests/hiding/exact.h"

using namespace blk;

namespace {

Look show(const std::vector<F> &values) {
  Look l;
  for (size_t i = 0; i < values.size(); i++) l.view.push_back(values[i]), l.at.push_back(i);
  return l;
}
F w(unsigned witness, unsigned i = 0) {
  ToyCoins from(100 + witness);
  F x = F::zero();
  for (unsigned k = 0; k <= i; k++) x = from.f();
  return x;
}

int failures = 0;
unsigned cases = 0;
// Compares witnesses 0 and 1. If `same` is false, the verdict's reason must contain `words`; an
// empty `words` accepts any reason.
void expect(const char *name, bool same, const char *words, const World &world) {
  Exact x(world, 1);
  const Exact::Verdict v = x.compare(1);
  cases++;
  if (v.same == same && (same || v.why.find(words) != std::string::npos)) return;
  failures++;
  std::printf("exact_self_test: %s: %s, %s\n", name, v.same ? "the same law" : "not the same law", v.why.c_str());
}

} // namespace

int main() {
  // ---- equal laws ----
  expect("a mask on a value", true, "", [](unsigned n, Coins &c) { return show({c.f() + w(n)}); });
  expect("a product of two masked values under a further mask, the values shown", true, "", [](unsigned n, Coins &c) {
    const F s = c.f() + w(n, 0), t = c.f() + w(n, 1);
    return show({s, t, s * t + c.f() + w(n, 2)});
  });
  expect("a relation of degree two among the entries: s, t and s t", true, "", [](unsigned n, Coins &c) {
    const F s = c.f() + w(n, 0), t = c.f() + w(n, 1);
    return show({s, t, s * t});
  });
  // ---- different laws ----
  expect("a bare value beside a masked one", false, "no coins give", [](unsigned n, Coins &c) { return show({c.f() + w(n, 0), w(n, 1)}); });
  expect("one mask on two values", false, "no coins give", [](unsigned n, Coins &c) {
    const F a = c.f();
    return show({a + w(n, 0), a + w(n, 1)});
  });
  // Coins that give witness 1 the view of witness 0 exist, which is not enough: witness 0's views
  // are only about half the field's values.
  expect("a square for the first witness alone", false, "do not fix the curved coins", [](unsigned n, Coins &c) {
    const F a = c.f();
    return show({a + (n == 0 ? w(0) : F::zero()) * a * a});
  });
  expect("a product of coins for the second witness alone", false, "curved in different coins", [](unsigned n, Coins &c) {
    const F a = c.f(), b = c.f();
    return show({a, n == 0 ? b : a * b + w(n)});
  });
  expect("a square for both, with the witness as its factor: nothing affine fixes the coin", false, "", [](unsigned n, Coins &c) {
    const F a = c.f();
    return show({a + w(n) * a * a});
  });
  // The four products form a 2 x 2 matrix of rank at most one.
  expect("a matrix of rank one over the witness", false, "", [](unsigned n, Coins &c) {
    const F a = c.f(), b = c.f(), x = c.f(), y = c.f();
    return show({a * x + w(n, 0), a * y + w(n, 1), b * x + w(n, 2), b * y + w(n, 3)});
  });
  expect("a mask from a generator of the prover's own", false, "no coins give", [](unsigned n, Coins &c) { return show({c.f(), from_u64(12345) + w(n)}); });
  // ---- equal laws for which the test cannot vouch: it says so and does not pass ----
  expect("a product of two coins that are shown nowhere, under a mask", false, "do not fix the curved coins", [](unsigned n, Coins &c) {
    const F a = c.f(), b = c.f();
    return show({a * b + c.f() + w(n)});
  });
  // ---- nothing tested ----
  expect("a world that has lost its witness", false, "NOTHING WAS TESTED", [](unsigned, Coins &c) { return show({c.f() + w(0), w(0, 1)}); });

  if (failures) return std::printf("exact_self_test: FAILED (%d of %u)\n", failures, cases), 1;
  std::printf("exact_self_test: ok (%u small provers with known laws: 3 equal, 7 different, 1 equal but outside the shape that the test can vouch for, 1 that tests nothing: each judged as it must be)\n", cases);
  return 0;
}
