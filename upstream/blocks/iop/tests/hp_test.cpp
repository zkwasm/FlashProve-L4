// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests HP, the hiding polynomial U: the random polynomial added to a sumcheck's summand
// (blocks/iop/hp.h). Enumerates over K = GF(3), GF(5), GF(7), for lists of one to three variables
// (`Weight`) from all five public constructors.
//   A. run(): the fast formulas (sum, round polynomials, value, linear form hp_form()) agree with
//      the definition of U. With 3000 or more choices of coefficients, only every 97th is compared.
//   B. run(): for every choice of challenges the map (random coefficients) -> (sum, round
//      polynomials) is one to one, and its image passes the verifier's checks. The tuples that pass
//      number |K|^(number of coefficients), so the map is onto them, and uniform coefficients give
//      a uniform tuple among those that pass.
//   C. main(): known answers that another prover must match: which coefficient of a round
//      polynomial is not sent (`Weight::fixed`), for each constructor, and the order of the random
//      coefficients (hp_place()).
#include "toy_support.h"

using namespace blk;

static int failures = 0;
#define CHECK(cond, ...)                \
  do {                                  \
    if (!(cond)) {                      \
      ++failures;                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                \
    }                                   \
  } while (0)

// U at a point, by its definition.
template <class K>
K U_at(const std::vector<Weight<K>>& vars, const std::vector<std::vector<K>>& w, const std::vector<K>& x) {
  K r = K::zero();
  for (size_t l = 0; l < vars.size(); ++l) {
    K term = poly_at(w[l], x[l]);
    for (size_t j = l + 1; j < vars.size(); ++j) term = term * poly_at(vars[j].psi, x[j]);
    r = r + term;
  }
  return r;
}
// Round polynomial of variable j by the definition: earlier variables at prefix, later ones summed.
template <class K>
std::vector<K> U_in_variable(const std::vector<Weight<K>>& vars, const std::vector<std::vector<K>>& w, size_t j,
                             std::vector<K> prefix);
template <class K>
K U_rest(const std::vector<Weight<K>>& vars, const std::vector<std::vector<K>>& w, size_t j, const std::vector<K>& prefix) {
  return j == vars.size() ? U_at(vars, w, prefix) : vars[j].apply(U_in_variable(vars, w, j, prefix));
}
template <class K>
std::vector<K> U_in_variable(const std::vector<Weight<K>>& vars, const std::vector<std::vector<K>>& w, size_t j,
                             std::vector<K> prefix) {
  std::vector<K> values;
  prefix.push_back(K::zero());
  for (unsigned n = 0; n < vars[j].d; ++n) {
    prefix.back() = small_int<K>(n);
    values.push_back(U_rest(vars, w, j + 1, prefix));
  }
  return poly_from_values_at_0_1_2(std::move(values));
}

template <class K>
std::vector<K> digits(u64 n, size_t count) {
  std::vector<K> v(count);
  for (K& x : v) x = K::from(n % K::size), n /= K::size;
  return v;
}
static u64 power(u64 b, size_t e) {
  u64 r = 1;
  while (e--) r *= b;
  return r;
}

template <class K>
void run(const char* name, const std::vector<Weight<K>>& vars, u64& tuples) {
  const size_t k = vars.size(), n = hp_entries(vars);
  const u64 all_w = power(K::size, n), all_gamma = power(K::size, k);
  for (u64 gi = 0; gi < all_gamma; ++gi) {
    const std::vector<K> gamma = digits<K>(gi, k);
    const std::vector<K> form = hp_form(vars, gamma);
    std::vector<bool> seen(all_w, false);
    for (u64 wi = 0; wi < all_w; ++wi) {
      const std::vector<K> entries = digits<K>(wi, n);
      HidingPolynomial<K> U(vars, entries);
      // The tuple as a verifier would receive it: the sum, then each round polynomial without the
      // coefficient that the check fixes. That makes n scalars: the base-|K| digits of `code`.
      u64 code = U.sum().v;
      K running = U.sum();
      std::vector<std::vector<K>> q;
      for (size_t j = 0; j < k; ++j) {
        q.push_back(U.round());
        CHECK(vars[j].apply(q[j]) == running, "%s: check of round %zu", name, j);
        std::vector<K> again = q[j];
        again[vars[j].fixed] = K::zero();
        vars[j].complete(again, running);
        CHECK(again == q[j], "%s: the fixed coefficient of round %zu", name, j);
        for (unsigned t = 0; t < vars[j].d; ++t)
          if (t != vars[j].fixed) code = code * K::size + q[j][t].v;
        running = poly_at(q[j], gamma[j]);
        U.bind(gamma[j]);
      }
      CHECK(!seen[code], "%s: two choices of coefficients give one tuple", name);
      seen[code] = true;
      ++tuples;
      K by_form = K::zero();
      for (size_t i = 0; i < n; ++i) by_form = by_form + form[i] * entries[i];
      CHECK(U.value() == running && by_form == running, "%s: value at the challenges", name);
      if (wi % 97 == 0 || all_w < 3000) {  // the slow comparison with the definition, on a part of the cases
        std::vector<std::vector<K>> w(k);
        for (unsigned l = 0; l < k; ++l) {
          w[l].assign(vars[l].d, K::zero());
          for (unsigned t = (l + 1 == k ? 0 : 1); t < vars[l].d; ++t) w[l][t] = entries[hp_place(vars, l, t)];
        }
        CHECK(U_rest(vars, w, 0, {}) == HidingPolynomial<K>(vars, entries).sum(), "%s: the sum against the definition", name);
        for (size_t j = 0; j < k; ++j)
          CHECK(U_in_variable(vars, w, j, std::vector<K>(gamma.begin(), gamma.begin() + j)) == q[j],
                "%s: round %zu against the definition", name, j);
        CHECK(U_at(vars, w, gamma) == running, "%s: the value against the definition", name);
      }
    }
  }
}

int main() {
  // The test uses small fields of its own and no hash.
  u64 tuples = 0;
  {
    using K = Fp<7>;
    const std::vector<K> D = {K::from(1), K::from(2), K::from(4)};
    CHECK(Weight<K>::plain(4).fixed == 1, "a plain sum leaves out the linear coefficient");
    CHECK(Weight<K>::equality(K::from(3), 4).fixed == 0 && Weight<K>::equality(K::zero(), 4).fixed == 0,
          "equality weights leave out the constant, at every point");
    CHECK(Weight<K>::at_zero(3).fixed == 0, "the value at 0 leaves out the constant");
    CHECK(Weight<K>::domain(D, 4).fixed == 0 && Weight<K>::lagrange(D, K::from(5), 4).fixed == 0,
          "a sum over a set of points leaves out the constant");
    // Three variables with 3, 4, 3 coefficients. Only the last has a random constant term.
    // `want`: variable, degree, place.
    const std::vector<Weight<K>> v = {Weight<K>::plain(3), Weight<K>::equality(K::from(2), 4), Weight<K>::at_zero(3)};
    CHECK(hp_entries(v) == 8, "1 + 2 + 3 + 2 coefficients");
    const size_t want[8][3] = {{0, 1, 0}, {0, 2, 1}, {1, 1, 2}, {1, 2, 3}, {1, 3, 4}, {2, 0, 5}, {2, 1, 6}, {2, 2, 7}};
    for (auto& w : want) CHECK(hp_place(v, unsigned(w[0]), unsigned(w[1])) == w[2], "the place of a coefficient");
  }
  {
    using K = Fp<3>;
    run<K>("GF(3): plain, plain, plain", {Weight<K>::plain(3), Weight<K>::plain(3), Weight<K>::plain(3)}, tuples);
    for (u64 z = 0; z < 3; ++z)
      run<K>("GF(3): equality, plain, at zero",
             {Weight<K>::equality(K::from(z), 3), Weight<K>::plain(2), Weight<K>::at_zero(3)}, tuples);
    run<K>("GF(3): one variable", {Weight<K>::at_zero(3)}, tuples);
  }
  {
    using K = Fp<5>;
    const std::vector<K> D = {K::from(0), K::from(1), K::from(2)};
    run<K>("GF(5): domain of 3 points, equality, at zero",
           {Weight<K>::domain(D, 3), Weight<K>::equality(K::from(3), 2), Weight<K>::at_zero(2)}, tuples);
    run<K>("GF(5): Lagrange weights, plain", {Weight<K>::lagrange(D, K::from(4), 4), Weight<K>::plain(3)}, tuples);
    run<K>("GF(5): plain, equality at 0, plain",
           {Weight<K>::plain(2), Weight<K>::equality(K::zero(), 3), Weight<K>::plain(2)}, tuples);
  }
  {
    using K = Fp<7>;
    run<K>("GF(7): plain of 4 coefficients, equality", {Weight<K>::plain(4), Weight<K>::equality(K::from(5), 2)}, tuples);
  }
  if (failures)
    std::printf("hp_test: FAILED (%d checks failed, see above)\n", failures);
  else
    std::printf("hp_test: ok (%llu tuples enumerated, all distinct for each choice of challenges)\n", (unsigned long long)tuples);
  return failures ? 1 : 0;
}
