// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests FOLD's arithmetic (blocks/iop/fold_algebra.h, blocks/iop/fold_rounds.h) over prime fields
// defined here (`Zq`): Z/(2^61 - 1) with random data, Z/3 and Z/5 by enumeration; never over the
// library's field or its extension (blocks/field/field.h). FOLD reduces a linear claim of value
// `target` on 2^k main rows and 2^h mask rows to a final claim on three folded rows, by a sumcheck
// of H = U + eps * (the claim's terms): U is the prover's hiding polynomial, eps a challenge.
//  1. fold_mask_table() and fold_mask_weights() give back U's value, for all h <= k <= 5.
//  2. Honest rounds: each polynomial is the sum that defines it (k <= 4); the final claim is true.
//  3. A false `target`, or a changed sent coefficient in any round, ends in a false final claim.
//  4. Hiding (k <= 3): for every witness, eps and challenges tried, the map from the prover's
//     2k + 1 private random values (`coins`) to the 2k + 1 scalars that it sends (U's sum; each
//     round's constant and quadratic coefficients) is one to one, so uniform coins give uniform
//     scalars, whatever the witness.
//  5. Why eps must come after U's sum: a prover that knows eps when it sends the sum gets a true
//     final claim out of a false `target`, with honest rounds.
//  6. Kept (unfolded) rows: their closed form, one term of H, gives the same messages and, on the
//     kept rows, the same final claim as bundling a copy of each kept row with every main row.
// No run of this test shows that part 4 finds a map that is not one to one, as with two
// coefficients that share a coin.
#include <cstdio>
#include <random>
#include <set>

#include "iop/fold_rounds.h"

using namespace blk;

// Z/Q for a prime Q below 2^63: + and - form sums of up to 2Q in a u64, so the library's prime,
// 2^64 - 2^32 + 1, does not fit.
template <u64 Q>
struct Zq {
  u64 v = 0;
  static Zq zero() { return Zq{0}; }
  static Zq one() { return Zq{1}; }
  Zq operator+(Zq o) const { return Zq{(v + o.v) % Q}; }
  Zq operator-(Zq o) const { return Zq{(v + Q - o.v) % Q}; }
  Zq operator*(Zq o) const { return Zq{(u64)((u128)v * o.v % Q)}; }
  bool operator==(Zq o) const { return v == o.v; }
};
template <u64 Q>
Zq<Q> inv(Zq<Q> x) {
  Zq<Q> r = Zq<Q>::one();
  for (u64 e = Q - 2; e; e >>= 1, x = x * x)
    if (e & 1) r = r * x;
  return r;
}

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

template <class K, class Rng>
std::vector<K> random_vector(size_t n, Rng &rng, u64 q) {
  std::vector<K> v(n);
  for (K &x : v) x = K{rng() % q};
  return v;
}

// One instance of FOLD (coefficients c, d on main rows m and mask rows mu) and its honest run.
template <class K>
struct Run {
  unsigned k, h;
  size_t len, len_zk;
  std::vector<K> c, m, d, mu, coins, g;
  K eps, target;
  K kept = K::zero();  // the value of the kept rows' term
  // filled by `play()`
  K mask_sum;
  std::vector<std::vector<K>> q;
  std::vector<K> c_end, m_end;
  FoldMaskEnd<K> mask_end;

  K true_target() const { return inner(c, m) + inner(d, mu) + kept; }
  void play() {
    FoldProverRounds<K> p(k, h, len_zk, fold_mask_from_coins(k, coins), d, mu, kept);
    mask_sum = p.mask_sum();
    p.start(eps);
    c_end = c, m_end = m;
    q.clear();
    for (unsigned j = 0; j < k; j++) {
      q.push_back(p.round(pair_products(c_end, m_end)));
      p.bind(g[j]);
      fold_low_once(c_end, g[j]);
      fold_low_once(m_end, g[j]);
    }
    for (K &x : c_end) x = x * eps;
    mask_end = p.end();
  }
  // The left side of the final claim, with the verifier's coefficients on the two mask rows.
  K end_value() const {
    return inner(c_end, m_end) + inner(fold_mask_form(k, h, d, eps, g), mask_end.mu) +
           inner(fold_mask_weights(k, h, len_zk, g), mask_end.fresh) + fold_kept_factor(eps, g) * kept;
  }
  K verifier_claim(const std::vector<std::vector<K>> &sent) const {
    FoldVerifierRounds<K> v(mask_sum, eps, target);
    for (unsigned j = 0; j < k; j++) v.round(sent[j][0], sent[j][2], g[j]);
    return v.claim();
  }
};

template <class K, class Rng>
Run<K> random_run(unsigned k, unsigned h, Rng &rng, u64 q) {
  Run<K> r;
  r.k = k, r.h = h, r.len = 2, r.len_zk = fold_mask_columns(k, h) + 1;
  r.c = random_vector<K>((size_t(1) << k) * r.len, rng, q);
  r.m = random_vector<K>((size_t(1) << k) * r.len, rng, q);
  r.d = random_vector<K>((size_t(1) << h) * r.len_zk, rng, q);
  r.mu = random_vector<K>((size_t(1) << h) * r.len_zk, rng, q);
  r.coins = random_vector<K>(2 * k + 1, rng, q);
  r.g = random_vector<K>(k, rng, q);
  r.eps = K{rng() % q};
  r.kept = K{rng() % q};
  r.target = r.true_target();
  return r;
}

// H at x, from its definition.
template <class K>
K summand(const Run<K> &r, const std::vector<K> &x) {
  const std::vector<K> w = fold_mask_from_coins(r.k, r.coins);
  const auto vars = fold_mask_variables<K>(r.k);
  K u = K::zero();
  for (unsigned l = 0; l < r.k; l++) {
    K wl = K::zero(), power = K::one();
    for (unsigned t = 0; t < 3; t++, power = power * x[l])
      if (t >= 1 || l + 1 == r.k) wl = wl + w[hp_place(vars, l, t)] * power;
    for (unsigned j = l + 1; j < r.k; j++) wl = wl * x[j];
    u = u + wl;
  }
  K mask = inner(fold_low(r.d, x.data(), r.h), fold_low(r.mu, x.data(), r.h));
  for (unsigned t = r.h; t < r.k; t++) mask = mask * x[t];
  K kept = r.kept;
  for (unsigned t = 0; t < r.k; t++) kept = kept * (K::one() - x[t]);
  return u + r.eps * (inner(fold_low(r.c, x.data(), r.k), fold_low(r.m, x.data(), r.k)) + mask + kept);
}

template <class K>
void check_by_definition(const Run<K> &r) {
  for (unsigned j = 0; j < r.k; j++)
    for (u64 at = 0; at < 3; at++) {  // three points fix a polynomial of degree 2
      K sum = K::zero();
      for (u64 b = 0; b < (u64(1) << (r.k - j - 1)); b++) {
        std::vector<K> x(r.g.begin(), r.g.begin() + j);
        x.push_back(K{at});
        for (unsigned t = 0; t + j + 1 < r.k; t++) x.push_back(K{(b >> t) & 1});
        sum = sum + summand(r, x);
      }
      CHECK(poly_at(r.q[j], K{at}) == sum, "a round polynomial differs from the sum that defines it");
    }
}

// Part 6, for slots - 1 kept rows.
template <class K, class Rng>
void check_kept_rows(unsigned k, unsigned h, size_t slots, Rng &rng, u64 q) {
  Run<K> closed = random_run<K>(k, h, rng, q);
  closed.kept = K::zero();
  std::vector<std::vector<K>> kc(slots), km(slots);  // index 0 is not used
  for (size_t j = 1; j < slots; j++) {
    kc[j] = random_vector<K>(closed.len, rng, q), km[j] = random_vector<K>(closed.len, rng, q);
    closed.kept = closed.kept + inner(kc[j], km[j]);
  }
  closed.target = closed.true_target();
  closed.play();
  CHECK(closed.verifier_claim(closed.q) == closed.end_value(), "kept rows: an honest run ends in a false claim");

  // Entry (bundle b, slot j, place c) is at index b + 2^k (j + slots * c).
  Run<K> written = closed;
  const size_t rows = size_t(1) << k;
  written.kept = K::zero(), written.len = slots * closed.len;
  written.c.assign(rows * written.len, K::zero()), written.m.assign(rows * written.len, K::zero());
  for (size_t b = 0; b < rows; b++)
    for (size_t c = 0; c < closed.len; c++) {
      written.c[b + rows * (slots * c)] = closed.c[b + rows * c], written.m[b + rows * (slots * c)] = closed.m[b + rows * c];
      for (size_t j = 1; j < slots; j++) {
        written.m[b + rows * (j + slots * c)] = km[j][c];
        if (b == 0) written.c[b + rows * (j + slots * c)] = kc[j][c];
      }
    }
  CHECK(written.true_target() == closed.target, "kept rows: the bundles state another claim");
  written.play();
  CHECK(written.mask_sum == closed.mask_sum && written.q == closed.q, "kept rows: the closed form sends other messages");
  const K factor = fold_kept_factor(closed.eps, closed.g);
  for (size_t c = 0; c < closed.len; c++)
    for (size_t j = 1; j < slots; j++) {
      CHECK(written.m_end[j + slots * c] == km[j][c], "kept rows: the fold of the copies is not the row");
      CHECK(written.c_end[j + slots * c] == factor * kc[j][c], "kept rows: the new coefficients are not the closed form's");
    }
}

int main() {
  // 2^61 - 1, a prime
  using Big = Zq<2305843009213693951ull>;
  const u64 big = 2305843009213693951ull;
  std::mt19937_64 rng(1);

  for (unsigned k = 1; k <= 5; k++)
    for (unsigned h = 0; h <= k; h++) {
      // 1
      const size_t len = fold_mask_columns(k, h) + (k & 1);
      const std::vector<Big> w = random_vector<Big>(2 * k + 1, rng, big), g = random_vector<Big>(k, rng, big);
      HidingPolynomial<Big> u(fold_mask_variables<Big>(k), w);
      for (unsigned j = 0; j < k; j++) u.bind(g[j]);
      CHECK(inner(fold_mask_weights(k, h, len, g), fold_low(fold_mask_table(k, h, len, w), g.data(), h)) == u.value(),
            "table and weights do not give the hiding polynomial's value");
      // 2
      Run<Big> r = random_run<Big>(k, h, rng, big);
      r.play();
      if (k <= 4) check_by_definition(r);
      CHECK(r.q[0][0] + r.q[0][0] + r.q[0][1] + r.q[0][2] == r.mask_sum + r.eps * r.target, "the first check fails");
      CHECK(r.verifier_claim(r.q) == r.end_value(), "an honest run ends in a false claim");
      CHECK(fold_mask_form(k, h, r.d, r.eps, r.g) == r.mask_end.d, "prover and verifier fold d differently");
      // 3
      Run<Big> f = r;
      f.target = f.target + Big::one();
      CHECK(!(f.verifier_claim(f.q) == f.end_value()), "a false target ends in a true claim");
      // 5
      f.mask_sum = f.mask_sum + f.eps * (f.true_target() - f.target);
      CHECK(f.verifier_claim(f.q) == f.end_value(), "the attack that the order of messages prevents does not work as described");
      // 3, a changed coefficient
      for (unsigned j = 0; j < k; j++)
        for (unsigned t = 0; t < 3; t += 2) {
          auto sent = r.q;
          sent[j][t] = sent[j][t] + Big::one();
          CHECK(!(r.verifier_claim(sent) == r.end_value()), "a changed round polynomial ends in a true claim");
        }
    }

  // 1, the known answer, which fixes the order of fold_mask_table()'s columns: k = 3, h = 1; U's
  // seven coefficients are 1 .. 7, in hp_place()'s order (blocks/iop/hp.h); `weights` is
  // fold_mask_weights() at the challenges 2, 3, 5. `row0` and `row1` are also in Part 3 of
  // docs/format/toys/1-product.md, "2.1 The prover: the fresh mask, and one element".
  {
    std::vector<Big> w, g{Big{2}, Big{3}, Big{5}};
    for (u64 i = 1; i <= 7; i++) w.push_back(Big{i});
    const u64 row0[7] = {3, 0, 4, 6, 7, 5, 0}, row1[7] = {4, 2, 4, 6, 7, 5, 0}, weights[7] = {15, 30, 45, 5, 25, 1, 0};
    const std::vector<Big> table = fold_mask_table(3, 1, 7, w), u = fold_mask_weights(3, 1, 7, g);
    for (size_t c = 0; c < 7; c++)
      CHECK(table[0 + 2 * c] == Big{row0[c]} && table[1 + 2 * c] == Big{row1[c]} && u[c] == Big{weights[c]}, "the table's columns are not in the agreed order");
  }

  // 6
  for (unsigned k = 1; k <= 3; k++)
    for (unsigned h = 0; h <= k; h++)
      for (size_t slots = 2; slots <= 4; slots *= 2) check_kept_rows<Big>(k, h, slots, rng, big);

  // 4: all coins, over Z/5 (k <= 2) and Z/3 (k = 3); three random instances for each k, h
  bool one_to_one = true;
  auto hiding = [&](auto tag, u64 q, unsigned k, unsigned h) {
    using K = decltype(tag);
    for (int trial = 0; trial < 3; trial++) {
      Run<K> r = random_run<K>(k, h, rng, q);
      u64 all = 1;
      for (unsigned i = 0; i < 2 * k + 1; i++) all *= q;
      std::set<u64> seen;
      for (u64 code = 0; code < all; code++) {
        u64 rest = code;
        for (K &x : r.coins) x = K{rest % q}, rest /= q;
        r.play();
        u64 shown = r.mask_sum.v;
        for (unsigned j = 0; j < k; j++) shown = (shown * q + r.q[j][0].v) * q + r.q[j][2].v;
        seen.insert(shown);
      }
      if (seen.size() != all) one_to_one = false;
    }
  };
  for (unsigned k = 1; k <= 2; k++)
    for (unsigned h = 0; h <= k; h++) hiding(Zq<5>{}, 5, k, h);
  for (unsigned h = 0; h <= 3; h++) hiding(Zq<3>{}, 3, 3, h);
  CHECK(one_to_one, "the shown scalars are not uniform");

  std::printf("fold_rounds_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
