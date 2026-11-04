// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/code/code.h, the linear codes that encode rows (n = 2^log_n symbols from ell message
// entries and r random ones, the random tail): Encoder against the defining sum, decode(), the
// evaluation points, min_distance() and hiding_positions() by ranks of small matrices, rows over
// the extension field E, folds, mask positions and what code_ok() rejects. Prints one summary
// line; returns 0 if no check failed.
#include "code/code.h"
#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>
using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      if (failures++ < 20) std::printf("code_test: line %d fails: %s\n", __LINE__, #cond); \
    } \
  } while (0)

// SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.
static u64 state = 2;
static u64 word() {
  u64 z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static F rf() {
  F x;
  while (!F_from_word(word(), x)) {
  }
  return x;
}
static E re() {
  return E::from(rf(), rf(), rf());
}
// The rank of a matrix of rows, by elimination.
static size_t rank(std::vector<std::vector<F>> m) {
  size_t rk = 0;
  const size_t cols = m.empty() ? 0 : m[0].size();
  for (size_t c = 0; c < cols && rk < m.size(); c++) {
    size_t p = rk;
    while (p < m.size() && is_zero(m[p][c])) p++;
    if (p == m.size()) continue;
    std::swap(m[p], m[rk]);
    const F i = inv(m[rk][c]);
    for (size_t q = rk + 1; q < m.size(); q++) {
      const F t = m[q][c] * i;
      if (is_zero(t)) continue;
      for (size_t k = c; k < cols; k++) m[q][k] -= t * m[rk][k];
    }
    rk++;
  }
  return rk;
}
// Calls f on every subset of size k <= n of {0 .. n - 1}, or on `some` random ones if more exist.
template <class Fn> static void subsets(size_t n, size_t k, size_t some, Fn f) {
  double count = 1;
  for (size_t i = 0; i < k; i++) count = count * (double)(n - i) / (double)(i + 1);
  std::vector<size_t> s(k);
  if (count <= (double)some) {
    for (size_t i = 0; i < k; i++) s[i] = i;
    for (;;) {
      f(s);
      size_t i = k;
      while (i > 0 && s[i - 1] == n - k + i - 1) i--;
      if (i == 0) return;
      s[i - 1]++;
      for (size_t j = i; j < k; j++) s[j] = s[j - 1] + 1;
    }
  }
  for (size_t t = 0; t < some; t++) {
    std::set<size_t> pick;
    while (pick.size() < k) pick.insert(word() % n);
    std::copy(pick.begin(), pick.end(), s.begin());
    f(s);
  }
}

// If `ranks`, also runs the two rank checks, which need n <= 64.
static void one_code(const Code &c, bool ranks) {
  CHECK(code_ok(c));
  const size_t n = c.n();
  const Encoder enc(c);
  std::vector<F> m(c.ell), t(c.r), w(n), gm(c.ell), gt(c.r), f(64);
  for (auto &x : m) x = rf();
  for (auto &x : t) x = rf();
  enc.encode(m.data(), t.data(), w.data());

  // Message and tail fill disjoint runs of coefficients below dim().
  CHECK(c.message_at() + c.ell <= c.dim() && c.tail_at() + c.r <= c.dim() && c.dim() <= n && c.dim() >= c.ell + c.r);
  CHECK(c.r == 0 || c.message_at() + c.ell <= c.tail_at() || c.tail_at() + c.r <= c.message_at());
  CHECK(((u64)1 << c.log_dim()) >= c.dim() && (c.log_dim() == 0 || ((u64)1 << (c.log_dim() - 1)) < c.dim()));

  // Every position if n <= 64, as the rank checks need; otherwise the ends and a random choice.
  std::vector<size_t> at;
  for (size_t x = 0; x < n; x++)
    if (n <= 64 || x < 2 || x + 2 >= n || word() % (n / 32) == 0) at.push_back(x);
  std::vector<std::vector<F>> tail_rows(n), all_rows(n);
  for (size_t x : at) {
    // Definition: sum_j coefficient_j * basis_j; basis_j = product of f[i] over j's set bits i.
    factors(c, x, c.log_dim(), f.data());
    F direct = F::zero();
    for (size_t j = 0; j < c.ell; j++) direct += m[j] * basis_at(f.data(), c.log_dim(), c.message_at() + j);
    for (size_t j = 0; j < c.r; j++) direct += t[j] * basis_at(f.data(), c.log_dim(), c.tail_at() + j);
    CHECK(w[x] == direct);
    // The formula with message_scale and tail_scale, in three ways.
    CHECK(w[x] == symbol_at(c, x, m.data(), t.data()));
    generator_row(c, x, gm.data(), gt.data());
    F dense = F::zero();
    for (size_t j = 0; j < c.ell; j++) dense += m[j] * gm[j];
    for (size_t j = 0; j < c.r; j++) dense += t[j] * gt[j];
    CHECK(w[x] == dense);
    const GeneratorRow g = generator_at(c, x);
    CHECK(!is_zero(g.message_scale) && !is_zero(g.tail_scale));
    for (size_t j = 0; j < c.ell; j++) CHECK(gm[j] == g.message_scale * product_index(g.f, c.log_ell(), j));
    for (size_t j = 0; j < c.r; j++) CHECK(gt[j] == g.tail_scale * product_index(g.f, c.log_r(), j));
    for (unsigned i = 0; i < c.log_dim(); i++) CHECK(!is_zero(f[i]));
    tail_rows[x] = gt;
    all_rows[x].resize(c.dim());
    for (size_t j = 0; j < c.dim(); j++) all_rows[x][j] = basis_at(f.data(), c.log_dim(), j);
  }
  // decode() inverts encode(); unless every word is a codeword, it rejects w with a symbol changed.
  std::vector<F> m2(c.ell), t2(c.r);
  CHECK(enc.decode(w.data(), m2.data(), t2.data()));
  CHECK(std::equal(m.begin(), m.end(), m2.begin(), [](F a, F b) { return a == b; }));
  CHECK(std::equal(t.begin(), t.end(), t2.begin(), [](F a, F b) { return a == b; }));
  if (c.ell + c.r < n) {
    std::vector<F> bent = w;
    bent[word() % n] += F::one();
    CHECK(!enc.decode(bent.data(), m2.data(), t2.data()));
  }
  // A row over E is three rows over F.
  std::vector<E> me(c.ell), te(c.r), we(n);
  for (auto &x : me) x = re();
  for (auto &x : te) x = re();
  enc.encode(me.data(), te.data(), we.data());
  for (int k = 0; k < 3; k++) {
    for (size_t j = 0; j < c.ell; j++) m[j] = me[j].c[k];
    for (size_t j = 0; j < c.r; j++) t[j] = te[j].c[k];
    encode(c, m.data(), t.data(), w.data());
    for (size_t x = 0; x < n; x++) CHECK(we[x].c[k] == w[x]);
  }
  CHECK(we[n - 1] == symbol_at(c, n - 1, me.data(), te.data()) && we[0] == symbol_at(c, 0, me.data(), te.data()));
  // Encoding commutes with a fold of two rows at a point of E.
  {
    std::vector<F> a(c.ell), b(c.ell), ta(c.r), tb(c.r), wa(n), wb(n);
    for (auto &x : a) x = rf();
    for (auto &x : b) x = rf();
    for (auto &x : ta) x = rf();
    for (auto &x : tb) x = rf();
    enc.encode(a.data(), ta.data(), wa.data());
    enc.encode(b.data(), tb.data(), wb.data());
    const E g = re();
    for (size_t j = 0; j < c.ell; j++) me[j] = fold_pair(a[j], b[j], g);
    for (size_t j = 0; j < c.r; j++) te[j] = fold_pair(ta[j], tb[j], g);
    enc.encode(me.data(), te.data(), we.data());
    for (size_t x = 0; x < n; x++) CHECK(we[x] == fold_pair(wa[x], wb[x], g));
  }
  if (!ranks) return;
  // hiding_positions(): at every set of that many positions the tail's part of the generator matrix
  // has full rank, so a uniform tail makes the symbols there uniform. Each rank check tries at most
  // 300 sets per code (see subsets()), the same in each run.
  const size_t hp = hiding_positions(c);
  CHECK(hp <= c.r && hp + 1 >= c.r && tail_for_positions(hp) <= c.r);
  if (hp > 0 && hp <= n)
    subsets(n, hp, 300, [&](const std::vector<size_t> &s) {
      std::vector<std::vector<F>> sub;
      for (size_t x : s) sub.push_back(tail_rows[x]);
      CHECK(rank(sub) == hp);
    });
  // min_distance() = d: a non-zero codeword has at most n - d zeros, that is: on every set of
  // n - d + 1 positions the evaluation of the dim() basis functions is injective.
  const size_t d = min_distance(c), need = n - d + 1;
  CHECK(d >= 1 && need >= c.dim());
  if (need <= n)
    subsets(n, need, 300, [&](const std::vector<size_t> &s) {
      std::vector<std::vector<F>> sub;
      for (size_t x : s) sub.push_back(all_rows[x]);
      CHECK(rank(sub) == c.dim());
    });
}

int main() {
  // Every valid shape (log_n, ell, r) up to length 16, some up to 2048; the rank checks up to 32.
  size_t shapes = 0;
  for (unsigned ln = 1; ln <= 4; ln++)
    for (size_t ell = 1; ell <= ((size_t)1 << ln); ell++)
      for (size_t r = 0; r <= ((size_t)1 << ln); r++) {
        const Code c{ln, ell, r};
        if (!code_ok(c)) continue;
        one_code(c, true);
        shapes++;
      }
  for (unsigned ln = 5; ln <= 11; ln++)
    for (size_t ell : {(size_t)1, (size_t)2, (size_t)16, (size_t)1 << (ln - 2), (size_t)1 << (ln - 1), ((size_t)1 << (ln - 1)) - 3})
      for (size_t r : {(size_t)0, (size_t)1, (size_t)2, (size_t)5, (size_t)8, (size_t)21, (size_t)40, (size_t)1 << (ln - 2)}) {
        const Code c{ln, ell, r};
        if (!code_ok(c)) continue;
        one_code(c, ln <= 5);
        shapes++;
      }

  // Lengths up to 2^10. The evaluation points (factor 0) are all different.
  // Factor i does not depend on the bits below i of the position, and flipping bit i negates it.
  for (unsigned ln = 1; ln <= 10; ln++) {
    const Code c{ln, 1, 0};
    const size_t n = c.n();
    std::vector<std::vector<F>> f(n, std::vector<F>(ln));
    std::set<std::pair<u64, u64>> seen;
    for (size_t x = 0; x < n; x++) {
      factors(c, x, ln, f[x].data());
      seen.insert({f[x][0].v, ln > 1 ? f[x][1].v : 0});
    }
    CHECK(seen.size() == n);
    for (size_t x = 0; x < n; x++)
      for (unsigned i = 0; i < ln; i++) {
        CHECK(!is_zero(f[x][i]));
        CHECK(f[x][i] == f[x >> i << i][i]);
        CHECK(f[x][i] == -f[x ^ ((size_t)1 << i)][i]);
      }
  }


  // A mask code is read at the main code's position modulo its own length.
  {
    const Code main{10, 256, 20}, mask{6, 16, 9}, longer{11, 16, 9};
    CHECK(mask_fits(main, mask) && mask_fits(main, main) && !mask_fits(main, longer));
    for (u64 x : {(u64)0, (u64)63, (u64)64, (u64)65, (u64)1023}) CHECK(mask_position(main, mask, x) == x % 64 && short_position(10, 6, x) == x % 64);
  }
  for (u64 t = 0; t < 40; t++) {
    const u64 r = tail_for_positions(t);
    CHECK(hiding_positions(Code{20, 1, (size_t)r}) >= t && (r == 0 || hiding_positions(Code{20, 1, (size_t)(r - 1)}) < t));
  }

  // What code_ok() rejects, and two extreme shapes that it accepts.
  CHECK(!code_ok(Code{0, 1, 0}) && !code_ok(Code{CODE_MOST_LOG_N + 1, 1, 0}) && !code_ok(Code{4, 0, 3}));
  CHECK(!code_ok(Code{4, 17, 0}) && !code_ok(Code{4, 16, 1}) && !code_ok(Code{4, 1, 17}) && code_ok(Code{4, 16, 0}));
  CHECK(!code_ok(Code{4, ~(size_t)0, 2}) && !code_ok(Code{4, 2, ~(size_t)0}) && !code_ok(Code{4, ~(size_t)0, ~(size_t)0}));
  CHECK(code_ok(Code{CODE_MOST_LOG_N, 1, 0}));

  // Lengths up to 2^CODE_MOST_LOG_N, far beyond what transform_table() can store: factors() only.
  // Each factor is the square of the one before; the position's top bits pick the coset.
  for (unsigned ln = 3; ln <= CODE_MOST_LOG_N; ln++)
    for (int k = 0; k < 8; k++) {
      const u64 x = word() & (((u64)1 << ln) - 1);
      F a[64];
      const unsigned count = ln < 40 ? ln : 40;
      factors(Code{ln, 1, 0}, x, count, a);
      for (unsigned i = 0; i + 1 < count; i++) CHECK(a[i + 1] == a[i] * a[i]);
      const unsigned s = gl64_subgroup_log(ln);
      // The point a second way: 7^(1 + c) w^e, c = x >> s, e = the low s bits of x reversed;
      // w = 7^(5 (p - 1) / 2^s) generates the subgroup H of order 2^s.
      u64 reversed = 0;
      for (unsigned i = 0; i < s; i++)
        if ((x >> i) & 1) reversed |= (u64)1 << (s - 1 - i);
      const F w = pow(pow(pow(F{7}, (P - 1) >> 32), 5), (u64)1 << (32 - s));
      CHECK(a[0] == pow(F{7}, 1 + (x >> s)) * pow(w, reversed));
      CHECK(pow(a[0], (u64)1 << s) == pow(F{GL64_GENERATOR}, (1 + (x >> s)) << s)); // the point lies in the coset 7^(1 + c) H
    }
  // Length 2^33, two cosets of 2^32 points: dim() may reach 2^32 and no more, positions x and
  // x + 2^32 have different points, and symbol_at() gives the defining sum on both.
  {
    const Code c{33, 1000, 37};
    CHECK(code_ok(c) && code_ok(Code{33, ((size_t)1 << 32) - 37, 37}));
    CHECK(!code_ok(Code{33, ((size_t)1 << 32) - 36, 37}) && !code_ok(Code{34, ((size_t)1 << 32) + 1, 0}));
    std::vector<F> m(c.ell), t(c.r);
    for (auto &x : m) x = rf();
    for (auto &x : t) x = rf();
    for (int k = 0; k < 20; k++) {
      const u64 x = word() & (((u64)1 << 32) - 1), y = x + ((u64)1 << 32);
      F fx[33], fy[33];
      factors(c, x, 33, fx);
      factors(c, y, 33, fy);
      CHECK(fx[0] != fy[0] || fx[1] != fy[1]);
      for (u64 p : {x, y}) {
        F f[33], v = F::zero();
        factors(c, p, c.log_dim(), f);
        for (size_t j = 0; j < c.ell; j++) v += m[j] * basis_at(f, c.log_dim(), c.message_at() + j);
        for (size_t j = 0; j < c.r; j++) v += t[j] * basis_at(f, c.log_dim(), c.tail_at() + j);
        CHECK(v == symbol_at(c, p, m.data(), t.data()));
      }
    }
  }

  // The longest code encoded here, n = 2^16 at rate about 1/8: 200 symbols against symbol_at().
  {
    const Code c{16, (size_t)1 << 13, 301};
    const Encoder enc(c);
    std::vector<F> m(c.ell), t(c.r), w(c.n());
    for (auto &x : m) x = rf();
    for (auto &x : t) x = rf();
    enc.encode(m.data(), t.data(), w.data());
    for (int i = 0; i < 200; i++) {
      const u64 x = i < 4 ? (u64[]){0, 1, c.n() - 1, c.n() / 2}[i] : word() % c.n();
      CHECK(w[x] == symbol_at(c, x, m.data(), t.data()));
    }
  }
  std::printf("code_test (BLK_FIELD=%d): %zu shapes: %s\n", BLK_FIELD, shapes, failures ? "FAILED" : "ok");
  return failures != 0;
}
