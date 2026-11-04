// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks GL64_GENERATOR and GL64_ROOT_32 (blocks/code/cosets.h) by computation, and tests codes on
// several cosets: BLK_GL64_SUBGROUP_LOG = 4 limits a coset to 2^4 points. With cosets of 2^32
// points, blocks/code/tests/code_test.cpp spot-checks length 2^33 (two cosets).
// The field's modulus is p = 2^64 - 2^32 + 1 (`P` in the code).
#define BLK_GL64_SUBGROUP_LOG 4
#include "code/code.h"
#include <cstdio>
#include <set>
#include <vector>
using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      if (failures++ < 20) std::printf("cosets_test: line %d fails: %s\n", __LINE__, #cond); \
    } \
  } while (0)

// SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.
static u64 state = 4;
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
// gl64_point() for GL64_SUBGROUP_LOG = 32, implemented a second time: 7^(1 + c) w^rev, c = I >> s,
// rev = the low s bits of I, reversed.
static u64 point_32(unsigned log_n, u64 I) {
  const unsigned s = log_n < 32 ? log_n : 32;
  u64 rev = 0;
  for (unsigned i = 0; i < s; i++)
    if ((I >> i) & 1) rev |= 1ull << (s - 1 - i);
  const u64 w = gl64_pow(GL64_ROOT_32, 1ull << (32 - s)); // of order 2^s
  return GL64::mul(gl64_pow(7, 1 + (I >> s)), gl64_pow(w, rev));
}

int main() {
  // 7 generates the multiplicative group: 7^((p - 1) / q) != 1 for every prime q dividing p - 1.
  u64 rest = P - 1;
  for (u64 q : {2ull, 3ull, 5ull, 17ull, 257ull, 65537ull}) {
    CHECK(rest % q == 0 && gl64_pow(GL64_GENERATOR, (P - 1) / q) != 1);
    while (rest % q == 0) rest /= q;
  }
  CHECK(rest == 1);
  CHECK(GL64_ROOT_32 == gl64_pow(gl64_pow(GL64_GENERATOR, (P - 1) >> 32), 5));
  CHECK(gl64_pow(GL64_ROOT_32, 1ull << 31) == P - 1); // order exactly 2^32
  for (unsigned i = 0; i < 5; i++) CHECK(gl64_pow(GL64_ROOT_32, 1ull << (26 + i)) == 1ull << (3u << i)); // orders 64 .. 4: 2^3, 2^6, 2^12, 2^24, 2^48

  // Lengths up to 2^8 (16 cosets). H: the subgroup of 2^s points, s = min(ln, 4). Coset number
  // c = x >> s. m: a row's ell message entries; t: its random tail, r entries.
  size_t shapes = 0;
  for (unsigned ln = 1; ln <= 8; ln++) {
    const size_t n = (size_t)1 << ln;
    std::set<u64> seen;
    for (size_t x = 0; x < n; x++) {
      const u64 z = gl64_point(ln, x);
      seen.insert(z);
      CHECK(z != 0);
      const unsigned s = gl64_subgroup_log(ln);
      CHECK(gl64_pow(z, 1ull << s) == gl64_pow(GL64_GENERATOR, (1 + (x >> s)) << s)); // in the coset 7^(1 + c) H
    }
    CHECK(seen.size() == n);
    for (size_t ell = 1; ell <= 16; ell++)
      for (size_t r = 0; r <= 16; r++) {
        const Code c{ln, ell, r};
        CHECK(code_ok(c) == (ell + r <= n && ell + r <= 16)); // the dimension fits one coset
        if (!code_ok(c)) continue;
        const Encoder enc(c);
        std::vector<F> m(ell), t(r), w(n), m2(ell), t2(r);
        for (auto &x : m) x = rf();
        for (auto &x : t) x = rf();
        enc.encode(m.data(), t.data(), w.data());
        for (size_t x = 0; x < n; x++) {
          // The polynomial sum_j t_j Z^j + Z^r sum_j m_j Z^j at the point, by Horner's rule.
          const F z{gl64_point(ln, x)};
          F v = F::zero();
          for (size_t j = ell; j-- > 0;) v = v * z + m[j];
          for (size_t j = r; j-- > 0;) v = v * z + t[j];
          CHECK(w[x] == v && w[x] == symbol_at(c, x, m.data(), t.data()));
        }
        CHECK(enc.decode(w.data(), m2.data(), t2.data()));
        for (size_t j = 0; j < ell; j++) CHECK(m2[j] == m[j]);
        for (size_t j = 0; j < r; j++) CHECK(t2[j] == t[j]);
        if (ell + r < n) { // a change in the last coset alone is seen too
          w[n - 1] += F::one();
          CHECK(!enc.decode(w.data(), m2.data(), t2.data()));
        }
        shapes++;
      }
  }
  // gl64_point() uses the subgroup's size only through s, and here s <= 4: it is compared with
  // point_32() only at lengths up to 2^4, where both use one coset.
  for (unsigned ln = 1; ln <= 4; ln++)
    for (u64 x = 0; x < (1ull << ln); x++) CHECK(gl64_point(ln, x) == point_32(ln, x));
  std::printf("cosets_test (BLK_FIELD=2): %zu shapes: %s\n", shapes, failures ? "FAILED" : "ok");
  return failures != 0;
}
