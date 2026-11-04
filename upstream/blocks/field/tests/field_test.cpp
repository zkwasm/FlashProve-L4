// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the field F = Z/p and its cubic extension E = F[X]/(X^3 - NR): arithmetic against slow
// arithmetic on edge and random values, byte encodings, F_from_word(); then the multilinear
// helpers. It tests the CPU build only, not the GPU build.
// Prints `ok` or `FAILED` last; returns 0 if no check failed.
#include "field/field.h"
#include "field/mle.h"
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>
using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      if (failures++ < 20) std::printf("field_test: line %d fails: %s\n", __LINE__, #cond); \
    } \
  } while (0)

// SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.
static u64 state = 1;
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
// Slow arithmetic (prefix s): sums and products reduced by % on 128-bit numbers.
static u64 sadd(u64 a, u64 b) {
  return (u64)(((u128)a + b) % P);
}
static u64 smul(u64 a, u64 b) {
  return (u64)((u128)a * b % P);
}
static E smulE(const E &a, const E &b) {
  u64 t[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) t[i + j] = sadd(t[i + j], smul(a.c[i].v, b.c[j].v));
  return E::from(F{sadd(t[0], smul(NR, t[3]))}, F{sadd(t[1], smul(NR, t[4]))}, F{t[2]});
}

int main() {
  // Operands near which carries, borrows and reduction steps begin or stop.
  std::vector<u64> edge = {0, 1, 2, 3, 4, 5, 7, 8, P - 1, P - 2, P - 3, P - 5, P / 2, P / 2 + 1, P / 3, (1ull << 32) - 1, 1ull << 32,
                           (1ull << 32) + 1, (1ull << 31), (1ull << 60), (1ull << 60) - 1, (1ull << 60) + 1, 0xFFFFFFFFull << 29};
  if (F_BITS == 64)
    for (u64 v : std::initializer_list<u64>{1ull << 63, (1ull << 63) - 1, (1ull << 63) + 1, 0xFFFFFFFF00000000ull, 0xFFFFFFFEFFFFFFFFull, 0xFFFFFFFE00000001ull,
                  0xFFFFFFFE00000002ull, 0x00000001FFFFFFFFull, 0x8000000080000000ull})
      edge.push_back(v);
  for (u64 &v : edge) v %= P;
  for (int i = 0; i < 40; i++) edge.push_back(rf().v);

  for (u64 a : edge)
    for (u64 b : edge) {
      const F x{a}, y{b};
      CHECK((x + y).v == sadd(a, b));
      CHECK((x - y).v == sadd(a, P - b));
      CHECK((x * y).v == smul(a, b));
      CHECK((x + y).v < P && (x - y).v < P && (x * y).v < P);
      for (u64 c : std::initializer_list<u64>{edge[8], edge[9], edge[3], edge[edge.size() - 1]})
        CHECK(BaseField::dot3(a, b, b, c, c, a) == sadd(sadd(smul(a, b), smul(b, c)), smul(c, a)));
    }
  for (u64 a : edge) {
    const F x{a};
    CHECK((-x).v == (P - a) % P);
    CHECK(BaseField::times_nr(a) == smul(NR, a));
    if (a) CHECK(x * inv(x) == F::one());
    CHECK(pow(x, 0) == F::one() && pow(x, 1) == x && pow(x, 5) == x * x * x * x * x);
    CHECK(pow(x, P) == x); // Fermat
  }
  CHECK(inv(F::zero()) == F::zero() && is_zero(inv(E::zero())));
  // reduce128() on all pairs of extreme words, some larger than any product of words below p.
  for (u64 hi : std::initializer_list<u64>{0ull, 1ull, ~0ull, ~0ull - 1, 0xFFFFFFFFull, 0xFFFFFFFF00000000ull, 1ull << 63, 1ull << 58, (1ull << 58) - 1})
    for (u64 lo : std::initializer_list<u64>{0ull, 1ull, ~0ull, ~0ull - 1, 0xFFFFFFFFull, 0xFFFFFFFF00000000ull, 0xFFFFFFFF00000001ull, 1ull << 61, (1ull << 61) - 1})
      CHECK(BaseField::reduce128(hi, lo) == (u64)((((u128)hi << 64) | lo) % P));
  for (int i = 0; i < 100000; i++) {
    const u64 hi = word(), lo = word();
    CHECK(BaseField::reduce128(hi, lo) == (u64)((((u128)hi << 64) | lo) % P));
  }
  { // extreme 32-bit halves: all four borrow and carry cases of `GL64::reduce128`, some by just 1
    const u64 half[] = {0, 1, 2, 0x7FFFFFFFull, 0x80000000ull, 0xFFFFFFFEull, 0xFFFFFFFFull};
    for (u64 a : half)
      for (u64 b : half)
        for (u64 c : half)
          for (u64 d : half) {
            const u64 hi = a << 32 | b, lo = c << 32 | d;
            CHECK(BaseField::reduce128(hi, lo) == (u64)((((u128)hi << 64) | lo) % P));
          }
  }

  // Integers.
  for (i64 v : std::initializer_list<i64>{(i64)0, (i64)1, (i64)-1, (i64)5, (i64)-5, INT64_MAX, INT64_MIN, INT64_MIN + 1, (i64)1 << 61, -((i64)1 << 61), ((i64)1 << 61) - 1,
                -(((i64)1 << 61) - 1), (i64)1 << 62}) {
    __int128 m = (__int128)v % (__int128)P;
    if (m < 0) m += P;
    CHECK(from_i64(v).v == (u64)m);
  }
  for (u64 v : std::initializer_list<u64>{0ull, 1ull, P - 1, P, P + 1, ~0ull, ~0ull - 1, 2 * (P / 2) + 2}) CHECK(from_u64(v).v == v % P);

  // The extension: X^3 - NR is irreducible, since 3 divides p - 1 and NR is not a cube.
  CHECK((P - 1) % 3 == 0);
  CHECK(pow(F{NR}, (P - 1) / 3) != F::one());
  CHECK(pow(F{5}, (P - 1) / 3) == F::one()); // 5 is a cube modulo 2^64 - 2^32 + 1
  CHECK(E::theta(1) * E::theta(1) == E::theta(2) && E::theta(1) * E::theta(2) == E::from(F{NR}) && E::theta(0) == E::one());
  std::vector<E> ee;
  for (u64 a : std::initializer_list<u64>{0ull, 1ull, P - 1, P - 2, P / 2})
    for (u64 b : std::initializer_list<u64>{0ull, 1ull, P - 1, P / 2 + 1})
      for (u64 c : std::initializer_list<u64>{0ull, 1ull, P - 1, P - 3}) ee.push_back(E::from(F{a}, F{b}, F{c}));
  for (int i = 0; i < 40; i++) ee.push_back(re());
  for (const E &x : ee) {
    for (const E &y : ee) {
      CHECK(x * y == smulE(x, y));
      CHECK(x + y - y == x);
    }
    if (!is_zero(x)) CHECK(x * inv(x) == E::one());
    CHECK(x * F{P - 1} == -x && F{3} * x == x + x + x && x + F{2} == x + E::from(F{2}) && x - F{2} == x - E::from(F{2}));
    // x^(p^3) = x
    CHECK(pow(pow(pow(x, P), P), P) == x);
    CHECK(pow(x, 0) == E::one() && pow(x, 3) == x * x * x);
  }
  for (int i = 0; i < 2000; i++) {
    const E x = re(), y = re(), z = re();
    CHECK((x * y) * z == x * (y * z) && x * (y + z) == x * y + x * z && x * y == smulE(x, y));
  }

  // Bytes: each element has one valid encoding.
  {
    u8 b[24];
    F x;
    E y;
    for (u64 v : std::initializer_list<u64>{0ull, 1ull, P - 1}) {
      word_to_bytes(v, b);
      CHECK(F_from_bytes(b, x) && x.v == v);
    }
    for (u64 v : std::initializer_list<u64>{P, P + 1, ~0ull, P | (1ull << 63), P + (1ull << 31)}) {
      if (v < P) continue;
      word_to_bytes(v, b);
      x = F::one();
      CHECK(!F_from_bytes(b, x) && x == F::zero());
    }
    F_to_bytes(F{0x0102030405060708ull % P}, b);
    CHECK(b[0] == 0x08 && b[1] == 0x07 && b[6] == 0x02); // low byte first
    for (int i = 0; i < 1000; i++) {
      const E e = re();
      E_to_bytes(e, b);
      CHECK(E_from_bytes(b, y) && y == e);
      for (int k = 0; k < 3; k++) CHECK(word_from_bytes(b + 8 * k) == e.c[k].v);
    }
    for (int bad = 0; bad < 3; bad++) { // one coordinate not below p makes the encoding invalid
      E_to_bytes(re(), b);
      word_to_bytes(P, b + 8 * bad);
      CHECK(!E_from_bytes(b, y) && is_zero(y));
      word_to_bytes(~0ull, b + 8 * bad);
      CHECK(!E_from_bytes(b, y) && is_zero(y));
    }
  }

  // Draws: F_from_word() takes the low F_BITS bits and rejects ("refuses") them if not below p; it
  // never reduces them modulo p. `refused` is the share of all words that it rejects.
  {
    F x;
    const u64 mask = F_BITS == 64 ? ~0ull : (1ull << F_BITS) - 1;
    for (u64 w : std::initializer_list<u64>{0ull, 1ull, P - 1, P, P + 1, ~0ull, mask, mask - 1, P | (7ull << 61), (P - 1) | (7ull << 61), 1ull << 63}) {
      const u64 low = w & mask;
      CHECK(F_from_word(w, x) == (low < P));
      CHECK(x.v == (low < P ? low : 0));
    }
    CHECK(std::fabs(LOG2_F - (double)std::log2((long double)P)) < 1e-12 && LOG2_E == 3 * LOG2_F);
    CHECK((long double)LOG2_F < std::log2((long double)P)); // LOG2_F is rounded down, not up
    const long double refused = (long double)(mask - P + 1) / (std::pow(2.0L, (int)F_BITS));
    CHECK(std::log2(refused) <= LOG2_DRAW_REFUSAL && std::log2(refused) > LOG2_DRAW_REFUSAL - 1e-6);
  }

  // Multilinear helpers. A vector v has the value sum_j eq(j, z) v[j] at z.
  for (unsigned k = 0; k <= 6; k++) {
    const size_t N = (size_t)1 << k;
    std::vector<E> z(k + 1), w(k + 1), t(N), ve(N);
    std::vector<F> vf(N), f(k + 1), pt(N);
    for (auto &x : z) x = re();
    for (auto &x : w) x = re();
    for (auto &x : ve) x = re();
    for (auto &x : vf) x = rf();
    for (auto &x : f) x = rf();
    eq_table(z.data(), k, t.data());
    E sum = E::zero(), both = E::zero();
    std::vector<E> tw(N);
    eq_table(w.data(), k, tw.data());
    for (size_t j = 0; j < N; j++) {
      CHECK(t[j] == eq_index(j, z.data(), k));
      sum += t[j];
      both += t[j] * tw[j];
    }
    CHECK(sum == E::one() && both == eq_eval(z.data(), w.data(), k));
    // eq(z, w) is the value at w of the vector eq(., z).
    CHECK(eq_eval(z.data(), w.data(), k) == mle_eval(t.data(), N, w.data(), k));
    // mle_eval() at the bits of j returns entry j.
    for (size_t j = 0; j < N; j++) {
      std::vector<E> bits(k + 1);
      for (unsigned i = 0; i < k; i++) bits[i] = ((j >> i) & 1) ? E::one() : E::zero();
      CHECK(mle_eval(ve.data(), N, bits.data(), k) == ve[j] && mle_eval(vf.data(), N, bits.data(), k) == E::from(vf[j]));
    }
    product_table(f.data(), k, N, pt.data());
    for (size_t len = 0; len <= N; len++) {
      E a = E::zero(), b = E::zero(), c = E::zero();
      F d = F::zero();
      for (size_t j = 0; j < len; j++) a += t[j] * ve[j], b += t[j] * vf[j], c += ve[j] * pt[j], d += vf[j] * pt[j];
      CHECK(mle_eval(ve.data(), len, z.data(), k) == a && mle_eval(vf.data(), len, z.data(), k) == b);
      CHECK(product_dot(ve.data(), len, f.data(), k) == c && product_dot(vf.data(), len, f.data(), k) == d);
    }
    E pm = E::zero();
    for (size_t j = 0; j < N; j++) {
      CHECK(pt[j] == product_index(f.data(), k, j));
      pm += t[j] * pt[j];
    }
    CHECK(product_mle(f.data(), z.data(), k) == pm);
    E pe = E::zero(), xp = E::one();
    for (size_t j = 0; j < N; j++) pe += ve[j] * xp, xp *= z[0];
    CHECK(poly_eval(ve.data(), N, z[0]) == pe && is_zero(poly_eval(ve.data(), 0, z[0])));
  }
  std::printf("field_test (BLK_FIELD=%d): %s\n", BLK_FIELD, failures ? "FAILED" : "ok");
  return failures != 0;
}
