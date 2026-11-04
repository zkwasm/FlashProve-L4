// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Compares the verifier's field arithmetic (verifier/src/vf_common.h) with plain division by p, in
// F_p, p = 2^64 - 2^32 + 1, and in E = F_p[x] / (x^3 - w), w = 2.
// One mistake has a check of its own: a signed integer reduced by (i64)p, which is -(2^32 - 1), is
// right only for |v| < 2^32 - 1. Honest proofs need not reveal it: prover and verifier may share
// it, or a test's numbers may be too small.
// Build command: see SUITE.md. It sets -DBLK_FIELD=2, the number of this field (blocks/base.h).
#include "vf_subproof_engine.h" // with vf_common.h
#include <functional>
#include <set>
#include <tuple>

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      failures++; \
    } \
  } while (0)

static u64 state = 1;
static u64 word() { return state = mix64(state); }
static const u64 p = 0xFFFFFFFF00000001ull, w = 2; // repeated on purpose: compared with FIELD_P, FIELD_W
static u64 slow(u128 v) { return (u64)(v % p); }
// The product in E from its definition; every step reduces with slow().
static Ex product(Ex x, Ex y) {
  auto m = [](u64 a, u64 b) { return slow((u128)a * b); };
  auto s = [](u64 a, u64 b) { return slow((u128)a + b); };
  return Ex(s(m(x.c[0], y.c[0]), m(w, s(m(x.c[1], y.c[2]), m(x.c[2], y.c[1])))), s(s(m(x.c[0], y.c[1]), m(x.c[1], y.c[0])), m(w, m(x.c[2], y.c[2]))),
            s(s(m(x.c[0], y.c[2]), m(x.c[1], y.c[1])), m(x.c[2], y.c[0])));
}
static Ex some() { return Ex(word() % p, word() % p, word() % p); }
static Ex greatest() { return Ex(p - 1, p - 1, p - 1); }

int main() {
  CHECK(FIELD_P == p && FIELD_W == w && LE == 192);
  {
    // E is a field: 3 divides p - 1 and w^((p - 1) / 3) != 1, so w is not a cube modulo p.
    u64 r = 1, a = w;
    for (u64 e = (p - 1) / 3; e; e >>= 1, a = slow((u128)a * a))
      if (e & 1) r = slow((u128)r * a);
    CHECK((p - 1) % 3 == 0 && r != 1);
  }
  vector<u64> edge = {0, 1, 2, 5, (1ull << 32) - 2, (1ull << 32) - 1, 1ull << 32, (1ull << 32) + 1, p / 2, p / 2 + 1, p - 2, p - 1};
  for (int i = 0; i < 150; i++) edge.push_back(word() % p);
  for (u64 a : edge)
    for (u64 b : edge) {
      CHECK(fadd(a, b) == slow((u128)a + b) && fsub(a, b) == slow((u128)a + p - b) && fmul(a, b) == slow((u128)a * b));
      CHECK(fmad1(~0ull, a, b) == slow((u128)a * b + ~0ull));
      CHECK(fmad2(b, a, b, p - 1, a) == slow(slow((u128)a * b + b) + (u128)(p - 1) * a));
      CHECK(fmad3(a, p - 1, p - 1, p - 1, b, a, p - 1) == slow(slow(slow((u128)(p - 1) * (p - 1) + a) + (u128)(p - 1) * b) + (u128)a * (p - 1)));
    }
  for (int i = 0; i < 100000; i++) {
    const u128 v = ((u128)word() << 64 | word()) >> (i % 3 == 0 ? i % 128 : 0);
    CHECK(fredW(v) == slow(v));
    CHECK(fred(v) == slow(v));
  }
  // fint() and eI() on signed integers of every bit length: verifier/src/vf_norm_digit_rows.h passes
  // 2^47 and 2^48 to eI().
  for (int b = 0; b < 63; b++)
    for (i64 d = -1; d <= 1; d++)
      for (int sign = 0; sign < 2; sign++) {
        const i64 v = sign ? -((i64)(1ull << b) + d) : (i64)(1ull << b) + d;
        const u64 size = v < 0 ? 0 - (u64)v : (u64)v;
        CHECK(fint(v) == (v < 0 ? (p - size % p) % p : size % p));
        CHECK(eI(v) == eB(fint(v)));
      }
  CHECK(fint(INT64_MIN) == (p - (1ull << 63) % p) % p && fint(INT64_MAX) == (u64)INT64_MAX % p);
  // Products in E, also of the largest components, and inverses. operator* on Ex calls the product
  // of blocks/field/field.h, so this loop tests that code too.
  for (int i = 0; i < 20000; i++) {
    const Ex x = i == 0 ? greatest() : some(), y = i < 2 ? greatest() : some();
    CHECK(x * y == product(x, y));
    if (i < 300 && !isZ(x)) {
      CHECK(x * eInv(x) == eO() && product(x, eInv(x)) == eO());
    }
  }
  printf(failures ? "field_setting_test: %d FAILURES\n" : "field_setting_test, setting %d: all passed\n", failures ? failures : BLK_FIELD);
  return failures ? 1 : 0;
}
