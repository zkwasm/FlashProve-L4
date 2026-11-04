// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The accounting of the rounds' soundness errors (blocks/numbers/setting.h, blocks/numbers/codes.h)
// restates the field's and the codes' parameters. This test compares them with
// blocks/field/field.h and blocks/code/code.h, for setting_goldilocks() only: the prime, the
// extension degree, element sizes in bytes; for many shapes of codes, length, dimension, distance
// and how many positions a random tail (a row's r random entries) hides; and the tail length needed
// to hide t positions. A shape that code_ok() rejects is skipped; the last line of output says how
// many shapes were compared.
#include "code/code.h"
#include "numbers/codes.h"
#include "numbers/setting.h"
#include <cstdio>
using namespace blk;

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      if (failures++ < 20) std::printf("numbers_test: line %d fails: %s\n", __LINE__, #cond); \
    } \
  } while (0)

int main() {
  const Setting s = setting_goldilocks();
  CHECK(!s.circle);
  CHECK(s.p == Big::two_to(32) * Big(P >> 32) + Big(P & 0xFFFFFFFFull));
  CHECK(s.degree == E_DEGREE && s.f_bytes == F_BYTES && s.e_bytes == E_BYTES);
  size_t shapes = 0;
  auto one = [&](unsigned log_n, u64 ell, u64 r) {
    const Code c{log_n, (size_t)ell, (size_t)r};
    if (!code_ok(c)) return;
    CodeSize z;
    z.log_n = log_n, z.ell = ell, z.r = r;
    CHECK(z.n() == c.n() && z.coordinates() == ell + r && z.dim(s) == c.dim());
    // The accounting's distance, n - k, equals min_distance() for plain Reed-Solomon codes
    // (`circle` false). The check also accepts 1 below (conservative), never above.
    CHECK(z.n() - z.k(s) <= min_distance(c) && z.n() - z.k(s) + 1 >= min_distance(c));
    CHECK(hidden_positions(s, z) == hiding_positions(c));
    shapes++;
  };
  for (unsigned log_n = 1; log_n <= 7; log_n++)
    for (u64 ell = 1; ell <= ((u64)1 << log_n); ell++)
      for (u64 r = 0; r <= ((u64)1 << log_n); r++) one(log_n, ell, r);
  for (unsigned log_n : {10u, 16u, 20u, 24u, 30u, 32u, 33u, 40u})
    for (unsigned a = 0; a < log_n; a++)
      for (u64 ell : {(u64)1 << a, ((u64)1 << a) + 1, ((u64)3 << a) - 1})
        for (u64 r : {(u64)0, (u64)1, (u64)2, (u64)37, (u64)64, (u64)65, (u64)100, (u64)127, (u64)128, (u64)129, (u64)665, (u64)4096, (u64)1 << a}) one(log_n, ell, r);
  for (u64 t = 0; t < 3000; t++) CHECK(tail_for(s, t) == tail_for_positions(t));
  std::printf("numbers_test (BLK_FIELD=%d): %zu shapes: %s\n", BLK_FIELD, shapes, failures ? "FAILED" : "ok");
  return failures != 0;
}
