// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks Send::form() (blocks/iop/send.h) entry by entry against its definition: the random linear
// form, with seed (z, beta), that the block SEND takes of an oracle message as its out-of-domain
// sample. Also checks where Send::pad() puts the pad, the random element that masks the answer.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixtures build plans by hand (blocks/numbers/plan.h)
#include <set>

#include "iop/tests/send_world.h"

using namespace blk;

namespace {

// The coefficients in `row.terms` and `row.units` (all that Send::form() fills) on one part, main
// or mask, of `o`.
std::vector<E> written_out(const Statement &S, const Row &row, OracleId o, Part part) {
  std::vector<E> c(size_t(1) << S.table_bits(o, part), E::zero());
  for (const Term &t : row.terms)
    if (t.oracle == o && t.part == part) t.v.add_to(c.data());
  for (const Unit &u : row.units)
    if (u.oracle == o && u.part == part) c[u.index] += u.coeff;
  return c;
}

int one_shape(const Codes &codes, OracleShape shape, Coins &coins) {
  Statement S(codes);
  const OracleId o = S.sent(shape);
  const SampleCube cube = Send::cube(codes, shape);
  const std::vector<PadPlace> pad = Send::pad(codes, shape);

  // The pad: per answer one element of E, the cubic extension of F (three entries in a row over F),
  // at the end of the last real mask row or, without mask rows, main row.
  const Part pad_part = shape.mask_rows ? Part::MASK : Part::MAIN;
  const size_t pad_rows = shape.mask_rows ? shape.mask_rows : shape.main_rows;
  const unsigned each = (shape.mask_rows || shape.main_deg == 3) ? 1 : 3;
  TEST_CHECK(pad.size() == size_t(each) * shape.answers);
  const unsigned row_bits = S.row_bits(o, pad_part);
  const size_t len = size_t(1) << S.place_bits(o, pad_part);
  for (size_t i = 0; i < pad.size(); i++) {
    TEST_CHECK(pad[i].part == pad_part);
    TEST_CHECK((pad[i].index & ((u64(1) << row_bits) - 1)) == pad_rows - 1);
    TEST_CHECK((pad[i].index >> row_bits) == len - pad.size() + i);
    TEST_CHECK(pad[i].answer == i / each && pad[i].basis == E::theta(i % each));
  }

  std::vector<SampleSeed> seeds(shape.answers);
  for (SampleSeed &s : seeds) {
    s.z = random_point(cube.bits + 2, coins);  // a shared seed may be longer than one message needs
    do s.beta = coins.e();
    while (is_zero(s.beta));
  }
  std::set<u64> places_on_the_cube;
  for (unsigned answer = 0; answer < shape.answers; answer++) {
    const Row row = Send::form(S, o, seeds[answer], answer);
    for (Part part : {Part::MAIN, Part::MASK}) {
      if (!S.has(o, part)) continue;
      const std::vector<E> c = written_out(S, row, o, part);
      const size_t rows = part == Part::MAIN ? shape.main_rows : shape.mask_rows;
      for (u64 j = 0; j < c.size(); j++) {
        if ((j & ((u64(1) << S.row_bits(o, part)) - 1)) >= rows) continue;  // beyond the real rows
        const PadPlace *p = nullptr;
        for (const PadPlace &q : pad)
          if (q.part == part && q.index == j) p = &q;
        if (p) {
          // pad entry: beta * `basis` in its answer's form, else 0; no term in z
          TEST_CHECK(c[j] == (p->answer == answer ? seeds[answer].beta * p->basis : E::zero()));
          continue;
        }
        // Elsewhere the equality polynomial eq(on_cube, z); one cube for both parts: top bit 0 for
        // main, 1 for mask.
        const bool both = shape.main_rows && shape.mask_rows;
        const u64 on_cube = j + ((both && part == Part::MASK) ? u64(1) << (cube.bits - 1) : 0);
        TEST_CHECK(on_cube >> cube.bits == 0);
        TEST_CHECK(c[j] == eq_index(on_cube, seeds[answer].z.data(), cube.bits));
        TEST_CHECK(!is_zero(c[j]));  // every entry counts (true but for a negligible set of seeds)
        if (answer == 0) TEST_CHECK(places_on_the_cube.insert(on_cube).second);  // no place twice
      }
    }
  }
  return 0;
}

}  // namespace

int main() {
  ToyCoins coins(11);
  Codes codes;
  codes.base = {Code{6, 8, 3}, Code{5, 2, 3}};
  codes.mask = {Code{5, 4, 3}, Code{5, 16, 3}};
  unsigned shapes = 0;
  for (CodeId code : {0u, 1u})
    for (CodeId mask_code : {0u, 1u})
      for (size_t main_rows : {0, 1, 2, 3, 5})
        for (size_t mask_rows : {0, 1, 2, 3})
          for (unsigned deg : {1u, 3u})
            for (unsigned answers : {1u, 2u}) {
              if (main_rows + mask_rows == 0) continue;
              if (!mask_rows && deg == 1 && codes.base[code].ell < 3 * answers) continue;  // no room for the pad
              OracleShape s;
              s.name = "o";
              s.code = code, s.mask_code = mask_code;
              s.main_rows = main_rows, s.main_deg = deg, s.mask_rows = mask_rows, s.answers = answers;
              if (one_shape(codes, s, coins)) return 1;
              shapes++;
            }
  std::printf("ok: the sample's form, entry by entry, on %u shapes\n", shapes);
  return 0;
}
