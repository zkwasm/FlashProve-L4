// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test of blocks/iop/laid.h, which places matrices as aligned power-of-two blocks in one table.
// For random blocks in tables of at most 8 rows of 32 places, the `Tensor` (a coefficient vector
// in product form, blocks/iop/tensor.h) from laid_tensor() is compared entry by entry with weights
// placed by the rule at `LaidBlock`. The weights on both matrix indices are product forms, or
// those on one index are a written-out vector. laid_ok() must accept a valid layout and reject it
// after each of nine changes (printed: "lawful", "refused"). Not called: laid_form(),
// IndexWeights::powers(), public_bytes(). No input that a BLK_REQUIRE in laid.h rejects is tried.
#include <cstdio>

#include "iop/laid.h"

using namespace blk;

static u64 state = 11;  // SplitMix64 (Steele, Lea, Flood, OOPSLA 2014)
static u64 word() {
  u64 z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static std::vector<E> elements(size_t n) {
  std::vector<E> v(n);
  for (E &x : v)
    for (F &c : x.c)
      while (!F_from_word(word(), c)) {
      }
  return v;
}
// The weight of `index`, from the definition at `IndexWeights`.
static E weight(const IndexWeights &w, u64 index) {
  if (w.values) return (*w.values)[index];
  E r = E::one();
  for (unsigned t = 0; t < w.lo.size(); t++) r = r * (((index >> t) & 1) ? w.hi[t] : w.lo[t]);
  return r;
}

int main() {
  int failures = 0;
  for (int trial = 0; trial < 400; trial++) {
    // A table of 2^k rows of 2^places places; entry (row, place) has index row + (place << k).
    const unsigned k = word() % 4, places = 1 + word() % 5;
    LaidBlock b;
    b.row_bits = word() % (k + 1), b.place_bits = word() % (places + 1);
    b.bits_i = word() % (b.row_bits + b.place_bits + 1), b.bits_c = b.row_bits + b.place_bits - b.bits_i;
    b.i_is_low = word() & 1;
    b.first_row = (word() % (u64(1) << (k - b.row_bits))) << b.row_bits;
    b.first_place = (word() % (u64(1) << (places - b.place_bits))) << b.place_bits;
    b.first_i = (word() % 3) << b.bits_i, b.first_c = (word() % 3) << b.bits_c;
    IndexWeights on_i = IndexWeights::eq(elements(b.bits_i + 2)), on_c;
    on_c.lo = elements(b.bits_c + 2), on_c.hi = elements(b.bits_c + 2);
    if (trial % 3 == 1) on_i = IndexWeights::written(std::make_shared<const std::vector<E>>(elements(size_t(3) << b.bits_i)));
    if (trial % 3 == 2) on_c = IndexWeights::written(std::make_shared<const std::vector<E>>(elements(size_t(3) << b.bits_c)));
    const E scalar = elements(1)[0];
    std::vector<E> want(size_t(1) << (k + places), E::zero()), got(want.size(), E::zero());
    for (u64 i = 0; i < (u64(1) << b.bits_i); i++)
      for (u64 c = 0; c < (u64(1) << b.bits_c); c++) {
        const u64 x = b.i_is_low ? (c << b.bits_i) + i : (i << b.bits_c) + c;
        want[(b.first_row + (x >> b.place_bits)) + ((b.first_place + (x & ((u64(1) << b.place_bits) - 1))) << k)] =
            scalar * weight(on_i, b.first_i + i) * weight(on_c, b.first_c + c);
      }
    laid_tensor(b, k, places, on_i, on_c, scalar).add_to(got.data());
    if (got != want) failures++, std::printf("FAILED: trial %d: a block's tensor differs from the weights laid out\n", trial);
    // The same written-out vector, given as a range (from entry 5) of a longer vector that only a
    // prover's engine can read (`Tensor::held`; the engine, blocks/engine/engine.h, keeps large
    // tables). The tensor must record that vector's id and length, have no values, return 0 from
    // at() outside the block, and give the same weights once with_table() supplies the entries. In
    // some trials the block is one element wide in the written-out index: the tensor then has no
    // dense variable.
    if (trial % 3) {
      const std::vector<E> &stretch = *(trial % 3 == 1 ? on_i : on_c).values;
      auto table = std::make_shared<std::vector<E>>(elements(5));
      table->insert(table->end(), stretch.begin(), stretch.end());
      table->push_back(elements(1)[0]);
      const IndexWeights held = IndexWeights::held_by_engine(7, 5, stretch.size(), table->size());
      const Tensor t = laid_tensor(b, k, places, trial % 3 == 1 ? held : on_i, trial % 3 == 2 ? held : on_c, scalar);
      std::vector<E> again(want.size(), E::zero());
      bool good = t.is_held() && t.held == 7 && t.held_length == table->size() && !t.values;
      t.with_table(table).add_to(again.data());
      for (u64 j = 0; j < want.size(); j++)
        if ((j & t.pick_mask()) != t.pick_value()) good = good && is_zero(t.at(j)) && is_zero(want[j]);
      if (!good || again != want) failures++, std::printf("FAILED: trial %d: a block's tensor on a vector in the engine's hands differs from the weights laid out\n", trial);
    }
  }
  {
    // Two matrices in a table of 4 rows of 8 places: 3 x 4 in three blocks, 2 x 2 in one.
    auto block = [](u64 i, unsigned bi, u64 c, unsigned bc, u64 row, unsigned rb, u64 place, unsigned pb) {
      LaidBlock b;
      b.first_i = i, b.bits_i = bi, b.first_c = c, b.bits_c = bc, b.first_row = row, b.row_bits = rb, b.first_place = place, b.place_bits = pb;
      return b;
    };
    const std::vector<LaidMatrix> good = {{3, 4, {block(0, 1, 0, 2, 0, 1, 0, 2), block(2, 0, 0, 1, 2, 0, 0, 1), block(2, 0, 2, 1, 2, 0, 2, 1)}},
                                          {2, 2, {block(0, 1, 0, 1, 0, 0, 4, 2)}}};
    if (!laid_ok(good, 2, 3)) failures++, std::printf("FAILED: a lawful layout is refused\n");
    for (const LaidMatrix &m : good)
      for (const LaidBlock &b : m.blocks)
        if (public_fault(b)) failures++, std::printf("FAILED: a lawful block has a fault by itself\n");
    for (int change = 0; change < 9; change++) {
      std::vector<LaidMatrix> bad = good;
      if (change == 0) bad[0].blocks.pop_back();                 // an element in no block
      if (change == 1) bad[0].blocks[2] = bad[0].blocks[1];      // an element in two blocks (and one in none)
      if (change == 2) bad[1].blocks[0].first_place = 0;         // two elements in one entry of the table
      if (change == 3) bad[1].blocks[0].first_place = 2;         // not aligned
      if (change == 4) bad[1].blocks[0].first_row = 4;           // outside the table
      if (change == 5) bad[1].size_c = 1;                        // outside its matrix
      // Changes 6 to 8 each break exactly one condition of laid_ok(), so this test fails if that
      // condition is removed. The blocks stay aligned and inside the table, no two blocks overlap
      // there, and the blocks' element counts add up to the matrix's. Changes 5 and 1 break those
      // of 6 and 7, and a second one each. With changes 0 to 5 only, this test still printed ok
      // with any one of the three conditions removed.
      if (change == 6) bad[0].blocks[2].first_i = 3;    // outside its matrix: (3, 2), (3, 3)
      if (change == 7) bad[0].blocks[2].first_c = 0;    // (2, 0), (2, 1) are in two blocks
      if (change == 8) bad[1].blocks[0].place_bits = 1; // 4 elements in 2 entries of the table
      if (change == 8 && !public_fault(bad[1].blocks[0])) failures++, std::printf("FAILED: a block of another size in the table than in its matrix has no fault by itself\n");
      if (laid_ok(bad, 2, 3)) failures++, std::printf("FAILED: an unlawful layout is accepted (change %d)\n", change);
    }
  }
  std::printf("laid_test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
