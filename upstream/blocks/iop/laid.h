// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Describes how matrices are placed (`laid` in identifiers) in the rows of one table, checks such a
// layout, and turns the weights of a linear claim on a matrix into a Form (a structured coefficient
// vector, tensor.h) on the table. The weight of element (i, c) must be a product on_i[i] * on_c[c].
//
// A prover commits to matrices of many shapes, but the main rows of an oracle message (encoded rows
// sent as Merkle roots), which are the table here, all have one length. A public layout cuts each
// matrix into blocks whose sides are powers of two. Each block fills an aligned rectangle of the
// table: adjacent rows, and in each the same range of places (indices within a row).
//  * A layout is input to a verifier, which must check it with laid_ok() first. laid_tensor()
//    sees one block only, and ends the process unless laid_block_ok() accepts it.
//  * A matrix element is one committed table entry. Where several cells (small integers) are packed
//    into an entry, no Form can weight a single cell.
//  * Table entries that no block covers get coefficient 0 here, but an oracle message's
//    out-of-domain samples (random linear forms, Send::form() in send.h) weight them too.
#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "iop/tensor.h"

namespace blk {

// The elements (row first_i + i, column first_c + c) of a matrix, for i < 2^bits_i and
// c < 2^bits_c, stored in 2^row_bits table rows at 2^place_bits places each. Element number
// x = c * 2^bits_i + i (if i_is_low) or i * 2^bits_c + c is in row first_row + (x >> place_bits) at
// place first_place + (x mod 2^place_bits). In a valid block,
// row_bits + place_bits = bits_i + bits_c, each first_ field is a multiple of its side and below
// 2^48, and each bits field is below 48.
struct LaidBlock {
  u64 first_i = 0, first_c = 0, first_row = 0, first_place = 0;
  unsigned bits_i = 0, bits_c = 0, row_bits = 0, place_bits = 0;
  bool i_is_low = false;
};

// Needed, with public_fault(), to bind a block into a hash chain (blocks/chain/public.h).
inline void public_bytes(Writer &w, const LaidBlock &b) {
  for (u64 x : {b.first_i, b.first_c, b.first_row, b.first_place, (u64)b.bits_i, (u64)b.bits_c, (u64)b.row_bits, (u64)b.place_bits, (u64)b.i_is_low}) w.put_u64(x);
}

// Returns nullptr if the block is valid (see LaidBlock). laid_block_ok() also checks that it is
// inside the table, and laid_ok() that it is inside its matrix.
inline const char *public_fault(const LaidBlock &b) {
  auto aligned = [](u64 first, unsigned bits) { return bits < 48 && first >> 48 == 0 && (first >> bits << bits) == first; };
  return b.row_bits + b.place_bits == b.bits_i + b.bits_c && aligned(b.first_i, b.bits_i) && aligned(b.first_c, b.bits_c) && aligned(b.first_row, b.row_bits) && aligned(b.first_place, b.place_bits)
             ? nullptr
             : "laid: a block that is not aligned, or of another size in the table than in its matrix";
}
inline bool laid_block_ok(const LaidBlock &b, unsigned table_row_bits, unsigned table_place_bits) {
  auto aligned = [](u64 first, unsigned bits) { return bits < 48 && first >> 48 == 0 && (first >> bits << bits) == first; };
  return b.row_bits + b.place_bits == b.bits_i + b.bits_c && b.row_bits <= table_row_bits && b.place_bits <= table_place_bits &&
         aligned(b.first_i, b.bits_i) && aligned(b.first_c, b.bits_c) && aligned(b.first_row, b.row_bits) &&
         aligned(b.first_place, b.place_bits) && (b.first_row >> table_row_bits) == 0 && (b.first_place >> table_place_bits) == 0;
}

// One matrix of a layout: size_i rows, size_c columns, and its blocks.
struct LaidMatrix {
  u64 size_i = 0, size_c = 0;
  std::vector<LaidBlock> blocks;
};

// True if every block is valid and inside its matrix and the table, every element of every matrix
// is in exactly one block, and no table entry gets two elements. Not checked: that the blocks' rows
// are sent (an oracle message may have fewer than 2^table_row_bits main rows).
inline bool laid_ok(const std::vector<LaidMatrix> &matrices, unsigned table_row_bits, unsigned table_place_bits) {
  auto meet = [](u64 a, unsigned a_bits, u64 b, unsigned b_bits) { return a < b + (u64(1) << b_bits) && b < a + (u64(1) << a_bits); };
  std::vector<const LaidBlock *> all;
  for (const LaidMatrix &m : matrices) {
    u128 covered = 0;
    for (size_t x = 0; x < m.blocks.size(); x++) {
      const LaidBlock &b = m.blocks[x];
      if (!laid_block_ok(b, table_row_bits, table_place_bits)) return false;
      if (b.first_i + (u64(1) << b.bits_i) > m.size_i || b.first_c + (u64(1) << b.bits_c) > m.size_c) return false;
      for (size_t y = 0; y < x; y++)
        if (meet(b.first_i, b.bits_i, m.blocks[y].first_i, m.blocks[y].bits_i) && meet(b.first_c, b.bits_c, m.blocks[y].first_c, m.blocks[y].bits_c))
          return false;
      covered += (u128)1 << (b.bits_i + b.bits_c);
      all.push_back(&b);
    }
    if (covered != (u128)m.size_i * m.size_c) return false;  // the blocks are disjoint and inside, so this is full cover
  }
  for (size_t x = 0; x < all.size(); x++)
    for (size_t y = 0; y < x; y++)
      if (meet(all[x]->first_row, all[x]->row_bits, all[y]->first_row, all[y]->row_bits) &&
          meet(all[x]->first_place, all[x]->place_bits, all[y]->first_place, all[y]->place_bits))
        return false;
  return true;
}

// Weights on one index (i or c) of a matrix. Product form: index j has the weight
// prod over t < lo.size() of (bit t of j ? hi[t] : lo[t]); lo and hi must have equal length, which
// is not checked. Written out, if `values` is set: index j has the weight (*values)[j].
struct IndexWeights {
  std::vector<E> lo, hi;
  std::shared_ptr<const std::vector<E>> values;
  // Prover only, instead of `values`: the weight of index j < held_count is entry held_first + j of
  // table `held` (held_length entries), which only the prover's engine (CPU or GPU back end)
  // can read: see Tensor::held.
  TableId held = Tensor::NOT_HELD;
  u64 held_first = 0, held_count = 0, held_length = 0;
  bool written_out() const { return values || held != Tensor::NOT_HELD; }

  static IndexWeights eq(const std::vector<E> &point) {  // point[t] belongs to bit t of the index
    IndexWeights w;
    for (const E &z : point) w.lo.push_back(E::one() - z), w.hi.push_back(z);
    return w;
  }
  static IndexWeights powers(E base, unsigned bits) {  // index i -> base^i, for i below 2^bits
    IndexWeights w;
    for (unsigned t = 0; t < bits; t++, base = base * base) w.lo.push_back(E::one()), w.hi.push_back(base);
    return w;
  }
  static IndexWeights written(std::shared_ptr<const std::vector<E>> values) {
    IndexWeights w;
    w.values = std::move(values);
    return w;
  }
  static IndexWeights held_by_engine(TableId table, u64 first, u64 count, u64 table_length) {
    BLK_REQUIRE(table != Tensor::NOT_HELD && first <= table_length && count <= table_length - first, "laid: no table, or the stretch does not lie in the table");
    IndexWeights w;
    w.held = table, w.held_first = first, w.held_count = count, w.held_length = table_length;
    return w;
  }
};

// scalar * on_i[i] * on_c[c] on the table entry of each matrix element (i, c) in the block and 0
// elsewhere, as one Tensor on a table of 2^table_row_bits rows of 2^table_place_bits places. The
// Tensor's variables are the table index's bits, row bits lowest. Each carries a bit of i or c, or
// picks a fixed bit of the block's position. At most one of on_i and on_c may be written out.
inline Tensor laid_tensor(const LaidBlock &b, unsigned table_row_bits, unsigned table_place_bits, const IndexWeights &on_i,
                          const IndexWeights &on_c, const E &scalar) {
  BLK_REQUIRE(laid_block_ok(b, table_row_bits, table_place_bits), "laid: a block that is not aligned or does not fit the table");
  BLK_REQUIRE(!(on_i.written_out() && on_c.written_out()), "laid: only one of the two weights can be a written-out vector");
  Tensor t;
  t.scalar = scalar;
  std::vector<std::pair<unsigned, unsigned>> dense;  // (bit of the written-out vector's index, variable)
  const unsigned low_bits = b.i_is_low ? b.bits_i : b.bits_c;
  auto push_bit_of_x = [&](unsigned x_bit) {  // appends the variable that carries this bit of the element number
    const bool of_i = (x_bit < low_bits) == b.i_is_low;
    const unsigned bit = x_bit < low_bits ? x_bit : x_bit - low_bits;
    const IndexWeights &w = of_i ? on_i : on_c;
    if (w.written_out()) {
      dense.push_back({bit, t.vars()});
      t.push(E::one(), E::one());
    } else {
      BLK_REQUIRE(bit < w.lo.size(), "laid: a product form with too few bits for its index");
      t.push(w.lo[bit], w.hi[bit]);
    }
  };
  for (unsigned x = b.place_bits; x < b.place_bits + b.row_bits; x++) push_bit_of_x(x);  // low row bits: the high bits of x
  t.insert_choice(t.vars(), table_row_bits - b.row_bits, b.first_row >> b.row_bits);
  for (unsigned x = 0; x < b.place_bits; x++) push_bit_of_x(x);  // low place bits: the low bits of x
  t.insert_choice(t.vars(), table_place_bits - b.place_bits, b.first_place >> b.place_bits);
  // Index bits above the block's side are constant on the block: their factors go into the scalar.
  auto above = [&](const IndexWeights &w, u64 first, unsigned bits) {
    if (w.written_out()) return;
    BLK_REQUIRE(w.lo.size() >= 64 || (first >> w.lo.size()) == 0, "laid: a product form with too few bits for its index");
    for (unsigned i = bits; i < w.lo.size(); i++) t.scalar = t.scalar * (((first >> i) & 1) ? w.hi[i] : w.lo[i]);
  };
  above(on_i, b.first_i, b.bits_i), above(on_c, b.first_c, b.bits_c);
  if (const std::shared_ptr<const std::vector<E>> &values = on_i.values ? on_i.values : on_c.values) {
    const u64 first = on_i.values ? b.first_i : b.first_c;
    if (dense.empty()) {
      BLK_REQUIRE(first < values->size(), "laid: the written-out vector is too short");
      t.scalar = t.scalar * (*values)[first];
    } else {
      t.dense_vars.assign(dense.size(), 0);
      for (const std::pair<unsigned, unsigned> &d : dense) t.dense_vars[d.first] = d.second;
      t.set_values(values, first);
    }
  } else if (on_i.written_out() || on_c.written_out()) {  // in the engine's table: not read here
    const IndexWeights &w = on_i.written_out() ? on_i : on_c;
    const u64 first = on_i.written_out() ? b.first_i : b.first_c;
    BLK_REQUIRE(dense.size() < 64 && first <= w.held_count && (u64(1) << dense.size()) <= w.held_count - first, "laid: the written-out vector is too short");
    t.dense_vars.assign(dense.size(), 0);
    for (const std::pair<unsigned, unsigned> &d : dense) t.dense_vars[d.first] = d.second;
    t.set_held(w.held, w.held_first + first, w.held_length);
  }
  return t;
}

// The sum of laid_tensor() over `blocks`, the blocks of one matrix.
inline Form laid_form(const std::vector<LaidBlock> &blocks, unsigned table_row_bits, unsigned table_place_bits, const IndexWeights &on_i,
                      const IndexWeights &on_c, const E &scalar) {
  Form f;
  for (const LaidBlock &b : blocks) f.add(laid_tensor(b, table_row_bits, table_place_bits, on_i, on_c, scalar));
  return f;
}

}  // namespace blk
