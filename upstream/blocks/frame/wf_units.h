// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives what WF, the committer's range proof by lookups, looks up (`WfShape` in wf_frame.h) from
// the row layout of the committed matrix. Prover and verifier each call wf_shape_of_layout() on the
// layout that they derive themselves, so that one rule gives both the same lookups.
// wf_shape_of_units() ends the process on a faulty list: for values read from a file, call
// wf_units_fault() or wf_layout_fault() first. Numbers stored in four digits get no range, and
// nothing here checks which numbers those are: see wf_units_of_layout().
#pragma once
#include <vector>

#include "frame/wf_frame.h"

namespace blk {

// The matrix's cells are signed bytes. From row 0 on, its rows form consecutive units; the rows
// after the last unit are single bytes. A unit of D digits (`digits`) and R rows (`rows`) stores
// R / D rows of numbers, digit by digit: digit q of the numbers' row i is the unit's row
// q * (R / D) + i. D is 2 (a weight: low + 256 * high) or 4.
struct WfUnit {
  u64 first_row = 0, rows = 0;
  unsigned digits = 0;
};

// The widths of the range proof's two lookups, and the radix of a two-digit number. A program whose
// own bounds rely on them should compare its constants with these in a static_assert: no run-time
// check would catch a difference.
constexpr unsigned WF_BYTE_BITS = 8;     // a cell is in -2^7 .. 2^7 - 1
constexpr unsigned WF_WEIGHT_BITS = 13;  // a number of a two-digit unit is in -2^12 .. 2^12 - 1
constexpr u64 WF_SECOND_DIGIT = u64(1) << WF_BYTE_BITS;  // the weight of a number's second digit

// Returns null or a description of the fault. For values read from a file. The first message is
// also returned for a matrix of 2^40 rows or more, which its text does not mention.
inline const char *wf_units_fault(u64 matrix_rows, u64 ranged_rows, const std::vector<WfUnit> &units) {
  if (units.empty() || ranged_rows > matrix_rows || ranged_rows % 2 || matrix_rows >> 40) return "no unit, or the ranged rows are an odd number or more than the matrix has";
  u64 end = 0;
  for (const WfUnit &u : units) {  // consecutive, so that no row is in two entries
    if ((u.digits != 2 && u.digits != 4) || !u.rows || u.rows % u.digits) return "a unit has not two or four digits, or not the same number of rows for each";
    if (u.first_row < end || u.rows > ranged_rows || u.first_row > ranged_rows - u.rows) return "the units do not follow one another within the ranged rows";
    if (u.first_row != end) return "rows between two units belong to no entry";
    end = u.first_row + u.rows;
  }
  return nullptr;
}

// Builds two lookups. `bytes`: every cell of rows 0 .. ranged_rows - 1 is in -128 .. 127.
// `weights`: every number of a two-digit unit is in -4096 .. 4095. Rows from ranged_rows on are in
// no lookup, and wf_fault() in blocks/iop/wf.h rejects lookups that leave a row of the matrix out.
// The matrix has 2^k_col columns.
// An entry is a pair of rows. `weights` has the entries of item 1; `bytes` has those of items 1, 2
// and 3, in this order:
//   1  for each two-digit unit, in list order: digits (0, 1) of each row of numbers;
//   2  for each four-digit unit: digits (0, 1) of each row of numbers, then digits (2, 3) of each;
//   3  the rows from the last unit's end to ranged_rows, in adjacent pairs.
// In `bytes` the two rows of an entry give separate lookup keys, and a part of the lookup (`WfPart`
// in blocks/iop/wf.h: consecutive entries that a prover can hold at once) has both rows of its
// entries: a prover that unpacks a weight gets both bytes at once.
// `WfLookup::mu` and `live`, which size the multiplicities, stay 0: wf_lines() in
// blocks/frame/wf_frame.h derives them from the parts. With 2^9 columns or more, the last sumcheck
// on each part's keys binds column bits 2, 1, 0, 5, 4, 3, 8, 7, 6 first (`first_bound`): a fast
// prover takes them three at a time.
inline WfShape wf_shape_of_units(unsigned k_col, u64 matrix_rows, u64 ranged_rows, const std::vector<WfUnit> &units) {
  const std::vector<unsigned> first_bound = k_col >= 9 ? std::vector<unsigned>{2, 1, 0, 5, 4, 3, 8, 7, 6} : std::vector<unsigned>{};
  const char *fault = wf_units_fault(matrix_rows, ranged_rows, units);
  BLK_REQUIRE(!fault, fault ? fault : "");  // for units from a file, call `wf_units_fault` first
  WfShape s;
  s.k_col = k_col, s.matrix_rows = matrix_rows;
  WfShapeLookup bytes, weights;
  bytes.lookup.name = "bytes", bytes.lookup.d = WF_BYTE_BITS;
  bytes.lookup.digit_weight = {F::one()}, bytes.lookup.offset = from_u64(u64(1) << (WF_BYTE_BITS - 1)), bytes.lookup.first_bound = first_bound, bytes.log_rows_apart = 1;
  weights.lookup.name = "weights", weights.lookup.d = WF_WEIGHT_BITS;
  weights.lookup.digit_weight = {F::one(), from_u64(WF_SECOND_DIGIT)}, weights.lookup.offset = from_u64(u64(1) << (WF_WEIGHT_BITS - 1)), weights.lookup.first_bound = first_bound;
  u64 end = 0;
  for (const WfUnit &u : units) {
    if (u.digits == 2) weights.entries.push_back(WfRun{0, u.first_row, u.rows / 2, u.rows / 2});
    end = u.first_row + u.rows;
  }
  bytes.entries = weights.entries;
  for (const WfUnit &u : units)
    for (u64 h = u.rows / 4, q = 0; u.digits == 4 && q < 2; q++) bytes.entries.push_back(WfRun{0, u.first_row + 2 * q * h, h, h});
  for (u64 y = end; y + 1 < ranged_rows; y += 2) bytes.entries.push_back(WfRun{0, y, 1, 1});
  s.lookups = {bytes, weights};
  return s;
}

// The three templates below are for a program with its own layout type: any type with m0 (the
// matrix's rows) and a list U of units with off (first row), rows and nlw (number of digits).
// All rows are ranged (ranged_rows = m0): the rows that a layout adds after the last unit to round
// m0 up get the `bytes` lookup.
// Not checked here: which units have four digits. A four-digit unit gets the `bytes` lookup and no
// statement on its numbers. The layout is taken as it is, because prover and verifier each derive
// it from the public files and the lookups are hashed into the first query (the hash input that
// fixes the parameters) of the key file's first hash chain (Fiat-Shamir transcript). The programs'
// layouts give four digits to the embedding only. The committer rejects any other four-digit unit,
// by way of hf::range_units() (prover/hash_commitment/hf_range_plan.h), and the verifier program
// cp_verify stops on one (verifier/src/vf_segments.inc, `internal-weight-limbs`: a verification call
// then ends NOT CHECKED). Any other program that relies on this should check its own layout.
template <class Layout>
std::vector<WfUnit> wf_units_of_layout(const Layout &Y) {
  std::vector<WfUnit> units;
  for (const auto &u : Y.U) units.push_back(WfUnit{(u64)u.off, (u64)u.rows, (unsigned)u.nlw});
  return units;
}
// Returns null or a description of the fault. Call it before wf_shape_of_layout().
template <class Layout>
const char *wf_layout_fault(const Layout &Y) {
  return wf_units_fault((u64)Y.m0, (u64)Y.m0, wf_units_of_layout(Y));
}
template <class Layout>
WfShape wf_shape_of_layout(const Layout &Y, unsigned k_col) {
  return wf_shape_of_units(k_col, (u64)Y.m0, (u64)Y.m0, wf_units_of_layout(Y));
}

}  // namespace blk
