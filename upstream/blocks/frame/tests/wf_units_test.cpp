// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/frame/wf_units.h, which derives from a matrix's units what WF, the committer's range
// proof by lookups, looks up (`WfShape`). A unit {first_row, rows, digits} stores numbers of 2 or 4
// digits: the rows of digit 0, then those of digit 1, and so on. An entry is a pair of rows.
//   lookups[0], "bytes":   each cell of the first ranged_rows rows (here all) gives one lookup key;
//   lookups[1], "weights": each number of a 2-digit unit gives one, from its two digits.
// Main checks: which rows the entries pair up; the parts from wf_cut() reach each such row once;
// wf_units_fault() reports a fault for each invalid list in main(). Not checked: how "bytes" pairs
// rows outside 2-digit units; WfLookup::first_bound; a valid list with ranged_rows < matrix_rows.
// `mu` and `live` (how many bits a multiplicity has) stay 0: wf_lines() sets them, which is tested
// in blocks/frame/tests/wf_frame_test.cpp. Last line "wf_units_test: ok", exit code 0; otherwise 1.
#include <cstdio>
#include <map>

#include "frame/wf_units.h"
using namespace blk;

static int failures = 0;
#define CHECK(x, what) \
  do {                 \
    if (!(x)) std::printf("FAILED: %s\n", what), failures++; \
  } while (0)

// Entry i of a `WfRun` (blocks/iop/wf.h) is rows first_row + i and first_row + i + stride.
// "weights" must have one run per 2-digit unit, in list order; "bytes" begins with the same runs.
static void covered_once(const std::vector<WfUnit> &units, u64 rows, unsigned k_col) {
  CHECK(!wf_units_fault(rows, rows, units), "an honest list of units is refused");
  for (unsigned part_log : {1u, 2u, 3u, 5u, 9u}) {
    const WfShape s = wf_shape_of_units(k_col, rows, rows, units);
    std::map<u64, int> seen;
    for (const WfRun &r : s.lookups[0].entries)
      for (u64 i = 0; i < r.places; i++) seen[r.first_row + i]++, seen[r.first_row + i + r.stride]++;
    bool once = seen.size() == rows;
    for (const auto &x : seen) once = once && x.first < rows && x.second == 1;
    CHECK(once, "a ranged row stands in no entry of the bytes' look-up, or in two");
    u64 pairs = 0;
    size_t at = 0;
    for (const WfUnit &u : units)
      if (u.digits == 2) {
        const WfRun &r = s.lookups[1].entries[at], &b = s.lookups[0].entries[at];
        CHECK(r.first_row == u.first_row && r.places == u.rows / 2 && r.stride == u.rows / 2, "a unit of two digits is not one run of the weights' entries");
        CHECK(b.first_row == r.first_row && b.places == r.places && b.stride == r.stride, "the bytes' entries do not begin with the weights'");
        pairs += r.places, at++;
      }
    CHECK(at == s.lookups[1].entries.size(), "the weights' look-up has entries of other units");
    // wf_cut() cuts the entries into parts of 2^part_log places; a place of "bytes" reads one row.
    std::map<u64, int> cut;
    const std::vector<WfPart> parts = wf_cut(s.lookups[0].entries, part_log, s.lookups[0].log_rows_apart);
    for (const WfPart &p : parts)
      for (const WfRun &r : p.runs)
        for (u64 i = 0; i < r.places; i++) cut[r.first_row + i]++;
    once = cut.size() == rows;
    for (const auto &x : cut) once = once && x.second == 1;
    CHECK(once, "behind the cut a row is reached by no place of the bytes' parts, or by two");
    for (const WfShapeLookup &l : s.lookups) CHECK(l.lookup.live == 0 && l.lookup.mu == 0, "the shape names planes: they follow from the parts (wf_lines)");
    {  // `wf_fault`'s check that no row is left out: on these parts, then without the last row
      WfPublic x;
      x.lookups = {s.lookups[0].lookup};
      x.lookups[0].parts = parts;
      CHECK(wf_covers_all_rows(x, 0, WfMatrix{k_col, rows}), "wf_covers_all_rows() says no to the bytes' parts");
      u64 last = 0;  // the run that reaches the matrix's last row
      WfRun *at_the_end = nullptr;
      for (WfPart &p : x.lookups[0].parts)
        for (WfRun &r : p.runs)
          if (r.first_row + r.places > last) last = r.first_row + r.places, at_the_end = &r;
      at_the_end->places--;
      CHECK(!wf_covers_all_rows(x, 0, WfMatrix{k_col, rows}), "wf_covers_all_rows() says yes though the last row is left out");
    }
    // "weights", cut as in wf_lines(): a place reads two rows, so a part has half as many places.
    if (part_log >= 1) {
      std::map<u64, u64> second_of;  // digit 0's row -> digit 1's row, as the units say
      for (const WfUnit &u : units)
        for (u64 i = 0; u.digits == 2 && i < u.rows / 2; i++) second_of[u.first_row + i] = u.first_row + u.rows / 2 + i;
      std::map<u64, int> met;
      bool right = true;
      for (const WfPart &p : wf_cut(s.lookups[1].entries, part_log - 1, s.lookups[1].log_rows_apart)) {
        right = right && p.log_places == part_log - 1;
        for (const WfRun &r : p.runs)
          for (u64 i = 0; i < r.places; i++) {
            const auto pair = second_of.find(r.first_row + i);
            right = right && pair != second_of.end() && pair->second == r.first_row + i + r.stride && r.first_place + i < (u64(1) << p.log_places);
            met[r.first_row + i]++;
          }
      }
      right = right && met.size() == second_of.size() && met.size() == pairs;
      for (const auto &x : met) right = right && x.second == 1;
      CHECK(right, "behind the cut the weights' parts do not hold each number of a unit of two digits once, as its two digits' rows");
    }
    // Literals, to catch a changed constant in blocks/frame/wf_units.h.
    const WfLookup &b = s.lookups[0].lookup, &w = s.lookups[1].lookup;
    CHECK(b.d == 8 && b.offset == from_u64(128) && b.digit_weight == std::vector<F>{F::one()} && b.padding_key == F::zero(), "the bytes' table, offset, digit weight or padding key");
    CHECK(w.d == 13 && w.offset == from_u64(4096) && w.digit_weight == (std::vector<F>{F::one(), from_u64(256)}) && w.padding_key == F::zero(), "the weights' table, offset, digit weights or padding key");
    CHECK(s.lookups[0].log_rows_apart == 1 && s.lookups[1].log_rows_apart == 0, "which look-up takes an entry's rows as keys of their own");
  }
}

int main() {
  covered_once({{0, 12, 2}, {12, 6, 2}}, 22, 4);  // the toy matrix of blocks/iop/tests/wf_world.h
  covered_once({{0, 96, 2}, {96, 40, 2}}, 144, 11);
  covered_once({{0, 8, 2}, {8, 16, 4}, {24, 2, 2}, {26, 4, 4}}, 36, 3);   // units of four digits between the others
  covered_once({{0, 4, 4}}, 4, 2);                                        // no weight at all
  CHECK(wf_units_fault(10, 10, {}), "no unit");
  CHECK(wf_units_fault(10, 9, {{0, 4, 2}}), "an odd number of ranged rows");
  CHECK(wf_units_fault(10, 12, {{0, 4, 2}}), "more ranged rows than the matrix has");
  CHECK(wf_units_fault(10, 10, {{0, 4, 3}}), "three digits");
  CHECK(wf_units_fault(10, 10, {{0, 6, 4}}), "rows that are no multiple of the digits");
  CHECK(wf_units_fault(10, 10, {{0, 4, 2}, {2, 4, 2}}), "units that overlap");
  CHECK(wf_units_fault(10, 10, {{0, 4, 2}, {6, 4, 2}}), "a gap between two units");
  CHECK(wf_units_fault(10, 10, {{0, 12, 2}}), "a unit beyond the ranged rows");
  CHECK(wf_units_fault(10, 10, {{2, 4, 2}}), "rows before the first unit");
  {  // any layout type with m0 (matrix rows), `U` (units), `off` (first row), `rows`, nlw (digits)
    struct Unit {
      size_t off, rows;
      int nlw;
    };
    struct Layout {
      size_t m0;
      std::vector<Unit> U;
    };
    const Layout Y{22, {{0, 12, 2}, {12, 6, 2}}}, bad{22, {{0, 12, 3}}};
    CHECK(!wf_layout_fault(Y) && wf_layout_fault(bad), "a layout's fault");
    const WfShape a = wf_shape_of_layout(Y, 4), b = wf_shape_of_units(4, 22, 22, {{0, 12, 2}, {12, 6, 2}});
    bool same = a.k_col == b.k_col && a.matrix_rows == b.matrix_rows && a.lookups.size() == b.lookups.size();
    for (size_t i = 0; same && i < a.lookups.size(); i++) {
      same = a.lookups[i].entries.size() == b.lookups[i].entries.size() && a.lookups[i].lookup.live == b.lookups[i].lookup.live;
      for (size_t r = 0; same && r < a.lookups[i].entries.size(); r++)
        same = a.lookups[i].entries[r].first_row == b.lookups[i].entries[r].first_row && a.lookups[i].entries[r].places == b.lookups[i].entries[r].places;
    }
    CHECK(same, "the shape of a layout is not the shape of its units");
  }
  std::printf(failures ? "wf_units_test: %d FAILED\n" : "wf_units_test: ok\n", failures);
  return failures != 0;
}
