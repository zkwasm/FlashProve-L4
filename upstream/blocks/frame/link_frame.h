// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Derives, from the block LINK's free parameters (`LinkLines`), what a run needs: shapes, codes,
   numbers of columns and positions, stages of the proximity test (`iopp` in identifiers) and the
   hash chain's `Plan`. LINK (blocks/iop/link.h) is to show that a leaf table (`further`) contains
   the same entries as the anchor (`first`), the first committed table (`Index` in identifiers),
   except pads and random tails. Its one group (oracle messages tested together) contains the two
   folded rows: each table's rows, combined at one random point.

   The numbers of columns follow from the tables' distances (`TableSize`). The anchor's number comes
   from the bound of a batch's test, which presumes that the anchor passed its one-time test (the
   committer's test on it) at these distances: not checked here. The leaf table's number comes from
   the one-time test's distance, because nothing has been shown about that table before.

   No other test opens the group, so `solve_group` (blocks/frame/derive.h) sizes its rows' random
   tails for one list of positions. */
#pragma once
#include <string>
#include <vector>

#include "frame/derive.h"
#include "frame/lin_frame.h"
#include "iop/link.h"

namespace blk {

struct LinkLines {
  // Both tables: 2^table_rows_log rows; code length 2^table_log_n; a row's message entries (pads
  // included) and random coordinates (its random tail). rows_high, masks_ahead, masks_ride: copied
  // into LinkShape.
  unsigned table_rows_log = 0, table_log_n = 0;
  u64 table_data = 0, table_random = 0;
  GroupLine folded;  // the group of the two folded rows
  bool rows_high = true, masks_ahead = true, masks_ride = true;
};

// The committer proves LINK, so its group uses the parameter file's lines `once-folded`, like the
// one-time test's folded row.
inline LinkLines link_lines(const Parameters &p) {
  LinkLines L;
  L.table_rows_log = p.table_rows_log, L.table_log_n = p.table_log_n, L.table_data = p.table_data, L.table_random = p.table_random;
  L.folded = p.once_folded;
  return L;
}

struct LinkSized {
  LinkLines lines;
  LinkShape S;
  TableSize table;
  TestSizes linked;
  std::string refused;  // not empty: no admissible numbers follow
};

// The tables, with distances given as radii in positions: radius, of the one-time test;
// later_radius, of a later test. Only tests call this. Radii are not checked here;
// link_sized() applies table_admitted().
inline TableSize link_table(const Setting &s, const LinkLines &L, u64 radius, u64 later_radius) {
  TableSize T;
  T.code.name = "table", T.code.log_n = L.table_log_n, T.code.ell = L.table_data, T.code.r = L.table_random;
  T.v = L.table_rows_log, T.answers = s.samples;
  T.distances.radius = radius, T.distances.later_radius = later_radius;
  return T;
}

// table: the tables that L describes, with their distances; for a parameter file, derive_table()
// makes it. codes: the chain's list; LINK's codes are added if missing.
inline LinkSized link_sized(const Setting &s, const LinkLines &L, const TableSize &table, Codes &codes) {
  LinkSized x;
  x.lines = L, x.table = table;
  std::string &why = x.refused;
  LinkShape &S = x.S;

  // ---- the tables and their columns ----
  const TableSize &T = x.table;
  if (T.code.log_n != L.table_log_n || T.code.ell != L.table_data || T.code.r != L.table_random || T.v != L.table_rows_log)
    return why = "table: the distances belong to another table", x;
  if (T.code.dim(s) > T.code.n()) return why = "table: data and random coordinates exceed the code's length", x;
  const TableAnalysis A = analysis(s, T);
  if (!table_admitted(T, A)) return why = "table: the distances are not admitted", x;
  S.columns_first = scope_positions(s, Scope::ONCE, A.later.agree);
  S.columns_further = scope_positions(s, Scope::ONCE, A.once.agree);
  if (!S.columns_first || !S.columns_further) return why = "table: no number of columns reaches the target", x;

  // ---- shapes ----
  IndexShape I;
  I.code = Code{L.table_log_n, (size_t)L.table_data, (size_t)L.table_random}, I.v = L.table_rows_log, I.answers = s.samples, I.salt_bytes = s.salt_bytes, I.distances = T.distances;
  if (index_fault(I)) return why = std::string("table: ") + index_fault(I), x;
  S.table = I;
  S.rows_high = L.rows_high;
  S.masks_ahead = L.masks_ahead, S.choice.masks_ride = L.masks_ride;
  S.rows = lin_rows_of(S.first(s.samples).entries() + s.samples, u64(1) << L.folded.log_ell, L.rows_high);
  const IoppStage group = S.test()[0];  // its numbers of row variables; the codes come below
  x.linked.group = shape_of("linked", L.folded, group.k, true);
  x.linked.group.h = group.h;

  // ---- positions and random coordinates; stages; codes ----
  // batched_rows: for each table, one linear claim per record answer and one per drawn column; the
  // equality of the two folded rows; one per answer to their samples.
  GroupLoad load;
  load.cube_bits = stage_cube_bits(x.linked.group), load.messages = 2;
  load.batched_rows = 2 * (u64)I.answers + S.columns_first + S.columns_further + 1 + 2 * (u64)s.samples;
  if (!solve_group(s, Scope::ONCE, load, [](u64 t) { return t; }, x.linked, why)) return x;
  if (!solve_test(s, Scope::ONCE, L.folded, x.linked, why)) return x;
  const std::vector<IoppStage> test = iopp_stages(x.linked.stages, codes);
  BLK_REQUIRE(test.size() >= 2 && test[0].k == group.k && test[0].h == group.h, "link_frame: the solved first stage is not the group");
  S.code = test[0].code, S.mask_code = test[0].mask_code;
  S.later.assign(test.begin() + 1, test.end());
  if (const char *f = link_fault(codes, S, s.samples)) why = std::string("LINK: ") + f;
  return x;
}

// Adds LINK's rounds to p, up to the proximity test's end. table_tree: sizes of the tree over one
// table's columns. Not checked: x.refused is empty. p's setting and `codes` must be link_sized()'s,
// and p's scope Scope::ONCE: otherwise a BLK_REQUIRE may end the process.
inline void link_plan(Plan &p, const Codes &codes, const LinkSized &x, const TreeSize &table_tree) {
  const LinkShape &S = x.S;
  u64 first = 0, further = 0;
  plan_link(p, "link", x.table, SampleSize{Send::cube(codes, S.first(p.set.samples).u).bits, {x.linked.stages[0].g}}, table_tree, first, further);
  if (!p.refused()) BLK_REQUIRE(first == S.columns_first && further == S.columns_further, "link_frame: the plan draws other numbers of columns");
  {  // `plan_iopp` computes errors from x's solved stages: they must match S's under `codes`
    const char *fault = iopp_sized_fault(codes, S.test(), x.linked.stages);
    BLK_REQUIRE(!fault, fault ? fault : "");
  }
  plan_iopp(p, "the two folded rows", x.linked.stages, tree_sizes(codes, link_message(S, p.set.samples)),
            2 * (u64)S.table.answers + first + further + 1 + 2 * (u64)p.set.samples, S.masks_ahead ? 0 : 1, S.choice.masks_ride);
}

}  // namespace blk
