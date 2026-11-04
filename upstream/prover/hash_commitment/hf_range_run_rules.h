// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Index rules, second half, for host and GPU code: whether a chunk contains the weight at a given
// row and column of the committed weight matrix W, and at which position. A pass over the weights
// delivers tensors in chunks (`Piece` in hf_commit_walk.h). The first half, hf_range_fold_rules.h,
// defines units (rows of W for one or two tensors), weight rows, index groups, classes and sources.
//
// A weight's position in the chunk's source, in elements, is run_row().at + run_half().at for each
// half of its column c (RunShape): it is linear in each group's field, bits cb0 .. cb0 + take - 1
// of c. The only caller, hf_range_windows.cuh (lookup keys of the committer's range proof, WF),
// uses run_shape(lk, lk) and the low half only: a run is a whole row; no field is in both halves.
#pragma once
#include "hash_commitment/hf_range_fold_rules.h"
namespace hfw {
// A run: R neighbouring column positions. Column c has a high and a low half: c = c_hi * R + c_lo.
struct RunShape {
  int lk = 0, s = 0; // W has k = 2^lk columns; a run has R = 2^s column positions
  u64 width = 0;     // min(R, k): W has a column only at c_lo < width
};
inline RunShape run_shape(int lk, int s) {
  RunShape S;
  S.lk = lk, S.s = s;
  S.width = (u64)1 << (s < lk ? s : lk);
  return S;
}

// ---- a weight row ---------------------------------------------------------------------------------------------------
// run_row()'s result for weight row r < u.weight_rows (not checked) of u, a unit or a part of one.
// `at`: u.bias + what r adds to its weights' positions. `in`: false if u has none of r's weights.
struct RunRow {
  i64 at = 0;
  bool in = false;
};
HD RunRow run_row(const FoldUnit &u, u64 r) {
  RunRow o;
  o.at = u.bias;
  for (int g = 0; g < u.groups; g++) {
    const u64 lf = r % u.g[g].left;
    r /= u.g[g].left;
    if (g == u.pg && (lf < u.row_from || lf >= u.row_to)) return o;
    if (u.g[g].cls == 'A' && !u.g[g].take && lf >= u.g[g].extent) return o;
    o.at += u.g[g].step * (i64)fold_row_bits(u.g[g], lf);
  }
  o.in = true;
  return o;
}

// ---- a column, in two halves ----------------------------------------------------------------------------------------
// The unit or part u has weights only in columns with from <= group i's field < to.
HD void run_field_range(const FoldUnit &u, int i, u64 &from, u64 &to) {
  const FoldGroup &g = u.g[i];
  from = 0, to = (u64)1 << g.take;
  if (g.cls == 'A' && g.extent < to) to = g.extent;
  if (i == u.pg) {
    if (u.field_from > from) from = u.field_from;
    if (u.field_from + u.field_count < to) to = u.field_from + u.field_count;
  }
}
// How many bits of group g's field are below bit s of the column number.
HD int run_low_bits(const FoldGroup &g, int s) {
  const int below = s - g.cb0;
  return below < 0 ? 0 : below > g.take ? g.take : below;
}
// run_half()'s result for the columns with c_lo = v < R (not checked), or with c_hi = v if `high`.
// `in` is false if u has no weight in them. Limit: `in` ignores a field that straddles bit s.
struct RunHalf {
  i64 at = 0;   // what the half adds to a weight's position
  u64 part = 0; // the half's bits of a field that straddles bit s; nothing reads or checks them
  bool in = false;
};
HD RunHalf run_half(const RunShape &S, const FoldUnit &u, u64 v, bool high) {
  RunHalf o;
  const u64 c = high ? v << S.s : v; // the other half's bits are 0
  if (c >> u.ncb) return o;          // the unit has no cell from column 2^ncb on
  for (int i = 0; i < u.groups; i++) {
    const FoldGroup &g = u.g[i];
    if (!g.take) continue;
    const int low = run_low_bits(g, S.s);
    if (high ? low == g.take : low == 0) continue; // the field is in the other half
    const u64 field = (c >> g.cb0) & (((u64)1 << g.take) - 1);
    o.at += g.step * (i64)(g.cls == 'P' ? field << (g.z - g.take) : field);
    if (high ? low != 0 : low != g.take) {
      o.part = field;
      continue;
    }
    u64 from, to;
    run_field_range(u, i, from, to);
    if (field < from || field >= to) return o;
  }
  o.in = true;
  return o;
}

// ---- the launches of a chunk ----------------------------------------------------------------------------------------
struct RunLaunch {
  FoldUnit u;
  bool wide = false;
};
// Appends to `out` the kernel launches that read one chunk, each with its part of the unit `whole`.
// The chunk has the weights with index first .. first + count - 1 in group pg (pg = -1: all), and
// `src` starts at index `first`; its first `planes` pointers are read. RunLaunch::wide: a thread
// may load `wide` neighbouring columns at once. Returns false if group pg has class P.
inline bool run_launches(const RunShape &S, const FoldUnit &whole, int pg, u64 first, u64 count, const FoldSource &src,
                         int planes, int cell_bytes, int wide, std::vector<RunLaunch> &out) {
  std::vector<std::pair<u64, u64>> ranges = {{first, count}};
  if (pg >= 0) ranges = fold_ranges(whole.g[pg], first, count);
  for (auto &range : ranges) {
    RunLaunch x;
    x.u = whole;
    if (pg >= 0 && !fold_restrict(x.u, pg, range.first, range.second, first)) return false;
    x.wide = S.width % (u64)wide == 0 && fold_wide(x.u, src, planes, cell_bytes, wide);
    out.push_back(x);
  }
  return true;
}
} // namespace hfw
