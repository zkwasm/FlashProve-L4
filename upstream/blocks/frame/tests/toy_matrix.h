// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A matrix within the ranges that toy_ranges() (blocks/frame/tests/toy_file.h) sets for the
// committer's range proof (WF). For tests calling make_key() (blocks/frame/check_key.h).
#pragma once
#include "iop/tests/wf_world.h"

namespace blk {

// Every cell is in -4 .. 3; in the row pairs of a two-digit lookup, low + 8 * high is in
// -16 .. 15. The ranges are hard-coded; x only locates the pairs. `key` seeds the test data.
inline LinMatrix toy_matrix_in_range(const LinShape &S, const WfPublic &x, u64 key) {
  LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
  M.a.resize(M.rows * M.columns);
  TestCoins data(key);
  for (F &cell : M.a) cell = wf_signed((long long)(data.word() % 8) - 4);
  for (const WfLookup &l : x.lookups)
    if (l.digit_weight.size() == 2)
      for (const WfPart &p : l.parts)
        for (const WfRun &r : p.runs)
          for (u64 i = 0; i < r.places; i++)
            for (u64 col = 0; col < M.columns; col++) {
              const long long weight = (long long)(data.word() % 32) - 16, low = ((weight + 4) % 8 + 8) % 8 - 4;
              M.a[(r.first_row + i) * M.columns + col] = wf_signed(low), M.a[(r.first_row + i + r.stride) * M.columns + col] = wf_signed((weight - low) / 8);
            }
  return M;
}

}  // namespace blk
