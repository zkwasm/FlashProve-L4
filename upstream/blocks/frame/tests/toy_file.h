// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A toy-size parameter file (lines defined in blocks/frame/parameters.h) with the target of 128
// bits, `sec 128`, and the lookups of the committer's range proof (WF), for tests on whole proof,
// key and leaf files.
#pragma once
#include <string>

#include "frame/version.h"
#include "frame/wf_frame.h"

namespace blk {

inline std::string toy_file() {
  std::string t = "version " + version_text(SERVED_VERSION);
  t += "\nsec 128\nreading 1\nrowcost 1\nslots 2\nanswers 1\nrecord 1\n"
       "work field 0 0 0\nwork sample 0 0 0\nwork closeness 0 0 0\nwork positions 4 6 8\nwork smallness 0 0 0\nsalt 40\nlisted 0\n"
       "group link 1 13 9 13 5\nstages link 2 1 13 13 0 0 15 15\n";
  /* Beside folded rows, a single mask row's code must have at most the main code's rate
     (`shape_fault`, blocks/numbers/blocks.h). The mask rows of `own` and `once-own` hold only 2^3
     entries, so that an index-mask message (masks for the pending claims on the committed table)
     takes more than one: by `sized` (blocks/frame/files.h), with `toy_model`, 2 in a request and 3
     in the key file. BLK_FIELD is always 2 (blocks/base.h stops any other build): the values for
     BLK_FIELD == 1 are never used. */
  const std::string n = BLK_FIELD == 1 ? "14" : "13", mask = n + " 4", own_mask = BLK_FIELD == 1 ? "13 5" : "12 3", row = BLK_FIELD == 1 ? " 8 " : " 7 ";
  for (const char *once : {"", "once-"}) {
    t += std::string("group ") + once + "folded 1 " + n + " 9 " + mask + "\nstages " + once + "folded 2 1 13 13 0 0 15 15\n";
    t += std::string("group ") + once + "omega 1 " + n + row + mask + "\nstages " + once + "omega 2 1 13 13 0 0 15 15\n";
    t += std::string("group ") + once + "own 1 " + n + row + own_mask + "\nstages " + once + "own 2 1 13 13 0 0 15 15\n";
  }
  t += "group once-ranges 1 14 9 14 8\nstages once-ranges 2 1 14 14 0 0 15 15\nranges-part 5\n";
  return t + (BLK_FIELD == 1 ? "table 3 15 2403 1001 0 6\n" : "table 3 14 803 1001 0 6\n");
}

/* matrix_rows must be even (not checked here): rows i (low) and i + matrix_rows / 2 (high) form
   entry i. Lookup `bytes`: every cell is in -4 .. 3 (three bits). Lookup `weights`: in the first
   fifth of the entries, low + 8 * high (two digits) is in -16 .. 15.
   `mu`, `live` (a multiplicity's bits) stay 0: `wf_lines` (blocks/frame/wf_frame.h) derives them
   from the lookup's parts. */
inline WfShape toy_ranges(u64 matrix_rows) {
  WfShape s;
  s.matrix_rows = matrix_rows;
  WfShapeLookup bytes, weights;
  bytes.lookup.name = "bytes", bytes.lookup.d = 3, bytes.lookup.digit_weight = {F::one()}, bytes.lookup.offset = from_u64(4);
  bytes.lookup.padding_key = from_u64(3), bytes.lookup.first_bound = {1, 0}, bytes.log_rows_apart = 1;
  bytes.entries = {WfRun{0, 0, matrix_rows / 2, matrix_rows / 2}};
  weights.lookup.name = "weights", weights.lookup.d = 5, weights.lookup.digit_weight = {F::one(), from_u64(8)}, weights.lookup.offset = from_u64(16);
  weights.lookup.padding_key = from_u64(17);
  weights.entries = {WfRun{0, 0, matrix_rows / 10, matrix_rows / 2}};
  s.lookups = {bytes, weights};
  return s;
}

}  // namespace blk
