// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tail: the matrices (TMat) of one sub-proof's approximate range check (`tail` in identifiers).
#pragma once
#include "vf_common.h"
// A matrix: m cell rows, k columns. wp, its symbol tree's w, is 0: no parameter-file line sets it.
struct TMat {
  string name;
  u64 m = 0, k = 0;
  // 0: this sub-proof commits to it; 1: an earlier one. shapeF() rejects others (2: weight matrix).
  int kind = 0;
  // For `kind` 1 only. `foreign` indexes ft, shapeF()'s list (vf_smallness_test_members.inc) of earlier sub-proofs'
  // symbol trees. ttOff is the matrix's first column in that tree; shapeF() rejects a nonzero one.
  int foreign = -1;
  u64 ttOff = 0;
  // aId names the matrix: 256 gid + mu (gid: the owning sub-proof's number; mu: its index there).
  u64 aId = 0;
  u32 wp = 0;
};
struct Tail {
  vector<TMat> mats;
  u64 foldColumns() const {
    u64 s = 0;
    for (auto &M : mats) s += M.k;
    return s;
  }
};
