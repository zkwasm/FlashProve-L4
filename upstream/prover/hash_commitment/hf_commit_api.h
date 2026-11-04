// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// What the committer program cp_commit stores of each weight table's Merkle tree: the nodes of one
// level s (`kept` in identifiers means stored). A weight table contains the weights as long rows,
// each encoded together with masks (random coefficients). It is never stored. For each opened
// column, an opening recomputes the 2^s columns under the stored node above it, hashes them up to
// level s, compares the result with that node, and takes the rest of the Merkle path from level s
// upward (show_columns(), hf_tree.cuh).
//
// Levels count from the leaves: tree leaf I is node(0, I), and the root is node(ln, 0). The table
// has 2^ln columns (HfParams::ln). Each table has a file hcnodes.bin, in isk/anchor/ (a
// commitment's first weight table) or isk/leaf/<k>/ (a further one, not a tree leaf); isk is the
// secret key directory. The file contains the 2^(ln - s) digests node(s, y), in order of y, 32
// bytes each (DW = 8 words), without a header. On loading, kept_rebuild() (hf_tree.cuh) recomputes
// the levels above s. An unreadable file, another length, or a root other than the table's gives
// the error [hc-nodes-root].
#pragma once
#include "hash_commitment/hf_params.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace hf {
// The stored nodes in memory: the tree from level s up, as a heap of 2^(ln - s + 1) digests, which
// the struct does not own. Digest 0 is all zero, and digest 2^(ln - l) + y is node(l, y). The
// heap's second half has the same bytes as hcnodes.bin. node() does not check its arguments. `DW`
// is defined in prover/common/setting.h.
//
// cp_commit fills the heap part by part; a part is the 2^lp columns under one node of level lp
// (phase A: make_table(), prover/hash_commitment/hf_commit_table.cuh).
//
// s is cp_commit's option -keep. It is recorded in isk.txt (line `keep`) and is no part of what a
// verifier sees. The weight matrix has 2^lk columns and the table 2^b rows (HfParams::lk,
// HfParams::b): table row r contains the matrix columns c with c >> lcl = r, where lcl = lk - b.
// With this layout one vector-matrix product with the weights gives all 2^lcl table columns under a
// node of level lcl, except for the masks' share. So the default s is the lower of lcl and
// HfParams::keep_level_most(); cp_commit rejects an s above the latter (`hc-keep-level`). An
// opening's Merkle paths must end at level s or above: show_columns() exits otherwise. In the
// parameter file policy.txt, b, ln and lk are the first, second and sixth numbers of the line
// `table`. At full size (prover/examples/target_full_f2: 11, 32 and 18, so lcl = 7), level 7 has
// 2^(32 - 7) = 2^25 digests of 32 bytes: 2^30 bytes (1.1 GB) in the file and 2^31 bytes (2.1 GB) as
// a heap. These sizes are computed from that line, not measured.
struct HfKept {
  int s = 0;                      // the lowest kept level
  const uint32_t *heap = nullptr; // host (pinned in the serving process)
  size_t digests(const HfParams &P) const { return (size_t)2 << (P.ln - s); }
  size_t bytes(const HfParams &P) const { return digests(P) * DW * 4; }
  const uint32_t *node(const HfParams &P, int level, uint64_t y) const { // s <= level <= ln, y < 2^(ln - level)
    return heap + ((((uint64_t)1 << (P.ln - level)) + y) * DW);
  }
  const uint32_t *root() const { return heap + DW; }
};
} // namespace hf
