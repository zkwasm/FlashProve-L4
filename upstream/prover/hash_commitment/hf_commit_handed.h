// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Argument types of once_make() (hf_commit_once.cuh), which builds the key file, and entry_make()
// (hf_entry_make.cuh), which builds a leaf file (`entry` in identifiers). The committer program
// cp_commit (hf_commit.cu) fills them. The key file contains cp_commit's one-time proofs about the
// anchor, a commitment's first weight table (rows of weights and masks, encoded, under a Merkle
// tree). A leaf file contains a proof (LINK) that ties a leaf table, a further table of the same
// weights with its own masks, to the anchor. This file is separate from hf_commit_once.cuh so that
// hf_range_windows.cuh can use `Committed`.

// Include this file after prover/weights_commitment/common.cuh,
// prover/weights_commitment/arguments.h, prover/hash_commitment/hf_shape.cuh and
// prover/hash_commitment/hf_commit_weights.cuh.
#pragma once
#include "hash_commitment/hf_commit_api.h"
#include "hash_commitment/hf_commit_walk.h"
#include "hash_commitment/hf_seed.h"
#include <map>
#include <string>
#include <vector>

namespace hf {
// Not in use: nothing reads or fills a MadeFiles.
typedef std::map<std::string, std::vector<u32>> MadeFiles;
// T, "the table", is the anchor for once_make() and the leaf table for entry_make(). O: the command
// line. PO: the parameter file's values. Each callee calls weights.release(), then sizes its GPU
// memory pools by what is free, so GPU memory that the caller still holds shrinks them.
// DW: the number of 32-bit words in a digest (prover/common/setting.h).
struct Committed {
  const rf::Opt &O;
  const rf::PolC &PO;
  const rf::HfShape &shape;
  PinnedWeights &weights;
  // The table's key kM, from which its masks, salts and pads are derived (hf_seed.h). Host memory,
  // valid for the whole call; locked only if mlock() succeeded (lock_in_memory(), hf_commit.cu).
  // With a null key no table is made; cp_commit never passes one.
  const hf_key32 *key;
  const uint8_t *key_again; // the key's second copy, 32 bytes, for hf_mask_own.cuh's self-checks
  HfKept kept;      // T's tree from the level kept.s up, in host memory
  const u32 *h0;    // not read. DW words: `h_0` of T's commitment record (`rf::HfRecord`)
  const u32 *h_par; // not read. DW words: `configuration_digest` of the configuration files
  MadeFiles *files; // see MadeFiles
  u64 *attempts;    // not read
};
// What entry_make() needs of the anchor.
struct LinkAnchor {
  const hf_key32 *key;      // the anchor's key; see Committed::key
  const uint8_t *key_again; // its second copy
  HfKept kept;              // the anchor's tree from the kept level up
  const u32 *h_card;        // not read. DW words: the digest of the anchor's commitment record
};
} // namespace hf
