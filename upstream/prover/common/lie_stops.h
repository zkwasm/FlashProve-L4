// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A hook for a malicious prover built for tests: where it returns true, its caller ignores a failed
// self-check (`stop` in identifiers), so that the verifier, not the prover, decides on a proof
// about weights that were not committed. The only caller is the check "own-node" in show_columns()
// (prover/hash_commitment/hf_tree.cuh), which is compiled into prover/hash_commitment/hf_keeper.cu,
// the translation unit that owns the weight table (all weights, encoded, under a Merkle tree). That
// unit does not include prover/common/lie.cuh, which has the test hooks for the values that the
// prover falsifies.
// In this repository passes_in_another_unit() is constant false in every build, so a failed
// self-check always counts. The test build that gives it another body is not part of this
// repository. That body does not check that the run is a test: it relies on earlier checks in the
// prover program `cp_prove`. One is in this repository: with the variable LIE set, read_inputs()
// (prover/prover/inputs.cuh) requires the test option -test-coins. The other is that test build's
// own: it stops in read_isk() (prover/prover/isk.cuh), before it opens a model file, unless
// `isk.txt` (in the secret key directory) specifies synthetic weights and has the line "seed test".
#pragma once
#include <cstdio>

#include "common/setting.h"

namespace lie {
static constexpr bool passes_in_another_unit(const char *) {
  return false;
}
} // namespace lie
