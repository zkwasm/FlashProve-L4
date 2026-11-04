// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes an honest proof for each toy-size protocol in whole_runs.h and verifies it in memory.
// 70 is prove()'s `key`, from which it derives witness and randomness.
#include <cstdio>
#include <vector>

#include "tests/whole_runs.h"

using namespace blk;

int main() {
  const RunChoice choice = small_choice();
  size_t bytes = 0;
  for (size_t which = 0; which < WHOLE_RUNS; which++) {
    const std::unique_ptr<WholeRun> run = whole_run(which, choice);
    const std::vector<u8> proof = run->prove(70);
    const char *why = nullptr;
    if (!run->verify(proof, &why)) {
      std::printf("FAILED: %s: the honest proof is refused: %s\n", run->name(), why);
      return 1;
    }
    bytes += proof.size();
  }
  std::printf("proof file: ok: field %d, hash %d: %zu bytes in %zu proofs accepted\n", BLK_FIELD, BLK_HASH, bytes, WHOLE_RUNS);
  return 0;
}
