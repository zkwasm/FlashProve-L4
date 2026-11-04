// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Turns an `UnprovedTable` (a committed table read without its proofs) into a `TestedTable`.
// as_if_tested() is private; only check_entry_if_tested() and check_proof_if_tested() call it.
#include "iop/foldeq.h"
using namespace blk;
void play(const UnprovedTable &unproved) {
#ifndef CONTROL
  const TestedTable taken = unproved.as_if_tested();
  (void)taken;
#endif
  (void)unproved.reader();
}
