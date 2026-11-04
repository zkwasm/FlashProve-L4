// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «invalid initialization of reference of type». With -DCONTROL it builds.
// The anchor (first table) passed to check_entry() is an untested `IndexKey`, not a `TestedTable`.
#include "frame/check_entry.h"
using namespace blk;
ProofVerdict play(const Configured &c, const IndexKey &untested, const TestedTable &tested, Reader &file, TestedTable &entry, const char *&why) {
#ifndef CONTROL
  return check_entry(c, untested, file, entry, why);
#else
  (void)untested;
  return check_entry(c, tested, file, entry, why);
#endif
}
