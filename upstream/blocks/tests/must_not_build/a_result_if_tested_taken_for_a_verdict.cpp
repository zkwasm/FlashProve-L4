// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «cannot convert». With -DCONTROL it builds.
// Stores an `IfTested` (blocks/iop/foldeq.h), which has no unconditional yes, as a `ProofVerdict`.
#include "frame/check_entry.h"
using namespace blk;
bool play(const Configured &c, const UnprovedTable &first, Reader &file, const char *&why) {
#ifndef CONTROL
  const ProofVerdict verdict = check_entry_if_tested(c, first, file, why);
  return verdict != ProofVerdict::NO;
#else
  return check_entry_if_tested(c, first, file, why) != IfTested::NO;
#endif
}
