// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «cannot convert». With -DCONTROL it builds.
// As a bool, `IfTested` (blocks/iop/foldeq.h) would lose its condition: that the table is tested.
#include "frame/check_entry.h"
using namespace blk;
bool play(const Configured &c, const UnprovedTable &first, Reader &file, const char *&why) {
#ifndef CONTROL
  const bool yes = check_entry_if_tested(c, first, file, why);
  return yes;
#else
  return check_entry_if_tested(c, first, file, why) != IfTested::NO;
#endif
}
