// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Marks a caller's `IndexKey` as tested without calling check_key_under() or check_entry_under(),
// through the private constructor of `TestedTable`.
#include "iop/foldeq.h"
using namespace blk;
void play(const IndexKey &by_hand, const Setting &set) {
  TestedTable none_yet;
#ifndef CONTROL
  TestedTable made(by_hand, set, std::vector<u8>());
  (void)made;
#endif
  (void)none_yet, (void)by_hand, (void)set;
}
