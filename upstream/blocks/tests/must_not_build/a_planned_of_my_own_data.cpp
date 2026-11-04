// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context|no matching function». With -DCONTROL it builds.
// Calls the private constructor of `Planned` (a finished plan), which is for `Plan::finish()`.
#include "numbers/plan.h"
using namespace blk;
Planned play(const Planned &q) {
#ifndef CONTROL
  return Planned(&*q);
#endif
  return q;
}
