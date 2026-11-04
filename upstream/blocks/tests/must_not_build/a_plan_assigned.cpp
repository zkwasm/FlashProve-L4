// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «use of deleted function». With -DCONTROL it builds.
// Assignment would replace `Plan::set`, the const `Setting` that planning reads.
#include "numbers/plan.h"
using namespace blk;
void play(Plan &p, const Plan &other) {
  Plan copy(other);
#ifndef CONTROL
  p = other;
#endif
  (void)p, (void)copy;
}
