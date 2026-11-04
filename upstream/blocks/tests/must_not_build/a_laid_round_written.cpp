// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «read-only». With -DCONTROL it builds.
// Changes a round after Plan::finish() has checked the plan (`Planned`).
#include "numbers/plan.h"
using namespace blk;
u64 play(Planned &q) {
#ifndef CONTROL
  q->rounds_[0].draw_positions = 1;
#endif
  return q->rounds_[0].draw_positions;
}
