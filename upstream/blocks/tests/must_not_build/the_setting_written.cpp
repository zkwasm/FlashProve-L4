// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «read-only». With -DCONTROL it builds.
// Changes `set.target` (error target in bits), const in `Plan` and `Planned`:
// plans are checked against it.
#include "numbers/plan.h"
using namespace blk;
unsigned play(Plan &p, Planned &q) {
#ifndef CONTROL
  p.set.target = 40, q->set.target = 40;
#endif
  return p.set.target + q->set.target;
}
