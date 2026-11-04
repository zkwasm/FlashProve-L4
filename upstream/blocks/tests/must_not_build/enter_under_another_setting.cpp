// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Copies rounds with error bounds computed for one `Setting` (blocks/numbers/setting.h) into a
// `Plan` for another, where they need not hold.
#ifdef CONTROL  // the test-only macro below makes `Plan::enter` public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(const Planned &p, const Setting &other) {
  Plan q(other);
  for (PlanRound r : p->rounds_) q.enter(r);
  return q.finish();
}
