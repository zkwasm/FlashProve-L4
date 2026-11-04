// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Replaces a finished plan (`Planned`: a proof's rounds with their error bounds) by one made of
// altered copies of its rounds, error bounds unchanged.
// Fails to compile only because `Plan::enter` is private.
#ifdef CONTROL  // the test-only macro below makes `Plan::enter` public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(Planned p) {
  Plan q(p->set);
  for (PlanRound r : p->rounds_) {
    r.combinations = 1;
    q.enter(r);
  }
  p = q.finish();
  return p;
}
