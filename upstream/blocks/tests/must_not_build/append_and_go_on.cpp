// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Calls the private Plan::append() and close_none() to add a round after a finished plan's rounds.
// Either alone stops the build; blocks/numbers/tests/plan_types_test.cpp tests each separately.
#ifdef CONTROL  // control: the test-only macro below makes both public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(const Planned &done) {
  Plan q(done->set);
  q.append(done);
  q.close_none("a round laid behind a finished plan's rounds");
  return q.finish();
}
