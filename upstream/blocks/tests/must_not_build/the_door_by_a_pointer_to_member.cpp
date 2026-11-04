// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Alters an unfinished `Plan` (a protocol run's list of rounds) through a pointer to the private
// member Plan::rounds_.
#ifdef CONTROL  // control: the test-only macro below makes the member public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
void play(Plan &p) {
  std::vector<PlanRound> Plan::*door = &Plan::rounds_;
  (p.*door)[0].draw_positions = 1;
}
