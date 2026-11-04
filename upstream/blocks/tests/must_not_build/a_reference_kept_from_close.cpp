// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Calls the private Plan::close() and, after Plan::finish(), changes the new round through the
// returned reference.
#ifdef CONTROL  // control: the test-only macro below makes `close` public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(Plan &p) {
  PlanRound &kept = p.close(R_LOC, RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-129)), Priced::AS_SAID);
  const Planned q = p.finish();
  kept.draw_positions = 1;
  return q;
}
