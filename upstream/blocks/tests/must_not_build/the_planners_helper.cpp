// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// `MayLay` (blocks/numbers/blocks.h), `Plan`'s only friend, is the private helper of planners
// such as plan_seal(). The function below, no friend of `MayLay`, calls MayLay::close() to add a
// round with its choice of error bound (0) and draw.
#include "numbers/plan.h"
using namespace blk;
void play(Plan &p) {
  plan_seal(p, "a planner's call");
#ifndef CONTROL
  PlanRound r;
  r.draw_positions = 1;
  MayLay::close(p, R_LOC, "", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD, r);
#endif
}
