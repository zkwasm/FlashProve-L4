// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «was not declared in this scope». With -DCONTROL it builds.
// No plan_close() exists; MayLay::close() (blocks/numbers/blocks.h), which adds rounds, is private.
#include "numbers/plan.h"
using namespace blk;
void play(Plan &p) {
  plan_seal(p, "a planner's call");
#ifndef CONTROL
  plan_close(p, R_LOC, "", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_positions = 1;
#endif
}
