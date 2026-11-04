// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Fills a `Plan` (a proof's messages and rounds) by hand, through private members that only its
// friend `MayLay` (blocks/numbers/blocks.h) may call.
#ifdef CONTROL  // control: the test-only macro below makes them public
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(const Setting &s) {
  Plan p(s);
  p.send_root("root"), p.send_scalars("scalars", 3), p.send_answers("answers", 1);
  p.close(R_LOC, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD).draw_positions = 1;
  p.file("paths", MessageKind::PATHS, 32, 32.0, 1), p.last_message_at("nowhere");
  return p.finish();
}
