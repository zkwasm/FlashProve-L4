// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «no matching function for call to». With -DCONTROL it builds.
// Constructs a writable `Plan` from a finished plan (`Planned`), to alter copies of its rounds.
#ifdef CONTROL  // the macro below, meant for tests, adds that constructor
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Plan play(const Planned &q) { return Plan(q); }
