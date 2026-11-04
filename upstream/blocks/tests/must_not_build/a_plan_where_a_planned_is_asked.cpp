// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «could not convert|conversion from|cannot convert». With -DCONTROL it builds.
// A writable `Plan` does not convert to a finished `Planned`: a caller must call Plan::finish().
#ifdef CONTROL  // control: the macro below, meant for tests, adds this conversion
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#endif
#include "numbers/plan.h"
using namespace blk;
Planned play(const Plan &p) { return p; }
