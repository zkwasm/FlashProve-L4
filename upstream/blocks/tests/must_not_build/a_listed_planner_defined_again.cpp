// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «redefinition of». With -DCONTROL it builds.
// Defines plan_zero() again. `MayLay` (blocks/numbers/blocks.h) names it a friend by signature, so
// the new body could add its own rounds to a `Plan`.
#include "numbers/plan.h"
using namespace blk;
#ifndef CONTROL
namespace blk {
void plan_zero(Plan &p, const std::string &where, unsigned) { MayLay::none(p, "a round of my own", where); }
}  // namespace blk
#endif
