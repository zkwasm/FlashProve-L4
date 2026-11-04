// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «redefinition of». With -DCONTROL it builds.
// Defines its own `MayLay`, the friend class that adds rounds to a `Plan`, before including
// blocks/numbers/plan.h, which includes the real definition (blocks/numbers/blocks.h).
#ifndef CONTROL
namespace blk {
class Plan;
class MayLay {
 public:
  static void lay(Plan &p);
};
}  // namespace blk
#endif
#include "numbers/plan.h"
using namespace blk;
