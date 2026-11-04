// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «redefinition of». With -DCONTROL it builds.
// (tests/new/run_new.py parses the line above.)
// Defines `MayLay`, the friend class of `Plan`, to add a round with the private Plan::close_none().
// The header included below includes blocks/numbers/blocks.h, which already defines `MayLay`.
#include "numbers/plan.h"
using namespace blk;
#ifndef CONTROL
namespace blk {
class MayLay {
 public:
  static void lay(Plan &p) { p.close_none("a round of my own"); }
};
}  // namespace blk
#endif
