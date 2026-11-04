// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «is private within this context». With -DCONTROL it builds.
// Deletes a finished plan's data, shared by `Planned` handles: Planned::Data::~Data() is private.
#include "numbers/plan.h"
using namespace blk;
void play(const Planned &q) {
#ifndef CONTROL
  delete &*q;
#endif
  (void)q;
}
