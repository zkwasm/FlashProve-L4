// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «in read-only object». With -DCONTROL it builds.
// Changes the distances of a table's test; later tests' error bounds assume them.
// `TestedTable` has no public data member; reader() returns a const reference.
#include "iop/foldeq.h"
using namespace blk;
u64 play(TestedTable &tested) {
#ifndef CONTROL
  tested.reader().shape().distances.radius = 1;
#endif
  return tested.reader().shape().distances.radius;
}
