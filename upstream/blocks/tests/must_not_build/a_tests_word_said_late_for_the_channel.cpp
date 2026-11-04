// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// tests/new/run_new.py reads that line. In production builds (`SERVING_BUILD`),
// blocks/fence.h rejects the test-only macros it lists, but only if defined on the command line
// or before blocks/base.h is first included. BLK_TEST_PROVER_MESSAGES_OUTSIDE (lets a prover's
// challenges ignore its messages) is defined later, so blocks/chain/rounds.h, which reads it,
// stops the build with BLK_MARK_NOT_FOR_SERVING(), here a static_assert with the words above.
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE
#endif
#include "chain/rounds.h"
