// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// tests/new/run_new.py parses that line.
// This production build (`SERVING_BUILD`) defines the test-only macro
// `BLK_ENTRY_UNDER_FOR_A_TEST` after including blocks/base.h. blocks/fence.h, included there,
// rejects it only if defined on the command line or before that include. So
// blocks/frame/check_entry.h, where the macro adds a public constructor to `EntryUnder`, expands
// blocks/fence.h's `BLK_MARK_NOT_FOR_SERVING` itself: here a failing static_assert.
// The control omits only the macro.
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_ENTRY_UNDER_FOR_A_TEST
#endif
#include "frame/check_entry.h"
