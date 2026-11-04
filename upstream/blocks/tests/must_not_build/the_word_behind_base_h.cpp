// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD with -DSERVING_BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// Defines a test-only macro after blocks/base.h includes blocks/fence.h, too late for that
// header's check for such macros. Under SERVING_BUILD (a production build), blocks/numbers/plan.h,
// which reads the macro, must stop the build with BLK_MARK_NOT_FOR_SERVING(). For the control,
// tests/new/run_new.py omits -DSERVING_BUILD.
#include "base.h"
#define BLK_PLANS_BY_HAND_FOR_A_TEST
#include "numbers/plan.h"
