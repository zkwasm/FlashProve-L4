// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// A production build (`SERVING_BUILD`) defines a test-only macro after including blocks/base.h, too
// late for the check in blocks/fence.h. blocks/iop/foldeq.h, which reads the macro, must stop the
// build itself, with BLK_MARK_NOT_FOR_SERVING().
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_TESTED_TABLE_FOR_A_TEST
#endif
#include "iop/foldeq.h"
