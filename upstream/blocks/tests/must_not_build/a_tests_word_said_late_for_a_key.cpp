// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// A production build (SERVING_BUILD) defines the test-only macro BLK_KEY_UNDER_FOR_A_TEST after
// including blocks/base.h. blocks/fence.h, which blocks/base.h includes, rejects such a macro only
// if it is defined on the command line or before that include. So
// blocks/frame/check_key.h, which reads the macro, must stop the build itself: it expands
// BLK_MARK_NOT_FOR_SERVING(), here the static_assert quoted above. The control omits the macro.
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_KEY_UNDER_FOR_A_TEST
#endif
#include "frame/check_key.h"
