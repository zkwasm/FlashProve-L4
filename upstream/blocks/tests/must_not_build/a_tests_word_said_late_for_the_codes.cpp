// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// The check for test-only macros in blocks/fence.h sees only those defined on the command line
// or before blocks/base.h is first included. This production build (`SERVING_BUILD`) afterwards
// sets `BLK_GL64_SUBGROUP_LOG`, a parameter of the Reed-Solomon code's evaluation points, to a
// test-only value (any but 32). So blocks/code/cosets.h, which reads it, stops the build with
// BLK_MARK_NOT_FOR_SERVING() (blocks/fence.h). The control only omits that #define.
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_GL64_SUBGROUP_LOG 5
#endif
#include "code/cosets.h"
