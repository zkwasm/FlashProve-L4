// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «a header for tests alone in a program that serves». With -DCONTROL it builds.
// In production builds (`SERVING_BUILD`) blocks/fence.h rejects the test-only macros it lists if
// defined on the command line or before blocks/base.h is first included. This one comes later, so
// blocks/iop/wf.h, which reads it, stops the build with `BLK_MARK_NOT_FOR_SERVING()`.
// `BLK_WF_ROWS_FOR_A_TEST` lets a test run WF, the committer's range proof, without its proximity
// test. tests/new/run_new.py compiles the file both ways: the control, the same production build
// without the macro, shows that the macro alone stops the build.
#define SERVING_BUILD 1
#include "base.h"
#ifndef CONTROL
#define BLK_WF_ROWS_FOR_A_TEST
#endif
#include "iop/wf.h"
