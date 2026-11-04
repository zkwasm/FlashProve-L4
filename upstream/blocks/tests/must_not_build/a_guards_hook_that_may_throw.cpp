// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// MUST NOT BUILD, by the compiler's words «invalid conversion from .void \(\*\)\(const char\*\). to .void \(\*\)\(const char\*\) noexcept.». With -DCONTROL it builds.
// A failed guard (`BLK_REQUIRE`, blocks/base.h) calls the hook blk::at_a_failed_guard, if set, then
// abort(). The hook's pointer type is noexcept, so that no caller can catch an exception from the
// hook and continue past the failed guard. Here my_end() is not noexcept, so the assignment in
// main() must not compile; the control (-DCONTROL) adds noexcept.
#include "base.h"
#ifdef CONTROL
static void my_end(const char *) noexcept {}
#else
static void my_end(const char *) {}
#endif
int main() { blk::at_a_failed_guard = my_end; }
