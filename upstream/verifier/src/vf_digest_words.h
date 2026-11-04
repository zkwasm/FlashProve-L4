// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

/* DIGEST_WORDS is the width of a digest of the hash H (verifier/src/vf_earlier_hash.h) in 32-bit words:
   8 (32 bytes). It is a compile-time constant: no run-time option or input file sets it.
   verifier/build.sh passes -DDIGEST_WORDS=8; any other value fails the static_assert below.
   verifier/src/vf_widths_and_bounds.h calls it `DW`, so an expression `DW == 8 ? a : b` always gives a:
   b is dead code. */
#pragma once
#ifndef DIGEST_WORDS
#define DIGEST_WORDS 8
#endif
static_assert(DIGEST_WORDS == 8, "a digest has 8 words");
