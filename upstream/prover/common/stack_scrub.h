// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Zeroes the part of the calling thread's stack that a function has just used. Host code only; any
// thread may call it.
// The headers in prover/second_writing/, a second implementation of how private randomness
// (`coins` in identifiers) is derived, leave ChaCha20 keys and states in local arrays, and must
// stay byte-identical to their copies in prover/tests/coins_second_text/
// (prover/second_writing/README.md). So a program that derives secret values with them must
// (1) make the calls inside a separate, never-inlined function, so that their stack use stays below
// the caller's frame, and (2) call scrub_the_stack_below() from that function's caller, right after
// it returns, on every path, including failure (some of these headers throw).
#pragma once
#include <cstddef>
#include <string.h>

// scrub_the_stack_below() zeroes STACK_SCRUBBED bytes (32 KiB) and needs slightly more free stack.
// If the caller's stack pointer is the same at this call as at the call before it (not checked),
// both frames start at the same address, and the array `below` covers what the earlier call left,
// down to about STACK_SCRUBBED bytes. How deep the earlier call reached is not checked either.
// Not erased: the callers' frames; padding above the array; stack below it; registers, which can
// hold an entire key (a later register save writes it to the stack again); other threads' stacks.
constexpr size_t STACK_SCRUBBED = (size_t)32 << 10;
__attribute__((noinline)) static void scrub_the_stack_below() {
  unsigned char below[STACK_SCRUBBED];
  explicit_bzero(below, sizeof below);
}
