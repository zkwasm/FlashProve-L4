// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Stubs of fault-injection hooks for tests of the prover program cp_prove. No program built
// from these sources parses this file's switches, which are values of the environment variable
// COINS_REFUSED_DRAW, or injects a fault at one: every build from these sources has only the stubs below,
// which inject nothing. Nothing here is compiled under the macro REFUSAL_TEST_WORDS. A production build
// (SERVING_BUILD) with the macro does not compile (prover/common/setting.h). Nothing exercises a
// switch: no target of prover/build.sh sets the macro, and no test in this tree compiles with it.
// prover/prover/closed_front.h requires CLOSED_KEEPS_THE_TEST_WORD=1 together with it, and with
// that macro cp_prove restarts itself without that variable (ClosedFront::enter()). spells_p()
// (prover/common/element_from_hash.h) reads that variable, though not in a production build.
//
// Public sources: fault injection at named points, compiled in only for tests, set through one
// environment variable and stopped by a malformed action, as described in: The Rust crate 'fail',
// documentation on docs.rs: the crate's page https://docs.rs/fail ('Usage considerations'),
// https://docs.rs/fail/latest/fail/struct.FailScenario.html (setup) and
// https://docs.rs/fail/latest/fail/fn.cfg.html.
// Public sources: failing the N-th call, as documented for Linux's fail-nth: The Linux kernel
// documentation, 'Fault injection capabilities infrastructure', 'proc entries';
// https://docs.kernel.org/fault-injection/fault-injection.html.
// The places are this program's own.
#pragma once
#ifndef REFUSAL_TEST_WORDS
#define REFUSAL_TEST_WORDS 0
#endif
#include "common/setting.h" // DW
#include <cstdint>

namespace refusal_test {
inline bool knows(const char *) {
  return false;
}
inline unsigned seat_bits(int, const char *, long, bool) {
  return 0;
}
inline unsigned fold_bits() {
  return 0;
}
inline bool leaves_pinned() {
  return false;
}
inline void pinned_leaf(int, uint32_t *) {}
inline unsigned between_bits(bool) {
  return 0;
}
inline bool strikes(const char *, long) {
  return false;
}
inline void a_try_begins(const char * = "", int = -1) {}
inline void a_try_ends() {}
inline bool nth_call(const char *) {
  return false;
}
} // namespace refusal_test
