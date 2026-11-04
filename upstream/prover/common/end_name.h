// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The name of the reason why the program ends, such as invalid input, a failed self-check or a
// failed CUDA call. Set it before calling exit() or returning a nonzero code up to main().
// The prover, cp_prove, prints it in its last line unless the exit code is 0. The production build
// (the default of prover/build.sh) prints "p258 end <status> <name>" (status: the exit code;
// prover/prover/closed_front.h); a test build prints "ended: cause .., by <name>"
// (prover/prover/ends.h).
// A program that prints no such line, such as the committer cp_commit, never reads the name.
// A name must be a string literal, so that it depends only on which line of the program chose it.
// Nothing here checks that.
// Once the production build holds a secret, its last line is all that it prints about its end.
// Apart from the exit code, no number, count or sentence goes with the name, because those could
// depend on hidden memory or on a request.
// prover/docs/ends-in-short.md, "The names in the last line", explains the names.
#pragma once
#include <atomic>
inline std::atomic<const char *> g_end_name{nullptr};
static_assert(std::atomic<const char *>::is_always_lock_free, "g_end_name is read in signal handlers");
// The translation unit that owns the weight table (`keeper` in identifiers) is linked into cp_prove
// three times, with local symbols (prover/build.sh, keeper_unit()), so each copy has a separate
// g_end_name. A copy's set-up points its g_end_name_at to the main program's, the one printed.
inline std::atomic<const char *> *g_end_name_at = &g_end_name;
// The first name stays: the first fault is the one to look for, and what fails after it may be its
// consequence. cp_prove also calls this with the name of a failed BLK_REQUIRE_NAMED() check
// (blocks/base.h).
inline void name_the_end(const char *literal) {
  const char *none = nullptr;
  g_end_name_at->compare_exchange_strong(none, literal);
}
// Returns the name, or null if none was set. A name set later is lost: if none was set, it stores
// "", which readers take as no name, so that nothing that fails during shutdown can set one.
inline const char *take_the_ends_name() {
  name_the_end("");
  const char *const name = g_end_name_at->load();
  return *name ? name : nullptr;
}
// Whether the name is that of a self-check on secret random values or keys: those names begin
// "own-test-". cp_prove exits with code 9 after such a name (prover/prover/ends.h).
inline bool names_a_self_check(const char *name) {
  const char *head = "own-test-";
  while (name && *head && *name == *head) name++, head++;
  return name && !*head;
}
// For a function that has printed why it stops and returns 2 up to main():
// return refused("cp-isk");
inline int refused(const char *literal) {
  name_the_end(literal);
  return 2;
}
