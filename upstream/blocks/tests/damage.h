// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test helpers that damage the bytes of a valid proof or file. damage() passes many damaged copies
// to the verifier under test, which must reject each by returning false; a verifier that ends the
// process instead, by a signal or a nonzero exit code, fails the test too. in_child() and
// shared_loop() run verifiers in child processes, so that a test outlives one that aborts and can
// report it. Only at DamageChoice::stride = 1 is a bit of every byte flipped.
#pragma once
#include <sched.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "base.h"
#include "field/field.h"
#include "hash/hash.h"

namespace blk {

// ---- child processes ----
// Runs work() in a child process. Returns the child's exit code (the low 8 bits of work()'s
// result), or -1 if a signal ended the child.
inline int in_child(const std::function<int()> &work) {
  std::fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    const int status = work();
    std::fflush(stdout);
    _exit(status);
  }
  int st = 0;
  BLK_REQUIRE(waitpid(child, &st, 0) == child, "tests: a child process is lost");
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

// Children per shared_loop(): the environment variable BLK_TEST_JOBS, or 4; at least 1, and at
// most the processors that this process may run on.
inline unsigned test_jobs() {
  const char *v = std::getenv("BLK_TEST_JOBS");
  cpu_set_t mine;
  const long n = v ? std::atol(v) : 4, most = std::max(1L, sched_getaffinity(0, sizeof mine, &mine) == 0 ? (long)CPU_COUNT(&mine) : sysconf(_SC_NPROCESSORS_ONLN));
  return n < 1 ? 1 : (unsigned)std::min(n, most);
}

// Calls one(i) for every i below count, in jobs = test_jobs() children: child j takes the i with
// i mod jobs = j. On a fault, one(i) prints a line and returns false. What one(i) writes to memory
// is lost with its child. Returns how many children had a fault or were ended by a signal.
inline unsigned shared_loop(size_t count, const std::function<bool(size_t)> &one) {
  const unsigned jobs = test_jobs();
  std::fflush(stdout);
  std::vector<pid_t> children;
  for (unsigned j = 0; j < jobs; j++) {
    const pid_t child = fork();
    BLK_REQUIRE(child >= 0, "tests: no child process");
    if (child == 0) {
      const rlimit no_core{0, 0}; // no core file if the child aborts
      setrlimit(RLIMIT_CORE, &no_core);
      int status = 0;
      for (size_t i = j; i < count; i += jobs)
        if (!one(i)) status = 1;
      std::fflush(stdout);
      _exit(status);
    }
    children.push_back(child);
  }
  unsigned bad = 0;
  for (pid_t child : children) {
    int st = 0;
    BLK_REQUIRE(waitpid(child, &st, 0) == child, "tests: a child process is lost");
    if (!WIFEXITED(st)) std::printf("THE PROCESS WAS ENDED BY SIGNAL %d WHILE A VERIFIER READ A DAMAGED PROOF\n", WTERMSIG(st));
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) bad++;
  }
  return bad;
}

// ---- damage ----
// `stride` must be at least 1 (not checked).
struct DamageChoice {
  unsigned bits = 1; // 1 to 8
  size_t stride = 1;
  enum { ALL, WORDS, CUTS } only = ALL; // WORDS: replaced words only; CUTS: truncated copies only
};
// The default: one bit at up to `places` bytes (`places` > 0, not checked), so a longer input has
// bytes that no flip touches. BLK_TEST_DAMAGE="bits stride" overrides it ("8 1": every bit).
inline DamageChoice damage_choice(size_t bytes, size_t places) {
  DamageChoice c;
  c.stride = 1 + bytes / places;
  if (const char *v = std::getenv("BLK_TEST_DAMAGE")) {
    unsigned long stride = 0;
    BLK_REQUIRE(std::sscanf(v, "%u %lu", &c.bits, &stride) == 2 && c.bits >= 1 && c.bits <= 8 && stride >= 1, "tests: BLK_TEST_DAMAGE is \"bits stride\", bits from 1 to 8");
    c.stride = stride;
  }
  return c;
}
// Copies by type of damage. damage() computes `flips`, `words`, `spellings` and `cuts` from its
// arguments, not as copies are tried: if a signal ended a child early, they include copies that
// were never tried.
struct DamageCount {
  size_t flips = 0, words = 0, spellings = 0, cuts = 0, tails = 0;
  unsigned bad = 0; // children with an accepted copy or a fatal signal, plus accepts in the parent
  void operator+=(const DamageCount &o) { flips += o.flips, words += o.words, spellings += o.spellings, cuts += o.cuts, tails += o.tails, bad += o.bad; }
  std::string text(const DamageChoice &c) const {
    char line[300];
    std::snprintf(line, sizeof line, "%zu with one bit flipped (%u of 8 bits of every %zu-th byte), %zu with a word at or above p (%zu of them w + p), %zu cut short, %zu with bytes behind the end",
                  flips, c.bits, c.stride, words, spellings, cuts, tails);
    return line;
  }
};
// A word at or above p (the field's prime 2^64 - 2^32 + 1, `P` in blocks/field/field.h) for every
// w: w + p (counted in `spellings`) if that fits in 64 bits (w < 2^32 - 1), else 64 ones.
inline u64 at_or_above_p(u64 w) { return w <= ~(u64)0 - P ? w + P : ~(u64)0; }

// Calls `judge` (the verifier under test; true: accepted) on these copies of `good`, or on one type
// of them (`c.only`). In child processes, for every multiple i of c.stride below good.size():
//   - c.bits copies with one bit of byte i flipped, starting at bit i mod 8 (at stride 1, a byte
//     that the verifier does not check shows up as an accepted copy);
//   - where 8 bytes remain, a copy in which at_or_above_p(w) replaces the 8 bytes from i, read as a
//     little-endian word w (no copy if w is 64 ones). Where these bytes hold a field element, w + p
//     is a second encoding of it, which a reader that reduces mod p takes for w, so the check w < p
//     in F_from_bytes() (blocks/field/field.h) must reject it; no one-bit flip gives w + p. In a
//     trial with the prime 2^61 - 1 and a scratch reader that reduced words mod that prime instead
//     of checking them (neither is part of these programs), a toy proof was accepted with any one
//     of about 3,000 of its 5,000 words replaced by a second encoding. For these programs' p, only
//     elements below 2^32 - 1 have a second encoding, so `good` may offer none, and damage() does
//     not report it: `spellings` also counts w + p where the 8 bytes are not an element.
//     blocks/tests/every_byte_test.cpp tries the word 5 + p directly;
//   - the first i bytes only.
// Then, in the calling process: all but the last byte; all bytes followed by a 0, a 1, DIGEST_BYTES
// (32) zeros, or themselves.
// Prints a line beginning with ACCEPTED for each accepted copy. The test failed if `bad` is not 0.
inline DamageCount damage(const std::string &what, const std::vector<u8> &good, const DamageChoice &c, const std::function<bool(const u8 *, size_t)> &judge) {
  DamageCount n;
  const size_t places = (good.size() + c.stride - 1) / c.stride;
  n.bad = shared_loop(places, [&](size_t at) {
    const size_t i = at * c.stride;
    std::vector<u8> damaged = good;
    bool well = true;
    for (unsigned b = 0; c.only == DamageChoice::ALL && b < c.bits; b++) {
      const u8 bit = (u8)(1u << ((i + b) % 8));
      damaged[i] ^= bit;
      if (judge(damaged.data(), damaged.size())) {
        std::printf("ACCEPTED WITH A BIT FLIPPED: %s: byte %zu of %zu, bit value %u\n", what.c_str(), i, good.size(), bit);
        well = false;
      }
      damaged[i] ^= bit;
    }
    if (c.only != DamageChoice::CUTS && i + 8 <= good.size() && at_or_above_p(word_from_bytes(good.data() + i)) != word_from_bytes(good.data() + i)) {
      word_to_bytes(at_or_above_p(word_from_bytes(good.data() + i)), damaged.data() + i);
      if (judge(damaged.data(), damaged.size())) {
        std::printf("ACCEPTED WITH A WORD AT OR ABOVE p: %s: the eight bytes from byte %zu of %zu on\n", what.c_str(), i, good.size());
        well = false;
      }
    }
    if (c.only != DamageChoice::WORDS && judge(good.data(), i)) {
      std::printf("ACCEPTED CUT SHORT: %s: %zu of %zu bytes\n", what.c_str(), i, good.size());
      well = false;
    }
    return well;
  });
  for (size_t i = 0; c.only != DamageChoice::CUTS && i + 8 <= good.size(); i += c.stride) {
    const u64 w = word_from_bytes(good.data() + i);
    n.words += at_or_above_p(w) != w, n.spellings += w <= ~(u64)0 - P;
  }
  if (c.only == DamageChoice::WORDS) return n;
  n.cuts = places + 1;
  if (!good.empty() && judge(good.data(), good.size() - 1)) {
    std::printf("ACCEPTED CUT SHORT: %s: without its last byte\n", what.c_str());
    n.bad++;
  }
  if (c.only == DamageChoice::CUTS) return n;
  n.flips = places * c.bits;
  for (const std::vector<u8> &more : {std::vector<u8>{0}, std::vector<u8>{1}, std::vector<u8>(DIGEST_BYTES, 0), good}) {
    std::vector<u8> longer = good;
    longer.insert(longer.end(), more.begin(), more.end());
    if (judge(longer.data(), longer.size())) {
      std::printf("ACCEPTED WITH BYTES BEHIND THE END: %s: %zu more\n", what.c_str(), more.size());
      n.bad++;
    }
    n.tails++;
  }
  return n;
}

} // namespace blk
