// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Fault-injection hooks (`door` in identifiers) for tests (a lost copy, a lost kernel launch), in
// addition to the one in lost_coins_door.cuh. They are called where code generates or copies
// private random values (such as mask entries, added to sent scalars) or launches a kernel that
// copies, adds or checks them. After an injected fault the self-checks must withhold the file. A
// fault that none notices ends up in a file that cp_prove writes out. Each hook
// reads one environment variable:
//   ROWS_LOSE_COPY=<name>:<call>:<first>:<words>   strike() zeroes 64-bit words of the copy
//   ROWS_LOSE_LAUNCH=<name>:<call>                 left_out() tells the caller to skip the launch
//   <name>   the call site's name
//   <call>   which call with that name in this translation unit, counted from 0, or * for all
//   <first>  the first word zeroed
//   <words>  how many words are zeroed; the part beyond the copy's end is ignored
// For each injected fault a hook prints one line, "lost copy: .." or "lost launch: ..", so that a
// test can tell such a fault from a variable that selected nothing.
//
// The production build (SERVING_BUILD) has only empty hooks. In any other build they act only after
// lost_coins_door::open_if() has recorded that nothing is hidden in this run, which requires
// synthetic weights and a test seed. In such a run a malformed value, or a <words> of 0, makes the
// first call of its hook exit with code 2.
#pragma once
#include "hiding/lost_coins_door.cuh"
namespace lost_copies_door {
#ifdef SERVING_BUILD
static inline void strike(const char *, void *, size_t, bool) {}
static inline bool left_out(const char *) {
  return false;
}
#else
// Returns true if `word`, the value of the environment variable `variable` (null: not set), selects
// this call from the call site `name`. `numbers` receives the `further` (0 or 2) remaining fields.
static bool named(const char *word, const char *variable, int further, const char *name, unsigned long long numbers[2]) {
  if (!word || !lost_coins_door::nothing_is_hidden) return false;
  char place[64] = "", which[24] = "", again[160] = "";
  numbers[0] = numbers[1] = 0;
  const int fields = sscanf(word, "%63[^:]:%23[^:]:%llu:%llu", place, which, &numbers[0], &numbers[1]);
  if (further)
    snprintf(again, sizeof again, "%s:%s:%llu:%llu", place, which, numbers[0], numbers[1]);
  else
    snprintf(again, sizeof again, "%s:%s", place, which);
  if (fields != 2 + further || strcmp(word, again) || (further && !numbers[1]) ||
      (strcmp(which, "*") && which[strspn(which, "0123456789")])) {
    fprintf(stderr, "%s takes <name>:<call, or *>%s\n", variable, further ? ":<first word>:<words>" : "");
    exit(2);
  }
  if (strcmp(name, place)) return false;
  static std::mutex one_at_a_time;
  static std::map<std::string, unsigned long long> calls; // per variable: each has one <name>
  std::lock_guard<std::mutex> lock(one_at_a_time);
  const unsigned long long this_call = calls[variable]++;
  return !strcmp(which, "*") || strtoull(which, nullptr, 10) == this_call;
}
// Call it after the copy. `copy`, the destination, has `words` 64-bit words, on the GPU (the copy
// must be queued on the calling thread's cudaStreamPerThread) or in host memory (the copy must have
// completed).
static void strike(const char *name, void *copy, size_t words, bool on_the_gpu) {
  unsigned long long lost[2];
  if (!named(TESTENV("ROWS_LOSE_COPY"), "ROWS_LOSE_COPY", 2, name, lost) || lost[0] >= words) return;
  const unsigned long long to = lost[1] < words - lost[0] ? lost[0] + lost[1] : words;
  if (!on_the_gpu)
    memset((uint64_t *)copy + lost[0], 0, 8 * (to - lost[0]));
  else if (cudaMemsetAsync((uint64_t *)copy + lost[0], 0, 8 * (to - lost[0]), cudaStreamPerThread) != cudaSuccess) {
    fprintf(stderr, "ROWS_LOSE_COPY: the words could not be zeroed\n");
    exit(2);
  }
  printf("lost copy: %s: the words %llu to %llu are zeroed behind the copy\n", name, lost[0], to - 1);
}
// Call it before the launch: if (!left_out("..")) kernel<<<..>>>(..).
static bool left_out(const char *name) {
  unsigned long long none[2];
  if (!named(TESTENV("ROWS_LOSE_LAUNCH"), "ROWS_LOSE_LAUNCH", 0, name, none)) return false;
  printf("lost launch: %s: the launch is left out\n", name);
  return true;
}
#endif
} // namespace lost_copies_door
