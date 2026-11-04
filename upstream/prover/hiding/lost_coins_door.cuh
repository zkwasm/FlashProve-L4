// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A fault-injection hook for tests: it zeroes private random values (`coins` in names and messages)
// in GPU memory right after they are generated, as after a lost write. The self-checks of
// cp_commit and cp_prove must then stop the run or withhold the file that it was
// producing; docs/how-hiding-is-checked.md tells of them. The values
// are those of a row commitment (hz::Tree in prover/hiding/trees.cuh; a "tree" here). The
// environment variable ROWS_LOSE_COINS selects which are zeroed; a_draw() explains its fields.
//
// The production build (SERVING_BUILD) has only empty stubs. In any other build the hook is
// disabled until open_if(true), which is only for a run that has nothing to hide. The state is
// static: a translation unit that does not call open_if() injects nothing.
#pragma once
#include "common/made_up_alone.h"
#include "common/setting.h" // TESTENV
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
namespace lost_coins_door {
enum Part { MASK = 0, ROWS = 1, SALT = 2 }; // a part's stream is the tree's first stream plus this number
#ifdef SERVING_BUILD
struct Draw {};
static inline void open_if(bool) {}
static inline Draw a_draw(const std::string &) {
  return Draw();
}
static inline void strike(const Draw &, int, void *, uint64_t, size_t) {}
#else
// True while the hook is enabled; prover/hiding/lost_copies_door.cuh reads it too. Pass open_if()
// true only in a run that has nothing to hide: its seed or its random values come from a test
// (cp_commit passes true under -test-seed, cp_prove under -test-coins). Call made_up_alone::say()
// (prover/common/made_up_alone.h) first, in the same translation unit: open_if() reads its answer
// at the call. The hook stays disabled unless both of say()'s arguments were true, synthetic
// weights (`made_up` in identifiers) and a test seed, so that open_if(true) alone does not enable
// it on a model's files.
static bool nothing_is_hidden = false;
static inline void open_if(bool nothing_is_hidden_in_this_run) {
  nothing_is_hidden = nothing_is_hidden_in_this_run && made_up_under_a_tests_seed();
}
// What ROWS_LOSE_COINS selects in one draw, that is, one call of a_draw(). Unless `struck`,
// strike() does nothing.
struct Draw {
  bool struck = false;
  int part = 0;
  unsigned long long first = 0, cells = 0;
  std::string tree;
};
// Counts one draw of the tree `name` and returns what to zero in it. Thread-safe. One function
// calls a_draw(), and strike() on each buffer just filled: hz::Tree::slot_pads(),
// which fills the mask entries (`pads` in identifiers) added to a sub-proof's
// messages and calls strike() with ROWS: the parts mask and salt select nothing.
// ROWS_LOSE_COINS=<tree>:<draw>:<part>:<first>:<cells>
//   <tree>   the tree's name, hz::Tree::nm
//   <draw>   which call of a_draw() with that name in this translation unit, from 0, or * for all;
//            calls made while the hook is disabled are not counted
//   <part>   mask (hz::Tree::mask: the blinding coordinates, random values encoded with each data
//            row), rows (hz::Tree::ext: the random rows B and h_i) or salt
//   <first>  the first lost cell (a 64-bit value), by its number in the part's stream of values
//   <cells>  how many cells are lost, at least 1; cells beyond the generated range are ignored
// A malformed value makes the program exit with code 2.
static Draw a_draw(const std::string &name) {
  Draw draw;
  const char *word = nothing_is_hidden ? TESTENV("ROWS_LOSE_COINS") : nullptr;
  if (!word) return draw;
  char tree[64] = "", which[24] = "", part[8] = "", again[160] = "";
  const int fields = sscanf(word, "%63[^:]:%23[^:]:%7[^:]:%llu:%llu", tree, which, part, &draw.first, &draw.cells);
  snprintf(again, sizeof again, "%s:%s:%s:%llu:%llu", tree, which, part, draw.first, draw.cells);
  if (fields != 5 || strcmp(word, again) || !draw.cells || (strcmp(which, "*") && which[strspn(which, "0123456789")]) ||
      (strcmp(part, "mask") && strcmp(part, "rows") && strcmp(part, "salt"))) {
    fprintf(stderr, "ROWS_LOSE_COINS takes <tree>:<draw, or *>:<mask, rows or salt>:<first cell>:<cells>; the program ends\n");
    exit(2);
  }
  if (name != tree) return draw;
  static std::mutex one_at_a_time;
  static unsigned long long draws = 0; // counts only draws of the tree that the variable names
  std::lock_guard<std::mutex> lock(one_at_a_time);
  const unsigned long long this_draw = draws++;
  draw.struck = !strcmp(which, "*") || strtoull(which, nullptr, 10) == this_draw;
  draw.part = !strcmp(part, "mask") ? MASK : !strcmp(part, "rows") ? ROWS : SALT;
  draw.tree = name;
  return draw;
}
// Zeroes the cells that `draw` selects among those just generated. `stretch` is the GPU address of
// cell number `first` of the part's stream; `cells` cells were generated there. Call it on the
// host thread that launched the fill: it queues the memset on that thread's default CUDA stream
// (cudaStreamPerThread), on which the fill must be; this is not checked. It prints one line,
// "lost coins: ..", for each range that it zeroes: without that line, the fault has hit no cell.
static void strike(const Draw &draw, int part, void *stretch, uint64_t first, size_t cells) {
  const unsigned long long drawn_to = first + cells; // one past the stretch's last cell
  if (!draw.struck || part != draw.part || draw.first >= drawn_to) return;
  const unsigned long long from = draw.first > first ? draw.first : first,
                           to = draw.cells < drawn_to - draw.first ? draw.first + draw.cells : drawn_to;
  if (from >= to) return;
  if (cudaMemsetAsync((uint64_t *)stretch + (from - first), 0, 8 * (to - from), cudaStreamPerThread) != cudaSuccess) {
    fprintf(stderr, "ROWS_LOSE_COINS: the cells could not be zeroed; the program ends\n");
    exit(2);
  }
  printf("lost coins: tree %s, part %s: the cells %llu to %llu are zeroed behind their draw\n", draw.tree.c_str(),
         part == MASK ? "mask" : part == ROWS ? "rows" : "salt", from, to - 1);
  fflush(stdout); // keeps the line whole in a log that also receives standard error
}
#endif
} // namespace lost_coins_door
