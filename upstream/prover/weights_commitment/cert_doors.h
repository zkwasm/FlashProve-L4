// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Hooks (namespace `cert_doors`) for tests of the committer program cp_commit, controlled by
// environment variables that TESTENV (prover/common/setting.h) reads. In the production build
// (SERVING_BUILD) they do nothing. No test of this repository sets one of the variables that their
// acting form reads, so none shows this under a set variable; prover/build.sh installs no
// production program one of whose printable strings equals the name of such a variable
// (prover/tests/serving_build_words.txt). cp_commit calls only once_key_fault(), and cp_prove calls
// none: neither calls the other hooks, and the self-checks that they would trigger are not in
// them. Include after prover/hiding/trees.cuh, which defines
// hz::Tree.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace cert_doors {
// A hook's argument g: THE_FILE is the committer's range proof, lookups meant to show that every
// committed weight is in its range; THE_RUN is cp_commit's whole run. `coins` in identifiers: the
// committer's private random values. A pad is one that masks a value before it is sent.
constexpr u32 THE_FILE = 0, THE_RUN = 1;
// What the hooks answer. Where one injects a fault, a self-check of its caller must stop the run.
//   on()              whether mark() and note() print
//   stop_behind()     whether the caller stops after step `step` of g, as at an internal error
//   by_strips()       1 or 0 as ROWS_STRIPS says, -1 if it is not set; nothing acts on the answer
//   pattern()         always -1
//   again()           the byte that fills the lookup calls' shared scratch before each use, or -1
//   bends_U(f)        whether entry 0 of lookup call f's vector U_f is altered before it is checked
//   dumps(g)          whether dump() stores g's hidden tables
//   lost_pads()       which random values of the range proof's crossing-pad tree are lost: zeroed,
//                     as after a GPU write that never arrived (see LostPads)
//   lost_slot_pads()  the same for the range proof's pad tree
//   lost_link_pads()  the same for the LINK proof's pad tree in each leaf file (`ivk/leaf/<k>.bin`)
//                     (this proof ties the file's leaf table to the anchor, the first weight table)
//   once_key_fault()  KEY_OF_ANOTHER_SEED: the key for the key file's random values is derived
//                     from a seed other than the stored one. Nothing acts on KEY_CHANGES_LATER.
enum OnceKeyFault { KEY_SOUND = 0, KEY_OF_ANOTHER_SEED = 1, KEY_CHANGES_LATER = 2 };
struct LostPads {
  // lose_coins(), after a tree's values are drawn and before it is built, zeroes `count` cells from
  // cell `first` of pad row `row` (EDEG planes of `sh.Kc` cells) or, with `salts`, of the salts.
  bool on_the_gpu = false, salts = false;
  // lose_engine_pads() zeroes `count` cells, from cell `first` on, in the copy of the pads that the
  // GPU's engine masks with. There a pad is EDEG (the extension field's degree) adjacent cells.
  bool engine = false;
  // The caller zeroes the host's copy of pad `first`, before or after its self-check on the values.
  bool copy_before = false, copy_behind = false;
  unsigned long long row = 0, first = 0, count = 0;
};
#ifdef SERVING_BUILD
static OnceKeyFault once_key_fault() {
  return KEY_SOUND;
}
static LostPads lost_pads() {
  return LostPads();
}
static LostPads lost_slot_pads() {
  return LostPads();
}
static LostPads lost_link_pads() {
  return LostPads();
}
static void lose_coins(const ::hz::Tree &, const LostPads &) {}
static void lose_engine_pads(void *, size_t, const LostPads &) {}
static int again() {
  return -1;
}
static bool bends_U(u32) {
  return false;
}
static bool on() {
  return false;
}
static bool stop_behind(u32, int) {
  return false;
}
static int by_strips() {
  return -1;
}
static int pattern() {
  return -1;
}
static bool dumps(u32) {
  return false;
}
static void keep(const char *, const void *, size_t) {}
#else
// True if nothing in this run is hidden: cp_commit sets it for synthetic weights under -test-seed.
// Dumps show hidden tables, and notes say which self-check failed; they, marks, the lost-pad hooks
// and once_key_fault() act only then. The other hooks do not read it.
static bool nothing_is_hidden = false;
static bool on() {
  return TESTENV("ROWS_REPORT") != nullptr && nothing_is_hidden;
}
// ROWS_STOP_BEHIND=<g>:<step>. With <g>:<step>:exit the process exits here instead, with code 4, as
// functions of prover/hiding do at an internal error.
static bool stop_behind(u32 g, int step) {
  const char *word = TESTENV("ROWS_STOP_BEHIND");
  unsigned file = 0;
  int at = 0;
  char how[8] = "";
  if (!word || sscanf(word, "%u:%d:%7s", &file, &at, how) < 2 || file != g || at != step) return false;
  if (!strcmp(how, "exit")) exit(4);
  return true;
}
// ROWS_STRIPS=0 or 1.
static int by_strips() {
  const char *word = TESTENV("ROWS_STRIPS");
  return word ? atoi(word) != 0 : -1;
}
// Always -1: with LENT_POISON set, lent::take() (lent_memory.h) fills every buffer that it returns.
static int pattern() {
  return -1;
}
// LENT_POISON's byte.
static int again() {
  return ::lent::poison_byte();
}
// ROWS_BEND_U=<f>.
static bool bends_U(u32 f) {
  const char *word = TESTENV("ROWS_BEND_U");
  char *end = nullptr;
  return word && *word && strtoul(word, &end, 10) == f && !*end;
}
// The forms of ROWS_LOSE_*: <first>:<count>, row:<row>:<first>:<count>, salt:<first>:<count>; where
// `more` has the letter, e: engine:<first>:<count>; c: copy:<pad> (before the self-check); h:
// host:<pad> (after it). Other forms: exit code 2. The three hooks act only if `nothing_is_hidden`:
// with no test seed, a lost pad would be left to the caller's self-check on the drawn values alone.
static LostPads lost_cells(const char *name, const char *word, const char *more) {
  LostPads lost;
  if (!word) return lost;
  char behind = 0;
  if (strchr(word, '-') || strchr(word, '+') || strchr(word, ' ')) word = "";
  if (sscanf(word, "row:%llu:%llu:%llu%c", &lost.row, &lost.first, &lost.count, &behind) == 3 && lost.count)
    lost.on_the_gpu = true;
  else if (sscanf(word, "salt:%llu:%llu%c", &lost.first, &lost.count, &behind) == 2 && lost.count)
    lost.on_the_gpu = lost.salts = true;
  else if (strchr(more, 'e') && sscanf(word, "engine:%llu:%llu%c", &lost.first, &lost.count, &behind) == 2 && lost.count)
    lost.engine = true;
  else if (sscanf(word, "%llu:%llu%c", &lost.first, &lost.count, &behind) == 2 && lost.count)
    lost.on_the_gpu = true;
  else if (strchr(more, 'c') && sscanf(word, "copy:%llu%c", &lost.first, &behind) == 1)
    lost.copy_before = true;
  else if (strchr(more, 'h') && sscanf(word, "host:%llu%c", &lost.first, &behind) == 1)
    lost.copy_behind = true;
  else {
    fprintf(stderr, "%s takes one of: <first cell>:<cells>; that behind row:<row>: or salt:%s%s%s\n", name,
            strchr(more, 'e') ? " or engine:" : "", strchr(more, 'c') ? "; copy:<pad>" : "", strchr(more, 'h') ? "; host:<pad>" : "");
    exit(2);
  }
  return lost;
}
// ROWS_ONCE_KEY=seed or later; acts only if `nothing_is_hidden`. Used by take_the_key_files(); the
// self-check is hold_the_key_files(), error own-test-key (prover/hash_commitment/hf_commit_keys.h).
static OnceKeyFault once_key_fault() {
  const char *word = nothing_is_hidden ? TESTENV("ROWS_ONCE_KEY") : nullptr;
  if (!word) return KEY_SOUND;
  if (!strcmp(word, "seed")) return KEY_OF_ANOTHER_SEED;
  if (!strcmp(word, "later")) return KEY_CHANGES_LATER;
  fprintf(stderr, "ROWS_ONCE_KEY must be the word seed or the word later\n");
  exit(2);
}
static LostPads lost_pads() {
  return lost_cells("ROWS_LOSE_PADS", nothing_is_hidden ? TESTENV("ROWS_LOSE_PADS") : nullptr, "ch");
}
static LostPads lost_slot_pads() {
  return lost_cells("ROWS_LOSE_SLOT_PADS", nothing_is_hidden ? TESTENV("ROWS_LOSE_SLOT_PADS") : nullptr, "ce");
}
static LostPads lost_link_pads() {
  return lost_cells("ROWS_LOSE_LINK_PADS", nothing_is_hidden ? TESTENV("ROWS_LOSE_LINK_PADS") : nullptr, "e");
}
static void lose_coins(const ::hz::Tree &tree, const LostPads &lost) {
  if (!lost.on_the_gpu) return;
  const size_t cells = lost.salts ? (size_t)tree.nsalt * tree.sh.n : tree.ext_cells();
  const size_t first = lost.first + (lost.salts ? 0 : (size_t)lost.row * EDEG * tree.sh.Kc);
  if (first >= cells) {
    fprintf(stderr, "a test's variable names lost cells that the tree does not have\n");
    exit(2);
  }
  u64 *at = (lost.salts ? tree.cw + tree.salt_row() * tree.sh.n : tree.ext) + first;
  CK(cudaMemset(at, 0, 8 * (lost.count < cells - first ? lost.count : cells - first)));
}
static void lose_engine_pads(void *pads, size_t cells, const LostPads &lost) {
  if (!lost.engine) return;
  if (lost.first >= cells) {
    fprintf(stderr, "a test's variable names lost cells that the pads do not have\n");
    exit(2);
  }
  CK(cudaMemset((u64 *)pads + lost.first, 0, 8 * (lost.count < cells - lost.first ? lost.count : cells - lost.first)));
}
// ROWS_DUMP=<an existing directory>; acts only if `nothing_is_hidden`. ROWS_DUMP_FILES=<numbers g,
// separated by commas> restricts the dumps to these.
static bool dumps(u32 g) {
  if (!TESTENV("ROWS_DUMP") || !nothing_is_hidden) return false;
  const char *list = TESTENV("ROWS_DUMP_FILES");
  if (!list) return true;
  while (*list) {
    char *end;
    const unsigned long file = strtoul(list, &end, 10);
    if (end != list && file == g) return true;
    list = *end ? end + 1 : end;
  }
  return false;
}
static void keep(const char *name, const void *on_the_host, size_t bytes) {
  char path[4096];
  snprintf(path, sizeof path, "%s/%s", TESTENV("ROWS_DUMP"), name);
  FILE *f = fopen(path, "wb");
  const bool written = f && fwrite(on_the_host, 1, bytes, f) == bytes;
  if (!f || fclose(f) || !written) {
    fprintf(stderr, "cp_commit: the dump %s cannot be written\n", path);
    exit(2);
  }
}
#endif
// Prints named 32-bit words, so that two committers' logs can be compared.
static void mark(const char *name, const u32 *words, int n) {
  if (!on()) return;
  printf("cert mark %-22s", name);
  for (int i = 0; i < n; i++) printf(" %08x", words[i]);
  printf("\n");
}
// Prints one line for a developer.
static void note(const char *format, ...) {
  if (!on()) return;
  va_list arguments;
  va_start(arguments, format);
  printf("cert note ");
  vprintf(format, arguments);
  printf("\n");
  va_end(arguments);
}
// If dumps(g), stores a hidden table as cert-<g, three digits>.<name>.bin in ROWS_DUMP's directory.
// keep() replaces an existing
// file of that name.
static void dump(u32 g, const char *name, const void *on_the_host, size_t bytes) {
  if (!dumps(g)) return;
  char whole[64];
  snprintf(whole, sizeof whole, "cert-%03u.%s.bin", g, name);
  keep(whole, on_the_host, bytes);
}
} // namespace cert_doors
