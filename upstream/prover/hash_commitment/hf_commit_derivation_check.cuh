// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// [own-test-second-text] A self-check of the committer program cp_commit. The two messages of
// stop_if_noted() print this name.
// Values derived from the seed (the provider's 32 secret bytes) are to hide the weights in
// cp_commit's files: each weight table's masks, its tree leaves' salts and its record's pads, and
// the private randomness (`coins` in identifiers) of the key file and of each leaf file (`entry` in
// identifiers). No verifier can tell whether such a value is what the derivation rule gives.
// cp_commit's start-up self-checks run under keys written in the source; its other comparisons by
// value derive the value again with the programs' code and keys, so an error that both derivations
// share passes them. This file compares masks, salts, pads and the keys of that randomness, before
// a byte of the files built with them leaves the process, with what prover/second_writing/ ("the
// library", namespace `coins`) derives: a second implementation of the derivation rule, which
// starts from the seed and includes no file of the programs; its ChaCha20 and BLAKE2b are adapted
// from the same reference code as theirs. A difference ends cp_commit with exit code 4.
// Terms and limits: above `namespace derivation_check`.
//
// Include this file first. Shared headers call slab_masks(), table_salts() and table_pads() through
// the macro HELD_TO_THE_DERIVATION_CHECK, which prover/common/derivation_check_seam.h defines as empty
// if it is not yet defined.
#pragma once
#include "common/stack_scrub.h"
#include "second_writing/mask_generator.h"
#include "second_writing/schedule_keys.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <system_error>
#include <thread>
#include <vector>

#define HELD_TO_THE_DERIVATION_CHECK(call) ::derivation_check::call;

// A weight table is all the weights as a matrix of rows, each encoded together with random masks,
// under a Merkle tree with a salt in each leaf. The anchor is a commitment's first weight table; a
// leaf table (`copy` in some identifiers) is a further table of the same weights with its own
// masks. A table's masks, salts and pads are derived from its making key (`mask_seed`). A run is
// the creation of the key file (`ivk/key.bin`) or of one leaf file (`ivk/leaf/<k>.bin`); its
// private randomness is derived from its process key. The library derives every key itself
// (prover/second_writing/schedule_keys.h), from this file's copy of the seed.
//
// Pads: table_pads() compares every pad of every row, each
// time a part of cp_commit has generated them or checks them again, and counts the comparisons.
// the_table_ends() requires two while a table is built. a_run_ends() requires only one for each
// table that the run opens, and there the comparisons for the record's answers count as well.
//
// No comparison in this file covers: once a table is built, whether the pads that a caller passes
// belong to the table that it works on (they are compared under whichever table's key the caller
// names); the seed, which the library gets from cp_commit's memory
// (docs/limits-all-rows.tsv, row H3.20), and the 32 bytes from the operating system that enter a
// key; the configuration
// digest (a hash of the public configuration files), the shape words, a leaf table's number, the
// commitment ID, a table's name (root, configuration digest, shape words) and a leaf table's stored
// answers, which cp_commit passes to both derivations; an error that the library and the programs
// share; a value after its comparison.
// The library is not independent of the programs.
// The library includes no file of the programs; nothing else here shows how far it is separate
// from them. What it is known to share with them is listed in
// prover/second_writing/README.md, "How far it is independent of the programs".
// Its key schedule follows the table at the top of prover/common/coin_keys.h: a misreading of that
// table that both sides share passes. Its ChaCha20 block function and BLAKE2b (compression, feeding and finishing)
// are adapted from the same reference code as the programs': D. J. Bernstein's chacha-merged.c, and
// blake2b-ref.c of the BLAKE2 reference source code package (sources:
// THIRD_PARTY.md, "ChaCha20" and "BLAKE2b"). So the comparisons check the key schedule, the words
// that enter each call, the order of calls and how a block becomes masks or a salt. In the two
// functions they catch only a slip on one side: an error of the reference code, or a slip in
// adapting it that both sides share, passes. Known answers check the library's functions:
// prover/tests/coins_rfc8439_test.cpp runs its block function (the duplicate in
// prover/tests/coins_second_text/) on RFC 8439's test vectors, and
// prover/tests/coin_keys_test.cpp compares keys derived through its BLAKE2b with stored values
// computed with Python's hashlib.
// The code that passes the library the seed, the public inputs and the values (this file,
// prover/hash_commitment/hf_commit_keys.h, the macro's callers) belongs to cp_commit and has no
// second implementation.
namespace derivation_check {
enum Table { NO_TABLE, ANCHOR, LEAF };
struct State {
  coins::Key seed{}, mask_seed[3] = {}; // making keys, by Table; [LEAF]: the current leaf table's
  bool seeded = false;
  coins::schedule::Octets configuration;   // the configuration digest, 32 bytes: public
  std::vector<std::uint64_t> shape;        // the shape words in a weight table's name: public
  unsigned log_rows = 0;                   // a weight table has 2^log_rows rows,
  std::uint64_t masks_a_row = 0;           // each with so many masks: public
  std::uint64_t salts_held[3] = {0, 0, 0}; // salts compared in the current run, by table
  // Comparisons of all pads of a table. [NO_TABLE]: of the table being built; the other two: since
  // the last run's end, by table.
  std::uint64_t pads_compared[3] = {0, 0, 0};
  Table table = NO_TABLE;                  // the table being built
  std::uint64_t rows_held = 0;             // its rows below this index were compared in full
  std::uint64_t *row = nullptr;            // host buffer for one row of masks
  std::size_t row_bytes = 0;
  std::atomic<bool> differs{false}, not_by_the_list{false};
  std::mutex lock; // guards salts_held and pads_compared's counts by table
};
static State &state() {
  static State &s = *[] {
    State *made = new State; // never destroyed, so that the exit handler can wipe it
    if (mlock(made, sizeof *made)) // the same warning as `lock_in_memory` in hf_commit.cu
      fprintf(stderr, "cp_commit: warning: a key's memory cannot be locked (%s): the system may write it to its swap space. Raise the limit on locked memory (ulimit -l)\n",
              strerror(errno));
    return made;
  }();
  return s;
}
static void forget_the_table() {
  State &s = state();
  if (s.row) explicit_bzero(s.row, s.row_bytes);
  s.table = NO_TABLE, s.rows_held = 0, s.pads_compared[NO_TABLE] = 0;
}
// Erases every secret that this file stores. stop() calls it, and so does wipe_the_keys()
// (prover/hash_commitment/hf_commit.cu), at cp_commit's end and from its exit handler. The file
// stores the seed and the anchor's making key until this call, a leaf table's making key until
// an_entry_ends(), and one row of masks until a table's last row is compared. `State` is locked in
// memory; if mlock() fails, cp_commit warns and goes on. cp_commit disables core dumps, and zeroes
// heap memory that the library frees (prover/common/zeroing_heap.h).
static void wipe() {
  State &s = state();
  forget_the_table();
  coins::wipe(s.seed);
  for (coins::Key &k : s.mask_seed) coins::wipe(k);
  s.seeded = false;
}
[[noreturn]] static void stop(const char *sentence) {
  wipe();
  fputs(sentence, stderr);
  exit(4);
}
// Exits with code 4 if a flag is set. `differs`: a compared value is not the library's, or a
// row of masks could not be copied from the GPU or derived by the library.
// `not_by_the_list`: the calls were not the expected ones: out of order, with another row length,
// or with a comparison missing. The messages print no count, index or value.
static void stop_if_noted() {
  if (state().differs)
    stop("cp_commit: internal error (own-test-second-text): a self-check failed: random masking values of this run are not what a "
         "second, separately written form of their rule (prover/second_writing/) derives from the seed. This is a fault of the "
         "program or of the machine, not of the input. Nothing of the step in hand, the anchor or this leaf, is written or "
         "published. What earlier steps or calls have published stays; while a leaf is added, the anchor's own values may be "
         "among those that differ. Run cp_commit again, on another GPU if there is one. If it happens again while leaves are "
         "added, make a new commitment. Report this line\n");
  if (state().not_by_the_list)
    stop("cp_commit: internal error (own-test-second-text): a self-check failed: the random masking values that this step has "
         "compared with a second, separately written form of their rule (prover/second_writing/) are not exactly those that the "
         "rule lists. This is a fault of the program, not of the input. Nothing of the step in hand, the anchor or this leaf, is "
         "written or published; what earlier steps or calls have published stays. Report this line\n");
}
// by_the_library() runs f, which calls the library, in a separate stack frame, then zeroes the
// STACK_SCRUBBED bytes of stack below its own frame (prover/common/stack_scrub.h): the library
// leaves keys in its local arrays. Limit: if f throws (out of memory), the stack is not zeroed.
template <class F> __attribute__((noinline)) static void a_frame_of_its_own(F &f) { f(); }
template <class F> static void by_the_library(F f) {
  a_frame_of_its_own(f);
  scrub_the_stack_below();
}
static coins::Key key_of(const std::uint8_t bytes[32]) {
  std::uint8_t b[32];
  memcpy(b, bytes, 32);
  const coins::Key k = coins::key_from_bytes(b);
  explicit_bzero(b, sizeof b);
  return k;
}

// ---- a table's size, and the seed ----
// The size of every weight table of this commitment, as cp_commit has taken it from the parameter
// file.
static void a_table_has(unsigned log_rows, std::uint64_t masks_a_row) {
  State &s = state();
  s.log_rows = log_rows, s.masks_a_row = masks_a_row;
}
// words: the seed as cp_commit has it in memory; the self-check [own-test-key] (hf_commit_keys.h)
// relates that copy to the seed's file when the key file is created. configuration: the
// configuration digest as eight 32-bit words.
static void the_seed_is(const std::uint32_t words[8], const std::uint32_t configuration[8], const std::uint64_t *shape, std::size_t shape_words) {
  State &s = state();
  for (int i = 0; i < 8; i++) s.seed[i] = words[i];
  s.configuration.clear();
  for (int i = 0; i < 32; i++) s.configuration.push_back((std::uint8_t)(configuration[i / 4] >> (8 * (i % 4))));
  s.shape.assign(shape, shape + shape_words);
  by_the_library([&] { s.mask_seed[ANCHOR] = coins::schedule::first_table(s.seed, s.configuration, s.shape); });
  s.seeded = true;
}
// ---- a table's masks and salts ----
// For each table, in this order: the_anchors_table_begins() or a_leafs_table_begins(); slab_masks()
// until all rows are compared; the_table_ends(). The first two exit unless the_seed_is() has run
// and no other table is being built.
static void the_anchors_table_begins() {
  State &s = state();
  if (!s.seeded || s.table != NO_TABLE) s.not_by_the_list = true, stop_if_noted();
  s.table = ANCHOR, s.rows_held = 0;
}
// fresh: the 32 bytes that enter leaf table k's making key: from the operating system, or zeros
// under -test-seed (test builds) or -public-weights (production builds).
static void a_leafs_table_begins(std::uint32_t k, const std::uint8_t fresh[32]) {
  State &s = state();
  if (!s.seeded || s.table != NO_TABLE) s.not_by_the_list = true, stop_if_noted();
  by_the_library([&] {
    coins::Key f = key_of(fresh);
    s.mask_seed[LEAF] = coins::schedule::copy_table(s.seed, k, f, s.configuration, s.shape);
    coins::wipe(f);
  });
  s.table = LEAF, s.rows_held = 0;
}
// The bitwise OR of part(from, to) over shares of [0, n), on at most 64 host threads.
template <class Part> static std::uint64_t on_threads(std::uint64_t n, Part part) {
  const std::uint64_t threads = std::max<std::uint64_t>(1, std::min<std::uint64_t>({std::thread::hardware_concurrency(), 64, n >> 12}));
  const std::uint64_t share = (n + threads - 1) / threads;
  std::vector<std::uint64_t> sums(threads, 0);
  std::vector<std::thread> others;
  for (std::uint64_t t = 1; t < threads; t++) {
    auto a_share = [&, t] { sums[t] = part(std::min(n, t * share), std::min(n, (t + 1) * share)); };
    try {
      others.emplace_back(a_share);
    } catch (const std::system_error &) { // no thread available: the calling thread does the share
      a_share();
    }
  }
  sums[0] = part(0, std::min(n, share));
  for (std::thread &other : others) other.join();
  std::uint64_t sum = 0;
  for (std::uint64_t x : sums) sum |= x;
  return sum;
}
// Compares the masks of rows r0 .. r0 + rows - 1 (a slab) of the table being built, in the GPU
// buffer from which the rows are encoded: on_the_gpu[r * stride + j] must be mask j < nm of row
// r0 + r. The work that wrote them must have been queued on st. Call it on the thread that builds
// the table, with the slabs in row order and each row once. Exits at a difference, and with code 2
// if the row buffer cannot be allocated and page-locked. threshold: a block of eight 64-bit values
// is good, and gives eight masks, if all are below it (coins::masks()); read only by a test build
// (HC_MASK_LOW_LIMIT). Not checked: nm >= 1 (with 0 no row buffer is allocated).
//
// PartMaker::parts() (hf_commit_table.cuh) passes every mask of every row, but only in its first
// call for a table; each call is a turn. Not covered: the masks of the later turns, each of which
// derives all masks again by the same kernel launches.
static void slab_masks(std::uint32_t r0, std::uint32_t rows, std::uint64_t nm, std::uint64_t threshold,
                       const std::uint64_t *on_the_gpu, std::size_t stride, cudaStream_t st) {
  State &s = state();
  if (s.table == NO_TABLE || r0 != s.rows_held || nm != s.masks_a_row) s.not_by_the_list = true, stop_if_noted();
  const std::size_t bytes = (nm * 8 + 4095) / 4096 * 4096;
  if (s.row_bytes < bytes) { // once per process: every table of a commitment has the same nm
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (s.row || p == MAP_FAILED || cudaHostRegister(p, bytes, cudaHostRegisterDefault) != cudaSuccess) {
      fprintf(stderr, "cp_commit: cannot allocate and page-lock %zu bytes of host memory for a row of masks. Free host memory, or raise the "
                      "limit on locked memory (ulimit -l)\n",
              bytes);
      exit(2);
    }
    (void)madvise(p, bytes, MADV_DONTDUMP);
    s.row = (std::uint64_t *)p, s.row_bytes = bytes;
  }
#if !defined(HC_MASK_LOW_LIMIT) || !HC_MASK_LOW_LIMIT
  threshold = coins::kTableThreshold; // the library's constant: the argument is not read
#endif
  const coins::Key &seed = s.mask_seed[s.table];
  for (std::uint32_t r = 0; r < rows; r++) {
    memset(s.row, 0xFF, 64); // all ones is no mask: so a copy that never arrived is noticed
    if (cudaMemcpyAsync(s.row, on_the_gpu + (std::size_t)r * stride, nm * 8, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess)
      s.differs = true;
    const std::uint64_t *came = s.row;
    if (on_threads((nm + 7) / 8, [&](std::uint64_t from, std::uint64_t to) {
          std::uint64_t difference = 0;
          by_the_library([&] {
            try {
              for (std::uint64_t g = from; g < to; g++) {
                const std::array<std::uint64_t, 8> m = coins::masks(seed, r0 + r, (coins::Word)g, threshold);
                for (unsigned i = 0; i < 8 && 8 * g + i < nm; i++) difference |= m[i] ^ came[8 * g + i];
              }
            } catch (...) { // the library finds no good block
              difference = 1;
            }
          });
          return difference;
        }))
      s.differs = true;
    stop_if_noted();
  }
  s.rows_held += rows;
  if (s.rows_held == (std::uint64_t)1 << s.log_rows) explicit_bzero(s.row, s.row_bytes); // the later turns bring nothing here
}
// How many answers a table's record has for each row: the fifth shape word.
static std::uint64_t stored_answers() { return state().shape.size() > 4 ? state().shape[4] : 0; }
// Call it when the table's tree is built. Exits unless the masks of all its rows were compared and
// all its pads twice.
static void the_table_ends() {
  State &s = state();
  if (s.table == NO_TABLE || s.rows_held != (std::uint64_t)1 << s.log_rows) s.not_by_the_list = true;
  if (s.pads_compared[NO_TABLE] < 2) s.not_by_the_list = true;
  stop_if_noted();
  forget_the_table();
}
// Returns which of the library's two making keys equals key_again, cp_commit's second copy of a
// table's key. Exits if neither does.
static int the_table_under(const std::uint8_t *key_again) {
  State &s = state();
  if (!key_again) s.not_by_the_list = true, stop_if_noted();
  int table = NO_TABLE;
  by_the_library([&] {
    coins::Key key = key_of(key_again);
    for (int t : {ANCHOR, LEAF}) {
      coins::Word difference = 0;
      for (int i = 0; i < 8; i++) difference |= key[i] ^ s.mask_seed[t][i];
      if (!difference) table = t;
    }
    coins::wipe(key);
  });
  if (table == NO_TABLE) s.differs = true, stop_if_noted(); // `key_again` is neither making key
  return table;
}
// Compares all pads of a table: pads[r * count + j] must be pad j < count of row r < rows of the
// table whose key's second copy is key_again. threshold: as for slab_masks(). At a difference it
// sets *differs and returns, so that the caller can erase its buffers before it ends the run or
// withholds what it has computed. It exits itself on a call before the seed, on a `rows` or `count`
// other than this file's own, and on a key that is neither table's. pads_own_test() (hf_pads.cuh)
// calls it.
static void table_pads(const std::uint8_t *key_again, const std::uint64_t *pads, std::uint32_t rows, unsigned count, std::uint64_t threshold, bool *differs) {
  State &s = state();
  if (!s.seeded || rows != (std::uint64_t)1 << s.log_rows || count != 3 * stored_answers()) s.not_by_the_list = true, stop_if_noted();
  const int table = the_table_under(key_again);
  if (s.table != NO_TABLE && table != s.table) { // while a table is built: only that table's pads
    *differs = true;
    return;
  }
#if !defined(HC_MASK_LOW_LIMIT) || !HC_MASK_LOW_LIMIT
  threshold = coins::kTableThreshold; // the library's own constant: the argument is ignored
#endif
  std::uint64_t difference = 0;
  by_the_library([&] {
    try {
      for (std::uint32_t r = 0; r < rows; r++)
        for (unsigned g = 0; 8 * g < count; g++) {
          const std::array<std::uint64_t, 8> p = coins::pads(s.mask_seed[table], r, g, threshold);
          for (unsigned i = 0; i < 8 && 8 * g + i < count; i++) difference |= p[i] ^ pads[(std::size_t)r * count + 8 * g + i];
        }
    } catch (...) { // the library finds no good block
      difference = 1;
    }
  });
  if (difference) *differs = true;
  std::lock_guard<std::mutex> hold(s.lock);
  s.pads_compared[s.table != NO_TABLE ? NO_TABLE : table]++;
}
// Compares a sample of a table's salts: those of the groups of adjacent columns that a run
// recomputes in order to open columns (fibres_taken(), hf_tree.cuh). cp_commit compares each
// group's recomputed node with the stored one, which relates the Merkle tree's salts at those
// columns to the salts compared here. Not covered: the salts of all other columns.
// salts: five 64-bit words for each of `count` tree leaves, as they were hashed. position(j): tree
// leaf j's column index. key_again: cp_commit's second copy of the table's key; it must equal one
// of the library's two making keys. Exits at a difference. Both pointers: host memory.
// salts_own_test() (hf_mask_own.cuh) calls it.
template <class Position>
static void table_salts(const std::uint8_t *key_again, const std::uint64_t *salts, std::size_t count, Position position) {
  State &s = state();
  if (!s.seeded) return; // before `the_seed_is`: nothing is compared
  const int table = the_table_under(key_again);
  std::uint64_t difference = 0;
  by_the_library([&] {
    for (std::size_t j = 0; j < count; j++) {
      const std::array<std::uint64_t, 5> salt = coins::table_salt(s.mask_seed[table], position(j));
      for (int x = 0; x < 5; x++) difference |= salt[x] ^ salts[5 * j + x];
    }
  });
  if (difference) s.differs = true;
  {
    std::lock_guard<std::mutex> hold(s.lock);
    s.salts_held[table] += count;
  }
  stop_if_noted(); // at once: the opening's words are written after this function returns
}

// ---- the keys of cp_commit's two kinds of run, and the end of each ----
// Exits unless in_hand, a run's process key as cp_commit has derived it, equals derived(), the
// library's. `RunKeys` (prover/hash_commitment/hf_commit_keys.h) calls the two functions below
// before private randomness is drawn under the key. Not covered: the values derived under a process
// key; EngineCoins::hold() (prover/hiding/engine_coins.cuh) compares a sample of them with the
// library's generator.
template <class Derived> static void a_runs_key_is(const std::uint8_t in_hand[32], Derived derived) {
  State &s = state();
  if (!s.seeded) s.not_by_the_list = true, stop_if_noted();
  by_the_library([&] {
    coins::Key theirs = key_of(in_hand), mine = derived();
    coins::Word difference = 0;
    for (int i = 0; i < 8; i++) difference |= theirs[i] ^ mine[i];
    if (difference) s.differs = true;
    coins::wipe(theirs), coins::wipe(mine);
  });
  stop_if_noted();
}
// of_the_system: the run's 32 bytes from the operating system (zeros under -test-seed or
// -public-weights). id: the commitment ID, 32 bytes.
// an_entrys_key_is() also takes leaf table k's name, and answers: the table's stored answers to
// samples of its rows.
static void the_key_files_key_is(const std::uint8_t in_hand[32], const std::uint8_t of_the_system[32], const std::uint8_t id[32]) {
  a_runs_key_is(in_hand, [&] {
    coins::Key f = key_of(of_the_system);
    const coins::Key k = coins::schedule::key_file(state().seed, f, coins::schedule::Octets(id, id + 32));
    coins::wipe(f);
    return k;
  });
}
static void an_entrys_key_is(const std::uint8_t in_hand[32], std::uint32_t k, const std::uint8_t of_the_system[32], const std::uint8_t id[32],
                             const coins::schedule::Octets &name, const coins::schedule::Octets &answers) {
  a_runs_key_is(in_hand, [&] {
    coins::Key f = key_of(of_the_system);
    const coins::Key key = coins::schedule::entry(state().seed, k, f, coins::schedule::Octets(id, id + 32), name, answers);
    coins::wipe(f);
    return key;
  });
}
// Call it when a run's last message is built, before a byte of its file leaves the process. Exits
// unless table_salts() has compared a salt of every table that the run opens: the anchor for the
// key file, the anchor and the leaf table for a leaf file. The sample's size is not checked
// otherwise, and nothing verifies that a_run_ends() has run.
static void a_run_ends(bool a_copy_too) {
  State &s = state();
  {
    std::lock_guard<std::mutex> hold(s.lock);
    if (!s.salts_held[ANCHOR] || (a_copy_too && !s.salts_held[LEAF])) s.not_by_the_list = true;
    if (!s.pads_compared[ANCHOR] || (a_copy_too && !s.pads_compared[LEAF])) s.not_by_the_list = true;
    s.salts_held[ANCHOR] = s.salts_held[LEAF] = s.pads_compared[ANCHOR] = s.pads_compared[LEAF] = 0;
  }
  stop_if_noted();
}
static void the_key_file_ends() { a_run_ends(false); }
// The leaf table is built and its leaf file's openings are done: its making key is erased.
static void an_entry_ends() {
  a_run_ends(true);
  coins::wipe(state().mask_seed[LEAF]);
}
} // namespace derivation_check
