// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes the bytes of a leaf file (`ivk/leaf/<k>.bin`; `entry` in identifiers and messages) in the
// committer program `cp_commit`; `ivk` is the commitment's public key directory. A leaf file ties a
// leaf table to the commitment's anchor. The anchor ("the first table") is the commitment's first
// weight table; a leaf table (`copy`, "further") is one more table of the same weights with its own
// masks (each row's random entries).
// The file's parts (blocks/frame/check_entry.h): header | the leaf table's root | its record's
// answers | the hash chain (Fiat-Shamir transcript) of a LINK proof ("the link"). A record is a
// table's stored answers to samples of its rows. A row has three pads for each answer: random
// entries at the end of its message. The LINK proof is the argument that the rows of both tables
// agree in everything but their pads and masks. blk::make_entry(), in that header, runs the protocol. entry_make() plans
// GPU memory, wraps each table in an `HfOnceTable` and passes both to it. Its helpers are those of
// once_make() (prover/hash_commitment/hf_commit_once.cuh), which makes the key file (`ivk/key.bin`:
// the anchor's root and the committer's proofs about the anchor).
// Include this file where prover/hash_commitment/hf_commit.cu does: it needs headers included
// before it there, among them prover/weights_commitment/lent_memory.h for the `lent` pools.
#pragma once
#include "frame/check_entry.h"
#include "hash_commitment/hf_commit_once.cuh"

namespace hf {
// Appends a leaf file to `file`, which the caller passes empty (not checked). `c`: the leaf table
// as just committed. `anchor`: the anchor's mask key and the upper levels of its Merkle tree
// (`Committed`, `LinkAnchor`: defined in prover/hash_commitment/hf_commit_handed.h).
// `configured`: what blk::configure() derived from the configuration's public files; the LINK
// proof's plan, codes and shape and the configuration's digest come from it. `first_commitment`,
// `copy_commitment`: the anchor's and the leaf table's root, 32 bytes each. `first_record`,
// `copy_record`: their records, from make_the_record() (prover/hash_commitment/hf_commit.cu).
// `count_copy`: adds its argument, a number of columns, to the leaf table's column count; false if
// the table's masks do not cover them or the count cannot be stored.
// Returns 0: `file` holds the leaf file. 2, with `file` untouched: the inputs fail the first check
// below, there is too little GPU or host memory, or a column count cannot be raised. 4: `file` was
// written, but a self-check of a table's callbacks (`seams` in messages) has failed or a failure
// flag (prover/common/slot_flags.h) is set: its bytes must not be used. A root of another length, a
// LINK shape for another table and the other cases of hf_once_internal() below exit the process
// with code 4.
//
// Column counts. Every opened column of a table counts against a limit (`bears` below), and the
// LINK proof opens columns of both tables. The anchor's are added here to `isk/anchor/count.bin`
// (`isk`: the secret key directory), which also counts those of the key file's one-time test (the
// committer's proximity test on the anchor) and of every other LINK proof. make_entry() raises the
// leaf table's count first, then the anchor's, before it draws a random value or writes to `file`.
// If the anchor's cannot be raised, the leaf table's stays raised: too high at worst. Nothing here
// lowers a count if the run fails later.
//
// Private randomness (`coins`). The caller has derived the generator's key for this run
// (RunKeys::take_an_entrys()) from the seed, public values and 32 fresh system bytes (zeros under
// `-test-seed` or `-public-weights`), which are stored nowhere except under
// `-keep-for-the-audit`. The generator made here must be the only one under that key: a second one
// would give the same values again. No code enforces that in a run;
// prover/tests/coin_contexts_listed.py checks it on test runs.
//
// GPU memory. One slab (`HfOnceSlab`) in two pools, taken before make_entry() is called:
// `lent::borrowed` for both tables' folded rows (a table's rows summed with weights from a
// challenge), `lent::own` for the walks' buffers and the other items of `own_pool` below. A walk
// is one pass over all the weights: each table takes one to fold it and one or more to open its
// columns. The tables share no walk, though they hold the same weights: that costs time, and no
// byte of the file depends on it. At least HF_ONCE_ENGINE_ROOM bytes outside the slab are left
// free for the engine (`blk::EngineGpu`: the `Engine` of the library blocks/, on the GPU).
//
// No test in SUITE.md runs `cp_commit` at full size.
static int entry_make(const Committed &c, const LinkAnchor &anchor, const blk::Configured &configured, const std::vector<uint8_t> &first_commitment,
                      const blk::IndexRecord &first_record, const std::vector<uint8_t> &copy_commitment, const blk::IndexRecord &copy_record,
                      std::function<bool(uint64_t)> count_copy, blk::Writer &file) {
  using namespace rf;
  const HfParams &P = c.shape.P;
  const LayW &Y = c.shape.Y;
  if (!configured.refused().empty() || P.m0 != Y.m0 || P.k != Y.k || !c.key || !c.key_again || !anchor.key || !anchor.key_again || !c.kept.heap || !anchor.kept.heap) {
    fprintf(stderr, "the entry: the configured instance, the parameters, the layout and what the two tables' phases A have left do not fit each other\n");
    return 2;
  }
  const blk::LinkShape &S = configured.sized().link.S;
  // The LINK proof compares the two tables' rows only up to the first pad: each table has its own
  // pads.
  if (!blk::link_same_table_shape(S.table, hf_index_shape(P)) || S.cut() != (blk::u64)P.pad_entry(0)) hf_once_internal("the configured link speaks of another table than the committed one");
  if (first_commitment.size() != blk::DIGEST_BYTES || copy_commitment.size() != blk::DIGEST_BYTES) hf_once_internal("a table's commitment is its root: 32 bytes");
  if (live_flags()) hf_once_internal("a failure flag is set that belongs to no part of the entry");

  auto last = std::chrono::steady_clock::now();
  auto lap = [&last](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    printf("the entry: %s: %.1f s\n", stage, std::chrono::duration<double>(now - last).count());
    fflush(stdout);
    last = now;
  };

  // What the process keeps until it exits (hf_once_ready()) and the weight walker's buffers are
  // allocated from the CUDA driver first, because the slab is sized by the memory then free.
  c.weights.release();
  hf_once_ready(Y);
  c.weights.reserve(::hf::WeightWalker::WHOLE_UNITS);
  // The slab is declared before the engine so that it is destroyed after it. In this translation
  // unit cudaMalloc and cudaFree are macros (prover/weights_commitment/lent_memory.h): while the
  // slab is in use, anything that the engine allocates by these names comes from the slab's pools,
  // which must still exist when the engine is destroyed and frees it. The `HfOncePoolsBack`,
  // declared first, is destroyed last: it returns the engine's pool's unused memory to the driver.
  const HfOncePoolsBack pools_go_back_at_every_return;
  HfOnceSlab slab;
  blk::EngineGpu engine;
  // Returns to the driver the freed memory that the engine's pool keeps (tens of GB at full size,
  // which the driver does not report as free), before anything is sized by the free memory. The
  // pool belongs to the process, not to this engine.
  engine.release_work_space();
  const std::unique_ptr<blk::CardStreamCoins> coins = ::hz::lng::engine_coins_once(::hz::lng::ENGINE_ENTRY);
  const blk::NonceSearch search = [&engine](const blk::ChainPrefix &prefix, unsigned bits) { return engine.search_nonce(prefix, bits); };

  // ---- the two tables: [0] the anchor, [1] the leaf table
  const uint64_t columns[2] = {(uint64_t)S.columns_first, (uint64_t)S.columns_further};
  const int kept_level[2] = {anchor.kept.s, c.kept.s};
  HfOpen mode[2]; // for `hf_side_slab_bytes` below; only the `HfShown` part and `once` are set
  for (int t = 0; t < 2; t++) static_cast<HfShown &>(mode[t]) = hf_shown(P, columns[t]), mode[t].once = true;
  // A `UnitW` (prover/common/weight_layout.h) is one layer's tensors of one kind, or the embedding
  // or the output head.
  HfOncePanels panels(c); // the panel: a buffer for one `UnitW` of weights, used by all walks
  // Both folded rows, each as the LINK proof's oracle message (rows sent as Merkle roots) holds it,
  // and a staging buffer for each.
  const size_t for_tables = 2 * HfOnceTable::lent_bytes(S.rows, configured.sized().entry_codes.base[S.code].ell);
  // u: the coefficients of a table's folded row. It is kept until the table's last walk is over:
  // the row's copy in `lent::borrowed` is released for the walks that open columns and then filled
  // again from u. Above HF_ONCE_U_ON_THE_GPU_MOST bytes u is in page-locked host memory, otherwise
  // in `lent::own`.
  const bool large = hf_side_u_elements(P) * sizeof(Ext) > HF_ONCE_U_ON_THE_GPU_MOST;
  HfOnceHostU host[2];
  if (large)
    for (int t = 0; t < 2; t++)
      if (int rc = host[t].take(hf_side_u_elements(P))) return rc;
  // `held_bytes`: what `lent::own` holds besides the walks' buffers and the reserve
  // HF_ONCE_SLAB_SPARE: the panel and, for each table, one extension-field element per row of the
  // weight matrix, plus 1 MiB to spare.
  const size_t held_bytes = 2 * HfOnceTable::own_bytes(P) + panels.bytes() + (1 << 20);
  const size_t may = HfOnceSlab::most(c.O.gpu_memory, HF_ONCE_ENGINE_ROOM);
  if (!may) { // `take` would read a cap of 0 as no limit
    fprintf(stderr, "the entry: the GPU has too little free memory: nothing is left for the link beside 2 GiB and the %zu bytes that the engine is left, whatever the size of the model\n", HF_ONCE_ENGINE_ROOM);
    return 2;
  }
  const size_t room = may - std::min(may, for_tables + held_bytes + HF_ONCE_SLAB_SPARE);
  // fibres[t]: how many of table t's drawn columns one walk serves at most: all of them, then
  // halved until both tables' needs, counted together, fit or both numbers are 1. The sum errs on
  // the safe side: rf::hf_side_make_columns() (hf_open_side.cuh) frees the buffers sized by
  // fibres[t] before it returns, and the anchor's columns are opened before the leaf table's.
  int fibres[2] = {(int)columns[0], (int)columns[1]};
  auto for_walks = [&] { return (hf_side_slab_bytes(P, Y, mode[0], kept_level[0], fibres[0], !large) + hf_side_slab_bytes(P, Y, mode[1], kept_level[1], fibres[1], !large)) * 9 / 8; };
  while ((fibres[0] > 1 || fibres[1] > 1) && for_walks() > room) {
    const int t = fibres[1] >= fibres[0] ? 1 : 0; // the larger one gives way
    fibres[t] = (fibres[t] + 1) / 2;
  }
  if (const char *word = TESTENV("ROWS_ONCE_GROUP")) // a switch for tests; no byte of a file depends on it
    for (int t = 0; t < 2; t++) fibres[t] = std::max(1, std::min(atoi(word), (int)columns[t]));
  // The slab takes only what was computed above: the engine allocates from the driver as it goes,
  // through its pool (cudaMallocFromPoolAsync(), which the macros named above do not replace), and
  // no formula here gives its need.
  const size_t own_pool = held_bytes + for_walks() + HF_ONCE_SLAB_SPARE;
  if (int rc = slab.take(for_tables, own_pool, std::min(may, (for_tables + 511) / 512 * 512 + (own_pool + 511) / 512 * 512))) return rc; // too little memory: `take` has printed the numbers
  struct ClosePanels { // frees the panel before the slab, which contains it, is freed
    HfOncePanels &panels;
    ~ClosePanels() { panels.close(); }
  } close_panels_at_every_return{panels};
  panels.open();
  once_pool_peak(blk::card_pool(), true); // resets the pool's peak, which the last line prints
  printf("the entry: GPU memory in bytes: the slab %zu, of which %zu for the two folded rows; the panel %zu; host memory for the folded vectors: %zu and %zu; %llu + %llu columns, %d and %d to a walk\n",
         slab.bytes, for_tables, panels.bytes(), host[0].bytes, host[1].bytes, (unsigned long long)columns[0], (unsigned long long)columns[1], fibres[0], fibres[1]);
  lap("set-up");

  HfHeld held[2];
  for (int t = 0; t < 2; t++) {
    held[t].seed = t ? c.key : anchor.key, held[t].key_again = t ? c.key_again : anchor.key_again;
    held[t].nodes = t ? c.kept.heap : anchor.kept.heap, held[t].s = kept_level[t];
    held[t].u_host = host[t].u, held[t].u_host_elements = host[t].u ? hf_side_u_elements(P) : 0;
  }
  // The columns of one weight table that may be opened in all: what a row's masks are to hide.
  const uint64_t bears = blk::hiding_positions(S.table.code);
  const std::string isk = c.O.isk;
  // The count's directory is opened relative to the descriptor that the run keeps of its secret
  // working directory (making::via(), hf_commit_making.h). So whoever renames that directory during
  // the run, or puts a symbolic link under its name, cannot redirect the count to another one.
  // Not checked here: if the run kept no such descriptor, via() would return the whole path.
  const std::string anchor_folder = ::cp::isk_anchor_dir(isk);
  auto count_first = [isk, anchor_folder, bears](uint64_t shown) {
    const ::cp::Counts k = ::cp::count_first_table_columns(making::via(anchor_folder).fd, making::via(anchor_folder).name, shown, bears);
    if (k.say == ::cp::Counts::FAULT)
      fprintf(stderr, "the entry: the first table's count (%s) cannot be raised: %s\n", ::cp::isk_anchor_count(isk).c_str(), k.error ? strerror(k.error) : "it is not the plain file that it must be");
    if (k.say == ::cp::Counts::SPENT)
      fprintf(stderr, "the entry: the first table has shown %llu columns, and its random entries bear %llu in all: %llu more are refused\n", (unsigned long long)k.begun, (unsigned long long)bears,
              (unsigned long long)shown);
    return k.say == ::cp::Counts::ROOM;
  };
  // Each table has its own rf::HfWalk (one walk: hf_walk.h). It first tells the hook for a test's
  // malicious committer (commit_lie.cuh; in this repository walking() is empty, passes() false)
  // which table is read and why: a table's first walk folds it, later ones read columns (`fibres`).
  // LINKFOLD and LINKFIBRES let a test alter what is read of the anchor for a leaf file only.
  int walks_of[2] = {0, 0};
  const rf::HfWalk walk_first = [&panels, &walks_of](const HfUnits &units) {
    commit_lie::walking(0, walks_of[0]++ ? commit_lie::LINKFIBRES : commit_lie::LINKFOLD);
    panels.walk(units);
  };
  const rf::HfWalk walk_copy = [&panels, &walks_of](const HfUnits &units) {
    commit_lie::walking(1, walks_of[1]++ ? commit_lie::FIBRES : commit_lie::FOLD);
    panels.walk(units);
  };
  // `HfOnceTable` (hf_keeper_once.cuh): a table as the library's `blk::IndexHolder`, which folds
  // it, opens its columns and counts them. At every return the destructors release the folded row,
  // and erase a u on the GPU and the opened columns' words in host memory (rf::hf_side_forget()).
  // `true`: print each step's time.
  HfOnceTable first(P, Y, columns[0], walk_first, held[0], fibres[0], true, first_commitment.data(), first_commitment.size(), first_record, count_first, engine);
  HfOnceTable copy(P, Y, columns[1], walk_copy, held[1], fibres[1], true, copy_commitment.data(), copy_commitment.size(), copy_record, std::move(count_copy), engine);

  // After the last walk of both tables, what only the walks needed goes back to the driver, for
  // the engine's proximity tests.
  int tables_walking = 2;
  const std::function<void()> a_tables_walks_are_over = [&] {
    if (--tables_walking) return;
    panels.close(); // also frees the weight walker's buffers
    slab.the_walks_are_over();
    size_t free_bytes = 0, total = 0;
    CK(cudaMemGetInfo(&free_bytes, &total));
    printf("the entry: the walks are over: %zu bytes of GPU memory are free\n", free_bytes);
    fflush(stdout);
  };
  first.when_the_walks_are_over(a_tables_walks_are_over), copy.when_the_walks_are_over(a_tables_walks_are_over);
  if (!blk::make_entry(configured, first.holder(), copy.holder(), engine, *coins, search, file)) {
    fprintf(stderr, "the entry: refused: a table's masks do not cover the link's columns (the copy's count, or the first table's lasting count, says no)\n");
    return 2;
  }
  ::lent::borrow = false;
  lap("the link");
  char pools[160]; // for the last line; here, so that its CUDA call precedes the flags' read below
  snprintf(pools, sizeof pools, "the engine's memory pool held %zu", once_pool_peak(blk::card_pool(), false));
  for (const HfOnceTable *t : {&first, &copy})
    if (t->failed() != OWN_GOOD && !commit_lie::passes("seams")) {
      fprintf(stderr, "the entry: internal error: an own test of the opening's seams has failed (%d, the %s table)\n", t->failed(), t == &first ? "first" : "further");
      return 4;
    }
  if (int rc = hf_once_flags("the entry")) return rc; // reads the CUDA status and the failure flags
  // No kernel launch may follow that read in this function, which would not see its failure.
  // Before the bytes leave this function, EngineCoins::hold() (prover/hiding/engine_coins.cuh)
  // compares samples of the random values drawn with a second implementation's and ends the
  // process if one differs. hold() launches no kernel: it reads GPU memory by copies to the host.
  if (auto *own = dynamic_cast<::hz::lng::EngineCoins *>(coins.get())) {
    if (!own->hold()) hf_once_internal("no random value of the entry was sampled: the coins' own test by value has seen nothing"); // an entry draws masks, pads and salts: none sampled means that the test looked at nothing
  } else
    hf_once_internal("the entry's coins are not the program's own generator");
  printf("the entry: %zu bytes; %llu + %llu columns; %d walks over the weights; GPU memory at the peak in lent's pools: %zu bytes and %zu; beside the slab, at the most, %s; the least that is left to the engine: %zu\n",
         file.bytes().size(), (unsigned long long)columns[0], (unsigned long long)columns[1], panels.walks, ::lent::own.peak, ::lent::borrowed.peak, pools, HF_ONCE_ENGINE_ROOM);
  return 0;
}
} // namespace hf
