// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Phase A of the committer program cp_commit (phases: hf_commit.cu): the Merkle tree over a weight
// table T. T is never stored: make_table() hashes it part by part. T has 8 R n bytes: 7 * 10^13,
// computed for R = 2^11, n = 2^32 (the `table` line of prover/examples/target_full_f2/policy.txt).
// M holds the weights as R rows: a row's KD coefficients are l cells (signed bytes, the weights'
// limbs), nm masks and zeros. Row r of T is the codeword of row r of M: n = 2^fln symbols (fln is
// HfParams::ln). Tree leaf I hashes T's column at position I and a salt. Levels count from the tree
// leaves, node(l, y) is node y of level l, and part h is the 2^lp positions under node(lp, h).
// Masks, salts and pads are derived from `key`, the table's secret key.
//
// On a part the factors f_lp, f_(lp + 1), .. of the code's basis are constant (hf_code.cuh, which
// calls a part a run), so a row's KD coefficients are first folded to 2^lp sums,
//   V[i] = the sum over x of high[x] M[r][x 2^lp + i]   (high: run_high(), products of factors),
// and the transform's last lp layers turn V into the row's symbols on the part. A tree leaf's hash
// (`chain` in identifiers) takes in a 16-row slab per step, so parts() takes slabs in rising order:
//   for each slab: cells (RowCells), masks -> fold (then the pads' terms:
//   hf_pads.cuh) -> transform -> a step of each tree leaf's hash;
//   then the salts and tree leaves, and the part's nodes level by level up to node(lp, h).
// The nodes of the levels s .. lp (s: the lowest level kept for openings) go to the host's heap
// (HfKept in hf_commit_api.h), and the host computes the levels above lp. A slab's cells and masks
// do not depend on the part, so np parts are built together from one computation of them. Where the
// weights are copied from host memory for every pass over the slabs, np = 2 halves the copies. The
// root does not depend on lp, np or s. The large GPU buffers and their sizes at full size are
// listed at part_bits_that_fit().
//
// Self-check of the masks, [own-test-masks] (hf_mask_own.cuh; a tag in brackets is the name that a
// check's message prints). A table built with masks that are not the key's passes every verifier,
// with its opened columns unmasked, if the openings compute the same wrong masks (a wrong copy of
// the key). If only the table has them (a buffer never filled), an opening stops at [hc-own-node].
// So the first call of parts() compares at most 64 blocks of eight masks in every row, in
// the buffer that the fold reads, with masks that separate code derives from a second copy of the
// key (TablePlan::key_again). parts() exits if that copy is missing, and make_table()
// returns true only after that call's last comparison. The masks of later calls are not checked.
//
// Hidden values: cells, masks, pads, V, the symbols and the hash states. PartMaker::close(), which
// the destructor calls, zeroes V, `state`, `masks` and both copies of the pads; masks_own_test()
// zeroes own_room. A failed [own-test-masks] or [own-test-pads] calls close() before it exits; a
// failed [own-test-second-text] does not. At any other exit (exit(), abort(), a failed CK()) the
// destructor does not run and the buffers are not zeroed. Digests, `tw` and `high` are not hidden.
#pragma once
#include "hash_commitment/hf_code.cuh"
#include "hash_commitment/hf_code_groups.cuh" // the faster transform on a long run
#include "code/code.h" // blocks/code/code.h: where a row's message and random tail start
#include "hash_commitment/hf_commit_api.h"
#include "hash_commitment/hf_commit_cells.cuh"
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_pads.cuh"
#include "hash_commitment/hf_tree.cuh"
#include "common/derivation_check_seam.h"
#include "common/made_up_alone.h" // whether a test's fault injection may act in this run
#include <chrono>
#include <functional>
namespace hf {
// Fault injection for tests: `ROWS_TABLE_FAULT` in the environment makes every call of parts() lose
// masks or hash other salts. If its openings shared the fault, such a table would verify: without
// masks its opened columns would hide nothing; under salts that anyone knows a guess at a column
// that is not opened could be tested against its leaf. So the program must exit before publishing
// anything.
// `entries-before` zeroes a slab's masks before [own-test-masks] and [own-test-second-text],
// `entries-behind` after them; `salts` hashes the salts of KNOWN_KEY into the tree's leaves.
// parts() detects only the first; an opening must stop the others.
// Eight more values zero the rows' pads, which mask the record's published answers. Each of the
// pads' four users generates its own copy (hf_pads.cuh):
// `pads-before` zeroes every copy before its first [own-test-pads]; `pads-making`, `pads-answers`,
// `pads-fold`, `pads-columns` and `pads-all` zero one user's copy in host memory, or all four,
// right after it; `pads-making-gpu` and `pads-columns-gpu` zero only the copy in GPU memory.
// A test that is not part of this repository runs each value and names the check that must stop it.
// Acts only on synthetic weights with a test seed (made_up_alone::say()), in no production build.
struct TableFault {
  enum Kind { NONE, ENTRIES_BEFORE, ENTRIES_BEHIND, SALTS, PADS_BEFORE, PADS_MAKING, PADS_ANSWERS, PADS_FOLD, PADS_COLUMNS, PADS_ALL, PADS_MAKING_GPU, PADS_COLUMNS_GPU };
#ifdef SERVING_BUILD
  static Kind planted() { return NONE; }
#else
  // Public, so that anyone knows its salts. Not the zero key: refuse_the_zero_key() ends on that.
  static constexpr hf_key32 KNOWN_KEY{{{1}}};
  static Kind planted() {
    static const Kind kind = [] {
      const char *word = made_up_under_a_tests_seed() ? TESTENV("ROWS_TABLE_FAULT") : nullptr;
      if (!word) return NONE;
      const Kind k = !strcmp(word, "entries-before")   ? ENTRIES_BEFORE
                     : !strcmp(word, "entries-behind") ? ENTRIES_BEHIND
                     : !strcmp(word, "salts")          ? SALTS
                     : !strcmp(word, "pads-before")    ? PADS_BEFORE
                     : !strcmp(word, "pads-making")    ? PADS_MAKING
                     : !strcmp(word, "pads-answers")   ? PADS_ANSWERS
                     : !strcmp(word, "pads-fold")      ? PADS_FOLD
                     : !strcmp(word, "pads-columns")   ? PADS_COLUMNS
                     : !strcmp(word, "pads-all")       ? PADS_ALL
                     : !strcmp(word, "pads-making-gpu")  ? PADS_MAKING_GPU
                     : !strcmp(word, "pads-columns-gpu") ? PADS_COLUMNS_GPU
                                                       : NONE;
      if (k == NONE) {
        fprintf(stderr, "cp_commit: ROWS_TABLE_FAULT takes entries-before, entries-behind, salts, pads-before, pads-making, pads-answers, pads-fold, pads-columns, pads-all, pads-making-gpu or pads-columns-gpu\n");
        exit(2);
      }
      fprintf(stderr, "cp_commit: ROWS_TABLE_FAULT=%s: a planted fault: this run loses these blinding values on purpose\n", word);
      return k;
    }();
    return kind;
  }
#endif
};
#ifndef SERVING_BUILD
// The two hooks that hf_pads.cuh calls (`pads_fault`, `pads_fault_on_the_gpu`), answered from
// ROWS_TABLE_FAULT.
static bool pads_fault_planted(PadsUser who, bool before_the_look) {
  const TableFault::Kind k = TableFault::planted();
  if (before_the_look) return k == TableFault::PADS_BEFORE;
  return k == TableFault::PADS_ALL || k == (who == PADS_FOR_THE_MAKING    ? TableFault::PADS_MAKING
                                            : who == PADS_FOR_THE_ANSWERS ? TableFault::PADS_ANSWERS
                                            : who == PADS_FOR_THE_FOLD    ? TableFault::PADS_FOLD
                                                                          : TableFault::PADS_COLUMNS);
}
static bool pads_fault_on_the_gpu_planted(PadsUser who) {
  const TableFault::Kind k = TableFault::planted();
  return (who == PADS_FOR_THE_MAKING && k == TableFault::PADS_MAKING_GPU) || (who == PADS_FOR_THE_COLUMNS && k == TableFault::PADS_COLUMNS_GPU);
}
static const bool pads_fault_is_set = (pads_fault = pads_fault_planted, pads_fault_on_the_gpu = pads_fault_on_the_gpu_planted, true);
#endif
struct TablePlan {
  int lp = 0;               // a part has 2^lp positions; s <= lp <= fln, 1 <= lp <= 30
  int np = 1;               // number of parts built together in one pass over the slabs
  int s = 0;                // the lowest kept level
  bool lines = true;        // print progress lines: times and public numbers only
  const uint8_t *key_again = nullptr; // the key's second copy
};
// Whether all parts share one small transform table (hf_code_groups.cuh),
// or each part has its own.
static inline constexpr bool parts_share_a_table(int lp) {
  return lp >= code_groups::LEAST_HEIGHT;
}
static inline size_t part_table_bytes(int lp, int np) { // 2^(lp - 1) words shared, or 2^lp a part
  return parts_share_a_table(lp) ? (size_t)4 << lp : (size_t)np * ((size_t)8 << lp);
}
static inline size_t part_bytes(const HfParams &P, int lp, int np) {
  return ((size_t)128 << lp) + part_table_bytes(lp, np) + (size_t)np * chain_bytes((u64)1 << lp) +
         16 * 64 * (size_t)mask_blocks(P.nm) + pads_card_bytes(P) + 4096;
}
// The largest lp from max(s, 1) to min(fln, most) with part_bytes() <= `bytes`; if none, max(s, 1).
// part_bytes() counts PartMaker's large GPU buffers, not those of RowCells: V, 128 * 2^lp bytes;
// a slab's `masks`, 1024 ceil(nm / 8); the transform's table `tw`, part_table_bytes(); each part's
// hash states (`state`), 80 * 2^lp. At full size (target_full_f2) with lp = 27:
// 17.2 + 5.6 + 0.5 + 10.7 np GB.
static inline int part_bits_that_fit(const HfParams &P, int s, int np, size_t bytes, int most = 28) {
  int lp = std::min(P.ln, most);
  while (lp > std::max(s, 1) && part_bytes(P, lp, np) > bytes) lp--;
  return lp;
}
// Folds a slab's 16 rows for one part: V[t 2^lp + i] = the sum over x < nx of high[x] c[x 2^lp + i]
// for t < 16, i < 2^lp. Coefficient c[j] of row t is cells[t * cell_stride + j - data_at] for
// data_at <= j < data_at + l, masks[t * mask_stride + j - mask_at] for mask_at <= j < mask_at + nm,
// and 0 elsewhere; the two ranges must not overlap. A row t >= `rows` gives zeros.
// Launch: any number of blocks in x, 16 in y.
// A zero cell skips its product (`if (c)`), so the kernel's running time depends on the weights.
// The loop form is public: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", NVIDIA Technical Blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void __launch_bounds__(256)
    k_fold_slab(const int8_t *__restrict__ cells, size_t cell_stride, const u64 *__restrict__ masks, size_t mask_stride,
                u64 l, u64 data_at, u64 mask_at, u64 nm, int lp, const u64 *__restrict__ high, int nx, int rows,
                u64 *__restrict__ V) {
  const int t = blockIdx.y;
  const u64 A = (u64)1 << lp;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < A; i += (u64)gridDim.x * blockDim.x) {
    u64 sum = 0;
    if (t < rows)
      for (int x = 0; x < nx; x++) {
        const u64 at = ((u64)x << lp) + i;
        if (at >= data_at && at - data_at < l) {
          const int c = cells[(size_t)t * cell_stride + (at - data_at)];
          if (c) sum = gf_add(sum, gf_mul(high[x], c > 0 ? (u64)c : HF_P - (u64)(-c)));
        } else if (at >= mask_at && at - mask_at < nm)
          sum = gf_add(sum, gf_mul(high[x], masks[(size_t)t * mask_stride + (at - mask_at)]));
      }
    V[((size_t)t << lp) + i] = sum;
  }
}

#define SPLIT_MARK(step) ((void)0) // a stub of a timing hook: does nothing

// Builds parts of T's tree on one GPU. All calls must come from one host thread.
struct PartMaker {
  const HfParams &P;
  const Domain D;
  const int lp, s, np;
  const hf_key32 *const key; // host
  const u64 limit;
  const u64 KD;
  // Where a row's cells and masks start: message_at() and tail_at() of the library's code
  // (blocks/code/code.h) of length 2^fln with i0 message entries and a random tail of nm masks.
  // cp_commit, cp_prove and the verifier all take these two offsets from that one definition.
  const blk::Code code;
  const TreeTags tags;
  u64 *V = nullptr, *tw = nullptr, *state = nullptr, *masks = nullptr, *high = nullptr, *own_room = nullptr;
  // All rows' pads, in GPU and in host memory, from the first call of parts() until close(). Like
  // every user of a table's pads, this object must generate its own copy and not be given one:
  // hf_pads.cuh explains why.
  u64 *pads = nullptr;
  TablePads pads_host;
  const uint8_t *key_again = nullptr; // the key's second copy (TablePlan::key_again)
  bool made = false;         // parts() has been called
  bool masks_passed = false; // the first call of parts() passed the masks' self-check
  bool pads_looked_again = false; // look_at_the_pads_again() has run, or the run is ending
  // mask_stride: words per row of `masks`. nx = ceil(KD / 2^lp): number of terms in a fold's sum.
  size_t mask_stride = 0;
  int nx = 0;
  cudaStream_t st = cudaStreamPerThread;
  PartMaker(const HfParams &P_, int lp_, int s_, const hf_key32 *key_, int np_ = 1)
      : P(P_), D(domain(P_.ln)), lp(lp_), s(s_), np(np_), key(key_), limit(mask_limit(key_)), KD(P_.i0 + P_.nm),
        code{(unsigned)P_.ln, (size_t)P_.i0, (size_t)P_.nm}, tags(P_.configuration, HF_NUM_F, (u32)P_.R) {
    if (lp < 1 || lp > 30 || lp > P.ln || s > lp || s < 0 || np < 1) {
      fprintf(stderr, "cp_commit: internal error: a part of 2^%d positions with nodes kept from level %d\n", lp, s);
      exit(4);
    }
    if (code.dim() != KD || P.l + P.pads() > P.i0) {
      fprintf(stderr, "cp_commit: internal error: a row of %llu coefficients where the code has %llu\n", (unsigned long long)KD,
              (unsigned long long)code.dim());
      exit(4);
    }
    nx = (int)((KD + (((u64)1 << lp) - 1)) >> lp);
    mask_stride = 8 * (size_t)mask_blocks(P.nm);
    commit_room(&V, (size_t)128 << lp);
    commit_room(&tw, part_table_bytes(lp, np));
    if (parts_share_a_table(lp)) code_groups::small_table(D, lp, tw, st);
    commit_room(&state, np * chain_bytes((u64)1 << lp));
    commit_room(&masks, 16 * 8 * mask_stride);
    commit_room(&own_room, masks_own_room_bytes(16));
    commit_room(&high, 8 * (size_t)nx * np);
  }
  [[noreturn]] void pads_failed(int end) {
    pads_looked_again = true;
    close(); // erased first: a print can block
    fprintf(stderr, "%s: a self-check failed: the random pads of the weight table's rows, as they lie in memory, are not what a second computation from the table's key gives. This is a fault of "
                    "the machine or of the program. No table is made, and nothing of this run is kept. Run cp_commit again, on another GPU if there is one\n", own_stop(end));
    exit(4);
  }
  // Generates all rows' pads, runs [own-test-pads] and copies them to GPU memory. A failed
  // self-check ends the run.
  void make_pads() {
    pads_kernels_ready(); // can end the run, so it comes before any pad exists
    int end = take_table_pads(P, key, key_again, PADS_FOR_THE_MAKING, pads_host);
    commit_room(&pads, pads_card_bytes(P));
    if (end == OWN_GOOD) end = pads_to_card(pads_host, P, PADS_FOR_THE_MAKING, pads, st);
    if (end != OWN_GOOD) pads_failed(end);
  }
  // [own-test-pads] again, after the last read of both copies. No code writes either in between.
  // Not detected: a fault that changed a value and changed it back.
  void look_at_the_pads_again() {
    if (!pads || pads_looked_again) return;
    int end = pads_own_test(P, key, key_again, pads_host);
    if (end == OWN_GOOD) end = pads_card_test(pads_host, P, pads, st);
    if (end != OWN_GOOD) pads_failed(end);
    pads_looked_again = true;
  }
  u64 *table_of(int j) const { return tw + ((size_t)j << lp); }
  u64 *state_of(int j) const { return state + j * (chain_bytes((u64)1 << lp) / 8); }
  u64 *high_of(int j) const { return high + (size_t)j * nx; }
  // Builds the parts h0 .. h0 + count - 1, count <= np (not checked): writes their nodes of the
  // levels s .. lp into heap (host memory, HfKept's order) and returns when they have arrived.
  void parts(u64 h0, int count, RowCells &cells, u32 *heap) {
    const u64 A = (u64)1 << lp;
    if (!key_again) {
      fprintf(stderr, "%s: a table under a key is made only with the key's second copy at hand; no table is made\n",
              own_stop(OWN_KEY_FILE));
      exit(4);
    }
    if (!pads) make_pads();
    for (int j = 0; j < count; j++) {
      const std::vector<u64> weights = run_high(D, lp, h0 + j, (u64)nx);
      CK(cudaMemcpyAsync(high_of(j), weights.data(), 8 * (size_t)nx, cudaMemcpyHostToDevice, st));
      CK(cudaStreamSynchronize(st)); // `weights` is pageable memory that this iteration frees
      if (!parts_share_a_table(lp)) run_table(D, lp, h0 + j, table_of(j), st);
    }
    std::vector<code_groups::PartFactors> factors; // a part's scalars; kernels take them by value
    if (parts_share_a_table(lp))
      for (int j = 0; j < count; j++) factors.push_back(code_groups::part_factors(D, lp, h0 + j, tw));
    SPLIT_MARK(TABLES);
    const u32 slabs = slabs_of((u32)P.R);
    for (u32 b = 0; b < slabs; b++) {
      const int rows = (int)std::min<size_t>(16, P.R - 16 * (size_t)b);
      size_t cell_stride = 0;
      const int8_t *c = cells.rows(16 * b, 16, &cell_stride, st);
      SPLIT_MARK(CELLS);
      masks_of_rows(key, limit, 16 * b, (u32)rows, 0, mask_blocks(P.nm), P.nm, masks, mask_stride, st);
      if (TableFault::planted() == TableFault::ENTRIES_BEFORE) CK(cudaMemsetAsync(masks, 0, (size_t)rows * mask_stride * 8, st));
      SPLIT_MARK(MASKS);
      if (!masks_passed) {
        const int end = masks_own_test(key_again, limit, 16 * b, (u32)rows, 0, mask_blocks(P.nm), P.nm, masks,
                                       mask_stride, own_room, st);
        if (end != OWN_GOOD) {
          fprintf(stderr, "%s: the masks of the rows from %u on are not those of the key's second copy; no table is made\n",
                  own_stop(end), 16 * b);
          pads_looked_again = true; // the run is ending: close() erases without comparing again
          close();
          exit(4);
        }
        // [own-test-second-text]: cp_commit checks all the slab's masks by a second implementation.
        HELD_TO_THE_DERIVATION_CHECK(slab_masks(16 * b, (u32)rows, P.nm, hf_mask_limit_in_use(limit), masks, mask_stride, st))
        SPLIT_MARK(MASKS_TEST);
      }
      if (TableFault::planted() == TableFault::ENTRIES_BEHIND) CK(cudaMemsetAsync(masks, 0, (size_t)rows * mask_stride * 8, st));
      for (int j = 0; j < count; j++) {
        k_fold_slab<<<dim3((unsigned)std::min<u64>((A + 255) / 256, 8192), 16), 256, 0, st>>>(
            c, cell_stride, masks, mask_stride, P.l, code.message_at(), code.tail_at(), P.nm, lp, high_of(j), nx, rows, V);
        CK(cudaGetLastError());
        pads_into_part(pads, P, code.message_at() + P.pad_entry(0), 16 * b, rows, lp, high_of(j), V, st);
        SPLIT_MARK(FOLD);
        if (parts_share_a_table(lp))
          code_groups::run_encode(V, factors[j], lp, 16, st);
        else
          run_encode(V, table_of(j), lp, 16, st);
        SPLIT_MARK(TRANSFORM);
        chain_rows(state_of(j), A, V, A, b, (u32)rows, tags, P.ln, (h0 + j) << lp, st);
        SPLIT_MARK(CHAIN);
      }
    }
    // Tree leaves, then level after level, in V, now free: two regions of 32 * 2^lp bytes in turn.
    for (int j = 0; j < count; j++) {
      const u64 h = h0 + j;
      u32 *now = (u32 *)V, *next = (u32 *)V + A * DW;
      // Zeroes both regions, which hold hidden symbols of T or the previous part's digests.
      // The kernels below write every digest before it is read, so without a fault the zeros are
      // never read. If a faulty GPU drops one of these kernels' writes, the old content stays
      // where that digest belongs and may be copied to the heap: zeros or a lower level's digest,
      // never a symbol.
      CK(cudaMemsetAsync(V, 0, (size_t)2 * A * DW * 4, st));
      chain_leaves(state_of(j), A, SERVING_OR(key, TableFault::planted() == TableFault::SALTS ? &TableFault::KNOWN_KEY : key), h << lp, (u32)P.R, tags, now, st);
      SPLIT_MARK(LEAVES);
      for (int l = 0;; l++) {
        const u64 nodes = A >> l;
        if (l >= s)
          CK(cudaMemcpyAsync(heap + kept_at(P.ln, l, h << (lp - l)) * DW, now, nodes * DW * 4, cudaMemcpyDeviceToHost, st));
        if (l == lp) break;
        nodes_up(tags, now, nodes / 2, next, st);
        std::swap(now, next);
      }
      SPLIT_MARK(NODES);
    }
    CK(cudaStreamSynchronize(st));
    made = true;
    masks_passed = true; // after the last slab's comparison: a failed one has exited
  }
  void close() {
    if (!V) return;
    look_at_the_pads_again();
    pads_host.wipe();
    CK(cudaMemsetAsync(V, 0, (size_t)128 << lp, st));
    CK(cudaMemsetAsync(state, 0, np * chain_bytes((u64)1 << lp), st));
    CK(cudaMemsetAsync(masks, 0, 16 * 8 * mask_stride, st));
    if (pads) CK(cudaMemsetAsync(pads, 0, pads_card_bytes(P), st));
    CK(cudaStreamSynchronize(st));
    for (u64 *p : {V, tw, state, masks, high, own_room, pads})
      if (p) CK((cudaFree)(p));
    V = tw = state = masks = high = own_room = pads = nullptr;
  }
  ~PartMaker() { close(); }
};

// Builds T's tree from level plan.s up, on the calling thread's current GPU, into heap: host memory
// of kept_digests(fln, s) digests of DW words each, in HfKept's order; heap + DW is then the root.
// make_cells() returns the source of M's rows, which is deleted at the end. An error exits the
// process. Returns whether [own-test-masks] has passed.
static inline bool make_table(const HfParams &P, const TablePlan &plan, const hf_key32 *key,
                              const std::function<RowCells *(int device)> &make_cells, u32 *heap) {
  const u64 parts = (u64)1 << (P.ln - plan.lp);
  const auto begin = std::chrono::steady_clock::now();
  bool made = false, passed = false;
  int device = 0;
  CK(cudaGetDevice(&device));
  RowCells *cells = make_cells(device);
  {
    PartMaker maker(P, plan.lp, plan.s, key, plan.np);
    maker.key_again = plan.key_again;
    for (u64 h = 0; h < parts; h += (u64)plan.np) {
      const auto from = std::chrono::steady_clock::now();
      const int count = (int)std::min<u64>((u64)plan.np, parts - h);
      maker.parts(h, count, *cells, heap);
      if (plan.lines)
        printf("cp_commit: parts %llu to %llu of %llu (2^%d positions each): %.2f s\n", (unsigned long long)h + 1,
               (unsigned long long)h + count, (unsigned long long)parts, plan.lp,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count());
    }
    made = maker.made, passed = maker.masks_passed;
  }
  cells->release();
  delete cells;
  CK(cudaGetLastError()); // finds an unread launch error (prover/docs/cuda-status.md, "The rule")
  kept_rebuild(heap, P.ln, plan.lp, TreeTags(P.configuration, HF_NUM_F, (u32)P.R));
  if (plan.lines)
    printf("cp_commit: the table's tree: %llu parts: %.1f s\n", (unsigned long long)parts,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
  return made && passed;
}
} // namespace hf
