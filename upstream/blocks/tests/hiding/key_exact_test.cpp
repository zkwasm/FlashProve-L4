// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Exact test of hiding (blocks/tests/hiding/exact.h: do two witnesses give equally distributed
// views?) on a whole key file: three hash chains with the committer's one-time proofs that the
// entries of its committed table's matrix are in range (WF) and that the table is close to one
// codeword (LIN). The fixture (`world` in identifiers) is wf_small_whole_world(), at toy size: a
// 4 x 4 matrix, one lookup, a record (stored sample answers) of one answer for each row. The
// witnesses are random matrices with entries in range (-2 .. 1); a pass covers only this pair and
// one list of challenges. The coins are all field elements that the committer draws, for the table
// and the proofs. The view is every field element of the file, the table's record and opened
// columns included.
// Only the CPU engine runs (`EngineCpu`, `IndexCpu`, `WfCpu`, `HeavyCpu`): a pass says nothing of
// the GPU engine, on which cp_commit makes key files. Two switches that change only the prover's
// channel fix the challenges (see world_of()); no verifier accepts such a file.
// main() asks, by the linear system and with no further run, that neither the table's random tails
// nor any row's pads could be constants. With the argument "pads-are-zeros" it injects one fault
// into the coins, and the two matrices must be told apart. The project's own control builds the test from a copy of blocks/iop/send.h in which send_prove()
// draws no pad for any sample; that copy is not part of this repository. A control, one for each of
// main()'s two cases, passes only if its case fails and the program prints the entry of the view at
// which the two matrices are told apart.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_WF_ROWS_FOR_A_TEST 1
#define BLK_KEY_UNDER_FOR_A_TEST 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixtures build plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets a fixture declare a table tested (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>

#include "engine/engine_cpu.h"
#include "frame/check_key.h"
#include "iop/tests/wf_small_world.h"
#include "tests/hiding/sessions.h"

using namespace blk;

namespace {

// To keep the challenges the same in all runs, BLK_TEST_PROVER_MESSAGES_OUTSIDE keeps what the
// prover sends out of the chains' hash inputs, first_query_stands_still (set in main()) keeps the
// table's root and record out of their first queries, and `NamedAs` (sessions.h) replaces the
// table's root and name by constants.
World world_of(const WfWorld &w, const KeyFixed &f) {
  return [&w, &f](unsigned witness, Coins &coins) {
    const KeyUnder under(KeyPlans{w.lin.first, w.lin.shared, w.lin.second}, f);
    const LinMatrix M = wf_small_matrix(w.lin.S, 300 + witness);
    IndexCpu index(w.lin.S.fold.index, lin_table_rows(w.lin.S, M), coins);
    NamedAs keeper(index);
    const LinMatrix *m = &M;
    WfCpu ranges(w.wf, wf_matrix_of(w.lin.S), [m](u64 row, u64 column) { return m->at(row, column); });
    HeavyCpu products(w.lin.S, M);
    EngineCpu engine;
    Writer file;
    file.keep_list();
    BLK_REQUIRE(make_key_under(under, keeper, ranges, products, engine, coins, nullptr, file), "no key is made");
    return look_at(file);
  };
}

} // namespace

// Arguments: "long" runs the test; "-v" then prints a line for each draw; "two-mask-rows" picks the
// second case; "pads-are-zeros" runs the negative control on the chosen case, in place of the
// comparison, and `make` does not run it. Without "long" the program makes one key, prints NOT RUN
// and exits with status 77,
// which blocks/Makefile counts as a failure; "make test" and
// "make long" run it with "long".
// The first case takes about nine minutes and 5.4 GB with four processes: the view is not affine in
// some of about 11,000 coins (printed as "curved"), so each coin costs six runs.
// The second case is there so that a test of hiding covers an index-mask message (rows of random
// masks only) with more than one mask row: two instead of one. WF's message (printed as "the
// message on the ranges") has nine mask rows instead of three, over a shorter mask code. This case
// has about 7,300 coins and takes four minutes and 2.5 GB.
// `make` does not run it.
int main(int argc, char **argv) {
  bool verbose = false, long_run = false, two_mask_rows = false, pads_are_zeros = false;
  for (int i = 1; i < argc; i++) {
    verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
    two_mask_rows = two_mask_rows || !std::strcmp(argv[i], "two-mask-rows");
    pads_are_zeros = pads_are_zeros || !std::strcmp(argv[i], "pads-are-zeros");
  }
  first_query_stands_still = true;  // the record's answers stand in first queries and move with the coins
  const WfWorld w = wf_small_whole_world(two_mask_rows);
  KeyFixed f;
  f.configuration = w.lin.configuration;
  f.codes = w.lin.codes, f.lin = w.lin.S, f.index_masks = w.lin.input, f.ranges = w.wf, f.binding = w.lin.binding;
  BLK_REQUIRE(!key_fault(f), "the small world's values are refused");
  if (!long_run) {
    ListedCoins learn;
    const Look l = world_of(w, f)(0, learn);
    std::printf("key_exact_test: NOT RUN (one key was made: %zu coins, %zu field entries; the test takes nine minutes: say \"long\")\n", learn.elements().size(), l.view.size());
    return 77;
  }
  Exact x(world_of(w, f), 1);
  const IndexShape &table = w.lin.S.fold.index;
  if (pads_are_zeros) {
    const std::vector<size_t> pads = pad_coins(x, table);
    x.spoil([pads](std::vector<F> &list) {
      for (size_t i : pads) list[i] = F::zero();
    }, pads);
    const Exact::Verdict v = x.compare(1);
    print_comparison("key_exact_test", "the control: the rows' pads are zeros", x, v, verbose);
    const bool noticed = !v.same && v.entry >= 0;
    std::printf("key_exact_test: %s (the control: the rows' pads are zeros: %s)\n", noticed ? "ok" : "FAILED", noticed ? ("told apart: " + x.where(v.entry)).c_str() : "NOT AS FORETOLD");
    return noticed ? 0 : 1;
  }
  const Exact::Verdict v = x.compare(1);
  if (verbose || !v.same) print_comparison("key_exact_test", two_mask_rows ? "a small whole key, two mask rows" : "a small whole key", x, v, verbose);
  int failures = !v.same;
  if (v.same && !v.draw_matters[0]) failures++, std::printf("key_exact_test: the table's random tails could be constants\n");
  for (u64 r = 0; v.same && r < table.rows(); r++)
    if (!v.draw_matters[1 + r]) failures++, std::printf("key_exact_test: the pads of row %llu of the table could be constants\n", (unsigned long long)r);
  std::printf("key_exact_test: %s (a whole key file, %zu coins, %zu entries of the view, rank %zu, %zu coins curved: "
              "two random matrices within the range have the same law; one pair, one list of challenges; "
              "mask rows: %zu in the index-mask message, %zu in the message on the ranges)\n",
              failures ? "FAILED" : "ok", x.coins(), x.entries(), v.rank, v.curved, (size_t)w.lin.input.shape.mask_rows, (size_t)w.wf.message.shape.mask_rows);
  return failures ? 1 : 0;
}
