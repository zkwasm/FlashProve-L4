// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs LIN (blocks/iop/lin.h), the test on the committed table (`Index` in identifiers): one linear
// claim per request of a batch, at toy message lengths but at the real target of 128 bits.
// blocks/frame/lin_frame.h derives the numbers of positions and columns, the random tails of LIN's
// codes (`r` random coefficients per row) and the stages of its proximity tests. The three hash
// chains (Fiat-Shamir transcripts) run on plans made by the library's planners: lists of rounds
// with their error bounds (`priced` in names and messages). Each plan must hold: every round
// reaches the target with its proof of work. So the accounting printed from such plans is that of
// the protocol that runs.
//   lin_frame_test [print]     print: also the accounting of the shared chain

#define BLK_PLANS_BY_HAND_FOR_A_TEST  // a test may build or alter plans: blocks/numbers/plan.h
#define BLK_TESTED_TABLE_FOR_A_TEST   // a test may vouch for a table: blocks/iop/foldeq.h
#include <cstring>

#include "chain/tests/death.h"
#include "frame/lin_frame.h"
#include "iop/tests/lin_world.h"
#include "numbers/accounting.h"

using namespace blk;

// like_full: shapes as at full size: row counts that are not powers of two, and a W (a request's
// oracle message) whose entries fill its rows, so that its pad, which masks its sample's answer, is
// in a mask row.
static TableSize toy_table(const Setting &set, bool like_full) {
  TableSize T;
  T.code.name = "table", T.code.log_n = 14, T.code.ell = like_full ? 1100 : 803, T.code.r = 1001;
  T.v = 3, T.answers = set.samples;
  T.distances.radius = T.distances.later_radius = 9000;
  return T;
}
// one_code: the groups `folded` and `omega` get the same code sizes.
static LinLines toy_lines(bool one_code, bool like_full) {
  LinLines L;
  L.k_lo = 3, L.matrix_rows = like_full ? 80 : 100;
  L.requests = 2;
  L.omega.log_n = L.own.log_n = 13, L.omega.log_ell = L.own.log_ell = like_full ? 5 : 7;
  L.folded.log_n = 13, L.folded.log_ell = one_code ? L.omega.log_ell : 9;
  // The code of a single mask row beside folded rows must have at most the main code's rate
  // (shape_fault(), blocks/numbers/blocks.h).
  L.folded.mask_log_n = L.omega.mask_log_n = 13, L.folded.mask_log_ell = L.omega.mask_log_ell = like_full ? 5 : 4;
  L.own.mask_log_n = 12, L.own.mask_log_ell = 4;
  L.folded.stages = {StageChoice{2, 1, 13, 13}, StageChoice{0, 0, 15, 15}};
  L.omega.stages = {StageChoice{2, 1, 13, 13}, StageChoice{0, 0, 15, 15}};
  L.own.stages = {StageChoice{2, 1, 13, 13}, StageChoice{0, 0, 15, 15}};
  return L;
}

// One-time test ("the key's test" in messages): the committer's own LIN run on a new table. Later
// test: a batch's (`TableTest`). For a later test, lin_verify_view() vouches that the table has
// passed its one-time test (blocks/iop/tests/lin_world.h).

int main(int argc, char **argv) {
  const bool print = argc > 1 && !std::strcmp(argv[1], "print");
  for (int which = 0; which < 4; which++) {
    {
      const int one_code = which & 1, like_full = which >> 1;
      Setting set = setting_goldilocks();
      set.instances = 2;
      set.set_work(ErrorKind::QUERY, Scope::REQUEST, 4), set.set_work(ErrorKind::QUERY, Scope::SHARED, 6);
      LinWorld w(TableTest::LATER);
      Plan first(set), shared(set), second(set);
      const LinSized x = lin_sized(set, toy_lines(one_code, like_full), toy_table(set, like_full), w.codes);
      if (!x.refused.empty()) {
        std::printf("FAILED: one code %d, like the full size %d: no such parameters: %s\n", one_code, like_full, x.refused.c_str());
        failures++;
        continue;
      }
      CHECK(x.batches >= 2, "the toy table bears too few batches for this test");
      w.S = x.S, w.input = x.input, w.binding.log_leaves = 1;
      const std::vector<TreeSize> input_trees = tree_sizes(w.codes, {w.input});
      plan_chain_start(first, "first chain", true);
      plan_send(first, "a request's masks", {SampleSize{Send::cube(w.codes, w.input.shape).bits, {x.own.stages[0].g}}}, (unsigned)input_trees.size());
      plan_scalars(first, "the claim's target", "stand-in for a request's own proof", 1);
      lin_plan_first(first, w.codes, x);
      plan_seal(first, "first chain");
      TreeSize table;
      table.name = "table", table.log_n = x.table.code.log_n, table.leaf_bytes = (u64)set.f_bytes << x.table.v;
      shared.begin_scope(Scope::SHARED);
      lin_plan_shared(shared, w.codes, x, table);
      plan_seal(shared, "shared chain");
      plan_chain_start(second, "second chain", true);
      lin_plan_second(second, w.codes, x, input_trees);
      w.first = first.finish(), w.shared = shared.finish(), w.second = second.finish();
      for (const Planned &p : {w.first, w.shared, w.second}) {
        if (p->refused()) std::printf("refused: %s\n", p->why_not().c_str());
        CHECK(p->holds(), "a planned round does not hold at the target");
      }
      if (print) print_accounting(account(w.shared), "shared chain");
      {  // planning under a code whose random tail is 1 longer than sized must abort
        Codes other = w.codes;
        other.base[x.S.test_u[0].code].r += 1;
        CHECK(ends_with("a stage of the proximity test is sized for another main code than it names", [&] {
                Plan q(set);
                q.begin_scope(Scope::SHARED);
                lin_plan_shared(q, other, x, table);
              }),
              "the shared chain is planned under codes that its proximity test is not sized for");
        Codes own = w.codes;
        own.mask[x.S.test_own[0].mask_code].r += 1;
        CHECK(ends_with("a stage of the proximity test is sized for another mask code than it names", [&] {
                Plan q(set);
                lin_plan_second(q, own, x, input_trees);
              }),
              "a second chain is planned under codes that its proximity test is not sized for");
      }

      LinCase t{};
      t.name = "sized", t.requests = 2;
      const LinMatrix M = lin_matrix(w.S, 11);
      TestCoins coins(21);
      IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), coins);
      const IndexKey key = index.key();
      const LinSession s = lin_session(t, w, index, key.rec, M, 31, 41);
      CHECK(s.served, "the keeper does not serve");
      for (unsigned c = 0; c < t.requests && s.served; c++) {
        const char *why;
        CHECK(lin_verify_view(w, key, 31, c, s.views[c], nullptr, &why), "an honest view is refused");
      }
      LinSessionOptions off;
      off.off_target = E::one();
      const LinSession f = lin_session(t, w, index, key.rec, M, 32, 42, off);
      CHECK(f.served && !lin_verify_view(w, key, 32, 0, f.views[0]) && lin_verify_view(w, key, 32, 1, f.views[1]), "a false target");
      const PartSize planned = w.shared->size_of(0);
      std::printf("one code %d, like the full size %d: %llu columns, %llu batches a table; positions %llu / %llu / %llu; shared part %zu bytes (planned: at most %llu, "
                  "expected %.0f), first %zu, second %zu\n",
                  one_code, like_full, (unsigned long long)w.S.fold.columns, (unsigned long long)x.batches, (unsigned long long)x.folded.t,
                  (unsigned long long)x.omega.t, (unsigned long long)x.own.t, s.served ? s.views[0].shared.size() : 0, (unsigned long long)planned.bytes,
                  planned.expected, s.served ? s.views[0].first.size() : 0, s.served ? s.views[0].second.size() : 0);
    }
  }
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
