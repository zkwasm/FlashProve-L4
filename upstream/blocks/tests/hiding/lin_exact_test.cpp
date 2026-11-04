// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The exact test of hiding (exact.h) on whole sessions of LIN, the block that proves one linear
// claim per request about a committed table (`Index` in identifiers). It asks whether two random
// matrices in the table give the view the same distribution ("law" in the output). The view is what
// a coalition of all requests of the table's batches sees; world_of() defines session and view.
// Limits beyond those stated in exact.h: toy sizes; only the CPU prover (`EngineCpu`, `IndexCpu`).
// A pass says nothing about the GPU engine (`EngineGpu`), on which `cp_commit` and `cp_prove` make
// every proof. The challenges are fixed on the prover's side alone (see world_of()), by a switch in
// its channel (`ProverRounds`) and constant public inputs: no verifier is changed, none accepts
// such a proof, and none runs on the compared sessions.
// Every run has a negative control, with some masks set to zero.
// Arguments: "-v" prints a line for each case and draw; "long" runs all cases (17 minutes and 7 GB
// with eight processes) instead of only the cases "two requests" and "a record of two answers".
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `lin_world` builds plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets a fixture declare a table tested (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>
#include <string>

#include "chain/tests/death.h"
#include "iop/tests/lin_world.h"
#include "tests/hiding/sessions.h"

using namespace blk;

namespace {

// The table's random tails (random entries encoded with each row, to hide opened positions) cover
// exactly the batches that run on it (one, or two in one view where the name says so), so a tail
// one entry too short would show. With `tight_tails` the session's new vectors (the folded row,
// each request's W, and Omega: each matrix row evaluated at the batch's shared challenge point) are
// opened at as many lists of positions as their tails cover.
// The columns below are `LinCase`'s fields (blocks/iop/tests/lin_world.h).
const LinCase CASES[] = {
    // name                                   v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws ahead  tight tails
    {"one request",                           1, 2,   4,   2,   1,  3,  1,  2,    8,  8,  4, false, 1, 0,    2, false, true},
    {"two requests",                          1, 2,   4,   2,   1,  3,  1,  2,    8,  8,  4, false, 2, 1,    2, false, true},
    {"three requests",                        2, 2,   3,   1,   1,  4,  1,  2,    8,  8,  8, false, 3, 2,    3, false, true},
    {"two requests, three codes",             2, 1,   5,   3,   1,  3,  1,  2,   16,  4,  8, false, 2, 1,    3, false, true},
    {"three requests, masks ahead",           2, 2,   3,   1,   1,  4,  1,  2,   16,  8,  8, true,  3, 2,    3, true,  true},
    {"a table's whole life: two batches",     1, 2,   4,   2,   1,  3,  2,  2,    8,  8,  4, false, 1, 0,    2, false, true},
    {"a record of two answers",               1, 2,   4,   2,   2,  3,  1,  2,    8,  8,  4, false, 1, 0,    2, false, true},
    {"two requests, rows high",               2, 2,   3,   0,   1,  4,  1,  2,    8,  8,  8, true,  2, 1,    3, false, true},
    {"two batches, masks ahead",              2, 1,   5,   0,   1,  3,  2,  2,   16,  4,  8, true,  2, 1,    3, true,  true},
};
constexpr size_t DEFAULT_CASE = 1; // two requests: the smallest coalition of more than one
constexpr size_t TWO_ANSWERS_CASE = 6;  // the smallest with two answers for each row

// A session is the committer's run, then `t.batches` batches in turn on one table, each with one
// shared hash chain (Fiat-Shamir transcript) and a first and a second chain per request. The
// witness is the table's matrix; the claims' linear forms are public and the same for every
// witness. The coins are all field elements that committer and prover draw. The view is every field
// element of every chain, including the claims' masked values and the table's opened columns, and
// the table's record (answers to samples of its rows, each masked by a pad).
// The challenges must be the same in all runs: `BLK_TEST_PROVER_MESSAGES_OUTSIDE` keeps the
// prover's messages out of the chains' hash queries, and in the first queries (hashed public
// inputs) constants from sessions.h replace the table's root (`NamedAs`) and record
// (made_up_record()).
World world_of(const LinCase &t, const LinWorld &w) {
  return [&t, &w](unsigned witness, Coins &coins) {
    const LinMatrix M = session_matrix<LinMatrix>(witness, [&w](u64 key) { return lin_matrix(w.S, key); });
    IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), coins);
    NamedAs keeper(index);
    const IndexRecord record = keeper.record(), in_queries = made_up_record(w.S.fold.index);
    SessionLook seen;
    seen.add_record(record);
    for (unsigned batch = 0; batch < t.batches; batch++) {
      const std::string which = "batch " + std::to_string(batch) + ", ";
      LinSessionOptions opt;
      opt.coins = &coins;
      opt.look = [&seen, &which](const char *chain, unsigned request, const Writer &bytes) { seen.add((which + chain).c_str(), request, bytes); };
      opt.query_record = &in_queries;
      const LinSession s = lin_session(t, w, keeper, record, M, 31 + batch, 41 + 1000 * batch, opt);
      BLK_REQUIRE(s.served, "the table's keeper does not serve");
    }
    return seen.all;
  };
}

} // namespace

// In a build of this test that is not part of this repository the batch has one request more than
// the codes were sized for. Omega is opened by the shared proximity test and by each request's own,
// and its random tail covers one list for each planned test. That program returns 0 only if (1) the
// prover's process aborts at the guard that counts opened positions, and with that guard disabled
// through a switch that is not part of this repository either (2) the verifier, its own guards
// enabled, accepts every request's view and (3) the exact test tells two matrices apart.
// The session's new vectors have no mask rows, so the pads are all that hides their samples'
// answers (README.md, "What these tests do not show").
// Here judge_session() (sessions.h) decides whether a case passes; it also requires that no row's
// pads could be constants.
int main(int argc, char **argv) {
  bool verbose = false, long_run = false;
  for (int i = 1; i < argc; i++) verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
  int failures = 0;
  std::string ran;
  for (const LinCase &t : CASES) {
    if (!long_run && &t != &CASES[DEFAULT_CASE] && &t != &CASES[TWO_ANSWERS_CASE]) continue;
    const LinWorld w = lin_world(t, TableTest::LATER);
    size_t coins = 0;
    if (!judge_session("lin_exact_test", t.name, world_of(t, w), hiding_positions(w.S.fold.index.code) == (u64)t.columns * t.batches, verbose, coins, 1 + (size_t)w.S.fold.index.rows()))
      failures++;
    ran += std::string(ran.empty() ? "" : "; ") + t.name + " (" + std::to_string(coins) + " coins)";
  }
  // With zero pads the record's answers are exact linear functions of the rows: the two matrices
  // must be told apart.
  {  // the control: a table whose rows' pads are zeros
    const LinCase &t = CASES[TWO_ANSWERS_CASE];
    const LinWorld w = lin_world(t, TableTest::LATER);
    const IndexShape &s = w.S.fold.index;
    Exact x(world_of(t, w), 1);
    BLK_REQUIRE(x.draws().size() > s.rows() && x.draws()[0].count == s.rows() * s.code.r && x.draws()[1].count == s.pads(), "a making draws its rows' random tails in one draw, then each row's pads");
    std::vector<size_t> pads;
    for (u64 r = 0; r < s.rows(); r++)
      for (size_t i = 0; i < s.pads(); i++) pads.push_back(x.draws()[1 + r].first + i);
    x.spoil([pads](std::vector<F> &list) {
      for (size_t i : pads) list[i] = F::zero();
    }, pads);
    const Exact::Verdict v = x.compare(1);
    if (verbose || v.same || v.entry < 0) print_comparison("lin_exact_test", "the control: the rows' pads are zeros", x, v, verbose);
    if (v.same || v.entry < 0) failures++, std::printf("lin_exact_test: the control: NOT AS FORETOLD: a record on pads that are zeros is not told apart\n");
    ran += "; the control, pads that are zeros: told apart: " + x.where(v.entry);
  }
  std::printf("lin_exact_test: %s (two random matrices have the same law in all that the requests of a table's batches see together; the view is affine in coins and matrix, so the pair speaks for all matrices; the table's "
              "random tails serve all the batches that they were sized for and are needed; one list of challenges; the cases that ran: %s)\n",
              failures ? "FAILED" : "ok", ran.c_str());
  return failures ? 1 : 0;
}
