// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The exact test of hiding (exact.h) on whole sessions on a committed table (`Index` in
// identifiers): the table is built and one batch proved. Each request starts with `PENDING` linear
// claims ("pending rows" in the output) on the table's matrix; member_first_prove() batches them
// into one. The witness is the matrix: do two random matrices, under the same public linear forms,
// give the view (all field elements that a batch's requests see together) the same distribution
// ("law" in the output)? Coins, fixed challenges and new vectors: as in lin_exact_test.cpp.
// judge_session() (sessions.h) decides whether a case passes.
// Limits beyond exact.h's: toy sizes; only the CPU prover (`EngineCpu`, `IndexCpu`); no verifier
// runs; the committer's one-time test of the table is not run, so nothing of it is in the view. A
// pass says nothing about the GPU engine (`EngineGpu`), the only engine that cp_commit and cp_prove
// use. To fix the challenges, BLK_TEST_PROVER_MESSAGES_OUTSIDE changes the prover's channel
// (`ProverRounds`), `NamedAs` (sessions.h) gives the table a constant name, and in first queries a
// constant record (made_up_record()) stands in the table's record; no verifier is changed, so none
// accepts such a proof. The requests' numbers stay in their chains' first queries, so each request
// draws challenges of its own.
// The view also has the table's record, which everybody sees in the key.
// Two negative controls, at which two matrices must be told apart: a table whose rows' pads are
// zeros; and an Omega whose random tail covers one list for each chain that opens it but for the
// requests after the first, so that it hides enough only if all requests draw the same positions.
// Arguments: "-v" prints a line per comparison and draw; "long" runs all cases, not only
// "two requests".
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `member_world` builds plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // fixtures vouch for untested tables (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>
#include <string>

#include "engine/engine_cpu.h"
#include "iop/tests/member_world.h"
#include "tests/hiding/sessions.h"

using namespace blk;

namespace {
// Columns: `LinCase`'s fields (blocks/iop/tests/lin_world.h).
// The table's random tails (random entries encoded with each row) are sized for one batch; with
// `tight_tails`, each new vector's tail only for its own openings.
const LinCase CASES[] = {
    // name                                   v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws ahead  tight tails
    {"one request",                           1, 2,   4,   2,   1,  3,  1,  2,    8,  8,  4, false, 1, 0,    2, false, true},
    {"two requests",                          1, 2,   4,   2,   1,  3,  1,  2,    8,  8,  4, false, 2, 1,    2, false, true},
    {"three requests",                        2, 2,   3,   1,   1,  4,  1,  2,    8,  8,  8, false, 3, 2,    3, false, true},
    {"three requests, masks ahead",           2, 2,   3,   1,   1,  4,  1,  2,   16,  8,  8, true,  3, 2,    3, true,  true},
};
// member_first_prove() (blocks/iop/member.h) batches only if this exceeds 1.
constexpr size_t PENDING = 3;
constexpr size_t DEFAULT_CASE = 1; // two requests: the smallest coalition of more than one

World world_of(const LinCase &t, const LinWorld &w) {
  return [&t, &w](unsigned witness, Coins &coins) {
    const LinMatrix M = session_matrix<LinMatrix>(witness, [&w](u64 key) { return lin_matrix(w.S, key); });
    IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), coins);
    NamedAs keeper(index);
    const IndexRecord record = keeper.record(), in_queries = made_up_record(w.S.fold.index);
    SessionLook seen;
    seen.add_record(record);
    MemberOptions opt;
    opt.query_record = &in_queries;
    opt.coins = &coins;
    opt.look = [&seen](const char *chain, unsigned request, const Writer &bytes) { seen.add(chain, request, bytes); };
    EngineCpu engine;
    const std::vector<LinView> views = member_session(t, w, PENDING, keeper, record, M, M, 31, 41, engine, opt);
    BLK_REQUIRE(views.size() == t.requests, "the table's keeper does not serve");
    return seen.all;
  };
}

} // namespace

int main(int argc, char **argv) {
  bool verbose = false, long_run = false;
  for (int i = 1; i < argc; i++) verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
  int failures = 0;
  std::string ran;
  for (const LinCase &t : CASES) {
    if (!long_run && &t != &CASES[DEFAULT_CASE]) continue;
    const LinWorld w = member_world(t, PENDING);
    size_t coins = 0;
    if (!judge_session("member_exact_test", t.name, world_of(t, w), hiding_positions(w.S.fold.index.code) == (u64)t.columns * t.batches, verbose, coins, 1 + (size_t)w.S.fold.index.rows())) failures++;
    ran += std::string(ran.empty() ? "" : "; ") + t.name + " (" + std::to_string(coins) + " coins)";
  }
  const LinCase &t = CASES[DEFAULT_CASE];
  const LinWorld w = member_world(t, PENDING);
  // Sets these coins to zero; the two matrices must then be told apart at a named entry.
  auto control = [&](const char *name, Exact &x, const std::vector<size_t> &zeros) {
    x.spoil([zeros](std::vector<F> &list) {
      for (size_t i : zeros) list[i] = F::zero();
    }, zeros);
    const Exact::Verdict v = x.compare(1);
    const std::string said = std::string("the control: ") + name;
    if (verbose || v.same || v.entry < 0) print_comparison("member_exact_test", said.c_str(), x, v, verbose);
    if (v.same || v.entry < 0) failures++, std::printf("member_exact_test: %s: NOT AS FORETOLD: not told apart\n", said.c_str());
    ran += "; " + said + ": told apart: " + x.where(v.entry);
  };
  // With zero pads the record's answers are exact linear functions of the rows.
  {
    Exact x(world_of(t, w), 1);
    const std::vector<size_t> pads = pad_coins(x, w.S.fold.index);
    control("the rows' pads are zeros", x, pads);
  }
  // Omega is opened by the shared test and by every request's own, and its random tail has an entry
  // for each position of these lists. The last ones become zeros, as many as the lists of all
  // requests but the first have positions: what is left covers the shared list and one other.
  {
    Exact x(world_of(t, w), 1);
    const Code &omega = w.codes.base[w.S.omega.code];
    BLK_REQUIRE(w.S.omega.main_rows == 1 && omega.r == (1 + (u64)t.requests) * t.draws, "Omega's tail is not one row's, sized for each list");
    const size_t count = E_DEGREE * omega.r, zero_entries = (size_t)(t.requests - 1) * t.draws;
    std::vector<const Exact::Draw *> tails;
    for (const Exact::Draw &d : x.draws())
      if (d.count == count) tails.push_back(&d);
    BLK_REQUIRE(tails.size() == 1, "no other draw has the size of Omega's random tail");
    std::vector<size_t> zeros;
    for (size_t i = count - E_DEGREE * zero_entries; i < count; i++) zeros.push_back(tails[0]->first + i);
    control("Omega's random tail covers one request's list alone", x, zeros);
  }
  std::printf("member_exact_test: %s (%zu pending rows a request; two random matrices have the same law in all that the requests of a batch see together; the view is affine in coins and matrix, so the pair speaks for all "
              "matrices; the table's random tails were sized for the one batch that is run and are needed; one list of challenges; the cases that ran: %s)\n",
              failures ? "FAILED" : "ok", PENDING, ran.c_str());
  return failures ? 1 : 0;
}
