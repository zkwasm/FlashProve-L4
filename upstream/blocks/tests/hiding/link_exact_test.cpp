// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the exact test of hiding (exact.h) on a whole LINK proof at toy size. LINK is to show that
// two committed tables contain the same entries: the anchor (`first`) and a leaf table (`further`;
// printed: copy). Witness: the rows that both tables encode. Coins: all field elements drawn to
// commit both tables and in link_prove(). View: both tables' records (stored sample answers, which
// everybody sees in the keys) and all field elements of the proof, both tables' opened columns
// included. A pass covers one pair of random witnesses and one list of challenges. The prover runs
// on the CPU (`EngineCpu`, `IndexCpu`): a pass says nothing about the GPU engine (`EngineGpu`), with
// which cp_commit and cp_prove make their proofs. The challenges are fixed on the prover's side
// only, by two switches in its channel (details at world_of()): no verifier runs here, and none
// accepts such a proof.
// In every case both witnesses must give the view the same law (distribution), and each table's
// random tails, the random entries encoded with each row, and each row's pads must be needed for
// that. At CASES[0], whose tails are sized for one LINK proof, the witnesses must be told apart
// under three negative controls (expect()).
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `link_world` builds a plan by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // fixtures vouch for untested tables (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>

#include "iop/tests/link_world.h"
#include "tests/hiding/sessions.h"

using namespace blk;

namespace {

// Columns: `LinkCase`'s fields (blocks/iop/tests/link_world.h).
const LinkCase CASES[] = {
    // name                              v data spare ans  c0  cj links rate  u   high  draws ahead
    {"tails for one link",               2, 11,  2,    1,   3,  5,  1,    2,   8,  false, 3,   false},
    {"tails for two links",              2, 11,  2,    1,   3,  5,  2,    2,   8,  false, 3,   false},
    {"rows high, masks ahead",           2, 11,  2,    1,   3,  5,  2,    2,   8,  true,  3,   true},
};

// To fix the challenges, `BLK_TEST_PROVER_MESSAGES_OUTSIDE` keeps the prover's messages out of its
// hash queries, first_query_stands_still (set in main()) keeps the tables' roots and records out of
// its first query, and `NamedAs` (sessions.h) keeps the samples' seeds the same in all runs.
World world_of(const LinkCase &t, const LinkWorld &w) {
  return [&t, &w](unsigned witness, Coins &coins) {
    const Table rows = link_rows(t, w, 300 + witness);
    IndexCpu first(w.S.table, rows, coins, w.configuration), further(w.S.table, rows, coins, w.configuration);
    NamedAs first_keeper(first), further_keeper(further);
    SessionLook seen;
    seen.add_record(first_keeper.record()), seen.add_record(further_keeper.record());
    EngineCpu engine;
    Writer link;
    link.keep_list();
    BLK_REQUIRE(link_prove(w.plan, w.configuration, w.codes, w.S, first_keeper, further_keeper, engine, coins, link), "a table's keeper does not serve");
    seen.add("the LINK proof", link);
    return seen.all;
  };
}

bool verbose = false;  // `failures`: the counter of `CHECK`, blocks/iop/tests/iopp_support.h
std::string summary;
// apart: the witnesses must be told apart at a named entry (`v.entry`); a comparison that gives up
// names none.
// tails: what replaces the leaf table's random tails. 0: nothing. 1: the anchor's tails, so the two
// tables differ only by their pads, and are opened at both lists of positions. 2: zeros. 3: nothing;
// all pads of both tables are zeros instead.
void expect(const LinkCase &t, const char *what, int tails, bool apart) {
  const LinkWorld w = link_world(t);
  Exact x(world_of(t, w), 1);
  const size_t making = 1 + (size_t)w.S.table.rows();  // a making's draws; the two makings come first
  std::vector<size_t> pads = pad_coins(x, w.S.table), further_pads = pad_coins(x, w.S.table, making);
  pads.insert(pads.end(), further_pads.begin(), further_pads.end());
  if (tails == 3) {
    x.spoil([pads](std::vector<F> &list) {
      for (size_t i : pads) list[i] = F::zero();
    }, pads);
  } else if (tails) {
    const size_t from = x.draws()[0].first, to = x.draws()[making].first, count = x.draws()[0].count;
    std::vector<size_t> held;
    for (size_t i = 0; i < count; i++) held.push_back(to + i);
    x.spoil([=](std::vector<F> &list) {
      for (size_t i = 0; i < count; i++) list[to + i] = tails == 1 ? list[from + i] : F::zero();
    }, held);
  }
  const Exact::Verdict v = x.compare(1);
  const std::string name = std::string(t.name) + (*what ? ", " : "") + what;
  const bool as_foretold = apart ? (!v.same && v.entry >= 0) : v.same;
  if (verbose || !as_foretold) print_comparison("link_exact_test", name.c_str(), x, v, verbose);
  if (!as_foretold) failures++, std::printf("link_exact_test: %s: NOT AS FORETOLD\n", name.c_str());
  if (!apart && v.same && !(v.draw_matters[0] && v.draw_matters[making])) failures++, std::printf("link_exact_test: %s: a table's random tails could be constants\n", name.c_str());
  for (size_t d = 0; !apart && v.same && d < 2 * making; d++)
    if (d % making && !v.draw_matters[d]) failures++, std::printf("link_exact_test: %s: the pads of row %zu of a table could be constants\n", name.c_str(), d % making - 1);
  summary += std::string(summary.empty() ? "" : "; ") + name + ": " + (v.same ? "the same law" : "told apart") + " (" + std::to_string(x.coins()) + " coins, " + std::to_string(x.entries()) +
             " entries of the view)";
}

} // namespace

// Arguments: "-v" prints a line for each comparison and draw.
// The two oracle messages with the tables' folded rows have no mask row, so once a table's opened
// columns use up its tails, the pad is the only random entry left to mask the answer.
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) verbose = verbose || !std::strcmp(argv[i], "-v");
  first_query_stands_still = true;  // the records' answers stand in the first query and move with the coins
  for (const LinkCase &t : CASES) expect(t, "", 0, false);
  expect(CASES[0], "the copy made with the first table's random tails", 1, true);
  expect(CASES[0], "the copy's random tails are zeros", 2, true);
  expect(CASES[0], "both tables' pads are zeros", 3, true);
  std::printf("link_exact_test: %s (%s)\n", failures ? "FAILED" : "ok", summary.c_str());
  return failures ? 1 : 0;
}
