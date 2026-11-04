// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Applies the exact test of hiding (blocks/tests/hiding/exact.h) to a committed table (`Index` in
// identifiers): rows encoded as codewords under one Merkle tree. The rows are encoded once, with
// random tails (coefficients that are to hide opened positions; printed: random entries) sized for
// the columns (all rows' symbols at one position) of a fixed number of batches, the table's budget.
// Its holder (`IndexHolder`, here `IndexCpu`; printed: `keeper`) counts the columns promised in
// begin(), which returns false if the tails cover no further batch. No verifier can detect a holder
// that miscounts, a table encoded again with the same private randomness (`coins` in identifiers),
// or two tables with the same tails. IndexCpu::open() does not check the count, which protects only
// callers that call begin() first, as lin_begin() and Link::prove() (blocks/iop/) do.
// A record is the stored answer to an out-of-domain sample of each row (a linear form of its
// message, set by a seed), masked by a pad at the end of that message.
// The fixture (`World`) runs only `IndexCpu`, on the CPU, at toy size: no protocol block and no
// verifier runs. A pass says nothing about the GPU holder `HfKeeper` (prover/hash_commitment/), the
// only one that cp_commit and cp_prove use. The witness is the table's rows; the view is its
// records and the columns opened at fixed positions. For each set-up (`Play`), expect() compares
// the view's law (distribution) for one pair of pseudo-random witnesses.
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "iop/index_cpu.h"
#include "tests/hiding/exact.h"

using namespace blk;

namespace {

// A batch here is one open() at COLUMNS positions; a table's budget is BATCHES batches.
constexpr u64 COLUMNS = 3, BATCHES = 2;

// A toy table: 2 rows of 8 entries, and a record of one answer for each row.
IndexShape shape() {
  IndexShape s;
  s.v = 1, s.answers = 1;
  s.code.ell = 8 + s.pads(), s.code.r = tail_for_positions(COLUMNS * BATCHES);
  s.code.log_n = s.code.log_dim() + 2;
  BLK_REQUIRE(!index_fault(s), "the test's numbers are not a table's shape in this setting");
  return s;
}

struct Play {
  unsigned makings = 1;      // tables made one after the other
  bool other_rows = false;   // the second holds other rows than the first
  u64 batches = BATCHES;     // batches opened from each table
  bool same_lists = false;   // the second is opened where the first is
  bool count = true;         // `begin` is called before every batch
  unsigned records = 1;      // records of each table, each at separate fixed seeds
};

// Positions count up from 1, through both tables unless `same_lists`: no table is opened twice at
// one position. The seeds of IndexCpu::record() depend on the root, but the exact test requires
// verifier's messages that do not depend on what is sent: so the seeds are fixed here, and
// record_value() (blocks/iop/foldeq.h) gives the answers.
World world_of(const Play &p) {
  return [p](unsigned witness, Coins &coins) {
    const IndexShape s = shape();
    Writer shown;
    shown.keep_list();
    u64 next = 1;
    for (unsigned making = 0; making < p.makings; making++) {
      Table rows(1, s.rows(), s.plain());
      ToyCoins(100 + witness + (p.other_rows ? 50 * making : 0)).fill(rows.a.data(), rows.a.size());
      IndexCpu table(s, rows, coins);
      for (unsigned record = 0; record < p.records; record++) {
        ToyCoins listed(900 + 10 * making + record);
        SampleSeed seed;
        for (unsigned i = 0; i < s.cube_bits(); i++) seed.z.push_back(listed.e());
        seed.beta = listed.e();
        BLK_REQUIRE(seed.beta != E::zero(), "a sample's factor is not 0");
        for (u64 r = 0; r < s.rows(); r++) {
          std::vector<F> message(s.code.ell);
          for (size_t j = 0; j < message.size(); j++) message[j] = table.message_entry(r, j);
          shown.put(record_value(s, seed, 0, message.data()));
        }
      }
      if (p.same_lists) next = 1;
      for (u64 b = 0; b < p.batches; b++) {
        BLK_REQUIRE(!p.count || table.begin(COLUMNS), "the keeper does not serve");
        std::vector<u64> at;
        for (u64 i = 0; i < COLUMNS; i++) at.push_back(next++);
        std::vector<F> columns;
        table.open(PositionList::from_draws(at, s.code.log_n), shown, columns);
      }
    }
    return look_at(shown);
  };
}

int failures = 0, controls = 0, noticed = 0;
std::string summary;
// In expect(), `apart` is the expected verdict. "Told apart" counts only if `Verdict::entry` is
// set, not if the exact test fails for another reason. A set-up with `apart` is a negative control.
// IndexCpu draws all rows' random tails in one draw, then each row's pads in a separate draw.
enum Shared { NOTHING, TAILS, PADS };
void expect(const char *what, const Play &p, Shared shared, bool apart) {
  Exact x(world_of(p), 1);
  if (shared != NOTHING) { // the second table's tails or pads take the first table's values
    const size_t each = x.draws().size() / 2;
    BLK_REQUIRE(p.makings == 2 && each == 1 + shape().rows(), "the makings' draws are not where this control looks for them");
    std::vector<std::pair<size_t, size_t>> tied; // (to, from)
    for (size_t d = shared == TAILS ? 0 : 1; d < (shared == TAILS ? 1 : each); d++)
      for (size_t i = 0; i < x.draws()[d].count; i++) tied.emplace_back(x.draws()[each + d].first + i, x.draws()[d].first + i);
    std::vector<size_t> held;
    for (const auto &t : tied) held.push_back(t.first);
    x.spoil([tied](std::vector<F> &list) {
      for (const auto &t : tied) list[t.first] = list[t.second];
    }, held);
  }
  const Exact::Verdict v = x.compare(1);
  const bool as_foretold = apart ? (!v.same && v.entry >= 0) : v.same;
  if (!as_foretold)
    std::printf("table_exact_test: %s: %s: NOT AS FORETOLD (%zu coins, %zu entries of the view, rank %zu)\n", what, v.same ? "the same law" : "TOLD APART", x.coins(), x.entries(), v.rank);
  if (!as_foretold) failures++;
  if (apart) controls++, noticed += as_foretold;
  summary += std::string(summary.empty() ? "" : "; ") + (apart ? "control: " : "") + what + ": " + (v.same ? "the same law" : "told apart");
}

} // namespace

// First the set-ups that keep the rules: the same law of the view is expected in each, and false
// from begin() beyond the budget. Then five negative controls: the program returns 0 only if the
// witnesses are told apart in each. A control is only a faulty set-up of this test: no fault is
// injected into the library.
int main() {
  Play one, two, others;
  two.makings = others.makings = 2, others.other_rows = true, others.same_lists = true;
  expect("a table serves its whole budget", one, NOTHING, false);
  expect("made again with fresh random entries and pads, each making serves its budget", two, NOTHING, false);
  expect("two tables with fresh random entries and pads, opened at the same lists", others, NOTHING, false);
  {
    const IndexShape s = shape();
    Table rows(1, s.rows(), s.plain());
    ToyCoins coins(1);
    IndexCpu table(s, rows, coins);
    bool served = true;
    for (u64 b = 0; b < BATCHES; b++) served = served && table.begin(COLUMNS);
    if (!served || table.begin(COLUMNS)) failures++, std::printf("table_exact_test: the keeper does not serve its budget, or serves more\n");
  }
  { // the results above apply to `IndexCpu::record` only if it equals `record_value` at its seeds
    const IndexShape s = shape();
    Table rows(1, s.rows(), s.plain());
    ToyCoins coins(1);
    coins.fill(rows.a.data(), rows.a.size());
    IndexCpu table(s, rows, coins);
    const IndexRecord own = table.record();
    const std::vector<SampleSeed> seeds = record_seeds(s, table.name());
    bool equal = own.y.size() == s.rows() && seeds.size() == 1;
    for (u64 r = 0; equal && r < s.rows(); r++) {
      std::vector<F> message(s.code.ell);
      for (size_t j = 0; j < message.size(); j++) message[j] = table.message_entry(r, j);
      equal = own.y[r] == record_value(s, seeds[0], 0, message.data());
    }
    if (!equal) failures++, std::printf("table_exact_test: the keeper's own record is not what the test reckons from the keeper's rows\n");
  }
  summary += "; the keeper says no to one batch more";
  { // one batch more than the tails cover is opened, and begin() is not called, as after a count
    // that was set back
    Play p;
    p.batches = BATCHES + 1, p.count = false;
    expect("one batch more than the random entries cover", p, NOTHING, true);
  }
  { // the same rows encoded again with the same tails and a count of 0; a whole budget each
    Play p;
    p.makings = 2;
    expect("made again with the same random entries, each making serves its budget", p, TAILS, true);
  }
  { // two tables with different rows and the same tails; one batch each
    Play p;
    p.makings = 2, p.other_rows = true, p.batches = 1, p.same_lists = true;
    expect("two tables with the same random entries, one batch each", p, TAILS, true);
  }
  { // a second record, at other seeds, on the pads of one table
    Play p;
    p.records = 2;
    expect("two records on one table's pads", p, NOTHING, true);
  }
  { // two tables with different rows, fresh tails and the same pads
    Play p;
    p.makings = 2, p.other_rows = true;
    expect("two tables with the same pads", p, PADS, true);
  }
  std::printf("table_exact_test: %s (%s; %d of %d controls noticed)\n", failures ? "FAILED" : "ok", summary.c_str(), noticed, controls);
  return failures ? 1 : 0;
}
