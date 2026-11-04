// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Helpers for the exact tests of hiding (method and limits: exact.h), which compare the
// distribution of a prover's view (the field elements sent or opened) for two witnesses under equal
// challenges. Most are for whole sessions (the committer's run, then batches of requests) on a
// committed table (`Index` in identifiers), whose matrix is the witness.
#pragma once
#include <string>
#include <vector>

#include "iop/foldeq.h"
#include "iop/index_cpu.h"
#include "tests/hiding/exact.h"

namespace blk {

// A committed table under a constant name. Two matrices have two Merkle roots, and the root is in
// hash chains' first queries (a chain's starting input), on which all challenges depend. In the
// interactive protocol the root is an oracle and reveals nothing. So commitment() and name() return
// constant bytes. No verifier sees such a proof.
// A record stores answers to samples of each row, each masked by a random pad in that row. The
// samples' seeds are a function of name(), so record() computes the answers at constant seeds, from
// the real table's rows. No other record of such a table is made: two on the same pads would reveal
// linear functions of the rows. The answers vary with the coins (private random field elements) and
// the witness, and they too are in first queries. The caller keeps them out of these queries (not
// checked), by made_up_record() or by first_query_stands_still (blocks/chain/rounds.h), and adds
// them to the view (add_record()).
class NamedAs : public IndexHolder {
public:
  explicit NamedAs(IndexCpu &real) : real_(real) {}
  const IndexShape &shape() const override { return real_.shape(); }
  std::vector<u8> commitment() const override { return std::vector<u8>(real_.commitment().size(), 7); }
  std::vector<u8> name() const override { return std::vector<u8>(real_.name().size(), 7); }
  Digest configuration() const override { return real_.configuration(); }
  IndexRecord record() override {
    const IndexShape &s = real_.shape();
    const std::vector<SampleSeed> seeds = record_seeds(s, name());
    IndexRecord rec;
    std::vector<F> message(s.code.ell);
    for (u64 r = 0; r < s.rows(); r++) {
      for (size_t p = 0; p < message.size(); p++) message[p] = real_.message_entry(r, p);
      for (unsigned i = 0; i < s.answers; i++) rec.y.push_back(record_value(s, seeds[i], i, message.data()));
    }
    return rec;
  }
  bool begin(u64 columns) override { return real_.begin(columns); }
  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override { real_.fold(gamma, message, tail, blocks); }
  TableId fold_table(const E *gamma, const FoldedWant &want, BlockSums *blocks) override { return real_.fold_table(gamma, want, blocks); }
  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override { real_.open(at, file, columns); }
  void forget() override { real_.forget(); }

private:
  IndexCpu &real_;
};
// A constant record to pass as LinSessionOptions::query_record (blocks/iop/tests/lin_world.h).
inline IndexRecord made_up_record(const IndexShape &s) { return IndexRecord{std::vector<E>(s.rows() * s.answers, E::one())}; }

// The coins of a table's pads, for a control that sets them. A making draws its rows' random tails
// in one draw, then each row's pads; `first` is the number of the making's first draw.
inline std::vector<size_t> pad_coins(const Exact &x, const IndexShape &s, size_t first = 0) {
  BLK_REQUIRE(x.draws().size() > first + s.rows() && x.draws()[first].count == s.rows() * s.code.r, "a making draws its rows' random tails in one draw");
  std::vector<size_t> pads;
  for (u64 r = 0; r < s.rows(); r++) {
    const Exact::Draw &d = x.draws()[first + 1 + r];
    BLK_REQUIRE(d.count == s.pads(), "a making draws each row's pads in one draw, behind the tails");
    for (size_t i = 0; i < d.count; i++) pads.push_back(d.first + i);
  }
  return pads;
}

// The views of a session's chains, joined in the order of the add() calls. Byte offsets continue
// from one chain to the next.
struct SessionLook {
  Look all;
  size_t bytes_before = 0;
  void add(const char *chain, unsigned request, const Writer &bytes) { add(std::string(chain) + " chain of request " + std::to_string(request), bytes); }
  // A table's record is in its key (`IndexKey`): everybody sees it.
  void add_record(const IndexRecord &rec) {
    Writer answers;
    answers.keep_list();
    for (const E &y : rec.y) answers.put(y);
    add("the table's record", answers);
  }
  void add(const std::string &where, const Writer &bytes) {
    const Look l = look_at(bytes);
    all.marks.push_back(Writer::Mark{bytes_before, where});
    for (const Writer::Mark &m : l.marks) all.marks.push_back(Writer::Mark{bytes_before + m.at, where + ": " + m.words});
    for (size_t place : l.at) all.at.push_back(bytes_before + place);
    all.view.insert(all.view.end(), l.view.begin(), l.view.end());
    all.frame.insert(all.frame.end(), l.frame.begin(), l.frame.end());
    bytes_before += bytes.size();
  }
};

// Prints one line for a comparison; if `verbose`, first one for each draw of coins.
inline void print_comparison(const char *test, const char *name, const Exact &x, const Exact::Verdict &v, bool verbose) {
  if (verbose)
    for (size_t i = 0; i < x.draws().size(); i++) {
      size_t alone = 0;
      for (size_t c = x.draws()[i].first; c < x.draws()[i].first + x.draws()[i].count; c++) alone += v.coin_matters[c];
      std::printf("  %s: draw %zu, coins %zu to %zu: %s; %zu of its coins are needed one by one\n", name, i, x.draws()[i].first, x.draws()[i].first + x.draws()[i].count - 1,
                  v.draw_matters[i] ? "needed" : "could be constants", alone);
    }
  std::printf("%s: %s: %s (%zu coins in %zu draws; %zu entries of the view, rank %zu; %zu coins curved, %zu idle; needed: %zu coins one by one, %zu draws; %u steps)%s%s%s%s\n", test, name,
              v.same ? "the same law" : "NOT THE SAME LAW", x.coins(), x.draws().size(), x.entries(), v.rank, v.curved, v.idle, v.count(v.coin_matters), v.count(v.draw_matters), v.steps,
              v.same ? "" : ": ", v.why.c_str(), v.entry >= 0 ? "; " : "", x.where(v.entry).c_str());
}

// The witnesses that judge_session() expects of its `world`. 0 and 1: random matrices M0 and M1;
// 10: M0 + M1; 11: zeros.
template <class Matrix, class Make> Matrix session_matrix(unsigned witness, Make random) {
  if (witness < 10) return random(300 + witness);
  Matrix sum = random(300);
  const Matrix other = random(301);
  for (size_t i = 0; i < sum.a.size(); i++) sum.a[i] = witness == 10 ? sum.a[i] + other.a[i] : F::zero();
  return sum;
}

// What lin_exact_test.cpp and member_exact_test.cpp require of one case. True if all hold.
//  (1) Witnesses 0 and 1 give the view the same distribution ("the same law" in the output).
//  (2) This random pair speaks for all matrices, under the challenges that were run. Both views are
//      affine in the coins with one matrix J (Verdict::short_way). J's entries are polynomials in
//      the witness: if J depended on it, two random witnesses would share a J with probability
//      about the degree over |F|. At fixed coins the view is affine in the matrix, which is tested
//      only on this pair, at one list of coins and one of challenges:
//      view(M0 + M1) - view(M0) - view(M1) + view(0) = 0. So the view is J coins + L matrix + k,
//      and two matrices give the same distribution if and only if L times their difference is in
//      J's column space. Were there a pair of matrices without that, the chance that a pair drawn at
//      random has it would be 1 / |F| or less.
//  (3) The tails of the table's rows (random entries, encoded along with a row) are the first draw and are needed
//      (Verdict::draw_matters). So are the draws after it, one for each row's pads; makings_draws
//      counts all these draws.
//  (4) If `tight` (the tails hide exactly as many columns as are drawn), every coin of the tails is
//      needed. A list that draws a column twice leaves the tails one entry to spare, so up to three
//      lists of challenges are tried; (1), (3) and J's part of (2) are required under each.
inline bool judge_session(const char *test, const char *name, const World &world, bool tight, bool verbose, size_t &coins, size_t makings_draws) {
  for (u64 key = 1; key <= 3; key++) {
    Exact x(world, key);
    const Exact::Verdict v = x.compare(1);
    coins = x.coins();
    if (verbose || !v.same) print_comparison(test, name, x, v, verbose);
    if (!v.same) return false;
    if (!v.short_way) return std::printf("%s: %s: the view is not affine in the coins with one matrix for both witnesses: one pair of matrices speaks for itself alone\n", test, name), false;
    if (key == 1) {
      const std::vector<F> sum = x.view_of(10), a = x.view_of(0), b = x.view_of(1), zero = x.view_of(11);
      for (size_t r = 0; r < sum.size(); r++)
        if (sum[r] - a[r] - b[r] + zero[r] != F::zero()) return std::printf("%s: %s: the view is not affine in the matrix: one pair of matrices speaks for itself alone\n", test, name), false;
    }
    if (!v.draw_matters[0]) return std::printf("%s: %s: the table's random tails could be constants\n", test, name), false;
    for (size_t d = 1; d < makings_draws; d++)
      if (!v.draw_matters[d]) return std::printf("%s: %s: the pads of row %zu of the table could be constants\n", test, name, d - 1), false;
    if (!tight) return true;
    bool each = true;
    for (size_t c = x.draws()[0].first; c < x.draws()[0].first + x.draws()[0].count; c++) each = each && v.coin_matters[c];
    if (each) return true;
  }
  return std::printf("%s: %s: under none of three lists of challenges is every coin of the table's random tails needed, though the tails have none to spare\n", test, name), false;
}

} // namespace blk
