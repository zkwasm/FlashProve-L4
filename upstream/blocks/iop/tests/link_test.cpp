// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests of the block LINK (blocks/iop/link.h) at toy size, on the fixture (`world` in names) of
// blocks/iop/tests/link_world.h. A LINK proof ("link" in names and messages) is to show that a
// further committed table (`Index` in names; a commitment's leaf table) has the same entries as the
// first (the anchor) before the cut: the offset in a row's message where the pads begin. A table's
// record stores answers to samples (random linear forms) of its rows, each masked by a pad.
// The prover sends each table's folded row (its rows combined
// at the fold point gamma) as an oracle message: encoded, under a Merkle root. The linear claims
// ("rows" in names and messages) on the two folded rows include, for each table, one per draw of a
// column and one per answer that its record has per row, and the equality claim: both agree before
// the cut. The final proximity test is to reject ("refuse") a false claim (`LINK_REFUSED_TEST`).
//
// The program tests the honest prover only (see main()), so it does not show that LINK rejects a
// copy that differs before the cut; examples/protocol-toy-false-files/ holds one such stored file.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets a test write or alter a plan (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets a test declare a table tested (blocks/iop/foldeq.h)
#include "chain/tests/death.h"
#include "iop/tests/link_world.h"

using namespace blk;

// link_commit() and link_make() take a seed for their private randomness (`coins` in names).

// The cases cover both layouts of a folded row (high) and
// different numbers of drawn columns for the two tables. The columns are `LinkCase`'s fields. v: a
// table has 2^v rows; ans: its record's answers per row; c0, cj: columns drawn of
// the first and of the further table; links: the number of proofs for which the rows' random tails
// are sized. A random tail is the random coefficients that are encoded with a row's message and are
// to hide a bounded number of opened columns.
static const LinkCase CASES[] = {
    // name                              v data spare ans  c0  cj links rate  u   high  draws ahead
    {"rows low",                         2, 11,  2,    1,   3,  5,  2,    2,   8,  false, 3,   false},
    {"rows high",                        2, 12,  1,    1,   3,  4,  2,    2,   8,  true,  3,   false},
    {"eight rows",                       3,  9,  3,    1,   2,  6,  3,    3,  16,  true,  2,   false},
    {"two answers",                      1,  7,  2,    2,   4,  3,  2,    2,   8,  false, 2,   false},
    {"rows high, masks ahead",           2, 11,  2,    1,   3,  5,  2,    2,   8,  true,  3,   true},
    {"masks ahead, rows low",             2, 12,  1,    1,   5,  3,  2,    2,  16,  false, 3,   true},
    {"many columns, some drawn twice",   1,  5,  0,    1,  30, 40,  2,    2,  32,  true,  2,   false},
};
[[maybe_unused]] static const size_t SMALL = 3;  // the case whose proof is altered byte by byte

// A reader (`IndexReader`: what a verifier has of a table) and a keeper (`IndexHolder`: the
// prover's object that holds a table) that forward to a real one but do not override
// configuration(). The base class then returns zeros in place of the configuration digest from
// which the table's tree starts.
class [[maybe_unused]] SilentReader : public IndexReader {
 public:
  explicit SilentReader(const IndexKey &real) : real_(real) {}
  const IndexShape &shape() const override { return real_.shape(); }
  const IndexRecord &record() const override { return real_.record(); }
  std::vector<u8> commitment() const override { return real_.commitment(); }
  std::vector<u8> name() const override { return real_.name(); }
  bool read(Reader &file, const PositionList &at, std::vector<F> &columns, VerifierSession &session) const override { return real_.read(file, at, columns, session); }

 private:
  const IndexKey &real_;
};
class [[maybe_unused]] SilentKeeper : public IndexHolder {
 public:
  explicit SilentKeeper(IndexHolder &real) : real_(real) {}
  const IndexShape &shape() const override { return real_.shape(); }
  std::vector<u8> commitment() const override { return real_.commitment(); }
  std::vector<u8> name() const override { return real_.name(); }
  IndexRecord record() override { return real_.record(); }
  bool begin(u64 columns) override { return real_.begin(columns); }
  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override { real_.fold(gamma, message, tail, blocks); }
  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override { real_.open(at, file, columns); }

 private:
  IndexHolder &real_;
};

// ---------------------------------------------------------------------------------------------------------------------
// For each case: an honest proof is accepted; it is rejected under another key,
// record or configuration; an untested first table, or a shape (`LinkShape`) and a plan (`Plan`:
// the rounds and their error bounds) that do not fit each other, aborts the process; a table serves
// only as many proofs as its random tails allow. ends_with() (blocks/chain/tests/death.h) runs a
// call in a child process and is true if it aborts with the given text in its stderr.
int main() {
  size_t bytes_of_a_link = 0;
  for (const LinkCase &t : CASES) {
    const LinkWorld w = link_world(t);
    CHECK(!link_fault(w.codes, w.S, w.plan->set.samples), "an honest shape is refused");
    {  // malformed shapes: `link_fault` or `public_fault` must report each
      LinkShape bad = w.S;
      bad.later[0].k++;
      CHECK(link_fault(w.codes, bad, w.plan->set.samples) != nullptr, "a later stage that does not take the folded message exactly is admitted");
      bad = w.S, bad.rows = 1;
      CHECK(w.S.rows == 1 || link_fault(w.codes, bad, w.plan->set.samples) != nullptr, "an oracle message that is too small for a folded row is admitted");
      bad = w.S, bad.mask_code++;
      CHECK(link_fault(w.codes, bad, w.plan->set.samples) != nullptr, "a mask code that is not there is admitted");
      bad = w.S, bad.later.clear();
      CHECK(public_fault(bad) != nullptr && link_fault(w.codes, bad, w.plan->set.samples) != nullptr, "a test of one stage is admitted");
      bad = w.S, bad.columns_further = 0;
      CHECK(public_fault(bad) != nullptr, "a link that draws no column of the further table is admitted");
      CHECK(!public_fault(w.S), "an honest shape is refused");
      Codes short_masks = w.codes;  // the first stage's mask rows too short for the fold's hiding polynomial
      short_masks.mask[w.S.mask_code].ell = 2;
      CHECK(link_fault(short_masks, w.S, w.plan->set.samples) != nullptr, "mask rows that are too short for the hiding polynomial of the first fold are admitted");
    }
    const Table rows = link_rows(t, w, 11);
    LinkTable first = link_commit(w, rows, 21), further = link_commit(w, rows, 22);
    CHECK(first.key.commitment() != further.key.commitment(), "two makings of a table have one root");

    const LinkProof p = link_make(w, first, further, 31);
    const char *why = "";
    CHECK(p.made && link_check(w, first.key, further.key, p.bytes, &why), "an honest link is refused");
    if (failures) std::printf("%s: %s\n", t.name, why);
    bytes_of_a_link = p.bytes.size();
    CHECK(p.column_rows[0] == t.columns_first && p.column_rows[1] == t.columns_further, "not one row for each drawn column of each table");
    CHECK(p.other_false_rows == 0 && p.false_column_rows[0] + p.false_column_rows[1] == 0, "a row of an honest prover is false");
    {  // the pads that mask the samples of the two folded rows' oracle messages, not tables' pads:
       // a verifier cannot notice a constant pad, so the prover's memory is read
      LinkTable first_other = link_commit(w, rows, 21), further_other = link_commit(w, rows, 22);
      const LinkProof q = link_make(w, first_other, further_other, 32);  // the same tables, other coins
      CHECK(!p.pads.empty() && p.pads.size() % 2 == 0 && q.pads.size() == p.pads.size(), "a folded row's oracle message has no pad");
      for (size_t i = 0; i < p.pads.size() && i < q.pads.size(); i++) {
        CHECK(!is_zero(p.pads[i]) && !(p.pads[i] == q.pads[i]), "a pad is zero, or the same under other coins");
        for (size_t j = 0; j < i; j++) CHECK(!(p.pads[i] == p.pads[j]), "two pads of one run are equal");
      }
    }
    {  // the library's `link_prove`, on the same tables and coins
      LinkTable first_again = link_commit(w, rows, 21), further_again = link_commit(w, rows, 22);
      TestCoins coins(31);
      EngineCpu engine;
      Writer bytes;
      CHECK(link_prove(w.plan, w.configuration, w.codes, w.S, *first_again.index, *further_again.index, engine, coins, bytes) && bytes.bytes() == p.bytes,
            "link_prove gives other bytes than the tests' walk through its steps");
    }
    {  // when `link_verify` reads the first table's columns it must abort, even on an honest proof,
       // unless that table has passed its test under the plan's `Setting`: the proof's error bound
       // needs that (blocks/iop/foldeq.h). link_check() runs no such test: it vouches for the first
       // table by TableRead::on_a_tests_word(). The further table is under test: any reader serves.
      Setting weaker = w.plan->set;
      weaker.target--;
      auto link_on = [&](const TableRead &first_table) {
        Reader bytes(p.bytes.data(), p.bytes.size());
        VerifierSession session;
        (void)link_verify(w.plan, w.configuration, w.codes, w.S, first_table, further.key, bytes, session);
      };
      CHECK(ends_with("this reader has passed none", [&] { link_on(first.key); }), "a link is checked on a first table that its keeper has handed out");
      CHECK(ends_with("tested under another setting", [&] { link_on(TableRead::on_a_tests_word(first.key, weaker)); }), "a link is checked on a first table that was tested at a lower target");
    }
    {  // keys that do not fit are rejected, and the process does not abort
      IndexKey wrong = further.key;
      wrong.table.v++;
      CHECK(!link_check(w, first.key, wrong, p.bytes), "a key of a table with other rows is taken");
      CHECK(!link_check(w, further.key, first.key, p.bytes), "a link is accepted with the two tables' places changed");
      wrong = further.key, wrong.rec.y.pop_back();
      CHECK(!link_check(w, first.key, wrong, p.bytes), "a record that is cut short is taken");
      wrong = first.key, wrong.rec.y[0] += E::one();
      CHECK(!link_check(w, wrong, further.key, p.bytes), "a link is accepted under another record of the first table");
    }
    // Two tables' pads are drawn separately, so the equality claim extended to the message's end
    // (`long_row_holds`) is false.
    CHECK(!p.long_row_holds, "the equality row run on to the message's end: not as foretold");
    CHECK(w.S.cut() == w.S.table.code.ell - 3 * (size_t)t.answers, "the cut is not where the pads begin");

    {
      LinkWorld elsewhere = link_world(t);
      CHECK(link_check(elsewhere, first.key, further.key, p.bytes), "control: a second making of the same world refuses");
      elsewhere.configuration.w[1] ^= 1;
      const char *said = "";
      CHECK(!link_check(elsewhere, first.key, further.key, p.bytes, &said), "a link is accepted under another configuration");
      CHECK(std::string(said) == "a key's tree starts from another configuration's digest than the chain names", "the refusal comes from another step than foretold");
      LinkTable made_elsewhere = link_commit(elsewhere, rows, 22);  // the same rows and coins, under the other digest: another root, another name
      CHECK(!(made_elsewhere.key.root == further.key.root) && !link_check(w, first.key, made_elsewhere.key, p.bytes), "a copy made under another configuration is taken");
    }
    {  // the base classes' zeros: such a reader is rejected, such a keeper aborts the prover
      const SilentReader silent_first(first.key), silent_further(further.key);
      for (int place = 0; place < 2; place++) {
        Reader bytes(p.bytes.data(), p.bytes.size());
        VerifierSession session;
        LinkWhy said;
        CHECK(!link_verify(w.plan, w.configuration, w.codes, w.S, place ? (const IndexReader &)first.key : silent_first, place ? (const IndexReader &)silent_further : further.key, bytes, session, &said),
              "a link is accepted from a reader that names no configuration");
        CHECK(std::string(said.step) == LINK_REFUSED_KEYS && std::string(said.reason) == "a key's tree starts from another configuration's digest than the chain names",
              "the refusal of a reader that names no configuration comes from another step than foretold");
      }
      for (int place = 0; place < 2; place++)
        CHECK(ends_with("LINK: a table's tree starts from another configuration's digest than the chain names", [&] {
                LinkTable a = link_commit(w, rows, 21), b = link_commit(w, rows, 22);
                SilentKeeper silent(place ? *b.index : *a.index);
                TestCoins coins(31);
                EngineCpu engine;
                Writer out;
                (void)link_prove(w.plan, w.configuration, w.codes, w.S, place ? (IndexHolder &)*a.index : silent, place ? (IndexHolder &)silent : *b.index, engine, coins, out);
              }),
              "a link is made on a keeper that names no configuration");
    }
    {  // the shape splits the drawn columns differently from the plan, same total: both sides abort
      LinkWorld other = w;
      other.S.columns_first += 1, other.S.columns_further -= 1;
      const char *const words = "rounds: the round's first list has another number of draws than its planned error counts";
      CHECK(!link_fault(other.codes, other.S, other.plan->set.samples), "control: the shape with the other cut is a shape");
      CHECK(ends_with(words, [&] { (void)link_check(other, first.key, further.key, p.bytes); }), "a verifier runs under a shape that cuts the drawn columns otherwise than the plan");
      CHECK(ends_with(words, [&] {
              LinkTable a = link_commit(w, rows, 21), b = link_commit(w, rows, 22);
              (void)link_make(other, a, b, 31);
            }),
            "a prover runs under a shape that cuts the drawn columns otherwise than the plan");
    }
    for (int bend = 0; bend < 12; bend++) {  // plans that do not fit the table: both sides abort
      // The altered round names another table than the shape's (`how` 0 to 3: one message entry,
      // tail entry, row variable or answer more), no table (4) or another test (5).
      const bool at_the_point = bend >= 6;  // both rounds name the table; each check works alone
      const int how = bend % 6;
      LinkWorld other = w;
      Plan bent(w.plan);
      for (PlanRound &r : bent.rounds_)
        if (r.name == (at_the_point ? R_LINK_POINT : R_LINK)) {
          r.table.code.ell += how == 0, r.table.code.r += how == 1, r.table.v += how == 2, r.table.answers += how == 3, r.table.said = how != 4;
          if (how == 5) r.table.test = TableTest::LATER;
        }
      other.plan = bent;
      const char *const words = "rounds: the round draws columns of another table than its planned error is priced for, or the plan names none";
      CHECK(ends_with(words, [&] { (void)link_check(other, first.key, further.key, p.bytes); }), "a verifier runs under a plan that is priced for another table");
      CHECK(ends_with(words, [&] {
              LinkTable a = link_commit(w, rows, 21), b = link_commit(w, rows, 22);
              (void)link_make(other, a, b, 31);
            }),
            "a prover runs under a plan that is priced for another table");
    }
    LinkTable another = link_commit(w, rows, 23), another_first = link_commit(w, rows, 24);
    CHECK(!link_check(w, first.key, another.key, p.bytes), "a link is accepted under another further table's key");
    CHECK(!link_check(w, another_first.key, further.key, p.bytes), "a link is accepted under another first table's key");

    unsigned served = 1;  // LINK proofs that the first table has served so far
    for (; served < t.links; served++) {
      LinkTable next = link_commit(w, rows, 40 + served);
      CHECK(link_make(w, first, next, 50 + served).made, "the first table's keeper refuses too early");
    }
    if ((u64)t.columns_first * (t.links + 1) > hiding_positions(w.S.table.code)) {
      LinkTable next = link_commit(w, rows, 60);
      CHECK(!link_make(w, first, next, 61).made, "the first table's keeper serves more links than its random tails bear");
    }
    {  // `link_prove` itself must count both tables' columns against their random tails.
       // The loop above makes its proofs step by step through LinkAccess::prove()
       // (blocks/iop/tests/link_world.h), which calls begin() itself and so does not test
       // Link::prove()'s calls. This block fails if Link::prove() leaves out first.begin() or
       // counts one column too few for the further table. No other test in blocks/ does: none reads
       // columns_left(), not even blocks/frame/tests/check_entry_test.cpp, which also goes through
       // Link::prove(). By design, IndexCpu::open() does not check the count, which guards only
       // callers of begin() (blocks/tests/hiding/table_exact_test.cpp,
      // the control "one batch more than the random entries cover").
      LinkTable a = link_commit(w, rows, 71);
      const u64 whole = a.index->columns_left();
      unsigned made = 0;
      for (;; made++) {
        LinkTable b = link_commit(w, rows, 80 + made);
        const u64 left_a = a.index->columns_left(), left_b = b.index->columns_left();
        TestCoins coins(90 + made);
        EngineCpu engine;
        Writer proof;
        const bool served = link_prove(w.plan, w.configuration, w.codes, w.S, *a.index, *b.index, engine, coins, proof);
        if (!served) {
          CHECK(left_a < w.S.columns_first, "link_prove refuses while the first table's random tails bear a further link");
          CHECK(proof.bytes().empty() && a.index->columns_left() == left_a, "a link that is refused writes bytes or spends columns of the first table");
          break;
        }
        CHECK(a.index->columns_left() == left_a - w.S.columns_first, "link_prove does not count the first table's columns");
        CHECK(b.index->columns_left() == left_b - w.S.columns_further, "link_prove does not count the further table's columns");
        CHECK(link_check(w, a.key, b.key, proof.bytes()), "a link that link_prove has made is refused");
        if (made > 64) break;  // the tails allow many more proofs: these are enough
      }
      CHECK(made > 64 || made == whole / w.S.columns_first, "link_prove serves another number of links than the first table's random tails bear");
    }
  }

  {  // CASES[SMALL]: one bit flipped in each byte in turn; one byte appended; one byte cut off
    const LinkCase &t = CASES[SMALL];
    const LinkWorld w = link_world(t);
    const Table rows = link_rows(t, w, 11);
    LinkTable first = link_commit(w, rows, 21), further = link_commit(w, rows, 22);
    std::vector<u8> b = link_make(w, first, further, 31).bytes;
    size_t accepted = 0, tried = 0;
    for (size_t i = 0; i < b.size(); i++) {
      b[i] ^= (u8)(1u << (i % 8));
      accepted += link_check(w, first.key, further.key, b), tried++;
      b[i] ^= (u8)(1u << (i % 8));
    }
    b.push_back(0);
    accepted += link_check(w, first.key, further.key, b), tried++;
    b.pop_back(), b.pop_back();
    accepted += link_check(w, first.key, further.key, b), tried++;
    if (accepted) std::printf("%zu of %zu altered links are accepted\n", accepted, tried);
    CHECK(accepted == 0, "an altered link is accepted");
  }
  std::printf("%s: %zu cases, last link %zu bytes\n", failures ? "FAILED" : "ok", sizeof CASES / sizeof CASES[0], bytes_of_a_link);
  return failures != 0;
}
