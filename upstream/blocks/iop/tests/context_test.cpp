// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/iop/context.h: which Merkle tree an oracle message (encoded main and mask rows) is
// put in, and what an opening at a list (the sorted drawn positions) returns. A proof with one bit
// flipped must be rejected (`refused` in messages); layouts and runs that break one of that
// header's rules must abort the process (BLK_REQUIRE). Each rule tried here that both
// open_oracles() (prover) and read_opened() (verifier) check is tried on both sides.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // allows plans built by hand (blocks/numbers/plan.h)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "chain/session.h"
#include "chain/tests/death.h"
#include "engine/engine_cpu.h"
#include "iop/context.h"
#include "iop/send.h"
#include "toy/toy_coins.h"

using namespace blk;

static Digest made_up_configuration() {
  Digest d{};
  d.w[0] = 77;
  return d;
}
static Table random_table(Coins &coins, unsigned deg, size_t rows, size_t len) {
  Table t(deg, rows, len);
  coins.fill(t.a.data(), t.a.size());
  return t;
}
// One row's symbol at position x of 2^list_log_n; a shorter codeword is read at x mod its length.
static E symbol_of(const Code &code, const Table &message, const Table &tail, size_t row, unsigned list_log_n, u64 x) {
  std::vector<E> m(code.ell), t(code.r);
  for (size_t j = 0; j < code.ell; j++) m[j] = message.at(row, j);
  for (size_t j = 0; j < code.r; j++) t[j] = tail.at(row, j);
  return symbol_at<E>(code, short_position(list_log_n, code.log_n, x), m.data(), t.data());
}

// [own-test-mask-rows] Private randomness that records every fill(), so that it can answer
// last_draw() and not_the_coins(). It can alter one value of one draw after recording it, as a
// stray memory write would. not_the_coins() returns 0 or 1, not a count, and ignores `left_out`,
// which is empty wherever this generator is used: the messages of one_case() have no pad.
struct NumberedCoins : ToyCoins {
  using ToyCoins::ToyCoins;
  std::vector<std::vector<F>> drawn;
  long loses = -1;          // number of the fill() call of which one value is altered; -1: none
  bool last_value = false;  // the altered value is the draw's last, not its first
  void fill(F *out, size_t n) override {
    ToyCoins::fill(out, n);
    drawn.emplace_back(out, out + n);
    if (n && (long)drawn.size() - 1 == loses) out[last_value ? n - 1 : 0] += F::one();
  }
  u32 last_draw() const override { return (u32)drawn.size() - 1; }
  u64 not_the_coins(u32 draw, const F *values, u64 n, const std::vector<u64> &) const override {
    u64 differ = drawn.at(draw).size() != n;
    for (u64 i = 0; i < n && !differ; i++) differ += !(values[i] == drawn[draw][i]);
    return differ;
  }
};

// Sends `sent` in one prover message, opens those numbered in `ask` at one list, and verifies.
// flip < 0: an honest run; returns 0 if accepted with both sides' symbols equal to the encodings.
// flip >= 0: the low bit of proof byte flip (mod the length) is flipped; returns 0 if rejected.
// numbered: null, or the prover's private randomness in place of ToyCoins(key). noted: null, or
// where to put the numbers that the context has noted for each sent message with mask rows: of the
// draw that filled the rows and of the draw that filled their random tail.
static int one_case(const char *name, const Codes &codes, const std::vector<OracleNotice> &sent, unsigned list_log_n,
                    const std::vector<size_t> &ask, u64 key, long flip, NumberedCoins *numbered = nullptr, std::vector<u32> *noted = nullptr) {
  const Setting set = setting_goldilocks();
  Plan plan(set);
  for (size_t i = 0; i < tree_sizes(codes, sent).size(); i++) plan.send_root("root");
  PlanRound &round = plan.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  round.draw_positions = 5, round.log_n = list_log_n;

  ToyCoins unnumbered(key);
  Coins &coins = numbered ? static_cast<Coins &>(*numbered) : unnumbered;
  EngineCpu engine;
  Writer proof;
  Binder prover_binder(9, plan);
  prover_binder.configuration(made_up_configuration());
  const Public<Codes> my_codes = prover_binder.parameter("codes", codes);
  prover_binder.parameter("the oracle messages", sent);
  ProverRounds pch(prover_binder, random_digest(coins), proof);
  ProverContext pr(my_codes, pch, engine, coins);
  std::vector<OracleDraft> drafts;
  std::vector<Table> mains, masks;
  for (const OracleNotice &n : sent) {
    OracleDraft d;
    d.shape = n.shape, d.leaf_class = n.leaf_class;
    Table m;
    if (n.shape.main_rows) m = random_table(coins, n.shape.main_deg, n.shape.main_rows, codes.base[n.shape.code].ell), d.main = engine.put(m);
    if (n.shape.mask_rows) d.mask = random_table(coins, E_DEGREE, n.shape.mask_rows, codes.mask[n.shape.mask_code].ell), d.mask_draw = coins.last_draw();
    mains.push_back(m), masks.push_back(d.mask);
    drafts.push_back(std::move(d));
  }
  const std::vector<OracleId> ids = send_oracles(pr, std::move(drafts));
  if (noted)
    for (OracleId o : ids)
      if (pr.st.oracle(o).shape.mask_rows) noted->push_back(pr.behind[o].mask_draw), noted->push_back(pr.behind[o].tail_draw);
  const PositionList at = pch.close_positions("positions").positions(list_log_n);
  std::vector<OracleId> asked;
  for (size_t a : ask) asked.push_back(ids[a]);
  Opened mine;
  open_oracles(pr, asked, at, mine);
  pch.finish();

  std::vector<u8> bytes = proof.bytes();
  if (flip >= 0) bytes[(size_t)flip % bytes.size()] ^= 1;
  Reader reader(bytes.data(), bytes.size());
  Binder verifier_binder(9, plan);
  verifier_binder.configuration(made_up_configuration());
  const Public<Codes> their_codes = verifier_binder.parameter("codes", codes);
  verifier_binder.parameter("the oracle messages", sent);
  VerifierSession session;
  VerifierRounds vch(verifier_binder, reader, session);
  VerifierContext vf(their_codes, vch);
  std::vector<OracleId> their_ids;
  Opened theirs;
  bool accepted = receive_oracles(vf, sent, their_ids);
  const PositionList their_list = vch.close_positions("positions").positions(list_log_n);
  accepted = accepted && read_opened(vf, asked, their_list, theirs);
  accepted = vch.finish() && accepted;
  if (flip >= 0) return accepted ? (std::printf("%s: ACCEPTED with byte %ld changed\n", name, flip), 1) : 0;
  if (!accepted) return std::printf("%s: the honest opening is refused: %s\n", name, vch.why()), 1;
  if (their_list != at) return std::printf("%s: the two sides' lists differ\n", name), 1;

  for (size_t w = 0; w < asked.size(); w++) {
    const OracleShape &s = sent[ask[w]].shape;
    const TreeBook::Place place = pr.book.place(asked[w]);
    for (size_t i = 0; i < at.size(); i++) {
      const Opened::Column &a = mine.of[w][i], &b = theirs.of[w][i];
      if (b.main.size() != s.main_rows || b.mask.size() != s.mask_rows) return std::printf("%s: a wrong number of rows\n", name), 1;
      for (size_t row = 0; row < s.main_rows; row++) {
        const E symbol = symbol_of(codes.base[s.code], mains[ask[w]], engine.tail(pr.commit[place.tree], place.part), row, list_log_n, at[i]);
        if (symbol != a.main[row] || symbol != b.main[row]) return std::printf("%s: a wrong main symbol\n", name), 1;
      }
      for (size_t row = 0; row < s.mask_rows; row++) {
        const E symbol = symbol_of(codes.mask[s.mask_code], masks[ask[w]], pr.behind[asked[w]].mask_tail, row, list_log_n, at[i]);
        if (symbol != a.mask[row] || symbol != b.mask[row]) return std::printf("%s: a wrong mask symbol\n", name), 1;
      }
    }
  }
  return 0;
}

// deg 1: main rows over F; deg 3: over the extension field E. Only oracle messages of one
// leaf_class share a tree.
static OracleNotice notice(const char *name, CodeId code, size_t rows, unsigned deg, size_t mask_rows, CodeId mask_code, unsigned leaf_class) {
  OracleNotice n;
  n.shape.name = name, n.shape.code = code, n.shape.main_rows = rows, n.shape.main_deg = deg;
  n.shape.mask_rows = mask_rows, n.shape.mask_code = mask_code, n.shape.answers = 0;
  n.leaf_class = leaf_class;
  return n;
}

// The mask of one step (an oracle message without an out-of-domain sample: `answers` = 0) shares
// a tree with a message whose shape has a sample. Such a tree may be opened once, and by one hash
// chain (Fiat-Shamir transcript). who: 0 = the prover opens the tree at two lists; 1 = the
// verifier reads it at two; 2 = another `TreeBook` adopts the tree.
static void wrong_layout(const Codes &codes, int who) {
  const Setting set = setting_goldilocks();
  OracleNotice rows = notice("a", 0, 3, 1, 1, 0, 0);
  rows.shape.answers = 1;
  const std::vector<OracleNotice> sent = {rows, notice("the mask of one step", NONE_ID, 0, 1, 1, 0, 0)};
  Plan plan(set);
  plan.send_root("root");
  for (const char *name : {"first list", "second list"}) {
    PlanRound &round = plan.close(name, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    round.draw_positions = 3, round.log_n = 8;
  }
  ToyCoins coins(1);
  EngineCpu engine;
  Writer proof;
  Binder prover_binder(9, plan);
  prover_binder.configuration(made_up_configuration());
  const Public<Codes> my_codes = prover_binder.parameter("codes", codes);
  ProverRounds pch(prover_binder, random_digest(coins), proof);
  ProverContext pr(my_codes, pch, engine, coins);
  std::vector<OracleDraft> drafts(2);
  drafts[0].shape = sent[0].shape, drafts[0].main = engine.put(random_table(coins, 1, 3, codes.base[0].ell));
  drafts[0].mask = random_table(coins, E_DEGREE, 1, codes.mask[0].ell);
  drafts[1].shape = sent[1].shape, drafts[1].mask = random_table(coins, E_DEGREE, 1, codes.mask[0].ell);
  const std::vector<OracleId> ids = send_oracles(pr, std::move(drafts));
  Opened opened;
  open_oracles(pr, ids, pch.close_positions("first list").positions(8), opened);
  const PositionList second = pch.close_positions("second list").positions(8);
  if (who == 0) open_oracles(pr, ids, second, opened);
  pch.finish();
  if (who == 2) {
    Statement other(codes);
    TreeBook book(set.salt_bytes, pch.run(), pch.configuration());
    std::vector<OracleId> theirs;
    book.adopt(book.enter(other, sent, theirs), pr.book.tree(0));
  }
  Reader reader(proof.data(), proof.size());
  Binder verifier_binder(9, plan);
  verifier_binder.configuration(made_up_configuration());
  const Public<Codes> their_codes = verifier_binder.parameter("codes", codes);
  VerifierSession session;
  VerifierRounds vch(verifier_binder, reader, session);
  VerifierContext vf(their_codes, vch);
  std::vector<OracleId> got;
  if (!receive_oracles(vf, sent, got) || !read_opened(vf, got, vch.close_positions("first list").positions(8), opened)) return;
  (void)read_opened(vf, got, vch.close_positions("second list").positions(8), opened);
}

// Random main rows for SEND (blocks/iop/send.h), which sends oracle messages and takes their
// out-of-domain samples. The pad: the entries with which SEND masks the samples' answers.
static Table table_with_pad(const Codes &codes, const OracleShape &shape, Coins &coins) {
  Table t = random_table(coins, shape.main_deg, shape.main_rows, codes.base[shape.code].ell);
  if (!shape.mask_rows) // the pad is in the main rows: its places arrive empty (0), SEND draws them
    for (const PadPlace &place : Send::pad(codes, shape)) t.set_entry(place.index, E::zero());
  return t;
}

// Makes and verifies an honest proof. If `planted`, the session's record of the values seen under
// the tree's root first gets another value at the root's left child. Returns the verifier's
// reason, or nullptr if it accepts.
static const char *remembers(const Codes &codes, bool planted) {
  const Setting set = setting_goldilocks();
  const std::vector<OracleNotice> sent = {notice("a", 0, 3, 1, 1, 0, 0)};
  Plan plan(set);
  plan.send_root("root");
  PlanRound &round = plan.close("list", RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
  round.draw_positions = 7, round.log_n = 8; // the root's two children are always computed or given, whatever is drawn
  ToyCoins coins(5);
  EngineCpu engine;
  Writer proof;
  Binder prover_binder(9, plan);
  prover_binder.configuration(made_up_configuration());
  const Public<Codes> my_codes = prover_binder.parameter("codes", codes);
  ProverRounds pch(prover_binder, random_digest(coins), proof);
  ProverContext pr(my_codes, pch, engine, coins);
  std::vector<OracleDraft> drafts(1);
  drafts[0].shape = sent[0].shape, drafts[0].main = engine.put(random_table(coins, 1, 3, codes.base[0].ell));
  drafts[0].mask = random_table(coins, E_DEGREE, 1, codes.mask[0].ell);
  const std::vector<OracleId> ids = send_oracles(pr, std::move(drafts));
  Opened opened;
  open_oracles(pr, ids, pch.close_positions("list").positions(8), opened);
  pch.finish();
  Reader reader(proof.data(), proof.size());
  Binder verifier_binder(9, plan);
  verifier_binder.configuration(made_up_configuration());
  const Public<Codes> their_codes = verifier_binder.parameter("codes", codes);
  VerifierSession session;
  VerifierRounds vch(verifier_binder, reader, session);
  VerifierContext vf(their_codes, vch);
  std::vector<OracleId> got;
  static std::string words;
  bool ok = receive_oracles(vf, sent, got);
  if (ok && planted) (void)session.record(vf.book.tree(0).root).put(1, 0, made_up_configuration());
  ok = ok && read_opened(vf, got, vch.close_positions("list").positions(8), opened);
  (void)vch.finish();
  if (ok && session.accepted()) return nullptr;
  words = vch.why() ? vch.why() : "no reason given";
  return words.c_str();
}

// Checks in blocks/iop/context.h that protect hiding (`guard` in main()), each with a layout or
// run that breaks its rule. No verifier and no honest run can see a breach: a leaf opened at a list
// that is not its own, or a row opened at more positions than its random tail (the r random
// entries encoded with it) hides, gives secrets away in proofs that are accepted. So main() names
// each check by its message: without the check, its case fails.
//
// Oracle messages a and b share a tree; SEND samples them so that they can join groups (a group:
// the oracle messages that one proximity test reads at one list).
// what: 0 = a and b join different groups; 1 = both join two groups, and the mask of one step
// shares their tree; 2 = b is left out of the opening; 3 (prover only) = list after list opens the
// tree until the positions exceed what a random tail hides: the main rows' (mask_rows = 0), or
// that of a's mask row where `codes` makes it the shorter.
static void breach(const Codes &codes, int what, bool verifier_side, CodeId main_code, size_t mask_rows) {
  Setting set = setting_goldilocks();
  OracleNotice a = notice("a", main_code, 2, 1, mask_rows, 0, 0), b = notice("b", main_code, 1, 1, 0, 0, 0);
  a.shape.answers = b.shape.answers = set.samples;
  std::vector<OracleNotice> sent = {a, b};
  if (what == 1) sent.push_back(notice("the mask of one step", NONE_ID, 0, 1, 1, 0, 0));
  const unsigned log_n = codes.base[main_code].log_n;
  const unsigned lists = what == 3 ? 4 : 1;
  Plan plan(set);
  plan_send(plan, "a, b", {SampleSize{Send::cube(codes, a.shape).bits, no_sized_group()}, SampleSize{Send::cube(codes, b.shape).bits, no_sized_group()}}, 1);
  for (unsigned l = 0; l < lists; l++) {
    PlanRound &round = plan.close("list " + std::to_string(l), RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
    round.draw_positions = 7, round.log_n = log_n;
  }
  auto groups = [&](Statement &st, const std::vector<OracleId> &ids) {
    const GroupId g = st.new_group("first", main_code, 0), h = st.new_group("second", main_code, 0);
    st.join(g, ids[0]);
    st.join(what == 0 ? h : g, ids[1]);
    if (what == 1) st.join(h, ids[0]), st.join(h, ids[1]);
  };
  ToyCoins coins(3);
  EngineCpu engine;
  Writer proof;
  Binder prover_binder(9, plan);
  prover_binder.configuration(made_up_configuration());
  const Public<Codes> my_codes = prover_binder.parameter("codes", codes);
  ProverRounds pch(prover_binder, random_digest(coins), proof);
  ProverContext pr(my_codes, pch, engine, coins);
  std::vector<OracleDraft> drafts(sent.size());
  for (size_t i = 0; i < sent.size(); i++) {
    drafts[i].shape = sent[i].shape;
    if (sent[i].shape.main_rows) drafts[i].main = engine.put(table_with_pad(codes, sent[i].shape, coins));
    if (sent[i].shape.mask_rows)
      drafts[i].mask = sent[i].shape.answers ? Send::fresh_mask(codes, sent[i].shape, coins) // leaves the pad's places empty
                                             : random_table(coins, E_DEGREE, sent[i].shape.mask_rows, codes.mask[0].ell);
  }
  const std::vector<OracleId> ids = send_prove(pr, std::move(drafts));
  // To reach the verifier's checks the prover must pass its own: it joins no groups and opens all.
  if (!verifier_side) groups(pr.st, ids);
  Opened opened;
  for (unsigned l = 0; l < lists; l++) {
    const PositionList at = pch.close_positions("list " + std::to_string(l)).positions(log_n);
    std::vector<OracleId> ask = ids;
    if (what == 2 && !verifier_side) ask.erase(ask.begin() + 1);
    open_oracles(pr, ask, at, opened);
  }
  pch.finish();
  if (!verifier_side) return;
  Reader reader(proof.data(), proof.size());
  Binder verifier_binder(9, plan);
  verifier_binder.configuration(made_up_configuration());
  const Public<Codes> their_codes = verifier_binder.parameter("codes", codes);
  VerifierSession session;
  VerifierRounds vch(verifier_binder, reader, session);
  VerifierContext vf(their_codes, vch);
  std::vector<OracleId> got;
  if (!send_verify(vf, sent, got)) return;
  groups(vf.st, got);
  std::vector<OracleId> ask = got;
  if (what == 2) ask.erase(ask.begin() + 1);
  (void)read_opened(vf, ask, vch.close_positions("list 0").positions(log_n), opened);
}

int main() {
  Codes codes;
  // Code{log_n, ell, r}: length 2^log_n, ell message entries, r random tail entries;
  // hiding_positions() returns r
  codes.base = {Code{8, 16, 21}, Code{7, 8, 21}, Code{8, 8, 21}};
  codes.mask = {Code{6, 4, 21}, Code{7, 4, 21}, Code{8, 4, 21}};
  Codes short_mask_tail = codes; // its main rows' tails hide all 28 positions of four lists, its first mask code's tail does not
  short_mask_tail.base[0] = Code{8, 16, 41}, short_mask_tail.mask[0] = Code{6, 4, 11};
  for (const std::vector<Code> *list : {&codes.base, &codes.mask})
    for (const Code &code : *list)
      if (!code_ok(code)) return std::printf("FAILED: the test's numbers are not a code in this setting\n"), 1;
  struct Case {
    const char *name;
    std::vector<OracleNotice> sent;
    unsigned log_n;
    std::vector<size_t> ask;
  };
  const std::vector<Case> cases = {
      {"rows over F with a mask row and rows over E with two, in one tree", {notice("a", 0, 3, 1, 1, 0, 0), notice("b", 0, 2, 3, 2, 0, 0)}, 8, {0, 1}},
      {"the same, asked for in reverse and one of them twice", {notice("a", 0, 3, 1, 1, 0, 0), notice("b", 0, 2, 3, 2, 0, 0)}, 8, {1, 0, 1}},
      {"mask rows alone, sent first, ride in a later tree",
       {notice("m", NONE_ID, 0, 1, 2, 0, 0), notice("a", 0, 3, 1, 0, 0, 0), notice("b", 2, 1, 3, 1, 0, 0)}, 8, {0, 1, 2}},
      {"mask rows alone in a tree of their own", {notice("m", NONE_ID, 0, 1, 2, 0, 0)}, 8, {0}},
      {"three leaf classes",
       {notice("a", 0, 3, 1, 1, 0, 0), notice("b", 0, 2, 3, 2, 0, 1), notice("m", NONE_ID, 0, 1, 1, 0, 1), notice("n", NONE_ID, 0, 1, 1, 1, 2)}, 8, {3, 2, 1, 0}},
      {"a shorter tree in the same class", {notice("a", 0, 3, 1, 1, 0, 0), notice("s", 1, 2, 3, 0, 0, 0)}, 8, {1, 0}},
      {"mask rows as long as the main rows", {notice("s", 1, 2, 3, 1, 1, 0)}, 7, {0}},
  };
  int bad = 0;
  for (const Case &c : cases)
    for (u64 key = 1; key <= 3; key++) bad += one_case(c.name, codes, c.sent, c.log_n, c.ask, key, -1);
  long flips = 0;
  for (; flips < 6000; flips++) bad += one_case("three leaf classes", codes, cases[4].sent, cases[4].log_n, cases[4].ask, 4, flips);
  for (int who : {0, 1})
    if (!ends_with("a tree that holds the mask of one step is opened at a second list", [&] { wrong_layout(codes, who); }))
      bad += 1, std::printf("%s goes on where the mask of one step is shown at a second list\n", who ? "the verifier" : "the prover");
  if (!ends_with("a tree that holds the mask of one step is read in no other chain", [&] { wrong_layout(codes, 2); }))
    bad += 1, std::printf("a tree that holds the mask of one step can be adopted into another chain\n");
  // counts the three wrong_layout() cases above
  int guards = 3;
  auto guard = [&](const char *words, auto &&run) {
    guards++;
    if (!ends_with(words, run)) bad += 1, std::printf("NO STOP: %s\n", words);
  };
  guard("mask rows of two lengths in one tree", [&] { (void)lay_trees(codes, {notice("a", 0, 1, 1, 1, 0, 0), notice("b", 0, 1, 1, 1, 1, 0)}); });
  guard("mask rows longer than the main rows above them", [&] { (void)lay_trees(codes, {notice("s", 1, 1, 1, 1, 2, 0)}); });
  guard("a tree is adopted with all the oracle messages that lie in it", [&] {
    OracleNotice a = notice("a", 0, 1, 1, 0, 0, 0), b = notice("b", 0, 1, 1, 0, 0, 0);
    a.shape.answers = b.shape.answers = 1;
    Digest some = made_up_configuration();
    Statement theirs(codes), mine(codes);
    TreeBook from(40, some, some), to(40, some, some);
    std::vector<OracleId> ids;
    const u32 whole = from.enter(theirs, {a, b}, ids);
    to.adopt(to.enter(mine, {a}, ids), from.tree(whole));
  });
  for (bool verifier_side : {false, true}) {
    guard("oracle messages that share a leaf belong to different groups", [&] { breach(codes, 0, verifier_side, 0, 1); });
    guard("the mask of one step shares a leaf with an oracle message that is read at several lists", [&] { breach(codes, 1, verifier_side, 0, 1); });
    guard("an oracle message that shares a tree with those asked for is left out", [&] { breach(codes, 2, verifier_side, 0, 1); });
  }
  guard("main rows are opened at more positions than their random tail hides", [&] { breach(codes, 3, false, 0, 0); });
  guard("mask rows are opened at more positions than their random tail hides", [&] { breach(short_mask_tail, 3, false, 0, 1); });
  // An adopted tree shares the counts of its openings and of its opened positions.
  {
    OracleNotice a = notice("a", 0, 1, 1, 0, 0, 0);
    a.shape.answers = 1;
    Digest some = made_up_configuration();
    Statement theirs(codes), mine(codes);
    TreeBook from(40, some, some), to(40, some, some);
    std::vector<OracleId> ids;
    const u32 sent_there = from.enter(theirs, {a}, ids);
    from.set_root(sent_there, some);
    const u32 here = to.enter(mine, {a}, ids);
    to.adopt(here, from.tree(sent_there));
    if (to.tree(here).lists != from.tree(sent_there).lists || to.tree(here).positions != from.tree(sent_there).positions)
      bad += 1, std::printf("an adopted tree has counts of its own\n");
  }
  // read_opened() compares every opening with the session's record of its root: with `planted` the
  // honest proof must be rejected with "two values under one root", without it accepted. A
  // verifier that ignored the record would accept both.
  for (bool planted : {false, true}) {
    const char *why = remembers(codes, planted);
    if (!planted && why) bad += 1, std::printf("the honest opening is refused: %s\n", why);
    if (planted && (!why || !std::strstr(why, "two values under one root")))
      bad += 1, std::printf("a second value at a vertex under one root: %s\n", why ? why : "ACCEPTED");
  }
  // [own-test-mask-rows] One value, the first or the last, of each draw of field elements in turn
  // is altered right after the draw. If the draw filled a message's mask rows or their random tail,
  // open_oracles() must abort. Every other altered draw is the control: the run must go through and
  // be accepted, as a run with altered mask rows or an altered tail would be without the check.
  {
    const Case &c = cases[4];
    NumberedCoins honest(5);
    std::vector<u32> noted;
    bad += one_case(c.name, codes, c.sent, c.log_n, c.ask, 5, -1, &honest, &noted);
    unsigned with_mask_rows = 0, stopped = 0, passed = 0;
    for (const OracleNotice &n : c.sent) with_mask_rows += n.shape.mask_rows > 0;
    for (long d = 0; d < (long)honest.drawn.size(); d++)
      for (bool last_value : {false, true}) {
        auto run = [&] {
          NumberedCoins coins(5);
          coins.loses = d, coins.last_value = last_value;
          if (one_case(c.name, codes, c.sent, c.log_n, c.ask, 5, -1, &coins)) _exit(1);
        };
        const bool compared = std::find(noted.begin(), noted.end(), (u32)d) != noted.end();
        if (compared ? ends_with("own-test-mask-rows", run) : runs_through(run))
          (compared ? stopped : passed)++;
        else
          bad += 1, std::printf("draw %ld, its %s value altered: %s\n", d, last_value ? "last" : "first", compared ? "the message is opened all the same" : "the run does not go through");
      }
    std::printf("one altered value: %u runs on the mask rows and random tails of %u messages end by own-test-mask-rows, %u runs on other draws go through\n", stopped, with_mask_rows, passed);
    if (noted.size() != 2 * with_mask_rows || stopped != 4 * with_mask_rows || passed == 0) bad += 1, std::printf("not all mask rows and random tails are compared, or no other draw was tried\n");
  }
  if (bad) return std::printf("FAILED: %d\n", bad), 1;
  std::printf("context: ok: %zu arrangements, %ld flips refused, %d guards stop at their planted breach\n", cases.size(), flips, guards);
  return 0;
}
