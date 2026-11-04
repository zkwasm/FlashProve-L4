// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shows why six run-time guards for hiding in blocks/iop/context.h are needed. A guard here is a
// BLK_REQUIRE (blocks/base.h) that aborts a prover before it opens a Merkle leaf at a list of
// positions at which some oracle message in that leaf is not meant to be read, or a row at more
// positions than its random tail (the random entries encoded with it) hides. An oracle message is a
// set of encoded rows; several can share a tree's leaves. No check that a verifier makes on a
// proof's bytes detects such a breach. The four guards on leaves depend only on the layout, which
// is public, so the verifier's code has them too; the two counts of opened positions are only in
// the prover's open_oracles(). blocks/iop/tests/context_test.cpp shows that each guard aborts at a
// breach; a build of this test that is not part of this repository shows what happens without the
// guards (see main()). The guards in TreeBook::adopt() are not covered. The headers read the two
// macros below.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // this test writes its plan by hand (blocks/numbers/plan.h)
#include <cstdio>
#include <string>
#include <vector>

#include "chain/closed.h"
#include "chain/tests/death.h"
#include "engine/engine_cpu.h"
#include "iop/context.h"
#include "iop/send.h"
#include "tests/hiding/exact.h"

using namespace blk;

namespace {

constexpr unsigned LOG_N = 8, DRAWS = 7;
constexpr u64 LABEL = 9; // of the small protocol's chain
constexpr size_t ONE_LIST = 7, TWO_LISTS = 15; // tail lengths that hide one list or two

// The small protocol. One prover message, sent by send_prove() (blocks/iop/send.h), has the oracle
// message a (two rows over F) and, if `Layout` says so, b (one row) and "the mask of one step"
// (`rider`): one mask row without an out-of-domain sample, meant to be read at one list only. The
// prover then sends one scalar: the sum of a's first row plus the rider's first entry, or 0 without
// a rider. Then DRAWS positions are drawn for each of two lists, one per group of oracle messages:
// the first group's messages and the rider are to be read at the first list, the second group's at
// the second.
struct Layout {
  size_t main_tail = ONE_LIST, mask_tail = ONE_LIST;
  bool a_in_both = false;    // a belongs to both groups (else to the first)
  bool a_mask_row = false;   // a has a mask row, with the pad that masks its sample's answer
  bool b = false;            // there is b, in the second group
  bool b_apart = false;      // .. in leaves of its own
  bool rider = false;        // there is the mask of one step
  bool rider_apart = false;  // .. in leaves of its own
  bool own_only = false;     // an opening omits messages that only share a tree with those it reads
  bool a_again = false;      // the second list reads a too, though a is in the first group only
};

struct Small {
  Setting set;
  Codes codes;
  std::vector<OracleNotice> sent; // a [, b] [, rider]
  Plan plan;
  Layout lay;
  explicit Small(const Layout &l) : set(setting_goldilocks()), plan(set), lay(l) {
    codes.base = {Code{LOG_N, 16, l.main_tail}};
    codes.mask = {Code{LOG_N, 4, l.mask_tail}};
    BLK_REQUIRE(code_ok(codes.base[0]) && code_ok(codes.mask[0]), "the test's numbers are not codes in this setting");
    auto notice = [&](const char *name, size_t rows, size_t mask_rows, unsigned answers, unsigned leaf_class) {
      OracleNotice n;
      n.shape.name = name, n.shape.code = rows ? 0 : NONE_ID, n.shape.main_rows = rows, n.shape.main_deg = 1;
      n.shape.mask_rows = mask_rows, n.shape.mask_code = 0, n.shape.answers = answers, n.leaf_class = leaf_class;
      return n;
    };
    sent.push_back(notice("a", 2, l.a_mask_row ? 1 : 0, set.samples, 0));
    if (l.b) sent.push_back(notice("b", 1, 0, set.samples, l.b_apart ? 1 : 0));
    if (l.rider) sent.push_back(notice("the mask of one step", 0, 1, 0, l.rider_apart ? 2 : 0));
    std::vector<SampleSize> samples;
    for (const OracleNotice &n : sent)
      if (n.shape.answers) samples.push_back(SampleSize{Send::cube(codes, n.shape).bits, no_sized_group()});
    plan_send(plan, "a, b", samples, (unsigned)tree_sizes(codes, sent).size());
    plan.send_scalars("a masked scalar", 1);
    for (const char *name : {"first list", "second list"}) {
      PlanRound &round = plan.close(name, RoundKind::POSITIONS, ErrorKind::QUERY, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);
      round.draw_positions = DRAWS, round.log_n = LOG_N;
    }
  }
  // Group membership and what each opening asks for, as one word for the first query (the hash
  // chain's first input). `codes` and `sent` have the rest of `Layout`.
  static constexpr u64 A_IN_BOTH = 1, B = 2, RIDER = 4, OWN_ONLY = 8, A_AGAIN = 16;
  u64 reading() const { return A_IN_BOTH * lay.a_in_both + B * lay.b + RIDER * lay.rider + OWN_ONLY * lay.own_only + A_AGAIN * lay.a_again; }
};

// Handles to the public inputs that bind() writes into the first query. Of `Small`, decide() gets
// only these (run_verifier(), blocks/chain/closed.h).
struct Bound {
  Public<Codes> codes;
  Public<std::vector<OracleNotice>> sent;
  Public<u64> reading;
};
Bound bind(Binder &binder, const Small &s) {
  Digest configuration{};
  configuration.w[0] = 77;
  binder.configuration(configuration);
  Bound in;
  in.codes = binder.parameter("codes", s.codes), in.sent = binder.parameter("the oracle messages", s.sent), in.reading = binder.parameter("who reads what", s.reading());
  return in;
}
void join_groups(Statement &st, const std::vector<OracleId> &ids, u64 reading) {
  const GroupId first = st.new_group("first", 0, 0), second = st.new_group("second", 0, 0);
  st.join(first, ids[0]);
  if (reading & Small::A_IN_BOTH) st.join(second, ids[0]);
  if (reading & Small::B) st.join(second, ids[1]);
}
// The oracle messages that the opening at list 0 or 1 asks for.
std::vector<OracleId> asked(const std::vector<OracleNotice> &sent, const std::vector<OracleId> &ids, u64 reading, unsigned list) {
  std::vector<bool> wanted(sent.size(), false);
  if (list == 0 || (reading & (Small::A_IN_BOTH | Small::A_AGAIN))) wanted[0] = true;
  if (list == 1 && (reading & Small::B)) wanted[1] = true;
  if (list == 0 && (reading & Small::RIDER)) wanted[(reading & Small::B) ? 2 : 1] = true;
  std::vector<OracleId> ask;
  for (size_t i = 0; i < sent.size(); i++) {
    bool shares = false;
    for (size_t j = 0; j < sent.size(); j++) shares = shares || (wanted[j] && sent[j].leaf_class == sent[i].leaf_class);
    if (wanted[i] || (shares && !(reading & Small::OWN_ONLY))) ask.push_back(ids[i]);
  }
  return ask;
}

void prove(const Small &s, unsigned witness, Coins &coins, Writer &proof) {
  ToyCoins data(100 + witness);
  EngineCpu engine;
  Binder binder(LABEL, s.plan);
  const Bound in = bind(binder, s);
  ProverRounds ch(binder, random_digest(coins), proof);
  ProverContext pr(in.codes, ch, engine, coins);
  std::vector<OracleDraft> drafts(s.sent.size());
  E scalar = E::zero();
  for (size_t i = 0; i < s.sent.size(); i++) {
    const OracleShape &shape = s.sent[i].shape;
    drafts[i].shape = shape, drafts[i].leaf_class = s.sent[i].leaf_class;
    if (shape.main_rows) {
      Table t(1, shape.main_rows, s.codes.base[0].ell);
      data.fill(t.a.data(), t.a.size());
      if (!shape.mask_rows) // the pad is in the main rows: zero here, drawn by `send_prove`
        for (const PadPlace &place : Send::pad(s.codes, shape)) t.set_entry(place.index, E::zero());
      if (i == 0 && s.lay.rider)
        for (size_t j = 0; j < t.len; j++) scalar = scalar + t.at(0, j);
      drafts[i].main = engine.put(t);
    }
    if (shape.mask_rows && shape.answers) drafts[i].mask = Send::fresh_mask(s.codes, shape, coins);
    if (shape.mask_rows && !shape.answers) {
      drafts[i].mask = Table(E_DEGREE, shape.mask_rows, s.codes.mask[0].ell);
      coins.fill(drafts[i].mask.a.data(), drafts[i].mask.a.size());
      scalar = scalar + drafts[i].mask.at(0, 0);
    }
  }
  const std::vector<OracleId> ids = send_prove(pr, std::move(drafts));
  join_groups(pr.st, ids, s.reading());
  ch.send(scalar);
  Opened opened;
  unsigned list = 0;
  for (const char *name : {"first list", "second list"}) open_oracles(pr, asked(s.sent, ids, s.reading(), list++), ch.close_positions(name).positions(LOG_N), opened);
  ch.finish();
}

// The verifier: checks the openings against the roots, but neither the samples' answers nor the
// scalar.
bool decide(VerifierRounds &ch, const Bound &in, Notes *) {
  VerifierContext vf(in.codes, ch);
  const std::vector<OracleNotice> &sent = ch.get(in.sent);
  const u64 reading = ch.get(in.reading);
  std::vector<OracleId> ids;
  E scalar;
  if (!send_verify(vf, sent, ids)) return false;
  join_groups(vf.st, ids, reading);
  if (!ch.recv(scalar)) return false;
  Opened opened;
  unsigned list = 0;
  for (const char *name : {"first list", "second list"})
    if (!read_opened(vf, asked(sent, ids, reading, list++), ch.close_positions(name).positions(LOG_N), opened)) return false;
  return true;
}
bool verify(const Small &s, const Writer &proof) { return run_verifier(LABEL, s.plan, s, bind, decide, proof.data(), proof.size()); }

// A proof that a verifier can accept. BLK_TEST_PROVER_MESSAGES_OUTSIDE otherwise keeps a prover's
// messages out of its hash chain's queries (blocks/chain/rounds.h), so that all runs of the exact
// test get the same challenges.
void plain_proof(const Small &s, Writer &proof) {
  prover_messages_outside = false;
  ToyCoins coins(5);
  prove(s, 0, coins, proof);
  prover_messages_outside = true;
}
bool accepted(const Small &s) {
  Writer proof;
  plain_proof(s, proof);
  return verify(s, proof);
}

// Runs the exact test of hiding on witnesses 0 and 1 only, at one list of challenges.
Exact::Verdict exact(const Small &s, std::string &where) {
  Exact x([&s](unsigned witness, Coins &coins) {
    Writer proof;
    proof.keep_list();
    prove(s, witness, coins, proof);
    return look_at(proof);
  }, 1);
  const Exact::Verdict v = x.compare(1);
  where = x.where(v.entry);
  return v;
}

const char *const LEFT_OUT = "open: an oracle message that shares a tree with those asked for is left out";
const char *const TWO_GROUPS = "trees: oracle messages that share a leaf belong to different groups, so they are read at different lists";
const char *const RIDER_SHARES = "trees: the mask of one step shares a leaf with an oracle message that is read at several lists";
const char *const SECOND_LIST = "trees: a tree that holds the mask of one step is opened at a second list";
const char *const MAIN_COUNT = "open: main rows are opened at more positions than their random tail hides";
const char *const MASK_COUNT = "open: mask rows are opened at more positions than their random tail hides";

struct Planted {
  const char *planted, *name; // the case's short name; printed description
  Layout breach, twin;
  std::vector<const char *> guards; // in the order in which they stop the prover
};

// In the four cases on leaves the breach ends in a row opened at more positions than its random
// tail hides, so a count of positions is their last guard.
std::vector<Planted> rows() {
  std::vector<Planted> r;
  {
    Planted x{"OPEN_MAIN_ROWS_TOO_OFTEN", "main rows with random entries for one list are read at two", {}, {}, {MAIN_COUNT}};
    x.breach.a_in_both = x.twin.a_in_both = true;
    x.twin.main_tail = TWO_LISTS;
    r.push_back(x);
  }
  {
    Planted x{"OPEN_MASK_ROWS_TOO_OFTEN", "a mask row with random entries for one list is read at two: it holds the pad of the sample", {}, {}, {MASK_COUNT}};
    x.breach.a_in_both = x.twin.a_in_both = true, x.breach.a_mask_row = x.twin.a_mask_row = true;
    x.breach.main_tail = x.twin.main_tail = TWO_LISTS;
    x.twin.mask_tail = TWO_LISTS;
    r.push_back(x);
  }
  {
    Planted x{"LEAVES_OF_TWO_GROUPS", "two oracle messages of different groups share their leaves", {}, {}, {TWO_GROUPS, MAIN_COUNT}};
    x.breach.b = x.twin.b = true;
    x.twin.b_apart = true;
    r.push_back(x);
  }
  {
    Planted x{"OPEN_LEAVES_A_PART_OUT", "the same, and each opening names its own group's message alone", {}, {}, {LEFT_OUT, TWO_GROUPS, MAIN_COUNT}};
    x.breach.b = x.twin.b = true, x.breach.own_only = x.twin.own_only = true;
    x.twin.b_apart = true;
    r.push_back(x);
  }
  {
    Planted x{"STEP_MASK_IN_SHARED_LEAVES", "the mask of one step rides in leaves that a second list opens", {}, {}, {RIDER_SHARES, SECOND_LIST, MASK_COUNT}};
    x.breach.a_in_both = x.twin.a_in_both = true, x.breach.rider = x.twin.rider = true;
    x.breach.main_tail = x.twin.main_tail = TWO_LISTS;
    x.twin.rider_apart = true;
    r.push_back(x);
  }
  {
    Planted x{"STEP_MASK_AT_A_SECOND_LIST", "the tree of one group's message, in which the mask of one step rides, is read once more", {}, {}, {SECOND_LIST, MASK_COUNT}};
    x.breach.a_again = x.twin.a_again = true, x.breach.rider = x.twin.rider = true;
    x.breach.main_tail = x.twin.main_tail = TWO_LISTS;
    x.twin.rider_apart = true;
    r.push_back(x);
  }
  return r;
}

} // namespace

// A build of this test that is not part of this repository, with a main() of its own, takes one
// case (printed: row) of rows() and returns 0 only if:
//   1. a child process that proves and then verifies the faulty layout (`breach`) aborts at each
//      listed guard, with that guard's message, in the listed order: each is then disabled
//      (by a switch that is not part of this repository either; printed: asleep), and
//      the next must abort the run;
//   2. with all of them disabled, on the verifier's side too, the verifier accepts. With its own
//      guards enabled, the verifier aborts on such a proof at the case's first guard if that is a
//      guard on leaves, and accepts if it is a count;
//   3. without guards, the exact test of hiding (blocks/tests/hiding/exact.h) tells two witnesses
//      apart, in an opening: the breach gives something away.
// This main() runs the valid counterpart (`twin`) of every faulty layout with all
// guards enabled: each must be accepted and give both witnesses the same distribution (printed:
// law), which shows that the exact test can also pass.
int main() {
  int bad = 0;
  const std::vector<Planted> all = rows();
  for (const Planted &row : all) {
    const Small twin(row.twin);
    if (!runs_through([&] {
          if (!accepted(twin)) _exit(1);
        }))
      bad++, std::printf("guards_test: %s: the lawful twin is stopped or refused\n", row.name);
    std::string where;
    const Exact::Verdict fine = exact(twin, where);
    if (!fine.same) bad++, std::printf("guards_test: %s: the lawful twin's witnesses are told apart: %s; %s\n", row.name, fine.why.c_str(), where.c_str());
  }
  if (bad) return std::printf("guards_test: FAILED (%d)\n", bad), 1;
  std::printf("guards_test: ok (the lawful twins of %zu planted layouts: no guard stops them, the verifier says yes, two witnesses have the same law)\n", all.size());
  return 0;
}
