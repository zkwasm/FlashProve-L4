// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests that calls which break the order of messages or the rules of use of three protocol blocks
// are stopped. SEND sends oracle messages and takes an out-of-domain sample of those with
// `answers` > 0; BAT batches linear claims (`Row` in code, "row" in messages); SPLIT splits a claim
// on two groups into one on each. A sample claim states a sample's answer. A group: the oracle
// messages that one proximity test reads.
// Some wrong programs do not compile: the static_asserts show it. Every other wrong call is made in
// a child process, by stops() (send_world.h), and must abort with the message that the test names;
// controls, by runs(), make a valid call and must finish. A case's changes are lost with its child.
// Limits: only the prover's side runs; the checks stop mistakes, not intent (Send::prove() accepts
// any channel type).
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets tests write a `Plan` by hand (blocks/numbers/plan.h)
#include <type_traits>

#include "iop/split.h"
#include "iop/tests/send_world.h"

using namespace blk;

namespace {

// ---- what does not compile ----
template <class T, class = void> struct draws_seed : std::false_type {};
template <class T> struct draws_seed<T, std::void_t<decltype(std::declval<T &>().seed(1u, 1u))>> : std::true_type {};
template <class T, class = void> struct draws_positions : std::false_type {};
template <class T> struct draws_positions<T, std::void_t<decltype(std::declval<T &>().positions(1u))>> : std::true_type {};
template <class T, class = void> struct draws_element : std::false_type {};
template <class T> struct draws_element<T, std::void_t<decltype(std::declval<T &>().draw())>> : std::true_type {};
template <class T, class = void> struct draws_batching : std::false_type {};
template <class T> struct draws_batching<T, std::void_t<decltype(std::declval<T &>().batching())>> : std::true_type {};

// The verifier's message with a sample's seed contains nothing else: no positions, and no element
// of the extension field E that could serve as a point at which something is folded and compared
// with the answer, or as any other challenge.
static_assert(draws_seed<SeedDraw>::value, "the control");
static_assert(!draws_positions<SeedDraw>::value && !draws_element<SeedDraw>::value && !draws_batching<SeedDraw>::value,
              "a seed shares its message with nothing");
// Conversely, none of the three other `Draw` classes (blocks/chain/rounds.h) has seed().
static_assert(draws_positions<PositionsDraw>::value && draws_element<AlgebraicDraw>::value, "the controls");
static_assert(!draws_seed<PositionsDraw>::value && !draws_seed<AlgebraicDraw>::value && !draws_seed<BitsDraw>::value,
              "a seed comes in a round of seeds only");
// Statement::sampled() marks an oracle message as sampled and releases its mask entries; only
// `Send` can make the `SendAccess` that it takes.
static_assert(!std::is_default_constructible<SendAccess>::value && !std::is_aggregate<SendAccess>::value, "SEND's key is SEND's");
// A caller cannot default-construct or aggregate-initialize a `RunInput`, a proximity test's input:
// Statement::begin_run() returns it, and rows() gives read access only.
static_assert(!std::is_default_constructible<RunInput>::value && !std::is_aggregate<RunInput>::value &&
                  std::is_same<decltype(std::declval<RunInput &>().rows()), const std::vector<Row> &>::value,
              "the rows of a proximity test are the statement's");
static_assert(!std::is_constructible<SampleReceipt, u32, u32, u32>::value && !std::is_copy_constructible<SampleReceipt>::value,
              "a receipt is made by SEND, once");
// No mask entry (a random element that masks one sent scalar) can be named by two `MaskSlot`s. A
// claim can still name one by its index, without a slot, as unit_row() does below.
static_assert(!std::is_constructible<MaskSlot, u32, u32>::value && !std::is_copy_constructible<MaskSlot>::value &&
                  !std::is_copy_constructible<MaskLedger>::value && !std::is_copy_constructible<Statement>::value,
              "a mask entry is handed out once");

// ---- what is stopped ----
Codes the_codes() {
  Codes c;
  c.base = {Code{6, 4, 3}, Code{5, 4, 3}};
  c.mask = {Code{5, 4, 3}};
  return c;
}
OracleNotice notice(const char *name, CodeId code, size_t main_rows, unsigned answers, unsigned leaf_class) {
  OracleNotice n;
  n.shape.name = name, n.shape.code = code, n.shape.main_rows = main_rows, n.shape.mask_rows = 1, n.shape.answers = answers;
  n.leaf_class = leaf_class;
  return n;
}

// Five oracle messages, one mask row each; `o1` and `o2` also have a main row, under different
// base codes.
struct World {
  Codes codes = the_codes();
  std::vector<OracleNotice> sent = {notice("o1", 0, 1, 1, 0), notice("o2", 1, 1, 1, 1), notice("both", NONE_ID, 0, 1, 2),
                                    notice("unsampled", NONE_ID, 0, 0, 3), notice("spare", NONE_ID, 0, 1, 4)};
  Plan plan{test_setting()};
  ToyCoins coins{1};
  PlantEngine engine;
  Writer proof;
  std::unique_ptr<TestBinder> binder;
  std::unique_ptr<ProverRounds> ch;
  std::unique_ptr<ProverContext> pr;
  unsigned bits = 0;

  World() {
    std::vector<SampleSize> samples;
    for (const OracleNotice &n : sent)
      if (n.shape.answers) {
        samples.push_back(SampleSize{Send::cube(codes, n.shape).bits, no_sized_group()});
        bits = std::max(bits, samples.back().cube_bits);
      }
    plan_send(plan, "", samples, (unsigned)lay_trees(codes, sent).shapes.size());
    plan_split(plan, "");
    plan_bat(plan, "first", no_sized_group(), 3);   // three sample claims, or two and one added
    plan_bat(plan, "second", no_sized_group(), 2);  // two sample claims
    binder = std::make_unique<TestBinder>(plan, codes, sent);
    ch = std::make_unique<ProverRounds>(binder->b, test_bytes(coins), proof);
    pr = std::make_unique<ProverContext>(binder->codes, *ch, engine, coins);
  }
  std::vector<OracleDraft> drafts() {
    std::vector<OracleDraft> d(sent.size());
    for (size_t i = 0; i < sent.size(); i++) {
      d[i].shape = sent[i].shape, d[i].leaf_class = sent[i].leaf_class;
      if (sent[i].shape.main_rows) d[i].main = engine.put(random_table(1, 1, 4, coins));
      d[i].mask = Send::fresh_mask(codes, sent[i].shape, coins);
    }
    return d;
  }
};

Row unit_row(OracleId o, Part part, u64 index) {
  Row r;
  r.name = "one entry";
  r.add_unit(o, part, index, E::one());
  return r;
}

int between_seed_and_answer() {
  World w;
  send_oracles(*w.pr, w.drafts());
  SeedDraw from = w.ch->close_seed(R_SEND_SEED);
  from.seed(w.bits, 4);
  const char *first = "after a round of seeds the prover's message begins with the answers";
  TEST_CHECK(stops(first, [&] { w.ch->send(E::one()); }));
  TEST_CHECK(stops(first, [&] { w.ch->send_root(Digest{}); }));
  TEST_CHECK(stops(first, [&] { w.ch->send_hiding_sum(E::one()); }));
  // No round closes while one of the four answers is missing.
  const char *owed = "a round is closed while answers to a sample are owed";
  TEST_CHECK(stops(owed, [&] { w.ch->close_algebraic(R_BAT); }));
  TEST_CHECK(stops(owed, [&] {
    for (int i = 0; i < 3; i++) w.ch->send_answer(E::one());
    w.ch->close_algebraic(R_BAT);
  }));
  TEST_CHECK(stops("an answer that no seed has asked for", [&] {
    for (int i = 0; i < 5; i++) w.ch->send_answer(E::one());
  }));
  // Once the prover has sent an answer, `from` draws no further seed.
  TEST_CHECK(stops("this verifier message is over", [&] {
    w.ch->send_answer(E::one());
    from.seed(w.bits, 1);
  }));
  TEST_CHECK(runs([&] {
    for (int i = 0; i < 4; i++) w.ch->send_answer(E::one());
    w.ch->send(E::one());
  }));
  return 0;
}

int before_the_sample() {
  World w;
  Statement &S = w.pr->st;
  const std::vector<OracleId> id = send_oracles(*w.pr, w.drafts());  // sent, and nothing more
  const GroupId g = S.new_group("g", 0);
  TEST_CHECK(stops("only a sampled oracle message can join a group", [&] { S.join(g, id[0]); }));
  TEST_CHECK(stops("this oracle message has no mask entries to give", [&] { S.masks(id[0]).take(); }));
  TEST_CHECK(stops("a row on a group names members of that group only", [&] { S.add_row(g, unit_row(id[0], Part::MAIN, 0)); }));
  // SEND samples at least one message, in the order of sending: prover and verifier then go through
  // the same list of answers.
  auto nothing = [](const Row &) { return E::zero(); };
  TEST_CHECK(stops("nothing to sample", [&] { Send::prove(w.pr->ch, S, {}, nothing); }));
  TEST_CHECK(stops("nothing to sample", [&] { Send::prove(w.pr->ch, S, {id[3]}, nothing); }));  // a message that has no answers
  TEST_CHECK(stops("oracle messages come in the order of sending", [&] { Send::prove(w.pr->ch, S, {id[1], id[0]}, nothing); }));
  TEST_CHECK(stops("oracle messages come in the order of sending", [&] { Send::prove(w.pr->ch, S, {id[0], id[0]}, nothing); }));
  TEST_CHECK(runs([&] { Send::prove(w.pr->ch, S, id, nothing); }));
  // Each answer has its own pad (the random element that masks it) in the last row: a row of four
  // entries allows at most four answers.
  OracleShape many = w.sent[2].shape;
  many.answers = 5;
  TEST_CHECK(stops("the row is too short for the pad", [&] { Send::pad(w.codes, many); }));
  many.answers = 4;
  TEST_CHECK(runs([&] { Send::pad(w.codes, many); }));
  // A sample claim needs a whole seed: a coordinate of z for each bit of the message's cube (which
  // indexes its entries), and a factor beta on the pad that is not 0 (with 0 the answer would
  // reveal an unmasked linear form of the message).
  SampleSeed seed;
  seed.z = random_point(w.bits, w.coins);
  TEST_CHECK(runs([&] { Send::form(S, id[0], seed, 0); }));
  TEST_CHECK(stops("the seed is too short", [&] {
    SampleSeed shorter = seed;
    shorter.z.resize(Send::cube(w.codes, S.oracle(id[0]).shape).bits - 1);
    Send::form(S, id[0], shorter, 0);
  }));
  TEST_CHECK(stops("beta is 0", [&] {
    SampleSeed bare = seed;
    bare.beta = E::zero();
    Send::form(S, id[0], bare, 0);
  }));
  return 0;
}

int after_the_sample() {
  World w;
  ProverContext &pr = *w.pr;
  Statement &S = pr.st;
  const std::vector<OracleId> id = send_prove(pr, w.drafts());
  const OracleId o1 = id[0], o2 = id[1], both = id[2], unsampled = id[3], spare = id[4];
  const GroupId g1 = S.new_group("first", 0), g2 = S.new_group("second", 1);
  S.join(g1, o1), S.join(g1, both), S.join(g2, o2), S.join(g2, both);
  Row across = unit_row(o1, Part::MAIN, 0);
  across.add_unit(o2, Part::MAIN, 0, E::one());

  TEST_CHECK(stops("only a sampled oracle message can join a group", [&] { S.join(g1, unsampled); }));
  TEST_CHECK(stops("this oracle message has no mask entries to give", [&] { S.masks(unsampled).take(); }));
  TEST_CHECK(stops("a masking oracle is never sampled", [&] { S.spend_as_mask(o1); }));
  TEST_CHECK(stops("a masking oracle serves one block only", [&] { S.spend_as_mask(unsampled), S.spend_as_mask(unsampled); }));
  TEST_CHECK(stops("an oracle message is sampled once", [&] { Send::prove(pr.ch, S, {o1}, [](const Row &) { return E::zero(); }); }));
  TEST_CHECK(stops("a group has one base code", [&] { S.join(g1, o2); }));
  TEST_CHECK(stops("a row on a group names members of that group only", [&] { S.add_row(g1, unit_row(o2, Part::MAIN, 0)); }));
  // A sample claim cannot be added, batched early or split.
  const Row sample = S.oracle(o1).sample_rows[0];
  TEST_CHECK(stops("sample rows come from SEND alone", [&] { S.add_row(g1, sample); }));
  TEST_CHECK(stops("a sample row stays with its oracle message", [&] { bat_sigma_prove(pr, {sample}, "early"); }));
  TEST_CHECK(stops("only a plain row can be split", [&] { split_prove(pr, sample, g1, g2, both); }));
  // Only claims on sampled messages are batched: a message's sample is meant to pin it down.
  TEST_CHECK(stops("a row names an oracle message that has not been sampled",
                   [&] { bat_sigma_prove(pr, {unit_row(unsampled, Part::MASK, 0)}, "early"); }));
  // A sample is copied from another statement only.
  TEST_CHECK(stops("the other statement is this one", [&] { S.sampled_elsewhere(unsampled, S, both); }));
  // A run of coefficients must fit its table, also when its last index overflows 64 bits.
  TEST_CHECK(stops("no such entry", [&] {
    Row r;
    r.add_run(o1, Part::MAIN, 0, u64(1) << 63, {E::one(), E::one(), E::one()});
    S.add_row(g1, r);
  }));
  // No claim but the sample claim reads a pad.
  const PadPlace pad = Send::pad(w.codes, S.oracle(o1).shape)[0];
  TEST_CHECK(stops("a row reads the pad of a sample", [&] { S.add_row(g1, unit_row(o1, pad.part, pad.index)); }));
  TEST_CHECK(stops("a row reads the pad of a sample", [&] {
    Row r;
    r.add(o1, Part::MASK, Tensor::eq(random_point(2, w.coins).data(), 2));
    S.add_row(g1, r);
  }));
  TEST_CHECK(runs([&] { S.add_row(g1, unit_row(o1, pad.part, pad.index - 1)); }));
  // A mask entry is given out once, and the pad never.
  TEST_CHECK(stops("the mask rows have no free entry left", [&] { S.masks(o1).take(4); }));
  TEST_CHECK(runs([&] { S.masks(o1).take(3); }));
  TEST_CHECK(stops("this slot has been moved away", [&] {
    MaskSlot s = S.masks(o1).take();
    Row r;
    r.add_mask(std::move(s), E::one());
    (void)s.index();
  }));
  TEST_CHECK(stops("the mask entry comes from a message with mask rows only", [&] { split_prove(pr, across, g1, g2, o1); }));
  TEST_CHECK(stops("the mask message is a member of both groups", [&] { split_prove(pr, across, g1, g2, spare); }));
  TEST_CHECK(stops("a member of exactly one of the two groups", [&] {
    Row r = across;
    r.add_unit(both, Part::MASK, 0, E::one());
    split_prove(pr, r, g1, g2, both);
  }));
  TEST_CHECK(runs([&] { split_prove(pr, across, g1, g2, both); }));

  // A group left untested would leave its claims unchecked.
  TEST_CHECK(stops("the protocol ends with a group that has not been tested", [&] { S.require_all_tested(); }));

  // A case that does not split still sends SPLIT's planned scalar and closes its round.
  auto in_the_place_of_split = [&] {
    pr.ch.send(E::one());
    pr.ch.close_empty(R_SPLIT);
  };
  // A case that runs `then` once the proximity test on the first group has begun.
  auto begun = [&](const std::function<void()> &then) {
    return [&, then] {
      in_the_place_of_split();
      S.add_row(g1, unit_row(o1, Part::MAIN, 0));
      bat_prove(pr, g1);
      then();
    };
  };
  TEST_CHECK(runs(begun([] {})));
  // Two claims here; the plan's error counts three.
  TEST_CHECK(stops("the round batches another number of rows than its planned error counts", [&] {
    in_the_place_of_split();
    bat_prove(pr, g1);
  }));
  TEST_CHECK(stops("the rows of a proximity test are fixed before its first challenge", begun([&] { S.add_row(g1, unit_row(o1, Part::MAIN, 0)); })));
  TEST_CHECK(stops("one proximity test for each group", begun([&] { S.begin_run(g1); })));
  TEST_CHECK(stops("the proximity test on this group has begun", begun([&] { S.join(g1, spare); })));
  TEST_CHECK(runs([&] { S.join(g1, spare); }));
  TEST_CHECK(stops("a proximity test on this oracle message has begun", begun([&] { S.masks(o1).take(); })));
  // The mask message shared with the second group gives no more entries either, so no claim can be
  // split any more.
  TEST_CHECK(stops("a proximity test on this oracle message has begun", begun([&] { S.masks(both).take(); })));
  TEST_CHECK(stops("a proximity test on this oracle message has begun", begun([&] { split_prove(pr, across, g1, g2, both); })));
  // With both groups tested, `spare`, sampled but a member of no group, is still noticed.
  auto both_begun = [&] {
    in_the_place_of_split();
    bat_prove(pr, g1);
    bat_prove(pr, g2);
    S.require_all_tested();
  };
  TEST_CHECK(stops("the protocol ends with an oracle message that is in no group", [&] {
    S.add_row(g1, unit_row(o1, Part::MAIN, 0));
    both_begun();
  }));
  TEST_CHECK(runs([&] {
    S.join(g1, spare);
    both_begun();
  }));
  return 0;
}

}  // namespace

int main() {
  if (between_seed_and_answer() || before_the_sample() || after_the_sample()) return 1;
  std::printf("ok: calls out of order are stopped, each by its own check\n");
  return 0;
}
