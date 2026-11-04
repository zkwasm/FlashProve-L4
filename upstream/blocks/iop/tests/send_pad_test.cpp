// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Looks in the prover's memory for pads that SEND did not draw or used twice: no proof reveals such
// a fault, and the verifier still accepts. SEND (blocks/iop/send.h) sends oracle messages (main and
// mask rows, sent as Merkle roots) and takes out-of-domain samples of them (random linear forms of
// a sent message). A pad is the random element of the extension field E that masks one sample's
// answer; its places, listed by Send::pad(), are the entries of the prover's tables that store it:
// one, or three in a table over the base field F. For each list of oracle message shapes, the test
// runs SEND twice on the same data with different private randomness (`coins` in identifiers):
//   - the verifier (here only send_verify()) accepts both runs; every answer differs between them;
//   - the places that send_prove() records are those of Send::pad(), none contains 0, and no value
//     occurs twice, within or across runs (pads_fault()); main and mask tables differ nowhere else;
//   - send_prove() aborts the process if a pad's place, which its caller must leave 0, contains 1.
// How the pads are distributed is not tested. This project's negative controls for pads_fault()
// build a form of the test from copies of send.h in which no pad is drawn, or one draw serves every
// place and run; such a build passes only if pads_fault() reports that fault for every list and the
// verifier accepts, which shows why the test must read the prover's memory. These copies are not
// part of this repository.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // lets a test write plans by hand (blocks/numbers/plan.h)
#include "chain/tests/death.h"
#include "iop/tests/send_world.h"

using namespace blk;

namespace {

Codes the_codes() {
  Codes c;
  c.base = {Code{8, 8, 7}};
  c.mask = {Code{7, 4, 15}};
  return c;
}

// deg: 1 for main rows over F, E_DEGREE for E. answers: samples taken of the message.
OracleShape a_shape(const char *name, size_t main_rows, unsigned deg, size_t mask_rows, unsigned answers) {
  OracleShape s;
  s.name = name, s.code = main_rows ? 0 : NONE_ID, s.main_rows = main_rows, s.main_deg = deg, s.mask_rows = mask_rows, s.answers = answers;
  return s;
}

struct Ran {
  std::vector<PadLook> pads;
  std::vector<E> answers;
  std::vector<Table> main, mask;  // the prover's tables after SEND, one per oracle message
  bool accepted = false;
  size_t draws = 0;  // of elements, in the prover's run
};
// [own-test-pad-kept] A generator (`Coins`, blocks/engine/table.h) that records each draw of field
// elements, and so can answer last_draw() and not_the_coins() (with 0 or 1, not a count). fill()
// adds 1 to the first element of one draw after recording it. If that draw is a pad's,
// send_prove() has the altered value in the table and in its own copy of the pad, and only
// not_the_coins() shows the fault; send_prove() then aborts with "own-test-pad-kept". Each pad's
// place, in main or mask rows, has its own draw.
struct Recalling : ToyCoins {
  using ToyCoins::ToyCoins;
  std::vector<std::vector<F>> drawn;
  long loses = -1;  // the number of that draw; -1: none
  void fill(F *out, size_t n) override {
    ToyCoins::fill(out, n);
    drawn.emplace_back(out, out + n);
    if (n && (long)drawn.size() - 1 == loses) out[0] = out[0] + F::one();
  }
  u32 last_draw() const override { return (u32)drawn.size() - 1; }
  u64 not_the_coins(u32 draw, const F *values, u64 n, const std::vector<u64> &) const override {
    u64 differ = drawn.at(draw).size() != n;
    for (u64 i = 0; i < n && !differ; i++) differ += !(values[i] == drawn[draw][i]);
    return differ;
  }
};
// The verifier's side: it sees only the bound public inputs and the proof (blocks/chain/closed.h).
bool decide(VerifierRounds &ch, const TestBound &x, Notes *notes) {
  VerifierContext vf(x.codes, ch);
  std::vector<OracleId> ids;
  if (!send_verify(vf, ch.get(x.sent), ids)) return false;
  for (OracleId o : ids)
    for (const Row &r : vf.st.oracle(o).sample_rows)
      if (notes) notes->emplace_back("answer", r.target);
  return true;
}

// [own-test-pad-kept] An engine that loses a pad on its way into a main table. how == 2: writes the
// first entry of each set_entries() call one column lower, in the same row; how == 3: omits it.
// After sending, send_prove() checks that every pad is at its place as drawn: it must abort with
// "own-test-pad-kept". Pads in mask rows do not pass through the engine.
struct Astray : EngineCpu {
  int how = 0;
  void set_entries(TableId id, const std::vector<std::pair<u64, E>> &changed) override {
    std::vector<std::pair<u64, E>> c = changed;
    if (how == 2 && !c.empty()) c[0].first -= u64(1) << ceil_log2(get(id).rows);  // the same row, one place lower (a pad's places lie at a row's end)
    if (how == 3 && !c.empty()) c.erase(c.begin());
    EngineCpu::set_entries(id, c);
  }
};
// spoil: 0 none; 1: the test writes 1 into the first sampled message's first pad place; 2, 3: the
// engine's fault, `Astray::how`. loses: `Recalling::loses`.
Ran run(const Codes &codes, const std::vector<OracleNotice> &sent, u64 coin_seed, int spoil = 0, long loses = -1) {
  Ran out;
  Setting set = test_setting();
  unsigned answers = 0;
  for (const OracleNotice &n : sent) answers = std::max(answers, n.shape.answers);
  set.samples = answers;
  Plan p(set);
  std::vector<SampleSize> samples;
  for (const OracleNotice &n : sent)
    if (n.shape.answers) samples.push_back(SampleSize{Send::cube(codes, n.shape).bits, no_sized_group()});
  plan_send(p, "", samples, (unsigned)lay_trees(codes, sent).shapes.size());
  Recalling coins(coin_seed);
  coins.loses = loses;
  ToyCoins data(5);  // the data are the same in every run
  Astray engine;
  engine.how = spoil;
  Writer proof;
  const TestInputs in{codes, sent, {}, {}};
  const TestBinder binder(p, in);
  ProverRounds ch(binder.b, test_bytes(coins), proof);
  ProverContext pr(binder.codes, ch, engine, coins);
  pr.pads = &out.pads;
  std::vector<OracleDraft> drafts;
  for (const OracleNotice &n : sent) {
    OracleDraft d;
    d.shape = n.shape, d.leaf_class = n.leaf_class;
    const std::vector<PadPlace> pad = n.shape.answers ? Send::pad(codes, n.shape) : std::vector<PadPlace>();
    auto is_pad = [&](Part part, u64 index) {
      for (const PadPlace &q : pad)
        if (q.part == part && q.index == index) return true;
      return false;
    };
    if (n.shape.main_rows) {
      Table t(n.shape.main_deg, n.shape.main_rows, codes.base[n.shape.code].ell);
      const unsigned bits = ceil_log2(t.rows);
      for (size_t r = 0; r < t.rows; r++)
        for (size_t c = 0; c < t.len; c++)
          if (!is_pad(Part::MAIN, r + (u64(c) << bits))) t.set(r, c, t.deg == 1 ? E::from(data.f()) : data.e());
      if (spoil == 1 && !pad.empty() && pad[0].part == Part::MAIN) t.set_entry(pad[0].index, E::one()), spoil = 0;
      d.main = engine.put(t);
    }
    if (n.shape.mask_rows) {
      d.mask = Send::fresh_mask(codes, n.shape, data);
      if (spoil == 1 && !pad.empty() && pad[0].part == Part::MASK) d.mask.set_entry(pad[0].index, E::one()), spoil = 0;
    }
    drafts.push_back(std::move(d));
  }
  const std::vector<OracleId> ids = send_prove(pr, std::move(drafts));
  for (OracleId o : ids) {
    for (const Row &r : pr.st.oracle(o).sample_rows) out.answers.push_back(r.target);
    out.main.push_back(pr.st.oracle(o).shape.main_rows ? engine.get(pr.behind[o].main) : Table());
    out.mask.push_back(pr.behind[o].mask);
  }
  ch.finish();
  out.draws = coins.drawn.size();
  Notes notes;
  out.accepted = run_verifier(TEST_LABEL, p, in, test_bind, decide, proof.bytes().data(), proof.bytes().size(), nullptr, &notes) && noted(notes, "answer") == out.answers;
  return out;
}

}  // namespace

int main() {
  const Codes codes = the_codes();
  // Pads in mask rows, in main rows over F and over E; one or two answers; in the last list, four
  // messages in one prover message, one not sampled.
  const std::vector<std::vector<OracleNotice>> lists = {
      {{a_shape("main rows over F, a mask row", 2, 1, 1, 1), 0}},
      {{a_shape("mask rows alone", 0, 1, 2, 1), 0}},
      {{a_shape("main rows over F alone", 3, 1, 0, 1), 0}},
      {{a_shape("main rows over E alone", 2, E_DEGREE, 0, 1), 0}},
      {{a_shape("over F alone, two answers", 1, 1, 0, 2), 0}},
      {{a_shape("over E alone, two answers", 2, E_DEGREE, 0, 2), 0}},
      {{a_shape("a mask row, two answers", 1, E_DEGREE, 1, 2), 0}},
      {{a_shape("first", 2, 1, 1, 1), 0}, {a_shape("second, over F alone", 2, 1, 0, 1), 0}, {a_shape("rides, not sampled", 0, 1, 1, 0), 0}, {a_shape("third, over E alone", 1, E_DEGREE, 0, 1), 0}},
  };
  size_t places = 0;
  for (const std::vector<OracleNotice> &sent : lists) {
    const Ran a = run(codes, sent, 11), b = run(codes, sent, 12);
    TEST_CHECK(a.accepted && b.accepted);
    size_t want = 0;
    for (const OracleNotice &n : sent)
      if (n.shape.answers) want += Send::pad(codes, n.shape).size();
    TEST_CHECK(a.pads.size() == want && want > 0);
    {  // the places are those of `Send::pad`, in its order; in a table over F each value is in F
      size_t i = 0;
      for (size_t o = 0; o < sent.size(); o++) {
        if (!sent[o].shape.answers) continue;
        for (const PadPlace &p : Send::pad(codes, sent[o].shape)) {
          const PadLook &seen = a.pads[i++];
          TEST_CHECK(seen.oracle == o && seen.in_mask_rows == (p.part == Part::MASK) && seen.place == p.index);
          if (p.part == Part::MAIN && sent[o].shape.main_deg == 1) TEST_CHECK(seen.value.c[1] == F::zero() && seen.value.c[2] == F::zero());
        }
      }
    }
    const char *fault = pads_fault(a.pads, b.pads);
    if (fault) std::printf("%s: %s\n", sent[0].shape.name.c_str(), fault);
    TEST_CHECK(!fault);
    TEST_CHECK(a.answers.size() == b.answers.size() && !a.answers.empty());
    for (size_t i = 0; i < a.answers.size(); i++) TEST_CHECK(a.answers[i] != b.answers[i]);
    // Each table differs between runs in as many entries as it has pads' places, which all differ.
    for (size_t o = 0; o < sent.size(); o++)
      for (int part = 0; part < 2; part++) {
        const Table &x = part ? a.mask[o] : a.main[o], &y = part ? b.mask[o] : b.main[o];
        TEST_CHECK(x.a.size() == y.a.size());
        size_t differ = 0;
        for (size_t r = 0; r < x.rows; r++)
          for (size_t c = 0; c < x.len; c++) differ += !(x.at(r, c) == y.at(r, c));
        size_t here = 0;
        for (const PadLook &n : a.pads) here += n.oracle == o && n.in_mask_rows == (part == 1);
        TEST_CHECK(differ == here);
      }
    TEST_CHECK(ends_with("send: a pad's place holds something", [&] { run(codes, sent, 13, 1); }));
    {  // `Astray`: the run must abort if a sampled message has no mask rows (pads in a main table)
      bool through_the_engine = false;
      for (const OracleNotice &n : sent) through_the_engine = through_the_engine || (n.shape.answers && !n.shape.mask_rows);
      for (int how : {2, 3})
        if (through_the_engine)
          TEST_CHECK(ends_with("own-test-pad-kept", [&] { run(codes, sent, 13, how); }));
        else
          TEST_CHECK(run(codes, sent, 13, how).accepted);
    }
    {  // `Recalling`: altering draw d aborts the run for as many d as there are pads' places
      size_t caught = 0;
      for (size_t d = 0; d < a.draws; d++) caught += ends_with("own-test-pad-kept", [&] { run(codes, sent, 11, 0, (long)d); });
      TEST_CHECK(a.draws > want && caught == want);
    }
    places += want;
  }
  std::printf("ok: %zu lists of shapes, %zu pads' places\n", lists.size(), places);
  return 0;
}
