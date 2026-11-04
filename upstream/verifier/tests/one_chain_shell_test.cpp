// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests verifier/src/vf_transcript_calls.h (all but `trTrits`, `trBatchOpen`, `trBatchRows`) on a
   `Tr` (verifier/src/vf_common.h), which wraps the verifier's channel (blk::VerifierRounds) of a
   hash chain, the Fiat-Shamir transcript of blocks/. `prove` and `check` follow one plan
   (blk::Plan: the chain's rounds): a zero-check's point, a sumcheck with a hiding polynomial
   (`PSC` in identifiers), the challenges of LINE (merges two evaluation claims into one), of a
   lookup and of BAT-sigma (batches linear claims into one). `check` must read what `prove` sent and
   draw the same challenges. Also: the extension-field types `Ex` and blk::E add and multiply alike;
   three altered proofs are detected.
   Build, from verifier/tests: g++ -std=c++17 -O2 -fno-ipa-modref -DDIGEST_WORDS=8 -DBLK_FIELD=2
   -DBLK_HASH=1 -I ../../blocks -I ../src one_chain_shell_test.cpp
   -fno-ipa-modref works around a bug in g++ 13.3.0: see verifier/build.sh. */
#include "vf_common.h"
#include "vf_transcript_calls.h"

#include "numbers/blocks.h"

static int failures = 0;
#define CHECK(c)                                                   \
  do {                                                             \
    if (!(c)) ++failures, printf("%s:%d: %s\n", __FILE__, __LINE__, #c); \
  } while (0)

static bool same(const Ex &a, const blk::E &b) {
  return a.c[0] == b.c[0].v && a.c[1] == b.c[1].v && a.c[2] == b.c[2].v;
}
static blk::E made(u64 i) {
  return blk::E::from(blk::from_u64(3 * i + 1), blk::from_u64(i * i + 7), blk::from_u64(0x123456789abcdefull ^ i));
}

static blk::Plan plan_of(const blk::Setting &s) {
  blk::Plan p(s);
  blk::PscSize x;
  x.degree.assign(3, 3);
  blk::plan_zero(p, "", 4);
  blk::plan_psc(p, "", x);
  blk::plan_line(p, "");
  blk::plan_shift(p, "", 1 << 10);
  blk::plan_lkp(p, "", 10);
  blk::plan_bat_sigma(p, "", 5);
  return p;
}

struct Spoken {
  vector<blk::E> sent, drawn;
};
// The prover's side. The scalars are arbitrary: the channel checks no equation of a sumcheck.
static Spoken prove(const blk::Planned &p, blk::Writer &w) {
  blk::Binder first(1, p);
  first.configuration(blk::Digest{{1, 2, 3, 4, 5, 6, 7, 8}});  // made up; the same on both sides
  blk::ProverRounds ch(first, blk::Digest{{1, 2, 3, 4, 5, 6, 7, 8}}, w);
  Spoken t;
  u64 i = 0;
  auto send = [&](size_t n) {
    while (n--) t.sent.push_back(made(++i)), ch.send(t.sent.back());
  };
  {
    blk::AlgebraicDraw d = ch.close_algebraic(blk::R_ZERO);
    for (int j = 0; j < 4; j++) t.drawn.push_back(d.draw());
  }
  t.sent.push_back(made(++i)), ch.send_hiding_sum(t.sent.back());
  t.drawn.push_back(ch.close_epsilon(blk::R_PSC_EPSILON));
  for (int j = 0; j < 3; j++) send(2), t.drawn.push_back(ch.close_algebraic(blk::R_PSC_ROUND).draw());
  send(2), t.drawn.push_back(ch.close_algebraic(blk::R_PSC_Y).draw_nonzero());
  send(2), ch.close_empty(blk::R_PSC_VALUES);  // the two values; the verifier's message is empty
  t.drawn.push_back(ch.close_algebraic(blk::R_LINE).draw());
  t.drawn.push_back(ch.close_algebraic(blk::R_LKP_SHIFT).draw());
  t.drawn.push_back(ch.close_algebraic(blk::R_LKP).draw());
  {
    blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
    from.require_rows(5);
    t.drawn.push_back(from.draw());
  }
  ch.finish();
  return t;
}
// The verifier's side: reads and draws through vf_transcript_calls.h only. False if a value differs from `t`.
static bool check(const blk::Planned &p, const vector<blk::u8> &bytes, const Spoken &t) {
  blk::Reader file(bytes.data(), bytes.size());
  blk::Binder first(1, p);
  first.configuration(blk::Digest{{1, 2, 3, 4, 5, 6, 7, 8}});  // made up; the same on both sides
  blk::VerifierSession session;
  blk::VerifierRounds ch(first, file, session);
  Tr tr{ch};
  size_t s = 0, d = 0;
  bool all = true;
  // A failed comparison does not stop the run: finish() ends the process if planned rounds remain.
  auto sent = [&](const Ex &x) { all = same(x, t.sent[s++]) && all; };
  auto drawn = [&](const Ex &x) { all = same(x, t.drawn[d++]) && all; };
  auto read = [&](size_t n) {
    for (const Ex &x : trRead(tr, n)) sent(x);
  };
  for (const Ex &x : trPoint(tr, RoundName::ZERO, 4)) drawn(x);
  sent(trSum(tr));
  drawn(trAnswer(tr, RoundName::PSC_EPSILON));
  for (int j = 0; j < 3; j++) read(2), drawn(trAnswer(tr, RoundName::PSC_ROUND));
  read(2), drawn(trAnswerNotZero(tr, RoundName::PSC_Y));
  read(2), trEmpty(tr, RoundName::PSC_VALUES);
  drawn(trAnswer(tr, RoundName::LINE));
  drawn(trAnswer(tr, RoundName::LKP_SHIFT));
  drawn(trAnswer(tr, RoundName::LKP));
  drawn(trBatch(tr, 5)[1]); // 5 rows, as planned; of the returned powers, [1] is the challenge
  return ch.finish() && all && s == t.sent.size() && d == t.drawn.size();
}
// The rejection name (`Refusal::name`) with which `run` ends; empty if it is not rejected.
template <class Run>
static string refusal(Run run) {
  refuseThrows = true;
  string name;
  try {
    run();
  } catch (const Refusal &r) {
    name = r.name;
  }
  refuseThrows = false;
  return name;
}

int main() {
  for (u64 i = 1; i < 200; i++) {  // `Ex` and blk::E are the same field in the same basis
    const blk::E a = made(i), b = made(1000 + i);
    CHECK(same(fromBlkE(a) * fromBlkE(b), a * b) && same(fromBlkE(a) + fromBlkE(b), a + b));
  }
  // `work`: bits of proof of work in every round of this plan that draws a challenge (all are
  // ErrorKind::FIELD)
  for (unsigned work : {0u, 6u}) {
    blk::Setting s = blk::setting_of_this_build();
    s.set_work(blk::ErrorKind::FIELD, blk::Scope::REQUEST, work);
    const blk::Planned p = plan_of(s).finish();
    blk::Writer w;
    const Spoken t = prove(p, w);
    CHECK(w.size() == p->size_of(0).bytes + 32);  // + 32: the chain's salt
    CHECK(refusal([&] { CHECK(check(p, w.bytes(), t)); }).empty());
    // First bytes: chain's salt (32); with work, round ZERO's nonce (8); the hiding polynomial's
    // sum (24). Without work nothing rejects `flipped`: only the comparison with `t` fails.
    vector<blk::u8> flipped = w.bytes(), big = w.bytes(), cut(w.bytes().begin(), w.bytes().end() - 1);
    flipped[40] ^= 1;  // in the sum; changes the challenges after it
    for (int j = 0; j < 8; j++) big[32 + (work ? 8 : 0) + j] = 0xff;  // coordinate 0 not canonical
    if (work)
      CHECK(refusal([&] { check(p, flipped, t); }) == "chain-refused");  // a later nonce fails
    else
      CHECK(refusal([&] { CHECK(!check(p, flipped, t)); }).empty());
    CHECK(refusal([&] { check(p, big, t); }) == "chain-refused");
    CHECK(refusal([&] { check(p, cut, t); }) == "chain-refused");
  }
  printf(failures ? "%d checks FAILED\n" : "ok\n", failures);
  return failures ? 1 : 0;
}
