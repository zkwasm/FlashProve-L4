// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests how a hash chain (the Fiat-Shamir transcript: `ChainT`, blocks/chain/chain.h) draws an
// element of E, the cubic extension of the field F: three coordinates, one 64-bit word each. The
// cases tested are too rare to occur with the true hash function, `Blake2bF`, so `Doctored` forces
// them: a word that F_from_word() (blocks/field/field.h) rejects (probability below 2^-32), the
// caps on retries, a 0 where none is allowed. CPU only; no round here has proof of work.
// Exit code 1: a check failed; 2: nothing was tested, because one of the true function's own words
// is rejected or 0.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "chain/chain.h"

using namespace blk;

// doctor[i]: what `Doctored` does to word i of the verifier's message (the 64-bit words after the
// chain's new value in a round's answer): 0 nothing, 1 makes it a rejected word, 2 makes it 0.
static std::vector<int> doctor;
static int doing(size_t i) { return i < doctor.size() ? doctor[i] : 0; }
// The test's round sends E::one(), three 64-bit words, and has no proof of work, so the input of
// hash call j (each call gives eight words of the round's answer) is one block: the tag (the hash's
// domain number, with the chain's label, `FirstQuery::label`, in the bits above it), the round
// number, the three words, a word with j in bits 16 and up, the nonce (0 without proof of work).
// `Doctored` reads j at that place, so the prover's message, if changed, must keep three words.
// Call 0 gives the chain's value and words 0 to 3 of the verifier's message; call j > 0, words
// 8 j - 4 to 8 j + 3.
struct Doctored : Blake2bF {
  static void compress(u32 cv[CVW], const u32 block[BLKW], const Tweak &t) {
    Blake2bF::compress(cv, block, t);
    if ((block[0] & 0xFFFF) != (u32)Domain::CHAIN_QUERY) return; // higher bits: the chain's label
    const u64 j = block[10] >> 16 | (u64)block[11] << 16;
    for (int w = j == 0 ? 4 : 0; w < 8; w++) {
      const int what = doing((size_t)(8 * j + w) - 4);
      if (what == 1) cv[2 * w] = cv[2 * w + 1] = 0xFFFFFFFFu;
      if (what == 2) cv[2 * w] = cv[2 * w + 1] = 0;
    }
  }
  // The same for the 64-bit form, which round_and_draw() and round_and_shut() use.
  static void compress64(u64 h[8], const u64 m[16], const Tweak &t) {
    Blake2bF::compress64(h, m, t);
    if ((m[0] & 0xFFFF) != (u64)Domain::CHAIN_QUERY) return;
    const u64 j = m[5] >> 16;
    for (int w = j == 0 ? 4 : 0; w < 8; w++) {
      const int what = doing((size_t)(8 * j + w) - 4);
      if (what == 1) h[w] = ~(u64)0;
      if (what == 2) h[w] = 0;
    }
  }
};

static int failures = 0;
#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      failures++; \
    } \
  } while (0)

int main() {
  FirstQuery q;
  // `q`: a default `FirstQuery`, what a chain fixes before the prover speaks (label 0, no
  // statement). ChainKind::ALGEBRAIC: a round whose verifier's message is elements of E.
  // `row`: the first 92 words of the verifier's message on the true function, taken from
  // chain_call() (blocks/hash/hash.h) directly, not through next_word().
  ChainT<Blake2bF> plain(q);
  ChainAbsorbT<Blake2bF> message = plain.message();
  message.push(E::one());
  const ChainPrefixT<Blake2bF> p = ChainT<Blake2bF>::prefix(message, ChainKind::ALGEBRAIC, 0);
  std::vector<u64> row;
  for (u64 j = 0; j < 12; j++)
    for (int w = j == 0 ? 4 : 0; w < 8; w++) row.push_back(chain_call(p, 0, j).w[w]);
  for (u64 w : row) {
    F f;
    if (!F_from_word(w, f) || w == 0) return std::printf("border_test: an unlucky word of the true function; change the message\n"), 2;
  }
  auto chain = [&] {
    ChainT<Doctored> c(q);
    ChainAbsorbT<Doctored> m = c.message();
    m.push(E::one());
    c.close(m, ChainKind::ALGEBRAIC);
    return c;
  };
  auto element = [&](size_t a, size_t b, size_t c) { // coordinates row[a], row[b], row[c]
    E x;
    CHECK(F_from_word(row[a], x.c[0]) && F_from_word(row[b], x.c[1]) && F_from_word(row[c], x.c[2]));
    return x;
  };
  // round_and_draw() must agree with close() and separate draws: two elements, value(), rounds();
  // round_and_shut() must give the same value().
  auto whole_agrees = [&](bool nonzero) {
    ChainT<Doctored> a = chain(), b(q);
    const E one = E::one();
    const u64 words[3] = {one.c[0].v, one.c[1].v, one.c[2].v};
    E out[2];
    b.round_and_draw(words, ChainKind::ALGEBRAIC, out, nonzero);
    const E first = nonzero ? a.draw_E_nonzero() : a.draw_E(), second = nonzero ? a.draw_E_nonzero() : a.draw_E();
    ChainT<Doctored> silent(q);
    silent.round_and_shut(words, ChainKind::ALGEBRAIC);
    // With one element, round_and_draw() may take the shortcut draw_E_straight().
    ChainT<Doctored> single(q);
    E one_out[1];
    single.round_and_draw(words, ChainKind::ALGEBRAIC, one_out, nonzero);
    // draw_E_straight() itself returns the ordinary draw's element or false, and false exactly if
    // one of the first three words is rejected, or the element is 0 and `nonzero` is set.
    ChainT<Doctored> words_of = chain();
    const u64 w0 = words_of.next_word(), w1 = words_of.next_word(), w2 = words_of.next_word();
    E straight;
    const bool taken = draw_E_straight(w0, w1, w2, nonzero, straight);
    E of_the_words;
    const bool all_good = F_from_word(w0, of_the_words.c[0]) && F_from_word(w1, of_the_words.c[1]) && F_from_word(w2, of_the_words.c[2]);
    const bool roads_agree = taken ? straight == first && all_good : (!all_good || (nonzero && of_the_words == E::zero()));
    return out[0] == first && out[1] == second && a.value() == b.value() && a.rounds() == b.rounds() && silent.value() == a.value() &&
           one_out[0] == first && single.value() == a.value() && roads_agree;
  };
  CHECK(chain().value() == [&] { ChainT<Blake2bF> c = plain; ChainAbsorbT<Blake2bF> m = c.message(); m.push(E::one()); c.close(m, ChainKind::ALGEBRAIC); return c.value(); }());

  // A rejected word is skipped, not reduced; the element is not restarted; also across hash calls.
  for (unsigned pattern = 0; pattern < 256; pattern++) {
    doctor.assign(8, 0);
    std::vector<size_t> good;
    for (size_t i = 0; i < 8; i++) doctor[i] = (pattern >> i) & 1;
    bool capped = false; // six rejected words in a row for one coordinate: the cap, tested below
    for (size_t i = 0, run = 0; i < 8; i++) run = doctor[i] ? run + 1 : 0, capped = capped || run >= DRAW_TRIES;
    if (capped) continue;
    for (size_t i = 0; i < row.size(); i++)
      if (!doing(i)) good.push_back(i);
    ChainT<Doctored> c = chain();
    CHECK(c.draw_E() == element(good[0], good[1], good[2]) && c.draw_E() == element(good[3], good[4], good[5]));
    CHECK(whole_agrees(false));
  }
  // The cap DRAW_TRIES = 6 (blocks/base.h): after five rejected words the sixth is taken; after six
  // the coordinate is 0, the six words are used up, and the next coordinate starts at the seventh.
  for (size_t at = 0; at < 3; at++) { // the coordinate whose words are rejected
    doctor.assign(at + DRAW_TRIES - 1, 0);
    for (size_t i = at; i < doctor.size(); i++) doctor[i] = 1;
    size_t next[3] = {0, 1, 2};
    for (size_t k = at; k < 3; k++) next[k] = k + DRAW_TRIES - 1;
    ChainT<Doctored> c = chain();
    CHECK(c.draw_E() == element(next[0], next[1], next[2]));
    doctor.push_back(1);
    for (size_t k = at; k < 3; k++) next[k] = k + DRAW_TRIES - 1; // `next` is not read after this
    E want = element(0, 1, 2);
    want.c[at] = F::zero();
    for (size_t k = at + 1; k < 3; k++) CHECK(F_from_word(row[k + DRAW_TRIES - 1], want.c[k]));
    c = chain();
    CHECK(c.draw_E() == want);
    CHECK(c.next_word() == row[2 + DRAW_TRIES]); // 3 coordinates, of which one took six words
    CHECK(whole_agrees(false) && whole_agrees(true));
  }
  {
    // All 18 words rejected: the element is 0, and the next word is word 18.
    doctor.assign(3 * DRAW_TRIES, 1);
    ChainT<Doctored> c = chain();
    CHECK(c.draw_E() == E::zero() && c.next_word() == row[3 * DRAW_TRIES]);
    // For draw_E_nonzero() that 0 is the first of NONZERO_TRIES = 2 elements; the second is taken.
    c = chain();
    CHECK(c.draw_E_nonzero() == element(18, 19, 20));
    // Both are 0, by rejected words: the result is 1, after 36 words.
    doctor.assign(6 * DRAW_TRIES, 1);
    c = chain();
    CHECK(c.draw_E_nonzero() == E::one() && c.next_word() == row[6 * DRAW_TRIES]);
    CHECK(whole_agrees(false) && whole_agrees(true));
  }
  {
    // Three words 0 make the element 0: draw_E_nonzero() skips it, and returns 1 after six words 0.
    doctor.assign(3, 2);
    ChainT<Doctored> c = chain(), d = chain();
    CHECK(c.draw_E() == E::zero());
    CHECK(d.draw_E_nonzero() == element(3, 4, 5) && d.draw_E() == element(6, 7, 8));
    doctor.assign(6, 2);
    c = chain();
    CHECK(c.draw_E_nonzero() == E::one() && c.next_word() == row[6]);
    CHECK(whole_agrees(false) && whole_agrees(true));
  }
  std::printf(failures ? "border_test: %d FAILURES\n" : "border_test: all passed\n", failures);
  return failures ? 1 : 0;
}
