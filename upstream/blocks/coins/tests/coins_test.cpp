// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/coins/coins.h: the prover's private random values ("coins"), which no verifier can
// check. That header defines key, context, stream, call, lane and retry. Run from blocks/.
//  1. ChaCha20's block function against the vector of RFC 8439, 2.3.2.
//  2. blocks/coins/tests/coin_values.txt, which blocks/coins/tests/coin_values.py computes from the
//     RFC and the rules stated in coins.h, using none of its code: keys, contexts, words, and
//     elements of the field. At least three of the listed words are not below
//     p = 2^64 - 2^32 + 1, so the generator rejects ("refuses") them and retries.
//  3. Forced rejections: block functions (Refusing, Edge) that overwrite some of ChaCha20's words.
//  4. StreamCoins: each fill() or bytes() opens the next stream, from COIN_FIRST_STREAM = 16384
//     up; again(), ledger(), last_draw(), not_the_coins(); let_go() and take_over(); misuse aborts.
//  5. Elements are below p, and each of their bits is set about half the time (p used as a bit
//     mask would show here).
//  6. fresh32(): which faults of the system's random source abort the process, and which do not.
#include "coins/coins.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
using namespace blk;

static int bad = 0, checks = 0;
static void check(bool ok, const std::string &what) {
  checks++;
  if (ok) return;
  bad++;
  printf("\nFAILED: %s", what.c_str());
}
template <class Call> static bool aborts(Call f) { // in a child process whose messages are dropped
  fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    if (!freopen("/dev/null", "w", stderr)) _exit(3);
    f();
    _exit(0);
  }
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
static bool same(const CoinKey &a, const CoinKey &b) { return !memcmp(a.w, b.w, 32); }
static CoinKey key_of(std::istringstream &s) {
  CoinKey k;
  for (u32 &w : k.w) s >> std::hex >> w;
  return k;
}

// coin_values.py's seed, fresh bytes and digest of inputs, and their keys.
struct TestKeys {
  CoinKey seed, child, process;
  u8 fresh[32];
  Digest inputs;
  TestKeys() {
    for (u32 i = 0; i < 8; i++) {
      seed.w[i] = 0x01010101u * (i + 1);
      const u32 w = 0x10203040u + i;
      for (int b = 0; b < 4; b++) fresh[4 * i + b] = (u8)(w >> 8 * b);
      inputs.w[i] = 0x0A0B0C0Du * (i + 1);
    }
    child = seed_child(seed, 6, 0);
    process = process_key(child, fresh, inputs);
  }
};
// coin_values.txt's second `context` line; the file's `coin` lines are computed under this context.
static const CoinContext SECOND{5, 0x81902, 1, 3};

static void rfc_vector() {
  u32 key[8], out[16];
  for (u32 i = 0; i < 8; i++) key[i] = 0x03020100u + 0x04040404u * i;
  chacha20_block(out, key, 1, 0x09000000u, 0x4a000000u, 0);
  const u32 want[16] = {0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
                        0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2};
  check(!memcmp(out, want, 64), "the block of RFC 8439, 2.3.2");
}

static void known_answers() {
  const TestKeys t;
  const CoinKey K = context_key(t.process, SECOND);
  std::ifstream in("coins/tests/coin_values.txt");
  int keys = 0, contexts = 0, coins = 0, refused = 0;
  for (std::string l; std::getline(in, l);) {
    std::istringstream s(l);
    std::string kind;
    s >> kind;
    if (kind == "child") check(same(key_of(s), t.child), "a child of the seed"), keys++;
    if (kind == "process") check(same(key_of(s), t.process), "a run's process key"), keys++;
    if (kind == "making") {
      Digest table, other;
      for (u32 i = 0; i < 8; i++) table.w[i] = other.w[i] = 0x0F0E0D0Cu * (i + 1);
      other.w[7] ^= 1;
      check(same(key_of(s), making_key(t.child, t.fresh, table)), "the secret of a table's making"), keys++;
      check(!same(making_key(t.child, t.fresh, other), making_key(t.child, t.fresh, table)), "another table has another secret");
    }
    if (kind == "context") {
      CoinContext c;
      s >> std::hex >> c.counter >> c.version >> c.object >> c.member;
      check(same(key_of(s), context_key(t.process, c)), "a context's key"), contexts++;
    }
    if (kind == "coin") {
      u32 stream;
      u64 i, word, e[2];
      int k[2];
      s >> std::hex >> stream >> i >> word >> e[0] >> e[1] >> std::dec >> k[0] >> k[1];
      check(coin_word(K, stream, i) == word, "a word of a stream");
      check(coin_element(K, stream, i).v == e[1], "an element of a stream"); // the file's second column of elements and of retries is this field's
      coins++, refused += k[1] > 0;
    }
  }
  check(keys == 3 && contexts == 3 && coins >= 17, "coin_values.txt has its lines");
  check(refused >= 3, "coin_values.txt holds places that are really refused");
}

// ChaCha20's block, but at some retry words `c` of calls 1 and 2 every lane is 64 ones, which the
// field rejects. `a` is the low word of the call's number; `c` is 0, or 8 k + lane at a lane's k-th
// retry.
struct Refusing {
  void operator()(u32 out[16], const u32 key[8], u32 a, u32 b, u32 c, u32 d) const {
    chacha20_block(out, key, a, b, c, d);
    const bool all = c == 0 && (a == 1 || a == 2), again = a == 2 && (c == 8 + 1 || c == 8 + 3 || c == 16 + 3);
    if (all || again)
      for (int i = 0; i < 8; i++) out[i] = ~0u;
  }
};
static void forced_refusals() {
  const TestKeys t;
  const CoinKey K = context_key(t.process, SECOND);
  for (u64 i = 0; i < 16; i++) {
    const u64 call = i / 4;
    const u32 lane = (u32)(i % 4);
    u32 retry = 0; // the retry word whose lane 0 must be taken
    if (call == 1) retry = 8 + lane;
    if (call == 2) retry = lane == 1 ? 16 + 1 : lane == 3 ? 24 + 3 : 8 + lane;
    F want;
    check(F_from_word(retry ? coin_lane(K, 16384, call, retry, 0) : coin_lane(K, 16384, call, 0, lane), want), "the test's own word is admitted");
    check(coin_element(K, 16384, i, Refusing()) == want, "after refusals: lane 0 of the call with the retry word 8 k + lane");
  }
  StreamCoinsT<Coins, Refusing> coins(t.process, SECOND);
  F got[16];
  coins.fill(got, 16);
  for (u64 i = 0; i < 16; i++) check(got[i] == coin_element(K, 16384, i, Refusing()), "fill() takes the same way after a refusal");
}

// ChaCha20's block, but at retry 0 every lane is `word`.
struct Edge {
  u64 word;
  void operator()(u32 out[16], const u32 key[8], u32 a, u32 b, u32 c, u32 d) const {
    chacha20_block(out, key, a, b, c, d);
    if (c == 0)
      for (int i = 0; i < 4; i++) out[2 * i] = (u32)word, out[2 * i + 1] = (u32)(word >> 32);
  }
};
static void edges() {
  const TestKeys t;
  const CoinKey K = context_key(t.process, SECOND);
  const u64 bits = ~(u64)0; // bits that `F_from_word` reads
  for (u64 i = 0; i < 4; i++) {
    F retried;
    check(F_from_word(coin_lane(K, 16384, 0, 8 + (u32)i, 0), retried), "the test's own retry word is admitted");
    check(coin_element(K, 16384, i, Edge{P - 1}) == F{P - 1} && coin_element(K, 16384, i, Edge{0}) == F::zero() && coin_element(K, 16384, i, Edge{1}) == F{1},
          "0, 1 and p - 1 are taken as they are");
    check(coin_element(K, 16384, i, Edge{P}) == retried, "the word p is refused: it does not become 0");
    check(coin_element(K, 16384, i, Edge{~(u64)0}) == retried, "the word of 64 ones is refused");
    if (P + 1 <= bits) check(coin_element(K, 16384, i, Edge{P + 1}) == retried, "the word p + 1 is refused: it does not become 1");
  }
}

static void streams() {
  const TestKeys t;
  const CoinKey K = context_key(t.process, SECOND);
  StreamCoins coins(t.process, SECOND);
  F a[10], one, again[4];
  u8 b[37], c[16], part[9];
  coins.fill(a, 10), coins.bytes(b, 37), coins.fill(&one, 1), coins.bytes(c, 16);
  bool ok = true;
  for (u64 i = 0; i < 10; i++) ok = ok && a[i] == coin_element(K, 16384, i);
  for (u64 i = 0; i < 37; i++) ok = ok && b[i] == (u8)(coin_word(K, 16385, i / 8) >> 8 * (i % 8));
  ok = ok && one == coin_element(K, 16386, 0);
  for (u64 i = 0; i < 16; i++) ok = ok && c[i] == (u8)(coin_word(K, 16387, i / 8) >> 8 * (i % 8));
  check(ok, "every call draws from a stream of its own, in the order of the calls");
  check(coins.last_stream() == 16387, "the last stream");
  coins.again(16384, 5, again, 4), coins.again(16385, 11, part, 9);
  check(!memcmp(again, a + 5, sizeof again) && !memcmp(part, b + 11, 9), "again() gives the same coins");
  {  // `Coins` (blocks/engine/table.h): the last draw's number; how many values differ from a draw
    F lost[10];
    std::copy(a, a + 10, lost);
    lost[0] = lost[0] + F::one(), lost[9] = lost[9] + F::one();
    const Coins &asked = coins;
    check(asked.last_draw() == 16387 && asked.not_the_coins(16384, a, 10, {}) == 0 && asked.not_the_coins(16384, lost, 10, {}) == 2 && asked.not_the_coins(16384, lost, 10, {0}) == 1 &&
              asked.not_the_coins(16384, lost, 10, {0, 9}) == 0,
          "not_the_coins() counts each lost value, but at a place that is left out");
    struct Numbers : Coins {  // numbers its draws but keeps the default `not_the_coins`
      void fill(F *, size_t) override {}
      void bytes(u8 *, size_t) override {}
      u32 last_draw() const override { return 7; }
    } numbers;
    check(numbers.not_the_coins(7, a, 10, {}) == 10, "the default answer is «all differ»: a look that sees nothing is loud");
  }
  const std::vector<CoinDraw> &l = coins.ledger();
  check(l.size() == 4 && l[0].stream == 16384 && l[0].elements && l[0].count == 10 && !l[1].elements && l[1].count == 37 &&
            l[2].count == 1 && l[3].stream == 16387 && l[3].count == 16, "the ledger");
  check(aborts([&] { coins.again(16388, 0, again, 1); }), "again() on a stream that was never drawn is refused");
  check(aborts([&] { coins.again(16384, 7, again, 4); }), "again() beyond what was drawn is refused");
  check(aborts([&] { coins.again(16385, 0, again, 1); }), "again() for elements on a stream of bytes is refused");
  check(aborts([&] { coins.again(16383, 0, again, 1); }), "a stream below 16,384 is refused");
  check(aborts([&] {
          std::thread other([&] { coins.fill(again, 1); });
          other.join();
        }) &&
            aborts([&] {
              std::thread other([&] { coins.last_stream(); });
              other.join();
            }) &&
            aborts([&] {
              std::thread other([&] { coins.again(16384, 0, again, 1); });
              other.join();
            }),
        "a call by another thread than the holder ends the process");
  check(!aborts([&] {
          coins.let_go();
          std::thread other([&] {
            coins.take_over();
            coins.fill(again, 1);
            coins.let_go();
          });
          other.join();
          coins.take_over();
          coins.fill(again, 1);
        }),
        "from hand to hand and back: whoever has taken over draws");
  check(aborts([&] {
          std::thread other([&] { coins.take_over(); });
          other.join();
        }),
        "a thread that takes over while the holder has not let go ends the process");
  check(aborts([&] { coins.let_go(), coins.fill(again, 1); }), "whoever has let go draws no more");
  check(aborts([&] {
          std::thread other([&] { coins.let_go(); });
          other.join();
        }),
        "only the holder lets go");
  check(aborts([&] { coins.take_over(); }), "the holder itself cannot take over a second time");
  check(!std::is_copy_constructible<StreamCoins>::value && !std::is_copy_assignable<StreamCoins>::value, "coins cannot be copied");
  F x[4][3];
  CoinContext other = SECOND;
  other.member++;
  u8 fresh2[32];
  memcpy(fresh2, t.fresh, 32), fresh2[31] ^= 0x80;
  Digest inputs2 = t.inputs;
  inputs2.w[7] ^= 1;
  StreamCoins c0(t.process, SECOND), c1(t.process, other), c2(process_key(t.child, fresh2, t.inputs), SECOND), c3(process_key(t.child, t.fresh, inputs2), SECOND);
  c0.fill(x[0], 3), c1.fill(x[1], 3), c2.fill(x[2], 3), c3.fill(x[3], 3);
  bool differ = true;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < i; j++) differ = differ && memcmp(x[i], x[j], sizeof x[i]);
  check(differ && !memcmp(x[0], a, sizeof x[0]), "another member, other fresh bytes or other inputs give other coins; the same give the same");
}

static void distribution() {
  const TestKeys t;
  StreamCoins coins(t.process, SECOND);
  std::vector<F> e(1 << 16);
  coins.fill(e.data(), e.size());
  u64 ones[64] = {0};
  bool below = true;
  for (const F &x : e) {
    below = below && x.v < P;
    for (int b = 0; b < 64; b++) ones[b] += x.v >> b & 1;
  }
  check(below, "every element is below p");
  bool fair = true;
  for (unsigned b = 0; b < F_BITS; b++) fair = fair && ones[b] > e.size() * 15 / 32 && ones[b] < e.size() * 17 / 32;
  for (unsigned b = F_BITS; b < 64; b++) fair = fair && ones[b] == 0;
  check(fair, "each of the element's bits is set about half the time");
}

static void the_source() {
  u8 out[32];
  // gives() makes stand-ins for getrandom().
  auto gives = [](int bytes, u8 value, int interrupts) {
    return [=](void *p, size_t, unsigned) mutable -> ssize_t {
      if (interrupts-- > 0) {
        errno = EINTR;
        return -1;
      }
      if (bytes < 0) {
        errno = EIO;
        return -1;
      }
      memset(p, 0, 32);
      static_cast<u8 *>(p)[31] = value;
      return bytes;
    };
  };
  check(aborts([&] { fresh32(out, gives(-1, 1, 0)); }), "a source that fails ends the process");
  check(aborts([&] { fresh32(out, gives(31, 1, 0)); }), "a source that gives 31 bytes ends the process");
  check(aborts([&] { fresh32(out, gives(32, 0, 0)); }), "a source that gives 32 zero bytes ends the process");
  check(!aborts([&] { fresh32(out, gives(32, 1, 0)); }), "31 zero bytes and one other are taken");
  check(!aborts([&] { fresh32(out, gives(32, 1, 1)); }), "a call that a signal interrupted is made again");
  u8 a[32], b[32];
  fresh32(a), fresh32(b);
  check(memcmp(a, b, 32) != 0, "two calls of the system's source differ");
}

// No run in this repository shows that coins_own_test() or edges() catches a generator that rejects
// no word.
int main() {
  coins_own_test();
  rfc_vector();
  known_answers();
  forced_refusals();
  edges();
  streams();
  distribution();
  the_source();
  printf("%sthe prover's coins, field %d, %d checks: %s\n", bad ? "\n" : "", BLK_FIELD, checks, bad ? "SOME FAILED" : "all passed");
  return bad != 0;
}
