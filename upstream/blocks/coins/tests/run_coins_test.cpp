// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests how one run of a program gets its private random values ("coins"):
// blocks/coins/run_coins.h, and blocks/coins/tests/repeatable_coins.h, which derives repeatable
// coins from a public number and compiles only after the #define below. Of run_coins() and
// run_coins_without_a_seed() the test checks only that two calls with the same arguments differ.
// Build and run from blocks/: make OUT=<directory> test
#define BLK_REPEATABLE_COINS_HIDE_NOTHING 1
#include "coins/tests/repeatable_coins.h"

#include <cstdio>
#include <set>

using namespace blk;
static int checks = 0, bad = 0;
static void check(bool ok, const char *what) {
  checks++;
  if (!ok) bad++, printf("FAILED: %s\n", what);
}
static std::string hex(const Digest &d) {
  u8 b[DIGEST_BYTES];
  digest_to_bytes(d, b);
  std::string s;
  for (u8 x : b) s += "0123456789abcdef"[x >> 4], s += "0123456789abcdef"[x & 15];
  return s;
}
int main() {
  coins_own_test();
  // One known answer, from Python's hashlib; 176 is Domain::COIN_INPUTS (blocks/hash/domains.h):
  //   import hashlib,struct; q=lambda x: struct.pack('<Q',x)
  //   m=q(176)+q(0)+q(4)+b'text'+q(3)+b'abc'+q(1)+b'w'+q(0)
  //   print(hashlib.blake2b(m, digest_size=32).hexdigest())
  check(hex(CoinInputs().add("text", std::string("abc")).add("w", std::string()).finish()) == "2e1351a95d65528ac482dbf13439ac508771d627138c3162d944ae83c9b24bd4", "the inputs' digest is the page's hash of the items");
  std::set<std::string> seen;
  seen.insert(hex(CoinInputs().finish()));
  seen.insert(hex(CoinInputs().add("a", std::string("bc")).finish()));
  seen.insert(hex(CoinInputs().add("ab", std::string("c")).finish()));
  seen.insert(hex(CoinInputs().add("a", std::string("b")).add("c", std::string()).finish()));
  seen.insert(hex(CoinInputs().add("a", std::string("b")).add("", std::string("c")).finish()));
  seen.insert(hex(CoinInputs().add("a", std::string()).add("b", std::string("c")).finish()));
  check(seen.size() == 6, "items that are cut otherwise give other digests");
  const Digest in = CoinInputs().add("text", std::string("abc")).finish(), other = CoinInputs().add("text", std::string("abd")).finish();
  F a[8], b[8];
  auto differ = [&] {
    bool d = false;
    for (int i = 0; i < 8; i++) d |= a[i] != b[i];
    return d;
  };
  run_coins_that_hide_nothing(5, in)->fill(a, 8), run_coins_that_hide_nothing(5, in)->fill(b, 8);
  check(!differ(), "a run that can be repeated repeats");
  run_coins_that_hide_nothing(6, in)->fill(b, 8);
  check(differ(), "another number, other coins");
  run_coins_that_hide_nothing(5, other)->fill(b, 8);
  check(differ(), "other inputs, other coins");
  run_coins_that_hide_nothing(5, in, CoinContext{1, 0, 0, 0})->fill(b, 8);
  check(differ(), "another context, other coins");
  run_coins_without_a_seed(in)->fill(a, 8), run_coins_without_a_seed(in)->fill(b, 8);
  check(differ(), "two runs on the system's bytes differ");
  CoinKey seed{};
  seed.w[3] = 9;
  run_coins(seed, 6, in)->fill(a, 8), run_coins(seed, 6, in)->fill(b, 8);
  check(differ(), "two runs under one seed and the same inputs differ by their fresh bytes");
  { // `run_coins_that_hide_nothing` for 5, by the steps of blocks/coins/coins.h; w[7] spells "TEST"
    CoinKey s{};
    s.w[0] = 5, s.w[7] = 0x54534554u;
    const u8 zeros[32] = {};
    StreamCoins by_hand(process_key(seed_child(s, 0, 0), zeros, in), CoinContext{});
    by_hand.fill(a, 8), run_coins_that_hide_nothing(5, in)->fill(b, 8);
    check(!differ(), "seed, child, process key, context: coins.h's way");
  }
  printf("the coins of a run, %d checks: %s\n", checks, bad ? "SOME FAILED" : "all passed");
  return bad != 0;
}
