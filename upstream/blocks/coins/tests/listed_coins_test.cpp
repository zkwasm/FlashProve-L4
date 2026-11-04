// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests ListedCoins (listed_coins.h): test-chosen values instead of a prover's private randomness
// (`Coins` in identifiers). The learning run (no lists) returns zeros and records each draw. Listed
// entries come back in list order; drawing beyond a list's end aborts the process; drawing less
// leaves all_drawn() false.
// Build and run from blocks/: make OUT=<directory> test
#define BLK_LISTED_COINS_FOR_A_TEST 1
#include "coins/tests/listed_coins.h"

#include <cstdio>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace blk;
static int checks = 0, bad = 0;
static void check(bool ok, const char *what) {
  checks++;
  if (!ok) bad++, printf("FAILED: %s\n", what);
}
template <class Call> static bool aborts(Call f) { // in a child process whose messages are dropped
  fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (!child) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    if (!freopen("/dev/null", "w", stderr)) _exit(9);
    f();
    _exit(0);
  }
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}
// A mock prover: five draws; e() draws 3 elements of F.
static void run(Coins &coins, F got[7], u8 salt[7]) {
  coins.fill(got, 3);
  coins.bytes(salt, 5);
  const E e = coins.e();
  for (int i = 0; i < 3; i++) got[3 + i] = e.c[i];
  coins.bytes(salt + 5, 2);
  got[6] = coins.f();
}
int main() {
  F got[7];
  u8 salt[7];
  ListedCoins first;
  run(first, got, salt);
  bool zeros = true;
  for (const F &x : got) zeros &= x == F::zero();
  for (u8 b : salt) zeros &= b == 0;
  check(zeros && first.elements().size() == 7 && first.bytes().size() == 7 && first.all_drawn(), "the learning run gives zeros and learns both lengths");
  // `Draw`: field elements (true) or bytes, first index in that list, count.
  const ListedCoins::Draw want[5] = {{true, 0, 3}, {false, 0, 5}, {true, 3, 3}, {false, 5, 2}, {true, 6, 1}};
  bool same = first.ledger().size() == 5;
  for (int i = 0; same && i < 5; i++) same = first.ledger()[i].elements == want[i].elements && first.ledger()[i].first == want[i].first && first.ledger()[i].count == want[i].count;
  check(same, "the ledger has a line a draw");
  std::vector<F> elements;
  std::vector<u8> bytes;
  for (u64 i = 0; i < 7; i++) elements.push_back(F{100 + i}), bytes.push_back((u8)(200 + i));
  ListedCoins next(elements, bytes);
  run(next, got, salt);
  bool played = next.all_drawn();
  for (u64 i = 0; i < 7; i++) played &= got[i] == F{100 + i} && salt[i] == 200 + i;
  check(played, "a played run gives the listed entries in the order of the draws");
  check(aborts([&] {
          ListedCoins few(std::vector<F>(6, F::zero()), bytes);
          run(few, got, salt);
        }) &&
            aborts([&] {
              ListedCoins few(elements, std::vector<u8>(6, 0));
              run(few, got, salt);
            }),
        "a draw beyond a list's end ends the process");
  ListedCoins more(std::vector<F>(8, F::zero()), bytes);
  run(more, got, salt);
  check(!more.all_drawn(), "a run that draws less than was listed is noticed");
  printf("listed coins, %d checks: %s\n", checks, bad ? "SOME FAILED" : "all passed");
  return bad != 0;
}
