// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests acctLinesH() and acctReadH() (verifier/src/vf_accounting_record.h): the text record in which a
// stage of the verifier program `cp_verify`, a child process, passes its error accounting
// (blk::Accounting: a summary of the rounds' soundness errors) to the parent process. A record
// read back must print exactly as the accounting that it was written from; an accounting that does
// not hold gives no record; text with no complete record, or with more than one, is not read.
// No `cp_verify` process is run.
// Usage: plan_record_test. Exit code: 0 if every check passed, 1 if not, 2 if printed() could
// not capture any output.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "numbers/accounting.h"
using blk::u32;
using blk::u64;
using std::floor;
using std::string;
using std::vector;
#include "vf_accounting_record.h"

static int failed = 0, made = 0;
static void want(bool ok, const string &what) {
  made++;
  if (ok) return;
  failed++;
  printf("not as expected: %s\n", what.c_str());
}
// Returns what print() writes to standard output, captured in a temporary file in TMPDIR, or in
// /tmp. If nothing can be captured, the test exits with code 2: two empty strings would compare
// as equal in main().
template <class Print> static string printed(Print print) {
  const char *set = getenv("TMPDIR");
  const string folder = set && *set ? set : "/tmp";
  string place = folder + "/plan_record_XXXXXX";
  char *name = &place[0];
  const int file = mkstemp(name);
  if (file < 0) {
    printf("plan_record_test: stopped: %s: no temporary file could be made in %s. Set TMPDIR to a folder that can be written\n", strerror(errno), folder.c_str());
    exit(2);
  }
  const int out = dup(1);
  if (out < 0) {
    unlink(name);
    fprintf(stderr, "plan_record_test: stopped: %s: standard output could not be copied\n", strerror(errno));
    exit(2);
  }
  fflush(stdout);
  dup2(file, 1);
  print();
  fflush(stdout);
  dup2(out, 1);
  close(out);
  string text;
  lseek(file, 0, SEEK_SET);
  char c;
  while (read(file, &c, 1) == 1) text.push_back(c);
  close(file);
  unlink(name);
  if (text.empty()) {
    printf("plan_record_test: stopped: nothing was caught in %s (is the disk full?)\n", folder.c_str());
    exit(2);
  }
  return text;
}
static vector<string> linesOf(const string &text) {
  vector<string> l(1);
  for (char c : text)
    if (c == '\n')
      l.emplace_back();
    else
      l.back().push_back(c);
  l.pop_back();
  return l;
}
static string textOf(const vector<string> &l) {
  string t;
  for (auto &x : l) t += x + "\n";
  return t;
}
// A synthetic accounting with eight groups of rounds (blk::AccountGroup). Bit k of `way` sets:
// 0 proof of work; 1 a count of hash calls, and `on_a_word`; 2 without bit 4, near_closeness[1];
// 3 `circle`; 4 proximity terms (`closeness`: error from rows only close to codewords) beyond a
// third of a code's distance; 5 an approximate range check (printed: `smallness test`).
static blk::Accounting madeUp(unsigned way) {
  blk::Accounting a;
  const bool work = way & 1;
  a.formed = a.holds = true, a.needs_work = work, a.target = 128, a.digest_bits = 256;
  a.rounds = 36767, a.queries = 30011, a.planned_calls = 2400000;
  struct Row {
    double bits;
    unsigned work;
    u64 rounds;
    const char *name;
  } rows[] = {{108.004, 20, 25, "positions (link #)"}, {96.031234, 32, 1, "columns of the table (a key)"}, {128.31, 0, 7, "a challenge"},   {129.5, 0, 1, "a sample"},
              {130.25, 0, 3, "a fold's round #"},     {131.279, 0, 900, "a sum-check's round"},           {140.0, 0, 12, "the seventh"}, {150.125, 0, 2, "the eighth"}};
  for (const Row &r : rows) {
    blk::AccountGroup g;
    g.bits = work ? r.bits : r.bits + r.work, g.work = work ? r.work : 0, g.rounds = r.rounds, g.name = r.name, g.list = r.work != 0;
    a.groups.push_back(g);
  }
  a.lists = work ? a.groups[1] : a.groups[0];
  a.worst = "positions (link 17)";
  a.sum = 123.624901, a.sum_lists = work ? 94.21 : 123.624901;
  a.counted = way & 2, a.calls = a.counted ? 2392828 : 0;
  if (work) a.work[3][0] = 20, a.work[3][1] = 28, a.work[3][2] = 32, a.work[2][2] = 5;
  a.circle = way & 8;
  a.closeness[0] = 40;
  if (way & 16) a.closeness[1] = 58;
  a.near = 40, a.near_small = way & 32 ? 25 : 0, a.near_closeness[0] = 1, a.near_closeness[1] = way & 16 ? 7 : way & 4 ? 2 : 0, a.by_work = work ? 26 : 0, a.on_a_word = way & 2 ? 311 : 0;
  if (way & 32) a.small_columns = 24150, a.small_union.bits = work ? 104.440001 : 112.440001, a.small_union.work = work ? 8 : 0, a.small_union.rounds = 12, a.small_union.name = "RNG (G0, smallness test)";
  return a;
}
int main() {
  // Round trip, with other lines around the record. A record keeps only the first ACCT_GROUPS_MOST
  // groups, which are all that blk::print_accounting() prints by default.
  for (unsigned way = 0; way < 64; way++) {
    const blk::Accounting a = madeUp(way);
    blk::Accounting b;
    const bool read = acctReadH("a line before\n" + acctLinesH(a) + "result: a line behind\n", b);
    want(read && printed([&] { blk::print_accounting(a, "a test"); }) == printed([&] { blk::print_accounting(b, "a test"); }), "way " + std::to_string(way) + ": the record read back prints as the accounting did");
    want(read && b.groups.size() == ACCT_GROUPS_MOST && b.needs_work == a.needs_work && b.rounds == a.rounds && b.queries == a.queries && b.planned_calls == a.planned_calls && b.calls == a.calls && b.near == a.near && b.near_small == a.near_small && b.near_closeness[0] == a.near_closeness[0] && b.near_closeness[1] == a.near_closeness[1] && b.near_closeness[2] == a.near_closeness[2] && b.by_work == a.by_work &&
             b.on_a_word == a.on_a_word,
         "way " + std::to_string(way) + ": the six largest groups, the word on work and the counts");
  }
  // Bits (-log2 of an error) are truncated to six decimals, never rounded up: a value read back
  // must not state a smaller error than the original.
  {
    blk::Accounting a = madeUp(1), b;
    a.lists.bits = a.groups[1].bits = 96.0399999999, a.sum = 123.9999999;
    want(acctReadH(acctLinesH(a), b) && b.lists.bits <= a.lists.bits && b.lists.bits > 96.0399 && b.sum <= a.sum && b.sum > 123.9999, "bits are cut to six decimals");
  }
  // No record of an accounting that was not formed or does not hold (`holds`: every round meets
  // the target, proof of work counted).
  {
    blk::Accounting a = madeUp(3), b;
    a.holds = false;
    want(acctLinesH(a).empty() && !acctReadH(acctLinesH(a), b) && !b.formed, "an accounting that does not hold gives no record");
    a = madeUp(3), a.formed = false;
    want(acctLinesH(a).empty(), "an accounting that was not formed gives no record");
    want(!acctReadH("", b) && !acctReadH("result: nothing\n", b), "a text without a record");
  }
  // Text with an incomplete, malformed or repeated record is rejected. Lines 0 to 4 of `whole`:
  // plan-sums, plan-rests, plan-near, plan-worst, plan-lists; then ACCT_GROUPS_MOST lines plan-round.
  // Not caught: acctReadH() accepts 1 to ACCT_GROUPS_MOST plan-round lines and a plan-union line or none,
  // so a record that has lost some of these is still read.
  const vector<string> whole = linesOf(acctLinesH(madeUp(31)));
  want(whole.size() == 5 + ACCT_GROUPS_MOST && linesOf(acctLinesH(madeUp(63))).size() == 6 + ACCT_GROUPS_MOST, "a record has eleven lines, and one more where the plan has a smallness test");
  blk::Accounting b;
  for (size_t i = 0; i < 5; i++) {
    vector<string> l = whole;
    l.erase(l.begin() + (long)i);
    want(!acctReadH(textOf(l), b) && !b.formed, "without its line " + std::to_string(i + 1));
    l = whole, l.push_back(whole[i]);
    want(!acctReadH(textOf(l), b), "with its line " + std::to_string(i + 1) + " twice");
  }
  {
    vector<string> l(whole.begin(), whole.begin() + 5);
    want(!acctReadH(textOf(l), b), "without a group of rounds");
    l = whole, l.push_back(whole.back());
    want(!acctReadH(textOf(l), b), "with a seventh group");
    want(!acctReadH(textOf(whole) + textOf(whole), b), "two records");
    l = whole, l[0] = " " + l[0];
    want(!acctReadH(textOf(l), b), "a head that does not begin its line is none");
  }
  auto with = [&](size_t line, const string &text) {
    vector<string> l = whole;
    l[line] = text;
    return acctReadH(textOf(l), b);
  };
  want(with(1, whole[1]) && !with(1, whole[1] + " 0") && !with(1, whole[1].substr(0, whole[1].rfind(' '))), "the line plan-rests has 24 numbers, no more and no fewer");
  // " 40 " is the line's fourth number, closeness[0].
  want(!with(1, "plan-rests: 128 256 2" + whole[1].substr(whole[1].find(" 40 "))) && !with(1, "plan-rests: 1000 256 1" + whole[1].substr(whole[1].find(" 40 "))), "a flag that is neither 0 nor 1; a target beyond all digests");
  want(!with(1, whole[1].substr(0, whole[1].rfind(' ')) + " 65") && !with(1, whole[1].substr(0, whole[1].rfind(' ')) + " -1") && !with(1, whole[1].substr(0, whole[1].rfind(' ')) + " x"), "work bits that are none");
  want(!with(0, "plan-sums: 0 0 123.0 94.0 1 5 5") && !with(0, "plan-sums: 10 9 nan 94.0 1 5 5") && !with(0, "plan-sums: 10 9 123.0 inf 1 5 5") && !with(0, "plan-sums: 10 9 123.0 94.0 2 5 5") &&
           !with(0, "plan-sums: 10 9 123.0 94.0 1 5"),
       "the line plan-sums: no rounds; numbers that are none; a flag of 2; a number short");
  want(with(2, "plan-near: 40 0 1 7 26 311") && b.near == 40 && b.near_closeness[1] == 7 && b.by_work == 26 && b.on_a_word == 311 && !with(2, "plan-near: 40 0 1 7 26") && !with(2, "plan-near: 40 0 1 7 26 311 0") &&
           !with(2, "plan-near: 40 0 1 x 26 311"),
       "the line plan-near has six numbers, no more and no fewer");
  want(!with(2, "plan-near: 7 0 1 7 26 311") && !with(2, "plan-near: 40 0 1 7 0 311"), "the parts are more than the whole; work is needed, and no round holds by its work");
  {
    const vector<string> plain = linesOf(acctLinesH(madeUp(30)));  // no work
    vector<string> l = plain;
    l[2] = "plan-near: 40 0 1 7 1 0";
    want(acctReadH(textOf(plain), b) && !b.needs_work && !b.by_work && !acctReadH(textOf(l), b), "no work is needed, and a round holds by its work");
  }
  want(!with(3, "plan-worst: "), "a worst round without a name");
  want(!with(4, "plan-lists: nan 32 1 1 a name") && !with(4, "plan-lists: 96.0 32 0 1 a name") && !with(4, "plan-lists: 96.0 32 1 2 a name") && !with(4, "plan-lists: 96.0 32 1 1 ") && !with(4, "plan-lists: 96.0 32 1 1"),
       "a group: bits that are no number; no rounds; a flag of 2; no name");
  want(!with(6, "plan-round: inf 0 1 0 a name"), "a group of rounds at infinity");
  {
    const string good = "plan-union: 112.44 0 12 24150 a name\n";
    want(acctReadH(textOf(whole) + good, b) && b.small_columns == 24150 && !acctReadH(textOf(whole) + good + good, b), "the line plan-union: once, not twice");
    for (const char *bad : {"plan-union: nan 0 12 24150 a name\n", "plan-union: 112.44 0 0 24150 a name\n", "plan-union: 112.44 0 12 0 a name\n", "plan-union: 112.44 0 12 24150 \n", "plan-union: 112.44 0 12\n"})
      want(!acctReadH(textOf(whole) + bad, b), string("not read: ") + bad);
  }
  printf("a stage's record of its accounting: %d checks, %d failed%s\n", made, failed, failed ? "" : ": all as expected");
  return failed ? 1 : 0;
}
