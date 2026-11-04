// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The numbers in this file are the program's own estimate. They are not a security level and must
// not be quoted as one.
//
// Each stage of the verifier program `cp_verify` is a child process. This file carries a stage's
// accounting of soundness errors (blk::Accounting) to the parent, as text lines in the stage's
// output: acctLinesH() writes this record and acctReadH() reads it back. blk::account() forms an
// accounting and blk::print_accounting() prints it (blocks/numbers/accounting.h, which states its
// rule). Tested by verifier/tests/plan_record_test.cpp.

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include "numbers/accounting.h" // ../../blocks
static constexpr size_t ACCT_GROUPS_MOST = 6;
/* The stages `key` and `leaf` send the accounting of the plan (the list of rounds) of the key file
   (`ivk/key.bin`) and a leaf file (`ivk/leaf/<k>.bin`). The last stage, `request`, sends one
   accounting of those two plans and the proof file's (printed: "key, entry and proof"), formed from
   the configuration only, as a planning call (`-plan accounting`) forms it. So the parent prints
   the last stage's record, with all processes' hash calls added.
   Bits are -log2 of an error, cut to six decimals, never rounded up. Work is bits of proof of work.
   A round holds if its error times 2^-work ("work counted") is at most 2^-target. A group
   (blk::AccountGroup) is the rounds with one name, work and printed error. The enumerations are
   defined in blocks/numbers/setting.h, `Priced` in blocks/numbers/plan.h.
     plan-sums:  <rounds> <rounds with a verifier's message>
                 <all rounds' errors added, in bits, with the work> <without>
                 <1: the next number counts a run's hash calls> <that number> <the planned calls>
     plan-rests: <target> <a digest's bits> <1: some round needs its work>
                 <rounds with a proximity term (printed: "closeness term"): two numbers, by
                 CountSource>
                 <1: circle codes>
                 <bits of work by ErrorKind and `Scope`: 18 numbers, `Scope` varying fastest>
     plan-near:  <rounds that hold with less than 0.01 bit to spare, work counted>
                 <of these: the approximate range check's (printed: "smallness test")>
                 <of these: with a proximity term, by CountSource: two numbers>
                 <rounds that hold only because their work is counted>
                 <rounds marked Priced::ON_THE_CALLERS_WORD: the error rests on a number not checked
                 at that round>
     plan-worst: <the name of the round with the largest error, work counted>
     plan-lists: <bits without the work> <work> <rounds> <1: the error of a list of positions>
                 <name>     the group with the largest error if work is not counted
     plan-round: the same, for at most ACCT_GROUPS_MOST groups: the largest errors, work counted,
                 largest first
     plan-union: <bits without the work> <work> <rounds> <columns> <name>     only if the plan has a
                 range check: its worst round under a union bound over its matrix's columns
   No record is written of an accounting that does not hold: a stage rejects earlier if a round of
   its file's plan does not. */
static std::string acctLinesH(const blk::Accounting &a) {
  if (!a.formed || !a.holds) return "";
  char b[256];
  auto cut = [](double x) { return floor(1e6 * x) / 1e6; };
  snprintf(b, sizeof b, "plan-sums: %llu %llu %.6f %.6f %d %llu %llu\n", (unsigned long long)a.rounds, (unsigned long long)a.queries, cut(a.sum), cut(a.sum_lists),
           a.counted ? 1 : 0, (unsigned long long)a.calls, (unsigned long long)a.planned_calls);
  std::string o = b;
  snprintf(b, sizeof b, "plan-rests: %u %u %d %llu %llu %d", a.target, a.digest_bits, a.needs_work ? 1 : 0, (unsigned long long)a.closeness[0],
           (unsigned long long)a.closeness[1], a.circle ? 1 : 0);
  o += b;
  for (auto &kind : a.work)
    for (unsigned g : kind) o += " " + std::to_string(g);
  snprintf(b, sizeof b, "\nplan-near: %llu %llu %llu %llu %llu %llu", (unsigned long long)a.near, (unsigned long long)a.near_small, (unsigned long long)a.near_closeness[0],
           (unsigned long long)a.near_closeness[1], (unsigned long long)a.by_work, (unsigned long long)a.on_a_word);
  o += b;
  o += "\nplan-worst: " + a.worst + "\n";
  auto group = [&](const char *head, const blk::AccountGroup &g) {
    snprintf(b, sizeof b, "%s %.6f %u %llu %d ", head, cut(g.bits), g.work, (unsigned long long)g.rounds, g.list ? 1 : 0);
    o += b + g.name + "\n";
  };
  group("plan-lists:", a.lists);
  for (size_t i = 0; i < a.groups.size() && i < ACCT_GROUPS_MOST; i++) group("plan-round:", a.groups[i]);
  if (a.small_columns) {
    snprintf(b, sizeof b, "plan-union: %.6f %u %llu %llu ", cut(a.small_union.bits), a.small_union.work, (unsigned long long)a.small_union.rounds, (unsigned long long)a.small_columns);
    o += b + a.small_union.name + "\n";
  }
  return o;
}
// Reads the one record in a stage's output. false: a line is missing (plan-union may be), occurs
// too often, does not parse, has a number out of range, or contradicts another number (plan-near's
// parts exceed its total; a.by_work against a.needs_work). true: a.holds is set; no error is
// compared. A call passes only if each stage's record is read (cpStagesRun() in cp_verify.cpp).
static bool acctReadH(const std::string &out, blk::Accounting &a) {
  a = blk::Accounting();
  std::istringstream in(out);
  std::string line;
  bool sums = false, rests = false, near = false, worst = false, lists = false;
  auto group = [](const std::string &s, blk::AccountGroup &g) {
    unsigned long long r = 0;
    int used = 0, list = 0;
    if (sscanf(s.c_str(), "%lf %u %llu %d %n", &g.bits, &g.work, &r, &list, &used) != 4 || !r || !std::isfinite(g.bits) || list < 0 || list > 1) return false;
    g.rounds = r, g.list = list == 1, g.name = s.substr((size_t)used);
    return !g.name.empty();
  };
  while (std::getline(in, line)) {
    if (!line.rfind("plan-sums: ", 0)) {
      unsigned long long r = 0, q = 0, c = 0, p = 0;
      int k = 0;
      if (sums || sscanf(line.c_str() + 11, "%llu %llu %lf %lf %d %llu %llu", &r, &q, &a.sum, &a.sum_lists, &k, &c, &p) != 7 || !r || k < 0 || k > 1 ||
          !std::isfinite(a.sum) || !std::isfinite(a.sum_lists))
        return false;
      a.rounds = r, a.queries = q, a.counted = k == 1, a.calls = c, a.planned_calls = p, sums = true;
    } else if (!line.rfind("plan-rests: ", 0)) {
      std::istringstream w(line.substr(12));
      unsigned long long x[24];
      for (auto &y : x)
        if (!(w >> y)) return false;
      std::string more;
      if (rests || (w >> more) || x[0] > 512 || x[1] > 512 || x[2] > 1 || x[5] > 1) return false;
      a.target = (unsigned)x[0], a.digest_bits = (unsigned)x[1], a.needs_work = x[2] == 1;
      for (int i = 0; i < 2; i++) a.closeness[i] = x[3 + i];
      a.circle = x[5] == 1;
      for (int i = 0; i < 18; i++) {
        if (x[6 + i] > 64) return false;
        a.work[i / 3][i % 3] = (unsigned)x[6 + i];
      }
      rests = true;
    } else if (!line.rfind("plan-near: ", 0)) {
      std::istringstream w(line.substr(11));
      unsigned long long x[6];
      for (auto &y : x)
        if (!(w >> y)) return false;
      std::string more;
      if (near || (w >> more) || x[1] + x[2] + x[3] > x[0]) return false;
      a.near = x[0], a.near_small = x[1], a.by_work = x[4], a.on_a_word = x[5], near = true;
      for (int i = 0; i < 2; i++) a.near_closeness[i] = x[2 + i];
    } else if (!line.rfind("plan-worst: ", 0)) {
      if (worst || line.size() == 12) return false;
      a.worst = line.substr(12), worst = true;
    } else if (!line.rfind("plan-lists: ", 0)) {
      if (lists || !group(line.substr(12), a.lists)) return false;
      lists = true;
    } else if (!line.rfind("plan-round: ", 0)) {
      blk::AccountGroup g;
      if (!group(line.substr(12), g)) return false;
      a.groups.push_back(g);
    } else if (!line.rfind("plan-union: ", 0)) {
      unsigned long long r = 0, c = 0;
      int used = 0;
      if (a.small_columns || sscanf(line.c_str() + 12, "%lf %u %llu %llu %n", &a.small_union.bits, &a.small_union.work, &r, &c, &used) != 4 || !r || !c || !std::isfinite(a.small_union.bits))
        return false;
      a.small_union.rounds = r, a.small_columns = c, a.small_union.name = line.substr(12 + (size_t)used);
      if (a.small_union.name.empty()) return false;
    }
  }
  a.formed = a.holds = sums && rests && near && worst && lists && !a.groups.empty() && a.groups.size() <= ACCT_GROUPS_MOST && (a.by_work != 0) == a.needs_work;
  return a.formed;
}
