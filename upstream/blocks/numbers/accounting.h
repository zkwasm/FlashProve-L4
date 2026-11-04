// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The numbers that this file prints are the program's own estimate. They are not a security level
// and must not be quoted as one.

/* The soundness-error accounting of a plan (`Planned`, plan.h: a protocol's rounds, each with an
   upper bound on its error). `account` computes it from the plan alone and reads no proof;
   `print_accounting` and the other print functions write it to standard output.

   The rule. The measure is round-by-round soundness (Canetti, Chen, Holmgren, Lombardi, Rothblum,
   Rothblum, Wichs, "Fiat-Shamir: from practice to theory", STOC 2019; for oracle proofs Chiesa,
   Manohar, Spooner, ePrint 2019/834; STIR, ePrint 2024/390, section 6.2, and
   WHIR, ePrint 2024/1586, section 6, choose their parameters so). Every round's error, proof of
   work counted, must be at most 2^-target (`Setting::target`, in bits), so the largest round error
   decides. The comparison is exact, on the upper bounds (`holds` in plan.h). Printed figures are
   cut to two decimals toward the weaker side and decide nothing.

   Proof of work. g bits before a verifier's message are counted as a factor 2^-g on that round's
   error (after Starkware, "ethSTARK documentation", ePrint 2021/582, Theorem 6, which tests the
   present round's work only; here all work so far is tested). For this program's hash chains the
   factor is an assumption: the theorem is proved for another setting.

   The compiled bound. With the verifier's messages taken from a random oracle by a hash chain and
   the oracles committed by Merkle trees with digests of D bits, a prover that makes t calls of the
   compression function (printed: hash calls) gets a false statement accepted with probability
   at most
       (t + q_V + 1) eps + c_h (t + q_V)^2 / 2^D,
   where eps is the largest round error, q_V the number of the verifier's own calls and c_h = 17/2
   (`COLLISION_CONSTANT`, below). The shape is that of Ben-Sasson, Chiesa, Spooner, "Interactive
   oracle proofs", ePrint 2016/116, as sharpened for round-by-round soundness in Chiesa, Yogev,
   "Building cryptographic proofs from hash functions"; for this protocol's compiler the bound is
   argued on paper, in a technical report that is not public, and it has not been
   refereed.

   The unit of all figures is one request's proof, not a batch's, as long as `Setting::instances`
   is 1 (see the last line that `print_accounting` prints). */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "numbers/blocks.h"
#include "numbers/plan.h"

namespace blk {

// Rounds with the same name, bits of proof of work and error (to two decimals). In the name each
// run of digits is replaced by '#'. `holds`: every round of the group holds at the target, work
// counted.
struct AccountGroup {
  double bits = 0;  // minus lg of the error without work
  unsigned work = 0;
  u64 rounds = 0;
  std::string name;
  bool holds = true;
  bool list = false;  // the error is that of a list of positions
  double all() const { return bits + work; }
};
// What account() returns. `formed`: some round has an error above 0; if none has, `worst`, `lists`
// and the sums are not set. `holds` is also false if the parameters were rejected or a round's
// `Priced` mark (plan.h: from what its error was computed) contradicts the round
// (PlanRound::word_fault()), so that it agrees with the plan's own verdict on both. `worst` is a
// round's full name, digits kept. `work`: Setting::work, bits of proof of work by ErrorKind and
// Scope.
struct Accounting {
  bool formed = false;
  bool holds = false;               // every round holds at the target, work counted (exact)
  bool needs_work = false;          // some round is above the target without its work (exact)
  std::string refused;              // why the parameters were rejected ("not admitted"), or empty
  std::string why_not;              // if !holds: the whole plan's first cause, as text (plan.h)
  unsigned target = 0;
  unsigned digest_bits = 256;
  u64 rounds = 0, queries = 0;      // all rounds; those whose kind is not EMPTY
  std::vector<AccountGroup> groups; // groups with an error above 0, the largest first, work counted
  std::string worst;                // name of the round with the largest error, work counted
  AccountGroup lists;               // the group with the largest error if work is not counted
  double sum = 0, sum_lists = 0;    // minus lg of all rounds' errors added, with work and without
  u64 planned_calls = 0;            // the verifier's hash calls as planned: chains and openings only
  bool counted = false;             // `calls` is a checking run's own count; `account` sets neither
  u64 calls = 0;
  unsigned work[6][3] = {};
  bool any_work() const {  // the setting has bits of work for one of the five kinds of error to which a file can give work
    for (int i = 0; i < 5; ++i)
      if (work[i][0] || work[i][1] || work[i][2]) return true;
    return false;
  }
  // A proximity term (ErrorKind::CLOSENESS; printed: closeness term) is the part of a round's error
  // for the event that a random combination of rows is close to the code although the rows are not
  // jointly close. `closeness` tells which bound the terms rest on; `circle` is Setting::circle;
  // only tests turn it on. `CountSource` is in setting.h.
  u64 closeness[2] = {};            // rounds with a proximity term, by CountSource
  bool circle = false;
  /* The round of the approximate range check (RNG; printed: `smallness test`) that has the largest
     error, work counted, under a union bound over its matrix's columns: columns times the round's
     error, one event per column (discussed in blocks/iop/rng.h). At an error of 2^-N that is
     N - log2(columns) bits. The figure decides nothing; the round's name keeps its digits.
     small_columns: that round's columns; 0 if the plan has no such round. */
  AccountGroup small_union;
  u64 small_columns = 0;
  /* near: rounds that hold with less than 0.01 bit to spare, work counted; near_small: those of the
     range check; near_closeness[i]: those with a proximity term on CountSource i. by_work: rounds
     whose error without their work is above the target; it is nonzero if and only if needs_work is
     set. Rounds with error 0 are in none of these counts. The counts are for printed lines and
     decide nothing here; `acctReadH` (verifier/src/vf_accounting_record.h) rejects a record, an
     `Accounting` passed as text between processes, in which by_work contradicts needs_work, or
     near is less than near_small and near_closeness together.
     on_a_word counts open items: at a round marked ON_THE_CALLERS_WORD the error is computed from
     a number that a caller supplied and that nothing checks at that round. */
  u64 near = 0, near_small = 0, near_closeness[2] = {0, 0}, by_work = 0;
  u64 on_a_word = 0;  // rounds marked Priced::ON_THE_CALLERS_WORD (plan.h), also at error 0
};

inline std::string digits_as_hash(const std::string &s) {
  std::string o;
  for (char c : s)
    if (c >= '0' && c <= '9') {
      if (o.empty() || o.back() != '#') o.push_back('#');
    } else
      o.push_back(c);
  return o;
}

// Forms the accounting of one part of the plan (an index into its `parts`: a file or part of a
// file), or of the whole plan if part < 0; `refused` and `why_not` are the whole plan's either way.
// A round's error is whole_error() (blocks/numbers/plan.h): its bound plus its draws' distance
// from uniform. Not checked: that all rounds on the same two codes assume one distance
// (Cause::TWO_DISTANCES). A plan from merged() (blocks/frame/files.h) spans several chains, where
// stages of different proximity tests may validly share both codes; configure()
// (blocks/frame/configure.h) and `Cursor` (blocks/numbers/plan.h) check each chain's own plan.
inline Accounting account(const Planned &p, int part = -1) {
  Accounting a;
  if (p->cause() == Cause::REFUSED) a.refused = p->why_not();
  a.target = p->set.target, a.digest_bits = 8 * p->set.digest_bytes;
  a.circle = p->set.circle;
  const PlanRound *worst = nullptr;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j) a.work[i][j] = p->set.work[i][j];
  Real sum, sum_lists;
  a.holds = p->cause() != Cause::REFUSED;  // Cause::TWO_DISTANCES is not checked: see above
  const unsigned block = p->set.hash_block_bytes;
  const Real unit = p->capped();
  for (auto &r : p->rounds_) {
    if (part >= 0 && (int)r.part != part) continue;
    if (r.word_fault()) a.holds = false;  // as in the plan's own verdict (plan.h), also at error 0
    if (r.priced == Priced::ON_THE_CALLERS_WORD) a.on_a_word += r.count;  // also at error 0
    a.rounds += r.count;
    if (r.kind != RoundKind::EMPTY) a.queries += r.count;
    /* The compression-function calls of this round's query, its input to the hash chain: one per
       hash_block_bytes, rounded up, of the prover's message plus chain_extra_bytes. Without proof
       of work these also give the first chain_free_words 64-bit words of the verifier's message;
       with it they give none, because the work's test block takes that place. Every eight further
       words cost one call, whatever the message's length. An element of the extension field E
       takes one word per coordinate, a position one word, a matrix entry (`draw_trits`) two bits.
       Words that a draw skips are not planned: a run whose draws skip words can make more calls in
       a round than are counted here. */
    const u64 words = r.draw_E * p->set.degree + r.draw_positions + (r.draw_trits + 31) / 32, free = r.work_bits ? 0 : p->set.chain_free_words;
    a.planned_calls += r.count * ((r.message_bytes + p->set.chain_extra_bytes + block - 1) / block + (words > free ? (words - free + 7) / 8 : 0));
    const Real error = p->whole_error(r, unit);
    if (error.zero()) continue;
    if (!p->holds(r, unit)) a.holds = false;
    if (!(error <= Real::two_to(-(long)p->set.target))) a.needs_work = true, a.by_work += r.count;
    if (p->holds(r, unit) && bits_of(error) + r.work_bits < p->set.target + 0.01) {
      a.near += r.count;
      if (r.why == ErrorKind::SMALL) a.near_small += r.count;
      if (r.why == ErrorKind::CLOSENESS) a.near_closeness[(int)r.rests] += r.count;
    }
    if (r.why == ErrorKind::CLOSENESS) a.closeness[(int)r.rests] += r.count;
    /* The range check's round. `plan_rng` (blocks/numbers/blocks.h) sets draw_trits = columns x N
       and combinations = N, so the division recovers `columns`. A run's matrix has the same shape:
       `require_combinations` (blocks/chain/rounds.h) ends the process if the run's N is not the
       plan's, and the channel (`Rounds`, same file), through which a protocol block draws, requires
       that exactly draw_trits entries are drawn. */
    if (r.draw_trits) {
      const u64 N = r.combinations;
      const u64 columns = N ? r.draw_trits / N : 0;
      const double bits = bits_of(Real::mul(error, Real(columns), Dir::UP));
      if (columns && (!a.small_columns || bits + r.work_bits < a.small_union.all())) {
        a.small_columns = columns;
        a.small_union.bits = bits, a.small_union.work = r.work_bits, a.small_union.rounds = r.count;
        a.small_union.name = r.where.empty() ? r.name : r.name + " (" + r.where + ")";
      }
    }
    if (!worst || Real::mul(p->whole_error(*worst, unit), Real::two_to(-(long)worst->work_bits), Dir::UP) < Real::mul(error, Real::two_to(-(long)r.work_bits), Dir::UP)) worst = &r;
    const Real all = Real::mul(error, Real(r.count), Dir::UP);
    sum_lists = Real::add(sum_lists, all, Dir::UP);
    sum = Real::add(sum, Real::mul(all, Real::two_to(-(long)r.work_bits), Dir::UP), Dir::UP);
    AccountGroup g;
    g.bits = bits_of(error), g.work = r.work_bits, g.rounds = r.count, g.holds = p->holds(r, unit), g.list = r.why == ErrorKind::QUERY;
    g.name = digits_as_hash(r.where.empty() ? r.name : r.name + " (" + r.where + ")");
    a.groups.push_back(g);
  }
  for (auto &m : p->messages_)
    if (part < 0 || (int)m.part == part) a.planned_calls += m.hash_calls * m.count;
  if (!a.holds) a.why_not = p->why_not();
  if (a.groups.empty()) return a;
  /* Rounds with the same name and work whose errors print alike are merged into one group, which
     keeps the largest of their errors. The groups are then sorted by their unrounded errors, so
     that the first group contains the round named in `worst` even if two groups print alike. */
  std::stable_sort(a.groups.begin(), a.groups.end(), [](const AccountGroup &x, const AccountGroup &y) {
    if (x.name != y.name) return x.name < y.name;
    if (x.work != y.work) return x.work < y.work;
    return x.bits < y.bits;
  });
  std::vector<AccountGroup> o;
  for (auto &x : a.groups)
    if (!o.empty() && o.back().name == x.name && o.back().work == x.work && cut2(o.back().bits) == cut2(x.bits))
      o.back().rounds += x.rounds, o.back().holds = o.back().holds && x.holds;
    else
      o.push_back(x);
  a.groups.swap(o);
  std::stable_sort(a.groups.begin(), a.groups.end(), [](const AccountGroup &x, const AccountGroup &y) {
    if (x.all() != y.all()) return x.all() < y.all();
    return x.name < y.name;
  });
  a.lists = a.groups[0];
  for (auto &x : a.groups)
    if (x.bits < a.lists.bits) a.lists = x;
  a.sum = bits_of(sum), a.sum_lists = bits_of(sum_lists);
  a.worst = worst->where.empty() ? worst->name : worst->name + " (" + worst->where + ")";
  a.formed = true;
  return a;
}

// c_h, the constant in the compiled bound's second term. It also covers the calls of the tree hash
// TH() (verifier/src/vf_common.h), which forms a request's digest. 17/2 holds under every parameter
// file that the programs accept: `lt`, log2 of the tokens in a request's window, is at most 16, and
// a chunk of TH() holds at most 16 digests. A planning call prints this accounting for every such
// file. A checking call can pass only with lt at most 10: then no chunk under a request's digest
// holds more than 9 digests. If only these chunks count, c_h = 5 is enough, and the printed level
// is 0.3 bit (at w = 128) below the one that the bound gives. TH() also hashes the configuration's
// files (policy.txt, KL.bin, layout.txt), in chunks of up to 16 digests at any lt.
constexpr double COLLISION_CONSTANT = 8.5;
// The compiled bound (top of this file) as one number: the largest L such that a prover with
// T = t + q_V + 1 <= 2^L hash calls, the verifier's included, succeeds with probability at most
// T / 2^L; that is, the largest L with 2^-L >= 2^-w + c_h 2^(L - D), to a double's precision.
// w: bits of the largest round error; D: digest_bits. At w = 128 and D = 256 it is 126.21, where
// the rule gives 128.
inline double compiled_level(double w, unsigned digest_bits = 256) {
  double lo = 0, hi = digest_bits - 1.0;
  for (int i = 0; i < 200; i++) {
    const double L = (lo + hi) / 2;
    (std::pow(2.0, -L) >= std::pow(2.0, -w) + COLLISION_CONSTANT * std::pow(2.0, L - (double)digest_bits) ? lo : hi) = L;
  }
  return lo;
}

inline void print_account_group(const char *head, const AccountGroup &g) {
  if (g.work)
    std::printf("%s  2^-%.2f = %.2f by the %s + %u by work, %llu round(s): %s%s\n", head, cut2(g.all()), cut2(g.bits), g.list ? "lists" : "bound", g.work,
                (unsigned long long)g.rounds, g.name.c_str(), g.holds ? "" : "  ABOVE THE TARGET");
  else
    std::printf("%s  2^-%.2f, %llu round(s): %s%s\n", head, cut2(g.all()), (unsigned long long)g.rounds, g.name.c_str(),
                g.holds ? "" : "  ABOVE THE TARGET");
}

// Prints every group of rounds, the largest error first.
inline void print_rounds(const Accounting &a) {
  for (auto &g : a.groups) print_account_group("round:", g);
}

/* Prints the accounting; the verifier program `cp_verify` prints its own through this function
   too. what: which files it is about. rounds_listed: `print_rounds` has printed every group above.
   top: at most how many groups are listed. The function compares no error with the target: a.holds
   and a.needs_work are the exact comparison's results. The stricter readings of the same rounds
   that it prints near the end decide nothing: the largest error without work, all errors added
   with the work and without, and the range check's union bound. If `any_work` returns false, the
   first is left out and the sum is printed once. The last line states the unit of all figures. */
inline void print_accounting(const Accounting &a, const char *what, bool rounds_listed = false, size_t top = 6) {
  if (!a.refused.empty()) {
    std::printf("accounting: THE PARAMETERS ARE NOT ADMITTED: %s\naccounting: THE ROUNDS: NO: the parameters are not admitted\n", a.refused.c_str());
    return;
  }
  if (!a.formed) {
    std::printf("accounting: none was formed: the plan has no round with an error\n");
    return;
  }
  static const char *const kind[6] = {"field", "sample", "closeness", "positions", "smallness", ""};
  const double w = a.groups[0].all(), D = a.digest_bits;
  const bool work = a.any_work();  // if false, no line below says that work is counted
  std::printf("accounting: %s: soundness round by round%s\n", what, rounds_listed ? " (every round's error is printed above, in the lines \"round:\")" : "");
  std::printf("accounting: target: %u bits in every round%s\n", a.target, work ? ", proof of work counted; every list of positions is derived from the target less the work before its draw"
                                                                              : "; no round has proof of work; every list of positions is derived from the target");
  std::printf("accounting: work, bits before a verifier's message, by the kind of its error (a request's own rounds / a batch's shared / the "
              "committer's):");
  for (int i = 0; i < 5; ++i)
    if (a.work[i][0] || a.work[i][1] || a.work[i][2]) std::printf(" %s %u / %u / %u;", kind[i], a.work[i][0], a.work[i][1], a.work[i][2]);
  std::printf(work ? "\n" : " none\n");
  std::printf("accounting: THE LARGEST ROUND ERROR: 2^-%.2f, in the round \"%s\". Each round's error is compared with the target by itself: the rounds' errors are not added to one another. There are %llu rounds, %llu "
              "of them with a verifier's message.\n", cut2(w), a.worst.c_str(), (unsigned long long)a.rounds, (unsigned long long)a.queries);
  // The round named above is seldom the only one at the target: sizes are derived to just meet it.
  std::printf("accounting: rounds that hold with less than 0.01 bit to spare%s: %llu: %llu of the smallness test, %llu with a closeness term (%llu of these on the theorem on "
              "codes, ePrint 2025/2055, Theorem 4.6, taken as given), %llu others\n", work ? ", work counted" : "", (unsigned long long)a.near, (unsigned long long)a.near_small, (unsigned long long)(a.near_closeness[0] + a.near_closeness[1]),
              (unsigned long long)a.near_closeness[1], (unsigned long long)(a.near - a.near_small - a.near_closeness[0] - a.near_closeness[1]));
  std::printf("accounting: the rounds with the largest errors (# stands for a number):\n");
  for (size_t i = 0; i < a.groups.size() && i < top; ++i) print_account_group("accounting: ", a.groups[i]);
  /* The result for the rounds of these parameters. It is not a verdict on a proof or a call: a call
     of `cp_verify` that prints `HOLD` or `HOLD WITH WORK` here can still end with `NO` or
     `NOT-A-VERIFICATION`. A target reached only with proof of work is reported separately, with
     both figures: how far the rounds go without the work is for a proof's reader to weigh, who did
     not choose the bits of proof of work. */
  if (!a.holds) std::printf("accounting: THE ROUNDS: NO: %s\n", a.why_not.c_str());
  else if (!a.needs_work) std::printf("accounting: THE ROUNDS: HOLD: every round holds at the target, and none needs proof of work for it\n");
  else
    std::printf("accounting: THE ROUNDS: HOLD WITH WORK: %.2f + %u: all rounds hold at the target, %llu of them only because their proof of work is counted; the pair is one round's: of all "
                "rounds the one with the largest error when work is left out, 2^-%.2f, and its bits of work\n"
                "work: %.2f + %u (%s: %s)\n", cut2(a.lists.bits), a.lists.work, (unsigned long long)a.by_work, cut2(a.lists.bits), cut2(a.lists.bits), a.lists.work, what, a.lists.name.c_str());
  if (a.holds) {  // after a NO these figures would mean nothing
    if (a.counted)
      std::printf("accounting: hash calls: t, the prover's, is a free variable; q_V, the verifier's own in this run: %llu\n",
                  (unsigned long long)a.calls);
    else
      std::printf("accounting: hash calls: t, the prover's, is a free variable; q_V, the verifier's own, is counted in a checking run (a planning call "
                  "reads no file)\naccounting:   planned for chains and openings: %llu (not planned: hashing of public files and tables; words that a draw skips); it stands for q_V in the lines below\n", (unsigned long long)a.planned_calls);
    const double qv = (double)(a.counted ? a.calls : a.planned_calls);  // a plan has no count
    // verifier/tests/call_accounting_test.py fails if a checking call's count is above the plan's.
    // Each of these lines, read alone, must show that proof of work is counted in it.
    char bare[96] = "";
    if (work) std::snprintf(bare, sizeof bare, "; in the first, proof of work is counted (without it: 2^-%.2f)", cut2(a.lists.bits));
    std::printf("accounting: compiled bound: (t + q_V + 1) 2^-%.2f + 17/2 (t + q_V)^2 / 2^%u; the second term is the digests' (%u bits)%s\n", cut2(w), a.digest_bits, a.digest_bits,
                bare);
    for (int lt : {64, 96, 120}) {
      const double q = std::ldexp(1.0, lt) + qv;
      std::printf("accounting:   at t = 2^%d: 2^-%.2f + 2^-%.2f%s\n", lt, cut2(w - std::log2(q + 1.0)),
                  cut2(D - std::log2(COLLISION_CONSTANT) - 2.0 * std::log2(q)), work ? " (proof of work counted)" : "");
    }
    // The rule's figure is not printed as a number of bits here: a reader would quote it alone,
    // without the stricter readings that follow.
    if (work) std::snprintf(bare, sizeof bare, ", proof of work counted (without it: %.2f bits)", cut2(compiled_level(a.lists.bits, a.digest_bits)));
    std::printf("accounting: as one number, by the compiled bound: %.2f bits%s: the largest L such that a prover with at most 2^L hash calls, the verifier's counted in, "
                "succeeds with probability at most (its calls) / 2^L\n", cut2(compiled_level(w, a.digest_bits)), bare);
  }
  // The aim is one fact per printed line, so that each is read.
  std::printf("accounting: what stands behind it:\naccounting:   of the rounds with a closeness term %llu rest on a lemma of the paper that this protocol builds on (see THIRD_PARTY.md), proved there (within a third of "
              "a code's distance; the same bound is also in other papers: blocks/numbers/codes.h names them)\n", (unsigned long long)a.closeness[0]);
  if (a.closeness[1])
    std::printf("accounting:   %llu of them rest on the theorem on codes, Theorem 4.6 of ePrint 2025/2055, which we take as given and of which we have verified no proof ourselves (blocks/numbers/codes.h says what the posting of November 6, 2025 prints behind it): TAKEN AS GIVEN\n",
                (unsigned long long)a.closeness[1]);
  else
    std::printf("accounting:   none of them rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6)\n");
  const bool beyond = a.closeness[1];  // some term beyond a third of the distance
  if (beyond)
    std::printf("accounting:   beyond a third a fold's round has no factor for the number of interleaved rows (a bound argued on paper for this protocol, for linear codes over E and two tables combined by "
                "one challenge; see docs/limits.md)\n");
  if (beyond || a.closeness[0]) std::printf("accounting:   a fold of 2^v rows at one point counts v closeness terms (a bound argued on paper for this protocol; see docs/limits.md)\n");
  if (beyond && a.circle) std::printf("accounting:   over 2^61 - 1 each term beyond a third has one list size more, since the encoding's image is a subcode (a bound argued on paper for this protocol; see docs/limits.md)\n");
  if (beyond || a.closeness[0]) std::printf("accounting:   the bounds for this protocol's own steps are argued on paper, "
                                            "in a technical report that is not public and not refereed: see docs/limits.md#what-the-theorem-assumes\n");
  if (a.small_columns)
    std::printf("accounting:   the smallness test's round counts one column: the first that no small integer column explains, which is fixed before the draw (argued on paper for this protocol; see docs/limits.md); a union "
                "bound over all columns stands among the stricter readings\n");
  std::printf("accounting:   the blocks' round errors and the compiled bound are cited from published work, some of it not peer-reviewed, or argued on paper in a technical report that is not public and not refereed; they are not proved for these programs (the Lean library proves its own verifier's bound, not this one)\n");
  if (a.on_a_word)
    std::printf("accounting:   %llu round(s) have an error that is computed from a number which the parameter file or this program's own code supplies and which no check ties to the run at that round "
                "(blocks/numbers/plan.h, ON_THE_CALLERS_WORD, says which numbers; some are held at a later place; see docs/limits.md#how-the-programs-count). Not counted: the rule and the integer m under which a group's "
                "distance stands: they are the analysis's choice and no number of a run; that all rounds of one chain on the same two codes were priced under one choice is not asked by this account: the committer, the prover and the verifier ask it of each chain when they read the public files\n", (unsigned long long)a.on_a_word);
  else
    std::printf("accounting:   that all rounds of one chain on the same two codes were priced under one choice of distance is not asked by this account: the committer, the prover and the verifier ask it of each chain when they read the public files\n");
  if (work)
    std::printf("accounting:   that g bits of work before a message multiply its round's error by 2^-g is assumed for this program's hash chains; it is proved "
                "for another setting in ePrint 2021/582, Theorem 6\n");
  std::printf("accounting:   that the programs compute the protocol is not proved; no figure here is a security level\n");
  std::printf("accounting: hiding is a separate bound, in the numbers of queries and of proofs on one commitment, and is no part of these "
              "figures\n");
  std::printf("accounting: FOR INFORMATION, stricter readings of the same rounds, which decide nothing. Each starts from the rounds' errors as they are charged above; in some rounds the charge is the largest of "
              "several events' chances and not their sum: so none is a union bound over all events\n");
  if (work) {
    std::printf("accounting:   the largest round error without the work: 2^-%.2f (%s)\n", cut2(a.lists.bits), a.lists.name.c_str());
    std::printf("accounting:   all rounds' errors added: 2^-%.2f with the work, 2^-%.2f without\n", cut2(a.sum), cut2(a.sum_lists));
  } else
    std::printf("accounting:   all rounds' errors added: 2^-%.2f\n", cut2(a.sum));
  if (a.small_columns)
    std::printf("accounting:   the smallness test's round by a union bound over the %llu columns of its matrix (their number times its error: one event a column: blocks/iop/rng.h): 2^-%.2f%s (%s)\n",
                (unsigned long long)a.small_columns, cut2(a.small_union.all()), a.small_union.work ? " with its work" : "", a.small_union.name.c_str());
  // The text is constant: nothing here computes the joint figure. The library can, through
  // Setting::instances (blocks/numbers/setting.h). No header in blocks/frame/ sets that field;
  // blocks/numbers/cp_plan.cpp does, from a line `instances`, and this text does not follow it.
  std::printf("accounting:   the unit of all figures above: one key's, one entry's and ONE request's proof; a round that the requests of a batch share is counted for all of them at once; counted jointly for a full "
              "batch of N requests, each round of a request's own chains reads log2(N) bits less (6 bits at 64)\n");
}

}  // namespace blk
