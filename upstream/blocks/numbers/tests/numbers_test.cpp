// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Unit tests of the headers in blocks/numbers/: the exact arithmetic (real.h) against identities;
   list sizes, counts and numbers of positions (codes.h) against exact integer comparisons and known
   values; the protocol blocks' rounds, scalars and round errors, and the stages of a proximity test
   (blocks.h), against closed formulas; `Plan` and `Cursor` (plan.h); `account` (accounting.h).
   blocks.h explains the blocks' capital names (SEND, BAT ...).

   Terms. A plan lists all rounds with a bound on each round's error. It holds only if every error
   times 2^-work (work: bits of proof of work) is at most 2^-target; a planner rejects it
   (`refused`) if sizes are outside a formula's range. A group is 2^k main rows and 2^h mask rows,
   each under one code, tested at one relative distance delta = 1 - agree. List size (`list`): a
   bound on how many codewords lie within delta of a word. count / |E|: the error of mutual
   correlated agreement, the proximity term (`CLOSENESS`, `closeness` in identifiers). "The lemma"
   (`CountSource::THIRD`): a proved bound on count below a third of the code's distance. "The
   theorem", or "the theorem on codes" (`CountSource::JOHNSON`; codes.h cites it): the assumed
   bound up to the Johnson radius; a larger integer m (>= 3) puts delta closer to that radius. */

/* Makes `Plan`'s private members public and lets a `Plan` and a `Planned` be made from each other,
   so that a test can write or alter rounds by hand. A build with SERVING_BUILD defined rejects it
   (blocks/fence.h). */
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // read by blocks/numbers/plan.h
#include <cstdio>
#include <functional>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "chain/tests/death.h"
#include "numbers/accounting.h"
#include "numbers/blocks.h"

using namespace blk;

static int failures = 0, checks = 0;
#define CHECK(c)                                                   \
  do {                                                             \
    ++checks;                                                      \
    if (!(c)) ++failures, std::printf("%s:%d: %s\n", __FILE__, __LINE__, #c); \
  } while (0)

static u64 state = 88172645463325252ull;
static u64 rnd() {  // xorshift64 (Marsaglia, "Xorshift RNGs", 2003): for test data only
  state ^= state << 13, state ^= state >> 7, state ^= state << 17;
  return state;
}
static Big random_big(unsigned limbs) {
  Big x;
  for (unsigned i = 0; i < limbs; ++i) x = x.shl(32) + Big(rnd() & 0xffffffffu);
  return x;
}

static void test_big() {
  CHECK(Big::pow(Big(10), 30).dec() == "1000000000000000000000000000000");
  CHECK((Big::two_to(61) - Big(1)).dec() == "2305843009213693951");
  CHECK(Big::pow(Big::two_to(61) - Big(1), 3).bits() == 183);
  for (int i = 0; i < 2000; ++i) {
    const Big a = random_big(1 + rnd() % 24), b = random_big(1 + rnd() % 12) + Big(1);
    Big q, r;
    Big::divmod(a, b, q, r);
    CHECK(q * b + r == a && r < b);
    const Big s = Big::isqrt(a);
    CHECK(s * s <= a && a < (s + Big(1)) * (s + Big(1)));
    const size_t n = rnd() % 100;
    CHECK(a.shl(n).shr(n) == a);
    CHECK((a + b) - b == a);
  }
  // Divisors just below a power of 2^32, meant to reach the correction steps of Big::divmod().
  for (int i = 0; i < 2000; ++i) {
    Big b = Big::two_to(32 * (2 + rnd() % 4)) - Big(1 + rnd() % 3), q0 = random_big(1 + rnd() % 4), r0 = Big(rnd() % 5), q, r;
    Big::divmod(q0 * b + r0, b, q, r);
    CHECK(q == q0 && r == r0);
  }
}

static void test_real() {
  const Bound two(2), root = Bound::sqrt(two);
  CHECK(Real::mul(root.lo, root.lo, Dir::UP) <= Real(2) && Real(2) <= Real::mul(root.hi, root.hi, Dir::DOWN));
  CHECK(Real::sub(root.hi, root.lo, Dir::UP) <= Real::two_to(-180));
  const Bound third = Bound::ratio(Big(1), Big(3));
  CHECK(third.lo < third.hi && Real::mul(third.lo, Real(3), Dir::UP) <= Real(1) && Real(1) <= Real::mul(third.hi, Real(3), Dir::DOWN));
  CHECK(Real::cmp(Real::pow(Real::two_to(-1), 128, Dir::UP), Real::two_to(-128)) == 0);
  CHECK((Bound(1) - Bound(2)).bad && (Bound(1) / Bound((u64)0)).bad && (Bound(3) + (Bound(1) - Bound(2))).bad);
  CHECK(std::fabs(Real(Big::pow(Big(3), 100)).log2() - 100 * std::log2(3.0)) < 1e-9);
  CHECK(Bound::ratio(Big(7), Big(2)).hi.ceil() == Big(4) && Bound::ratio(Big(7), Big(2)).lo.floor() == Big(3));
  for (int i = 0; i < 300; ++i) {
    const Bound a = Bound::ratio(Big(1 + rnd() % 1000), Big(1 + rnd() % 1000)), b = Bound::ratio(Big(1 + rnd() % 1000), Big(1 + rnd() % 1000));
    for (const Bound &x : {a + b, a * b, a / b, Bound::sqrt(a), Bound::pow(a, 1 + rnd() % 50)}) CHECK(!x.bad && x.lo <= x.hi);
  }
  /* `Real::div` and `Real::sqrt` round to the requested side (checked in exact integers) and, in
     the first loop, to within one unit in the last of REAL_BITS bits. In `Real::div` the step that
     adds 1 where a remainder is left changes the result only if the bits that `cut` then drops are
     all 0, and `cut` drops two or three bits more than the dividend has. So the second loop takes
     small dividends, where that is one case in 8 to one in about 50. `Real::sqrt` does not call
     `cut`: there the step changes the result wherever the root is not exact in REAL_BITS bits. */
  unsigned uneven = 0;
  for (int i = 0; i < 4000; ++i) {
    const Big a = random_big(1 + rnd() % 8) + Big(1), b = random_big(1 + rnd() % 8) + Big(1);
    const Real up = Real::div(Real(a), Real(b), Dir::UP), down = Real::div(Real(a), Real(b), Dir::DOWN);
    CHECK(Real::cmp(Real(up.m * b, up.e), Real(a)) >= 0 && Real::cmp(Real(down.m * b, down.e), Real(a)) <= 0);
    CHECK(Real::cmp(Real::sub(up, down, Dir::UP), Real::two_to(down.top() - (long)REAL_BITS)) <= 0);
    const Real high = Real::sqrt(Real(a), Dir::UP), low = Real::sqrt(Real(a), Dir::DOWN);
    CHECK(Real::cmp(Real(high.m * high.m, 2 * high.e), Real(a)) >= 0 && Real::cmp(Real(low.m * low.m, 2 * low.e), Real(a)) <= 0);
    CHECK(Real::cmp(Real::sub(high, low, Dir::UP), Real::two_to(low.top() - (long)REAL_BITS)) <= 0);
    uneven += Real::cmp(up, down) != 0;
  }
  CHECK(uneven > 3000);
  for (u64 small = 1; small < 4096; ++small) {
    const Big a(1 + small % 7), b = random_big(1 + rnd() % 4) + Big(9);
    const Real up = Real::div(Real(a), Real(b), Dir::UP), down = Real::div(Real(a), Real(b), Dir::DOWN);
    CHECK(Real::cmp(Real(up.m * b, up.e), Real(a)) >= 0 && Real::cmp(Real(down.m * b, down.e), Real(a)) <= 0);
    const Real high = Real::sqrt(Real(small), Dir::UP), low = Real::sqrt(Real(small), Dir::DOWN);
    CHECK(Real::cmp(Real(high.m * high.m, 2 * high.e), Real(small)) >= 0 && Real::cmp(Real(low.m * low.m, 2 * low.e), Real(small)) <= 0);
  }
  /* Does an operator of `Bound` ever return lo > hi? Not in the cases tried here (a sample, not a
     proof). The constructor marks such a pair bad, so a valid computation must never reach one
     through rounding. Chains of operations on numbers of many sizes: after each step lo <= hi, and
     the result is bad exactly where the operator's rule says so (`may_refuse`). */
  auto drawn = [&] {
    switch (rnd() % 5) {
      case 0: return Bound((u64)(rnd() % 3));
      case 1: return Bound(Real::two_to((long)(rnd() % 800) - 400));
      case 2: return Bound::ratio(Big(1 + rnd() % 1000), Big(1 + rnd() % 1000));
      case 3: return Bound::ratio(Big::two_to(rnd() % 200) + Big(rnd() % 7), Big::two_to(rnd() % 200) + Big(1));
      default: return Bound::sqrt(Bound::ratio(Big(1 + rnd() % 1000), Big(1 + rnd() % 1000)));
    }
  };
  unsigned steps = 0, refusals = 0;
  for (int i = 0; i < 3000; ++i) {
    Bound x = drawn();
    CHECK(!x.bad && x.lo <= x.hi);
    for (int step = 0; step < 8 && !x.bad; ++step, ++steps) {
      const Bound y = drawn();
      bool may_refuse = false;
      switch (rnd() % 6) {
        case 0: x = x + y; break;
        case 1: x = x * y; break;
        case 2: may_refuse = y.lo.zero(), x = x / y; break;
        case 3: may_refuse = !(y.hi < x.lo), x = x - y; break;
        case 4: x = Bound::sqrt(x); break;
        default: x = Bound::pow(x, 1 + rnd() % 9); break;
      }
      CHECK(x.bad == may_refuse && (x.bad || x.lo <= x.hi));
      refusals += x.bad;
    }
  }
  CHECK(steps > 10000 && refusals > 500 && refusals < 3000);  // both kinds of steps were met
}

// (A / B)^t <= 2^-bits in integers.
static bool power_below(const Big &A, const Big &B, u64 t, u64 bits) { return Big::pow(A, t).shl(bits) <= Big::pow(B, t); }

// setting_of_this_build() is setting_goldilocks(), the programs' only field. Here setting_m61()
// (`Setting::p` = 2^61 - 1, circle codes) is only another set of parameters for the formulas.
static void test_positions() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    for (unsigned log_n : {11u, 16u, 19u, 23u})
      for (unsigned rate : {1u, 2u, 3u, 4u})
        for (unsigned bits : {96u, 100u, 108u, 128u, 139u}) {
          CodeSize c;
          c.log_n = log_n, c.ell = c.n() >> rate, c.r = 0;
          const Big n(c.n()), k(c.k(s));
          // 1 - delta = (n + k + 1) / 2n, a rational
          const u64 t = positions(Bound::ratio(n + k + Big(1), Big(2) * n), bits);
          CHECK(power_below(n + k + Big(1), Big(2) * n, t, bits) && !power_below(n + k + Big(1), Big(2) * n, t - 1, bits));
          // Johnson: (1 - delta)^2 = (k / n) ((m + 1) / m)^2, a rational
          for (u64 m : {3ull, 10ull, 73ull, 1189ull}) {
            const Proximity x = johnson(s, c, m);
            if (x.bad()) continue;
            const u64 u = positions(x.agree, bits);
            const Big A = k * Big(m + 1) * Big(m + 1), B = n * Big(m) * Big(m);
            CHECK(power_below(A, B, u, 2 * bits) && !power_below(A, B, u - 1, 2 * bits));
          }
        }
  }
}

// A group's sizes with its analysis under a setting; no struct of the library stores both.
struct Both : GroupSize, Analysis {
  Both(const Setting &s, const GroupSize &g) : GroupSize(g), Analysis(analysis(s, g)) {}
};
static Both lone(const Setting &s, unsigned log_n, u64 ell, u64 r, Scope scope) {
  GroupSize g;
  g.main.log_n = log_n, g.main.ell = ell, g.main.r = r;
  return Both(s, size_group(s, scope, g, GroupLoad()));
}

// Known values, all under setting_m61(): none here is for the programs' field.
static void test_known() {
  Setting s = setting_m61();
  const Both l = lone(s, 23, 1 << 22, 0, Scope::REQUEST), h = lone(s, 19, 1 << 16, 0, Scope::REQUEST);
  CHECK(l.at.m == 73 && h.at.m == 84);
  CHECK(lone(s, 20, 1 << 17, 0, Scope::REQUEST).at.m == 73 && lone(s, 16, 1 << 12, 0, Scope::REQUEST).at.m == 104 &&
        lone(s, 11, 1 << 7, 0, Scope::REQUEST).at.m == 209);
  CHECK(positions(l.at.agree, 128) == 267 && positions(h.at.agree, 128) == 87 && positions(l.at.agree, 108) == 225 && positions(l.at.agree, 139) == 290);
  Setting w = s;
  w.set_work(ErrorKind::CLOSENESS, Scope::REQUEST, 20);
  const Both lw = lone(w, 23, 1 << 22, 0, Scope::REQUEST), hw = lone(w, 19, 1 << 16, 0, Scope::REQUEST);
  CHECK(lw.at.m == 1189 && hw.at.m == 1366);
  CHECK(positions(lw.at.agree, 108) == 217 && positions(hw.at.agree, 108) == 73);
  // The table (see test_table()): rate 0.052, one-time distance 1/8.
  CodeSize T;
  T.log_n = 33, T.ell = 402653184, T.r = 44040191;
  CHECK(T.dim(s) == 446693375);
  const Bound eighth = Bound::ratio(Big(1), Big(8)), once = Bound(1) - eighth;
  CHECK(positions(once, 132) == 686 && positions(once, 128) == 665 && positions(once, 96) == 499 && positions(once, 100) == 520);
  CHECK(at_distance(s, T, once).source == CountSource::THIRD && at_distance(s, T, once).count.hi.floor() == Big::two_to(30) + Big(1));
  // The edge of the slack: m holds, m + 1 does not.
  for (const Both &g : {l, h, lw}) {
    const Setting &x = g.at.m > 1000 ? w : s;
    GroupSize shape;
    shape.main = g.main;
    CHECK(group_holds(x, Scope::REQUEST, group_of_m(shape, g.at.m), GroupLoad()));
    CHECK(!group_holds(x, Scope::REQUEST, group_of_m(shape, g.at.m + 1), GroupLoad()));
  }
  // A distance made from m gives back m or m + 1: at_distance() rounds sqrt(rho) / eta = m upwards.
  const Proximity back = at_distance(s, l.main, l.at.agree);
  CHECK(back.source == CountSource::JOHNSON && (back.m == 73 || back.m == 74));
}

static void test_circle_dimension() {
  const Setting c = setting_m61(), p = setting_goldilocks();
  CodeSize x;
  x.log_n = 10, x.ell = 16, x.r = 100;
  CHECK(x.dim(c) == 228 && x.dim(p) == 116 && x.coordinates() == 116);
  x.ell = 4, x.r = 37;
  CHECK(x.dim(c) == 101 && x.k(c) == 101 && x.k(p) == 40);
  CHECK(hidden_positions(c, x) == 37 && tail_for(c, 36) == 37 && tail_for(c, 37) == 37 && tail_for(p, 36) == 36 && tail_for(c, 0) == 0);
  x.r = 38;
  CHECK(hidden_positions(c, x) == 37 && hidden_positions(p, x) == 38);
}

// The most sibling digests for t distinct leaves in a tree of depth d, by a search over all ways
// to split the leaves.
static u64 most_by_search(unsigned d, u64 t, std::map<std::pair<unsigned, u64>, u64> &memo) {
  if (!t || !d) return 0;
  auto key = std::make_pair(d, t);
  auto it = memo.find(key);
  if (it != memo.end()) return it->second;
  const u64 half = (u64)1 << (d - 1);
  u64 best = 0;
  for (u64 a = t > half ? t - half : 0; a <= t && a <= half; ++a) {
    const u64 v = (a == 0 || a == t ? 1 : 0) + most_by_search(d - 1, a, memo) + most_by_search(d - 1, t - a, memo);
    best = std::max(best, v);
  }
  return memo[key] = best;
}
static void test_paths() {
  std::map<std::pair<unsigned, u64>, u64> memo;
  for (unsigned d = 1; d <= 7; ++d) {
    u64 so_far = 0;
    for (u64 t = 1; t <= ((u64)1 << d); ++t) {
      so_far = std::max(so_far, most_by_search(d, t, memo));  // t draws hit at most t distinct leaves
      CHECK(path_digests_most(d, t) == so_far);
    }
  }
  CHECK(path_digests_most(23, 335) == 4867);
  // The average against an experiment.
  const unsigned d = 10;
  const u64 t = 60;
  double sum = 0;
  const int runs = 4000;
  for (int i = 0; i < runs; ++i) {
    std::vector<char> on((size_t)2 << d, 0);
    for (u64 j = 0; j < t; ++j)
      for (size_t v = ((size_t)1 << d) + rnd() % ((u64)1 << d); v; v >>= 1) on[v] = 1;
    for (size_t v = 2; v < on.size(); v += 2) sum += on[v] != on[v + 1];
  }
  CHECK(std::fabs(sum / runs - path_digests_expected(d, t)) < 1.0);
  CHECK(path_digests_expected(23, 335) <= (double)path_digests_most(23, 335));
  CHECK(std::fabs(expected_distinct(10, 60) - 1024 * (1 - std::pow(1023.0 / 1024, 60))) < 1e-6);
}

static u64 scalars_of(const Plan &p) {
  u64 b = 0;
  for (auto &m : p.messages_)
    if (m.kind == MessageKind::SCALARS) b += m.bytes * m.count;
  return b / p.set.e_bytes;
}

// The rounds, repeats counted, whose verifier's message is not empty.
static u64 spoken(const Plan &p) {
  u64 k = 0;
  for (auto &r : p.rounds_)
    if (r.kind != RoundKind::EMPTY) k += r.count;
  return k;
}

// Counts by block: rounds, rounds with a verifier's message, scalars; also under `compact` (equal
// rounds entered once, with a repeat count).
static void test_counts() {
  const Setting s = setting_goldilocks();
  auto counts = [&](std::function<void(Plan &)> f, u64 all, u64 with_message, u64 scalars) {
    for (bool compact : {false, true}) {
      Plan p(s);
      p.compact = compact;
      f(p);
      CHECK(p.finish()->rounds() == all && spoken(p) == with_message && scalars_of(p) == scalars);
    }
  };
  GroupSize g;
  g.main.log_n = 16, g.main.ell = 1 << 12, g.main.r = 101, g.mask.log_n = 12, g.mask.ell = 64, g.mask.r = 101, g.mask.over_E = true, g.k = 6, g.h = 2;
  g = size_group(s, Scope::REQUEST, g, GroupLoad());
  CHECK(!analysis(s, g).bad());
  counts([&](Plan &p) { plan_send(p, "", {SampleSize{20, {g}}}, 1); }, 2, 1, 0);
  counts([&](Plan &p) { plan_bat(p, "", {g}, 100); }, 1, 1, 0);
  counts([&](Plan &p) { plan_bat_sigma(p, "", 100); }, 1, 1, 0);
  counts([&](Plan &p) { plan_split(p, ""); }, 1, 0, 1);
  counts([&](Plan &p) { plan_zero(p, "", 20); }, 1, 1, 0);
  for (unsigned k : {0u, 1u, 7u}) {
    PscSize x;
    x.degree.assign(k, 3);
    counts([&](Plan &p) { plan_psc(p, "", x); }, k + 3, k + 2, 2 * k + 5);  // 1 + 2 k + 2 + 2
  }
  for (unsigned v : {1u, 5u, 28u})
    counts([&](Plan &p) { plan_product_tree(p, "", v, 0); }, (u64)v * (v - 1) / 2 + 4 * v, (u64)v * (v - 1) / 2 + 3 * v, (u64)v * v + 4 * v);
  counts([&](Plan &p) { plan_fold(p, "", g, 1); }, g.k + 1, g.k + 1, 2 * g.k + 1);
  counts([&](Plan &p) { plan_loc(p, "", g, {}); }, 1, 1, 0);
  counts([&](Plan &p) { plan_cs(p, "", g, g, 20, 1, 3, {}, 1); }, 2, 2, 0);
  counts([&](Plan &p) { plan_shift(p, "", 1 << 20); }, 1, 1, 0);
  counts([&](Plan &p) { plan_join(p, "", 5); }, 1, 1, 0);
  {  // One lookup, with multiplicities of w = 2^mu bits. 4 challenges: shift, alpha, the point that
     // joins the claims on the w bit positions (plan_join()), the zero-check's point. w - 1
     // sumchecks (PSC, plan_psc()) tie the keys' root to the table side's w roots, one for each bit
     // position (7 scalars, then 6 each). Then two product trees, the leaves, the bits' check.
    const u64 v = 20, d = 12, mu = 5, w = 32;
    const u64 tree_q = v * (v - 1) / 2 + 3 * v + d * (d - 1) / 2 + (3 + mu) * d, tree_s = v * v + 4 * v + d * d + 4 * d + 2 * mu * d;
    const u64 queries = 4 + 2 * (w - 1) + tree_q + (1 + mu + d) + (2 + mu + d), empty = (w - 1) + v + d + 1;
    counts([&](Plan &p) { plan_lookup(p, "", LookupSize{(unsigned)v, (unsigned)d, (unsigned)mu, true}); }, queries + empty, queries,
           7 + 6 * (w - 2) + tree_s + (1 + mu + 2 * d) + (1 + 2 * (mu + d) + 2 + 1));
    for (const LookupSize &bad : {LookupSize{32, 12, 5, false}, LookupSize{20, 33, 5, false}, LookupSize{1, 1, 0, false}}) {
      Plan p(s);
      plan_lookup(p, "", bad);
      CHECK(p.refused());
    }
    {  // 40 of 64 bit positions in use (`live`), keys in two parts: 39 sumchecks tie the roots,
       // and one more, without variables, splits the product of the parts' roots.
      const u64 L = 40, m6 = 6, v0 = 16, v1 = 18, k = 2;
      const u64 keys_q = v0 * (v0 - 1) / 2 + 3 * v0 + v1 * (v1 - 1) / 2 + 3 * v1, table_q = d * (d - 1) / 2 + (3 + m6) * d;
      const u64 q = 3 + 2 * (L - 1) + 2 * (k - 1) + keys_q + table_q + (1 + m6 + d) + (2 + m6 + d), none = (L - 1) + (k - 1) + v0 + v1 + d + 1;
      counts([&](Plan &p) { plan_lookup(p, "", LookupSize{0, (unsigned)d, (unsigned)m6, false, (unsigned)L, {(unsigned)v0, (unsigned)v1}}); }, q + none, q,
             7 + 6 * (L - 2) + 5 * (k - 1) + v0 * v0 + 4 * v0 + v1 * v1 + 4 * v1 + d * d + 4 * d + 2 * m6 * d + (1 + m6 + 2 * d) + (1 + 2 * (m6 + d) + 2 + 1));
      for (const LookupSize &bad : {LookupSize{20, 12, 5, false, 20, {}}, LookupSize{20, 12, 5, false, 33, {}}, LookupSize{0, 12, 5, false, 21, {20, 20}}, LookupSize{20, 12, 6, false, 53, {}}}) {
        Plan p(s);
        plan_lookup(p, "", bad);
        CHECK(p.refused());
      }
    }
    Plan edge(s);  // 2^32 (2^32 - 1) = p - 1
    plan_lookup(edge, "", LookupSize{20, 32, 5, false});
    CHECK(!edge.refused());
  }
  GroupSize one = g;
  one.k = one.h = 0;
  counts([&](Plan &p) { plan_base(p, "", one, {}, 1); }, 2, 2, 1 + one.main.coordinates() + one.mask.coordinates());
}

static void test_stages() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()})
    {
      const Setting &x = s;
      Plan p(x);
      GroupSize g;
      g.name = "g", g.main.log_n = 23, g.main.ell = (u64)1 << 22, g.main.r = 341, g.mask.log_n = 17, g.mask.ell = 1 << 13, g.mask.r = 341, g.mask.over_E = true, g.k = 8;
      g = size_group(x, Scope::REQUEST, g, GroupLoad());
      const std::vector<StageChoice> choice = {{5, 3, 20, 16}, {5, 3, 16, 16}, {5, 3, 13, 13}, {0, 0, 15, 15}};
      const u64 t1 = plan_positions(p, analysis(x, g).at.agree);
      const auto st = solve_stages(p, g, t1, choice);
      CHECK(!p.refused() && st.size() == 5);
      u64 folds = 0;
      for (size_t i = 0; i + 1 < st.size(); ++i) {
        const GroupSize &a = st[i].g, &b = st[i + 1].g;
        folds += a.k;
        CHECK(a.main.ell == b.main.ell << b.k);
        CHECK(b.mask.ell << b.h == 2 * a.mask.ell + a.main.r + 2 * a.mask.r + st[i + 1].r_ood);
        CHECK(st[i + 1].r_ood >= x.samples);
        CHECK((b.mask.ell & (b.mask.ell - 1)) == 0 && (!b.k || b.mask.ell >= 2 * b.k + 1 - b.h));
        CHECK(hidden_positions(x, b.main) >= st[i + 1].t && hidden_positions(x, b.mask) >= st[i + 1].t);
        CHECK(b.main.dim(x) <= b.main.n() && b.mask.dim(x) <= b.mask.n() && b.mask.log_n <= b.main.log_n);
      }
      plan_iopp(p, "test", st, {}, 100, 1);
      CHECK(p.finish()->holds());
      // Hiding: plan_iopp() rejects a stage whose random coordinates were sized (`Stage::t`) for
      // one position fewer than are drawn on it.
      for (size_t i = 0; i < st.size(); ++i) {
        std::vector<Stage> less = st;
        less[i].t -= 1;
        Plan q(x);
        plan_iopp(q, "test", less, {}, 100, 1);
        CHECK(q.refused() && q.why().find("a list is longer than the stage's random coordinates were sized for") != std::string::npos);
      }
      // BAT (batches the claims) 1; each stage but the last: FOLD (sumcheck over its rows) k + 1,
      // the code switch (CS, plan_cs()) 2; BASE (the last stage) 2. No round is empty.
      CHECK(spoken(p) == 1 + folds + (st.size() - 1) * 3 + 2 && p.finish()->rounds() == spoken(p));
      for (auto &r : p.rounds_)  // no draw is longer than a stage of its code length was sized for
        if (r.kind == RoundKind::POSITIONS) {
          bool found = false;
          for (auto &q : st) found = found || (q.t >= r.draw_positions && q.g.main.log_n == r.log_n);
          CHECK(found);
        }
    }
}

static void test_refusals() {
  const Setting s = setting_goldilocks();
  auto shape = [](unsigned log_n, u64 ell, u64 r, unsigned mask_log_n, u64 mask_ell, u64 mask_r, unsigned k, unsigned h) {
    GroupSize g;
    g.main.log_n = log_n, g.main.ell = ell, g.main.r = r, g.mask.log_n = mask_log_n, g.mask.ell = mask_ell, g.mask.r = mask_r, g.mask.over_E = true, g.k = k, g.h = h;
    return g;
  };
  {  // a default-constructed group; a group for which `size_group` finds no m
    Plan p(s), q(s);
    plan_loc(p, "", GroupSize(), {});
    CHECK(p.refused() && !p.finish()->holds());
    GroupSize huge;
    huge.main.log_n = 47, huge.main.ell = (u64)1 << 46;
    Setting tight = s;
    tight.target = 190;
    const GroupSize g = size_group(tight, Scope::REQUEST, huge, GroupLoad());
    CHECK(!g.m && analysis(tight, g).bad());
    plan_loc(q, "", g, {});
    CHECK(q.refused());
  }
  // group_holds() is false for: a mask code longer than the main code; h > k; a single mask row
  // (h = 0 < k) whose code has the larger rate.
  CHECK(!group_holds(s, Scope::REQUEST, size_group(s, Scope::REQUEST, shape(12, 256, 50, 14, 8192, 50, 2, 1), GroupLoad()), GroupLoad()));
  CHECK(!group_holds(s, Scope::REQUEST, size_group(s, Scope::REQUEST, shape(16, 4096, 50, 12, 64, 50, 1, 2), GroupLoad()), GroupLoad()));
  CHECK(!group_holds(s, Scope::REQUEST, size_group(s, Scope::REQUEST, shape(16, 4096, 50, 12, 1024, 50, 4, 0), GroupLoad()), GroupLoad()));
  CHECK(group_holds(s, Scope::REQUEST, size_group(s, Scope::REQUEST, shape(16, 4096, 50, 12, 64, 50, 4, 0), GroupLoad()), GroupLoad()));
  CHECK(group_holds(s, Scope::REQUEST, size_group(s, Scope::REQUEST, shape(16, 4096, 50, 12, 1024, 50, 4, 2), GroupLoad()), GroupLoad()));  // the mask code has the larger rate; its term is counted
  {  // a last stage whose mask code has the larger rate: 0.0326 against the main code's 0.0091
    const GroupSize last = shape(15, 256, 44, 15, 1024, 44, 0, 0);
    CHECK(mask_rules(s, last));
    const Both g(s, group_of_m(last, 100));
    bool bad = false;
    const Big at_mask = johnson_list(g.at.agree, rho_of(s, last.mask), bad), at_main = johnson_list(g.at.agree, rho_of(s, last.main), bad);
    CHECK(!g.bad() && !bad && g.list == at_mask && at_main < at_mask);  // the list is that of the larger rate
    // The distance that the main code alone would give is beyond the mask code's own Johnson
    // radius: no lemma bounds the mask rows there.
    const Analysis beyond = analysis_at(s, last, johnson(s, last.main, 100));
    CHECK(surely_lt(Bound(1) - Bound::sqrt(rho_of(s, last.mask)), beyond.at.delta) && beyond.bad() && !beyond.at.bad() && beyond.mask_at.bad());
  }
  {  // From a third of a code's minimum distance up, the count is the theorem on codes'.
    const GroupSize plain = shape(16, 4096, 50, 12, 64, 50, 4, 2);
    const u64 n = plain.main.n(), half = (n - plain.main.k(s) - 1) / 2;  // (d - 1) div 2 positions, d the minimum distance
    // At that distance at_distance() and a group made by group_at() use the theorem on codes.
    CHECK(at_distance(s, plain.main, Bound::ratio(Big(n - half), Big(n))).source == CountSource::JOHNSON);
    CHECK(Both(s, group_at(plain, half)).at.source == CountSource::JOHNSON);
    // A group under Radius::JOHNSON with a main rate of about 3/8: its distance is between a third
    // and a half of the mask code's minimum distance, and the mask code uses the theorem on codes.
    const Both j(s, size_group(s, Scope::REQUEST, shape(16, 24576, 50, 12, 64, 50, 4, 2), GroupLoad()));
    CHECK(!j.bad() && j.at.source == CountSource::JOHNSON && j.mask_at.source == CountSource::JOHNSON);
  }
  {  /* The mask code's term in `fold_round_error` and `base_error`, written a second time, for
        a middle stage and a last stage whose mask code has the larger rate. The term must matter:
        without it the round's error is lower by more than a bit. */
    const Both stage(s, group_of_m(shape(21, 131072, 54, 12, 256, 54, 5, 2), 200)), last(s, group_of_m(shape(14, 512, 51, 14, 1024, 51, 0, 0), 200));
    CHECK(!stage.bad() && !last.bad() && mask_rules(s, stage) && mask_rules(s, last));
    auto near = [](const Bound &a, const Bound &b) { return std::fabs(a.hi.log2() - b.hi.log2()) < 1e-9; };
    for (unsigned j = 1; j <= stage.k; ++j) {
      const Bound main = mca_error(s, stage.at) + Bound(stage.list) * field_error(s, 2);
      const Bound whole = j <= stage.h ? main + Bound(2) * mca_error(s, stage.mask_at) : main;
      CHECK(near(fold_round_error(s, stage, stage, j), whole));
      if (j <= stage.h) CHECK(main.hi.log2() + 1 < whole.hi.log2());
    }
    const Bound main = mca_error(s, last.at) + Bound(last.list) * field_error(s, 1);
    CHECK(near(base_error(s, last, last), main + mca_error(s, last.mask_at)) && main.hi.log2() + 1 < base_error(s, last, last).hi.log2());
  }
  {  // a mask code 16 times shorter than the main one: its count is taken at the main code's length
    const GroupSize sixteen = shape(16, 4096, 50, 12, 64, 50, 4, 2);
    const Both t(s, group_at(sixteen, (sixteen.main.n() - sixteen.main.k(s) - 1) / 3));  // (d - 1) div 3 positions
    const u64 errors = (t.at.delta * Bound(Big(t.main.n()))).hi.floor().to_u64();
    CHECK(!t.bad() && t.mask_at.source == CountSource::THIRD && !(t.mask_at.count.hi < Real(errors + 1)) && !(Real(errors + 1) < t.mask_at.count.hi) && errors > t.mask.n());
    const Both j(s, group_of_m(sixteen, 50));
    const Proximity native = at_distance(s, sixteen.mask, j.at.agree);
    CHECK(!j.bad() && j.mask_at.source == CountSource::JOHNSON && native.source == CountSource::JOHNSON);
    CHECK(Real::mul(native.count.hi, Real(15), Dir::UP) < j.mask_at.count.hi && j.mask_at.count.hi < Real::mul(native.count.hi, Real(17), Dir::UP));
    // By hand: the theorem's count at length n and the mask's rate, and one list size.
    const Word rep = repeated(s, sixteen.mask, 16);
    CHECK(rep.n() == 65536 && rep.k() == 16 * sixteen.mask.k(s));
    const Bound by_hand = johnson_count(rep, rho_of(s, sixteen.mask), j.at.delta, j.mask_at.m) + Bound(j.mask_at.list);
    CHECK(std::fabs(by_hand.hi.log2() - j.mask_at.count.hi.log2()) < 1e-9);
  }
  {  // below a third of the distance both codes' counts are those of a lemma of the paper that this
     // protocol builds on (codes.h: THIRD), at the very edge
    const GroupSize edge = shape(16, 4096, 50, 12, 64, 50, 4, 0);
    const u64 n = edge.main.n(), d = n - edge.main.k(s), e = (d - 1) / 3;
    const Both g(s, group_at(edge, e));
    CHECK(weakest_source(g) == CountSource::THIRD && g.list == Big(1) && 3 * e < d && 3 * (e + 1) >= d);
    CHECK(at_distance(s, g.main, Bound::ratio(Big(n - e), Big(n))).source == CountSource::THIRD);
    CHECK(at_distance(s, g.main, Bound::ratio(Big(n - e - 1), Big(n))).source != CountSource::THIRD);
  }
  {  // `solve_stages` rejects: h > k; a middle stage with k = 0; a last stage with k > 0; no stage
    const GroupSize g = size_group(s, Scope::REQUEST, shape(20, 1 << 17, 101, 14, 256, 101, 4, 0), GroupLoad());
    for (const std::vector<StageChoice> &bad : {std::vector<StageChoice>{{1, 2, 18, 16}, {0, 0, 14, 14}}, {{0, 0, 18, 16}, {0, 0, 14, 14}}, {{3, 1, 18, 16}}, {}}) {
      Plan p(s);
      solve_stages(p, g, 100, bad);
      CHECK(p.refused());
    }
  }
  {  // more proof of work than the nonce has room for; `push_repeat` of 0
    Setting w = s;
    w.set_work(ErrorKind::FIELD, Scope::REQUEST, 59);  // 58 is the most that a nonce of 8 bytes has room for
    Plan p(w), q(s);
    plan_line(p, "");
    CHECK(p.refused());
    q.push_repeat(0);
    q.pop_repeat();
    CHECK(q.refused());
  }
}

// The committed table's blocks, with its record (stored answers to samples of its rows).
// A table gets a one-time test, at its one-time distance, and after it later tests.
static void test_table() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    TableSize T;
    T.code.log_n = 33, T.code.ell = 402653184, T.code.r = 44040191, T.v = 11;
    const Bound quarter = Bound::ratio(Big(1), Big(4));
    T.distances.radius = T.distances.later_radius = T.code.n() / 4;
    CHECK(table_admitted(s, T) && analysis(s, T).once.source == CountSource::THIRD);
    // eleven proximity terms of (2^31 + 1) / |E| and a list term
    const Bound fold = table_fold_error(s, T, analysis(s, T).once), by_hand = Bound::ratio(Big(11) * (Big::two_to(31) + Big(1)) + Big(11), s.card_E());
    CHECK(fold.hi <= Real::mul(by_hand.hi, Real::add(Real(1), Real::two_to(-150), Dir::UP), Dir::UP) && by_hand.lo <= fold.hi);
    const SampleSize made_up{24, no_sized_group()};
    LinSize x{made_up, made_up, made_up};
    x.instances = 64, x.k_lo = 18, x.rows_log = 22;
    TreeSize tree;
    tree.log_n = 33, tree.leaf_bytes = 16384;
    Setting w = s;
    w.set_work(ErrorKind::QUERY, Scope::SHARED, 28), w.set_work(ErrorKind::QUERY, Scope::ONCE, 32);
    Plan batch(w), once(w), link(w);
    batch.begin_scope(Scope::SHARED), once.begin_scope(Scope::ONCE), link.begin_scope(Scope::ONCE);
    CHECK(plan_lin_shared(batch, "", T, x, tree) == 241 && batch.finish()->holds() && spoken(batch) == 5 && batch.finish()->rounds() == 7);
    x.one_time = true, x.instances = 1;
    CHECK(plan_lin_shared(once, "", T, x, tree) == 232 && once.finish()->holds());
    u64 a = 0, b = 0;
    plan_link(link, "", T, x.u, tree, a, b);
    CHECK(a == 232 && b == 232 && link.finish()->holds() && spoken(link) == 4 && link.finish()->rounds() == 5);
    for (auto &r : once.rounds_)
      if (r.why == ErrorKind::CLOSENESS) CHECK(r.rests == CountSource::THIRD);
    {  // The record's round is no query of the plan's hash chain (a `Cursor` skips it); more
       // answers lower its error; a later test beyond the one-time distance is rejected.
      TableSize R = T;
      R.distances.radius = R.distances.later_radius = 0, R.distances.m_once = 100, R.distances.m_later = 90;
      Plan p(w);
      plan_record(p, "", R);
      CHECK(table_admitted(s, R) && p.finish()->holds() && p.rounds_.size() == 1 && p.rounds_[0].kind == RoundKind::RECORD && p.rounds_[0].work_bits == 0);
      CHECK(p.finish()->size_of(0).bytes == (u64)2048 * 24);  // the answers; the seeds are in no file
      if (s.circle == setting_of_this_build().circle) {  // a `Cursor` takes only a plan over this build's field
        Cursor walk(p);
        CHECK(walk.at_end());
      }
      TableSize two = R;
      two.answers = 2;
      Plan q(w);
      plan_record(q, "", two);
      CHECK(q.finish()->size_of(0).bytes == 2 * p.finish()->size_of(0).bytes && q.rounds_[0].draw_E == 2 * p.rounds_[0].draw_E && q.rounds_[0].error < p.rounds_[0].error);
      std::swap(R.distances.m_once, R.distances.m_later);
      Plan z(w);
      plan_record(z, "", R);
      CHECK(!table_admitted(s, R) && z.refused());
      /* The lower limit on the slack (Johnson radius minus distance). The record's round takes the
         list size at the largest distance at which the table is ever read, and it has no proof of
         work. At a distance so close to the Johnson radius that the squared list size outweighs the
         sample's error, the plan misses the target. */
      TableSize near = T;
      near.distances.radius = near.distances.later_radius = 0, near.distances.m_once = near.distances.m_later = (u64)1 << 40;
      Plan y(w);
      plan_record(y, "", near);
      CHECK(table_admitted(s, near) && !y.refused() && y.finish()->cause() == Cause::ABOVE_THE_TARGET);
    }
  }
}

/* The analysis of a group or of the table is a function of sizes, choices and setting, and
   `GroupSize`, `TableSize`, `SampleSize` and `CodeSize`, which a caller fills, have no member
   for it. The static_asserts below check that certain inputs that contradict each other cannot be
   written, such as a distance, a count or a list size beside codes of which it is untrue. Each
   message but "the looking glass is blind" (a control) names what would become possible. */
#define MEMBER_SEEN(m)                                                  \
  template <class X, class = void> struct has_##m : std::false_type {}; \
  template <class X> struct has_##m<X, std::void_t<decltype(std::declval<X>().m)>> : std::true_type {}
MEMBER_SEEN(at);
MEMBER_SEEN(mask_at);
MEMBER_SEEN(list);
MEMBER_SEEN(fault);
MEMBER_SEEN(once);
MEMBER_SEEN(later);
MEMBER_SEEN(repeated_k);
MEMBER_SEEN(k);  // the control: a member that is there is seen
static_assert(has_k<GroupSize>::value && has_at<Analysis>::value && has_once<TableAnalysis>::value, "the looking glass is blind");
static_assert(!has_at<GroupSize>::value && !has_mask_at<GroupSize>::value && !has_list<GroupSize>::value && !has_fault<GroupSize>::value, "a group stores a result of the analysis");
static_assert(!has_once<TableSize>::value && !has_later<TableSize>::value, "the table stores a result of the analysis");
static_assert(!has_list<SampleSize>::value, "a sample is handed a list as a bare number");
static_assert(!std::is_default_constructible<TestedIn>::value && !std::is_default_constructible<SampleSize>::value && !std::is_default_constructible<LinSize>::value, "the groups whose lists count can be left empty without the name");
static_assert(!std::is_convertible<std::vector<GroupSize>, TestedIn>::value && !std::is_constructible<TestedIn, std::initializer_list<GroupSize>>::value, "a list of any length, the empty one too, can be handed in for the groups");
static_assert(std::is_convertible<GroupSize, TestedIn>::value && std::is_constructible<TestedIn, GroupSize, GroupSize>::value, "the looking glass is blind");
static_assert(!has_repeated_k<CodeSize>::value, "a code can say of itself that it is a repetition");
// (Who may write a plan's rounds: plan_types_test.cpp, which does not define this file's macro.)
static_assert(!std::is_default_constructible<Word>::value && !std::is_constructible<Word, unsigned, u64>::value, "the formulas' k can be written by hand");
static_assert(!std::is_convertible<Bound, u64>::value && !std::is_invocable<decltype(static_cast<GroupSize (*)(const GroupSize &, u64)>(group_at)), GroupSize, Bound>::value, "an interval can be handed in as a distance");
static_assert(!std::is_invocable<decltype(plan_bat), Plan &, std::string, Big, u64>::value, "BAT is handed a list as a bare number");

static void test_analysis() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    GroupSize shape;
    shape.main.log_n = 20, shape.main.ell = 1 << 17, shape.main.r = 101, shape.mask.log_n = 14, shape.mask.ell = 1024, shape.mask.r = 101, shape.k = 4, shape.h = 2;
    const GroupSize makers[] = {size_group(s, Scope::REQUEST, shape, GroupLoad()), group_of_m(shape, 50), group_at(shape, shape.main.n() / 4)};
    const GroupSize other = group_of_m(shape, 40);
    for (const GroupSize &g : makers) {
      CHECK(!analysis(s, g).bad());  // the control
      {  // What a plan records with a round (`PlanRound::group`, with `said` set): the whole group
         // at FOLD's rounds and BASE's first round; only the two codes where positions are drawn;
         // nothing at the other rounds.
        const std::vector<TreeSize> none;
        Plan q(s);
        plan_fold(q, "", g, 0), plan_loc(q, "", g, none), plan_cs(q, "", g, other, 10, 3, 3, none, 1), plan_base(q, "", g, none, 0);
        unsigned whole = 0, codes = 0, silent = 0;
        for (const PlanRound &r : q.rounds_) {
          const bool begins = r.name == R_FOLD_EPSILON || r.name == R_FOLD_ROUND || r.name == R_BASE_LAMBDA, draws = r.kind == RoundKind::POSITIONS;
          if (begins) whole += r.group.said && r.group == priced(g);
          if (draws) codes += r.group.said && r.group == priced_codes(g) && r.group.main == priced(g.main) && r.group.mask == priced(g.mask) && !r.group.k && !r.group.h && r.group.slots == 1;
          if (!begins && !draws) silent += !r.group.said;
          if (begins || draws) CHECK(!r.table.said);
        }
        CHECK(whole == 2 + g.k && codes == 3 && silent + 5 + g.k == q.rounds_.size());
        // `Priced` (blocks/numbers/plan.h): what a round's error was computed from. A round that
        // records a group has AS_SAID: from what is recorded. The code switch's round has
        // ON_THE_CALLERS_WORD: also from a number that a caller passed in and nothing checks at
        // that round, here the next stage's list size in the batching term.
        for (const PlanRound &r : q.rounds_) CHECK(!r.word_fault() && (r.group.said ? r.priced == (r.name == R_CS ? Priced::ON_THE_CALLERS_WORD : Priced::AS_SAID) : true));
      }
      const std::vector<TreeSize> none;
      auto refused = [&](const GroupSize &bent, const char *what) {
        Plan fold(s), rounds(s), loc(s), from(s), to(s), base(s), bat(s), send(s);
        plan_fold(fold, "", bent, 0), plan_fold_rounds(rounds, "", bent), plan_loc(loc, "", bent, none), plan_cs(from, "", bent, g, 10, 3, 3, none, 1), plan_cs(to, "", g, bent, 10, 3, 3, none, 1);
        plan_base(base, "", bent, none, 0), plan_bat(bat, "", {g, bent}, 5), plan_send(send, "", {SampleSize{10, {g, bent}}}, 1);
        const bool all = fold.refused() && rounds.refused() && loc.refused() && from.refused() && to.refused() && base.refused() && bat.refused() && send.refused() && !fold.finish()->holds();
        if (!all || !analysis(s, bent).bad()) std::printf("not refused: %s (rule %d, m %llu)\n", what, (int)g.rule, (unsigned long long)g.m);
        CHECK(all && analysis(s, bent).bad());
      };
      GroupSize b;
      b = g, b.mask.log_n = 11, b.mask.ell = 1024, b.mask.r = 1024, refused(b, "the mask code at rate 1");
      b = g, b.mask.ell = (u64)1 << 15, refused(b, "a mask code with more entries than positions");
      b = g, b.h = b.k + 1, refused(b, "h = k + 1");  // every analysis checks the shape rules
      // Error bounds (`price` in identifiers) follow the codes: changing a code changes analysis
      // and, except under `Radius::AT`, positions.
      b = g, b.main.ell *= 4;
      {
        const Analysis was = analysis(s, g), is = analysis(s, b);
        Plan p(s), q(s);
        const u64 t = plan_loc(p, "", g, none), u = plan_loc(q, "", b, none);
        if (g.rule == Radius::AT) CHECK(is.bad() || is.at.source != was.at.source);  // the named distance is no longer within a third of the code's
        else CHECK(!is.bad() && was.at.agree.hi < is.at.agree.lo && t < u && t == plan_positions(p, was.at.agree) && u == plan_positions(q, is.at.agree));
      }
      // k, h and slots may change under one sized group, as between a request's segments
      // (`ReqLink` in blocks/iop/req.h): the list size stays the same.
      b = g, b.k = 7, b.h = 1, b.slots = 16;
      CHECK(!analysis(s, b).bad() && analysis(s, b).list == analysis(s, g).list);
    }
    {  // A sample's list size is the largest of its groups'; with `no_sized_group`: 1.
      Plan one(s), two(s), zero(s);
      plan_send(one, "", {SampleSize{10, {makers[1]}}}, 1), plan_send(two, "", {SampleSize{10, {makers[2], makers[1]}}}, 1), plan_send(zero, "", {SampleSize{10, no_sized_group()}}, 1);
      CHECK(Big(1) < analysis(s, makers[1]).list && analysis(s, makers[2]).list == Big(1));
      CHECK(Real::cmp(one.rounds_[0].error, two.rounds_[0].error) == 0 && zero.rounds_[0].error < one.rounds_[0].error);
    }
    {  // A `Bound` with lo > hi is bad; so is a code with more entries than positions.
      const Bound a = Bound::ratio(Big(74), Big(100)), z = Bound::ratio(Big(76), Big(100));
      CHECK(!Bound(a.lo, z.hi).bad && Bound(z.lo, a.hi).bad && !Bound(a.lo, a.lo).bad);
      // An interval of some width is valid: agree^t takes its upper end, the count its lower end.
      CHECK(!at_distance(s, shape.main, Bound(a.lo, z.hi)).bad() && at_distance(s, shape.main, Bound(z.lo, a.hi)).bad());
      CHECK((Bound(z.lo, a.hi) + a).bad && (a * Bound(z.lo, a.hi)).bad);  // and what is made of it is bad
      CodeSize none;
      none.log_n = 4, none.ell = 20;
      CHECK(at_distance(s, none, Bound::ratio(Big(3), Big(4))).bad() && johnson(s, none, 10).bad());  // length 16, dimension 20
      none.ell = 16, none.r = 1;
      CHECK(at_distance(s, none, Bound::ratio(Big(3), Big(4))).bad());
    }
    for (u64 m : {0ull, 1ull, 2ull}) CHECK(analysis(s, group_of_m(shape, m)).bad());  // the theorem's integer is at least 3

    TableSize T;
    T.code.log_n = 33, T.code.ell = 402653184, T.code.r = 44040191, T.v = 11, T.distances.radius = T.distances.later_radius = T.code.n() / 4;
    TableSize R = T;  // distances by the theorem's integers
    R.answers = 2, R.distances.radius = R.distances.later_radius = 0, R.distances.m_once = 40, R.distances.m_later = 20;
    for (const TableSize &t : {T, R}) {
      CHECK(table_admitted(s, t));
      auto refused = [&](const TableSize &bent) {
        const SampleSize made_up{24, no_sized_group()};
        LinSize x{made_up, made_up, made_up};
        x.instances = 64, x.k_lo = 18, x.rows_log = 22;
        TreeSize tree;
        tree.log_n = 33, tree.leaf_bytes = 16384;
        Plan lin(s), link(s), record(s);
        u64 a = 0, b = 0;
        plan_lin_shared(lin, "", bent, x, tree), plan_link(link, "", bent, x.u, tree, a, b), plan_record(record, "", bent);
        CHECK(!table_admitted(s, bent) && lin.refused() && link.refused() && record.refused());
      };
      TableSize b;
      b = t, b.code.ell = b.code.n(), refused(b);  // more entries than positions
      if (!t.distances.m_once) b = t, b.distances.radius = t.code.n(), refused(b);
      if (t.distances.m_once) b = t, b.distances.m_later = t.distances.m_once + 1, refused(b);  // a later test beyond the one-time distance
      if (t.distances.m_once) b = t, b.distances.m_later = 2, refused(b);
      if (t.distances.m_once) b = t, b.answers = 0, refused(b);
    }
    {  // The theorem's integers are accepted: a table of rate 0.21, m_once = 40, m_later = 20.
      TableSize H;
      H.code.log_n = 14, H.code.ell = 3438, H.v = 3, H.distances.m_once = 40, H.distances.m_later = 20;
      CHECK(table_admitted(s, H));
    }
    {  // the one-time agreement follows from code and radius
      const TableAnalysis a = analysis(s, T);
      CHECK(Real::cmp(a.once.agree.hi, Bound::ratio(Big(3), Big(4)).hi) == 0);
    }
  }
}

// Boundary cases of blocks/numbers/codes.h: known answers where one comparison decides which bound
// or which integer m applies.
static void test_edges() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    // The bound within a third of the distance needs 3 e < d (e: the tested distance in positions;
    // d: the code's). At e = 100 of 1,024 positions and d = 301, 300, 299, only d = 301 qualifies;
    // its count is e + 1.
    for (u64 d : {301, 300, 299}) {
      CodeSize c;
      c.log_n = 10;
      for (c.ell = 600; c.ell < 900 && c.n() - c.k(s) != d; ++c.ell) {}
      const Proximity x = at_distance(s, c, Bound(1) - Bound::ratio(Big(100), Big(1024)));
      CHECK(c.n() - c.k(s) == d && !x.bad() && (x.source == CountSource::THIRD) == (d == 301));
      if (d == 301) CHECK(x.count.hi.floor() == Big(101) && x.count.lo.ceil() == Big(101) && x.list == Big(1));
      const Proximity most = at_distance(s, c, Bound::ratio(Big(c.n() - (d - 1) / 3), Big(c.n())));  // the largest distance below a third: (d - 1) div 3 positions
      CHECK(most.source == CountSource::THIRD && most.count.hi.floor() == Big((d - 1) / 3 + 1));
    }
    // The count of the theorem on codes, written out again:
    // (2 M^5 + 3 M delta rho) / (3 rho^(3/2)) n + M / sqrt(rho), M = m + 1/2.
    // The comparison, to one part in 2^150, sees both terms.
    CodeSize c;
    c.log_n = 20, c.ell = 1 << 17, c.r = 101;
    const Bound rho = Bound::ratio(Big(c.k(s)), Big(c.n())), root = Bound::sqrt(rho), n(Big(c.n()));
    auto count = [&](u64 m, const Bound &delta) {
      const Bound M = Bound::ratio(Big(2 * m + 1), Big(2));
      return (Bound(2) * M * M * M * M * M + Bound(3) * M * delta * rho) / (Bound(3) * rho * root) * n + M / root;
    };
    auto alike = [](const Real &x, const Real &y) { return Real::sub(std::max(x, y), std::min(x, y), Dir::UP) <= Real::mul(y, Real::two_to(-150), Dir::DOWN); };
    const Proximity j = johnson(s, c, 50);
    CHECK(!j.bad() && alike(j.count.hi, count(50, j.delta).hi) && alike(j.agree.hi, (root + root / Bound(50)).hi));
    // m is at least 3: it is 3 even at a distance so far below the Johnson radius that
    // sqrt(rho) / eta is below 3 (`eta`: the slack, 1 - sqrt(rho) - delta).
    CodeSize thin;
    thin.log_n = 20, thin.ell = 1 << 10;
    const Proximity far_below = at_distance(s, thin, Bound::ratio(Big(1), Big(2)));
    const Bound thin_rho = Bound::ratio(Big(thin.k(s)), Big(thin.n())), thin_root = Bound::sqrt(thin_rho);
    CHECK(far_below.source == CountSource::JOHNSON && surely_lt(thin_root / far_below.eta, Bound(1)) && far_below.m == 3);
    const Bound M = Bound::ratio(Big(7), Big(2));
    CHECK(alike(far_below.count.hi, ((Bound(2) * M * M * M * M * M + Bound(3) * M * far_below.delta * thin_rho) / (Bound(3) * thin_rho * thin_root) * Bound(Big(thin.n())) + M / thin_root).hi));
    for (u64 m : {0, 1, 2}) CHECK(johnson(s, c, m).bad());
    CHECK(!johnson(s, c, 3).bad());
  }
}

/* Every block's round errors, computed a second time; not RECORD's (a table's stored sample
   answers), and LIN's and LINK's (the tests on the committed table) only at distances named in
   positions. Prover and verifier both read blocks/numbers/blocks.h, so comparing
   the two never shows a wrong formula there. Here each round's error is written out as quotients
   over |E|, without the error functions of that file or of blocks/numbers/codes.h (`field_error`,
   `sample_error`, `fold_round_error` ...). Taken from them: `analysis`, for a group's or a table's
   distance, counts and list size, which other tests check, and each plan's own numbers of positions
   and combinations.
   This shows that a formula has not changed since the test was written: a term dropped, a list size
   left out, a sum turned into a maximum. It does not show that a formula is right: the formulas
   here were written from the comments and code of blocks/numbers/blocks.h, not from a source
   outside this code.
   The sizes are chosen so that no two terms of a sum are equal. */
static void test_prices() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    const Bound E(s.card_E());
    auto over_E = [&](u64 x) { return Bound(Big(x)) / E; };
    auto is = [&](const PlanRound &r, const char *name, const Bound &by_hand) {
      // Equal up to rounding, which depends on the order of operations: within one part in 2^150
      // (numbers have 192 bits). So a dropped term is noticed unless it is below 2^-150 of its sum.
      const Real &x = r.error, &y = by_hand.hi;
      const bool fine = r.name == name && !by_hand.bad && !x.zero() && Real::sub(std::max(x, y), std::min(x, y), Dir::UP) <= Real::mul(y, Real::two_to(-150), Dir::DOWN);
      if (!fine) std::printf("price: %s (%s): 2^%.6f, by hand 2^%.6f\n", name, r.name.c_str(), r.error.log2(), by_hand.hi.log2());
      return fine;
    };
    auto round_of = [&](const Plan &p, const char *name, unsigned which = 0) -> const PlanRound & {
      for (const PlanRound &r : p.rounds_)
        if (r.name == name && !which--) return r;
      CHECK(false);
      return p.rounds_.front();
    };
    GroupSize shape;
    shape.main.log_n = 20, shape.main.ell = 1 << 17, shape.main.r = 101, shape.mask.log_n = 14, shape.mask.ell = 1024, shape.mask.r = 101, shape.k = 4, shape.h = 2, shape.slots = 4;
    const GroupSize g = group_of_m(shape, 50), other = group_of_m(shape, 9);
    const Analysis a = analysis(s, g), b = analysis(s, other);
    const Bound L(a.list), M(b.list);
    CHECK(!a.bad() && !b.bad() && Big(1) < b.list && b.list < a.list);
    const std::vector<TreeSize> none;
    {  // SEND (oracle messages, each sampled out of domain): the sum over one prover message's
       // oracle messages, each with its own list size and cube_bits (the variables of its sample's
       // point), of (list^2 / 2) (cube_bits / |E|)^samples, for cube_bits >= 2
      Plan p(s);
      plan_send(p, "", {SampleSize{20, {g}}, SampleSize{13, {other}}, SampleSize{7, no_sized_group()}}, 1);
      CHECK(s.samples == 1 && is(round_of(p, R_SEND_SEED), R_SEND_SEED, L * L / Bound(2) * over_E(20) + M * M / Bound(2) * over_E(13) + Bound(1) / Bound(2) * over_E(7)));
      Setting two = s;
      two.samples = 2;
      Plan q(two);
      plan_send(q, "", {SampleSize{20, {g}}}, 1);
      CHECK(is(round_of(q, R_SEND_SEED), R_SEND_SEED, L * L / Bound(2) * over_E(20) * over_E(20)));
    }
    {  // BAT: list (rows - 1) / |E|; BAT-sigma: (rows - 1) / |E|
      Plan p(s);
      plan_bat(p, "", {g}, 1000), plan_bat_sigma(p, "", 1000);
      CHECK(is(round_of(p, R_BAT), R_BAT, L * over_E(999)) && is(round_of(p, R_BAT_SIGMA), R_BAT_SIGMA, over_E(999)));
    }
    {  // PSC. epsilon: 1 / |E|. A round with `degree` d, whose polynomial has degree
       // below d: (d - 1) / |E|. Y, drawn from E without 0: degree_Y / (|E| - 1).
      PscSize x;
      x.degree = {3, 5, 2}, x.degree_Y = 4;
      Plan p(s);
      plan_psc(p, "", x);
      CHECK(is(round_of(p, R_PSC_EPSILON), R_PSC_EPSILON, over_E(1)));
      CHECK(is(round_of(p, R_PSC_ROUND, 0), R_PSC_ROUND, over_E(2)) && is(round_of(p, R_PSC_ROUND, 1), R_PSC_ROUND, over_E(4)) && is(round_of(p, R_PSC_ROUND, 2), R_PSC_ROUND, over_E(1)));
      CHECK(is(round_of(p, R_PSC_Y), R_PSC_Y, Bound(4) / Bound(s.card_E() - Big(1))));
    }
    {  // ZERO (a zero-check's point), JOIN: variables / |E|; LINE (two evaluation claims into one):
       // 1 / |E|; a lookup's shift: (its tables' entries) / |E|; a lookup's alpha: (keys - 1) / |E|
      Plan p(s);
      plan_zero(p, "", 23), plan_join(p, "", 11), plan_line(p, ""), plan_shift(p, "", 4096), plan_lkp(p, "", 16);
      CHECK(is(round_of(p, R_ZERO), R_ZERO, over_E(23)) && is(round_of(p, R_JOIN), R_JOIN, over_E(11)) && is(round_of(p, R_LINE), R_LINE, over_E(1)));
      CHECK(is(round_of(p, R_LKP_SHIFT), R_LKP_SHIFT, over_E(4096)) && is(round_of(p, R_LKP), R_LKP, over_E(65535)));
      Plan q(s);
      plan_lookup(q, "", LookupSize(0, 12, 5, true, 0, {14, 12, 9}));  // three parts: 2^14 + 2^12 + 2^9 keys; a table of 2^12 entries
      CHECK(!q.refused() && is(round_of(q, R_LKP_SHIFT), R_LKP_SHIFT, over_E(4096)) && is(round_of(q, R_LKP), R_LKP, over_E(16384 + 4096 + 512 - 1)));
    }
    // A proximity term: count / |E|; for circle codes (setting_m61()) with one list size more
    // beyond a third of the distance (the encoding's image is a subcode).
    auto closeness = [&](const Proximity &x) { return (s.circle && x.source != CountSource::THIRD ? x.count + Bound(x.list) : x.count) / E; };
    {  // FOLD. epsilon: list / |E|. Round j: the main rows' proximity term + 2 list / |E| and,
       // while the mask rows are folded too (j <= h), twice their proximity term.
      Plan p(s);
      plan_fold(p, "", g, 0);
      CHECK(is(round_of(p, R_FOLD_EPSILON), R_FOLD_EPSILON, L * over_E(1)));
      for (unsigned j = 1; j <= g.k; ++j)
        CHECK(is(round_of(p, R_FOLD_ROUND, j - 1), R_FOLD_ROUND, closeness(a.at) + L * over_E(2) + (j <= g.h ? Bound(2) * closeness(a.mask_at) : Bound((u64)0))));
    }
    {  // LOC, a draw of t positions: agree^t. CS: agree^t of the stage that ends
       // + (the next stage's list) (rows - 1) / |E|; rows: the old rows, the samples and the
       // generator rows at each position.
      Plan p(s);
      const u64 t = plan_loc(p, "", g, none);
      CHECK(t > 50 && is(round_of(p, R_LOC), R_LOC, Bound::pow(a.at.agree, t)));
      Plan q(s);
      const u64 u = plan_cs(q, "", g, other, 10, 7, 3, none, 1);
      const PlanRound &r = round_of(q, R_CS);
      CHECK(u >= t && r.rows == 7 + s.samples + 3 * u && r.draw_positions == u);
      const Bound positions = Bound::pow(a.at.agree, u), batching = M * over_E(r.rows - 1);
      // The batching term is much smaller than the positions' term, so the sum is compared exactly,
      // upper bound with upper bound; the round's error exceeds the positions' term by the batching
      // term.
      CHECK(Real::cmp(r.error, (positions + batching).hi) == 0 && positions.hi < r.error);
      CHECK(std::fabs(Real::sub(r.error, positions.hi, Dir::UP).log2() - batching.hi.log2()) < 1e-6);
    }
    {  // BASE. lambda: the two proximity terms, once each, + list / |E|. The positions: agree^t.
      GroupSize last = g;
      last.k = last.h = 0, last.slots = 1;
      const Analysis z = analysis(s, last);
      Plan p(s);
      plan_base(p, "", last, none, 0);
      CHECK(is(round_of(p, R_BASE_LAMBDA), R_BASE_LAMBDA, closeness(z.at) + closeness(z.mask_at) + Bound(z.list) * over_E(1)));
      CHECK(is(round_of(p, R_BASE), R_BASE, Bound::pow(z.at.agree, round_of(p, R_BASE).draw_positions)));
    }
    {  // RNG, the approximate range check (`SMALL`, `small` in identifiers), with N combinations:
       // 2^-N
      Plan p(s);
      plan_rng(p, "", 1 << 10);
      CHECK(is(round_of(p, R_RNG), R_RNG, Bound(Real::two_to(-(long)round_of(p, R_RNG).combinations))));
    }
    {  // LIN, the test on the committed table. The shared point: requests (v + k_lo) / |E| and v
       // proximity terms + list v / |E|. The nu columns: agree^nu. zeta: rows_log / |E|.
      TableSize T;
      T.code.log_n = 33, T.code.ell = 402653184, T.code.r = 44040191, T.v = 11, T.distances.radius = T.code.n() / 4, T.distances.later_radius = T.code.n() / 8;
      const TableAnalysis A = analysis(s, T);
      const SampleSize made_up{24, no_sized_group()};
      LinSize x{made_up, made_up, made_up};
      x.instances = 64, x.k_lo = 18, x.rows_log = 22;
      TreeSize tree;
      tree.log_n = 33, tree.leaf_bytes = 16384;
      Plan later(s), once(s);
      const u64 nu = plan_lin_shared(later, "", T, x, tree);
      const Bound fold_later = Bound(11) * closeness(A.later) + Bound(A.later.list) * over_E(11);
      CHECK(is(round_of(later, R_LIN_POINT), R_LIN_POINT, Bound(64) * over_E(11 + 18) + fold_later) && is(round_of(later, R_LOC), R_LOC, Bound::pow(A.later.agree, nu)) && is(round_of(later, R_LIN_ZETA), R_LIN_ZETA, over_E(22)));
      x.one_time = true, x.instances = 1;
      const u64 mu = plan_lin_shared(once, "", T, x, tree);
      const Bound fold = Bound(11) * closeness(A.once) + Bound(A.once.list) * over_E(11);
      CHECK(mu < nu && is(round_of(once, R_LIN_POINT), R_LIN_POINT, over_E(11 + 18) + fold) && is(round_of(once, R_LOC), R_LOC, Bound::pow(A.once.agree, mu)));
      // LINK ties a further committed table (a leaf table) to the first one (the anchor); `further`
      // and `first` are their numbers of drawn columns. The point: v / |E| + `fold` for the leaf
      // table + `fold_later` for the anchor. The two folded rows' samples: twice one sample's error. The two lists of columns:
      // the larger of their errors, not the sum and not the smaller.
      // phi: (a row's variables) / |E|.
      Plan link(s);
      u64 first = 0, further = 0;
      plan_link(link, "", T, SampleSize{24, {g}}, tree, first, further);
      const Bound e0 = Bound::pow(A.later.agree, first), ej = Bound::pow(A.once.agree, further);
      CHECK(is(round_of(link, R_LINK_POINT), R_LINK_POINT, over_E(11) + fold + fold_later) && is(round_of(link, R_SEND_SEED), R_SEND_SEED, Bound(2) * (L * L / Bound(2) * over_E(24))));
      CHECK(Real::cmp(e0.hi, ej.hi) != 0 && Real::cmp(round_of(link, R_LINK).error, std::max(e0.hi, ej.hi)) == 0 && is(round_of(link, R_LINK_PHI), R_LINK_PHI, over_E(29)));
    }
    {  // A challenge drawn from E without 0 has |E| - 1 values. Beside the real |E| the 1 is below
       // the tolerance of `is`, hence a field of 101 elements (a plan is computed, not run).
      Setting tiny = s;
      tiny.p = Big(101), tiny.degree = 1;
      PscSize x;
      x.degree = {3}, x.degree_Y = 4;
      Plan p(tiny);
      plan_psc(p, "", x), plan_send(p, "", {SampleSize{1, no_sized_group()}}, 1);
      CHECK(is(round_of(p, R_PSC_Y), R_PSC_Y, Bound::ratio(Big(4), Big(100))) && is(round_of(p, R_PSC_ROUND), R_PSC_ROUND, Bound::ratio(Big(2), Big(101))));
      CHECK(is(round_of(p, R_SEND_SEED), R_SEND_SEED, Bound::ratio(Big(1), Big(200))));  // cube_bits = 1: the pad factor's 1 / (|E| - 1) exceeds 1 / |E|; list size 1: list^2 / 2 = 1 / 2
    }
    {  // a request's rounds count once for each request that shares a session
      Setting many = s;
      many.instances = 24;
      Plan p(many);
      plan_zero(p, "", 23);
      CHECK(p.scope() == Scope::REQUEST && is(round_of(p, R_ZERO), R_ZERO, Bound(24) * over_E(23)));
    }
  }
}

/* One stage of a proximity test, one distance. A group's `rule` and `m` choose the distance at
   which its errors are bounded. A stage whose proximity terms are computed at a near distance and
   whose positions at a far one is bounded at neither. The plan records the choice with every round
   that records a group, and `holds` requires that all rounds on the same two codes record one
   choice. `account` does not require it: it also gets plans over several hash chains, where two
   stages of different tests may validly share both codes. */
static void test_one_distance() {
  for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
    GroupSize shape;
    shape.main.log_n = 20, shape.main.ell = 1 << 17, shape.main.r = 101, shape.mask.log_n = 14, shape.mask.ell = 1024, shape.mask.r = 101, shape.k = 4, shape.h = 2, shape.slots = 4;
    const std::vector<TreeSize> none;
    const GroupSize near = group_at(shape, (shape.main.n() - shape.main.k(s) - 1) / 3), far = group_of_m(shape, 50), farther = group_of_m(shape, 51), named = group_at(shape, 300000);
    auto stage = [&](const GroupSize &head, const GroupSize &positions) {
      Plan p(s);
      plan_fold(p, "", head, 0), plan_loc(p, "", positions, none);
      CHECK(!p.refused() && account(p).holds);
      for (const PlanRound &r : p.rounds_) CHECK(p.finish()->holds(r, p.finish()->capped()) && !r.word_fault());  // every round by itself is fine: it is the two together that are not
      return p.finish()->holds();
    };
    for (const GroupSize &g : {near, far, farther, named}) CHECK(stage(g, g));
    CHECK(!stage(near, far) && !stage(far, near) && !stage(far, farther) && !stage(near, named) && !stage(named, far));
    {  // What a mixed plan would gain: the near distance's bound within a third for the proximity terms,
       // and the far distance's positions, fewer than half as many.
      Plan a(s), b(s);
      CHECK(plan_loc(a, "", near, none) > 2 * plan_loc(b, "", far, none));
    }
    {  // two stages of one proximity test: other codes, so another choice is valid
      GroupSize next = shape;
      next.main.log_n = 19, next.main.ell = 1 << 15;
      Plan p(s);
      plan_fold(p, "", far, 0), plan_cs(p, "", far, group_of_m(next, 9), 10, 7, 3, none, 1), plan_fold(p, "", group_of_m(next, 9), 0), plan_loc(p, "", group_of_m(next, 9), none);
      CHECK(!p.refused() && p.finish()->holds());
    }
  }
}

// Every round has a `Priced` value (`word` in identifiers) that states what its error was computed from (the five
// values: `enum class Priced`, blocks/numbers/plan.h). holds() checks it against the round, by PlanRound::word_fault().
// A number is "recorded" with a round if it is set in a field that PlanRound::says() reads. `Accounting::on_a_word`
// counts the rounds with ON_THE_CALLERS_WORD (from a caller's number that nothing checks there). A
// draw of positions, error 0, under the three `uncounted` values below is a fault: such a round does not hold at the target (128).
static void test_words() {
  const Setting s = setting_of_this_build();
  const Bound small(Real::two_to(-140)), none((u64)0);
  // say: which of the six fields that PlanRound::says() reads is set; 0: none
  auto one = [&](const Bound &error, Priced word, int say) {
    Plan p(s);
    PlanRound &r = p.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, error, word);
    if (say == 1) r.rows = 3;
    if (say == 2) r.first_list = 1;
    if (say == 3) r.group.said = true;
    if (say == 4) r.table.said = true;
    if (say == 5) r.combinations = 7;
    if (say == 6) r.sharing = 64;
    return p.finish()->holds();
  };
  CHECK(one(none, Priced::NOTHING, 0) && !one(small, Priced::NOTHING, 0));  // NOTHING needs error 0
  {  // A draw of positions or of a matrix always has an error. One position, error 0, under NOTHING, must not
     // hold at the target (128): such a round is a fault (Cause::WORD) unless ON_THE_CALLERS_WORD. Control: E.
    Plan p(s), q(s), control(s);
    p.close("a test's", RoundKind::POSITIONS, ErrorKind::QUERY, none, Priced::NOTHING).draw_positions = 1;
    q.close("a test's", RoundKind::BITS, ErrorKind::SMALL, none, Priced::NOTHING).draw_trits = 64;
    control.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, none, Priced::NOTHING).draw_E = 1;
    CHECK(p.finish()->cause() == Cause::WORD && !account(p).holds && q.finish()->cause() == Cause::WORD && control.finish()->holds());
    Plan by_hand(s);  // under ON_THE_CALLERS_WORD the same round passes, and `on_a_word` counts it
    by_hand.close("a test's", RoundKind::POSITIONS, ErrorKind::QUERY, none, Priced::ON_THE_CALLERS_WORD).draw_positions = 1;
    CHECK(by_hand.finish()->holds() && account(by_hand).on_a_word == 1);
    for (const Priced uncounted : {Priced::SETTING_ALONE, Priced::THE_ROUND_ITSELF, Priced::AS_SAID}) {  // `on_a_word` does not count these three: without the fault, error 0 would go unnoticed
      Plan positions(s), matrix(s);
      positions.close("a test's", RoundKind::POSITIONS, ErrorKind::QUERY, none, uncounted).draw_positions = 1;
      matrix.close("a test's", RoundKind::BITS, ErrorKind::SMALL, none, uncounted).draw_trits = 64;
      if (uncounted == Priced::AS_SAID) positions.rounds_[0].rows = 1, matrix.rounds_[0].combinations = 1;  // AS_SAID with nothing recorded would be a fault of its own
      CHECK(positions.finish()->cause() == Cause::WORD && matrix.finish()->cause() == Cause::WORD && !account(positions).holds && !account(matrix).holds);
    }
  }
  {  // `holds` and `account` agree on a `word_fault`, also at a round with error 0: `account`
     // checks the `Priced` value. On one distance for a stage they differ on purpose (`test_one_distance`).
    Plan p(s);
    p.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-140)), Priced::SETTING_ALONE);
    CHECK(p.finish()->holds() && account(p).holds);
    p.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, none, Priced::AS_SAID);  // AS_SAID, and nothing is recorded
    CHECK(p.finish()->cause() == Cause::WORD && !account(p).holds);
  }
  for (int say = 1; say <= 6; ++say) {
    CHECK(one(none, Priced::NOTHING, say));  // with NOTHING and error 0 a round may record numbers
    CHECK(!one(small, Priced::SETTING_ALONE, say) && !one(small, Priced::THE_ROUND_ITSELF, say) && one(small, Priced::AS_SAID, say) && one(small, Priced::ON_THE_CALLERS_WORD, say));
  }
  CHECK(one(small, Priced::SETTING_ALONE, 0) && one(small, Priced::THE_ROUND_ITSELF, 0) && !one(small, Priced::AS_SAID, 0) && one(small, Priced::ON_THE_CALLERS_WORD, 0));
  CHECK(one(none, Priced::AS_SAID, 1) && one(none, Priced::THE_ROUND_ITSELF, 0));  // a formula may give 0: one row to batch, a degree of 1
  {  // A plan keeps its setting through every copy and move. `Plan::set` is const, so a Plan
     // cannot be assigned (tested by plan_types_test.cpp).
    Setting other = s;
    other.target = 100;
    auto made = [&] {
      Plan p(other);
      p.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-110)), Priced::SETTING_ALONE);
      return p;
    };
    const Plan a = made();
    CHECK(a.set.target == 100 && a.finish()->holds() && a.finish()->set.target == 100);
    Plan *source = new Plan(a);
    Plan copied(*source), moved(std::move(*source));
    delete source;
    for (const Plan *p : {&copied, &moved}) CHECK(p->set.target == 100 && p->rounds_.size() == 1 && p->finish()->holds());
    std::vector<Plan> many(3, a);
    many.push_back(made());
    for (const Plan &p : many) CHECK(p.set.target == 100 && p.finish()->holds());
    {  // `append` copies a finished plan's rounds into another, for output over several chains. A
       // Plan made from a Planned (tests only) gets copies too: the Planned does not change.
      Plan whole(other);
      whole.append(made());
      CHECK(!whole.refused() && whole.rounds_.size() == made().rounds_.size());
      const Planned done = made().finish();
      Plan again(done);
      again.rounds_[0].error = Real::two_to(-10);
      CHECK(done->holds() && !again.finish()->holds() && again.finish()->cause() == Cause::ABOVE_THE_TARGET && again.set.target == 100);
      // append() of rounds computed under another setting makes the plan refused(). Below, each
      // field of the setting is altered by itself.
      Plan under_another(s);
      under_another.append(made());
      CHECK(under_another.refused() && !under_another.finish()->holds() && !account(under_another).holds);
      auto taken = [&](const Setting &bent) {
        Plan p(bent);
        p.append(made());
        return !p.refused();
      };
      CHECK(taken(other));  // the control
      std::vector<Setting> bent(19, other);
      bent[0].field_name += "x", bent[1].p = bent[1].p + Big(2), bent[2].degree++, bent[3].circle = !bent[3].circle, bent[4].f_bytes++, bent[5].e_bytes++, bent[6].digest_bytes++, bent[7].salt_bytes++;
      bent[8].nonce_bytes--, bent[9].position_bytes++, bent[10].hash_block_bytes++, bent[11].label_bytes++, bent[12].chain_extra_bytes++, bent[13].chain_free_words++, bent[14].target++;
      bent[15].instances++, bent[16].samples++, bent[17].saved_coefficient = !bent[17].saved_coefficient;
      bent[18].set_work(ErrorKind::QUERY, Scope::ONCE, 1);
      for (const Setting &x : bent) CHECK(!taken(x));
    }
    // A Cursor takes only a plan that holds() and whose field is this build's (p, circle, degree);
    // otherwise the process ends (ends_with(): blocks/chain/tests/death.h).
    CHECK(ends_with("reckoned over another field", [&] { Setting six = s; six.degree = 6; Plan p(six); Cursor c(p); }));
  }
  {  // LIN's shared point records its table and how many requests share it. Every round that
     // records a table has ON_THE_CALLERS_WORD. Of the 12 rounds planned here, seven have it: LIN's
     // shared point, its columns and its two SEND seeds; LINK's point, its SEND seed and its round
     // with the two lists of columns. Over all planners, blocks/numbers/blocks.h sets
     // ON_THE_CALLERS_WORD at 12 places, under ten round names: the five here, R_BAT, R_LKP_SHIFT,
     // R_LKP, R_CS and R_RECORD. These counts come from that file; this test checks none of them.
     // The callers' unchecked numbers are listed at `Priced` in blocks/numbers/plan.h.
    TableSize T;
    T.code.log_n = 33, T.code.ell = 402653184, T.code.r = 44040191, T.v = 11, T.distances.radius = T.code.n() / 4;
    const SampleSize made_up{24, no_sized_group()};
    LinSize x{made_up, made_up, made_up};
    x.instances = 64, x.k_lo = 18, x.rows_log = 22;
    TreeSize tree;
    tree.log_n = 33, tree.leaf_bytes = 16384;
    Plan p(s);
    plan_lin_shared(p, "", T, x, tree);
    unsigned points = 0;
    for (const PlanRound &r : p.rounds_) {
      const bool point = r.name == R_LIN_POINT;
      points += point;
      CHECK(point ? r.sharing == 64 && r.table.said && r.priced == Priced::ON_THE_CALLERS_WORD : !r.sharing);  // checked at run time: table, test, sharing; on the caller's word: the distances
      if (r.table.said) CHECK(r.priced == Priced::ON_THE_CALLERS_WORD);
    }
    CHECK(points == 1);
    Plan link(s);  // likewise LINK's point and its two lists of columns
    u64 first = 0, further = 0;
    plan_link(link, "", T, made_up, tree, first, further);
    unsigned said = 0;
    for (const PlanRound &r : link.rounds_) {
      if (!r.table.said) continue;
      said++;
      CHECK(r.priced == Priced::ON_THE_CALLERS_WORD);
    }
    CHECK(said == 2);
  }
  {  // RNG's round records its number of combinations
    Plan p(s);
    plan_rng(p, "", 1 << 10);
    const PlanRound &r = p.rounds_.back();
    CHECK(r.name == R_RNG && r.combinations && r.draw_trits == r.combinations << 10 && r.priced == Priced::AS_SAID && r.combinations == rng_combinations(p.set, p.scope()));
    Plan by_hand(s);  // `small_columns` uses the round's `combinations`, not the setting's
    PlanRound &matrix = by_hand.close("a test's", RoundKind::BITS, ErrorKind::SMALL, Bound(Real::two_to(-140)), Priced::AS_SAID);
    matrix.draw_trits = 70, matrix.combinations = 7;
    CHECK(account(by_hand).small_columns == 10);
  }
  {  // m above 2^63: 2 m + 1 is computed without a bound on size, so the count keeps growing
    CodeSize c;
    c.log_n = 20, c.ell = 1 << 17, c.r = 101;
    const u64 huge = ((u64)1 << 63) + 2;
    const Proximity big = johnson(s, c, (u64)1 << 40), bigger = johnson(s, c, huge);
    CHECK(!big.bad() && !bigger.bad() && big.count.hi < bigger.count.lo && Real::two_to(300) < bigger.count.lo && bigger.eta.hi < big.eta.lo);
    GroupSize g;
    g.main = c, g.k = 4;
    Plan a(s), b(s);
    plan_fold(a, "", group_of_m(g, 50), 0), plan_fold(b, "", group_of_m(g, huge), 0);
    CHECK(a.finish()->holds() && !b.refused() && b.finish()->cause() == Cause::ABOVE_THE_TARGET);  // not `refused`: its proximity rounds miss the target
  }
}

// An element of E has 24 bytes, a root 32. SEND's seed is a point of the largest cube (10
// variables) and one pad factor: 11 elements.
static void test_cursor() {
  Plan p(setting_of_this_build());
  plan_send(p, "a", {SampleSize{10, no_sized_group()}, SampleSize{8, no_sized_group()}}, 1);
  PscSize x;
  x.degree.assign(2, 3);
  plan_psc(p, "b", x);
  plan_split(p, "c");
  plan_bat(p, "d", no_sized_group(), 5);
  plan_seal(p, "e");
  Cursor c(p);
  const PlanRound &seed = c.next_round(R_SEND_SEED);
  CHECK(seed.kind == RoundKind::SEED && seed.message_bytes == 32 && seed.roots == 1 && seed.answers == 0 && seed.draw_E == 11);
  const PlanRound &answers = c.next_round(R_SEND_ANSWER);  // the two answers have a query of their own
  CHECK(answers.kind == RoundKind::EMPTY && answers.message_bytes == 2 * 24 && answers.answers == 2 && !answers.work_bits && answers.error.zero());
  const PlanRound &eps = c.next_round(R_PSC_EPSILON);
  CHECK(eps.kind == RoundKind::EPSILON && eps.message_bytes == 24 && eps.answers == 0 && eps.roots == 0);
  CHECK(c.next_round(R_PSC_ROUND).message_bytes == 48 && c.next_round(R_PSC_ROUND).message_bytes == 48);
  CHECK(c.next_round(R_PSC_Y).message_bytes == 48);
  CHECK(c.next_round(R_PSC_VALUES).message_bytes == 2 * 24 && c.next_round(R_SPLIT).message_bytes == 24);  // two values; the split's target
  CHECK(c.next_round(R_BAT).message_bytes == 0);
  CHECK(!c.at_end() && c.next_round(R_SEAL).kind == RoundKind::SEAL && c.at_end());
}

// A round's kind follows from its name.
static void test_names() {
  RoundKind kind;
  CHECK(kind_of_round(R_CS, kind) && kind == RoundKind::POSITIONS && kind_of_round(R_REPLY, kind) && kind == RoundKind::EMPTY && !kind_of_round("a test's", kind));
  Plan p(setting_goldilocks());
  plan_send(p, "", {SampleSize{10, no_sized_group()}}, 1), plan_line(p, ""), plan_seal(p, "");
  CHECK(p.finish()->names_of_the_format());
  p.close("a test's", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING);
  CHECK(!p.finish()->names_of_the_format());
}

// A seal bears no proof of work (docs/format/FORMAT.md, R5.3). No public file sets a bit in the row
// ErrorKind::ZERO of `Setting::work`; if a test sets one, Plan::close() ends the process at a seal.
static void test_seal_without_work() {
  for (Scope scope : {Scope::REQUEST, Scope::SHARED, Scope::ONCE}) {
    Setting bent = setting_goldilocks();
    bent.set_work(ErrorKind::ZERO, scope, 1);
    CHECK(ends_with("a seal bears no proof of work", [&] { Plan p(bent); p.begin_scope(scope), plan_seal(p, ""); }));
    CHECK(runs_through([&] { Plan p(setting_goldilocks()); p.begin_scope(scope), plan_seal(p, ""); }));  // the control
  }
}

// `account`, mostly on plans written by hand; RNG's rounds come from plan_rng(). `groups`: rounds
// with the same name, work and printed error, largest error first, work counted; all(): bits +
// work. `lists`: the group with the largest error without work.
// compiled_level(w): the largest L with 2^-L >= 2^-w + c 2^(L - D); D = 256, the digests' bits;
// c = COLLISION_CONSTANT = 17/2 (both defined in blocks/numbers/accounting.h). The first check
// compares it at w = 128 with the closed form.
static void test_accounting() {
  CHECK(std::fabs(compiled_level(128.0) - (128.0 - std::log2((1.0 + std::sqrt(35.0)) / 2.0))) < 1e-9);  // x = 1 + (17/2) / x for x = 2^(128 - L): 126.21
  Setting s = setting_m61();
  s.set_work(ErrorKind::QUERY, Scope::REQUEST, 20);
  Plan p(s);
  p.push_repeat(4);
  p.close("a", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-108)), Priced::ON_THE_CALLERS_WORD);
  p.pop_repeat();
  p.close("b", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-130)), Priced::ON_THE_CALLERS_WORD);
  const Accounting a = account(p);
  CHECK(a.formed && a.holds && a.needs_work && a.worst == "a" && a.rounds == 5 && a.groups.size() == 2 && a.groups[0].name == "a" && a.groups[0].rounds == 4);
  CHECK(a.groups[0].all() == 128 && a.lists.bits == 108 && a.lists.name == "a");
  CHECK(std::fabs(a.sum - (-std::log2(4 * std::exp2(-128.0) + std::exp2(-130.0)))) < 1e-9 && std::fabs(a.sum_lists - 106) < 1e-5);
  PartSize z = p.finish()->size_of(0);
  CHECK(z.bytes == 4 * s.nonce_bytes);
  Plan q(s);
  q.close("c", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::add(Real::two_to(-128), Real::two_to(-300), Dir::UP)), Priced::ON_THE_CALLERS_WORD);
  CHECK(q.finish()->cause() == Cause::ABOVE_THE_TARGET && !account(q).holds);  // above the target by one part in 2^172
  {  /* Two groups that print the same (128.00) are sorted by their unrounded errors, whatever their
        names: the first contains the round named in `worst`. Equality passes: the rule is "at most
        2^-target", compared exactly (on upper bounds, rounded up), with no tolerance. */
    Plan t(s);
    t.push_repeat(4);
    t.close("an earlier name", RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound(Real::div(Real::two_to(-128), Real(1005), Dir::UP)) * Bound(1000), Priced::ON_THE_CALLERS_WORD);  // 2^-128.007
    t.pop_repeat();
    t.close("the later name", RoundKind::BITS, ErrorKind::SMALL, Bound(Real::two_to(-128)), Priced::ON_THE_CALLERS_WORD);
    const Accounting b = account(t);
    CHECK(b.holds && !b.needs_work && b.groups.size() == 2 && b.worst == "the later name" && b.groups[0].name == "the later name" && b.groups[0].all() == 128 && b.groups[0].rounds == 1);
    CHECK(b.groups[1].rounds == 4 && b.groups[1].all() > 128.006 && b.groups[1].all() < 128.008 && cut2(b.groups[1].all()) == cut2(b.groups[0].all()));
    CHECK(!b.small_columns);  // a round that draws no matrix has no columns
  }
  {  /* `needs_work`, true and false, and the rounds that the output names. The round with the
        largest error with work counted (x: 108 + 20 bits) is not the one with the largest error
        without work (y: 100 + 40), and neither is the plan's first or last round. */
    Setting w = setting_m61();
    w.set_work(ErrorKind::QUERY, Scope::REQUEST, 20), w.set_work(ErrorKind::SAMPLE, Scope::REQUEST, 40);
    Plan t(w);
    t.close("first", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-131)), Priced::ON_THE_CALLERS_WORD);
    t.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-108)), Priced::ON_THE_CALLERS_WORD);
    t.close("y", RoundKind::SEED, ErrorKind::SAMPLE, Bound(Real::two_to(-100)), Priced::ON_THE_CALLERS_WORD);
    t.close("last", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-129)), Priced::ON_THE_CALLERS_WORD);
    const Accounting b = account(t);
    CHECK(b.holds && b.needs_work && b.worst == "x" && b.groups.size() == 4 && b.groups[0].name == "x" && b.groups[0].all() == 128 && b.groups[0].work == 20);
    CHECK(b.lists.name == "y" && b.lists.bits == 100 && b.lists.work == 40 && b.groups[1].name == "last" && b.groups[3].name == "y");
    // Work that is not needed: every round meets the target without it, one of them exactly.
    Plan u(w);
    u.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-128)), Priced::ON_THE_CALLERS_WORD);
    u.close("y", RoundKind::SEED, ErrorKind::SAMPLE, Bound(Real::two_to(-140)), Priced::ON_THE_CALLERS_WORD);
    const Accounting c = account(u);
    CHECK(c.holds && !c.needs_work && c.lists.name == "x" && c.lists.bits == 128 && c.worst == "x" && c.groups[0].all() == 148);
    // Only the last of many rounds needs its work, by one part in 2^172.
    Plan v(w);
    v.push_repeat(1000);
    v.close("many", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-128)), Priced::ON_THE_CALLERS_WORD);
    v.pop_repeat();
    CHECK(account(v).holds && !account(v).needs_work);
    v.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::add(Real::two_to(-128), Real::two_to(-300), Dir::UP)), Priced::ON_THE_CALLERS_WORD);
    const Accounting d = account(v);
    CHECK(d.holds && d.needs_work && d.worst == "many");
    // Work that is not enough: the plan does not hold, and `needs_work` is still true.
    Plan n(w);
    n.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-107)), Priced::ON_THE_CALLERS_WORD);
    CHECK(n.finish()->cause() == Cause::ABOVE_THE_TARGET && !account(n).holds && account(n).needs_work);
  }
  {  // `small_union`: RNG's round under a union bound over its columns: N - log2(columns) bits. Of
     // several rounds, the one with the largest such error. It does not affect `holds`.
    Plan t(setting_m61());
    plan_rng(t, "few", 1 << 10);
    plan_rng(t, "many", 24576);
    plan_rng(t, "fewer", 3);
    const Accounting b = account(t);
    CHECK(b.holds && !b.needs_work && b.groups[0].all() == 128 && b.small_columns == 24576 && b.small_union.name == "RNG (many)" && b.small_union.work == 0);
    CHECK(std::fabs(b.small_union.bits - (128 - std::log2(24576.0))) < 1e-9);
    Setting w = setting_m61();
    w.set_work(ErrorKind::SMALL, Scope::REQUEST, 8), w.instances = 24;  // 120 + 5 combinations; the round's error is 24 x 2^-125
    Plan u(w);
    plan_rng(u, "with work", 1 << 12);
    const Accounting c = account(u);
    CHECK(c.holds && c.small_columns == 1 << 12 && c.small_union.work == 8 && std::fabs(c.small_union.bits - (125 - std::log2(24.0) - 12)) < 1e-9);
  }
  {  /* The counts, worked out by hand, for a plan of two parts with repeats. A query hashes the
        prover's message and 32 more bytes in blocks of 128, one hash call each. Its first call also
        gives four 64-bit words of the verifier's message, or none in a round with work; every eight
        further words cost a call. An element of E takes three words, a position one.
         part "one":  3 x (10 scalars: 240 bytes, 3 calls; 30 elements: 90 words, 86 beyond the four
                      free ones, 11 calls; paths: 7 calls) = 3 x 21 = 63
                      a round with an empty verifier's message: 1 call; not counted in `queries`
                      5 x (a proximity round on the theorem on codes; one element: 3 words, free) = 5
         part "two":  100 positions with 20 bits of work: 100 words, none free: 13 calls + 1 = 14 */
    Setting w = setting_m61();
    w.set_work(ErrorKind::QUERY, Scope::REQUEST, 20);
    Plan t(w);
    t.begin_part("one");
    t.push_repeat(3);
    t.send_scalars("scalars", 10);
    t.close("elements", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-140)), Priced::ON_THE_CALLERS_WORD).draw_E = 30;
    t.file("paths", MessageKind::PATHS, 320, 320.0, 7);
    t.pop_repeat();
    t.close_none("silent");
    t.push_repeat(5);
    {
      PlanRound &r = t.close("close", RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, Bound(Real::two_to(-135)), Priced::ON_THE_CALLERS_WORD);
      r.draw_E = 1, r.rests = CountSource::JOHNSON;
    }
    t.pop_repeat();
    t.begin_part("two");
    t.close("positions", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-110)), Priced::ON_THE_CALLERS_WORD).draw_positions = 100;
    const Accounting all = account(t), one = account(t, t.finish()->part_index("one")), two = account(t, t.finish()->part_index("two"));
    CHECK(all.holds && all.rounds == 10 && all.queries == 9 && all.closeness[1] == 5 && all.planned_calls == 63 + 1 + 5 + 14);
    CHECK(one.rounds == 9 && one.queries == 8 && one.closeness[1] == 5 && one.planned_calls == 69 && !one.needs_work);
    CHECK(two.rounds == 1 && two.queries == 1 && two.closeness[1] == 0 && two.planned_calls == 14 && two.needs_work);
    // A `refused` plan does not hold, whatever its rounds are like.
    Plan refused = t;
    refused.refuse("a test's");
    CHECK(!account(refused).holds && account(refused).refused == "a test's");
    // Rounds with the same name and work whose errors print the same form one group, with the
    // largest of their errors.
    Plan alike(w);
    const Bound base(Real::two_to(-130));
    alike.close("twins", RoundKind::ALGEBRAIC, ErrorKind::FIELD, base * Bound::ratio(Big(1000), Big(1003)), Priced::ON_THE_CALLERS_WORD);  // 2^-130.0043
    alike.close("twins", RoundKind::ALGEBRAIC, ErrorKind::FIELD, base * Bound::ratio(Big(1000), Big(1001)), Priced::ON_THE_CALLERS_WORD);  // 2^-130.0014
    alike.close("twins", RoundKind::ALGEBRAIC, ErrorKind::FIELD, base * Bound::ratio(Big(1000), Big(1002)), Priced::ON_THE_CALLERS_WORD);  // 2^-130.0029
    const Accounting g = account(alike);
    CHECK(g.groups.size() == 1 && g.groups[0].rounds == 3 && std::fabs(g.groups[0].bits - (130 + std::log2(1.001))) < 1e-9);
  }
  {  /* `near`: rounds within 0.01 bit of the target, work counted. `by_work`: rounds above the
        target without their work. Near: x (108 + 20, four times), RNG's matrix (128), a proximity
        round on THIRD and one on JOHNSON (128.006). Not near: a proximity round at 128.02, a field
        round at 129. */
    Setting w = setting_m61();
    w.set_work(ErrorKind::QUERY, Scope::REQUEST, 20);
    Plan t(w);
    t.push_repeat(4);
    t.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-108)), Priced::ON_THE_CALLERS_WORD);
    t.pop_repeat();
    plan_rng(t, "cells", 1 << 10);
    const Bound close = Bound(Real::div(Real::two_to(-128), Real(1000), Dir::UP)) * Bound(996), far = Bound(Real::div(Real::two_to(-128), Real(1000), Dir::UP)) * Bound(986);  // 2^-128.0058, 2^-128.0203
    t.close("third", RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, close, Priced::ON_THE_CALLERS_WORD).rests = CountSource::THIRD;
    t.close("johnson", RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, close, Priced::ON_THE_CALLERS_WORD).rests = CountSource::JOHNSON;
    t.close("johnson, further off", RoundKind::ALGEBRAIC, ErrorKind::CLOSENESS, far, Priced::ON_THE_CALLERS_WORD).rests = CountSource::JOHNSON;
    t.close("field", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-129)), Priced::SETTING_ALONE);
    t.close("nothing", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::ON_THE_CALLERS_WORD);  // error 0: counted in `on_a_word`, not in `near` or `by_work`
    const Accounting b = account(t);
    CHECK(b.holds && b.needs_work && b.by_work == 4 && b.near == 7 && b.near_small == 1);
    CHECK(b.near_closeness[0] == 1 && b.near_closeness[1] == 1 && b.closeness[1] == 2);
    // Not in `on_a_word`: RNG's round (AS_SAID) and the round "field" (SETTING_ALONE).
    CHECK(b.on_a_word == 8);  // x four times, 3 proximity rounds, the round with error 0
    // A round above the target is not in `near`; its error without the work is above the target
    // too, so `by_work` counts it.
    Plan above(w);
    above.close("x", RoundKind::POSITIONS, ErrorKind::QUERY, Bound(Real::two_to(-107)), Priced::ON_THE_CALLERS_WORD);
    above.close("far", RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound(Real::two_to(-40)), Priced::ON_THE_CALLERS_WORD);
    const Accounting n = account(above);
    CHECK(!n.holds && n.near == 0 && n.by_work == 2);
  }
}

// Trees that may be opened at only one list of positions; rounds with an empty verifier's message;
// the cap on draws.
static void test_one_list_trees_and_draw_cap() {
  TreeSize tree;
  tree.name = "tree", tree.log_n = 10, tree.leaf_bytes = 124;
  {
    Plan p(setting_goldilocks());
    plan_open(p, "", tree, 7), plan_open(p, "", tree, 7);  // a tree of sampled rows may be read at several lists
    CHECK(!p.refused() && p.finish()->size_of(0).bytes == 2 * (7 * (124 + 40) + path_digests_most(10, 7) * 32));
  }
  {  // A tree with the mask of one step (`one_list`) may be opened at one list only, in the chain
     // that sent it. Copies of a marked TreeSize are the same tree. Each Plan counts by itself.
    TreeSize rider = tree, other = tree;
    rider.one_list = true, other.one_list = true;
    const std::vector<TreeSize> kept(2, rider);
    Plan once(setting_goldilocks()), twice(setting_goldilocks()), elsewhere(setting_goldilocks());
    plan_open(once, "a", kept[0], 7), plan_open(once, "a", other, 7);
    CHECK(!once.refused());
    plan_open(twice, "a", kept[0], 7);
    CHECK(!twice.refused());
    plan_open(twice, "b", kept[1], 7);
    CHECK(twice.refused() && twice.why() == "b: the tree tree holds the mask of one step and is opened at a second list");
    Plan many(setting_goldilocks());  // like a proximity test's stages: each mark makes a new tree
    for (int i = 0; i < 1000; ++i) {
      TreeSize stage = tree;
      stage.one_list = true;
      plan_open(many, "s", stage, 7);
    }
    CHECK(!many.refused());
    TreeSize adopted = tree;
    adopted.one_list = true, adopted.from_another_chain = true;
    plan_open(elsewhere, "c", adopted, 7);
    CHECK(elsewhere.refused() && elsewhere.why() == "c: the tree tree holds the mask of one step and is read in another chain");
  }
  // Planned's queries() counts a round with an empty verifier's message (plan_none());
  // `Accounting::queries` does not.
  {
    const Setting s = setting_of_this_build();
    Plan p(s);
    plan_scalars(p, "values", "", 2);
    plan_none(p, R_PSC_VALUES, "");
    plan_scalars(p, "more", "", 1);
    plan_line(p, "");
    CHECK(p.finish()->rounds() == 2 && p.finish()->queries() == 2 && p.finish()->holds());
    CHECK(p.rounds_[0].message_bytes == 48 && p.rounds_[1].message_bytes == 24);
    const Planned done = p.finish();
    Cursor walk(done);
    CHECK(&walk.next_round(R_PSC_VALUES) == &done->rounds_[0] && &walk.next_round(R_LINE) == &done->rounds_[1] && walk.at_end());
    Plan q(s);  // a root may not be in a round with an empty verifier's message
    plan_roots(q, "", 1);
    plan_none(q, R_SPLIT, "");
    CHECK(q.refused());
  }
  // capped(): a bound on the chance that, for some coordinate of a drawn element of E, six words in
  // a row (DRAW_TRIES; of each, as many bits as p has) are not below p; the coordinate is then 0.
  for (const Setting &s : {setting_goldilocks(), setting_m61()}) {
    Plan p(s);
    const double bits = -bits_of(p.finish()->capped());  // three coordinates, six words each: 3 (2^-32)^6, 3 (2^-61)^6
    CHECK(std::fabs(bits + (s.circle ? 366 : 192) - std::log2(3.0)) < 1e-6);
    plan_send(p, "", {SampleSize{20, no_sized_group()}}, 1);
    CHECK(p.rounds_[0].draw_E == 21 && p.rounds_[0].draw_nonzero == 1 && most_words(p.rounds_[0]) == 18 * 22);
    CHECK(p.rounds_[0].error < p.finish()->whole_error(p.rounds_[0], p.finish()->capped()));
    PlanRound positions;
    positions.kind = RoundKind::POSITIONS, positions.draw_positions = 219, positions.draw_E = 1;
    PlanRound trits;
    trits.kind = RoundKind::BITS, trits.draw_trits = 65;
    PlanRound seal;
    seal.kind = RoundKind::SEAL;
    CHECK(most_words(positions) == 219 + 18 && most_words(trits) == 3 && most_words(seal) == 4);
  }
}

int main() {
  test_one_list_trees_and_draw_cap();
  test_big();
  test_real();
  test_positions();
  test_known();
  test_circle_dimension();
  test_paths();
  test_counts();
  test_stages();
  test_refusals();
  test_table();
  test_analysis();
  test_edges();
  test_one_distance();
  test_prices();
  test_words();
  test_cursor();
  test_names();
  test_seal_without_work();
  test_accounting();
  if (failures) std::printf("%d of %d checks FAILED\n", failures, checks);
  else std::printf("ok, %d checks\n", checks);
  return failures ? 1 : 0;
}
