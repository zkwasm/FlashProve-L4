// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Do the scalars sent by PSC (blocks/iop/psc.h), the sumcheck with a hiding polynomial
// (`hp` in identifiers), depend on the witness? An exhaustive count over small fields
// (histogram()), with negative controls.
//
// What it shows, for the shapes and fields in main(), the witnesses taken and the lists of the
// verifier's challenges (`coins` in code and output) that were run: given uniform mask entries (the
// prover's random values) used nowhere else, the scalars that PSC sends are uniform. One PSC is run
// under all lists that the verifier can draw; the chain PSC, LINE, PSC (not a hash chain; LINE
// joins two claims into one) under one list (24 with `long`): nothing is shown about the others.
// It shows nothing about the protocol's large field as such, other shapes or other witnesses;
// nothing about what the rest of the protocol reveals of the mask entries (the out-of-domain sample
// of the oracle message that contains them, the opened positions of their codewords, the later
// proof of the linear claims that PSC outputs); nothing about running time or memory. It checks the
// program against the hiding property; it does not prove the property.
// Arguments: `long`: more witnesses, targets and lists; `-v`: a line for each shape and control.
#include <cstring>
#include <deque>

#include "toy_support.h"

using namespace blk;

static int failures = 0;
static unsigned long long provers = 0;
static bool long_run = false, verbose = false;
static unsigned shapes = 0, controls = 0, noticed = 0;
static size_t chain_lists = 0;
#define SAY(...)                         \
  do {                                   \
    if (verbose) std::printf(__VA_ARGS__); \
  } while (0)

static u64 power(u64 b, size_t e) {
  u64 r = 1;
  while (e--) r *= b;
  return r;
}
template <class K>
std::vector<K> digits(u64 n, size_t count) {
  std::vector<K> v(count);
  for (K& x : v) x = K::from(n % K::size), n /= K::size;
  return v;
}

// How the mask row is built from the enumerated entries. The honest filling is the identity.
template <class K>
struct Filling {
  const char* name;
  size_t free;                                                  // entries enumerated
  std::function<std::vector<K>(const std::vector<K>&)> row;     // -> the whole mask row
};

// The prover under test, for one witness and one list of challenges: mask row -> scalars sent.
template <class K>
using Tape = std::function<std::vector<K>(const std::vector<K>& mask_row)>;

// Runs `tape` once for every value of the enumerated entries and counts how often each tuple of
// sent scalars occurs. Hiding asks that the counts are the same for all witnesses. With the honest
// filling there are as many scalars as mask entries, and the tests ask for more: every tuple occurs
// exactly once, that is, the messages are uniform.
template <class K>
std::vector<u32> histogram(const Tape<K>& tape, const Filling<K>& filling, size_t scalars) {
  std::vector<u32> h(power(K::size, scalars), 0);
  const u64 all = power(K::size, filling.free);
  std::vector<K> entries(filling.free, K::zero());
  for (u64 i = 0; i < all; ++i) {
    if (i)  // count up
      for (K& x : entries) {
        x = x + K::one();
        if (!(x == K::zero())) break;
      }
    const std::vector<K> t = tape(filling.row(entries));
    BLK_REQUIRE(t.size() == scalars, "histogram: number of scalars");
    u64 code = 0;
    for (const K& x : t) code = code * K::size + x.v;
    ++h[code];
    ++provers;
  }
  return h;
}

// ---- One PSC ----
// A public statement without its target (the claimed sum). `Weight` (blocks/iop/hp.h) defines how a
// variable is summed over. `Summand` (blocks/iop/summand.h) has shown maps, whose values at the
// challenge point (end values) PSC sends masked, and kept maps, whose values are not sent. A
// witness is the maps' tables, concatenated, shown maps first.
template <class K>
struct Statement {
  std::string name;
  std::vector<Weight<K>> vars;
  Summand<K> S;
  unsigned maps;        // shown + kept
  unsigned table_bits;  // each map is the multilinear extension of a table of 2^table_bits entries
};

template <class K>
std::vector<ToyMap<K>> maps_of(const Statement<K>& st, const std::vector<K>& witness) {
  std::vector<ToyMap<K>> m;
  const size_t n = size_t(1) << st.table_bits;
  for (unsigned i = 0; i < st.maps; ++i)
    m.push_back(multilinear(std::vector<K>(witness.begin() + i * n, witness.begin() + (i + 1) * n)));
  return m;
}
template <class K>
K sum_of(const Statement<K>& st, const std::vector<K>& witness) {
  return DefinitionSumcheck<K>(st.vars, st.S, maps_of(st, witness)).sum();
}

// `coins`: eps (PSC sums the hiding polynomial plus eps times the summand), one challenge gamma_j
// for each variable and, if a map is shown, beta, the challenge of the further variable Y that PSC
// adds to mask the end values. `inner`, the sumcheck without masks, is the same for every mask
// row: it is run once and replayed.
template <class K>
Tape<K> psc_tape(const Statement<K>& st, const std::vector<K>& witness, const std::vector<K>& coins, K target) {
  auto replay = std::make_shared<ReplaySumcheck<K>>();
  DefinitionSumcheck<K> inner(st.vars, st.S, maps_of(st, witness));
  for (size_t j = 0; j < st.vars.size(); ++j) {
    replay->rounds.push_back(inner.round());
    inner.bind(coins[1 + j]);
  }
  replay->ends = inner.end_values();
  return [st, replay, coins, target](const std::vector<K>& row) {
    replay->at = 0;
    MaskLedger ledger(SendAccess::make(0, 0, u32(row.size())));
    ToyProverChannel<K> ch(listed_coins(coins), false);
    SumClaim<K> claim;
    claim.vars = st.vars;
    claim.target = target;
    psc_prove(ch, std::move(claim), st.S, *replay, ledger.take(row.size()), mask_row(row));
    return ch.tape;
  };
}

// Up to `count` witnesses whose sum is `target`, spread evenly over all of them in counting order.
// `count` must be at least 2 (not checked).
template <class K>
std::vector<std::vector<K>> witnesses_with_sum(const Statement<K>& st, K target, size_t count) {
  const size_t n = size_t(st.maps) << st.table_bits;
  std::vector<std::vector<K>> all, some;
  for (u64 i = 0; i < power(K::size, n); ++i) {
    std::vector<K> w = digits<K>(i, n);
    if (sum_of(st, w) == target) all.push_back(w);
  }
  if (all.size() <= count) return all;
  for (size_t i = 0; i < count; ++i) some.push_back(all[i * (all.size() - 1) / (count - 1)]);
  return some;
}

// Calls f with every list of `plain` challenges, each extended, if with_nonzero, by every non-zero
// value in turn.
template <class K>
void all_coins(size_t plain, bool with_nonzero, const std::function<void(const std::vector<K>&)>& f) {
  for (u64 i = 0; i < power(K::size, plain); ++i) {
    std::vector<K> c = digits<K>(i, plain);
    if (!with_nonzero) {
      f(c);
      continue;
    }
    c.push_back(K::zero());
    for (u64 b = 1; b < K::size; ++b) c.back() = K::from(b), f(c);
  }
}

// Runs one PSC under all lists of challenges, so a passing run also covers a verifier that chooses
// each challenge after seeing the messages so far: the count of a tuple under such a verifier is
// its count under the list that the tuple itself leads to.
template <class K>
void test_statement(const Statement<K>& st, size_t witnesses_each, u64 targets) {
  SumClaim<K> shape;
  shape.vars = st.vars;
  const size_t n = psc_masks(shape, st.S), w_entries = n - st.S.shown;
  const Filling<K> honest{"honest", n, [](const std::vector<K>& e) { return e; }};
  unsigned long long counts = 0;
  bool uniform = true;
  for (u64 t = 0; t < targets; ++t) {
    const K target = K::from(t);
    const auto ws = witnesses_with_sum(st, target, witnesses_each);
    if (ws.size() < 2) continue;  // nothing to compare
    all_coins<K>(1 + st.vars.size(), st.S.shown > 0, [&](const std::vector<K>& coins) {
      for (const auto& w : ws) {
        for (u32 c : histogram(psc_tape(st, w, coins, target), honest, n)) uniform = uniform && c == 1;
        ++counts;
      }
    });
  }
  if (!uniform || !counts) ++failures;
  ++shapes;
  SAY("  %-58s |K| = %llu, %zu masks: %s (%llu counts)\n", st.name.c_str(), (unsigned long long)K::size, n,
      uniform && counts ? "uniform for every witness taken and all coins" : "NOT UNIFORM: FAIL", counts);

  // Controls: with spoilt masks, some two witnesses must be told apart for some list of challenges;
  // otherwise the test would be blind. The last control instead sets beta to 0, which PSC excludes.
  // The mask row: the hiding polynomial's coefficients in hp_place() order, then one inner mask for
  // each shown map; beta times it is added to the map's end value.
  std::vector<Filling<K>> spoilt;
  const std::vector<Weight<K>> vars = psc_vars(shape, st.S);
  // Each coefficient of the hiding polynomial in turn is fixed to zero. Without the linear
  // coefficient of Y, the linear coefficient of the last round polynomial is eps times a
  // combination of the inner masks whose coefficients depend on the witness. If Y is the only
  // variable, the constant is the hiding polynomial's sum and enters no other message: fixing it
  // reveals nothing, so that shape has no such control.
  std::deque<std::string> names;  // (a Filling keeps a pointer)
  for (unsigned l = 0; l < vars.size(); ++l)
    for (unsigned t = l + 1 == vars.size() ? 0 : 1; t < vars[l].d; ++t) {
      if (vars.size() == 1 && t == 0) continue;
      names.push_back("hiding polynomial: coefficient " + std::to_string(t) + " of variable " + std::to_string(l) + " = 0");
      const size_t place = hp_place(vars, l, t);
      spoilt.push_back({names.back().c_str(), n - 1, [place](std::vector<K> e) {
                          e.insert(e.begin() + place, K::zero());
                          return e;
                        }});
    }
  if (st.S.shown >= 1)
    spoilt.push_back({"no inner masks", w_entries, [&](std::vector<K> e) {
                        e.resize(n, K::zero());
                        return e;
                      }});
  if (st.S.shown >= 2)
    spoilt.push_back({"one inner mask for two end values", n - 1, [](std::vector<K> e) {
                        e.push_back(e.back());
                        return e;
                      }});
  const K target = K::zero();
  const auto ws = witnesses_with_sum(st, target, 6);
  BLK_REQUIRE(ws.size() >= 2, "controls: two witnesses are needed");
  auto told_apart = [&](const Filling<K>& f, bool beta_zero) {
    bool apart = false;
    auto with = [&](const std::vector<K>& coins) {
      if (apart) return;
      const auto h0 = histogram(psc_tape(st, ws[0], coins, target), f, n);
      for (size_t i = 1; i < ws.size() && !apart; ++i) apart = histogram(psc_tape(st, ws[i], coins, target), f, n) != h0;
    };
    if (beta_zero)
      all_coins<K>(1 + st.vars.size(), false, [&](std::vector<K> c) { c.push_back(K::zero()), with(c); });
    else
      all_coins<K>(1 + st.vars.size(), st.S.shown > 0, with);
    return apart;
  };
  for (const Filling<K>& f : spoilt) {
    const bool apart = told_apart(f, false);
    if (!apart) ++failures;
    ++controls, noticed += apart;
    SAY("      control, %-46s %s\n", f.name, apart ? "witnesses told apart, as it must be" : "NOT NOTICED: FAIL");
  }
  if (st.S.shown) {
    const bool apart = told_apart(honest, true);
    if (!apart) ++failures;
    ++controls, noticed += apart;
    SAY("      control, %-46s %s\n", "the further variable's challenge = 0", apart ? "witnesses told apart, as it must be" : "NOT NOTICED: FAIL");
  }
}

// ---- A chain: PSC, LINE, PSC on a product tree with four leaves ----
// Public: the product of the leaves. The first PSC sends masked values of the two halves of the
// middle level, which is never sent; LINE (blocks/iop/line.h) joins the two claims; the second PSC
// takes the joined claim, which still contains the first PSC's inner masks, down to the leaves.
// `coins`: eps, beta (first PSC); rho (LINE); eps, gamma, beta (second PSC).
template <class K>
Tape<K> chain_tape(const std::vector<K>& leaves, const std::vector<K>& coins, K root) {
  const Summand<K> S{2, 0, 2, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[1]}; }};
  auto top = std::make_shared<ReplaySumcheck<K>>(), low = std::make_shared<ReplaySumcheck<K>>();
  top->ends = {leaves[0] * leaves[2], leaves[1] * leaves[3]};
  {
    std::vector<Weight<K>> vars = {Weight<K>::equality(coins[2], 3)};
    DefinitionSumcheck<K> inner(vars, S, {multilinear<K>({leaves[0], leaves[1]}), multilinear<K>({leaves[2], leaves[3]})});
    low->rounds.push_back(inner.round());
    inner.bind(coins[4]);
    low->ends = inner.end_values();
  }
  return [S, top, low, coins, root](const std::vector<K>& row) {
    low->at = 0;
    MaskLedger ledger(SendAccess::make(0, 0, u32(row.size())));
    ToyProverChannel<K> ch(listed_coins(coins), false);
    SumClaim<K> first;
    first.target = root;
    PscOutput<K> out = psc_prove(ch, std::move(first), S, *top, ledger.take(5), mask_row(row));
    PointClaim<K> joined = line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
    psc_prove(ch, SumClaim<K>::at_point(std::move(joined), {3}), S, *low, ledger.take(7), mask_row(row));
    return ch.tape;
  };
}

// `witnesses_each` must be at least 2 (not checked).
template <class K>
void test_chain(size_t coin_lists, size_t witnesses_each) {
  const size_t n = 12;
  const Filling<K> honest{"honest", n, [](const std::vector<K>& e) { return e; }};
  // The inner masks are entries 3, 4 (first PSC) and 10, 11 (second). Control: entry 5, the second
  // hiding polynomial's first, repeats entry 3, so that one entry masks two things.
  const Filling<K> reused{"reused", n - 1, [](std::vector<K> e) {
                            e.insert(e.begin() + 5, e[3]);
                            return e;
                          }};
  TestRng rng(11);
  bool uniform = true, apart = false;
  unsigned long long counts = 0;
  for (size_t i = 0; i < coin_lists; ++i) {
    const std::vector<K> coins = {rng.element<K>(), rng.nonzero<K>(), rng.element<K>(),
                                  rng.element<K>(), rng.element<K>(), rng.nonzero<K>()};
    const K root = K::from((i + 1) % K::size);
    std::vector<std::vector<K>> all, ws;  // witnesses spread evenly over all with this product
    for (u64 j = 0; j < power(K::size, 4); ++j) {
      std::vector<K> w = digits<K>(j, 4);
      if (w[0] * w[1] * w[2] * w[3] == root) all.push_back(w);
    }
    for (size_t j = 0; j < witnesses_each; ++j) ws.push_back(all[j * (all.size() - 1) / (witnesses_each - 1)]);
    for (const auto& w : ws) {
      for (u32 c : histogram(chain_tape(w, coins, root), honest, n)) uniform = uniform && c == 1;
      ++counts;
    }
    if (!apart) {  // the control: go through the witnesses until two differ
      const auto h0 = histogram(chain_tape(all[0], coins, root), reused, n);
      for (size_t j = 1; j < all.size() && !apart; ++j) apart = histogram(chain_tape(all[j], coins, root), reused, n) != h0;
    }
  }
  if (!uniform || !apart) ++failures;
  ++shapes, ++controls, noticed += apart;
  chain_lists = coin_lists;
  SAY("  %-58s |K| = %llu, %zu masks: %s (%llu counts)\n", "chain PSC, LINE, PSC on a product tree of four leaves",
      (unsigned long long)K::size, n, uniform ? "uniform for every witness taken, for the lists of coins run" : "NOT UNIFORM: FAIL", counts);
  SAY("      control, %-46s %s\n", "one entry used by both sum-checks", apart ? "witnesses told apart, as it must be" : "NOT NOTICED: FAIL");
}

template <class K>
Summand<K> product() {
  return {2, 0, 2, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[1]}; }};
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    long_run = long_run || !std::strcmp(argv[i], "long");
    verbose = verbose || !std::strcmp(argv[i], "-v");
  }
  // The test uses small fields of its own and no hash.
  const size_t few = long_run ? 4 : 2, some = long_run ? 6 : 3;
  {
    using K = Fp<3>;
    test_statement<K>({"product of two maps, one variable, equality weights", {Weight<K>::equality(K::from(2), 3)}, product<K>(), 2, 1}, some, 3);
    test_statement<K>({"product of two maps, one variable, plain weights", {Weight<K>::plain(3)}, product<K>(), 2, 1}, some, 3);
    test_statement<K>({"product of two maps, two variables, equality weights",
                       {Weight<K>::equality(K::from(2), 3), Weight<K>::equality(K::from(1), 3)}, product<K>(), 2, 2}, few, long_run ? 3 : 1);
    test_statement<K>({"product with one map kept, two variables",
                       {Weight<K>::equality(K::from(2), 3), Weight<K>::equality(K::from(0), 3)},
                       {1, 1, 1, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{K::zero(), y[0]}; }}, 2, 2}, some, 3);
    test_statement<K>({"affine in both maps, plain weights, nothing shown",
                       {Weight<K>::plain(3), Weight<K>::plain(3)},
                       {0, 2, 0, [](const std::vector<K>& x, const std::vector<K>&) {
                          return std::vector<K>{K::zero(), x[0] + K::one(), x[1] + x[1] + K::one()};
                        }}, 2, 2}, some, 3);
    test_chain<K>(long_run ? 24 : 1, long_run ? 3 : 2);
  }
  {
    using K = Fp<5>;
    test_statement<K>({"product of two values, no variable", {}, product<K>(), 2, 0}, some, 5);
    test_statement<K>({"degree three in one value, no variable", {},
                       {1, 0, 3, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[0] * y[0] + y[0]}; }}, 1, 0}, some, 5);
  }
  if (failures)
    std::printf("psc_hiding_test: FAILED (%d of %u shapes and %u controls went wrong; run with -v to see which)\n", failures, shapes, controls);
  else
    std::printf("psc_hiding_test: ok (%u shapes: the sent scalars are uniform for every witness taken, under all verifier's coins, the "
                "chain under %zu list(s) of coins; all %u runs with spoilt masks told witnesses apart, as they must; %llu runs of the prover)\n",
                shapes, chain_lists, noticed, provers);
  return failures ? 1 : 0;
}
