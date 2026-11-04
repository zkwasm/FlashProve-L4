// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests PSC, the sumcheck with a hiding polynomial (blocks/iop/psc.h), and LINE, which merges two
// claims on a map's halves into one (blocks/iop/line.h). Every run is over `K`, the integers mod
// 2^61 - 1 (`Fp`, a test field): none uses the library's fields (blocks/field/field.h). A map is a
// witness polynomial. PSC's output claims are called rows in names and messages: (E_j) on each
// shown map, whose masked end value the prover sends, and (U), with the kept maps as terms. Nothing
// is committed or hashed: the test evaluates the claims on its maps and mask row (random masks).
//
// The test runs honest provers on cases() and product trees. The verifier must accept, the
// output claims must hold, and both sides must log the same events (for the first case, a fixed
// list) and, in run_case(), compute the same claims. A PSC must send 1 + sum_j (d_j - 1) + `degree`
// + `shown` scalars and take as many mask entries (d_j: `Weight::d`, coefficients in variable j).
//
// Every run here is honest. Malicious provers belong to a test that is not part of this repository,
// where every run must be rejected: by psc_verify(), by scalars left unread, or by a failing output
// claim. (Most fail a claim: psc_verify() uses its checks to fill in the coefficients not sent.)
// Exception: one run there, where the challenge eps is the prover's guess, must be accepted.
#include "toy_support.h"

using namespace blk;
using K = Fp<(u64(1) << 61) - 1>;

static int failures = 0;
static int runs = 0;
static bool verbose = false;  // set by any argument; nothing reads it
[[maybe_unused]] static int other_books = 0;  // nothing sets or reads it

struct Case {
  std::string name;
  std::vector<Weight<K>> vars;
  Summand<K> S;
  std::vector<ToyMap<K>> maps;      // shown first, then kept
  unsigned incoming = 0;            // mask entries in the claim's own mask form
  std::vector<Weight<K>> true_vars; // if not empty: the weights with the number of coefficients that the summand really has
};

// A `Summand` (blocks/iop/summand.h) is initialized with `shown`, `kept`, `degree`, `at`.
static Summand<K> product_of_two() {
  return {2, 0, 2, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] * y[1]}; }};
}

static std::vector<Case> cases(TestRng& rng) {
  std::vector<Case> all;
  auto table = [&](unsigned k) { return multilinear(rng.vector<K>(size_t(1) << k)); };
  auto equality = [&](unsigned k, unsigned d) {
    std::vector<Weight<K>> v;
    for (unsigned j = 0; j < k; ++j) v.push_back(Weight<K>::equality(rng.element<K>(), d));
    return v;
  };
  all.push_back({"product of two maps, equality weights", equality(3, 3), product_of_two(), {table(3), table(3)}, 0, {}});
  all.push_back({"the same with a mask form in the claim", equality(2, 3), product_of_two(), {table(2), table(2)}, 2, {}});
  all.push_back({"no variable: a product of two values", {}, product_of_two(), {table(0), table(0)}, 1, {}});
  {
    const std::vector<K> z = rng.vector<K>(3);
    Summand<K> S{2, 0, 2, [z](const std::vector<K>& x, const std::vector<K>& y) {
                   return std::vector<K>{eq_at(z, x) * y[0] * y[1]};
                 }};
    all.push_back({"plain weights, the equality factor inside the summand",
                   std::vector<Weight<K>>(3, Weight<K>::plain(4)), S, {table(3), table(3)}, 1, {}});
  }
  {
    Summand<K> S{3, 0, 3, [](const std::vector<K>& x, const std::vector<K>& y) {
                   return std::vector<K>{y[0] * y[1] * y[2] - y[2] + x[0] * y[0]};
                 }};
    all.push_back({"degree three with a public coefficient", equality(2, 4), S, {table(2), table(2), table(2)}, 0, {}});
  }
  {
    Summand<K> S{1, 1, 1, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{K::zero(), y[0]}; }};
    all.push_back({"product with one map kept", equality(3, 3), S, {table(3), table(3)}, 1, {}});
  }
  {
    const std::vector<K> p0 = rng.vector<K>(2), p1 = rng.vector<K>(2), chi = rng.vector<K>(2);
    Summand<K> S{0, 2, 0, [=](const std::vector<K>& x, const std::vector<K>&) {
                   return std::vector<K>{K::zero(), chi[0] * eq_at(p0, x), chi[1] * eq_at(p1, x)};
                 }};
    all.push_back({"affine in all maps: nothing shown, no further variable",
                   std::vector<Weight<K>>(2, Weight<K>::plain(3)), S, {table(2), table(2)}, 2, {}});
  }
  {
    const std::vector<K> D = {K::from(0), K::from(1), K::from(2), K::from(3)};
    std::vector<Weight<K>> sum_over_D = {Weight<K>::domain(D, 7), Weight<K>::equality(rng.element<K>(), 3)};
    std::vector<Weight<K>> value_on_D = {Weight<K>::lagrange(D, rng.element<K>(), 7), Weight<K>::equality(rng.element<K>(), 3)};
    auto m = [&] { return lagrange_then_multilinear(rng.vector<K>(8), D); };
    all.push_back({"first variable over a set of four points, summed", sum_over_D, product_of_two(), {m(), m()}, 0, {}});
    all.push_back({"first variable over a set of four points, Lagrange weights", value_on_D, product_of_two(), {m(), m()}, 1, {}});
  }
  {
    const std::vector<K> D = {K::from(5), K::from(9), K::from(2)};
    auto m = [&](unsigned rest) { return lagrange_then_multilinear(rng.vector<K>(D.size() << rest), D); };
    all.push_back({"a sum over three points and a plain sum, with a mask form in the claim",
                   {Weight<K>::domain(D, 5), Weight<K>::plain(3)}, product_of_two(), {m(1), m(1)}, 1, {}});
    Summand<K> kept_only{0, 1, 0, [](const std::vector<K>& x, const std::vector<K>&) {
                           return std::vector<K>{x[0] + K::from(7), x[1]};
                         }};
    all.push_back({"nothing shown, Lagrange weights and a plain sum, a term without a map",
                   {Weight<K>::lagrange(D, rng.element<K>(), 3), Weight<K>::plain(3)}, kept_only, {m(1)}, 1, {}});
  }
  {
    Summand<K> S{1, 1, 1, [](const std::vector<K>&, const std::vector<K>& y) { return std::vector<K>{y[0] + K::from(3), y[0]}; }};
    all.push_back({"no variable, one value shown and one kept", {}, S, {table(0), table(0)}, 1, {}});
  }
  {
    Summand<K> S{2, 2, 2, [](const std::vector<K>& x, const std::vector<K>& y) {
                   return std::vector<K>{x[0] * y[0] * y[1] + K::from(5), y[0], x[1] + K::from(2)};
                 }};
    const std::vector<ToyMap<K>> four = {table(2), table(2), table(2), table(2)};
    all.push_back({"two shown and two kept, plain sums", {Weight<K>::plain(4), Weight<K>::plain(3)}, S, four, 2, {}});
    all.push_back({"the same, a plain sum and equality weights at 0", {Weight<K>::plain(4), Weight<K>::equality(K::zero(), 3)}, S, four, 2, {}});
    all.push_back({"the same, equality weights at 1 and a plain sum", {Weight<K>::equality(K::one(), 4), Weight<K>::plain(3)}, S, four, 1, {}});
  }
  return all;
}

struct Outcome {
  bool accepted;
  std::string why;  // the check that rejected
};

// One run of PSC. target_shift: added to the true target (a false claim if not zero).
// coins: the verifier's challenges, if the test chooses them.
static Outcome run_case(const Case& c, u64 seed, K target_shift, const std::vector<K>* coins = nullptr,
                        [[maybe_unused]] const std::vector<std::string>* expected_log = nullptr) {
  TestRng rng(seed);
  SumClaim<K> shape;
  shape.vars = c.vars;
  const size_t fresh = psc_masks(shape, c.S);
  {  // `psc_size` (a plan's sizes) must count the same scalars: the sum, d - 1 a round, the values
    const PscSize size = psc_size(shape, c.S);
    size_t planned = 1;
    for (unsigned d : size.degree) planned += d - 1;
    if (size.with_Y) planned += size.degree_Y + size.values;
    BLK_REQUIRE(planned == fresh && size.with_Y == (c.S.shown > 0), "psc_test: psc_size and psc_masks count different scalars");
  }
  const std::vector<K> row = rng.vector<K>(c.incoming + fresh);  // the "committed" mask row
  const MaskValues<K> masks = mask_row(row);
  const std::vector<K> form = rng.vector<K>(c.incoming);

  // The claim is built twice, by the same public steps: each side has its own `MaskLedger`
  // (blocks/iop/mask.h), which gives out each mask entry once.
  auto make_claim = [&](MaskLedger& ledger, K target) {
    SumClaim<K> claim;
    claim.vars = c.vars;
    for (unsigned i = 0; i < c.incoming; ++i) claim.masks.push_back({ledger.take(), form[i]});
    claim.target = target;
    return claim;
  };
  const std::vector<Weight<K>>& true_vars = c.true_vars.empty() ? c.vars : c.true_vars;
  K target = DefinitionSumcheck<K>(true_vars, c.S, c.maps).sum() + target_shift;
  for (unsigned i = 0; i < c.incoming; ++i) target = target + form[i] * row[i];

  auto coin_source = [&] { return coins ? listed_coins(*coins) : random_coins<K>(seed ^ 0xC01); };
  MaskLedger pl(SendAccess::make(0, 0, u32(row.size())));
  ToyProverChannel<K> pch(coin_source());
  DefinitionSumcheck<K> inner(true_vars, c.S, c.maps);
  SumClaim<K> pclaim = make_claim(pl, target);
  PscOutput<K> pout = psc_prove(pch, std::move(pclaim), c.S, inner, pl.take(fresh), masks);

  MaskLedger vl(SendAccess::make(0, 0, u32(row.size())));
  ToyVerifierChannel<K> vch(pch.tape, coin_source());
  SumClaim<K> vclaim = make_claim(vl, target);
  PscOutput<K> vout;
  ++runs;
  if (!psc_verify(vch, std::move(vclaim), c.S, vl.take(fresh), vout)) return {false, "the verifier routine"};
  if (!vch.at_end()) return {false, "scalars left unread"};
  const std::string bad = failing_row(vout, c.S, c.maps, masks);
  if (!bad.empty()) return {false, "row " + bad};
  if (!same_output(pout, vout)) return {false, "TEST: the two sides computed different rows"};
  if (pch.log != vch.log) return {false, "TEST: the two sides saw different orders of events"};
  size_t count = 1 + c.S.degree + c.S.shown;
  for (const Weight<K>& v : c.vars) count += v.d - 1;
  if (pch.tape.size() != count || fresh != count) return {false, "TEST: the number of scalars or of masks"};
  for (size_t i = 0; i < count - c.S.shown; ++i)
    if (vout.row_masks[i].slot.index() != c.incoming + i) return {false, "TEST: the hiding polynomial's entries come first, in order"};
  for (size_t j = 0; j < c.S.shown; ++j)
    if (vout.ends[j].masks[0].slot.index() != c.incoming + count - c.S.shown + j) return {false, "TEST: the inner masks come last, in order"};
  if (expected_log && vch.log != *expected_log) return {false, "TEST: the order of events is not the written-out one"};
  if (pl.left() != 0 || vl.left() != 0) return {false, "TEST: mask entries left over"};
  return {true, ""};
}

// ---- A product tree: PSC and LINE in turn, for maps that are never sent ----
// Entry x of level i is the product of entries x and x + 2^i of level i + 1; level n holds the 2^n
// leaves. Public claim: the product of all leaves. Each level above the leaves costs one PSC
// (product of the two halves of the level below, under equality weights at the point reached so
// far) and one LINE. What remains is one claim on the leaves at a point.
struct Tree {
  std::vector<std::vector<K>> level;  // level[i] has 2^i entries
  explicit Tree(std::vector<K> leaves) {
    level.push_back(std::move(leaves));
    while (level.front().size() > 1) {
      const std::vector<K>& below = level.front();
      std::vector<K> up(below.size() / 2);
      for (size_t x = 0; x < up.size(); ++x) up[x] = below[x] * below[x + up.size()];
      level.insert(level.begin(), up);
    }
  }
};
static std::vector<ToyMap<K>> halves(const std::vector<K>& v) {
  return {multilinear(std::vector<K>(v.begin(), v.begin() + v.size() / 2)),
          multilinear(std::vector<K>(v.begin() + v.size() / 2, v.end()))};
}

[[maybe_unused]] static Outcome run_tree(unsigned n, u64 seed, K target_shift) {
  TestRng rng(seed);
  const Tree tree(rng.vector<K>(size_t(1) << n));
  size_t total = 0;
  for (unsigned i = 0; i < n; ++i) total += 2 * i + 5;  // scalars and masks of a level with i variables
  const std::vector<K> row = rng.vector<K>(total);
  const MaskValues<K> masks = mask_row(row);
  const Summand<K> S = product_of_two();
  const K root = tree.level[0][0] + target_shift;

  MaskLedger pl(SendAccess::make(0, 0, u32(total))), vl(SendAccess::make(0, 0, u32(total)));
  ToyProverChannel<K> pch(random_coins<K>(seed ^ 0xC01));
  PointClaim<K> pc{{}, {}, root};
  for (unsigned i = 0; i < n; ++i) {
    SumClaim<K> claim = SumClaim<K>::at_point(std::move(pc), std::vector<unsigned>(i, 3));
    DefinitionSumcheck<K> inner(claim.vars, S, halves(tree.level[i + 1]));
    const size_t fresh = psc_masks(claim, S);
    PscOutput<K> out = psc_prove(pch, std::move(claim), S, inner, pl.take(fresh), masks);
    pc = line(pch, std::move(out.ends[0]), std::move(out.ends[1]));
  }
  ToyVerifierChannel<K> vch(pch.tape, random_coins<K>(seed ^ 0xC01));
  PointClaim<K> vc{{}, {}, root};
  ++runs;
  for (unsigned i = 0; i < n; ++i) {
    SumClaim<K> claim = SumClaim<K>::at_point(std::move(vc), std::vector<unsigned>(i, 3));
    const size_t fresh = psc_masks(claim, S);
    PscOutput<K> out;
    if (!psc_verify(vch, std::move(claim), S, vl.take(fresh), out)) return {false, "the verifier routine"};
    // Claim (U) has only mask entries here, so it is checked now. The end claims go on to LINE.
    if (!(mask_form(out.row_masks, masks) == out.row_target)) return {false, "row (U) of level " + std::to_string(i)};
    vc = line(vch, std::move(out.ends[0]), std::move(out.ends[1]));
  }
  if (!vch.at_end()) return {false, "scalars left unread"};
  if (!(mle_at(tree.level[n], vc.point) + mask_form(vc.masks, masks) == vc.value)) return {false, "the row on the leaves"};
  if (pch.log != vch.log) return {false, "TEST: the two sides saw different orders of events"};
  if (pch.tape.size() != total || vl.left() != 0) return {false, "TEST: scalars or masks are not 2i + 5 a level"};
  return {true, ""};
}

static void expect(bool want_accept, const Outcome& o, const std::string& what) {
  if (o.accepted != want_accept) {
    ++failures;
    std::printf("FAIL: %s: %s%s\n", what.c_str(), o.accepted ? "ACCEPTED" : "refused by ", o.why.c_str());
  }
}

int main(int argc, char**) {
  verbose = argc > 1;
  TestRng rng(12);
  std::vector<Case> all = cases(rng);
  const K one = K::one(), zero = K::zero();
  (void)one;
  (void)zero;
  for (const Case& c : all)
    for (u64 s = 1; s <= 5; ++s) expect(true, run_case(c, s, zero), c.name);
  for (unsigned n = 1; n <= 4; ++n) expect(true, run_tree(n, n, zero), "product tree");
  {  // all[0]: three variables under equality weights, a product of two shown maps
    std::vector<std::string> log = {"hiding sum", "eps PSC epsilon"};
    for (int j = 0; j < 3; ++j) log.insert(log.end(), {"scalar", "scalar", "alg PSC round", "draw"});
    log.insert(log.end(), {"scalar", "scalar", "alg PSC Y", "draw, not zero", "scalar", "scalar", "empty PSC values"});
    expect(true, run_case(all[0], 1, zero, nullptr, &log), "the written-out order of events");
  }
  const char* what = "honest provers, all accepted";
  if (failures)
    std::printf("psc_test: FAILED (%d of %d runs went otherwise than they must, see above)\n", failures, runs);
  else
    std::printf("psc_test: ok (%s; %d runs)\n", what, runs);
  return failures ? 1 : 0;
}
