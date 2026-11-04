// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests PSC (the sumcheck with a hiding polynomial) and LINE (two claims on a map's halves become
// one) over the challenge field E, on a hash chain (a Fiat-Shamir transcript) whose rounds a `Plan`
// lists. The verifier's side runs inside run_verifier() (blocks/chain/closed.h): it gets the
// channel (`VerifierRounds`) and handles to what the `Binder` (blocks/chain/public.h) wrote into
// the chain's first query (hashed before the prover speaks), none of the test's own data. There
// are no commitments: the leaves or maps and the mask row (random elements of E), on which the
// verifier checks the linear claims (`Row`, `row_u` in identifiers; printed: row), are items of
// that query's statement. So this test says nothing about hiding.
// What main() runs: rows_on_a_table() and a product tree of eight leaves, three levels of PSC
// and LINE. The honest proof is accepted; it is one digest (the chain's salt) and one element of E
// per mask entry. It is rejected (printed: refused) for another root, with the lowest or highest
// bit of any byte flipped, without its last byte, and with a byte added.
// Malicious provers: in a test outside this repository. One leaves mu', the hiding polynomial's
// sum, out of its chain's query and fits it to a false target; the verifier hashes mu' and rejects.
// A scratch copy whose verifier left mu' out accepted it once; no build of this test repeats that.
#include "chain/closed.h"
#include "chain/public.h"
#include "chain/rounds.h"
#include "field/field.h"
#include "iop/psc_rows.h"
#include "numbers/blocks.h"
#include "toy_support.h"

using namespace blk;

static E random_E(TestRng& rng) { return E::from(from_u64(rng.word()), from_u64(rng.word()), from_u64(rng.word())); }
static std::vector<E> random_Es(TestRng& rng, size_t n) {
  std::vector<E> v(n);
  for (E& x : v) x = random_E(rng);
  return v;
}
static Setting setting() { return setting_goldilocks(); }
// A chain's first query needs a configuration digest (a hash of the public files); here a constant.
static Digest made_up_configuration() {
  Digest d{};
  d.w[0] = 0x70736363u;
  return d;
}
// y[0] * y[1]: two `shown` maps (PSC sends their masked end values), none `kept`, degree 2.
static Summand<E> product_of_two() {
  return {2, 0, 2, [](const std::vector<E>&, const std::vector<E>& y) { return std::vector<E>{y[0] * y[1]}; }};
}

// The product tree and the mask row. level[i] has 2^i elements. The PSC of level i uses 2 i + 5
// mask entries: psc_masks().
struct World {
  unsigned levels = 3;
  std::vector<std::vector<E>> level;  // level[i][x] = level[i+1][x] * level[i+1][x + 2^i]
  std::vector<E> row;
  E root;
  explicit World(TestRng& rng) {
    level.push_back(random_Es(rng, size_t(1) << levels));
    while (level.front().size() > 1) {
      const std::vector<E>& below = level.front();
      std::vector<E> up(below.size() / 2);
      for (size_t x = 0; x < up.size(); ++x) up[x] = below[x] * below[x + up.size()];
      level.insert(level.begin(), up);
    }
    size_t masks = 0;
    for (unsigned i = 0; i < levels; ++i) masks += 2 * i + 5;
    row = random_Es(rng, masks);
    root = level[0][0];
  }
};
static std::vector<ToyMap<E>> halves(const std::vector<E>& v) {
  return {multilinear(std::vector<E>(v.begin(), v.begin() + v.size() / 2)),
          multilinear(std::vector<E>(v.begin() + v.size() / 2, v.end()))};
}
static Plan tree_plan(const World& w) {
  Plan p(setting());
  plan_product_tree(p, "tree", w.levels, 0);
  return p;
}

// The tree's public inputs; the handles to them once bound.
struct TreeInputs {
  E root;
  std::vector<E> leaves, row;
};
struct TreeBound {
  Public<Digest> configuration;
  Public<E> root;
  Public<std::vector<E>> leaves, row;
};
static TreeBound tree_bind(Binder& binder, const TreeInputs& in) {
  TreeBound h;
  h.configuration = binder.configuration(made_up_configuration());
  h.root = binder.statement("root", in.root);
  h.leaves = binder.statement("the leaves, in place of a commitment", in.leaves);
  h.row = binder.statement("the mask row, in place of a commitment", in.row);
  return h;
}
static TreeInputs tree_inputs(const World& w, E root) { return {root, w.level[w.levels], w.row}; }

[[maybe_unused]] static std::vector<u8> tree_prove(const World& w, E root) {
  const Planned plan = tree_plan(w).finish();
  Writer proof;
  Binder binder(7, plan);
  const TreeBound in = tree_bind(binder, tree_inputs(w, root));
  ProverRounds ch(binder, Digest{}, proof);
  MaskLedger ledger(SendAccess::make(0, 0, u32(w.row.size())));
  const Summand<E> S = product_of_two();
  PointClaim<E> c{{}, {}, ch.get(in.root)};
  for (unsigned i = 0; i < w.levels; ++i) {
    SumClaim<E> claim = SumClaim<E>::at_point(std::move(c), std::vector<unsigned>(i, 3));
    DefinitionSumcheck<E> inner(claim.vars, S, halves(w.level[i + 1]));
    const size_t fresh = psc_masks(claim, S);
    PscOutput<E> out = psc_prove(ch, std::move(claim), S, inner, ledger.take(fresh), mask_row(w.row));
    c = line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
  }
  ch.finish();
  return proof.bytes();
}
// Records which check rejected, as a note's name: run_verifier() takes plain function pointers,
// which capture no variable of the test.
static bool refused_by(Notes* notes, const char* who) {
  if (notes) notes->push_back({who, E::zero()});
  return false;
}
// "row (U)": the linear claim that each PSC leaves on the mask row.
static bool tree_decide(VerifierRounds& ch, const TreeBound& in, Notes* notes) {
  const std::vector<E>& leaves = ch.get(in.leaves);
  const std::vector<E>& row = ch.get(in.row);
  unsigned levels = 0;
  while ((size_t(1) << levels) < leaves.size()) ++levels;
  MaskLedger ledger(SendAccess::make(0, 0, u32(row.size())));
  const Summand<E> S = product_of_two();
  const MaskValues<E> masks = mask_row(row);
  PointClaim<E> c{{}, {}, ch.get(in.root)};
  bool row_u = true;
  for (unsigned i = 0; i < levels; ++i) {
    SumClaim<E> claim = SumClaim<E>::at_point(std::move(c), std::vector<unsigned>(i, 3));
    const size_t fresh = psc_masks(claim, S);
    PscOutput<E> out;
    if (!psc_verify(ch, std::move(claim), S, ledger.take(fresh), out)) return refused_by(notes, "the verifier routine");
    row_u = row_u && mask_form(out.row_masks, masks) == out.row_target;
    c = line(ch, std::move(out.ends[0]), std::move(out.ends[1]));
  }
  if (!row_u) return refused_by(notes, "row (U)");
  if (!(mle_at(leaves, c.point) + mask_form(c.masks, masks) == c.value)) return refused_by(notes, "the row on the leaves");
  return true;
}
// Returns "" if the proof is accepted, otherwise the name of the check that rejected it.
template <class Inputs, class Bound>
static std::string verdict(const Planned& plan, const Inputs& inputs, Bound (*bind)(Binder&, const Inputs&), bool (*decide)(VerifierRounds&, const Bound&, Notes*),
                           const std::vector<u8>& bytes) {
  Notes notes;
  const char* why = nullptr;
  if (run_verifier(7, plan, inputs, bind, decide, bytes.data(), bytes.size(), &why, &notes)) return "";
  return !notes.empty() ? notes.back().first : std::string("the channel: ") + (why ? why : "no reason given");
}
[[maybe_unused]] static std::string tree_verify(const World& w, E root, const std::vector<u8>& bytes) {
  return verdict(tree_plan(w).finish(), tree_inputs(w, root), tree_bind, tree_decide, bytes);
}

// Checks the linear claims that blocks/iop/psc_rows.h makes of a PSC's output; this PSC is
// proved, not verified. `table`, four rows of eight elements, stands for the main rows of oracle
// message 1, `mask` for the mask rows of oracle message 0.
// Sum claim: sum_x eq(z, x) a(x) b(x) = target, a = row 1 (`shown`), b = row 2 (`kept`: not sent).
[[maybe_unused]] static std::string rows_on_a_table(TestRng& rng) {
  const std::vector<E> table = random_Es(rng, 32);  // row r's element c is table[r + 4 c]
  auto row_of = [&](size_t r) {
    std::vector<E> v(8);
    for (size_t c = 0; c < 8; ++c) v[c] = table[r + 4 * c];
    return v;
  };
  const Summand<E> S{1, 1, 1, [](const std::vector<E>&, const std::vector<E>& y) { return std::vector<E>{E::zero(), y[0]}; }};
  SumClaim<E> claim;
  const std::vector<E> z = random_Es(rng, 3);
  for (int j = 0; j < 3; ++j) claim.vars.push_back(Weight<E>::equality(z[j], 3));
  const std::vector<ToyMap<E>> maps = {multilinear(row_of(1)), multilinear(row_of(2))};
  DefinitionSumcheck<E> inner(claim.vars, S, maps);
  claim.target = DefinitionSumcheck<E>(claim.vars, S, maps).sum();
  const size_t fresh = psc_masks(claim, S);
  const std::vector<E> mask = random_Es(rng, fresh);
  Plan plan(setting());
  plan_psc(plan, "one", psc_size(claim, S));
  Writer proof;
  Binder binder(7, plan.finish());
  (void)binder.configuration(made_up_configuration());
  const Public<std::vector<E>> bound_point = binder.statement("point", z);
  const Public<E> bound_target = binder.statement("target", claim.target);
  ProverRounds ch(binder, Digest{}, proof);
  MaskLedger ledger(SendAccess::make(0, 0, u32(fresh)));
  SumClaim<E> bound;
  for (const E& x : ch.get(bound_point)) bound.vars.push_back(Weight<E>::equality(x, 3));
  bound.target = ch.get(bound_target);
  PscOutput<E> out = psc_prove(ch, std::move(bound), S, inner, ledger.take(fresh), mask_row(mask));
  ch.finish();
  std::vector<Row> rows;
  rows.push_back(psc_row_u(out, {row_of_table(1, Part::MAIN, 2, 2)}, "U"));
  rows.push_back(point_row(std::move(out.ends[0]), row_of_table(1, Part::MAIN, 2, 1), "E0"));
  for (const Row& r : rows) {
    E left = E::zero();
    for (const Term& t : r.terms) {
      if (t.oracle != 1 || t.part != Part::MAIN || t.v.vars() != 5) return "a term of row " + r.name + " is misplaced";
      for (u64 j = 0; j < 32; ++j) left = left + t.v.at(j) * table[j];
    }
    for (const Unit& u : r.units) {
      if (u.oracle != 0 || u.part != Part::MASK) return "a mask of row " + r.name + " is misplaced";
      left = left + u.coeff * mask[u.index];
    }
    if (!(left == r.target)) return "row " + r.name + " fails";
  }
  // block_of_table(): the upper half of the table read as a vector, at a point of four coordinates.
  const std::vector<E> point = random_Es(rng, 4);
  Row half;
  block_of_table(1, Part::MAIN, 5, 1)(half, point, E::one());
  E left = E::zero();
  for (u64 j = 0; j < 32; ++j) left = left + half.terms[0].v.at(j) * table[j];
  if (!(left == mle_at(std::vector<E>(table.begin() + 16, table.end()), point))) return "block_of_table";
  return "";
}

int main() {
  TestRng rng(3);
  int failures = 0;
  size_t changed = 0, bytes = 0;
  const World w(rng);
  const std::vector<u8> proof = tree_prove(w, w.root);
  bytes = proof.size();
  if (!tree_verify(w, w.root, proof).empty()) ++failures, std::printf("FAIL: the honest proof is refused\n");
  if (proof.size() != DIGEST_BYTES + w.row.size() * E_BYTES) ++failures, std::printf("FAIL: the length of the proof\n");
  if (tree_verify(w, w.root + E::one(), proof).empty()) ++failures, std::printf("FAIL: accepted for another root\n");
  const std::string rows = rows_on_a_table(rng);
  if (!rows.empty()) ++failures, std::printf("FAIL: rows of a statement: %s\n", rows.c_str());
  for (size_t i = 0; i < proof.size(); ++i)
    for (u8 flip : {u8(1), u8(0x80)}) {
      std::vector<u8> other = proof;
      other[i] ^= flip;
      ++changed;
      if (tree_verify(w, w.root, other).empty()) ++failures, std::printf("FAIL: accepted with byte %zu changed\n", i);
    }
  std::vector<u8> shorter(proof.begin(), proof.end() - 1), longer = proof;
  longer.push_back(0);
  if (tree_verify(w, w.root, shorter).empty()) ++failures, std::printf("FAIL: accepted without its last byte\n");
  if (tree_verify(w, w.root, longer).empty()) ++failures, std::printf("FAIL: accepted with a byte added\n");
  if (failures)
    std::printf("psc_chain_test: FAILED (%d checks failed, see above)\n", failures);
  else
    std::printf("psc_chain_test: ok (proof of %zu bytes accepted; %zu copies with one bit flipped, a shorter and a longer one refused)\n", bytes,
                changed);
  return failures ? 1 : 0;
}
