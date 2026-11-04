// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests BATT (blocks/iop/batt.h), the block that batches J linear claims on the committed matrix M
// into one claim at a fresh point, over a hash chain (a Fiat-Shamir transcript,
// blocks/chain/rounds.h). A claim (a `row` in the code) has weights g on M's rows, a point psi for
// its columns, coefficients on entries of the mask row (random elements of the field E), and a
// `target`, the value that it states. In every run here, bind_inputs() puts all four into the
// chain's first query, the hash input fixed before the prover speaks. Nothing is committed here:
// M, which is small, and the mask row are the test's own vectors (`World`). BATT's verifier checks
// nothing itself, so verify() also tests the claim that BATT outputs.

// PSC (blocks/iop/psc.h): the sumcheck with a hiding polynomial that BATT runs.
#include "chain/closed.h"
#include "chain/public.h"
#include "chain/rounds.h"
#include "engine/sumcheck_cpu.h"
#include "field/field.h"
#include "iop/batt.h"
#include "numbers/blocks.h"
#include "toy_support.h"

using namespace blk;

static E random_E(TestRng &rng) { return E::from(from_u64(rng.word()), from_u64(rng.word()), from_u64(rng.word())); }
static std::vector<E> random_Es(TestRng &rng, size_t n) {
  std::vector<E> v(n);
  for (E &x : v) x = random_E(rng);
  return v;
}
static Setting setting() { return setting_goldilocks(); }

// A matrix of matrix_rows <= 2^k_row rows and 2^k_col columns, with `rows` claims on it.
struct Case {
  unsigned k_row, k_col;
  size_t matrix_rows, rows;  // of the matrix; J
  bool twice;                // a second BATT batches the first one's output with one more claim
};
// A claim without its mask entries: each side takes those from its own `MaskLedger`
// (blocks/iop/mask.h), which gives out each entry once, in order.
struct PublicRow {
  std::vector<Tensor> g;
  std::vector<E> psi, mask_coeff;
  E target;  // the true one; what a prover claims is bound separately
};
static void public_bytes(Writer &w, const PublicRow &r) {
  w.put_u64(r.g.size());
  for (const Tensor &t : r.g) public_bytes(w, t.scalar), public_bytes(w, t.los()), public_bytes(w, t.his());
  public_bytes(w, r.psi), public_bytes(w, r.mask_coeff);
}
// Called by `Binder`: nullptr, or why r is inconsistent or public_bytes() cannot write it.
static const char *public_fault(const PublicRow &r) {
  if (r.g.empty() || r.psi.empty()) return "a row without weights or without a column variable";
  for (const Tensor &t : r.g) {
    if (t.vars() != r.g[0].vars()) return "a row's weights differ in their number of variables";
    if (t.has_dense() || t.values) return "a row's weights have a written-out factor, which public_bytes(PublicRow) does not write";
  }
  return nullptr;
}
// A TensorRow as copyable values: each MaskSlot, which cannot be copied, is replaced by its index.
struct Plain {
  std::vector<Tensor> g;
  std::vector<E> psi;
  std::vector<std::pair<u32, E>> terms;  // (entry of the one mask row, coefficient)
  E target = E::zero();
};
static Plain plain_of(const TensorRow &r) {
  Plain p;
  p.g = r.g, p.psi = r.psi, p.target = r.target;
  for (const MaskTerm<E> &m : r.masks) p.terms.push_back({m.slot.index(), m.coeff});
  return p;
}
[[maybe_unused]] static bool same(const Plain &a, const Plain &b) {
  if (a.g.size() != b.g.size() || a.psi != b.psi || a.terms != b.terms || !(a.target == b.target)) return false;
  for (size_t i = 0; i < a.g.size(); i++)
    if (!(a.g[i].scalar == b.g[i].scalar) || a.g[i].los() != b.g[i].los() || a.g[i].his() != b.g[i].his()) return false;
  return true;
}
struct World {
  Case c;
  std::vector<std::vector<E>> M;  // M[y][col]
  std::vector<E> mask;            // the mask row, taken in order: the claims' entries, then BATT's
  std::vector<PublicRow> row;     // c.rows of them, and one more for the second BATT
  World(const Case &c_, TestRng &rng) : c(c_) {
    for (size_t y = 0; y < c.matrix_rows; y++) M.push_back(random_Es(rng, size_t(1) << c.k_col));
    size_t entries = 0;
    for (size_t j = 0; j < c.rows + 1; j++) {
      PublicRow r;
      for (size_t t = 0; t < 1 + j % 2; t++) {
        const std::vector<E> z = random_Es(rng, c.k_row);
        r.g.push_back(Tensor::eq(z.data(), c.k_row, random_E(rng)));
      }
      r.psi = random_Es(rng, c.k_col);
      r.mask_coeff = random_Es(rng, 1 + (j / 2) % 2);
      entries += r.mask_coeff.size();
      row.push_back(std::move(r));
    }
    mask = random_Es(rng, entries + 2 * batt_masks(c.k_col));
    size_t at = 0;
    for (PublicRow &r : row) {
      r.target = on_matrix(r.g, r.psi);
      for (const E &a : r.mask_coeff) r.target += a * mask[at++];
    }
  }
  // g's weights on row indices from matrix_rows on are not used.
  std::vector<E> weights_times_matrix(const std::vector<Tensor> &g) const {  // g^T M
    std::vector<E> w(size_t(1) << c.k_col, E::zero());
    for (size_t y = 0; y < c.matrix_rows; y++) {
      E a = E::zero();
      for (const Tensor &t : g) a += t.at(y);
      for (size_t col = 0; col < w.size(); col++) w[col] += a * M[y][col];
    }
    return w;
  }
  E on_matrix(const std::vector<Tensor> &g, const std::vector<E> &psi) const { return mle_at(weights_times_matrix(g), psi); }
  bool holds(const Plain &r) const {
    E left = on_matrix(r.g, r.psi);
    for (const auto &t : r.terms) left += t.second * mask.at(t.first);
    return left == r.target;
  }
  Plan plan() const {
    Plan p(setting());
    plan_batt(p, "first", c.rows, c.k_col);
    if (c.twice) plan_batt(p, "second", 2, c.k_col);
    return p;
  }
};
// A run's public inputs, as handles to values in the chain's first query (blocks/chain/public.h).
// `rows` has a group for each run of BATT: the first run's claims; the claim that joins the second.
struct Inputs {
  Public<std::vector<std::vector<PublicRow>>> rows;  // without their targets
  Public<std::vector<E>> targets;       // as the prover claims them
  std::vector<E> loose_targets;         // targets in no query; no run here fills it
  bool loose = false;
};
struct Given {  // what `bind_inputs` reads; the verifier's side never sees it
  const World &w;
  const std::vector<E> &targets;
  bool with_targets;  // false: targets in no query (no run here passes false)
};
static Inputs bind_inputs(Binder &b, const Given &given) {
  const World &w = given.w;
  const std::vector<E> &targets = given.targets;
  const bool with_targets = given.with_targets;
  Inputs x;
  Digest made_up{};  // no public file here: an arbitrary value serves as the configuration digest
  made_up.w[0] = 0x74746162;
  b.configuration(made_up);
  std::vector<std::vector<PublicRow>> groups{std::vector<PublicRow>(w.row.begin(), w.row.begin() + w.c.rows)};
  if (w.c.twice) groups.push_back({w.row[w.c.rows]});
  x.rows = b.statement("the rows", groups);
  if (with_targets)
    x.targets = b.statement("the rows' targets", targets);
  else
    x.loose_targets = targets, x.loose = true;
  return x;
}
template <class Ch>
static std::vector<TensorRow> take_rows(Ch &ch, const Inputs &x, MaskLedger &ledger, size_t group) {
  const std::vector<std::vector<PublicRow>> &groups = ch.get(x.rows);
  const std::vector<E> &targets = x.loose ? x.loose_targets : ch.get(x.targets);
  std::vector<TensorRow> rows;
  if (group >= groups.size()) return rows;
  size_t at = 0;  // the targets are in the groups' order
  for (size_t earlier = 0; earlier < group; earlier++) at += groups[earlier].size();
  for (const PublicRow &row : groups[group]) {
    TensorRow r;
    r.g = row.g, r.psi = row.psi, r.target = targets.at(at++);
    for (const E &a : row.mask_coeff) r.masks.push_back({ledger.take(), a});
    rows.push_back(std::move(r));
  }
  return rows;
}
static std::vector<E> true_targets(const World &w) {
  std::vector<E> t;
  for (const PublicRow &r : w.row) t.push_back(r.target);
  return t;
}
static BattSumcheck cpu_sumcheck(const World &w, std::vector<std::vector<Tensor>> g) {
  return [&w, g = std::move(g)](const Summand<E> &S, const std::vector<Weight<E>> &vars, const std::vector<E> &, const std::vector<std::vector<E>> &) {
    std::vector<std::vector<E>> maps;
    for (const std::vector<Tensor> &x : g) maps.push_back(w.weights_times_matrix(x));
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(S, vars, std::move(maps)));
  };
}
static std::vector<std::vector<Tensor>> weights_of(const std::vector<TensorRow> &rows) {
  std::vector<std::vector<Tensor>> g;
  for (const TensorRow &r : rows) g.push_back(r.g);
  return g;
}

// targets: as the prover claims them. 11 is the chain's label; *out gets the output claim.
static std::vector<u8> prove(const World &w, const std::vector<E> &targets, TensorRow *out = nullptr, bool with_targets = true) {
  const Planned plan = w.plan().finish();
  Writer proof;
  Binder b(11, plan);
  const Inputs x = bind_inputs(b, Given{w, targets, with_targets});
  ProverRounds ch(b, Digest{}, proof);
  MaskLedger ledger(SendAccess::make(0, 0, u32(w.mask.size())));
  std::vector<TensorRow> rows = take_rows(ch, x, ledger, 0), more = take_rows(ch, x, ledger, 1);
  BattSumcheck heavy = cpu_sumcheck(w, weights_of(rows));
  TensorRow one = batt_prove(ch, std::move(rows), ledger, mask_row(w.mask), heavy);
  if (!more.empty()) {
    more.insert(more.begin(), std::move(one));
    heavy = cpu_sumcheck(w, weights_of(more));
    one = batt_prove(ch, std::move(more), ledger, mask_row(w.mask), heavy);
  }
  ch.finish();
  if (out) *out = std::move(one);
  return proof.bytes();
}
// The verifier's steps. The test has no oracle message, so it makes the `MaskLedger` itself
// (through `SendAccess` in toy_support.h), with as many entries as the bound claims and the runs of
// BATT take.
static bool verifier_steps(VerifierRounds &ch, const Inputs &x, Plain &out) {
  const std::vector<std::vector<PublicRow>> &groups = ch.get(x.rows);
  if (groups.empty() || groups[0].empty()) return false;
  size_t entries = groups.size() * batt_masks(unsigned(groups[0][0].psi.size()));
  for (const std::vector<PublicRow> &g : groups)
    for (const PublicRow &r : g) entries += r.mask_coeff.size();
  MaskLedger ledger(SendAccess::make(0, 0, u32(entries)));
  std::vector<TensorRow> rows = take_rows(ch, x, ledger, 0), more = take_rows(ch, x, ledger, 1);
  TensorRow one;
  if (!batt_verify(ch, std::move(rows), ledger, one)) return false;
  if (!more.empty()) {
    more.insert(more.begin(), std::move(one));
    if (!batt_verify(ch, std::move(more), ledger, one)) return false;
  }
  out = plain_of(one);
  return true;
}
// For run_verifier() (blocks/chain/closed.h): it gets the channel (`VerifierRounds` in the code)
// and x but nothing of the `World`.
static bool decide(VerifierRounds &ch, const Inputs &x, Plain &out, Notes *) { return verifier_steps(ch, x, out); }
// Runs the verifier's side, which also rejects a proof that is not read to its end, then tests the
// output claim on the matrix and the mask row, which that side never sees; in the protocol the
// commitment and a later proximity test do this.
// Returns "" if the proof is accepted, otherwise which of the two rejected it.
static std::string verify(const World &w, const std::vector<E> &targets, const std::vector<u8> &bytes, Plain *out = nullptr) {
  Plain one;
  const char *why = nullptr;
  if (!run_verifier(11, w.plan().finish(), Given{w, targets, true}, bind_inputs, decide, bytes.data(), bytes.size(), one, &why)) return std::string("the verifier's side: ") + (why ? why : "");
  if (!w.holds(one)) return "the row that comes out";
  if (out) *out = std::move(one);
  return "";
}

static const Case CASES[] = {{2, 1, 4, 1, false}, {2, 3, 3, 2, false}, {3, 2, 5, 4, false}, {1, 4, 2, 3, true}, {3, 3, 8, 5, true}};

// What main() checks: the honest proof is accepted; both sides output the same claim; the
// proof is as long as the chain's salt and one element of E for each mask entry that BATT takes; a
// second BATT takes the first one's output among its inputs; the proof is rejected with the lowest
// or the highest bit of any one byte flipped, without its last byte, and with a byte added.
int main() {
  TestRng rng(20261015);
  int cases = 0;
  [[maybe_unused]] int played = 0;
  for (const Case &c : CASES) {
    const World w(c, rng);
    [[maybe_unused]] std::vector<E> targets = true_targets(w);
    cases++;
    TensorRow made;
    Plain theirs;
    const std::vector<u8> proof = prove(w, targets, &made);
    const std::string who = verify(w, targets, proof, &theirs);
    if (!who.empty()) return printf("case %d: the honest proof is refused by %s\n", cases, who.c_str()), 1;
    if (!same(plain_of(made), theirs)) return printf("case %d: prover and verifier put out different rows\n", cases), 1;
    if (proof.size() != DIGEST_BYTES + (c.twice ? 2 : 1) * batt_masks(c.k_col) * E_BYTES) return printf("case %d: the proof has %zu bytes\n", cases, proof.size()), 1;
    for (size_t i = 0; i < proof.size(); i++)
      for (u8 bit : {u8(1), u8(128)}) {
        std::vector<u8> other = proof;
        other[i] ^= bit;
        if (verify(w, targets, other).empty()) return printf("case %d: ACCEPTED with byte %zu altered\n", cases, i), 1;
      }
    std::vector<u8> shorter(proof.begin(), proof.end() - 1), longer = proof;
    longer.push_back(0);
    if (verify(w, targets, shorter).empty() || verify(w, targets, longer).empty()) return printf("case %d: ACCEPTED with a byte lost or added\n", cases), 1;
  }
  printf("ok: %d cases\n", cases);
  return 0;
}
