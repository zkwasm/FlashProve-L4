// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/iop/batt_shown.h: the form of the block BATT (batches linear claims on a committed
// matrix M into one) whose sumcheck (PSC, blocks/iop/psc.h) sends its end value. It runs over a
// hash chain (Fiat-Shamir transcript) and a `Plan`, arranged as in a request's proof. Each of L
// segments (a sub-proof's part of the proof; `links`, "link" in messages) starts with J claims
// (`rows`; value: `target`), each masked by an entry of the segment's mask row. The block batches
// them into claim (E), on M and the entry c of the request's index-mask message that masks the end
// value, and leaves claim (U), on entries of the segment's mask row only. If L > 1, BATT
// (blocks/iop/batt.h) batches the L claims (E) into one.
// Oracle message 0 is the index-mask message, 1 to L the segments' mask rows. Toy sizes (`CASES`).
// M and the mask rows are plain vectors, not commitments. The programs `cp_prove` and `cp_verify`
// do not include batt_shown.h, so this does not test them.
#include "chain/closed.h"
#include "chain/public.h"
#include "chain/rounds.h"
#include "engine/sumcheck_cpu.h"
#include "field/field.h"
#include "iop/batt_shown.h"
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

struct Case {
  unsigned k_row, k_col;
  size_t matrix_rows, links, rows;  // rows: J, of each link
};
// An incoming claim without its value: weights, column point, its mask entry's coefficient.
struct PublicRow {
  std::vector<Tensor> g;
  std::vector<E> psi;
  E mask_coeff;
};
static void public_bytes(Writer &w, const PublicRow &r) {
  w.put_u64(r.g.size());
  for (const Tensor &t : r.g) public_bytes(w, t.scalar), public_bytes(w, t.los()), public_bytes(w, t.his());
  public_bytes(w, r.psi), public_bytes(w, r.mask_coeff);
}
// For Binder (blocks/chain/public.h): nullptr, or why r is inconsistent or cannot be written.
static const char *public_fault(const PublicRow &r) {
  if (r.g.empty() || r.psi.empty()) return "a row without weights or without a column variable";
  for (const Tensor &t : r.g) {
    if (t.vars() != r.g[0].vars()) return "a row's weights differ in their number of variables";
    if (t.has_dense() || t.values) return "a row's weights have a written-out factor, which public_bytes(PublicRow) does not write";
  }
  return nullptr;
}
// A claim in copyable form (a `MaskSlot`, blocks/iop/mask.h, cannot be copied). Empty g: a claim on
// mask entries only. `terms`: ((oracle message, entry), coefficient).
struct Plain {
  std::vector<Tensor> g;
  std::vector<E> psi;
  std::vector<std::pair<std::pair<u32, u32>, E>> terms;
  E target = E::zero();
};
static Plain plain_of(const std::vector<Tensor> &g, const std::vector<E> &psi, const std::vector<MaskTerm<E>> &masks, const E &target) {
  Plain p;
  p.g = g, p.psi = psi, p.target = target;
  for (const MaskTerm<E> &m : masks) p.terms.push_back({{m.slot.oracle(), m.slot.index()}, m.coeff});
  return p;
}
struct HandedOn {
  std::vector<Plain> own, on;  // by segment: claim (U); claim (E)
  Plain last;
};
struct World {
  Case c;
  std::vector<std::vector<E>> M;
  std::vector<std::vector<E>> mask;  // by oracle message
  std::vector<PublicRow> row;        // links * rows, link by link
  std::vector<E> target;             // the true ones
  World(const Case &c_, TestRng &rng) : c(c_) {
    for (size_t y = 0; y < c.matrix_rows; y++) M.push_back(random_Es(rng, size_t(1) << c.k_col));
    mask.push_back(random_Es(rng, c.links + batt_masks(c.k_col)));
    for (size_t l = 0; l < c.links; l++) mask.push_back(random_Es(rng, c.rows + batt_shown_own_masks(c.k_col) + 1));
    for (size_t l = 0; l < c.links; l++)
      for (size_t j = 0; j < c.rows; j++) {
        PublicRow r;
        for (size_t t = 0; t < 1 + j % 2; t++) {
          const std::vector<E> z = random_Es(rng, c.k_row);
          r.g.push_back(Tensor::eq(z.data(), c.k_row, random_E(rng)));
        }
        r.psi = random_Es(rng, c.k_col), r.mask_coeff = random_E(rng);
        target.push_back(on_matrix(r.g, r.psi) + r.mask_coeff * mask[1 + l][j]);
        row.push_back(std::move(r));
      }
  }
  std::vector<E> weights_times_matrix(const std::vector<Tensor> &g) const {
    std::vector<E> w(size_t(1) << c.k_col, E::zero());
    for (size_t y = 0; y < c.matrix_rows; y++) {
      E a = E::zero();
      for (const Tensor &t : g) a += t.at(y);
      for (size_t col = 0; col < w.size(); col++) w[col] += a * M[y][col];
    }
    return w;
  }
  E on_matrix(const std::vector<Tensor> &g, const std::vector<E> &psi) const { return mle_at(weights_times_matrix(g), psi); }
  MaskValues<E> values() const {
    return [this](u32 oracle, u32 index) {
      BLK_REQUIRE(oracle < mask.size() && index < mask[oracle].size(), "no such mask entry");
      return mask[oracle][index];
    };
  }
  bool holds(const Plain &r) const {
    E left = r.g.empty() ? E::zero() : on_matrix(r.g, r.psi);
    for (const auto &t : r.terms) left += t.second * values()(t.first.first, t.first.second);
    return left == r.target;
  }
  Plan plan() const {
    Plan p(setting());
    for (size_t l = 0; l < c.links; l++) plan_batt_shown(p, "link", c.rows, c.k_col);
    if (c.links > 1) plan_batt(p, "the request's end", c.links, c.k_col);
    return p;
  }
};
static bool only(const Plain &row, u32 oracle) {
  for (const auto &t : row.terms)
    if (t.first.first != oracle) return false;
  return true;
}
struct Inputs {
  Public<std::vector<std::vector<PublicRow>>> rows;  // by segment; gives both sides L and J
  Public<std::vector<E>> targets;                    // the claims' values, in the order of `rows`
};
struct Given {  // what `bind_inputs` reads; `decide` never sees it
  const World &w;
  const std::vector<E> &targets;
};
static Inputs bind_inputs(Binder &b, const Given &given) {
  const World &w = given.w;
  const std::vector<E> &targets = given.targets;
  Inputs x;
  Digest made_up{};  // the test has no public files: this replaces the configuration digest
  made_up.w[0] = 0x74746162;
  b.configuration(made_up);
  std::vector<std::vector<PublicRow>> links;
  for (size_t l = 0; l < w.c.links; l++) links.emplace_back(w.row.begin() + l * w.c.rows, w.row.begin() + (l + 1) * w.c.rows);
  x.rows = b.statement("the rows", links), x.targets = b.statement("the rows' targets", targets);
  return x;
}
template <class Ch>
static std::vector<TensorRow> take_rows(Ch &ch, const Inputs &x, size_t l, MaskLedger &own) {
  const std::vector<std::vector<PublicRow>> &links = ch.get(x.rows);
  size_t at = 0;
  for (size_t earlier = 0; earlier < l; earlier++) at += links[earlier].size();
  std::vector<TensorRow> rows;
  for (const PublicRow &p : links.at(l)) {
    TensorRow r;
    r.g = p.g, r.psi = p.psi, r.target = ch.get(x.targets).at(at++);
    r.masks.push_back({own.take(), p.mask_coeff});
    rows.push_back(std::move(r));
  }
  return rows;
}
static BattSumcheck cpu_sumcheck(const World &w, const std::vector<TensorRow> &rows, [[maybe_unused]] bool lie) {
  std::vector<std::vector<E>> maps;
  for (const TensorRow &r : rows) maps.push_back(w.weights_times_matrix(r.g));
  return [maps](const Summand<E> &S, const std::vector<Weight<E>> &vars, const std::vector<E> &, const std::vector<std::vector<E>> &) {
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(S, vars, maps));
  };
}
struct Ledgers {
  std::vector<std::unique_ptr<MaskLedger>> of;
  explicit Ledgers(const World &w) {
    for (u32 o = 0; o < w.mask.size(); o++) of.emplace_back(new MaskLedger(SendAccess::make(o, 0, u32(w.mask[o].size()))));
  }
};

static std::vector<u8> prove(const World &w, const std::vector<E> &targets, TensorRow *last = nullptr) {
  const Planned plan = w.plan().finish();
  Writer proof;
  Binder b(12, plan);
  const Inputs x = bind_inputs(b, Given{w, targets});
  ProverRounds ch(b, Digest{}, proof);
  Ledgers ledger(w);
  std::vector<TensorRow> pending;
  const size_t links = ch.get(x.rows).size();
  for (size_t l = 0; l < links; l++) {
    std::vector<TensorRow> rows = take_rows(ch, x, l, *ledger.of[1 + l]);
    const BattSumcheck heavy = cpu_sumcheck(w, rows, l == 0);
    MaskValues<E> values = w.values();
    pending.push_back(batt_shown_prove(ch, std::move(rows), *ledger.of[1 + l], *ledger.of[0], values, heavy).row);
  }
  TensorRow one;
  if (links > 1) {
    const BattSumcheck heavy = cpu_sumcheck(w, pending, false);
    one = batt_prove(ch, std::move(pending), *ledger.of[0], w.values(), heavy);
  } else {
    one = std::move(pending[0]);
  }
  ch.finish();
  if (last) *last = std::move(one);
  return proof.bytes();
}
// The verifier's side: run_verifier() (blocks/chain/closed.h) gives it only the channel and the
// bound handles. No oracle message is sent, so it makes each `MaskLedger` (gives out each mask
// entry once), sized from the bound claims. It puts the claims that the blocks leave into `out`,
// unchecked, for verify().
static bool decide(VerifierRounds &ch, const Inputs &x, HandedOn &out, Notes *) {
  const std::vector<std::vector<PublicRow>> &rows = ch.get(x.rows);
  const size_t links = rows.size();
  if (links == 0 || rows[0].empty()) return false;
  const unsigned k_col = unsigned(rows[0][0].psi.size());
  std::vector<std::unique_ptr<MaskLedger>> ledger;
  ledger.emplace_back(new MaskLedger(SendAccess::make(0, 0, u32(links + (links > 1 ? batt_masks(k_col) : 0)))));
  for (size_t l = 0; l < links; l++) ledger.emplace_back(new MaskLedger(SendAccess::make(u32(1 + l), 0, u32(rows[l].size() + batt_shown_own_masks(k_col)))));
  std::vector<TensorRow> pending;
  for (size_t l = 0; l < links; l++) {
    BattShown s;
    if (!batt_shown_verify(ch, take_rows(ch, x, l, *ledger[1 + l]), *ledger[1 + l], *ledger[0], s)) return false;
    out.own.push_back(plain_of({}, {}, s.own_masks, s.own_target)), out.on.push_back(plain_of(s.row.g, s.row.psi, s.row.masks, s.row.target));
    pending.push_back(std::move(s.row));
  }
  TensorRow one;
  if (links > 1) {
    if (!batt_verify(ch, std::move(pending), *ledger[0], one)) return false;
  } else {
    one = std::move(pending[0]);
  }
  out.last = plain_of(one.g, one.psi, one.masks, one.target);
  return true;
}
// Stands in for the commitments and the later proximity tests: checks the claims from decide()
// against M and the mask rows, which decide() never sees. Accepts if run_verifier() passes (the
// proof must be read to its end), every claim is true, each (U) names entries of its segment's mask
// row only, each (E) names exactly one entry, of message 0, and the last claim names entries of
// message 0 only. Returns "", or what rejected the proof (printed: refused).
static std::string verify(const World &w, const std::vector<E> &targets, const std::vector<u8> &bytes, Plain *last = nullptr) {
  HandedOn out;
  const char *why = nullptr;
  if (!run_verifier(12, w.plan().finish(), Given{w, targets}, bind_inputs, decide, bytes.data(), bytes.size(), out, &why)) return std::string("the verifier's side: ") + (why ? why : "");
  for (size_t l = 0; l < out.own.size(); l++) {
    if (!only(out.own[l], u32(1 + l))) return "PROGRAM: a link's row (U) names another message's entry";
    if (out.on[l].terms.size() != 1 || !only(out.on[l], 0)) return "PROGRAM: the row that is handed on does not name one entry of the index-mask message";
  }
  for (size_t l = 0; l < out.own.size(); l++) {
    if (!w.holds(out.own[l])) return "a link's row (U)";
    if (!w.holds(out.on[l])) return "the row that is handed on";
  }
  if (!only(out.last, 0)) return "PROGRAM: the last row names an entry outside the index-mask message";
  if (!w.holds(out.last)) return "the last row";
  if (last) *last = std::move(out.last);
  return "";
}

static const Case CASES[] = {{2, 1, 4, 1, 1}, {2, 3, 3, 1, 2}, {3, 2, 5, 2, 3}, {1, 4, 2, 3, 2}, {3, 3, 8, 4, 4}};

// What main() checks: honest proofs are accepted (see verify()); both sides compute the
// same last claim (weights not compared); after the chain's salt, the proof has one extension-field
// element for each mask entry that the blocks take; a proof with the lowest or highest bit of any
// one byte flipped, its last byte removed or a byte appended is rejected.
int main() {
  TestRng rng(20261016);
  int cases = 0;
  [[maybe_unused]] int played = 0;
  for (const Case &c : CASES) {
    const World w(c, rng);
    [[maybe_unused]] std::vector<E> targets = w.target;
    cases++;
    [[maybe_unused]] const char *by = "a link's row (U)";
    TensorRow made;
    Plain theirs;
    const std::vector<u8> proof = prove(w, targets, &made);
    const std::string who = verify(w, targets, proof, &theirs);
    if (!who.empty()) return printf("case %d: the honest proof is refused by %s\n", cases, who.c_str()), 1;
    const Plain mine = plain_of(made.g, made.psi, made.masks, made.target);
    if (!(mine.target == theirs.target) || mine.psi != theirs.psi || mine.terms != theirs.terms) return printf("case %d: the two sides' last rows differ\n", cases), 1;
    const size_t scalars = c.links * (batt_shown_own_masks(c.k_col) + 1) + (c.links > 1 ? batt_masks(c.k_col) : 0);
    if (proof.size() != DIGEST_BYTES + scalars * E_BYTES) return printf("case %d: the proof has %zu bytes\n", cases, proof.size()), 1;
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
