// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Shared by the sub-proof checks.
// A hidden value v is not sent: the verifier is shown v plus a public factor times an entry
// of the committed mask row (vf_sumcheck_psc.h). A slot is such an entry's number. A claim: a form
// (Aff) is 0. HFork batches the claims into one equation and does not test it: doorH()
// (vf_subproof_segment.h) passes it to ChainCheck (vf_request_chain.h), which tests it at its end.
// Each rejection name here (refuse()'s argument) has `internal-`, meant for program faults: no
// verdict (NOT CHECKED).
// Source for showing a hidden value under an entry of a committed mask row: Dalal, Hemo, Rabinovich, Rothblum, "VEIL:
// Lightweight Zero-Knowledge for Hash-Based Multilinear Proof Systems", Cryptology ePrint Archive 2026/683 (version
// of April 13, 2026), Figure 7 (there the value plus the entry; here the value plus a public factor times the entry);
// adding the claims up into one equation follows its Remark 4.4.
#pragma once
#include <functional>
#include <array>
#include <map>
#include <memory>
#include "vf_smallness_packing.h"

// The smallest power of two >= v. It requires v <= 2^63: for a larger v the loop never ends.
static u64 powerOfTwoAtLeast(u64 v) {
  u64 p = 1;
  while (p < v) p <<= 1;
  return p;
}
// A virtual row of an array (HTree): the vector sum_r coef[r] * (row r). Forms read its entries
// (Ent). L: the one equation's coefficients on its entries. `Ex` (vf_common.h): an extension-field
// element.
struct HRow {
  vector<Ex> coef;
  vector<Ex> L;
  // L[i]: entry L0 + i. doorH() (vf_subproof_segment.h) requires L0 = 0 and |L| = K.
  u64 L0 = 0;
  // With rowForm, coef must be a product over the bits of the row number r, since blocks/iop/laid.h
  // takes at most one of a term's row weights and column weights (here L) written out. One of:
  //   rowPoint: coef[rowFirst + r] = eq(rowPoint, r) for r < 2^|rowPoint|, 0 on all other rows; the
  //     point's coordinate 0 belongs to r's top bit, and rowFirst is a multiple of 2^|rowPoint|;
  //   rowOnes: coef[r] = 1;   rowLo, rowHi: coef[r] = prod_t (bit t of r ? rowHi[t] : rowLo[t]).
  // doorH() compares coef with the product. Without rowForm: one term for each nonzero coef[r].
  bool rowForm = false, rowOnes = false;
  vector<Ex> rowPoint, rowLo, rowHi;
  u64 rowFirst = 0;
};
// kind: 'N', a plain table; 'P', a matrix of the approximate range check, up to 3 cells per symbol.
struct HTree { // an array of R rows and K columns that a sub-proof's claims read
  string name;
  char kind = 'N';
  u64 R = 0, K = 0;
  const u32 *root = nullptr;
  bool owner = false;
  vector<HRow> rows;
};
// HTree::root: not null once rootInH() marks it committed. owner: this sub-proof commits to it.

// Ent: the term coef * (entry idx of row HFork::trees[tree].rows[row]). Aff, an affine form:
// e's terms + sum_{x in s} x.second * (hidden value of slot x.first) + c. affAdd(): a += k * b.
struct Ent {
  u32 tree = 0, row = 0;
  u64 idx = 0;
  Ex coef = eZ();
};
struct Aff {
  vector<Ent> e;
  vector<std::pair<u32, Ex>> s;
  Ex c = eZ();
};
static Aff affC(Ex v) {
  Aff a;
  a.c = v;
  return a;
}
static Aff affSlot(u32 slot) {
  Aff a;
  a.s.push_back({slot, eO()});
  return a;
}
static Aff affEnt(u32 tree, u32 row, u64 idx) {
  Aff a;
  Ent x;
  x.tree = tree;
  x.row = row;
  x.idx = idx;
  x.coef = eO();
  a.e.push_back(x);
  return a;
}
static void affAdd(Aff &a, const Aff &b, Ex k) {
  for (auto &x : b.e) {
    Ent y = x;
    y.coef = x.coef * k;
    a.e.push_back(y);
  }
  for (auto &x : b.s) a.s.push_back({x.first, x.second * k});
  a.c = a.c + b.c * k;
}
static Aff operator+(Aff a, const Aff &b) {
  affAdd(a, b, eO());
  return a;
}
static Aff operator-(Aff a, const Aff &b) {
  affAdd(a, b, eZ() - eO());
  return a;
}
static Aff operator*(const Aff &a, Ex k) {
  Aff r;
  affAdd(r, a, k);
  return r;
}

#include "vf_transcript_calls.h"
// One sub-proof's state. G_: the prefix of its rejection names.
struct HFork {
  string G_;
  vector<HTree> trees;
  u32 ns = 0;
  // The hidden value of slot j < ns is vp[j] - mf[j] * (entry j of the mask row): vp[j] was shown,
  // mf[j] is a public factor. LS[j], c0: the one equation's coefficient on it; its constant.
  vector<Ex> vp, mf, LS;
  Ex c0 = eZ();
  u32 nClaim = 0;
  void rootTaken(size_t t, const u32 *w) {
    trees[t].root = w;
  }
  // gotF[j] is set when slot j gets its value. A second value is rejected ("internal-slot-twice"):
  // a mask entry hides one shown scalar, and a form refers to a slot and is evaluated only at the
  // end. publicPart(), called when a sumcheck (`Psc`) begins, rejects a form whose slot has no
  // value; settle() rejects such a slot if its coefficient is not 0 (both "internal-slot-order").
  vector<u8> gotF;
  // Sets up n slots without values. Call it before the first message: hide() rejects otherwise.
  void slots(u32 n) {
    ns = n;
    vp.assign(n, eZ());
    mf.assign(n, eZ());
    LS.assign(n, eZ());
    gotF.assign(n, 0);
  }
  // Gives the free slot j its hidden value: shownScalar - factor * (mask entry j).
  void hide(u32 j, Ex shownScalar, Ex factor) {
    if (j >= ns || gotF.size() != ns) refuse(G_ + "internal-slot-marks");
    if (gotF[j]) refuse(G_ + "internal-slot-twice");
    gotF[j] = 1;
    vp[j] = shownScalar;
    mf[j] = factor;
  }
  // What was shown for slot j.
  Ex shown(u32 j) const {
    if (j >= ns || !gotF[j]) refuse(G_ + "internal-slot-order");
    return vp[j];
  }
  // The form a with each hidden value replaced by what was shown for it, without the terms in a.e.
  Ex publicPart(const Aff &a) const {
    Ex v = a.c;
    for (auto &x : a.s) v = v + x.second * shown(x.first);
    return v;
  }
  // Reads n hidden values, each sent as value + mask entry, as one message into slots `slot` ..
  // `slot` + n - 1. Returns the first slot.
  u32 claimsIn(Tr &tr, u32 &slot, u32 n) {
    const vector<Ex> v = trRead(tr, n);
    for (u32 i = 0; i < n; i++) hide(slot + i, v[i], eO());
    slot += n;
    return slot - n;
  }
  // Adds w * a to the one equation (the rows' L vectors, LS and c0).
  void push(Ex w, const Aff &a) {
    if (doorShutF) refuse(G_ + "internal-f-door");
    for (auto &x : a.e) {
      if (x.tree >= trees.size() || x.row >= trees[x.tree].rows.size()) refuse(G_ + "internal-L-row");
      HRow &R_ = trees[x.tree].rows[x.row];
      vector<Ex> &L = R_.L;
      // The window: columns [R_.L0, R_.L0 + L.size()).
      // An index outside the window is rejected before any write; the test
      // is ordered so that the subtraction cannot wrap.
      if (x.idx < R_.L0 || x.idx - R_.L0 >= L.size()) refuse(G_ + "internal-L-index");
      L[x.idx - R_.L0] = L[x.idx - R_.L0] + w * x.coef;
    }
    for (auto &x : a.s) {
      if (x.first >= ns) refuse(G_ + "internal-L-slot");
      LS[x.first] = LS[x.first] + w * x.second;
    }
    c0 = c0 + w * a.c;
  }
  // Claim c, counted from 0 in the order in which the claims are added, has the batching
  // coefficient kappa_c = theta^c. kapAt() rejects until batchBy() has stored the challenge theta.
  Ex theta = eZ();
  bool thetaDrawnF = false;
  void batchBy(Ex drawn) {
    if (thetaDrawnF) refuse(G_ + "internal-f-draws");
    theta = drawn;
    thetaDrawnF = true;
  }
  Ex kapAt(u32 c) const {
    if (!thetaDrawnF) refuse(G_ + "internal-f-draws");
    Ex power = eO(), square = theta;
    for (u32 e = c; e; e >>= 1, square = square * square)
      if (e & 1) power = power * square;
    return power;
  }
  // The planned number of claims (such as `HPlan2::nLin`). doorH() requires nClaim, the number of
  // claims added, to equal it. kapMapF must have this many entries, or kapAllTakenF() rejects.
  u32 planClaimsF = 0;
  // kapMapF[c] is set once kappa_c has been taken: the bound planClaimsF by itself would not catch
  // a repeated index.
  vector<u8> kapMapF;
  // Set by doorH() (vf_subproof_segment.h), which a sub-proof's code calls after its
  // last claim. From then on kapTake() and push() reject ("internal-f-door"),
  // so a later lin() or t2Lin() is rejected.
  bool doorShutF = false;
  // Gives kappa_c to a claim. Once kapMapF is set up (t2Batch(), vf_subproof_segment.h), it rejects c at or
  // above planClaimsF and a second call for the same c; while kapMapF is empty it checks neither.
  Ex kapTake(u32 c) {
    if (doorShutF) refuse(G_ + "internal-f-door");
    if (!kapMapF.empty()) {
      if (c >= planClaimsF || kapMapF.size() != planClaimsF || kapMapF[c]) refuse(G_ + "internal-f-draws");
      kapMapF[c] = 1;
    }
    return kapAt(c);
  }
  // Rejects unless every c below planClaimsF has been taken.
  void kapAllTakenF() const {
    if (kapMapF.size() != planClaimsF) refuse(G_ + "internal-f-draws");
    for (u8 x : kapMapF)
      if (x != 1) refuse(G_ + "internal-f-draws");
  }
  // A claim that a function leaves for its caller to add is recorded with owe(); the code that adds
  // it sets `added`, or settle() rejects. This catches a forgotten claim, not a wrong one.
  struct Owed {
    string what;
    bool added = false;
  };
  vector<std::shared_ptr<Owed>> owed;
  std::shared_ptr<Owed> owe(const string &what) {
    owed.push_back(std::make_shared<Owed>(Owed{what, false}));
    return owed.back();
  }
  Ex kap() { return kapTake(nClaim++); }
  // Test support for the checkers under verifier/tests (claims_report.h there);
  // cp_verify never sets valuesF. With valuesF (the prover's virtual rows and mask row), each claim
  // is recorded when added, and judgeAll() lists those whose value is not 0.
  // No decision of the verifier reads any of this.
  struct TestProverValues {
    struct TestProverVirtualRow {
      vector<Ex> cv, D; // a virtual row of the prover's: weights cv, D[c] = sum_r cv[r] cell(r, c)
    };
    vector<vector<TestProverVirtualRow>> pairs;                            // pairs[tree]
    vector<Ex> mask;                                         // the mask row's entries
  };
  std::shared_ptr<const TestProverValues> valuesF;
  struct TestClaimFoundFalse {
    u32 claim = 0;
    string section;
  };
  vector<TestClaimFoundFalse> falseF;
  Ex valuesSumF = eZ(); // sum over the claims of kappa * value
  // When a claim is added, a slot's value, and the HRow::coef by which judgeAll() finds a virtual
  // row's pair (the TestProverVirtualRow with cv equal to coef), may still be missing. So a claim
  // is stored as its constant plus its entries on arrays with one pair, for each virtual row of an
  // array with several pairs a sum on each pair, and its slot terms.
  struct TestUnresolvedRowSums {
    u32 tree = 0, row = 0;
    vector<Ex> byPair; // the sum of coefficient * D_p[column] for each pair p of the array
  };
  struct TestClaimValueParts {
    Ex part = eZ();
    vector<TestUnresolvedRowSums> open;
  };
  struct TestRecordedClaim {
    u32 claim = 0;
    string section;
    Ex kappa;
    TestClaimValueParts value;
    vector<std::pair<u32, Ex>> slots;
  };
  vector<TestRecordedClaim> judgedF;
  mutable vector<vector<u8>> readF; // readF[tree][virtual row]: some claim has read it
  string sectionF;
  void section(const string &name) { sectionF = name; }
  // Adds coefficient * (entry 'column' of virtual row 'row' of tree 'tree') to a claim's value.
  void take(TestClaimValueParts &j, u32 tree, u32 row, u64 column, const Ex &coefficient) const {
    if (tree >= trees.size() || tree >= valuesF->pairs.size() || row >= trees[tree].rows.size() || valuesF->pairs[tree].empty()) {
      fprintf(stderr, "claims: a claim reads virtual row %u of tree number %u (%s): there are %zu trees, and the prover's file has %zu pairs for it\n", row, tree,
              tree < trees.size() ? trees[tree].name.c_str() : "?", trees.size(), tree < valuesF->pairs.size() ? valuesF->pairs[tree].size() : (size_t)0);
      refuse(G_ + "internal-claim-values");
    }
    const vector<TestProverValues::TestProverVirtualRow> &pairs = valuesF->pairs[tree];
    if (column >= pairs[0].D.size()) {
      fprintf(stderr, "claims: a claim reads column %llu of virtual row %u of tree %s, whose values have %zu columns\n", (unsigned long long)column, row, trees[tree].name.c_str(), pairs[0].D.size());
      refuse(G_ + "internal-claim-values");
    }
    if (readF.size() != trees.size()) readF.resize(trees.size());
    if (readF[tree].size() != trees[tree].rows.size()) readF[tree].assign(trees[tree].rows.size(), 0);
    readF[tree][row] = 1;
    if (pairs.size() == 1) {
      j.part = j.part + coefficient * pairs[0].D[column];
      return;
    }
    TestUnresolvedRowSums *o = nullptr;
    for (size_t i = j.open.size(); i-- > 0 && !o;)
      if (j.open[i].tree == tree && j.open[i].row == row) o = &j.open[i];
    if (!o) j.open.push_back({tree, row, vector<Ex>(pairs.size(), eZ())}), o = &j.open.back();
    for (size_t p = 0; p < pairs.size(); p++) o->byPair[p] = o->byPair[p] + coefficient * pairs[p].D[column];
  }
  // The form's constant and its entries of virtual rows; not its slots.
  TestClaimValueParts valueOf(const Aff &a) const {
    TestClaimValueParts j;
    j.part = a.c;
    for (const Ent &x : a.e) take(j, x.tree, x.row, x.idx, x.coef);
    return j;
  }
  // Records a claim. `value`: valueOf(a); t2Lin() (vf_subproof_segment.h) adds runs of entries with take().
  void judged(u32 claim, const Ex &kappa, TestClaimValueParts &&value, const Aff &a) { judgedF.push_back({claim, sectionF, kappa, std::move(value), a.s}); }
  // After the last claim: fills falseF and valuesSumF.
  void judgeAll() {
    falseF.clear(), valuesSumF = eZ();
    // partner[t][r]: the pair of array t whose cv equals row r's coef.
    vector<vector<size_t>> partner(trees.size());
    for (size_t t = 0; t < readF.size(); t++) {
      partner[t].assign(readF[t].size(), 0);
      for (size_t r = 0; r < readF[t].size(); r++) {
        if (!readF[t][r]) continue;
        const vector<Ex> &coef = trees[t].rows[r].coef;
        size_t found = valuesF->pairs[t].size();
        for (size_t p = 0; p < valuesF->pairs[t].size() && found == valuesF->pairs[t].size(); p++) {
          const vector<Ex> &cv = valuesF->pairs[t][p].cv;
          bool same = cv.size() == coef.size();
          for (size_t i = 0; same && i < coef.size(); i++) same = cv[i] == coef[i];
          if (same) found = p;
        }
        if (found == valuesF->pairs[t].size()) {
          fprintf(stderr, "claims: claims read virtual row %zu of tree %s, and none of the door's %zu pairs on that array has the row's weights on the array's rows (%zu weights at the end)\n", r,
                  trees[t].name.c_str(), valuesF->pairs[t].size(), coef.size());
          refuse(G_ + "internal-claim-values");
        }
        partner[t][r] = found;
      }
    }
    for (const TestRecordedClaim &j : judgedF) {
      Ex v = j.value.part;
      for (const TestUnresolvedRowSums &o : j.value.open) v = v + o.byPair[partner[o.tree][o.row]];
      for (const auto &x : j.slots) {
        if (x.first >= ns || x.first >= valuesF->mask.size() || !gotF[x.first]) {
          fprintf(stderr, "claims: claim %u (%s) names slot %u: the sub-proof has %u slots, the mask row's values %zu entries, and under that number %s\n", j.claim, j.section.c_str(),
                  x.first, ns, valuesF->mask.size(), x.first < ns && gotF[x.first] ? "something stands" : "NOTHING STANDS at the end");
          refuse(G_ + "internal-claim-values");
        }
        v = v + x.second * (vp[x.first] - mf[x.first] * valuesF->mask[x.first]);
      }
      valuesSumF = valuesSumF + j.kappa * v;
      if (!isZ(v)) falseF.push_back({j.claim, j.section});
    }
  }
  // Adds the claim that the form f is 0.
  void lin(const Aff &f) {
    const u32 number = nClaim;
    const Ex kappa = kap();
    push(kappa, f);
    if (valuesF) judged(number, kappa, valueOf(f), f);
  }
  // The one equation, once all claims are in, written on the mask row itself:
  //   (the rows' L vectors on their rows) + sum_j maskCoef[j] * (mask entry j) + constant = 0.
  // `used`: how many slots the sub-proof's messages have taken. Each must have a value by now: one
  // that was shown, or a coefficient of the hiding polynomial that a sumcheck adds to its summand
  // (U in vf_sumcheck_psc.h), which only pscRow() sets: "internal-slot-unused" rejects a sumcheck without
  // that call. The count of claims and, after owe(), `added` check that the returned form is added.
  void settle(vector<Ex> &maskCoef, Ex &constant, u32 used) const {
    if (used > ns || gotF.size() != ns) refuse(G_ + "internal-slot-marks");
    for (const auto &o : owed)
      if (!o->added) refuse(G_ + "internal-claim-not-added-" + o->what);
    for (u32 j = 0; j < used; j++)
      if (!gotF[j]) refuse(G_ + "internal-slot-unused");
    maskCoef.assign(ns, eZ());
    constant = c0;
    for (u32 j = 0; j < ns; j++) {
      if (isZ(LS[j])) continue;
      if (!gotF[j]) refuse(G_ + "internal-slot-order");
      maskCoef[j] = eZ() - LS[j] * mf[j];
      constant = constant + LS[j] * vp[j];
    }
  }
};
/* A plan (such as `HPlan2`, vf_layers_plan.inc: a sub-proof's arrays and counts, made without reading a
 * proof and copied by its check) keeps its trees without the L vectors to save memory: `trimL`
 * frees them and `fillL` allocates them again, all zero. */
static void trimL(HFork &F) {
  for (auto &X : F.trees)
    for (auto &r : X.rows) vector<Ex>().swap(r.L);
}
static void fillL(HFork &F) {
  for (auto &X : F.trees)
    for (auto &r : X.rows) r.L.assign(X.K, eZ());
}

#include "vf_sumcheck_psc.h"  // Psc, pscBegin, pscEnds, pscRow: the masked sumcheck
#include "vf_sumcheck_receive.h" // roundIn: receives a sumcheck round
#include "vf_sumcheck_token.h"  // tscIn: the token sumcheck
// A product layer: sum_x eq(rho, x) vL(x) vR(x) = C. r: challenges. sL, sR: slots of vL(r), vR(r).
// tau: a challenge. handDown(): the next layer's rho = (tau, r), C = (1 - tau) vL(r) + tau vR(r).
struct ProductLevel {
  vector<Ex> r;
  u32 sL = 0, sR = 0;
  Ex tau = eZ();
  Aff row; // the linear claim that the layer's sumcheck leaves
};
// Runs the mu rounds of one product layer in the given order, then receives vL, vR and draws tau.
static ProductLevel readProductLevelRounds(HFork &F, Tr &tr, u32 &slot, int mu, const vector<Ex> &rho, const Aff &C,
                     const vector<int> &order) {
  ProductLevel o;
  o.r.resize(mu);
  Psc p = pscBegin(F, tr, slot, C);
  for (int j = 0; j < mu; j++) {
    const int v = order[j];
    o.r[mu - 1 - v] = roundIn(F, tr, slot, RoundForm::EQ2, p, rho[mu - 1 - v]);
  }
  o.sL = pscEnds(F, tr, slot, p, 2, 2);
  o.sR = o.sL + 1;
  o.tau = trAnswer(tr, RoundName::LINE);
  o.row = pscRow(F, p, F.shown(o.sL) * F.shown(o.sR)); // the summand is the product of the two end values
  return o;
}
static void handDown(const ProductLevel &o, vector<Ex> &rho, Aff &C) {
  rho.assign(1, o.tau);
  rho.insert(rho.end(), o.r.begin(), o.r.end());
  C = affSlot(o.sL) * (eO() - o.tau) + affSlot(o.sR) * o.tau;
}
// Adds the layer's claim, ProductLevel::row.
static void layerObject(HFork &F, const ProductLevel &o) {
  F.lin(o.row);
}
#ifndef VF_WITHOUT_LOOKUPS  // unit tests under verifier/tests/ define it
#include "vf_lookup_helpers.h"
#endif
