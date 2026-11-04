// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of the lookup argument by product trees (blocks/iop/lkp.h), on hidden values
 * (verifier/src/vf_sumcheck_psc.h). A group of calls (`Call`: 2^n keys, a public table of 2^k rows)
 * runs as one lookup inside the caller's sub-proof (`HFork`).
 * Key side: one vector in trees of 2^dT keys ("forest" in identifiers).
 * Table side: one table, one vector of multiplicities per call.
 * Indexes, high bits first: keys (x | y), x the place in its tree, y the tree's number; table rows
 * (t | c), t the row of a call's table (a shorter table is repeated), c the call's list index;
 * planes b[(t | c | j)], j a multiplicity's bit. Their sizes, and the indexes that no call uses:
 * docs/format/11-1-machinery.md, "A lookup", "The key side's leaves" and "The table side's leaves"
 * (its dL is d).
 * With several calls the verifier first draws a shift theta. Call c's keys and rows are moved by c
 * theta, which is meant to keep a key of one call from passing as a row of another's table.
 * Messages, one run on the caller's transcript (`Tr`, hash chain); one sumcheck (`Psc`) a level:
 *   docs/format/11-1-machinery.md, "A lookup", steps 1 to 9.
 * What the caller must ensure. Nothing here enters the values below into the transcript, and a
 * prover that can choose one of them after a challenge is drawn proves a false lookup.
 * (1) Committed, and their samples answered, before theta (before alpha if there is no theta): the
 *     planes; whatever the key vectors are read from; and the mask row on whose entries all claims
 *     here are tested: each sumcheck's challenge eps must follow it (reason: blocks/iop/psc.h).
 * (2) Fixed by what the transcript contains before that same challenge, every one of:
 *       the calls' order in the list and their number (c in c theta; first, e; which table fills
 *       the numbers from NC on; whether theta is drawn); lid; what each key form KF[c] reads;
 *       of each call n, k, t0 and the table (the range table, or all its 2^k row values);
 *       pub[y] for every y < NT (it enters both the public factor a and the value of the claim on
 *       the number of keys); lm, which committed matrix TB and TP (`forestPlaneClaims`) read and
 *       by which map of rows and columns, and that it has no column for a number c >= NC;
 *       the mask row's root and length, the lookup's first slot, and the number of claims added
 *       before the first of these (the powers of the batching element go by a claim's number);
 *       the constants that `forestShape` tests (32 planes, no heavy rows), the field and hash.
 *     The transcript's plan does not fix these: `forestPlan` enters only v, d, the number of
 *     planes and whether there is a shift. (dT, rK) = (10, 2) and (11, 1), or (kT, cK) = (6, 1) and
 *     (5, 2), have one plan, but other orders of rounds, other weights, other tables.
 * (3) All claims added: `forestObjects`, `forestPlaneLins` (three that read the planes) and, by
 *     the caller, `forestJoin`'s form (on the key vectors). `HFork::settle` rejects unless each
 *     of these three is marked as added. A mark does not see what was added, if anything. */
#pragma once
#include <memory>

#include "numbers/blocks.h"

// One group's lookup: calls, sizes, padding, challenges; points and claims from forestPad().
struct HForest {
  vector<Call *> cl; // The lookup calls (`Call`: vf_common.h) in the caller's list order.
  int dT = 0, kT = 0, NT = 0, rK = 0, NC = 0, cK = 0;
  // Call c (list index) has 2^iota[c] trees, from number first[c]. callOf[y], y < NT: tree y's c.
  vector<int> iota, first, callOf;
  vector<u64> pub; // pub[y], y < NT: the number of padding keys in tree y.
  Ex theta = eZ(), alpha = eZ();
  Ex a = eO();          // the product of alpha - (padding key) over all padding keys
  vector<Aff> rows;   // claims, in order: tie (31), key levels (v), table levels (d), bits (1)
  Psc leaves;         // the table leaves' sumcheck; `forestPlaneClaims` makes its claim
  Ex leavesA = eZ();    // alpha - 1 - (the table at zb[0]): the factor of the planes' value there
  u32 sBit = 0;         // the slot (a sent scalar's number) of the planes' value at zb[1]
  vector<Ex> zx, zy, e; // the key side's point; e[c], list order: weight of call c's trees at zy
  Aff Cend;           // the hidden value of the key side's leaves at (zx | zy)
  vector<Ex> zb[2];     // (t | c | j): the points at which the planes are read
  vector<Ex> tT;        // tT[c], list order: call c's table at zb[0]'s last k_c coordinates of t
  // Marks of the claims that forestPad() leaves to the caller (HFork::owe()); null before that.
  std::shared_ptr<HFork::Owed> owedObjects, owedPlanes, owedJoin;
  int v() const { return dT + rK; }
  int d() const { return kT + cK; }
  int b(int y) const { return y - first[callOf[y]]; } // which tree of its call (not the planes' b)
  u64 emptyKeys() const { return ((1ull << rK) - (u64)NT) << dT; }
  u64 padding() const {
    u64 s = emptyKeys();
    for (u64 x : pub) s += x;
    return s;
  }
  // A padding key of call c, moved by the shift. The trees of padding alone hold call 0's.
  Ex padKey(int c) const { return eB(cl[c]->tauAt(cl[c]->t0)) + eS(theta, (u64)c); }
};
// The last key level's order of rounds (mu = v - 1 variables): x's bits, positions rK and above, as
// in orderRange(), then y's from the top down. Needs mu - rK >= 9: forestShape() tests dT >= 10.
// The order in which a level's rounds bind its variables carries nothing for soundness — a
// sum-check is sound under any order fixed in advance — but it belongs to the format all the
// same: it fixes the sequence in which challenges are drawn and read, so prover and verifier must
// share it and it is among the public things fixed before the first draw; the two special orders
// exist only for fast kernels on narrow keys. (They are this one and orderRange().)
// That this order is fixed before the first challenge is the caller's duty, point (2) at the top:
// it follows from mu and rK, and the transcript's plan (forestPlan()) does not fix rK.
static vector<int> orderForestRange(int mu, int rK) {
  vector<int> o;
  for (int v : {2, 1, 0, 5, 4, 3, 8, 7, 6}) o.push_back(rK + v);
  for (int v = mu - 1; v >= rK + 9; v--) o.push_back(v);
  for (int v = rK - 1; v >= 0; v--) o.push_back(v);
  return o;
}
// A group's sizes and the checks on them. lid caps dT.
static HForest forestShape(const vector<Call *> &cl, int lid) {
  HForest h;
  h.cl = cl;
  for (auto cp : cl) {
    const Call &c = *cp;
    if (c.k < 5 || c.k > 16 || c.fn < 0 || c.fn > 2 || (c.t0 >> c.k) || c.n > 62 || c.n < 0 ||
        (c.tau && c.tau->size() != (1ull << c.k)))
      refuse("shape-pt-words-" + c.name);
    // A row value must be below the prime: some functions here reduce their arguments, some do not.
    if (c.tau)
      for (u64 x : *c.tau)
        if (x >= FIELD_P) refuse("forest-row-value-" + c.name);
    if (c.n < 10) refuse("hid-call-floor");
    if (c.kappa != 5 || !c.heavy.empty()) refuse("hid-planes-" + c.name);
  }
  const int NC = (int)cl.size();
  if (NC < 1 || NC > 16) refuse("forest-calls");
  h.NC = NC;
  while ((1 << h.cK) < NC) h.cK++;
  h.dT = lid;
  for (auto c : cl) {
    h.dT = std::min(h.dT, c->n);
    h.kT = std::max(h.kT, c->k);
  }
  if (h.dT < 10) refuse("forest-depth");
  u64 nt = 0;
  h.iota.resize(NC);
  for (int c = 0; c < NC; c++) {
    h.iota[c] = cl[c]->n - h.dT;
    // n <= 62 and dT >= 10, so a term is at most 2^52 and 16 of them cannot wrap.
    nt += 1ull << h.iota[c];
  }
  if (nt > 256) refuse("forest-trees");
  h.NT = (int)nt;
  while ((1 << h.rK) < h.NT) h.rK++;
  // A 32-bit multiplicity has room for all keys; the multiplicities' sum cannot wrap in the field.
  if (h.v() >= 32) refuse("forest-keys");
  if (((u128)1 << (h.d() + 32)) >= FIELD_P) refuse("no-wrap-pin-forest");
  // Largest calls first (ties keep the list's order), so that first[c] is a multiple of 2^iota[c].
  vector<int> ord(NC);
  for (int c = 0; c < NC; c++) ord[c] = c;
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return h.iota[a] > h.iota[b]; });
  h.first.assign(NC, 0);
  h.callOf.assign(h.NT, -1);
  int at = 0;
  for (int c : ord) {
    h.first[c] = at;
    if (at % (1 << h.iota[c])) refuse("internal-forest-place");
    for (int b = 0; b < (1 << h.iota[c]); b++) h.callOf[at + b] = c;
    at += 1 << h.iota[c];
  }
  if (at != h.NT) refuse("internal-forest-place");
  return h;
}
// Adds count padding places of call c (list index), from place first, to pub (NT entries, by y).
// Place i's tree: docs/format/11-1-machinery.md, "The trees and the calls".
static void forestPubRun(const HForest &h, vector<u64> &pub, int c, u64 first, u64 count) {
  if (pub.size() != (size_t)h.NT || c < 0 || (size_t)c >= h.iota.size()) refuse("internal-forest-pub");
  const u64 N = 1ull << (h.dT + h.iota[c]), D = 1ull << h.dT;
  if (first > N || count > N - first) refuse("internal-forest-pub");
  while (count) {
    const u64 b = first >> h.dT, take = std::min(count, (b + 1) * D - first);
    pub[(u64)h.first[c] + b] += take;
    first += take;
    count -= take;
  }
}
/* Rejects counts of padding keys pub[y] with more padding keys in a tree than places, or with a
   call that is all padding (single trees may be). A caller's key form puts the padding key in at a
   padding place and does not read what is committed there, so a yes would say nothing about such a
   call; `lkp_fault` (blocks/iop/lkp.h) rejects it too. Every caller runs this where its plan fills
   pub, so that runs without a proof reject such a parameter file too. cl: the calls in list order,
   for their names (a plan may have cleared h.cl). prefix: HFork::G_, put before "internal-..". */
static void forestPubAsk(const HForest &h, const vector<Call *> &cl, const vector<u64> &pub, const string &prefix = "") {
  if (pub.size() != (size_t)h.NT || cl.size() != (size_t)h.NC) refuse(prefix + "internal-forest-pub");
  for (u64 x : pub)
    if (x > (1ull << h.dT)) refuse("pt-pub-forest");
  for (int c = 0; c < h.NC; c++) {
    u64 padded = 0;
    for (int b = 0; b < (1 << h.iota[c]); b++) padded += pub[(size_t)h.first[c] + b];
    if (padded >> (h.dT + h.iota[c])) refuse("forest-only-padding-" + cl[c]->name);
  }
}
// Counts the lookup's slots: one per scalar sent. The mask entry with a slot's number is a
// coefficient of a hiding polynomial or the mask of a shown value (vf_sumcheck_psc.h).
static size_t forestElems(const HForest &h) {
  size_t s = sumcheckSlots(RoundForm::EQ2, 0, 3, 3) + 30 * sumcheckSlots(RoundForm::EQ2, 0, 2, 3);
  for (int mu = 0; mu < h.v(); mu++) s += layerSlots(mu);
  for (int l = 0; l < h.d(); l++) s += layerSlots(5 + l);
  // The scalars of docs/format/11-1-machinery.md, "A lookup", step 7:
  s += 1 + 2 * (size_t)h.d() + 5;                                  // the leaves
  return s + sumcheckSlots(RoundForm::EQ2, 5 + h.d(), 1, 2);  // the bits
}
// Adds the lookup's rounds to the transcript's plan.
static void forestPlan(blk::Plan &p, const string &where, const HForest &h) {
  blk::LookupSize x;
  x.v = (unsigned)h.v(), x.d = (unsigned)h.d(), x.mu = 5, x.shifted = h.NC > 1;
  blk::plan_lookup(p, where, x);
}
// The number of claims that forestObjects() adds.
static size_t forestLins(const HForest &h) {
  return 31 + (size_t)h.v() + (size_t)h.d() + 1;
}
// Draws theta (several calls only) and alpha; computes a. Needs h.pub.
static void forestDraws(HForest &h, Tr &tr) {
  if (h.NC > 1) h.theta = trAnswer(tr, RoundName::LKP_SHIFT);
  h.alpha = trAnswer(tr, RoundName::LKP);
  h.a = ePow(h.alpha - h.padKey(0), h.emptyKeys());
  for (int y = 0; y < h.NT; y++) h.a = h.a * ePow(h.alpha - h.padKey(h.callOf[y]), h.pub[y]);
}
// The tie and the point that joins the planes' roots. sZ: the slot of the key side's root.
// rhoT, CT: the top table level's point and start form.
static void forestTie(HFork &F, Tr &tr, u32 &slot, HForest &h, u32 &sZ, vector<Ex> &rhoT, Aff &CT) {
  /* The tie: Z = a * prod_j G_j^(2^j) (Z: the key side's root; G_j: plane j's root), in steps:
     P_31 = G_31, P_j = P_(j+1)^2 G_j, Z = a P_0. None is sent unmasked. What each step shows:
     docs/format/11-1-machinery.md, "Step 3: the tie". */
  u32 sP = 0, sG[32];
  for (int j = 0; j < 31; j++) {
    Psc p = pscBegin(F, tr, slot, j ? affSlot(sP) : affC(eZ()));
    u32 at = pscEnds(F, tr, slot, p, 3, j ? 2 : 3);
    if (!j) sZ = at++;
    sP = at, sG[j] = at + 1;
    const Ex power = F.shown(sP) * F.shown(sP) * F.shown(sG[j]);
    h.rows.push_back(pscRow(F, p, j ? power : F.shown(sZ) - h.a * power));
  }
  sG[31] = sP;
  // The planes' 32 roots are the table side's top level.
  const vector<Ex> byVariable = trPoint(tr, RoundName::LKP_PLANES, 5); // the lowest bit's coordinate first
  rhoT.assign(byVariable.rbegin(), byVariable.rend());
  const vector<Ex> ej = eqTab(rhoT.data(), 5);
  CT = Aff();
  for (int j = 0; j < 32; j++) affAdd(CT, affSlot(sG[j]), ej[j]);
}
// Runs the lookup's messages (order and caller's duties: at the top). Sets each call's alpha, zk.
static HForest forestPad(HFork &F, Tr &tr, const vector<Call *> &cl, const vector<u64> &pub, u32 &slot, int lid) {
  HForest h = forestShape(cl, lid);
  const int NC = h.NC, dT = h.dT, kT = h.kT, rK = h.rK, cK = h.cK, v = h.v(), d = h.d(), m = 5 + d;
  forestPubAsk(h, cl, pub, F.G_);
  h.pub = pub;
  const u32 slot0 = slot;
  h.owedObjects = F.owe("look-up-mask-claims"), h.owedPlanes = F.owe("look-up-plane-claims"), h.owedJoin = F.owe("look-up-join");
  forestDraws(h, tr);
  u32 sZ = 0;
  vector<Ex> rhoT;
  Aff CT;
  forestTie(F, tr, slot, h, sZ, rhoT, CT);
  {
    vector<Ex> rho;
    Aff C = affSlot(sZ);
    for (int mu = 0; mu < v; mu++) {
      const ProductLevel o = readProductLevelRounds(F, tr, slot, mu, rho, C, mu == v - 1 ? orderForestRange(mu, rK) : orderDefault(mu));
      handDown(o, rho, C);
      h.rows.push_back(o.row);
    }
    h.zx.assign(rho.begin(), rho.begin() + dT);
    h.zy.assign(rho.begin() + dT, rho.end());
    h.Cend = C;
  }
  for (int l = 0; l < d; l++) {
    const ProductLevel o = readProductLevelRounds(F, tr, slot, 5 + l, rhoT, CT, orderDefault(5 + l));
    handDown(o, rhoT, CT);
    h.rows.push_back(o.row);
  }
  if (rhoT.size() != (size_t)m) refuse(F.G_ + "internal-forest-point");
  // The table leaves are 1 + b (alpha - 1 - row): affine in the planes, so no value is shown.
  // Format: docs/format/11-1-machinery.md, "Step 7: the leaves".
  h.leaves = pscBegin(F, tr, slot, CT);
  h.zb[0].resize(m);
  for (int j = 0; j < m; j++) h.zb[0][j] = roundIn(F, tr, slot, RoundForm::EQ, h.leaves, rhoT[j], j < d ? 2 : 1);
  // The bits are bits: sum eq(point, .) (b^2 - b) = 0 at a fresh point.
  {
    const vector<Ex> byVariable = trPoint(tr, RoundName::ZERO, m);
    const vector<Ex> r5(byVariable.rbegin(), byVariable.rend());
    Psc p = pscBegin(F, tr, slot, affC(eZ()));
    h.zb[1].resize(m);
    for (int j = 0; j < m; j++) h.zb[1][j] = roundIn(F, tr, slot, RoundForm::EQ2, p, r5[j]);
    h.sBit = pscEnds(F, tr, slot, p, 2, 1);
    h.rows.push_back(pscRow(F, p, F.shown(h.sBit) * F.shown(h.sBit) - F.shown(h.sBit)));
  }
  if (slot != slot0 + (u32)forestElems(h) || h.rows.size() != forestLins(h)) refuse(F.G_ + "internal-slot-count");
  // e_c = eq(zy's first hi coordinates, the top hi bits of first[c]); zk_c = (the rest of zy | zx).
  h.e.assign(NC, eZ());
  h.tT.assign(NC, eZ());
  for (int c = 0; c < NC; c++) {
    Call &k = *cl[c];
    const int hi = rK - h.iota[c];
    if (hi < 0) refuse(F.G_ + "internal-forest-place");
    const u64 pre = (u64)h.first[c] >> h.iota[c];
    Ex e = eO();
    for (int i = 0; i < hi; i++) e = e * (((pre >> (hi - 1 - i)) & 1) ? h.zy[i] : eO() - h.zy[i]);
    h.e[c] = e;
    k.alpha = h.alpha;
    k.zk.assign(h.zy.begin() + hi, h.zy.end());
    k.zk.insert(k.zk.end(), h.zx.begin(), h.zx.end());
    if (k.zk.size() != (size_t)k.n) refuse(F.G_ + "internal-forest-point");
    const Ex *u = h.zb[0].data() + (kT - k.k);
    Ex tt = eZ();
    if (!k.tau) {
      for (int i = 0; i < k.k; i++) tt = tt + eS(u[i], 1ull << (k.k - 1 - i));
    } else {
      const vector<Ex> et = eqTab(u, k.k);
      for (size_t t = 0; t < et.size(); t++) tt = tt + eS(et[t], (*k.tau)[t]);
    }
    h.tT[c] = tt;
  }
  {
    const vector<Ex> ec = eqTab(h.zb[0].data() + kT, cK);
    Ex row = eZ();
    for (int c = 0; c < (1 << cK); c++) {
      const int of = std::min(c, NC - 1);
      row = row + ec[c] * (h.tT[of] + eS(h.theta, (u64)of));
    }
    h.leavesA = h.alpha - eO() - row;
  }
  return h;
}
// Adds the claims on mask entries only (h.rows).
static void forestObjects(HFork &F, const HForest &h) {
  if (!h.owedObjects) refuse("internal-forest-no-proof-read");
  // Section names, for tests that list the false claims (../tests/claims_report.h).
  const string before = F.sectionF, whose = before.empty() ? "" : before + ", ";
  const size_t v = (size_t)h.v(), d = (size_t)h.d();
  for (size_t i = 0; i < h.rows.size(); i++) {
    F.section(whose + "look-up: " + (i < 31 ? "tie " + std::to_string(i) : i < 31 + v ? "key level " + std::to_string(i - 31) : i < 31 + v + d ? "table level " + std::to_string(i - 31 - v) : string("bits")));
    F.lin(h.rows[i]);
  }
  F.section(before);
  h.owedObjects->added = true;
}
// Section names of the plane claims and the join.
static const char *const FOREST_PLANE_SECTIONS[3] = {"look-up: leaves", "look-up: planes at the bits' point", "look-up: number of keys"};
static const char *const FOREST_JOIN_SECTION = "look-up: join";
// The padding is described twice: by the counts pub[y] and by the places where the caller's key
// forms put in the padding key (or read a cell that must hold it), which its own loops count in
// `laid`[y]. Both are public, so a difference is a bug, not an attack. A padding key is a row of
// its call's table, so (argued, not proved) a difference cannot let in a key from outside a table.
// In the layer and output head both come from the lookup segments' column lists: an error in a key
// form is not seen. settle() does not ask for this.
static void forestPaddingLaid(const HForest &h, const vector<u64> &laid) {
  if (laid != h.pub) refuse("internal-forest-padding-places");
}
// For a caller that itself adds the plane claims or the join. A mark does not see what was added.
static void forestPlanesAdded(const HForest &h) {
  if (!h.owedPlanes) refuse("internal-forest-no-proof-read");
  h.owedPlanes->added = true;
}
static void forestJoinAdded(const HForest &h) {
  if (!h.owedJoin) refuse("internal-forest-no-proof-read");
  h.owedJoin->added = true;
}
/* The claim on the calls' key vectors: the key side's leaves are alpha - (shifted key), so
     sum_c e_c (alpha - c theta - KF_c) + sum_(y >= NT) eq(zy, y) (alpha - padding key) - C_end = 0,
   KF[c], list order: the caller's form for call c's key vector at zk_c (not checked here). */
static Aff forestJoin(const HForest &h, const vector<Aff> &KF) {
  if (KF.size() != h.cl.size() || h.e.size() != h.cl.size()) refuse("internal-forest-join");
  forestJoinAdded(h); // marked where the form is made: that the caller also adds it is not seen
  Aff f;
  for (size_t c = 0; c < KF.size(); c++) affAdd(f, affC(h.alpha - eS(h.theta, (u64)c)) - KF[c], h.e[c]);
  const vector<Ex> ey = eqTab(h.zy.data(), h.rK);
  for (int y = h.NT; y < (1 << h.rK); y++) f.c = f.c + ey[y] * (h.alpha - h.padKey(0));
  return f - h.Cend;
}
// The number of columns of the planes' matrix of 2^lm rows.
static u64 forestPlaneCols(const HForest &h, int lm) {
  if (lm < 5 || lm > h.kT + 5) refuse("forest-planes");
  return (u64)h.NC << (h.kT + 5 - lm);
}
/* The three claims that read the planes, in the order of docs/format/11-1-machinery.md, "The three
 * claims on the planes". The planes' matrix and the caller's forms TB(e, col), TP(col) on it (row
 * weights: `forestPhiB` at e, `forestPhiP`): docs/format/11-1-machinery.md, "The planes' value as
 * a form". The caller adds all three and marks them. */
template <class FB, class FP> static vector<Aff> forestPlaneClaims(HFork &F, HForest &h, int lm, FB TB, FP TP) {
  (void)forestPlaneCols(h, lm);
  const int w = h.kT + 5 - lm;
  Aff at[2];
  for (int e = 0; e < 2; e++) {
    const vector<Ex> ec = eqTab(h.zb[e].data() + h.kT, h.cK), et = eqTab(h.zb[e].data(), w);
    for (u64 c = 0; c < (u64)h.NC; c++)
      for (u64 th = 0; th < (1ull << w); th++) affAdd(at[e], TB(e, (c << w) + th), ec[c] * et[th]);
  }
  Aff f = affC(eZ() - eB((1ull << h.v()) - h.padding()));
  for (u64 col = 0; col < ((u64)h.NC << w); col++) f = f + TP(col);
  return {pscRow(F, h.leaves, eO(), at[0] * h.leavesA), affSlot(h.sBit) - at[1], f};
}
template <class FB, class FP> static void forestPlaneLins(HFork &F, HForest &h, int lm, FB TB, FP TP) {
  const string before = F.sectionF, whose = before.empty() ? "" : before + ", ";
  const vector<Aff> claims = forestPlaneClaims(F, h, lm, TB, TP);
  if (claims.size() != 3) refuse("internal-forest-join");
  for (size_t i = 0; i < 3; i++) F.section(whose + FOREST_PLANE_SECTIONS[i]), F.lin(claims[i]);
  F.section(before);
  forestPlanesAdded(h);
}
// The row weights phiB_e and phiP of docs/format/11-1-machinery.md, "The planes' value as a form".
static vector<Ex> forestPhiB(const HForest &h, int lm, int e) {
  (void)forestPlaneCols(h, lm);
  vector<Ex> z(h.zb[e].begin() + (h.kT + 5 - lm), h.zb[e].begin() + h.kT);
  z.insert(z.end(), h.zb[e].begin() + h.kT + h.cK, h.zb[e].end());
  return eqTab(z.data(), lm);
}
static vector<Ex> forestPhiP(int lm) {
  vector<Ex> p(1ull << lm);
  for (u64 r = 0; r < p.size(); r++) p[r] = eB(1ull << (r & 31));
  return p;
}
