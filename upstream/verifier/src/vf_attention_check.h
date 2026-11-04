// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks the attention sub-proof A_g of layer group g: scores, causal softmax and weighted sum.
// Format: docs/format/11-3-attention-segment.md. Its SE, VM, VD, SF, SD, HB, Mx, ib, Zd are scE,
// vpm, vpd, scF, scd (SCD), scH, m, ix, Z here. Fill AttentionSubProof::A.cf and `in`, call plan(),
// then run() with currentSegment (vf_subproof_segment.h) set to A_g's segment of the proof (`link`
// in identifiers). run() reads A_g's messages from the request's first hash chain, through a `Tr`
// (vf_common.h), and adds linear claims. It tests none, so its return accepts nothing:
// doorH() passes their weighted sum to the chain, which tests it at its end. To reject, the code
// calls refuse(name), which exits. A name with "internal-" marks a self-check of this program: the
// call then ends NOT CHECKED, not NO. A `tree` is 2^dT keys of a lookup (`forest`), or a committed
// array. A message slot (`slot`) numbers a scalar that the prover sends and its mask entry
// (vf_sumcheck_psc.h). "Claim n" is a group of claims, numbered as in F.section()'s strings (3 to
// 8, 10); "A-x.y": item y of claim x. At query token i: s = <q, kx> is the score of key token j; m,
// the row maximum, attained at j = ix; e, the softmax weight; Z, the sum of e; acc, the sum of e v;
// o, acc / Z rounded.
#pragma once
#include "vf_attention_shape.h"
#include "vf_smallness_matrices.h"
#include "vf_subproof_engine.h"
#include "vf_smallness_test.h"
#include "attn/plan_rounds.h" // prover/attn/plan_rounds.h: one source for prover and verifier
#include <functional>
#include <map>

// Returns the shape of A_g's one lookup group, all of A.calls. It runs A.shape(), so only A.cf must
// be filled, and reads no proof. It rejects a call that is all padding, and sets pub, each call's
// iota and A.kbA, the columns of bA_g (multiplicity bits).
static HForest aforkShapeHid(AFork &A) {
  A.shape();
  vector<Call *> cl;
  for (auto &c : A.calls) cl.push_back(&c.c);
  HForest h = forestShape(cl, LIDB);
  // run() builds one table, TA (TO), from the last lmS (lt + ldh) coordinates of several calls' key
  // points (zk). Only the last dT coordinates, zx, are the same in all calls.
  if (h.dT < std::max(A.cf.lmS, A.cf.lt + A.cf.ldh)) refuse("forest-depth-A");
  A.kbA = forestPlaneCols(h, A.cf.lmb); // rejects unless 5 <= lmbA <= kT + 5 ("forest-planes")
  for (size_t q = 0; q < A.calls.size(); q++) A.calls[q].c.iota = h.iota[q];
  // Call c's padding keys are its keys tot_c .. 2^n_c - 1; pub[y] counts those in tree y.
  h.pub.assign((size_t)h.NT, 0);
  for (size_t q = 0; q < A.calls.size(); q++)
    forestPubRun(h, h.pub, (int)q, A.calls[q].tot, (1ull << A.calls[q].c.n) - A.calls[q].tot);
  forestPubAsk(h, cl, h.pub); // here, on the public numbers, and not only when a proof is read
  return h;
}
// plan()'s and run()'s calls into the chain and into the approximate range check on the committed
// cells (`tail`, `tl`; printed: "smallness test").
struct AttentionSeam {
  HTailF2 tl;
  // m: the range check's matrices. The one of kind 1, qa_g (q, kx, v, o), is in an array that the
  // layer sub-proof, its owner, committed. It must fill the array: col0F, its first column there,
  // must be 0, and ktO, the array's column count, its k. aO counts the sub-proofs that use it.
  void shape(vector<TMat> m, u32 tau, u64 col0F, u64 ktO, u32 aO) {
    u64 kq = 0;
    for (auto &M : m) {
      if (M.kind == 1) {
        M.foreign = 0;
        M.ttOff = 0;
        kq = M.k;
      } else if (M.kind != 0)
        refuse("hid-a-weight-matrix");
    }
    if (col0F || ktO != kq) refuse("internal-a-TX-place");
    T2Foreign f;
    f.name = "PXo";
    f.K = ktO;
    f.a = aO;
    // f.m, f.wp: A_g's values for qa_g. vf_layers_plan.inc compares the resulting shape with the owner's.
    for (auto &M : m)
      if (M.kind == 1) {
        f.m = M.m;
        f.wp = M.wp;
      }
    const bool hasU[4] = {false, false, true, false}; // of U1 to U4, A_g has only U3
    tl.shape(m, vector<int>(m.size(), -1), {f}, hasU, tau);
  }
  // The index in HFork::trees of U3, the evaluation tables (weighted sums of a matrix's rows).
  u32 iU3() const { return (u32)tl.ix.U[2]; }
  // The number of scalars that the range check sends.
  size_t smallElems() const { return tl.smallElems(); }
  // Fp, Tp: the plan; T: a run's copy. PK_i: own matrix i; ZZ, BZ: the range check's; PXo: qa_g.
  HFork Fp;
  Tail2 Tp, T;
  u64 nU[3] = {0, 0, 0};
  void plan(u64 nU3, u32 hb) {
    nU[2] = nU3;
    Fp = HFork();
    Tp = Tail2();
    tl.addTrees(Fp, Tp, nU, {}, hb);
    if (Fp.trees.size() != 1 + tl.pkMat.size() + 2 + 1) refuse("internal-a-tree-list"); // The list: U3, PK_0 .., ZZ, BZ, PXo.
    t2Plan(Fp, Tp);
  }
  // Starts a run. The chain has already received and sampled the segment's first oracle messages.
  void start(HFork &F, u32 ns) {
    F.trees = Fp.trees;
    fillL(F); // the plan keeps no L vector
    T = Tp;
    tl.funClosed = false;
    tl.foldDrawn = 0;
    tl.runT = &T;
    F.slots(ns);
  }
  // Reads the range check's messages.
  void transcript(HFork &F, Tr &tr, u32 &slot) { tl.transcript(F, tr, slot); }
  // Draws the element whose powers batch the claims, after the last message. used: the
  // slots taken so far, which must be all. Claims can be added only after batch().
  void batch(HFork &F, Tr &tr, u32 used) { t2Batch(F, T, tr, used); }
  // Adds the range check's claims.
  void objects(HFork &F, const vector<HFun> &fun) { tl.objectsF(F, fun); }
  // Runs after all claims are added up into one equation: sets coef, a virtual row's (HRow) weights
  // on an array's rows, and passes the equation to the chain.
  void finish(HFork &F, Tr &tr) {
    // No entry of U3 and no column that a virtual row is meant to read may have weight (L) 0.
    tl.selfTest(F, nU);
    tl.coefficients(F);
    // The range check's tiling self-test reads the claims' coefficients and the rows' weights.
    tl.rkSelfTest(F);
    doorH(F, T, tr, tl.equationsLaidF());
  }
};

// One batching element, and its powers 0 .. n - 1.
static vector<Ex> coinsOm(Tr &tr, size_t n) {
  return trBatch(tr, n);
}

#include "vf_sumcheck_attention.h" // the sumchecks SC3 and SCD

struct AttentionSubProofInputs {
  u32 hb = 0, gid = 0;
  u64 ktO = 0, col0Q = 0;
  u32 aO = 0;
  u64 midQ = 0;
  // hb: for the arrays' placement records. gid: A_g's number. ktO, col0Q, aO: AttentionSeam::shape().
  // midQ: qa_g's TMat::aId. Next, TMat::wp by matrix: no parameter-file line sets it, so all are 0.
  u32 wpS = 0, wph = 0, wpo = 0, wpbA = 0, wpq = 0;
};
struct AttentionSubProof {
  AFork A;
  // aforkShapeHid(A); plan() sets it. It points to the calls in this object's A, so a copy of this
  // object must run plan() again before it is used.
  HForest fs;
  AttentionSubProofInputs in;
  AttentionSeam seam;
  u32 ns = 0;
  size_t nMsg = 0, nLin = 0, NC = 0, NP = 0, nKRp = 0, nKFp = 0; // NP: pairs (layer, query head a)
  /* A_g's rounds, for the plan of the request's first chain. `rounds` adds those that follow commit
     point `moment` of the segment: 0, the cells; 1, U3; 2, the range check's digits. `rounds`
     points to this object: see fs. */
  blk::AttentionRounds own;
  std::function<void(blk::Plan &, unsigned moment)> rounds;
  u64 nW = 0;
  u32 nClaim = 0;
  // The last run's timings. Nothing in verifier/src reads them.
  vector<std::pair<string, double>> times;
  bool holds(const ACall &c, int seg) const { return std::find(c.segs.begin(), c.segs.end(), seg) != c.segs.end(); }
  size_t npOf(const ACall &c, int seg) const {
    size_t n = 0;
    for (auto &pc : c.pcs) n += pc.seg == seg;
    return n;
  }
  // Reads no proof. A.cf and `in` must be filled. cf.Ty, a cell's bound in A.shape()'s no-overflow
  // tests, must be the range check's.
  void plan() {
    if (!in.hb) refuse("hid-hb");
    if (!in.aO) refuse("hid-a-owner-uses");
    if (A.cf.Ty != ((u128)1 << (1 + LBZ))) refuse("internal-a-Ty"); // Ty must equal 2 beta_z.
    fs = aforkShapeHid(A);
    const ACfg &c = A.cf;
    const u64 T = A.T, kPart = A.kAs / c.NAs;
    NC = A.calls.size();
    NP = A.Lam * A.HA;
    // The range check's matrices and their cells: docs/format/11-3-attention-segment.md, "The
    // arrays and what each cell holds".
    vector<TMat> m;
    auto mat = [&](const string &n, u64 mm, u64 k, int kind, u64 mid, u32 wp) {
      TMat M;
      M.name = n;
      M.m = mm;
      M.k = k;
      M.kind = kind;
      M.aId = mid;
      M.wp = wp;
      if (kind == 1) M.foreign = 0;
      m.push_back(M);
    };
    for (u64 q = 0; q < c.NAs; q++)
      mat("as" + std::to_string(q), 1ull << c.lmS, kPart, 0, (u64)in.gid * 256 + q, in.wpS);
    mat("ah", c.wcH * T, A.fillH / c.wcH, 0, (u64)in.gid * 256 + c.NAs, in.wph);
    mat("ao", c.wcO * T, A.fillO / c.wcO, 0, (u64)in.gid * 256 + c.NAs + 1, in.wpo);
    mat("bA", 1ull << c.lmb, A.kbA, 0, (u64)in.gid * 256 + c.NAs + 2, in.wpbA);
    mat("qa", c.wcQ * T, A.fillQ / c.wcQ, 1, in.midQ, in.wpq);
    if (c.NAs + 4 > 256) refuse("hid-a-fold-list");
    seam.shape(m, T2TAU, in.col0Q, in.ktO, in.aO);
    // Count the slots (msg). A lookup segment (`sRM` to `sIX`) is a named list of a call's keys.
    const size_t lt = c.lt, Lam = A.Lam;
    size_t msg = 0, sumNp = 0;
    // The slots: docs/format/11-3-attention-segment.md, "The rounds of the segment", rows 3 to 15
    // and 21.
    msg += forestElems(fs); // all slots of the group's lookup
    nKRp = nKFp = 0;
    own = blk::AttentionRounds();
    for (auto &k : A.calls) {
      if (k.kind == "KR")
        for (int s : {sRM, sRD})
          if (holds(k, s)) {
            nKRp++;
            msg += 3;
          }
      if (k.kind == "KFD")
        for (int s : {sFM, sFD})
          if (holds(k, s)) {
            nKFp++;
            const size_t np = npOf(k, s);
            sumNp += np;
            msg += 7 + sc3Slots(nvKF(s), np);
            own.score.push_back({(unsigned)nvKF(s), (unsigned)np});
          }
    }
    msg += 7 + lt + sc3Slots(nvE(), NP);
    msg += 2 + sc3Slots(nvVM(), NP) + sc3Slots(nvVD(), NP);
    msg += NP + sc3Slots(nvF(), NP) + sumcheckSlotsDegrees(scdDegrees(), Lam + NP * lt, (int)lt + 1) +
           sc3Slots(nvH(), NP);
    msg += seam.smallElems();
    nMsg = msg;
    // Count the claims (nLin). Pair b < NP: (layer lam, query head a < HA of each key/value head).
    {
      const size_t segH = c.wcH, segO = (size_t)(c.wcO * T >> c.la), segQ = (size_t)(c.wcQ * T >> c.la);
      // The terms in order: claims 1, 2 (the lookup's); 3 to 5; A-J, the join of the keys to the
      // lookup; 6; 7 and 8; 9 (3 claims); one claim for each of the nKFp + 5 SC3s; one for the SCD.
      nLin = forestLins(fs) + 3 * nKRp + (6 * nKFp + 2 * sumNp) + (5 + lt + 2 * NP) + 1 + (1 + 4 * NP) +
             (4 * NP + Lam + NP * lt) + 3 + (nKFp + 5 + 1);
      if (seam.tl.NM != c.NAs + 4) refuse("internal-a-claim-count");
      // lins() takes the number of reads. A table in U3 with a vector of length L on a matrix of m
      // rows has m / L: segH, segO, segQ on ah_g, ao_g, qa_g, and 1 on a part of as_g and on bA_g.
      const size_t reads = 3 * c.NAs + (nKFp + 6) * segH + segO + (2 * nKFp + 5) * segQ + 3;
      nLin += seam.tl.lins(reads);
      // The range check's equations have no challenge of their own: they get further powers
      // of the element that batches the claims, so that element's round counts them too.
      own.rows = nLin + seam.tl.equationsF(reads);
    }
    own.lt = (unsigned)lt, own.lgk = (unsigned)c.lgk, own.pairs = (unsigned)NP, own.Lam = (unsigned)Lam;
    own.plane_messages = (unsigned)nKRp;
    own.e = (unsigned)nvE(), own.value_m = (unsigned)nvVM(), own.value_d = (unsigned)nvVD(),
    own.attained = (unsigned)nvF();
    own.bits = (unsigned)nvH();
    rounds = [this](blk::Plan &p, unsigned moment) {
      const string where = "attention " + std::to_string(in.gid);
      if (moment == 0) {
        forestPlan(p, where + ", look-up", fs);
        blk::plan_attention_own(p, where, own);
      }
      if (moment >= 1) seam.tl.planRounds(p, where + ", smallness test", moment - 1); // its matrix; then the rest
      if (moment == 2) blk::plan_attention_end(p, where, own);
    };
    if (msg >= (1ull << 30)) refuse("hid-a-slots");
    nW = A.tableWords() / WPE;
    seam.plan(nW, in.hb);
    ns = (u32)msg;
  }
  /* An SC3 (vf_sumcheck_attention.h) is the sumcheck of sum_b sum_x W_b(x) A_b(x) B_b(x), W_b
     public; A*_b, B*_b: its end values. `nvKF` to `nvH` give its number of variables, in plan() and
     in run(). */
  int nvKF(int seg) const { return seg == sFM ? A.cf.lgk + 2 * A.lnT - 1 + A.cf.ldh : A.cf.lgk + A.lnT + A.cf.ldh; }
  int nvE() const { return A.cf.lgk + A.cf.lt; }
  int nvVM() const { return A.cf.lgk + 2 * A.lnT - 1 + A.cf.lB; }
  int nvVD() const { return A.cf.lgk + A.lnT + A.cf.lB; }
  int nvF() const { return A.cf.lt + A.cf.lgk + A.cf.ldh; }
  int nvH() const { return A.cf.lt + A.cf.lgk + 4; }
  // The SCD's rounds and their degrees: docs/format/11-3-attention-segment.md, "The sumcheck SD".
  u64 hsScd() const { return 2 * (u64)A.cf.lt + (u64)A.cf.lgk; }
  vector<int> scdDegrees() const {
    const int lt = A.cf.lt, lgk = A.cf.lgk;
    vector<int> d;
    for (int j = 0; j < (int)hsScd(); j++) d.push_back(j < lt ? 2 : j < lt + lgk ? lt + 2 : lt + 1);
    return d;
  }
  /* Checks the sub-proof. G_: a prefix for rejection names. `tr` only wraps the channel of the
     request's first hash chain, which reads the proof itself. That the proof contains exactly this
     sub-proof's scalars is not tested here: the channel compares every round's message with the
     plan (`rounds`); `check_proof` (blocks/frame/check.h) ends each chain, and the last chain's
     `finish` rejects bytes left over (blocks/chain/rounds.h).
     Steps 1 and 3 to 10 below: this file's own numbers, which only mark the order. */
  void run(const string &G_) {
    const ACfg &cf = A.cf;
    const int lt = cf.lt, lB = cf.lB, lgk = cf.lgk, ldh = cf.ldh, lmS = cf.lmS, lnT = A.lnT, ls = A.ls, nM = A.nM,
              nD = A.nD;
    const u64 H = cf.H, T = A.T, B = A.B, nT = A.nT, hP = A.hP, G = A.G, HA = A.HA, dh = A.dh, dq = A.dq, Lam = A.Lam;
    vector<ACall> &calls = A.calls;
    double tl0 = now();
    times.clear();
    auto lap = [&](const char *n) {
      const double t = now();
      times.push_back({n, t - tl0});
      tl0 = t;
    };
    A.bindTau();
    HFork F;
    F.G_ = G_;
    // gid is not sent: it follows from the configuration, whose digest is in the chain's first
    // query (its first hash input).
    if (!currentSegment || !currentSegment->V) refuse("plan-link");
    Tr tr{currentSegment->V->ch};
    seam.start(F, ns);
    u32 slot = 0;
    // Step 1 (the next is step 3): the lookup's messages, for all calls as one group. They set
    // every call's key point zk.
    vector<Call *> cl1;
    for (auto &c : calls) cl1.push_back(&c.c);
    HForest g1 = forestPad(F, tr, cl1, fs.pub, slot, LIDB);
    lap("start, calls of group 1");
    struct PW {
      u64 lam = 0, a = 0;
      Ex wp;
    };
    // weighed[q][i]: pieces() has returned piece i of call q.
    vector<vector<char>> weighed(calls.size());
    for (size_t q = 0; q < calls.size(); q++) weighed[q].assign(calls[q].pcs.size(), 0);
    auto pieces = [&](const ACall &c, int seg) {
      vector<PW> v;
      for (auto &pc : c.pcs)
        if (pc.seg == seg) {
          weighed[(size_t)(&c - &calls[0])][(size_t)(&pc - &c.pcs[0])] = 1;
          PW x;
          x.lam = pc.lam;
          x.a = pc.a;
          x.wp = eqAt(c.c.zk.data(), c.c.n - pc.b, pc.off >> pc.b);
          v.push_back(x);
        }
      std::stable_sort(v.begin(), v.end(),
                       [](const PW &x, const PW &y) { return x.lam != y.lam ? x.lam < y.lam : x.a < y.a; });
      return v;
    };
    /* Pieces, their weights wp and inner points: docs/format/11-3-attention-segment.md, "The calls:
       tables, keys and lengths". `off` follows the placement in AFork::shape, but sums take the
       pieces by (lam, a): hence the sort. */
    auto inner = [&](const ACall &c, int seg) {
      const int b = A.segB(seg);
      return vector<Ex>(c.c.zk.end() - b, c.c.zk.end());
    };
    auto sumW = [&](const vector<PW> &v) {
      Ex s = eZ();
      for (auto &x : v) s = s + x.wp;
      return s;
    };
    // Reads n hidden values sent outside a sumcheck; returns the first's slot.
    auto send = [&](size_t n) {
      return F.claimsIn(tr, slot, (u32)n);
    };
    // A lookup segment's np plane values, from slot s0: plane x of as_g at Z. isD: region D, not M.
    struct PlaneClaim {
      size_t call = 0;
      int seg = 0;
      vector<Ex> Z;
      bool isD = false;
      vector<PW> pw;
      u32 s0 = 0;
      int np = 0;
    };
    // Step 3. 3 plane values per RM or RD (M: score tiles of side B below the diagonal; D: on it).
    vector<PlaneClaim> pcl;
    for (size_t q = 0; q < NC; q++)
      if (calls[q].kind == "KR")
        for (int seg : {sRM, sRD}) {
          if (!holds(calls[q], seg)) continue;
          PlaneClaim p;
          p.call = q;
          p.seg = seg;
          p.Z = inner(calls[q], seg);
          p.isD = seg == sRD;
          p.pw = pieces(calls[q], seg);
          p.s0 = send(3);
          p.np = 3;
          pcl.push_back(p);
        }
    // Step 4. Per lookup segment FM or FD: 5 plane values, then m's and s's values; then s's SC3.
    struct KF {
      size_t call = 0, claim = 0;
      int seg = 0;
      vector<PW> pw;
      vector<Ex> zg, zP, zs, zI, zi, zj, WI, W;
      u32 s0 = 0;
      Degree3SumcheckState sc;
      u64 tM = 0, tQ = 0, tK = 0;
    };
    vector<KF> kfs;
    for (size_t q = 0; q < NC; q++)
      if (calls[q].kind == "KFD")
        for (int seg : {sFM, sFD}) {
          if (!holds(calls[q], seg)) continue;
          KF k;
          k.call = q;
          k.seg = seg;
          k.pw = pieces(calls[q], seg);
          const vector<Ex> Z = inner(calls[q], seg);
          const Ex *z = Z.data();
          k.zg.assign(z, z + lgk);
          z += lgk;
          if (seg == sFM) {
            k.zP.assign(z, z + lnT - 1);
            z += lnT - 1;
            k.zs.assign(z, z + lnT);
            z += lnT;
          } else {
            k.zI.assign(z, z + lnT);
            z += lnT;
          }
          k.zi.assign(z, z + lB);
          z += lB;
          k.zj.assign(z, z + lB);
          k.s0 = send(7);
          PlaneClaim p;
          p.call = q;
          p.seg = seg;
          p.Z = Z;
          p.isD = seg == sFD;
          p.pw = k.pw;
          p.s0 = k.s0;
          p.np = 5;
          k.claim = pcl.size();
          pcl.push_back(p);
          k.WI.assign(nT, eZ());
          if (seg == sFM) {
            const vector<Ex> eP = eqTab(k.zP.data(), lnT - 1), es = eqTab(k.zs.data(), lnT);
            for (u64 P = 0; P < hP; P++)
              for (u64 sg = 0; sg + 1 < nT; sg++) {
                u64 I = 0, J = 0;
                A.cover(P, sg, I, J);
                k.WI[I] = k.WI[I] + eP[P] * es[sg];
              }
          } else
            k.WI = eqTab(k.zI.data(), lnT);
          k.sc = sc3In(F, tr, slot, nvKF(seg), k.pw.size(), affSlot(k.s0 + 6));
          {
            // k.W: the SC3's weights at its end point r, each piece's wp times f. For FM, f
            // leaves out sg = nT - 1, where no row P of region M has a tile (AFork::cover()).
            const Ex *r = k.sc.r.data();
            Ex f = eqPts(k.zg.data(), r, lgk);
            if (seg == sFM)
              f = f * eqPts(k.zP.data(), r + lgk, lnT - 1) *
                  (eqPts(k.zs.data(), r + lgk + lnT - 1, lnT) -
                   onesOf(k.zs.data(), lnT) * onesOf(r + lgk + lnT - 1, lnT));
            else
              f = f * eqPts(k.zI.data(), r + lgk, lnT);
            k.W.resize(k.pw.size());
            for (size_t b = 0; b < k.W.size(); b++) k.W[b] = k.pw[b].wp * f;
          }
          sc3Row(F, k.sc, k.W, calls[q].c.name);
          kfs.push_back(std::move(k));
        }
    // Step 5. ACC, ZO, ZC: acc, Z o, Z at WL's inner point; RH, RHP: rh, rhp there. The SC3 of Z o.
    ACall *cWL = nullptr, *cIX = nullptr, *cW = nullptr;
    for (auto &c : calls) {
      if (holds(c, sWL)) cWL = &c;
      if (holds(c, sIX)) cIX = &c;
      if (holds(c, sWH)) cW = &c;
    }
    if (!cWL || !cIX || !cW) refuse("internal-af-calls");
    const vector<PW> pwW = pieces(*cWL, sWL), pwX = pieces(*cIX, sIX), pwH = pieces(*cW, sWH);
    if (pwW.size() != NP || pwX.size() != NP || pwH.size() != NP) refuse("internal-af-calls");
    const vector<Ex> ZW = inner(*cWL, sWL), ZX = inner(*cIX, sIX), ZH = inner(*cW, sWH);
    const Ex zsl = ZW[0];
    const Ex *zgW = ZW.data() + 1, *ziW = zgW + lgk, *zcW = ziW + lt, *zgX = ZX.data(), *ziX = zgX + lgk;
    // IB_k, RH2, RHP2: docs/format/11-3-attention-segment.md, "The scalars by name", row 5.
    const Ex *zgH = ZH.data(), *ziH = zgH + lgk, *zcH = ziH + lt;
    const u32 s5 = send(5), sIB = send((size_t)lt), s5h = send(2);
    const u32 sACC = s5, sZO = s5 + 1, sZC = s5 + 2, sRH = s5 + 3, sRHP = s5 + 4;
    const Ex LWW = sumW(pwW), LWX = sumW(pwX), LWH = sumW(pwH);
    auto timesW = [&](const vector<PW> &pw, Ex f) {
      vector<Ex> W(pw.size());
      for (size_t b = 0; b < W.size(); b++) W[b] = pw[b].wp * f;
      return W;
    };
    Degree3SumcheckState scE = sc3In(F, tr, slot, nvE(), NP, affSlot(sZO));
    sc3Row(F, scE, timesW(pwW, eqPts(zgW, scE.r.data(), lgk) * eqPts(ziW, scE.r.data() + lgk, lt)), "e");
    // Step 6. Sumchecks of e (v + gam): vpm over region M, vpd over D.
    const vector<Ex> gamP = trBatch(tr, 2); // two rows, numerator and denominator, under 1 and gam = gamP[1]
    const Ex *zihi = ziW, *zilo = ziW + lnT;
    const u32 sCl = send(2);
    Degree3SumcheckState vpm = sc3In(F, tr, slot, nvVM(), NP, affSlot(sCl));
    vector<Ex> WJv(nT, eZ());
    {
      const Ex *r = vpm.r.data();
      const vector<Ex> eIh = eqTab(zihi, lnT), eP = eqTab(r + lgk, lnT - 1), es = eqTab(r + lgk + lnT - 1, lnT);
      Ex f = eZ();
      for (u64 P = 0; P < hP; P++)
        for (u64 sg = 0; sg + 1 < nT; sg++) {
          u64 I = 0, J = 0;
          A.cover(P, sg, I, J);
          f = f + eP[P] * es[sg] * eIh[I];
          WJv[J] = WJv[J] + eP[P] * es[sg];
        }
      sc3Row(F, vpm, timesW(pwW, f * eqPts(zgW, r, lgk)), "VPM");
    }
    Degree3SumcheckState vpd = sc3In(F, tr, slot, nvVD(), NP, affSlot(sCl + 1));
    sc3Row(F, vpd, timesW(pwW, eqPts(zgW, vpd.r.data(), lgk) * eqPts(zihi, vpd.r.data() + lgk, lnT)), "VPD");
    // Step 7. Claim 7: the maximum is attained, m(i) = s(i, ix(i)). Claim 8: ix's bits are bits.
    // One message of the verifier has claim 7's point (tg | ti) and claim 8's: tk and (tbg | tbi).
    const vector<Ex> points = trPoint(tr, RoundName::ZERO, (size_t)(2 * lgk + 2 * lt + 4));
    auto part = [&](int first, int n) { return vector<Ex>(points.begin() + first, points.begin() + first + n); };
    const vector<Ex> tg = part(0, lgk), ti = part(lgk, lt), tk = part(lgk + lt, 4), tbg = part(lgk + lt + 4, lgk),
                     tbi = part(2 * lgk + lt + 4, lt);
    const u32 sM6 = send(NP);
    const vector<Ex> om = coinsOm(tr, NP);
    Aff startF; // sum_b om_b M6_b
    for (size_t b = 0; b < NP; b++) affAdd(startF, affSlot(sM6 + (u32)b), om[b]);
    Degree3SumcheckState scF = sc3In(F, tr, slot, nvF(), NP, startF);
    const Ex *ri = scF.r.data(), *rg = ri + lt, *rc = rg + lgk;
    {
      vector<Ex> WF(NP);
      const Ex f = eqPts(tg.data(), rg, lgk) * eqPts(ti.data(), ri, lt);
      for (size_t b = 0; b < NP; b++) WF[b] = om[b] * f;
      sc3Row(F, scF, WF, "f");
    }
    const vector<Ex> om4 = coinsOm(tr, NP);
    // scF's end value B*_b is kx at token ix(i), which no array contains. The SCD reduces the sum
    // of om4_b B*_b to KC_lam (kx), lam < Lam, and IBP_(b,k) (bit k of ix), b outer, k inner.
    Aff startD;
    for (size_t b = 0; b < NP; b++) affAdd(startD, sc3B(scF, b), om4[b]);
    VariedDegreeSumcheckState scd = scdIn(F, tr, slot, scdDegrees(), lt + 1, Lam + NP * (size_t)lt, startD);
    const Ex *pj = scd.r.data(), *pg = pj + lt, *pi = pg + lgk;
    const u32 sKC = scd.sEnd;
    auto sIb = [&](size_t b, int k) {
      return sKC + (u32)(Lam + b * (size_t)lt + (size_t)k);
    };
    {
      // S0 of docs/format/11-3-attention-segment.md, "The sumcheck SD", on the scalars sent
      // (F.shown()).
      Ex s0 = eZ();
      for (size_t b = 0; b < NP; b++) {
        Ex t = om4[b] * F.shown(sKC + (u32)(b / HA));
        for (int k = 0; k < lt; k++) {
          const Ex y = pj[lt - 1 - k];
          t = t * (F.shown(sIb(b, k)) * (y + y - eO()) + (eO() - y));
        }
        s0 = s0 + t;
      }
      scd.rowU = pscRow(F, scd.p, eqPts(ri, pi, lt) * eqPts(rg, pg, lgk) * s0);
    }
    const vector<Ex> om5 = coinsOm(tr, NP);
    Degree3SumcheckState scH = sc3In(F, tr, slot, nvH(), NP, affC(eZ()));
    const Ex *ri2 = scH.r.data(), *rg2 = ri2 + lt, *rk = rg2 + lgk;
    {
      vector<Ex> WH_(NP);
      const Ex f = eqPts(tbi.data(), ri2, lt) * eqPts(tbg.data(), rg2, lgk) * eqPts(tk.data(), rk, 4);
      for (size_t b = 0; b < NP; b++) WH_[b] = om5[b] * f;
      sc3Row(F, scH, WH_, "h");
    }
    lap("steps 3-7");
    // Step 8. The chain receives the commitment to U3. Its tables (there "parts"), added below:
    // docs/format/11-3-attention-segment.md, "The parts of `U3`".
    rootInH(F, tr, {(int)seam.iU3()});
    const u64 La = 1ull << cf.la, mS = 1ull << lmS, S2 = 1ull << ls, GSo = A.fillO * T >> cf.la,
              GSq = A.fillQ * T >> cf.la, mb = 1ull << cf.lmb, kAs = A.kAs, kPart = kAs / cf.NAs, kbA = A.kbA;
    const int iAH = (int)cf.NAs, iAO = iAH + 1, iBA = iAH + 2, iQA = iAH + 3;
    vector<HFun> fun;
    u64 p = 0;
    // Self-check in add(): elems = k * (m / L), which is what the range check's reads of the table
    // cover (HTailF2::objectsF()). So no two tables overlap and, as the last must end at nW, U3's
    // length, no entry of U3 is outside every table. tabLen: the tables' starts and lengths.
    std::map<u64, u64> tabLen;
    // Appends a table (matrix M, phi over L rows); returns its start in U3.
    // fn.tree = 2 is U3's number (U1 is 0), not an index into HFork::trees.
    auto add = [&](int M, u64 L, const vector<Ex> &phi, u64 elems, const string &nm) {
      const u64 at = p;
      seam.tl.funOpen();
      if (M < 0 || (size_t)M >= seam.tl.mats.size() || !L || seam.tl.mats[M].m % L ||
          elems != seam.tl.mats[M].k * (seam.tl.mats[M].m / L))
        refuse(G_ + "internal-a-table-length");
      HFun fn;
      fn.M = M;
      fn.L = L;
      fn.phi = phi;
      fn.tree = 2;
      fn.at = p;
      fn.name = G_ + nm;
      fun.push_back(std::move(fn));
      tabLen[at] = elems;
      p += elems;
      return at;
    };
    // A table on as_g has kAs entries: the tables of its NAs parts one after the other, part mu's
    // columns from mu * kPart on. tabLen maps the first part's start to the parts' total length.
    auto addAs = [&](const vector<Ex> &phi, const string &nm) {
      const u64 at = p;
      for (u64 q = 0; q < cf.NAs; q++) add((int)q, mS, phi, kPart, nm + "-part" + std::to_string(q));
      tabLen[at] = p - at;
      return at;
    };
    auto lastN = [&](const vector<Ex> &z, int n) {
      return eqTab(z.data() + (z.size() - n), n);
    };
    vector<Ex> ptM(vpm.r.begin(), vpm.r.begin() + lgk + 2 * lnT - 1), ptD(vpd.r.begin(), vpd.r.begin() + lgk + lnT);
    ptM.insert(ptM.end(), zilo, zilo + lB);
    ptM.insert(ptM.end(), vpm.r.end() - lB, vpm.r.end());
    ptD.insert(ptD.end(), zilo, zilo + lB);
    ptD.insert(ptD.end(), vpd.r.end() - lB, vpd.r.end());
    // One table TA serves the lookup segments RM, RD, FM and FD: each inner point ends in
    // zx's last lmS coordinates. That follows from dT >= lmS ("forest-depth-A") and lmS
    // <= nD, the shorter inner point's length (AFork::shape()), and is tested again here.
    for (auto &pc : pcl)
      for (int i = 1; i <= lmS; i++)
        if (!(pc.Z[pc.Z.size() - (size_t)i] == g1.zx[g1.zx.size() - (size_t)i])) refuse(G_ + "internal-a-merged-point");
    const u64 tA = addAs(lastN(g1.zx, lmS), "as@zx");
    const u64 tVM = addAs(lastN(ptM, lmS), "as@VPM"), tVD = addAs(lastN(ptD, lmS), "as@VPD");
    auto tok = [&](const vector<Ex> &hi, const Ex *lo) {
      const vector<Ex> el = eqTab(lo, lB);
      vector<Ex> v(T);
      for (u64 I = 0; I < nT; I++)
        for (u64 x = 0; x < B; x++) v[I * B + x] = hi[I] * el[x];
      return v;
    };
    for (auto &k : kfs)
      k.tM = add(iAH, T, tok(k.WI, k.zi.data()), A.fillH,
                 "ah@phiM-" + calls[k.call].c.name + (k.seg == sFD ? "-D" : "-M"));
    const u64 hW = add(iAH, T, eqTab(ziW, lt), A.fillH, "ah@ziW"), hX = add(iAH, T, eqTab(ziX, lt), A.fillH, "ah@ziX"),
              hE = add(iAH, T, eqTab(scE.r.data() + lgk, lt), A.fillH, "ah@riE"),
              hT = add(iAH, T, eqTab(ti.data(), lt), A.fillH, "ah@ti"),
              hPi = add(iAH, T, eqTab(pi, lt), A.fillH, "ah@pi"), hR2 = add(iAH, T, eqTab(ri2, lt), A.fillH, "ah@ri2");
    auto lng = [&](const Ex *zc, const vector<Ex> &ptk) {
      if (ptk.size() != T || La != S2 * T) refuse("plan-tensor-layout");
      return blockRows(eqTab(cutLast(zc, ldh, ls).lo, ls), ptk);
    };
    // One table TO serves WL's and WH's reads of ao_g: (ziW | zcW) and (ziH | zcH) are
    // both zx's last lt + ldh coordinates, since dT >= lt + ldh ("forest-depth-A").
    for (int i = 0; i < lt + ldh; i++)
      if (!(ziW[i] == ziH[i]) || !(ziW[i] == g1.zx[g1.zx.size() - (size_t)(lt + ldh) + (size_t)i]))
        refuse(G_ + "internal-a-merged-point");
    const u64 tO = add(iAO, La, lng(zcW, eqTab(ziW, lt)), GSo, "ao@zx");
    for (auto &k : kfs) {
      const Ex *r = k.sc.r.data(), *rcK = r + (k.sc.r.size() - ldh);
      vector<Ex> WIr(nT, eZ()), WJr(nT, eZ());
      if (k.seg == sFM) {
        const vector<Ex> eP = eqTab(r + lgk, lnT - 1), es = eqTab(r + lgk + lnT - 1, lnT);
        for (u64 P = 0; P < hP; P++)
          for (u64 sg = 0; sg + 1 < nT; sg++) {
            u64 I = 0, J = 0;
            A.cover(P, sg, I, J);
            WIr[I] = WIr[I] + eP[P] * es[sg];
            WJr[J] = WJr[J] + eP[P] * es[sg];
          }
      } else
        WIr = WJr = eqTab(r + lgk, lnT);
      k.tQ = add(iQA, La, lng(rcK, tok(WIr, k.zi.data())), GSq, "qa@phiQ-" + calls[k.call].c.name);
      k.tK = add(iQA, La, lng(rcK, tok(WJr, k.zj.data())), GSq, "qa@phiK-" + calls[k.call].c.name);
    }
    const u64 qO = add(iQA, La, lng(zcW, eqTab(scE.r.data() + lgk, lt)), GSq, "qa@o");
    const u64 qVM = add(iQA, La, lng(zcW, tok(WJv, vpm.r.data() + (vpm.r.size() - lB))), GSq, "qa@VPM");
    const u64 qVD =
        add(iQA, La, lng(zcW, tok(eqTab(vpd.r.data() + lgk, lnT), vpd.r.data() + (vpd.r.size() - lB))), GSq, "qa@VPD");
    const u64 qF = add(iQA, La, lng(rc, eqTab(ri, lt)), GSq, "qa@f-q"),
              qKx = add(iQA, La, lng(rc, eqTab(pj, lt)), GSq, "qa@f-kx");
    // bA_g's three tables: TB_0, TB_1 and TP (claim 9).
    const u64 tBit[2] = {add(iBA, mb, forestPhiB(g1, cf.lmb, 0), kbA, "bA-bits-0"),
                         add(iBA, mb, forestPhiB(g1, cf.lmb, 1), kbA, "bA-bits-1")};
    const u64 tPin = add(iBA, mb, forestPhiP(cf.lmb), kbA, "bA-pin");
    if (p != nW) refuse("internal-af-W-length");
    // The list of tables is closed before the range check's random coefficients are drawn:
    // closeFuns() records its length and a hash, which funsHeld() compares after the claims.
    seam.tl.closeFuns(fun);
    lap("tables' weights (phi)");
    // Step 9. The range check's messages.
    seam.transcript(F, tr, slot);
    lap("tail: ZS, group 2, its root tests");
    if (slot != nMsg) refuse(G_ + "internal-slot-count");
    // Step 10. The element whose powers batch the claims: the sub-proof's last challenge.
    F.planClaimsF = (u32)nLin; // The planned number of claims.
    seam.batch(F, tr, slot);
    {
      lap("the batching element");
      // ---- The claims: docs/format/11-3-attention-segment.md, "The claims, in order" ----
      // That list's group 1 is claims 1 and 2; groups 2 to 4 are claims 3 to 5; group 5 is A-J.
      // A claim "X = Y" is added as lin(X - Y). plane() reads as_g, shV() ah_g, XL() ao_g or qa_g.
      const u32 iU3 = seam.iU3();
      // U3at(t, i): entry i of the table that starts at t. Claims 1 to 9 read U3 only through it.
      // Self-check: t starts a table and i is within it, so the range check covers what they read.
      // It does not see an i in another tensor, limb or layer of the table, or another table's t.
      auto U3at = [&](u64 t, u64 i) {
        const auto it = tabLen.find(t);
        if (it == tabLen.end() || i >= it->second) refuse(G_ + "internal-a-table-index");
        return uEnt(F, iU3, t + i);
      };
      auto S_ = [&](u32 s) {
        return affSlot(s);
      };
      const Ex m1 = eZ() - eO();
      auto plane = [&](u64 t, int x, u64 lam, u64 a, const vector<Ex> &Z, bool isD) {
        const int nb_ = isD ? nD : nM;
        const u64 a0 = (isD ? 5 * Lam * HA << nM : 0) + ((((u64)x * Lam + lam) * HA + a) << nb_);
        const vector<Ex> eh = eqTab(Z.data(), nb_ - lmS);
        Aff s;
        for (u64 hi = 0; hi < eh.size(); hi++) affAdd(s, U3at(t, (a0 >> lmS) + hi), eh[hi]);
        return s;
      };
      auto shV = [&](u64 t, u64 OFF, int nl, u64 J, u64 lam, u64 j) {
        Aff s;
        u64 sc = 1;
        for (int l = 0; l < nl; l++, sc *= 256) affAdd(s, U3at(t, OFF + ((u64)l * Lam + lam) * J + j), eI((i64)sc));
        return s;
      };
      auto XL = [&](u64 t, const TenAt &X, u64 lam, u64 hd, const Ex *zc) {
        const PointHighLow k = cutLast(zc, ldh, ls);
        const vector<Ex> ec = eqTab(k.hi, k.nhi);
        const u64 j0 = headFeat(X, hd, dh);
        Aff s;
        for (u64 ch = 0; ch < ec.size(); ch++) {
          u64 sc = 1;
          for (u64 l = 0; l < X.nl; l++, sc *= 256)
            affAdd(s, U3at(t, tenCell(X, l, lam, j0) / S2 + ch), eS(ec[ch], sc));
        }
        return s;
      };
      auto overG = [&](const Ex *zg, auto f) {
        const vector<Ex> eg = eqTab(zg, lgk);
        Aff s;
        for (u64 g = 0; g < G; g++) affAdd(s, f(g), eg[g]);
        return s;
      };
      // KF[q]: call q's key form. Keys, tables and key forms:
      // docs/format/11-3-attention-segment.md, "The calls: tables, keys and lengths" and
      // "The claims, in order", group 5.
      vector<Aff> KF(NC);
      vector<char> joined(NC, 0);
      // Claims 1 and 2. F.section() only names the next claims for a test's report (../tests/claims_report.h).
      forestObjects(F, g1);
      // One claim per plane: the value sent equals table TA's.
      auto linPlanes = [&](const PlaneClaim &pc) {
        for (int x = 0; x < pc.np; x++) {
          Aff f = S_(pc.s0 + (u32)x);
          for (auto &w : pc.pw) affAdd(f, plane(tA, x, w.lam, w.a, pc.Z, pc.isD), m1 * w.wp);
          F.lin(f);
        }
      };
      F.section("attention: claim 3: the records' planes of the calls KR (A-3.1)");
      // Claim 3: KR. A key is (c0 + 128) + 2^20 (c1 + 128) + 2^40 c2. The table has flag fl = 1 iff
      // r >= 2^sh. c_x is plane x's cell (0, 1: r's bytes; 2: fl; 3, 4: e's bytes), a signed byte:
      // planes 0, 1, 3 store a byte minus 128. A key's constant is multiplied by LW = sumW(pw).
      for (size_t q = 0; q < NC; q++)
        if (calls[q].kind == "KR") {
          Aff v;
          for (auto &pc : pcl)
            if (pc.call == q) {
              linPlanes(pc);
              const Ex LW = sumW(pc.pw);
              v = v + (S_(pc.s0) + affC(eS(LW, 128))) + (S_(pc.s0 + 1) + affC(eS(LW, 128))) * eI(1ll << 20) +
                  S_(pc.s0 + 2) * eI(1ll << 40);
            }
          KF[q] = v;
          joined[q]++;
        }
      F.section("attention: claim 4: scores against q and k, and the maximum, for the calls KFD (A-4.1 to A-4.4)");
      // Claim 4: KFD. Key: D' - r + 2^32 fl + 2^33 e. DP: D' = m - s (+ Om if j > i); C if no tile.
      for (size_t q = 0; q < NC; q++)
        if (calls[q].kind == "KFD") {
          const ACall &c = calls[q];
          const AFork::ShD d = A.shd(c.sh);
          Ex wall = eZ();
          for (auto &pc : c.pcs) wall = wall + eqAt(c.c.zk.data(), c.c.n - pc.b, pc.off >> pc.b);
          // The padding keys' term: docs/format/11-3-attention-segment.md,
          // "The claims, in order", group 5.
          Aff v = affC(eS(eO() - wall, c.c.tauAt(c.c.t0)));
          for (auto &k : kfs)
            if (k.call == q) {
              const u32 s = k.s0;
              linPlanes(pcl[k.claim]);
              {
                Aff f = S_(s + 5);
                for (auto &w : k.pw)
                  affAdd(f, overG(k.zg.data(), [&](u64 g) { return shV(k.tM, A.oMr, 4, H, w.lam, g * HA + w.a); }),
                         m1 * w.wp);
                F.lin(f);
              }
              F.lin(k.sc.rowU);
              const Ex *rgK = k.sc.r.data(), *rcK = rgK + (k.sc.r.size() - ldh);
              for (size_t b = 0; b < k.pw.size(); b++) {
                const PW &w = k.pw[b];
                F.lin(sc3A(k.sc, b) - overG(rgK, [&](u64 g) { return XL(k.tQ, A.atQ, w.lam, g * HA + w.a, rcK); }));
                F.lin(sc3B(k.sc, b) - overG(rgK, [&](u64 g) { return XL(k.tK, A.atK, w.lam, g, rcK); }));
              }
              const Ex LW = sumW(k.pw), L128 = eS(LW, 128);
              const Aff DP = S_(s + 5) - S_(s + 6) +
                               affC(k.seg == sFM ? eS(LW, d.C) * onesOf(k.zs.data(), lnT)
                                                 : eS(LW, d.Om) * (eO() - LEof(k.zj.data(), k.zi.data(), lB)));
              v = v + (DP - (S_(s) + affC(L128)) - (S_(s + 1) + affC(L128)) * eI(256) + S_(s + 2) * eI(1ll << 32) +
                       (S_(s + 3) + affC(L128) + S_(s + 4) * eI(256)) * eI(1ll << 33));
            }
          KF[q] = v;
          joined[q]++;
        }
      F.section("attention: claim 5: the division: Z o, its remainders, the index's bits (A-5.1 to A-5.8)");
      // First, ZC, RH, RHP, the IB_k, RH2 and RHP2 against U3's tables.
      {
        Aff zc_ = S_(sZC), rh = S_(sRH), rhp = S_(sRHP), r2 = S_(s5h), r2p = S_(s5h + 1);
        for (auto &w : pwW) {
          affAdd(zc_, overG(zgW, [&](u64 g) { return shV(hW, A.oZr, 4, H, w.lam, g * HA + w.a); }), m1 * w.wp);
          affAdd(rh, overG(zgW, [&](u64 g) { return XL(tO, TenAt{A.oRh, 1, Lam, dq}, w.lam, g * HA + w.a, zcW); }),
                 m1 * w.wp);
          affAdd(rhp, overG(zgW, [&](u64 g) { return XL(tO, TenAt{A.oRhp, 1, Lam, dq}, w.lam, g * HA + w.a, zcW); }),
                 m1 * w.wp);
        }
        for (auto &w : pwH) {
          affAdd(r2, overG(zgH, [&](u64 g) { return XL(tO, TenAt{A.oRh, 1, Lam, dq}, w.lam, g * HA + w.a, zcH); }),
                 m1 * w.wp);
          affAdd(r2p, overG(zgH, [&](u64 g) { return XL(tO, TenAt{A.oRhp, 1, Lam, dq}, w.lam, g * HA + w.a, zcH); }),
                 m1 * w.wp);
        }
        // These claims follow the order in which the values are sent.
        F.lin(zc_);
        F.lin(rh);
        F.lin(rhp);
        for (int k = 0; k < lt; k++) {
          Aff f = S_(sIB + (u32)k);
          for (auto &w : pwX)
            affAdd(f,
                   overG(zgX, [&](u64 g) { return shV(hX, A.oIb, 1, (u64)lt * H, w.lam, (u64)k * H + g * HA + w.a); }),
                   m1 * w.wp);
          F.lin(f);
        }
        F.lin(r2);
        F.lin(r2p);
        const Aff REM = S_(sACC) * eI(2) + S_(sZC) - S_(sZO) * eI(2);
        Aff ib;
        for (int k = 0; k < lt; k++) affAdd(ib, S_(sIB + (u32)k), eI(1ll << k));
        // tw0, tw1: WL's keys; tIX: IX's. docs/format/11-3-attention-segment.md,
        // "The claims, in order", group 5.
        const Aff tw0 = REM - (S_(sRH) + affC(eS(LWW, 128))) * eI(65536),
              tw1 = S_(sZC) * eI(2) - affC(LWW) - REM - (S_(sRHP) + affC(eS(LWW, 128))) * eI(65536);
        const Aff tWL = tw0 * (eO() - zsl) + tw1 * zsl;
        const Aff tIX = (affC(LWX * idOf(ziX, lt)) - ib) * eI(1ll << (16 - lt));
        for (size_t q = 0; q < NC; q++)
          if (calls[q].kind == "R16") {
            Aff v;
            if (holds(calls[q], sWL)) v = v + tWL;
            if (holds(calls[q], sIX)) v = v + tIX;
            KF[q] = v;
            joined[q]++;
          }
        // WH's key: (rh + 128) + 2^20 (rhp + 128), from RH2 and RHP2.
        const size_t qW = (size_t)(cW - &calls[0]);
        KF[qW] = (S_(s5h) + affC(eS(LWH, 128))) + (S_(s5h + 1) + affC(eS(LWH, 128))) * eI(1ll << 20);
        joined[qW]++;
        F.lin(scE.rowU);
        const Ex *rgE = scE.r.data();
        for (size_t b = 0; b < NP; b++) {
          const PW &w = pwW[b];
          F.lin(sc3A(scE, b) - overG(rgE, [&](u64 g) { return shV(hE, A.oZr, 4, H, w.lam, g * HA + w.a); }));
          F.lin(sc3B(scE, b) - overG(rgE, [&](u64 g) { return XL(qO, A.atO, w.lam, g * HA + w.a, zcW); }));
        }
      }
      // A-J, the join, over all calls in list order: forestJoin(). Before it, the test that every
      // call's KF has been set exactly once.
      for (char j : joined)
        if (j != 1) refuse(G_ + "af-unclosed-call");
      /* The keys that no weighed piece covers must match each tree's padding count
         (forestPaddingLaid()). The forms of KR, R16 and WH have no term for these, so read 0: that
         must be the padding key. KFD's has one. */
      {
        vector<u64> laid((size_t)g1.NT, 0);
        for (size_t q = 0; q < NC; q++) {
          const ACall &c = calls[q];
          if (c.kind != "KFD" && c.c.tauAt(c.c.t0) != 0) refuse(G_ + "internal-af-padding-key");
          u64 upto = 0;  // the end of the last weighed piece; pieces follow each other without gaps
          for (size_t i = 0; i < c.pcs.size(); i++) {
            if (!weighed[q][i]) {
              forestPubRun(g1, laid, (int)q, c.pcs[i].off, 1ull << c.pcs[i].b);  // padding keys
              continue;
            }
            upto = std::max<u64>(upto, c.pcs[i].off + ((u64)1 << c.pcs[i].b));
          }
          forestPubRun(g1, laid, (int)q, upto, (1ull << c.c.n) - upto);
        }
        forestPaddingLaid(g1, laid);
      }
      F.section(FOREST_JOIN_SECTION);
      F.lin(forestJoin(g1, KF));
      F.section("attention: claim 6: the accumulator and the sum of the weights against e and v");
      // Format: docs/format/11-3-attention-segment.md, "The claims, in order", group 6.
      F.lin(S_(sCl) + S_(sCl + 1) - S_(sACC) - S_(sZC) * gamP[1]);
      F.lin(vpm.rowU);
      for (size_t b = 0; b < NP; b++) {
        const PW &w = pwW[b];
        F.lin(sc3A(vpm, b) - (plane(tVM, 3, w.lam, w.a, ptM, false) + affC(eI(128)) +
                                plane(tVM, 4, w.lam, w.a, ptM, false) * eI(256)));
        F.lin(sc3B(vpm, b) - (affC(gamP[1]) + overG(vpm.r.data(), [&](u64 g) { return XL(qVM, A.atV, w.lam, g, zcW); })));
      }
      F.lin(vpd.rowU);
      for (size_t b = 0; b < NP; b++) {
        const PW &w = pwW[b];
        F.lin(sc3A(vpd, b) -
              (plane(tVD, 3, w.lam, w.a, ptD, true) + affC(eI(128)) + plane(tVD, 4, w.lam, w.a, ptD, true) * eI(256)));
        F.lin(sc3B(vpd, b) - (affC(gamP[1]) + overG(vpd.r.data(), [&](u64 g) { return XL(qVD, A.atV, w.lam, g, zcW); })));
      }
      // Source for a maximum taken from the prover and proved by two conditions: Liu, Xie, Zhang, "zkCNN: Zero
      // Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy", Cryptology ePrint Archive
      // 2021/673, section 4.3. There by bits and a product that must be 0; here by looked-up differences and a
      // committed index. Only the second, that it is attained, was checked against the paper.
      // The maximum of a softmax by looked-up differences is also in: Chen, Waiwitlikhit, Stoica, Kang, "ZKML: An
      // Optimizing System for ML Inference in Zero-Knowledge Proofs", EuroSys 2024, section 5.1 (there of two values,
      // with a product that must be 0).
      F.section("attention: claim 7: the score at the index is the maximum");
      // Format: docs/format/11-3-attention-segment.md, "The claims, in order", group 7.
      for (size_t b = 0; b < NP; b++) {
        const u64 lam = b / HA, a = b % HA;
        F.lin(S_(sM6 + (u32)b) - overG(tg.data(), [&](u64 g) { return shV(hT, A.oMr, 4, H, lam, g * HA + a); }));
      }
      F.lin(scF.rowU);
      for (size_t b = 0; b < NP; b++) {
        const u64 lam = b / HA, a = b % HA;
        F.lin(sc3A(scF, b) - overG(rg, [&](u64 g) { return XL(qF, A.atQ, lam, g * HA + a, rc); }));
      }
      // scd: the claim that it leaves, then its end values.
      {
        F.lin(scd.rowU);
        for (u64 lam = 0; lam < Lam; lam++)
          F.lin(S_(sKC + (u32)lam) - overG(pg, [&](u64 g) { return XL(qKx, A.atK, lam, g, rc); }));
        for (size_t b = 0; b < NP; b++) {
          const u64 lam = b / HA, a = b % HA;
          for (int k = 0; k < lt; k++)
            F.lin(S_(sIb(b, k)) -
                  overG(pg, [&](u64 g) { return shV(hPi, A.oIb, 1, (u64)lt * H, lam, (u64)k * H + g * HA + a); }));
        }
      }
      F.section("attention: claim 8: the index's cells are bits");
      // Format: docs/format/11-3-attention-segment.md, "The claims, in order", group 8.
      F.lin(scH.rowU);
      {
        const vector<Ex> ek = eqTab(rk, 4);
        for (size_t b = 0; b < NP; b++) {
          const u64 lam = b / HA, a = b % HA;
          F.lin(sc3B(scH, b) - sc3A(scH, b) + affC(eO()));
          Aff f = sc3A(scH, b);
          for (int k = 0; k < lt; k++)
            affAdd(f, overG(rg2, [&](u64 g) { return shV(hR2, A.oIb, 1, (u64)lt * H, lam, (u64)k * H + g * HA + a); }),
                   m1 * ek[k]);
          F.lin(f);
        }
      }
      // Claim 9, on the multiplicities' bits: docs/format/11-3-attention-segment.md,
      // "The claims, in order", group 9.
      forestPlaneLins(
          F, g1, cf.lmb, [&](int e, u64 col) { return U3at(tBit[e], col); },
          [&](u64 col) { return U3at(tPin, col); });
      lap("walk WA1-WA8");
      F.section("attention: claim 10: the smallness test");
      // The range check's claims.
      seam.objects(F, fun);
      lap("walk WA9 (ring products, X3, MODQ, X4)");
      /* Self-check: every message slot must be in a claim (LS[j] is 0 if none has slot j, else with
         probability at most about nLin / |E|). It guards hiding; do not remove it: only so can
         cp_verify see an end value sent unmasked. cp_prove tests no finished proof, so whoever hands
         a proof on must have verified it. */
      for (u32 j = 0; j < (u32)nMsg; j++)
        if (isZ(F.LS[j])) refuse(G_ + "internal-a-free-slot");
      seam.tl.funsHeld(F, fun); // runs after all claims are listed
    } // ---- the end of the list of claims ----
    // After finish() no claim can be added.
    seam.finish(F, tr);
    lap("self-test, closing steps, the one equation, columns");
    if (F.nClaim != F.planClaimsF) refuse(G_ + "internal-f-draws");
    F.kapAllTakenF();
    seam.tl.foldAllTakenF(F);
    nClaim = F.nClaim;
    if (nClaim != nLin) refuse(G_ + "internal-a-claim-count");
    // The range check's coefficients must have been drawn exactly once.
    if (seam.tl.foldDrawn != 1) refuse(G_ + "internal-af-rows-booked");
    // The range check's lookup group must have run; the 1 counts this sub-proof's own.
    if (1 + seam.tl.groupsRun() != 2) refuse(G_ + "internal-a-calls-booked");
  }
};
