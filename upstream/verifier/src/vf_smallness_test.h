// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's approximate range check (`tail` in identifiers, "smallness test" in
 * docs/format/11-1-machinery.md; what it shows: blocks/iop/rng.h) in a layer, attention or
 * output-head sub-proof (`fork` in identifiers). It is meant to show, for every cell of its
 * matrices, |cell| < 2^(LBZ + 1) = 2^18. Z is one row's sum of coefficient * cell, each column's
 * coefficient a random -1, 0 or 1, in one of tau combinations. A tie is a linear claim per matrix.
 * A read, one claim per functional (HFun: L weights phi) and L rows of its matrix, equates
 * sum_i phi[i] Z with the same combinations of the committed sum_i phi[i] cell, in U3. Both:
 * docs/format/11-1-machinery.md, "The equations".
 * A "tree" is a committed array; a virtual row (HRow) weights its rows by coef, its columns by L.
 * PK_i, PX_i, PF_i: symbol trees (the packed matrices). U1 .. U3: unit trees (one row over the
 * field E). ZZ: two digits per Z. BZ: the bits of their lookup multiplicities. */
#pragma once
#include "vf_smallness_matrices.h"
#include "vf_subproof_engine.h"
#include "vf_subproof_segment.h"
/* tau, the number of random combinations: the parameter file's line `tau` (TAU, vf_common.h), which
 * cp_verify.cpp checks (`tauServed`: 1 or 2 `t2BL` blocks); else `piecesF` ends NOT CHECKED. */
#define T2TAU ((u32)TAU)
/* The bits of the low and the high digit of Z + 2^LBZ (LBZ: vf_common.h):
 * docs/format/11-1-machinery.md, "The array `ZZ`". Their lookup calls: R10z (low), R8d (high). */
static constexpr int T2D0 = 10, T2D1 = LBZ + 1 - T2D0;
static_assert(T2D0 >= 5 && T2D1 >= 5, "the table of a lookup has at least 2^kappa = 32 rows");
static_assert(T2D1 == 8,
              "the lookup call of the high digit is named R8d, and the tiling self-test has its k written out as 8");
/* Splits rows 0 .. R - 1 into blocks (first row, length), each length the largest power of two that
 * still fits: 160 rows give (0, 128) and (128, 32). */
static vector<std::pair<u64, u64>> t2BL(u64 R) {
  vector<std::pair<u64, u64>> o;
  u64 r0 = 0;
  while (r0 < R) {
    u64 Rp = 1;
    while (2 * Rp <= R - r0) Rp *= 2;
    if (r0 & (Rp - 1)) refuse("internal-t2f-blocks");
    o.push_back({r0, Rp});
    r0 += Rp;
  }
  return o;
}
/* theta^c, one factor at a time: the batching coefficient kappa of claim c (theta: the sub-proof's
 * batching element). Only `tileSelfTestF` calls it, to recompute weights that claims got through
 * `HFork::kap` (vf_subproof_engine.h), which squares and multiplies; a difference ends NOT CHECKED. */
static Ex t2fKappaByText(const Ex &theta, u32 c) {
  Ex power = eO();
  for (u32 i = 0; i < c; i++) power = power * theta;
  return power;
}
/* A reader's description of a foreign matrix: one that another sub-proof, its owner, commits to
 * and this one reads. `shapeF` compares it with the matrix, `t2SameShape` and `samePlaceF` with the
 * owner's. Its tree PF_i is the owner's border tree PX_i (one that others read): K columns;
 * a = 1 + the number of readers. */
struct T2Foreign {
  string name;
  u64 K = 0;
  u32 a = 0;
  u64 m = 0;
  u32 wp = 0; // m: the owner's cell rows. wp: the owner's width parameter; no parameter-file line sets it, so it is 0.
};
struct T2Map {
  int U[4] = {-1, -1, -1, -1};
  int TT = -1, BT = -1, ZZ = -1, YY = -1, KK = -1, BZ = -1;
  vector<int> TX, FT, PK; // PX_i, PF_i, PK_i. U to PK: indices in HFork::trees; -1: no such tree
  int count = 0;
};
// One sub-proof's range check (rest: vf_smallness_test_members.inc). Planning code calls shape(), then addTrees().
struct HTailF2 {
  // mats[mu]: the matrices checked. NM: their number. mz: their rows, summed.
  u64 mz = 0, NM = 0;
  vector<TMat> mats;
  /* Per matrix: off, its first row among the mz; gtTree, its symbol tree; gtRow and col0, always 0.
   * kTX: columns per PX_i. ft: one struct per PF_i. z[1]: R10z; z[2]: R8d; z[0], z[3]: empty. */
  vector<u64> off, col0;
  vector<int> gtTree;
  vector<u32> gtRow;
  vector<u64> kTX;
  vector<T2Foreign> ft;
  T2Map ix;
  HCallZ z[4];
  /* The lookup calls' forest (vf_lookup_argument.h). `shape` sets its sizes and pub but not cl: a
   * plan's copy would point into the plan. `transcript` sets all fields. */
  HForest f9;
  bool f9Ran = false;
  size_t groupsRun() const { return f9Ran ? 1 : 0; }
  vector<signed char> cfF;        // the combinations' coefficients (`drawMatrixF`), and
  vector<u64> cfFirst;            // the number of the first column of each matrix among all columns
  vector<Ex> vt, vp; // vt = vp = (theta^u), the combinations' weights in reads and ties (`powersF`)
  /* The number of the forest's join (`linJoin9`, which sets it) among the sub-proof's claims: the
   * join's kappa is theta^joinObj9. `tileSelfTestF` reads it. */
  mutable u32 joinObj9 = ~0u;
#include "vf_smallness_test_members.inc"

  /* m: the matrices, in the order of mu. TMat::kind 0: this sub-proof commits to the matrix; 1: it
   * is foreign, described by f[TMat::foreign]. Any other kind is rejected ("t2f-weight-class"):
   * kind 2, a weight matrix, cannot occur in a sub-proof of a request. For kind 0, txOf[mu] < 0
   * gives a tree PK_i (no other sub-proof reads it), txOf[mu] = i the border tree PX_i, with
   * i = 0, 1, .. in m's order. hasU[n]: the sub-proof has the unit tree U(n+1); hasU[3] must be
   * false. tau must equal T2TAU. */
  void shape(const vector<TMat> &m, const vector<int> &txOf, const vector<T2Foreign> &f, const bool hasU[4], u32 tau) {
    shapeF(m, txOf, f, hasU, tau);
  }
  // The number of linear claims that objectsF() adds, for nX4 reads.
  // Callers add lins() to the number of their other claims; a run with
  // another total ends NOT CHECKED (status 5: the verifier itself failed; no verdict).
  size_t lins(size_t nX4) const { return linsF(nX4); }
  // The number of BZ's columns (KB): docs/format/11-1-machinery.md, "Where a bit of a multiplicity
  // is in `BZ`".
  u64 bzColumns() const { return 1ull << (f9.kT + f9.cK); }
  // The check's number of message slots (one per scalar sent): its lookup's.
  size_t smallElems() const { return forestElems(f9); }
  // ---- The trees in list order ----
  // Adds U1 .. U3 (those that the sub-proof has), the PK_i, the PX_i, ZZ, BZ and the PF_i, in this
  // order. ZZ: docs/format/11-1-machinery.md, "The array `ZZ`". nU[n]: the length of U(n+1).
  // aTX[i]: PX_i's a (see T2Foreign). hb is only stored, in plOwn and plFor.
  void addTrees(HFork &F, Tail2 &T, const u64 nU[3], const vector<u32> &aTX, u32 hb) { addTreesF(F, T, nU, aTX, hb); }
  // Runs the check's rounds: docs/format/11-1-machinery.md, first piece, "The smallness test".
  void transcript(HFork &F, Tr &tr, u32 &slot) { transcriptF(F, tr, slot); }
  static const Ex *zrOf(const HCallZ &c, const HPiece &p) { return c.c.zk.data() + (c.c.n - p.b); }
  // Sets rows[..].coef, the row weights of every virtual row.
  void coefficients(HFork &F) const { coefficientsF(F); }
  /* The forest's join, one linear claim on both calls (compare `forestJoin`): LJ in
   * docs/format/11-1-machinery.md, "The claims, in order"; its key forms: "The lookup on the
   * digits". That page's theta is f9.theta; its chi is HFork::theta. A piece (HPiece) splits zk,
   * both calls' point, as (zp | zr | zc); `zrOf` returns (zr | zc); its row has
   * coef[r0 + r] = eq(zr, r). */
  void linJoin9(HFork &F) const {
    // At a padding key's index the claim reads nothing: right only if that key, unshifted, is 0.
    for (int q = 0; q < 2; q++)
      if (f9.cl[q]->tauAt(f9.cl[q]->t0) != 0) refuse(F.G_ + "internal-f-padding-key");
    vector<vector<Ex>> ec;
    vector<Run2> runs;
    Aff a;
    for (int q = 0; q < 2; q++) {
      const HCallZ &c = z[1 + q];
      a.c = a.c + f9.e[q] * (f9.alpha - eS(f9.theta, (u64)q));
      for (size_t i = 0; i < c.pc.size(); i++) ec.push_back(eqTab(zrOf(c, c.pc[i]) + c.pc[i].a, clog2(c.pc[i].CP)));
    }
    {
      const vector<Ex> ey = eqTab(f9.zy.data(), f9.rK);
      for (int y = f9.NT; y < (1 << f9.rK); y++) a.c = a.c + ey[y] * (f9.alpha - f9.padKey(0));
    }
    // Padding is described twice: f9.pub[y] counts forest tree y's padding keys for the lookup's
    // public factor and key count, and the runs below skip them. read[y]: the keys read in tree y;
    // the rest count as padding (`laid[y]`). Key i of a call is in that call's tree i >> dT.
    vector<u64> read(f9.NT, 0);
    size_t at = 0;
    for (int q = 0; q < 2; q++) {
      const HCallZ &c = z[1 + q];
      for (size_t i = 0; i < c.pc.size(); i++, at++) {
        const HPiece &p = c.pc[i];
        Run2 r;
        r.tree = (u32)p.tree;
        r.row = (u32)p.row;
        r.first = p.c0;
        r.coef = eZ() - f9.e[q] * eqOne(c.c.zk.data(), c.c.n - p.b, p.off >> p.b);
        r.w = &ec[at];
        r.n = p.Cp;
        runs.push_back(r);
        for (u64 rr = 0; rr < p.Rp; rr++)
          for (u64 first = p.off + rr * p.CP, left = r.n; left;) {
            const u64 b = first >> f9.dT, take = std::min<u64>(left, ((b + 1) << f9.dT) - first);
            if (b >> f9.iota[q]) refuse(F.G_ + "internal-f-join-place");
            read[(u64)f9.first[q] + b] += take;
            first += take, left -= take;
          }
      }
    }
    {
      vector<u64> laid(f9.NT, 0);
      for (int y = 0; y < f9.NT; y++) {
        if (read[y] > 1ull << f9.dT) refuse(F.G_ + "internal-f-join-place");
        laid[y] = (1ull << f9.dT) - read[y];
      }
      forestPaddingLaid(f9, laid); // NOT CHECKED unless the two descriptions agree (vf_lookup_argument.h)
    }
    // t2Lin() gives the join the number F.nClaim.
    joinObj9 = F.nClaim;
    F.section(string("smallness test, ") + FOREST_JOIN_SECTION);
    t2Lin(F, a - f9.Cend, runs);
    forestJoinAdded(f9); // records the join as added; else `settle` (vf_subproof_engine.h) ends NOT CHECKED
  }
  /* Three claims on the planes b, the multiplicities' bits in BZ (compare `forestPlaneLins`), on
   * virtual rows 0, 1, 2: LP0 to LP2 in docs/format/11-1-machinery.md, "The claims, in order".
   * A bit's place: docs/format/11-1-machinery.md, "Where a bit of a multiplicity is in `BZ`". */
  void linBits9(HFork &F) {
    const u64 KB = bzColumns();
    auto on = [&](u32 row, const Ex &coef, const vector<Ex> &w) {
      Run2 r;
      r.tree = (u32)ix.BZ;
      r.row = row;
      r.first = 0;
      r.coef = coef;
      r.w = &w;
      r.n = KB;
      return r;
    };
    const vector<Ex> w0 = eqTab(f9.zb[0].data() + 5, f9.kT + f9.cK), w1 = eqTab(f9.zb[1].data() + 5, f9.kT + f9.cK);
    bitFactor9 = f9.leaves.eps * f9.leavesA;
    F.section(string("smallness test, ") + FOREST_PLANE_SECTIONS[0]);
    t2Lin(F, pscRow(F, f9.leaves, eO(), Aff()), {on(0, bitFactor9, w0)});
    F.section(string("smallness test, ") + FOREST_PLANE_SECTIONS[1]);
    t2Lin(F, affSlot(f9.sBit), {on(1, eZ() - eO(), w1)});
    vector<Ex> w(KB);
    for (u64 u = 0; u < KB; u++) w[u] = eB(1ull << (u & 31));
    F.section(string("smallness test, ") + FOREST_PLANE_SECTIONS[2]);
    t2Lin(F, affC(eZ() - eB((1ull << f9.v()) - f9.padding())), {on(2, eO(), w)});
    forestPlanesAdded(f9); // likewise: the three claims above
  }
  Ex bitFactor9 = eZ(); // the factor of b at zb[0] in the first claim above, for `tileSelfTestF`
  // Checks, once all claims are added, that in U1 .. U3, ZZ, BZ and the symbol trees no column has
  // weight 0 in a virtual row meant to read it: no claim would constrain it. A chance 0 fails too.
  void selfTest(const HFork &F, const u64 nU[3]) const { selfTestF(F, nU); }
  // ---- The tiling self-test ----
  static Ex eqAtKey(const Ex *zp, int n, u64 idx) {
    Ex p = eO();
    for (int v = 0; v < n; v++) p = p * (((idx >> (n - 1 - v)) & 1) ? zp[v] : eO() - zp[v]);
    return p;
  }
  void rkSelfTest(const HFork &F) const { tileSelfTestF(F); }
};
