// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The verifier's side of the approximate range check ("tail" or "smallness test" in names) of a
   sub-proof (`fork` in names, as in `HFork`) on the cells of its matrices (mats, the fold list).
   blocks/iop/rng.h describes the protocol and says what it shows. Included inside struct HTailF2
   (verifier/src/vf_smallness_test.h). The main calls, in order: for the plan shapeF, addTreesF; in a run
   closeFuns, transcriptF, objectsF, selfTestF, coefficientsF, tileSelfTestF.
   The check's equations and claims are written out as formulas (chi there is HFork::theta) in
   docs/format/11-1-machinery.md, "The smallness test: values, equations and claims".
   A tree is a committed array; a lin is a linear claim. A symbol packs three cells of a column. A
   symbol tree is PF_i if an earlier sub-proof owns it, PX_i if another also reads it, else PK_i. Cf
   is the matrix that rng.h calls Theta (it is not HFork::theta): tau = T2TAU combinations of a
   matrix's columns. ZZ has their digits (row u: combination u); BZ has the bits of the
   multiplicities in the digits' lookup. A tie is the lin on one matrix's symbols and digits. A
   functional (HFun) is a weight vector phi of length L with its table in the committed vector U3:
   entry (c, sg) is sum_i phi[i] cell[sg L + i][c]. A read is the lin on the entries and digits of
   one row segment sg (cell rows sg L .. sg L + L - 1). */

vector<size_t> pkMat, pxMat; // The matrix (index into mats) of PK_i and of PX_i.
// Per matrix: its symbol tree's shape. Set by addTreesF.
vector<ShP> shp;
vector<u32> nXof;      // nXof[mu] = mu. Written, but nothing reads it.
/* Per matrix (powersF). rowX (vf_smallness_packing.h's Hi): the weights of its symbol rows in its tie, powers of
   theta^Np (Np: `stepF`). loX (Lo) = (1): each tree row is one symbol row. theta is the batching
   element: claim c enters the sub-proof's one equation (doorH, vf_subproof_segment.h) times theta^c. */
vector<vector<Ex>> rowX, loX;
vector<u64> eqFirst; // [mu]: the number of matrix mu's first equation; [NM]: of the first read's
vector<vector<Ex>> gamX, gamV; // gamX[mu][c] = sum_u xi[u] Cf[c][u], xi = vp; gamV: with vt for xi.
// tieObj[mu]: the claim number of mu's tie. tiesOn[t]: ties on tree t.
vector<u32> tieObj, tiesOn;
// objectsF's list of functionals, which tileSelfTestF reads again.
// The caller keeps it alive and unchanged until then (funsHeld).
const vector<HFun> *funLaidF = nullptr;
u64 nUF[3] = {0, 0, 0}; // The lengths of U1 .. U3 (addTreesF).
/* The table parts: the ranges of U1 ([0]) and U3 ([1]) that hold the tables. addTreesF sets U1's
   empty and U3's to all of U3. */
u64 tabFirst[2] = {0, 0}, tabLen[2] = {0, 0};
size_t nX4Laid = 0, funAtClose = 0;
bool funClosed = false;
u32 foldDrawn = 0;             // How many times this run has drawn Cf; each sub-proof requires 1.
u32 slotA = 0, slotB = 0;      // The check's message slots are [slotA, slotB).
vector<Ex> lsTail;             // F.LS on [slotA, slotB) at the end of objectsF.
vector<PlaceF> plOwn, plFor; // Placement records: plOwn per PX_i, plFor per PF_i.
/* A copy of shp that addTreesF takes while the plan is built, before any word of a proof is read.
   symShapesHeldF compares it with shp and Tail2's shapes. */
vector<ShP> shapeRec;
void symShapesHeldF(const HFork &F, const Tail2 &T) const {
  if (shapeRec.size() != NM) refuse(F.G_ + "internal-sym-shape");
  for (size_t mu = 0; mu < NM; mu++) {
    const N2 &n = T.nt[gtTree[mu]];
    if (!n.P || !n.p.same(shapeRec[mu]) || !shp[mu].same(shapeRec[mu])) refuse(F.G_ + "internal-sym-shape");
  }
}

// The ties and reads compute a column of ZZ only with zzCol and a table entry's index only with
// tabAt; both check the range. Keep it so: tileSelfTestF does not recompute the tables' indices.
// ZZ's column for cell row i of matrix mu. plane: 0 for the low digit, 1 for the high digit.
u64 zzCol(size_t mu, int plane, u64 i) const {
  if (mu >= NM || plane < 0 || plane > 1 || i >= mats[mu].m) refuse("internal-f-zzcol");
  return (u64)plane * mz + off[mu] + i;
}
// The index in U1 or U3 of entry (c, sg) of Fn's table.
u64 tabAt(const HFun &Fn, u64 c, u64 sg) const {
  if (Fn.M < 0 || (size_t)Fn.M >= NM || !Fn.L) refuse("internal-t2f-functional-" + Fn.name);
  const TMat &M = mats[Fn.M];
  const u64 nsg = M.m / Fn.L;
  if (c >= M.k || sg >= nsg) refuse("internal-f-table-index");
  return Fn.at + c * nsg + sg;
}
// A virtual row (HRow) weights a tree's rows (coef) and columns (L). ZZ's in order: one per piece
// of lookup call R10z (low digits), then of R8d (high digits), the reads' (coef = vt), the ties'
// (coef = xi). A piece (HPiece) is a block of ZZ's rows (t2BL) by one call's mz columns.
static int zzBlocks() { return (int)t2BL(T2TAU).size(); }
static u32 zzRowReads() { return 2 * (u32)zzBlocks(); }
static u32 zzRowTies() { return 2 * (u32)zzBlocks() + 1; }
// Builds the lookup calls z[1] (R10z) and z[2] (R8d), their pieces, and f9.
void piecesF() {
  auto pc = [](int tree, int row, u64 r0, u64 Rp, u64 c0, u64 Cp) {
    HPiece p;
    p.tree = tree;
    p.row = row;
    p.r0 = r0;
    p.Rp = Rp;
    p.c0 = c0;
    p.Cp = Cp;
    return p;
  };
  for (auto &x : z) x = HCallZ();
  const vector<std::pair<u64, u64>> bz = t2BL(T2TAU);
  if (bz.empty() || bz.size() > 2) refuse("internal-t2f-blocks");
  for (size_t i = 0; i < bz.size(); i++) z[1].pc.push_back(pc(ix.ZZ, (int)i, bz[i].first, bz[i].second, 0, mz));
  for (size_t i = 0; i < bz.size(); i++) z[2].pc.push_back(pc(ix.ZZ, (int)(bz.size() + i), bz[i].first, bz[i].second, mz, mz));
  placeZ(z[1], "R10z", T2D0, LIDB);
  placeZ(z[2], "R8d", T2D1, LIDB);
  for (int q = 1; q <= 2; q++)
    if (z[q].c.n < 10) refuse("hid-call-floor");
  /* f9: the lookup forest (product trees) that both calls share. f9.pub[y] counts the padding keys
   * of f9's tree y: columns Cp .. CP - 1 of each piece's rows, and all keys after the last piece.
   * Nothing here reads placeZ's pub. */
  f9 = forestShape({&z[1].c, &z[2].c}, LIDB);
  f9.cl.clear();
  f9Ran = false;
  f9.pub.assign(f9.NT, 0);
  for (int q = 0; q < 2; q++) {
    const HCallZ &c = z[1 + q];
    u64 end = 0;
    for (auto &p : c.pc) {
      if (p.Cp < p.CP)
        for (u64 rr = 0; rr < p.Rp; rr++) forestPubRun(f9, f9.pub, q, p.off + rr * p.CP + p.Cp, p.CP - p.Cp);
      end = std::max<u64>(end, p.off + (1ull << p.b));
    }
    forestPubRun(f9, f9.pub, q, end, (1ull << c.c.n) - end);
  }
  forestPubAsk(f9, {&z[1].c, &z[2].c}, f9.pub); // at plan time, not only when a proof is read
}

/* Sets the check's sizes and tree list (ix) from the fold list m. TMat::kind 0: own matrix; 1:
   foreign, an earlier sub-proof's, in tree f[TMat::foreign]. txOf[mu] = i >= 0: own matrix mu is
   PX_i. */
void shapeF(const vector<TMat> &m, const vector<int> &txOf, const vector<T2Foreign> &f, const bool hasU[4], u32 tau) {
  mats = m;
  ft = f;
  NM = mats.size();
  mz = 0;
  off.clear();
  col0.clear();
  gtTree.clear();
  gtRow.clear();
  kTX.clear();
  pkMat.clear();
  pxMat.clear();
  shp.clear();
  nXof.clear();
  if (tau != T2TAU) refuse("hid-tau");
  if (txOf.size() != NM || hasU[3]) refuse("internal-t2f-lists"); // no sub-proof has U4
  for (size_t mu = 0; mu < NM; mu++)
    if (mats[mu].kind == 0) {
      if (txOf[mu] >= 0) {
        if ((size_t)txOf[mu] != pxMat.size()) refuse("internal-t2f-TX-order");
        pxMat.push_back(mu);
        kTX.push_back(mats[mu].k);
      } else
        pkMat.push_back(mu);
    }
  if (NM >= 256) refuse("plan-link");
  // The list of trees, in order: U1 .. U3 (those that the sub-proof has), PK_i, PX_i, ZZ, BZ, PF_i.
  ix = T2Map();
  for (int n = 0; n < 4; n++)
    if (hasU[n]) ix.U[n] = ix.count++;
  for (size_t i = 0; i < pkMat.size(); i++) ix.PK.push_back(ix.count++);
  for (size_t i = 0; i < pxMat.size(); i++) ix.TX.push_back(ix.count++);
  ix.ZZ = ix.count++;
  ix.BZ = ix.count++;
  for (size_t i = 0; i < ft.size(); i++) ix.FT.push_back(ix.count++);
  size_t nextF = 0, nPK = 0;
  plOwn.assign(pxMat.size(), PlaceF());
  plFor.assign(ft.size(), PlaceF());
  for (size_t mu = 0; mu < NM; mu++) {
    const TMat &M = mats[mu];
    if (!M.m) refuse("shape-matrix-" + M.name);
    if (!M.k || M.k > KCAP) refuse("hid-k-range");
    off.push_back(mz);
    mz += M.m;
    col0.push_back(0);
    gtRow.push_back(0);
    if (M.kind == 0 && txOf[mu] < 0) {
      gtTree.push_back(ix.PK[nPK]);
      nXof.push_back(48 + (u32)nPK);
      nPK++;
    } else if (M.kind == 0) {
      gtTree.push_back(ix.TX[txOf[mu]]);
      nXof.push_back(32 + (u32)txOf[mu]);
      PlaceF &pl = plOwn[txOf[mu]];
      pl.col0 = col0.back();
      pl.k = M.k;
      pl.m = M.m;
      pl.haveMatrix = true;
    } else if (M.kind == 1) {
      /* A foreign matrix must fill the tree that ft[M.foreign] describes: the same m, k and wp,
         first column 0. ft is this sub-proof's own description: the plan compares the tree made
         from it with the owner's (t2SameShape, vf_subproof_segment.h). One PF_i per matrix, in ft's order. */
      if (M.foreign < 0 || (size_t)M.foreign >= ft.size() || M.ttOff || M.k != ft[M.foreign].K ||
          M.m != ft[M.foreign].m || M.wp != ft[M.foreign].wp)
        refuse("shape-foreign-" + M.name);
      if ((size_t)M.foreign != nextF) refuse("internal-t2f-ZF-order");
      nextF++;
      gtTree.push_back(ix.FT[M.foreign]);
      nXof.push_back(10 + (u32)M.foreign);
      {
        PlaceF &pl = plFor[M.foreign];
        pl.col0 = M.ttOff;
        pl.k = M.k;
        pl.m = M.m;
        pl.haveMatrix = true;
      }
    } else
      refuse("t2f-weight-class");
  }
  if (nextF != ft.size()) refuse("internal-t2f-ZF-order");
  // Replaces the values pushed above.
  for (size_t mu = 0; mu < NM; mu++) nXof[mu] = (u32)mu;
  if (2 * mz >= (1ull << 32)) refuse("t2f-columns-range");
  piecesF();
}
/* The number of lins: docs/format/11-1-machinery.md, "The claims, in order". The 4: the join
   (linJoin9) and three on BZ (linBits9). */
size_t linsF(size_t nX4) const {
  return forestLins(f9) + 4 + (size_t)NM + nX4;
}
/* The plan records here which tree PF_i, i = iF, is: the owner's number, and the tree's position
   among the owner's symbol trees (PK_i, then PX_i). They can be set only once. The plan then
   compares plFor[iF] with the owner's record (samePlaceF, vf_smallness_packing.h). */
void borderMeans(size_t iF, int ownerNo, int index) {
  if (iF >= plFor.size() || plFor[iF].owner >= 0 || ownerNo < 0 || index < 0) refuse("internal-border-count");
  plFor[iF].owner = ownerNo;
  plFor[iF].rootIndex = index;
}
// no: this sub-proof's number as an owner.
void ownerIs(int no) {
  for (auto &pl : plOwn) pl.owner = no;
}
/* The owner's placement record of the matrix named "qa" (role "qa") or "xbO" (any other role).
 * Call it on the owner's object: the record must never be looked up through a field of a reader,
 * a sub-proof that reads the matrix. */
const PlaceF &ownEntry(const char *role) const {
  const string want = !strcmp(role, "qa") ? "qa" : "xbO";
  for (size_t i = 0; i < pxMat.size(); i++)
    if (mats[pxMat[i]].name == want) {
      if (!plOwn[i].whole()) refuse("internal-border-count");
      return plOwn[i];
    }
  refuse("internal-border-count");
}
// The sub-proof sets the pointer at its start.
mutable const Tail2 *runT = nullptr;

// ---- The trees in list order. ----
void addTreesF(HFork &F, Tail2 &T, const u64 nU[3], const vector<u32> &aTX, u32 hb) {
  // This must run before the sub-proof's first tree is added.
  if (!T.nt.empty()) refuse("internal-t2-order");
  T.link = true;
  if (aTX.size() != pxMat.size()) refuse("internal-t2f-lists");
  // Bounds the parameter file's line `sec` (statedTarget.hb), not the argument hb (less by POW).
  // cp_verify.cpp makes the same test, under the same name.
  if ((u32)T2TAU < statedTarget.hb) refuse("policy-tau-hb");
  auto must = [&](u32 got, int want) {
    if ((int)got != want) refuse(F.G_ + "internal-t2f-map");
  };
  const char *un[4] = {"U1", "U2", "U3", "U4"};
  for (int n = 0; n < 3; n++) nUF[n] = nU[n];
  tabFirst[0] = tabLen[0] = tabFirst[1] = 0;
  tabLen[1] = ix.U[2] >= 0 ? nU[2] : 0;
  if (ix.U[3] >= 0) refuse(F.G_ + "internal-t2f-lists");
  for (int n = 0; n < 3; n++)
    if (ix.U[n] >= 0)
      must(t2AddN(F, T, un[n], 1, nU[n], CE, 1, false, 1), ix.U[n]);
  for (size_t i = 0; i < pkMat.size(); i++) {
    const TMat &M = mats[pkMat[i]];
    must(t2AddP(F, T, "PK" + std::to_string(i), M.m, M.k, M.wp, 1, false), ix.PK[i]);
  }
  for (size_t i = 0; i < pxMat.size(); i++) {
    const TMat &M = mats[pxMat[i]];
    must(t2AddP(F, T, "PX" + std::to_string(i), M.m, M.k, M.wp, aTX[i], false), ix.TX[i]);
  }
  must(t2AddN(F, T, "ZZ", T2TAU, 2 * mz, 1, 1, false, 2 * zzBlocks() + 2), ix.ZZ);
  must(t2AddN(F, T, "BZ", 32, bzColumns(), 1, 1, false, 3), ix.BZ);
  for (size_t i = 0; i < ft.size(); i++)
    must(t2AddP(F, T, ft[i].name, ft[i].m, ft[i].K, ft[i].wp, ft[i].a, true), ix.FT[i]);
  // Fills pl from the shape of tree t.
  auto declared = [&](PlaceF &pl, int t) {
    const N2 &n = T.nt[t];
    pl.sh = n.p;
    pl.haveShape = true;
    pl.hb = hb;
    pl.mS = n.p.mS;
    pl.lanes = LANES;
    pl.padRows = n.p.R * n.p.J;
    pl.J = n.p.J;
    pl.K = n.p.k;
    pl.off = 0;
    pl.w = (u32)n.p.w;
  };
  for (size_t i = 0; i < pxMat.size(); i++) {
    declared(plOwn[i], ix.TX[i]);
    // The position of PX_i among the owner's symbol trees (the PK_i, then the PX_i).
    plOwn[i].rootIndex = (int)(pkMat.size() + i);
  }
  for (size_t i = 0; i < ft.size(); i++) declared(plFor[i], ix.FT[i]);
  shp.assign(NM, ShP());
  for (size_t mu = 0; mu < NM; mu++) {
    shp[mu] = T.nt[gtTree[mu]].p;
    if (shp[mu].m != mats[mu].m || shp[mu].k != mats[mu].k)
      refuse(F.G_ + "internal-t2f-map");
  }
  shapeRec = shp;
}
/* The sub-proof closes its list of functionals here, before Cf is drawn: transcriptF requires it.
 * Its function that adds a functional calls `funOpen` first. */
void closeFuns(const vector<HFun> &fun) {
  if (funClosed) refuse("internal-f-fun-closed");
  funClosed = true;
  funAtClose = fun.size();
  funDigest(fun, funHash);
}
/* The list of functionals must not change after it is closed. closeFuns records its length
 * (funAtClose) and a hash (funHash). funsHeld computes both again and compares them: objectsF calls
 * it at its start, and each sub-proof calls it again after its last claim. The hash is internal to
 * this program and is sent nowhere. */
u32 funHash[8] = {0, 0, 0, 0, 0, 0, 0, 0};
/* The hash covers five fields of every functional: M, L, tree, at and phi. It is computed with
 * mix64 (vf_common.h) and guards against a bug in this program, not against an adversary: only this
 * program writes the list. */
static void funDigest(const vector<HFun> &fun, u32 out[8]) {
  vector<u64> d(fun.size(), 0);
  parFor(fun.size(), [&](u64 i) {
    const HFun &f = fun[i];
    u64 h = mix64(0x46313846ull + i);
    for (u64 v : {(u64)(i64)f.M, f.L, (u64)f.tree, f.at, (u64)f.phi.size()}) h = mix64(h + v);
    // One chain per word of an element, so that the CE calls of mix64 can overlap.
    u64 c[CE];
    for (int k = 0; k < CE; k++) c[k] = mix64(h + (u64)k);
    for (auto &x : f.phi)
      for (int k = 0; k < CE; k++) c[k] = mix64(c[k] + eC(x, k));
    for (int k = 0; k < CE; k++) h = mix64(h + c[k]);
    d[i] = h;
  });
  u64 a = mix64(fun.size()), b = mix64(~(u64)fun.size());
  for (u64 v : d) {
    a = mix64(a + v);
    b = mix64(b ^ v);
  }
  for (int i = 0; i < 8; i++) out[i] = 0;
  out[0] = (u32)a;
  out[1] = (u32)(a >> 32);
  out[2] = (u32)b;
  out[3] = (u32)(b >> 32);
}
void funsHeld(const HFork &F, const vector<HFun> &fun) const {
  u32 h[8];
  funDigest(fun, h);
  if (!funClosed || fun.size() != funAtClose || memcmp(h, funHash, 32)) refuse(F.G_ + "internal-f-fun-moved");
}
void funOpen() const {
  if (funClosed) refuse("internal-f-fun-closed");
}

/* Draws Cf, the coefficients in {-1, 0, 1} of the tau combinations (0 with probability 1/2), into
 * cfF: one vector for the entire sub-proof, in the order of blocks/iop/rng.h: matrix by matrix in
 * the fold list's order, column by column, tau entries a column. They are one round of the
 * hash chain. */
void drawMatrixF(Tr &tr) {
  cfFirst.assign(NM + 1, 0);
  for (size_t mu = 0; mu < NM; mu++) cfFirst[mu + 1] = cfFirst[mu] + mats[mu].k;
  cfF = trTrits(tr, (size_t)cfFirst[NM], (size_t)T2TAU);
}
// Plans the check's rounds on the hash chain. which = 0: Cf, after the out-of-domain sample of the
// reads' message. which = 1: the lookup on the digits, after the sample of ZZ and BZ. The equations
// have no round: theta's round must be planned with (number of claims) + equationsF(nX4) rows.
void planRounds(blk::Plan &p, const std::string &where, unsigned which) const {
  if (which == 0) {
    u64 columns = 0;
    for (auto &M : mats) columns += M.k;
    if (blk::rng_combinations(p) != (unsigned)T2TAU) refuse("plan-tau");
    blk::plan_rng(p, where, columns);
    return;
  }
  if (which != 1) refuse("internal-plan-moment");
  forestPlan(p, where + ", digits", f9);
}
/* The numbers of the check's equations and their powers of theta:
 * docs/format/11-1-machinery.md, "The equations". Its tauP (Np below) is stepF(), its first_mu is
 * eqFirst[mu], its NCm is planClaimsF, its chi is theta, and its Q is equationsF(nX4) for NX = nX4
 * reads. */
static u64 stepF() { return 1ull << clog2((u64)T2TAU); }
u64 equationsF(size_t nX4) const {
  u64 n = 0;
  for (size_t mu = 0; mu < NM; mu++) n += stepF() << clog2((mats[mu].m + 2) / 3);
  return n + (u64)nX4 * stepF();
}
// equationsF() for the nX4Laid reads that objectsF has added.
// The matrices' share is no second count: powersF computes it by the same arithmetic.
u64 equationsLaidF() const {
  if (eqFirst.size() != NM + 1) refuse("internal-f-draws");
  return eqFirst[NM] + (u64)nX4Laid * stepF();
}
// Once theta is drawn: sets vt = vp = (theta^u), rowX, loX, eqFirst.
void powersF(const HFork &F) {
  if (!F.thetaDrawnF || !F.planClaimsF) refuse(F.G_ + "internal-f-draws");
  vt.assign(T2TAU, eO());
  for (u32 u = 1; u < T2TAU; u++) vt[u] = vt[u - 1] * F.theta;
  vp = vt;
  const Ex step = ePow(F.theta, stepF());
  eqFirst.assign(NM + 1, 0);
  rowX.assign(NM, {});
  loX.assign(NM, {});
  for (size_t mu = 0; mu < NM; mu++) {
    const ShP &s = shp[mu];
    if (s.J != 1 || s.Rd != 0 || s.R != s.mS || s.Rr != s.mS || s.mS != (mats[mu].m + 2) / 3) refuse(F.G_ + "internal-sym-shape");
    eqFirst[mu + 1] = eqFirst[mu] + (stepF() << clog2(s.mS));
    rowX[mu].assign(s.mS, eO());
    for (u64 i = 1; i < s.mS; i++) rowX[mu][i] = rowX[mu][i - 1] * step;
    loX[mu].assign(1, eO());
  }
}
// The factor that raises the next claim's theta^nClaim to theta^(planClaimsF + first).
Ex leadF(const HFork &F, u64 first) const {
  if (F.nClaim >= F.planClaimsF) refuse(F.G_ + "internal-f-draws");
  return ePow(F.theta, (u64)(F.planClaimsF - F.nClaim) + first);
}

/* Called just before Cf is drawn. Every array that Cf multiplies must be fixed by then: each
 * matrix's symbol tree, the sub-proof's own or an earlier sub-proof's, and the committed vectors
 * over E (U1 .. U3; U3 contains the reads' tables). For each, the oracle message that contains it
 * in this sub-proof's segment of the hash chain (`link` in names) must have been received on this
 * chain, with the answers to its sample in a prover's message (`blk::Oracle::sampled`);
 * rng_shape_check (blocks/iop/rng.h) requires the same. Otherwise a prover could add to a table's
 * entries a vector that Cf maps to 0: every equation would hold, but the entries would be the sums
 * of no small cells. An array's commit point (`moment` in names) follows from this program's own
 * lists (verifier/src/vf_segments.inc); `layout.txt` chooses only the oracle message at that commit
 * point. So a failure is a bug in this program: cp_verify ends `NOT CHECKED`, not `NO`. */
void fixedBeforeTheMatrixF(const HFork &F) const {
  auto no = [&]() { refuse(F.G_ + "internal-f-matrix-order"); };
  if (!currentSegment || !currentSegment->check || !currentSegment->V || !currentSegment->arrayOf || !currentSegment->layout) no();
  const blk::ReqFrame &frame = currentSegment->check->frame();
  const size_t here = currentSegment->link;
  if (frame.link() != here || here >= frame.shape().link.size()) no();
  auto fixed = [&](int tree) {
    if (tree < 0 || (size_t)tree >= currentSegment->arrayOf->size()) no();
    const std::pair<size_t, string> &at = (*currentSegment->arrayOf)[(size_t)tree];
    const blk::ReqArray *a = at.first <= here ? currentSegment->layout->find(at.first, at.second) : nullptr;
    if (!a || a->message >= frame.shape().link[at.first].messages.size()) no();
    const blk::ReqMessage &m = frame.shape().link[at.first].messages[a->message];
    // This segment's own array: its commit point has come. An earlier segment's: its rows are kept
    // up to this segment, so they were sent again in this segment's first prover message. Otherwise
    // ReqFrame::where() would end the process.
    if (at.first == here ? m.moment >= currentSegment->moments : at.first + m.life < here) no();
    if (!currentSegment->V->st.oracle(frame.where(at.first, a->message)).sampled) no();
  };
  if (gtTree.size() != NM) no();
  for (size_t mu = 0; mu < NM; mu++) fixed(gtTree[mu]);
  for (int n = 0; n < 4; n++)
    if (ix.U[n] >= 0) fixed(ix.U[n]);
}

// ---- transcript: Cf, the commitment to ZZ and BZ, the lookup on the digits ----
void transcriptF(HFork &F, Tr &tr, u32 &slot) {
  if (!funClosed) refuse(F.G_ + "internal-f-fun-open");
  if (shp.size() != NM) refuse(F.G_ + "internal-t2f-map");
  slotA = slot;
  /* The error 2^-tau of Cf's round, with no factor for a list of codewords, rests on this order: Cf
   * is drawn only after the phi are fixed (above) and every oracle message with the check's cells
   * or reads has had its sample answered: the sub-proof's first prover message (its packed
   * matrices; earlier sub-proofs' kept ones, sent again) and the reads' (its second commit point).
   * Cells and reads are then one fixed object. fixedBeforeTheMatrixF checks that these samples are
   * answered. The order also rests on the callers (rootInH on U3 is their last call on the chain
   * before this one); on the plan (planRounds; verifier/src/vf_segments.inc says which array is
   * committed at which commit point); on rootInH (verifier/src/vf_subproof_segment.h), which accepts a
   * commitment only if the trees named are exactly the layout's arrays at that commit point; and on
   * the channel (blocks/chain/rounds.h), which accepts only the next planned round and closes no
   * round while answers to a sample are missing. */
  fixedBeforeTheMatrixF(F);
  drawMatrixF(tr);
  foldDrawn++;
  rootInH(F, tr, {ix.ZZ, ix.BZ});
  // Both roots must be known before the first challenge of the lookup calls.
  if (!F.trees[ix.ZZ].root || !F.trees[ix.BZ].root) refuse(F.G_ + "internal-f-order");
  // The lookup on the digits (vf_lookup_argument.h).
  {
    const vector<u64> pub9 = f9.pub;
    f9 = forestPad(F, tr, {&z[1].c, &z[2].c}, pub9, slot, LIDB);
    f9Ran = true;
  }
  if (!runT) refuse(F.G_ + "internal-root-behind-sd");
  slotB = slot;
}
// Sets the row weights of the virtual rows: coef, written out, and the same as a product over the
// bits of the row's number (HRow::rowForm).
void coefficientsF(HFork &F) const {
  auto point = [](HRow &row, const Ex *z, int n, u64 first) {
    row.rowForm = true, row.rowOnes = false;
    row.rowPoint.assign(z, z + n);
    row.rowFirst = first;
  };
  auto powers = [](HRow &row, Ex base, u64 rows) { // coef[r] = base^r: bit t of r has the pair (1, base^(2^t)), bit 0 first
    row.rowForm = true, row.rowOnes = false;
    row.rowPoint.clear();
    row.rowFirst = 0;
    row.rowLo.assign((size_t)clog2(rows), eO());
    row.rowHi.clear();
    for (int t = 0; t < clog2(rows); t++, base = base * base) row.rowHi.push_back(base);
  };
  auto ones = [](HRow &row) {
    row.rowForm = true, row.rowOnes = true;
    row.rowPoint.clear();
    row.rowFirst = 0;
  };
  for (int n = 0; n < 4; n++)
    if (ix.U[n] >= 0 && F.trees[ix.U[n]].kind == 'N') {
      F.trees[ix.U[n]].rows[0].coef.assign(1, eO());
      ones(F.trees[ix.U[n]].rows[0]);
    }
  for (int q = 1; q <= 2; q++)
    for (auto &p : z[q].pc) {
      const vector<Ex> e = eqTab(zrOf(z[q], p), p.a);
      vector<Ex> w(F.trees[p.tree].R, eZ());
      for (u64 r = 0; r < p.Rp; r++) w[p.r0 + r] = e[r];
      F.trees[p.tree].rows[p.row].coef = w;
      point(F.trees[p.tree].rows[p.row], zrOf(z[q], p), p.a, p.r0);
    }
  F.trees[ix.ZZ].rows[zzRowReads()].coef = vt;
  F.trees[ix.ZZ].rows[zzRowTies()].coef = vp;
  for (u32 r : {(u32)zzRowReads(), (u32)zzRowTies()}) powers(F.trees[ix.ZZ].rows[r], F.theta, (u64)T2TAU);
  for (int e = 0; e < 2; e++) {
    F.trees[ix.BZ].rows[e].coef = eqTab(f9.zb[e].data(), 5);
    point(F.trees[ix.BZ].rows[e], f9.zb[e].data(), 5, 0);
  }
  F.trees[ix.BZ].rows[2].coef.assign(32, eO());
  ones(F.trees[ix.BZ].rows[2]);
  // A symbol tree's one virtual row is for its tie.
  for (size_t mu = 0; mu < NM; mu++) {
    F.trees[gtTree[mu]].rows[0].coef = rowX[mu];
    powers(F.trees[gtTree[mu]].rows[0], ePow(F.theta, stepF()), shp[mu].mS);
  }
}
/* One flag per matrix: foldSums ends without a verdict on a second call for the same matrix, and
 * each sub-proof checks at its end (foldAllTakenF) that every matrix has had its call. */
mutable vector<u8> foldTakenF;
void foldAllTakenF(const HFork &F) const {
  if (foldTakenF.size() != NM) refuse(F.G_ + "internal-f-draws");
  for (u8 x : foldTakenF)
    if (x != 1) refuse(F.G_ + "internal-f-draws");
}
/* For each column c of matrix mu: gx[c] = sum_u vp[u] Cf[c][u], for the tie, and gv[c], with vt for
 * vp, for the reads (Cf is in cfF). Only tileSelfTestF computes these sums a second time. */
void foldSums(size_t mu, vector<Ex> &gx, vector<Ex> &gv) const {
  if (foldTakenF.size() != NM) foldTakenF.assign(NM, 0);
  if (mu >= NM || foldTakenF[mu]) refuse("internal-f-draws");
  foldTakenF[mu] = 1;
  const u64 k = mats[mu].k;
  gx.assign(k, eZ());
  gv.assign(k, eZ());
  parFor((k + 255) / 256, [&](u64 b) {
    for (u64 c = b * 256; c < k && c < (b + 1) * 256; c++) {
      const signed char *cf = cfF.data() + (size_t)(cfFirst[mu] + c) * T2TAU;
      u128 px[CE] = {0}, nx[CE] = {0}, pv[CE] = {0}, nv[CE] = {0};
      for (u32 u = 0; u < T2TAU; u++) {
        if (!cf[u]) continue;
        const bool neg = cf[u] < 0;
        for (int i = 0; i < CE; i++) {
          (neg ? nx : px)[i] += eC(vp[u], i);
          (neg ? nv : pv)[i] += eC(vt[u], i);
        }
      }
      for (int i = 0; i < CE; i++) { // four 128-bit sums a component (+1 and -1 terms of gx, of gv), each reduced once
        eSet(gx[c], i, fsub(fredW(px[i]), fredW(nx[i])));
        eSet(gv[c], i, fsub(fredW(pv[i]), fredW(nv[i])));
      }
    }
  });
}
// ---- the claims, in this order: the lookup's, the ties, the reads ----
void objectsF(HFork &F, const vector<HFun> &fun) {
  // `fun` must be the list that closeFuns closed (see funHash).
  if (!funClosed || fun.size() != funAtClose) refuse(F.G_ + "internal-f-fun-closed");
  funsHeld(F, fun); // its hash must also match the one recorded at closing
  funLaidF = &fun;
  // Slot check: the weights on the check's message slots (one per value sent masked) must come from
  // objectsF's claims only. First half: F.LS, the claims' weights on slots, is 0 on [slotA, slotB).
  for (u32 j = slotA; j < slotB; j++)
    if (!isZ(F.LS[j])) refuse(F.G_ + "internal-f-tail-slot");
  const Ex m1 = eZ() - eO();
  // Run2: weight coef * w[w0+i] on column first+i of a virtual row, for i < n.
  auto run = [](int tree, u32 row, u64 first, Ex coef, const vector<Ex> &w, u64 w0, u64 n) {
    Run2 r;
    r.tree = (u32)tree;
    r.row = row;
    r.first = first;
    r.coef = coef;
    r.w = &w;
    r.w0 = w0;
    r.n = n;
    return r;
  };
  // `laid` checks the number of claims added since `before`.
  // The numbers: docs/format/11-1-machinery.md, "The claims, in order".
  auto laid = [&](u32 before, size_t want) {
    if ((size_t)(F.nClaim - before) != want) refuse(F.G_ + "internal-f-exist");
  };
  if (!f9Ran || f9.cl.size() != 2 || f9.cl[0] != &z[1].c || f9.cl[1] != &z[2].c) refuse(F.G_ + "internal-f-exist");
  const string sectionBefore = F.sectionF;  // sections only label claims for tests (../tests/claims_report.h)
  {
    u32 b = F.nClaim;
    F.section("smallness test");  // `forestObjects` prefixes its own section names with this one
    forestObjects(F, f9);
    laid(b, 31 + (size_t)f9.v() + (size_t)f9.d() + 1);
    b = F.nClaim;
    linJoin9(F);
    linBits9(F);
    laid(b, 4);
  }
  powersF(F);
  gamX.assign(NM, {});
  gamV.assign(NM, {});
  for (size_t mu = 0; mu < NM; mu++) foldSums(mu, gamX[mu], gamV[mu]);
  Ex sx = eZ(), sv = eZ();
  for (auto &x : vp) sx = sx + x;
  for (auto &x : vt) sv = sv + x;
  // The ties, mu ascending, one linear claim each: digits (wl) - offset - symbols (wr) = 0.
  // The offset 2^LBZ: val in docs/format/11-1-machinery.md, "Notation".
  tieObj.assign(NM, ~0u);
  tiesOn.assign(F.trees.size(), 0);
  for (size_t mu = 0; mu < NM; mu++) {
    const ShP &s = shp[mu];
    const vector<Ex> wl = tieLeft(s, rowX[mu], loX[mu]), wr = tieRight(s, loX[mu], gamX[mu]);
    Ex sw = eZ();
    for (auto &x : wl) sw = sw + x;
    tieObj[mu] = F.nClaim;
    F.section("smallness test: tie of matrix " + std::to_string(mu));
    const Ex lead = leadF(F, eqFirst[mu]);
    t2Lin(F, affC(eZ() - lead * eS(sx * sw, 1ull << LBZ)),
          {run(ix.ZZ, zzRowTies(), zzCol(mu, 0, 0), lead, wl, 0, s.m),
           run(ix.ZZ, zzRowTies(), zzCol(mu, 1, 0), eS(lead, 1ull << T2D0), wl, 0, s.m),
           run(gtTree[mu], 0, 0, m1 * lead, wr, 0, s.J * s.k)});
    tiesOn[gtTree[mu]]++;
  }
  /* The reads (X4): one linear claim per functional and row segment sg. Order: mu ascending;
   * functionals on U1, then those on U3; sg ascending. The reads must cover every entry of the
   * table parts (tabFirst, tabLen) exactly once, with a non-zero weight, and nothing else. */
  vector<char> rdU[2];
  for (int e = 0; e < 2; e++) rdU[e].assign(tabLen[e], 0);
  nX4Laid = 0;
  for (size_t mu = 0; mu < NM; mu++) {
    const TMat &M = mats[mu];
    const vector<Ex> &gc = gamV[mu];
    for (u32 un : {0u, 2u})
      for (auto &Fn : fun) {
        if ((size_t)Fn.M != mu || Fn.tree != un) continue;
        if (ix.U[un] < 0) refuse("internal-t2f-unit-" + Fn.name);
        if (!Fn.L || M.m % Fn.L || Fn.phi.size() != Fn.L) refuse("shape-functional-" + Fn.name);
        const u64 nsg = M.m / Fn.L;
        vector<char> &rdT = rdU[un / 2];
        const u64 t0 = tabFirst[un / 2];
        Ex sphi = eZ();
        for (auto &x : Fn.phi) sphi = sphi + x;
        for (u64 sg = 0; sg < nsg; sg++) {
          const Ex lead = leadF(F, eqFirst[NM] + (u64)nX4Laid * stepF());
          Aff f = affC(eZ() - lead * eS(sphi * sv, 1ull << LBZ));
          for (u64 c = 0; c < M.k; c++) {
            const u64 at = tabAt(Fn, c, sg);
            if (at < t0 || at - t0 >= rdT.size() || rdT[at - t0] || isZ(gc[c])) refuse(F.G_ + "internal-f-table-read");
            rdT[at - t0] = 1;
            affAdd(f, uEnt(F, (u32)ix.U[un], at), eZ() - lead * gc[c]);
          }
          F.section("smallness test: read " + Fn.name + ", segment " + std::to_string(sg));
          t2Lin(F, f,
                {run(ix.ZZ, zzRowReads(), zzCol(mu, 0, sg * Fn.L), lead, Fn.phi, 0, Fn.L),
                 run(ix.ZZ, zzRowReads(), zzCol(mu, 1, sg * Fn.L), eS(lead, 1ull << T2D0), Fn.phi, 0, Fn.L)});
          nX4Laid++;
        }
      }
  }
  F.section(sectionBefore);
  for (auto &Fn : fun)
    if (Fn.M < 0 || (size_t)Fn.M >= NM || (Fn.tree != 0 && Fn.tree != 2)) refuse("internal-t2f-functional-" + Fn.name);
  for (int e = 0; e < 2; e++)
    for (char c : rdU[e])
      if (!c) refuse(F.G_ + "internal-f-table-read");
  /* Sorted by first entry, the tables must follow each other without gap or overlap
   * and fill their table part. iv: per table, first entry, entries, row segments.
   * nsgSum: the expected number of reads. */
  size_t nsgSum = 0;
  for (u32 un : {0u, 2u}) {
    vector<std::array<u64, 3>> iv;
    for (auto &Fn : fun)
      if (Fn.tree == un) iv.push_back({Fn.at, mats[Fn.M].k * (mats[Fn.M].m / Fn.L), mats[Fn.M].m / Fn.L});
    std::sort(iv.begin(), iv.end());
    u64 at = tabFirst[un / 2];
    for (auto &x : iv) {
      if (!x[1]) refuse(F.G_ + "internal-t2f-table-outside");
      if (x[0] < at) refuse(F.G_ + "internal-t2f-tables-overlap");
      if (x[0] > at) refuse(F.G_ + "internal-f-tables-gap");
      at += x[1];
      nsgSum += x[2];
    }
    if (at != tabFirst[un / 2] + tabLen[un / 2]) refuse(F.G_ + "internal-f-tables-gap");
    if (ix.U[un] >= 0 && at > nUF[un]) refuse(F.G_ + "internal-t2f-table-outside");
  }
  /* Counts the claims: one tie on each symbol tree (kind 'P'), none elsewhere; NM ties, distinct
   * in tree and claim number; nsgSum reads. */
  {
    size_t ties = 0;
    for (size_t t = 0; t < F.trees.size(); t++) {
      const bool sym = F.trees[t].kind == 'P';
      if (tiesOn[t] != (sym ? 1u : 0u)) refuse(F.G_ + "internal-f-exist");
      ties += tiesOn[t];
    }
    if (ties != NM || nX4Laid != nsgSum) refuse(F.G_ + "internal-f-exist");
    for (size_t mu = 0; mu < NM; mu++)
      for (size_t nu_ = mu + 1; nu_ < NM; nu_++)
        if (tieObj[mu] == tieObj[nu_] || gtTree[mu] == gtTree[nu_]) refuse(F.G_ + "internal-f-exist");
  }
  // Kept for the second half of the slot check, in selfTestF.
  lsTail.assign(F.LS.begin() + slotA, F.LS.begin() + slotB);
}
// ---- self-tests of the claims' weights, run after the sub-proof's last claim ----
// selfTestF stops at a column weight (L) of 0: no claim then constrains that column through that
// virtual row. It covers U1 .. U3 (lengths nU) and the symbol trees; each column of ZZ in every
// virtual row of its lookup call, in the reads' row and in the ties' row; each column of BZ in all
// three rows. It does not read the row weights (coef), which coefficientsF sets later. L is a sum
// over all claims: a nonzero L shows that some claim reads the column, not that each claim that
// should read it does. Weights depend on the plan and the challenges (in a layer, also on the
// selector values as sent, masked: `shown` in identifiers, vf_layers_check.inc). A 0 is a bug or, with
// negligible probability, chance; the verifier, cp_verify, then ends `NOT CHECKED`
// (internalName, vf_common.h), which is no accept.
void selfTestF(const HFork &F, const u64 nU[3]) const {
  auto need = [&](int t, u32 r, u64 c) {
    if (isZ(F.trees[t].rows[r].L[c])) refuse(F.G_ + "internal-t2f-untyped-" + F.trees[t].name);
  };
  for (int n = 0; n < 3; n++)
    if (ix.U[n] >= 0) {
      const u64 Kw = F.trees[ix.U[n]].K;
      for (u64 x = 0; x < nU[n]; x++) need(ix.U[n], (u32)(x / Kw), x % Kw);
    }
  const u32 blocksZZ = (u32)zzBlocks(), readsZZ = zzRowReads(), tiesZZ = zzRowTies(); // each makes a list to count it: once, not at every column
  for (u64 c = 0; c < mz; c++) {
    for (u32 b = 0; b < blocksZZ; b++) {
      need(ix.ZZ, b, c);
      need(ix.ZZ, blocksZZ + b, mz + c);
    }
    need(ix.ZZ, readsZZ, c);
    need(ix.ZZ, readsZZ, mz + c);
    need(ix.ZZ, tiesZZ, c);
    need(ix.ZZ, tiesZZ, mz + c);
  }
  for (u64 x = 0; x < bzColumns(); x++) {
    need(ix.BZ, 0, x);
    need(ix.BZ, 1, x);
    need(ix.BZ, 2, x);
  }
  for (size_t mu = 0; mu < NM; mu++)
    for (u64 c = 0; c < shp[mu].J * shp[mu].k; c++) need(gtTree[mu], 0, c);
  // Slot check, second half: no claim after objectsF has changed F.LS on these slots.
  if (lsTail.size() != (size_t)(slotB - slotA)) refuse(F.G_ + "internal-f-tail-slot");
  for (u32 j = slotA; j < slotB; j++)
    if (!(F.LS[j] == lsTail[j - slotA])) refuse(F.G_ + "internal-f-tail-slot");
}
/* The tiling self-test; call it after coefficientsF. The pieces of the two lookup calls must tile
 * ZZ. A second computation checks the join's weights on ZZ, the ties' left and right weights, and
 * BZ's row and column weights. It cannot detect an error in mz, NM or ix: both computations take
 * these from shapeF. Of the reads it checks vt, gamV and the column weights on ZZ, and with these
 * the power of theta of each read whose weights phi are not all 0. In the head's sub-proof
 * (vf_output_head_check.inc), if the request's range is the last position alone (r = T - 1, e = T), hf@phiZ,
 * hr@phiC and hr@rhoC have phi = 0: their reads add nothing to ZZ, and no self-test compares their
 * powers. It does not check the reads' weights on U1 and U3, their constant terms, or that a claim
 * and a read use the same index for a table entry. These rest on objectsF's one computation, its
 * checks of the tables' tiling and the number of reads, and selfTestF. An error there would make
 * honest proofs fail, unless the prover's code had it too. */
void tileSelfTestF(const HFork &F) const {
  auto bad = [&]() {
    refuse(F.G_ + "internal-t2f-untiled");
  };
  struct Want {
    int call;
    size_t i;
    int tree;
    string tn;
    int row;
    u64 r0, Rp, c0, Cp;
  };
  vector<Want> W;
  /* The pieces, written out here and not taken from piecesF or t2BL: one per call and block of
   * rows. Blocks (tau0, tau1 rows): docs/format/11-1-machinery.md, "The lookup on the digits". */
  const u64 rows1 = 1ull << (63 - __builtin_clzll((u64)T2TAU)), rows2 = (u64)T2TAU - rows1;
  const int nbW = rows2 ? 2 : 1;
  if (rows2 & (rows2 - 1)) bad();
  for (int q = 0; q < 2; q++) {
    W.push_back({q, 0, ix.ZZ, "ZZ", q * nbW, 0, rows1, q * mz, mz});
    if (rows2) W.push_back({q, 1, ix.ZZ, "ZZ", q * nbW + 1, rows1, rows2, q * mz, mz});
  }
  // Also written out here: the calls' names cn and digit widths ck. jo: the join's claim number.
  const HCallZ *cs[2] = {&z[1], &z[2]};
  const char *cn[2] = {"R10z", "R8d"};
  const int ck[2] = {10, 8};
  const u32 jo = joinObj9;
  if (jo == ~0u) bad();
  if (!z[0].pc.empty() || !z[3].pc.empty() || F.trees[ix.ZZ].rows.size() != (size_t)(2 * nbW + 2) ||
      F.trees[ix.BZ].rows.size() != 3)
    bad();
  /* The lookup forest f9's numbers, recomputed without forestShape. Their values:
   * docs/format/11-1-machinery.md, "The lookup on the digits" (n9: n_Z; dW: dT; eW: e_0, e_1). */
  Ex eW[2];
  {
    const int n9 = z[1].c.n, dW = n9 < LIDB ? n9 : LIDB, io = n9 - dW;
    if (z[2].c.n != n9 || f9.dT != dW || f9.kT != 10 || f9.NT != (1 << (io + 1)) || f9.rK != io + 1 ||
        f9.iota != vector<int>{io, io} || f9.first != vector<int>{0, 1 << io} || (int)f9.zy.size() != io + 1 ||
        (int)f9.zx.size() != dW || f9.e.size() != 2)
      bad();
    eW[0] = eO() - f9.zy[0];
    eW[1] = f9.zy[0];
    if (!(f9.e[0] == eW[0]) || !(f9.e[1] == eW[1])) bad();
    for (int q = 1; q <= 2; q++) {
      if ((int)z[q].c.zk.size() != n9 || !(z[q].c.alpha == f9.alpha)) bad();
      for (int i = 0; i < n9; i++)
        if (!(z[q].c.zk[i] == (i < io ? f9.zy[1 + i] : f9.zx[i - io]))) bad();
    }
  }
  // Each call's constants, and its pieces' key ranges: aligned, within the call's keys, disjoint.
  for (int q = 0; q < 2; q++) {
    const HCallZ &c = *cs[q];
    if (c.c.name != cn[q] || c.c.fn != 0 || c.c.k != ck[q] || c.c.kappa != 5 || c.c.t0 != 0 || c.pc.size() != (size_t)nbW ||
        c.c.n < 10 || c.c.n > 44 || (int)c.c.zk.size() != c.c.n)
      bad();
    u64 tot = 0;
    for (auto &p : c.pc) {
      if (!p.Rp || !p.Cp || p.a < 0 || p.a > 40 || (1ull << p.a) != p.Rp || (p.r0 & (p.Rp - 1)) || p.CP < p.Cp ||
          (p.CP & (p.CP - 1)) || p.b != p.a + clog2(p.CP) || p.b > c.c.n)
        bad();
      if ((p.off & ((1ull << p.b) - 1)) || p.off + (1ull << p.b) > (1ull << c.c.n)) bad();
      tot += 1ull << p.b;
    }
    if (nbW == 2) {
      const HPiece &a = c.pc[0], &b = c.pc[1];
      if (a.off < b.off + (1ull << b.b) && b.off < a.off + (1ull << a.b)) bad();
    }
    if (c.c.n != (clog2(tot) < 10 ? 10 : clog2(tot)) || c.c.iota != (c.c.n > LIDB ? c.c.n - LIDB : 0)) bad();
  }
  u128 area = 0;
  for (size_t x = 0; x < W.size(); x++) {
    const Want &w = W[x];
    const HPiece &p = cs[w.call]->pc[w.i];
    if (w.tree < 0 || (size_t)w.tree >= F.trees.size() || F.trees[w.tree].name != w.tn) bad();
    if (p.tree != w.tree || p.row != w.row || p.r0 != w.r0 || p.Rp != w.Rp || p.c0 != w.c0 || p.Cp != w.Cp) bad();
    if (w.r0 + w.Rp > F.trees[w.tree].R || w.c0 + w.Cp > F.trees[w.tree].K) bad();
    area += (u128)w.Rp * w.Cp;
    for (size_t y2 = x + 1; y2 < W.size(); y2++)
      if (w.r0 < W[y2].r0 + W[y2].Rp && W[y2].r0 < w.r0 + w.Rp && w.c0 < W[y2].c0 + W[y2].Cp && W[y2].c0 < w.c0 + w.Cp)
        bad();
  }
  if (area != (u128)F.trees[ix.ZZ].R * F.trees[ix.ZZ].K || F.trees[ix.ZZ].R != (u64)T2TAU || F.trees[ix.ZZ].K != 2 * mz)
    bad();
  /* The join's weights: on a piece, coef[r] L[cc] must be -theta^jo eW[call] eq(zk, key (r, cc)).
   * Each side is a factor of r times a factor of cc, and a 0 stops the run, so the piece's first
   * row and first column decide all of it; its last corner is compared too. */
  for (auto &w : W) {
    const HCallZ &c = *cs[w.call];
    const HPiece &p = c.pc[w.i];
    const HTree &T = F.trees[w.tree];
    const vector<Ex> &cv = T.rows[w.row].coef, &L = T.rows[w.row].L;
    if (cv.size() != T.R || L.size() != T.K) bad();
    const Ex ka = t2fKappaByText(F.theta, jo) * eW[w.call], mk = eZ() - ka;
    const Ex *zk = c.c.zk.data();
    const int n = c.c.n;
    if (isZ(ka) || isZ(cv[w.r0]) || isZ(L[w.c0])) bad();
    parFor((w.Cp + 4095) / 4096, [&](u64 blk) {
      EqRun eq(zk, n, p.off + blk * 4096);
      for (u64 cc = blk * 4096; cc < w.Cp && cc < (blk + 1) * 4096; cc++, eq.next()) {
        const Ex v = mk * eq.at();
        if (isZ(v) || !(cv[w.r0] * L[w.c0 + cc] == v)) bad();
      }
    });
    for (u64 r = 0; r < w.Rp; r++) {
      const Ex v = mk * eqAtKey(zk, n, p.off + r * p.CP);
      if (isZ(v) || !(cv[w.r0 + r] * L[w.c0] == v)) bad();
    }
    if (!(cv[w.r0 + w.Rp - 1] * L[w.c0 + w.Cp - 1] == mk * eqAtKey(zk, n, p.off + (w.Rp - 1) * p.CP + (w.Cp - 1))))
      bad();
    for (u64 r = 0; r < T.R; r++)
      if ((r < w.r0 || r >= w.r0 + w.Rp) && !isZ(cv[r])) bad();
    for (u64 cc = 0; cc < T.K; cc++)
      if ((cc < w.c0 || cc >= w.c0 + w.Cp) && !isZ(L[cc])) bad();
  }
  // The ties' weights. The expected values are recomputed from mats, cfF and the batching element
  // F.theta; the values in use (coef, L, rowX, loX, gamX, gamV) are only compared with them.
  auto badT = [&]() {
    refuse(F.G_ + "internal-f-tie-weights");
  };
  {
    const vector<Ex> &c4 = F.trees[ix.ZZ].rows[2 * nbW].coef, &c5 = F.trees[ix.ZZ].rows[2 * nbW + 1].coef;
    if (c4.size() != (size_t)T2TAU || c5.size() != (size_t)T2TAU) badT();
    for (u32 u = 0; u < T2TAU; u++)
      if (!(c4[u] == ePow(F.theta, u)) || !(c5[u] == c4[u])) badT();
  }
  const vector<Ex> &L5 = F.trees[ix.ZZ].rows[2 * nbW + 1].L;
  vector<Ex> xi2(T2TAU), vt2(T2TAU); // the self-test's own derivation of xi and vt, done once
  for (u32 u = 0; u < T2TAU; u++) {
    xi2[u] = vt2[u] = ePow(F.theta, u);
  }
  if (gamX.size() != NM || gamV.size() != NM) badT();
  u64 colsSeen = 0;
  for (size_t mu = 0; mu < NM; mu++) {
    const ShP &s = shp[mu];
    const HTree &X = F.trees[gtTree[mu]];
    if (X.kind != 'P' || X.rows.size() != 1 || X.R != s.Rr || X.K != s.Kc || tieObj[mu] == ~0u) badT();
    /* The equations' numbering (see stepF), recomputed by loops: np2 = Np, first2 = eqFirst[mu].
       ka: docs/format/11-1-machinery.md, "The claims, in order" (chi: F.theta). */
    u64 np2 = 1, first2 = 0;
    while (np2 < (u64)T2TAU) np2 *= 2;
    for (size_t nu = 0; nu < mu; nu++) {
      u64 span = 1;
      while (span < (mats[nu].m + 2) / 3) span *= 2;
      first2 += np2 * span;
    }
    const Ex ka = ePow(F.theta, (u64)F.planClaimsF + first2);
    /* t2AddP (verifier/src/vf_subproof_segment.h) gives a symbol tree one symbol row per tree row, no dummy
       row and no unused entry (J = 1, R = Rr = m', Rd = 0, K1 = Kc = k): checked here and assumed
       below. verifier/src/vf_smallness_packing.h's placement and tie functions allow any J; a test that is not
       part of this repository runs them at five values of J, from 2 to 768. */
    const u64 mS2 = (mats[mu].m + 2) / 3;
    if (mS2 != s.mS || s.J != 1 || s.R != mS2 || s.Rr != mS2 || s.Rd != 0 || s.k != mats[mu].k || s.K1 != s.k || s.Kc != s.k || rowX[mu].size() != mS2 || loX[mu].size() != 1 ||
        !(loX[mu][0] == eO()) || tieObj[mu] >= F.planClaimsF)
      refuse(F.G_ + "internal-sym-rowvec");
    vector<Ex> hi(mS2);
    for (u64 R = 0; R < mS2; R++) hi[R] = ePow(F.theta, np2 * R);
    // Right side (symbols): ce[R] must be hi[R]. Two equal entries would permit a forgery. Whether
    // entries are equal or zero depends only on theta; a committed false matrix meets one fixed
    // polynomial in theta, which the batching round's error already counts. At a zero or a repeat
    // the tests below end the run without a verdict.
    auto badR = [&]() {
      refuse(F.G_ + "internal-sym-rowvec");
    };
    const vector<Ex> &ce = X.rows[0].coef;
    if (ce.size() != s.Rr) badR();
    for (u64 R = 0; R < s.R; R++)
      if (!(ce[R] == hi[R]) || isZ(ce[R])) badR();
    {
      vector<std::array<u64, CE>> v(s.R);
      for (u64 R = 0; R < s.R; R++)
        for (int i = 0; i < CE; i++) v[R][i] = eC(ce[R], i);
      std::sort(v.begin(), v.end());
      for (u64 R = 1; R < s.R; R++)
        if (v[R] == v[R - 1]) badR();
    }
    // Left side (digits; L5): docs/format/11-1-machinery.md, "The weights that the test puts on the
    // arrays", TIE_mu (chi: F.theta).
    const u64 c0 = off[mu], c1 = mz + off[mu];
    parFor((s.mS + 4095) / 4096, [&](u64 blk) {
      for (u64 iS = blk * 4096; iS < s.mS && iS < (blk + 1) * 4096; iS++) {
        Ex v = ka * hi[iS];
        for (u64 l = 0; l < 3 && 3 * iS + l < s.m; l++) {
          if (isZ(v) || !(L5[c0 + 3 * iS + l] == v) || !(L5[c1 + 3 * iS + l] == eS(v, 1ull << T2D0))) badT();
          v = eS(v, RADB);
        }
      }
    });
    colsSeen += s.m;
    // The symbol tree's column weights: LX[c] = -ka g, g recomputed term by term.
    auto badW = [&]() {
      refuse(F.G_ + "internal-sym-weight");
    };
    const vector<Ex> &LX = X.rows[0].L;
    if (LX.size() != s.k) badW();
    const Ex mka = eZ() - ka;
    parFor(s.k, [&](u64 c) {
      u64 before = 0; // cfFirst[mu], recomputed: the number of the earlier matrices' columns
      for (size_t nu = 0; nu < mu; nu++) before += mats[nu].k;
      Ex g = eZ(), gv = eZ();
      for (u32 u = 0; u < T2TAU; u++) {
        const signed char x = cfF[(size_t)((before + c) * T2TAU + u)];
        if (x) {
          g = x < 0 ? g - xi2[u] : g + xi2[u];
          gv = x < 0 ? gv - vt2[u] : gv + vt2[u];
        }
      }
      if (gamX[mu].size() != s.k || gamV[mu].size() != s.k || !(gamX[mu][c] == g) || !(gamV[mu][c] == gv))
        refuse(F.G_ + "internal-f-fold-array"); // foldSums' two sums must equal the recomputed ones
      if (!(LX[c] == mka * g)) badW();
    });
  }
  if (colsSeen != mz || L5.size() != 2 * mz) badT();
  /* The reads' weights on ZZ. Read q, in objectsF's order (matrices ascending; functionals on U1,
   * then those on U3; row segments ascending), stands for the equations number firstR + q np2 + u,
   * u < tau: its whole factor kr is theta^(planClaimsF + firstR + q np2), whatever its claim's
   * number. firstR is eqFirst[NM], recomputed by loops: the first number behind the ties'. So no
   * two reads, and no read and tie, share a number: np2 is at least tau. Were two reads under one
   * power, errors in two tables could cancel. The column weights: docs/format/11-1-machinery.md,
   * "The weights that the test puts on the arrays", READ_q (chi: F.theta). No value of theta makes
   * this test fail: it compares two computations of one polynomial in theta. */
  auto badQ = [&]() {
    refuse(F.G_ + "internal-f-read-weights");
  };
  {
    const vector<Ex> &L4 = F.trees[ix.ZZ].rows[2 * nbW].L;
    if (!funLaidF || L4.size() != 2 * mz) badQ();
    u64 np2 = 1, firstR = 0;
    while (np2 < (u64)T2TAU) np2 *= 2;
    for (size_t nu = 0; nu < NM; nu++) {
      u64 span = 1;
      while (span < (mats[nu].m + 2) / 3) span *= 2;
      firstR += np2 * span;
    }
    const Ex byStep = ePow(F.theta, np2);
    vector<Ex> want(2 * mz, eZ());
    u64 q = 0, before = 0; // before: off[mu], recomputed: the number of the earlier matrices' cell rows
    for (size_t mu = 0; mu < NM; before += mats[mu].m, mu++)
      for (u32 un : {0u, 2u})
        for (const HFun &Fn : *funLaidF) {
          if (Fn.M < 0 || (size_t)Fn.M != mu || Fn.tree != un) continue;
          const u64 m = mats[mu].m, at = before;
          if (!m || !Fn.L || m % Fn.L || Fn.phi.size() != Fn.L) badQ();
          const u64 nsg = m / Fn.L; // at least 1
          vector<Ex> kr(nsg); // the whole factors of this functional's reads, q .. q + nsg - 1
          kr[0] = ePow(F.theta, (u64)F.planClaimsF + firstR + q * np2);
          for (u64 sg = 1; sg < nsg; sg++) kr[sg] = kr[sg - 1] * byStep;
          parFor((m + 4095) / 4096, [&](u64 blk) { // one functional's reads share no column
            for (u64 x = blk * 4096; x < m && x < (blk + 1) * 4096; x++) {
              const Ex v = kr[x / Fn.L] * Fn.phi[x % Fn.L];
              Ex &lo = want[at + x], &hi = want[mz + at + x];
              lo = lo + v, hi = hi + eS(v, 1ull << T2D0);
            }
          });
          q += nsg;
        }
    if (q != nX4Laid || before != mz) badQ();
    for (u64 x = 0; x < 2 * mz; x++)
      if (!(L4[x] == want[x])) badQ();
  }
  // BZ's weights: docs/format/11-1-machinery.md, "The weights that the test puts on the arrays"
  // (chi: F.theta).
  auto badB = [&]() {
    refuse(F.G_ + "internal-f-bz-weights");
  };
  const HTree &B = F.trees[ix.BZ];
  if (B.R != 32 || B.K != bzColumns()) badB();
  for (u64 r = 0; r < 32; r++)
    if (!(B.rows[2].coef[r] == eO())) badB();
  {
    const int lKB = f9.kT + f9.cK;
    if (f9.NC != 2 || f9.cK != 1 || B.K != (1ull << lKB)) badB();
    for (int e = 0; e < 2; e++) {
      const vector<Ex> &cv = B.rows[e].coef, &L = B.rows[e].L;
      if (cv.size() != 32 || L.size() != B.K || (int)f9.zb[e].size() != 5 + lKB) badB();
      for (u64 r = 0; r < 32; r++)
        if (!(cv[r] == eqPlainF(f9.zb[e].data(), 5, r))) badB();
      const Ex mk = t2fKappaByText(F.theta, jo + 1 + (u32)e) * (e ? eZ() - eO() : bitFactor9);
      UnscaledEqIterator st(f9.zb[e].data() + 5, lKB, 0);
      for (u64 x = 0; x < B.K; x++, st.next())
        if (!(L[x] == mk * st.at())) badB();
    }
    const Ex kp = t2fKappaByText(F.theta, jo + 3);
    if (B.rows[2].L.size() != B.K) badB();
    for (u64 x = 0; x < B.K; x++)
      if (!(B.rows[2].L[x] == eS(kp, 1ull << (x & 31)))) badB();
  }
  if (!runT) refuse(F.G_ + "internal-sym-shape");
  symShapesHeldF(F, *runT); // after the last claim
}
