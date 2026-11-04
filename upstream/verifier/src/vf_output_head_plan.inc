// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The plan PH2 of sub-proof H (`fork` in names; the output head): array shapes, numbers of message
 * slots and claims, rounds. No proof is read here. Included in `runStage` (cp_verify.cpp) after
 * vf_layers_plan.inc; uses hidPlans2, borderCmpF, statement h, layouts y, HLy, sizes; T = 2^lt tokens. */
struct HPlanH2 {
  HTailF2 tl;
  HFork Fp;
  Tail2 Tp;
  u64 nU[3] = {0, 0, 0};
  u32 ns = 0;
  size_t nMsg = 0, nLin = 0;
  // ns = nMsg message slots, nLin linear claims, NCL weight claims, lamS rounds over the features.
  size_t NCL = 1;
  /* `rounds` adds to p, a hash chain's plan, the rounds after H's commit point `moment`: 0, the
   * matrices; 1, U3; 2, the digits of the approximate range check ("smallness test" in strings).
   * It points to PH2 and HLy.fs. Order: as in `runForkHidH2` (vf_output_head_check.inc). `own`: H's sizes. */
  blk::HeadRounds own;
  std::function<void(blk::Plan &, unsigned moment)> rounds;
  int lamS = 0;
};
HPlanH2 PH2;
{
  HPlanH2 &P = PH2;
  const u32 hb = statedTarget.hb - (u32)POW; // line `sec` less a request's proof of work on positions
  const HL &L = HLy;
  const HForest &fs =
      L.fs; // H's lookup calls, L.calls, run as one lookup argument: a `forest` (vf_lookup_argument.h)
  P.Fp.G_ = "H:";
  // A token's clog2(Vv)-bit index must fit in an lkW-bit column index of the model's weight table.
  if ((u32)clog2(Vv) > lkW) refuse("hid-vocab-cols");
  // Self-check of sizes set in runStage(): fs's calls, fs.pub (one per tree), L.kb (bH's columns).
  if (fs.cl.size() != L.calls.size() || fs.pub.size() != (size_t)fs.NT || L.kb != forestPlaneCols(fs, (int)h.lmbH))
    refuse("internal-h-forest");
  /* The range check's matrices (tl). H commits to hf (cells of the final norm, logit margins and
   * exponentials; NHf parts), hd (normalized activations), hr (a digit row per token, vf_norm_digit_rows.h)
   * and bH (fs's multiplicity bits); only H reads them. gidH: H's number in labels (TMat::aId). */
  vector<TMat> mats;
  const u64 gidH = 1 + NGr;
  /* `shape` rejects 256 or more matrices. wp is 0 (vf_smallness_matrices.h). xbL, the residual stream after the
   * last layer group, is foreign (kind 1), with its owner's label: that group's sub-proof G_(NGr-1)
   * commits to it in its last array that others read (`TX`). fx describes it; a = 2: owner, H. */
  auto req = [&](const string &n, u64 m, u64 k, u32 wp) {
    TMat M;
    M.name = n;
    M.m = m;
    M.k = k;
    M.wp = wp;
    M.aId = gidH * 256 + mats.size();
    mats.push_back(M);
  };
  for (u64 p = 0; p < h.NHf; p++) req("hf" + std::to_string(p), L.m, L.k, 0);
  req("hd", L.mD, L.kD, 0);
  req("hr", L.mR, L.kR, 0);
  req("bH", 1ull << h.lmbH, L.kb, 0);
  req("xbL", y.mxb, y.kxb, 0);
  {
    TMat &M = mats.back();
    M.kind = 1;
    M.foreign = 0;
    M.ttOff = 0;
    M.aId = hidPlans2[NGr - 1].tl.mats[iXO].aId;
  }
  T2Foreign fx;
  fx.name = "PXf0";
  fx.K = y.kxb;
  fx.a = 2;
  fx.m = y.mxb;
  fx.wp = 0;

  const bool hasU[4] = {false, false, true, false}; // of the vectors U1 to U4, H has only U3
  P.tl.shape(mats, vector<int>(mats.size(), -1), {fx}, hasU, h.tau);
  // msg counts H's message slots: each numbers one scalar sent and one mask entry. First, fs's.
  size_t msg = forestElems(fs);
  P.lamS = lPd;
  const int muVh = lPd + lt;
  if (muVh < 9) refuse("shape-product-layer-H");
  P.NCL = 1;
  // Self-check: runForkHidH2() must use exactly msg slots, else NOT CHECKED: H:internal-slot-count.
  msg += P.tl.smallElems() + (size_t)ledNS() + (size_t)layerSlots(muVh) + (size_t)ledNP() +
         (size_t)tscSlots(lt, (u64)ledNO(), 2) + (size_t)tscSlots(lt, 2, 2) +
         (size_t)tscSlots(lt, 2, 2, P.lamS) + wsForkSlots((int)lkW);

  // An evaluation table sums runs of a matrix's rows under public weights. GSx: the size of xbL's.
  const u64 GSx = y.kxb * (y.mxb >> la);
  // U3 holds the tables. Self-check in runForkHidH2() ("internal-W-length-H"): their lengths sum to
  // nU[2]. hf has two: one at the point Z that all lookup calls share, one under the weights phiZ.
  const u64 nhfZ = 2;
  P.nU[2] = nhfZ * (L.filled * T >> la) + 2 * GSx + 2 * L.fillD + 6 * L.fillR + 3 * L.kb;
  // readsH: one read per evaluation table and run. nLin's summands: fs's claims except its join and
  // those on bH; 3 on the norm's sum of squares; on its digit row, 2 on the row products' sumcheck
  // and 1 per end value, carry and further equation; 3 on the stream times the reciprocal; fs's
  // join; 4 on the acceptance policy; 3 on bH; 2 on the logits' sumcheck; 1 on the merge.
  const size_t readsH = nhfZ * h.NHf * (L.m >> la) + 2 * (L.mD >> lt) + 6 * (L.mR >> lt) + 3 + 2 * (y.mxb >> la);
  P.nLin = forestLins(fs) + 3 + (size_t)(1 + 1 + ledNO() + ledNS() + ledLines()) + 3 + 1 + 4 + 3 + 2 + 1 +
           P.tl.lins(readsH); // the range check's claims

  P.tl.addTrees(P.Fp, P.Tp, P.nU, {}, hb);
  // Self-check. The arrays: U3; a PK_i per matrix that only H reads; ZZ, BZ (range check); xbL's.
  if (P.Tp.nt.size() != (1 + P.tl.pkMat.size() + 2 + 1)) refuse("internal-h-tree-list");
  t2Plan(P.Fp, P.Tp);

  /* The shape that this plan gives xbL's array (`tree` in names) must equal the one in its owner's
   * plan. A difference is a bug in this program, never a fault of a proof. */
  {
    const HPlan2 &O = hidPlans2[NGr - 1];
    if (O.tl.ix.TX.empty()) refuse("internal-h-owner-TX");
    /* Compares the owner's last `TX` array with H's one foreign array (`FT`), then the two records
     * of where xbL is (PlaceF, vf_smallness_packing.h). `runStage` checks the count borderCmpF. */
    if (!t2SameShape(O.Tp.nt[O.tl.ix.TX.back()], O.Fp.trees[O.tl.ix.TX.back()], P.Tp.nt[P.tl.ix.FT[0]],
                     P.Fp.trees[P.tl.ix.FT[0]]))
      refuse("internal-h-owner-shape");
    {
      P.tl.borderMeans(0, (int)NGr - 1,
                       (int)(O.tl.pkMat.size() + O.tl.ix.TX.size()) - 1); // H reads "stream" of G_(NGr-1).
      const char *df = samePlaceF(hidPlans2[NGr - 1].tl.ownEntry("stream"), P.tl.plFor[0]);
      if (*df) refuse(string("internal-h-owner-place-") + df);
      borderCmpF++;
    }
  }
  if (msg >= (1ull << 30)) refuse("hid-h-slots");
  P.nMsg = msg;
  P.own.lt = (unsigned)lt, P.own.lPd = (unsigned)lPd, P.own.lkW = (unsigned)lkW;
  P.own.sums = (unsigned)ledNS(), P.own.products = (unsigned)ledNP(), P.own.row_values = (unsigned)ledNO();
  // The range check's equations have no separate challenge: they take further powers of the element
  // that batches the claims, so that element's round counts them too.
  P.own.rows = P.nLin + P.tl.equationsF(readsH);
  {
    HPlanH2 *plan = &PH2;
    const HForest *lookup = &HLy.fs;
    P.rounds = [plan, lookup](blk::Plan &p, unsigned moment) {
      const string where = "head";
      if (moment == 0) {
        forestPlan(p, where + ", look-up", *lookup);
        blk::plan_head_own(p, where, plan->own);
      }
      if (moment >= 1) plan->tl.planRounds(p, where + ", smallness test", moment - 1); // its matrix; then the rest
      if (moment == 2) blk::plan_head_end(p, where, plan->own);
    };
  }
  P.ns = (u32)msg;
}
