// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Plans the layer sub-proof G_g (`fork` in identifiers) of each layer group g < NGr, and compares
 * what one sub-proof reads from another with the owner's plan. It reads no word of a proof.
 * cp_verify.cpp includes it inside `runStage`, whose locals it uses (p3: routed experts).
 * vf_layers_check.inc checks G_g with the plan; if its own counts differ, the call ends NOT CHECKED. */
/* tl: G_g's approximate range check (printed "smallness test"; `tail` in identifiers). Fp, Tp: the
   committed arrays (`trees`) and their shape records. nU: lengths of U1 to U3, vectors over E.
   `rounds`: adds to a hash chain's plan G_g's own rounds after one commit point (`moment`). */
struct HPlan2 {
  HTailF2 tl;
  HFork Fp;
  Tail2 Tp;
  u64 nU[3] = {0, 0, 0};
  u32 ns = 0, used = 0; // ns: message slots, one per scalar sent; used: those taken (equal)
  std::function<void(blk::Plan &, unsigned)> rounds;
  size_t NCL = 0, nLin = 0; // NCL: claims on the weights; nLin: the sub-proof's claims, all linear
  u64 nBatched = 0; // nLin plus the range check's equations; one element's powers batch them all
  int lamS = 0;
};
vector<HPlan2> hidPlans2;
size_t borderCmpF = 0; // Comparisons of placement records made; `runStage` checks the total.
{
  const u32 hb = statedTarget.hb - (u32)POW; // line `sec` minus `work positions`[0]; for PlaceF::hb
  hidPlans2.resize(NGr);
  for (u64 g = 0; g < NGr; g++) {
    HPlan2 &P = hidPlans2[g];
    P.Fp.G_ = "G" + std::to_string(g) + ":";
    vector<TMat> mats = tails[g].mats;
    vector<int> txOf(mats.size(), -1);
    vector<T2Foreign> ft;
    vector<u32> aTX;
    /* A border matrix (read by another sub-proof) is committed as array PX_i (`TX` in identifiers).
     * A foreign one (kind 1) is read from its owner's PX. a = 1 + readers; `runStage` tests it. */
    for (size_t mu = 0; mu < mats.size(); mu++) {
      TMat &M = mats[mu];
      // Border matrices: the output xbO, and qa for its sub-proof A_g.
      if (M.kind == 0 && (mu == iXO || mu == iQA)) {
        txOf[mu] = (int)aTX.size();
        aTX.push_back(2);
      } else if (M.kind == 1) {
        T2Foreign f;
        f.name = "PXf" + std::to_string(ft.size());
        f.K = M.k;
        f.a = 2;
        f.m = M.m;
        f.wp = M.wp;
        M.foreign = (int)ft.size();
        M.ttOff = 0;
        ft.push_back(f);
      } else if (M.kind == 2)
        refuse("internal-weight-matrix"); // No sub-proof has a matrix of the weight class.
    }
    const bool hasU[4] = {false, false, true, false}; // of U1 to U4, G_g has only U3, the reads
    P.tl.shape(mats, txOf, ft, hasU, h.tau);
    // msg counts the message slots. First: those of y.fo, the one lookup for all of y.calls.
    const HForest &fo = y.fo;
    size_t msg = forestElems(fo);
    const size_t nT = tscCalls.empty() ? 0 : 1; // one token sumcheck, TSC(zt), for all of tscCalls
    // lamS, the batched sumcheck's rounds: docs/format/11-2-feed-forward-segment.md, "Two derived
    // numbers".
    {
      const int wi = clog2(Lam);
      u64 tot =
          4 * (1ull << (wi + clog2(d))) + (1ull << (wi + clog2(dq))) + (1ull << (wi + clog2(f))) + 2 * (1ull << wi);
      if (p3) {
        const int wm = clog2(LamM), le = clog2(NE);
        tot += (1ull << (wm + clog2(d))) + (1ull << wm) + 2 * (1ull << (wm + le)) +
               LamM * (2 * (1ull << (le + clog2(d))) + (1ull << (le + clog2(fe))));
      }
      P.lamS = clog2(tot);
    }
    P.NCL = 6 * Lam + (p3 ? 4 * LamM : 0) + (g == 0 ? 1 : 0); // last in G_0: the embedding's claim

    {
      /* The claims, in vf_layers_check.inc's order; that file compares its count with nLin.
       * The terms follow the rows of docs/format/11-2-feed-forward-segment.md, "The claims, in
       * order"; row 13 is added below. */
      const size_t lin = forestLins(fo) + 3 + (2 + ((u64)ledNO() + (u64)ledNS() + (u64)ledLines()) * NI) +
                         (nT ? NN * Lam + (p3 ? LamM * (1 + NE) : 0) : 0) + 3 + (p3 ? 3 : 0) + (p3 ? LamM : 0) + 3 +
                         (g == 0 ? 1 : 0) + 3 + 1;
      // nX4: the range check's reads: NX in docs/format/11-2-feed-forward-segment.md, "The claims,
      // in order".
      const size_t nX4 =
          rdAF.size() * (size_t)y.parts[0] * (size_t)(y.mrow[0] >> la) + rdQA.size() * (size_t)(y.mrow[3] >> la) +
          (rdXI.size() + (g == 0 ? 1 : 0) + rdXO.size()) * (size_t)(y.mxb >> la) +
          rdAD.size() * (size_t)y.parts[1] * (size_t)(y.mrow[1] >> lt) + rdAR.size() * (size_t)(y.mrow[2] >> lt) +
          rdQS.size() * (size_t)(y.mrow[3] >> lt) + 3;
      P.nLin = lin + P.tl.lins(nX4);
      P.nBatched = P.nLin + P.tl.equationsF(nX4);
    }
    {
      // The rounds after each commit point: docs/format/11-2-feed-forward-segment.md, "The rounds,
      // in order". blk::plan_layer_own() (prover/prover/plan_rounds.h) adds steps 2 to 16.
      blk::LayerRounds x;
      x.lt = (unsigned)lt, x.ls = (unsigned)ls, x.lNI = (unsigned)lNI, x.lPd = (unsigned)lPd, x.ll = (unsigned)ll;
      x.lPf = (unsigned)lPf, x.lfe = (unsigned)lfe, x.lamS = (unsigned)P.lamS, x.lkW = (unsigned)lkW;
      x.NI = NI, x.NS = (u64)ledNS(), x.NP = (u64)ledNP(), x.NO = (u64)ledNO();
      x.with_terms = nT != 0;
      x.shown_by_the_terms = nT ? NN * Lam + (p3 ? LamM * (1 + NE) : 0) : 0;
      x.never_sent = 1 + (p3 ? LamM : 0);
      x.weight_claims = P.NCL - (g == 0 ? 1 : 0);
      x.first_group = g == 0, x.routed = p3;
      const string where = "G" + std::to_string(g);
      const HTailF2 *small = &P.tl; // stays valid: hidPlans2 is not resized after this
      const u64 claims = P.nBatched;
      const HForest *look = &y.fo;
      P.rounds = [look, x, where, small, claims](blk::Plan &p, unsigned moment) {
        if (moment == 0) {
          forestPlan(p, where + ", look-up", *look);
          blk::plan_layer_own(p, where, x);
          return;
        }
        small->planRounds(p, where + ", smallness test", moment - 1);
        if (moment == 2) blk::plan_bat_sigma(p, where + ", claims", claims);
      };
    }

    /* The other message slots, in the messages' order (vf_layers_check.inc):
     * docs/format/11-2-feed-forward-segment.md, "The rounds, in order", column "scalars sent",
     * steps 4 to 18. */
    msg += (size_t)ledNS() * NI + (size_t)layerSlots(muV) + (size_t)ledNP() * NI;
    msg += (size_t)tscSlots(lt, (u64)ledNO() * NI, 2);
    if (nT) msg += (size_t)tscSlots(lt, NN * Lam + (p3 ? LamM * (1 + NE) : 0), 1);
    msg += (size_t)sumcheckSlots(RoundForm::SUM2, P.lamS, 2 + 1 + (p3 ? LamM : 0) + P.NCL - (g == 0 ? 1 : 0), 2);
    msg += (size_t)layerSlots(muH) + (p3 ? (size_t)layerSlots(muH2) : 0) + (g == 0 ? 1 : 0) +
           wsForkSlots((int)lkW) + P.tl.smallElems();

    P.nU[2] = wW / WPE + (g == 0 ? mats[iXI].k * (mats[iXI].m >> la) : 0); // xb_0 has one more long table, at Z_E.
    P.tl.addTrees(P.Fp, P.Tp, P.nU, aTX, hb);
    t2Plan(P.Fp, P.Tp);
    /* A layer sub-proof reads at most one foreign array: G_0 none, G_g (g > 0) the last border
     * array of G_(g-1). That array's shape in G_g's plan must equal its shape in the owner's.
     * An `internal-` name is meant for a fault of the program: a checking call ends NOT CHECKED. */
    if (P.tl.ix.FT.size() != (g ? 1u : 0u)) refuse("internal-g-owner-shape");
    if (g) {
      const HPlan2 &O = hidPlans2[g - 1];
      if (O.tl.ix.TX.empty() || P.tl.ix.FT.empty() ||
          !t2SameShape(O.Tp.nt[O.tl.ix.TX.back()], O.Fp.trees[O.tl.ix.TX.back()], P.Tp.nt[P.tl.ix.FT[0]],
                       P.Fp.trees[P.tl.ix.FT[0]]))
        refuse("internal-g-owner-shape");
    }
    /* The reader's placement record for that array (PlaceF, vf_smallness_packing.h) must equal the
     * owner's. "stream" means xbO. */
    P.tl.ownerIs((int)g);
    if (g) {
      const HPlan2 &O = hidPlans2[g - 1];
      P.tl.borderMeans(0, (int)g - 1, (int)(O.tl.pkMat.size() + O.tl.ix.TX.size()) - 1);
      const char *df = samePlaceF(hidPlans2[g - 1].tl.ownEntry("stream"), P.tl.plFor[0]);
      if (*df) refuse(string("internal-g-owner-place-") + df);
      borderCmpF++;
    }

    // A message slot's index is a u32. The head's and attention's plans have the same limit, 2^30.
    if (msg >= (1ull << 30)) refuse("hid-g-slots");
    P.used = (u32)msg;
    P.ns = (u32)msg;
    trimL(P.Fp);
  }

  /* A_g reads qa from G_g's first border array, PX_0: its shape in A_g's plan must equal the
   * owner's. `runStage` has built the attention plans (AH2) before this file. */
  for (u64 g = 0; g < NGr; g++) {
    const HPlan2 &O = hidPlans2[g];
    const AttentionSubProof &X = AH2[g];
    if (O.tl.ix.TX.empty() || X.seam.tl.ix.FT.empty() ||
        !t2SameShape(O.Tp.nt[O.tl.ix.TX[0]], O.Fp.trees[O.tl.ix.TX[0]], X.seam.Tp.nt[X.seam.tl.ix.FT[0]],
                     X.seam.Fp.trees[X.seam.tl.ix.FT[0]]))
      refuse("internal-a-owner-shape");
  }
  for (u64 g = 0; g < NGr; g++) {
    AttentionSubProof &X = AH2[g];
    // The placement records, as for G_g above.
    X.seam.tl.borderMeans(0, (int)g, (int)hidPlans2[g].tl.pkMat.size());
    const char *df = samePlaceF(hidPlans2[g].tl.ownEntry("qa"), X.seam.tl.plFor[0]);
    if (*df) refuse(string("internal-a-owner-place-") + df);
    borderCmpF++;
  }
}
