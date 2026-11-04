// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md. "Slot" and "tail" mean other things here: see the comment below.

/* Checks one layer sub-proof (`fork` in names), G_g: layer group g, without the attention function.
   docs/format/11-2-feed-forward-segment.md specifies it. It is "11-2" below, and "step n" is
   row n of its table of rounds. This file is included in `runStage` (cp_verify.cpp) and uses its
   locals: the layout y, p3 (the odd layers have routed experts) and others. `refuse` (vf_common.h)
   prints `NO: <name>` and exits; a name with "internal-", meant for
   a fault of this program, becomes NOT CHECKED (status 5). Every sumcheck is the masked one (PSC)
   of vf_sumcheck_psc.h. No round rejects by itself: the unsent coefficient is filled in from the
   running claim. A hidden value is "shown": sent under a mask, in a slot (a scalar's number in the
   sub-proof). `F.shown` gives what was sent, `affSlot` the hidden value as an affine form ("form"
   below). Each algebraic check becomes a linear claim on hidden values, tested only at the hash
   chain's end. Of U3, the vector of all evaluation tables (weighted sums of committed cells;
   "reads" in section names), only the root arrives. Never sent: each feature's squares (HV, for the
   norms) and products (Hh1, Hh2, for the gated feed-forward), summed over the tokens. A product
   layer checks each. The weight claims leave by the merge (vf_weights_merge.h): a slot v_a each,
   then the merged value, masked by an entry of the request's index-mask message. In names, "stack"
   is the batched sumcheck, "tail" the approximate range check (printed: "smallness test"). */
// Indexed by sub-proof: G_g at g, the output head at NGr and, in forkRan alone, A_g at NGr + 1 + g.
vector<WsForkOut> forkOut(NGr + 1);
vector<char> forkRan(2 * NGr + 1, 0);
// g: the layer group, 0 <= g < NGr. hp: its plan (shapes, counts; vf_layers_plan.inc). Fills forkOut[g].
auto runForkHidG2 = [&](u64 g) {
  const double tf0 = now();
  const HPlan2 &hp = hidPlans2[g];
  Tail &tw = tails[g];
  // The chain's current segment (`link` in names) must be this sub-proof's. The segments are G_0,
  // A_0, G_1, A_1, .., H (buildLinksH() in vf_segments.inc).
  if (!currentSegment || !currentSegment->V || !currentSegment->check || currentSegment->link != 2 * (size_t)g) refuse("internal-g-link");
  HFork F;
  F.G_ = "G" + std::to_string(g) + ":";
  F.trees = hp.Fp.trees;
  fillL(F);
  Tail2 T2 = hp.Tp;
  HTailF2 tl = hp.tl;
  const u32 iU3 = (u32)tl.ix.U[2]; // U3's index in F.trees; a layer sub-proof has no U1 or U2
  Tr tr{currentSegment->V->ch};
  // tr (a `Tr`: vf_common.h) wraps the request's first hash chain. The sub-proof runs after its
  // segment's first commit point. It has no start state: the chain's parameters list the segments.
  tl.runT = &T2;
  F.slots(hp.ns);
  Lay yy = y;
  setTau(yy, y);
  u32 slot = 0;
  // One lookup (`forest` in names) for all calls. Lookup segment (Seg): named columns of one call.
  vector<Call *> cl1;
  for (auto &c : yy.calls) cl1.push_back(&c.c);
  // From here on, the code is in the order of messages, challenges and claims.
  HForest fo = forestPad(F, tr, cl1, y.fo.pub, slot, LIDB);
  const size_t NC = yy.calls.size();
  // The run's lookup must have the sizes from which b and U3 were sized and the slots counted.
  if (fo.dT != y.fo.dT || fo.kT != y.fo.kT || fo.NT != y.fo.NT || fo.rK != y.fo.rK || fo.first != y.fo.first ||
      fo.iota != y.fo.iota || fo.callOf != y.fo.callOf)
    refuse(F.G_ + "internal-g-forest");
  // Step 2: P_V, the norms' token point.
  const vector<Ex> PV = trPoint(tr, RoundName::ZERO, lt);
  // Step 3, in group 0 only: the embedding's points Z_E and tkh.
  const bool hasE = g == 0;
  vector<Ex> ZE, tkh;
  if (hasE) {
    ZE = trPoint(tr, RoundName::ZERO, ls + lt);
    tkh = trPoint(tr, RoundName::ZERO, lPd - ls);
  }
  auto zl = [&](CallD &c) {
    return c.c.zk.data();
  };
  auto zj = [&](CallD &c) {
    return c.c.zk.data() + ll;
  };
  auto callOf = [&](const string &sn) -> CallD & {
    for (auto &c : yy.calls)
      for (auto &s : c.segs)
        if (s.name == sn) return c;
    refuse("internal-segment-" + sn);
  };
  auto callNo = [&](const string &cn) -> size_t {
    for (size_t q = 0; q < NC; q++)
      if (yy.calls[q].c.name == cn) return q;
    refuse("internal-call-" + cn);
  };
  /* A call's point zk = (zl | zj | token part) has ll + lJ + lt coordinates. As dT >= la, its last
     la are those of zx, the lookup's point in a tree. Z and zt rely on this, so it is tested. */
  if (fo.dT < la) refuse("forest-depth-G");
  for (auto &c : yy.calls)
    for (int i = 0; i < la; i++)
      if (!(c.c.zk[c.c.zk.size() - la + i] == fo.zx[fo.dT - la + i])) refuse(F.G_ + "internal-g-merged-point");
  const vector<Ex> ztF(fo.zx.end() - lt, fo.zx.end());
  // The norms' sums of squares. An instance is one norm of one layer. It has NSl convolution sums
  // (W, from slot sW) and NPl row products: 7 and 3, or 9 and 7 with five limbs (vf_norm_digit_rows.h).
  const u64 NSl = (u64)ledNS(), NPl = (u64)ledNP(), NRl = (u64)ledNR();
  const u32 sW = F.claimsIn(tr, slot, (u32)(NSl * NI));
  const vector<Ex> gp = trBatch(tr, (size_t)NSl); // gam^0 .. gam^(NSl - 1): they batch the limbs' convolution sums
  /* The vector HV[i Pd + k] = sum_t eq(P_V, t) (sum_l gam^l x_l)^2 is never sent; x_l is limb l of
     instance i's input at token t, feature k. The product layer (sumcheck `readProductLevelRounds`) starts from
       sum_i eq(rhoI, i) sum_k HV[i Pd + k] = sum_i eq(rhoI, i) sum_s gam^s W_(i,s).
     The plain sum over k is 2^lPd times the eq-weighted sum at (1/2, .., 1/2), so the layer's point
     has 1/2 there and the start form is divided by 2^lPd. rhoI is sampled after gam and the W. */
  const vector<Ex> rhoI = lNI ? trPoint(tr, RoundName::JOIN, lNI) : vector<Ex>(); // no message if there is one instance
  ProductLevel wpl;
  {
    const vector<Ex> e = eqTab(rhoI.data(), lNI);
    const Ex half = eInv(eI(2));
    Ex scale = eO();
    for (int k = 0; k < lPd; k++) scale = scale * half;
    Aff C;
    for (u64 i = 0; i < NI; i++) {
      const Ex w = scale * e[i];
      for (u64 s = 0; s < NSl; s++) affAdd(C, affSlot(sW + (u32)(NSl * i + s)), w * gp[s]);
    }
    vector<Ex> rho = rhoI;
    rho.insert(rho.end(), (size_t)lPd, half);
    rho.insert(rho.end(), PV.begin(), PV.end());
    wpl = readProductLevelRounds(F, tr, slot, muV, rho, C, orderDefault(muV));
  }
  // Step 7: the row products (from slot sPv) and their coefficients omV.
  const u32 sPv = F.claimsIn(tr, slot, (u32)(NPl * NI));
  const vector<Ex> omV = trBatch(tr, (size_t)(NPl * NI));
  const bool hasT = !tscCalls.empty();
  auto isMoe = [&](u64 lam) {
    return p3 && lam % 2 == 1;
  };
  // The tables' points by name, entered as they become known. Z, zt: the last la, lt of zx.
  std::map<string, vector<Ex>> pt;
  auto lastLa = [&](const vector<Ex> &z) {
    return lastOf(z, la);
  };
  pt["Z"] = lastLa(fo.zx);
  pt["zt"] = ztF;
  pt["Z_V"] = lastLa(wpl.r);
  pt["P_V"] = PV;
  if (hasE) pt["Z_E"] = ZE;
  vector<HFun> fun;
  u64 p = 0;
  std::map<string, u64> tab;
  std::map<u64, u64> tabLen; // offset where one `addFun` call began -> number of entries it added
  // One table (HFun) per matrix, `parts` from tw.mats[M0] on. An entry: <phi, a run of L cells>.
  auto addFun = [&](size_t M0, u64 parts, u64 L, const vector<Ex> &phi, const string &nm) {
    tl.funOpen();
    const u64 first = p;
    for (u64 q = 0; q < parts; q++) {
      const TMat &M = tw.mats[M0 + q];
      HFun fn;
      fn.M = (int)(M0 + q);
      fn.L = L;
      fn.phi = phi;
      fn.tree = 2; // 2 means U3 (0 would be U1); not an index into F.trees
      fn.at = p;
      fn.name = F.G_ + nm;
      fun.push_back(fn);
      p += M.k * (M.m / L);
    }
    if (p == first) refuse("internal-W-length"); // an empty table would share its start with the next
    tab[nm] = first;
    tabLen[first] = p - first;
  };
  // The only function through which this file's claims refer to an entry of U3: entry i of the
  // tables that one addFun() call added from offset t on. It stops at an i at or beyond their
  // length and a t at which no such call began. It does not catch an index that strays to another
  // tensor, layer or limb of the same matrix, or a t that starts other tables than those meant.
  auto U3at = [&](u64 t, u64 i) {
    const auto it = tabLen.find(t);
    if (it == tabLen.end() || i >= it->second) refuse(F.G_ + "internal-g-table-index");
    return uEnt(F, iU3, t + i);
  };
  const u64 La = 1ull << la;
  // The tables get their offsets in U3 here, because the batched sumcheck's forms name their
  // entries. tables() leaves phi empty, to be set before U3's root: Z_a, Z_b are not known yet.
  struct LaterPhi {
    size_t first = 0, count = 0;
    string point;
    bool lng = false;
  };
  vector<LaterPhi> laterPhi;
  auto tables = [&](const char *mn, size_t M0, u64 parts, const vector<string> &pts, bool lng) {
    for (auto &n : pts) {
      laterPhi.push_back({fun.size(), (size_t)parts, n, lng});
      addFun(M0, parts, lng ? La : T, vector<Ex>(), string(mn) + "@" + n);
    }
  };
  tables("af", iAF, y.parts[0], rdAF, true);
  tables("qa", iQA, 1, rdQA, true);
  tables("xbI", iXI, 1, rdXI, true);
  // A long table's run is 2^la cells (T tokens of S2 features); a short one's, one feature's T.
  // In group 0, xbI has one more long table, at Z_E; no other matrix has a table at Z_E.
  if (hasE) tables("xbI", iXI, 1, {"Z_E"}, true);
  tables("xbO", iXO, 1, rdXO, true);
  tables("ad", iAD, y.parts[1], rdAD, false);
  tables("ar", iAR, 1, rdAR, false);
  // qa also has a short table, at zt.
  tables("qa", iQA, 1, rdQS, false);
  const u64 mb = 1ull << h.lmb;
  // Three tables on b, the lookup's multiplicity bits: TB_0, TB_1, TP. A run is a whole column.
  for (int e = 0; e < 2; e++) addFun(iB, 1, mb, forestPhiB(fo, (int)h.lmb, e), e ? "b-bits-1" : "b-bits-0");
  addFun(iB, 1, mb, forestPhiP((int)h.lmb), "b-pin");
  if (p != hp.nU[2]) refuse("internal-W-length");
  auto ewOf = [&](const string &sn) {
    CallD &c = callOf(sn);
    vector<Ex> e = eqTab(zj(c), c.lJ);
    Seg &s = c.seg(sn);
    vector<Ex> w(s.n);
    for (u64 x = 0; x < s.n; x++) w[x] = e[s.pos[x]];
    return w;
  };
  auto ewh = [&](const string &sn) {
    CallD &c = callOf(sn);
    const PointHighLow k = cutLast(zj(c), c.lJ, ls);
    vector<Ex> e = eqTab(k.hi, k.nhi);
    Seg &s = c.seg(sn);
    vector<Ex> w(s.n / S2);
    for (u64 xh = 0; xh < w.size(); xh++) w[xh] = e[s.pos[S2 * xh] / S2];
    return w;
  };
  vector<Ex> AH, lfc;
  {
    CallD &c = callOf("QK");
    const PointHighLow k = cutLast(zj(c), c.lJ, ldh);
    vector<Ex> e = eqTab(k.hi, k.nhi);
    Seg &s = c.seg("QK");
    AH.assign(64, eZ());
    // QK: q, then kx. AH[hd_]: eq at the block of head hd_'s dh columns; lfc: eq within a block.
    const TenAt tq = y.T_("q").at(), tk = y.T_("kx").at();
    for (u64 hd_ = 0; hd_ < h.H + h.Gk; hd_++)
      AH[hd_] = e[s.pos[hd_ < h.H ? headFeat(tq, hd_, dh) : tq.J + headFeat(tk, hd_ - h.H, dh)] / dh];
    lfc = eqTab(k.lo, ldh);
  }
  auto isAcc = [](const string &s) {
    return s == "G" || s == "U" || s == "V" || s == "O" || s == "D" || s == "LR";
  };
  vector<size_t> AC, TS;
  for (size_t q = 0; q < NC; q++) {
    bool a = false;
    for (auto &s : yy.calls[q].segs)
      if (isAcc(s.name)) a = true;
    if (a) AC.push_back(q);
  }
  for (auto &n : tscCalls) TS.push_back(callNo(n));
  auto tOf = [&](u32 k) {
    return k < 16 ? 16 - k : 0u;
  };
  auto pw = [&](u32 k) {
    return eI(1ll << k);
  };
  auto shortA = [&](u64 t, const Ten &X, u64 i, u64 j) {
    Aff s;
    u64 sc = 1;
    for (int l = 0; l < X.nl; l++, sc *= 256) affAdd(s, U3at(t, tenCell(X.at(), l, i, j)), eI((i64)sc));
    return s;
  };
  auto tabAt = [&](const string &n) {
    auto it = tab.find(n);
    if (it == tab.end()) refuse("internal-table-" + n);
    return it->second;
  };
  // Tensors read from the tables, as forms; limbs at radix 256. jh, kh: which block of S2 features.
  auto longA = [&](u64 t, const Ten &X, u64 i, u64 jh) {
    Aff s;
    u64 sc = 1;
    for (int l = 0; l < X.nl; l++, sc *= 256) affAdd(s, U3at(t, tenCell(X.at(), l, i, 0) / S2 + jh), eI((i64)sc));
    return s;
  };
  auto xbA = [&](u64 t, u64 kh) {
    Aff s;
    u64 sc = 1;
    for (u64 l = 0; l < y.xb.nl; l++, sc *= 256) affAdd(s, U3at(t, tenCell(y.xb, l, 0, 0) / S2 + kh), eI((i64)sc));
    return s;
  };
  auto sumE = [&](const vector<Ex> &v) {
    Ex s = eZ();
    for (auto &x : v) s = s + x;
    return s;
  };
  const char *mname[4] = {"af", "ad", "ar", "qa"};
  auto SA = [&](const vector<Ex> &wh, const string &X, u64 i, const string &Z) {
    const Ten &t = y.T_(X);
    const u64 tb = tabAt(string(mname[t.mat]) + "@" + Z);
    Aff s;
    for (u64 xh = 0; xh < wh.size(); xh++) affAdd(s, longA(tb, t, i, xh), wh[xh]);
    return s;
  };
  auto SxsA = [&](const vector<Ex> &wh, const string &Z, u64 sg) {
    const u64 tb = tabAt(string(sg == 0 ? "xbI" : sg == NW * Lam ? "xbO" : "af") + "@" + Z);
    const Ten &txs = y.T_("xs");
    Aff s;
    for (u64 xh = 0; xh < d / S2; xh++)
      affAdd(s, (sg == 0 || sg == NW * Lam) ? xbA(tb, xh) : longA(tb, txs, sg - 1, xh), wh[xh]);
    return s;
  };
  auto sumR = [&](const vector<Ex> &v, u64 a, u64 b) {
    Ex s = eZ();
    for (u64 i = a; i < b; i++) s = s + v[i];
    return s;
  };
  auto SBA = [&](const vector<Ex> &wh, u64 a, u64 b, const string &X, u64 i, const string &Z, u64 x0) {
    const Ten &t = y.T_(X);
    const u64 tb = tabAt(string(mname[t.mat]) + "@" + Z);
    Aff s;
    for (u64 xh = a / S2; xh < b / S2; xh++) affAdd(s, longA(tb, t, i, xh - x0 / S2), wh[xh]);
    return s;
  };
  auto shA = [&](const string &X, const string &P, u64 i, u64 j) {
    const Ten &t = y.T_(X);
    return shortA(tabAt(string(mname[t.mat]) + "@" + P), t, i, j);
  };
  const Ex o1 = eI(128), o2 = eI(32896);
  struct RowA {
    Aff Dg[74], V, c[8], s0, sig, l1, h1, Rlo, Rhi, R, p1, rr, rq;
  };
  const Ten &trow = y.T_("row");
  // Digit row i of the norms, with four limbs, at the point P: 11-2, "A norm's row of digits".
  auto rowAt = [&](const string &P, u64 i) {
    const u64 t = tabAt("ar@" + P);
    RowA r;
    for (int j = 0; j < 74; j++) r.Dg[j] = U3at(t, trow.OFF + i * 74 + j) + affC(eI(128));
    auto dig = [&](int a, int n) {
      Aff s;
      u64 sc = 1;
      for (int j = 0; j < n; j++, sc *= 256) affAdd(s, r.Dg[a + j], eI((i64)sc));
      return s;
    };
    r.V = dig(0, 8);
    for (int s = 1; s <= 6; s++) r.c[s] = dig(8 + 6 * (s - 1), 6) - affC(eI(1ll << 47));
    r.c[7] = r.Dg[7];
    r.s0 = r.Dg[45];
    r.sig = r.Dg[44] * eI(2) + r.Dg[45] + r.Dg[46] * eI(256) + r.Dg[47] * eI(65536) + r.Dg[48] * eI(1 << 24);
    r.l1 = dig(49, 4);
    r.h1 = dig(53, 4);
    r.Rlo = dig(57, 3);
    r.Rhi = dig(60, 3);
    r.R = r.Rlo + r.Rhi * eI(1 << 24);
    r.p1 = dig(63, 3);
    r.rr = dig(66, 4);
    r.rq = dig(70, 4);
    return r;
  };
  // The same with five limbs: 99 digits.
  auto row5At = [&](const string &P, u64 i) {
    const u64 t = tabAt("ar@" + P);
    return row5<Aff>([&](int j) { return U3at(t, trow.OFF + i * 99 + (u64)j); });
  };
  // The reciprocal R of digit row i at the point P.
  auto rowR = [&](const string &P, u64 i) {
    return residualStreamLimbs == 5 ? row5At(P, i).R : rowAt(P, i).R;
  };
  /* Steps 8 and 9: the two token sumchecks (vf_sumcheck_token.h). TSC(P_V) starts from the slot
     tV.sCl. NOl: its end values for each instance (11-2, "The scalars by name"). */
  const u64 NOl = (u64)ledNO();
  TokenSumcheckState tV = tscIn(F, tr, slot, PV.data(), lt, (u32)(NOl * NI), 2);
  pt["r_tV"] = tV.r;
  /* TSC(zt), unless tscCalls is empty: it starts from the slot tT.sCl and ends at the point rt.
     Each of its terms is a product of two vectors over the tokens. One factor of each product is
     shown, unless it is public (the rotary tables): R of every instance; then, with p3, sc of every
     mixture-of-experts layer, then the selector's NE entries of each. The summand is affine in the
     other factors. KT is the sum of the terms at rt, with the shown values in place of the shown
     factors. It is no form in committed values, because the other factors contain products with
     weights. So it enters the batched sumcheck's start form. */
  u32 sSc = 0, sSel = 0;
  TokenSumcheckState tT;
  Aff KT;
  const u64 nEndT = NN * Lam + (p3 ? LamM * (1 + NE) : 0);
  if (hasT) {
    tT = tscIn(F, tr, slot, ztF.data(), lt, (u32)nEndT, 1);
    sSc = tT.sEnd + (u32)(NN * Lam);
    sSel = sSc + (u32)LamM;
    pt["Y"] = longPoint(fo.zx.data() + (fo.dT - la), ls, tT.r);
    pt["rt"] = tT.r;
    // The sumcheck's claim is u + eps KT = 0, u being pscRow()'s form without the terms of KT.
    // eps = 0 is rejected here. The error of the round that samples eps accounts for that event.
    if (isZ(tT.p.eps)) refuse("coin-zero");
    KT = pscRow(F, tT.p, eZ()) * (eZ() - eInv(tT.p.eps));
  }
  // site(), high(): SITE and HIGH of 11-2, "The keys of the lookup of step 1", where A is the
  // accumulator. normK(): K1 or K2 there.
  auto normK = [&](u64 l, bool first) {
    return kl(l, first ? LK1 : LK2);
  };
  auto site = [&](const Ex &w, const Aff &SQ, u32 k, int form, const std::function<Aff()> &Srh) {
    Aff v = (affC(w * pw(k - 1)) - SQ * pw(k)) * pw(tOf(k));
    if (form) v = v - (affC((form == 1 ? o1 : o2) * w) + Srh()) * pw(16);
    return v;
  };
  auto high = [&](const Ex &w, const Aff &Srh, u32 k, int form) {
    const Aff s = affC((form == 1 ? o1 : o2) * w) + Srh;
    const u32 top = form == 1 ? 24 : 32;
    return s * pw(top - (k > 16 ? k : 16));
  };
  // The part of lookup segment sn at the group's layer lam in the join, which is the lookup's claim
  // on all keys (JF below): a form, without the accumulators. Each column has a key for each token.
  // On 11-2 it is LINE_SEG(lam), under the same heading as SITE; the page writes up, sl, an, mn for
  // the tensors u, s, a, m.
  auto LINE = [&](const string &sn, u64 lam) -> Aff {
    const string Z = "Z", z = "zt";
    const u64 l = g * Lam + lam;
    const vector<Ex> ew = ewOf(sn);
    const u64 n = ew.size();
    const Ex w = sumE(ew);
    vector<Ex> wh;
    if (n % S2 == 0 && sn != "ROW") wh = ewh(sn);
    auto S = [&](const string &X) {
      return SA(wh, X, lam, Z);
    };
    const u64 lamM = lam / 2;
    const u32 kq = kl(l, Lkq), kk = kl(l, Lkk), kv = kl(l, Lkv), ko = kl(l, Lko), kd = kl(l, Lkd),
              kr = isMoe(lam) ? klM(l, 0) : 0;
    if (sn == "G" || sn == "U") {
      const bool up = sn == "U";
      const int form = up ? fU : fG;
      const char *Qn = up ? "u" : "gi", *rh = up ? "ru" : "rg";
      // The quotient on the indices [a, b).
      auto Q = [&](u64 a, u64 b) {
        if (up || !GCL) return SBA(wh, a, b, Qn, lam, Z, 0);
        return SBA(wh, a, b, "g0", lam, Z, 0) + SBA(wh, a, b, "gh", lam, Z, 0) * eI(256);
      };
      if (!isMoe(lam)) return site(w, Q(0, f), kl(l, up ? Lku : Lkg), form, [&]() { return S(rh); });
      const Aff a = site(sumR(ew, 0, fe), Q(0, fe), kl(l, up ? Lku : Lkg), form,
                           [&]() { return SBA(wh, 0, fe, rh, lam, Z, 0); });
      return a + site(sumR(ew, fe, f), Q(fe, f), klM(l, up ? 2 : 1), form,
                      [&]() { return SBA(wh, fe, f, rh, lam, Z, 0); });
    }
    if (sn == "GH" || sn == "UH") {
      const bool up = sn == "UH";
      const char *rh = up ? "ru" : "rg";
      const int form = up ? fU : fG;
      if (!isMoe(lam)) return high(w, S(rh), kl(l, up ? Lku : Lkg), form);
      const Aff a = high(sumR(ew, 0, fe), SBA(wh, 0, fe, rh, lam, Z, 0), kl(l, up ? Lku : Lkg), form);
      return a + high(sumR(ew, fe, f), SBA(wh, fe, f, rh, lam, Z, 0), klM(l, up ? 2 : 1), form);
    }
    if (sn == "UT") return S("u") + affC(pw(LBU) * w);
    if (sn == "QK" || sn == "QKH") {
      Aff a;
      for (int b = 0; b < 2; b++) {
        const u64 A0 = b ? dq : 0, B0 = b ? nqk : dq;
        const u32 k = b ? kk : kq;
        const Ex wB = sumR(ew, A0, B0);
        if (sn == "QK")
          a = a + site(wB, SBA(wh, A0, B0, "qkw", lam, Z, 0), k, fQK,
                       [&]() { return SBA(wh, A0, B0, "rqk", lam, Z, 0); });
        else
          a = a + high(wB, SBA(wh, A0, B0, "rqk", lam, Z, 0), k, fQK);
      }
      return a;
    }
    if (sn == "N1" || sn == "N2") {
      const bool n1 = sn == "N1";
      return site(w, S(n1 ? "aw" : "mw"), normK(l, n1), 2, [&]() { return S(n1 ? "ra" : "rm"); });
    }
    if (sn == "N1H" || sn == "N2H") return high(w, S(sn == "N1H" ? "ra" : "rm"), normK(l, sn == "N1H"), 2);
    if (sn == "O") return site(w, SxsA(wh, Z, 2 * lam + 1) - SxsA(wh, Z, 2 * lam), ko, fO, [&]() { return S("ro"); });
    if (sn == "D")
      return site(w, SxsA(wh, Z, 2 * lam + 2) - SxsA(wh, Z, 2 * lam + 1), kd, fD, [&]() { return S("rd"); });
    if (sn == "V") return site(w, S("vw"), kv, fV, [&]() { return S("rv"); });
    if (sn == "OH") return high(w, S("ro"), ko, fO);
    if (sn == "DH") return high(w, S("rd"), kd, fD);
    if (sn == "VH") return high(w, S("rv"), kv, fV);
    if (sn == "ROW") {
      Aff a;
      const u64 t = tabAt("ar@" + z);
      for (u64 x = 0; x < n; x++) {
        const u64 np = x / NRl, idx = x % NRl, mu = ledMu((int)idx);
        affAdd(a, U3at(t, trow.OFF + (NN * lam + np) * NRl + idx) + affC(eI(128)), eS(ew[x], mu));
      }
      return a;
    }
    // Lookup table FC's row for x: (x + 2^15) + 2^28 (clamp(x, +-2047) + 2^11), hence 2^15 + 2^39.
    if (sn == "CN1" || sn == "CN2") {
      Aff b;
      for (u64 x = 0; x < d; x++) affAdd(b, shA(sn == "CN1" ? "a" : "m", z, lam, x), ew[x]);
      return S(sn == "CN1" ? "aw" : "mw") + b * pw(28) + affC(w * eI((1ll << 15) + (1ll << 39)));
    }
    if (sn == "CQK")
      return S("qkw") + (SBA(wh, 0, dq, "q", lam, Z, 0) + SBA(wh, dq, nqk, "kx", lam, Z, dq)) * pw(28) +
             affC(w * eI((1ll << 15) + (1ll << 39)));
    if (sn == "CV") return S("vw") + S("v") * pw(28) + affC(w * eI((1ll << 15) + (1ll << 39)));
    if (sn == "F")
      return (GCL ? S("g0") + S("gc") * eI(256) : S("gi")) + S("s") * pw(28) +
             affC(w * eI((1ll << 15) + (1ll << 40)));
    // G0's lookup key g0 + 128 limits g0 to a byte. CG's must be a row of lookup table FS. That
    // alone does not give gc = clamp(gh, +-127): CG, F and G0 together do (F reads g0 and gc too).
    if (sn == "G0") return S("g0") + affC(w * eI(128));
    if (sn == "CG") return S("gh") + S("gc") * pw(28) + affC(w * eI(49152 + (1ll << 35) + (1ll << 40)));
    // Lookup segments of mixture-of-experts layers only: joinPart() skips them on the other layers.
    if (sn == "XR") {
      Aff b;
      for (u64 x = 0; x < d; x++) affAdd(b, shA("xr", z, lamM, x), ew[x]);
      return affC(pw(15) * w) - b * pw(16);
    }
    if (sn == "LR") {
      Aff q;
      for (u64 x = 0; x < NE; x++)
        affAdd(q, RCL ? shA("ly", z, lamM, x) * eI(8) + shA("l3", z, lamM, x) : shA("lr", z, lamM, x), ew[x]);
      return site(w, q, kr, fR, [&]() {
        Aff s;
        for (u64 x = 0; x < NE; x++) affAdd(s, shA("rl", z, lamM, x), ew[x]);
        return s;
      });
    }
    if (sn == "LT") {
      Aff a;
      for (u64 x = 0; x < NE; x++) affAdd(a, shA("lr", z, lamM, x) + affC(pw(14)), eS(ew[x], 2));
      return a;
    }
    if (sn == "DD") {
      Aff a, tail_;
      for (u64 x = NE; x-- > 0;) {
        affAdd(a, shA("lr", z, lamM, x) + tail_, eZ() - ew[x]);
        tail_ = tail_ + shA("sel", z, lamM, x);
      }
      return a;
    }
    if (sn == "LH") {
      Aff s;
      for (u64 x = 0; x < NE; x++) affAdd(s, shA("rl", z, lamM, x), ew[x]);
      return high(w, s, kr, fR);
    }
    if (sn == "SEL") {
      Aff a;
      for (u64 x = 0; x < NE; x++) affAdd(a, shA("sel", z, lamM, x), eS(ew[x], 128));
      return a;
    }
    if (sn == "MR") return (affC(pw(GCL ? 14 : 15)) + shA("sc", z, lamM, 0) * pw(17)) * ew[0];
    // L3's key 32 l3 limits l3 to 0 .. 7. CLR's key is FC's row for ly exactly if lr - l3 = 8 clamp(ly, +-2047).
    if (sn == "L3") {
      Aff a;
      for (u64 x = 0; x < NE; x++) affAdd(a, shA("l3", z, lamM, x), eS(ew[x], 32));
      return a;
    }
    if (sn == "CLR") {
      Aff a;
      for (u64 x = 0; x < NE; x++)
        affAdd(a, shA("ly", z, lamM, x) + (shA("lr", z, lamM, x) - shA("l3", z, lamM, x)) * pw(25), ew[x]);
      return a + affC(w * eI((1ll << 15) + (1ll << 39)));
    }
    refuse("hid-profile-segment-" + sn);
  };
  /* With e_c = fo.e[c], JF = sum over all calls c of e_c (alpha - c theta - tau_c(t0_c) PADW_c
   *        - sum_lam el_c[lam] sum_sn `LINE`(sn, lam)) + the trees of padding alone - Cl - C_end.
   * It is the lookup's join (`forestJoin`) with the accumulators moved to the batched sumcheck's
   * side. Cl (slot tT.sCl) is absent if tscCalls is empty. `joinPart` is one call's bracket. */
  // The padding is described twice: fo.pub counts it by tree (CallD::padding()), and joinPart()
  // finds its places. It marks the columns (lam, s.pos[x]) of each lookup segment whose form it
  // adds, and stops at a second mark. Any other column is padding: its weight goes into PADW_c and
  // T keys into its tree's count, which forestPaddingLaid() compares with fo.pub. Marks and
  // CallD::used both come from s.pos, not from inside LINE(): a slip in a form is not seen.
  vector<u64> laid(fo.NT, 0);
  auto joinPart = [&](CallD &c, size_t q) {
    const vector<Ex> el = eqTab(zl(c), ll), e = eqTab(zj(c), c.lJ);
    vector<char> weighed(Lam * c.J, 0);
    Aff lines;
    for (u64 lam = 0; lam < Lam; lam++)
      for (auto &s : c.segs) {
        if (s.moe && !isMoe(lam)) continue;
        affAdd(lines, LINE(s.name, lam), eZ() - el[lam]);
        for (u64 x = 0; x < s.n; x++)
          if (weighed.at(lam * c.J + s.pos[x])++) refuse("internal-g-join-place"); // a fault of the program, not a verdict: the call ends NOT CHECKED (vf_common.h, internalName)
      }
    Ex padW = eZ();
    for (u64 lam = 0; lam < Lam; lam++) {
      Ex s = eZ();
      for (u64 j = 0; j < c.J; j++)
        if (!weighed[lam * c.J + j]) s = s + e[j], laid.at((u64)fo.first[q] + ((lam * c.J + j) >> (fo.dT - lt))) += T;
      padW = padW + el[lam] * s;
    }
    return affC(fo.alpha - eS(fo.theta, (u64)q) - eS(padW, c.c.tauAt(c.c.t0))) + lines;
  };
  // om[0] = 1 for the join; om[1], a challenge, for KT, only if [ListT] = 1: tscCalls is not empty.
  const size_t NFC = 1 + (hasT ? 1 : 0);
  const vector<Ex> om = NFC > 1 ? trBatch(tr, NFC) : vector<Ex>(1, eO()); // no message if there is one claim
  // The batched sumcheck: sum_x A[x] B[x]. A panel (SBlk): 2^w aligned x, ni rows of ny entries.
  u32 nextV = 0; // The number of the next weight claim, in list order.
  struct SBlk {
    bool tri = false; // A(e, yv) = sl[e] xx[yv]
    long v0 = -1; // For a weight panel: the number of its first claim. Otherwise -1.
    size_t claim = 0; // om's index: 0, a panel of the join (loop over AC); 1, of KT (loop over TS)
    Ex ec = eZ();     // e_c of the panel's call.
    u64 ni = 0, ny = 0;
    int wy = 0, w = 0;
    u64 off = 0;
    std::function<Aff(u64, u64)> A, B;
    vector<Ex> wt;
    vector<Ex> sl;    // tri: the selector's values at rt, as TSC(zt) has shown them (public numbers)
    vector<Aff> xx; // tri: the column factor, unless it is never sent
    long virt = -1;   // never-sent vector (lookup segment D): 0, A is Hh1; 1 + lm, xx is Hh2(lm, .)
    size_t listNo = 0;
  };
  vector<SBlk> bl;
  // 2^lift(k) of 11-2, for the shift k.
  auto kT = [&](u64 l, int i) {
    return pw(tOf(kl(l, i)));
  };
  auto kTM = [&](u64 l, int i) {
    return pw(tOf(klM(l, i)));
  };
  const int le = clog2(NE ? NE : 1);
  /* `plain` appends a two-factor panel. Its row i is the group's layer lam = i or, if moeSet, the
   * mixture-of-experts layer 2 i + 1. A, B, wt take lam. Empty B: B is weights (a weight panel). */
  Ex ecNow = eZ(); // e_c of the call whose panels are being listed
  auto plain = [&](size_t claim, bool moeSet, u64 ny, std::function<Aff(u64, u64)> A,
                   std::function<Aff(u64, u64)> B, std::function<Ex(u64)> wt) {
    SBlk b;
    b.claim = claim;
    b.ec = ecNow;
    b.ni = moeSet ? LamM : Lam;
    b.ny = ny;
    b.wy = clog2(ny);
    b.w = clog2(b.ni) + b.wy;
    if (A)
      b.A = [A, moeSet](u64 i, u64 yv) {
        return A(moeSet ? 2 * i + 1 : i, yv);
      };
    if (B)
      b.B = [B, moeSet](u64 i, u64 yv) {
        return B(moeSet ? 2 * i + 1 : i, yv);
      };
    for (u64 i = 0; i < b.ni; i++) b.wt.push_back(wt(moeSet ? 2 * i + 1 : i));
    if (!B) {
      b.v0 = nextV;
      nextV += (u32)b.ni;
    }
    b.listNo = bl.size();
    bl.push_back(b);
  };
  /* `tri` appends a three-factor panel ("selector" in 11-2): NE rows of nk, one coefficient wt and
   * one weight claim. */
  auto tri = [&](size_t claim, u64 nk, vector<Ex> sl, vector<Aff> xx, long virt, Ex wt) {
    SBlk b;
    b.tri = true;
    b.claim = claim;
    b.ec = ecNow;
    b.ni = NE;
    b.ny = nk;
    b.wy = clog2(nk);
    b.w = le + b.wy;
    b.sl = std::move(sl);
    b.xx = std::move(xx);
    b.virt = virt;
    b.wt.push_back(wt);
    b.v0 = nextV++;
    b.listNo = bl.size();
    bl.push_back(b);
  };
  auto lay = [&](u64 lam) {
    return g * Lam + lam;
  };
  for (size_t ci = 0; ci < AC.size(); ci++) {
    CallD &c = yy.calls[AC[ci]];
    const vector<Ex> el = eqTab(zl(c), ll);
    const string z = "zt";
    ecNow = fo.e[AC[ci]];
    for (auto &s : c.segs) {
      const string sn = s.name;
      if (sn == "G" || sn == "U") {
        const bool up = sn == "U";
        plain(
            0, false, d, [&, z](u64 lam, u64 k2) { return shA("m", z, lam, k2); },
            std::function<Aff(u64, u64)>(),
            [&, el, up](u64 lam) { return el[lam] * kT(lay(lam), up ? Lku : Lkg); });
      } else if (sn == "V")
        plain(
            0, false, d, [&, z](u64 lam, u64 k2) { return shA("a", z, lam, k2); },
            std::function<Aff(u64, u64)>(), [&, el](u64 lam) { return el[lam] * kT(lay(lam), Lkv); });
      /* Lookup segment O: A is attention's output o at zt, from qa's short table. */
      else if (sn == "O") {
        plain(
            0, false, dq, [&, z](u64 lam, u64 k2) { return shA("o", z, lam, k2); },
            std::function<Aff(u64, u64)>(), [&, el](u64 lam) { return el[lam] * kT(lay(lam), Lko); });
      } else if (sn == "D") {
        // A is never sent: Hh1(lam, k2) = sum_t eq(zt, t) hh(lam, k2, t), hh = s u, for all k2 < f.
        // (On a mixture-of-experts layer B is 0 for k2 >= fe.) Here e_c and wt multiply B, not A,
        // so that A's end value is om[0] times Hh1's multilinear extension at a point.
        plain(0, false, f, std::function<Aff(u64, u64)>(), std::function<Aff(u64, u64)>(),
              [&, el](u64 lam) { return el[lam] * kT(lay(lam), Lkd); });
        bl.back().virt = 0;
      } else if (sn == "LR")
        plain(
            0, true, d, [&, z](u64 lam, u64 k2) { return shA("m", z, lam, k2); }, std::function<Aff(u64, u64)>(),
            [&, el](u64 lam) { return el[lam] * kTM(lay(lam), 0); });
    }
  }
  for (size_t ti = 0; ti < TS.size(); ti++) {
    CallD &c = yy.calls[TS[ti]];
    const size_t claim = 1;
    const vector<Ex> el = eqTab(zl(c), ll);
    const string r = "rt", Y = "Y";
    ecNow = fo.e[TS[ti]];
    for (auto &s : c.segs) {
      const string sn = s.name;
      if (sn == "N1" || sn == "N2") {
        const u64 n2 = sn == "N2";
        const vector<Ex> wh = ewh(sn);
        plain(
            claim, false, 1, [&, n2](u64 lam, u64) { return affC(F.shown(tT.sEnd + (u32)(2 * lam + n2))); },
            [&, wh, Y, n2](u64 lam, u64) { return SxsA(wh, Y, 2 * lam + n2); }, [el](u64 lam) { return el[lam]; });
      } else if (sn == "QK") {
        plain(
            claim, false, d, [&, r](u64 lam, u64 k2) { return shA("a", r, lam, k2); },
            std::function<Aff(u64, u64)>(), [&, el](u64 lam) { return el[lam] * kT(lay(lam), Lkq); });
      } else if (!p3)
        continue;
      else if (sn == "XR") {
        const vector<Ex> ew = ewOf("XR");
        plain(
            claim, true, 1, [&](u64 lam, u64) { return affC(F.shown(sSc + (u32)(lam / 2))); },
            [&, r, ew](u64 lam, u64) {
              Aff b;
              for (u64 x = 0; x < d; x++) affAdd(b, shA("m", r, lam, x), ew[x]);
              return b;
            },
            [el](u64 lam) { return el[lam]; });
      } else if (sn == "DD" || sn == "MR") {
        const vector<Ex> ew = ewOf(sn);
        const Ex wS = sn == "DD" ? sumE(ew) : ew[0];
        plain(
            claim, true, NE, [&](u64 lam, u64 e) { return affC(F.shown(sSel + (u32)((lam / 2) * NE + e))); },
            [&, r](u64 lam, u64 e) { return shA("lr", r, lam / 2, e); },
            [el, wS](u64 lam) { return el[lam] * wS; });
      } else if (sn == "G" || sn == "U" || sn == "D")
        for (u64 lm = 0; lm < LamM; lm++) {
          const u64 lam = 2 * lm + 1;
          vector<Ex> sl;
          vector<Aff> xx;
          for (u64 e = 0; e < NE; e++) sl.push_back(F.shown(sSel + (u32)(lm * NE + e)));
          if (sn == "D") {
            // xx is never sent: Hh2(lm, k2) = sum_t eq(rt, t) hh(2 lm + 1, fe + k2, t).
            tri(claim, fe, sl, xx, 1 + (long)lm, el[lam] * kT(lay(lam), Lkd));
          } else {
            const bool up = sn == "U";
            for (u64 k2 = 0; k2 < d; k2++) xx.push_back(shA("xr", r, lm, k2));
            tri(claim, d, sl, xx, -1, el[lam] * kTM(lay(lam), up ? 2 : 1));
          }
        }
    }
  }
  // Sets `off`, a panel's first x. Largest panels first, so each is aligned; ties in list order.
  vector<size_t> ord(bl.size());
  for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
  std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return bl[a].w > bl[b].w; });
  {
    u64 off = 0;
    for (size_t i : ord) {
      bl[i].off = off;
      off += 1ull << bl[i].w;
    }
    if (clog2(off) != hp.lamS) refuse(F.G_ + "internal-stack");
  }
  /* The start form is cstart = JF + om[1] KT (JF: see `joinPart`). All its coefficients are fixed
     by now. Its entries of U3 are committed later, with U3's root. They are functions of the cells alone only if the points of the U3 tables that it names are drawn before this sumcheck's eps (pscBegin() below): by a reading of the code they are, and no test or self-check enforces that.
     Then come lamS rounds without an equality factor; st.rs[v] is the coordinate of bit v. */
  HStack st;
  st.lam = hp.lamS;
  {
    Aff cstart;
    for (size_t q = 0; q < NC; q++) affAdd(cstart, joinPart(yy.calls[q], q), fo.e[q]);
    forestPaddingLaid(fo, laid);
    {
      // The trees that hold only padding, as in forestJoin() (vf_lookup_argument.h).
      const vector<Ex> ey = eqTab(fo.zy.data(), fo.rK);
      for (int t = fo.NT; t < (1 << fo.rK); t++) cstart.c = cstart.c + ey[t] * (fo.alpha - fo.padKey(0));
    }
    if (hasT) cstart = cstart - affSlot(tT.sCl);
    cstart = cstart - fo.Cend;
    if (hasT) affAdd(cstart, KT, om[1]);
    st.p = pscBegin(F, tr, slot, cstart);
  }
  stackRounds(F, tr, slot, st);
  /* The end values, in the order sent: A0; H1, for Hh1; with routed experts one for each
     Hh2(lm, .); B0; the values v_a of the panels' NCLp weight claims. What each is:
     docs/format/11-2-feed-forward-segment.md, "The scalars by name" (step 11) and "The panels of
     step 11". */
  const u64 NVIRT = 1 + (p3 ? LamM : 0), NCLp = hp.NCL - (hasE ? 1 : 0);
  if (nextV != NCLp) refuse(F.G_ + "internal-claim-count");
  const u32 sA0 = pscEnds(F, tr, slot, st.p, 2, (u32)(2 + NVIRT + NCLp));
  const u32 sH1 = sA0 + 1, sB0 = sH1 + (u32)NVIRT, sV = sB0 + 1;
  Aff Alin, Blin; // what A0 and B0 must equal
  Aff rowS;       // the claim that the sumcheck leaves
  int wD = 0;       // w of the panel whose A is Hh1
  {
    Ex As = F.shown(sA0), Bs = F.shown(sB0);
    for (auto &b : bl) {
      const Ex sb = eqLowAt(st.rs, b.w, st.lam, b.off >> b.w);
      const vector<Ex> EY = eqLow(st.rs, 0, b.wy), EI = eqLow(st.rs, b.wy, b.w);
      const Ex co = sb * om[b.claim] * b.ec;
      if (b.tri) {
        Ex s1 = eZ();
        for (u64 e = 0; e < NE; e++) s1 = s1 + b.sl[e] * EI[e];
        if (b.virt >= 0) { // xx is Hh2: hd2's point takes the lowest lfe coordinates
          if (b.wy != lfe) refuse(F.G_ + "internal-stack");
          As = As + co * b.wt[0] * s1 * F.shown(sH1 + (u32)b.virt);
        } else
          for (u64 k2 = 0; k2 < b.ny; k2++) affAdd(Alin, b.xx[k2], co * b.wt[0] * s1 * EY[k2]);
        Bs = Bs + sb * F.shown(sV + (u32)b.v0);
        continue;
      }
      if (b.virt >= 0) { // A is Hh1: the A side is om[0] A, the B side e_c wt[i] B
        if (b.v0 < 0 || wD) refuse(F.G_ + "internal-stack");
        wD = b.w;
        As = As + sb * om[b.claim] * F.shown(sH1 + (u32)b.virt);
        for (u64 i = 0; i < b.ni; i++) Bs = Bs + sb * EI[i] * b.ec * b.wt[i] * F.shown(sV + (u32)b.v0 + (u32)i);
        continue;
      }
      for (u64 i = 0; i < b.ni; i++) {
        for (u64 yv = 0; yv < b.ny; yv++) {
          affAdd(Alin, b.A(i, yv), co * b.wt[i] * EI[i] * EY[yv]);
          if (b.v0 < 0) affAdd(Blin, b.B(i, yv), sb * EI[i] * EY[yv]);
        }
        if (b.v0 >= 0) Bs = Bs + sb * EI[i] * F.shown(sV + (u32)b.v0 + (u32)i);
      }
    }
    if (wD != ll + lPf) refuse(F.G_ + "internal-stack");
    rowS = pscRow(F, st.p, As * Bs);
  }
  // The product layer for hh = s u starts from H1. Its point is (st.rs[wD - 1 .. 0] | zt).
  ProductLevel hd;
  {
    vector<Ex> rho;
    for (int v = wD - 1; v >= 0; v--) rho.push_back(st.rs[v]);
    rho.insert(rho.end(), ztF.begin(), ztF.end());
    hd = readProductLevelRounds(F, tr, slot, muH, rho, affSlot(sH1), orderRange(muH));
  }
  pt["Z_a"] = lastLa(hd.r);
  // With routed experts, the second product layer:
  // docs/format/11-2-feed-forward-segment.md, steps 13 and 14.
  ProductLevel hd2;
  if (p3) {
    if (!hasT) refuse(F.G_ + "internal-g-routed-without-terms");
    vector<Ex> rho;
    if (ll > 1) rho = trPoint(tr, RoundName::JOIN, ll - 1);
    const vector<Ex> e = eqTab(rho.data(), ll - 1);
    Aff C;
    for (u64 lm = 0; lm < LamM; lm++) affAdd(C, affSlot(sH1 + 1 + (u32)lm), e[lm]);
    for (int v = lfe - 1; v >= 0; v--) rho.push_back(st.rs[v]);
    rho.insert(rho.end(), tT.r.begin(), tT.r.end());
    hd2 = readProductLevelRounds(F, tr, slot, muH2, rho, C, orderRange(muH2));
    pt["Z_b"] = lastLa(hd2.r);
  }
  // The merge's weight claims follow the weight panels in bl's order (unsorted): a two-factor panel
  // gives ni claims, i ascending, and a three-factor panel gives one. In G_0 the embedding's is
  // last. A claim has a factor per index group (WsGrp) of its unit, a weight tensor (vf_weights_layout.h).
  vector<WsClaim> claims;
  {
    std::map<std::pair<u32, u32>, u32> uiOf;
    for (size_t i = 0; i < wsLay.U.size(); i++) uiOf[{wsLay.U[i].X, wsLay.U[i].l}] = (u32)i;
    const int le_ = clog2(NE ? NE : 1);
    // The factor FR(lo0, wtot): docs/format/11-2-feed-forward-segment.md, "The weight claims".
    auto facRs = [&](const WsGrp &G, int lo0, int wtot) {
      WsFac x;
      for (int i = 0; i < G.z; i++) x.pt.push_back(st.rs[lo0 + G.z - 1 - i]);
      const vector<Ex> e = eqLow(st.rs, lo0 + G.z, lo0 + wtot);
      for (u64 hi = 0; hi < G.o; hi++) x.HV.push_back(e[hi]);
      return x;
    };
    // The factor FE(sn, x0): docs/format/11-2-feed-forward-segment.md, "The weight claims".
    // Not tested here: by place(), pos keeps runs of 2^z aligned, in order.
    auto facEw = [&](const WsGrp &G, const string &sn, u64 x0) {
      CallD &c = callOf(sn);
      Seg &s = c.seg(sn);
      WsFac x;
      for (int i = 0; i < G.z; i++) x.pt.push_back(zj(c)[c.lJ - G.z + i]);
      const vector<Ex> e = eqTab(zj(c), c.lJ - G.z);
      for (u64 xh = 0; xh < G.o; xh++) x.HV.push_back(e[s.pos[x0 + (xh << G.z)] >> G.z]);
      return x;
    };
    auto claim = [&](u32 X, u64 lam, std::function<WsFac(const WsGrp &)> fac) {
      auto it = uiOf.find({X, (u32)(g * Lam + lam)});
      if (it == uiOf.end()) refuse(F.G_ + "internal-claim-unit");
      WsClaim c;
      c.ui = it->second;
      for (auto &G : wsLay.U[c.ui].g) c.f.push_back(fac(G));
      claims.push_back(std::move(c));
    };
    for (size_t ci = 0; ci < AC.size(); ci++) {
      CallD &c = yy.calls[AC[ci]];
      for (auto &s : c.segs) {
        const string sn = s.name;
        if (!isAcc(sn)) continue;
        for (u64 i = 0; i < (sn == "LR" ? LamM : Lam); i++) {
          const u64 lam = sn == "LR" ? 2 * i + 1 : i;
          const bool mo = isMoe(lam);
          const u32 X = sn == "G"   ? (mo ? 5u : 3u)
                        : sn == "U" ? (mo ? 6u : 4u)
                        : sn == "V" ? 1u
                        : sn == "O" ? 2u
                        : sn == "D" ? 9u
                                    : 11u;
          const int wy = clog2(sn == "O" ? dq : sn == "D" ? f : d);
          claim(X, lam, [&](const WsGrp &G) { return !strcmp(G.name, "kk") ? facRs(G, 0, wy) : facEw(G, sn, 0); });
        }
      }
    }
    for (size_t ti = 0; ti < TS.size(); ti++) {
      CallD &c = yy.calls[TS[ti]];
      for (auto &s : c.segs) {
        const string sn = s.name;
        if (sn == "QK") {
          const vector<Ex> et = eqTab(tT.r.data(), lt);
          for (u64 lam = 0; lam < Lam; lam++) {
            vector<Ex> kap(dh);
            const u64 l = g * Lam + lam;
            for (u64 i = 0; i < dh / 2; i++) {
              Ex Ct = eZ(), St = eZ();
              for (u64 t = 0; t < T; t++) {
                const i64 cv = ropeC(l, t, i, false), sv = ropeC(l, t, i, true);
                if (cv) Ct = Ct + eS(et[t], fint(cv));
                if (sv) St = St + eS(et[t], fint(sv));
              }
              kap[2 * i] = Ct * lfc[2 * i] + St * lfc[2 * i + 1];
              kap[2 * i + 1] = Ct * lfc[2 * i + 1] - St * lfc[2 * i];
            }
            claim(0, lam, [&](const WsGrp &G) {
              WsFac x;
              if (!strcmp(G.name, "hd"))
                x.AV = AH;
              else if (!strcmp(G.name, "cd"))
                x.AV = kap;
              else
                x = facRs(G, 0, clog2(d));
              return x;
            });
          }
        } else if (p3 && (sn == "G" || sn == "U" || sn == "D"))
          for (u64 lm = 0; lm < LamM; lm++) {
            const int wk = clog2(sn == "D" ? fe : d);
            claim(sn == "G"   ? 7u
                  : sn == "U" ? 8u
                              : 10u,
                  2 * lm + 1, [&](const WsGrp &G) {
                    return !strcmp(G.name, "kk")  ? facRs(G, 0, wk)
                           : !strcmp(G.name, "e") ? facRs(G, wk, le_)
                                                  : facEw(G, sn, sn == "D" ? 0 : fe);
                  });
          }
      }
    }
    // The embedding's claim (unit X = 12): v_E = sum_t eq(ztE, t) sum_(k < d) phiE[k] Emb[tok_t][k]
    // The factor of group "v" is psi[v] = sum_(t: tok_t = v) eq(ztE, t).
    // The other factor, phiE, is eq at (tkh | zsE), top bit first.
    if (hasE) {
      auto it = uiOf.find({12u, (u32)h.L});
      if (it == uiOf.end()) refuse(F.G_ + "internal-claim-unit");
      WsClaim c;
      c.ui = it->second;
      const PointHighLow kE = cutLast(ZE.data(), (int)ZE.size(), lt); // Z_E = (zsE | ztE): kE.hi, kE.lo
      vector<Ex> pnt = tkh;
      pnt.insert(pnt.end(), kE.hi, kE.hi + kE.nhi);
      const vector<Ex> et = eqTab(kE.lo, lt);
      for (auto &G : wsLay.U[c.ui].g) {
        WsFac x;
        if (!strcmp(G.name, "v")) {
          x.AV.assign(Vv, eZ());
          for (u64 t = 0; t < T; t++) x.AV[tokv[t]] = x.AV[tokv[t]] + et[t];
        } else {
          for (int i = 0; i < G.z; i++) x.pt.push_back(pnt[lPd - G.z + i]);
          const vector<Ex> e = eqTab(pnt.data(), lPd - G.z);
          for (u64 hi = 0; hi < G.o; hi++) x.HV.push_back(e[hi]);
        }
        c.f.push_back(std::move(x));
      }
      claims.push_back(std::move(c));
    }
    if (claims.size() != hp.NCL) refuse(F.G_ + "internal-claim-count");
  }
  // The panels' claims have their values from the batched sumcheck's end; v_E is sent separately.
  vector<u32> sVa;
  for (u64 a = 0; a < NCLp; a++) sVa.push_back(sV + (u32)a);
  u32 sVE = 0;
  if (hasE) sVa.push_back(sVE = F.claimsIn(tr, slot, 1));
  WsForkOut wo = wsForkEnd(F, tr, slot, wsLay, claims, sVa);
  // All points are known: laterPhi's tables get phi. Then the root of U3 (W in rejection names).
  for (auto &x : laterPhi) {
    const auto it = pt.find(x.point);
    if (it == pt.end() || it->second.size() != (size_t)(x.lng ? la : lt)) refuse(F.G_ + "internal-g-table-point");
    const vector<Ex> phi = x.lng ? longPhi(it->second.data(), ls, lt) : eqTab(it->second.data(), lt);
    for (size_t q = 0; q < x.count; q++) fun[x.first + q].phi = phi;
  }
  rootInH(F, tr, {(int)iU3});
  tl.closeFuns(fun);
  // The range check's messages (tl: an `HTailF2`, vf_smallness_test.h).
  tl.transcript(F, tr, slot);
  if (slot != hp.used) refuse(F.G_ + "internal-slot-count");
  // hp.nLin: the planned number of claims; vf_layers_plan.inc computes it from the shape parameters.
  F.planClaimsF = (u32)hp.nLin;
  // The challenge whose powers batch the claims, sampled after the last message.
  t2Batch(F, T2, tr, slot);
  // ---- The list of claims and the self-tests on it ----
  // F.lin(a) adds the claim a = 0. This block reads no message and samples no challenge.
  {
    // The lookup's claims that contain mask entries only.
    F.section("layer"); // the lookup prefixes its own section names with it
    forestObjects(F, fo);
    // The norms' sums of squares. (F.section names the claims that follow, for tests: ../tests/claims_report.h.)
    F.section("norms: the product layer of the squares");
    layerObject(F, wpl);
    F.section("norms: the layer's two end values are one vector's");
    F.lin(affSlot(wpl.sL) - affSlot(wpl.sR));
    F.section("norms: the squared vector is the stream's limbs under gamma");
    {
      const Ten &txs = y.T_("xs");
      // The loop below covers both tensors' limbs: each needs residualStreamLimbs limbs, and NSl = 2 residualStreamLimbs - 1.
      if ((u64)txs.nl != y.xb.nl || (int)y.xb.nl != residualStreamLimbs || NSl != 2 * y.xb.nl - 1) refuse("plan-tensor-layout");
      const u64 tI = tabAt("xbI@Z_V"), tA = tabAt("af@Z_V");
      const vector<Ex> e = hiTab(wpl.r, la, lNI + lPd - ls);
      Aff a = affSlot(wpl.sL);
      for (u64 i = 0; i < NI; i++)
        for (u64 kh = 0; kh < d / S2; kh++)
          for (u64 l = 0; l < y.xb.nl; l++)
            affAdd(a,
                   i == 0 ? U3at(tI, tenCell(y.xb, l, 0, 0) / S2 + kh)
                          : U3at(tA, tenCell(txs.at(), l, i - 1, 0) / S2 + kh),
                   eZ() - e[(i * Pd) / S2 + kh] * gp[l]);
      F.lin(a);
    }
    {
      /* TSC(P_V)'s claims, then each instance's equations on its digit row at P_V
         (vf_norm_digit_rows.h): docs/format/11-2-feed-forward-segment.md, "The claims, group by
         group", groups 3 to 6. */
      auto sh = [&](u64 i, u64 k) {
        return F.shown(tV.sEnd + (u32)(NOl * i + k));
      };
      auto is = [&](u64 i, u64 k, const Aff &form) {
        F.lin(affSlot(tV.sEnd + (u32)(NOl * i + k)) - form);
      };
      {
        Aff a = affSlot(tV.sCl);
        for (u64 i = 0; i < NPl * NI; i++) affAdd(a, affSlot(sPv + (u32)i), eZ() - omV[i]);
        F.section("norms: the batch of the row products");
        F.lin(a);
      }
      Ex S0 = eZ();
      if (residualStreamLimbs == 5) {
        if (NOl != 4) refuse("plan-tensor-layout");
        for (u64 i = 0; i < NI; i++) S0 = S0 + ledProducts5(&omV[7 * i], sh(i, 0), sh(i, 1), sh(i, 2), sh(i, 3));
        F.section("norms: the token sum-check of the row products");
        F.lin(pscRow(F, tV.p, S0));
        F.section("norms: the shown row values are reads");
        for (u64 i = 0; i < NI; i++) {
          const Row5<Aff> r = row5At("r_tV", i);
          is(i, 0, r.a), is(i, 1, r.b), is(i, 2, r.Rlo), is(i, 3, r.Rhi);
        }
        vector<Row5<Aff>> rP;
        for (u64 i = 0; i < NI; i++) rP.push_back(row5At("P_V", i));
        F.section("norms: the carries of the convolution sums");
        for (u64 i = 0; i < NI; i++)
          for (int s = 0; s < 9; s++) F.lin(ledCarry5<Aff>(rP[i], s, affSlot(sW + (u32)(9 * i + (u64)s))));
        F.section("norms: the row ledger's lines (root, reciprocal)");
        for (u64 i = 0; i < NI; i++) {
          const int n = (int)(i % 2); // 0: the layer's first norm; 1: its second
          Aff Pr[7], out[8];
          for (u32 j = 0; j < 7; j++) Pr[j] = affSlot(sPv + (u32)(7 * i) + j);
          ledLines5<Aff>(rP[i], Pr, eI((i64)h.eps[n]), affC(eI(1ll << (kl(g * Lam + i / NN, n ? LM2 : LM1) + 1))),
                              out);
          for (const Aff &o : out) F.lin(o);
        }
      } else {
        if (NOl != 3) refuse("plan-tensor-layout");
        for (u64 i = 0; i < NI; i++)
          S0 = S0 + omV[3 * i] * sh(i, 0) * sh(i, 0) + omV[3 * i + 1] * sh(i, 1) * sh(i, 0) + omV[3 * i + 2] * sh(i, 2) * sh(i, 0);
        F.section("norms: the token sum-check of the row products");
        F.lin(pscRow(F, tV.p, S0));
        F.section("norms: the shown row values are reads");
        for (u64 i = 0; i < NI; i++) {
          const RowA r = rowAt("r_tV", i);
          is(i, 0, r.sig), is(i, 1, r.Rlo), is(i, 2, r.Rhi);
        }
        vector<RowA> rP;
        for (u64 i = 0; i < NI; i++) rP.push_back(rowAt("P_V", i));
        F.section("norms: the carries of the convolution sums");
        for (u64 i = 0; i < NI; i++)
          for (int s = 0; s < 7; s++)
            F.lin(affSlot(sW + (u32)(7 * i + s)) + rP[i].c[s] - rP[i].Dg[s] - rP[i].c[s + 1] * eI(256));
        F.section("norms: the row ledger's lines (root, reciprocal)");
        for (u64 i = 0; i < NI; i++) {
          const RowA &r = rP[i];
          const int n = (int)(i % 2); // 0: the layer's first norm; 1: its second
          const u32 Mn = kl(g * Lam + i / NN, n ? LM2 : LM1);
          F.lin(r.V + affC(eI((i64)h.eps[n])) - affSlot(sPv + (u32)(3 * i)) - r.l1);
          F.lin(r.l1 + r.h1 - r.sig * eI(2));
          F.lin(affSlot(sPv + (u32)(3 * i + 2)) - r.p1);
          const Aff nM = affC(eI(1ll << (Mn + 1)));
          F.lin(nM + r.sig - r.s0 - affSlot(sPv + (u32)(3 * i + 1)) * eI(2) - r.p1 * eI(1 << 25) - r.rr * eI(2));
          F.lin(r.rr + r.rq - r.sig + affC(eO()));
        }
      }
    }
    // What TSC(zt) has shown must equal the tables' values at rt. (Its claim is in cstart, as KT.)
    if (hasT) {
      F.section("the second token sum-check's shown values are reads (reciprocals, scores, selectors)");
      for (u64 i = 0; i < NN * Lam; i++) F.lin(affSlot(tT.sEnd + (u32)i) - rowR("rt", i));
      if (p3) {
        for (u64 lm = 0; lm < LamM; lm++) F.lin(affSlot(sSc + (u32)lm) - shA("sc", "rt", lm, 0));
        for (u64 lm = 0; lm < LamM; lm++)
          for (u64 e = 0; e < NE; e++) F.lin(affSlot(sSel + (u32)(lm * NE + e)) - shA("sel", "rt", lm, e));
      }
    }
    // The product layers for hh = s u ("sl and u" in section names). Their end values: -s, -u.
    F.section("gated product: the product layer");
    layerObject(F, hd);
    F.section("gated product: the layer's end values are sl and u");
    {
      const vector<Ex> e = hiTab(hd.r, la, ll + lPf - ls);
      const u64 t = tabAt("af@Z_a");
      const Ten &ts = y.T_("s"), &tu = y.T_("u");
      Aff a = affSlot(hd.sL), b = affSlot(hd.sR);
      for (u64 lam = 0; lam < Lam; lam++)
        for (u64 kh = 0; kh < f / S2; kh++) {
          const Ex w = e[lam * (Pf / S2) + kh];
          affAdd(a, longA(t, ts, lam, kh), w);
          affAdd(b, longA(t, tu, lam, kh), w);
        }
      F.lin(a);
      F.lin(b);
    }
    if (p3) {
      F.section("gated product of the routed experts: the product layer");
      layerObject(F, hd2);
      F.section("gated product of the routed experts: the layer's end values are sl and u");
      const vector<Ex> e = hiTab(hd2.r, la, ll - 1 + lfe - ls);
      const u64 t = tabAt("af@Z_b");
      const Ten &ts = y.T_("s"), &tu = y.T_("u");
      Aff a = affSlot(hd2.sL), b = affSlot(hd2.sR);
      for (u64 lm = 0; lm < LamM; lm++)
        for (u64 kh = 0; kh < fe / S2; kh++) {
          const Ex w = e[lm * (fe / S2) + kh];
          affAdd(a, longA(t, ts, 2 * lm + 1, fe / S2 + kh), w);
          affAdd(b, longA(t, tu, 2 * lm + 1, fe / S2 + kh), w);
        }
      F.lin(a);
      F.lin(b);
    }
    /* No call has a separate join claim: every join is inside JF. With routed experts, the one-hot
     * check for each mixture-of-experts layer, at zt: the selector's entries sum to 1. */
    if (p3) {
      F.section("the selector's entries sum to 1");
      const string z = "zt";
      for (u64 lm = 0; lm < LamM; lm++) {
        Aff a = affC(eZ() - eO());
        for (u64 e = 0; e < NE; e++) affAdd(a, shA("sel", z, lm, e), eO());
        F.lin(a);
      }
    }
    // The lookup's three claims on the multiplicity bits ("planes"): b's tables TB_0, TB_1, TP.
    {
      const u64 tB[2] = {tabAt("b-bits-0"), tabAt("b-bits-1")}, tP = tabAt("b-pin");
      auto col = [&](u64 x) {
        if (x >= y.kbm) refuse(F.G_ + "internal-g-table-index");
        return x;
      };
      F.section("layer");
      forestPlaneLins(
          F, fo, (int)h.lmb, [&](int e, u64 x) { return U3at(tB[e], col(x)); },
          [&](u64 x) { return U3at(tP, col(x)); });
    }
    // Format: docs/format/11-2-feed-forward-segment.md, "The claims, group by group", group 12.
    if (hasE) {
      const vector<Ex> e = eqTab(tkh.data(), lPd - ls);
      const u64 t = tabAt("xbI@Z_E");
      Aff a = affSlot(sVE) * (eZ() - eO());
      for (u64 kh = 0; kh < d / S2; kh++) affAdd(a, xbA(t, kh), e[kh]);
      F.section("the embedding's value is a read of the first stream");
      F.lin(a);
    }
    // The range check's claims. First: its matrix was sampled once; two lookups ran (fo, its own).
    if (tl.foldDrawn != 1) refuse(F.G_ + "internal-g-rows-booked");
    if (1 + tl.groupsRun() != 2) refuse(F.G_ + "internal-g-calls-booked");
    F.section("the smallness test");
    tl.objectsF(F, fun);
    F.section(string("layer, ") + FOREST_JOIN_SECTION + ", the second token sum-check and the inner products: the batched sum-check's one claim");
    F.lin(rowS);
    forestJoinAdded(fo); // the lookup's join, written out here, is cstart's first summand (JF)
    F.section("batched sum-check: A0 and B0 are reads");
    F.lin(affSlot(sA0) - Alin);
    F.lin(affSlot(sB0) - Blin);
    // The last claim: the merge's.
    F.section("the merge of the weight claims");
    wsForkLin(F, wo);
    // Self-test 1 (of this program, not the proof): every slot is in a claim, except the merged
    // value's (wo.sFv), whose claim leaves the sub-proof. It guards hiding and must stay: without
    // it, the program cp_prove's run of cp_verify may miss an unmasked end value. F.LS[j], slot j's
    // batched coefficient, is 0 if no claim has it or, on an honest proof, with chance about 1/|E|.
    for (u32 j = 0; j < hp.used; j++)
      if (isZ(F.LS[j]) != (j == wo.sFv)) refuse(F.G_ + "internal-g-free-slot");
    // Self-test 2: the number of claims listed is hp.nLin.
    if ((size_t)F.nClaim != hp.nLin) refuse(F.G_ + "internal-g-claim-count");
    tl.funsHeld(F, fun); // `fun` must be as at `closeFuns`; runs only after all claims are listed
    tl.selfTest(F, hp.nU);
    tl.coefficients(F);
    tl.rkSelfTest(F); // the range check's tiling self-test; limits at `tileSelfTestF`, vf_smallness_test_members.inc
  } // ---- the end of the list of claims ----
  doorH(F, T2, tr, tl.equationsLaidF());
  {
    if (F.nClaim != F.planClaimsF) refuse(F.G_ + "internal-f-draws");
    F.kapAllTakenF();
    tl.foldAllTakenF(F);
  }
  // The results: wo, for wsPending() (vf_weights_pending_claim.h), and the log's line.
  forkOut[g] = std::move(wo);
  forkRan[g] = 1;
  char ln[256];
  snprintf(
      ln, sizeof ln,
      "read: sub-proof G_%llu: %zu lookup calls in one forest, 2 more in the smallness test's, %zu matrices, %u claims listed (tested at the chain's end), %u slots  %.3f s\n",
      (unsigned long long)g, NC, (size_t)tl.NM, F.nClaim, F.ns, now() - tf0);
  forkLine[g] = ln;
};
