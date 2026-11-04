// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Checks H, the sub-proof (`fork` in names) of the output head: the final norm, the logits, and the
   acceptance policy on the checked tokens. Included in `runStage` (cp_verify.cpp), whose locals it
   uses, after vf_layers_check.inc, which declares forkOut. `refuse` (vf_common.h) rejects; a name with
   "internal-" is meant for a fault of this program, not of the proof. For a hidden value x the
   prover sends ("shows", `F.shown`) x plus a public factor times a mask entry, whose number is x's
   message slot (except fv', the merge's end value: vf_weights_merge.h). Sumchecks are masked sumchecks
   (PSC: vf_sumcheck_psc.h, vf_sumcheck_token.h), whose rounds test no equation. Any other check is a claim (`F.lin`):
   an affine form in hidden values and table entries must be 0. T = 2^lt token positions, d channels
   padded to Pd = 2^lPd, V (`Vv`) vocabulary entries. A feature is T cells, one per position; a
   block is S2 = 2^ls features, 2^la cells. hOn[t]: t is in the request's range ("covered").
   hOnC[t]: covered and t + 1 < T (not the next token y); only these hRC positions have a cost. */
// Specified in docs/format/11-4-output-head.md ("11-4" below). Its UH, Mg, act, chi, "checked",
// "pair" and "part" are U, D, a, F.theta, covered, place and table here; its gc, gb, gu, gk, gz, gs
// and BIG are c, b, u, k, z, zs and BIGH. Order of rounds: prover/head/plan_rounds.h. tr (`Tr`)
// wraps the channel of the request's first hash chain, which reads the proof.
auto runForkHidH2 = [&]() {
  const double tf0 = now();
  const HPlanH2 &hp = PH2;
  if (!currentSegment || !currentSegment->V || !currentSegment->check) refuse("plan-link");
  /* y, the next token. The chain receives it, unmasked, at the start of this sub-proof's part of
     the chain (`link` in identifiers), before that part's first commitments (blocks/iop/req.h).
     Here its form is tested, and it becomes the token checked at the last position, if covered.
     `check_proof` (blocks/frame/check.h) compares it with the token given to the client. */
  u32 yTok = 0;
  {
    const std::vector<blk::E> &plain = currentSegment->check->frame().plain();
    if (plain.size() != 1 || plain[0].c[1].v || plain[0].c[2].v || plain[0].c[0].v >= Vv) refuse("header-y");
    yTok = (u32)plain[0].c[0].v;
  }
  vector<u32> win = hWin; // win[t]: the token checked at the covered position t
  if (hOn[T - 1]) win[T - 1] = yTok;
  HFork F;
  F.G_ = "H:";
  F.trees = hp.Fp.trees;
  fillL(F); // allocates the `HRow::L` vectors, all zero: a plan keeps none
  Tail2 T2 = hp.Tp;
  HTailF2 tl = hp.tl;
  HL L = HLy;
  // A lookup call of kind FC (the clamp) reads the lookup table y.tauFC; `y` in code is a layout.
  for (auto &c : L.calls)
    if (c.kind == "FC") c.c.tau = &y.tauFC;
  // A call of kind FE keeps its pointer to HLy.tauFE, the layout's table of exponentials.
  const size_t NC = L.calls.size();
  const u32 iU3 = (u32)tl.ix.U[2];
  // Nothing is hashed at the start: the request's shape and layout are parameters of the chain's
  // first query (blocks/frame/check.h), and y has come in a prover's message on the chain.
  Tr tr{currentSegment->V->ch};
  {
    tl.runT = &T2;
    F.slots(hp.ns);

    const HPlan2 &O = hidPlans2[NGr - 1];
    if (O.tl.ix.TX.empty() || O.tl.kTX.back() != y.kxb) refuse("internal-h-owner-TX");
  }
  u32 slot = 0;
  // Reads n hidden values sent outside a sumcheck, as one message. Returns the first one's slot.
  auto send = [&](size_t n) {
    return F.claimsIn(tr, slot, (u32)n);
  };
  // The `callH` list's lookup calls run as one lookup forest. forestPad() sets each call's alpha
  // and key point zk = (zj | zt): zj over the columns of its lookup segments, zt over positions.
  vector<Call *> cl1;
  for (auto &c : L.calls) cl1.push_back(&c.c);
  HForest g1 = forestPad(F, tr, cl1, L.fs.pub, slot, LIDB);
  /* Every call's zk ends in g1.zx, which has dT >= la coordinates ("forest-depth-H"). So all calls
     share the last la, the long point Z (`Zl`), and with it zt. Table "hf@Z" relies on it. */
  const vector<Ex> Zl = lastOf(g1.zx, la);
  for (auto &c : L.calls)
    for (int i = 0; i < la; i++)
      if (!(c.c.zk[c.c.zk.size() - (size_t)la + (size_t)i] == Zl[(size_t)i])) refuse(F.G_ + "internal-h-merged-point");
  // P_V: the point over the positions at which the norm's equations are tested.
  const vector<Ex> PV = trPoint(tr, RoundName::ZERO, (size_t)lt);
  auto callOf = [&](const string &sn) -> CallD & {
    for (auto &c : L.calls)
      for (auto &s : c.segs)
        if (s.name == sn) return c;
    refuse("internal-segment-" + sn);
  };
  auto zj = [&](CallD &c) {
    return c.c.zk.data();
  };
  auto zt = [&](CallD &c) {
    return c.c.zk.data() + c.lJ;
  };
  // A plain sum over an index of n bits is run with the factor eq(z, index) at z = (1/2, .., 1/2),
  // which is 2^-n for every value of the index. eq: the equality polynomial (`eqTab`, vf_common.h).
  const Ex half = eInv(eI(2));
  Ex plainPd = eO(); // 2^-lPd
  for (int i = 0; i < lPd; i++) plainPd = plainPd * half;
  // The norm's sum of squares, a product layer: 11-4, step 4 and "Group 2".
  const u32 sW = send((size_t)ledNS()); // convolution sums: 7, or with five limbs 9 (vf_norm_digit_rows.h)
  const vector<Ex> gp = trBatch(tr, (size_t)ledNS()); // gam^0 .. : one power for each sum just sent
  ProductLevel wpl;
  {
    Aff C;
    for (int s = 0; s < ledNS(); s++) affAdd(C, affSlot(sW + (u32)s), gp[s] * plainPd);
    vector<Ex> rho((size_t)lPd, half);
    rho.insert(rho.end(), PV.begin(), PV.end());
    wpl = readProductLevelRounds(F, tr, slot, muVH, rho, C, orderDefault(muVH));
  }
  // Products of two numbers of the norm's row of digits (`led` in names), summed with eq(P_V, t).
  const u32 sPv = send((size_t)ledNP()); // 3, or with five limbs 7
  const vector<Ex> omV = trBatch(tr, (size_t)ledNP()); // the powers of one batching element
  // Their token sumcheck (TSC, vf_sumcheck_token.h). End values, summand: 11-4, EV_j and "Group 3".
  TokenSumcheckState tV = tscIn(F, tr, slot, PV.data(), lt, (u32)ledNO(), 2);
  {
    auto sh = [&](u32 j) { return F.shown(tV.sEnd + j); };
    tV.rowU = pscRow(F, tV.p,
                       residualStreamLimbs == 5 ? ledProducts5(omV.data(), sh(0), sh(1), sh(2), sh(3))
                                 : omV[0] * sh(0) * sh(0) + omV[1] * sh(1) * sh(0) + omV[2] * sh(2) * sh(0));
  }
  // TSC of the call with lookup segment NF. Shown: R and the norm's input combined over channels.
  CallD &cN = callOf("NF"), &cH = callOf("HD0");
  TokenSumcheckState tN = tscIn(F, tr, slot, zt(cN), lt, 2, 2);
  tN.rowU = pscRow(F, tN.p, F.shown(tN.sEnd) * F.shown(tN.sEnd + 1));
  /* The TSC of HD0's call. Cl_H is the logits' share of the margins in HD0's keys: sum_t
     sum_(v < V) eq(zt, t) on_t ew_HD0[v] (lam_t[win_t] - lam_t[v]) = sum_t eq(zt, t) sum_k a[t][k]
     U[k][t]. on_t = hOn[t], lam_t: the logits, a: the norm's output, Wh: the output matrix, U[k][t]
     = on_t (wx Wh[k][win_t] - sum_v ew_HD0[v] Wh[k][v]). Column win_t depends on t, so the sum does
     not split into a factor on t and one on k: after the rounds over t, ending at r_H = tH.r, come
     lam_s = lPd rounds over the channels k, summed plainly, so the first slot holds 2^-lPd Cl_H.
     rs[v]: the challenge for bit v of k. (11-4 calls the first slot's value Cl_H.) Shown last: a at
     (rs, r_H), and v_H, U's value there. */
  TokenSumcheckState tH = tscRounds(F, tr, slot, zt(cH), lt);
  HStack st;
  st.lam = hp.lamS;
  st.rs.assign((size_t)st.lam, eZ());
  for (int i = 0; i < st.lam; i++)
    st.rs[(size_t)(st.lam - 1 - i)] = roundIn(F, tr, slot, RoundForm::EQ2, tH.p, half);
  tH.sEnd = pscEnds(F, tr, slot, tH.p, 2, 2);
  tH.rowU = pscRow(F, tH.p, F.shown(tH.sEnd) * F.shown(tH.sEnd + 1));
  const u32 sV = tH.sEnd + 1;
  // The prover commits to U3, the evaluation tables: entry (c, g) = sum_i phi[i] M[g L + i][c].
  rootInH(F, tr, {(int)iU3});
  auto lastLa = [&](const vector<Ex> &z) {
    return lastOf(z, la);
  };
  const vector<Ex> ZV = lastLa(wpl.r);
  const vector<Ex> YN = longPoint(cutLast(zj(cN), cN.lJ, ls).lo, ls, tN.r);
  if (YN.size() != (size_t)la || la != ls + lt) refuse("plan-tensor-layout");
  vector<HFun> fun;
  u64 p = 0;
  // hf: cells by channel or vocabulary entry. hd, xbL: the norm's output, input. hr: digit rows.
  std::map<string, u64> tab;
  std::map<u64, u64> tabLen; // start of one addFun call's tables -> their total length (all parts)
  const size_t iHF = 0, iHD = h.NHf, iHR = iHD + 1, iBH = iHR + 1, iXL = iBH + 1;
  const u64 La = 1ull << la;
  // Adds one HFun per part matrix tl.mats[M0 + q] at running offset p; tab[nm] keeps the start.
  auto addFun = [&](size_t M0, u64 parts, u64 Lf, const vector<Ex> &phi, const string &nm) {
    tl.funOpen();
    const u64 first = p;
    for (u64 q = 0; q < parts; q++) {
      const TMat &M = tl.mats[M0 + q];
      HFun fn;
      fn.M = (int)(M0 + q);
      fn.L = Lf;
      fn.phi = phi;
      fn.tree = 2;
      fn.at = p;
      fn.name = F.G_ + nm + (parts > 1 ? "." + std::to_string(q) : "");
      fun.push_back(std::move(fn));
      p += M.k * (M.m / Lf);
    }
    if (p == first) refuse(F.G_ + "internal-h-table-index"); // an empty table would share its start with the next
    tab[nm] = first;
    tabLen[first] = p - first;
  };
  // The tables, in the order of 11-4, "The parts of `U3`".
  addFun(iHF, h.NHf, La, longPhi(Zl.data(), ls, lt), "hf@Z");
  // phiC[t] is eq(zt, t) at the positions with a cost and 0 elsewhere, with the zt of HD0's call.
  // ERC, rhoCv and hf@phiZ: 11-4, ERC, rhoC and part 2.
  Ex ERC = eZ();
  vector<Ex> phiC = eqTab(zt(cH), lt), rhoCv(T, eZ());
  for (u64 t = 0; t < T; t++) {
    if (hOnC[t]) {
      ERC = ERC + phiC[t];
      rhoCv[t] = eO();
    } else
      phiC[t] = eZ();
  }
  {
    vector<Ex> phiZ(La);
    for (u64 i = 0; i < La; i++) phiZ[i] = phiC[i & (T - 1)];
    addFun(iHF, h.NHf, La, phiZ, "hf@phiZ");
  }
  addFun(iXL, 1, La, longPhi(ZV.data(), ls, lt), "xbL@Z_V");
  addFun(iXL, 1, La, longPhi(YN.data(), ls, lt), "xbL@Y_N");
  addFun(iHD, 1, T, eqTab(zt(callOf("CNF")), lt), "hd@zt_FC");
  addFun(iHD, 1, T, eqTab(tH.r.data(), lt), "hd@r_H");
  addFun(iHR, 1, T, eqTab(zt(callOf("ROW")), lt), "hr@zt_ROW");
  addFun(iHR, 1, T, eqTab(PV.data(), lt), "hr@P_V");
  addFun(iHR, 1, T, eqTab(tV.r.data(), lt), "hr@r_tV");
  addFun(iHR, 1, T, eqTab(tN.r.data(), lt), "hr@rt_N");
  addFun(iHR, 1, T, phiC, "hr@phiC");
  addFun(iHR, 1, T, rhoCv, "hr@rhoC");
  const u64 mb = 1ull << h.lmbH;
  // bH holds the bits of the forest's multiplicities.
  addFun(iBH, 1, mb, forestPhiB(g1, (int)h.lmbH, 0), "bH-bits-0");
  addFun(iBH, 1, mb, forestPhiB(g1, (int)h.lmbH, 1), "bH-bits-1");
  addFun(iBH, 1, mb, forestPhiP((int)h.lmbH), "bH-pin");
  if (p != hp.nU[2]) refuse("internal-W-length-H");
  tl.closeFuns(fun);
  auto tabAt = [&](const string &n) {
    auto it = tab.find(n);
    if (it == tab.end()) refuse("internal-table-" + n);
    return it->second;
  };
  // Except in tl.objectsF(), the claims read U3 only through this function: entry i of the tables
  // that one call of addFun has added from offset t on, all parts together. A self-check sees an i
  // at or past their length and a t at which no such call began, but not an i that falls in another
  // segment, limb or digit of the same tables, or a t that starts other tables than those meant.
  auto U3at = [&](u64 t, u64 i) {
    const auto it = tabLen.find(t);
    if (it == tabLen.end() || i >= it->second) refuse(F.G_ + "internal-h-table-index");
    return uEnt(F, iU3, t + i);
  };
  // The messages of the approximate range check (`tl`, vf_smallness_test.h; printed: "smallness test").
  tl.transcript(F, tr, slot);
  // ewOf: w[x] = eq(zj, column x of segment sn). ewh: one coefficient for each block of S2 columns.
  auto ewOf = [&](const string &sn) {
    CallD &c = callOf(sn);
    const vector<Ex> e = eqTab(zj(c), c.lJ);
    Seg &s = c.seg(sn);
    vector<Ex> w(s.n);
    for (u64 x = 0; x < s.n; x++) w[x] = e[s.pos[x]];
    return w;
  };
  auto ewh = [&](const string &sn) {
    CallD &c = callOf(sn);
    const PointHighLow k = cutLast(zj(c), c.lJ, ls);
    const vector<Ex> e = eqTab(k.hi, k.nhi);
    Seg &s = c.seg(sn);
    vector<Ex> w(s.n / S2);
    for (u64 xh = 0; xh < w.size(); xh++) w[xh] = e[s.pos[S2 * xh] / S2];
    return w;
  };
  auto sumE = [&](const vector<Ex> &v) {
    Ex s = eZ();
    for (auto &x : v) s = s + x;
    return s;
  };
  // wx of segment HD0, HD1 or HD2: the sum of its columns' coefficients over the entries v < V.
  auto wxOf = [&](const vector<Ex> &ew) {
    Ex s = eZ();
    for (u64 v = 0; v < Vv; v++) s = s + ew[v];
    return s;
  };
  // The sum of eq(P, t) over the covered positions, for a point P.
  auto onSum = [&](const Ex *P) {
    const vector<Ex> e = eqTab(P, lt);
    Ex s = eZ();
    for (u64 t = 0; t < T; t++)
      if (hOn[t]) s = s + e[t];
    return s;
  };
  /* The margins' public terms, which are subtracted in HD0's key. A place (v, t) is real if v < V
     and position t is covered. The margin D: 11-4, "What the head computes, in integers", item 6.
     Each function returns the sum over the real places of ew[v] et[t] times its term, for the
     columns' coefficients ew and the positions' et at a point. placeTermH: [v < win_t], which makes
     the first of equal logits the largest. publicTermH: Pub, which is 0 in this format. */
  auto placeTermH = [&](const vector<Ex> &ew, const vector<Ex> &et) {
    vector<Ex> pre(Vv, eZ()); // pre[v] = the sum of ew[x] over x < v
    for (u64 v = 1; v < Vv; v++) pre[v] = pre[v - 1] + ew[v - 1];
    Ex s = eZ();
    for (u64 t = 0; t < T; t++)
      if (hOn[t]) s = s + et[t] * pre[win[t]];
    return s;
  };
  auto publicTermH = [&](const vector<Ex> &, const vector<Ex> &) {
    return eZ(); // Pub is 0 in this format
  };
  const vector<Ex> ew0 = ewOf("HD0");
  const Ex wx0 = wxOf(ew0);
  /* The one claim on the weights (`WsClaim`, vf_weights_layout.h), on unit X = 13, the output matrix Wh.
     Its factors x (AV) and kk (HV, pt): 11-4, "The weight claim". */
  vector<WsClaim> claims(1);
  {
    WsClaim &c = claims[0];
    size_t ui = wsLay.U.size();
    for (size_t i = 0; i < wsLay.U.size(); i++)
      if (wsLay.U[i].X == 13) ui = i;
    // The head's unit is the last one.
    if (ui + 1 != wsLay.U.size()) refuse(F.G_ + "internal-claim-unit");
    c.ui = (u32)ui;
    for (auto &G : wsLay.U[ui].g) {
      WsFac x;
      if (!strcmp(G.name, "x")) {
        const vector<Ex> eH = eqTab(tH.r.data(), lt);
        const Ex sR = onSum(tH.r.data());
        x.AV.resize(Vv);
        for (u64 v = 0; v < Vv; v++) x.AV[v] = eZ() - sR * ew0[v];
        for (u64 t = 0; t < T; t++) {
          if (!hOn[t]) continue;
          // win[t] indexes x.AV. A token of Vv or more was rejected before ("token-range" in
          // runStage(), "header-y" above); the test is repeated here, at the point of use.
          if (win[t] >= Vv) refuse("header-y");
          x.AV[win[t]] = x.AV[win[t]] + wx0 * eH[t];
        }
      } else if (!strcmp(G.name, "kk")) {
        for (int i = 0; i < G.z; i++) x.pt.push_back(st.rs[G.z - 1 - i]);
        const vector<Ex> e = eqLow(st.rs, G.z, st.lam);
        for (u64 hi = 0; hi < G.o; hi++) x.HV.push_back(e[hi]);
      } else
        refuse(F.G_ + "internal-claim-factors");
      c.f.push_back(std::move(x));
    }
  }
  if (claims.size() != hp.NCL) refuse(F.G_ + "internal-claim-count");
  WsForkOut wo = wsForkEnd(F, tr, slot, wsLay, claims, {sV});
  if (slot != hp.nMsg) refuse(F.G_ + "internal-slot-count");
  // The sub-proof's last challenge, F.theta: its powers batch the claims that are listed below.
  F.planClaimsF = (u32)hp.nLin; // the plan's number of claims
  t2Batch(F, T2, tr, slot);
  { // ==== the list of claims and the self-tests on it ====
    // S_(s): the hidden value of slot s, as a form. pw(k) = 2^k. o1, o2: offsets of 1 and 2 digits.
    auto S_ = [&](u32 s) {
      return affSlot(s);
    };
    auto pw = [&](u32 k) {
      return eI(1ll << k);
    };
    const Ex o1 = eI(128), o2 = eI(32896);
    // The forest's claims that read mask entries only.
    forestObjects(F, g1);
    // Sl: the form SL of 11-4, "Three short forms".
    auto Sl = [&](const string &sn, u64 OFF, int nl, u64 J) {
      const vector<Ex> wh = ewh(sn);
      const u64 t = tabAt("hf@Z");
      Aff s;
      for (u64 xh = 0; xh < wh.size(); xh++) {
        u64 sc = 1;
        for (int l = 0; l < nl; l++, sc *= 256) affAdd(s, U3at(t, (OFF + (u64)l * J) / S2 + xh), eS(wh[xh], sc));
      }
      return s;
    };
    auto xbA = [&](u64 t, u64 kh) {
      Aff s;
      u64 sc = 1;
      for (u64 l = 0; l < y.xb.nl; l++, sc *= 256) affAdd(s, U3at(t, tenCell(y.xb, l, 0, 0) / S2 + kh), eI((i64)sc));
      return s;
    };
    auto sh2 = [&](u64 t, u64 j) {
      return U3at(t, j) + U3at(t, d + j) * eI(256);
    };
    struct RowAH {
      Aff Dg[74], V, c[8], s0, sig, l1, h1, Rlo, Rhi, R, p1, rr, rq;
    };
    // The norm's row of digits ("ledger"), four limbs, in table "hr@" + P; a cell is digit - 128.
    auto rowAt = [&](const string &P) {
      const u64 t = tabAt("hr@" + P);
      RowAH r;
      for (int j = 0; j < 74; j++) r.Dg[j] = U3at(t, (u64)j) + affC(eI(128));
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
    // The same row with five limbs: 99 cells.
    auto row5At = [&](const string &P) {
      const u64 t = tabAt("hr@" + P);
      return row5<Aff>([&](int j) { return U3at(t, (u64)j); });
    };
    // F.section names the next claims for a test's report (../tests/claims_report.h); no decision reads it.
    F.section("head: the final norm: the sum of squares");
    // Claims: wpl's own; its two end values are equal, as it sums squares; sL as read in xbL@Z_V.
    layerObject(F, wpl);
    F.lin(S_(wpl.sL) - S_(wpl.sR));
    {
      if ((int)y.xb.nl != residualStreamLimbs) refuse("plan-tensor-layout"); // the number of sums follows the number of limbs
      const vector<Ex> e = hiTab(wpl.r, la, lPd - ls);
      const u64 t = tabAt("xbL@Z_V");
      Aff b = S_(wpl.sL);
      for (u64 kh = 0; kh < d / S2; kh++)
        for (u64 l = 0; l < y.xb.nl; l++) affAdd(b, U3at(t, tenCell(y.xb, l, 0, 0) / S2 + kh), eZ() - e[kh] * gp[l]);
      F.lin(b);
    }
    F.section("head: the final norm: the row ledger's lines (root, reciprocal, carries)");
    /* tV: its sum Cl is the combination by omV of the products at sPv; the claim that its sumcheck
       leaves; its end values, read in the row at r_tV. Then the row's equations, read at P_V. */
    {
      Aff a = S_(tV.sCl);
      for (u32 i = 0; i < (u32)ledNP(); i++) affAdd(a, S_(sPv + i), eZ() - omV[i]);
      F.lin(a);
      F.lin(tV.rowU);
    }
    // Five limbs: four end values (a, b, Rlo, Rhi), nine carry equations, eight more (vf_norm_digit_rows.h).
    if (residualStreamLimbs == 5) {
      const Row5<Aff> r = row5At("r_tV");
      F.lin(S_(tV.sEnd) - r.a);
      F.lin(S_(tV.sEnd + 1) - r.b);
      F.lin(S_(tV.sEnd + 2) - r.Rlo);
      F.lin(S_(tV.sEnd + 3) - r.Rhi);
      const Row5<Aff> q = row5At("P_V");
      for (int s = 0; s < 9; s++) F.lin(ledCarry5<Aff>(q, s, S_(sW + (u32)s)));
      Aff Pr[7], out[8];
      for (u32 j = 0; j < 7; j++) Pr[j] = S_(sPv + j);
      ledLines5<Aff>(q, Pr, eI((i64)h.epsF), affC(eI(1ll << (h.MF + 1))), out);
      for (const Aff &o : out) F.lin(o);
    } else {
      const RowAH r = rowAt("r_tV");
      F.lin(S_(tV.sEnd) - r.sig);
      F.lin(S_(tV.sEnd + 1) - r.Rlo);
      F.lin(S_(tV.sEnd + 2) - r.Rhi);
      const RowAH q = rowAt("P_V");
      for (int s = 0; s < 7; s++) F.lin(S_(sW + (u32)s) + q.c[s] - q.Dg[s] - q.c[s + 1] * eI(256));
      F.lin(q.V + affC(eI((i64)h.epsF)) - S_(sPv) - q.l1);
      F.lin(q.l1 + q.h1 - q.sig * eI(2));
      F.lin(S_(sPv + 2) - q.p1);
      // 2^MF + floor(sigma / 2) = R sigma + rr, doubled. rr: the remainder. s0: sigma's low bit.
      const Aff nF = affC(eI(1ll << (h.MF + 1)));

      F.lin(nF + q.sig - q.s0 - S_(sPv + 1) * eI(2) - q.p1 * eI(1 << 25) - q.rr * eI(2));
      F.lin(q.rr + q.rq - q.sig + affC(eO()));
    }
    F.section("head: the final norm: the stream times the reciprocal");
    // tN: the claim that its sumcheck leaves; its two end values.
    {
      const vector<Ex> wh = ewh("NF");
      const u64 t = tabAt("xbL@Y_N");
      Aff SX;
      for (u64 kh = 0; kh < d / S2; kh++) affAdd(SX, xbA(t, kh), wh[kh]);
      F.lin(tN.rowU);
      F.lin(S_(tN.sEnd) - (residualStreamLimbs == 5 ? row5At("rt_N").R : rowAt("rt_N").R));
      F.lin(S_(tN.sEnd + 1) - SX);
    }
    /* Number i of a position's six hidden numbers (i = 0 .. 5: c, b, u, k, z, zs; LEDG,
       vf_norm_digit_rows.h), summed over the positions with the coefficient vector P, "phiC" or "rhoC". A
       cell holds its digit minus 128, so each digit gets 128 times wsum, the sum of P's entries. */
    auto gapAt = [&](const string &P, int i, Ex wsum) {
      const u64 t = tabAt("hr@" + P), o = (u64)ledNR() + (u64)LEDG[i][0];
      Aff s;
      u64 sc = 1;
      for (int l = 0; l < LEDG[i][1]; l++, sc *= 256)
        affAdd(s, U3at(t, o + (u64)l) + affC(eS(wsum, 128)), eI((i64)sc));
      return s;
    };
    // ex is its EXC cells read at radix 256 plus offE: a cell holds a digit of ex - 1, minus 128.
    const u64 offE = 128 * (((1ull << (8 * EXC)) - 1) / 255) + 1;
    // Lookup segment sn: sum_(x, t) ew[x] eq(zt, t) key(x, t); keyForm adds NF's product term Cl_N.
    auto LINE = [&](const string &sn) -> Aff {
      const vector<Ex> ew = ewOf(sn);
      const Ex w = sumE(ew);
      if (sn == "NF")
        return affC(w * (pw(h.KF - 1) - pw(16) * o2)) - Sl(sn, L.oAw, 2, d) * pw(h.KF) - Sl(sn, L.oRa, 2, d) * pw(16);
      if (sn == "NFH") return (affC(o2 * w) + Sl(sn, L.oRa, 2, d)) * pw(32 - h.KF);
      /* HD0, HD1, HD2: the three digits of a margin D = K_2 + 2^sh K_1 + 2^(sh + 15) K_0
         (sh: h.digSh). HD2's key K_2 is the low byte (cell d2), HD1's K_1 the middle digit (cells
         d1). HD0's key K_0, the top digit, has no cell: it is D minus the two lower digits, times
         iw, the inverse of 2^(sh + 15) in the field. A real place has the coefficient
         ew[v] eq(zt, t); all together have wR. Sl reads d1 and d2 at every place, but the digits'
         offsets and D's terms are added at the real places only: at any other place the three keys
         are d1, d2 and -(2^sh d1 + d2) iw, which are in their lookup tables only if all three
         are 0. Of D's terms, Cl_H is the one in the logits. c and b are read under phiC, which has
         the positions' coefficients, so they are multiplied by wx, the columns' sum. */
      if (sn == "HD0" || sn == "HD1" || sn == "HD2") {
        const Ex wx = wxOf(ew), wR = wx * onSum(zt(callOf(sn)));
        if (sn == "HD1") return affC(o2 * wR) + Sl(sn, L.oD1, 2, L.JV);
        if (sn == "HD2") return affC(o1 * wR) + Sl(sn, L.oD2, 1, L.JV);
        const vector<Ex> et = eqTab(zt(cH), lt);
        const Ex iw = eInv(pw(h.digSh + 15));
        return (S_(tH.sCl) * pw((u32)lPd) + (gapAt("phiC", 0, ERC) + gapAt("phiC", 1, ERC) * eI((i64)BIGH)) * wx -
                affC(placeTermH(ew, et) + publicTermH(ew, et)) -
                (affC(o2 * wR) + Sl(sn, L.oD1, 2, L.JV)) * pw(h.digSh) - (affC(o1 * wR) + Sl(sn, L.oD2, 1, L.JV))) *
               iw;
      }
      /* HE0, HE1: the key at (v, t) is K_1 + 2^16 ex, looked up in the table FE, whose row u is
         u + 2^16 EX[u] (tauFE, cp_verify.cpp). Which offset gets which weight: 11-4, "The key
         forms". At a place that is not real d1 is 0, so ex must be EX[0]. HE1's column x is entry
         JVa + x. */
      if (sn == "HE0" || sn == "HE1") {
        const u64 v0 = sn == "HE1" ? L.JVa : 0;
        Ex wx = eZ();
        for (u64 x = 0; x < ew.size() && v0 + x < Vv; x++) wx = wx + ew[x];
        return affC(o2 * wx * onSum(zt(callOf(sn)))) + Sl(sn, L.oD1 + v0, 2, L.JV) +
               (affC(eS(w, offE)) + Sl(sn, L.oE + v0, EXC, L.JV)) * pw(16);
      }
      if (sn == "ROW") {
        Aff a;
        const u64 t = tabAt("hr@zt_ROW");
        for (int idx = 0; idx < ledNRH(); idx++) {
          const u64 mu = idx < ledNR() ? ledMu(idx) : 1; // the head's additional digits are whole bytes
          affAdd(a, U3at(t, (u64)idx) + affC(eI(128)), eS(ew[idx], mu));
        }
        return a;
      }
      if (sn == "CNF") {
        Aff b;
        const u64 t = tabAt("hd@zt_FC");
        for (u64 x = 0; x < d; x++) affAdd(b, sh2(t, x), ew[x]);
        // The key aw + 2^15 + 2^28 (a + 2^11) in table FC is meant to show a = clamp(aw; 2,047).
        return Sl(sn, L.oAw, 2, d) + b * pw(28) + affC(w * eI((1ll << 15) + (1ll << 39)));
      }
      refuse("internal-line-" + sn);
    };
    /* keyForm returns KF_c, the form for call c's key vector at zk_c: the padding key times padW,
       the sum of eq(zj, j) over the columns j that no segment lists; LINE of each segment of c; and
       Cl_N (tN.sCl) for cN, the call that has NF. In the columns v >= V the keys of HD0, HD1 and
       HD2 must be 0 and those of HE0 and HE1 the table's row 0: each is its call's padding key. The
       padding is described twice: by its number in each tree (g1.pub, computed in cp_verify.cpp
       from CallD::used and V), and by the columns counted here, T keys each: the unlisted ones and
       those of entries v >= V. `forestPaddingLaid` compares the two. Both come from the lists
       s.pos (`place` fills CallD::used from them), so a slip inside a LINE form is not caught. */
    vector<u64> laid((size_t)g1.NT, 0);
    auto keyForm = [&](CallD &c, size_t q) {
      const vector<Ex> e = eqTab(zj(c), c.lJ);
      vector<char> weighed(c.J, 0); // by column: 0 unlisted, 1 listed, 2 listed with entry v >= V
      Aff lines;
      for (auto &s : c.segs) {
        lines = lines + LINE(s.name);
        const bool entries = s.name == "HD0" || s.name == "HD1" || s.name == "HD2" || s.name == "HE0" || s.name == "HE1";
        const u64 v0 = s.name == "HE1" ? L.JVa : 0;
        for (u64 x = 0; x < s.n; x++) {
          if (weighed.at(s.pos[x])) refuse("internal-h-join-place");
          weighed[s.pos[x]] = entries && v0 + x >= Vv ? 2 : 1;
        }
      }
      Ex padW = eZ();
      for (u64 j = 0; j < c.J; j++) {
        if (!weighed[j]) padW = padW + e[j];
        if (weighed[j] != 1) laid.at((u64)g1.first[q] + (j >> (g1.dT - lt))) += T;
      }
      Aff v = affC(eS(padW, c.c.tauAt(c.c.t0)));
      if (&c == &cN) v = v + S_(tN.sCl);
      return v + lines;
    };
    // The forest's join (forestJoin(), vf_lookup_argument.h) on all calls' key forms, in the list's order.
    {
      vector<Aff> KF;
      for (size_t q = 0; q < L.calls.size(); q++) KF.push_back(keyForm(L.calls[q], q));
      forestPaddingLaid(g1, laid);
      F.section(FOREST_JOIN_SECTION);
      F.lin(forestJoin(g1, KF));
    }
    F.section("head: the four lines on the hidden numbers (the acceptance policy)");
    /* All six hidden numbers are at least 0 only by their digits' lookup (ROW); no equation wraps,
       by no-wrap-H-E2, -E3, -E4, -Z (cp_verify.cpp). (B2), (B3): plain sums over the hRC positions:
       c + C b + u adds up to theta hRC, and b + z to cap (to 0 if hRC = 0). (B4): c + k = C at each
       such position, so c <= C; it is tested at HD0's zt, drawn after the cells were committed. By
       its margins and (Z), a position with b = 0 has a cost of at most c; one with b > 0 is charged
       at least C. So the clipped costs sum to at most theta hRC, and at most cap costs exceed C. */
    {
      const Ex R_ = eI((i64)hRC);
      F.lin(gapAt("rhoC", 0, R_) + gapAt("rhoC", 1, R_) * eI((i64)h.clipC) + gapAt("rhoC", 2, R_) -
            affC(eS(R_, h.theta)));
      F.lin(gapAt("rhoC", 1, R_) + gapAt("rhoC", 4, R_) - affC(eI(hRC ? (i64)h.cap : 0)));
      F.lin(gapAt("phiC", 0, ERC) + gapAt("phiC", 3, ERC) - affC(eS(ERC, h.clipC)));
      /* (Z), tested like (B4) at one point (see prover/head/plan_rounds.h): at each position with a
         cost, zs and the position's V table values ex add up to 2^F (F: h.famF), so the values add
         up to at most 2^F. EX[0] = 2^F and every row of the table is at least 1, so no middle digit
         K_1 of the position is 0: c + b BIGH is at least the logit gap plus 2^sh. With EX[u] >=
         2^F exp(-u 2^(sh - sT)) and 2^sh K_1 at most the margin, c + b BIGH is also at least the
         token's negative log-probability at the exponent sT (h.tabS), in units of the integer
         logits. Table hf@phiZ sums ex's cells over all JV columns; a column v >= V has ex = 2^F. */
      {
        const u64 te = tabAt("hf@phiZ");
        Aff a = gapAt("phiC", 5, ERC) + affC(eS(ERC, offE) * eI((i64)L.JV) - eS(ERC, (L.JV - Vv + 1) << h.famF));
        u64 sc = 1;
        for (u64 l = 0; l < EXC; l++, sc *= 256)
          for (u64 xh = 0; xh < L.JV / S2; xh++) affAdd(a, U3at(te, (L.oE + l * L.JV) / S2 + xh), eI((i64)sc));
        F.lin(a);
      }
    }
    // The forest's three claims on the bits of its multiplicities, read in bH's tables.
    {
      const u64 tB[2] = {tabAt("bH-bits-0"), tabAt("bH-bits-1")}, tP = tabAt("bH-pin");
      auto col = [&](u64 t, u64 x) {
        if (x >= L.kb) refuse(F.G_ + "internal-h-table-index");
        return U3at(t, x);
      };
      forestPlaneLins(
          F, g1, (int)h.lmbH, [&](int e, u64 x) { return col(tB[e], x); }, [&](u64 x) { return col(tP, x); });
    }
    // The range check's claims.
    F.section("head: the smallness test");
    tl.objectsF(F, fun);
    F.section("head: the logits against the normalized activations and the weights");
    /* tH: the claim that its sumcheck leaves. Its first end value is Alin = sum_(kk < d)
       eq(rs, kk) af[kk], where af[kk], two limbs of table "hd@r_H", is the norm's output at (r_H,
       channel kk). The second is v_H, the weight claim's value, from which the merge starts. */
    {
      const vector<Ex> eq = eqLow(st.rs, 0, st.lam);
      const u64 t = tabAt("hd@r_H");
      Aff Alin;
      for (u64 kk = 0; kk < d; kk++) affAdd(Alin, sh2(t, kk), eq[kk]);
      F.lin(tH.rowU);
      F.lin(S_(tH.sEnd) - Alin);
    }
    F.section("head: the merge of the claims on the weights");
    // The merge's claim (vf_weights_merge.h), the last, uses its end value fv' as sent; forkOut takes fv'.
    wsForkLin(F, wo);
    /* Self-tests of this list of claims, not of the proof. Every message slot but that of fv' is in
       a claim. This test guards hiding: do not remove it. Only through a claim can cp_verify notice an
       end value that was sent without its mask. cp_prove tests no finished proof, so whoever hands a
       proof on must have verified it. */
    for (u32 j = 0; j < (u32)hp.nMsg; j++)
      if (isZ(F.LS[j]) != (j == wo.sFv)) refuse(F.G_ + "internal-h-free-slot");
    if (F.nClaim != hp.nLin) refuse(F.G_ + "internal-h-claim-count");
    if (tl.foldDrawn != 1) refuse(F.G_ + "internal-h-rows-booked");
    // Two lookups ran: this sub-proof's forest and the range check's.
    if (1 + tl.groupsRun() != 2) refuse(F.G_ + "internal-h-calls-booked");
    tl.funsHeld(F, fun); // Runs after the whole list of claims.
    tl.selfTest(F, hp.nU);
    tl.coefficients(F);
    /* `tl.selfTest`: no entry of U3 and no column that a virtual row is meant to read has column
       weight (L) 0; it does not read the row weights (coef), which `tl.coefficients` then sets for
       doorH. `tl.rkSelfTest` runs `tileSelfTestF`; vf_smallness_test_members.inc states what it cannot detect. */
    tl.rkSelfTest(F);
  }
  // ---- the end of the list of claims ----
  // doorH gives the chain the batched equation, tested at the chain's end, and closes the `link`.
  doorH(F, T2, tr, tl.equationsLaidF());
  {
    if (F.nClaim != F.planClaimsF) refuse(F.G_ + "internal-f-draws");
    F.kapAllTakenF();
    tl.foldAllTakenF(F);
  }
  forkOut[NGr] = std::move(wo);
  char ln[320];
  snprintf(
      ln, sizeof ln,
      "read: sub-proof H: %zu lookup calls in one forest, 2 more in the smallness test's, %zu matrices, %u claims listed (tested at the chain's end), %u slots  "
      "%.3f s\n",
      NC, (size_t)tl.NM, F.nClaim, F.ns, now() - tf0);
  forkLine[NGr] = ln;
};
