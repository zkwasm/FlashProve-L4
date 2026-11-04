// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The layout of the committed weight matrix W (WsLay), and a sub-proof's claims about W.
   Format: docs/format/11-5-weights-in-the-table.md.
   Identifiers call a sub-proof `fork`. A layer sub-proof or the sub-proof of the output head has
   claims about the weights (WsClaim), each about one unit: the rows of W that store one layer's
   weights of one kind, or the embedding, or the output head. `wsForkEnd` runs the merge
   (verifier/src/vf_weights_merge.h), which batches the claims and fixes W's columns at one point
   rm; coefficients on W's rows remain. From these,
   `wsPending` (verifier/src/vf_weights_pending_claim.h) builds the pending claim, the one linear claim on W
   that the sub-proof leaves for the block LIN (blocks/iop/lin.h, blocks/frame/lin_frame.h), which
   proves such claims on the committed table that stores W.
   From verifier/src/vf_common.h: `Ex`, an element of the extension field; `eqTab`, which returns
   the 2^n values eq(z, c) for a point z of n coordinates; `refuse`, which ends the process, or
   throws, with a rejection name. */
#pragma once
#include "vf_subproof_engine.h"
#include "vf_weights_merge.h" // the merge's messages and its claim
#include <algorithm>

/* L: layers. d: residual width. dh: head dimension. dkv, dq: widths of the value projection's
   output and the output projection's input. f, fe: feed-forward widths, dense and expert.
   NE: routed experts. V: vocabulary size. eh: 1 iff W has the embedding and the output head.
   D: W's number of rows is rounded up to a multiple of D.
   cp_verify.cpp sets MOE to 1 iff NE > 0. Layer l is a mixture-of-experts (MoE) layer, and moe[l]
   is nonzero (`wsSetMoe` writes 1), iff MOE is 1 and l is odd; `WsLay::build` rejects any other moe. */
struct WsShape {
  u64 D = 0, L = 0, d = 0, dh = 0, dkv = 0, dq = 0, f = 0, fe = 0, NE = 0, V = 0;
  u32 eh = 0, MOE = 0;
  vector<u8> moe;
};
// s.L must be set before the call.
static void wsSetMoe(WsShape &s, bool MOE) {
  s.MOE = MOE ? 1 : 0;
  s.moe.assign(s.L <= 4096 ? s.L : 0, 0);
  for (u64 l = 0; l < s.moe.size(); l++) s.moe[l] = MOE && (l & 1);
}

// ---- layout ----
/* An index group: one index of a unit's tensor, with R values. Its "taken" bits become column bits
   of W; the rest goes to the row index.
   Class E: R = o 2^z with o odd, and an index is hi 2^z + lo. The lowest t bits of lo are taken
   (`high` false) or the highest t (`high` true).
   Class A: the whole index is taken (t = ceil(log2 R)) or none of it (t = 0).
   cb0: the group's lowest column bit. `left`: the number of index values left to the rows.
   On docs/format/11-5-weights-in-the-table.md class E is "low" (`high` false) or "high" (`high`
   true), R is M and `left` is lambda.
   `name`: "kk" input, "x" output, "e" expert, "hd" attention head, "cd" coordinate in it, "v"
   token. */
struct WsGrp {
  const char *name = "";
  u64 R = 0, o = 0, left = 0;
  char cls = 'E';
  bool high = false;
  int z = 0, t = 0, cb0 = 0;
  bool takenBit(int b) const { return high ? b >= z - t : b < t; }
  // the i-th taken bit, lowest first
  int takenAt(int i) const { return high ? z - t + i : i; }
  // A class A group with R = 1 has t = 0 and counts as not taken: its one element is a row factor.
  bool aTaken() const { return cls == 'A' && t > 0; }
};
// One unit's rows in W. X: the kind of weights (0 to 13, see WsLay::build()). l: the layer (L for
// 12, 13). nlw: digits per weight. ncb: the number of column bits that g takes.
struct WsUnit {
  u32 X = 0, l = 0;
  int nlw = 2, ncb = 0;
  vector<WsGrp> g;
  u64 rows = 0, off = 0;
};
// W has at most 2^WS_ROWS_MOST_LOG2 rows, a constant of the format; WsLay rejects more.
static constexpr u32 WS_ROWS_MOST_LOG2 = 22;
// W has m0 rows and 2^lk columns. A weight is stored as digits (WS_SECOND_DIGIT), one to a cell.
// layoutW() (prover/common/weight_layout.h) is the committer's and prover's counterpart;
// verifier/tests/weights_twin_test.cpp compares the two, on small shapes.
struct WsLay {
  int lk = 0;
  u64 m0 = 0;
  vector<WsUnit> U;
  // R must not be 0: the loop would not end.
  static WsGrp E_(const char *n, u64 R, bool high) {
    WsGrp g;
    g.name = n;
    g.R = R;
    g.cls = 'E';
    g.high = high;
    g.o = R;
    g.z = 0;
    while (!(g.o & 1)) {
      g.o >>= 1;
      g.z++;
    }
    return g;
  }
  static WsGrp A_(const char *n, u64 R) {
    WsGrp g;
    g.name = n;
    g.R = R;
    g.cls = 'A';
    return g;
  }
  // Appends a unit, from row m0 on, and sets t, cb0 and `left` of its groups:
  // docs/format/11-5-weights-in-the-table.md, "Column bits".
  void add(u32 X, u32 l, int nlw, vector<WsGrp> g) {
    WsUnit u;
    u.X = X;
    u.l = l;
    u.nlw = nlw;
    int left = lk, cb = 0;
    u128 rows = (u128)nlw;
    for (auto &x : g) {
      int t = 0;
      if (x.cls == 'E')
        t = std::min(x.z, left);
      else {
        const int c = clog2(x.R);
        t = c <= left ? c : 0;
      }
      x.t = t;
      x.cb0 = cb;
      left -= t;
      cb += t;
      x.left = x.cls == 'E' ? x.R >> t : x.aTaken() ? 1 : x.R;
      rows *= x.left;
      if (rows > ((u128)1 << WS_ROWS_MOST_LOG2)) refuse("w-shape-mW");
    }
    u.ncb = cb;
    u.g = std::move(g);
    u.rows = (u64)rows;
    u.off = m0;
    m0 += u.rows;
    if (m0 > (1ull << WS_ROWS_MOST_LOG2)) refuse("w-shape-mW");
    U.push_back(std::move(u));
  }
  // Checks every number that reaches a shift, a loop or a product below: it relies on no check by
  // its caller.
  void build(const WsShape &s, u32 lkW) {
    if (lkW < 1 || lkW > 20) refuse("w-shape-lk");
    // D is 2,048 in the standard parameter form and 32 in toy models.
    if (s.D != 2048 && s.D != 32) refuse("shape-D");
    if (!s.L || s.L > 4096 || s.moe.size() != s.L || s.MOE > 1) refuse("shape-words");
    for (u64 l = 0; l < s.L; l++)
      if ((s.moe[l] != 0) != (s.MOE && (l & 1))) refuse("internal-shape-moe");
    for (u64 v : {s.d, s.dh, s.dkv, s.dq, s.f})
      if (!v || v > (1u << 24)) refuse("shape-words");
    // Only MoE layers use fe and NE, so these are checked only if there is one (else NE can be 0).
    bool anyMoe = false;
    for (u8 m : s.moe) anyMoe = anyMoe || m;
    if (anyMoe)
      for (u64 v : {s.fe, s.NE})
        if (!v || v > (1u << 24)) refuse("shape-words");
    if (s.eh > 1 || (s.eh && (!s.V || s.V > (1u << 24)))) refuse("shape-words");
    lk = (int)lkW;
    m0 = 0;
    U.clear();
    for (u32 l = 0; l < s.L; l++) {
      /* X, with its short name in cp_verify.cpp. 0 qk: queries and keys, as 64 heads: the H query
         heads, the Gk key heads, then zeros (cp_verify.cpp rejects a shape with H + Gk above 64,
         "shape-heads"). 1 v: values. 2 o: attention's output. 3 g, 4 u: gating, up. 5 gs, 6 us:
         the shared expert's. 7 gr, 8 ur: the routed experts'. 9 dn: down (in a MoE layer the
         shared expert's). 10 dnr: the routed experts' down. 11 r: router. */
      add(0, l, 2, {A_("hd", 64), A_("cd", s.dh), E_("kk", s.d, false)});
      add(1, l, 2, {E_("kk", s.d, false), E_("x", s.dkv, true)});
      add(2, l, 2, {E_("kk", s.dq, false), E_("x", s.d, true)});
      if (!s.moe[l]) {
        add(3, l, 2, {E_("kk", s.d, false), E_("x", s.f, true)});
        add(4, l, 2, {E_("kk", s.d, false), E_("x", s.f, true)});
        add(9, l, 2, {E_("kk", s.f, false), E_("x", s.d, true)});
      } else {
        add(5, l, 2, {E_("kk", s.d, false), E_("x", s.fe, true)});
        add(6, l, 2, {E_("kk", s.d, false), E_("x", s.fe, true)});
        add(7, l, 2, {E_("kk", s.d, false), E_("e", s.NE, false), E_("x", s.fe, true)});
        add(8, l, 2, {E_("kk", s.d, false), E_("e", s.NE, false), E_("x", s.fe, true)});
        add(9, l, 2, {E_("kk", s.fe, false), E_("x", s.d, true)});
        add(10, l, 2, {E_("kk", s.fe, false), E_("e", s.NE, false), E_("x", s.d, true)});
        add(11, l, 2, {E_("kk", s.d, false), E_("x", s.NE, true)});
      }
    }
    // 12: the embedding. 13: the output head.
    if (s.eh) {
      add(12, (u32)s.L, 4, {A_("v", s.V), E_("x", s.d, true)});
      add(13, (u32)s.L, 2, {A_("x", s.V), E_("kk", s.d, false)});
    }
    m0 = (m0 + s.D - 1) / s.D * s.D;
    if (m0 > (1ull << WS_ROWS_MOST_LOG2)) refuse("w-shape-mW");
  }
};

// ---- claims ----
/* A claim states the sum over one unit's weights, each times a product of factors, one for each
   index group. The layout splits each group between columns and rows, so claim a's coefficient on
   column c of the unit's row y is gamma_a[y] eps_a[c]. */

static Ex wsEq1(Ex p, Ex x) {
  return p * x + (eO() - p) * (eO() - x);
}
/* A claim's factor for one index group: HV and pt for class E, AV for class A.
   Format: docs/format/11-5-weights-in-the-table.md, "From a segment's claims to a row on the
   table". */
struct WsFac {
  vector<Ex> HV, pt, AV;
};
// ui: index into WsLay::U. f: one factor per group of that unit.
struct WsClaim {
  u32 ui = 0;
  vector<WsFac> f;
};
/* A factor's part on the rows. Class E: HV, and pl, the coordinates of pt for the bits not
   taken, highest bit first. Class A: AV if not taken, else nothing. */
struct WsRowFac {
  vector<Ex> HV, pl, AV;
};
// What remains of claim a once the columns are fixed at rm: its coefficient on row y of unit ui is
// ka gamma_a[y].
struct WsRecClaim {
  u32 ui = 0;
  Ex ka = eZ();
  vector<WsRowFac> f;
};

static void wsClaimShape(const WsLay &lay, const WsClaim &c) {
  if (c.ui >= lay.U.size()) refuse("internal-claim-unit");
  const WsUnit &u = lay.U[c.ui];
  if (c.f.size() != u.g.size()) refuse("internal-claim-factors");
  for (size_t gi = 0; gi < u.g.size(); gi++) {
    const WsGrp &g = u.g[gi];
    const WsFac &x = c.f[gi];
    if (g.cls == 'E' ? (x.HV.size() != g.o || x.pt.size() != (size_t)g.z || !x.AV.empty())
                     : (x.AV.size() != g.R || !x.HV.empty() || !x.pt.empty()))
      refuse("internal-claim-factors");
  }
}
// Returns the multilinear extension of c's eps_a at rm: e_a in
// docs/format/11-5-weights-in-the-table.md, "From a segment's claims to a row on the table".
// Not checked: that c has passed wsClaimShape(), that rm has lk elements.
static Ex wsEpsT(const WsLay &lay, const WsClaim &c, const vector<Ex> &rm) {
  const WsUnit &u = lay.U[c.ui];
  const int lk = lay.lk;
  auto X = [&](int b) {
    return rm[lk - 1 - b];
  };
  Ex p = eO();
  for (size_t gi = 0; gi < u.g.size(); gi++) {
    const WsGrp &g = u.g[gi];
    const WsFac &x = c.f[gi];
    if (g.cls == 'E') {
      for (int i = 0; i < g.t; i++) p = p * wsEq1(x.pt[g.z - 1 - g.takenAt(i)], X(g.cb0 + i));
    } else if (g.aTaken()) {
      // Coordinate 0 of eqTab() belongs to the top bit of i, which is column bit cb0 + t - 1.
      vector<Ex> zz(g.t);
      for (int s = 0; s < g.t; s++) zz[s] = X(g.cb0 + g.t - 1 - s);
      const vector<Ex> e = eqTab(zz.data(), g.t);
      Ex s = eZ();
      for (u64 i = 0; i < g.R; i++) s = s + x.AV[i] * e[i];
      p = p * s;
    }
  }
  for (int b = u.ncb; b < lk; b++) p = p * (eO() - X(b));
  return p;
}
// Not checked: that c has passed wsClaimShape().
static WsRecClaim wsRowPart(const WsLay &lay, const WsClaim &c, Ex ka) {
  const WsUnit &u = lay.U[c.ui];
  WsRecClaim r;
  r.ui = c.ui;
  r.ka = ka;
  r.f.resize(u.g.size());
  for (size_t gi = 0; gi < u.g.size(); gi++) {
    const WsGrp &g = u.g[gi];
    if (g.cls == 'E') {
      r.f[gi].HV = c.f[gi].HV;
      for (int b = g.z - 1; b >= 0; b--)
        if (!g.takenBit(b)) r.f[gi].pl.push_back(c.f[gi].pt[g.z - 1 - b]);
    } else if (!g.aTaken())
      r.f[gi].AV = c.f[gi].AV;
  }
  return r;
}
// A weight is the sum over j of WS_SECOND_DIGIT^j times its digit j.
static constexpr u32 WS_SECOND_DIGIT = 256;
// Adds wgt * ka * gamma_a[y] to Gam[u.off + y] for each row y of the claim's unit u. gamma_a[y] is
// WS_SECOND_DIGIT^j times the product of the row factors at y. y is a mixed-radix number: most
// significant first, the digit number j and then the groups in reverse order, each with the part
// of its index left to the rows. Not checked: c.ui, the sizes of c's factors.
static void wsGammaAdd(vector<Ex> &Gam, const WsLay &lay, const WsRecClaim &c, Ex wgt) {
  const WsUnit &u = lay.U[c.ui];
  vector<Ex> v(u.nlw);
  {
    Ex s = wgt * c.ka;
    for (int l = 0; l < u.nlw; l++) {
      v[l] = s;
      s = eS(s, WS_SECOND_DIGIT);
    }
  }
  for (size_t gi = u.g.size(); gi-- > 0;) {
    const WsGrp &g = u.g[gi];
    const WsRowFac &x = c.f[gi];
    vector<Ex> rf(g.left);
    if (g.cls == 'E') {
      const int zl = g.z - g.t;
      const vector<Ex> e = eqTab(x.pl.data(), zl);
      for (u64 hi = 0; hi < g.o; hi++)
        for (u64 lo = 0; lo < (1ull << zl); lo++) rf[(hi << zl) + lo] = x.HV[hi] * e[lo];
    } else if (g.aTaken())
      rf[0] = eO();
    else
      for (u64 i = 0; i < g.R; i++) rf[i] = x.AV[i];
    vector<Ex> n2(v.size() * rf.size());
    for (size_t i = 0; i < v.size(); i++)
      for (size_t q = 0; q < rf.size(); q++) n2[i * rf.size() + q] = v[i] * rf[q];
    v.swap(n2);
  }
  if (v.size() != u.rows || u.off + u.rows > Gam.size()) refuse("internal-gamma-rows");
  for (u64 y = 0; y < u.rows; y++) Gam[u.off + y] = Gam[u.off + y] + v[y];
}


// ---- the end of a sub-proof ----
// rec: one entry for each claim.
struct WsForkOut : WsMerged {
  vector<WsRecClaim> rec;
};
// Runs the merge and sets rec[a].ka to the merge's coefficient rho[a] times wsEpsT() at rm. sv[a]:
// the slot of claim a's value v_a. Slots number a sub-proof's sent scalars, and a form (`Aff`,
// verifier/src/vf_subproof_engine.h) refers to a scalar by its slot. The caller must still call wsForkLin(),
// which adds the merge's own claim.
static WsForkOut wsForkEnd(HFork &F, Tr &tr, u32 &slot, const WsLay &lay, const vector<WsClaim> &cl,
                           const vector<u32> &sv) {
  if (cl.empty() || cl.size() > 4096 || sv.size() != cl.size()) refuse(F.G_ + "internal-claim-count");
  for (auto &c : cl) wsClaimShape(lay, c);
  vector<Ex> rho;
  WsForkOut o;
  static_cast<WsMerged &>(o) = wsForkMerge(F, tr, slot, lay.lk, sv, rho);
  for (size_t a = 0; a < cl.size(); a++) o.rec.push_back(wsRowPart(lay, cl[a], rho[a] * wsEpsT(lay, cl[a], o.rm)));
  return o;
}
