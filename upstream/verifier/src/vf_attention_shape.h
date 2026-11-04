// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The shape of one layer group's attention sub-proof (`fork` in identifiers). From public
// parameters (ACfg), AFork::shape() derives sizes, lookup tables and calls, and tests no-overflow
// conditions against wrap-around in the field. tenCell() to headFeat() serve all sub-proofs.
#pragma once
#include "vf_common.h"
#include <algorithm>
#include <map>
// Lookup table FC (cp_verify.cpp) clamps queries q, keys kx and values v to -CLAMP .. CLAMP.
// The score bound Smax, the causal mask's margin Om and condition N6 use it too, to agree with FC.
static constexpr int CLAMP = 2047;
// A tensor's features (T cells each) in a committed matrix: nl limbs of cnt layers of J, from off.
struct TenAt {
  u64 off = 0, nl = 0, cnt = 0, J = 0;
};
// Returns the matrix's feature that stores limb l of the tensor's feature j at layer i.
static inline u64 tenCell(const TenAt &X, u64 l, u64 i, u64 j) {
  return X.off + (l * X.cnt + i) * X.J + j;
}
// A block is 2^ls adjacent features (the callers' S2) of T = 2^lt cells each; a long table has one
// entry per block. Returns the row (index in the block) for token t of a feature with low bits c2.
static inline size_t blockRow(size_t c2, size_t t, size_t T) {
  return c2 * T + t;
}
// The coefficients of a block's rows: a long table's entry is the sum of coefficient times cell.
static vector<Ex> blockRows(const vector<Ex> &low, const vector<Ex> &tok) {
  vector<Ex> v(low.size() * tok.size());
  for (size_t c2 = 0; c2 < low.size(); c2++)
    for (size_t t = 0; t < tok.size(); t++) v[blockRow(c2, t, tok.size())] = low[c2] * tok[t];
  return v;
}
// A block's point has la = ls + lt coordinates: those of the feature's low bits, then the token's.
static vector<Ex> longPoint(const Ex *low, int ls, const vector<Ex> &tok) {
  vector<Ex> p(low, low + ls);
  p.insert(p.end(), tok.begin(), tok.end());
  return p;
}
// A block's coefficients: row i gets the multilinear equality polynomial eq(p, i).
static vector<Ex> longPhi(const Ex *p, int ls, int lt) {
  return blockRows(eqTab(p, ls), eqTab(p + ls, lt));
}
// A point split in two: lo, its last k coordinates; hi, the nhi before them.
struct PointHighLow {
  const Ex *hi = nullptr, *lo = nullptr;
  int nhi = 0;
};
static PointHighLow cutLast(const Ex *z, int n, int k) {
  if (k < 0 || k > n) refuse("plan-tensor-layout");
  PointHighLow c;
  c.hi = z;
  c.lo = z + (n - k);
  c.nhi = n - k;
  return c;
}
// The last k coordinates of z: with k = la, its block's point.
static vector<Ex> lastOf(const vector<Ex> &z, int k) {
  const PointHighLow c = cutLast(z.data(), (int)z.size(), k);
  return vector<Ex>(c.lo, c.lo + k);
}
// A long table's coefficients at z: eq over the coordinates before its last la. Rejects unless
// there are exactly `want`.
static vector<Ex> hiTab(const vector<Ex> &z, int la, int want) {
  const PointHighLow c = cutLast(z.data(), (int)z.size(), la);
  if (c.nhi != want) refuse("plan-tensor-layout");
  return eqTab(c.hi, c.nhi);
}
// Returns head hd's first feature among a layer's J, with dh features a head.
static inline u64 headFeat(const TenAt &X, u64 hd, u64 dh) {
  if (!dh || X.J % dh || hd >= X.J / dh) refuse("plan-tensor-layout");
  return hd * dh;
}
// T = 2^lt tokens; a block (see blockRow()) has 2^la rows. Ty: the cell bound that shape()
// assumes. `lines`: the `callA` lines of docs/format/11-3-attention-segment.md, "The lookup calls".
struct ACfg {
  int lt = 0, lB = 0, lgk = 0, ldh = 0, ll = 0, S = 0, rx = 0, kFD = 0, lmS = 0, la = 0, lmb = 0, nmin = 0;
  u64 H = 0;
  vector<u32> sh; // one shift per layer; a softmax weight is Tt[(maximum - score) >> sh] or 0
  u64 wcH = 0, wcO = 0, wcQ = 0;
  u64 NAs = 1;
  u128 Ty = 0;
  bool oInQa = true;
  struct Line {
    string name, kind;
    vector<int> segs;
  };
  vector<Line> lines;
};
enum { sRM, sRD, sFM, sFD, sWL, sWH, sIX };
struct APiece {
  int seg = 0;
  u64 lam = 0, a = 0;
  int b = 0;
  u64 off = 0;
};
struct ACall {
  Call c;
  string kind;
  u32 sh = 0;
  vector<int> segs; // Its lookup segments (sRM to sIX), each once, in list order: the call's units.
  vector<APiece> pcs;
  u64 tot = 0; // The sum of the pieces' sizes: the keys from tot on are the padding keys.
  vector<u64> tau;
};
// eq of z with the n-bit index idx. Coordinate 0 is the top bit, as in idOf() and LEof() (x <= y).
static Ex eqAt(const Ex *z, int n, u64 idx) {
  Ex p = eO();
  for (int v = 0; v < n; v++) p = p * (((idx >> (n - 1 - v)) & 1) ? z[v] : eO() - z[v]);
  return p;
}
static Ex eqPts(const Ex *x, const Ex *y, int n) {
  Ex p = eO();
  for (int v = 0; v < n; v++) p = p * (x[v] * y[v] + (eO() - x[v]) * (eO() - y[v]));
  return p;
}
static Ex onesOf(const Ex *z, int n) {
  Ex p = eO();
  for (int v = 0; v < n; v++) p = p * z[v];
  return p;
}
static Ex idOf(const Ex *z, int n) {
  Ex s = eZ();
  for (int v = 0; v < n; v++) s = s + eS(z[v], 1ull << (n - 1 - v));
  return s;
}
// Source for the multilinear extension of a comparison of two integers given by their bits: Arun, Setty, Thaler,
// "Jolt: SNARKs for Virtual Machines via Lookups", Cryptology ePrint Archive 2023/1217, section 4.2.2, equations (5)
// to (7). There the strict comparison, as a sum over the bit positions; here the non-strict one, by a recurrence from
// the lowest bit.
static Ex LEof(const Ex *x, const Ex *y, int n) {
  Ex le = eO();
  for (int v = n - 1; v >= 0; v--) le = (eO() - x[v]) * y[v] + (x[v] * y[v] + (eO() - x[v]) * (eO() - y[v])) * le;
  return le;
}
// What shape() derives from cf: docs/format/11-3-attention-segment.md, "Numbers" and "Notation",
// where B is Bt.
struct AFork {
  ACfg cf;
  u64 T = 0, B = 0, nT = 0, hP = 0, G = 0, HA = 0, dh = 0, dq = 0, dkv = 0, Lam = 0, u0 = 0;
  int lnT = 0, nM = 0, nD = 0, ls = 0; // ls = cf.la - cf.lt: a block is 2^ls adjacent features
  vector<u64> Tt;
  vector<ACall> calls;
  // kbA, the number of columns of matrix bA (multiplicities' bits), depends on the calls' lookup
  // forest (their product trees), so aforkShapeHid() (vf_attention_check.h) sets it after shape().
  u64 kAs = 0, kbA = 0;
  u64 featH = 0, featO = 0, featQ = 0, fillH = 0, fillO = 0, fillQ = 0;
  u64 oMr = 0, oZr = 0, oIb = 0, oRh = 0, oRhp = 0, oQ = 0, oK = 0, oV = 0, oO = 0;
  TenAt atQ, atK, atV, atO; // q, kx, v, o in qa; cp_verify.cpp rejects unless qa's owner agrees
  struct ShD {
    u64 C = 0, rho = 0, nHD = 0, Om = 0;
  };
  u64 Smax = 0;
  ShD shd(u32 sh) const {
    ShD d;
    d.C = (u64)u0 << sh;
    d.rho = 65536 - (1ull << sh);
    d.nHD = (4 * Smax + 1 + d.rho - 1) / d.rho;
    d.Om = 2 * Smax + d.C;
    return d;
  }
  int segB(int s) const {
    switch (s) {
    case sRM:
    case sFM:
      return nM;
    case sRD:
    case sFD:
      return nD;
    case sWL:
      return 1 + cf.lgk + cf.lt + cf.ldh;
    case sWH:
      return cf.lgk + cf.lt + cf.ldh;
    default:
      return cf.lgk + cf.lt;
    }
  }
  // Reads no proof. Not tested: cf.NAs and cf.wcH, which are divisors here, must not be 0.
  void shape() {
    const int lt = cf.lt;
    // lt and lB are tested before their use as shift counts. lgk, ldh and ll are shift counts here
    // without a range test: a caller must bound them, as layout() in cp_verify.cpp does.
    if (lt < 2 || lt > 16 || cf.lB < 0 || cf.lB > lt - 2) refuse("af-shape-tiles");
    T = 1ull << lt;
    B = 1ull << cf.lB;
    lnT = lt - cf.lB;
    nT = 1ull << lnT;
    hP = nT / 2;
    G = 1ull << cf.lgk;
    if (!cf.H || cf.H % G) refuse("af-shape-heads");
    HA = cf.H / G;
    dh = 1ull << cf.ldh;
    dq = cf.H * dh;
    dkv = G * dh;
    Lam = 1ull << cf.ll;
    if (cf.sh.size() != Lam) refuse("af-shape-sh");
    ls = cf.la - lt;
    if (ls < 0 || ls > cf.ldh || cf.la > 16) refuse("af-shape-la");
    // Z, a row's sum of softmax weights, is at most T 2^S; remainders mod 2 Z must fit three bytes.
    if (cf.S < 1 || cf.S > 14 || cf.rx < 1 || cf.rx > 12 || lt + cf.S + 1 > 24) refuse("af-policy-softmax");
    Tt.clear();
    // Rounding is to nearest, ties to even: nearbyint under the default rounding mode.
    for (u64 u = 0;; u++) {
      double v = nearbyint(ldexp(1.0, cf.S) * exp(-(double)u / ldexp(1.0, cf.rx)));
      if (v == 0) {
        u0 = u;
        break;
      }
      Tt.push_back((u64)v);
      if (u > (1u << 20)) refuse("af-policy-softmax");
    }
    Smax = dh * CLAMP * CLAMP;
    nM = cf.lgk + 2 * lt - 1;
    nD = cf.lgk + lt + cf.lB;
    if (cf.lmS > nD || cf.lmS < 1) refuse("af-shape-lmS");
    const u128 P2 = (u128)1 << 60, p = ((u128)1 << 61) - 1, Ty = cf.Ty, Ty2 = Ty * 257, Ty4 = Ty * 16843009ull;
    vector<u32> shs(cf.sh);
    std::sort(shs.begin(), shs.end());
    shs.erase(std::unique(shs.begin(), shs.end()), shs.end());
    // kFD, tested before its use as a shift count, becomes Call::k: forestShape() needs k >= 5.
    if (cf.kFD < 8 || cf.kFD > 16) refuse("af-policy-kF");
    for (u32 sh : shs) {
      if (sh < 4 || sh > 12) refuse("af-policy-sh");
      ShD d = shd(sh);
      const u64 rows = 1ull << cf.kFD;
      if (rows < u0 + d.nHD || rows - u0 - d.nHD > d.nHD) refuse("af-policy-kF");
      if (d.C + 65536 > (1u << 24)) refuse("af-policy-C");
      if (!((u128)6 * Smax + 2 * d.C + 3 * 65536 + ((u128)1 << 32) +
                (((u128)1 << 33) * (Ty2 + 128 + ((u128)1 << cf.S))) <
            p))
        refuse("af-no-wrap-N3");
      if (!((u128)6 * Smax + 2 * d.C + 3 * 65536 < ((u128)1 << 32))) refuse("af-no-wrap-N4");
    }
    if (!(Ty4 + Smax < P2)) refuse("af-no-wrap-N1");
    if (!((((u128)1 << 40) * Ty + ((u128)1 << 20) * (Ty + 128) + Ty + 128 + ((u128)1 << 41) < p) &&
          Ty + 128 + 256 < ((u128)1 << 20)))
      refuse("af-no-wrap-N2");
    const u128 TS = (u128)T << cf.S;
    if (!(TS + Ty4 < P2)) refuse("af-no-wrap-N5");
    // N6: 2 TS CLAMP is for v, a clamp's output; 2 TS Ty2 for o, a rounded mean of values of v
    // (no clamp); 2 TS + 1 for WL's second key, which is 2 Z - 1 less the first key's remainder.
    if (!(2 * TS * CLAMP + TS + 2 * TS * Ty2 + 2 * TS + 1 + 65536 * (Ty + 128) + 65536 < P2)) refuse("af-no-wrap-N6");
    if (!(((u128)1 << (16 - lt)) * 2 * T < P2)) refuse("af-no-wrap-N9");
    // The expanded list. If sh has several values, a KR or KFD call's name ends in its sh.
    calls.clear();
    auto mk = [&](const string &n, const string &kind, u32 sh, const vector<int> &segs) {
      ACall c;
      c.c.name = n;
      c.kind = kind;
      c.sh = sh;
      c.segs = segs;
      calls.push_back(c);
    };
    const bool one = shs.size() == 1;
    auto sfx = [&](u32 sh) {
      return one ? string() : std::to_string(sh);
    };
    const vector<ACfg::Line> &ln = cf.lines;
    // Every line must have at least one lookup segment, so that no call's part of the join (the
    // claim that ties the keys to the committed cells: forestJoin(), vf_lookup_argument.h) is empty.
    for (auto &l : ln)
      if (l.segs.empty()) refuse("af-policy-call-list");
    // The rule for a valid list: docs/format/11-3-attention-segment.md, "The lookup calls".
    {
      int seen[7] = {0};
      for (auto &l : ln)
        for (int s : l.segs) {
          if (s < 0 || s > 6) refuse("af-policy-call-list");
          const bool ok = (l.kind == "KR" && (s == sRM || s == sRD)) || (l.kind == "KFD" && (s == sFM || s == sFD)) ||
                          (l.kind == "R16" && (s == sWL || s == sIX)) || (l.kind == "WH" && s == sWH);
          if (!ok || seen[s]++) refuse("af-policy-call-list");
        }
      for (int s = 0; s < 7; s++)
        if (!seen[s]) refuse("af-policy-call-list");
    }
    for (auto &l : ln) {
      if (l.kind[0] == 'K')
        for (u32 sh : shs) mk(l.name + sfx(sh), l.kind, sh, l.segs);
      else
        mk(l.name, l.kind, 0, l.segs);
    }
    if (cf.nmin < 10) refuse("af-policy-nmin");
    // A column of ao (wcO T cells) or qa (wcQ T) is a positive integer number of blocks.
    if ((1ull << cf.la) > cf.wcO * T || (1ull << cf.la) > cf.wcQ * T || cf.wcO % (1ull << ls) || cf.wcQ % (1ull << ls))
      refuse("af-shape-la");
    // At most twelve expanded calls. cp_verify.cpp makes the same test (`policy-call-list-A`).
    if (calls.size() > 12) refuse("af-policy-call-list");
    for (size_t q = 0; q < calls.size(); q++) {
      ACall &c = calls[q];
      const bool bySh = c.kind[0] == 'K';
      // A piece: 2^b keys of one lookup segment, for layer lam and query head a of each key/value
      // head. Placement: by decreasing b, then by (segment as listed, lam, a); so 2^b divides off.
      for (int s : c.segs)
        for (u64 lam = 0; lam < Lam; lam++) {
          if (bySh && cf.sh[lam] != c.sh) continue;
          for (u64 a = 0; a < HA; a++) c.pcs.push_back({s, lam, a, segB(s), 0});
        }
      std::stable_sort(c.pcs.begin(), c.pcs.end(), [](const APiece &x, const APiece &y) { return x.b > y.b; });
      u64 off = 0;
      for (auto &pc : c.pcs) {
        pc.off = off;
        off += 1ull << pc.b;
      }
      c.tot = off;
      Call &k = c.c;
      k.n = clog2(off);
      if (k.n < cf.nmin) k.n = cf.nmin;
      k.kappa = 5;
      // iota = n - dT depends on all calls' lookup forest: aforkShapeHid() (vf_attention_check.h) sets it.
      if (k.n < 10 || k.n > 44) refuse("af-shape-call-" + k.name);
      if (c.kind == "KR") {
        k.k = 16;
        k.fn = 2;
        c.tau.resize(65536);
        for (u64 t = 0; t < 65536; t++) c.tau[t] = (t & 255) + ((t >> 8) << 20) + ((u64)(t >= (1ull << c.sh)) << 40);
      } else if (c.kind == "WH") {
        k.k = 16;
        k.fn = 2;
        c.tau.resize(65536);
        for (u64 t = 0; t < 65536; t++) c.tau[t] = (t & 255) + ((t >> 8) << 20);
      } else if (c.kind == "R16") {
        k.k = 16;
        k.fn = 0;
      } else {
        ShD d = shd(c.sh);
        const u64 nH = d.nHD;
        k.k = cf.kFD;
        k.fn = 1;
        k.t0 = u0;
        c.tau.resize(1ull << k.k);
        for (u64 t = 0; t < c.tau.size(); t++)
          c.tau[t] = t < u0        ? (t << c.sh) + (Tt[t] << 33)
                     : t < u0 + nH ? d.C - (1ull << c.sh) + d.rho * (t - u0) + (1ull << 32)
                                   : d.C - (1ull << c.sh) + d.rho * (t - u0 - nH) + d.rho / 2 + (1ull << 32);
      }
    }
    // kAs, the columns of matrix `as`: docs/format/11-3-attention-segment.md, "The arrays".
    {
      const u64 cells = 5 * Lam * HA * ((1ull << nM) + (1ull << nD)), unit = cf.NAs << cf.lmS;
      kAs = (cells + unit - 1) / unit * cf.NAs;
    }
    oMr = 0;
    oZr = 4 * Lam * cf.H;
    oIb = 8 * Lam * cf.H;
    featH = oIb + Lam * lt * cf.H;
    oRh = 0;
    oRhp = Lam * dq;
    featO = 2 * Lam * dq; // ao has rh, rhp: third byte less 128 of remainder rem and 2 Z - 1 - rem
    oQ = 0;
    oK = 2 * Lam * dq;
    oV = oK + 2 * Lam * dkv;
    oO = oV + 2 * Lam * dkv;
    featQ = oO + (cf.oInQa ? 2 * Lam * dq : 0);
    atQ = {oQ, 2, Lam, dq};
    atK = {oK, 2, Lam, dkv};
    atV = {oV, 2, Lam, dkv};
    atO = {oO, 2, Lam, dq};
    fillH = (featH + cf.wcH - 1) / cf.wcH * cf.wcH;
    fillO = (featO + cf.wcO - 1) / cf.wcO * cf.wcO;
    fillQ = (featQ + cf.wcQ - 1) / cf.wcQ * cf.wcQ;
  }
  void bindTau() {
    for (auto &c : calls) c.c.tau = c.tau.empty() ? nullptr : &c.tau;
  }
  size_t tableWords() const {
    size_t nKF = 0;
    for (auto &c : calls)
      if (c.kind == "KFD") nKF += c.segs.size();
    // The tables' size in words. Its terms, in order: `as`, ah, ao, qa, bA.
    return WPE * (3 * kAs + (nKF + 6) * fillH + (fillO * T >> cf.la) + (2 * nKF + 5) * (fillQ * T >> cf.la) + 3 * kbA);
  }
  // The score tile (I, J), J < I, in slot sg of slot row P. Not tested: P < hP and sg < nT - 1.
  void cover(u64 P, u64 sg, u64 &I, u64 &J) const {
    if (sg < P) {
      I = P;
      J = sg;
    } else {
      I = nT - 1 - P;
      J = sg - P;
    }
  }
};
// The bound on a clamped cell in the no-overflow conditions of the layer and output-head sub-proofs
// (cp_verify.cpp), derived from CLAMP so that table FC and those conditions cannot disagree.
static constexpr u32 CLAMPED_CELL_BOUND = CLAMP + 1;
