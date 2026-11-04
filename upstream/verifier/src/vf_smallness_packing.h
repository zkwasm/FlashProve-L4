// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The approximate range check (`tail` in identifiers) is meant to show that every cell, a committed
// integer of a matrix, is small. Here: packing, placement and the tie, the linear claim that
// relates packed cells to the check's random combinations of columns. A symbol, one field element,
// packs up to three cells at radix B = 2^20: docs/format/11-1-machinery.md, "The array of a
// matrix". A symbol tree (kind 'P' in HFork::trees, vf_subproof_engine.h) is the array of a
// matrix's symbols, not a Merkle tree. No prover source includes this file, but the program
// `cp_commit` links cp_verify.cpp, and so this file, for cp_with_configured().
// vf_common.h and the prover's sources share headers of blocks/.
#pragma once
#include "vf_common.h"
// ---- constants ----
// By (f1) below, packing is one-to-one on cells |x| < 2 beta, beta = 2^LBZ. p = FIELD_P < 2^64.
static constexpr int LRAD = 20;
// Cell rows per symbol. Four break (f1): (4 beta - 2) B^3 > 2^78.
static constexpr int LANES = 3;
static constexpr u64 KCAP = 8191; // cap on a matrix's columns k; rejection name "hid-k-range"
static constexpr u64 RADB = 1ull << LRAD;
static_assert(LANES == 3, "(f1): exactly three lanes: with four, (4 beta - 2) B^3 is above p");
static_assert((1ull << LRAD) >= 4 * (1ull << LBZ) - 1, "(f1): B >= 4 beta - 1");
static_assert((unsigned __int128)(4 * (1ull << LBZ) - 2) *
                      ((unsigned __int128)1 + RADB + (unsigned __int128)RADB * RADB) <
                  (unsigned __int128)FIELD_P,
              "(f1): (4 beta - 2)(1 + B + B^2) < p");
// (f2), no abort: an honest prover needs -beta <= Z < beta for each entry Z of a random combination
// of k columns of signed bytes: P(|Z| >= beta) <= 2 exp(-beta^2 / (2^14 k)) <= 2 exp(-2^7).
static_assert(((unsigned __int128)(KCAP + 1) << (14 + 7)) <= ((unsigned __int128)1 << (2 * LBZ)),
              "(f2): the no-abort condition ties the limit on a folded matrix's columns to beta");

// ---- the shape of a symbol tree ----
// m, k: the matrix's cell rows and columns. R tree rows hold up to J symbol rows each, Rd dummy
// rows follow, Rr = R + Rd. K1 of a tree row's Kc entries hold data. `a` sub-proofs use the matrix
// (owner and readers). cJ = clog2(J), aX = clog2(R). In the programs only t2AddP (vf_subproof_segment.h) sets
// a shape: J = 1, R = Rr = mS, Rd = 0, K1 = Kc = k. Any other J, R, Rr or Rd ends NOT CHECKED
// (powersF, vf_smallness_test_members.inc). Nothing in this repository sets J above 1 or the rest: code length n =
// 2^ln = 2^(w + 1), t opened columns, mk = Kc - K1, P2 = 2^clog2(mk).
struct ShP {
  u64 m = 0, k = 0, mS = 0; // mS = m' = ceil(m / 3) symbol rows
  int w = 0, ln = 0, cJ = 0, aX = 0;
  u32 t = 0, a = 0;
  u64 n = 0, mk = 0, P2 = 0, K1 = 0, Kc = 0, J = 0, R = 0, Rd = 0, Rr = 0;
  bool same(const ShP &o) const {
    return m == o.m && k == o.k && mS == o.mS && w == o.w && ln == o.ln && cJ == o.cJ && aX == o.aX && t == o.t &&
           a == o.a && n == o.n && mk == o.mk && P2 == o.P2 && K1 == o.K1 && Kc == o.Kc && J == o.J && R == o.R &&
           Rd == o.Rd && Rr == o.Rr;
  }
};
// The symbol of symbol row iS and matrix column c is at entry `place` of tree row `row`. Each
// matrix has a symbol tree to itself, so its first column begins at entry 0. placeP and tieLeft
// divide by s.J without a check, and a new ShP has J = 0. placeP checks neither iS nor c.
struct PlP {
  u64 row = 0, place = 0;
};
static PlP placeP(const ShP &s, u64 iS, u64 c) {
  PlP p;
  p.row = iS / s.J;
  p.place = c * s.J + (iS % s.J);
  return p;
}

// ---- the placement record ----
// One sub-proof (the owner) commits a border matrix and another reads it; each keeps this record.
// If samePlaceF finds a mismatch, a checking call ends NOT CHECKED: "internal-g-owner-place-" (or
// -a-, -h-) + its label.
// owner: the owner's number among the layer sub-proofs. rootIndex: the tree's index among the
// owner's symbol trees. hb: the parameter file's line `sec` minus the proof-of-work bits POW. The
// layout inside a matrix is not covered; tensorsAgree (cp_verify.cpp) compares only matrix qa's.
struct PlaceF {
  int owner = -1, rootIndex = -1;
  // haveShape: sh has been set. haveMatrix: col0, k and m have been set.
  ShP sh;
  bool haveShape = false;
  u64 col0 = ~0ull, k = 0, m = 0; // first column, columns, cell rows
  bool haveMatrix = false;
  u32 hb = 0;
  // mS, J, w: as in sh. lanes = LANES, padRows = R J, K = sh.k, off = 0.
  u64 mS = 0, lanes = 0, padRows = 0, J = 0, K = 0, off = ~0ull;
  u32 w = 0;
  bool whole() const { return owner >= 0 && rootIndex >= 0 && haveShape && haveMatrix && hb; }
};
// Returns the label of the first group of fields in which the owner's o and the reader's r differ,
// or ""; also "P3" if o's columns run past o.K.
static const char *samePlaceF(const PlaceF &o, const PlaceF &r) {
  if (o.owner != r.owner) return "P0";
  if (!o.sh.same(r.sh)) return "P1";
  if (o.col0 != r.col0 || o.k != r.k || o.col0 + o.k > o.K) return "P3";
  if (o.m != r.m) return "P4";
  if (o.rootIndex != r.rootIndex) return "P5";
  if (o.hb != r.hb) return "P6";
  if (o.mS != r.mS || o.lanes != r.lanes || o.padRows != r.padRows || o.J != r.J || o.K != r.K || o.off != r.off ||
      o.w != r.w)
    return "P9";
  return "";
}

// ---- the tie ----
// Returns the tie's left coefficients wl, one for each cell row i < m. powersF
// (vf_smallness_test_members.inc) makes `rows` (Hi) and `lo` (Lo). The claim and its coefficients:
// docs/format/11-1-machinery.md, "The claim of a matrix" and "The weights that the test puts on the
// arrays". refuse() (vf_common.h) ends the process.
static vector<Ex> tieLeft(const ShP &s, const vector<Ex> &rows, const vector<Ex> &lo) {
  if (rows.size() != s.Rr || lo.size() != s.J) refuse("internal-sym-point");
  vector<Ex> w(s.m);
  parFor((s.mS + ENTRIES_PER_PARALLEL_JOB - 1) / ENTRIES_PER_PARALLEL_JOB, [&](u64 b) {
    for (u64 i = b * ENTRIES_PER_PARALLEL_JOB; i < s.mS && i < (b + 1) * ENTRIES_PER_PARALLEL_JOB; i++) {
      const Ex e = rows[i / s.J] * lo[i % s.J];
      u64 sc = 1;
      for (u64 l = 0; l < (u64)LANES && LANES * i + l < s.m; l++, sc <<= LRAD)
        w[LANES * i + l] = l ? eS(e, sc) : e;
    }
  });
  return w;
}
// Returns the tie's right coefficients wr, of a tree row's first J * k entries, from the column
// weights gam: docs/format/11-1-machinery.md, "The weights that the test puts on
// the arrays".
static vector<Ex> tieRight(const ShP &s, const vector<Ex> &lo, const vector<Ex> &gam) {
  if (lo.size() != s.J || gam.size() != s.k) refuse("internal-sym-point");
  vector<Ex> w(s.J * s.k);
  parFor(s.k, [&](u64 c) {
    for (u64 x = 0; x < s.J; x++) w[placeP(s, x, c).place] = gam[c] * lo[x];
  });
  return w;
}
// For self-tests only: eq(z, idx) as the plain product over idx's n bits, the top bit taking z[0].
// Like UnscaledEqIterator it uses only field arithmetic, not eqTab() (vf_common.h), whose tables it checks.
static Ex eqPlainF(const Ex *z, int n, u64 idx) {
  Ex p = eO();
  for (int v = 0; v < n; v++) p = p * (((idx >> (n - 1 - v)) & 1) ? z[v] : eO() - z[v]);
  return p;
}
// For self-tests only (tileSelfTestF, vf_smallness_test_members.inc): eq(z, idx) at consecutive
// idx. Only idx's low n bits count.
struct UnscaledEqIterator {
  const Ex *z;
  int n;
  u64 idx;
  vector<Ex> pre;
  UnscaledEqIterator(const Ex *z_, int n_, u64 i0) : z(z_), n(n_), idx(i0), pre((size_t)n_ + 1) {
    pre[0] = eO();
    fill(0);
  }
  void fill(int v0) {
    for (int v = v0; v < n; v++) pre[v + 1] = pre[v] * (((idx >> (n - 1 - v)) & 1) ? z[v] : eO() - z[v]);
  }
  Ex at() const { return pre[n]; }
  void next() {
    const u64 ch = idx ^ (idx + 1);
    idx++;
    int top = 63 - __builtin_clzll(ch);
    fill(top >= n ? 0 : n - 1 - top);
  }
};
