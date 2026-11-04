// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Lookups and helpers for sub-proofs (`fork` in identifiers). Included at the end of vf_subproof_engine.h.
#pragma once
// The affine form (`Aff`) for element x of a vector that F.trees[t] stores as a table of Kw
// columns: column x mod Kw of virtual row (`HRow`) floor(x / Kw), assumed to be that table row.
static Aff uEnt(const HFork &F, u32 t, u64 x) {
  const u64 Kw = F.trees[t].K;
  // Ent::row has 32 bits: a row number that does not fit is rejected (`refuse`) before the cast.
  // HFork::push() checks only the 32-bit value, under the same rejection name.
  if ((x / Kw) >> 32) refuse(F.G_ + "internal-L-row");
  return affEnt(t, (u32)(x / Kw), x % Kw);
}

#include "vf_lookup_argument.h" // the lookup on hidden values, for a group of calls of one sub-proof
#include "prover/plan_rounds.h" // in prover/: rounds of the layer sub-proof, shared with the prover
// The layer sub-proof's batched sumcheck (`Psc`, vf_sumcheck_psc.h): lam rounds with no equality factor,
// and their challenges rs. The output head's sub-proof (vf_output_head_check.inc) uses only lam and rs.
struct HStack {
  int lam = 0;
  vector<Ex> rs;
  Psc p; // the caller begins it (`pscBegin`) with the start form, ends it and takes its claim
};
// Reads the lam rounds, each of degree 2, from bit lam - 1 down. `Tr` (vf_common.h): the transcript.
static void stackRounds(HFork &F, Tr &tr, u32 &slot, HStack &st) {
  st.rs.assign(st.lam, eZ());
  for (int v = st.lam - 1; v >= 0; v--) st.rs[v] = roundIn(F, tr, slot, RoundForm::SUM2, st.p);
}
// eq over bits lo .. hi - 1, with rs[v] the coordinate of bit v. eqTab() expects
// coordinate 0 to be the top bit, so the coordinates are passed from the top bit down.
static vector<Ex> eqLow(const vector<Ex> &rs, int lo, int hi) {
  vector<Ex> z;
  for (int v = hi - 1; v >= lo; v--) z.push_back(rs[v]);
  return eqTab(z.data(), hi - lo);
}
// Entry c of eqLow(rs, lo, hi): bit j of c goes with rs[lo + j].
static Ex eqLowAt(const vector<Ex> &rs, int lo, int hi, u64 c) {
  vector<Ex> z;
  for (int v = hi - 1; v >= lo; v--) z.push_back(rs[v]);
  return eqOne(z.data(), hi - lo, c);
}
// Parts of the approximate range check that ends a sub-proof (vf_smallness_test.h).
// HPiece is one piece of a lookup call: a block of Rp rows by Cp columns that starts at row r0,
// column c0 of the array HFork::trees[tree]; `row` is the index of its virtual row in
// HTree::rows. placeZ() sets the rest: CP = Cp rounded up to a power of 2, a = ceil(log2 Rp),
// b = a + log2 CP. The piece takes the 2^b lookup keys from `off` on; its row r, column c has key
// off + r * CP + c. HCallZ is the call (`Call`, vf_common.h) with its pieces.
struct HPiece {
  int tree = 0, row = 0;
  u64 r0 = 0, Rp = 0, c0 = 0, Cp = 0, CP = 0, off = 0;
  int a = 0, b = 0;
};
struct HCallZ {
  Call c;
  vector<HPiece> pc;
};
// Places z's pieces, largest first, so that 2^b divides `off`; fills in z.c. lid: LIDB (vf_common.h).
static void placeZ(HCallZ &z, const string &name, int k, int lid) {
  for (auto &p : z.pc) {
    p.a = clog2(p.Rp);
    p.CP = powerOfTwoAtLeast(p.Cp);
    p.b = p.a + clog2(p.CP);
  }
  vector<size_t> ord(z.pc.size());
  for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
  std::stable_sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return z.pc[x].b > z.pc[y].b; });
  u64 off = 0;
  for (size_t i : ord) {
    z.pc[i].off = off;
    off += 1ull << z.pc[i].b;
  }
  Call &c = z.c;
  c.name = name;
  c.k = k;
  c.kappa = 5;
  c.fn = 0;
  c.t0 = 0;
  c.n = clog2(off) < 10 ? 10 : clog2(off);
  c.iota = c.n > lid ? c.n - lid : 0;
  // pub counts the padding keys (each Rp must be a power of 2): in each piece, columns Cp .. CP - 1
  // of rows 0 .. Rp - 1, and all keys after the last piece. Nothing in verifier/ reads Call::pub.
  const u64 I = c.I(), N = 1ull << c.n;
  c.pub.assign(I, 0);
  auto padRun = [&](u64 first, u64 count) {
    // Each key i of first .. first + count - 1
    // adds one to pub[i mod I].
    const u64 whole = count / I;
    for (u64 b = 0; b < I; b++) c.pub[b] += whole;
    for (u64 i = first + whole * I; i < first + count; i++) c.pub[i & (I - 1)]++;
  };
  for (auto &p : z.pc)
    if (p.Cp < p.CP)
      for (u64 rr = 0; rr < p.Rp; rr++) padRun(p.off + rr * p.CP + p.Cp, p.CP - p.Cp);
  if (off < N) padRun(off, N - off);
}
struct HFun {
  int M = 0;
  u64 L = 0;
  vector<Ex> phi;
  u32 tree = 0;
  u64 at = 0;
  string name;
};
// Header block HD: hb, the target in bits (the parameter file's line `sec`), at index `at` of the
// header's words. parseHeader() (cp_verify.cpp) sets both.
struct TargetBitsWord {
  size_t at = 0;
  u32 hb = 0;
};
static TargetBitsWord statedTarget;
