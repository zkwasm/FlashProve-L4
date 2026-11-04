// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Adds the linear claims of a sub-proof's approximate range check (`tail` in identifiers) to its
// list, a `Walk`. Keep the verifier's order: a claim's number sets its batching coefficient.
// rng(): a range of variables, not the range check.
#pragma once
#include "hiding/product_test.cuh"
#include "hiding/forest.cuh"
#include "hiding/walk.cuh"
#include "hiding/walk_forest.cuh"
#include "hiding/tail.cuh"
namespace hw {
// A read: a table Tab of weighted sums of matrix mu's m rows. L must divide m (not checked here).
//   Tab[col, sg] = sum over i < L of phi[i] M_mu[sg L + i][col], for each run sg < m / L of L rows
// Tab[col, sg] is variable off + col (m / L) + sg of the Walk's space `tree`: 1 or 3, or else
// HiddenTail::make_powers() exits. phi: GPU memory. No caller uses space 1: each leaves it empty,
// so Walk::emit() would exit.
struct TableRead {
  int mu;
  size_t L;
  const T *phi;
  size_t off;
  int tree = 3;
};
// Adds the claims in the order of objectsF() in verifier/src/vf_smallness_test_members.inc: those of walk_forest()
// (prover/hiding/walk_forest.cuh) for the lookup forest HT.F (both range lookups); its join (the
// lookups' keys are ZZ, the digits of the combinations Z = M Cf); three on its multiplicity bits,
// BZ; one tie per matrix; one per run of each read (by matrix; then space 1 before 3). slPZ: the
// forest's first message slot. ZZ and BZ are read through the virtual rows HT.vrows (weighted sums
// of an array's rows), each a space with one variable per column. A tie compares mu's digits with
// mu's columns, space spP0 + HT.pcOf[mu].
// W and HT must not move afterwards: W.prep keeps their addresses.
inline void walk_tail_f(Walk &W, HiddenTail &HT, int spP0, size_t slPZ, const std::vector<TableRead> &reads) {
  const int row0 = 5; // space 0: message slots; 1, 3: reads' tables; 2, 4: unused; 5+: virtual rows
  const int rRead = row0 + HT.rowX4, rTie = row0 + HT.rowTie, rBZ = row0 + HT.rowBZ0;
  constexpr int NCZ = HiddenTail::NCZ;
  const Forest &f = HT.F;
  const int rest = f.plan.d();
  const ForestForms ff = walk_forest(W, f, slPZ);
  {
    // zk = (piece | row | column) serves both lookups; their pieces (blocks of ZZ's rows) differ
    // only in column c0.
    const T *zk = HT.zk();
    std::vector<AF> KF(NCZ);
    for (size_t i = 0; i < HT.pc[0].size(); i++) {
      const HPieceG &p0 = HT.pc[0][i];
      const T *ec = W.eq(zk + (HT.n9 - p0.b) + l2i(p0.Rp), l2i(p0.CP));
      const T *at = W.eqAt(zk, HT.n9 - p0.b, p0.off >> p0.b);
      for (int q = 0; q < NCZ; q++) {
        const HPieceG &pc = HT.pc[q][i];
        KF[q] = KF[q] + rng(row0 + pc.row, pc.c0, 1, pc.Cp, ec, 1, Sy(at));
      }
    }
    W.lin(forest_join(f, ff, KF));
  }
  // BZ[r][x]: bit j < 32 of the multiplicity of lookup c's table row t; (t | c | j) = r width + x.
  // r is t's top five bits; x < width = 2^rest goes with the `rest` coordinates after zb's first
  // five. wc[x] = 2^j, for the sum of all multiplicities.
  const size_t width = HT.kb;
  T *wc = W.take(width);
  {
    const T *one = W.eq(f.claims->zb[0], 0); // `eq` over no coordinates: one element, 1
    W.prep.push_back([=] { k_wcol<<<(unsigned)std::min<size_t>((width + 255) / 256, 4096), 256>>>(one, 0, 5, width, wc); });
  }
  walk_forest_plane_claims(W, f, ff, rng(rBZ, 0, 1, width, W.eq(f.claims->zb[0] + 5, rest)),
                           rng(rBZ + 1, 0, 1, width, W.eq(f.claims->zb[1] + 5, rest)), rng(rBZ + 2, 0, 1, width, wc));
  // A tie or a read's claim combines many equations (numbered in HiddenTail). lead() is its extra
  // power of the batching challenge theta; make_powers() computes it, wtie and gamT once
  // Walk::finish() has drawn theta.
  const size_t mz = HT.mz;
  size_t all_reads = 0;
  for (auto &fn : reads) all_reads += HT.M[fn.mu].m / fn.L;
  HT.leads.clear();
  HT.room_for_leads((size_t)HT.NM + all_reads);
  auto lead = [&](unsigned long long first) {
    HT.leads.push_back({W.nclaim, first});
    return Sy(HT.lead + (HT.leads.size() - 1));
  };
  for (int mu = 0; mu < HT.NM; mu++)
    W.lin((rng(rTie, HT.off[mu], 1, HT.M[mu].m, HT.wtie[mu]) +
           rng(rTie, mz + HT.off[mu], 1, HT.M[mu].m, HT.wtie[mu], 1, Sy(hE(1ull << EZD0))) -
           rng(spP0 + HT.pcOf[mu], HT.tieX0[mu], 1, HT.tieN[mu], HT.tieR[mu])) *
          lead(HT.eq_first[mu]));
  size_t q = 0;
  for (int mu = 0; mu < HT.NM; mu++)
    for (int tree = 1; tree <= 3; tree += 2)
      for (auto &fn : reads) {
        if (fn.mu != mu || fn.tree != tree) continue;
        const size_t nsg = HT.M[mu].m / fn.L;
        for (size_t sg = 0; sg < nsg; sg++, q++)
          W.lin((rng(rRead, HT.off[mu] + sg * fn.L, 1, fn.L, fn.phi) +
                 rng(rRead, mz + HT.off[mu] + sg * fn.L, 1, fn.L, fn.phi, 1, Sy(hE(1ull << EZD0))) -
                 rng(fn.tree, fn.off + sg, (long long)nsg, HT.M[mu].k, HT.gamT[mu])) *
                lead(HT.eq_first[HT.NM] + q * HiddenTail::NP));
      }
  {
    Walk *walk = &W; // theta's address and nclaim are final at `Walk::seal`; the job runs later
    HiddenTail *tail = &HT;
    W.prep.push_back([walk, tail] { tail->make_powers(walk->theta, walk->nclaim); });
  }
}
} // namespace hw
