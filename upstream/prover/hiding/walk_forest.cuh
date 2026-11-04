// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Adds to a `Walk` (the prover's record of a sub-proof's linear claims) the claims of a lookup
// forest. `Forest` runs a group of lookup calls, each a key vector with its public table, as one
// lookup, with product trees over all keys (key side) and all table rows (table side).
// Planes: the multiplicities' bits b, indexed (t | c | j) = (row, call, bit), of kT, cK and 5 bits.
// The caller must add walk_forest()'s claims, the join (forest_join()) and the plane claims in
// the order of its sub-proof's verifier (not checked here): a claim's number sets its batching
// coefficient.
// Forms (AF: linear in committed values and mask entries) omit public constants, which a `Walk`
// ignores.
#pragma once
#include "hiding/product_test.cuh"
#include "hiding/forest.cuh"
#include "hiding/walk.cuh"
namespace hw {
// cend: the hidden value of the key side's leaves at (claims->zx | claims->zy) of `Forest::claims`.
// bit: the planes' hidden value at claims->zb[1], the end value of the bits' sumcheck
// (`psc` in identifiers) for b^2 = b.
// leaves, leaves_start: the sumcheck over the table side's leaves and its start form;
// walk_forest_plane_claims() adds its claim.
struct ForestForms {
  AF cend, bit, leaves_start;
  Rec leaves;
};
// Adds f's claims that read mask entries only, in forestObjects()'s order
// (verifier/src/vf_lookup_argument.h): the tie of the sides' roots (31 claims), one for each key level, each
// table level and the bits' sumcheck.
// sb: f's first message slot and mask entry. f: after Forest::init() (not checked): W keeps
// addresses into f's GPU memory, which Forest::prove() fills.
inline ForestForms walk_forest(Walk &W, const Forest &f, size_t sb) {
  ForestForms o;
  const ForestPlan &P = f.plan;
  const std::vector<Forest::Layer> L = f.layers();
  const LookupTieForms tie = walk_lookup_tie(W, sb + P.at_tie(), f.claims->rho5);
  size_t at = 0;
  AF C = tie.Z;
  for (int l = 0; l < P.v(); l++, at++) C = W.layer(W.psc(0, L[at].r_round, sb + L[at].slot, 2, 2, L[at].zeta), C);
  o.cend = C;
  C = tie.table;
  for (int l = 0; l < P.d(); l++, at++) C = W.layer(W.psc(0, L[at].r_round, sb + L[at].slot, 2, 2, L[at].zeta), C);
  o.leaves_start = C;
  {
    std::vector<int> degrees(P.d(), 2); // in t and c
    degrees.insert(degrees.end(), 5, 1); // in j
    o.leaves = W.psc_degrees(psc::Form::EQ, degrees, L[at].r_round, sb + L[at].slot, 0, 0, nullptr);
    at++;
  }
  const Rec bits = W.psc(0, L[at].r_round, sb + L[at].slot, 2, 1, nullptr);
  W.row(bits, AF());
  o.bit = W.shown(bits, 0);
  return o;
}
// The linear part of the join, which relates F.cend to committed data:
//   sum_c e_c (alpha - c theta - KF_c) + (padding-only trees' term) - F.cend.
// KF[c]: the form of call c's key vector at claims->zk[c], one per call, in f.calls' order
// (not checked).
inline AF forest_join(const Forest &f, const ForestForms &F, const std::vector<AF> &KF) {
  AF a;
  for (size_t c = 0; c < KF.size(); c++) a = a - KF[c] * Sy(&f.claims->e[c]);
  return a - F.cend;
}
// Adds the three claims that read the planes, in forestPlaneLins()'s order. at0, at1: the forms
// of the planes' values at claims->zb[0] and zb[1]. sum: the form of sum 2^j b over all b; public
// value: the count of non-padding keys.
inline void walk_forest_plane_claims(Walk &W, const Forest &f, const ForestForms &F, const AF &at0, const AF &at1, const AF &sum) {
  W.row(F.leaves, F.leaves_start, at0 * Sy(&f.claims->leaves_factor));
  W.lin(F.bit - at1);
  W.lin(sum);
}
// The same claims, where the planes are a matrix M of 2^lm rows, 5 <= lm <= kT + 5 (not checked):
// row (low lm - 5 bits of t | j), column (c | the other bits of t). Its weighted column sums TB_0,
// TB_1, TP, sum_row phi[row] M[row][column] per column, start at elements oTB0, oTB1, oTP of W's
// space sp.
inline void walk_forest_planes(Walk &W, const Forest &f, const ForestForms &F, int sp, size_t oTB0, size_t oTB1, size_t oTP, int lm) {
  const ForestPlan &P = f.plan;
  const int w = P.kT + 5 - lm;
  const size_t cols = (size_t)1 << w, off[2] = {oTB0, oTB1};
  AF at[2];
  for (int e = 0; e < 2; e++) {
    const T *zb = f.claims->zb[e]; // (t | c | j)
    const T *eth = W.eq(zb, w);
    for (int c = 0; c < P.NC; c++)
      at[e] = at[e] + rng(sp, off[e] + (size_t)c * cols, 1, cols, eth, 1, Sy(W.eqAt(zb + P.kT, P.cK, (u64)c)));
  }
  walk_forest_plane_claims(W, f, F, at[0], at[1], rng(sp, oTP, 1, (size_t)P.NC * cols, nullptr));
}
// The weights phi of TB_e and TP, 2^lm entries each. zb: claims->zb[e].
//   TB_e, k_forest_phi_bits():   phi[row] = eq((last lm - 5 coordinates of t | those of j), row)
//   TP, k_forest_phi_powers():   phi[row] = 2^(row mod 32)
// The loop form is public: the grid-stride loop; see k_wcol() (prover/hiding/walk.cuh).
__global__ void k_forest_phi_bits(const T *zb, int kT, int cK, int lm, T *phi) {
  const size_t n = (size_t)1 << lm;
  const int nt = lm - 5;
  for (size_t row = blockIdx.x * (size_t)blockDim.x + threadIdx.x; row < n; row += (size_t)gridDim.x * blockDim.x) {
    T v = F::one();
    for (int i = 0; i < nt; i++) {
      const T z = zb[kT - nt + i];
      v = F::mul(v, ((row >> (lm - 1 - i)) & 1) ? z : F::sub(F::one(), z));
    }
    for (int i = 0; i < 5; i++) {
      const T z = zb[kT + cK + i];
      v = F::mul(v, ((row >> (4 - i)) & 1) ? z : F::sub(F::one(), z));
    }
    phi[row] = v;
  }
}
__global__ void k_forest_phi_powers(int lm, T *phi) {
  const size_t n = (size_t)1 << lm;
  for (size_t row = blockIdx.x * (size_t)blockDim.x + threadIdx.x; row < n; row += (size_t)gridDim.x * blockDim.x)
    phi[row] = ext_base(1ull << (row & 31));
}
} // namespace hw
