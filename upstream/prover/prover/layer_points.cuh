// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Kernels that a sub-proof (the argument for part of the forward pass) launches outside its
// sumchecks. They absorb stored values into the transcript `Tr`, sample the verifier's messages
// (points, batching coefficients) and compute the sums from which product layers start. Include
// prover/hiding/product_test.cuh first. Launch each kernel with one thread, except k_norms_start().
#pragma once
// The point of the norms' product layer, a sumcheck (`PtWPL`) on sum_c eq(rho, c) L[c] R[c],
// c = (instance | feature | token): rho = (lNI sampled challenges | lPd times 1/2 | PV[0 .. lt)).
// An instance is one norm of one layer. eq at 1/2 is 1/2 for either bit, so each feature weighs
// halves[0] = 2^-lPd.
__global__ void k_rho_norms(Tr *tr, int lNI, int lPd, const T *PV, int lt, T *rho, T *halves) {
  Tr t = *tr;
  const T half = F::inv(F::small(2));
  T h = F::one();
  if (lNI) tr_point(t, Round::JOIN, rho, lNI); // no message if there is one instance
  for (int i = 0; i < lPd; i++) rho[lNI + i] = half, h = F::mul(h, half);
  for (int i = 0; i < lt; i++) rho[lNI + lPd + i] = PV[i];
  halves[0] = h;
  *tr = t;
}
// o[0] = halves[0] sum_(i < NI) eq(rho[0 .. lNI), i) sum_(k < Pd) HV[i Pd + k]: the sum that the
// norms' product layer claims; eq(r, i) pairs r[0] with i's top bit. Launch: one block of 256
// threads.
__global__ void __launch_bounds__(256)
    k_norms_start(const T *rho, int lNI, int NI, int Pd, const T *HV, const T *halves, T *o) {
  __shared__ T sh[256];
  T a = F::zero();
  for (int x = threadIdx.x; x < NI * Pd; x += 256) {
    const int i = x / Pd;
    T e = F::one();
    for (int q = 0; q < lNI; q++) e = F::mul(e, ((i >> (lNI - 1 - q)) & 1) ? rho[q] : F::sub(F::one(), rho[q]));
    a = F::add(a, F::mul(e, HV[x]));
  }
  sh[threadIdx.x] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    T s = F::zero();
    for (int w = 0; w < 256; w++) s = F::add(s, sh[w]);
    o[0] = F::mul(s, halves[0]);
  }
}
// Adds to the prover's current message the n values that earlier launches stored in `slots` with
// sc_door::claim() (prover/hiding/sumcheck_door.cuh), each from a different thread.
__global__ void k_absorb_slots(Tr *tr, const T *slots, int n) {
  Tr t = *tr;
  sc_door::absorb(t, slots, n);
  *tr = t;
}
// Samples the verifier's message, n elements, into o. tr_point() traps at Round::PSC_Y, and
// unless n is 1 to 255 (0 for Round::PSC_VALUES).
__global__ void k_point(Tr *tr, Round round, T *o, int n) {
  Tr t = *tr;
  tr_point(t, round, o, n);
  *tr = t;
}
// Batching coefficients for n claims: o[i] = x^i for one sampled x.
__global__ void k_batching_powers(Tr *tr, T *o, int n) {
  Tr t = *tr;
  T x;
  tr_point(t, Round::BAT_SIGMA, &x, 1);
  T p = F::one();
  for (int i = 0; i < n; i++, p = F::mul(p, x)) o[i] = p;
  *tr = t;
}
// o[0] = sum_(i < n) eq(rho[0 .. l), i) v[i].
__global__ void k_join_at(const T *rho, int l, const T *v, int n, T *o) {
  T a = F::zero();
  for (int i = 0; i < n; i++) {
    T e = F::one();
    for (int q = 0; q < l; q++) e = F::mul(e, ((i >> (l - 1 - q)) & 1) ? rho[q] : F::sub(F::one(), rho[q]));
    a = F::add(a, F::mul(e, v[i]));
  }
  o[0] = a;
}
