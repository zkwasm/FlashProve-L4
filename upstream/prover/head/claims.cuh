// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The part of the output head's sub-proof (`fork` in identifiers; prover/head/hidden_fork.cuh,
   which includes this file) that concerns the committed weights. It reduces the statement about
   the head's matrix Wh[kk][x] (kk < d a feature, x < V a vocabulary entry) to one claim about W,
   the single committed matrix that holds all weights, in three steps:
   1. SCH, the rounds over kk: the second half of one sumcheck over token positions and features.
      The caller's token sumcheck (prover/hiding/token_sumcheck.cuh) has begun it and run the
      rounds over the token positions; nothing is sent and no challenge sampled between the halves.
   2. The sumcheck's end. The second of its two end values (its summand's factors at the final
      point), v_HD, is a linear function of Wh.
   3. MERGE, later: a sumcheck of lkW rounds over the column bits of W. It reduces the claim on
      v_HD to the value, at a random point, of one public linear combination of W's rows. The
      opening of the weight commitment proves that value later.
   HeadClaim::rounds runs steps 1 and 2, HeadClaim::merge step 3. A challenge is derived from the
   transcript (`tr`, a `Tr`: prover/hiding/transcript.cuh). Not checked in this file: 13-bit
   weights, d <= 2^18, tb <= 15 (see k_head_columns_part). */
#pragma once
#include "head/claims_lib.cuh"
#include "head/feature_rounds.cuh"

// Fills SCH's arrays A = af and B = bhd (see HeadClaim::rounds()), zero-padded to Pd elements.
__global__ void k_head_arrays(const T *af_rows, size_t stride, const T *bhd, int d, int Pd, T *A, T *B) {
  GRID_LOOP(kk, Pd) {
    const bool in = (int)kk < d;
    A[kk] = in ? af_rows[kk * stride] : F::zero();
    B[kk] = in ? bhd[kk] : F::zero();
  }
}
/* The head's weights are in the columns (lo << cV) | x of W: the vocabulary entry x (cV bits) and
   lo, the lowest tb bits of kk. So this kernel and k_head_columns split kk = (kh << tb) | lo;
   eq_hi and eq_lo are the tables of eq, the multilinear equality polynomial, at SCH's challenges
   for kh and for lo.

   The kernel writes partial sums for k_head_columns:
     sum_s part[s][lo][x] = sum_kh eq_hi[kh] (w + 8192),  s < gridDim.y, w = Wh[(kh << tb) | lo][x].
   A weight is two cells (signed bytes), its low and high limb, stored in the order [limb][kk][x].
   F::smac takes an unsigned integer, so it needs w + 8192 >= 0. Nothing here checks that; it holds
   for a head weight of 13 bits, -4096 <= w < 4096 (line `wb_head` of the parameter file,
   `policy.txt`). The kernel requires a grid of (V / 256 rounded up, any number, 2^tb) blocks of
   256 threads. A grid's z extent is at most 65535, so the launch fails if tb > 15, which nothing
   here checks. */
// The products are added up unreduced and reduced once, in F::sred() (prover/hiding/pt_field.h).
// The sum does not overflow while (terms) x (greatest integer) <= 2^32. That holds for d <= 2^18
// (not checked here): a sum has at most d terms, and for 13-bit weights w + 8192 is below 2^14. The
// single reduction is specific to modular arithmetic.
__global__ void k_head_columns_part(const i8 *cells, int V, int d, int tb, const T *eq_hi, T *part) {
  const int x = blockIdx.x * 256 + threadIdx.x, lo = blockIdx.z, nh = d >> tb, per = (nh + gridDim.y - 1) / gridDim.y,
            k0 = blockIdx.y * per, k1 = min(nh, k0 + per);
  if (x >= V) return;
  const i8 *c0 = cells, *c1 = cells + (size_t)d * V;
  F::SAcc a = F::szero();
  for (int kh = k0; kh < k1; kh++) {
    const size_t at = (size_t)((kh << tb) | lo) * V + x;
    const int w = (int)c0[at] + 256 * (int)c1[at];
    F::smac(a, F::sprep(eq_hi[kh]), (u32)(w + 8192));
  }
  part[((size_t)blockIdx.y * gridDim.z + lo) * V + x] = F::sred(a);
}

/* Computes MERGE's two input vectors. In column c = (lo << cV) | x of W, Tc[c] is the sum of the
   partial sums minus 8192 eq_hi_sum (= sum_kh eq_hi[kh]), which removes the offset, and
   eps[c] = phiHD[x] eq_lo[lo] (phiHD: see HeadClaim::merge). Both are 0 in every other column.
   v_HD = sum_c eps[c] Tc[c]. */
__global__ void k_head_columns(const T *part, int ns, const T *eq_hi_sum, const T *phiHD, const T *eq_lo, int V, int cV,
                               int tb, int lk, T *Tc, T *eps) {
  const T off = F::mul(eq_hi_sum[0], F::small(8192));
  GRID_LOOP(c, (size_t)1 << lk) {
    const size_t x = c & (((size_t)1 << cV) - 1), lo = (c >> cV) & (((size_t)1 << tb) - 1), up = c >> (cV + tb);
    if (x >= (size_t)V || up) {
      Tc[c] = F::zero();
      eps[c] = F::zero();
      continue;
    }
    T a = F::zero();
    for (int s = 0; s < ns; s++) a = F::add(a, part[((size_t)s * ((size_t)1 << tb) + lo) * V + x]);
    Tc[c] = F::sub(a, off);
    eps[c] = tb ? F::mul(phiHD[x], eq_lo[lo]) : phiHD[x];
  }
}

// The prover's state for the claim about the head's weights. A, B: SCH's arrays, of Pd = 2^lam
// elements. sch_coins: SCH's challenges. slices: the number of partial sums part[s] (s < slices).
struct HeadClaim {
  int lk = 0;  // lkW: log2 of the number of columns of W
  int cV = 0;  // ceil(lg V): the bits of a column index that select the vocabulary entry
  int tb = 0;  // low bits of kk that move into the column index: min(trailing zero bits of d, lk - cV)
  int lam = 0; // ceil(lg d): the number of SCH's rounds
  int d = 0, V = 0, Pd = 0, slices = 8;
  T *msg = nullptr; // MERGE's message slots
  T *A = nullptr, *B = nullptr, *sch_coins = nullptr;
  TscEnd ends; // the end of the sumcheck over positions and features
  T *eq_hi = nullptr, *eq_lo = nullptr, *eq_hi_sum = nullptr, *part = nullptr;
  CSumcheck SC;
  CEnd end; // end.rm: MERGE's challenges

  // slots: the first of MERGE's end.merge_slots() message slots. A message slot holds one scalar of
  // the sub-proof's message. HeadClaim's own arrays are `Place::Fork` memory
  // (prover/prover/fork_memory.h): valid only during the sub-proof's run.
  void setup(int lkW, int d_, int V_, T *slots) {
    lk = lkW;
    d = d_;
    V = V_;
    msg = slots;
    lam = l2i(d);
    Pd = 1 << lam;
    cV = l2i(V);
    if (cV > lk) {
      printf("policy: lkW = %d is below ceil(log2 V) = %d for the vocabulary size V = %d; raise lkW [hid-vocab-cols]\n",
             lk, cV, V);
      exit(2);
    }
    tb = std::min(c_tz(d), lk - cV);
    A = dalH<T>(Pd, Place::Fork);
    B = dalH<T>(Pd, Place::Fork);
    sch_coins = dalH<T>(32, Place::Fork);
    eq_hi = dalH<T>((size_t)1 << (lam - tb), Place::Fork);
    eq_lo = dalH<T>((size_t)1 << tb, Place::Fork);
    eq_hi_sum = dalH<T>(1, Place::Fork);
    part = dalH<T>((size_t)slices * ((size_t)1 << tb) * V, Place::Fork);
    SC.setup();
    end.setup(lk, 1, 1, Place::Result);
    // The summand is the product of the two end values. The shape repeats the one that the caller
    // gives to its token sumcheck (Tsc::init()).
    ends.setup(TscShape{2, 2, lam}, {A, B}, {TscTerm{0, 1, nullptr, nullptr}});
  }

  /* Runs SCH and the sumcheck's end, directly after the `before` rounds over the token positions.
     - af_rows: d rows, `stride` elements apart. The first entry of row kk is af[kk], the normalized
       activations at the random point where the rounds over the token positions ended.
     - bhd[kk], d elements: the other factor's value at that point. It must be the sum over x of
       phiHD[x] Wh[kk][x] (phiHD: see `merge`).
     SCH runs under eq weights at the coordinate 1/2, so each kk < Pd has the weight 2^-lam
     (TscShape::more). The sumcheck's end values are A[0], the value of af at SCH's point, and
     B[0] = v_HD, the value of bhd there. The end (TscEnd) sends the round in the further variable
     Y, in which v + m Y replaces each end value v (m: its inner mask, an entry of the sub-proof's
     mask row), and then each v as v + beta m, where beta is that round's challenge. */
  void rounds(Tr *tr, int before, const T *af_rows, size_t stride, const T *bhd) {
    k_head_arrays<<<grid_blocks(Pd), 256>>>(af_rows, stride, bhd, d, Pd, A, B);
    k_head_rounds<<<1, 1024>>>(A, B, lam, before, tr, sch_coins);
    ends.run(tr, cudaStreamPerThread);
  }
  /* Runs MERGE, after `rounds`: it reads sch_coins.
     - phiHD: the claim's public vector over the vocabulary, V elements (AV in
       prover/head/hidden_fork.cuh).
     - cells: the head's committed cells, 2 d V bytes (order: see k_head_columns_part).
     - c: one element in GPU memory, an entry of the request's index-mask message (its masks for
       claims on W) and not of the sub-proof's own mask row. MERGE's end value fv is sent as
       fv + beta c, where beta is a challenge (CEnd::send_merge, prover/head/claims_lib.cuh).
     No kernel launch here is checked: the caller must test `cudaGetLastError` after `merge`. */
  void merge(Tr *tr, const T *phiHD, const i8 *cells, const T *c) {
    // SCH fixes the highest bit of kk first, so kh's challenges come before lo's.
    k_eq<<<gsl_blocks_of((size_t)1 << (lam - tb)), 256>>>(sch_coins, lam - tb, eq_hi);
    k_eq<<<1, 256>>>(sch_coins + (lam - tb), tb, eq_lo);
    kG_sum1<<<1, 1024>>>(eq_hi, (size_t)(d >> tb), eq_hi_sum);
    k_head_columns_part<<<dim3((V + 255) / 256, slices, 1 << tb), 256>>>(cells, V, d, tb, eq_hi, part);
    k_head_columns<<<grid_blocks(end.K), 256>>>(part, slices, eq_hi_sum, phiHD, eq_lo, V, cV, tb, lk, end.Tor, end.ep);
    end.send_merge(SC, tr, msg, c); // v_HD has been sent, as v_HD + beta m, at the sumcheck's end
  }
};
