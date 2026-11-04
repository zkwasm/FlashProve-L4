// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The first messages of the lookup argument (LKP, blocks/iop/lkp.h: every key is to occur in a
// public table) for a group of lookup calls, on the GPU: k_draws(), k_tie(), k_join(), in that
// order. prover/hiding/forest.cuh launches them, then proves the two sides' product trees from
// the roots down; verifier/src/vf_lookup_argument.h is the verifier's side. For a challenge alpha:
//   Z    the product of (alpha - key) over all keys, padding included: the key side's root
//   G_j  j < 32: the product of (alpha - row) over the table rows whose multiplicity has bit j set:
//        the table side's root for bit j (the bits are `PLANES` in identifiers)
//   a    public: the product of (alpha - padding key) over all padding keys, which the
//        multiplicities (how many keys equal each row) do not count
// The tie is to show Z = a prod_j G_j^(2^j). The kernels here send Z and the G_j only masked.
//
// Before k_draws() the transcript (`Tr`; a hash chain, blocks/chain/chain.h) must fix everything
// public about the lookup, and the commitments from which keys, multiplicities and mask entries
// (random elements that mask sent scalars) are read, with the answers to their out-of-domain
// samples: the caller's duty. Nothing here absorbs or checks them. `Tr`, `T` (an extension-field
// element) and `F` (its arithmetic) are from prover/hiding/transcript.cuh: include it first.
#pragma once
#include "common/lie.cuh"
#include "common/slot_flags.h"
#include "hiding/place.cuh"
#include "hiding/walk.cuh"

namespace lookup_tie {

// PLANES: a multiplicity's bits. SLOTS: the tie's message slots, one scalar each.
constexpr int PLANES = 32, STEPS = PLANES - 1, SLOTS = 7 + 6 * (STEPS - 1);
static_assert(SLOTS == 187, "7 slots for the first step, 6 for each of the 30 others");

// The public shape of the keys of NC calls: 2^rK trees of 2^dT keys each. Tree y < NT belongs to
// call call_of[y]; the other trees contain only padding. Call c pads with row pad_row[c] of
// table[c] (GPU memory); a null table[c] is the range table, whose row t is t. NC <= 16 and
// NT <= 256 are not checked here: forest_plan() (prover/hiding/forest_plan.h) exits on more.
struct Trees {
  int NC = 0, NT = 0, rK = 0, dT = 0;
  unsigned char call_of[256] = {};
  const u64 *table[16] = {};
  u32 pad_row[16] = {};
  __device__ T pad_key(int c) const { return F::small(table[c] ? table[c][pad_row[c]] : (u64)pad_row[c]); }
};

// x^e for a public e. Public source: A. Menezes, P. van Oorschot, S. Vanstone,
// Handbook of Applied Cryptography, CRC Press, 1996, chapter 14, section 14.6.1, Algorithm 14.76
// "Right-to-left binary exponentiation", https://cacr.uwaterloo.ca/hac/about/chap14.pdf.
__device__ __forceinline__ T power(T x, u64 e) {
  T p = F::one();
  for (; e; e >>= 1, x = F::mul(x, x))
    if (e & 1) p = F::mul(p, x);
  return p;
}

// Draws theta and alpha. One thread. Adding c theta to the keys and table rows of call c is to
// keep a key of one call from passing as a row of another call's table.
//   pub[y]                y < NT: the number of padding keys of tree y (public)
//   theta_alpha[0], [1]   theta (0 for a single call) and alpha
//   alpha_of_tree[y]      y < 2^rK: alpha - call_of[y] theta, used as alpha for the keys of tree y
//                         and its call's rows; alpha for a tree of padding only, whose keys are
//                         the padding key of call 0
//   a[0]                  the public factor a
//   also_alpha0, also_alpha1   if not null, each receives alpha
__global__ void k_draws(Tr *tr, const Trees g, const u32 *pub, T *theta_alpha, T *alpha_of_tree, T *a, T *also_alpha0,
                        T *also_alpha1) {
  Tr t = *tr;
  T theta = F::zero(), alpha;
  if (g.NC > 1) tr_point(t, Round::LKP_SHIFT, &theta, 1);
  tr_point(t, Round::LKP, &alpha, 1);
  *tr = t;
  theta_alpha[0] = theta, theta_alpha[1] = alpha;
  if (also_alpha0) *also_alpha0 = alpha;
  if (also_alpha1) *also_alpha1 = alpha;
  T factor = power(F::sub(alpha, g.pad_key(0)), (((u64)1 << g.rK) - (u64)g.NT) << g.dT);
  for (int y = 0; y < 1 << g.rK; y++) {
    if (y >= g.NT) {
      alpha_of_tree[y] = alpha;
      continue;
    }
    const int c = g.call_of[y];
    const T mine = F::sub(alpha, F::mul(F::small((u64)c), theta));
    alpha_of_tree[y] = mine;
    factor = F::mul(factor, power(F::sub(mine, g.pad_key(c)), pub[y]));
  }
  a[0] = factor;
}

// (v1 + c1 Y)^2 (v2 + c2 Y): its coefficients of Y, Y^2, Y^3.
__device__ __forceinline__ void square_times_in_y(T v1, T c1, T v2, T c2, T *o) {
  const T twice = F::mul(F::add(v1, v1), c1), c1c1 = F::mul(c1, c1);
  o[0] = F::add(F::mul(F::mul(v1, v1), c2), F::mul(twice, v2));
  o[1] = F::add(F::mul(twice, c2), F::mul(c1c1, v2));
  o[2] = F::mul(c1c1, c2);
}

// The tie, by squaring and multiplying: P_31 = G_31, P_j = P_(j+1)^2 G_j, Z = a P_0. Each of the
// 31 steps is a masked sumcheck (PSC, blocks/iop/psc.h) over no variables. A step sends the
// hiding polynomial's sum, 3 scalars for the round in Y (the extra variable of PSC) and its masked
// end values, one message slot each:
//   step 0              0 = Z - a P_1^2 G_0;    end values Z, P_1, G_0: 7 slots
//   step j = 1 .. 30    P_j = P_(j+1)^2 G_j;    end values P_(j+1), G_j: 6 slots
// Step j starts from the claim on P_j that step j - 1 leaves. Left over: claims on the hidden
// values Z and G_j, from which the two sides' sumchecks start, and 31 linear claims (`row` in
// identifiers) on mask entries (walk_lookup_tie()). A step has 3 rounds, 93 in all: its end values
// are a round of their own, in which the verifier's message is empty.
// Z[0], G[0 .. 32), a[0]: unmasked values. slots: the tie's SLOTS message slots, written here.
// Mask entries are read at slots + PADOFF and scratch is written at slots + 2 PADOFF
// (prover/hiding/sumcheck_door.cuh): only PADOFF != 0 is checked.
// If Z is not a prod_j G_j^(2^j), the two sides were not made from the same keys (a GPU fault, a
// key outside its table, a padding entry that is not the padding key): the failure bit OWN_TEST
// is set, and the request gets no proof.
// One warp. Lane 0 has the values and does the arithmetic; the first SEND2_N lanes share each
// round's hash, on a copy of the transcript object in shared memory.
__global__ void k_tie(Tr *tr, const T *Z, const T *G, const T *a, T *slots) {
  using namespace sc_door;
  const Site site = Site::FOREST_ROOTS;
  __shared__ __align__(16) unsigned char room[sizeof(Tr)];
  __shared__ T P[PLANES]; // P[j] = P_j; only the prover's self-check below reads P[0]
  Tr *const t = reinterpret_cast<Tr *>(room);
  const int lane = threadIdx.x;
  const bool leader = lane == 0;
  tr_copy_by_lanes(t, tr, lane, 32);
  if (leader) {
    P[PLANES - 1] = G[PLANES - 1];
    for (int j = PLANES - 2; j >= 0; j--) P[j] = F::mul(F::mul(P[j + 1], P[j + 1]), G[j]);
    // In a malicious-prover test build that is not part of this repository, a run can ignore the
    // failure, so that a verifier gets the proof (prover/common/lie.cuh). There lie::pass() records
    // its argument, this check's name, so it is called only on a failure; here it returns false.
    raise_flags(F::eq(Z[0], F::mul(a[0], P[0])) || lie::pass(lie::LOOKUP_TIE) ? 0u : (unsigned)OWN_TEST);
  }
  __syncwarp();
  if (lane < SEND2_N)
    for (int j = 0; j < STEPS; j++) {
      const int shown = j ? 2 : 3,
                first = shown - 2; // of `shown` end values, P_(j+1) is number `first`, G_j the next
      begin_by_lanes(site, t, slots, j ? P[j] : F::zero(), 3);
      T in_y[3], v[3];
      if (leader) {
        square_times_in_y(P[j + 1], inner_mask(*t, first), G[j], inner_mask(*t, first + 1), in_y);
        if (!j) { // Z - a P_1^2 G_0
          for (T &x : in_y) x = F::sub(F::zero(), F::mul(a[0], x));
          in_y[0] = F::add(in_y[0], inner_mask(*t, 0));
        }
        v[0] = j ? P[j + 1] : Z[0], v[1] = j ? G[j] : P[1], v[2] = G[0];
      }
      further_by_lanes<3>(site, t, in_y);
      if (j)
        ends_by_lanes<2>(site, t, v);
      else
        ends_by_lanes<3>(site, t, v);
      slots += 1 + 3 + shown;
    }
  __syncwarp();
  tr_copy_by_lanes(tr, t, lane, 32);
}

// Draws the point that joins the 32 claims on the G_j into one claim on the table side's top
// level. One thread, after k_tie(): the point must follow the last step's end values.
//   rho[0 .. 5)     the point, top bit of j first, as hw::Walk::eq() reads it (drawn low bit first)
//   table_start[0]  sum_j eq(rho, j) G_j, unmasked: the table side's top level, 32 entries, at rho
//   key_start[0]    Z, unmasked: the key side's root
__global__ void k_join(Tr *tr, const T *Z, const T *G, T *rho, T *table_start, T *key_start) {
  Tr t = *tr;
  T drawn[5];
  tr_point(t, Round::LKP_PLANES, drawn, 5);
  *tr = t;
  for (int i = 0; i < 5; i++) rho[i] = drawn[4 - i];
  T sum = F::zero();
  for (int j = 0; j < PLANES; j++) {
    T w = F::one();
    for (int i = 0; i < 5; i++) w = F::mul(w, (j >> i) & 1 ? drawn[i] : F::sub(F::one(), drawn[i]));
    sum = F::add(sum, F::mul(w, G[j]));
  }
  table_start[0] = sum;
  key_start[0] = Z[0];
}

} // namespace lookup_tie

// ---- The tie in a sub-proof's list of linear claims (hw::Walk, prover/hiding/walk.cuh) ----
// The two hidden values from which the two sides start, as affine forms.
struct LookupTieForms {
  hw::AF Z;     // the key side's root
  hw::AF table; // the table side's top level at the joining point
};
// Adds the tie's 31 linear claims to W, in the order of the steps. sb: the tie's first message
// slot. rho5: the joining point in GPU memory, as lookup_tie::k_join() writes it.
inline LookupTieForms walk_lookup_tie(hw::Walk &W, size_t sb, const T *rho5) {
  using namespace hw;
  LookupTieForms out;
  const T *weight = W.eq(rho5, 5); // eq(rho5, j), j < 32
  AF power;                        // the hidden P_j, from the previous step's end value
  for (int j = 0; j < lookup_tie::STEPS; j++) {
    const size_t shown = j ? 2 : 3, first = shown - 2;
    const Rec R = W.psc(0, {}, sb, 3, shown, nullptr);
    W.row(R, j ? power : AF());
    if (!j) out.Z = W.shown(R, 0);
    power = W.shown(R, first);
    out.table = out.table + W.shown(R, first + 1) * Sy(weight + j);
    sb += 1 + 3 + shown;
  }
  out.table = out.table + power * Sy(weight + lookup_tie::PLANES - 1); // P_31 is G_31
  return out;
}
