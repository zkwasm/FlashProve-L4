// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Digit rows of the model's norms: sizes, digit ranges, and the equations on a row when a value of
   the residual stream has five limbs (signed base-256 digits; parameter-file line "xl 5").

   For each token a norm needs V, the sum of the squares of the token's d stream values x,
   sigma = isqrt(V + eps) and R = floor((2^M + floor(sigma / 2)) / sigma); eps and M are public. The
   prover commits to the three, and to carries and remainders used to check them, as byte digits, a
   cell each: one digit row per norm and token (`led` in identifiers, `ledger` in rejection names).
   The equations (`Lines` in identifiers) are linear in the digits, in hidden products and in sums
   W_s of x_i x_j over the d values and all i + j = s (x_i: limb i of x). An equation is checked
   modulo p = 2^64 - 2^32 + 1, so it implies the integer one only if its sides differ by less than
   p. With four limbs V < 2^59 and sigma < 2^30 are one element each. With five, V can reach 2^78
   and sigma 2^39: V exists only as digits, and sigma = a + 2^24 b is split to keep every product
   below 2^48. This file checks nothing: the layer sub-proof (vf_layers_check.inc) and the output head's
   (vf_output_head_check.inc) claim the equations, range-check the digits and define the four-limb row. */
#pragma once
#include "vf_subproof_engine.h"

// Limbs of a stream value: 4, or 5 with the line "xl 5". Set when the parameter file is read.
static int residualStreamLimbs = 4;
// Counts per instance: one norm of one layer, or the final norm. gamma: the challenge batching W_s.
static int ledNS() { return 2 * residualStreamLimbs - 1; }          // convolution sums W_s, sent before gamma
static int ledNR() { return residualStreamLimbs == 5 ? 99 : 74; }   // digits of a row
static int ledNP() { return residualStreamLimbs == 5 ? 7 : 3; }     // hidden products, sent before their weights
static int ledNO() { return residualStreamLimbs == 5 ? 4 : 3; }     // end values of the products' sumcheck
static int ledLines() { return residualStreamLimbs == 5 ? 8 : 5; }  // equations after the carry equations
/* The multiplier mu of digit idx in the lookup that range-checks the row. The key mu x digit is a
   byte, so the digit is below 256 / mu, given a bound on its cell that keeps the key below p. */
static u64 ledMu4(int idx) {
  return idx == 7 ? 32 : idx == 44 ? 2 : idx == 45 ? 128 : idx == 48 ? 4 : 1;
}
static u64 ledMu5(int idx) {
  return idx == 9 ? 4 : idx == 58 ? 2 : idx == 59 ? 128 : idx == 63 ? 2 : idx == 76 ? 2 : idx == 79 ? 32 : 1;
}
static u64 ledMu(int idx) {
  return residualStreamLimbs == 5 ? ledMu5(idx) : ledMu4(idx);
}
/* The output head's row has LEDNE more digits after the final norm's (a layer's rows have none):
   six hidden numbers of a token position with a cost under the acceptance policy, low digit first.
   Each digit is range-checked as a byte, so each number is >= 0. c: the cost, if at most the
   clip C. b: above 0 if the cost is above C. c + b BIGH is added to the token position's logit
   margins. k = C - c, so c <= C. u, z: slack: over such token positions c + C b + u and b + z add
   up to public bounds. zs: 2^F minus the sum of the token position's exponential-table values,
   EXC cells each. LEDG[i]: offset after the norm's digits, and length, of c, b, u, k, z, zs. */
static constexpr int LEDNE = 21, EXC = 3;
static constexpr int LEDG[6][2] = {{0, 5}, {5, 1}, {6, 5}, {11, 5}, {16, 2}, {18, EXC}};
static constexpr u64 BIGH = 1ull << 37;
static int ledNRH() { return ledNR() + LEDNE; }
// LedK<T>::c(x): the constant x as a T (Aff or Ex).
template <class T> struct LedK;
template <> struct LedK<Aff> {
  static Aff c(Ex x) { return affC(x); }
};
template <> struct LedK<Ex> {
  static Ex c(Ex x) { return x; }
};
/* The five-limb row: its 99 digits Dg (cell + 128) and the numbers that row5() forms from them. In
   comments v_s is Dg[s], byte s of V; c_s is c[s]; sigma is sig; kappa is kap. Positions, ranges
   and meanings: docs/format/11-2-feed-forward-segment.md, "A norm's row of digits" (there V is
   Vsq). */
template <class T> struct Row5 {
  T Dg[99], c[10], Vlo, Vhi, a, b, sig, s0, l1, h1, Blo, Bhi, kap, Rlo, Rhi, R, q1, rr, rq;
};
// cell(j): the row's cell j as a T. A test that is not part of this repository uses T = Ex.
template <class T, class CellFn> static Row5<T> row5(CellFn cell) {
  typedef LedK<T> K;
  Row5<T> r;
  for (int j = 0; j < 99; j++) r.Dg[j] = cell(j) + K::c(eI(128));
  auto dig = [&](int at, int n) {
    T s = K::c(eZ());
    i64 sc = 1;
    for (int j = 0; j < n; j++, sc *= 256) s = s + r.Dg[at + j] * eI(sc);
    return s;
  };
  r.Vlo = dig(0, 6);
  r.Vhi = dig(6, 4);
  r.c[0] = K::c(eZ());
  for (int s = 1; s <= 8; s++) r.c[s] = dig(10 + 6 * (s - 1), 6) - K::c(eI(1ll << 47));
  r.c[9] = r.Dg[9];
  r.s0 = r.Dg[59];
  r.a = r.Dg[58] * eI(2) + r.Dg[59] + r.Dg[60] * eI(256) + r.Dg[61] * eI(65536);
  r.b = r.Dg[62] + r.Dg[63] * eI(256);
  r.sig = r.a + r.b * eI(1 << 24);
  r.l1 = dig(64, 5);
  r.h1 = dig(69, 5);
  r.Blo = dig(74, 3);
  r.Bhi = dig(77, 2);
  r.kap = r.Dg[79] - K::c(eI(4));
  r.Rlo = dig(80, 3);
  r.Rhi = dig(83, 3);
  r.R = r.Rlo + r.Rhi * eI(1 << 24);
  r.q1 = dig(86, 3);
  r.rr = dig(89, 5);
  r.rq = dig(94, 5);
  return r;
}
/* Carry equation s = 0 .. 8: W_s + c_s - v_s - 256 c_(s+1) = 0. The nine give V = sum_s 256^s W_s.
   |256 c_(s+1)| <= 2^55. The rejection `no-wrap-ledger` (cp_verify.cpp) keeps |W_s| < 2^53 if every
   |limb| < `Ty`, that file's bound on a cell. */
template <class T> static T ledCarry5(const Row5<T> &r, int s, const T &Ws) {
  return Ws + r.c[s] - r.Dg[s] - r.c[s + 1] * eI(256);
}
/* An instance's eight other equations, as differences that must be 0, in the order of their claims.
   P: the hidden products P0 .. P6 = a^2, a b, b^2, Rlo a, Rlo b, Rhi a, Rhi b. `two`: 2^(M + 1).
     E0  P1 = Blo + 2^23 Bhi
     E1  Vlo + eps - P0 - 2^25 Blo - l1 + 2^48 kappa = 0
     E2  Vhi - Bhi - P2 - kappa = 0
         sigma^2 = a^2 + 2^25 a b + 2^48 b^2, so E1 + 2^48 E2 with E0 is V + eps - sigma^2 = l1.
     E3  l1 + h1 = 2 sigma. With l1, h1 >= 0 this gives sigma^2 <= V + eps <= sigma^2 + 2 sigma.
     R1  P4 + P5 = q1          R2  P6 = 0
         R sigma = P3 + 2^24 (P4 + P5) + 2^48 P6.
     R3  2^(M+1) + sigma - s0 - 2 P3 - 2^25 q1 = 2 rr: 2 (2^M + floor(sigma / 2) - R sigma) = 2 rr.
     R4  rr + rq = sigma - 1. With rr, rq >= 0 this gives 0 <= rr < sigma.
   The digits' ranges give a, Rlo, Rhi < 2^24 and b < 2^15, so every product is below 2^48. With
   eps < 2^48 (`no-wrap-ledger-eps`) and M <= 47 (`KL-norm` for a layer's norms, `policy-norm-F` for
   the final norm) no term reaches 2^51. Honest rows meet R2: R sigma <= 2^M + sigma / 2 < 2^48. */
template <class T> static void ledLines5(const Row5<T> &r, const T P[7], Ex eps, const T &two, T out[8]) {
  typedef LedK<T> K;
  out[0] = P[1] - r.Blo - r.Bhi * eI(1 << 23);
  out[1] = r.Vlo + K::c(eps) - P[0] - r.Blo * eI(1 << 25) - r.l1 + r.kap * eI(1ll << 48);
  out[2] = r.Vhi - r.Bhi - P[2] - r.kap;
  out[3] = r.l1 + r.h1 - r.sig * eI(2);
  out[4] = P[4] + P[5] - r.q1;
  out[5] = P[6];
  out[6] = two + r.sig - r.s0 - P[3] * eI(2) - r.q1 * eI(1 << 25) - r.rr * eI(2);
  out[7] = r.rr + r.rq - r.sig + K::c(eO());
}
// The sum of om[j] P_j, the P_j formed from the sumcheck's end values as sent for a, b, Rlo, Rhi.
static Ex ledProducts5(const Ex *om, Ex a, Ex b, Ex lo, Ex hi) {
  return om[0] * a * a + om[1] * a * b + om[2] * b * b + om[3] * lo * a + om[4] * lo * b + om[5] * hi * a + om[6] * hi * b;
}
