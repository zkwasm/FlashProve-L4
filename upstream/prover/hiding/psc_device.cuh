// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The prover's arithmetic for the masked sumcheck (PSC). blocks/iop/psc.h defines the protocol:
// read it first. A sumcheck kernel computes what it would compute in a protocol that hides
// nothing; the functions here turn those values into the scalars that are sent.
//
// The sumcheck runs on U + eps * (summand). The summand is a polynomial in X and in maps F_j(X). U
// is a hiding polynomial (blocks/iop/hp.h) whose coefficients are entries of the committed mask row
// (random values). One further variable Y, read at 0 in all rounds but its own, carries an inner
// mask c_j Y on each shown map F_j: a map whose end value, at the challenge point, is sent, as
// F_j + beta c_j. eps and beta are challenges.
//
// The caller stores and sends the scalars and passes in the challenges, in the order given at
// `Stage`. Some faults of a caller are not caught, as noted at the declarations.
// No test of this repository compares these functions with the prover of blocks/iop/psc.h.
#pragma once
#include <cstddef> // size_t; <cstdlib> below is not always included
#ifndef HD
#define HD
#endif
// Two macros are for tests only. A PSC_STOP defined before this file replaces the stop below (a
// unit test that is not part of this repository makes it throw). PSC_CHECK_CLAIM=1 makes round()
// stop the program depending on hidden values (see `Form`). With SERVING_BUILD (the production
// build), either is a compile error; without it, an object file compiled with either contains the
// string of BLK_MARK_NOT_FOR_SERVING() (blocks/fence.h), by which a build script can recognise it.
// No test of this repository tests either of the two.
#ifdef SERVING_BUILD
#ifdef PSC_STOP
#error "SERVING_BUILD does not go with a PSC_STOP that is handed in: it takes the place of the stop at a call out of order, whose harm would be silent (a scalar over another, a mask entry used twice)"
#endif
#if defined(PSC_CHECK_CLAIM) && PSC_CHECK_CLAIM
#error "SERVING_BUILD does not go with PSC_CHECK_CLAIM=1, a test's build in which a round stops the program depending on hidden values"
#endif
#elif defined(PSC_STOP) || (defined(PSC_CHECK_CLAIM) && PSC_CHECK_CLAIM)
#include "fence.h"
BLK_MARK_NOT_FOR_SERVING();
#endif
// Stops the program. At a call out of order, or with a number of rounds or a degree out of range,
// the harm would otherwise be silent: a scalar stored over another, a mask entry used twice.
// On the GPU the stop aborts the kernel: the host gets an error from a later CUDA call.
#ifndef PSC_STOP
#ifdef __CUDA_ARCH__
#define PSC_STOP() __trap()
#else
#include <cstdlib>
#define PSC_STOP() abort()
#endif
#endif
namespace psc {
// F is the field's arithmetic (add, sub, mul, inv, eq, zero, one, small) on elements T. HD is the
// includer's function qualifiers, such as `__host__ __device__`, or empty.
//
// How a kernel describes a round's polynomial g, and how the variable is summed. c is the running
// claim, z the round's coordinate of the equality point (the sum is weighted by eq(z, X)). EQ2, EQ
// and EQC have equality weights (the sums with z), the others plain weights.
//   EQ2    degree 2; g(0) and the coefficient of X^2; c = (1 - z) g(0) + z g(1)
//   SUM2   degree 2; g(0) and the coefficient of X^2; c = g(0) + g(1)
//   SUM3   degree 3; g(0), g(-1) and the coefficient of X^3; c = g(0) + g(1)
//   NODES  degree d; g(0), g(2), .., g(d); c = g(0) + g(1)
//   EQ     degree d; g(0) and the coefficients of X^2, .., X^d; c = (1 - z) g(0) + z g(1).
//          (EQ2 is the case d = 2.)
//   EQC    degree d; the coefficients of X, X^2, .., X^d; summed as under EQ. These are the
//          coefficients that are sent, so nothing is recovered
//          from c: the running claim is not kept, no 1/z is made, and a coordinate 0 of the equality point is like any other. Same slots,
//          mask entries and scalars as EQ.
// The rounds of one sumcheck have one form; under NODES, EQ and EQC their degrees may differ. Two
// mixes are allowed. A round of degree 2 of a sumcheck that began as EQ may be run as EQ2. A round
// of one that began as EQ or EQ2 may be run as EQC, but then no later round as EQ or EQ2, since the
// claim is lost (round() stops the program). One that began as EQC stays EQC. `State` does not
// record the form: a mix such as a round by SUM2 after begin() by EQ2 is not caught.
// In a build with PSC_CHECK_CLAIM=1, for tests, a kernel passes g(0) too, after EQC's values, the
// claim is kept, and round() stops the program unless c = g(0) + z (g(1) - g(0)): a wrong table
// then shows at its round. (That build makes the 1/z again; the comments below ignore it.)
enum class Form : int { EQ2, SUM2, SUM3, NODES, EQ, EQC };
#ifndef PSC_CHECK_CLAIM
#define PSC_CHECK_CLAIM 0
#endif
constexpr int MAX_DEGREE = 20, MAX_ROUNDS = 64;
HD constexpr int degree_of(Form form, int degree) {
  return form == Form::SUM3 ? 3 : form == Form::NODES || form == Form::EQ || form == Form::EQC ? degree : 2;
}
HD constexpr bool equality_weights(Form form) {
  return form == Form::EQ2 || form == Form::EQ || form == Form::EQC;
}
// Does round() recover a coefficient from the running claim?
HD constexpr bool from_claim(Form form) {
  return form != Form::EQC;
}
// Scalars of a round = the polynomial's coefficients less the one that is not sent = its degree.
HD constexpr int round_scalars(Form form, int degree = 0) {
  return degree_of(form, degree);
}
// The number of slots, and of mask entries, of a sumcheck whose rounds all have one form and
// degree. `degree_in_y`: the summand's degree in the shown maps, 0 if none is shown.
//
// The caller's message array has three planes, `pad` elements apart: slots (one for each sent
// scalar), mask entries, scratch. A sumcheck's scalars take consecutive slots:
// mu' (the sum of U) | the rounds | the round in Y | the end values.
// It uses the mask entries with the same numbers, in this order: for each round the entries of X^1,
// X^2, ..; for the last variable (Y, or the last round if nothing is shown) also that of X^0,
// first; then the inner masks, so that c_j has the number of the slot of end value j. Scratch: eps
// at the slot of mu'; at a round's first slot 1/z (EQ2, EQ) or, under plain weights, the later
// variables' part of the sent constant (see round()); beta at the first slot of the round in Y.
// Not checked: `pad`; that the mask plane is filled before begin().
HD constexpr size_t slots(Form form, int rounds, size_t shown, int degree_in_y, int degree = 0) {
  return 1 + (size_t)rounds * round_scalars(form, degree) + degree_in_y + shown;
}

template <class T> struct State {
  T eps;            // joins the hiding polynomial to the claim
  T claim;          // the running claim of the sumcheck without masks, while claim_kept
  T beta;           // the challenge of Y
  T *slot;          // where the next scalar goes
  int rounds_left;  // rounds in X still to come
  int in_y;         // scalars of the round in Y: degree_in_y, then 0 from `took_beta` on
  int stage;        // Stage: which calls may come next
  int rounds;       // rounds in X in all
  int claim_kept;   // 0 from a round by EQC on
  unsigned char degree[MAX_ROUNDS];  // of each round's polynomial, as given to `begin`
};
// The order of the calls: begin(), took_eps(), (round(), bind()) for each round, then, if values
// are shown, further(), took_beta(), end_value(). Each of these, and inner_mask(), stops the
// program if it is called at another stage; round() and bind() also if the degree is not the one
// that begin() was given. begin() needs a `State` that is through(); a `State` of zeros is.
// One thread makes the calls that change a `State`: only begin_by_warp() synchronises.
enum Stage : int { NONE = 0, BEGUN, ROUNDS, SENT, IN_Y, ENDS };
template <class T> HD bool through(const State<T> &s) {
  return s.stage == NONE || s.stage == ENDS || (s.stage == ROUNDS && s.rounds_left == 0 && s.in_y == 0);
}

// The mask entries e of the variable whose first scalar goes to `slot`: e[t] belongs to X^t,
// t >= 1; e[0] only if the variable keeps its constant.
template <class T> HD const T *entries(const T *slot, long long pad, bool keeps_constant) {
  return slot + pad - (keeps_constant ? 1 : 2);
}

// Fills the state and the rounds' scratch and returns mu', to store at slots[0]. z_of_round(j): the
// equality point's coordinate of round j, read only under equality weights. degree_of_round(j): the
// degree of round j, used only under NODES, EQ and EQC. `claim`: the summand's true sum, without
// the masks that the starting claim may carry.
template <class F, class T, class Z, class D>
HD T begin(State<T> &s, Form form, T *slots, long long pad, int rounds, Z z_of_round, D degree_of_round, T claim,
           int degree_in_y) {
  if (!through(s) || rounds < 0 || rounds > MAX_ROUNDS || degree_in_y < 0 || degree_in_y > MAX_DEGREE) PSC_STOP();
  // Backwards over the variables. `after`: what the later variables' entries add to the round
  // polynomial of the present one.
  T *first = slots + 1;
  for (int j = 0; j < rounds; j++) first += round_scalars(form, degree_of_round(j));
  T after = degree_in_y ? entries(first, pad, true)[0] : F::zero();  // Y is summed by g -> g(0)
  for (int j = rounds - 1; j >= 0; j--) {
    const int d = degree_of(form, degree_of_round(j));
    if (d < 1 || d > MAX_DEGREE) PSC_STOP();
    s.degree[j] = (unsigned char)d;
    first -= d;
    const bool keeps = !degree_in_y && j == rounds - 1;
    const T *e = entries(first, pad, keeps);
    T high = F::zero();
    for (int t = 1; t <= d; t++) high = F::add(high, e[t]);
    if (equality_weights(form)) {  // phi(w) = w(0) + z (w(1) - w(0)), phi(1) = 1
      after = F::add(after, F::mul(z_of_round(j), high));
      if (keeps) after = F::add(after, e[0]);
    } else {  // phi(w) = w(0) + w(1), phi(1) = 2
      first[2 * pad] = after;
      after = F::add(F::add(after, after), high);
      if (keeps) after = F::add(after, F::add(e[0], e[0]));
    }
  }
  if (form == Form::EQC && PSC_CHECK_CLAIM) {  // the check needs 1/z; 0 for z = 0
    T *p = slots + 1;
    for (int j = 0; j < rounds; j++) {
      p[2 * pad] = F::eq(z_of_round(j), F::zero()) ? F::zero() : F::inv(z_of_round(j));
      p += degree_of(form, degree_of_round(j));
    }
  }
  if (equality_weights(form) && from_claim(form) && rounds) {
    // All 1/z with one inversion: the products z_0 .. z_(j-1) forwards, then backwards
    // (Montgomery's trick: P. L. Montgomery, "Speeding the Pollard and elliptic curve methods of
    // factorization", Math. Comp. 48 (1987), section 10.3.1).
    T run = F::one();
    T *p = slots + 1;
    for (int j = 0; j < rounds; j++) {
      p[2 * pad] = run;
      run = F::mul(run, z_of_round(j));
      p += degree_of(form, degree_of_round(j));
    }
    // A coordinate 0 (chance about rounds / |field|) leaves g(1) undetermined, since the claim is
    // then g(0) alone: the kernel's values do not give the round's polynomial, and no true proof
    // can be made from them.
    if (F::eq(run, F::zero())) PSC_STOP();
    T inverse = F::inv(run);
    for (int j = rounds - 1; j >= 0; j--) {
      p -= degree_of(form, degree_of_round(j));
      p[2 * pad] = F::mul(inverse, p[2 * pad]);
      inverse = F::mul(inverse, z_of_round(j));
    }
  }
  s.claim = claim;
  s.claim_kept = from_claim(form) || PSC_CHECK_CLAIM;
  s.slot = slots + 1;
  s.rounds_left = s.rounds = rounds;
  s.in_y = degree_in_y;
  s.stage = BEGUN;
  return after;
}
#ifdef __CUDACC__
// begin() under equality weights, by the 32 lanes of a warp: all call it with the same
// arguments, and s is in memory that they all see. Under these weights each round adds a term of
// its own to mu', so lane l takes rounds l and l + 32, with an inversion for each (none under EQC).
// The state, the scratch and the value are those of begin(); every lane gets the value.
// Not checked: that all 32 lanes call it; that threadIdx.x & 31 is the lane (a 1-D block).
template <class F, class T, class Z, class D>
__device__ T begin_by_warp(State<T> &s, Form form, T *slots, long long pad, int rounds, Z z_of_round, D degree_of_round, T claim,
                           int degree_in_y) {
  const int lane = threadIdx.x & 31;
  if (!equality_weights(form) || !through(s) || rounds < 0 || rounds > MAX_ROUNDS || degree_in_y < 0 || degree_in_y > MAX_DEGREE) PSC_STOP();
  __syncwarp();  // every lane has read the state before one writes to it
  T part = F::zero();
  int all = 0;  // the scalars of all rounds
  for (int j = 0; j < rounds; j++) {
    const int d = degree_of(form, degree_of_round(j));
    if (d < 1 || d > MAX_DEGREE) PSC_STOP();
    all += d;
  }
  for (int j = lane; j < rounds; j += 32) {
    int before = 0;  // the scalars of the rounds before round j
    for (int i = 0; i < j; i++) before += degree_of(form, degree_of_round(i));
    const int d = degree_of(form, degree_of_round(j));
    s.degree[j] = (unsigned char)d;
    T *first = slots + 1 + before;
    const bool keeps = !degree_in_y && j == rounds - 1;
    const T *e = entries(first, pad, keeps);
    T high = F::zero();
    for (int t = 1; t <= d; t++) high = F::add(high, e[t]);
    const T z = z_of_round(j);
    part = F::add(part, F::mul(z, high));
    if (keeps) part = F::add(part, e[0]);
    if (from_claim(form)) {
      // A coordinate 0 leaves g(1) undetermined: see begin().
      if (F::eq(z, F::zero())) PSC_STOP();
      first[2 * pad] = F::inv(z);
    } else if (PSC_CHECK_CLAIM) {
      first[2 * pad] = F::eq(z, F::zero()) ? F::zero() : F::inv(z);
    }
  }
  __syncwarp();
  if (lane == 0 && degree_in_y) part = F::add(part, entries(slots + 1 + all, pad, true)[0]);
  static_assert(sizeof(T) % 8 == 0, "an element is whole 64-bit words");
  for (int o = 16; o; o >>= 1) {
    T other = part;
    unsigned long long *w = reinterpret_cast<unsigned long long *>(&other);
    for (unsigned i = 0; i < sizeof(T) / 8; i++) w[i] = __shfl_xor_sync(~0u, w[i], o);
    part = F::add(part, other);
  }
  if (lane == 0) {
    s.claim = claim;
    s.claim_kept = from_claim(form) || PSC_CHECK_CLAIM;
    s.slot = slots + 1;
    s.rounds_left = s.rounds = rounds;
    s.in_y = degree_in_y;
    s.stage = BEGUN;
  }
  __syncwarp();
  return part;
}
#endif
template <class T> HD void took_eps(State<T> &s, long long pad, T eps) {
  if (s.stage != BEGUN) PSC_STOP();
  s.stage = ROUNDS;
  s.eps = eps;
  s.slot[2 * pad - 1] = eps;
}

// v: the kernel's values in the order of `Form`. `sent`: the round_scalars() scalars, to
// store at s.slot. c: the true coefficients, which bind() needs (degree + 1 of them; under EQC c[0]
// is not known and is set to 0).
//
// The coefficient that the verifier's check fixes is not sent: the constant under equality weights
// and for Y, the linear one under plain weights. Everything in a round polynomial of U that depends
// on earlier challenges, and the term of the starting claim's own masks, sits in just that
// coefficient. So
//  equality weights, Y: sent coefficient of X^t = (entry of X^t) + eps * (true coefficient), t >= 1
//  plain weights:       the same for t >= 2; sent constant = (a combination of the later variables'
//                       entries; for the last variable its entry of X^0) + eps * (true constant)
template <class F, class T> HD void round(State<T> &s, Form form, long long pad, const T *v, int degree, T *sent, T *c) {
  const int d = degree_of(form, degree);
  if (s.stage != ROUNDS || s.rounds_left <= 0 || d != (int)s.degree[s.rounds - s.rounds_left]) PSC_STOP();
  s.stage = SENT;
  const bool keeps = !s.in_y && s.rounds_left == 1;
  const T *e = entries(s.slot, pad, keeps);
  if (form == Form::EQC) {
    c[0] = F::zero();
    for (int t = 1; t <= d; t++) c[t] = v[t - 1], sent[t - 1] = F::add(e[t], F::mul(s.eps, c[t]));
#if PSC_CHECK_CLAIM
    T rise = F::zero();  // g(1) - g(0)
    for (int t = 1; t <= d; t++) rise = F::add(rise, c[t]);
    c[0] = v[d];
    const T inverse = s.slot[2 * pad], gap = F::sub(s.claim, c[0]);  // c - g(0) = z (g(1) - g(0))
    if (!s.claim_kept || !(F::eq(inverse, F::zero()) ? F::eq(gap, F::zero()) : F::eq(F::mul(gap, inverse), rise))) PSC_STOP();
#endif
    return;
  }
  if (!s.claim_kept) PSC_STOP();
  const T scratch = s.slot[2 * pad];
  if (equality_weights(form)) {
    c[0] = v[0];
    c[1] = F::mul(F::sub(s.claim, v[0]), scratch);  // g(1) - g(0) = (claim - g(0)) / z
    for (int t = 2; t <= d; t++) c[t] = v[t - 1], c[1] = F::sub(c[1], c[t]);
    for (int t = 1; t <= d; t++) sent[t - 1] = F::add(e[t], F::mul(s.eps, c[t]));
    return;
  }
  if (form == Form::SUM2) {
    c[0] = v[0], c[2] = v[1];
  } else if (form == Form::SUM3) {  // g(-1) + g(1) = 2 c0 + 2 c2 and g(1) = claim - c0
    c[0] = v[0], c[3] = v[2];
    c[2] = F::sub(F::mul(F::sub(F::add(v[1], s.claim), v[0]), F::inv(F::small(2))), v[0]);
  } else {
    // Newton's divided differences on the nodes 0 .. d, then the change to powers of X.
    T y[MAX_DEGREE + 1];
    y[0] = v[0], y[1] = F::sub(s.claim, v[0]);
    for (int k = 2; k <= d; k++) y[k] = v[k - 1];
    for (int step = 1; step <= d; step++) {
      const T by = F::inv(F::small(step));
      for (int i = d; i >= step; i--) y[i] = F::mul(F::sub(y[i], y[i - 1]), by);
    }
    for (int i = 0; i <= d; i++) c[i] = F::zero();
    for (int i = d; i >= 0; i--) {  // c = y[i] + (X - i) c
      const T node = F::small(i);
      for (int t = d; t >= 1; t--) c[t] = F::sub(c[t - 1], F::mul(node, c[t]));
      c[0] = F::sub(y[i], F::mul(node, c[0]));
    }
  }
  if (form != Form::NODES) {  // the linear coefficient, from the claim
    T rest = F::add(c[0], c[0]);
    for (int t = 2; t <= d; t++) rest = F::add(rest, c[t]);
    c[1] = F::sub(s.claim, rest);
  }
  sent[0] = F::add(scratch, F::mul(s.eps, c[0]));
  if (keeps) sent[0] = F::add(sent[0], e[0]);
  for (int t = 2; t <= d; t++) sent[t - 1] = F::add(e[t], F::mul(s.eps, c[t]));
}
// r: the round's challenge. c: as round() left it.
template <class F, class T> HD void bind(State<T> &s, Form form, int degree, const T *c, T r) {
  const int d = degree_of(form, degree);
  if (s.stage != SENT || d != (int)s.degree[s.rounds - s.rounds_left]) PSC_STOP();
  s.stage = ROUNDS;
  if (from_claim(form) || PSC_CHECK_CLAIM) {
    T g = c[d];
    for (int t = d - 1; t >= 0; t--) g = F::add(F::mul(g, r), c[t]);
    s.claim = g;
  } else {
    s.claim_kept = 0;
  }
  s.slot += d;
  s.rounds_left--;
}

// The inner mask c_j of shown map j. To be called after the last round in X. j is not checked, here
// or in end_value(): `State` does not record how many maps are shown.
template <class T> HD T inner_mask(const State<T> &s, long long pad, int j) {
  if (!(s.stage == ROUNDS && s.rounds_left == 0 && s.in_y > 0) && s.stage != IN_Y && s.stage != ENDS) PSC_STOP();
  return s.slot[pad + s.in_y + j];
}
// The round in Y. in_y[t - 1]: the coefficient of Y^t in the summand at the challenge point with
// F_j + c_j Y for each shown map, t = 1 .. s.in_y. `sent`: s.in_y scalars, to store at s.slot.
template <class F, class T> HD void further(State<T> &s, long long pad, const T *in_y, T *sent) {
  if (s.stage != ROUNDS || s.rounds_left != 0 || s.in_y <= 0) PSC_STOP();
  s.stage = IN_Y;
  for (int t = 1; t <= s.in_y; t++) sent[t - 1] = F::add(s.slot[pad + t - 1], F::mul(s.eps, in_y[t - 1]));
}
// beta is not checked: with beta = 0 (the protocol draws it nonzero) end_value() adds no mask.
template <class T> HD void took_beta(State<T> &s, long long pad, T beta) {
  if (s.stage != IN_Y) PSC_STOP();
  s.stage = ENDS;
  s.beta = beta;
  s.slot[2 * pad] = beta;
  s.slot += s.in_y;
  s.in_y = 0;
}
// End value j as it is sent, to store at s.slot[j].
template <class F, class T> HD T end_value(const State<T> &s, long long pad, T value, int j) {
  if (s.stage != ENDS) PSC_STOP();
  return F::add(value, F::mul(s.beta, s.slot[pad + j]));
}
// in_y for the product of two shown maps with the values a and b: (a + c0 Y)(b + c1 Y) without its
// constant.
//
// Sources for the masked end values and for the product in Y: Cramer and Damgard, "Zero-Knowledge
// Proofs for Finite Field Arithmetic, or: Can Zero-Knowledge Be for Free?", CRYPTO '98, LNCS 1462,
// pp. 424-441, section 2.3; Maurer, "Unifying Zero-Knowledge Proofs of Knowledge", AFRICACRYPT
// 2009, LNCS 5580, pp. 272-286, section 6.7; QuickSilver (ePrint 2021/076), section 3.1. Here an
// end value is sent as value + beta * (its inner mask, an entry of the committed mask row), under
// one challenge beta for all end values of the sumcheck. The product of two such values, with Y
// for beta, is a polynomial of degree 2 in Y. The prover sends eps times its coefficients of Y and
// Y^2, each plus a separate mask entry, before beta is drawn.
template <class F, class T> HD void product_in_y(T a, T b, T c0, T c1, T *in_y) {
  in_y[0] = F::add(F::mul(a, c1), F::mul(b, c0));
  in_y[1] = F::mul(c0, c1);
}

// ---- After the sumcheck: the linear claim (`row`) that it leaves ----
// The value of the hiding polynomial at the challenges, as a linear form in its entries: out[i] is
// the coefficient of the sumcheck's i-th mask entry (the inner masks not counted). `out` needs
// 1 + degree_in_y + (the rounds' scalars) elements: not checked. r_of_round(j): the challenge of
// round j. Returns eps * prod_j psi_j(r_j), the factor of the starting claim's own masks in the
// linear claim; psi = 1 under equality weights and for Y, psi = X under plain weights.
// psc_rows() in blocks/iop/psc.h builds the whole claim.
template <class F, class T, class R, class D>
HD T row_form(Form form, int rounds, R r_of_round, D degree_of_round, int degree_in_y, T eps, T beta, T *out) {
  int n = 1 + degree_in_y;
  for (int j = 0; j < rounds; j++) n += round_scalars(form, degree_of_round(j));
  T tail = F::one();  // prod of psi over the variables after the present one
  int at = n;
  if (degree_in_y) {
    at -= degree_in_y + 1;
    T power = F::one();
    for (int t = 0; t <= degree_in_y; t++, power = F::mul(power, beta)) out[at + t] = power;
  }
  for (int j = rounds - 1; j >= 0; j--) {
    const int d = degree_of(form, degree_of_round(j));
    const bool keeps = !degree_in_y && j == rounds - 1;
    const T r = r_of_round(j);
    at -= keeps ? d + 1 : d;
    T power = tail;
    if (keeps) out[at] = power;
    for (int t = 1; t <= d; t++) {
      power = F::mul(power, r);
      out[at + (keeps ? t : t - 1)] = power;
    }
    if (!equality_weights(form)) tail = F::mul(tail, r);
  }
  return F::mul(eps, tail);
}
}  // namespace psc
