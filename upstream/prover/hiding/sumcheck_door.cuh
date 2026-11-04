// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Namespace sc_door: the functions through which a sub-proof's sumchecks send; a sub-proof is the
// argument for one part of the forward pass. A kernel computes a round's values, or its arrays' end
// values (their values at the challenge point), unmasked, and continues with them. A function here
// masks them, stores them in the message array, absorbs them into the Fiat-Shamir transcript `Tr`
// and returns the challenge. No kernel is meant to send such a value itself, or to mask one, except
// a value that it passes to in_the_clear(). The protocol is the masked sumcheck (PSC, `psc` in
// identifiers; blocks/iop/psc.h): a sumcheck on U + eps * summand, U a random hiding polynomial.
// Its arithmetic: prover/hiding/psc_device.cuh. The calls, in order (some also by a warp's lanes):
//   begin()         sends the hiding polynomial's sum; returns the challenge eps
//   round()         sends one round's message; returns the round's challenge
//   inner_mask()    returns c_j, the mask of end value j, which the caller needs for further()
//   further()       sends the round in the extra variable Y; returns the challenge beta
//   end(), ends()   send end value j as (end value j) + beta c_j; after end(), call ends_sent()
// claim(), claims() and claim_alone() send a value outside a sumcheck as value + mask entry (see
// PADOFF); in_the_clear() sends a scalar as it is. The state of the running sumcheck is `Tr::p`.
// Only prover/hiding/transcript.cuh includes this file, after `Tr`, `T` (an element of the
// extension field) and `F` (the field's arithmetic) are defined. The functions that send reach the
// transcript through a few routines (see "The transcript, as the door uses it").
#pragma once
#include <type_traits>
// A sub-proof's message array has one slot for each scalar that it sends. The mask entry (`pad` in
// identifiers) with a slot's number, one random element of the committed mask row, is PADOFF
// elements after the slot, and a scratch cell 2 PADOFF after it (prover/hiding/psc_device.cuh gives
// the layout). padv_again() and pad_distance() trap if PADOFF is 0 (not set). No function here
// checks that the cell PADOFF after a slot contains a mask entry. The prover program cp_prove
// compares each sub-proof's mask entries in that place with their derivation after their last use
// (hz::Tree::slot_pads_agree(), prover/hiding/trees.cuh).
// Public source: CUTLASS's planar complex layout, two parts an offset apart: NVIDIA, CUTLASS
// documentation (4.4.2), 'CUTLASS Terminology', entry 'Planar Complex'.
// Public source: a global variable written with cudaMemcpyToSymbol: NVIDIA, CUDA C++ Programming
// Guide, section 'Using TMA to transfer multi-dimensional arrays', paragraph 'Host-to-device
// transfer'.
// See also: a global variable in place of a kernel parameter: NVIDIA, CUDA Programming Guide,
// "Asynchronous Data Copies", "Using TMA to transfer multi-dimensional arrays",
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html.
// Public source: cuBLASLt's one pointer and one distance between planes: NVIDIA, cuBLAS documentation
// (https://docs.nvidia.com/cuda/cublas/), cublasLtMatrixLayoutAttribute_t, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET.
__device__ long long PADOFF = 0;
// Each program that compiles this file has its own PADOFF. cp_prove writes PAD_STRIDE to it once,
// at set-up (prover/prover/inputs.cuh). One distance thus serves all its sub-proofs, although their
// numbers of slots differ. A sub-proof's mask row may have at most PAD_STRIDE coefficients, which
// each sub-proof's set-up checks.
static constexpr size_t PAD_STRIDE = (size_t)1 << 18;
// Returns v + (the mask entry of the slot at q). Writes nothing.
__device__ __forceinline__ T padv_again(T v, const T *q) {
  const long long o = PADOFF;
  if (!o) __trap();
  return F::add(v, q[o]);
}
// In a build with SLOT_COUNT=1, adds one to counter `word` of the slot at q: one of the three
// 64-bit words of the element 3 PADOFF after it. Only word 0, the number of writes to the slot, is
// used.
// Public sources for the counting: see the comment at the top of prover/hiding/slot_count.cuh,
// which clears and reads the counters.
__device__ __forceinline__ void slot_count(const T *q, int word) {
#if SLOT_COUNT
  atomicAdd((unsigned long long *)(q + 3 * PADOFF) + word, 1ull);
#else
  (void)q, (void)word;
#endif
}
// A build that traces the prover's time may define DOOR_MARK(site, round); nothing in this
// repository does. It is expanded in the thread that masks a message, before the message is hashed.
// round counts from 0 and is -1 where it is not known.
#ifndef DOOR_MARK
#define DOOR_MARK(site, round)
#endif
namespace sc_door {
// What a sent scalar is. ROUND: any scalar of a sumcheck that is not an end value (END). CLAIM:
// masked by claim(). CLEAR: sent by in_the_clear(). No code in this repository reads the kind.
enum class Kind : int { ROUND, END, CLAIM, CLEAR };
// How a kernel's values describe a round's polynomial g, and how the variable is summed: under EQ2,
// EQ and EQC with equality weights, (1 - z) g(0) + z g(1), z being the round's coordinate of the
// equality point; under SUM2, SUM3 and NODES as g(0) + g(1). psc_device.cuh lists each form's
// values.
using Form = psc::Form;
// The verifier's message after the prover's: none, or one challenge. COIN and CHAL have the same
// effect; the argument `round` of answer() and message() says which round it is.
enum class Draw : int { NONE, COIN, CHAL };
// Which code sends. Only a DOOR_MARK that a tracing build defines reads it. A product layer is one
// layer of the lookup argument's grand-product check (product_test.cuh). No code in this repository
// passes ROOT_ARRAYS, ROOT_TEST, OPENING_MERGE or OPENING_CLOSE.
enum class Site : int {
  LAYER_GRAM,     // a product layer, three rounds from a Gram matrix
  LAYER_TABLE,    // a product layer of the table side, one of its first three rounds
  LAYER_LATE,     // a product layer's late rounds and its two end values
  FOREST_ROOTS,   // the tie of a lookup's two roots
  ROOT_ARRAYS,    // the root test's arrays
  ROOT_TEST,      // the root test's rounds
  TOKEN,          // the token sumcheck
  TOKEN_SUMS,     // sums over the tokens from which token sumchecks start
  PAIRS,          // the sumcheck over many pairs of arrays: the batched inner products, the merge of weight claims
  WEIGHT_CLAIMS,  // the values of the claims about the weights
  ATTENTION_S3,   // attention's sumcheck of degree 3
  ATTENTION_SCD,  // attention's sumcheck whose degree differs by round
  ATTENTION_SENT, // attention's other hidden values
  OPENING_MERGE,  // the opening's messages: the merge of a request's claims
  OPENING_CLOSE,  // the closing sumcheck of the opening, of a link and of the range file
};
// The number of scalars in a round's message.
HD constexpr int round_values(Form form, int degree = 0) {
  return psc::round_scalars(form, degree);
}
// slots(form, rounds, end_values, degree_in_y, degree = 0): the number of slots, and of mask
// entries, of a sumcheck with `rounds` rounds of one form and one degree (degree is read for NODES,
// EQ and EQC only). degree_in_y: the summand's degree in the maps whose end values are sent; 0
// exactly if end_values is 0.
using psc::slots;
// The same for rounds whose degrees differ (NODES, EQ, EQC): degree[j] is the degree of round j.
HD size_t slots_degrees(const int *degree, int rounds, size_t end_values, int degree_in_y) {
  size_t n = 1 + (size_t)degree_in_y + end_values;
  for (int j = 0; j < rounds; j++) n += (size_t)degree[j];
  return n;
}
// The number of slots of a product layer's sumcheck with mu variables: the hiding polynomial's
// sum, two scalars a round, two for the round in Y, two end values.
HD constexpr size_t layer_slots(int mu) {
  return slots(Form::EQ2, mu, 2, 2);
}
static inline void cheat_from_the_environment() {}      // Six test hooks, down to cheat_is(), for
static inline void cheat_again() {}                     // a test build that is not part of this
static inline void cheat_say_seen(const char *, int) {} // repository. In every build of this
static inline bool cheat_struck() {                     // repository each is empty or returns
  return false;                                         // false, so code that needs a true result
}                                                       // never runs. Called at 13 places.
__device__ __forceinline__ bool cheat_at_layer_end(int, T, T) { // a test hook: always false here
  return false;
}
__device__ __forceinline__ constexpr bool cheat_is(int) { // a test hook: always false here
  return false;
}
// Copies a transcript object, the state of its running sumcheck included, by 'lanes' threads, of
// which the caller is number 'lane'. No barrier: the caller synchronises these threads before *to
// is read.
__device__ __forceinline__ void tr_copy_by_lanes(Tr *to, const Tr *from, int lane, int lanes) {
  static_assert(sizeof(Tr) % 4 == 0, "a transcript object is a whole number of words");
  for (int i = lane; i < (int)(sizeof(Tr) / 4); i += lanes) ((u32 *)to)[i] = ((const u32 *)from)[i];
}
// The same from global memory that another block of the same launch may have written (the loads
// bypass the L1 cache: __ldcg, CUDA Programming Guide, "C/C++ Language Extensions", "Load Functions
// Using Cache Hints",
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).
// The loads order nothing: the caller puts a grid barrier after those writes.
__device__ __forceinline__ void tr_fetch_by_lanes(Tr *to, const Tr *from, int lane, int lanes) {
  for (int i = lane; i < (int)(sizeof(Tr) / 4); i += lanes) ((u32 *)to)[i] = __ldcg((const u32 *)from + i);
}
// The distance from a slot to its mask entry. The kernel traps if it is not set.
__device__ __forceinline__ long long pad_distance() {
  const long long o = PADOFF;
  if (!o) __trap();
  return o;
}
// Stores a scalar in its slot and returns it; every function here that sends calls it. bare: the
// scalar without any mask, which no code in this repository reads.
template <Kind KIND> __device__ __forceinline__ T leave(Site site, T bare, T sent, T *slot) {
  (void)site, (void)bare;
  // A mask entry hides only one scalar, so each slot must be written at most once in a sub-proof.
  // Whether that holds depends only on the addresses. Every run of cp_prove checks what is sent: a
  // sub-proof's scalars are absorbed from consecutive slots (tr_open(),
  // prover/hiding/transcript_shell.cuh, traps otherwise), and ChainRun::close()
  // (prover/prover/chain_run.cuh) requires as many scalars as mask entries. Only a build with
  // SLOT_COUNT=1 counts the writes to each slot. That the addresses depend only on the shape, not
  // on a request's values, is not tested.
  // A sumcheck's mask entries have the numbers of its slots, so an entry is used once if its slot
  // is written once and the calls match what begin() was given. prover/hiding/psc_device.cuh stops
  // the program on a call at the wrong stage, one round too many or a round of another degree. It
  // does not know how many end values are sent, or in which form the sumcheck began. So it does
  // not stop an end value with too high a number. It stops a round of the right degree in a form
  // that psc::Form does not allow there only if the round needs the running claim and EQC has
  // dropped it. The caller must avoid both. A round in another form can send a scalar without a
  // mask entry: SUM2 after begin() by EQ2 sends 1/z + eps g(0). An end value with too high a number
  // is sent as value + beta times the cell PADOFF after its slot. Within the sub-proof's slots that
  // cell is a mask entry meant for a later scalar, so it can be used twice; beyond them it is 0 in
  // cp_prove, and the value is stored and absorbed unmasked. tr_open() or ChainRun::close() then
  // stops the run, because the sub-proof's slots no longer follow one another or differ in number
  // from its mask entries.
  slot_count(slot, 0);
  *slot = sent;
  return sent;
}
// ---- The transcript, as the door uses it ----
// `Tr` wraps a hash chain of the proof (prover/hiding/transcript_shell.cuh).
//   absorb()             adds scalars that are already stored in slots to the prover's message
//   answer()             ends that message and returns the verifier's message, one element; with
//                        Draw::NONE it does nothing and returns 0
//   answer_not_zero()    the same for the round PSC_Y, whose element is never 0
//   message()            absorb() and answer() for N scalars that the caller also passes in sent
//   message2_by_lanes()  message() for two scalars, by SEND2_N lanes of a warp (a constant defined
//                        with `Tr`); one warp of a block at a time
//   message_by_lanes()   the same for N scalars, read from lane 0 (N = 0 to 3 are in use); every
//                        lane gets the answer
//   values_by_lanes()    N end values as the prover's message, by the same lanes; the verifier's
//                        message is empty
//   tr_empty()           ends the prover's message; the verifier's is empty (defined with `Tr`)
// round: which round of which protocol block this is (`Round`). It is logged, and the host later
// compares the log with the plan, the list of the proof's rounds (Stretch::end()). The answer of
// the round PSC_Y is never 0, whichever function returns it.
// A lane's number is taken as threadIdx.x & 31: right if the block is one-dimensional; not checked.
__device__ __forceinline__ void absorb(Tr &t, const T *slots, int n) {
  tr_open(t, slots, n);
}
template <Draw DRAW> __device__ __forceinline__ T answer(Tr &t, Round round) {
  if constexpr (DRAW == Draw::NONE)
    return F::zero();
  else
    return tr_answer(t, round);
}
__device__ __forceinline__ T answer_not_zero(Tr &t, Round round) {
  return tr_answer_not_zero(t, round);
}
template <Draw DRAW, int N> __device__ __forceinline__ T message(Tr &t, Round round, const T *sent, const T *slots) {
  static_assert(DRAW != Draw::NONE, "a message without an answer is absorbed");
  return tr_round<N>(t, round, sent, slots);
}
__device__ __forceinline__ T message2_by_lanes(Tr *t, Round round, T a, T b, const T *slots) {
  return tr_round2_by_lanes(t, round, a, b, slots);
}
template <Draw DRAW, int N> __device__ __forceinline__ T message_by_lanes(Tr *t, Round round, const T *sent, const T *slots) {
  static_assert(DRAW != Draw::NONE, "a message without an answer is absorbed");
  return tr_round_by_lanes<N>(t, round, sent, slots);
}
template <int N> __device__ __forceinline__ void values_by_lanes(Tr *t, const T *sent, const T *slots) {
  tr_empty_by_lanes<N>(t, Round::PSC_VALUES, sent, slots);
}

// ---- A sumcheck ----
// Begins a sumcheck: sends the hiding polynomial's sum, which ends the prover's message, and
// returns the round's challenge, eps. slots: the sumcheck's first slot. z_of_round(j): the
// coordinate of the equality point that belongs to round j; called only for EQ2, EQ and EQC.
// degree_of_round(j): used only for NODES, EQ and EQC. claim: the true value of the claimed sum.
// EQ2, EQ, SUM3 and NODES need it to recover a round's polynomial from the kernel's values; a
// sumcheck of SUM2 rounds may pass anything. degree_in_y: see slots(). One thread. eps must not be
// known before the point, the claim and the sum are fixed, so it is drawn here and no caller can
// supply it.
template <Form FORM, class Z, class D>
__device__ __forceinline__ T begin_degrees(Site site, Tr &t, T *slots, int rounds, Z z_of_round, D degree_of_round, T claim,
                                           int degree_in_y) {
  const long long pad = pad_distance();
  DOOR_MARK(site, -1);
  T mu = psc::begin<F>(t.p, FORM, slots, pad, rounds, z_of_round, degree_of_round, claim, degree_in_y);
  mu = leave<Kind::ROUND>(site, mu, mu, slots);
  const T eps = message<Draw::CHAL, 1>(t, Round::PSC_EPSILON, &mu, slots);
  psc::took_eps(t.p, pad, eps);
  return eps;
}
// As begin_degrees(). degree: one number for all rounds, or a functor degree_of_round(j).
template <Form FORM, class Z, class D = int>
__device__ __forceinline__ T begin(Site site, Tr &t, T *slots, int rounds, Z z_of_round, T claim, int degree_in_y,
                                   D degree = 0) {
  if constexpr (std::is_integral<D>::value)
    return begin_degrees<FORM>(site, t, slots, rounds, z_of_round, [degree](int) { return (int)degree; }, claim, degree_in_y);
  else
    return begin_degrees<FORM>(site, t, slots, rounds, z_of_round, degree, claim, degree_in_y);
}
// begin() for the forms with equality weights, called by all 32 lanes of a warp with the same
// arguments (psc::begin_by_warp()), one warp of a block at a time. *t must be in memory that they
// all see. Lanes 0 to SEND2_N - 1 return eps, the others 0.
// Public sources: see round_by_lanes().
template <Form FORM, class Z, class D = int>
__device__ __forceinline__ T begin_by_warp(Site site, Tr *t, T *slots, int rounds, Z z_of_round, T claim, int degree_in_y, D degree = 0) {
  const long long pad = pad_distance();
  T mu;
  if constexpr (std::is_integral<D>::value)
    mu = psc::begin_by_warp<F>(t->p, FORM, slots, pad, rounds, z_of_round, [degree](int) { return (int)degree; }, claim, degree_in_y);
  else
    mu = psc::begin_by_warp<F>(t->p, FORM, slots, pad, rounds, z_of_round, degree, claim, degree_in_y);
  T eps = F::zero();
  if ((threadIdx.x & 31) < SEND2_N) {
    const bool leader = (threadIdx.x & 31) == 0;
    if (leader) {
      DOOR_MARK(site, -1);
      mu = leave<Kind::ROUND>(site, mu, mu, slots);
    }
    eps = message_by_lanes<Draw::CHAL, 1>(t, Round::PSC_EPSILON, &mu, slots);
    if (leader) psc::took_eps(t->p, pad, eps);
  }
  __syncwarp();
  return eps;
}
// A sumcheck with no variable except Y, by the first SEND2_N lanes of a warp, which make each call
// together, one warp of a block at a time. The values are read from lane 0. *t must be in memory
// that the lanes share. None of the three returns a challenge.
//   begin_by_lanes()       begin() with no round
//   further_by_lanes<N>()  further(); N is the degree_in_y that begin_by_lanes() was given
//   ends_by_lanes<N>()     ends<Draw::NONE>() for N end values
// Public sources: see round_by_lanes().
__device__ __forceinline__ void begin_by_lanes(Site site, Tr *t, T *slots, T claim, int degree_in_y) {
  const long long pad = pad_distance();
  const bool leader = (threadIdx.x & 31) == 0;
  T mu = F::zero();
  if (leader) {
    DOOR_MARK(site, -1);
    mu = psc::begin<F>(t->p, Form::EQ2, slots, pad, 0, [](int) { return F::zero(); }, [](int) { return 0; }, claim, degree_in_y);
    mu = leave<Kind::ROUND>(site, mu, mu, slots);
  }
  const T eps = message_by_lanes<Draw::CHAL, 1>(t, Round::PSC_EPSILON, &mu, slots);
  if (leader) psc::took_eps(t->p, pad, eps);
  __syncwarp((1u << SEND2_N) - 1);
}
// Sends one round's message and returns the round's challenge. v: the kernel's values, in the order
// that psc::Form gives. degree is read for Form::NODES, Form::EQ and Form::EQC only. One thread.
template <Form FORM, Draw DRAW>
__device__ __forceinline__ T round(Site site, int round_number, Tr &t, const T *v, int degree = 0) {
  constexpr bool BY_ROUND = FORM == Form::NODES || FORM == Form::EQ || FORM == Form::EQC; // the caller passes the degree
  constexpr int MOST = BY_ROUND ? psc::MAX_DEGREE : psc::degree_of(FORM, 0);
  const long long pad = pad_distance();
  DOOR_MARK(site, round_number);
  (void)round_number;
  T sent[MOST], c[MOST + 1];
  psc::round<F>(t.p, FORM, pad, v, degree, sent, c);
  const int n = round_values(FORM, degree);
  for (int i = 0; i < n; i++) sent[i] = leave<Kind::ROUND>(site, c[psc::equality_weights(FORM) || i ? i + 1 : 0], sent[i], t.p.slot + i);
  T r;
  if constexpr (BY_ROUND) {
    absorb(t, t.p.slot, n);
    r = answer<DRAW>(t, Round::PSC_ROUND);
  } else {
    r = message<DRAW, MOST>(t, Round::PSC_ROUND, sent, t.p.slot);
  }
  psc::bind<F>(t.p, FORM, degree, c, r);
  return r;
}
// round() for a form with two scalars, called together by the first SEND2_N lanes of a warp, one
// warp of a block at a time. a and b, the kernel's values, are read from lane 0. Every lane gets
// the challenge. *t must be in memory that the lanes share. Lane 0 writes *t after the call's last
// barrier, so the other lanes must synchronise before they read it.
// Public source (the idea is public): a leader lane acts once for all lanes: NVIDIA Technical Blog,
// "Using CUDA Warp-Level Primitives",
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/,
// "Opportunistic Warp-level Programming".
// Public source (the idea is public): in OpenMP, the COPYPRIVATE clause of the SINGLE directive:
// "OpenMP Directives: Data Scope Attribute Clauses: COPYPRIVATE Clause",
// https://hpc-tutorials.llnl.gov/openmp/copyprivate_clause/. Here the lane is fixed.
// Form::EQC: a round of degree 2. at_0: g(0), read only in a build with PSC_CHECK_CLAIM.
template <Form FORM>
__device__ __forceinline__ T round_by_lanes(Site site, int round_number, Tr *t, T a, T b, T at_0 = F::zero()) {
  static_assert(round_values(FORM, 2) == 2, "the lanes send two scalars");
  T sent[2] = {F::zero(), F::zero()}, c[3];
  (void)round_number;
  const bool leader = (threadIdx.x & 31) == 0;
  if (leader) {
    const T v[3] = {a, b, at_0};
    psc::round<F>(t->p, FORM, pad_distance(), v, 2, sent, c);
    DOOR_MARK(site, round_number);
    sent[0] = leave<Kind::ROUND>(site, c[psc::equality_weights(FORM) ? 1 : 0], sent[0], t->p.slot);
    sent[1] = leave<Kind::ROUND>(site, c[2], sent[1], t->p.slot + 1);
  }
  const T r = message2_by_lanes(t, Round::PSC_ROUND, sent[0], sent[1], t->p.slot);
  if (leader) psc::bind<F>(t->p, FORM, 2, c, r);
  return r;
}
// The inner mask c_j of end value j. Valid once all rounds but the one in Y are done; any thread,
// any copy of the `Tr`.
__device__ __forceinline__ T inner_mask(const Tr &t, int j) {
  return psc::inner_mask(t.p, pad_distance(), j);
}
// Sends the round in the extra variable Y. in_y[i]: the coefficient of Y^(i + 1) of the summand at
// the challenge point, with (end value j) + inner_mask(j) Y in place of each map whose end value is
// sent. Returns beta, the verifier's answer, from answer_not_zero(): beta = 0 would send the end
// values unmasked. One thread.
__device__ __forceinline__ T further(Site site, Tr &t, const T *in_y) {
  const long long pad = pad_distance();
  DOOR_MARK(site, -1);
  const int n = t.p.in_y;
  T sent[psc::MAX_DEGREE];
  psc::further<F>(t.p, pad, in_y, sent);
  for (int i = 0; i < n; i++) sent[i] = leave<Kind::ROUND>(site, in_y[i], sent[i], t.p.slot + i);
  absorb(t, t.p.slot, n);
  const T beta = answer_not_zero(t, Round::PSC_Y);
  psc::took_beta(t.p, pad, cheat_is(3) ? F::zero() : beta);
  return beta;
}
// Masks and stores end value j without absorbing it: ends_sent() absorbs the entire message,
// possibly in a later launch. After further(); any thread, any copy of the `Tr`.
__device__ __forceinline__ T end(Site site, const Tr &t, T v, int j) {
  return leave<Kind::END>(site, v, psc::end_value<F>(t.p, pad_distance(), v, cheat_is(2) && j == 1 ? 0 : j), t.p.slot + j);
}
// The address of the first end value's slot. Valid after further().
__device__ __forceinline__ const T *end_slots(const Tr &t) {
  return t.p.slot;
}
// Absorbs the n end values that end() has stored as the prover's message; the verifier's message is
// empty. One thread.
__device__ __forceinline__ void ends_sent(Tr &t, int n) {
  if (n < 1) __trap(); // a sumcheck that sends no end value has no such round
  absorb(t, t.p.slot, n);
  tr_empty(t, Round::PSC_VALUES);
}
// Sends the n end values as one message from one thread. Then, unless DRAW is NONE, runs the round
// `behind` with an empty prover's message and returns its challenge.
template <Draw DRAW> __device__ __forceinline__ T ends(Site site, Tr &t, const T *v, int n, Round behind = Round::LINE) {
  for (int j = 0; j < n; j++) (void)end(site, t, v[j], j);
  ends_sent(t, n);
  return answer<DRAW>(t, behind);
}
// Public sources: see round_by_lanes().
template <int N> __device__ __forceinline__ void further_by_lanes(Site site, Tr *t, const T *in_y) {
  const long long pad = pad_distance();
  const bool leader = (threadIdx.x & 31) == 0;
  T sent[N];
  const T *slots = nullptr;
  if (leader) {
    DOOR_MARK(site, -1);
    if (t->p.in_y != N) __trap();
    psc::further<F>(t->p, pad, in_y, sent);
    for (int i = 0; i < N; i++) sent[i] = leave<Kind::ROUND>(site, in_y[i], sent[i], t->p.slot + i);
    slots = t->p.slot;
  }
  const T beta = message_by_lanes<Draw::COIN, N>(t, Round::PSC_Y, sent, slots);
  if (leader) psc::took_beta(t->p, pad, cheat_is(3) ? F::zero() : beta);
  __syncwarp((1u << SEND2_N) - 1);
}
// Public sources: see round_by_lanes().
template <int N> __device__ __forceinline__ void ends_by_lanes(Site site, Tr *t, const T *v) {
  const bool leader = (threadIdx.x & 31) == 0;
  T sent[N];
  const T *slots = nullptr;
  if (leader) {
    for (int j = 0; j < N; j++) sent[j] = end(site, *t, v[j], j);
    slots = t->p.slot;
  }
  values_by_lanes<N>(t, sent, slots);
}
// The end of a product layer's sumcheck, whose summand is the product of its two end values a and
// b: sends the round in Y and the end values, and returns the challenge of the round LINE that
// follows. Lanes and memory as in round_by_lanes().
__device__ __forceinline__ T layer_ends_by_lanes(Site site, Tr *t, T a, T b) {
  T sent[2] = {F::zero(), F::zero()};
  const bool leader = (threadIdx.x & 31) == 0;
  const long long pad = pad_distance();
  if (leader) {
    DOOR_MARK(site, -1);
    if (cheat_at_layer_end(4, a, b)) {
      const T other = a;
      a = b, b = other;
    }
    T in_y[2];
    psc::product_in_y<F>(a, b, psc::inner_mask(t->p, pad, 0), psc::inner_mask(t->p, pad, 1), in_y);
    psc::further<F>(t->p, pad, in_y, sent);
    sent[0] = leave<Kind::ROUND>(site, in_y[0], sent[0], t->p.slot);
    sent[1] = leave<Kind::ROUND>(site, in_y[1], sent[1], t->p.slot + 1);
  }
  const T beta = message2_by_lanes(t, Round::PSC_Y, sent[0], sent[1], t->p.slot);
  if (leader) {
    psc::took_beta(t->p, pad, cheat_is(3) ? F::zero() : beta);
    sent[0] = end(site, *t, a, 0);
    sent[1] = end(site, *t, b, 1);
  }
  __syncwarp((1u << SEND2_N) - 1); // lane 0 has moved t->p.slot on
  // two rounds: the end values, with an empty verifier's message, then the challenge
  values_by_lanes<2>(t, sent, t->p.slot);
  return message_by_lanes<Draw::COIN, 0>(t, Round::LINE, nullptr, nullptr);
}

// ---- A scalar that is sent as it is ----
// For a scalar that is public or carries a mask from elsewhere. Nothing is added to it here: the
// caller answers for what it reveals. It takes a slot like every sent scalar, and the mask entry
// with that number stays unused. One thread.
__device__ __forceinline__ void in_the_clear(Site site, Tr &t, T v, T *slot) {
  (void)leave<Kind::CLEAR>(site, v, v, slot); // stores v in its slot, as it is
  absorb(t, slot, 1);
}

// ---- Hidden values outside a sumcheck: value + the mask entry of the slot ----
// Masks and stores v but does not absorb it: the caller calls absorb() on the slots, possibly in a
// later launch.
__device__ __forceinline__ T claim(Site site, T v, T *slot) {
  return leave<Kind::CLAIM>(site, v, padv_again(v, slot), slot);
}
// Mask, store and absorb n hidden values, or one. No verifier's message follows here.
__device__ __forceinline__ void claims(Site site, Tr &t, const T *v, int n, T *slots) {
  DOOR_MARK(site, -1);
  for (int i = 0; i < n; i++) (void)claim(site, v[i], slots + i);
  absorb(t, slots, n);
}
__device__ __forceinline__ void claim_alone(Site site, Tr &t, T v, T *slot) {
  DOOR_MARK(site, -1);
  (void)claim(site, v, slot);
  absorb(t, slot, 1);
}

} // namespace sc_door
