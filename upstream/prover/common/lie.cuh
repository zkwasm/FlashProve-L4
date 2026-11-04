// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Fault injection for tests of the verifier. A "lie" replaces one value of a run, usually an output
// of the model's integer arithmetic, by a false one; the prover (cp_prove) then proves the run, and
// the verifier must reject the proof. Active only in a test build for malicious-prover tests that
// is not part of this repository. Here every function is a test hook with no effect, in every
// build: empty, the identity or a constant. prover/'s sources call 16 of the 20, at 126 places.
//
// A kernel of the forward pass calls v = lie::at(value, row, token, v) as soon as it has computed a
// step's output v: before the step's cells (committed small integers) and lookup keys are derived
// from v and before v is stored for later steps. For four values the remainder is formed from the
// true value, not from the altered one: QUOTIENT, UP, GATE and LOGIT (see the kernels' comments in
// prover/prover/forward.cuh). Here at() returns v; in that test build it returns another value at a
// place that the test names through the environment. Later code runs unchanged on the false value,
// so a lie is meant to make one relation of the proof false: the one that ties this output to the
// step's inputs. The host first states which layer and part the next launches belong to:
// lie::here(layer, part, stream). In that test build layer and part are held in single variables
// without a lock, so overlapping work on another stream or host thread can read another step's
// layer and part, and nothing checks this.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/common.h"
#include "common/setting.h"

namespace lie {
// O: attention's output projection. D: the down projection. BORDER: the residual stream between two
// groups of layers.
enum Part : int { EMBEDDING, BORDER, NORM1, QKV, ATTENTION, O, NORM2, ROUTER, FFN, EXPERTS, D, HEAD, SMALLNESS, WEIGHTS, LOOKUP, PARTS };
enum Value : int {
  ACCUMULATOR, // a product with a weight matrix, before it is rounded
  QUOTIENT,    // a rounded quotient
  CLAMPED,     // what a clamp's table gives
  STREAM,      // the residual stream after an addition
  SIGMA,       // a norm's root
  RECIPROCAL,  // a norm's reciprocal
  ROTATED,     // a query's or key's accumulator after the rotation
  LOGIT,       // a router's logit
  EXPERT,      // the router's choice
  SCORE,       // the router's score
  SCALED,      // the routed expert's input: the norm's output times the score, rounded
  GATE,        // the gate's rounded value
  UP,          // the up projection's rounded value
  SILU,        // what the SiLU table gives
  HIDDEN,      // the gated product
  DIGIT,       // a digit of a norm's row of digits
  // values of attention (MAXIMUM to OUTPUT; prover/attn/) and of the output head (MARGIN to TOKEN;
  // prover/head/). These parts also use ACCUMULATOR, SCORE, LOGIT, DIGIT and others for their own
  // values, and a norm uses SUM.
  MAXIMUM,
  INDEX,
  WEIGHT,
  SUM,
  OUTPUT,
  MARGIN,
  EXPONENTIAL,
  PRICE,
  COST,
  TOKEN,
  // values of the approximate range check (part "smallness"). No code in this repository acts on
  // these two or on that part: the test build that does is not part of this repository
  COMBINATION, // a combination of one row's cells under the verifier's matrix
  SPLIT,       // how a combination is written in its two digits
  // more values of a layer (CODE: prover/weights_blocks/made_up_source.cuh)
  CODE,     // a weight's quantized code: this run reads other weights than the committed ones
  SELECTOR, // an entry of the router's selector row
  ROW,      // the row at which the score's table is read
  // more values of attention (prover/attn/forward_kernels.cuh, prover/attn/forward.cuh). RECORD:
  // the record gets a lower row of the table, that row's weight and a remainder larger to match;
  // its flag stays. EMPTY: row: the query head; token: the slot row; second: the place in the tile.
  // GROUP: row: the query heads' own key-value head; token, second: 0.
  RECORD, // by how many steps of the table a score's record is moved down
  BIT,    // a cell of the index's bits (second: which bit)
  EMPTY,  // the low byte's cell of the weight at an empty slot of region M
  GROUP,  // which key-value head's keys and values a group of query heads reads
  // CELL, with the part "head": a cell of the head's matrix hf (prover/head/forward.cuh) after the
  // forward pass (row: the feature; token: the token position), also at one outside the request's
  // range and at a row of the vocabulary from V on. With the part "smallness": a cell of one of a
  // sub-proof's own matrices as its symbol is committed (layer: the range check's number; row: the
  // matrix; token, second: the cell's row and column); no code in this repository acts on that use.
  CELL,   // two uses: see above
  // After ROUNDED the remainder, its digit cells and both keys follow the false quotient: the
  // rounding's equation holds, the remainder's range does not. No code in this repository acts on
  // COUNT or on the part "lookup".
  ROUNDED,  // a rounded quotient before its remainder is formed
  MASK,     // a private random field element in memory, before its use (`zeroed_in_memory`)
  COUNT,    // a lookup's multiplicity, after the counting
  PLACE,    // two tensors of the synthetic model exchanged as this run reads it (`place`)
  VALUES
};
enum How : int { ADDED, SET, SECOND };
// Stops: checks of the prover's own that a false value can fail. The test build named at the top
// of this file can go on past them, so that no test takes the prover's giving up for the verifier's
// rejection. A kernel calls pass(); host code would call passes(), but none does. Both return false
// here, in every run and build, so a failed stop always counts; that test build prints each stop
// passed. `k_tie`: prover/hiding/lookup_tie.cuh. INDEX_WINNER: prover/common/index_guard.cuh.
enum Stop : int {
  LOOKUP_TIE,          // `k_tie`: a lookup's two roots do not agree (a key in no row of its table)
  ATTENTION_REMAINDER, // attention's division: k_divide's *err
  HEAD_MARGIN,         // the head: a margin does not fit its digits (E_LR)
  HEAD_BOUNDS,         // the head: the request exceeds the acceptance policy's bounds (E_BOUNDS)
  HEAD_WINNER,         // the head: the checked token is no index of the vocabulary (INDEX_WINNER)
  STOPS
};
template <class I> __device__ __forceinline__ constexpr I at(Value, long long, long long, I v, long long = 0) {
  return v;
}
__device__ __forceinline__ constexpr bool wants_the_second(Value, long long, long long) {
  return false;
}
__device__ __forceinline__ void chosen(Value, long long, long long, long long, long long) {}
__device__ __forceinline__ constexpr bool pass(Stop) {
  return false;
}
static inline void arm() {}
static inline void here(int, Part, cudaStream_t) {}
static inline void seat(int) {}
static inline constexpr bool acts_now() {
  return false;
}
static inline constexpr int layer_now() {
  return 0;
}
template <class I, class Where> static inline void in_memory(int, Part, Value, cudaStream_t, Where) {}
template <class I, class Where> static inline void back_in_memory(int, Part, Value, cudaStream_t, Where) {}
template <class E, class Where> static inline void zeroed_in_memory(int, Part, Value, cudaStream_t, Where) {}
static inline constexpr long long on_the_host(int, Part, Value, long long, long long v, cudaStream_t) {
  return v;
}
static inline constexpr bool place(int &, int &, cudaStream_t, bool = false) {
  return false;
}
static inline constexpr bool experts(int, int, int &, int &, cudaStream_t, bool = false) {
  return false;
}
static inline constexpr bool on() {
  return false;
}
static inline constexpr bool passes(const char *) {
  return false;
}
static inline void from_the_environment() {}
static inline void report() {}
// For an index into a table of n entries that a step computes from a value returned by at().
// Returns i. In a test build that is not part of this repository an i outside 0 <= i < n gives 0,
// the table's first entry. A lie may name any number, and a read outside a table can fault the
// kernel and, on a machine that others share, bring down their work too. An honest index is within
// its table by the step's own arithmetic, so this repository's builds test nothing here. Unlike
// index_below() (prover/common/index_guard.cuh), inside() sets no flag: the run continues and
// writes its proof, for the verifier to check. Only the read is kept within the table: the step
// still derives its keys and cells from the false value.
template <class I> __device__ __forceinline__ constexpr I inside(I i, long long) {
  return i;
}
} // namespace lie
