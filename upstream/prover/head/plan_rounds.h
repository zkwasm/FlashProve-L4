// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The rounds of the output head's sub-proof (final norm, logits, acceptance policy) as the proof's
// plan (blk::Plan, blocks/numbers/plan.h) lists them, in the order in which
// prover/head/hidden_fork.cuh runs them and verifier/src/vf_output_head_check.inc reads them. Prover and
// verifier each fill HeadRounds from their own shape. Running a round whose name is not the next in
// the plan ends the process (Cursor::next_round()). Include paths: blocks/ and prover/.
//
// The sub-proof is one segment of a request's proof (`link` in identifiers: ReqLink,
// blocks/iop/req.h); req_plan() there plans its messages and the next token's round. The callers
// plan what follows each of its three commit points (`moment`: a message that commits arrays):
//   0   after the cells (committed small integers): the sub-proof's lookup, then plan_head_own()
//   1   after the reads (weighted sums of cells): the matrix of the approximate range check
//   2   after the range check's digits: its lookup on them, then plan_head_end()
#pragma once
#include <string>

#include "hiding/token_plan.h"
#include "numbers/blocks.h"

namespace blk {

struct HeadRounds {
  unsigned lt = 0, lPd = 0, lkW = 0; // log2 of: token positions, padded features, weight columns
  // Of the norm: 2 * limbs - 1 sums of limb products; 7 row products of 4 row values with 5 limbs,
  // else 3 of 3.
  unsigned sums = 0, products = 0, row_values = 0;
  // What the last round batches: the sub-proof's linear claims and its range check's equations.
  u64 rows = 0;
};

// Two equations of the acceptance policy have no round of their own. At every token position that
// has a cost (in the request's range, not the window's last), its hidden numbers satisfy c + k = C,
// and zs plus its table values is 2^F (`C`, `F`: parameter-file lines). Both are tested at zt
// (weights `phiC` in verifier/src/vf_output_head_check.inc): the lt token-position coordinates of the key
// point (where a lookup call's keys are evaluated) of the call with the lookup segment (named list
// of keys) HD0. The rounds that draw zt cover their error: in the last level of the lookup's
// product tree over the keys, the sumcheck (`psc` in identifiers) rounds (2/|E|, E the extension
// field) and, if its challenge is one of the coordinates, the plan_line() round (1/|E|).
//
// In round-by-round soundness a state is lost for a malicious prover while some false relation is
// still false, so a challenge rescues it only by rescuing every false relation at once: across such
// relations a round's error is the largest of their errors, not their sum. Within one relation the
// errors add wherever one verifier message gives it several ways out: a point of n coordinates
// (n/|E|), a batching by powers ((rows - 1)/|E|), codeword positions that all miss the error or a
// challenge drawn with them that cancels it.
//
// Three conditions must hold for each equation; each is argued below, not proved:
// (1) The polynomial is fixed before zt's first coordinate is drawn. It is the multilinear
//     extension over the token positions of the difference of the equation's two sides (0 at a
//     position without a cost; which positions have one is public). The matrices hr (cells of c, k,
//     zs) and hf (cells of the table values) are the sub-proof's own, so they are in the segment's
//     first message (Prover::chain_first_moment() in prover/prover/chain_frame.cuh), which is
//     received, and its out-of-domain sample answered, before the lookup's first challenge.
// (2) Each coordinate is the challenge of a planned round whose error's numerator is at least the
//     polynomial's degree in it, 1: every level of that tree replaces the whole point by its
//     rounds' challenges and its plan_line() challenge (handDown() in verifier/src/vf_subproof_engine.h),
//     and forestPad() in verifier/src/vf_lookup_argument.h cuts a call's key point from the last level's.
// (3) Each equation at zt is one of the segment's linear claims. These read not the cells but the
//     reads, which are sent after zt is drawn. The reads' sample is answered before the range
//     check's matrix is drawn; that matrix's round catches reads that do not match the cells; and
//     the check's equations are multiplied by further powers of the challenge that batches the
//     claims, whose round, the last of plan_head_end(), counts claims and equations.
//
// As in every round that tests an identity at a random point, the segment's first message must be
// one definite codeword when the challenges are drawn. Unique decoding does not give this (its
// group's list of close codewords may be longer than 1); the answered sample does: the codeword is
// the one in the list with that sample, which the seed round of plan_send() and the list's factor
// in plan_bat() account for. The message commits to packed entries; the cells are their one small
// unpacking, and where there is none the range check's relation is the false one.
//
// plan_head_own() plans the final norm's rounds, then the logits' sumcheck. `where` begins
// PlanRound::where, printed after each round's name.
inline void plan_head_own(Plan &p, const std::string &where, const HeadRounds &x) {
  plan_zero(p, where + ", the norm's point", x.lt);
  plan_scalars(p, "sums of limb products", where, x.sums);
  plan_bat_sigma(p, where + ", limbs", x.sums);
  PscSize layer; // sums squares over features (plain: equality weights at 1/2) and token positions
  layer.degree.assign(x.lPd + x.lt, 3), layer.degree_Y = 2, layer.values = 2;
  plan_psc(p, where + ", sum of squares", layer);
  plan_line(p, where + ", sum of squares");
  plan_scalars(p, "row products", where, x.products);
  plan_bat_sigma(p, where + ", row products", x.products);
  plan_token_sum(p, where + ", row products", x.lt, x.row_values, 2);
  plan_token_sum(p, where + ", stream times reciprocal", x.lt, 2, 2);
  plan_token_sum(p, where + ", activations times head", x.lt, 2, 2, x.lPd);
}
// The merge of the one claim on the weight table, which sends the value that goes to the opening of
// the weights; then the round that batches x.rows.
inline void plan_head_end(Plan &p, const std::string &where, const HeadRounds &x) {
  plan_bat_sigma(p, where + ", merge", 1);
  // Plain sums over the lkW column bits, round polynomials of degree 2 (PscSize::degree counts
  // their 3 coefficients, of which 2 are sent); one end value, in which the summand has degree 1.
  // 1 + 2 lkW + 1 + 1 scalars, as sc_door::slots(SUM2, lkW, 1, 1), prover/hiding/sumcheck_door.cuh.
  PscSize merge;
  merge.degree.assign(x.lkW, 3), merge.degree_Y = 1, merge.values = 1;
  plan_psc(p, where + ", merge", merge);
  plan_bat_sigma(p, where + ", all rows", x.rows);
}

} // namespace blk
