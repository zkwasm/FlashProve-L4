// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The rounds of the attention sub-proof (the argument for one layer group's attention) for the
// proof's plan (blk::Plan: the list of rounds that both sides follow), in the order in which the
// prover runs them (prover/attn/front.cuh; the last round, prover/attn/fork.cuh) and
// verifier/src/vf_attention_check.h reads them. Prover and verifier include this file and fill
// AttentionRounds from their own shape. Compile with -I blocks.
//
// The sub-proof is one segment of a request's proof (ReqLink, blocks/iop/req.h). After the SEND
// (oracle messages sent, then sampled) of each of its commit points (`moment`) the callers plan
// (lookup and range check by other functions):
//   0   the cells (committed small integers, packed): the sub-proof's lookup, then
//       plan_attention_own()
//   1   the reads (weighted sums of cells): the approximate range check's random matrix
//   2   the range check's digits: its lookup on them, then plan_attention_end()
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "numbers/blocks.h"

namespace blk {

// SC3: a sumcheck (`psc` in identifiers) of the sum over nv variables and `pairs` pairs (A, B) of
// W A B, the weights W public. A round polynomial has degree 3 (PscSize::degree: the degree plus
// 1). `values`: the end values sent, two for each pair; the summand has degree 2 in them.
inline PscSize attention_sc3(unsigned nv, unsigned pairs) {
  PscSize x;
  x.degree.assign(nv, 4), x.degree_Y = 2, x.values = 2 * pairs;
  return x;
}
// SCD: a sumcheck with rounds over the lt bits of the key token j (degree 2), the lgk bits of the
// key-value head g (lt + 2), the lt bits of the query token i (lt + 1). `values`: KC_lam (layer
// lam's keys), lam < Lam, and the lt index bits of each pair; the summand has degree lt + 1 in
// them.
inline PscSize attention_scd(unsigned lt, unsigned lgk, unsigned pairs, unsigned Lam) {
  PscSize x;
  x.degree.assign(lt, 3);
  x.degree.insert(x.degree.end(), lgk, lt + 3);
  x.degree.insert(x.degree.end(), lt, lt + 2);
  x.degree_Y = lt + 1, x.values = Lam + pairs * lt;
  return x;
}

// 2^lt tokens; 2^lgk key-value heads with HA query heads each; Lam layers; `pairs` pairs (layer,
// query head a < HA). A query token's index is the key token with its largest score. A lookup
// segment (enum ASeg, prover/attn/plan.cuh) is a named list of a lookup call's keys.
struct AttentionRounds {
  unsigned lt = 0, lgk = 0;
  unsigned pairs = 0, Lam = 0; // NP = Lam * HA, and Lam
  unsigned plane_messages = 0; // lookup segments RM, RD present in calls of kind KR; 3 scalars each
  // For each lookup segment FM, FD that is present, in the order of the calls: (variables, pieces).
  std::vector<std::pair<unsigned, unsigned>> score;
  // Variables of the SC3 for Z o, the value product's two, q k at the index, the index bits.
  unsigned e = 0, value_m = 0, value_d = 0, attained = 0, bits = 0;
  // Batched by the last round: the sub-proof's linear claims (a linear form of committed messages
  // has a public value) and its range check's equations.
  u64 rows = 0;
};

// The message "sums, index bits, remainders" is planned for AP::wd == 3 (three-byte division
// remainders, prover/attn/plan.cuh); for wd == 4 prover/attn/front.cuh sends four more scalars.
// The parameter file has no line wd: APlan::decodeS() reads its line `S` as S + 32 (wd - 3); the
// largest softmax weight is 2^S. cp_verify accepts three-byte remainders only: it rejects a line
// `S` of 32 or more (rejection name `af-policy-wd`) and lt + S + 1 > 24 (`af-policy-softmax`).
//
// An error event without a round of its own. The identity Z(h, i) = sum over j <= i of e(h, i, j)
// (the denominator is the sum of the softmax weights, at every head and token position) is tested
// at a point drawn by the lookup: the key point of the call with lookup segment WL, without its
// first inner coordinate and its last ldh, the channel bits (`wp`, `zgW`, `ziW` in
// verifier/src/vf_attention_check.h). The rounds that draw these coordinates carry its error: in the last
// level of the lookup's product tree over the keys, the sumcheck rounds (2/|E|, E the extension
// field) and the plan_line() round (1/|E|). The argument is informal.
//
// In round-by-round soundness a state is lost for a malicious prover while some false relation is
// still false, so a challenge rescues it only by rescuing every false relation at once: across
// such relations a round's error is the largest of their errors, not their sum. Within one
// relation the errors add wherever one verifier message gives it several ways out: a point of n
// coordinates (n/|E|), a batching by powers ((rows - 1)/|E|), codeword positions that all miss or
// a challenge drawn with them that cancels.
//
// Three conditions must hold; each is argued below from a reading of the code, not proved:
// (1) The polynomial, the multilinear extension of Z - sum_j e over heads and token positions, is
//     fixed before the first of these coordinates is drawn. Z's cells (matrix `ah`) and e's
//     (matrices `as`) are among the sub-proof's own matrices, which are sent in the segment's first
//     message (Prover::chain_first_moment(), prover/prover/chain_frame.cuh). Its sample is answered
//     before the lookup's first challenge. As for other identities tested at a drawn point, this
//     fixes the message as one codeword: not by unique decoding (the group's list may hold several
//     codewords) but as the one in the list with that sample, which SEND's seed round and the
//     list's factor in the segment's plan_bat() round account for. What is committed are packed
//     entries; the cells are their one small unpacking, and where there is none, the false relation
//     is the range check's.
// (2) Each coordinate is the challenge of a planned round whose error's numerator is at least the
//     polynomial's degree in it, 1. Every level of that tree replaces the whole point by its
//     rounds' challenges and its plan_line() challenge (handDown(), verifier/src/vf_subproof_engine.h), and
//     a call's key point is cut from the last level's (forestPad(), verifier/src/vf_lookup_argument.h).
// (3) What is claimed at the point reaches linear claims through planned rounds only. A linear
//     claim ties ZC, the value sent for Z, to Z's cells. Another equates ACC + gam ZC (ACC: the
//     value sent for the numerator; gam: the challenge of "numerator and denominator") with the
//     value product's two sums, whose sumchecks are planned below. Linear claims tie their end
//     values to e's and v's cells.
//     These claims read not the cells but the reads, an oracle message sent after the point is
//     drawn. Its sample is answered before the range check's matrix is drawn; that matrix's round
//     catches reads that are not the cells'; and the check's equations are multiplied by further
//     powers of the challenge that batches the claims, whose round, in plan_attention_end(), counts
//     both.
inline void plan_attention_own(Plan &p, const std::string &where, const AttentionRounds &x) {
  if (x.plane_messages) plan_scalars(p, "plane values", where, 3 * (u64)x.plane_messages);
  for (const auto &s : x.score) {
    plan_scalars(p, "plane values and the scores' sum", where, 7);
    plan_psc(p, where + ", scores", attention_sc3(s.first, s.second));
  }
  plan_scalars(p, "sums, index bits, remainders", where, 5 + (u64)x.lt + 2);
  plan_psc(p, where + ", Z o", attention_sc3(x.e, x.pairs));
  plan_bat_sigma(p, where + ", numerator and denominator", 2);
  plan_scalars(p, "the value product's two sums", where, 2);
  plan_psc(p, where + ", value product, off the diagonal", attention_sc3(x.value_m, x.pairs));
  plan_psc(p, where + ", value product, diagonal", attention_sc3(x.value_d, x.pairs));
  plan_zero(p, where + ", the maximum is attained; the index bits are bits", 2 * x.lgk + 2 * x.lt + 4);
  plan_scalars(p, "the maxima at the point", where, x.pairs);
  plan_bat_sigma(p, where + ", maxima", x.pairs);
  plan_psc(p, where + ", q k at the index", attention_sc3(x.attained, x.pairs));
  plan_bat_sigma(p, where + ", keys at the index", x.pairs);
  plan_psc(p, where + ", the key at the index", attention_scd(x.lt, x.lgk, x.pairs, x.Lam));
  plan_bat_sigma(p, where + ", index bits", x.pairs);
  plan_psc(p, where + ", index bits", attention_sc3(x.bits, x.pairs));
}
// After the sub-proof's last message.
inline void plan_attention_end(Plan &p, const std::string &where, const AttentionRounds &x) {
  plan_bat_sigma(p, where + ", all rows", x.rows);
}

} // namespace blk
