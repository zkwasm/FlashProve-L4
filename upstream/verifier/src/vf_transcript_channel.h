// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* What the functions of verifier/src/vf_transcript_calls.h call: reads and draws on the channel
   (blk::VerifierRounds, blocks/chain/rounds.h) of the hash chain (Fiat-Shamir transcript) that the
   sub-proofs run on. The channel itself reads the proof. A read also checks that the value is in
   canonical form and adds it to the chain's next query. A function that draws first closes the
   round, by name. Format: docs/format/FORMAT.md, "4.2 A round's query and its answer".

   The channel compares the name, the length of the prover's message and the number of elements
   drawn with the next round of the plan, the list of rounds fixed before the run
   (blocks/numbers/plan.h). A mismatch is a fault of this program, not of a proof, and ends the
   process. cp_verify, the verifier program, runs its stages as child processes. If one ends in this
   way, cp_verify prints `NOT CHECKED` (exit code 5), as it does for a rejection name that begins
   "internal-"; but if another stage rejects the proof, it prints that rejection. */
#pragma once
#include "chain/rounds.h"
#include "numbers/round_names.h"

// The round's name for the channel (blocks/numbers/round_names.h). LKP_PLANES, FOREST_POINT and
// JOIN share the name "JOIN": by name the channel cannot tell them apart.
static const char *roundName(RoundName r) {
  switch (r) {
    case RoundName::PSC_EPSILON: return blk::R_PSC_EPSILON;
    case RoundName::PSC_ROUND: return blk::R_PSC_ROUND;
    case RoundName::PSC_Y: return blk::R_PSC_Y;
    case RoundName::LINE: return blk::R_LINE;
    case RoundName::ZERO: return blk::R_ZERO;
    case RoundName::BAT_SIGMA: return blk::R_BAT_SIGMA;
    case RoundName::LKP: return blk::R_LKP;
    case RoundName::LKP_SHIFT: return blk::R_LKP_SHIFT;
    case RoundName::LKP_PLANES: return blk::R_LKP_PLANES;
    case RoundName::FOREST_POINT: return blk::R_FOREST_POINT;
    case RoundName::PSC_VALUES: return blk::R_PSC_VALUES;
    case RoundName::JOIN: return blk::R_JOIN;
  }
  refuse("internal-round-name");
}
/* `Ex`, this verifier's challenge field, and the library's blk::E must be the same cubic extension
   in the same basis: the same prime, and X^3 equal to the same constant. This asserts only the
   prime and Ex's degree. verifier/src/vf_common.h, which defines `Ex`, asserts the constant
   (FIELD_W == blk::NR); verifier/tests/one_chain_shell_test.cpp tests that products agree. */
static_assert(CE == 3 && blk::P == FIELD_P, "this verifier's field is not the field that blocks/ was built for");
static Ex fromBlkE(const blk::E &x) {
  return Ex(x.c[0].v, x.c[1].v, x.c[2].v);
}
/* Called here after a failed read and after each closed round. After a fault in a proof's bytes the
   channel's reads fail and its draws return fixed values. This verifier rejects at once instead:
   it prints the channel's reason and then rejects, under one name whatever the fault. */
static void chainHeld(const blk::VerifierRounds &ch) {
  if (ch.ok()) return;
  printf("chain: %s\n", ch.why());
  refuse("chain-refused");
}

static vector<Ex> chainRead(blk::VerifierRounds &ch, size_t n) {
  vector<Ex> v(n);
  for (Ex &x : v) {
    blk::E y;
    if (!ch.recv(y)) chainHeld(ch);
    x = fromBlkE(y);
  }
  return v;
}
static Ex chainSum(blk::VerifierRounds &ch) {
  blk::E y;
  if (!ch.recv_hiding_sum(y)) chainHeld(ch);
  return fromBlkE(y);
}
// Rejects BAT_SIGMA as this program's fault. Of this file's functions only chainBatch() and
// chainBatchOpen() draw BAT_SIGMA, which batches linear claims, because the channel must also be
// given the number of claims.
static void chainNoBatching(RoundName round) {
  if (round == RoundName::BAT_SIGMA) refuse("internal-batching-bare");
}
// One challenge. Before PSC_EPSILON (PSC: the sumcheck with a hiding polynomial) the last read
// must be chainSum(), that polynomial's sum, or the process ends.
static Ex chainAnswer(blk::VerifierRounds &ch, RoundName round) {
  chainNoBatching(round);
  const blk::E y = round == RoundName::PSC_EPSILON ? ch.close_epsilon(roundName(round)) : ch.close_algebraic(roundName(round)).draw();
  chainHeld(ch);  // a nonce is missing or fails the round's proof of work
  return fromBlkE(y);
}
// One challenge, never 0.
static Ex chainAnswerNotZero(blk::VerifierRounds &ch, RoundName round) {
  chainNoBatching(round);
  const blk::E y = ch.close_algebraic(roundName(round)).draw_nonzero();
  chainHeld(ch);
  return fromBlkE(y);
}
// The matrix of the approximate range check (RNG): entries in {-1, 0, 1}, `combinations` per
// column. The plan computes the round's error from `combinations`, but the channel compares only
// the total unless asked, as here.
static vector<signed char> chainTrits(blk::VerifierRounds &ch, size_t columns, size_t combinations) {
  auto from = ch.close_bits(blk::R_RNG);
  ch.require_combinations(combinations); // before an entry is drawn
  vector<signed char> v = from.trits(columns * combinations);
  chainHeld(ch);
  return v;
}
// Closes a round in which the verifier's message is empty.
static void chainEmpty(blk::VerifierRounds &ch, RoundName round) {
  ch.close_empty(roundName(round));
  chainHeld(ch);
}
// n challenges of one round.
static vector<Ex> chainPoint(blk::VerifierRounds &ch, RoundName round, size_t n) {
  chainNoBatching(round);
  blk::AlgebraicDraw draw = ch.close_algebraic(roundName(round));
  vector<Ex> v(n);
  for (Ex &x : v) x = fromBlkE(draw.draw());
  chainHeld(ch);
  return v;
}
/* BAT_SIGMA: one element x is drawn, and claim i (`rows` in identifiers) gets the coefficient x^i.
   The planned error depends on the number of claims, which the channel compares only if given it.
   So this returns x^0 .. x^(rows - 1): the vector has no coefficient for another claim. */
static vector<Ex> chainBatch(blk::VerifierRounds &ch, size_t rows) {
  if (!rows) refuse("internal-batching-bare");
  blk::AlgebraicDraw from = ch.close_algebraic(blk::R_BAT_SIGMA);
  from.require_rows(rows);
  const Ex x = fromBlkE(from.draw());
  chainHeld(ch);
  vector<Ex> v(rows, eO());
  for (size_t i = 1; i < rows; i++) v[i] = v[i - 1] * x;
  return v;
}
/* The same round where a sub-proof's claims are added after the draw. This returns x: the caller
   forms the powers. `chainBatchRows` must give the number of claims before the next read or
   close, or the process ends. */
static Ex chainBatchOpen(blk::VerifierRounds &ch) {
  const blk::E y = ch.close_algebraic(blk::R_BAT_SIGMA).draw();
  chainHeld(ch);
  return fromBlkE(y);
}
static void chainBatchRows(blk::VerifierRounds &ch, u64 rows) {
  ch.require_rows(rows);
}
