// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* The transcript, as the verifier's sub-proofs use it (the prover's: prover/hiding/transcript.cuh).
   A sub-proof argues one part of the forward pass. Tr (vf_common.h) wraps blk::VerifierRounds
   (blocks/chain/rounds.h), the channel of the proof's first hash chain.

     trRead           reads n field elements (Ex) of the prover's message
     trSum            reads a hiding polynomial's sum: must be the last value before PSC_EPSILON
     trAnswer         draws one element
     trAnswerNotZero  draws one element that is not 0
     trPoint          draws n elements
     trBatch          draws one element x to batch `rows` claims; returns x^0 .. x^(rows - 1)
     trBatchOpen      the same challenge for a caller that counts its claims later; returns x
     trBatchRows      states that count; must be called before anything else is read or drawn
     trEmpty          ends a prover's message that the verifier does not answer
     trTrits          draws a matrix with entries in {-1, 0, 1}

   A call that finds a fault in the proof's bytes calls refuse() (vf_common.h) and does not return. */
#pragma once
// RoundName names the kind of a call's round, not its place. The next round of the chain's plan (the
// list of rounds: blocks/numbers/plan.h) must have the same name, or the process aborts.
// Names as in enum Round (prover/hiding/transcript.cuh) and blocks/numbers/round_names.h (R_..).
// The latter gives LKP_PLANES, FOREST_POINT and JOIN one name, so this test cannot tell them apart.
// PSC: the sumcheck with a hiding polynomial; BAT: batching of claims; LKP: lookup. BAT_SIGMA is
// the round of trBatch and trBatchOpen. Another call that draws under this name is
// a fault of this program ("internal-batching-bare"): cp_verify ends NOT CHECKED, not NO.
enum class RoundName { PSC_EPSILON, PSC_ROUND, PSC_Y, LINE, ZERO, BAT_SIGMA, LKP, LKP_SHIFT, LKP_PLANES, FOREST_POINT, JOIN, PSC_VALUES };
#include "vf_transcript_channel.h" // needs RoundName; has the functions called below

static vector<Ex> trRead(Tr &tr, size_t n) { return chainRead(tr.one, n); }
static Ex trSum(Tr &tr) { return chainSum(tr.one); }
static Ex trAnswer(Tr &tr, RoundName round) { return chainAnswer(tr.one, round); }
static Ex trAnswerNotZero(Tr &tr, RoundName round) { return chainAnswerNotZero(tr.one, round); }
// The matrix of the approximate range check, column by column, `combinations` entries each. The
// chain aborts the process unless `combinations` is the planned number.
static vector<signed char> trTrits(Tr &tr, size_t columns, size_t combinations) { return chainTrits(tr.one, columns, combinations); }
// An empty round still makes its hash query.
static void trEmpty(Tr &tr, RoundName round) { chainEmpty(tr.one, round); }
static vector<Ex> trPoint(Tr &tr, RoundName round, size_t n) { return chainPoint(tr.one, round, n); }
// The chain aborts the process unless `rows` is the planned number.
static vector<Ex> trBatch(Tr &tr, size_t rows) { return chainBatch(tr.one, rows); }
static Ex trBatchOpen(Tr &tr) { return chainBatchOpen(tr.one); }
// After trBatchOpen, whose caller forms the powers of x itself: `rows` is the number of claims
// that these powers have batched. The chain aborts the process unless `rows` is the planned
// number.
static void trBatchRows(Tr &tr, u64 rows) { chainBatchRows(tr.one, rows); }
