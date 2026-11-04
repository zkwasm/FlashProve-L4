// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The verifier's check of a request's segment chain: the part of its proof outside the sub-proofs,
// each of which ends in one linear equation on the committed arrays. A segment (`link` in names) is
// one sub-proof's part of the proof, with one group of oracle messages. blk::ReqVerifier
// (blocks/iop/req.h), "the frame", reads the chain's own messages. Their order:
// docs/format/10-proof-file.md, "10.3 The first chain's rounds". Two forms of this check:
//   mb::ChainCheck     calls that the code checking a sub-proof makes itself; cp_verify uses it
//   mb::chain_verify() one mb::LinkHook per segment; used only by verifier/tests/req_chain_test.cpp
//
// True from ChainCheck::finish() or chain_verify() is meant to show that every segment's equation
// holds for what was committed, up to the errors that V.ch.plan() lists for its rounds. V is the
// caller's context; V.ch is its channel (blocks/chain/rounds.h) on the proof's first Fiat-Shamir
// hash chain. True does not cover the oracle messages in frame().at_start(), which belong to no
// segment; what follows on V.ch; bytes left over in the proof, which blk::VerifierRounds::finish()
// rejects; or the next token: the caller compares frame().plain() with the token that the client
// was given. In cp_verify, blk::check_proof() in blocks/frame/check.h sets up V, compares the
// token, and tests the at_start() messages on the proof's second chain (the last of three), where
// it calls finish().
#pragma once
#include <string>
#include <vector>

#include "iop/req.h"
#include "iop/req_rows.h"

namespace mb {

// One segment's sub-proof, as chain_verify() calls it. run() checks what follows commit point
// `moment`: a prover message of the segment that carries oracle messages (`sent`), counted from 0.
// After the last one, claim() and terms() give the segment's equation, a blk::ReqEquation
// (blocks/iop/req_rows.h).
// Not checked, for a hook or for ChainCheck::close(): the equation must be a function of public
// inputs read by V.ch.get(), of what was received on V.ch and of challenges drawn there. Nothing
// may come from a file, or from the proof by another route.
struct LinkHook {
  virtual ~LinkHook() {}
  virtual bool run(unsigned moment, blk::VerifierContext &V, const blk::ReqFrame &frame, const std::vector<blk::OracleId> &sent) = 0;
  virtual void claim(blk::E &target, std::vector<blk::E> &on_masks) = 0;
  virtual std::vector<blk::ReqTerm> terms() = 0;
};

// The request's shape and the layout of its arrays: fixed before a proof is read, and bound in the
// hash chain's first query, which is hashed before the prover speaks (blocks/chain/public.h).
// Prover and verifier must bind the same items there, under the same names and in the same order.
// For this, the prover program cp_prove calls blk::bind_request_parameters()
// (blocks/frame/check.h) itself, and cp_verify calls it through blk::check_proof().
struct ChainShape {
  blk::Public<blk::ReqShape> shape;
  blk::Public<blk::ReqLayout> layout;
};

// For each segment the caller calls first(), later() once for each further commit point, with the
// sub-proof's parts in between, and close(); after the last segment, finish(). A call out of turn
// returns false, where the frame would end the process, and so does every call after a false one.
// So finish() returns true only if every segment had all its commit points and exactly one close().
// Preconditions: `fixed.shape` and `fixed.layout` have passed blk::req_fault() and
// blk::req_layout_fault(), or the constructor ends the process. Not checked here: all hash chains
// of one proof must share one blk::VerifierSession (blocks/chain/session.h).
class ChainCheck {
public:
  ChainCheck(blk::VerifierContext &V, const ChainShape &fixed) : V_(V), frame_(V, fixed.shape), rows_(V, frame_, fixed.layout) {}
  // Starts the next segment: its plain scalars, commit point 0 and its samples; from the second
  // segment on, also the opening of the previous segment's group at the drawn codeword positions.
  // Plain scalars: public values sent before commit point 0; in cp_verify the next token alone.
  [[nodiscard]] bool first() {
    if (!good_ || in_link_ || frame_.link() >= links()) return good_ = false;
    in_link_ = true, moments_ = 1;
    return good_ = frame_.first(sent_);
  }
  // The oracle messages of the segment's next commit point and their samples.
  [[nodiscard]] bool later() {
    if (!good_ || !in_link_ || moments_ >= frame_.shape().link[frame_.link()].moments()) return good_ = false;
    moments_++;
    return good_ = frame_.later(sent_);
  }
  // The sub-proof is over: blk::ReqRows::row(), which the prover uses too, makes its equation a
  // linear claim (`row` in names) on the segment's group; the group's linear claims are batched and
  // folded into one. The process ends on a malformed equation: for example, if it names an array
  // that is not in the layout, or if a term's on_rows and on_columns are both written-out vectors.
  [[nodiscard]] bool close(const blk::ReqEquation &equation) {
    if (!good_ || !in_link_ || moments_ != frame_.shape().link[frame_.link()].moments()) return good_ = false;
    in_link_ = false;
    V_.st.add_row(frame_.group(), rows_.row(equation));
    return good_ = frame_.close();
  }
  // After the last close(): the opening of the last segment's group, and the proximity test.
  [[nodiscard]] bool finish() {
    if (!good_ || in_link_ || frame_.link() != links()) return good_ = false;
    const bool yes = frame_.finish();
    good_ = false;  // nothing is due any more
    return yes;
  }
  const blk::ReqFrame &frame() const { return frame_; }
  const std::vector<blk::OracleId> &sent() const { return sent_; }  // at the latest commit point

private:
  size_t links() const { return frame_.shape().link.size(); }
  blk::VerifierContext &V_;
  blk::ReqVerifier frame_;
  blk::ReqRows<blk::VerifierContext> rows_;  // declared after frame_: its constructor reads frame_
  std::vector<blk::OracleId> sent_;
  bool good_ = true, in_link_ = false;
  unsigned moments_ = 0;  // of the present link, so far
};

// The entire check, from the first segment's first message to the end of the proximity test.
// hooks[l], which must not be null (not checked), serves segment l.
[[nodiscard]] inline bool chain_verify(blk::VerifierContext &V, const ChainShape &fixed, const std::vector<LinkHook *> &hooks) {
  ChainCheck check(V, fixed);
  const blk::ReqShape &shape = check.frame().shape();
  BLK_REQUIRE(hooks.size() == shape.link.size(), "the chain: one hook for each link");
  for (size_t l = 0; l < shape.link.size(); l++) {
    LinkHook &hook = *hooks[l];
    if (!check.first() || !hook.run(0, V, check.frame(), check.sent())) return false;
    for (unsigned moment = 1; moment < shape.link[l].moments(); moment++)
      if (!check.later() || !hook.run(moment, V, check.frame(), check.sent())) return false;
    blk::ReqEquation equation;
    hook.claim(equation.target, equation.on_masks);
    equation.terms = hook.terms();
    if (!check.close(equation)) return false;
  }
  return check.finish();
}

} // namespace mb
