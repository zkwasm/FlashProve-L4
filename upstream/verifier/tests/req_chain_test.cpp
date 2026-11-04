// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests verifier/src/vf_request_chain.h, the verifier's check of a request's proof outside its sub-proofs
// (arguments for parts of the forward pass): mb::chain_verify(), which calls one hook per
// segment (a sub-proof's part of the proof; `link` in identifiers), and mb::ChainCheck, whose
// methods the caller calls itself. The library's prover (blocks/iop/req.h) proves the toy requests
// of blocks/iop/tests/req_rows_cases.h: each segment has one equation, on its matrices and those
// kept from up to two earlier segments; the last sends a plain (unmasked) scalar, the "reply".
// Both checks must accept honest proofs and reject (printed: "refused") a false value (`target`) of
// the first equation. chain_verify() must reject an honest proof if the first hook changes a weight
// of its equation, and the last hook alone must see the reply. A ChainCheck run that omits a due
// call must be rejected; a second close() is not tried.
// Build, from verifier/src: g++ -std=c++17 -O2 -fno-ipa-modref -DBLK_FIELD=2 -DBLK_HASH=1 -I . -I ../../blocks ../tests/req_chain_test.cpp
#include "vf_request_chain.h"

#include "iop/tests/req_rows_cases.h"

struct MadeUpHook : mb::LinkHook {
  const blk::ReqWorld &w;
  const blk::ReqLayout &layout;
  size_t link;
  blk::u64 seed;
  bool spoil;
  blk::E target;
  size_t plain_seen = 0;
  MadeUpHook(const blk::ReqWorld &world, const blk::ReqLayout &lay, size_t l, blk::u64 s, bool sp) : w(world), layout(lay), link(l), seed(s), spoil(sp) {}
  bool run(unsigned moment, blk::VerifierContext &V, const blk::ReqFrame &frame, const std::vector<blk::OracleId> &) override {
    plain_seen = frame.plain().size();
    return moment + 1 < w.shape.link[link].moments() || V.ch.recv(target);  // the equation's value
  }
  void claim(blk::E &t, std::vector<blk::E> &on_masks) override { t = target, on_masks = blk::rows_toy_equation(w, layout, link, seed, false).on_masks; }
  std::vector<blk::ReqTerm> terms() override { return blk::rows_toy_equation(w, layout, link, seed, spoil).terms; }
};

static int bad = 0;
static size_t had_to_be_refused = 0; // runs that must be rejected; 0 (an empty loop) fails the test
static bool accepted(bool verdict) { return had_to_be_refused++, verdict; }
static bool verify(const blk::ReqWorld &w, const blk::ReqLayout &layout, blk::u64 seed, const std::vector<blk::u8> &proof, bool spoil) {
  blk::Binder binder(1, w.plan);
  const blk::RowsPublic bound = blk::rows_toy_bind(binder, w, layout, seed);
  blk::Reader reader(proof.data(), proof.size());
  blk::VerifierSession session;  // one verdict, one chain
  blk::VerifierRounds channel(binder, reader, session);
  blk::VerifierContext V(bound.codes, channel);
  std::vector<MadeUpHook> made;
  for (size_t l = 0; l < w.shape.link.size(); l++) made.emplace_back(w, channel.get(bound.layout), l, channel.get(bound.seed), spoil && l == 0);
  std::vector<mb::LinkHook *> hooks;
  for (MadeUpHook &h : made) hooks.push_back(&h);
  if (!mb::chain_verify(V, mb::ChainShape{bound.shape, bound.layout}, hooks)) return false;
  for (size_t l = 0; l < made.size(); l++)
    if (made[l].plain_seen != (l + 1 == made.size() ? 1u : 0u)) bad++, std::printf("the reply is seen by another hook than the last link's\n");
  return channel.finish();
}

// Checks the proof through mb::ChainCheck. If no_close or no_moment is a segment's number, the run
// skips that segment's close(), or its last commit point (`moment`), as a sub-proof's function that
// returns early would.
static bool verify_by_calls(const blk::ReqWorld &w, const blk::ReqLayout &layout, blk::u64 seed, const std::vector<blk::u8> &proof, size_t no_close = ~size_t(0),
                            size_t no_moment = ~size_t(0)) {
  blk::Binder binder(1, w.plan);
  const blk::RowsPublic bound = blk::rows_toy_bind(binder, w, layout, seed);
  blk::Reader reader(proof.data(), proof.size());
  blk::VerifierSession session;  // one verdict, one chain
  blk::VerifierRounds channel(binder, reader, session);
  blk::VerifierContext V(bound.codes, channel);
  mb::ChainCheck check(V, mb::ChainShape{bound.shape, bound.layout});
  for (size_t l = 0; l < w.shape.link.size(); l++) {
    if (!check.first()) return false;
    for (unsigned moment = 1; moment + (l == no_moment ? 1 : 0) < w.shape.link[l].moments(); moment++)
      if (!check.later()) return false;
    if (l == no_close) continue;
    blk::ReqEquation equation = blk::rows_toy_equation(w, channel.get(bound.layout), l, channel.get(bound.seed), false);
    if (!V.ch.recv(equation.target) || !check.close(equation)) return false;
  }
  return check.finish() && channel.finish();
}

int main() {
  for (const blk::ReqCase &t : blk::rows_toy_cases()) {
    const blk::ReqWorld w = blk::req_toy_world(t, 1);
    const blk::ReqLayout layout = blk::rows_toy_layout(w);
    blk::EngineCpu engine, again;
    const std::vector<blk::u8> proof = blk::rows_prove(w, layout, 800, blk::E::zero(), engine);
    if (!verify(w, layout, 800, proof, false)) bad++, std::printf("an honest proof is refused: %s\n", t.name);
    if (!verify_by_calls(w, layout, 800, proof)) bad++, std::printf("an honest proof is refused by the check in the form of calls: %s\n", t.name);
    for (size_t l = 0; l < w.shape.link.size(); l++) {
      if (accepted(verify_by_calls(w, layout, 800, proof, l))) bad++, std::printf("A LINK IS LEFT WITHOUT ITS CLOSE, AND THE RUN IS ACCEPTED: %s, link %zu\n", t.name, l);
      if (accepted(verify_by_calls(w, layout, 800, proof, ~size_t(0), l))) bad++, std::printf("A MOMENT IS LEFT OUT, AND THE RUN IS ACCEPTED: %s, link %zu\n", t.name, l);
    }
    const std::vector<blk::u8> false_proof = blk::rows_prove(w, layout, 800, blk::E::one(), again);
    if (accepted(verify(w, layout, 800, false_proof, false)) | accepted(verify_by_calls(w, layout, 800, false_proof))) bad++, std::printf("A FALSE TARGET IS ACCEPTED: %s\n", t.name);
    if (accepted(verify(w, layout, 800, proof, true))) bad++, std::printf("ANOTHER WEIGHT IS ACCEPTED: %s\n", t.name);
  }
  if (!had_to_be_refused) bad++, std::printf("no run that had to be refused was played\n");
  std::printf("req_chain_test: %s (%zu runs had to be refused)\n", bad ? "FAILED" : "ok", had_to_be_refused);
  return bad ? 1 : 0;
}
