// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Prover and verifier for a segment chain at toy size, on the frame of blocks/iop/req.h. A segment
// (also `link` in identifiers and strings) is one sub-proof's part of a proof. The prover knows
// vectors x_1 and b_1 .. b_L over F and shows that it knows c_l = x_l * b_l (entry by entry) with
// x_{l+1} = c_l; masks are meant to hide the vectors. Segment l sends two oracle messages: b_l
// (after x_1 in segment 1) with the segment's mask row, and c_l. The next segment reads c_l's row,
// a kept row, as the frame sends it again. Each sub-proof is ZERO (the verifier's random point) and
// PSC (the masked sumcheck), as in blocks/toy/product.h; it leaves linear claims (`Row`) on these
// messages. A group is rows that are read at one list of positions and tested together. The frame
// reduces a group's claims to one (BAT batches linear claims into one; FOLD is a masked sumcheck
// over the group's rows) and carries it into the next segment's group; a full proximity test ends
// the run.
// A vector fills the lower half of a row. The upper half is zero, except that the row of a message
// without a mask row (c's) ends in the pad: random entries that mask the answers to its
// out-of-domain sample.
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "engine/engine.h"
#include "engine/sumcheck_cpu.h"
#include "iop/req.h"
#include "toy/product.h"

namespace blk {

struct SegmentsRun {
  Setting set;
  std::string text; // the public file: the run's parameters (`toy_text`)
  Codes codes;
  ReqShape req;
  Planned plan;
  explicit SegmentsRun(const Setting &s) : set(s) {}
  unsigned vars() const { return codes.base[0].log_ell() - 1; } // log2 of a vector's length
  size_t links() const { return req.link.size(); }
};

constexpr u64 SEGMENT_ROWS = 3; // the linear claims that a segment's sub-proof leaves

inline void segments_plan_link(Plan &plan, const std::string &where, unsigned vars) {
  plan_zero(plan, where, vars);
  plan_psc(plan, where, psc_size(toy_claim(std::vector<E>(vars)), toy_summand())); // sizes read off what is run
}

// log_ell: log2 of a main row's message length; that length is twice a vector's. The other sizes
// are those of toy_size(). run_words: the words after "run " in the public file's first line.
// If run.plan->holds() is false, segments_prove() ends the process and segments_verify() returns
// false. Not checked directly: that the pad (E_DEGREE * set.samples entries) fits a row's upper
// half; if not, send_prove() or, on either side, Statement::add_row() ends the process.
// The sizing follows toy_size() (blocks/toy/product.h).
inline SegmentsRun segments_size(const Setting &set, size_t links, unsigned log_ell, unsigned log_n, unsigned mask_log_n,
                                 const std::vector<StageChoice> &further, const char *run_words = "links") {
  SegmentsRun run(set);
  run.text = toy_text(run_words, set, {{"links", links}, {"log-length", log_ell}, {"log-n", log_n}, {"mask-log-n", mask_log_n}}, further);
  Plan plan(set);
  const unsigned vars = log_ell - 1;
  ReqShape &req = run.req;
  for (size_t l = 0; l < links; l++) {
    ReqLink link;
    link.name = "link " + std::to_string(l + 1);
    ReqMessage own, product;
    own.shape.name = link.name + ": b", own.shape.code = 0, own.shape.main_rows = l == 0 ? 2 : 1, own.shape.mask_rows = 1;
    own.shape.answers = set.samples;
    product.shape.name = link.name + ": c", product.shape.code = 0, product.shape.main_rows = 1, product.shape.answers = set.samples;
    product.life = l + 1 < links ? 1 : 0;
    link.messages = {own, product};
    req.link.push_back(link);
  }
  req.answers = set.samples;
  req.choice.masks_ride = true; // the last test's fresh masks too are sent with their stages' rows
  const std::vector<ReqGroup> groups = req_groups(req);

  GroupSize g;
  g.name = "links";
  for (size_t l = 0; l < links; l++) g.k = std::max(g.k, groups[l].k);
  g.h = 0, g.slots = 2; // a main row and one kept row side by side
  g.main.name = "rows", g.main.log_n = log_n, g.main.ell = u64(1) << log_ell;
  // The mask entries that PSC takes: the hiding polynomial's coefficients and a mask for each of
  // the two values (of x and b) that it sends.
  const u64 masks = 1 + 2 * u64(vars) + 2 + 2;
  g.mask.name = "mask row", g.mask.log_n = mask_log_n, g.mask.over_E = true;
  g.mask.ell = pow2_at_least(std::max<u64>(masks + set.samples, 2 * g.k + 1));
  // A row's tail (its `r` random coordinates) must hide as many positions as are drawn, and that
  // number, t, depends on the rate, so on the tail: t rises from 0 until it settles, for at most
  // 16 turns. Each t is the least that leaves 2^-(set.target + 30) of the allowed error for the
  // batching term of the last test's first code switch (plan_cs(), blocks/numbers/blocks.h).
  u64 t = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, t);
    GroupLoad load;
    load.cube_bits = stage_cube_bits(g);
    // An assumed bound: 16 for a sub-proof's claims and the sample claims, and four generator
    // claims (one for each symbol that a code switch reads: blocks/iop/cs.h) for each of
    // 4 * set.target positions. A plan that exceeds it is not thereby accepted
    // (`GroupLoad`, blocks/numbers/blocks.h).
    load.batched_rows = 16 + 4 * 4 * (u64)set.target; // rows of a sub-proof, samples, and four generator rows a position
    // size_group() chooses m (GroupSize::m), which sets the distance of the group's test: below
    // the Johnson radius by a slack that shrinks as m grows.
    g = size_group(set, plan.scope(), g, load);
    if (analysis(set, g).bad()) {  // size_group has found no m
      plan.refuse("no slack holds the links' rounds at the target");
      return run.plan = plan.finish(), run;
    }
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= t) break;
    t = more;
  }
  GroupSize last = g; // group of the last carried oracle message: first stage of the last test
  last.name = "last group", last.k = groups.back().k, last.h = groups.back().h, last.slots = 1;
  ReqSizes sizes;
  sizes.group = g;
  sizes.last = solve_stages(plan, last, t, further);
  if (plan.refused()) return run.plan = plan.finish(), run;
  sizes.rows.assign(links, SEGMENT_ROWS);
  req.last = iopp_stages(sizes.last, run.codes);
  if (const char *fault = req_fault(run.codes, req)) {
    plan.refuse(fault);
    return run.plan = plan.finish(), run;
  }
  plan_chain_start(plan, "segments", true);
  req_plan(plan, run.codes, req, sizes, [&](Plan &p, size_t l, unsigned) { segments_plan_link(p, req.link[l].name, vars); });
  return run.plan = plan.finish(), run;
}

// Handles to a run's public inputs. All are parameters in the hash chain's first query: the
// statement has no public part.
struct SegmentsPublic {
  Public<Digest> configuration;
  Public<Codes> codes;
  Public<ReqShape> req;
};
// This protocol's hash-chain label.
constexpr u64 SEGMENTS_LABEL = 2;
inline SegmentsPublic segments_bind(Binder &binder, const SegmentsRun &run) {
  SegmentsPublic in;
  in.configuration = binder.configuration(configuration_digest({run.text}));
  in.codes = binder.parameter("codes", run.codes);
  in.req = binder.parameter("the chain of links", run.req);
  return in;
}

// Coefficients on o's main table for `scalar` times the extension at `point` of row `row`'s vector.
inline Tensor segments_vector_at(const Statement &st, OracleId o, u64 row, const std::vector<E> &point, const E &scalar) {
  Tensor v = Tensor::eq(point.data(), (unsigned)point.size(), scalar);
  v.push(E::one(), E::zero()); // the lower half of the row
  v.insert_choice(0, st.row_bits(o, Part::MAIN), row);
  return v;
}

// Which oracle messages contain a segment's x, b and c, and in which rows; c is in row 0.
struct SegmentPlaces {
  OracleId x, b, c;
  u64 x_row, b_row;
};
inline SegmentPlaces segments_places(const ReqFrame &frame, const std::vector<OracleId> &own) {
  if (frame.link() == 0) return SegmentPlaces{own[0], own[0], own[1], 0, 1};
  return SegmentPlaces{frame.kept_before()[0], own[0], own[1], 0, 0};
}

// Adds what PSC leaves to the segment's group g, as SEGMENT_ROWS linear claims. Both sides call it.
inline void segments_rows(Statement &st, GroupId g, const SegmentPlaces &at, PscOutput<E> &&out) {
  Row u;
  u.name = "hiding polynomial and c";
  for (MaskTerm<E> &m : out.row_masks) u.add_mask(std::move(m));
  u.add(at.c, Part::MAIN, segments_vector_at(st, at.c, 0, out.point, out.row_maps[0].coeff));
  u.target = out.row_target;
  st.add_row(g, std::move(u));
  for (size_t j = 0; j < 2; j++) {
    Row e;
    e.name = j == 0 ? "x at the point" : "b at the point";
    e.add(j == 0 ? at.x : at.b, Part::MAIN, segments_vector_at(st, j == 0 ? at.x : at.b, j == 0 ? at.x_row : at.b_row, out.ends[j].point, E::one()));
    for (MaskTerm<E> &m : out.ends[j].masks) e.add_mask(std::move(m));
    e.target = out.ends[j].value;
    st.add_row(g, std::move(e));
  }
}

// x: x_1; b: b_1 .. b_L, run.links() vectors. All must have 2^run.vars() entries. Not checked.
inline void segments_prove(const SegmentsRun &run, std::vector<F> x, const std::vector<std::vector<F>> &b, Engine &engine, Coins &coins,
                           Writer &proof) {
  Binder binder(SEGMENTS_LABEL, run.plan);
  const SegmentsPublic in = segments_bind(binder, run);
  ProverRounds ch(binder, random_digest(coins), proof, [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); });
  ProverContext pr(in.codes, ch, engine, coins);
  ReqProver frame(pr, in.req);
  const Codes &codes = pr.st.codes();
  const ReqShape &req = ch.get(in.req);
  const unsigned vars = codes.base[req.code].log_ell() - 1;
  const size_t len = size_t(1) << vars, ell = codes.base[req.code].ell;
  const Summand<E> summand = toy_summand();
  for (size_t l = 0; l < req.link.size(); l++) {
    std::vector<F> c(len);
    for (size_t i = 0; i < len; i++) c[i] = x[i] * b[l][i];
    Table own(1, l == 0 ? 2 : 1, ell), product(1, 1, ell);
    for (size_t i = 0; i < len; i++) {
      if (l == 0) own.set(0, i, E::from(x[i]));
      own.set(l == 0 ? 1 : 0, i, E::from(b[l][i]));
      product.set(0, i, E::from(c[i]));
    }
    // The pad's entries in c's row stay zero: send_prove() requires that and fills them.
    const std::vector<OracleId> ids = frame.first({engine.put(own), engine.put(product)});
    const SegmentPlaces at = segments_places(frame, ids);

    std::vector<E> zeta(vars);
    {
      AlgebraicDraw from = ch.close_algebraic(R_ZERO);
      for (E &z : zeta) z = from.draw();
    }
    SumClaim<E> claim = toy_claim(zeta);
    std::vector<std::vector<E>> maps(3, std::vector<E>(len));
    for (size_t i = 0; i < len; i++) maps[0][i] = E::from(x[i]), maps[1][i] = E::from(b[l][i]), maps[2][i] = E::from(c[i]);
    SumcheckCpu<E> inner(summand, claim.vars, std::move(maps));
    std::vector<MaskSlot> fresh = pr.st.masks(ids[0]).take(psc_masks(claim, summand));
    PscOutput<E> out = psc_prove<E>(ch, std::move(claim), summand, inner, std::move(fresh),
                                    [&pr](u32 oracle, u32 index) { return pr.behind[oracle].mask.entry(index); });
    segments_rows(pr.st, frame.group(), at, std::move(out));
    frame.close();
    x = std::move(c);
  }
  frame.finish();
  ch.finish();
  pr.st.require_all_tested();
}

// Sizes and plans a run from its public file only. False unless the sized run's text equals `text`.
inline bool segments_from_text(const std::string &text, std::unique_ptr<SegmentsRun> &run, const char **why) {
  ToyNumbers n;
  Setting set;
  const bool read = toy_read_text(text, n) && toy_setting(n, set) && n.has("links") && n.has("log-length") && n.has("log-n") && n.has("mask-log-n") &&
                    n["links"] >= 1 && n["links"] <= 64 && n["log-length"] >= 2 && n["log-length"] <= 20 && n["log-n"] <= 24 && n["mask-log-n"] <= 24;
  if (read)
    run = std::make_unique<SegmentsRun>(segments_size(set, (size_t)n["links"], (unsigned)n["log-length"], (unsigned)n["log-n"],
                                                      (unsigned)n["mask-log-n"], n.further, n.run_words.c_str()));
  if (!read || run->text != text) {
    if (why) *why = "the public file is not a file of this run in its one spelling";
    return false;
  }
  return true;
}

// The verifier's side, for run_verifier() (blocks/chain/closed.h): it gets handles, not the run.
inline bool segments_decide(VerifierRounds &ch, const SegmentsPublic &in, Notes *notes) {
  VerifierContext vf(in.codes, ch);
  vf.notes = notes;
  ReqVerifier frame(vf, in.req);
  const ReqShape &req = ch.get(in.req);
  const unsigned vars = vf.st.codes().base[req.code].log_ell() - 1;
  const Summand<E> summand = toy_summand();
  for (size_t l = 0; l < req.link.size(); l++) {
    std::vector<OracleId> ids;
    if (!frame.first(ids)) return false;
    const SegmentPlaces at = segments_places(frame, ids);
    std::vector<E> zeta(vars);
    {
      AlgebraicDraw from = ch.close_algebraic(R_ZERO);
      for (E &z : zeta) z = from.draw();
    }
    SumClaim<E> claim = toy_claim(zeta);
    std::vector<MaskSlot> fresh = vf.st.masks(ids[0]).take(psc_masks(claim, summand));
    PscOutput<E> out;
    if (!psc_verify<E>(ch, std::move(claim), summand, std::move(fresh), out)) return false;
    segments_rows(vf.st, frame.group(), at, std::move(out));
    if (!frame.close()) return false;
  }
  if (!frame.finish()) return false;
  vf.st.require_all_tested();
  return true;
}
inline bool segments_verify(const SegmentsRun &run, const u8 *proof, size_t bytes, const char **why = nullptr, Notes *notes = nullptr) {
  return run_verifier(SEGMENTS_LABEL, run.plan, run, segments_bind, segments_decide, proof, bytes, why, notes);
}
// The complete verifier: its only inputs are the public file's text and the proof.
inline bool segments_verify_file(const std::string &text, const u8 *proof, size_t bytes, const char **why = nullptr, Notes *notes = nullptr) {
  std::unique_ptr<SegmentsRun> run;
  return segments_from_text(text, run, why) && segments_verify(*run, proof, bytes, why, notes);
}

} // namespace blk
