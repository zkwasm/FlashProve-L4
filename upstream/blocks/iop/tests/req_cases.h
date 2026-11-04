// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Synthetic requests for tests of the request frame (blocks/iop/req.h: a request's proof outside
// its sub-proofs), with stand-in sub-proofs (ToySub), a prover that runs on any `Engine`
// (blocks/engine/engine.h) and a verifier. With equal seeds two engines must give the same proof,
// byte for byte (compared in prover/engine/tests/last_run_test.cu).
// In identifiers, `link` is a segment (one sub-proof's part of a request's proof), `moment` a
// commit point (one of a segment's prover messages), `Row` a linear claim, `Row::target` its value.
#pragma once
#include "chain/closed.h"
#include "iop/req.h"
#include "iop/tests/iopp_support.h"

namespace blk {

// `life`: how many later segments can read an oracle message's main rows (ReqMessage::life).
struct ReqCase {
  const char *name;
  unsigned links;
  unsigned life;        // of the "kept cells", sent by segment l if l + life < links; 0: none
  bool at_start;        // a "rider" (oracle message, mask rows only) is in the first prover message
  unsigned target;      // bits
  unsigned work;        // bits of proof of work on draws of positions
  unsigned log_n;       // of the segments' base code
  bool reply_late = false;  // reply (see ToySub) sent after the last segment's first answers
  // Further kept oracle messages (main rows, ReqMessage::life), sent on the same condition as "kept
  // cells"; over F if the life is odd, else over E.
  std::vector<std::pair<size_t, unsigned>> more_kept = {};
};

// The test fixture (`world` in identifiers).
struct ReqWorld {
  Codes codes;
  ReqShape shape;
  ReqSizes sizes;
  Planned plan;
};

constexpr unsigned REQ_TOY_ROWS = 3;  // linear claims that a toy sub-proof leaves

// rows: linear claims per segment; ToySub needs the default. sub: plans what sub-proofs send; by
// default the claims' values after each segment's last commit point and the reply if `reply_late`.
inline ReqWorld req_toy_world(const ReqCase &t, unsigned rows = REQ_TOY_ROWS, const ReqSubPlan &sub = ReqSubPlan()) {
  Setting set = setting_goldilocks();
  set.target = t.target;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, t.work);
  ReqWorld w;
  Plan plan(set);

  // The segments' two codes. The random tails (r) must hide as many positions as are drawn, and the
  // number of draws depends on the rate: iterate upward from 0 draws. That 16 turns suffice is not
  // checked here.
  GroupSize g;
  g.name = "links", g.k = 4, g.h = 0;
  g.main.name = "rows", g.main.log_n = t.log_n, g.main.ell = u64(1) << (t.log_n - 2);
  g.mask.name = "mask row", g.mask.log_n = t.log_n - 1, g.mask.ell = 16, g.mask.over_E = true;
  u64 draws = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, draws);
    GroupLoad load;
    load.cube_bits = 40, load.batched_rows = u64(1) << 16;
    g = size_group(set, plan.scope(), g, load);
    BLK_REQUIRE(g.m, "test: no slack holds the group's rounds at the target");
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= draws) break;
    draws = more;
  }
  w.sizes.group = g;
  GroupSize last = g;
  last.name = "last group", last.k = 1;
  for (unsigned n = t.log_n - 2;; n++) {  // first n whose later-stage codes `solve_stages` accepts
    BLK_REQUIRE(n < t.log_n + 8, "test: no lengths serve the later stages");
    Plan trial(set);
    w.sizes.last = solve_stages(trial, last, draws, {StageChoice{3, 2, n, n - 1}, StageChoice{0, 0, n + 1, n + 1}});
    if (!trial.refused()) break;
  }
  w.shape.last = iopp_stages(w.sizes.last, w.codes);  // the segments' two codes get number 0
  w.shape.choice.masks_ride = true;

  auto message = [&](const char *name, size_t rows, unsigned deg, size_t mask_rows, unsigned moment, unsigned life) {
    ReqMessage m;
    m.shape.name = name, m.shape.code = 0, m.shape.main_rows = rows, m.shape.main_deg = deg, m.shape.mask_rows = mask_rows;
    m.shape.mask_code = 0, m.shape.answers = set.samples;
    m.moment = moment, m.life = life;
    return m;
  };
  for (unsigned l = 0; l < t.links; l++) {
    ReqLink link;
    link.name = "link " + std::to_string(l + 1);
    link.messages.push_back(message("cells", 3, 1, 1, 0, 0));
    if (t.life && l + t.life < t.links) link.messages.push_back(message("kept cells", 2, 1, 0, 0, t.life));
    for (const std::pair<size_t, unsigned> &x : t.more_kept)
      if (l + x.second < t.links) link.messages.push_back(message("more kept cells", x.first, x.second % 2 ? 1 : E_DEGREE, 0, 0, x.second));
    link.messages.push_back(message("reads", 1, E_DEGREE, 0, 1, 0));
    link.messages.push_back(message("digits", 2, 1, 0, 2, 0));
    link.messages.push_back(message("multiplicities", 1, 1, 0, 2, 0));
    if (l + 1 == t.links && !t.reply_late) link.plain = 1;  // the reply
    w.shape.link.push_back(link);
    w.sizes.rows.push_back(rows);
  }
  // No proximity test is run on the rider here. m = 3 is meant to give its group another list size
  // than the segments', so that req_plan()'s choice of group for the rider's sample shows.
  if (t.at_start) {
    OracleNotice n;
    n.shape.name = "masks for later", n.shape.mask_rows = 1, n.shape.mask_code = 0, n.shape.answers = set.samples;
    n.leaf_class = 1;
    w.shape.at_start.push_back(n);
    w.sizes.at_start = group_of_m(g, 3), w.sizes.at_start.name = "the rider's group";
  }
  plan_chain_start(plan, "request", true);
  req_plan(plan, w.codes, w.shape, w.sizes, sub ? sub : ReqSubPlan([&](Plan &p, size_t l, unsigned moment) {
    if (t.reply_late && l + 1 == t.links && moment == 0) plan_scalars(p, "a late reply", w.shape.link[l].name, 1);
    if (moment + 1 == w.shape.link[l].moments()) plan_scalars(p, "targets", w.shape.link[l].name, rows);
  }));
  return w.plan = plan.finish(), w;
}

// Handles to a run's public inputs in the hash chain's first query (hashed before any message).
// Both sides derive the claims' coefficients from `seed`.
struct ReqPublic {
  Public<Codes> codes;
  Public<ReqShape> shape;
  Public<u64> seed;
};
inline ReqPublic req_toy_bind(Binder &b, const ReqWorld &w, u64 seed) {
  ReqPublic p;
  b.configuration(toy_configuration());
  p.codes = b.parameter("codes", w.codes), p.shape = b.parameter("request", w.shape);
  p.seed = b.statement("seed of the rows", seed);
  return p;
}

struct ReqProved {
  std::vector<u8> proof;
  size_t trees = 0;
};

constexpr u64 REQ_TOY_REPLY = 7, REQ_TOY_REPLIES = 1000;  // default; well-formed replies are < 1000

// The toy sub-proof: random tables; after a segment's last commit point, three linear claims
// (toy_rows(), iopp_support.h) on its oracle messages and earlier segments' kept ones, whose values
// the prover sends.
// The reply is a public value that the prover sends only in the last segment: unless `reply_late`,
// a plain (unmasked) scalar in a round of its own before that segment's first prover message. Of
// the claims' coefficients, only that segment's depend on it.
struct ToySub {
  u64 seed;
  std::vector<OracleId> on;  // the current segment's oracle messages so far
  E late = E::zero();        // the reply, if `reply_late`
  bool is_last(const ReqFrame &frame) const { return frame.link() + 1 == frame.shape().link.size(); }
  bool last_moment(const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &sent) {
    if (moment == 0) on.clear();
    on.insert(on.end(), sent.begin(), sent.end());
    if (moment + 1 < frame.shape().link[frame.link()].moments()) return false;
    on.insert(on.end(), frame.kept_before().begin(), frame.kept_before().end());
    return true;
  }
  E reply(const ReqFrame &frame) const { return frame.plain().empty() ? late : frame.plain()[0]; }
  u64 seed_of_rows(const ReqFrame &frame) const { return seed + 10 + frame.link() + (is_last(frame) ? 1000 * reply(frame).c[0].v : 0); }
};
struct ToySubProver : ReqSubProver, ToySub {
  const ReqWorld &w;
  Engine &engine;
  Coins &coins;
  TestCoins data;
  unsigned false_at;  // this segment's first claim gets a false value; out of range: none
  u64 says;           // the reply
  ToySubProver(const ReqWorld &world, u64 s, Engine &e, Coins &c, unsigned f, u64 r)
      : ToySub{s, {}}, w(world), engine(e), coins(c), data(s + 1), false_at(f), says(r) {}
  std::vector<TableId> tables(size_t l, unsigned moment) override {
    std::vector<TableId> main;
    for (const ReqMessage &m : w.shape.link[l].messages)
      if (m.moment == moment) main.push_back(toy_draft(w.codes, OracleNotice{m.shape, 0}, engine, coins, data).main);
    return main;
  }
  std::vector<OracleDraft> at_start() override {
    std::vector<OracleDraft> d;
    for (const OracleNotice &n : w.shape.at_start) d.push_back(toy_draft(w.codes, n, engine, coins, data));
    return d;
  }
  std::vector<E> plain(size_t l) override { return std::vector<E>(w.shape.link[l].plain, E::from(from_u64(says))); }
  void run(ProverContext &P, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &sent) override {
    if (is_last(frame) && moment == 0 && frame.plain().empty()) P.ch.send(late = E::from(from_u64(says)));
    if (!last_moment(frame, moment, sent)) return;
    bool head = true;
    for (Row &row : toy_rows(P.st, on, seed_of_rows(frame))) {
      row.target = row_value(P, row) + (head && frame.link() == false_at ? E::one() : E::zero());
      head = false;
      P.ch.send(row.target);
      P.st.add_row(frame.group(), std::move(row));
    }
  }
};
// Of the reply it checks only the form (see ReqInputs for why) and passes it on in `received`.
struct ToySubVerifier : ReqSubVerifier, ToySub {
  bool replied = false;
  E received = E::zero();
  explicit ToySubVerifier(u64 s) : ToySub{s, {}} {}
  bool run(VerifierContext &V, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &sent) override {
    if (is_last(frame) && moment == 0) {
      if (frame.plain().empty() && !V.ch.recv(late)) return false;
      const E r = reply(frame);
      if (!is_zero(r.c[1]) || !is_zero(r.c[2]) || r.c[0].v >= REQ_TOY_REPLIES) return false;  // the form of a reply
      replied = true, received = r;
    }
    if (!last_moment(frame, moment, sent)) return true;
    for (Row &row : toy_rows(V.st, on, seed_of_rows(frame))) {
      if (!V.ch.recv(row.target)) return false;
      V.st.add_row(frame.group(), std::move(row));
    }
    return true;
  }
};

inline ReqProved req_toy_prove(const ReqWorld &w, u64 seed, unsigned false_at, Engine &engine, u64 reply = REQ_TOY_REPLY) {
  TestCoins coins(toy_coins_seed(seed));
  Writer proof;
  Binder binder(1, w.plan);
  const ReqPublic bound = req_toy_bind(binder, w, seed);
  ProverRounds ch(binder, toy_random_bytes(seed + 2), proof);
  ProverContext P(bound.codes, ch, engine, coins);
  P.pads = toy_pads;
  ToySubProver sub(w, ch.get(bound.seed), engine, coins, false_at, reply);
  req_prove(P, bound.shape, sub);
  ch.finish();
  ReqProved out;
  out.proof = proof.bytes();
  out.trees = P.book.trees();
  return out;
}

// The verifier's side runs under run_verifier() (blocks/chain/closed.h): req_toy_decide() gets the
// channel and handles to what the first query contains, not the fixture. The reply that the client
// was given is not in the first query, so it cannot reach req_toy_decide(): the reply as received
// goes out, and the caller compares after the verdict.
struct ReqInputs {
  const ReqWorld &w;
  u64 seed;
};
struct ReqHandedOn {
  bool replied = false;
  E reply = E::zero();
};
inline ReqPublic req_toy_bind_inputs(Binder &b, const ReqInputs &in) { return req_toy_bind(b, in.w, in.seed); }
inline bool req_toy_decide(VerifierRounds &ch, const ReqPublic &bound, ReqHandedOn &out, Notes *) {
  VerifierContext V(bound.codes, ch);
  ToySubVerifier sub(ch.get(bound.seed));
  if (!req_verify(V, bound.shape, sub)) return false;
  out.replied = sub.replied, out.reply = sub.received;
  return true;
}
// reply: what the client was given. compare = false: accepts any well-formed reply in the proof.
inline bool req_toy_verify(const ReqWorld &w, u64 seed, const std::vector<u8> &proof, u64 reply = REQ_TOY_REPLY, bool compare = true) {
  ReqHandedOn out;
  if (!run_verifier(1, w.plan, ReqInputs{w, seed}, req_toy_bind_inputs, req_toy_decide, proof.data(), proof.size(), out)) return false;
  return !compare || (out.replied && out.reply == E::from(from_u64(reply)));
}

inline std::vector<ReqCase> req_toy_cases() {
  return {
      {"two links", 2, 0, false, 128, 0, 12},
      {"three links, kept for one", 3, 1, false, 128, 0, 12},
      {"four links, kept for two, work, a rider", 4, 2, true, 128, 8, 13},
  };
}

}  // namespace blk
