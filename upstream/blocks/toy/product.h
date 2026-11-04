// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A complete run of the proof system at toy size. The prover knows three vectors a, b, c over F
// with a[i] * b[i] = c[i] for all i and shows that; masks are meant to hide them. The steps:
//   SEND   one oracle message (encoded rows under a Merkle tree): main rows a, b, c and one mask
//          row; its out-of-domain sample; in the same prover message and tree, the fresh mask of
//          the first fold of the proximity test (`iopp` in identifiers);
//   ZERO   the verifier's point zeta: the claim becomes  sum_i eq(zeta, i) (a[i] b[i] - c[i]) = 0;
//   PSC    the sumcheck with a hiding polynomial, on that sum; it leaves linear claims with public
//          values (`Row`, `target` in identifiers) on the oracle message;
//   the proximity test on the oracle message's group (the oracle messages that one test reads),
//   which takes in those claims and the sample's.
// toy_size() (the plan), toy_prove() and toy_decide() (the verifier) take the steps in this order.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "chain/closed.h"
#include "engine/engine.h"
#include "engine/sumcheck_cpu.h"
#include "frame/heads.h"
#include "iop/bat.h"
#include "iop/context.h"
#include "iop/iopp.h"
#include "iop/psc.h"
#include "iop/send.h"
#include "numbers/blocks.h"

namespace blk {

// A toy run's only public file: the numbers from which the run is sized and planned.
// Its digest is the run's configuration digest.
inline std::string toy_text(const char *run, const Setting &set, const std::vector<std::pair<const char *, u64>> &numbers,
                            const std::vector<StageChoice> &further) {
  std::string text = std::string("run ") + run + "\nformat " + version_text(SERVED_VERSION) + "\n";
  auto line = [&text](const char *name, u64 x) { text += std::string(name) + " " + std::to_string(x) + "\n"; };
  line("target", set.target), line("work-on-positions", set.work[(int)ErrorKind::QUERY][(int)Scope::REQUEST]);
  line("salt-bytes", set.salt_bytes), line("nonce-bytes", set.nonce_bytes), line("position-bytes", set.position_bytes), line("samples", set.samples);
  for (const auto &n : numbers) line(n.first, n.second);
  for (const StageChoice &s : further)
    text += "stage " + std::to_string(s.k) + " " + std::to_string(s.h) + " " + std::to_string(s.log_n) + " " + std::to_string(s.mask_log_n) + "\n";
  return text;
}

// A parsed public file. toy_read_text() is lenient (any order, unknown or repeated names, leading
// zeros): a caller must compare the sized run's text with the text read, as toy_from_text() does,
// so that a file has one spelling. Numbers have at most 6 digits, to keep a file from asking for
// a huge run.
struct ToyNumbers {
  std::string run_words, format;
  std::vector<std::pair<std::string, u64>> numbers;
  std::vector<StageChoice> further;
  bool has(const char *name) const {
    for (const auto &n : numbers)
      if (n.first == name) return true;
    return false;
  }
  u64 operator[](const char *name) const {
    for (const auto &n : numbers)
      if (n.first == name) return n.second;
    return 0;
  }
};
inline bool toy_read_text(const std::string &text, ToyNumbers &out) {
  out = ToyNumbers();
  size_t at = 0;
  while (at < text.size()) {
    const size_t end = text.find('\n', at);
    if (end == std::string::npos) return false;
    const std::string line = text.substr(at, end - at);
    at = end + 1;
    const size_t space = line.find(' ');
    if (space == std::string::npos) return false;
    const std::string name = line.substr(0, space), rest = line.substr(space + 1);
    if (name == "run") {
      out.run_words = rest;
    } else if (name == "format") {
      out.format = rest;
    } else {
      std::vector<u64> x;
      for (size_t i = 0; i < rest.size();) {
        u64 v = 0;
        const size_t first = i;
        for (; i < rest.size() && rest[i] >= '0' && rest[i] <= '9' && i - first < 6; i++) v = 10 * v + (u64)(rest[i] - '0');
        if (i == first || (i < rest.size() && rest[i] != ' ')) return false;
        x.push_back(v);
        i++;
      }
      if (name == "stage" && x.size() == 4)
        out.further.push_back(StageChoice{(unsigned)x[0], (unsigned)x[1], (unsigned)x[2], (unsigned)x[3]});
      else if (name != "stage" && x.size() == 1)
        out.numbers.emplace_back(name, x[0]);
      else
        return false;
    }
  }
  return true;
}
// The `Setting` of a public file: the Goldilocks field, the build's hash, the file's six numbers,
// of which `position-bytes` must be 0: a proof lists no drawn position.
inline bool toy_setting(const ToyNumbers &n, Setting &set) {
  for (const char *name : {"target", "work-on-positions", "salt-bytes", "nonce-bytes", "position-bytes", "samples"})
    if (!n.has(name)) return false;
  if (n.format != version_text(SERVED_VERSION) || n["target"] > 256 || n["work-on-positions"] > 40 || n["position-bytes"] != 0) return false;
  set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = (unsigned)n["target"], set.set_work(ErrorKind::QUERY, Scope::REQUEST, (unsigned)n["work-on-positions"]);
  set.salt_bytes = n["salt-bytes"], set.nonce_bytes = n["nonce-bytes"], set.samples = n["samples"];
  return true;
}

struct ToyShape {
  Setting set;
  std::string text; // the public file
  Codes codes;      // base[0]: the rows a, b, c; mask[0]: the mask row
  unsigned vars() const { return codes.base[0].log_ell(); }
  OracleShape oracle() const {
    OracleShape s;
    s.name = "a, b, c";
    s.code = 0, s.main_rows = 3, s.main_deg = 1, s.mask_rows = 1, s.answers = set.samples;
    return s;
  }
};

// S = Y_1 Y_2 - U_1 (Y: the `shown` maps a, b; U: the `kept` map c): PSC sends the values of a
// and b at its challenge point under masks; c stays a term of a linear claim that PSC leaves.
inline Summand<E> toy_summand() {
  Summand<E> s;
  s.shown = 2, s.kept = 1, s.degree = 2;
  s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] * y[1], -E::one()}; };
  return s;
}
// 3: coefficients of a round polynomial (degree 2).
inline SumClaim<E> toy_claim(const std::vector<E> &zeta) {
  SumClaim<E> claim;
  for (const E &z : zeta) claim.vars.push_back(Weight<E>::equality(z, 3));
  claim.target = E::zero();
  return claim;
}

// Coefficients on o's main rows for `scalar` times row `row`'s multilinear extension at `point`.
inline Tensor toy_row_at(const Statement &st, OracleId o, u64 row, const std::vector<E> &point, const E &scalar) {
  Tensor v = Tensor::eq(point.data(), (unsigned)point.size(), scalar);
  v.insert_choice(0, st.row_bits(o, Part::MAIN), row);
  return v;
}

// Adds what PSC leaves to group g, as three linear claims. Both sides call it.
inline void toy_rows(Statement &st, GroupId g, OracleId o, PscOutput<E> &&out) {
  Row u;
  u.name = "hiding polynomial and c";
  for (MaskTerm<E> &m : out.row_masks) u.add_mask(std::move(m));
  for (const KeptTerm<E> &t : out.row_maps) u.add(o, Part::MAIN, toy_row_at(st, o, 2 + t.map, out.point, t.coeff));
  u.target = out.row_target;
  st.add_row(g, std::move(u));
  for (size_t j = 0; j < out.ends.size(); j++) {
    Row e;
    e.name = j == 0 ? "a at the point" : "b at the point";
    e.add(o, Part::MAIN, toy_row_at(st, o, j, out.ends[j].point, E::one()));
    for (MaskTerm<E> &m : out.ends[j].masks) e.add_mask(std::move(m));
    e.target = out.ends[j].value;
    st.add_row(g, std::move(e));
  }
}

// After the proximity test's first stage, a stage's fresh mask is sent with its rows, in one tree.
inline IoppChoice toy_choice() {
  IoppChoice choice;
  choice.masks_ride = true;
  return choice;
}
// The first prover message's oracle messages. Empty `stages`: no proximity test follows.
inline std::vector<OracleNotice> toy_first_message(const OracleNotice &abc, const std::vector<IoppStage> &stages) {
  std::vector<OracleNotice> sent = {abc};
  if (!stages.empty()) sent.push_back(fold_mask_notice(iopp_fold_shape(stages[0], GroupLayout())));
  return sent;
}

// Handles to a run's public inputs in the first query (hashed before the prover speaks) of the
// hash chain (the Fiat-Shamir transcript). All are parameters: the statement (the prover knows
// such vectors, of the length that the codes fix) has no public part.
struct ToyPublic {
  Public<Digest> configuration;
  Public<Codes> codes;
  Public<OracleNotice> oracle;
  Public<std::vector<IoppStage>> stages;
  Public<IoppChoice> choice;
};
constexpr u64 TOY_LABEL = 1;
inline ToyPublic toy_bind(Binder &binder, const ToyShape &shape, const std::vector<IoppStage> &stages) {
  ToyPublic in;
  in.configuration = binder.configuration(configuration_digest({shape.text}));
  in.codes = binder.parameter("codes", shape.codes);
  in.oracle = binder.parameter("the oracle message a, b, c", OracleNotice{shape.oracle(), 0});
  in.stages = binder.parameter("the stages of the proximity test", stages);
  in.choice = binder.parameter("the choices of the proximity test", toy_choice());
  return in;
}

// ---- the preamble: everything before the proximity test ----

inline void toy_plan_preamble(Plan &plan, const ToyShape &shape, const std::vector<IoppStage> &stages, const TestedIn &tested_in) {
  const std::vector<OracleNotice> sent = toy_first_message(OracleNotice{shape.oracle(), 0}, stages);
  plan_send(plan, "a, b, c", {SampleSize{Send::cube(shape.codes, shape.oracle()).bits, tested_in}},
            (unsigned)tree_sizes(shape.codes, sent).size());
  plan_zero(plan, "a b = c", shape.vars());
  plan_psc(plan, "a b = c", psc_size(toy_claim(std::vector<E>(shape.vars())), toy_summand())); // sizes read off what is run
}
// Linear claims that the proximity test takes in: three from PSC, one for each sample answer.
inline u64 toy_rows_batched(const Setting &set) { return 3 + set.samples; }

// abc: the rows a, b, c. If some a[i] * b[i] != c[i], psc_prove() ends the process, with
// overwhelming probability. Returns the group still to be tested. mask: receives the first fold's
// fresh mask, if sent.
inline GroupId toy_prove_preamble(ProverContext &pr, const ToyPublic &in, const Table &abc, FoldMask &mask) {
  const OracleNotice &notice = pr.ch.get(in.oracle);
  const std::vector<IoppStage> &stages = pr.ch.get(in.stages);
  OracleDraft draft;
  draft.shape = notice.shape, draft.leaf_class = notice.leaf_class;
  draft.main = pr.engine.put(abc);
  draft.mask = Send::fresh_mask(pr.st.codes(), draft.shape, pr.coins);
  std::vector<OracleDraft> drafts;
  drafts.push_back(std::move(draft));
  if (!stages.empty()) {
    const FoldShape fold = iopp_fold_shape(stages[0], GroupLayout());
    mask = fold_mask_make(fold, pr);
    drafts.push_back(fold_mask_draft(pr.st.codes(), fold, mask));
  }
  const std::vector<OracleId> sent = send_prove(pr, std::move(drafts));
  const OracleId o = sent[0];
  if (!stages.empty()) mask.sent_as(sent[1]);

  std::vector<E> zeta(pr.st.codes().base[notice.shape.code].log_ell());
  {
    AlgebraicDraw from = pr.ch.close_algebraic(R_ZERO);
    for (E &z : zeta) z = from.draw();
  }
  const Summand<E> summand = toy_summand();
  SumClaim<E> claim = toy_claim(zeta);
  std::vector<std::vector<E>> maps(3, std::vector<E>(abc.len));
  for (size_t row = 0; row < 3; row++)
    for (size_t i = 0; i < abc.len; i++) maps[row][i] = abc.at(row, i);
  SumcheckCpu<E> inner(summand, claim.vars, std::move(maps));
  std::vector<MaskSlot> fresh = pr.st.masks(o).take(psc_masks(claim, summand));
  PscOutput<E> out = psc_prove<E>(pr.ch, std::move(claim), summand, inner, std::move(fresh),
                                  [&pr](u32 oracle, u32 index) { return pr.behind[oracle].mask.entry(index); });
  const GroupId g = pr.st.new_group("a, b, c", 0);
  pr.st.join(g, o);
  toy_rows(pr.st, g, o, std::move(out));
  return g;
}

// psc_verify() compares no sum: a false product is left for the proximity test on g to reject.
inline bool toy_verify_preamble(VerifierContext &vf, const ToyPublic &in, GroupId &g, FoldMask &mask) {
  const OracleNotice &notice = vf.ch.get(in.oracle);
  std::vector<OracleId> ids;
  if (!send_verify(vf, toy_first_message(notice, vf.ch.get(in.stages)), ids)) return false;
  const OracleId o = ids[0];
  if (ids.size() > 1) mask.sent_as(ids[1]);
  std::vector<E> zeta(vf.st.codes().base[notice.shape.code].log_ell());
  {
    AlgebraicDraw from = vf.ch.close_algebraic(R_ZERO);
    for (E &z : zeta) z = from.draw();
  }
  const Summand<E> summand = toy_summand();
  SumClaim<E> claim = toy_claim(zeta);
  std::vector<MaskSlot> fresh = vf.st.masks(o).take(psc_masks(claim, summand));
  PscOutput<E> out;
  if (!psc_verify<E>(vf.ch, std::move(claim), summand, std::move(fresh), out)) return false;
  g = vf.st.new_group("a, b, c", 0);
  vf.st.join(g, o);
  toy_rows(vf.st, g, o, std::move(out));
  return true;
}

// ---- the whole run ----

// What is fixed before a run; toy_size() makes it.
struct ToyRun {
  ToyShape shape;
  Planned plan;
  std::vector<IoppStage> stages;
  explicit ToyRun(const Setting &set) { shape.set = set; }
};

// log_len: log2 of the vectors' length. log_n, mask_log_n: log2 of the first stage's two code
// lengths. further: log2 of each later stage's numbers of rows (`k`, `h`) and code lengths; the
// last has k = h = 0. run_words: the public file's first line after "run ". If the parameters do
// not fit or miss the target (set.target bits), run.plan->holds() is false: toy_prove() then ends
// the process, and toy_verify() rejects.
inline ToyRun toy_size(const Setting &set, unsigned log_len, unsigned log_n, unsigned mask_log_n,
                       const std::vector<StageChoice> &further, const char *run_words = "product") {
  ToyRun run(set);
  run.shape.text = toy_text(run_words, set, {{"log-length", log_len}, {"log-n", log_n}, {"mask-log-n", mask_log_n}}, further);
  Plan plan(set);
  GroupSize g;
  g.name = "a, b, c";
  g.k = 2, g.h = 0; // three rows in a table of four; one mask row
  g.main.name = "rows", g.main.log_n = log_n, g.main.ell = u64(1) << log_len;
  const u64 masks = 1 + 2 * u64(log_len) + 2 + 2; // what PSC takes: the hiding polynomial's coefficients and two inner masks
  g.mask.name = "mask row", g.mask.log_n = mask_log_n, g.mask.over_E = true;
  g.mask.ell = pow2_at_least(std::max<u64>(masks + set.samples, 2 * g.k + 1 - g.h));
  // The random tails must hide as many positions as are drawn, and that number depends on the
  // rate, so on the tails: t rises from 0 until it settles.
  u64 t = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, t);
    GroupLoad load;
    load.cube_bits = stage_cube_bits(g);
    load.batched_rows = toy_rows_batched(set);
    g = size_group(set, plan.scope(), g, load);
    if (analysis(set, g).bad()) {  // size_group has found no m
      plan.refuse("no slack holds the first stage's rounds at the target");
      return run.plan = plan.finish(), run;
    }
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= t) break;
    t = more;
  }
  const std::vector<Stage> stages = solve_stages(plan, g, t, further);
  if (plan.refused()) return run.plan = plan.finish(), run;
  run.stages = iopp_stages(stages, run.shape.codes);
  if (const char *fault = iopp_sized_fault(run.shape.codes, run.stages, stages)) return plan.refuse(fault), run.plan = plan.finish(), run; // required before `plan_iopp` (see blocks/iop/iopp.h)
  plan_chain_start(plan, "toy run", true);
  toy_plan_preamble(plan, run.shape, run.stages, {g});
  plan_iopp(plan, g.name, stages, tree_sizes(run.shape.codes, toy_first_message(OracleNotice{run.shape.oracle(), 0}, run.stages)),
            toy_rows_batched(set), 0, true);
  return run.plan = plan.finish(), run;
}

inline void toy_prove(const ToyRun &run, const Table &abc, Engine &engine, Coins &coins, Writer &proof) {
  Binder binder(TOY_LABEL, run.plan);
  const ToyPublic in = toy_bind(binder, run.shape, run.stages);
  ProverRounds ch(binder, random_digest(coins), proof, [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); });
  ProverContext pr(in.codes, ch, engine, coins);
  FoldMask mask;
  const GroupId g = toy_prove_preamble(pr, in, abc, mask);
  iopp_prove(pr, g, in.stages, in.choice, std::move(mask));
  ch.finish();
  pr.st.require_all_tested();
}

// Sizes and plans a run from its public file only. False unless the sized run's text equals `text`.
inline bool toy_from_text(const std::string &text, std::unique_ptr<ToyRun> &run, const char **why) {
  ToyNumbers n;
  Setting set;
  const bool read = toy_read_text(text, n) && toy_setting(n, set) && n.has("log-length") && n.has("log-n") && n.has("mask-log-n") &&
                    n["log-length"] >= 1 && n["log-length"] <= 20 && n["log-n"] <= 24 && n["mask-log-n"] <= 24;
  if (read) run = std::make_unique<ToyRun>(toy_size(set, (unsigned)n["log-length"], (unsigned)n["log-n"], (unsigned)n["mask-log-n"], n.further, n.run_words.c_str()));
  if (!read || run->shape.text != text) {
    if (why) *why = "the public file is not a file of this run in its one spelling";
    return false;
  }
  return true;
}

inline ToyPublic toy_bind_run(Binder &binder, const ToyRun &run) { return toy_bind(binder, run.shape, run.stages); }
// The verifier's side, for run_verifier() (blocks/chain/closed.h): it gets the channel and handles
// to the first query's items, not the run. notes: if not null, receives intermediate values
// (VerifierContext::note()); no decision reads them.
inline bool toy_decide(VerifierRounds &ch, const ToyPublic &in, Notes *notes) {
  VerifierContext vf(in.codes, ch);
  vf.notes = notes;
  GroupId g = 0;
  FoldMask mask;
  if (!toy_verify_preamble(vf, in, g, mask) || !iopp_verify(vf, g, in.stages, in.choice, std::move(mask))) return false;
  vf.st.require_all_tested();
  return true;
}
// The second half of toy_verify_file(); tests call it with parameters other than the prover's.
inline bool toy_verify(const ToyRun &run, const u8 *proof, size_t bytes, const char **why = nullptr, Notes *notes = nullptr) {
  return run_verifier(TOY_LABEL, run.plan, run, toy_bind_run, toy_decide, proof, bytes, why, notes);
}
// The complete verifier: the public file's text, the proof, nothing else.
inline bool toy_verify_file(const std::string &text, const u8 *proof, size_t bytes, const char **why = nullptr, Notes *notes = nullptr) {
  std::unique_ptr<ToyRun> run;
  return toy_from_text(text, run, why) && toy_verify(*run, proof, bytes, why, notes);
}

} // namespace blk
