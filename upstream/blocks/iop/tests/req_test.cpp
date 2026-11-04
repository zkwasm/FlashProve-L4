// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the frame of a request's proof, the part outside the sub-proofs (blocks/iop/req.h): prover
// against verifier on the small synthetic requests of blocks/iop/tests/req_cases.h, with the real
// hash chain, Merkle trees and CPU engine. Random arrays and random linear claims stand in for the
// sub-proofs; the prover sends the claims' values, so that the hash chain binds them before they
// are batched. Only the shapes are small: distances, numbers of positions, random tails, the last
// test's stages, round errors and proof of work come from blocks/numbers/ at each case's target.
//
// Words. A segment (`link` in identifiers and messages) is one sub-proof's part of the proof; its
// oracle messages (encoded rows, sent as Merkle roots) go out at commit points (`moment`) and join
// its group, whose rows are opened at the same positions. A claim (`Row`, "row") has a public value
// (its "target"; the other target is a plan's, in bits). Closing a segment reduces its claims to
// one, which the code switch (CS) carries into the next segment in a new oracle message, the
// carrying one. The last test is the proximity test (`iopp` in identifiers), in stages, on the last
// carrying oracle message. A "rider" is an `at_start` oracle message: mask rows that go in the
// request's first prover message and join no segment's group. A plan (`Plan`,
// blocks/numbers/plan.h) lists every round with its error bound; both sides follow it as they run.
// Messages and identifiers say "refused" for rejected, "priced" for computed (of an error bound)
// and `bent` for altered.
//
// Each block of main() says what it tests. Other malicious provers are not part of this repository.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // plans are built and altered by hand (blocks/numbers/plan.h)
#include <sys/resource.h>

#include <cstring>

#include "chain/tests/death.h"
#include "iop/tests/req_cases.h"

using namespace blk;

// Checks that `ReqSizes` has the group `at_start` and no member `at_start_list`. Sizes store
// groups: two codes, numbers of rows, a rule and its integer `m`. A group's distance and list size
// (a bound on the number of codewords within the distance) are computed from these and not stored:
// a stored result could stay an honest code's while the code's numbers beside it change. With sizes
// that stored both, the verifier accepted a false value in 10 of 16 runs (of a malicious prover, in
// a test that is not part of this repository). The planning functions compute both where they need
// them: analysis() in blocks/numbers/blocks.h. `HasAtStart` is the control: the detection finds a
// member that exists. For `GroupSize` the same is tested by blocks/numbers/tests/numbers_test.cpp.
template <class T, class = void> struct HasAtStartList : std::false_type {};
template <class T> struct HasAtStartList<T, std::void_t<decltype(std::declval<T &>().at_start_list)>> : std::true_type {};
template <class T, class = void> struct HasAtStart : std::false_type {};
template <class T> struct HasAtStart<T, std::void_t<decltype(std::declval<T &>().at_start)>> : std::true_type {};
static_assert(HasAtStart<ReqSizes>::value && !HasAtStartList<ReqSizes>::value, "ReqSizes holds a bare list for what rides at the start, or the group is gone");

// Further malicious provers run on the same requests in a test that is not part of this repository,
// at toy size and on synthetic data. Bad results seen, each with the check that answers it:
//   * Public inputs that contradict each other. The plan is made first; then the middle stage of
//     the last test gets a mask code of rate 1, which both sides bind in the first query and run.
//     Without the two checks described at STOP_AT_A_STAGES_HEAD the verifier accepted a false value
//     in 11 of 16 such runs; with them, both sides abort (main() tests this for other bent codes).
//   * Sizes that stored a group's distance and list size: 10 of 16 runs, see `HasAtStartList`.
//   * Two further ways to a false accept cannot be written, so nothing runs them. In both, the
//     library's own sizing functions made every size, and every number of a code was true. (1) The
//     segments' mask code, of rate 1, was marked to be read as a repetition, in a member of
//     `CodeSize` (`repeated_k`) that replaced the dimension in every formula and that no run-time
//     check compared. (2) The segments' group was sized at a distance given as an interval whose
//     ends had crossed, so that one formula read "near 1" and the error bound "tiny". In a form of
//     the library that allowed these, the verifier accepted a false value in 8 and 13 of 16 runs
//     for (1) and in 5 and 9 of 128 runs for (2) (each pair: over the Goldilocks field, then over
//     the prime field of 2^61 - 1 elements, for which this library cannot be built). `CodeSize` has
//     no such member, a distance is passed as an integer number of positions, and `Bound`'s
//     constructor marks a crossed pair bad: blocks/numbers/tests/numbers_test.cpp checks that by
//     static_asserts and a test.

// Runs body in a child process, as ends_with() does (blocks/chain/tests/death.h). Stores the
// child's standard error in `said`; returns true if SIGABRT ended the child.
template <class Body> static bool ended_saying(std::string &said, Body body) {
  std::fflush(stdout);
  int fd[2];
  if (pipe(fd) != 0) return false;
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    close(fd[0]);
    dup2(fd[1], 2);
    body();
    _exit(0);
  }
  close(fd[1]);
  said.clear();
  char buffer[256];
  for (ssize_t n; (n = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(fd[0]);
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

// The first words of the messages of two run-time checks of the channel (`ProverRounds`,
// `VerifierRounds` in blocks/chain/rounds.h: each side sends, receives and gets challenges through
// it). A plan records, beside a round, the group that the round's error was computed for
// (`PlanRound::group`). require_group() compares the record with the group that runs, where a stage
// begins and in its fold's rounds; require_codes() compares it with the two codes of the rows on
// which positions are drawn. Each aborts on a difference. The records are not among the plan's
// bytes in the first query (the hash chain's first input, which fixes the public inputs): a
// malicious prover can run under a copy of the plan with other records (`lying` below) and still
// share the verifier's hash chain, so the verifier must make these checks itself.
static const char *const STOP_AT_A_STAGES_HEAD = "rounds: the round begins a stage on another group than its planned errors are priced for";
static const char *const STOP_AT_POSITIONS = "rounds: the round draws positions on rows of other codes than its planned error is priced for";
// True if body aborts with either message. Both checks cover a stage's codes, and a run aborts at
// the one that it reaches first.
template <class Body> static bool ends_at_a_stop(Body body) {
  std::string said;
  return ended_saying(said, body) && (said.find(STOP_AT_A_STAGES_HEAD) != std::string::npos || said.find(STOP_AT_POSITIONS) != std::string::npos);
}

static ReqProved prove(const ReqWorld &w, u64 seed, unsigned false_at) {
  EngineCpu engine;
  return req_toy_prove(w, seed, false_at, engine);
}

int main() {
  const std::vector<ReqCase> cases = req_toy_cases();
  const unsigned nowhere = ~0u;
  size_t runs = 0, refused = 0;
  for (ReqCase t : cases) {
    const ReqWorld w = req_toy_world(t);
    const char *fault = req_fault(w.codes, w.shape);
    if (fault) std::printf("%s: %s\n", t.name, fault);
    CHECK(!fault, "the configuration is refused");
    if (fault) continue;
    CHECK(w.plan->holds(), "a round of the plan does not hold at the target");
    const u64 seed = 300;
    const ReqProved p = prove(w, seed, nowhere);
    CHECK(req_toy_verify(w, seed, p.proof), "an honest proof is refused");
    if (const char *pads = toy_pads_fault([&] { (void)prove(w, seed, nowhere); })) {  // the pad (it masks a sample's answer) of every sampled message: the segments', carrying and kept ones, the rider
      std::printf("%s: %s\n", t.name, pads);
      CHECK(false, "a pad is not drawn, or is drawn once for two places or two runs");
    }
    for (unsigned l = 0; l < t.links; l++) {
      const ReqProved f = prove(w, seed, l);
      CHECK(!req_toy_verify(w, seed, f.proof), "a false target is accepted");
    }
    // A proof made under another plan: a lower target, more proof of work, or longer codes.
    ReqCase fewer = t, other_work = t, other_rate = t;
    fewer.target = t.target - 8, other_work.work = t.work + 4, other_rate.log_n = t.log_n + 1;
    for (const ReqCase &o : {fewer, other_work, other_rate}) {
      const ReqWorld other = req_toy_world(o);
      const ReqProved q = prove(other, seed, nowhere);
      CHECK(req_toy_verify(other, seed, q.proof), "an honest proof under the other plan is refused");
      runs++;
      if (!req_toy_verify(w, seed, q.proof)) refused++;
      else std::printf("ACCEPTED a proof made under another plan: %s\n", t.name);
    }
    // The proof's length against the plan's upper bound and its expected length.
    {
      const PartSize z = w.plan->size_of(0);
      if (p.proof.size() > z.bytes || p.proof.size() < 0.97 * z.expected || p.proof.size() > 1.03 * z.expected)
        std::printf("%s: %zu bytes; the plan: expected %.0f, at most %llu\n", t.name, p.proof.size(), z.expected, (unsigned long long)z.bytes);
      CHECK(p.proof.size() <= z.bytes, "a proof is longer than the plan's bound");
      CHECK(p.proof.size() >= 0.97 * z.expected && p.proof.size() <= 1.03 * z.expected, "a proof is more than 3 % off the plan's expected length");
    }
    // The reply: a public value that the prover announces only in the last segment, as a plain
    // scalar in a round of its own before that segment's first prover message. The coefficients of
    // the last segment's claims depend on it. req_toy_verify() compares it with the reply that the
    // client was given, unless told not to.
    {
      EngineCpu engine;
      const ReqProved other = req_toy_prove(w, seed, nowhere, engine, REQ_TOY_REPLY + 1);  // an honest proof of another reply
      CHECK(!req_toy_verify(w, seed, other.proof), "a proof of another reply than the client's is accepted");
      CHECK(req_toy_verify(w, seed, other.proof, REQ_TOY_REPLY, false), "control: without the comparison that proof must pass, or the comparison was not what refused it");
      EngineCpu again;
      CHECK(!req_toy_verify(w, seed, req_toy_prove(w, seed, nowhere, again, REQ_TOY_REPLIES).proof, REQ_TOY_REPLIES), "a reply that has not the form of one is accepted");
      // With the scalar changed in the proof's bytes all later challenges differ, so even the
      // verifier that does not compare rejects.
      u8 scalar[E_BYTES];
      E_to_bytes(E::from(from_u64(REQ_TOY_REPLY)), scalar);
      size_t at = p.proof.size(), found = 0;
      for (size_t i = 0; i + E_BYTES <= p.proof.size(); i++)
        if (std::equal(scalar, scalar + E_BYTES, p.proof.begin() + i)) at = i, found++;
      CHECK(found == 1, "the reply's scalar does not stand once in the proof");
      if (found == 1) {
        std::vector<u8> changed = p.proof;
        changed[at] = (u8)(REQ_TOY_REPLY + 1);
        CHECK(!req_toy_verify(w, seed, changed, REQ_TOY_REPLY, false), "the reply is changed in the proof, and nothing else notices: the chain does not hold it");
      }
      // One message late: after the answers of the last segment's first commit point. That is
      // another plan, and so another first query.
      ReqCase late = t;
      late.reply_late = true;
      const ReqWorld lw = req_toy_world(late);
      const ReqProved lp = prove(lw, seed, nowhere);
      CHECK(req_toy_verify(lw, seed, lp.proof), "control: an honest proof in the world of the late reply is refused");
      CHECK(!req_toy_verify(w, seed, lp.proof), "a reply that comes one message late is accepted");
    }
    std::printf("%s: %zu bytes, %zu trees\n", t.name, p.proof.size(), p.trees);
  }
  {  // No segment runs twice within a request. The rider holds masks that a segment's sub-proof
     // uses once; run again with the same rider, the segment would send a second value under the
     // same mask. `ReqProver` has no way back: each such call aborts. (A `ReqCase` lists: name,
     // segments, `life` of the kept rows, rider, target in bits, bits of proof of work, `log_n`.)
    const ReqWorld w = req_toy_world({"two links, a rider", 2, 0, true, 24, 0, 11});
    auto run = [&w](int again) {
      EngineCpu engine;
      TestCoins coins(5);
      Writer proof;
      Binder binder(1, w.plan);
      const ReqPublic bound = req_toy_bind(binder, w, 5);
      ProverRounds ch(binder, toy_random_bytes(7), proof);
      ProverContext P(bound.codes, ch, engine, coins);
      ToySubProver sub(w, ch.get(bound.seed), engine, coins, ~0u, REQ_TOY_REPLY);
      ReqProver frame(P, bound.shape);
      sub.run(P, frame, 0, frame.first(sub.tables(0, 0), sub.at_start(), sub.mask_row(0), sub.plain(0)));
      if (again == 1) frame.first(sub.tables(0, 0), sub.at_start(), sub.mask_row(0), sub.plain(0));
      for (unsigned moment = 1; moment < w.shape.link[0].moments(); moment++) sub.run(P, frame, moment, frame.later(sub.tables(0, moment)));
      if (again == 2) frame.later(sub.tables(0, 1));
      frame.close();
      if (again == 3) frame.close();
      if (again == 4) frame.first(sub.tables(0, 0), sub.at_start(), sub.mask_row(0), sub.plain(0));  // starts the request again, with a new rider
    };
    CHECK(runs_through([&] { run(0); }), "the first link of an honest run does not run through");
    CHECK(ends_with("a link's first prover message is not due", [&] { run(1); }), "a link's first prover message goes out twice");
    CHECK(ends_with("the link has no further moment", [&] { run(2); }), "a moment of a link goes out twice");
    CHECK(ends_with("moments of the link are missing", [&] { run(3); }), "a link is closed twice");
    CHECK(ends_with("what rides at the start is fixed", [&] { run(4); }), "a request starts again inside its own frame");
  }
  {  // A sample's error is computed with the list size of the group that its oracle message belongs
     // to. The error of the first prover message's samples is computed here a second time and
     // compared.
    const ReqWorld w = req_toy_world({"four links, kept for two, a rider", 4, 2, true, 24, 0, 11});
    std::vector<SampleSize> expected;
    for (const ReqMessage &m : w.shape.link[0].messages)
      if (m.moment == 0) expected.push_back(SampleSize{Send::cube(w.codes, m.shape).bits, {w.sizes.group}});
    for (const OracleNotice &n : w.shape.at_start) expected.push_back(SampleSize{Send::cube(w.codes, n.shape).bits, {w.sizes.at_start}});
    const Big links_list = analysis(w.plan->set, w.sizes.group).list;
    CHECK(w.shape.at_start.size() == 1 && analysis(w.plan->set, w.sizes.at_start).list != links_list, "the case has no rider, or the rider's list is the links': the comparison below would show nothing");
    Plan second(w.plan->set);
    plan_send(second, "written a second time", expected, 1);
    auto first_seed = [](const Planned &p) {
      for (const PlanRound &r : p->rounds_)
        if (r.name == R_SEND_SEED) return r.error;
      return Real();
    };
    CHECK(Real::cmp(first_seed(w.plan), first_seed(second)) == 0, "the samples of a request's first prover message are not priced each with its own group's list");
    // The last carrying oracle message belongs to the last test's first stage. In this case that
    // stage has the segments' list size, which would show nothing. So the plan is made once more
    // from sizes in which that stage has another `m`, and with it another list size.
    ReqSizes other = w.sizes;
    other.last.front().g = group_of_m(other.last.front().g, 3);
    CHECK(analysis(w.plan->set, other.last.front().g).list != links_list, "the first stage's list is the links' still: the comparison below would show nothing");
    Plan again(w.plan->set), third(w.plan->set);
    plan_chain_start(again, "request", true);
    req_plan(again, w.codes, w.shape, other, [&](Plan &p, size_t l, unsigned moment) {
      if (moment + 1 == w.shape.link[l].moments()) plan_scalars(p, "targets", w.shape.link[l].name, REQ_TOY_ROWS);  // as `req_toy_world` plans the synthetic sub-proofs
    });
    plan_send(third, "written a second time", {SampleSize{Send::cube(w.codes, req_groups(w.shape).back().target.shape).bits, {other.last.front().g}}}, 1);
    Real last_seed;
    for (const PlanRound &r : again.rounds_)
      if (r.name == R_SEND_SEED && r.where == "last group") last_seed = r.error;
    CHECK(Real::cmp(last_seed, first_seed(third)) == 0, "the sample of the last carrying oracle message is not priced with the list of the last test's first stage");
  }
  {  // req_plan() computes the last test's errors from the solved stages (`ReqSizes::last`); a run
     // folds and opens the shape's stages (`ReqShape::last`) under `Codes`. req_plan() aborts
     // unless the two agree (iopp_sized_fault()).
    const ReqCase t = {"four links, kept for two, a rider", 4, 2, true, 24, 0, 11};
    const ReqWorld w = req_toy_world(t);
    auto plan_with = [&](const Codes &codes, const ReqShape &shape, const ReqSizes &sizes) {
      Plan p(w.plan->set);
      plan_chain_start(p, "request", true);
      req_plan(p, codes, shape, sizes, [&](Plan &q, size_t l, unsigned moment) {
        if (moment + 1 == shape.link[l].moments()) plan_scalars(q, "targets", shape.link[l].name, REQ_TOY_ROWS);  // as `req_toy_world` plans the synthetic sub-proofs
      });
      return p.finish()->holds();
    };
    CHECK(!iopp_sized_fault(w.codes, w.shape.last, w.sizes.last) && plan_with(w.codes, w.shape, w.sizes), "control: the world's own sizes, stages and codes are not taken, or their plan does not hold");
    struct Bent {
      const char *what, *says;
      std::function<void(Codes &, ReqShape &, ReqSizes &)> bend;
    };
    const std::vector<Bent> bent = {
        {"a stage's main code has more random entries than were priced", "sized for another main code", [](Codes &c, ReqShape &s, ReqSizes &) { c.base[s.last[1].code].r += 16; }},
        {"a stage's mask code has more random entries than were priced", "sized for another mask code", [](Codes &c, ReqShape &s, ReqSizes &) { c.mask[s.last[1].mask_code].r += 1; }},
        {"the sizes price a stage's main code with fewer random entries than it has", "sized for another main code", [](Codes &, ReqShape &, ReqSizes &z) { z.last[1].g.main.r -= 1; }},
        {"the sizes price a stage with a shorter message", "sized for another main code", [](Codes &, ReqShape &, ReqSizes &z) { z.last[1].g.main.ell /= 2; }},
        {"the sizes price a stage with fewer rows", "sized for another number of rows", [](Codes &, ReqShape &, ReqSizes &z) { z.last[1].g.k -= 1; }},
        {"the sizes have a stage fewer", "sized for another number of stages", [](Codes &, ReqShape &, ReqSizes &z) { z.last.pop_back(); }},
    };
    for (const Bent &b : bent) {
      Codes codes = w.codes;
      ReqShape shape = w.shape;
      ReqSizes sizes = w.sizes;
      b.bend(codes, shape, sizes);
      const char *fault = iopp_sized_fault(codes, shape.last, sizes.last);
      if (!fault || !std::strstr(fault, b.says)) std::printf("%s: %s\n", b.what, fault ? fault : "passes");
      CHECK(fault && std::strstr(fault, b.says), "sizes that are not the stages under the codes pass, or are refused for something else");
      CHECK(ends_with(b.says, [&] { (void)plan_with(codes, shape, sizes); }), "req_plan makes a plan from sizes that are not the shape's stages under the list's codes");
    }
  }
  {  // Altered after planning: the plan is made, then one number of the codes or stages in the
     // first query is changed. No comparison at planning time can see that. Either req_fault()
     // rejects the configuration, or the honest prover aborts where the first stage begins that
     // runs on another group than its errors were computed for (require_group()), and so does the
     // verifier.
    const ReqWorld w = req_toy_world(cases[0]);
    struct Bend {
      const char *what;
      std::function<void(ReqWorld &)> bend;
    };
    std::vector<Bend> bends;
    for (size_t stage : {size_t(0), size_t(1)})
      for (size_t more : {size_t(1), size_t(4), size_t(16)}) {
        bends.push_back({"a stage's main code has more random entries", [=](ReqWorld &x) { x.codes.base[x.shape.last[stage].code].r += more; }});
        bends.push_back({"a stage's mask code has more random entries", [=](ReqWorld &x) { x.codes.mask[x.shape.last[stage].mask_code].r += more; }});
      }
    bends.push_back({"a later stage's main code has a shorter message at the same length", [](ReqWorld &x) { x.codes.base[x.shape.last[1].code].ell /= 2; }});
    bends.push_back({"a later stage's main code has a longer message at the same length", [](ReqWorld &x) { x.codes.base[x.shape.last[1].code].ell *= 2; }});
    bends.push_back({"a later stage's mask code is shorter", [](ReqWorld &x) { x.codes.mask[x.shape.last[1].mask_code].log_n -= 1; }});
    bends.push_back({"a later stage has a row variable more", [](ReqWorld &x) { x.shape.last[1].k += 1; }});
    bends.push_back({"a later stage has a row variable less", [](ReqWorld &x) { x.shape.last[1].k -= 1; }});
    bends.push_back({"a later stage has a mask rows' variable more", [](ReqWorld &x) { x.shape.last[1].h += 1; }});
    bends.push_back({"a later stage has a mask rows' variable less", [](ReqWorld &x) { x.shape.last[1].h -= 1; }});
    size_t by_form = 0, stopped = 0, verifier_stopped = 0, no_proof = 0;
    for (const Bend &b : bends) {
      ReqWorld x = w;
      b.bend(x);
      if (const char *fault = req_fault(x.codes, x.shape)) {
        std::printf("  refused by its form: %s: %s\n", b.what, fault);
        by_form++;
        continue;
      }
      const bool ended = ends_with(STOP_AT_A_STAGES_HEAD, [&] { (void)prove(x, 300, nowhere); });
      if (!ended) std::printf("%s: the prover's process does not end at the stage's head\n", b.what);
      CHECK(ended, "a request runs under codes or stages that were bent behind the planning");
      stopped += ended;
      // The verifier's side. It reaches the altered stage only on a proof that is valid up to
      // there: one made under a copy of the plan that records the altered numbers (see
      // STOP_AT_A_STAGES_HEAD). Where the change also alters the rounds themselves (a row variable
      // more or less), no such proof can be made.
      ReqWorld lying = x;
      auto numbers = [](const Code &c) { return PricedCode{c.log_n, (u64)c.ell, (u64)c.r}; };
      const IoppStage &was = w.shape.last[1], &is = x.shape.last[1];
      Plan lies(x.plan);
      for (PlanRound &r : lies.rounds_) {
        if (!r.group.said) continue;
        if (r.group == group_as_run(w.codes, was.code, was.mask_code, true, was.k, was.h, 1)) r.group.k = is.k, r.group.h = is.h;
        for (size_t c = 0; c < w.codes.base.size(); c++)
          if (r.group.main == numbers(w.codes.base[c])) {
            r.group.main = numbers(x.codes.base[c]);
            break;
          }
        for (size_t c = 0; c < w.codes.mask.size(); c++)
          if (r.group.mask == numbers(w.codes.mask[c])) {
            r.group.mask = numbers(x.codes.mask[c]);
            break;
          }
      }
      lying.plan = lies;
      std::string said;
      const bool aborted = ended_saying(said, [&] {
        const ReqProved p = prove(lying, 300, nowhere);
        std::fprintf(stderr, "the prover under the other plan has proved\n");
        std::fprintf(stderr, req_toy_verify(x, 300, p.proof) ? "THE VERIFIER HAS ACCEPTED\n" : "the verifier has refused\n");
      });
      if (said.find("the prover under the other plan has proved") == std::string::npos) {
        CHECK(aborted, "a prover under the other plan neither proves nor ends");
        no_proof++;
        continue;
      }
      const bool verifier_ended = aborted && said.find(STOP_AT_A_STAGES_HEAD) != std::string::npos;
      if (!verifier_ended) std::printf("%s: the verifier's process does not end at the stage's head: %s\n", b.what, said.c_str());
      CHECK(verifier_ended, "a verifier runs under codes or stages that were bent behind the planning");
      verifier_stopped += verifier_ended;
    }
    std::printf("bent behind the planning: %zu bends: %zu refused by their form; the prover stopped where the stage begins at %zu, the verifier at %zu; no proof can be made for it at %zu\n", bends.size(), by_form, stopped, verifier_stopped, no_proof);
    CHECK(by_form + stopped == bends.size() && verifier_stopped + no_proof == stopped && verifier_stopped >= 10, "a bend is not accounted for, or too few reach the run");
  }
  {  // Only the plan's record at a round that draws positions is altered; codes, shape and the rest
     // of the plan stay. Each side of the code switch calls require_codes() for itself
     // (blocks/iop/cs.h): the prover aborts under such a plan, and so does the verifier, on an
     // honest proof made under the true plan (the first query is the same: see
     // STOP_AT_A_STAGES_HEAD). Tried at a segment's positions (`R_LOC`) and at a stage's (`R_CS`),
     // for each of the six numbers of the two codes.
    const ReqWorld w = req_toy_world(cases[0]);
    const ReqProved honest = prove(w, 300, nowhere);
    size_t provers = 0, verifiers = 0, bends = 0;
    for (const char *name : {R_LOC, R_CS})
      for (int bend = 0; bend < 6; bend++) {
        ReqWorld x = w;
        PlanRound *round = nullptr;
        Plan bent(w.plan);
        for (PlanRound &r : bent.rounds_)
          if (!round && r.name == name) round = &r;
        CHECK(round && round->group.said, "a round that draws positions says no codes");
        if (!round) continue;
        PricedCode &code = bend < 3 ? round->group.main : round->group.mask;
        (bend % 3 == 0 ? code.log_n += 1 : bend % 3 == 1 ? code.ell += 1 : code.r += 1);
        x.plan = bent;
        bends++;
        provers += ends_with(STOP_AT_POSITIONS, [&] { (void)prove(x, 300, nowhere); });
        verifiers += ends_with(STOP_AT_POSITIONS, [&] { (void)req_toy_verify(x, 300, honest.proof); });
      }
    std::printf("what the plan says where positions are drawn, bent by itself: %zu bends: the prover's process ends at %zu, the verifier's at %zu\n", bends, provers, verifiers);
    CHECK(bends == 12 && provers == bends && verifiers == bends, "a side of the switch draws positions under a plan that says other codes there");
  }
  {  // Configurations that the frame cannot run: req_fault() reports each, with its segment, and
     // does not abort.
    const ReqWorld w = req_toy_world({"four links, kept for two, a rider", 4, 2, true, 24, 0, 11});
    size_t at = 99;
    CHECK(!req_fault(w.codes, w.shape, &at), "a lawful configuration is found faulty");
    struct Wrong {
      const char *what;
      size_t link;  // 4: none
      std::function<void(ReqShape &)> spoil;
      const char *says = nullptr;  // the fault's first words, where the case aims at one check
    };
    const std::vector<Wrong> wrong = {
        {"rows over a field of degree 2", 1, [](ReqShape &s) { s.link[1].messages[0].shape.main_deg = 2; }},
        {"an oracle message without rows", 2, [](ReqShape &s) { s.link[2].messages.back().shape.main_rows = 0; }},
        {"two answers in one message", 3, [](ReqShape &s) { s.link[3].messages[0].shape.answers = 2; }},
        {"more answers than a row is long", 4, [](ReqShape &s) {  // everywhere alike, so that the shape does not contradict itself: only the codes show it
           s.answers = 1u << 30;
           for (ReqLink &l : s.link)
             for (ReqMessage &m : l.messages) m.shape.answers = s.answers;
           for (OracleNotice &n : s.at_start) n.shape.answers = s.answers;
         }},
        {"the shape's answers differ from its messages'", 0, [](ReqShape &s) { s.answers = 2; }},
        {"no answer to any sample", 4, [](ReqShape &s) {  // everywhere alike: the shape does not contradict itself, and no message would be sampled
           s.answers = 0;
           for (ReqLink &l : s.link)
             for (ReqMessage &m : l.messages) m.shape.answers = 0;
           for (OracleNotice &n : s.at_start) n.shape.answers = 0;
         }, "no answer to a sample"},
        {"one message of a link without an answer", 2, [](ReqShape &s) { s.link[2].messages[0].shape.answers = 0; }, "an oracle message of the link has another number of answers"},
        {"the rider in the links' leaf class", 4, [](ReqShape &s) { s.at_start[0].leaf_class = 0; }},
        {"the rider under a mask code that is not there", 4, [](ReqShape &s) { s.at_start[0].shape.mask_code = 77; }},
        {"the rider with main rows", 4, [](ReqShape &s) { s.at_start[0].shape.main_rows = 1; }},
        {"2^63 + 1 rows", 0, [](ReqShape &s) { s.link[0].messages[0].shape.main_rows = (size_t(1) << 63) + 1; }},
        {"kept beyond the last link", 2, [](ReqShape &s) { s.link[2].messages[0].life = 2, s.link[2].messages[0].shape.mask_rows = 0; }},
        {"a moment left out", 1, [](ReqShape &s) { s.link[1].messages.back().moment = 5; }},
        {"a kept message with a mask row", 1, [](ReqShape &s) {
           for (ReqMessage &m : s.link[1].messages)
             if (m.life > 0) m.shape.mask_rows = 1;
         }, "a kept oracle message has no main rows, or has mask rows"},
        {"a kept message of mask rows alone", 1, [](ReqShape &s) {
           for (ReqMessage &m : s.link[1].messages)
             if (m.life > 0) m.shape.main_rows = 0, m.shape.mask_rows = 1;
         }, "a kept oracle message has no main rows, or has mask rows"},
        {"a message of a later moment is kept", 1, [](ReqShape &s) { s.link[1].messages.back().life = 1; }, "a kept oracle message does not go in its link's first prover message"},
        {"a first link of mask rows and kept rows alone", 0, [](ReqShape &s) {  // no other check rejects it; the first closing would have no main row to fold
           std::vector<ReqMessage> left;
           for (ReqMessage &m : s.link[0].messages)
             if (m.moment == 0) left.push_back(m);
           for (ReqMessage &m : left)
             if (m.life == 0) m.shape.main_rows = 0;
           s.link[0].messages = left;
         }},
    };
    for (const Wrong &x : wrong) {
      ReqShape s = w.shape;
      x.spoil(s);
      const char *fault = req_fault(w.codes, s, &at);
      if (!fault || at != x.link) std::printf("%s: %s, link %zu\n", x.what, fault ? fault : "passes", at);
      CHECK(fault && at == x.link, "a wrong configuration passes, or the fault is laid on another link");
      if (fault && x.says && std::string(fault).compare(0, std::strlen(x.says), x.says)) {
        std::printf("%s: %s\n", x.what, fault);
        CHECK(false, "a wrong configuration is refused by another test than the one that the case is there for");
      }
    }
  }
  // At a target of 10 bits: an honest proof with one bit flipped, in every fifth byte in turn.
  {
    const ReqCase t = {"two links, low target", 2, 1, false, 10, 0, 9};
    const ReqWorld w = req_toy_world(t);
    const ReqProved p = prove(w, 400, nowhere);
    CHECK(req_toy_verify(w, 400, p.proof), "an honest proof is refused");
    for (size_t i = 0; i < p.proof.size(); i += 5) {
      std::vector<u8> changed = p.proof;
      changed[i] ^= (u8)(1u << (i % 8));
      runs++;
      if (!req_toy_verify(w, 400, changed)) refused++;
      else std::printf("ACCEPTED a proof with byte %zu changed\n", i);
    }
  }
  CHECK(refused == runs, "something that had to be refused was accepted");
  std::printf("req_test: %s (%zu of %zu refused)\n", failures ? "FAILED" : "ok", refused, runs);
  return failures ? 1 : 0;
}
