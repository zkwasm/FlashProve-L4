// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the exact test of hiding (blocks/tests/hiding/exact.h) on RNG alone, the approximate range
// check on committed cells: with the challenges fixed, do two witnesses give the same law
// (distribution) of the view over the prover's coins, the field elements that it draws?
// The fixture `rng_world` (blocks/iop/tests/rng_world.h) proves its rng_small_case(), six cells
// within -1 .. 1, on the CPU engine (`EngineCpu`). The witness is the cells; sizes, bound, layouts
// and the reads' weights are public and the same for all witnesses.
// The prover sends four oracle messages by SEND, each with an out-of-domain sample: packed cells,
// reads (weighted sums of cells), digits, multiplicities. A short run stops after RNG's batching
// element: nothing is opened, and the view is the samples' answers and RNG's scalars.
// A pass covers only the pairs compared, at one list of challenges. It says nothing about
// `cp_prove`, which proves on the GPU engine (`EngineGpu`) and does not call rng_prove(), RNG's
// prover here. Short runs do not cover the openings of a whole run's proximity test (`iopp` in
// identifiers), and so not whether the rows' random tails are long enough for them: only "long"
// adds whole_run(), and `make test` (blocks/Makefile) passes no argument.
// A pass in the shape in which every message has a mask row does not show that the pads are drawn
// (README.md, "What these tests do not show").
// Arguments: "-v" prints a line for each comparison and draw. "long" compares five pairs in each
// short run, not two, and adds a whole run in both of main()'s shapes (two pairs; five minutes).

// Adds the switch `prover_messages_outside` (blocks/chain/rounds.h), true at first. While it is
// true, the prover's messages stay out of its hash chain, so the challenges do not depend on what
// it sends. Only the prover's channel (`ProverRounds`) reads the switch: no verifier is changed,
// and none accepts such a proof.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `whole_run` writes a plan by hand (blocks/numbers/plan.h)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function" // rng_world.h has functions that are unused here
#include "iop/tests/rng_world.h"
#pragma GCC diagnostic pop
#include "iop/tests/iopp_support.h"
#include "tests/hiding/sessions.h"

using namespace blk;

// The rounds of rng_world::block_run(), then a proximity test of two stages, with `draws` positions
// a stage and random tails just long enough for that many. Codes and plan are written by hand,
// because the fixture's own sizing, rng_world::size_run_at(), rejects so small a case.
// toy_plan_iopp() writes 0 in place of its rounds' error bounds, so the plan says nothing about
// soundness.
static rng_world::RangeRun whole_run(unsigned draws) {
  using namespace rng_world;
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = the_case.target;
  RangeRun run(set);
  Plan plan(set);
  run.combinations = rng_combinations(plan);
  const RngShape R = rng_shape(set, run.combinations);
  const LkpPublic L = rng_lookup(R);
  BLK_REQUIRE(lkp_masks(L) + set.samples <= u64(1) << the_case.log_mask, "the mask row is too short for the look-up");
  const size_t tail = tail_for_positions(draws);
  run.codes.mask.push_back(toy_code(size_t(1) << the_case.log_mask, tail, 0));
  run.codes.base.push_back(toy_code(size_t(1) << the_case.log_len, tail, run.codes.mask.back().log_n));
  run.stages = {IoppStage{0, 0, 3, the_case.reference ? 1u : 2u}};
  toy_later_stages(run.codes, run.stages, {Later{0, 0}}, draws, set.samples);
  plan_chain_start(plan, "range check, whole, by hand", true);
  std::vector<std::vector<OracleNotice>> sent = prover_messages(set);
  sent.push_back({R.digits, R.multiplicities});
  for (size_t i = 0; i < sent.size(); i++) {
    if (i == 2) plan_rng(plan, "range check", R.all_columns());
    std::vector<SampleSize> samples;
    for (const OracleNotice &n : sent[i]) samples.push_back(SampleSize{Send::cube(run.codes, n.shape).bits, no_sized_group()});
    plan_send(plan, sent[i][0].shape.name, samples, (unsigned)tree_sizes(run.codes, sent[i]).size());
  }
  plan_lookup(plan, "digits", LookupSize{L.parts[0], L.d, L.mu, true, L.live});
  plan_bat_sigma(plan, "range check", R.equations());
  toy_plan_iopp(plan, run.codes, run.stages, draws, IoppChoice(), false, rows_left(R) + 4 * set.samples);
  return run.plan = plan.finish(), run;
}

int main(int argc, char **argv) {
  bool verbose = false, long_run = false;
  for (int i = 1; i < argc; i++) verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
  int failures = 0;
  unsigned compared = 0;
  std::string summary;
  // In one shape, every message has a mask row. In the reference shape the digits' and the
  // multiplicities' messages have none, as in the segments of a request's proof, so the pads that
  // mask the answers to their samples are in main rows, in tables that RNG builds itself. (The
  // reads keep their mask row in both shapes; in a segment only the first message has one.)
  for (unsigned shape = 0; shape < (long_run ? 4u : 2u); shape++) {
    const bool reference = shape % 2, whole = shape >= 2;
    rng_world::the_case = rng_world::rng_small_case(reference);
    rng_world::the_case.whole = whole;
    const rng_world::RangeRun run = whole ? whole_run(2) : rng_world::size_run();
    const std::string name = std::string(rng_world::the_case.name) + (whole ? ", a whole run" : "");
    if (whole) { // the verifier must accept a plain proof (`prover_messages_outside` off)
      BLK_REQUIRE(!run.plan->refused(), "no such parameters");
      prover_messages_outside = false;
      rng_world::Seen prover, verifier;
      const std::vector<u8> proof = rng_world::prove(run, 5, prover);
      prover_messages_outside = true;
      if (!rng_world::verify(run, proof, verifier)) failures++, std::printf("rng_exact_test: %s: the verifier says no to this build's plain proof\n", name.c_str());
    }
    Exact x([&run](unsigned witness, Coins &coins) {
      Writer proof;
      proof.keep_list();
      rng_world::Seen seen;
      rng_world::prove(run, rng_world::rng_witness(witness), coins, proof, seen);
      return look_at(proof);
    }, 1);
    // rng_witness(): below 10 random; 10 zeros; 11 at the bound; 12 witness 0, one cell changed.
    for (unsigned second : long_run && !whole ? std::vector<unsigned>{1, 2, 10, 11, 12} : std::vector<unsigned>{1, 10}) {
      const Exact::Verdict v = x.compare(second);
      compared++;
      const bool as_foretold = v.same;
      if (verbose || !as_foretold) print_comparison("rng_exact_test", (name + ", witnesses 0 and " + std::to_string(second)).c_str(), x, v, verbose);
      if (!as_foretold) failures++;
      if (second == 1)
        summary += std::string(summary.empty() ? "" : "; ") + name + ": " + (v.same ? "the same law" : "told apart") + ", " + std::to_string(x.coins()) + " coins, " +
                   std::to_string(x.entries()) + " entries of the view, rank " + std::to_string(v.rank) + ", " + std::to_string(v.curved) + " coins curved";
    }
  }
  std::printf("rng_exact_test: %s (%u pairs of witnesses compared, random and special ones, at one list of challenges; the first of each shape: %s)\n", failures ? "FAILED" : "ok", compared, summary.c_str());
  return failures ? 1 : 0;
}
