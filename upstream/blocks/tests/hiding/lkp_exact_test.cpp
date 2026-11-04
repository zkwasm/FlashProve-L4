// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the exact test of hiding (blocks/tests/hiding/exact.h) on LKP alone, the lookup argument
// (every key occurs in a public table), with the CPU prover (`EngineCpu`) of
// blocks/iop/tests/lkp_world.h on small_world(): do two witnesses give the view the same law
// (distribution) over the coins? A pass says nothing about the GPU engine (`EngineGpu`,
// prover/engine/engine_gpu.cuh), the only engine that `cp_prove` uses.
// Under block_run()'s plan, SEND sends one oracle message (the keys, the bits of the
// multiplicities, one mask row) and takes an out-of-domain sample of it; LKP's rounds and a linear
// claim on the keys follow. The coins are all field elements that the prover draws; the view is
// the sample's answers and LKP's scalars. This plan has no proximity test (`iopp` in identifiers),
// so nothing is opened and that test's openings are not covered. With "long", whole_run()'s plan
// covers them, for two of the five pairs of witnesses. Each plan is tried at one list of
// challenges.
// The witness is the keys (lkp_witness()). The prover computes the multiplicities from them, and
// LKP's public inputs are the same for all witnesses. Of the eight keys, 0 and 1 are table row 1
// in every witness, 6 and 7 are padding, and none is the table's last row: keys 2 to 5 vary over
// rows 0 to 2. Witnesses 0 to 2 are random; in 1000 these four keys are row 0, in 1001 row 2; 1002
// is witness 0 with the byte that selects key 2 raised by one.
// Arguments: "-v": a line for each comparison and draw; "long": also whole_run() (two minutes).

// Adds the switch `prover_messages_outside` (blocks/chain/rounds.h), true at first. While it is
// true, the prover's messages stay out of its hash chain, so the challenges do not depend on what
// it sends. Only the prover's channel (`ProverRounds`) reads the switch, and no verifier accepts a
// proof made while it is true.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // to write a plan by hand (blocks/numbers/plan.h)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function" // the header below has functions unused here
#include "iop/tests/lkp_world.h"
#pragma GCC diagnostic pop
#include "iop/tests/iopp_support.h"
#include "tests/hiding/sessions.h"

using namespace blk;

// Like lkp_world::block_run(), with a proximity test of two stages after LKP. Codes and plan are
// written by hand. Each stage draws `draws` positions, and the rows' random tails (`tail`) are just
// long enough for that many.
static lkp_world::LookupRun whole_run(const lkp_world::World &w, unsigned log_len, unsigned log_mask, unsigned draws) {
  using namespace lkp_world;
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 32;
  LookupRun run(w, set);
  Plan plan(set);
  run.whole = true;
  LOG_LEN = log_len;
  const size_t tail = tail_for_positions(draws);
  run.codes.mask.push_back(toy_code(size_t(1) << log_mask, tail, 0));
  run.codes.base.push_back(toy_code(size_t(1) << log_len, tail, run.codes.mask.back().log_n));
  run.stages = {IoppStage{0, 0, 1, 0}};
  toy_later_stages(run.codes, run.stages, {Later{0, 0}}, draws, set.samples);
  const std::vector<OracleNotice> sent = {{oracle_shape(set), 0}};
  plan_chain_start(plan, "look-up, whole, by hand", true);
  plan_send(plan, "keys, multiplicities", {SampleSize{Send::cube(run.codes, oracle_shape(set)).bits, no_sized_group()}}, (unsigned)tree_sizes(run.codes, sent).size());
  plan_lookup(plan, "look-up", LookupSize{w.fixed.parts[0], w.fixed.d, w.fixed.mu, w.fixed.shifted, w.fixed.live, w.fixed.parts});
  toy_plan_iopp(plan, run.codes, run.stages, draws, IoppChoice(), false, rows_left(w) + set.samples);
  return run.plan = plan.finish(), run;
}

int main(int argc, char **argv) {
  bool verbose = false, long_run = false;
  for (int i = 1; i < argc; i++) verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
  int failures = 0;
  unsigned compared = 0;
  std::string summary;
  const lkp_world::World w = lkp_world::small_world();
  for (bool whole : {false, true}) {
    if (whole && !long_run) break;
    const lkp_world::LookupRun run = whole ? whole_run(w, 7, 7, 2) : lkp_world::block_run(w, 7, 7);
    const std::string name = whole ? "a whole run" : "the run that ends behind the block";
    if (whole) { // the verifier must accept a plain proof (`prover_messages_outside` off)
      prover_messages_outside = false;
      lkp_world::Seen prover, verifier;
      const std::vector<u8> proof = lkp_world::prove(run, 5, prover);
      prover_messages_outside = true;
      if (!lkp_world::verify(run, proof, verifier)) failures++, std::printf("lkp_exact_test: %s: the verifier says no to a plain proof\n", name.c_str());
    }
    Exact x([&](unsigned witness, Coins &coins) {
      Writer proof;
      proof.keep_list();
      lkp_world::Seen seen;
      lkp_world::prove(run, lkp_world::lkp_witness(w.fixed, witness), coins, proof, seen);
      return look_at(proof);
    }, 1);
    for (unsigned second : whole ? std::vector<unsigned>{1, 1000} : std::vector<unsigned>{1, 2, 1000, 1001, 1002}) {
      const Exact::Verdict v = x.compare(second);
      compared++;
      if (verbose || !v.same) print_comparison("lkp_exact_test", (name + ", witnesses 0 and " + std::to_string(second)).c_str(), x, v, verbose);
      if (!v.same) failures++;
      if (second == 1)
        summary += std::string(summary.empty() ? "" : "; ") + name + ": " + (v.same ? "the same law" : "not the same law") + ", " + std::to_string(x.coins()) + " coins, " + std::to_string(x.entries()) +
                   " entries of the view, rank " + std::to_string(v.rank) + ", " + std::to_string(v.curved) + " coins curved";
    }
  }
  std::printf("lkp_exact_test: %s (%u pairs of witnesses compared, random and special ones, at one list of challenges; the first of each run: %s)\n", failures ? "FAILED" : "ok", compared, summary.c_str());
  return failures ? 1 : 0;
}
