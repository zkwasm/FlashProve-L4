// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The exact test of hiding (blocks/tests/hiding/exact.h) on WF alone: the committer's range proof,
// by lookups, on a committed matrix's entries. wf_small_prove() (blocks/iop/tests/wf_small_world.h)
// sends the index-mask message (masks for WF's claims on the matrix) and the multiplicities by the
// block SEND, with an out-of-domain sample of each, runs one lookup and stops. So the test covers
// neither the openings of WF's proximity test nor the linear claims that WF leaves on the matrix
// (both sides compute them, neither sends them) nor what later blocks do with them. It has no
// negative control. It runs the prover on the CPU engine (`EngineCpu`, `WfCpu`): a pass says
// nothing about the GPU engine (`EngineGpu`, `WfGpu`), the only one on which cp_commit runs WF.
// The witness is the matrix: 4 x 4 random entries in -2 .. 1; the public statement does not depend
// on them. Three pairs of matrices are compared, at one list of challenges: their views must have
// the same law (distribution). The coins are all field elements that the prover draws. The view is
// the samples' answers and the lookup's scalars: nothing is opened.
// Arguments: "-v" prints a line for each comparison and draw.
// BLK_TEST_PROVER_MESSAGES_OUTSIDE: challenges do not depend on the prover's messages. It changes
// only the prover's channel (`ProverRounds`, blocks/chain/rounds.h); no verifier accepts such runs.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#define BLK_WF_ROWS_FOR_A_TEST 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixture builds its plan by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets a fixture declare a table tested (blocks/iop/foldeq.h)
#include <cstdio>
#include <cstring>
#include <string>

#include "engine/engine_cpu.h"
#include "iop/tests/wf_small_world.h"
#include "tests/hiding/sessions.h"

using namespace blk;

int main(int argc, char **argv) {
  const bool verbose = argc > 1 && !std::strcmp(argv[1], "-v");
  int failures = 0;
  unsigned compared = 0;
  std::string summary;
  const WfSmall w = wf_small_world();
  Exact x([&w](unsigned witness, Coins &coins) {
    Writer proof;
    proof.keep_list();
    EngineCpu engine;
    BLK_REQUIRE(wf_small_prove(w, wf_small_matrix(w, 300 + witness), engine, coins, proof), "the matrix is out of its range");
    return look_at(proof);
  }, 1);
  for (unsigned second : {1u, 2u, 3u}) {
    const Exact::Verdict v = x.compare(second);
    compared++;
    if (verbose || !v.same) print_comparison("wf_exact_test", ("witnesses 0 and " + std::to_string(second)).c_str(), x, v, verbose);
    if (!v.same) failures++;
    if (second == 1)
      summary = std::string(v.same ? "the same law" : "not the same law") + ", " + std::to_string(x.coins()) + " coins, " + std::to_string(x.entries()) + " entries of the view, rank " +
                std::to_string(v.rank) + ", " + std::to_string(v.curved) + " coins curved";
  }
  std::printf("wf_exact_test: %s (%u pairs of random matrices within the range compared, at one list of challenges; the first: %s)\n", failures ? "FAILED" : "ok", compared, summary.c_str());
  return failures ? 1 : 0;
}
