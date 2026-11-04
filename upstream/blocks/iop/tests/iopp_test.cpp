// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the entire proximity test (`iopp`; meant to show that rows are close to codewords) on one
// group at toy size, prover against verifier, with the real hash chain, trees and CPU engine. A run
// (blocks/iop/tests/iopp_cases.h) sends the group's oracle messages and answers an out-of-domain
// sample of each (SEND), states three linear claims, batches them with the sample claims (BAT),
// folds and re-encodes each stage but the last (FOLD, then the code switch, CS), and checks the
// last (BASE). iopp_toy_world() derives the stage sizes itself, from the number of draws (positions
// drawn per stage); iopp_fault() (blocks/iop/iopp.h) must accept them.
//
// No prover here commits to rows far from the code. The test checks, for each case:
//   - true claims are accepted; one with a false value (`target`) is rejected (`refused` in
//     identifiers and output);
//   - the spare entries ("free room") of the first code switch's oracle message are not all zero;
//     no pad (it masks a sample's answer) is zero or repeats; no oracle message is opened at more
//     positions than one stage draws;
//   - for one small case: a proof with a bit flipped in any byte, or a byte shorter or longer, is
//     rejected;
// and that iopp_fault() finds stage sizes that do not fit ("do not close").
//
// The hand-written plans (blocks/iop/tests/iopp_support.h) give each round the error bound 0: error
// accounting is not tested.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // permits hand-written plans (blocks/numbers/plan.h)
#include "iop/tests/iopp_cases.h"

using namespace blk;

static IoppProved prove(const IoppCase &t, u64 seed, const E &off_target) {
  EngineCpu engine;
  return iopp_toy_prove(t, seed, off_target, engine);
}

int main() {
  const unsigned D = 6;
  const std::vector<IoppCase> cases = iopp_toy_cases(D);
  size_t runs = 0, refused = 0;
  for (const IoppCase &t : cases) {
    const IoppWorld w = iopp_toy_world(t);
    const char *fault = iopp_fault(w.codes, w.stages, t.answers);
    if (fault) std::printf("%s: %s\n", t.name, fault);
    CHECK(!fault, "the sizes of the stages do not close");
    if (fault) continue;
    const u64 seeds = 3;
    for (u64 seed = 100; seed < 100 + seeds; seed++) {
      const IoppProved p = prove(t, seed, E::zero());
      CHECK(iopp_toy_verify(t, seed, p.proof), "an honest proof is refused");
      CHECK(p.tables_fine, "an oracle is opened at a second list");
      if (const char *pads = toy_pads_fault([&] { (void)prove(t, seed, E::zero()); })) {
        std::printf("%s: %s\n", t.name, pads);
        CHECK(false, "a pad is not drawn, or is drawn once for two places or two runs");
      }
      {
        EngineCpu engine;
        CHECK(iopp_toy_prove(t, seed, E::zero(), engine, true).tables_fine, "a switch leaves its free room at zero, or draws no pad");
      }
      const IoppProved f = prove(t, seed, E::one());
      CHECK(!iopp_toy_verify(t, seed, f.proof), "a false target is accepted");
      if (seed == 100 && &t == &cases[1]) {
        for (size_t i = 0; i < p.proof.size(); i++) {
          std::vector<u8> changed = p.proof;
          changed[i] ^= (u8)(1u << (i % 8));
          runs++;
          if (!iopp_toy_verify(t, seed, changed)) refused++;
          else std::printf("ACCEPTED a proof with byte %zu changed: %s\n", i, t.name);
        }
        std::vector<u8> shorter(p.proof.begin(), p.proof.end() - 1), longer = p.proof;
        longer.push_back(0);
        CHECK(!iopp_toy_verify(t, seed, shorter), "a proof cut short is accepted");
        CHECK(!iopp_toy_verify(t, seed, longer), "a proof with a byte added is accepted");
      }
    }
  }
  {  // Sizes that do not close.
    const IoppWorld w = iopp_toy_world(cases[3]);
    auto faulty = [&](size_t stage, void (*change)(IoppStage &)) {
      std::vector<IoppStage> st = w.stages;
      change(st[stage]);
      size_t at = ~size_t(0);
      const char *fault = iopp_fault(w.codes, st, 1, &at);
      CHECK(fault != nullptr, "sizes that do not close pass");
      if (fault && at != stage) std::printf("stage %zu was changed, stage %zu is named: %s\n", stage, at, fault);
      CHECK(at == stage, "the fault is laid on another stage than the one that was changed");
    };
    faulty(0, [](IoppStage &s) { s.k = 70; });
    faulty(0, [](IoppStage &s) { s.h = s.k + 1; });
    faulty(0, [](IoppStage &s) { s.code = 99; });
    faulty(1, [](IoppStage &s) { s.k += 1; });
    faulty(1, [](IoppStage &s) { s.k = 0; });
    faulty(1, [](IoppStage &s) { s.h = s.k + 1; });  // a larger h <= k only adds room: no fault
    faulty(2, [](IoppStage &s) { s.mask_code = 0; });
    faulty(3, [](IoppStage &s) { s.k = 1; });
    CHECK(iopp_fault(w.codes, {w.stages[0]}, 1) != nullptr, "one stage passes");
    CHECK(iopp_fault(w.codes, w.stages, 0) != nullptr, "no answer passes");
    CHECK(iopp_fault(w.codes, w.stages, 1000000) != nullptr, "more answers than room passes");
  }
  CHECK(refused == runs, "something that had to be refused was accepted");
  std::printf("iopp_test: %s (%zu of %zu refused)\n", failures ? "FAILED" : "ok", refused, runs);
  return failures ? 1 : 0;
}
