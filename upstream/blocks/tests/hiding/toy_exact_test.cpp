// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the exact test of hiding (blocks/tests/hiding/exact.h) on whole proofs of toy size: do two
// witnesses give the view, every field element in the proof (such as scalars, sample answers,
// opened columns and the last step's rows), the same law (distribution)?
// Digests and salts are not in the view; a pass covers only the pairs and challenges that were run.
// An oracle message (`OracleShape`, blocks/iop/relation.h) has main rows, such as the witness's
// vectors here, and mask rows, which the prover draws from its coins. E, the cubic extension of the
// field F, has elements of three coordinates, so a mask entry is three coins.
//   product   blocks/toy/product.h: vectors with a[i] b[i] = c[i], by the blocks SEND (oracle
//             messages and their out-of-domain samples), ZERO (the verifier's random point), PSC
//             (a sumcheck with a hiding polynomial) and a whole proximity test;
//   links     blocks/toy/segments.h: three products in a row, x_(l+1) = x_l b_l, each a segment
//             (`link` in identifiers) of the frame of blocks/iop/req.h: a chain of segments, each
//             closed into the next, with one proximity test at the end. This run has kept rows
//             (main rows that a later segment sends again) and oracle messages whose pad (the
//             random element that masks a sample's answer) is in a main row.
// The provers are toy_prove() and segments_prove() on the CPU engine (`EngineCpu`): a pass says
// nothing about the GPU engine (`EngineGpu`, prover/engine/engine_gpu.cuh), the only engine that
// `cp_prove` uses. The coins are all field elements that the prover under test draws from its
// `Coins` object, such as mask rows, pads, random tails (random entries encoded with each row), the
// masks of the proximity test's folds and last step.

// BLK_TEST_PROVER_MESSAGES_OUTSIDE (blocks/chain/rounds.h) compiles in the switch
// `prover_messages_outside`, which is on except where test() turns it off. While it is on, the
// verifier's messages are fixed: nothing that the prover sends enters the queries of its hash chain
// (the Fiat-Shamir transcript), so it runs the interactive protocol under a fixed list of
// challenges. No verifier is changed, and none accepts such a proof.
#define BLK_TEST_PROVER_MESSAGES_OUTSIDE 1
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine_cpu.h"
#include "tests/hiding/exact.h"
#include "toy/product.h"
#include "toy/segments.h"

using namespace blk;

namespace {

Setting tiny_setting() {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 10; // bits: no security level; hiding does not depend on it
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 6);
  return set;
}

// Witnesses. Numbers below 10 are pseudo-random. The others are special, since a leak may show for
// special witnesses only: 10 all zeros; 11 witness 0 with one entry of `first` changed; 12 `first`
// zero; 13 every other entry of `others` zero; 14 all ones.
// first: a (product) or x (links). others: b, or b_1 .. b_L one after the other.
void fill_witness(unsigned witness, F *first, F *others, size_t len, size_t count) {
  ToyCoins from(100 + (witness == 11 ? 0 : witness));
  from.fill(first, len), from.fill(others, len * count);
  if (witness == 11) first[0] = first[0] + F::one();
  for (size_t i = 0; i < len; i++)
    if (witness == 10 || witness == 12) first[i] = F::zero();
    else if (witness == 14) first[i] = F::one();
  for (size_t i = 0; i < len * count; i++)
    if (witness == 10 || (witness == 13 && i % 2)) others[i] = F::zero();
    else if (witness == 14) others[i] = F::one();
}
const unsigned SPECIAL[] = {10, 11, 12, 13, 14};

// Writes a whole proof for witness number `witness`.
using Maker = std::function<void(unsigned witness, Coins &coins, Writer &proof)>;

Maker product_maker(const ToyRun &run) {
  return [&run](unsigned witness, Coins &coins, Writer &proof) {
    Table abc(1, 3, run.shape.codes.base[0].ell);
    fill_witness(witness, abc.a.data(), abc.a.data() + abc.len, abc.len, 1);
    for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));
    EngineCpu engine;
    toy_prove(run, abc, engine, coins, proof);
  };
}

using LinksEngine = EngineCpu;

Maker links_maker(const SegmentsRun &run) {
  return [&run](unsigned witness, Coins &coins, Writer &proof) {
    std::vector<F> x(size_t(1) << run.vars()), all(x.size() * run.links());
    fill_witness(witness, x.data(), all.data(), x.size(), run.links());
    std::vector<std::vector<F>> b;
    for (size_t l = 0; l < run.links(); l++) b.emplace_back(all.begin() + (std::ptrdiff_t)(l * x.size()), all.begin() + (std::ptrdiff_t)((l + 1) * x.size()));
    LinksEngine engine;
    segments_prove(run, x, b, engine, coins, proof);
  };
}

World world_of(const Maker &make) {
  return [make](unsigned witness, Coins &coins) {
    Writer proof;
    proof.keep_list();
    make(witness, coins, proof);
    return look_at(proof);
  };
}

bool verbose = false, long_run = false;
int failures = 0;
unsigned compared = 0;
std::string summary;

// Whether the witnesses must be told apart in `run`: in no run of this repository's builds. What
// this project's builds of this test with a fault in the library require, and what a pass therefore
// does not show: README.md, "What these tests do not show".
bool foretold(const char *run) {
  return (void)run, false;
}

// levels: the number of the run's sumchecks whose summand multiplies two maps (polynomials that
// extend two of the prover's vectors), both with their end values, the values at the challenges,
// sent under masks (`Summand::shown`);
// sampled: the number of its sampled oracle messages whose mask row Send::fresh_mask() draws.
void test(const char *name, const Maker &make, const std::function<bool(const Writer &)> &verify, size_t levels, size_t sampled) {
  { // a plain proof (`prover_messages_outside` off): the verifier must accept it
    prover_messages_outside = false;
    ToyCoins coins(5);
    Writer proof;
    make(0, coins, proof);
    prover_messages_outside = true;
    if (!verify(proof)) failures++, std::printf("toy_exact_test: %s: the verifier says no to this build's plain proof\n", name);
  }
  const World world = world_of(make);
  for (u64 key = 1; key <= (long_run ? 11u : 1u); key += 10) {
    Exact x(world, key);
    std::vector<unsigned> seconds = {1};
    if (long_run) seconds.push_back(2), seconds.insert(seconds.end(), std::begin(SPECIAL), std::end(SPECIAL));
    for (unsigned second : seconds) {
      const Exact::Verdict v = x.compare(second);
      if (verbose)
        for (size_t i = 0; i < x.draws().size(); i++) {
          size_t alone = 0;
          for (size_t c = x.draws()[i].first; c < x.draws()[i].first + x.draws()[i].count; c++) alone += v.coin_matters[c];
          std::printf("  %s: draw %zu, coins %zu to %zu: %s; %zu of its coins are needed one by one\n", name, i, x.draws()[i].first,
                      x.draws()[i].first + x.draws()[i].count - 1, v.draw_matters[i] ? "needed" : "could be constants", alone);
        }
      const bool as_foretold = foretold(name) ? (!v.same && v.entry >= 0) : v.same;
      if (verbose || !as_foretold)
        std::printf("toy_exact_test: %s, challenges %llu, witnesses 0 and %u: %s (%zu coins in %zu draws; %zu entries of the view, rank %zu; %zu coins curved, %zu idle; needed: %zu "
                  "coins one by one, %zu draws; %u steps)%s%s%s%s\n",
                  name, (unsigned long long)key, second, v.same ? "the same law" : "TOLD APART", x.coins(), x.draws().size(), x.entries(), v.rank, v.curved, v.idle,
                  v.count(v.coin_matters), v.count(v.draw_matters), v.steps, v.same ? "" : ": ", v.why.c_str(), v.entry >= 0 ? "; " : "", x.where(v.entry).c_str());
      if (!as_foretold) failures++;
      compared++;
      // The test also fails if the numbers of curved and idle coins differ from those accounted for
      // here, so that a change in how the prover uses its coins does not pass unnoticed.
      // Curved coins (the view is not affine in them): the two mask entries that each of the
      // `levels` sumchecks multiplies with each other, three coordinates each. Idle coins (they
      // change nothing in the view): the pad's place in the mask row of each of the `sampled`
      // messages, three coordinates: Send::fresh_mask() draws it and sets it to zero, and
      // send_prove() then draws the pad.
      if (v.curved != 6 * levels || v.idle != 3 * sampled)
        failures++, std::printf("toy_exact_test: %s: %zu coins curved and %zu idle, where %zu and %zu are understood\n", name, v.curved, v.idle, 6 * levels, 3 * sampled);
      if (key == 1 && second == 1)
        summary += std::string(summary.empty() ? "" : "; ") + name + ": " + (v.same ? "the same law" : "told apart") + ", " + std::to_string(x.coins()) + " coins, " +
                   std::to_string(x.entries()) + " entries of the view, rank " + std::to_string(v.rank) + ", " + std::to_string(v.curved) + " coins curved";
    }
  }
}

} // namespace

// Arguments: "-v" prints a line for each comparison and each of the prover's draws; "long" adds a
// second list of challenges, a third random witness and five special ones (two minutes);
// "product" or "links" selects that run only; with both, no run is made and the program still
// prints ok and exits with code 0. Without "long": witnesses 0 and 1, one list of challenges.
int main(int argc, char **argv) {
  bool product = true, links = true;
  for (int i = 1; i < argc; i++) {
    verbose = verbose || !std::strcmp(argv[i], "-v"), long_run = long_run || !std::strcmp(argv[i], "long");
    if (!std::strcmp(argv[i], "product")) links = false;
    if (!std::strcmp(argv[i], "links")) product = false;
  }
  const Setting set = tiny_setting();
  const ToyRun one = toy_size(set, 4, 7, 7, {StageChoice{1, 1, 8, 8}, StageChoice{0, 0, 9, 9}}, "product, tiny edition: a test size that reaches no security level");
  const SegmentsRun chain = segments_size(set, 3, 6, 9, 9, {StageChoice{1, 1, 9, 9}, StageChoice{0, 0, 10, 10}}, "links, tiny edition: a test size that reaches no security level");
  if (one.plan->refused() || !one.plan->holds() || chain.plan->refused() || !chain.plan->holds()) {
    std::printf("toy_exact_test: FAILED (no such parameters)\n");
    return 1;
  }
  if (product) test("product", product_maker(one), [&](const Writer &p) { return toy_verify(one, p.data(), p.size()); }, 1, 1);
  if (links) test("links", links_maker(chain), [&](const Writer &p) { return segments_verify(chain, p.data(), p.size()); }, chain.links(), chain.links());
  std::printf("toy_exact_test: %s (%u pairs of witnesses compared%s; the first: %s)\n", failures ? "FAILED" : "ok", compared, long_run ? ", random and special ones, at two lists of challenges" : ", random ones, at one list of challenges",
              summary.c_str());
  return failures ? 1 : 0;
}
