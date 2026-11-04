// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests RNG, the approximate range check on committed cells (blocks/iop/rng.h): it is meant to show
// |cell| < 2 D for the prover's small integers. The runs are complete, but for the small case, the
// `before_rng` runs in main() and runs that abort, and at a low error target (at most
// 33 bits) to keep them short. blocks/iop/tests/rng_world.h has cases, prover and verifier.
// A run: the prover sends the packed cells (three to an entry), then the reads (weighted partial
// sums of the cells). In RNG the verifier draws a matrix with entries in {-1, 0, 1}; each of its N
// columns gives one combination of the cells' columns; the prover sends the two digits of each
// combination + D (`b0` and `b1` bits: 2 D = 2^(`b0` + `b1`)) and, for the lookup on the digits'
// ranges (LKP, blocks/iop/lkp.h), the multiplicities. A batching element turns RNG's equations into
// one linear claim (`Row` in identifiers, "row" in messages). A proximity test on all four oracle
// messages (encoded rows, sent as Merkle roots) ends the run.
// This test covers honest runs, proofs with a flipped bit, which must be rejected ("refused"
// in messages and identifiers), and RNG's aborts on invalid input: see whole_test() and main().
// No malicious prover runs in this test: in no run here
// does the verifier get a proof on a cell beyond the bound.
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "iop/tests/rng_world.h"

using namespace blk;
using namespace rng_world;

static int failures = 0;
#define CHECK(cond, text) \
  do { \
    if (!(cond)) failures++, std::printf("FAILED %s:%d %s\n", __FILE__, __LINE__, text); \
  } while (0)

// `FALSE_ROWS`: the names of the verifier's claims that must be false on what the prover sent
// (false_rows()). The prover here is honest, so there are none.
static const char *const FALSE_ROWS = "";

// Runs f in a child process. True if SIGABRT killed the child and its stderr contains `text`.
static bool aborts_saying(const char *text, const std::function<void()> &f) {
  int fd[2];
  if (pipe(fd)) return false;
  std::fflush(stdout);
  const pid_t child = fork();
  BLK_REQUIRE(child >= 0, "tests: no child process");
  if (child == 0) {
    const rlimit no_core{0, 0}; // no core file if the child aborts
    setrlimit(RLIMIT_CORE, &no_core);
    dup2(fd[1], 2), close(fd[0]), close(fd[1]);
    f();
    _exit(0);
  }
  close(fd[1]);
  std::string said;
  char buffer[512];
  for (ssize_t n; (n = read(fd[0], buffer, sizeof buffer)) > 0;) said.append(buffer, (size_t)n);
  close(fd[0]);
  int status = 0;
  BLK_REQUIRE(waitpid(child, &status, 0) == child, "tests: a child process is lost");
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && said.find(text) != std::string::npos;
}

// `runs` counts the runs that must be rejected, `refused` those that were: main() compares them.
static size_t runs = 0, refused = 0;
static unsigned combinations_run = 0;
static int whole_test();

// A case with one matrix of 3 rows (one packed row) and 2 D = 128. If `contradicting`, shape and
// plan (the rounds with their errors) have different numbers of combinations but the same number
// of entries of the verifier's matrix, 33 (see `name`). In pair 1 neither number is 1.
[[maybe_unused]] static RngCase two_numbers(int pair, bool contradicting) {
  RngCase c;
  c.name = pair == 0 ? (contradicting ? "33 columns x 1 combination under a plan for 1 column x 33" : "33 columns x 33 combinations in shape and plan")
                     : (contradicting ? "11 columns x 3 combinations under a plan for 3 columns x 11" : "11 columns x 11 combinations in shape and plan");
  c.b0 = 4, c.b1 = 3, c.log_radix = 8, c.digit_columns = 4;
  c.target = pair == 0 ? 33 : 11;
  c.matrices = {{3, c.target}}, c.reads = {{0, 3}}, c.cell_bound = 1;  // at most 33 x 1 < D = 64: every honest combination stays in [-D, D)
  if (contradicting) c.shape_combinations = pair == 0 ? 1 : 3, c.planned_columns = c.shape_combinations;
  return c;
}
int main() {
  for (bool reference : {false, true}) {
    the_case = RngCase(), the_case.reference = reference;
    if (whole_test()) return 1;
  }
  // The small case (rng_small_case(), for blocks/tests/hiding/rng_exact_test.cpp) ends right after
  // RNG's claim is made. With and without mask rows in RNG's own two messages
  // (`RngCase::reference`), the prover's run completes for five witnesses; the proof's bytes depend
  // on the witness and on the private randomness (`coins`), its length on neither; no pad (the
  // random element that masks a sample's answer) is 0 or occurs twice in two runs with different
  // `coins`. No verifier runs: the oracle messages get no proximity test.
  for (bool reference : {false, true}) {
    the_case = rng_small_case(reference);
    const RangeRun run = size_run();
    std::vector<std::vector<u8>> proofs;
    for (unsigned witness : {0u, 1u, 10u, 11u, 12u}) {
      ToyCoins coins(500);
      Writer proof;
      Seen seen;
      prove(run, rng_witness(witness), coins, proof, seen);
      proofs.push_back(proof.bytes());
      CHECK(!proofs.back().empty() && proofs.back().size() == proofs[0].size(), "the small case: a witness moves the proof's length");
      for (size_t i = 0; i + 1 < proofs.size(); i++) CHECK(proofs[i] != proofs.back(), "the small case: two witnesses give one proof under the same coins");
    }
    Seen one, other;
    const std::vector<u8> a = prove(run, 501, one), b = prove(run, 502, other);
    const char *fault = pads_fault(one.pads, other.pads);
    CHECK(!fault && a != b && a.size() == b.size(), "the small case: a pad is not drawn afresh, or the coins move nothing");
    std::printf("  %s: %u combinations, %zu bytes, %zu mask entries for the look-up\n", the_case.name, run.combinations, a.size(), lkp_masks(rng_lookup(rng_shape(run.set, run.combinations))));
  }
  // The number of combinations is in the shape and in the plan. The plan computes the error of the
  // matrix's round from its own number and fixes only the number of the matrix's entries. For
  // inputs in which the two numbers differ and the entries' numbers agree, rng_plan_fault() must
  // report a fault, and both sides must abort where the matrix would be drawn. The verifier gets
  // there on the bytes of the two messages before RNG (`before_rng`).
  for (int pair : {0, 1}) {
    the_case = two_numbers(pair, true);
    const RangeRun run = size_run();
    const RngShape S = rng_shape(run.set, run.combinations);
    CHECK(!run.plan->refused() && run.plan->holds() && !rng_fault(S), "contradicting inputs: the plan does not hold or the shape is unfit by itself: then the stops below show nothing");
    const char *fault = rng_plan_fault(S, run.plan);
    CHECK(fault && std::strstr(fault, "another number of combinations"), "a shape with another number of combinations than the plan's is not called a fault");
    CHECK(aborts_saying("another number of combinations", [&] { Seen s; prove(run, 70, s); }), "RNG's prover drew a matrix of another number of combinations than the plan prices");
    ToyCoins coins(70);
    Writer before;
    Seen s;
    prove(run, rng_witness(0), coins, before, s, false, nullptr, true);
    CHECK(!before.bytes().empty() && aborts_saying("another number of combinations", [&] { Seen v; verify(run, before.bytes(), v); }), "RNG's verifier drew a matrix of another number of combinations than the plan prices");
    the_case = two_numbers(pair, false);
    const RangeRun agreeing = size_run();
    const RngShape A = rng_shape(agreeing.set, agreeing.combinations);
    Seen p, v;
    CHECK(!rng_plan_fault(A, agreeing.plan) && verify(agreeing, prove(agreeing, 70, p), v), "one number of combinations in shape and plan: an honest proof is refused");
    // With these numbers equal, the channel's count (blocks/chain/rounds.h) of the matrix's entries
    // fixes the number of the cells' columns, so the accounting reads the run's own number from the
    // plan (`small_columns`: `small` is the range check).
    CHECK(account(agreeing.plan).small_columns == A.all_columns(), "the accounting reads another number of columns off the plan than the shape has");
  }
  CHECK(refused == runs && runs > 0, "something that had to be refused was accepted");
  std::printf("rng_test: %s (%u combinations; %zu of %zu refused)\n", failures ? "FAILED" : "ok", combinations_run, refused, runs);
  return failures ? 1 : 0;
}
// Runs `the_case` from plan to verdict, under two seeds (`key`) of the prover's private
// randomness. This test checks that the honest proof is accepted, and rejected with one bit
// flipped (at up to 300 places); that RNG's claim equals, coefficient by coefficient, the sum of
// its equations under the powers of the batching element; that no pad is 0 or occurs twice in two
// runs; and that RNG's prover aborts on invalid input (the verifier is not run on it): reads
// without an answered sample (a random linear form of a sent message), digits on a pad's place,
// invalid shapes.
static int whole_test() {
  const RangeRun run = size_run();
  if (run.plan->refused()) {
    std::printf("FAILED: no such parameters: %s\n", run.plan->why_not().c_str());
    return 1;
  }
  combinations_run = run.combinations;
  for (u64 key = 60; key < 62; key++) {
    Seen p, v;
    const std::vector<u8> proof = prove(run, key, p);
    const char *why = "";
    const bool accepted = verify(run, proof, v, &why);
    if (false_rows(p, v) != FALSE_ROWS) std::printf("  false rows: %s\n", false_rows(p, v).c_str());
    CHECK(v.rows.size() == rows_left(rng_shape(run.set, run.combinations)) && false_rows(p, v) == FALSE_ROWS, "other rows are false on what was sent than this prover should leave");
    CHECK(accepted, "an honest proof is refused");
    if (!accepted) std::printf("  %s\n", why);
    CHECK(same_rows(p, v), "prover and verifier come to other rows");
    CHECK(p.structured_row_is_the_sum, "RNG's row is not the sum of its equations under the powers of the batching element");
    if (key == 60)
      for (size_t i = 0; i < proof.size(); i += proof.size() / 300 + 1) {
        std::vector<u8> changed = proof;
        changed[i] ^= (u8)(1u << (i % 8));
        Seen s;
        runs++;
        if (!verify(run, changed, s)) refused++;
        else std::printf("ACCEPTED a proof with byte %zu changed\n", i);
      }
  }
  {  // pads of all four messages, two runs on the same cells, different `coins`: none 0, none twice
    Seen one, other;
    prove(run, 70, one), prove(run, 71, other);
    const char *fault = pads_fault(one.pads, other.pads);
    if (fault) std::printf("  pads: %s\n", fault);
    size_t in_main = 0;
    for (const PadLook &p : one.pads) in_main += !p.in_mask_rows;
    CHECK(!fault && in_main == (the_case.reference ? 2 * 3 * (size_t)run.set.samples : 0), "a pad is not drawn afresh, or the pads do not lie where the shape puts them");
  }
  if (the_case.reference) {  // digits on a pad's place
    RngShape changed = rng_shape(run.set, run.combinations);
    for (LaidBlock &b : changed.digits_laid)
      if (b.first_row == 1 && b.first_c == 2 * the_case.digit_columns - 1 && b.first_i == 0) b.first_place = (u64(1) << the_case.log_len) - (u64(1) << b.place_bits);  // the last places of the last row
    handed_in = &changed;
    CHECK(aborts_saying("the digits cover a place of their message's pad", [&] { Seen s; prove(run, 60, s); }), "RNG lets digits lie on a pad's place");
    handed_in = nullptr;
  }
  CHECK(aborts_saying("the reads has no answered sample", [&] { Seen s; prove(run, 60, s, true); }), "RNG drew its matrix although the reads had no answered sample");
  // RNG's prover must abort with `text` on the run's shape as altered by `change`, such as a layout
  // with a hole or an overlap, a radix too small or too large, a number out of range.
  auto stops = [&](const char *text, const std::function<void(RngShape &)> &change) {
    RngShape changed = rng_shape(run.set, run.combinations);
    change(changed);
    handed_in = &changed;
    if (!aborts_saying(text, [&] { Seen s; prove(run, 60, s); })) failures++, std::printf("FAILED: RNG did not stop with the words: %s\n", text);
    handed_in = nullptr;
  };
  stops("packed matrices are not laid lawfully", [](RngShape &S) { S.matrices[1].laid[0] = S.matrices[0].laid[0]; });
  stops("packed matrices are not laid lawfully", [](RngShape &S) { S.matrices[0].laid.pop_back(); });
  stops("reads are not laid lawfully", [](RngShape &S) { S.reads_laid[1].first_place = S.reads_laid[0].first_place; });
  stops("digits are not laid lawfully", [](RngShape &S) { S.digits_laid.pop_back(); });
  stops("lies in a row that is not sent", [](RngShape &S) { S.digits.shape.main_rows = 3; for (LaidBlock &b : S.digits_laid) if (b.first_row == 1) b.first_row = 3; });
  stops("the radix is too small", [](RngShape &S) { S.log_radix = 5; });
  stops("could wrap around in the field", [](RngShape &S) { S.log_radix = 30; });
  stops("out of range", [](RngShape &S) { S.digit_columns = u64(1) << 63; });
  stops("too few columns", [](RngShape &S) { S.digit_columns = 8; });
  stops("no power of two", [](RngShape &S) { S.digit_columns = 17; });
  stops("gets another number on the channel", [](RngShape &S) { S.digits_in = 3; });
  stops("from no earlier message", [](RngShape &S) { S.masks_from = 2; });
  stops("no earlier than the digits", [](RngShape &S) { S.reads_in = 2; });
  stops("do not fit row 0", [](RngShape &S) { S.digit_bits[0] = 10, S.log_radix = 14; });
  {  // RNG's derived lookup is valid; its padding is exactly the key numbers that name no digit.
    const RngShape S = rng_shape(run.set, run.combinations);
    const LkpPublic L = rng_lookup(S);
    bool right = lkp_fault(L) == nullptr;
    for (u64 i = 0; right && i < L.all_keys(); i++) right = lkp_is_padding(L, 0, i) == ((i >> 1) / S.digit_columns >= S.combinations);
    CHECK(right, "the derived look-up is unfit, or its padding is not the key numbers without a digit");
  }
  return 0;
}
