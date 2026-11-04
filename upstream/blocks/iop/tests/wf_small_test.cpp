// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks the test fixture wf_small_world() (`world` in identifiers), which a test of hiding uses
// (blocks/tests/hiding/wf_exact_test.cpp). The fixture runs WF (the range proof on a matrix's
// entries, by lookups) without its proximity test (BLK_WF_ROWS_FOR_A_TEST), so runs prove nothing.
//   - wf_fault() accepts the parameters. The lookup covers every row and has 16 padding keys.
//   - Runs on two witnesses, each with two sets of private randomness (`Coins`), are well formed
//     and equally long, and each leaves two pending claims on the matrix (`rows` in messages). The
//     same inputs give the same bytes; a change of either gives other bytes. A run without its last
//     byte is not well formed. The prover rejects an out-of-range witness.
//   - A run's two pads (each masks one out-of-domain sample's answer), read from the prover's
//     memory in two runs with different randomness: none is 0, none repeats.
//   - The last line prints how many field elements (`F`) and plain bytes a run draws, and how many
//     bytes it sends.
#define BLK_WF_ROWS_FOR_A_TEST 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixture builds its plan by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets test code mark a table as tested (blocks/iop/foldeq.h)
#include "iop/tests/wf_small_world.h"

using namespace blk;

namespace {
// Counts what a run draws: blocks/tests/hiding/exact.h reruns the prover at least once per element
// of F drawn.
class CountingCoins : public Coins {
 public:
  explicit CountingCoins(u64 key) : inner_(key) {}
  void bytes(u8 *out, size_t n) override { inner_.bytes(out, n), bytes_ += n; }
  void fill(F *out, size_t n) override { inner_.fill(out, n), elements_ += n; }
  size_t elements() const { return elements_; }
  size_t plain_bytes() const { return bytes_; }

 private:
  TestCoins inner_;
  size_t elements_ = 0, bytes_ = 0;
};
}  // namespace

int main() {
  const WfSmall w = wf_small_world();
  const WfMatrix size = wf_matrix_of(w.lin.S);
  const char *fault = wf_fault(w.lin.codes, w.wf, size);
  if (fault) std::printf("%s\n", fault);
  CHECK(!fault, "the small world's parameters are refused");
  if (fault) return 1;
  CHECK(size.rows == 4 && size.k_col == 2 && wf_covers_all_rows(w.wf, 0, size), "the small matrix, or what the look-up covers");
  const LkpPublic told = wf_lookups(w.wf, size)[0];
  CHECK(told.padding.size() == 3 && lkp_padding_count(told) == 16, "the places of padding: one stretch in part 0, two in part 1, four places of four keys");

  EngineCpu engine;
  auto run = [&](u64 witness, u64 coin_key, std::vector<PadLook> *pads, size_t *elements = nullptr, size_t *plain = nullptr) {
    CountingCoins coins(coin_key);
    Writer out;
    out.keep_list();
    const bool made = wf_small_prove(w, wf_small_matrix(w, witness), engine, coins, out, pads);
    CHECK(made, "the honest prover refuses a witness in range");
    if (elements) *elements = coins.elements();
    if (plain) *plain = coins.plain_bytes();
    return out.bytes();
  };
  std::vector<PadLook> pads_a, pads_b;
  size_t elements = 0, plain = 0;
  const std::vector<u8> a = run(1, 11, &pads_a, &elements, &plain), b = run(1, 12, &pads_b), c = run(2, 11, nullptr), d = run(2, 12, nullptr), again = run(1, 11, nullptr);
  for (const std::vector<u8> *bytes : {&a, &b, &c, &d}) {
    Notes notes;
    CHECK(wf_small_well_formed(w, *bytes, &notes) && notes.size() == 2, "a run is not well formed, or leaves another number of rows than two");
    CHECK(bytes->size() == a.size(), "the runs differ in length");
  }
  CHECK(a == again && a != b && a != c, "the same witness and coins give other bytes, or others the same");
  const char *pads = pads_fault(pads_a, pads_b);
  if (pads) std::printf("%s\n", pads);
  CHECK(!pads && pads_a.size() == 2, "the pads of the two sampled messages");
  {
    LinMatrix wild = wf_small_matrix(w, 1);
    wild.a[5] = from_u64(2);  // an entry must be in -2 .. 1
    CountingCoins coins(13);
    Writer out;
    CHECK(!wf_small_prove(w, wild, engine, coins, out), "the prover runs on a witness that is out of its range");
  }
  std::vector<u8> bent = a;
  bent.pop_back();
  CHECK(!wf_small_well_formed(w, bent), "a run that is cut short counts as well formed");
  std::printf("%s: a run draws %zu elements of F and %zu plain bytes, and sends %zu bytes\n", failures ? "FAILED" : "ok", elements, plain, a.size());
  return failures != 0;
}
