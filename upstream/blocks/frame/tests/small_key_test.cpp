// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes and checks complete key files (a committed table's root and the committer's proofs about
// it: blocks/frame/check_key.h) at toy size. The fixture, wf_small_whole_world() in
// blocks/iop/tests/wf_small_world.h, has a 4 x 4 matrix (the witness), one lookup and hand-written
// plans: small enough for blocks/tests/hiding/key_exact_test.cpp to rerun the committer for each
// of its random field elements. Checks:
//   - two witnesses, each with two seeds of private randomness (`Coins`), give accepted files;
//   - the same witness and seed give the same bytes;
//   - a file with one flipped bit is rejected ("refused" in messages);
//   - nothing is written for a witness with an entry of 2 (valid: -2..1).
// Prints how many field elements and bytes one make_key_under() draws, the committed table's
// (`IndexCpu`) randomness not counted, and the file lengths (a round's repeated positions are
// opened once, which shortens a file; no round here draws two).
// The four macros enable test-only interfaces (listed in blocks/fence.h).
#define BLK_WF_ROWS_FOR_A_TEST 1
#define BLK_KEY_UNDER_FOR_A_TEST 1
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixture writes plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // fixture headers mark tables as tested (blocks/iop/foldeq.h)
#include "frame/check_key.h"
#include "iop/tests/wf_small_world.h"

using namespace blk;

namespace {
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
  const WfWorld w = wf_small_whole_world();
  KeyFixed f;
  f.configuration = w.lin.configuration;
  f.codes = w.lin.codes, f.lin = w.lin.S, f.index_masks = w.lin.input, f.ranges = w.wf, f.binding = w.lin.binding;
  const char *fault = key_fault(f);
  if (fault) std::printf("%s\n", fault);
  CHECK(!fault, "the small world's values are refused");
  if (fault) return 1;
  const KeyUnder under(KeyPlans{w.lin.first, w.lin.shared, w.lin.second}, f);
  EngineCpu engine;
  size_t elements = 0, plain = 0;
  auto make = [&](const LinMatrix &M, u64 coin_key, bool &made) {
    TestCoins commit_coins(31);  // fixed seed: the same witness gives the same table
    IndexCpu index(w.lin.S.fold.index, lin_table_rows(w.lin.S, M), commit_coins);
    const LinMatrix *m = &M;
    WfCpu ranges(w.wf, wf_matrix_of(w.lin.S), [m](u64 row, u64 column) { return m->at(row, column); });
    HeavyCpu products(w.lin.S, M);
    CountingCoins coins(coin_key);
    Writer file;
    made = make_key_under(under, index, ranges, products, engine, coins, nullptr, file);
    elements = coins.elements(), plain = coins.plain_bytes();
    return file.bytes();
  };
  auto check = [&](const std::vector<u8> &bytes) {
    Reader r(bytes.data(), bytes.size());
    TestedTable key;
    const char *why = nullptr;
    return check_key_under(under, r, key, why);
  };
  bool made = false;
  const LinMatrix one = wf_small_matrix(w.lin.S, 1), two = wf_small_matrix(w.lin.S, 2);
  const std::vector<u8> a = make(one, 11, made), again = make(one, 11, made), b = make(one, 12, made), c = make(two, 11, made), d = make(two, 12, made);
  for (const std::vector<u8> *bytes : {&a, &b, &c, &d}) CHECK(!bytes->empty() && check(*bytes), "an honest small key is refused");
  CHECK(a == again && a != b && a != c, "the same witness and coins give other bytes, or others the same");
  std::vector<u8> bent = a;
  bent[bent.size() / 2] ^= 4;
  CHECK(!check(bent), "a small key with a flipped bit is accepted");
  LinMatrix wild = one;
  wild.a[5] = from_u64(2);
  CHECK(make(wild, 13, made).empty() && !made, "a key is written for a witness out of its range");
  (void)make(one, 11, made);
  std::printf("%s: a making draws %zu elements of F and %zu plain bytes, and writes %zu bytes (others: %zu, %zu, %zu)\n", failures ? "FAILED" : "ok", elements, plain, a.size(), b.size(), c.size(),
              d.size());
  return failures != 0;
}
