// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests that the CPU engine (`EngineCpu`; no other engine is run) gives every Merkle-tree leaf its
// own salt, from its own bytes of the prover's private randomness (`Coins`). An opening can include
// digests of unopened leaves; without a fresh salt, such a digest lets a reader test a guess of the
// leaf's column (all rows' symbols at its position). The leaves over one short position
// (x mod 2^LOG_N_ZK, where the mask rows are read) hold the same mask-row symbols: if they shared a
// salt, opening one would leave only the others' main-row symbols to guess. The verifier only
// hashes opened leaves' salts (check_opening(), blocks/tree/opening.h), so it accepts shared,
// constant or reused salts; the exact test of hiding (blocks/tests/hiding/exact.h) leaves salts and
// digests out. So main() compares every leaf's salt with the bytes that the prover drew. It injects
// no fault: this project's forms of it, built from a copy of blocks/tree/opening.h with shared or
// zero salts, which pass only if the fault is detected, are not part of this repository.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "engine/engine_cpu.h"
#include "numbers/setting.h"
#include "tests/hiding/exact.h"

using namespace blk;

namespace {

constexpr unsigned LOG_N = 6, LOG_N_ZK = 4;

// Commits to two trees from one `Coins` object, opens both at every position and returns the
// openings' salts, one vector per tree. The openings are not verified.
std::vector<std::vector<u8>> salts_of(Coins &coins, size_t salt_bytes) {
  const Code code{LOG_N, 8, 3}, mask_code{LOG_N_ZK, 4, 3};
  BLK_REQUIRE(code_ok(code) && code_ok(mask_code), "the test's numbers are not codes in this setting");
  ToyCoins data(9);
  EngineCpu engine;
  std::vector<std::vector<u8>> salts;
  for (u32 tree = 0; tree < 2; tree++) {
    Table rows(1, 2, code.ell);
    data.fill(rows.a.data(), rows.a.size());
    OraclePart part;
    part.main = engine.put(rows), part.code = code, part.mask_code = mask_code;
    part.mask = Table(E_DEGREE, 1, mask_code.ell), part.mask_tail = Table(E_DEGREE, 1, mask_code.r);
    data.fill(part.mask.a.data(), part.mask.a.size()), data.fill(part.mask_tail.a.data(), part.mask_tail.a.size());
    OracleTreeShape shape;
    shape.log_n = LOG_N, shape.log_n_zk = LOG_N_ZK, shape.main_symbols = {2}, shape.mask_symbols = {E_DEGREE};
    shape.tree_number = tree, shape.salt_bytes = salt_bytes;
    shape.run.w[0] = 1, shape.parameters.w[0] = 2;
    Digest root;
    const CommitId commit = engine.commit({part}, shape, coins, root);
    std::vector<u64> all((size_t)1 << LOG_N);
    for (size_t x = 0; x < all.size(); x++) all[x] = x;
    Opening opening;
    engine.open(commit, PositionList::from_draws(all, LOG_N), opening);
    salts.push_back(opening.salts);
  }
  return salts;
}

} // namespace

// Records the draws of salts_of() in a `ListedCoins` (blocks/coins/tests/listed_coins.h: a test
// sets its values) and repeats them on chosen values. Checks: each tree draws its n salts in one
// draw (`stretches`), the second tree's bytes right after the first's; leaf x has bytes
// x * salt_bytes onward of its tree's draw; no two salts are equal; none is zeros.
int main() {
  const size_t salt_bytes = setting_goldilocks().salt_bytes, n = (size_t)1 << LOG_N;
  ListedCoins learn;
  salts_of(learn, salt_bytes);
  std::vector<ListedCoins::Draw> stretches;
  for (const ListedCoins::Draw &d : learn.ledger())
    if (!d.elements) stretches.push_back(d);
  int faults = 0;
  std::string seen_faults;
  auto expect = [&](bool ok, const char *what) {
    if (ok) return;
    seen_faults += std::string(faults++ ? "; " : "") + what;
  };
  expect(stretches.size() == 2 && stretches[0].count == n * salt_bytes && stretches[1].count == n * salt_bytes && stretches[1].first == n * salt_bytes,
         "a tree's salts are not drawn in one stretch of n times the salt's length, each tree its own");
  if (!faults) {
    std::vector<F> elements(learn.elements().size());
    std::vector<u8> bytes(learn.bytes().size());
    ToyCoins(1).fill(elements.data(), elements.size()), ToyCoins(2).bytes(bytes.data(), bytes.size());
    ListedCoins play(elements, bytes);
    const std::vector<std::vector<u8>> salts = salts_of(play, salt_bytes);
    std::set<std::vector<u8>> seen;
    bool own = true, zero = false;
    for (size_t tree = 0; tree < 2; tree++)
      for (size_t x = 0; x < n; x++) {
        const std::vector<u8> salt(salts[tree].begin() + (std::ptrdiff_t)(x * salt_bytes), salts[tree].begin() + (std::ptrdiff_t)((x + 1) * salt_bytes));
        const size_t at = stretches[tree].first + x * salt_bytes;
        own = own && salt == std::vector<u8>(bytes.begin() + (std::ptrdiff_t)at, bytes.begin() + (std::ptrdiff_t)(at + salt_bytes));
        zero = zero || salt == std::vector<u8>(salt_bytes, 0);
        seen.insert(salt);
      }
    expect(own, "a leaf's salt is not its own stretch of the coins' bytes");
    expect(!zero, "a leaf's salt is zeros");
    expect(seen.size() == 2 * n, "two leaves have one salt");
  }
  std::printf("salts_test: %s%s\n", faults ? "FAILED: " : "ok (2 trees of 64 leaves, 4 leaves over each short position: every leaf shows its own bytes of the coins; no two alike, none zero)", seen_faults.c_str());
  return faults ? 1 : 0;
}
