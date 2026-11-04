// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the honest committer against the library's public calls configure(), make_key() and
// check_key(). make_key() writes a key file: a committed table's root and the committer's
// proofs about that table. check_key() verifies it (blocks/frame/check_key.h).
// blocks/iop/tests/wf_test.cpp runs the committer's range proof (WF) under
// plans written by hand, with 0 for the proximity tests' error bounds (a plan lists a protocol's
// rounds and their error bounds). Here configure() computes the plans from two public files,
// toy_file() and toy_model(), which have toy message lengths but the real target of 128 bits and
// proofs of work.
// This test runs no malicious committer:
// check_key() must accept the honest committer's key file.
// The exit code is 0 only if every `CHECK` passes.

// WF is the committer's range proof, by lookups (LKP), on the cells of the matrix in the committed
// table (`Index` in identifiers); the lookup `bytes` allows -4 .. 3. LIN checks linear claims on
// that table. A key file has three hash chains (Fiat-Shamir transcripts): WF runs in the first, the
// one-time chain, and LIN ends in the last, the second chain.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the included fixtures build plans (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // the fixtures mark tables as tested (blocks/iop/foldeq.h)
#include "frame/check_key.h"
#include "frame/tests/toy_matrix.h"
#include "frame/tests/toy_shapes.h"

using namespace blk;

int main() {
  const Configured c = configure({toy_file(), toy_model()}, toy_shapes);
  if (!c.refused().empty()) std::printf("the public files: %s\n", c.refused().c_str());
  CHECK(c.refused().empty(), "the toy's public files are not admitted");
  if (!c.refused().empty()) return 1;
  const LinShape &S = c.sized().once.S;
  const WfPublic &x = c.sized().ranges.x;
  const WfMatrix size = wf_matrix_of(S);
  EngineCpu engine;
  const NonceSearch search = [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };

  const LinMatrix good = toy_matrix_in_range(S, x, 3);
  [[maybe_unused]] LinMatrix wild = good;
  wild.a[(wild.rows - 1) * wild.columns + 5] = wf_signed(4);  // only `bytes` covers the last row
  // `committed` fills the table; `ranges` and `products` read `spoken_of`; nothing reads `wild`.
  const LinMatrix &committed = good, &spoken_of = good;
  TestCoins commit_coins(24), coins(44);
  IndexCpu index(S.fold.index, lin_table_rows(S, committed), commit_coins, c.configuration());
  const LinMatrix *m = &spoken_of;
  WfCpu ranges(x, size, [m](u64 row, u64 column) { return m->at(row, column); });
  HeavyCpu products(S, spoken_of);
  Writer file;
  file.keep_list();
  const bool made = make_key(c, index, ranges, products, engine, coins, search, file);
  CHECK(made, "the committer's run does not go through");
  if (!made) return 1;
  Reader r(file.bytes().data(), file.bytes().size());
  TestedTable key;
  const char *why = nullptr;
  const char *place = nullptr;
  const ProofVerdict verdict = check_key(c, r, key, why, nullptr, &place);
  CHECK(verdict != ProofVerdict::NO && !place, "the control: an honest key is refused, or a yes names a place");
  std::printf("%s: the control: an honest key of %zu bytes under the public files\n", failures ? "FAILED" : "ok", file.bytes().size());
  return failures != 0;
}
