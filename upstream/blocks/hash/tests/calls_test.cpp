// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Counts the compressions (calls of the hash's compression function) that functions of
// blocks/hash/hash.h make. The expected counts restate blocks/hash/README.md, "The calls".
// Named there but never run here: leaf() on a digest, leaf_begin(), leaf_step(), leaf_last(),
// chain_call_cells(), chain_block(), work_test(), chain_work_passes().
// Every hashed string begins with a 16-byte label: the formulas' leading 16.
#define BLK_COUNT_HASH_CALLS 1
#include "hash/hash.h"
#include <vector>
using namespace blk;

static int bad = 0, checks = 0;
template <class Call> static void costs(u64 want, const char *what, Call call) {
  const u64 before = hash_calls_counted();
  call();
  const u64 made = hash_calls_counted() - before;
  checks++;
  if (made != want) bad++, printf("\n%s: %llu compressions, expected %llu", what, (unsigned long long)made, (unsigned long long)want);
}
int main() {
  const u64 B = HASH_BLOCK_BYTES;
  Digest d{};
  d.w[0] = 1;
  costs(0, "a tag", [&] { Tag t(Domain::STATEMENT), v(Domain::LEAF_MAIN, 5, d, 7, 40); (void)t, (void)v; });
  const Tag vertex_tag(Domain::TREE_NODE, 0, d);
  costs(1, "a vertex", [&] { vertex(vertex_tag, d, d); });
  costs(1, "a run's value", [&] { run_value(1, d, d); });
  costs(1, "made-up coins", [&] { test_coins(d, 3); });
  // The fill: zeros that move a leaf's salt entirely into the last block.
  for (u64 rows : {1, 6, 7, 8, 11, 12, 14, 15, 16, 125, 129, 140, 170, 2048})
    for (u64 salt : {40, 48, 64}) {
      const std::vector<F> column(rows, F::zero());
      const std::vector<u8> bytes(salt);
      const Tag leaf_tag(Domain::LEAF_MAIN, 5, d, rows, salt);
      costs((16 + 8 * rows + salt + B - 1) / B, "a leaf", [&] { leaf(leaf_tag, 77, column.data(), rows, bytes.data(), salt); }); // the fill adds no compression
      checks++, bad += (leaf_bytes(rows, salt) + B - 1) / B != (16 + 8 * rows + salt + B - 1) / B;
    }
  for (u64 sent : {0, 8, 24, 32, 48, 56, 64, 96, 104, 112, 120, 128, 200, 232, 240, 248, 1000}) { // a hash chain's round: the prover's bytes, then an 8-byte word, then the 8-byte nonce
    const std::vector<u8> bytes(sent);
    const u64 before_last_word = (16 + sent) / B, call = (16 + sent + 8) % B == 0 ? 2 : 1;
    ChainPrefix p;
    costs(before_last_word, "what a round's calls share", [&] {
      ChainAbsorb a(d, 4);
      a.push(bytes.data(), bytes.size()), a.push_u64(1);
      p = a.prefix();
    });
    for (u64 j : {0, 1, 9}) costs(call, "a call of a round", [&] { chain_call(p, 5, j); });
    checks++, bad += before_last_word + call != hash_calls(sent + 16);
  }
  for (u64 n : {0, 1, 127, 128, 129, 256, 257, 4096})
    costs((16 + n + B - 1) / B, "a hash of bytes", [&] {
      const std::vector<u8> bytes(n);
      Hasher h(Domain::STATEMENT, 0);
      h.push(bytes.data(), n);
      h.finish();
    });
  printf("%sthe compressions of the hash's calls, %d checks: %s\n", bad ? "\n" : "", checks, bad ? "NOT as README.md says" : "all as README.md says");
  return bad != 0;
}
