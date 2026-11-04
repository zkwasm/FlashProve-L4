// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Tests that a GPU Merkle tree returns 32 zero bytes for a digest that was never written, not
// whatever that memory contained before.
//
// CardTree::begin() (prover/engine/tree.cuh) zeroes a tree's arrays. A run without a fault
// cannot show this, because every digest is written before it is read. CardTree's caller writes the
// leaf digests, so this test writes all but one, as if the GPU had lost that store. It finishes the
// tree and opens it at the sibling leaf, so the unwritten digest comes first in the opening. It
// does this for a tree with all vertices and a lean tree, which drops the middle levels, on GPU
// memory in each of the four states of prover/engine/tests/filled_arena.cuh.
// Expected: tree_check() accepts the opening (a verifier does not notice the loss), and the first
// digest is zero.
// Control: in each state a plain buffer is allocated first. The test fails unless that buffer is
// non-zero in at least one state; the arena filled with 0xEE makes it so.
// Not tested: a lost store above the leaves, and so the zeroing of a lean tree's upper levels.
// k_tree_nodes() and k_subtrees() make those stores; this test cannot suppress one.
// Exit code 0 and a last line "tree_arrays_cleared_test: ok ..."; 1 on a failure; 3 without a GPU.
#include <cstdio>
#include <vector>

#include "engine/tests/filled_arena.cuh"
#include "engine/tree.cuh"
using namespace blk;

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
    std::printf("no GPU: nothing was tested\n");
    return 3;
  }
  constexpr unsigned log_n = 12;
  static_assert(log_n > TREE_LOWER_LEVELS, "a shorter tree is never lean");
  const u64 n = (u64)1 << log_n, unwritten = 1029; // a leaf in the second subtree of 2^10 leaves
  Digest start{};
  start.w[0] = 1;
  const TreeShape shape = TreeShape::plain(log_n, start);
  std::vector<Digest> leaves(n);
  for (u64 x = 0; x < n; x++)
    for (u32 i = 0; i < 8; i++) leaves[x].w[i] = 0x9E3779B9u * (u32)(8 * x + i + 1);
  const PositionList list = PositionList::from_draws({unwritten ^ 1}, log_n);
  int failures = 0, states_not_zero = 0;
  {
    FilledArena memory("tree_arrays_cleared_test");
    for (memory.first(); memory.more(); memory.next()) {
      const InArena in(memory.arena());
      {
        Card<Digest> plain(2 * n); // as long as the longest array of the trees below
        std::vector<Digest> found(plain.size());
        plain.to_host(found.data());
        bool zero = true;
        for (const Digest &d : found) zero = zero && d == Digest{};
        states_not_zero += !zero;
      }
      for (const bool lean : {false, true}) {
        auto expect = [&](bool ok, const char *what) {
          if (ok) return;
          failures++, memory.differs();
          std::printf("FAILED: %s (%s; %s)\n", what, lean ? "a lean tree" : "a tree with all vertices", memory.state());
        };
        CardTree tree;
        tree.begin(shape, lean);
        Digest *const bottom = tree.bottom();
        const u64 after = unwritten + 1;
        ENGINE_CK(cudaMemcpy(bottom, leaves.data(), unwritten * sizeof(Digest), cudaMemcpyHostToDevice));
        ENGINE_CK(cudaMemcpy(bottom + after, leaves.data() + after, (n - after) * sizeof(Digest), cudaMemcpyHostToDevice));
        const Digest root = tree.finish();
        const std::vector<Digest> given = tree.open(list);
        expect(given.size() == log_n && tree_check(root, shape, list, {leaves[unwritten ^ 1]}, given),
               "tree_check() rejects the opening next to the leaf that was not written");
        expect(!given.empty() && given[0] == Digest{}, "the digest that was never written is sent as something other than 32 zero bytes");
      }
    }
  }
  if (!states_not_zero) {
    failures++;
    std::printf("FAILED: a fresh buffer was zero in all four states of GPU memory, so nothing was tested\n");
  }
  std::printf("tree_arrays_cleared_test: %s (two trees of 2^%u leaves in four states of GPU memory; a fresh buffer was non-zero in %d)\n",
              failures ? "FAILED" : "ok", log_n, states_not_zero);
  return failures != 0;
}
