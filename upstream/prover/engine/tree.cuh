// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A Merkle tree in GPU memory (`Card` in identifiers means GPU): the perfect binary tree of
// blocks/tree/tree.h. Call begin(), fill bottom(), a GPU pointer, with the leaves' digests
// (position x at bottom()[x]) and call finish(); then open() as often as needed, and release().
// A failed check or CUDA call ends the process. The vertex at depth d and place i (its index
// within the depth) has the number 2^d + i, tree_index().
//
// What a tree of n = 2^a leaves keeps in GPU memory until release():
//   not lean  every vertex's value, in one array, by number: 64 n bytes;
//   lean      (a > 10; a smaller tree is kept whole) the bottom (depth a) and the depths 0 to
//             a - 10: 32 n + n / 16 bytes. finish() computes the nine levels between in shared
//             memory, a subtree of 2^10 leaves per thread block, and drops them. open() recomputes
//             the subtrees that it needs, in work space of up to 32 KB per drawn position. The
//             digests are the same.
#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "engine/device.cuh"
#include "hash/hash.h"
#include "tree/tree.h"

namespace blk {

// An inner vertex's hash takes its two children and `tag` (TreeShape::vertex_tag()), one tag for
// all vertices: a domain number and TreeShape::start, a constant of the instance. Neither the
// tree's number nor the vertex's place nor the run (one execution of the protocol) is in it: see
// vertex() in blocks/hash/hash.h.
template <class Fn> __global__ void k_tree_nodes(Digest *value, unsigned depth, TagT<Fn> tag) {
  const u64 width = (u64)1 << depth, i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  value[width + i] = vertex<Fn>(tag, value[2 * (width + i)], value[2 * (width + i) + 1]);
}

// 128 threads for 2^10 leaves: a subtree's upper levels occupy one warp, which waits for each hash
// in turn while the multiprocessor's other blocks work (`__launch_bounds__` asks for four or more).
// prover/engine/card_plan.h copies TREE_LOWER_LEVELS as CARD_LOWER_LEVELS: keep the two equal.
constexpr unsigned TREE_LOWER_LEVELS = 10, TREE_SUBTREE_THREADS = 128;
// One block computes one subtree of `levels` <= 10 levels (not checked) from the bottom's values
// (depth a). Its top is at depth cut = a - levels and place which[blockIdx.x], or blockIdx.x if
// `which` is null. If not null, `top` gets the top's value, at top[place], and `all` gets the
// vertex at depth cut + e (e < levels) and place j within the subtree, at
// all[blockIdx.x * 2^levels + 2^e + j].
template <class Fn>
__global__ __launch_bounds__(TREE_SUBTREE_THREADS, 4) void k_subtrees(const Digest *bottom, unsigned levels, const u64 *which, Digest *top,
                                                                   Digest *all, TagT<Fn> tag) {
  __shared__ Digest value[1u << TREE_LOWER_LEVELS];
  const u64 above = which ? which[blockIdx.x] : blockIdx.x; // the place of the subtree's top
  for (u32 j = threadIdx.x; j < (1u << levels); j += blockDim.x) value[j] = bottom[(above << levels) + j];
  __syncthreads();
  for (unsigned e = levels; e-- > 0;) // the depth cut + e
    // Each iteration computes one vertex per thread, from the left, and stores it in value[from ..]
    // after the barrier: later ones at this depth read only from value[2 (from + threads)] on.
    for (u32 from = 0; from < (1u << e); from += blockDim.x) {
      const u32 j = from + threadIdx.x;
      const bool mine = j < (1u << e);
      Digest v;
      if (mine) {
        v = vertex<Fn>(tag, value[2 * j], value[2 * j + 1]);
        if (all) all[((u64)blockIdx.x << levels) + (1u << e) + j] = v;
      }
      __syncthreads();
      if (mine) value[j] = v;
      __syncthreads();
    }
  if (top && threadIdx.x == 0) top[above] = value[0];
}
// out[i] = A[at[i] mod 2^61]; A is `work` if at[i] has bit 62, `bottom` if bit 61, else `upper`.
static __global__ void k_pick_lean(const Digest *upper, const Digest *bottom, const Digest *work, const u64 *at, size_t count, Digest *out) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const u64 a = at[i], index = a & (((u64)1 << 61) - 1);
  out[i] = (a >> 62 & 1) ? work[index] : (a >> 61 & 1) ? bottom[index] : upper[index];
}

class CardTree {
public:
  // Allocates and zeroes the arrays. `lean`: see the top of this file.
  void begin(const TreeShape &shape, bool lean = false) {
    BLK_REQUIRE(shape.valid() && shape.log_n <= 30, "tree on the GPU: not a shape, or more than 2^30 leaves");
    shape_ = shape;
    const unsigned a = shape.log_n;
    lean_ = lean && a > TREE_LOWER_LEVELS; // depth 10 or less: kept whole (at most 64 KB)
    cut_ = lean_ ? a - TREE_LOWER_LEVELS : a;
    if (lean_) {
      bottom_.resize((size_t)1 << a);
      value_.resize((size_t)2 << cut_);
    } else
      value_.resize((size_t)2 << a);
    // The arrays are zeroed before anything is written to them, because the memory still contains
    // what its last user left there, such as masks, salts or the model's inner values. A run
    // without a fault writes every digest before it is read, so the zeros change no file. If the
    // GPU silently loses a store, or a whole thread block of k_subtrees(), while the tree is built,
    // the unwritten digest is hashed into the root and can be sent in an opening that the verifier
    // accepts: it is then zero, not the last user's data. Nothing here detects the loss. Not
    // covered: a digest damaged later; other GPU memory, open()'s buffers included.
    ENGINE_CK(cudaMemsetAsync(value_.get(), 0, value_.size() * sizeof(Digest), 0));
    if (lean_) ENGINE_CK(cudaMemsetAsync(bottom_.get(), 0, bottom_.size() * sizeof(Digest), 0));
    value_.spent(), bottom_.spent(); // a test with card_test_byte still finds its byte, not zeros
  }
  Digest *bottom() const { return lean_ ? bottom_.get() : value_.get() + ((size_t)1 << shape_.log_n); }
  // Bytes of GPU memory: what this tree holds until release(); what a tree of `shape` will hold.
  size_t bytes() const { return (value_.size() + bottom_.size()) * sizeof(Digest); }
  static size_t bytes(const TreeShape &shape, bool lean) {
    const unsigned a = shape.log_n;
    if (!lean || a <= TREE_LOWER_LEVELS) return sizeof(Digest) * ((size_t)2 << a);
    return sizeof(Digest) * (((size_t)1 << a) + ((size_t)2 << (a - TREE_LOWER_LEVELS)));
  }

  // Computes the root from the filled bottom(); waits for `stream`. The caller must ensure two
  // things, which are not checked: the fill is complete or queued on `stream`; and, here and in
  // open(), `stream` is the default stream or keeps order with it, because Card<> (a GPU array,
  // prover/engine/device.cuh), begin()'s zeroing and open()'s cudaMemcpy() calls work on the
  // default stream.
  Digest finish(cudaStream_t stream = 0) {
    const Tag tag = shape_.vertex_tag();
    if (lean_) {
      k_subtrees<HashFn><<<1u << cut_, TREE_SUBTREE_THREADS, 0, stream>>>(bottom_.get(), TREE_LOWER_LEVELS, nullptr,
                                                                          value_.get() + ((size_t)1 << cut_), nullptr, tag);
      ENGINE_CK(cudaGetLastError());
    }
    for (unsigned d = cut_; d-- > 0;) { // depth cut_ is filled: above if lean, else by the caller
      k_tree_nodes<HashFn><<<blocks_for((u64)1 << d, 128), 128, 0, stream>>>(value_.get(), d, tag);
      ENGINE_CK(cudaGetLastError());
    }
    Digest root;
    ENGINE_CK(cudaMemcpyAsync(&root, value_.get() + 1, sizeof root, cudaMemcpyDeviceToHost, stream));
    ENGINE_CK(cudaStreamSynchronize(stream));
    return root;
  }

  // Returns the digests of an opening at `list`: the vertex values that a verifier cannot compute
  // from the listed leaves, in the order of tree_walk(). Waits for `stream`, unless the opening has
  // no digest.
  // GPU buffers are sized by bounds that follow from the shape and the number of draws (repeats
  // counted), not by what this list needs, which depends on how the drawn positions share paths.
  // A listed leaf needs at most one digest at each depth. A digest deeper than cut_ is the sibling
  // of a vertex on a listed leaf's path, so it is in that leaf's subtree: at most one subtree per
  // listed leaf is recomputed. So neither the size nor the number of the GPU allocations depends on
  // which positions were drawn. Entries beyond what this list needs are neither written nor read.
  std::vector<Digest> open(const PositionList &list, cudaStream_t stream = 0) const {
    struct Ask {
      struct Value {};
      std::vector<std::pair<unsigned, u64>> at; // depth, place
      Value leaf(size_t) { return {}; }
      bool given(unsigned d, u64 place, Value &) {
        at.push_back({d, place});
        return true;
      }
      Value node(unsigned, u64, const Value &, const Value &) { return {}; }
    } side;
    Ask::Value root;
    tree_walk(shape_, list, side, root);
    std::vector<Digest> out(side.at.size());
    const unsigned a = shape_.log_n;
    const size_t draws = list.draws().size(), digests_most = draws * a, subtrees_most = lean_ && cut_ + 1 < a ? (size_t)std::min<u64>(draws, (u64)1 << cut_) : 0;
    Card<u64> at(digests_most), which(subtrees_most);
    Card<Digest> picked(digests_most), work(subtrees_most << TREE_LOWER_LEVELS);
    if (out.empty()) return out;
    std::vector<u64> from(out.size()), subtrees; // where each digest is; subtree tops to recompute
    for (const auto &v : side.at)
      if (v.first > cut_ && v.first < a) subtrees.push_back(v.second >> (v.first - cut_));
    std::sort(subtrees.begin(), subtrees.end());
    subtrees.erase(std::unique(subtrees.begin(), subtrees.end()), subtrees.end());
    for (size_t i = 0; i < out.size(); i++) {
      const unsigned d = side.at[i].first;
      const u64 place = side.at[i].second;
      if (!lean_ || d <= cut_)
        from[i] = tree_index(d, place);
      else if (d == a)
        from[i] = (u64)1 << 61 | place;
      else {
        const u64 k = std::lower_bound(subtrees.begin(), subtrees.end(), place >> (d - cut_)) - subtrees.begin();
        from[i] = (u64)1 << 62 | ((k << TREE_LOWER_LEVELS) + ((u64)1 << (d - cut_)) + (place & (((u64)1 << (d - cut_)) - 1)));
      }
    }
    BLK_REQUIRE(out.size() <= digests_most && subtrees.size() <= subtrees_most, "tree: an opening needs more digests or subtrees than the room that is taken for them");
    ENGINE_CK(cudaMemcpy(at.get(), from.data(), from.size() * sizeof(u64), cudaMemcpyHostToDevice));
    if (!subtrees.empty()) {
      ENGINE_CK(cudaMemcpy(which.get(), subtrees.data(), subtrees.size() * sizeof(u64), cudaMemcpyHostToDevice));
      k_subtrees<HashFn><<<(unsigned)subtrees.size(), TREE_SUBTREE_THREADS, 0, stream>>>(bottom_.get(), TREE_LOWER_LEVELS, which.get(), nullptr,
                                                                                         work.get(), shape_.vertex_tag());
      ENGINE_CK(cudaGetLastError());
    }
    k_pick_lean<<<blocks_for(out.size(), 128), 128, 0, stream>>>(value_.get(), bottom_.get(), work.get(), at.get(), out.size(), picked.get());
    ENGINE_CK(cudaGetLastError());
    ENGINE_CK(cudaStreamSynchronize(stream));
    ENGINE_CK(cudaMemcpy(out.data(), picked.get(), out.size() * sizeof(Digest), cudaMemcpyDeviceToHost));
    return out;
  }
  void release() { value_.release(), bottom_.release(); }

private:
  TreeShape shape_;
  bool lean_ = false;
  unsigned cut_ = 0;       // value_ keeps depths 0 .. cut_, by number; if not lean, cut_ = log_n
  Card<Digest> value_, bottom_;
};

} // namespace blk
