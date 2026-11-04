// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the code that opens the weight table (weights and masks, encoded, under a Merkle tree) gets
// the weights. An opening (a proof's test on that table) needs sums over the entire weight matrix W
// two or more times, one after the other. It does not load W itself: each time, the program
// (cp_prove or cp_commit) supplies W in one pass over the weights ("walk" in identifiers), unit by
// unit; a unit (UnitW) is a range of W's rows. A callback of the opening (HfSeams, hf_open_api.h;
// implemented in hf_open_side.cuh) that needs such sums
//   1. creates an rf::HfPass (hf_passes.cuh: rf::HfFold or rf::HfColumns), which computes them;
//   2. calls HfSide::walk (hf_open_side.cuh) with the two functions (HfUnits) that add rows of W
//      to that object;
//   3. when it returns, calls rf::HfPass::finish() and reads the sums.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <cuda_runtime.h>

namespace rf {
struct UnitW; // common/weight_layout.h

struct HfUnits {
  // Adds rows of W: cells[c * ld + i] = W[first_row + i][c] for i < rows and every column c.
  // Arguments as for rf::HfPass::cells().
  std::function<void(const int8_t *cells, size_t ld, size_t rows, size_t first_row, cudaStream_t stream)> cells;
  // Adds an entire unit of routed experts' weights from its quantized blocks. Arguments as for
  // rf::HfPass::blocks(). Requires rf::HfPass::fits(u).
  std::function<void(const UnitW &u, const void *blocks, int type, const float *gains, float two_e, cudaStream_t stream)>
      blocks;
};
// The type of HfSide::walk. An implementation passes every row of every unit of W exactly once, in
// any order, to units.cells or units.blocks, and returns with the GPU idle.
// Not checked: at the call, every buffer of the opening and its callbacks must be in host memory or
// in the slab (the GPU memory pool that keeps its contents throughout an opening, selected by
// HfSeams::short_lived(false)), none in memory borrowed from the program (short_lived(true)).
// cp_prove's implementation is Unit::walk() in hf_keeper.cu; class Unit implements HfKeeperUnit
// (hf_keeper_unit.h). There one thread runs the opening and a second, which waits in next_walk(),
// loads the weights. Unit::walk() zeroes borrowed memory and treats every buffer in it as freed,
// because the second thread may take that memory back to load the weights. It then blocks until
// that thread has added the units on its own stream and called go_on(). If that thread calls
// abandon() instead, Unit::walk() throws rf::Abandoned; the callbacks' buffers are zeroed with the
// slab, by leaf_end().
// cp_commit's implementation, HfOncePanels::walk() (hf_commit_once.cuh), adds the units on the
// calling thread.
typedef std::function<void(const HfUnits &units)> HfWalk;
} // namespace rf
