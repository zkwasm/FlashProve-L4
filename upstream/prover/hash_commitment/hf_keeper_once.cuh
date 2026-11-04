// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// HfOnceTable presents, as a blk::IndexHolder (blocks/iop/foldeq.h; `keeper` in identifiers), a
// weight table that the committer program cp_commit has just committed: all the weights as rows,
// each encoded together with random entries (its masks), under a Merkle tree. The holder folds the
// rows at a point into one row (`u`), opens the drawn columns and counts them. blk::make_key()
// needs one holder for the key file (`ivk/key.bin`), blk::make_entry() two for a leaf file
// (`ivk/leaf/<k>.bin`; `entry` in identifiers). HfOnceTable works on the calling thread; the
// counterpart for batches is prover/hash_commitment/hf_keeper_unit.h.
//
// Include prover/weights_commitment/lent_memory.h and prover/weights_commitment/common.cuh (for
// rf::LayW) first. The first header's macros turn cudaMalloc and cudaFree, here too, into
// lent::take() and lent::give(). Between lent::begin_opening() and lent::end_opening(), take()
// allocates from the pool lent::borrowed if lent::borrow is true, otherwise from lent::own.
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "engine/engine_gpu.cuh"
#include "hash_commitment/hf_keeper.h"
#include "hash_commitment/hf_open_side.cuh"

namespace hf {

// Wraps an HfKeeper (`keeper_`, returned by holder(); prover/hash_commitment/hf_keeper.h). Its
// callbacks (`seams` in identifiers) are those that rf::hf_side_seams() makes of an rf::HfSide
// (`side_`; prover/hash_commitment/hf_open_side.cuh); the caller supplies each pass over all the
// weights W (`walk` in identifiers). "The engine" is the constructor's blk::EngineGpu, which
// commits to the prover's tables, under Merkle trees, on the GPU.
//
// GPU memory (`card` in identifiers). During its own passes this object holds nothing in
// lent::borrowed, as prover/hash_commitment/hf_walk.h requires.
//   holder().fold_table()  allocates omega, the sums over the folded row's blocks, in lent::own;
//                          one pass computes omega and the folded row. The row goes into a buffer
//                          from lent::borrowed, which the engine adopts in place.
//   holder().open()        removes the engine's own trees from GPU memory and frees the row's
//                          buffer; computes the drawn columns in one or more passes; fills a new
//                          buffer and tells the engine its address. The engine writes the pad of
//                          the row's out-of-domain sample again and rebuilds its trees: a root that
//                          has changed ends the process.
// fold_table() makes no room in GPU memory: before it the other parts of make_key() must have
// released their large buffers. The pools need lent_bytes() free in lent::borrowed, and in
// lent::own rf::hf_side_slab_bytes() + own_bytes() and whatever `walk` allocates: a pool that runs
// out ends the process with exit code 4.
//
// The engine allocates outside the pools, from a CUDA memory pool (blk::card_pool(),
// prover/engine/device.cuh) that keeps what it has taken. No formula gives its peak. Measured from
// the pools' set-up until make_key() or make_entry() returned, at full size (Llama 4 Maverick's
// shape, Goldilocks field, synthetic weights): 6,442,450,944 and 6,476,005,376 bytes for a key file
// under two seeds (one run each), 8,422,162,432 for a leaf file (one run); it varies with the
// challenges in steps of 32 MiB. The callers in cp_commit leave it at least 24 GiB
// (HF_ONCE_ENGINE_ROOM, prover/hash_commitment/hf_commit_once.cuh) when they set up the pools, and
// print each run's peak in their last line.
//
// The hash chain that follows LIN's shared part (LIN: the test of linear claims on the table,
// blocks/iop/lin.h) takes memory of its own: a key file's claim
// puts thousands of Tensors (vectors in product form) into one linear claim (`Row`), that of
// lin_dot_row(). From lin_written_out_from Tensors on, the prover first sums them into a dense one.
class HfOnceTable {
 public:
  // P: with `configuration` (the configuration digest) set. Y: the layout of W.
  // columns: how many columns the plan (the configuration's list of rounds) draws from this table
  //   (key file: LinShape::fold.columns). If holder().begin() or holder().open() is given another
  //   number, the process ends.
  // walk, held, fibres_a_walk, lines: as in rf::HfSide. `held` (HfHeld, hf_open_api.h) has the
  //   stored nodes of the table's tree and their lowest level, the masks' key (`seed`) with its
  //   second copy, and pinned host memory for `u` or null (then `u` is allocated in lent::own).
  // root: the table's Merkle root, 32 bytes, which identifies the table in first queries (the
  //   first hash input of each hash chain, a Fiat-Shamir transcript); holder().commitment() returns
  //   it.
  // record: the table's record, which the committer has computed before this call.
  //   holder().record() returns it.
  // count: the caller's persistent count of this table's opened columns, called by
  //   holder().begin(); false: the masks do not cover `columns` more. The committer's own openings
  //   count like a batch's.
  // engine: the one that make_key() or make_entry() is given.
  // P, Y, engine and what `held` points to must outlive the object.
  HfOnceTable(const HfParams &P, const rf::LayW &Y, uint64_t columns, rf::HfWalk walk, const HfHeld &held, int fibres_a_walk, bool lines, const uint8_t *root, size_t root_bytes,
              blk::IndexRecord record, std::function<bool(uint64_t)> count, blk::EngineGpu &engine)
      : engine_(engine) {
    static_cast<HfShown &>(mode_) = hf_shown(P, columns);  // `side_` reads only tcols, c and `once`
    mode_.once = true;
    side_.P = &P, side_.mode = &mode_, side_.Y = &Y;
    side_.held = held;
    side_.short_lived = [](bool yes) { ::lent::borrow = yes; };
    side_.walk = [walk = std::move(walk)](const rf::HfUnits &units) {  // allocates from lent::own
      const bool before = ::lent::borrow;
      ::lent::borrow = false;
      walk(units);
      ::lent::borrow = before;
    };
    side_.fibres_a_walk = fibres_a_walk, side_.lines = lines;
    seams_ = rf::hf_side_seams(side_);
    seams_.forget = [this, forget = seams_.forget] {
      forget();
      if (std::function<void()> call = std::move(walks_are_over_)) walks_are_over_ = nullptr, call();
    };
    HfCard card;
    card.take = [](size_t bytes) { return from_pool(bytes, true); };
    card.take_lasting = [](size_t bytes) { return from_pool(bytes, false); };
    card.give = [](void *p) { cudaFree(p); };
    card.fetch = [](void *host, const void *from, size_t bytes) {
      CK(cudaMemcpyAsync(host, from, bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
      CK(cudaStreamSynchronize(cudaStreamPerThread));
    };
    card.store = [](void *to, const void *host, size_t bytes) {
      CK(cudaMemcpyAsync(to, host, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
      CK(cudaStreamSynchronize(cudaStreamPerThread));
    };
    card.adopt = [this](const void *cells, size_t rows, size_t len) {
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      return engine_.adopt(cells, blk::Cell::WORD, blk::E_DEGREE, rows, len);
    };
    card.leave = [this] {
      left_ = engine_.trees_on_card();
      for (blk::CommitId c : left_) engine_.leave_card(c);
    };
    card.moved = [this](blk::TableId t, const void *cells) {
      CK(cudaStreamSynchronize(cudaStreamPerThread));
      engine_.moved(t, cells);
    };
    card.back = [this] {
      for (blk::CommitId c : left_) engine_.back_on_card(c);
      left_.clear();
    };
    keeper_.reset(new HfKeeper(P, mode_, seams_, std::move(card), std::vector<uint8_t>(root, root + root_bytes), std::move(record), std::move(count)));
  }
  HfOnceTable(const HfOnceTable &) = delete;
  HfOnceTable &operator=(const HfOnceTable &) = delete;
  // Zeroes and frees what the callbacks still hold, when the caller returns, also with a failure;
  // not when the process ends by exit() or abort(), as in rf::hf_side_internal(), at a failed CK()
  // or at a failed BLK_REQUIRE(). Not zeroed here: the folded row's buffer, which goes back to
  // lent::borrowed, and `held.u_host`. Their owners zero them.
  ~HfOnceTable() {
    keeper_.reset();
    rf::hf_side_forget(side_);
  }

  blk::IndexHolder &holder() { return *keeper_; }
  // Sets the largest number of drawn positions that one pass serves (0: all), for a caller that
  // learns only in mid-run how much GPU memory is free. Call it before holder().open(). No byte of
  // a file depends on it.
  void fibres_a_walk(int n) { side_.fibres_a_walk = n; }
  // `call` runs once, at the end of holder().forget(), which foldeq_columns_prove()
  // (blocks/iop/foldeq.h) calls right after holder().open(). By then this object has freed all that
  // it took from lent::own, and the folded row and the engine's trees are back in GPU memory. From
  // then on it holds that row in lent::borrowed and nothing else, allocates nothing and makes no
  // pass. So `call` may release whatever served only the passes, lent::own included, for the
  // engine's proximity tests and the hash chains that follow; where two tables share the pools
  // (make_entry()), only the later of their two calls may. It must zero lent::own first: omega,
  // which depends on the weights, was freed there without zeroing. If the run ends before forget(),
  // `call` never runs.
  void when_the_walks_are_over(std::function<void()> call) { walks_are_over_ = std::move(call); }
  // The code of the self-check on the drawn columns or on the rows' pads that failed
  // (hf::OWN_GOOD: none); forget() keeps it. A failed self-check on the folded masks is not
  // recorded here; it raises the failure flag OWN_TEST (prover/common/slot_flags.h).
  int failed() const { return side_.failed; }

  // Upper bound on what this object takes from lent::borrowed: the folded row as its oracle message
  // holds it (`rows` x `len` elements of the extension field E, blk::E) and one chunk of `window`
  // elements.
  static size_t lent_bytes(size_t rows, size_t len, size_t window = (size_t)1 << 22) { return hcp::up512(rows * len * sizeof(blk::E)) + hcp::up512(window * sizeof(blk::E)); }
  // What it takes from lent::own besides rf::hf_side_slab_bytes(): omega, one element of E for each
  // row of W.
  static size_t own_bytes(const HfParams &P) { return hcp::up512((size_t)P.m0 * sizeof(blk::E)); }

 private:
  static void *from_pool(size_t bytes, bool borrowed) {
    void *p = nullptr;
    const bool before = ::lent::borrow;
    ::lent::borrow = borrowed;
    CK(cudaMalloc(&p, bytes));
    ::lent::borrow = before;
    return p;
  }
  blk::EngineGpu &engine_;
  HfOpen mode_;
  rf::HfSide side_;
  HfSeams seams_;
  std::vector<blk::CommitId> left_;
  std::function<void()> walks_are_over_;
  std::unique_ptr<HfKeeper> keeper_;
};

}  // namespace hf
