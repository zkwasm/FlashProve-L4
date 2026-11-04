// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Interface of prover/hash_commitment/hf_keeper.cu (`keeper` in identifiers and messages), the only
// translation unit that handles the contents of a weight table T (one of a commitment's leaf
// tables, not a Merkle leaf; `leaf` in identifiers) during a batch's opening (LIN, the test on the
// weight table). It keeps the table's mask seed (its making key, from which masks, salts and pads
// derive: prover/hash_commitment/hf_mask.cuh, hf_pads.cuh), the stored nodes of its Merkle tree
// and the folded row u (T's rows combined at a challenge), and makes the passes over the weight
// matrix W (`walk` in identifiers). It is linked into the prover program cp_prove. Neither a hash
// chain (a Fiat-Shamir transcript) nor a statement passes through this interface, and the caller's
// engine is reached only through HfKeeperHooks; the protocol code sees a blk::IndexHolder
// (blocks/iop/foldeq.h). Order of calls and GPU memory: see HfKeeperUnit.
#pragma once
#include <cuda_runtime.h>
#include <atomic>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "hash_commitment/hf_keeper.h"

namespace hf {

// A batch's callbacks. The engine (blk::EngineGpu, prover/engine/engine_gpu.cuh) is the caller's
// object that holds a batch's tables on the GPU; `card` is a pointer into GPU memory. leaf_begin()
// requires `count`. Set `leave`, `moved` and `back` all three or none: this is not checked.
struct HfKeeperHooks {
  // Counts `columns` more opened columns of the leaf table; false: the table's masks do not cover
  // them. holder().begin() calls it. The hook's supplier must ensure, and nothing here
  // checks, that the count is written, so that every later reader sees it, before this call
  // returns, and is durable (on disk, or wherever the caller keeps it) before the opening thread
  // starts. A batch whose opening was begun counts in full.
  // In cp_prove, count_a_batch() (prover/common/cp_files.h) raises the count in the table's
  // `count.bin`, under a lock, before the batch reads any other file of the table, and the hook
  // returns true once for that batch; fold_begin() (prover/prover/fold_walk.cuh) ends the process
  // unless the count's sync has succeeded.
  std::function<bool(uint64_t columns)> count;
  std::function<blk::TableId(const void *card, size_t rows, size_t len)> adopt;  // the engine adopts the folded row where it is in GPU memory
  std::function<void(blk::TableId, const void *card)> moved;                     // the adopted row's new address after the second pass
  std::function<void()> leave, back;  // the engine releases, then takes again, all GPU memory that it holds for this opening; cp_prove, the one caller, sets both to do nothing
};

// Order of the calls on an instance. hf_keeper_unit() and plan(), once. Then, for each batch:
//   1. leaf_begin(); blk::lin_begin(unit.holder(), ..); the requests' first chains (a request's
//      hash chain before the batch's shared chain; its second chain follows that one);
//   2. lend_slab() (the slab: GPU memory that the instance keeps until leaf_end()) and lend();
//   3. the opening thread: hf_open_lin(.., unit.holder(), ..) (hf_open_lin.h), opening_is_over();
//      meanwhile the service thread (the caller's thread that loads the weights): the loop below;
//   4. the requests' second chains; leaf_end(). take_back() comes before or after step 4.
// plan(), lend_slab(), leaf_begin() and the add_*() calls return 0, or 2 with a line on stderr.
// Out of order, holder(), go_on() and abandon() fail a BLK_REQUIRE guard, which ends the process;
// so does leaf_end() inside a pass.
// leaf_begin() and blk::lin_begin() take no GPU memory, so the slab's memory can serve the first
// chains and be lent after them, but before the opening thread starts: lend_slab() rejects a late
// call only once the first pass has begun.
//
// Two threads. In hf_open_lin(), holder().fold_table() and holder().open() each need a pass over
// all of W, and only the service thread can load the weights. So these calls block while the
// service thread runs:
//     while (unit.next_walk()) {  // blocks until the opening thread waits in a pass;
//                                 // false: opening_is_over() was called
//       .. unit.add_*(..) for every weight unit, each once, in any order, on one stream ..
//       unit.add_rest(stream);    // ends this pass; with HfKeeperSetup::generator (synthetic
//                                 // weights) it first adds the units not yet added
//       unit.go_on();             // the opening thread continues
//     }
// An opening has walks() passes, one after the other: what the second computes depends on
// challenges drawn after the first.
//
// GPU memory (hf_walk.h). When a pass begins, the opening holds nothing in memory lent by lend():
// the service thread takes it back to load the weights. What outlives a pass is in the slab
// (lend_slab()) or in host memory. HfKeeperHooks::leave is called before the second pass, `moved`
// and `back` after it.
class HfKeeperUnit {
 public:
  virtual ~HfKeeperUnit() = default;

  // ---- what follows from the parameter file
  virtual const HfParams &parameters() const = 0;
  virtual int walks() const = 0;
  // A weight unit: a run of rows of W: one kind of weight of one layer, the embedding, or the head.
  struct Unit {
    int kind, layer;  // as add_* name them
    size_t rows;      // of W
  };
  virtual std::vector<Unit> units() const = 0;  // the layout's units: a walk adds each of them once
  // Comes before a size is asked for and before leaf_begin(). shown: hf_shown() (hf_keeper.h) of
  // the number of columns that an opening draws. folded_row_bytes: of the table that the engine
  // adopts (rows x length x 24).
  virtual int plan(const HfShown &shown, size_t folded_row_bytes) = 0;

  // ---- GPU memory: lent pieces, the slab, the walk room
  // The walk room is the lent memory that the add_*() calls need. Nothing in it lives from one call
  // to the next, so it may be taken back and lent again between any two of them. lend() ignores
  // memory that is null or not 512-byte aligned, with a line on stderr. take_back() ends the
  // process if the opening still has a buffer in lent memory
  // (prover/weights_commitment/lent_memory.h).
  virtual void lend(void *gpu_memory, size_t bytes) = 0;
  virtual void take_back() = 0;
  virtual size_t slab_bytes() const = 0;
  virtual size_t walk_room_bytes() const = 0;
  virtual int lend_slab(void *gpu_memory, size_t bytes) = 0;
  // The most lent memory that the instance needs at a time, and its longest single piece: in a pass
  // the walk room (with the generator also the cells of the largest weight unit), between and after
  // the passes the folded row. The engine's own needs in an opening are not in these numbers.
  virtual size_t lent_peak_bytes() const = 0;
  virtual size_t lent_longest_bytes() const = 0;

  // ---- the leaf table against which a batch is proved
  // table_root: the 32-byte root of the table's tree, a stored tree as blocks/ defines it: the
  // stored nodes are checked against it, and it is the table's commitment in first queries (the
  // hash input that starts a hash chain). hooks come with each batch because each batch has its own
  // engine. holder().open() writes the opened columns as a blk::Opening (blocks/tree/opening.h),
  // the form that blk::IndexKey reads.
  // record: the leaf table's record as its leaf file has it, which holder().record() returns.
  // leaf_end() leaves the stored nodes (digests: see forget()) in host memory until the next leaf
  // table's are read over them.
  virtual int leaf_begin(const uint8_t *table_root, size_t root_bytes, const blk::IndexRecord &record, const uint8_t *mask_seed, const char *nodes_file, HfKeeperHooks hooks) = 0;
  virtual blk::IndexHolder &holder() = 0;  // from leaf_begin to leaf_end
  virtual void leaf_end() = 0;  // zeroes both seed copies, u, lent GPU memory; not the stored nodes

  // ---- the service thread's side of a pass
  // X: the weight unit's kind. Without the generator add_rest() returns 2 if a unit is missing: add
  // it and call again, or call abandon(), which takes the place of go_on().
  virtual bool next_walk() = 0;
  // type: the blocks' type as in weights_blocks/decode.cuh (0 Q4_K, 1 Q5_K, 4 Q8_0, 5 F16, 6 BF16); any
  // other number is refused.
  virtual int add_blocks(int X, int layer, const void *blocks, int type, const float *gains, float two_e, cudaStream_t stream) = 0;
  virtual int add_planes(int X, int layer, int sources, const int8_t *const *low, const int8_t *const *high, const size_t *pitch, cudaStream_t stream) = 0;
  virtual int add_embedding(const int8_t *bytes, cudaStream_t stream) = 0;
  virtual int add_head(const short *weights, cudaStream_t stream) = 0;
  virtual int add_rest(cudaStream_t stream) = 0;
  virtual void go_on() = 0;
  // Called by the opening thread when hf_open_lin() has returned, also when it has failed. It gives
  // back the folded row's lent memory: from here on the opening holds nothing lent, and take_back()
  // may come.
  virtual void opening_is_over() = 0;
  virtual void abandon() = 0;  // service thread in a pass: the opening thread's blocked call throws

  // From opening_is_over() to leaf_end(): null, or the name, a constant string, of a failure that
  // comes from what the prover stores and not from chance: "hc-nodes-root" (the file of stored
  // nodes cannot be read or does not hash up to the table's root; a thread reads it in the
  // background, so leaf_begin()'s return value does not cover it) or "hc-own-node" (a stored node
  // differs from the node recomputed from the weights and the mask seed in use).
  virtual const char *keeps_fault() const = 0;
  // From opening_is_over() to leaf_end(): null, or the name, a constant string, of the self-check
  // on secret random values that has failed: "own-test-fold" (the masks' part of the folded row),
  // "own-test-salt" (a salt of a recomputed tree leaf), "own-test-key-copy" (the mask seed's second
  // copy, from which both are derived again, is missing or all zeros), "own-test-pads" (a pad of a
  // row; also a second copy that is missing or all zeros)
  // or "own-test-masks" (a result that the opening's code does not produce). The checks
  // compare masks, salts or pads with a second derivation and read no weight.
  // The service thread asks for the name here and sets it for the program's last line
  // (prover/common/end_name.h). The opening thread must not set it: at an exit of the process the
  // caller's forget() (left out in one case: see there) zeroes the second copy while that thread
  // may still run, so it can find zeros, and the last line would name a self-check that has not
  // failed.
  virtual const char *own_test_fault() const = 0;
  // Wall-clock seconds, for the operator's log line: reading the file of stored nodes, recomputing
  // the levels above them, and the second pass's wait for the thread that does both.
  virtual void loader_seconds(double *read, double *rebuild, double *waited) const = 0;
  // Zeroes only the mask seed's two copies, u, the slab (a pass's sums are there) and the lent
  // memory (in a pass, copies of weights): not the stored nodes, and not what the objects of an
  // opening in progress hold in host memory, such as the opened columns' words.
  // The stored nodes are digests of the Merkle tree, whose leaves are salted; in cp_prove they come
  // from a file in the secret key directory (`isk`), and an opening's Merkle paths publish some of
  // them.
  // Meant for an exit of the process; call it only between two other calls of the instance. If an
  // exit begins, on whichever thread, while the service thread is inside one of the instance's
  // calls, cp_prove leaves this call out (Prover::setup_opening(),
  // prover/prover/opening_in_process.cuh): the instance then zeroes nothing of the above, and that
  // exit's wipe counts as incomplete.
  // Returns 0, or 1 if the GPU part was not done (CUDA does not respond): the caller then tries to
  // zero what is lent, which also fails after a GPU fault, and frees nothing. Afterwards
  // leaf_begin(), lend_slab() and the add_*() calls fail.
  virtual int forget() = 0;

  // Returns and clears the failure flags that the instance's checks have raised since the last call
  // (prover/common/slot_flags.h); after "hc-nodes-root", OWN_TEST stays set until the next
  // leaf_begin(). A message made under a flag must not leave the process.
  virtual unsigned take_flags() = 0;
};

// `configuration` is required: blk::configuration_digest() of the public files, from which the
// hashes of the table's tree start.
struct HfKeeperSetup {
  std::string policy;                          // the parameter file's bytes
  const uint32_t *configuration = nullptr;     // DW 32-bit words; copied at set-up
  bool generator = false, test_run = false;    // synthetic weights; a test run (test builds only)
  int keep_level = 0;                          // tree level stored in a leaf table's file of nodes
  // Each copy of the object file has its own blk::at_a_failed_guard (blocks/base.h) and g_end_name
  // (prover/common/end_name.h). The caller passes its own, so that a guard that fails inside the
  // instance calls the program's handler, and a name set there before exit() reaches the program's
  // last line. end_name null: the copy keeps its own.
  void (*at_a_failed_guard)(const char *name) noexcept = nullptr; // the caller's handler
  std::atomic<const char *> *end_name = nullptr; // the caller's `&g_end_name`
};
// Returns null, with a line on stderr, if the set-up fails or this copy's instance still exists. If
// the mask generator fails its known-answer test, the guard "own-test-generator" ends the process.
std::unique_ptr<HfKeeperUnit> hf_keeper_unit(const HfKeeperSetup &setup);
// Several instances at once (two batches whose openings overlap need two). The translation unit
// keeps state in statics (pools of lent memory, failure flags on the host and on the GPU, hash
// buffers), so one copy of its object file serves one instance at a time. A program that needs more
// links further copies, in which only the name of hf_keeper_unit() differs and no other symbol is
// visible to the linker (prover/build.sh, keeper_unit()). Each copy has that state to itself: a
// fault of one batch neither sets nor clears another's flags. The copies share the GPU and its
// driver, the process's heap, standard output, and the disk, where cp_prove, not a copy, keeps a
// leaf table's count of opened columns: two instances that open one leaf table are each counted in
// its `count.bin` (see HfKeeperHooks::count). A test that is not part of this repository checks
// that two batches' openings of one leaf table, interleaved on two copies, write the same bytes as
// when run in turn, that each batch's columns are counted, and that a fault in one copy sets no
// flag in the other. A second such test checks for separate static variables, on the host and in
// GPU memory, but on a small object file copied by the steps of keeper_unit(), not on the
// translation unit's own. A program that names one of the two functions below must link that copy.
constexpr int HF_KEEPER_COPIES = 3;
std::unique_ptr<HfKeeperUnit> hf_keeper_unit_2(const HfKeeperSetup &setup);
std::unique_ptr<HfKeeperUnit> hf_keeper_unit_3(const HfKeeperSetup &setup);
inline std::unique_ptr<HfKeeperUnit> hf_keeper_unit_of_copy(int copy, const HfKeeperSetup &setup) {  // copy: 0, 1, 2. Whoever calls this links all three.
  BLK_REQUIRE(copy >= 0 && copy < HF_KEEPER_COPIES, "the keeper of the weight table: no such copy of the unit");
  return copy == 0 ? hf_keeper_unit(setup) : copy == 1 ? hf_keeper_unit_2(setup) : hf_keeper_unit_3(setup);
}

}  // namespace hf
