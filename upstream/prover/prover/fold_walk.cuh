// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Drives the shared chain of a batch's opening, the batch's test on the weight table (the weights
// in their committed form). A proof has three hash chains (Fiat-Shamir transcripts): the request's
// first chain, the shared chain (`fold` in identifiers), which all proofs of the batch repeat, and
// the request's second chain. The table owner does the shared chain's work on the weights, in two
// walks (its walks() gives the number); a walk is one pass over all the weights. The table owner is
// the hf::HfKeeperUnit at `closer` (prover/hash_commitment/hf_keeper_unit.h); messages call it "the
// keeper", identifiers and prover/prover/close_memory.cuh "the library".
//
// These functions run on the service's thread, which also proves the requests, and act on the
// current opening, `at_hand` (set by turn_to(), prover/prover/opening_in_process.cuh). fold_begin()
// starts a second thread, the opening's thread, which runs the shared chain. That thread blocks at
// each walk, because only the service's thread can provide the weights and GPU memory. Order of
// calls, after the requests' first chains: fold_begin(), fold_walks(), fold_over(), the requests'
// second chains, batch_end(). For the walks under -overlap see fold_walks().
//
// An int result is 0, 2 (the table owner rejected a call) or 4 (an internal error). A 0 does not
// say that the shared chain succeeded: end_the_batch() (prover/prover/service.cuh) reads that from
// the owner's take_flags().
#pragma once
#include "prover/close_memory.cuh"
// Lends the table owner the memory that it keeps throughout an opening (its `slab`), and the walk
// room, its workspace during a walk. An `InsideTheCloseLibrary` object (opening_in_process.cuh)
// marks a scope that calls the table owner: if the process ends inside one, the wipe at its end
// (`g_end_wipe`) skips that owner's forget() and counts its memory as not zeroed. See also
// docs/limits-all-rows.tsv, row R7.5.
inline int Prover::lend_the_slab() {
  // Waits for all GPU work, including weight loads that are still queued.
  CK(cudaDeviceSynchronize());
  const auto top = slab_for_the_library();
  InsideTheCloseLibrary inside;
  if (!top.first || closer->lend_slab(top.first, top.second)) {
    printf("internal error: the opening: the sub-proofs' slab has %zu bytes mapped, and the keeper "
           "could not be lent the %zu bytes of it that it keeps\n",
           fork_memory.whole_slab().second, closer->slab_bytes());
    return 4;
  }
  return lend_the_walk_room();
}

// One thread at a time holds the batch's generator of private randomness (`coins` in identifiers;
// a blk::CardStreamCoins, prover/engine/stream_coins.cuh) and its engine (blk::EngineGpu, which
// stores, encodes and hashes the chains' tables on the GPU). A draw or a call by another thread
// ends the process. Both are handed over, not taken: the holder's last call is let_go(), which for
// the engine also waits for the GPU work that it has queued; the next thread's first call is
// take_over(), which ends the process if a thread still holds the object (blocks/coins/coins.h,
// prover/engine/engine_gpu.cuh). The opening's thread holds both while it runs the shared chain,
// the service's thread before and after. For a generator of another type these two functions do
// nothing.
static inline void coins_let_go(blk::Coins *coins) {
  if (auto *held = dynamic_cast<blk::CardStreamCoins *>(coins)) held->let_go();
}
static inline void coins_take_over(blk::Coins *coins) {
  if (auto *held = dynamic_cast<blk::CardStreamCoins *>(coins)) held->take_over();
}

// Starts the opening's thread and returns when that thread is blocked at its first walk; from then
// on no request can join the batch (BatchOpening::open(), batch_opening.h). Before the thread
// starts, the_count_is_lasting() (isk.cuh) ends the process unless what the batch has added to its
// leaf table's column count (`count.bin`) is on disk; the leaf table is the weight table that the
// batch is proved against. Then lend_the_slab() waits for the GPU, so that the GPU work of the
// batch's requests is complete.
inline int Prover::fold_begin() {
  the_count_is_lasting(*at_hand); // no opening's thread starts before its batch's count is on disk
  const double began = epoch();
  if (int rc = lend_the_slab()) return rc;
  Opening &o = *at_hand;
  o.abandoned = false;
  o.engine->let_go();   // this thread's last call of the batch's engine until the join
  coins_let_go(o.coins); // and of the batch's generator
  o.thread.thread = std::thread([&o] {
    o.engine->take_over();
    coins_take_over(o.coins); // the batch's generator is this thread's from here to its end: the service's thread draws nothing from it meanwhile
    try {
      o.order->open();
    } catch (...) { // a call that is blocked in a walk throws after `abandon`
      o.abandoned = true;
    }
    coins_let_go(o.coins); // in either case: the thread that joins this one takes both over
    o.engine->let_go();
    // Reads this thread's own CUDA error state after the shared chain's launches. A launch that the
    // driver rejected leaves its error with the launching thread only: the thread that joins this
    // one, which puts the shared chain's bytes into the proof files, cannot see it
    // (prover/docs/cuda-status.md, "The rule").
    CK(cudaGetLastError());
    o.unit->opening_is_over();
  });
  bool walks;
  {
    InsideTheCloseLibrary inside;
    walks = closer->next_walk();
  }
  at_hand->ending.times.fold_begin = epoch() - began;
  if (!walks) {
    printf("internal error: the shared chain has ended before its first walk over the weights\n");
    return 4;
  }
  return 0;
}

// Fetches the table owner's list of units, on the first call only: the list depends only on the
// parameter file, which setup_opening() (opening_in_process.cuh) gives to every opening's table
// owner alike. A unit is one kind of weight of one layer, the embedding or the output head.
inline void Prover::list_the_units() {
  if (!unit_kind.empty()) return;
  InsideTheCloseLibrary inside;
  for (const hf::HfKeeperUnit::Unit &u : closer->units()) unit_kind.push_back(u.kind), unit_layer.push_back(u.layer);
}

// Passes every layer group's units to the table owner as WeightStream::pass() (weight_stream.cuh)
// loads them, then the embedding and the output head. A layer group is `shape.Lam` consecutive
// layers. The last argument of fold_group_for_opening() (opening_in_process.cuh) selects among a
// group's units: -2 all, -1 those that are not routed experts, a layer's number that layer's routed
// experts. Afterwards no group counts as loaded: call weight_stream.restart() before requests are
// proved again.
inline int Prover::fold_units_from_the_stream() {
  list_the_units();
  const std::vector<int> &kind = unit_kind, &layer = unit_layer;
  int rc = 0;
  weight_stream.pass(
      [&](int g) {
        stop_if_asked();
        if (!rc) rc = fold_group_for_opening(g, kind, layer, weight_stream.experts_by_layer ? -1 : -2);
      },
      [&](int routed_layer) {
        stop_if_asked();
        if (rc) return;
        if (routed_layer < 0) { // -memory rest-by-turns: the embedding (-1), then the head (-2), each after its load
          rc = fold_end_for_opening(-1 - routed_layer);
        } else
          rc = fold_group_for_opening(routed_layer / block_weights->shape.Lam, kind, layer, routed_layer);
      });
  if (rc) return rc;
  for (int which = 0; which < 2 && !weight_stream.ends_by_turns; which++)
    if (int r = fold_end_for_opening(which)) return r;
  return 0;
}

// One walk. With generated weights (args.stream_weights is false) nothing is passed: the table
// owner's add_rest() generates each unit itself, in lent memory. What fold_rest() lends stays lent
// until fold_next() or, after the last walk, batch_end(). Call fold_walk() only while the walk room
// is lent, no request is being proved and the GPU is idle.
inline int Prover::fold_walk() {
  if (args.stream_weights)
    if (int r = fold_units_from_the_stream()) return r;
  return fold_rest(false);
}

// Between two walks. Waits until the opening's thread is blocked at its next walk. The table owner
// then uses no lent memory except what it keeps throughout the opening, so the rest is taken back.
// That includes the weight stream's GPU buffers, into which the next walk loads the weights. Then
// the walk room is lent again.
inline int Prover::fold_next() {
  stop_if_asked();
  const double began = epoch();
  bool walks;
  {
    InsideTheCloseLibrary inside;
    walks = closer->next_walk();
  }
  take_back_what_is_lent(); // it waits for the GPU
  at_hand->ending.times.fold_next += epoch() - began;
  if (!walks) {
    printf("internal error: the shared chain has ended before the last of its walks over the weights\n");
    return 4;
  }
  return lend_the_walk_room();
}

// After the last walk: waits until the shared chain has ended and joins the opening's thread.
// Returns 4 if the chain asks for another walk or BatchOpening::open() threw.
inline int Prover::fold_over() {
  bool walks;
  {
    InsideTheCloseLibrary inside;
    walks = closer->next_walk();
  }
  if (walks) {
    printf("internal error: the shared chain asks for more walks over the weights than the keeper has announced\n");
    closer->abandon();
    at_hand->thread.join();
    coins_take_over(at_hand->coins);
    at_hand->engine->take_over();
    return 4;
  }
  at_hand->thread.join();
  coins_take_over(at_hand->coins); // this thread's again, after the join
  at_hand->engine->take_over();       // and so is the batch's engine
  CK(cudaDeviceSynchronize()); // what the opening's thread has launched is complete before this thread reads it
  return at_hand->abandoned ? 4 : 0;
}

// All walks of a batch's opening, after fold_begin(). With -overlap a walk can instead use the
// weights that are loaded for a later batch's requests: the_rider_folds() and
// the_riders_walk_ends() (service.cuh) then pass the units, and the latter calls fold_rest(true).
inline int Prover::fold_walks() {
  int walks;
  {
    InsideTheCloseLibrary inside;
    walks = closer->walks();
  }
  weight_stream.turns_run = false; // the walks' loads go by the copy engine (weight_stream.cuh, load())
  for (int w = 0; w < walks; w++) {
    if (int r = fold_walk()) return r;
    if (w + 1 < walks)
      if (int r = fold_next()) return r;
  }
  weight_stream.turns_run = true;
  return 0; // end_the_batch() waits for the shared chain's end (fold_over())
}

// Ends a walk, when every unit has been passed: lends the table owner the GPU memory that is idle
// at a batch's end and lets the opening's thread continue. in_a_pass (-overlap): the call comes
// while a later batch's requests are being proved. free_at_the_batchs_close() (close_memory.cuh)
// then omits `chain_dying_stretch`; nothing sets it, so the same memory is lent either way.
inline int Prover::fold_rest(bool in_a_pass) {
  CK(cudaDeviceSynchronize());
  stop_if_asked();
  lend_for_the_opening(in_a_pass);
  InsideTheCloseLibrary inside;
  if (int r = closer->add_rest(cudaStreamPerThread)) {
    printf(
        "the opening: the keeper rejected add_rest: a unit of the weights was never passed to it, or "
        "the call came out of order (code %d)\n",
        r);
    return 2;
  }
  closer->go_on();
  return 0;
}

// Ends the batch's opening: frees what the shared chain, if open() has returned, still keeps on the
// engine, has the table owner's leaf_end() zero both its copies of the mask seed, the leaf table's
// root, u (its combination of the table's rows) and all GPU memory lent to it, and takes that
// memory back. If it cannot be zeroed, the process ends. Nothing zeroes the stored nodes of the
// leaf table's Merkle tree: they stay in pinned host memory until a later leaf_begin() loads nodes
// over them. They are digests over salted leaves, read or recomputed from a file that stays in the
// commitment's secret key directory (`isk/leaf/<k>/hcnodes.bin`), and an opening publishes some of
// them. See also docs/limits-all-rows.tsv, row H11.6.1.1. `count` is reset so that no later batch
// passes the_count_is_lasting() on this batch's result. The engine is not freed here:
// end_the_batch() (service.cuh) frees it after it has cleared the slots. batch_end() may be called
// at any point after the table owner's leaf_begin(), also after a failed call, with one exception:
// an opening's thread that has not been joined must be blocked in a walk, or abandon() ends the
// process.
inline int Prover::batch_end() {
  Opening &o = *at_hand;
  if (o.thread.thread.joinable()) { // an end before the shared chain is over
    closer->abandon();
    o.thread.join();
    coins_take_over(o.coins);
    o.engine->take_over();
  }
  if (o.order && o.order->is_open()) o.order->end();
  o.order.reset();
  o.count = cp::CountSync(), o.counted = false; // waits for a sync that is still under way
  {
    InsideTheCloseLibrary inside;
    closer->leaf_end();
    closer->take_back();
  }
  back_from_the_close();
  return 0;
}
