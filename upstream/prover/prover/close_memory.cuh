// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Lends the GPU memory that is idle at a batch's end ("close" in identifiers and output) to the
// opening library (`closer`, an hf::HfKeeperUnit): the code, linked into the prover program
// `cp_prove`, that owns the weight table (all the weights, encoded, under a Merkle tree). The
// library works in that memory during the batch's opening: the test on the weight table, which
// follows the batch's last pass and needs more GPU memory than is otherwise free. A pass proves up
// to -in-step requests on one transfer of the weights to the GPU. No sub-proof (`fork` in
// identifiers) runs during an opening, except with -overlap, under which an opening can run during
// later batches' passes.
//
// The library zeroes each lent piece on receiving it and, after waiting for the GPU, before
// returning it (prover/weights_commitment/lent_memory.h). The passes must not rely on those zeros:
// they must zero or write whatever they read, as they must after another request. The pieces'
// addresses and lengths are meant to be fixed at set-up, by the parameter file and the options, and
// to depend on no token and no weight. But the length of the slab (the sub-proofs' scratch memory)
// is what is mapped at the call, and it has not been shown that no buffer's size depends on a
// request's values (prover/prover/fork_memory.h, note_what_is_taken()).
#pragma once
#include "prover/opening_in_process.cuh"

// The GPU must be idle at the call. Returns the idle pieces:
// - `chain_dying_stretch` ("the chain library's" in output): nothing sets it, so it is empty;
// - the rooms, the weight stream's GPU buffers, which then count as empty until
//   WeightStream::restart() or pass() loads them again;
// - the slab below the part that the library keeps throughout the opening (slab_without_it()),
//   without its upper end, the walk room, if lend_the_walk_room() has lent that. The library works
//   in the walk room during the opening's walks. In a walk the opening reads every weight once.
// Each piece starts at a multiple of 512 bytes, as the library's lend() requires. Contiguous
// neighbours are merged, so that rooms placed side by side
// (WeightStream::lay_the_rooms_back_to_back()) give one long piece.
// during_the_walk: omits the rooms and the slab, which hold weights during a walk. No caller passes
// true.
// in_a_pass: true only with -overlap. The call then comes between two turns of a later batch's
// pass, after its last layer group; a layer group is `Lam` consecutive layers, and a turn is one
// request's work on one layer group or, last, on the output head. Omits only
// `chain_dying_stretch`; as that is empty, the flag has no effect. The rooms then hold no weight
// that is still needed, and the slab is unused, as between any two turns under -overlap
// (prover/prover/in_step.cuh).
// Public sources for lending the rooms: see prover/prover/weight_stream.cuh, room().
inline std::vector<std::pair<char *, size_t>> Prover::free_at_the_batchs_close(bool during_the_walk, bool in_a_pass) {
  std::vector<std::pair<char *, size_t>> pieces;
  auto add = [&](char *first, size_t bytes) {
    char *const from = (char *)(((uintptr_t)first + 511) & ~(uintptr_t)511);
    if (!first || bytes <= (size_t)(from - first) + 511) return;
    const size_t whole = (bytes - (from - first)) & ~(size_t)511;
    if (!pieces.empty() && pieces.back().first + pieces.back().second == from)
      pieces.back().second += whole;
    else
      pieces.push_back({from, whole});
  };
  if (!in_a_pass) add(chain_dying_stretch.first, chain_dying_stretch.second);
  if (during_the_walk) return pieces;
  for (int r = 0; r < WeightStream::n_rooms(); r++) {
    const auto room = weight_stream.room(r);
    add(room.first, room.second);
  }
  const auto slab = slab_without_it();
  add(slab.first, walk_room_is_lent ? (size_t)(walk_room().first - slab.first) : slab.second);
  return pieces;
}

// Maps more of the slab at set-up if what will be lent at a batch's end is less than the opening
// needs (lent_peak_bytes() in all, lent_longest_bytes() in one piece:
// prover/hash_commitment/hf_keeper_unit.h), so that nothing is mapped at a batch's end. The slab
// must also hold the walk room and, below it, the largest unit of the weights that
// fold_group_for_opening() decodes during a walk (a unit: one kind of weight of one layer). Call it
// after the set-up of the sub-proofs and the weight stream, and after set_up_the_shared_chain(),
// before which the library's sizes are 0. Only the first call after setup_opening() has an effect.
// If the slab cannot grow, the program exits with code 1 (ForkMemory::map_up_to()). This function
// does not check that enough is mapped afterwards: ForkMemory::map_at_least() caps the request at
// the slab's reserved addresses.
inline void Prover::hold_the_close_to_its_room() {
  if (close_has_its_room || !g_opening_library_with_secrets[0]) return;
  close_has_its_room = true;
  // walk: the walk room, with slack for its alignment. An `InsideTheCloseLibrary`
  // (opening_in_process.cuh) marks a call into the library: stop() (ends.h), if reached during one,
  // does not call forget() on the library's copy in use, whose memory then counts as not zeroed.
  size_t peak, one_block, walk, claims;
  {
    InsideTheCloseLibrary inside;
    claims = 0; // nothing lent at a request's end: its part of the test runs on `Opening::engine`
    peak = closer->lent_peak_bytes();
    one_block = closer->lent_longest_bytes();
    walk = ((closer->walk_room_bytes() + 511) & ~(size_t)511) + 1024;
  }
  // With -overlap, what each copy of the library keeps throughout an opening, its own "slab", is a
  // separate buffer, not part of the sub-proofs' slab. It is allocated here so that a GPU without
  // memory for it fails before the first request. It is in ForkMemory's record, so stop() zeroes it
  // at the process's end (ForkMemory::wipe(); not after a GPU fault). It is in `laid_once`, so the
  // restoring of GPU memory between turns (memory_as_set_up.cuh) skips it: while it is lent it is
  // the library's, which zeroes it on receiving it and before returning it
  // (docs/limits-all-rows.tsv, row H11.4).
  // `chain_dying_stretch` is then not counted below: in_a_pass omits it.
  for (int i = 0; args.overlap && i < openings; i++) {
    size_t bytes;
    {
      InsideTheCloseLibrary inside;
      bytes = ((closer->slab_bytes() + 511) & ~(size_t)511) + 512;
    }
    char *buffer = nullptr;
    fork_memory.purpose = "the opening library's slab";
    CK(cudaMalloc(&buffer, bytes));
    fork_memory.purpose = "other buffers of the prover";
    fork_memory.laid_once.push_back({buffer, bytes});
    char *const first = (char *)(((uintptr_t)buffer + 511) & ~(uintptr_t)511);
    opening[i].slab = {first, (size_t)(buffer + bytes - first) & ~(size_t)511};
  }
  const size_t chains = args.overlap ? 0 : chain_dying_stretch.second;
  size_t rooms = 0, longest = chains;
  for (int r = 0; r < WeightStream::n_rooms(); r++) {
    const size_t bytes = weight_stream.room_bytes(r);
    rooms += bytes;
  }
  longest = std::max(longest, weight_stream.longest_stretch_of_rooms());
  const size_t beside_the_slab = rooms + chains;
  size_t kept = 0;
  if (!args.overlap) {
    InsideTheCloseLibrary inside;
    kept = closer->slab_bytes();
  }
  // at_most: what could be lent with all free GPU memory except 1 GiB mapped into the slab. "The
  // keeper" in the message is the library.
  {
    size_t free_now = 0, all = 0;
    CK(cudaMemGetInfo(&free_now, &all));
    const size_t spare = (size_t)1 << 30;
    const size_t at_most = beside_the_slab + slab_without_it().second + (free_now > spare ? free_now - spare : 0);
    if (peak > at_most) // only a warning: the library has no smaller plan
      printf("# the close: the keeper needs %.3f GB of lent memory at a walk's end, and at most %.3f GB can be lent\n", peak / 1e9, at_most / 1e9);
  }
  // The walk room is lent as a separate piece: the opening's longest buffer must fit below it,
  // unless a contiguous range of rooms holds it.
  fork_memory.map_at_least(kept + std::max({longest >= one_block ? (size_t)0 : one_block + walk,
                                            peak > beside_the_slab ? peak - beside_the_slab : (size_t)0,
                                            walk + decoding_needs_at_most(), claims}));
  printf("# the close: lent to the opening: the chain library's %.3f GB, the rooms' %.3f GB, the slab's %.3f GB below "
         "the %.3f GB that the opening library keeps: %.3f GB in all, of which the opening needs %.3f GB, and %.3f GB "
         "in one piece\n",
         chains / 1e9, rooms / 1e9, slab_without_it().second / 1e9, kept / 1e9,
         (beside_the_slab + slab_without_it().second) / 1e9, peak / 1e9, one_block / 1e9);
}

// Lends the opening library the walk room, until the library's take_back(). Returns 0, or 4 if the
// slab is too small. The walk room must not be lent already: neither this function nor the library
// checks that.
inline int Prover::lend_the_walk_room() {
  const auto room = walk_room();
  if (!room.first) {
    printf("internal error: the opening: the sub-proofs' slab has no room for the opening library's walk room\n");
    return 4;
  }
  lent_at_the_close.push_back(room);
  walk_room_is_lent = true;
  InsideTheCloseLibrary inside;
  closer->lend(room.first, room.second);
  return 0;
}

// Lends the opening library the pieces of free_at_the_batchs_close(), until the library's
// take_back().
inline void Prover::lend_for_the_opening(bool in_a_pass) {
  // Lending also zeroes the weights that were decoded into the slab.
  const auto more = free_at_the_batchs_close(false, in_a_pass);
  lent_at_the_close.insert(lent_at_the_close.end(), more.begin(), more.end());
  InsideTheCloseLibrary inside;
  for (auto &piece : more) closer->lend(piece.first, piece.second);
}

// Takes back every piece lent through the library's lend(), but not what the library keeps
// throughout the opening. If a piece still holds a buffer of a running opening, the library calls
// exit(4), so `cp_prove` exits with code 1 (ends.h); otherwise the library first waits for the GPU
// and zeroes the pieces. At a batch's end batch_end() does this instead (fold_walk.cuh).
inline void Prover::take_back_what_is_lent() {
  {
    InsideTheCloseLibrary inside;
    closer->take_back();
  }
  back_from_the_close();
}

// Call it after the library's take_back(): from then on the passes may use the pieces again. In a
// test build with LENT_BACK=scribbled in the environment, it first overwrites every piece in
// `lent_at_the_close` with 0xEE. If the run then writes the same files as a run without it, the
// passes did not rely on the returned zeros in that run. A test that is not part of this repository
// makes that comparison; it sets the variable only in runs with -overlap or -overlap-two
// (docs/limits-all-rows.tsv, row H11.6.1).
inline void Prover::back_from_the_close() {
  const char *test = TESTENV("LENT_BACK");
  if (test && !strcmp(test, "scribbled")) {
    size_t bytes = 0;
    for (auto &piece : lent_at_the_close) {
      CK(cudaMemset(piece.first, 0xEE, piece.second));
      bytes += piece.second;
    }
    CK(cudaDeviceSynchronize());
    printf("test mode: %zu pieces, %.3f GB, came back from the opening library and were overwritten with 0xEE\n",
           lent_at_the_close.size(), bytes / 1e9);
  }
  lent_at_the_close.clear();
  walk_room_is_lent = false;
}
