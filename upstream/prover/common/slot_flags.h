// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The bits of a request's failure word ("flags" in identifiers). A check that fails while a request
// is proved sets a bit here and stops nothing: the work continues, with a harmless substitute
// value where the check has one. With any bit set, the prover program cp_prove releases the same
// empty file <id>.failed instead of the request's proof file (<id>.proof) and token file (<id>.y).
// This is meant to keep other requests' files from showing that a check failed, where or why. The
// running time of the batch (the requests proved together) may show that one did
// (docs/what-is-not-hidden.md, "What a client learns").
//
// The variables and functions here are static, so each translation unit has its own set: in
// cp_prove, the main unit and each linked copy of the unit that owns the weight table
// (prover/hash_commitment/hf_keeper.cu).
#pragma once
#include <atomic>
#include <cuda_runtime.h>

// The committer program cp_commit prints its failure word in decimal and treats any bit as a
// failure (hf_once_flags() in prover/hash_commitment/hf_commit_once.cuh).
//
// A draw derives a challenge, a field element, from hash output
// (blocks/chain/chain.h). The approximate range check requires every entry of its
// combined vectors (random combinations of committed columns) to be in an interval, the window
// (prover/hiding/tail.cuh). No overflow of the counts (multiplicities) in its lookups is reported:
// forest_plan() (prover/hiding/forest_plan.h) rejects a group of lookups with 2^32 keys or more, so
// that 32-bit counts suffice (docs/limits-all-rows.tsv, row E24). OWN_TEST need not mean a fault of
// the program or the machine: a forward-pass value outside its range (UNFIT) can also fail the
// self-check in k_tie() (prover/hiding/lookup_tie.cuh). A proof-of-work search that finds no nonce
// sets no bit: blk::EngineGpu::search_nonce() (prover/engine/engine_gpu.cuh) ends the process.
enum SlotFlag : unsigned {
  NO_ELEMENT = 1,           // no draw raises it: test switches and an unlaunched kernel do (element_from_hash.h)
  // a value that the verifier requires to be non-zero is zero (kh_gam(), hiding/tail_gam.cuh; see element_from_hash.h)
  ZERO_ELEMENT = 2,
  WINDOW = 4,               // a combined vector's entry left its window
  INDEX = 8,                // an index was outside its table (prover/common/index_guard.cuh)
  OWN_TEST = 16,            // a self-check of the prover failed
  UNFIT = 32,               // a forward-pass value, or in cp_commit a lookup key, left its range
  // The request's R checked tokens (those that the proof is about) exceed a bound set by the lines
  // `C`, `theta` and `cap` of the parameter file `policy.txt`: their costs under the acceptance
  // policy, each clipped at `C`, sum to more than `theta` * R, or more than `cap` of them exceed
  // `C`. This is a result, not a fault. k_head_budget() (prover/head/forward.cuh), which finds it,
  // sets no other bit. The prover has no self-check on the two bounds.
  ABOVE = 64,
  FLAGS_UNREADABLE = 1u << 31 // the GPU's words could not be read or cleared; counts as a failure
};

// The live words: the GPU's words 0 and 1, and the host's word. Code that sets a bit need not know
// which request it works for: the code that switches between requests replaces a live word by one
// atomic exchange, so that no bit is lost. MemberState (prover/prover/member_state.cuh) exchanges
// the GPU's two in stream order, before and after every turn (a request's set-up, or its work on
// one layer group or the output head). enter_slot() and leave_slot() (prover/prover/slot.h)
// exchange the host's word. A kernel's bit is sure to reach its request's word only if the kernel's
// stream has joined the stream of the exchange before the word is taken: a later bit counts as the
// next request's or as no request's. MemberState joins only the streams given to
// side_streams::enter() (prover/common/side_streams.h).
//
// The GPU's words:
//   0     the failure bits
//   1     which uses of an index found it outside its table (the bits of
//         prover/common/index_guard.cuh); only tests read it (indexes_outside() in that file)
//   2, 3  the same two, for bits set between two requests
// Public source: compare POSIX signals (Linux man-pages, signal(7), "Signal mask and pending
// signals"): process-wide and per-thread pending sets. Here no sender names a target: a bit counts
// as no request's if the exchange that installs a request's word finds it, and it stays in what
// live_flags() and indexes_outside() return until the words are cleared; a request's saved word is
// read without it. Such a bit fails the entire batch.
static __device__ __align__(16) unsigned g_live_flags_device[4] = {0, 0, 0, 0};
static std::atomic<unsigned> g_live_flags_host{0};

// ORs bits into the live word of the calling side, host or GPU. It writes even if bits is 0, so
// that code on a path that must not reveal its outcome can call it without a branch. (Not named
// raise(): with an int argument that would silently call the C library's raise(int), which sends a
// signal.)
static __host__ __device__ __forceinline__ void raise_flags(unsigned bits) {
#ifdef __CUDA_ARCH__
  atomicOr(&g_live_flags_device[0], bits);
#else
  g_live_flags_host.fetch_or(bits);
#endif
}

// Returns the OR of this translation unit's failure words, the host's and the GPU's words 0 and 2,
// and clears nothing. It waits for the entire GPU first, so that no kernel in
// any stream can still set a bit for the work queued so far.
// Public source for comparison: see g_live_flags_device above.
static inline unsigned live_flags() {
  unsigned device[4] = {0, 0, 0, 0};
  const bool read = cudaDeviceSynchronize() == cudaSuccess &&
                    cudaMemcpyFromSymbol(device, g_live_flags_device, sizeof device) == cudaSuccess;
  return device[0] | device[2] | g_live_flags_host.load() | (read ? 0 : FLAGS_UNREADABLE);
}
// Returns live_flags() and clears all the words. For code that keeps no word per request (such as
// cp_commit and the weight table's unit) and, at a batch's end in cp_prove, for the bits of no
// request. A bit that a kernel sets between the read and the clearing is lost.
static inline unsigned take_live_flags() {
  unsigned bits = live_flags();
  const unsigned zero[4] = {0, 0, 0, 0};
  if (cudaMemcpyToSymbol(g_live_flags_device, zero, sizeof zero) != cudaSuccess) bits |= FLAGS_UNREADABLE;
  return bits | g_live_flags_host.exchange(0);
}
