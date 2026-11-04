// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// [own-test-slot-pads] A self-check of the prover program cp_prove: it compares every request's
// slot pads, by value, with a second implementation of their derivation (prover/second_writing/).
//
// A sub-proof (the argument for one part of the forward pass) masks the scalars that it sends with
// its slot pads: the entries of its mask row that it uses. Each must be a private random value (a
// "coin") derived from the batch's process key (a ChaCha20 key). No verifier sees
// whether they are the coins that the batch's key gives: a proof whose pads are zeros verifies like any other, and shows what it
// should hide. The GPU's checks by value (hz::Tree::slot_pads_agree(), prover/hiding/trees.cuh)
// derive the pads again with the functions that derived them, so they miss a fault in those
// functions. The self-check [own-test-coins] runs once, at the program's start, under other keys.
//
// Limits: the second implementation includes no file of the programs, but this file and
// prover/prover/slot_pads_derive.h, which pass it the key, the batch slot and the sub-proof's
// identifier gid and pair the two lists, are part of cp_prove and can share its errors. A
// misreading of the derivation rule that both implementations share is not detected.
// Further limits: prover/second_writing/README.md, "How far it is independent of the programs".
#pragma once
#include "common/end_name.h"
#include "prover/slot_pads_derive.h"
#include <sys/mman.h>

// Terms: a sub-proof's masked scalars are numbered y < slots ("slot" in identifiers and messages).
// Each pad is an element of the extension field E: EDEG (3) cells of 64 bits. The parameter `slot`
// is another thing: the request's index in its batch.
//
// Host memory: one pinned buffer contains the key, from take_the_key() to wipe_the_key(), and
// during a comparison the pads twice: as copied from the GPU and as derived. Outside the buffer,
// the second implementation's functions leave on the stack the cipher's state and the key that is
// derived from the process key for the request's blinding context (a label formed from the version
// word and the batch slot). That key gives all the request's slot pads, and with them every scalar
// that the proof masks with one. cp_prove draws nothing else under that key
// (hz::Tree::slot_pads(), prover/hiding/trees.cuh). What the request's proof draws through blocks/
// has another context's key, under the same process key (chain_coins_of_a_seat(),
// prover/prover/chain_coins.cuh). The cipher's state contains whichever key it ran under: in the
// first of the three steps that derive a context's key, the process key itself, which gives every
// private random value of the batch. Nothing in the second implementation erases that state, so
// the process key can remain on the stack too, unless a later call overwrites it. So right after
// slot_pads_derive() returns, scrub_the_stack_below() (prover/common/stack_scrub.h) zeroes 32 KiB
// of stack below the caller. No test of this repository searches the stack afterwards. Not erased:
// the registers, which keep the entire key of the request's blinding context. Whatever later saves
// registers on this thread's stack (a first call through the dynamic linker, a signal's frame)
// writes that key there.
// What the buffer contains is not erased if the process ends without its exit handler (a signal
// that kills, a loss of power).
// Not thread-safe: every call uses this one buffer, and there is no lock.
struct SlotPadsSecond {
  static constexpr size_t KEY_CELLS = 4; // eight words

  // Allocates the buffer, for sub-proofs of at most `slots` slots. Call it once, at set-up.
  // Returns false if cudaMallocHost() fails. cp_prove's exit handler zeroes the buffer
  // (ForkMemory::wipe()), but only if prover/prover/fork_memory.h, whose macro cudaMallocHost
  // records it, is included before this file. The two madvise() calls ask the kernel to keep the
  // buffer out of core files, which cp_prove also disables, and out of child processes.
  bool set_up(size_t slots) {
    if (!slots) slot_pads_stop("the set-up counts no slot");
    room = EDEG * slots;
    const size_t bytes = 8 * (KEY_CELLS + 2 * room);
    if (cudaMallocHost(&memory, bytes) != cudaSuccess) return false; // on ForkMemory's record, and zeroed
    const size_t page = 4096; // the two `madvise` calls below are hints for whole pages
    char *const first = (char *)((uintptr_t)memory & ~(uintptr_t)(page - 1));
    const size_t length = (size_t)((char *)memory + bytes - first);
    madvise(first, length, MADV_DONTDUMP); // hints, as in weight_stream.cuh: a failure is ignored
    madvise(first, length, MADV_DONTFORK);
    return true;
  }
  // Copies the batch's process key: 32 bytes as hz::key32_renew() takes them, where word i is
  // bytes 4 i to 4 i + 3, the low byte first. Call it at the batch's start, and wipe_the_key()
  // wherever hz::key32_wipe() erases the randomness generator's copy (prover/hiding/trees.cuh).
  // The two copies are separate, so a stale or overwritten key there shows as a difference. Not
  // detected: a process key that is wrongly derived, or the same in two batches. Both copies are
  // made from one derived key (Prover::begin_the_batch_on_a_leaf(), prover/prover/isk.cuh), so
  // every comparison then passes.
  void take_the_key(const uint8_t *process_key) {
    if (!memory) slot_pads_stop("the key comes before the buffer");
    memcpy(memory, process_key, 8 * KEY_CELLS);
    key_held = true;
  }
  void wipe_the_key() {
    if (memory) explicit_bzero(memory, 8 * KEY_CELLS);
    key_held = false;
  }

  // Compares the `slots` elements at `pads` in GPU memory (EDEG consecutive cells each, read on
  // the calling thread's stream) with the slot pads of sub-proof gid of the request in batch slot
  // `slot`. Call it right before the sub-proof's closing (hd::close(), prover/hiding/door.cuh),
  // after the pads' last use. Sub-proofs call it through ch::masking_pads_hold()
  // (prover/hiding/moments.cuh), which exits if nothing is bound to it.
  // gid is the sub-proof's public identifier (`ChainLink::gid`, prover/prover/chain_host.cuh), not
  // the number of the randomness stream that the pads came from. `version` (the parameter file's
  // version word) and `slot` select the request's blinding context, which the second
  // implementation computes itself (coins::request_proof_context()).
  //
  // Returns the number of pairs of cells read. `equal` becomes false if a pair differs or
  // slot_pads_derive() fails. The caller must fail the request if `equal` is false, or if at the
  // request's end a sub-proof was not compared exactly once or the returned numbers do not sum to
  // the number expected for all its sub-proofs (Prover::chain_bind(),
  // prover/prover/chain_frame.cuh; Prover::finish_request(), prover/prover/request.cuh).
  //
  // After a failed check the request is proved to its end like any other; cp_prove takes no further
  // batch and exits at the next end of a batch. A response that has no final file name when the
  // failing request ends gets none from this process (`self_check_failed`, prover/prover/prover.h):
  // a machine or a program that has lost one random value releases no further proof. Until then an
  // earlier batch's responses can still get their final names (docs/limits-all-rows.tsv, row C3).
  // Exception: if an earlier batch's responses are getting their final names at that moment, all of
  // them get one, because the flag that Spool::shut() sets (prover/prover/spool.h) is read once for
  // a batch.
  // Limit: the cause can be an overwrite of the pads that depends on hidden data, and then the fact
  // that the service ended reveals something about weights or tokens. It is not ruled out that a
  // kernel writes into the pads' array at an index that depends on data.
  //
  // Not detected: a correct pad that the kernels do not add, add to two scalars or read before it
  // is written (the check reads the array once, after its last use); the host's copy of the pads,
  // which ChainRun::first() fetches from this array and commits to (prover/prover/chain_run.cuh); a
  // wrong batch slot, version word or number of slots, which the prover's derivation shares with
  // this check; a gid that is wrong both in `ChainLink::gid` and in the sub-proof's stream numbers;
  // any private random value other than the slot pads (random tails, the rest of the mask rows,
  // salts, those of the batch's opening and of the committer program cp_commit); a bug in this file
  // that no test exercises.
  size_t hold(const void *pads, size_t slots, uint32_t gid, uint32_t version, uint32_t slot, bool &equal) {
    const size_t cells = EDEG * slots;
    if (!key_held) slot_pads_stop("a comparison while no batch's key is held");
    if (!slots) slot_pads_stop("a sub-proof without a slot");
    if (cells > room) slot_pads_stop("a sub-proof has more slots than the set-up counted");
    u64 *const came = memory + KEY_CELLS, *const want = came + room;
    memset(came, 0xFF, 8 * cells); // no cell is all ones, so a copy that does not arrive differs
    memset(want, 0, 8 * cells);
    CK(cudaMemcpyAsync(came, pads, 8 * cells, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    const bool derived = slot_pads_derive(memory, version, slot, gid, slots, want); // want[kappa * slots + y]
    scrub_the_stack_below();
    u64 difference = 0; // every pair is read, whatever it holds
    size_t read = 0;
    for (size_t y = 0; y < slots; y++)
      for (size_t kappa = 0; kappa < EDEG; kappa++) difference |= came[EDEG * y + kappa] ^ want[kappa * slots + y], read++;
    explicit_bzero(came, 8 * cells);
    explicit_bzero(want, 8 * cells);
    equal = derived && !difference;
    return read;
  }

private:
  u64 *memory = nullptr; // the key's eight words, then 'room' cells twice
  size_t room = 0;
  bool key_held = false;

  // Reports an error of the calling code, which depends on no hidden value. Because of the name set
  // here cp_prove exits with code 9, not 4, as after a failed self-check, unless a name was already
  // set: stop() in prover/prover/ends.h keeps the first one. exit() goes through cp_prove's exit
  // handler (end_on_exit()), which runs the wipe, within the bounds that prover/prover/ends.h tells.
  [[noreturn]] void slot_pads_stop(const char *what) {
    if (memory) explicit_bzero(memory, 8 * (KEY_CELLS + 2 * room));
    key_held = false;
    printf("internal error (own-test-slot-pads): %s\n", what);
    name_the_end("own-test-slot-pads");
    exit(4);
  }
};
static SlotPadsSecond slot_pads_second;
