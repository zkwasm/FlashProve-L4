// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A batch of requests in the prover program cp_prove, and the batch's slots, one for each request
// (`member` in identifiers). A request's proof has three hash chains (Fiat-Shamir transcripts):
// its first chain, ended by a digest, the seal; the shared chain, the same in every proof of the
// batch; and its second chain.
// A request's files are released only if its slot's failure word (`flags` in identifiers; bits:
// prover/common/slot_flags.h) is 0 and nothing has failed the whole batch. No other slot's word
// counts, but if, for example, the self-check at Slot::pads_differ fails for any slot, the service
// ends and none of the batch's requests is released (Prover::a_self_check_failed()).
// The batch's other requests still influence a client's files, at least through the
// binding tree (the Merkle tree over the batch's seals) and through the keys of the batch's private
// randomness (`coins` in identifiers). These keys have all the batch's requests among their inputs
// (prover/prover/isk.cuh). Timing is not hidden. Both limits are described in
// docs/what-is-not-hidden.md, "The other requests in a batch" and the list item "Time".
// cp_prove's own printed line on timing is listed in prover/docs/closed_front.md.
#pragma once
#include "common/lie.cuh" // a test hook, told which slot is entered: empty in this repository
#include "common/member_hand.h"
#include "common/slot_flags.h"
#include "prover/request_part.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

constexpr int SLOTS_MOST = 64; // the largest batch: some per-batch bit masks are 64-bit words
// The largest number of leaves of a binding tree, 2^8. Used only in a static_assert (inputs.cuh),
// which ties it to parameter_file::LEAFD_LIMIT, the limit on `leafd`. The tree has `slots` leaves
// (a line of the parameter file), and `leafd` is log2 of `slots`.
constexpr int PLACES_MOST = 256;
// The largest batch pass (struct `Wave`: requests proved on one pass over the weights) that
// set_up_the_seats() accepts. The option -in-step takes only 1 to 16, and the GPU memory for the
// requests' state buffers, which that function sets up, can allow fewer (in_step.cuh).
constexpr int SLOTS_IN_STEP_MOST = SLOTS_MOST;

// There is no state for a failed request: its slot stays Live and goes through every step,
// whatever its failure word.
enum class SlotState : int {
  Empty = 0,   // nothing is proved here
  Live = 2     // holds a request
};

// One place in a batch: it holds one request or is Empty. Not the "slot" of "slot pads" (at
// `pads_differ`), which is the index of one masked value that a sub-proof sends.
struct Slot {
  int j = 0; // the slot's index in the batch, 0 .. B - 1; it appears in no file
  SlotState state = SlotState::Empty;
  char id[65] = {0};              // only if Live: the request's identifier
  std::vector<uint32_t> tokens;   // T tokens, then r and e: the proof covers positions r to e - 1
  // The counter in the labels of the request's private randomness, under the batch's process key
  // (Batch::ForTheLibrary::process_key).
  uint64_t attempt = 0;           // equals j
  uint32_t y = 0;                 // the next token (the token file); `finish_request` sets it
  // begin_request() fills the GPU's two self-check counters with this byte, and finish_request()
  // subtracts that start value: a sum left by a recent request then does not pass for this one's.
  // See docs/limits-all-rows.tsv, row H18.1.1.
  unsigned char counts_start = 0; // 1 to 255
  // The generator of the private randomness that blocks/ draws for the request (chain_coins.cuh),
  // from begin_the_batch_on_a_leaf() to clear_slot(). Declared before `part`, whose context draws
  // from it: members are destroyed in reverse order.
  std::unique_ptr<blk::Coins> coins;
  // The request's own part of its proof: the first chain's bytes, later also its place in the
  // binding tree and the second chain's bytes. From the first chain's seal to the second chain it
  // also holds the request's prover context (request_part.h).
  RequestPart part;
  // Unused: nothing sets or reads `hand` or `record`; clear_slot() resets them.
  MemberHand hand;
  std::vector<uint32_t> record;
  // Points to the slot's failure word, in the Batch that constructed the slot; swap_the_slots()
  // moves the pointer with the slot. Only leave_slot() adds bits; finish_request() first ORs the
  // GPU's bits into the host's live word. So the word is final when seat_ends() (in_step.cuh: the
  // request's last turn) returns, after the first chain's seal. The second chain runs with no slot
  // entered (end_the_batch(), service.cuh): a bit raised there belongs to no slot and fails the
  // whole batch.
  unsigned *flags_host = nullptr;
  // [own-test-slot-pads] (a self-check's printed name) The request's mask entries (`pad` in
  // identifiers: random values, each added once to one sent value) are compared, segment by
  // segment, with a second implementation of their derivation (slot_pads_second.cuh). A segment
  // (`link` in identifiers) is a sub-proof's part of the first chain. finish_request() reads the
  // three fields: the number of 64-bit values compared; for each segment, the number of
  // comparisons; and `pads_differ`. Callbacks that chain_bind() installs (chain_frame.cuh) set
  // `pads_differ` if a comparison fails, a segment has a second comparison or no entry in
  // `pad_links_held`, or the mask entry that a sub-proof took last for its claim on the weights
  // (index_mask_entry()) is not in its GPU cell.
  unsigned long long pad_cells_held = 0;
  std::vector<unsigned char> pad_links_held;
  bool pads_differ = false;
};

struct Batch {
  uint64_t number = 0; // rises while the process lives; printed for the operator, in no response
  int B = 0;           // the batch's slots are slot[0 .. B - 1]; some of them may be Empty
  // A leaf table is one of the commitment's weight tables: the same weights, its own masks.
  uint32_t leaf = 0;   // k, the number of the batch's leaf table; public: it is in every proof file
  // A check word derived from the batch's process key (key32_word_of(), prover/hiding/trees.cuh)
  // for the self-check [own-test-key]. Only begin_the_batch_on_a_leaf() compares it, and exits on a
  // mismatch; it is zeroed at the batch's end.
  uint32_t key_word = 0;
  // What begin_the_batch_on_a_leaf() (prover/prover/isk.cuh) stores here for calls that it makes
  // before it returns; `waits`: stored and not yet passed on. `root` and `mask_seed` (the key from
  // which the leaf table's masks are derived) go to leaf_begin(), in the translation unit that owns
  // the weight table (prover/hash_commitment/hf_keeper.cu), and wipe() follows that call. `root`
  // and `table` come from the leaf table's leaf file (`entry.bin` in the secret key directory) and
  // are checked against that unit's own. `process_key`, the key of the batch's private randomness,
  // is copied by key32_renew() and take_the_key(), for the sub-proofs' randomness and its
  // self-check. It does not go to that unit and is zeroed right after its last reader.
  struct ForTheLibrary {
    bool waits = false;
    std::vector<uint8_t> root;    // the leaf table's root, 32 bytes
    blk::IndexName table;         // the leaf table's item in the chains' first queries
    uint8_t mask_seed[32] = {0}, process_key[32] = {0};
    void wipe() {
      explicit_bzero(mask_seed, sizeof mask_seed), explicit_bzero(process_key, sizeof process_key);
      root.clear(), table = blk::IndexName(), waits = false;
    }
  } for_the_library;
  // The generator of the private randomness that the batch's requests share (chain_coins.cuh): the
  // binding tree, the places in it, the shared chain. begin_the_batch_on_a_leaf() makes it,
  // end_the_batch() destroys it.
  std::unique_ptr<blk::Coins> coins;
  Slot slot[SLOTS_MOST];
  // The failure words, reached through Slot::flags_host.
  unsigned flags[SLOTS_MOST] = {0};
  Batch() {
    for (int j = 0; j < SLOTS_MOST; j++) slot[j].j = j, slot[j].flags_host = &flags[j];
  }
  Batch(const Batch &) = delete; // each slot's `flags_host` points into a Batch
  Batch &operator=(const Batch &) = delete;
};

// Exchanges two batches' requests. `number`, `flags` and the struct ForTheLibrary stay. Each
// `flags_host` moves with its slot, so a slot's word need not be in the Batch that holds the slot.
// Call it only while no slot is entered (enter_slot()) and neither batch's ForTheLibrary::waits is
// true; neither condition is checked. prove_the_batch() (service.cuh) calls it only with -overlap
// or -overlap-two: a batch whose opening has begun stays in `Opening::rode` while `Prover::batch`
// holds the next.
static inline void swap_the_slots(Batch &a, Batch &b) {
  for (int j = 0; j < SLOTS_MOST; j++) std::swap(a.slot[j], b.slot[j]);
  std::swap(a.B, b.B);
  std::swap(a.leaf, b.leaf);
  std::swap(a.key_word, b.key_word);
  std::swap(a.coins, b.coins);
}

// A batch pass: the slots first .. first + n - 1 of a batch, proved on one pass over the weights.
struct Wave {
  Batch *batch = nullptr;
  int first = 0, n = 0;
};

// The slot that is being worked on; null between two slots' work.
static Slot *g_slot_entered = nullptr;
[[noreturn]] static inline void slots_out_of_order(const char *what) {
  printf("internal error: %s\n", what);
  exit(4);
}
// The bits that host code sets (raise_flags(), prover/common/slot_flags.h) from enter_slot() to
// leave_slot() go to the entered slot's failure word. A bit found set at enter_slot() belongs to no
// slot: it is put aside here and restored at leave_slot(), so live_flags() sees it only while no
// slot is entered. The GPU's live word is exchanged elsewhere, in stream order
// (prover/prover/member_state.cuh).
static unsigned g_flags_between_slots = 0;
static inline void enter_slot(Slot &s) {
  if (g_slot_entered) slots_out_of_order("a slot is entered while another slot is still entered");
  g_flags_between_slots |= g_live_flags_host.exchange(0);
  g_slot_entered = &s;
  lie::seat(s.j);
}
static inline void leave_slot(Slot &s) {
  if (g_slot_entered != &s) slots_out_of_order("the slot that is left is not the one that was entered");
  *s.flags_host |= g_live_flags_host.exchange(g_flags_between_slots);
  g_flags_between_slots = 0;
  g_slot_entered = nullptr;
  lie::seat(-1);
}

// The slot's failure word so far (final after seat_ends(): see Slot::flags_host). It waits for the
// whole GPU and adds FLAGS_UNREADABLE if that fails.
static inline unsigned flags_of(const Slot &s) {
  return *s.flags_host | (cudaDeviceSynchronize() == cudaSuccess ? 0 : FLAGS_UNREADABLE);
}

// Erases the request from the slot and makes it Empty. `part.lin`, `part.seal` and `part.place`
// keep their values. GPU memory is zeroed or restored elsewhere, at the end of each batch pass
// (wipe_behind_the_wave(), request.cuh).
static inline void clear_slot(Slot &s) {
  s.part.withhold(), s.part.let_go(), s.part.first.reset();
  s.coins.reset(); // wipes its key; after `part`, whose context draws from it
  for (auto *words : {&s.tokens, &s.record}) {
    if (!words->empty()) explicit_bzero(words->data(), words->size() * sizeof(uint32_t));
    words->clear();
  }
  explicit_bzero(s.id, sizeof s.id);
  s.hand = MemberHand{};
  s.attempt = 0, s.y = 0, s.counts_start = 0;
  s.pad_cells_held = 0, s.pad_links_held.clear(), s.pads_differ = false;
  *s.flags_host = 0;
  s.state = SlotState::Empty;
}
