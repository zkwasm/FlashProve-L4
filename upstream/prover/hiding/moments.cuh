// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How a sub-proof (the argument for one part of the forward pass) commits to its arrays and uses
// the request's first hash chain (a Fiat-Shamir transcript). Not self-contained: `T` (an element of
// the extension field E), `Tr` (the GPU's transcript object; both prover/hiding/transcript.cuh),
// CK, hz::Tree and more must be declared before prover/hiding/tail.cuh includes this file.
//
// Each sub-proof of a request is one segment (`link` in identifiers and messages) of the request's
// proof. The frame is the code that runs the proof outside the sub-proofs ("chain" in messages;
// blocks/iop/req.h, on the GPU prover/prover/chain_run.cuh). It commits to a segment's arrays and,
// when the sub-proof calls hd::close() (prover/hiding/door.cuh), reduces the sub-proof's one linear
// equation on them to a claim that the next segment carries.
// A commit point (`moment`) is a numbered prover message of a segment: 0 has the segment's first
// arrays and its mask row, a row of random elements of E; 1 has U3, a vector over E; 2 has ZZ, the
// digits of the approximate range check, and BZ, its lookups' multiplicity bits.
// A `slot` is the index of one masked element of E in a sub-proof's messages, not a request's
// place in its batch. Its mask entry (`pad`) is the entry of the segment's mask row that masks it.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "common/end_name.h"
namespace blk {
class ProverRounds;
}
// `ch` ("chain" in messages): what a sub-proof uses to declare arrays and to call the frame.
namespace ch {
// One array of a commit point after 0, as a sub-proof's set-up declares it for the `plan`
// (ChainHost::plan, prover/prover/chain_host.cuh), which fixes the arrays of every commit point.
// HiddenTail::chain_arrays() (prover/hiding/tail_f.cuh) declares ZZ and BZ with it. U3
// (chain_vector(), prover/prover/chain_moments.cuh) and the arrays of commit point 0
// (Prover::chain_first_moment(), prover/prover/chain_frame.cuh) are declared without it.
// Element (r, c), r < R and c < K, is the cell at index offset + r * row_stride + c * column_stride
// from entry `pointer` of `at` (Host::later).
struct ArrayAt {
  std::string name;
  // Word64: an element of the base field F in 64 bits. Cell16: an unsigned 16-bit integer.
  enum Cells { Word64, Cell16 } cells = Word64;
  size_t R = 1, K = 1, offset = 0, row_stride = 0, column_stride = 1;
  int pointer = 0;
  char cls = 'N'; // 'W' or 'N'; nothing reads it
};
// The sub-proofs' callbacks into the frame, so that a sub-proof's object needs no pointer to the
// `Prover` (prover/prover/prover.h; "the driver" in messages), the struct with the state and the
// top-level steps of the prover program cp_prove. Before each sub-proof, Prover::chain_bind()
// (prover/prover/chain_frame.cuh) sets them for that sub-proof's segment, except `blob`, which the
// caller of chain_bind() sets. The sub-proof then calls, in this order: first(), kept_pads_agree(),
// moment() for commit points 1 and 2, masking_pads_hold(), hd::close(). [own-test-slot-pads] marks
// the parts of the self-check that fails under that name.
struct Host {
  // Runs the segment's next commit point after 0 (ChainHost::later(),
  // prover/prover/chain_host.cuh): commits to the arrays as oracle messages (encoded rows under
  // Merkle roots) and answers their out-of-domain samples (random linear forms of a message, chosen
  // after its roots are sent). `at`: pointers to the arrays in GPU memory, indexed by
  // ArrayAt::pointer. `tr`, here and in first(): the request's own transcript object, or the
  // program exits. `moment` is ignored: the frame counts commit points, and if one was left out the
  // program ends, in hd::close() at the latest. The arrays must stay allocated and unchanged until
  // hd::close() has returned for this segment.
  std::function<void(int moment, const std::vector<const void *> &at, Tr *tr)> later;
  // All sub-proofs of a request run on one hash chain, the request's first chain. The CPU runs it
  // and lends it to the GPU, as the transcript object `tr`, for the rounds of the sub-proofs'
  // sumchecks; first() and later() end by lending it. A round that only the CPU can run (seeds,
  // bits, proof of work) goes between take_back(), which returns the CPU's channel
  // (blocks/chain/rounds.h), and lend(). Taking the chain back, which later() and hd::close() also
  // do, ends the program if a sumcheck is still running on `tr`.
  std::function<blk::ProverRounds &()> take_back;
  std::function<void()> lend;
  // For a sub-proof that leaves a linear claim on the weight table (the committed weights). Copies
  // the next unused entry of the request's index-mask message (mask rows that the first segment's
  // first() sends) to `where` in GPU memory; ChainRun keeps the entry's blk::MaskSlot
  // (blocks/iop/mask.h) for that claim. Call it after first(). `where` must lie outside the
  // segment's mask entries (not checked), so that the value is in one oracle message only.
  std::function<void(T *where)> index_mask_entry;
  // [own-test-slot-pads] Checks that `cell` still has the value that index_mask_entry() copied
  // last. Call it after the segment's last kernel. On a difference the request gets no proof.
  std::function<void(const T *cell)> index_mask_entry_holds;
  // Runs the segment's commit point 0 and lends the hash chain to `tr` (ChainHost::first()). The
  // segment's mask entries must already be in GPU memory as elements of E. `y`: the next token in
  // GPU memory for the output head's sub-proof, null for every other.
  std::function<void(Tr *tr, const u32 *y)> first;
  // Returns the GPU address of the array from which the frame took the segment's mask entries: the
  // sub-proof's own (ChainLink::masks). ChainRun::first() fetches it to the CPU and commits to the
  // fetched copy; the frame keeps no copy on the GPU. `slots` is ignored.
  std::function<const void *(size_t slots)> kept_pads;
  // [own-test-slot-pads] Compares the `slots` mask entries at `pads` in GPU memory with what a
  // second implementation of their derivation gives (SlotPadsSecond::hold(),
  // prover/prover/slot_pads_second.cuh). On a difference Prover::finish_request()
  // (prover/prover/request.cuh) gives no proof.
  std::function<void(const void *pads, size_t slots)> masking_pads_hold;
  // Returns ROOTW 32-bit words in GPU memory; callers copy the last DW, a digest's length, as a
  // root (both constants: prover/common/setting.h). The frame returns no root to a sub-proof, so
  // chain_bind() binds this to zeros for all commit points.
  std::function<const u32 *(int moment)> words;
  // Copies the words that the sub-proof has written for this segment into one GPU buffer and
  // returns its address. Callers store it in a field, `host_words`, that nothing reads.
  std::function<const u32 *()> blob;
};
// One instance per translation unit. One is enough: a prover's sub-proofs run one after another,
// even when cp_prove's option -in-step interleaves requests.
static Host host;
// Self-check on the array of host.kept_pads: a kernel derives every mask entry again and counts the
// values that agree (hz::Tree::slot_pads_agree(), prover/hiding/trees.cuh);
// Prover::finish_request() checks the count. S: the hz::Tree that drew the entries. Call it after
// host.first(), before anything writes to the array. It does not see the copy that the frame
// fetches to the CPU and commits to, nor does any other self-check of cp_prove
// (docs/limits-all-rows.tsv, row H3.18). A fault in that copy gives a proof that does not verify,
// and no check in cp_prove notices it.
static void kept_pads_agree(const hz::Tree &S, size_t slots) {
  S.slot_pads_agree((const hz::T *)host.kept_pads(slots), slots);
}
// Runs host.masking_pads_hold on the array of mask entries that the sub-proof's kernels mask with.
// Call it exactly once for each segment, after the entries' last use and before hd::close(). If the
// call is missing, repeated or made through an earlier segment's binding, the request gets no
// proof. If the callback is unbound, the program exits rather than skip the check.
static void masking_pads_hold(const void *pads, size_t slots) {
  if (!host.masking_pads_hold) {
    fprintf(stderr, "internal error: chain: the slot pads' test by value was requested before the driver bound it\n");
    name_the_end("own-test-slot-pads");
    exit(4);
  }
  host.masking_pads_hold(pads, slots);
}
// Calls host.later. Of m it checks only that m >= 1.
static void moment(int m, const std::vector<const void *> &at, Tr *tr) {
  if (!host.later || m < 1) {
    fprintf(stderr, "internal error: chain: a moment was requested before the driver bound the "
                    "link's function, or the moment's number is below 1\n");
    exit(4);
  }
  host.later(m, at, tr);
}
} // namespace ch
// Commits to the vector of U.K elements of E at `vec` (U3 in every caller) as the segment's next
// commit point; m is that point's number (see ch::moment()). `vec` is copied to U.layE, so the
// caller may reuse it. `root`: DW words in GPU memory, which receive the last DW words of
// ch::host.words(m).
static inline void commit_unit(hz::TreeN &U, const T *vec, u32 *root, Tr *tr, [[maybe_unused]] int m) {
  CK(cudaMemcpyAsync(U.layE, vec, U.K * sizeof(T), cudaMemcpyDeviceToDevice, 0));
  ch::moment(m, {U.layE}, tr);
  CK(cudaMemcpyAsync(root, ch::host.words(m) + ROOTW - DW, 4 * DW, cudaMemcpyDeviceToDevice, 0)); // zeros: Host::words
}
