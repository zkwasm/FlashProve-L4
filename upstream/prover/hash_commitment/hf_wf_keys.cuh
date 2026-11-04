// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The committer's weights as the lookup keys of WF, its range proof on the committed matrix W
// (blocks/iop/wf.h; GPU side: prover/engine/wf_heavy.cuh).
//
// Here WF has two lookups: every cell of W is a signed byte; every weight stored as two digits is
// in [-4096, 4095]. Each lookup's keys are cut into parts, by rows of W. `RangeWindows`
// (prover/hash_commitment/hf_range_windows.cuh) turns weights into keys, one lookup call of a plan
// (prover/hash_commitment/hf_range_plan.h) at a time. So a part is a call. A call has 2^lwr
// places, each with one key per column of W; W's rows go in pairs, a weight's two digits together.
// For parts of up to 2^L rows:
//   bytes     the calls B of a plan with lwr = L:      a place's top bit selects a row of a pair,
//             its other bits the pair;
//   weights   the calls W of a plan with lwr = L - 1:  one pair of rows per place.
// A part of either kind reads about the same amount of packed weights.
// Include it after the headers that prover/hash_commitment/hf_range_windows.cuh needs, with
// blocks/ on the include path.
#pragma once
#include "engine/wf_heavy.cuh"
#include "hash_commitment/hf_range_plan.h"
#include "hash_commitment/hf_range_windows.cuh"
#include <memory>

namespace rf {

// Returns the plan's calls of `kind` as a lookup in WF's parameters, one part per call.
inline blk::WfLookup wf_lookup_of(const ::hf::HfRangePlan &plan, u32 kind, const char *name, unsigned mu, unsigned live, const std::vector<unsigned> &first_bound) {
  blk::WfLookup l;
  l.name = name, l.d = (unsigned)hf_range_k(kind), l.mu = mu, l.live = live, l.first_bound = first_bound;
  l.offset = blk::from_u64(hf_range_off(kind)), l.padding_key = blk::F::zero();
  l.digit_weight = {blk::F::one()};
  if (kind == HF_RANGE_W) l.digit_weight.push_back(blk::from_u64(HF_RANGE_SECOND_DIGIT));
  for (u32 f = 0; f < plan.NF; f++) {
    if (plan.kind(f) != kind) continue;
    blk::WfPart part;
    part.log_places = (unsigned)plan.lwr;
    for (u32 pl = 0; pl < plan.places(); pl++) {
      u32 row[2], wt[2];
      const int rows = plan.rows_at(f, pl, row, wt);
      if (!rows) continue;
      const blk::u64 stride = rows == 2 ? row[1] - row[0] : 0;
      if (!part.runs.empty() && part.runs.back().first_place + part.runs.back().places == pl && part.runs.back().first_row + part.runs.back().places == row[0] &&
          part.runs.back().stride == stride)
        part.runs.back().places++;
      else
        part.runs.push_back(blk::WfRun{pl, row[0], 1, stride});
    }
    l.parts.push_back(std::move(part));
  }
  return l;
}

// A piece of a part (`blk::LkpPiece`, prover/engine/lkp_pieces.cuh) is one call of the plan that
// HfRangePlan::pieces_of() makes from the lookup's plan.
// GPU memory: one `RangeWindows` at a time, set up again for another lookup or another cut into
// pieces. Its staging slots and the `WeightWalker` buffers
// (prover/hash_commitment/hf_commit_walk.h) hold weights; close() zeroes them, except the walker's
// norm gains: those stay on the GPU until PinnedWeights::end()
// (prover/hash_commitment/hf_commit_weights.cuh).
class WfWindows : public blk::WfKeySource {
public:
  // Lookup 0: the calls B of `bytes`; lookup 1: the calls W of `weights`. All three arguments must
  // outlive this object. Make the plans with identity_bits = 30: neither WF nor this file uses a
  // plan's identities (2^iota classes of a call's keys), but with one identity for a call of 2^36
  // keys (full size) the plan's 32-bit count of padding keys in an identity can overflow, and
  // HfRangePlan::make() then fails.
  WfWindows(const ::hf::Committed &committed, const ::hf::HfRangePlan &bytes, const ::hf::HfRangePlan &weights) : committed_(committed), whole_{&bytes, &weights} {}
  ~WfWindows() override { close(); }
  WfWindows(const WfWindows &) = delete;
  WfWindows &operator=(const WfWindows &) = delete;
  // Lends GPU memory, 16-byte aligned (not checked), that keeps the current part's packed weights,
  // as far as they fit, so that the many passes over a part do not copy them again; null: none.
  // RangeWindows::keep() can store some of the part's keys there too. After lend(nullptr, 0), lay()
  // can still read and write the old pool, until close().
  // Neither this class nor `RangeWindows` zeroes the pool, and close() does not touch it: its owner
  // zeroes it. On its normal path once_make() (prover/hash_commitment/hf_commit_once.cuh) calls
  // close(), then zeroes and frees the pool. If it returns earlier, as when blk::make_key() fails,
  // the destructors keep that order: ~WfWindows() calls close(), then ~HfOncePool() zeroes and frees the pool. An exit(),
  // as in open() or in a failed CK(), skips both steps: nothing is zeroed.
  void lend(void *pool, size_t bytes) { pool_ = pool, pool_bytes_ = bytes, kept_ = NONE; }

  // `keys`: GPU memory, 16-byte aligned (not checked).
  void lay(size_t lookup, const blk::LkpPiece &piece, blk::Key16 *keys) override {
    static_assert(sizeof(blk::Key16) == sizeof(u16), "a key has 16 bits");
    open(lookup, piece.log_places, piece.closed);
    const ::hf::HfRangePlan &whole = *whole_[lookup];
    const u32 more = (u32)whole.lwr - piece.log_places, call = (lookup ? whole.NB : 0) + piece.part;
    if (pool_ && kept_ != call) {
      win_->keep(pool_, pool_bytes_, call << more, (call + 1) << more);
      kept_ = call;
    }
    win_->lay((int)((call << more) | (u32)piece.number), (u16 *)keys, cudaStreamPerThread);
    laid++;
  }
  // RangeWindows::lay() writes a key outside its table as 0 and sets `UNFIT`
  // (prover/common/slot_flags.h).
  bool all_fit() override {
    CK(cudaDeviceSynchronize());
    const unsigned flags = take_live_flags();
    if (flags & ~(unsigned)UNFIT) raise_flags(flags & ~(unsigned)UNFIT);  // left for the caller
    return !(flags & UNFIT) || commit_lie::passes("unfit-key");  // false in this repository
  }
  void close() {
    if (win_) win_->end();
    win_.reset(), kept_ = NONE, lookup_ = NONE;
  }
  size_t laid = 0, set_up = 0;  // for a report

private:
  static constexpr u32 NONE = ~(u32)0;
  void open(size_t lookup, unsigned log_places, unsigned closed) {
    if (win_ && lookup_ == lookup && (unsigned)view_.lwr == log_places && (unsigned)view_.closed == closed) return;
    close();
    view_.pieces_of(*whole_[lookup], (int)log_places, (int)closed);
    win_.reset(new RangeWindows);
    if (win_->setup(committed_, view_)) {
      fprintf(stderr, "the ranges' keys: internal error: the windows could not be set up\n");
      exit(4);
    }
    lookup_ = (u32)lookup, set_up++;
  }
  const ::hf::Committed &committed_;
  const ::hf::HfRangePlan *whole_[2];
  ::hf::HfRangePlan view_;
  std::unique_ptr<RangeWindows> win_;
  u32 lookup_ = NONE, kept_ = NONE;
  void *pool_ = nullptr;
  size_t pool_bytes_ = 0;
};

} // namespace rf
