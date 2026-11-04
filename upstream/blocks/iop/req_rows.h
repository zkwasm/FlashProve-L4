// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Converts the equation that ends a sub-proof into a linear claim (`Row`, relation.h) on oracle
// messages (encoded rows, sent as Merkle roots).
//
// A sub-proof is the argument for one part of the computation; a segment (`link` in identifiers) is
// its part of a request's proof (req.h). A sub-proof works on arrays: named matrices of committed
// entries. A public layout cuts each array into aligned blocks and places them in the main rows of
// one of the segment's oracle messages (laid.h). The equation (`ReqEquation`) has rank-one weights
// on arrays and coefficients on the segment's mask row (random values that mask sent scalars).
// Prover and verifier build the claim with the same function, ReqRows::row().
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/laid.h"
#include "iop/req.h"

namespace blk {

struct ReqArray {
  std::string name;
  size_t message = 0;  // which of its segment's oracle messages contains it
  LaidMatrix laid;     // its size and its blocks; rows are numbered within that oracle message
};
// link[l]: the arrays that segment l sends.
struct ReqLayout {
  std::vector<std::vector<ReqArray>> link;
  const ReqArray *find(size_t l, const std::string &name) const {
    if (l < link.size())
      for (const ReqArray &a : link[l])
        if (a.name == name) return &a;
    return nullptr;
  }
};

// Canonical bytes and a check, as blocks/chain/public.h requires of a value in the first query of a
// Fiat-Shamir hash chain.
inline void public_bytes(Writer &w, const ReqArray &a) {
  public_bytes(w, a.name), w.put_u64(a.message), w.put_u64(a.laid.size_i), w.put_u64(a.laid.size_c), public_bytes(w, a.laid.blocks);
}
inline void public_bytes(Writer &w, const ReqLayout &l) { public_bytes(w, l.link); }
// Null, or what is wrong with the layout itself. Whether it fits the request: req_layout_fault().
inline const char *public_fault(const ReqArray &a) {
  if (a.name.empty()) return "an array has no name";
  if (!a.laid.size_i || !a.laid.size_c || a.laid.size_i >> 48 || a.laid.size_c >> 48) return "an array has no element, or 2^48 rows or columns or more";
  if (a.laid.blocks.empty()) return "an array has no block";
  return public_fault(a.laid.blocks);
}
inline const char *public_fault(const ReqLayout &l) {
  for (const std::vector<ReqArray> &link : l.link)
    for (size_t a = 0; a < link.size(); a++) {
      if (const char *fault = public_fault(link[a])) return fault;
      for (size_t b = 0; b < a; b++)
        if (link[a].name == link[b].name) return "two arrays of a link have one name";
    }
  return nullptr;
}

// Null if the layout fits shape `s` and the codes, else what is wrong. A layout is input to a
// verifier, which must call this first. Precondition, not checked: req_fault(codes, s) (req.h) is
// null. req_layout_fault() repeats only part of public_fault(): an empty name passes. Pad
// (send.h): the random entries of an oracle message that mask its out-of-domain sample's answer.
// If req_fault(codes, s) is not null, Send::pad() can terminate the process: a row can be shorter
// than its pad.
inline const char *req_layout_fault(const Codes &codes, const ReqShape &s, const ReqLayout &layout) {
  if (layout.link.size() != s.link.size()) return "the layout has another number of links than the request";
  for (size_t l = 0; l < s.link.size(); l++) {
    const std::vector<ReqMessage> &messages = s.link[l].messages;
    for (size_t a = 0; a < layout.link[l].size(); a++) {
      if (layout.link[l][a].message >= messages.size()) return "an array lies in an oracle message that its link does not have";
      for (size_t b = 0; b < a; b++)
        if (layout.link[l][a].name == layout.link[l][b].name) return "two arrays of a link have one name";
    }
    for (size_t m = 0; m < messages.size(); m++) {
      const OracleShape &o = messages[m].shape;
      std::vector<LaidMatrix> in_it;
      for (const ReqArray &a : layout.link[l])
        if (a.message == m) in_it.push_back(a.laid);
      if (in_it.empty()) continue;
      if (!o.main_rows) return "an array lies in an oracle message without main rows";
      const unsigned row_bits = ceil_log2(o.main_rows), place_bits = codes.base[o.code].log_ell();
      if (!laid_ok(in_it, row_bits, place_bits)) return "blocks that are not aligned, overlap, leave the table or do not cover their array";
      for (const LaidMatrix &x : in_it)
        for (const LaidBlock &b : x.blocks) {
          if (b.first_row + (u64(1) << b.row_bits) > o.main_rows) return "a block lies in rows that are not sent";
          for (const PadPlace &p : Send::pad(codes, o)) {  // no claim may read the pad of a sample
            const u64 row = p.index & ((u64(1) << row_bits) - 1), place = p.index >> row_bits;
            if (p.part == Part::MAIN && row >= b.first_row && row < b.first_row + (u64(1) << b.row_bits) && place >= b.first_place &&
                place < b.first_place + (u64(1) << b.place_bits))
              return "a block covers the pad of its oracle message's sample";
          }
        }
    }
  }
  return nullptr;
}

// One term: the weight scale * on_rows[i] * on_columns[c] on element (i, c) of the array `array`.
struct ReqTerm {
  size_t born = 0;    // the segment that sent the array: the current one or an earlier one
  std::string array;
  E scale = E::one();
  IndexWeights on_rows, on_columns;  // at most one of the two is a written-out vector
};
// Sum of `terms` + sum over i of on_masks[i] * (mask entry i) = `target`.
struct ReqEquation {
  std::vector<ReqTerm> terms;
  std::vector<E> on_masks;  // coefficients on the first entries of the segment's mask row
  E target = E::zero();     // public
};

// Builds a segment's linear claim. `Context`: `ProverContext` or `VerifierContext` (context.h).
// `bound` is the layout as it is in the chain's first query. `ctx`, `frame` and `bound` must
// outlive the object, which keeps references. The constructor terminates the process if
// req_layout_fault() finds a fault, so no user of ReqRows can omit that check. A caller that builds
// a `Row` from laid_form() (laid.h) by hand is not covered: laid_form() checks each block by
// itself, not overlap, full cover or the pad. Terminating is acceptable because a layout is public
// input, fixed before a proof is read; code that reads one from a file must call
// req_layout_fault() first and reject a faulty file.
template <class Context> class ReqRows {
 public:
  ReqRows(Context &ctx, const ReqFrame &frame, const Public<ReqLayout> &bound) : ctx_(ctx), frame_(frame), layout_(ctx.ch.get(bound)) {
    const char *fault = req_layout_fault(ctx.st.codes(), frame.shape(), layout_);
    BLK_REQUIRE(!fault, fault ? fault : "");
  }
  // The current segment's equation as a linear claim; the caller adds it to the segment's group
  // (ReqFrame::group()). The process terminates if an array's oracle message has not been sent or
  // the current segment cannot read it (ReqFrame::where()). The caller's duty, not checked:
  // weights, coefficients and `target` depend only on the first query, sent messages and
  // challenges. A sub-proof that uses `on_masks` numbers its mask entries itself, from 0 (how
  // these mask its sent scalars is not checked), and takes none from the `MaskLedger` (mask.h),
  // which gives out each entry once: row() takes them and terminates the process if any entry was
  // taken before.
  Row row(const ReqEquation &eq) const {
    Statement &S = ctx_.st;
    const ReqShape &shape = frame_.shape();
    Row row;
    row.name = "equation of " + shape.link[frame_.link()].name;
    for (const ReqTerm &t : eq.terms) {
      const ReqArray *a = layout_.find(t.born, t.array);
      BLK_REQUIRE(a != nullptr, "request: an equation names an array that the layout does not have");
      const OracleId o = frame_.where(t.born, a->message);
      row.add_form(o, laid_form(a->laid.blocks, S.row_bits(o, Part::MAIN), S.place_bits(o, Part::MAIN), t.on_rows, t.on_columns, t.scale));
    }
    if (!eq.on_masks.empty()) {
      const std::vector<ReqMessage> &messages = shape.link[frame_.link()].messages;
      size_t m = 0;
      while (m < messages.size() && !messages[m].shape.mask_rows) m++;
      BLK_REQUIRE(m < messages.size() && messages[m].shape.mask_rows == 1, "request: coefficients on a mask row, and the link has not one");
      const OracleId o = frame_.where(frame_.link(), m);
      const std::vector<MaskSlot> taken = S.masks(o).take(eq.on_masks.size());  // used up
      BLK_REQUIRE(taken.front().index() == 0 && taken.back().index() + 1 == taken.size(), "request: entries of the link's mask row were taken before its equation names them by number");
      row.add_run(o, Part::MASK, 0, 1, eq.on_masks);
    }
    row.target = eq.target;
    return row;
  }

 private:
  Context &ctx_;
  const ReqFrame &frame_;
  const ReqLayout &layout_;
};

}  // namespace blk
