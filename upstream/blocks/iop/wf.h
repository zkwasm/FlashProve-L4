// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// WF: the committer's range proof on a committed table (encoded rows under a Merkle tree). It is to
// show that every entry of the table's matrix M is in its range, by lookups (LKP) of keys, each a
// public linear map of entries of M plus a constant (WfLookup). The keys are never sent.
//
// Messages, all on one channel. (An oracle message is encoded rows, sent as Merkle roots.)
//   caller  the index-mask message (mask rows only; its entries mask claims on M), sent and
//           sampled out of domain (SEND) before WF begins
//   SEND    one oracle message: the multiplicities of all lookups, in bits (wf_layout()), and
//           mask rows; with it, if `masks_ride`, the mask of the proximity test's first fold
//   LKP     once for each lookup, in order
//   IOPP    the proximity test on the group that consists of that one oracle message
// Rounds and errors are those of these three blocks: wf_plan(), blocks/frame/wf_frame.h. WF has no
// check of its own on a proof. LKP's linear claims (`Row`) go into that group, and its claims on
// the keys become pending claims on M (WfPending). These show nothing about M unless the caller
// passes them to member_first_prove() or member_first_verify() (blocks/iop/member.h) and then
// completes LIN, the test on the committed table (its shared and second parts).
#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "iop/context.h"
#include "iop/iopp.h"
#include "iop/lkp.h"
#include "iop/member.h"
#include "iop/send.h"
#include "numbers/blocks.h"

namespace blk {

// A run of consecutive places of a part (WfPart). For digit q of its keys (WfLookup), place
// first_place + p, p < places, reads matrix row first_row + p + q * stride: the layout of a matrix
// that stores all digits 0 of a block of entries, then all digits 1, and so on.
struct WfRun {
  u64 first_place = 0;
  u64 first_row = 0;  // the matrix's row of the run's first place, digit 0
  u64 places = 0;
  u64 stride = 0;     // from a digit's row to the next digit's
};
// A part of a lookup's keys: 2^log_places places, each with one key for each of the 2^k_col columns
// of M. The key at (place, column) has the number column + 2^k_col * place. A place that no run
// covers is padding: its keys are the public WfLookup::padding_key and are not counted in the
// multiplicities. Parts exist for a prover that cannot hold all keys at once (lkp.h).
struct WfPart {
  unsigned log_places = 0;
  std::vector<WfRun> runs;  // by rising places, without overlap; a place in no run is padding
  u64 places() const {      // covered (after wf_fault: no wrap-around)
    u64 n = 0;
    for (const WfRun &r : runs) n += r.places;
    return n;
  }
};
// One lookup claims that every key at a covered place is a row of the range table 0 .. 2^d - 1:
//   key(place, column) = offset + sum over digits q of digit_weight[q] * M[row(place, q)][column].
// Examples. Every entry is a signed byte: one digit of weight 1, offset 128, d = 8.
// Every weight stored as two bytes is in [-4096, 4095]: weights 1 and 256, offset 4096, d = 13.
// A table row's multiplicity is the number of counted keys equal to it; plane j is bit j of all
// multiplicities. `first_bound`: the variables that the sumcheck on a part's last key level (the
// level of LKP's product tree next to the keys) binds first, in this order.
struct WfLookup {
  std::string name;
  unsigned d = 0, mu = 0, live = 0;  // the range table 0 .. 2^d - 1; 2^mu plane slots, of which 'live' are real (0: all)
  std::vector<F> digit_weight;       // one for each of a place's rows
  F offset = F::zero();
  F padding_key = F::zero();         // what a place of padding holds
  std::vector<WfPart> parts;
  std::vector<unsigned> first_bound; // LkpPublic::last_key_level_order
};
// The size of M: `rows` rows of 2^k_col elements of F. It is not a field of WfPublic: the hash
// chain's first query (it fixes the public inputs before the prover speaks) has it once, in the
// shape of the test on the committed table (LinShape), and the entry points make their WfMatrix
// from that LinShape only. Written twice, the sizes could differ: ranges that count one row fewer
// than the test would leave that row unchecked, and every check would pass.
struct WfMatrix {
  unsigned k_col = 0;
  u64 rows = 0;
  unsigned k_row() const { return ceil_log2(rows); }
};
inline WfMatrix wf_matrix_of(const LinShape &table) { return WfMatrix{table.k_col(), table.matrix_rows}; }

// WF's public parameters. The entry points take them as a Public<WfPublic>, so everything but the
// names, which are only labels, is in the chain's first query. `stages`, `choice`: of WF's
// proximity test.
struct WfPublic {
  std::vector<WfLookup> lookups;
  OracleNotice message;            // multiplicities and masks
  OracleId message_id = NONE_ID;   // the number that it gets on its channel
  std::vector<IoppStage> stages;
  IoppChoice choice;
};
inline void public_bytes(Writer &w, const WfRun &r) { public_bytes(w, r.first_place), public_bytes(w, r.first_row), public_bytes(w, r.places), public_bytes(w, r.stride); }
inline void public_bytes(Writer &w, const WfPart &p) { public_bytes(w, p.log_places), public_bytes(w, p.runs); }
inline void public_bytes(Writer &w, const WfLookup &l) {
  public_bytes(w, l.d), public_bytes(w, l.mu), public_bytes(w, l.live), public_bytes(w, l.digit_weight);  // not the name: it labels the lookup's rounds and messages in the plan (`wf_plan`)
  public_bytes(w, l.offset), public_bytes(w, l.padding_key), public_bytes(w, l.parts), public_bytes(w, l.first_bound);
}
inline void public_bytes(Writer &w, const WfPublic &x) {
  public_bytes(w, x.lookups), public_bytes(w, x.message), public_bytes(w, x.message_id);
  public_bytes(w, x.stages), public_bytes(w, x.choice);
}

// ---- what follows from the parameters ----

// The lookups in the form that LKP takes.
inline std::vector<LkpPublic> wf_lookups(const WfPublic &x, const WfMatrix &m) {
  std::vector<LkpPublic> all;
  for (const WfLookup &l : x.lookups) {
    LkpPublic p;
    p.d = l.d, p.mu = l.mu, p.live = l.live;
    for (unsigned number = 0; number < l.parts.size(); number++) {
      const WfPart &part = l.parts[number];
      p.parts.push_back(part.log_places + m.k_col);
      // Padding is every range of places before, between and after the runs. A place is 2^k_col
      // consecutive key numbers.
      auto stretch = [&](u64 from, u64 to) {
        if (from < to) p.padding.push_back(LkpPadding{number, from << m.k_col, 0, (to - from) << m.k_col, l.padding_key.v});
      };
      u64 upto = 0;
      for (const WfRun &r : part.runs) stretch(upto, r.first_place), upto = r.first_place + r.places;
      stretch(upto, u64(1) << part.log_places);
    }
    p.read_from = {x.message_id};
    p.keys = {LkpKeyBlock{0, 0, 0, 0}};
    p.tables = {LkpTablePart{0, l.d, 0, {}}};
    p.last_key_level_order = l.first_bound;
    all.push_back(std::move(p));
  }
  return all;
}
// Where the mask entries come from. The claims that the lookups add to WF's group name mask entries
// of WF's oracle message only. But the last key level of a part sends two masked values of a map
// that is linear in M, and the claim on them leaves WF as a pending claim, which only the test on
// the committed table reads. So the masks of these two values are entries of the index-mask
// message, which belongs to that test.
// Hiding requires, besides LKP's and SEND's own conditions:
//  - the index-mask message has 2 entries for each part of each lookup, besides those that BATT
//    (batt.h) takes when the caller batches the pending claims into one (wf_index_mask_entries());
//  - an execution of the prover that has broken off is never resumed with the same mask messages:
//    a new one draws all its private randomness (`coins` in identifiers) again. Two values of M
//    sent under one mask entry give, subtracted with the right factors, a linear function of M
//    without any mask.
//
// Entries of the index-mask message that WF takes.
inline size_t wf_index_masks(const WfPublic &x, const WfMatrix &m) {
  size_t n = 0;
  for (const LkpPublic &p : wf_lookups(x, m)) n += lkp_key_masks(p);
  return n;
}
// All entries of the index-mask message that are taken on WF's chain (a key file's one-time chain):
// WF's, and BATT's if there is more than one pending claim. Code that sizes that message, or checks
// that it has room, calls this, so that the count is written once.
inline size_t wf_index_mask_entries(const WfPublic &x, const WfMatrix &m) {
  size_t parts = 0;
  for (const WfLookup &l : x.lookups) parts += l.parts.size();
  return wf_index_masks(x, m) + (parts > 1 ? batt_masks(m.k_col) : 0);
}
// Entries of its own oracle message's mask rows that WF takes.
inline size_t wf_own_masks(const WfPublic &x, const WfMatrix &m) {
  size_t n = 0;
  for (const LkpPublic &p : wf_lookups(x, m)) n += lkp_masks(p) - lkp_key_masks(p);
  return n;
}
// Linear claims that WF's proximity test batches: the message's sample claims (one for each answer
// to its sample) and, for each lookup, those of LKP's steps (lkp.h), in the order of the sum below:
// tie, parts' roots, table levels, leaves, bits (2), number of keys, key levels.
inline u64 wf_rows_batched(const WfPublic &x, const WfMatrix &m) {
  u64 n = x.message.shape.answers;
  for (const LkpPublic &p : wf_lookups(x, m)) {
    n += (p.planes() - 1) + (p.part_bits().size() - 1) + p.d + 1 + 2 + 1;
    for (unsigned v : p.part_bits()) n += v;
  }
  return n;
}

// Layout of the multiplicities in the main rows. The live planes of a lookup are cut into aligned
// blocks of plane numbers (lkp_live_blocks()). A block of 2^b planes is a vector of 2^(b + d) bits,
// entry (plane - first) + 2^b * (row of the table). The blocks of all lookups are stored one after
// the other, the largest first (ties: in the order of the lookups and of their blocks); a block is
// within one main row. All sizes are powers of two, so every block begins at a multiple of its
// size. `row`, `first_place`: the main row, and the index in it, at which the block begins.
struct WfBlock {
  size_t lookup = 0;
  std::pair<unsigned, unsigned> planes;  // (first, log2 of the number)
  unsigned bits = 0;                     // log2 of its entries
  u64 row = 0, first_place = 0;
};
inline std::vector<WfBlock> wf_layout(const WfPublic &x, size_t ell) {
  std::vector<WfBlock> blocks;
  for (size_t i = 0; i < x.lookups.size(); i++)
    for (const std::pair<unsigned, unsigned> &b : lkp_live_blocks(x.lookups[i].live ? x.lookups[i].live : 1u << x.lookups[i].mu))
      blocks.push_back(WfBlock{i, b, b.second + x.lookups[i].d, 0, 0});
  std::stable_sort(blocks.begin(), blocks.end(), [](const WfBlock &a, const WfBlock &b) { return a.bits > b.bits; });
  u64 row = 0, place = 0;
  for (WfBlock &b : blocks) {
    if (place + (u64(1) << b.bits) > ell) row++, place = 0;
    b.row = row, b.first_place = place;
    place += u64(1) << b.bits;
  }
  return blocks;
}

// public_fault(): null, or how a value contradicts itself or is out of range, as far as the value
// by itself shows. None of these overloads ends the process. A WfPublic does not store which places
// are padding, or how many: both follow from the runs.
inline const char *public_fault(const WfRun &r) { return r.places < 1 ? "a run covers no place" : nullptr; }
inline const char *public_fault(const WfPart &p) {
  if (p.log_places > 40) return "a part has too many places";
  u64 upto = 0;  // the places below are covered or passed over
  for (const WfRun &r : p.runs) {
    if (const char *f = public_fault(r)) return f;
    if (r.first_place < upto || r.first_place >= (u64(1) << p.log_places)) return "a part's runs overlap, are not in rising order, or begin beyond the part";
    if (r.places > (u64(1) << p.log_places) - r.first_place) return "a run reaches beyond its part";
    upto = r.first_place + r.places;
  }
  return upto ? nullptr : "a part covers no place";
}
inline const char *public_fault(const WfLookup &l) {
  const unsigned planes = l.live ? l.live : 1u << std::min(l.mu, 6u);
  if (l.mu < 1 || l.mu > 6 || l.live >= (1u << l.mu) || planes < 2 || planes > 62 || l.d < 1 || l.d + planes >= F_BITS) return "a look-up's table or planes are not admitted";
  if (l.digit_weight.empty() || l.parts.empty()) return "a look-up without a digit or without a part";
  for (const F &w : l.digit_weight)
    if (w.v == 0) return "a digit of a look-up's key has the weight 0";  // its entries would enter no key: the look-up would say nothing of them
  if (!(l.padding_key.v >> l.d == 0)) return "a place of padding holds no row of the table";
  for (size_t i = 0; i < l.first_bound.size(); i++)
    for (size_t j = 0; j < i; j++)
      if (l.first_bound[i] == l.first_bound[j]) return "the order of the last key level names a variable twice";
  for (const WfPart &p : l.parts) {
    if (const char *f = public_fault(p)) return f;
    if (l.digit_weight.size() == 1)
      for (const WfRun &r : p.runs)
        if (r.stride) return "a run has a stride where a key has one digit";  // nothing would read it: one meaning, one encoding
    if (l.digit_weight.size() > 1)
      for (const WfRun &r : p.runs)
        if (!r.stride) return "the digits of a key lie in one row";  // a key of several digits would then be a multiple of one entry
  }
  return nullptr;
}
inline const char *public_fault(const WfPublic &x) {
  if (x.lookups.empty()) return "no look-up";
  const OracleShape &s = x.message.shape;
  if (s.main_rows == 0 || s.mask_rows == 0 || s.main_deg != 1 || s.answers < 1) return "the oracle message has not main rows over F and mask rows, or is not sampled";
  for (const WfLookup &l : x.lookups)
    if (const char *f = public_fault(l)) return f;
  return nullptr;
}
// Whether the keys of this lookup are offset + one entry (one digit, of weight 1) and its runs
// cover every row of the matrix.
inline bool wf_covers_all_rows(const WfPublic &x, size_t lookup, const WfMatrix &m) {
  const WfLookup &l = x.lookups[lookup];
  if (l.digit_weight.size() != 1 || !(l.digit_weight[0] == F::one())) return false;
  std::vector<std::pair<u64, u64>> stretch;
  for (const WfPart &p : l.parts)
    for (const WfRun &r : p.runs) stretch.push_back({r.first_row, r.first_row + r.places});
  std::sort(stretch.begin(), stretch.end());
  u64 upto = 0;
  for (const std::pair<u64, u64> &s : stretch) {
    if (s.first > upto) return false;
    upto = std::max(upto, s.second);
  }
  return upto >= m.rows;
}

// public_fault(x), then whether x fits the matrix and the codes: null, or the fault. A caller that
// reads parameters from a file calls this first: the entry points end the process on a fault.
inline const char *wf_fault(const Codes &codes, const WfPublic &x, const WfMatrix &m) {
  if (const char *f = public_fault(x)) return f;
  if (m.rows < 1 || m.k_row() > 40 || m.k_col > 40) return "a matrix without rows, or with too many rows or columns";
  for (const WfLookup &l : x.lookups) {
    const unsigned planes = l.live ? l.live : 1u << l.mu;
    u64 keys = 0;
    for (const WfPart &p : l.parts) {
      if (p.log_places + m.k_col < 1 || p.log_places + m.k_col >= planes) return "a part has no key, or more keys than a multiplicity can count";
      for (unsigned variable : l.first_bound)  // the last key level of a part of 2^v keys has v - 1 variables
        if (variable + 1 >= p.log_places + m.k_col) return "the order of the last key level names a variable that a part has not";
      for (const WfRun &r : p.runs) {
        // the last digit's last row: first_row + (places - 1) + (digits - 1) * stride
        if (r.stride > m.rows || r.first_row >= m.rows || r.places > m.rows - r.first_row) return "a run's places reach beyond the matrix";
        if ((l.digit_weight.size() - 1) * r.stride > m.rows - r.first_row - r.places) return "a run's places reach beyond the matrix";
      }
      keys += u64(1) << (p.log_places + m.k_col);
      if (keys >> planes) return "a look-up has more keys than a multiplicity can count";
    }
  }
  // Some lookup whose keys are single entries must cover every row: WF is about every entry.
  bool covered = false;
  for (size_t i = 0; i < x.lookups.size(); i++) covered = covered || wf_covers_all_rows(x, i, m);
  if (!covered) return "no look-up of the ranges covers every row of the matrix";
  const OracleShape &s = x.message.shape;
  if (s.code >= codes.base.size() || s.mask_code >= codes.mask.size()) return "the oracle message names a code that is not there";
  const size_t ell = codes.base[s.code].ell;
  const std::vector<WfBlock> blocks = wf_layout(x, ell);
  if ((u64(1) << blocks[0].bits) > ell) return "a block of planes is longer than a main row";  // the largest comes first
  if (blocks.back().row + 1 != s.main_rows) return "the oracle message has not the main rows that the multiplicities fill";
  if (wf_own_masks(x, m) > Send::free_mask_entries(s.mask_rows, codes.mask[s.mask_code].ell, s.answers)) return "the mask rows have no room for the look-ups' masks and the pad";
  if (x.stages.empty() || x.stages[0].code != s.code || x.stages[0].mask_code != s.mask_code) return "the first stage of the proximity test has not the oracle message's codes";
  return iopp_fault(codes, x.stages, s.answers);  // the code switches' (CS) messages get as many sample answers as this one: the plan's `set.samples`
}

// The first fold of WF's proximity test; its mask shares the tree of WF's oracle message.
inline FoldShape wf_first_fold(const WfPublic &x) {
  FoldShape F = iopp_fold_shape(x.stages[0], GroupLayout());
  F.mask_leaf_class = x.message.leaf_class;
  return F;
}
// The oracle messages of WF's prover message, in order. The verifier and wf_plan() read them here.
inline std::vector<OracleNotice> wf_message(const WfPublic &x) {
  std::vector<OracleNotice> m{x.message};
  if (x.choice.masks_ride) m.push_back(fold_mask_notice(wf_first_fold(x)));
  return m;
}

// The function by which LKP adds a vector on lookup i's multiplicities (mu + d variables, the plane
// in the low ones) to a linear claim. It adds the entries on live planes only, as LKP requires.
inline LkpPlanesAt wf_planes_at(const WfPublic &x, const Codes &codes, size_t i) {
  const OracleShape &s = x.message.shape;
  const unsigned log_ell = codes.base[s.code].log_ell(), row_bits = ceil_log2(s.main_rows), mu = x.lookups[i].mu;
  const OracleId o = x.message_id;
  std::vector<WfBlock> mine;
  for (const WfBlock &b : wf_layout(x, codes.base[s.code].ell))
    if (b.lookup == i) mine.push_back(b);
  return [mine, log_ell, row_bits, mu, o](Row &row, Tensor v) {
    for (const WfBlock &b : mine) {
      Tensor part = lkp_on_block(v, mu, b.planes);
      part.insert_choice(b.bits, log_ell - b.bits, b.first_place >> b.bits);  // where the block begins in its row
      part.insert_choice(0, row_bits, b.row);                                // the row's number is the low bits of a table's index
      row.add(o, Part::MAIN, std::move(part));
    }
  };
}

// Appends to g the row weights scale * values[first + i] on matrix rows first_row + i, i < count,
// and 0 elsewhere: one Tensor for each aligned block of rows, cut as in member_weights_on_rows()
// (member.h). `values` is shared, not copied. Not checked: that the rows are inside the matrix
// (wf_fault() checks the runs).
inline void wf_weights_on_rows(unsigned k_row, u64 first_row, const std::shared_ptr<const std::vector<E>> &values, u64 first, u64 count, const E &scale,
                               std::vector<Tensor> &g) {
  for (u64 at = first_row, end = first_row + count; at < end;) {
    unsigned d = 0;  // the largest aligned block that begins at 'at' and ends by 'end'
    while (d < k_row && (at & ((u64(2) << d) - 1)) == 0 && at + (u64(2) << d) <= end) d++;
    Tensor t;
    if (d == 0) {
      t.scalar = (*values)[first + (at - first_row)] * scale;
      t.insert_choice(0, k_row, at);
    } else {
      t.scalar = scale;
      t.push_dense(d);
      t.insert_choice(d, k_row - d, at >> d);
      t.set_values(values, first + (at - first_row));
    }
    g.push_back(std::move(t));
    at += u64(1) << d;
  }
}

// The pending claim of one part, made from LKP's claim on the part's keys (LkpKeys):
//   sum over the key numbers i of weights[i] * key[i] + <masks> = target,
// where weights[i] = eq(point, i) at a covered place and 0 at padding, whose share LKP has moved
// into the target. point = (z_col | z_place), the column in the low coordinates. With e[place] =
// eq(z_place, place), key = offset + sum over the digits, and since the weights eq(z_col, .) add up
// to 1:
//   sum weights * key = < g (x) eq(z_col, .), M >  +  offset * (sum of e over the covered places),
//   g[row(place, q)] = e[place] * digit_weight[q] on the covered places, added up where rows
//   coincide, 0 elsewhere.
// The covered places are read from the same runs from which wf_lookups() derives LKP's padding.
// Both sides check that the two agree, or the process ends: LKP's weights must add up to the sum of
// e over the places that the runs cover. (At a random point, two different sets of places give
// different sums except with small probability.) Each side computes all of e: 2^log_places
// elements of E, the extension field.
inline TensorRow wf_pending_row(const WfMatrix &m, const WfLookup &l, const WfPart &p, LkpKeys &&claim) {
  BLK_REQUIRE(claim.point.size() == size_t(m.k_col) + p.log_places, "WF: the claim's point has not one coordinate for each bit of a key's number");
  auto e = std::make_shared<std::vector<E>>(1, E::one());  // eq(z_place, .), coordinate t for bit t of the place
  for (unsigned t = 0; t < p.log_places; t++) {
    const E z = claim.point[m.k_col + t];
    const size_t half = e->size();
    e->resize(2 * half);
    for (size_t i = 0; i < half; i++) (*e)[half + i] = (*e)[i] * z, (*e)[i] = (*e)[i] - (*e)[half + i];
  }
  TensorRow row;
  E covered = E::zero();
  for (const WfRun &r : p.runs) {
    for (u64 i = 0; i < r.places; i++) covered += (*e)[r.first_place + i];
    for (size_t q = 0; q < l.digit_weight.size(); q++)
      wf_weights_on_rows(m.k_row(), r.first_row + q * r.stride, e, r.first_place, r.places, E::from(l.digit_weight[q]), row.g);
  }
  E handed = E::zero();  // the sum of all weights in LKP's claim
  for (const Tensor &t : claim.weights.tensors) {
    BLK_REQUIRE(!t.has_dense(), "WF: LKP's weights are product forms and single entries");
    E sum = t.scalar;
    for (unsigned i = 0; i < t.vars(); i++) sum *= t.lo(i) + t.hi(i);
    handed += sum;
  }
  for (const std::pair<u64, E> &entry : claim.weights.entries) handed += entry.second;
  BLK_REQUIRE(handed == covered, "WF: LKP's weights lie on other places than the runs cover");
  row.psi.assign(claim.point.begin(), claim.point.begin() + m.k_col);
  row.target = claim.target - covered * l.offset;
  row.masks = std::move(claim.masks);
  return row;
}

// ---- the prover's heavy part ----
// The prover's work that reads M, as an interface (WfCpu below implements it). One object serves
// one execution of the prover. Its sums and end values are hidden values: they go into LKP's calls
// and nowhere else. A call of prover() may free the LkpProver that the call before returned.
class WfHeavy {
 public:
  virtual ~WfHeavy() = default;
  // count[t]: how often row t of the lookup's table occurs among the keys at the covered places.
  // False: a key is in no row of the table.
  virtual bool multiplicities(size_t lookup, std::vector<u64> &count) = 0;
  virtual LkpProver &prover(size_t lookup) = 0;  // called once for each lookup, in order
};

// ---- the two sides ----

// The pending claims (`TensorRow`, batt.h) that WF leaves on the matrix: for each lookup in order,
// one for each of its parts in order. The caller passes them to member_first_prove() or
// member_first_verify() (member.h): BATT batches them into one claim if there are several, then
// LIN, the test on the committed table, begins. No honest execution would reveal that a caller
// dropped them. So take() returns all of them at once, and the destructor ends the process if they
// were not taken, unless the channel has rejected the proof by then.
// Every claim, its value (`target`) included, is fixed before the caller's batching challenge:
// wf_pending_row() makes it from parameters in the first query and LKP's output only, and WF has
// returned before the caller can draw.
class WfPending {
 public:
  WfPending() = default;
  WfPending(const WfPending &) = delete;
  WfPending &operator=(const WfPending &) = delete;
  ~WfPending() { BLK_REQUIRE(rows_.empty() || !channel_->ok(), "WF: the rows on the matrix were dropped: no test on the table has taken them"); }
  void put(TensorRow &&row, const Rounds &channel) { rows_.push_back(std::move(row)), channel_ = &channel; }  // the channel must outlive this
  size_t size() const { return rows_.size(); }
  const std::vector<TensorRow> &look() const { return rows_; }  // for tests and reports
  std::vector<TensorRow> take() {
    std::vector<TensorRow> all = std::move(rows_);
    rows_.clear();
    return all;
  }

 private:
  std::vector<TensorRow> rows_;
  const Rounds *channel_ = nullptr;
};
// What WF returns: the pending claims, and the numbers of its oracle message and of its group.
struct WfOutput {
  WfPending pending;
  OracleId message = NONE_ID;
  GroupId group = NONE_ID;
};

// Both sides begin with this. No challenge under which a value of the matrix is sent may be drawn
// before the index-mask message is sent and its sample answered: a mask that is chosen after its
// challenge can be fitted to any false value. If the sample is not answered yet, WF ends the
// process here, and so does LKP (tested in blocks/iop/tests/wf_test.cpp and
// blocks/iop/tests/lkp_test.cpp).
inline void wf_start_check(const Statement &st, const WfPublic &x, const WfMatrix &m, OracleId index_masks) {
  const char *fault = wf_fault(st.codes(), x, m);
  BLK_REQUIRE(!fault, fault ? fault : "");  // parameters are no part of a proof: see `wf_fault`
  BLK_REQUIRE(st.oracle(index_masks).sampled && st.oracle(index_masks).shape.main_rows == 0,
              "WF: the index-mask message is a sampled message of mask rows alone, sent before WF begins");
}

// The entry points. wf_prove() returns false, with nothing sent, if a key is in no row of its table
// or a multiplicity does not fit its planes. wf_verify() returns false if SEND, LKP or the
// proximity test rejects.
// What acceptance is to show, for the matrix that the test on the committed table pins: for every
// lookup, every covered place and every column, the key is in 0 .. 2^d - 1. wf_fault() adds that a
// lookup of single entries, with weight 1, covers every row of the matrix, that no digit has weight
// 0, and that the digits of a key are in different rows. (With weight 0, every key of a one-digit
// lookup would be the constant offset, and a key file would be accepted for any matrix.) Nothing
// checks that a key, an element of F, is the integer that it looks like: the reader concludes that
// from the parameters, since digits that are bytes and weights of a few bits cannot wrap around.
// Two of the orders on which soundness rests (the others: WfPending, wf_start_check()):
//  - The committed table is fixed before the first alpha (LKP's challenge). This rests on the
//    first query's statement item LinPublic::table, which WF does not read.
//  - The multiplicities are sent and sampled before the first alpha: one call sends, samples and
//    then draws, so a caller cannot reorder them. LKP's first error term rests on this. A prover
//    that knew alpha first would need planes of true bits with
//      prod over t of (alpha - t)^(multiplicity of row t) = (keys' root) / (padding's factor):
//    one equation in the multiplicative group of E in very many unknown bits. Solutions abound, and
//    given discrete logarithms in E, finding one is a dense knapsack problem in the exponents; the
//    claim on the number of keys adds one linear condition. Every claim of LKP would then hold. No
//    test has this prover: it cannot be run.
[[nodiscard]] inline bool wf_prove(ProverContext &P, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfHeavy &heavy, WfOutput &out);
[[nodiscard]] inline bool wf_verify(VerifierContext &V, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfOutput &out);
// wf_rows_prove() and wf_rows_verify() are WF without its proximity test. Without the test nothing
// is proved; a test of hiding on this block alone runs them
// (blocks/tests/hiding/wf_exact_test.cpp). So both take a WfRowsAlone, which only wf_prove(),
// wf_verify() and a file that defines BLK_WF_ROWS_FOR_A_TEST before it includes this header can
// construct. first_mask (output): the mask of the test's first fold, if `masks_ride`.
class WfRowsAlone {
#ifdef BLK_WF_ROWS_FOR_A_TEST
 public:
#endif
  WfRowsAlone() {}
  friend bool wf_prove(ProverContext &, const Public<WfPublic> &, const LinPublic &, OracleId, WfHeavy &, WfOutput &);
  friend bool wf_verify(VerifierContext &, const Public<WfPublic> &, const LinPublic &, OracleId, WfOutput &);
};
#ifdef BLK_WF_ROWS_FOR_A_TEST
BLK_MARK_NOT_FOR_SERVING();  // fence.h misses a #define made after base.h's first include
#endif
[[nodiscard]] inline bool wf_rows_prove(const WfRowsAlone &, ProverContext &P, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfHeavy &heavy,
                                        WfOutput &out, FoldMask &first_mask) {
  const WfPublic &x = P.ch.get(fixed);
  const WfMatrix m = wf_matrix_of(P.ch.get(table.shape));
  wf_start_check(P.st, x, m, index_masks);
  BLK_REQUIRE(x.message.shape.answers == P.ch.plan()->set.samples, "WF: the oracle message has another number of answers to a sample than the plan's setting");
  const Codes &codes = P.st.codes();
  const OracleShape &s = x.message.shape;
  Table main(1, s.main_rows, codes.base[s.code].ell);
  {
    std::vector<std::vector<u64>> count(x.lookups.size());
    for (size_t i = 0; i < x.lookups.size(); i++) {
      if (!heavy.multiplicities(i, count[i])) return false;
      BLK_REQUIRE(count[i].size() == size_t(1) << x.lookups[i].d, "WF: one multiplicity for each row of the table");
      const unsigned planes = x.lookups[i].live ? x.lookups[i].live : 1u << x.lookups[i].mu;
      for (u64 c : count[i])
        if (c >> planes) return false;
    }
    for (const WfBlock &b : wf_layout(x, main.len))
      for (u64 t = 0; t < u64(1) << x.lookups[b.lookup].d; t++)
        for (u64 j = 0; j < u64(1) << b.planes.second; j++)
          if ((count[b.lookup][t] >> (b.planes.first + j)) & 1) main.set(b.row, b.first_place + j + (t << b.planes.second), E::one());
  }
  std::vector<OracleDraft> drafts(1);
  drafts[0].shape = s, drafts[0].leaf_class = x.message.leaf_class;
  drafts[0].main = P.engine.put(main);
  drafts[0].mask = Send::fresh_mask(codes, s, P.coins), drafts[0].mask_draw = P.coins.last_draw();
  if (x.choice.masks_ride) {
    const FoldShape F = wf_first_fold(x);
    first_mask = fold_mask_make(F, P);
    drafts.push_back(fold_mask_draft(codes, F, first_mask));
  }
  const std::vector<OracleId> ids = send_prove(P, std::move(drafts));
  // A self-check, once, right after sending: the message's mask rows (SEND checks its pads in them
  // itself) and their random tail (random coefficients encoded with a row's entries) must be the
  // values that were drawn; `mask_draw` above records the draw. The message goes into a key file,
  // which is public for good, and its masks cover values that are computed from the model's
  // weights. What the check misses, and when it compares nothing: not_as_drawn(), send.h.
  BLK_REQUIRE_NAMED(not_as_drawn(P, ids[0]) == 0, "own-test-ranges-mask", "internal error (own-test-ranges-mask): values of the mask rows of the key file's message on the ranges are not the random values that were drawn for them. The program or the machine has a fault; nothing of this run may be used");
  // Of the first fold's mask, sent with it, only the random tail is compared here: the mask's table
  // is computed from several draws, which FOLD compares itself (fold_mask_as_drawn(), fold.h).
  if (x.choice.masks_ride) BLK_REQUIRE_NAMED(not_as_drawn(P, ids[1]) == 0, "own-test-ranges-mask", "internal error (own-test-ranges-mask): values of the random tail of the fold mask that goes with the key file's message on the ranges are not the random values that were drawn for it. The program or the machine has a fault; nothing of this run may be used");
  BLK_REQUIRE(ids[0] == x.message_id, "WF: the oracle message has another number on this channel than the parameters say");
  if (x.choice.masks_ride) first_mask.sent_as(ids[1]);
  out.message = ids[0];
  out.group = P.st.new_group("ranges", s.code, s.mask_code);
  P.st.join(out.group, out.message);

  const Public<std::vector<LkpPublic>> lookups = derive(fixed, table.shape, [](const WfPublic &r, const LinShape &t) { return wf_lookups(r, wf_matrix_of(t)); });
  for (size_t i = 0; i < x.lookups.size(); i++) {
    LkpOutput left;
    lkp_prove(P, element(lookups, i), heavy.prover(i), wf_planes_at(x, codes, i), out.message, out.group, left, index_masks);
    for (size_t part = 0; part < x.lookups[i].parts.size(); part++)
      out.pending.put(wf_pending_row(m, x.lookups[i], x.lookups[i].parts[part], left.keys.take(part)), P.ch);
  }
  return true;
}
[[nodiscard]] inline bool wf_prove(ProverContext &P, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfHeavy &heavy, WfOutput &out) {
  FoldMask first_mask;
  if (!wf_rows_prove(WfRowsAlone(), P, fixed, table, index_masks, heavy, out, first_mask)) return false;
  const WfPublic &x = P.ch.get(fixed);
  iopp_prove(P, out.group, x.stages, x.choice, std::move(first_mask));
  release_oracles(P, {out.message});
  return true;
}

[[nodiscard]] inline bool wf_rows_verify(const WfRowsAlone &, VerifierContext &V, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfOutput &out,
                                         FoldMask &first_mask) {
  const WfPublic &x = V.ch.get(fixed);
  const WfMatrix m = wf_matrix_of(V.ch.get(table.shape));
  wf_start_check(V.st, x, m, index_masks);
  BLK_REQUIRE(x.message.shape.answers == V.ch.plan()->set.samples, "WF: the oracle message has another number of answers to a sample than the plan's setting");
  const OracleShape &s = x.message.shape;
  std::vector<OracleId> ids;
  if (!send_verify(V, wf_message(x), ids)) return false;
  BLK_REQUIRE(ids[0] == x.message_id, "WF: the oracle message has another number on this channel than the parameters say");
  if (x.choice.masks_ride) first_mask.sent_as(ids[1]);
  out.message = ids[0];
  out.group = V.st.new_group("ranges", s.code, s.mask_code);
  V.st.join(out.group, out.message);

  const Public<std::vector<LkpPublic>> lookups = derive(fixed, table.shape, [](const WfPublic &r, const LinShape &t) { return wf_lookups(r, wf_matrix_of(t)); });
  for (size_t i = 0; i < x.lookups.size(); i++) {
    LkpOutput left;
    if (!lkp_verify(V, element(lookups, i), wf_planes_at(x, V.st.codes(), i), out.message, out.group, left, index_masks)) return false;
    for (size_t part = 0; part < x.lookups[i].parts.size(); part++)
      out.pending.put(wf_pending_row(m, x.lookups[i], x.lookups[i].parts[part], left.keys.take(part)), V.ch);
  }
  return true;
}
// Contract on the plan and the channel: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool wf_verify(VerifierContext &V, const Public<WfPublic> &fixed, const LinPublic &table, OracleId index_masks, WfOutput &out) {
  FoldMask first_mask;
  if (!wf_rows_verify(WfRowsAlone(), V, fixed, table, index_masks, out, first_mask)) return false;
  const WfPublic &x = V.ch.get(fixed);
  return iopp_verify(V, out.group, x.stages, x.choice, std::move(first_mask));
}

// ---- the heavy part on the CPU, by the definitions ----
// WfCpu computes each key from M's entries, at(row, column), and holds all keys of one lookup in
// memory at once.
class WfCpu : public WfHeavy {
 public:
  WfCpu(const WfPublic &x, const WfMatrix &m, std::function<F(u64 row, u64 column)> at) : x_(x), m_(m), fixed_(wf_lookups(x, m)), at_(std::move(at)) {}
  // All keys of a lookup, part after part, in the order of their numbers; a place of padding holds
  // WfLookup::padding_key. `padding`, if given, gets one flag for each key: true at padding.
  std::vector<E> keys(size_t lookup, std::vector<bool> *padding = nullptr) const {
    const WfLookup &l = x_.lookups[lookup];
    std::vector<E> k;
    const u64 columns = u64(1) << m_.k_col;
    for (const WfPart &p : l.parts) {
      const size_t first = k.size();
      k.resize(first + (columns << p.log_places), E::from(l.padding_key));
      if (padding) padding->resize(k.size(), true);
      for (const WfRun &r : p.runs)
        for (u64 i = 0; i < r.places; i++)
          for (u64 c = 0; c < columns; c++) {
            F key = l.offset;
            for (size_t q = 0; q < l.digit_weight.size(); q++) key += l.digit_weight[q] * at_(r.first_row + i + q * r.stride, c);
            k[first + c + (r.first_place + i) * columns] = E::from(key);
            if (padding) (*padding)[first + c + (r.first_place + i) * columns] = false;
          }
    }
    return k;
  }
  bool multiplicities(size_t lookup, std::vector<u64> &count) override {
    std::vector<bool> padding;
    const std::vector<E> k = keys(lookup, &padding);
    count.assign(size_t(1) << x_.lookups[lookup].d, 0);
    for (size_t i = 0; i < k.size(); i++) {
      if (padding[i]) continue;
      u64 row = k[i].c[0].v;
      if (row >> x_.lookups[lookup].d) return false;
      count[row]++;
    }
    return true;
  }
  LkpProver &prover(size_t lookup) override {
    std::vector<u64> count;
    BLK_REQUIRE(multiplicities(lookup, count), "WF: a key in no row of its table");
    const LkpPublic &p = fixed_[lookup];
    std::vector<E> bits(size_t(1) << (p.mu + p.d), E::zero());
    for (size_t t = 0; t < count.size(); t++)
      for (size_t j = 0; j < p.planes(); j++)
        if ((count[t] >> j) & 1) bits[j + (t << p.mu)] = E::one();
    now_.reset(new LkpCpu(p, keys(lookup), std::move(bits)));
    return *now_;
  }

 private:
  const WfPublic x_;
  const WfMatrix m_;
  const std::vector<LkpPublic> fixed_;
  std::function<F(u64, u64)> at_;
  std::unique_ptr<LkpCpu> now_;
};

}  // namespace blk
