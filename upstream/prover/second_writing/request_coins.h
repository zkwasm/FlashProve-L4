// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives the private random values (`coins`) of one request's proof from the batch's process key
// (`process_key`), as part of a second implementation: the program cp_prove compares each
// sub-proof's slot pads with slot_pads(), which prover/prover/slot_pads_derive.h calls. A sub-proof
// covers one part of the forward pass; its slot pads (mask entries) mask what it sends.
// Nothing outside this file calls the other methods or crossing_scalar_numbers(), so no test runs
// them. The constructor and the methods leave copies of keys on the stack:
// prover/common/stack_scrub.h states what a caller must do.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "coin_streams.h"

namespace coins {

using Salt = std::array<std::uint64_t, kSaltCells>;
// An element of the cubic extension field, by its coordinates in 1, x, x^2.
using Ext = std::array<std::uint64_t, 3>;

template <class F = ChaChaBlock>
class BasicRequestCoins {
 public:
  // `process_key` is the batch's, `slot` the request's index within its batch, `version` the
  // format's version word.
  BasicRequestCoins(const Key& process_key, Word version, Word slot, F f = F())
      : coins_(process_key, request_proof_context(version, slot), f) {}

  // Any coin of the context, by stream and index.
  const BasicCoins<F>& coins() const { return coins_; }

  // --- Row commitments with a serial number -----------------------------------

  // The mk blinding coordinates of a data row (class N, B or W): out[c] is the row's
  // coordinate Ks + c. A row of class C has no coins: it repeats those of the row
  // that it copies. `num`, `row`: see serial_stream() (coin_streams.h).
  void blinding_coordinates(Word num, Word row, std::uint64_t mk, std::uint64_t* out) const {
    coins_.cells(serial_stream(num, kKindBlindingCoordinates, row), 0, mk, out);
  }

  // Coordinates [first, first + count) of a blinding row (class M).
  void blinding_row(Word num, Word row, std::uint64_t first, std::uint64_t count,
                    std::uint64_t* out) const {
    coins_.cells(serial_stream(num, kKindBlindingRow, row), first, count, out);
  }

  // The salt of the tree leaf at `position`: five field elements.
  Salt leaf_salt(Word num, std::uint64_t position) const {
    Salt s;
    coins_.cells(serial_stream(num, kKindSalts, 0), kSaltCells * position, kSaltCells, s.data());
    return s;
  }

  // The salts of tree leaves [first, first + count), five elements each.
  void leaf_salts(Word num, std::uint64_t first, std::uint64_t count, std::uint64_t* out) const {
    coins_.cells(serial_stream(num, kKindSalts, 0), kSaltCells * first, kSaltCells * count, out);
  }

  // The salt at the start of the request's transcript.
  Salt transcript_salt() const { return leaf_salt(kNumTranscriptSalt, 0); }

  // Data coordinates [first, first + count) of row `row` of P, the pad commitment.
  void pad_row(Word row, std::uint64_t first, std::uint64_t count, std::uint64_t* out) const {
    coins_.cells(serial_stream(kNumP, kKindDerivedRow, row), first, count, out);
  }

  // The pad of masked scalar k: elements 3 k to 3 k + 2 of P, where element e is data coordinate
  // e mod kdp of row e / kdp.
  Ext masked_scalar_pad(std::uint64_t k, std::uint64_t kdp) const {
    if (kdp == 0) throw std::invalid_argument("coins: kdp, the number of data coordinates of a row of P, is 0");
    Ext pad;
    for (unsigned i = 0; i < 3; ++i) {
      const std::uint64_t e = 3 * k + i, row = e / kdp;
      if (row >> 14) throw std::invalid_argument("coins: the masked scalar would lie in row 2^14 of P or beyond");
      pad[i] = coins_.cell(serial_stream(kNumP, kKindDerivedRow, Word(row)), e % kdp);
    }
    return pad;
  }

  // --- The pads h of a sub-proof's message slots -------------------------------

  // The pad of message slot y < ns (not checked) of sub-proof gid.
  Ext slot_pad(Word gid, std::uint64_t ns, std::uint64_t y) const {
    Ext pad;
    for (unsigned kappa = 0; kappa < 3; ++kappa)
      pad[kappa] = coins_.cell(slot_pad_stream(gid), slot_pad_cell(ns, kappa, y));
    return pad;
  }

  // All ns pads, 3 ns values: out[kappa * ns + y] is coordinate kappa of message slot y's pad.
  void slot_pads(Word gid, std::uint64_t ns, std::uint64_t* out) const {
    coins_.cells(slot_pad_stream(gid), 0, 3 * ns, out);
  }

  // --- The crossing-pad commitment CP ------------------------------------------

  // Entry y < kc (not checked) of one of CP's three rows; `first_plane` is kPlaneHC, kPlaneRC0 or
  // kPlaneRC. The entries y < N_F of hC (N_F: the number of sub-proofs with weight claims) are not
  // coins of this stream: use crossing_pad().
  Ext crossing_row_entry(unsigned first_plane, std::uint64_t kc, std::uint64_t y) const {
    Ext e;
    for (unsigned kappa = 0; kappa < 3; ++kappa)
      e[kappa] = coins_.cell(rows_stream(kBaseCP), plane_cell(kc, first_plane + kappa, y));
    return e;
  }

  // hC[f] for f < N_F: a copy of the pad of masked scalar k_f
  // (k_f: see crossing_scalar_numbers()).
  Ext crossing_pad(std::uint64_t k_f, std::uint64_t kdp) const {
    return masked_scalar_pad(k_f, kdp);
  }

  // The salt of CP's tree leaf at `position` < n_c, CP's code length (not checked).
  Salt crossing_salt(std::uint64_t n_c, std::uint64_t position) const {
    Salt s;
    for (unsigned x = 0; x < kSaltCells; ++x)
      s[x] = coins_.cell(salts_stream(kBaseCP), salt_cell(n_c, x, position));
    return s;
  }

 private:
  BasicCoins<F> coins_;
};

using RequestCoins = BasicRequestCoins<>;

// One segment's input to crossing_scalar_numbers(): how many sum-check rounds close its long and
// its short stack.
struct SegmentRounds {
  bool has_weight_claims;  // a feed-forward segment or the head
  unsigned long_rounds;    // lR + lK
  unsigned short_rounds;   // lR2 + lK2
};

// Returns the number k(f) of the masked scalar fv'_f of every sub-proof f with weight claims, in
// the segments' order. Masked scalar 0 precedes the segments. Each segment then has, in this
// order: fv'_f if it has one, share', two for each long round, s', two for each short round, s2'.
// `total`, if given, receives NPW, the number of all masked scalars.
inline std::vector<std::uint64_t> crossing_scalar_numbers(const std::vector<SegmentRounds>& segments,
                                                          std::uint64_t* total = nullptr) {
  std::vector<std::uint64_t> k_of_f;
  std::uint64_t k = 1;
  for (const SegmentRounds& s : segments) {
    if (s.has_weight_claims) k_of_f.push_back(k++);
    k += 1 + 2 * std::uint64_t(s.long_rounds) + 1 + 2 * std::uint64_t(s.short_rounds) + 1;
  }
  if (total) *total = k;
  return k_of_f;
}

}  // namespace coins
