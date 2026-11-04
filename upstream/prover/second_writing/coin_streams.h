// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Numbers in the label (coin_generator.h) of a private random value ("coin"): blinding context
// (one proof or message), stream, and cell (a value's index in its stream). No verifier sees
// them. This header is part of a second implementation of the generator of the programs cp_commit
// and cp_prove (see coin_generator.h): the numbers matter because both implementations must derive
// the same values from one key.
//
// The prover program cp_prove calls only request_proof_context() and slot_pad_stream(), through
// prover/second_writing/request_coins.h. Only tests run the other functions, except five that nothing runs:
// num_of_level() and gid_of_segment() have no caller; coordinate_cell(), plane_cell() and
// salt_cell() have only callers that nothing calls.
// In the other contexts the programs run only their own generator, and only as a self-test,
// against a table that the test program coins_known_maker prints with the functions below.
#pragma once

#include <cstdint>
#include <stdexcept>

#include "coin_generator.h"

namespace coins {

// ---------------------------------------------------------------------------
// Blinding contexts. Label words: counter (4, 5), version (10), object (11), member (12).
// ---------------------------------------------------------------------------

// Of these objects the programs draw a proof's values in kObjectRequestProof alone. Their other
// coins have objects that this file does not list: 2 and 4 in cp_prove, 5 and 10 in cp_commit
// (prover/hiding/long_rows_words.h).
enum Object : Word {
  kObjectRequestProof = 1,
  kObjectSharedMessage = 6,
  kObjectFirstMessage = 7,
  kObjectSecondMessage = 8,
  kObjectRangeFile = 9,
  kObjectOnceSharedMessage = 22,
  kObjectOnceFirstMessage = 23,
  kObjectOnceSecondMessage = 24,
  kObjectLink = 38,
};

// A batch's contexts, under its process key (the key from which a run's private random values are
// derived: `process_key` in coin_generator.h). The messages are those of the weight table's
// opening. `slot` is the request's index in its batch, `leaf_indices` the number of leaves of
// the batch's binding tree (a Merkle tree; each request has a leaf).
constexpr Context request_proof_context(Word version, Word slot) {
  return {slot, version, kObjectRequestProof, slot};
}
constexpr Context first_message_context(Word version, Word slot) {
  return {std::uint64_t(1) + slot, version, kObjectFirstMessage, slot};
}
constexpr Context shared_message_context(Word version, Word leaf_indices) {
  return {std::uint64_t(leaf_indices) + 1, version, kObjectSharedMessage, 0};
}
constexpr Context second_message_context(Word version, Word leaf_indices, Word slot) {
  return {std::uint64_t(leaf_indices) + 2 + slot, version, kObjectSecondMessage, slot};
}

// Four contexts named for the committer's proofs in the key file, under one process key: the range
// file (a range proof on the weights) and one opening. Word 10 is the range file's format word in
// all four. cp_commit draws the key file's values in object 5 instead.
constexpr Context range_file_context(Word range_format) {
  return {1, range_format, kObjectRangeFile, 0};
}
constexpr Context once_first_message_context(Word range_format) {
  return {2, range_format, kObjectOnceFirstMessage, 0};
}
constexpr Context once_shared_message_context(Word range_format) {
  return {3, range_format, kObjectOnceSharedMessage, 0};
}
constexpr Context once_second_message_context(Word range_format) {
  return {4, range_format, kObjectOnceSecondMessage, 0};
}

// The LINK proof (`link` in identifiers) ties a leaf table (a further committed table of the same
// weights) to the anchor (the first). It has one context here, under the key of one creation of the
// proof: coins_known_maker uses link_process_key() (prover/tests/coins_second_text/process_keys.h).
// cp_commit draws a LINK proof's values in object 10 instead.
constexpr Context link_context(Word version) { return {1, version, kObjectLink, 0}; }

// ---------------------------------------------------------------------------
// First numbering: streams formed from a serial number, a kind and a row.
// For a request's row commitments: the pad commitment P, its blinding commitment PM, X_j, T(l,i).
// ---------------------------------------------------------------------------

// Ks: the index of a row's first blinding coordinate.
enum StreamKind : Word {
  kKindBlindingCoordinates = 0,  // cell c is the row's coordinate Ks + c
  kKindBlindingRow = 1,          // cell j is the row's coordinate j
  kKindSalts = 2,                // row 0; cell 5 I + x is element x of position I
  kKindDerivedRow = 3,           // cell y is data coordinate y (rows of P only)
};

// `num` is the commitment's serial number, `row` the row's index among its real rows (here: all but
// its zero rows), from 0. Both must be below 2^14: a larger one would name another stream.
constexpr Word serial_stream(Word num, StreamKind kind, Word row) {
  return (num >> 14) != 0 || (row >> 14) != 0
             ? throw std::invalid_argument("coins: a serial number or a row is 2^14 or more and names no stream")
             : kSerialStreams + (num << 16) + (Word(kind) << 14) + row;
}

constexpr Word kNumTranscriptSalt = 0;  // no row commitment has this number
constexpr Word kNumP = 1;
constexpr Word kNumPM = 2;

// X_j: the commitment of level j = 1 .. 69 (not checked) of a request's final recursion; from
// j = 70 on, 2 + j is the number of a T(l,i).
// Every level's tree takes its leaf salts from coins, though the levels
// from 2 up add no blinding coordinates or rows. Nothing calls num_of_level().
constexpr Word num_of_level(Word level) { return 2 + level; }
// T(l,i): commitment i = 1 .. 8 (not checked) of segment l = 1, 2, ..
// A segment: one sub-proof's part of the proof.
constexpr Word num_of_segment_commitment(Word segment, Word i) {
  return 64 + 8 * segment + (i - 1);
}

// Cells in one tree leaf's salt.
constexpr unsigned kSaltCells = 5;

// ---------------------------------------------------------------------------
// Second numbering: streams numbered from a base. A call gives four cells: cells_per_call().
// ---------------------------------------------------------------------------

// Returns gid, the public identifier of the sub-proof of segment l, in the form with
// attention and ng layer groups: group g < ng has the feed-forward segment l = 2 g + 1
// and the attention segment l = 2 g + 2; the output head is l = 2 ng + 1. Other l are
// not checked. Nothing calls gid_of_segment().
constexpr Word gid_of_segment(Word segment, Word ng) {
  return segment == 2 * ng + 1 ? 1 + ng                         // the head
         : segment % 2 == 1    ? 1 + (segment - 1) / 2          // feed-forward
                               : 2 + ng + (segment - 2) / 2;    // attention
}

// The one-time pads h (mask entries) of sub-proof gid: an element of E, the cubic extension field,
// for each of its ns message slots. slot_pad_cell(): coordinate kappa < 3 of pad y.
constexpr Word slot_pad_stream(Word gid) { return 4 * (4096 + 64 * gid) + 1; }
constexpr std::uint64_t slot_pad_cell(std::uint64_t ns, unsigned kappa, std::uint64_t y) {
  return kappa * ns + y;
}

// A commitment's streams, from its base: blinding coordinates, blinding rows, salts.
constexpr Word coordinates_stream(Word base) { return base; }
constexpr Word rows_stream(Word base) { return base + 1; }
constexpr Word salts_stream(Word base) { return base + 2; }

// Blinding coordinate c < mk of real row rho. Here real rows are rows over the base field: data
// rows first, then dummy rows.
constexpr std::uint64_t coordinate_cell(std::uint64_t mk, std::uint64_t rho, std::uint64_t c) {
  return rho * mk + c;
}
// Entry j < kc of a plane. A row of elements of E is three consecutive planes,
// coordinates kappa = 0, 1, 2.
constexpr std::uint64_t plane_cell(std::uint64_t kc, unsigned plane, std::uint64_t j) {
  return plane * kc + j;
}
// Salt element x < 5 of position I < n, where n is the code length.
constexpr std::uint64_t salt_cell(std::uint64_t n, unsigned x, std::uint64_t position) {
  return x * n + position;
}

// The crossing-pad commitment CP of a request's proof (object 1). It has no
// data rows, so no coin comes from the base stream itself.
constexpr Word kBaseCP = 4 * 4096 + 8;  // 16,392

// The bases of the opening's messages and of the committer's contexts, above those of gid < 60.
constexpr Word kBaseOpening = 4 * (4096 + 64 * 60);  // 31,744
constexpr Word kBaseS = kBaseOpening;                // S, S_i, S_R, LINK's S
constexpr Word kBaseUW = kBaseOpening + 4;           // UW_i
constexpr Word kBaseRA = kBaseOpening + 8;           // RA (range file)
constexpr Word kBaseZV = kBaseOpening + 16;          // ZV; ZF_i under object 8
constexpr Word kBaseBW = kBaseOpening + 68;          // BW (range file)
constexpr Word kBaseCPR = kBaseOpening + 80;         // CP_R (range file)
constexpr Word kBaseZU = kBaseOpening + 88;          // ZU; ZUA in LINK
constexpr Word kBaseZUC = kBaseOpening + 92;         // ZUC (LINK)

// Two single streams of a shared message's context: the binding tree's leaf salts and spare words,
// and the permutation of slots to leaves. Under the range file's context or the LINK proof's, the
// first gives the transcript's salt.
constexpr Word kStreamBindingWords = kBaseOpening + 20;  // 31,764
constexpr Word kStreamPermutation = kBaseOpening + 24;   // 31,768

// The first plane of each blinding row in rows_stream(base).
constexpr unsigned kPlaneB = 0;    // B_X of a table
constexpr unsigned kPlaneH = 3;    // h_X of a table
constexpr unsigned kPlaneHS = 0;   // hS of a pad commitment; h_i of ZF_i
constexpr unsigned kPlaneRS = 3;   // rS of a pad commitment
constexpr unsigned kPlaneHC = 0;   // hC of a crossing-pad commitment
constexpr unsigned kPlaneRC0 = 3;  // rC0
constexpr unsigned kPlaneRC = 6;   // rC

}  // namespace coins
