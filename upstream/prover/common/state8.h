// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The parameter file's (policy.txt) target and its bits of proof of work before draws of codeword
// positions, in the global `now`, for the host code of cp_commit and cp_prove. The target is the
// number of bits that the programs' error accounting must reach in every round, proof of work
// counted. set() reads both from the frame's lines (blocks/frame/parameters.h). `now` has no lock:
// call set() before threads read it.
#pragma once
#include "hash_commitment/hf_format.h" // the version word, hf_version_word()
#include <cstdint>
#include <map>
#include <string>

// The parameter file's target and bits of proof of work, and the bit counts derived from them.
namespace s8 {
// Suffixes below: _r, a request's own rounds; _s, the rounds that a batch's requests share; _c, the
// committer's rounds, among them the LINK proof, which ties a leaf table (a further weight table)
// to the anchor (the first).
struct Values {
  int hb = 0, fhb = 0; // targets in bits (fhb: for the weight table's columns); both are line `sec`
  int pow = 0;         // bits of proof of work, _r: 1st number of the line `work positions`
  int pows = 0;        // _s: its 2nd number
  int powc = 0;        // _c: its 3rd number
};
inline Values now;

inline uint32_t ver() { return hf_version_word(); } // the value that the line `version` must have

// The bits that set a list's number of positions: a target less the drawing round's proof of work.
// Of hb_r() to g_member(), only hb_r() is called outside this file (in prover/prover/); HfParams
// (prover/hash_commitment/hf_opening_params.h) computes the rest itself.
inline int hb_r() { return now.hb - now.pow; }
inline int hb_s() { return now.hb - now.pows; }
inline int hb_c() { return now.hb - now.powc; }
inline int fhb_c() { return now.fhb - now.powc + 1; } // +1: the LINK proof draws two lists at once
// For HfParams::ZV, which cp_commit and cp_prove leave unfilled: the shape of a table to be read in
// shared and in requests' own rounds, so the smaller proof of work applies.
inline int hb_zv() { return now.hb - (now.pow < now.pows ? now.pow : now.pows); }
// `once`: the committer's one-time opening, which uses powc; otherwise a batch's opening.
inline int hb_open(bool once) { return once ? hb_c() : hb_s(); }     // the shared message
inline int hb_member(bool once) { return once ? hb_c() : hb_r(); }   // a request's second message
inline int hb_zv(bool once) { return once ? hb_c() : hb_zv(); }
inline int g_open(bool once) { return once ? now.powc : now.pows; }  // g_: bits of proof of work
inline int g_member(bool once) { return once ? now.powc : now.pow; }

// Fills `now` from `frame`, a blk::Parameters as parse() (prover/common/parameter_file.h) accepted
// it. set() is a template so that this file need not include blocks/frame/parameters.h: setting.h
// includes this file ahead of the blocks' headers. parse() and blk::read_parameters() have bounded
// the target and the bits of proof of work. Returns the empty string, or leaves `now` unchanged
// and returns why the file is rejected, beginning with a name and a colon.
template <class Frame> inline std::string set(const Frame &frame) {
  if (hf_version_older(frame.version)) return std::string("version-word-other-format: ") + HF_VERSION_OLDER_TEXT;
  if (!hf_version_admitted(frame.version))
    return "version-word: the line version must be " + std::to_string(HF_VERSION_WORD) +
           ", the word of this build, whose digests have " + std::to_string(HF_DW) + " words";
  now.hb = now.fhb = (int)frame.sec;
  now.pow = (int)frame.work[3][0], now.pows = (int)frame.work[3][1], now.powc = (int)frame.work[3][2];
  return "";
}
} // namespace s8
