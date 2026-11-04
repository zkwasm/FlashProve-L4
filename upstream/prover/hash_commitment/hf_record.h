// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The commitment record of one weight table: nine header words, six of them the table's shape, then
// four digests. The table is the anchor (a commitment's first weight table) or a leaf table (a
// further table of the same weights). Host code only.
// The committer, `cp_commit`, builds a record in memory for each table, checks its tier word
// (the third header word) and prints its digest h_card. It writes no record to a file, and no key
// derivation or hash chain takes a record's digest.
// Identifiers say `card` for the record, `copy` for a leaf table, `once` for the committer's
// one-time test on the anchor, and T_F, RT_F or RT for the root of the table's Merkle tree.
// DW, a digest's length in 32-bit words, and the tree hash rf::TH() must be declared before this
// header, as prover/weights_commitment/common.cuh does. The HF_ constants come from
// prover/hash_commitment/hf_format.h.
//
// As a file, wcard.bin: the record, HF_RECORD_WORDS words, then RT_F, HF_TF_WORDS words. Public.
#pragma once
#include "hash_commitment/hc_create_file.h" // `hc::write_a_new_file`
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hf_params.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
namespace rf {
static_assert(HF_DW == DW, "hf_format.h's digest length is the tree's");
static_assert(HF_RECORD_FILE_WORDS == HF_RECORD_WORDS + HF_TF_WORDS, "the file is the record, then T_F");
// On an empty object call only read(), read_bytes() or write(): the other members check no sizes.
struct HfRecord {
  std::vector<uint32_t> w;       // the record: HF_RECORD_WORDS words, or empty
  std::vector<uint32_t> tw;      // T_F = RT_F: HF_TF_WORDS words, or empty
  const char *refusal = nullptr; // name of the check that failed in `read` or `read_bytes`, or null
  HfRecord() {}
  // Builds an anchor's record. fails() accepts no `tier` but HF_TIER_HIDDEN. Each of
  // unit_list_digest, h_par (blk::configuration_digest() of the configuration files) and RT has DW words.
  // The fourth digest, h_once, stays zero: `cp_commit` does not call set_h_once().
  HfRecord(uint32_t tier, const ::hf::HfParams &P, const uint32_t *unit_list_digest, const uint32_t *h_par,
           const uint32_t *RT)
      : w(HF_RECORD_WORDS, 0), tw(HF_TF_WORDS, 0) {
    memcpy(tw.data(), RT, 4 * DW);
    w[HF_REC_MAGIC] = HF_RECORD_MAGIC;
    w[HF_REC_VERSION] = hf_version_word();
    w[HF_REC_TIER] = tier;
    w[HF_REC_LK] = (uint32_t)P.lk;
    w[HF_REC_M0] = (uint32_t)P.m0;
    w[HF_REC_FB] = (uint32_t)P.b;
    w[HF_REC_FLN] = (uint32_t)P.ln;
    w[HF_REC_FI0] = (uint32_t)P.i0;
    w[HF_REC_FNM] = (uint32_t)P.nm;
    memcpy(at(HF_REC_UNIT_LIST), unit_list_digest, 4 * DW);
    memcpy(at(HF_REC_H_PAR), h_par, 4 * DW);
    memcpy(at(HF_REC_TH_TF), TH(tw).data(), 4 * DW);
  }
  void set_h_once(const uint32_t h_once[DW]) { memcpy(at(HF_REC_H_ONCE), h_once, 4 * DW); }
  // Makes the record a leaf table's. Call it before h_0(), which covers the changed first word.
  void set_copy() { w[HF_REC_MAGIC] = HF_RECORD_COPY_WORD; }
  const uint32_t *at(int offset) const { return w.data() + offset; }
  uint32_t *at(int offset) { return w.data() + offset; }
  uint32_t word(int offset) const { return w[offset]; }
  const uint32_t *RT() const { return tw.data(); } // RT_F, DW words
  // Level 0: the anchor's record; 1: a leaf table's.
  bool is_copy() const { return level() > 0; }
  int level() const { return hf_record_level(word(HF_REC_MAGIC)); } // 0 or 1; -1: no record's first word
  // h_0 = TH(the words before h_once).
  void h_0(uint32_t out[DW]) const {
    memcpy(out, TH(std::vector<uint32_t>(w.begin(), w.begin() + HF_REC_H_ONCE)).data(), 4 * DW);
  }
  // h_card = TH(the record).
  void h_card(uint32_t out[DW]) const { memcpy(out, TH(w).data(), 4 * DW); }
  // Takes wcard.bin's bytes. Returns false, with the object empty, unless `file` is non-null and
  // has exactly HF_RECORD_FILE_WORDS words. fails() checks their values. `cp_commit` calls
  // read_bytes() and fails() only in its code for adding leaf tables to an existing commitment, a
  // use that it rejects (`cp-commit-first-form-alone`).
  bool read_bytes(const uint8_t *file, size_t bytes) {
    w.clear(), tw.clear();
    refusal = "hc-record-length";
    if (!file || bytes != (size_t)HF_RECORD_FILE_WORDS * 4) return false;
    refusal = nullptr;
    w.resize(HF_RECORD_WORDS), tw.resize(HF_TF_WORDS);
    memcpy(w.data(), file, (size_t)HF_RECORD_WORDS * 4);
    memcpy(tw.data(), file + (size_t)HF_RECORD_WORDS * 4, (size_t)HF_TF_WORDS * 4);
    return true;
  }
  // The same from a file. Only prover/hash_commitment/tests/hf_record_test.cpp calls read() and
  // write().
  bool read(const std::string &path) {
    w.clear(), tw.clear();
    refusal = "input-unreadable-card";
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t file[(HF_RECORD_FILE_WORDS + 1) * 4];
    const size_t bytes = fread(file, 1, sizeof file, f);
    fclose(f);
    return read_bytes(file, bytes);
  }
  // Checks every field except h_once; the arguments are the caller's own values. Returns null, or
  // the name of the first check that fails. The root is compared only with the record's own digest
  // of it. Records of both levels pass; level() tells them apart.
  const char *fails(const ::hf::HfParams &P, const uint32_t *unit_list_digest, const uint32_t *h_par) const {
    if (level() < 0 || word(HF_REC_VERSION) != hf_version_word())
      return "hc-record-head";
    if (word(HF_REC_TIER) != HF_TIER_HIDDEN) return "hc-record-tier";
    if (word(HF_REC_LK) != (uint32_t)P.lk || word(HF_REC_M0) != (uint32_t)P.m0 || word(HF_REC_FB) != (uint32_t)P.b ||
        word(HF_REC_FLN) != (uint32_t)P.ln || word(HF_REC_FI0) != (uint32_t)P.i0 || word(HF_REC_FNM) != (uint32_t)P.nm)
      return "hc-record-shape";
    if (memcmp(at(HF_REC_UNIT_LIST), unit_list_digest, 4 * DW)) return "hc-record-unit-list";
    if (memcmp(at(HF_REC_H_PAR), h_par, 4 * DW)) return "hc-record-hpar";
    if (memcmp(at(HF_REC_TH_TF), TH(tw).data(), 4 * DW)) return "hc-record-tw";
    return nullptr;
  }
  std::vector<uint32_t> file_words() const {
    std::vector<uint32_t> words(w);
    words.insert(words.end(), tw.begin(), tw.end());
    return words;
  }
  // Writes the file, which must not exist. Returns false if the object is empty or the file cannot
  // be created or fully written.
  bool write(const std::string &path) const {
    if (w.size() != (size_t)HF_RECORD_WORDS || tw.size() != (size_t)HF_TF_WORDS) return false;
    const std::vector<uint32_t> words = file_words();
    return ::hc::write_a_new_file(path, words.data(), words.size() * 4);
  }
};
// The unit list, whose tree hash is in the record: five words for each unit (kind X, layer l, rows,
// row offset, column bits ncb). A unit (`UnitW`) is the rows of the weight matrix W that store the
// tensors of one kind and layer.
template <class Layout> static std::vector<uint32_t> hf_unit_list_words(const Layout &Y) {
  std::vector<uint32_t> list;
  for (auto &u : Y.U)
    for (size_t v : {(size_t)u.X, (size_t)u.l, (size_t)u.rows, (size_t)u.off, (size_t)u.ncb})
      list.push_back((uint32_t)v);
  return list;
}
} // namespace rf
