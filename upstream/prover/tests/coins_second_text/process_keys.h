// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives process keys and mask seeds from a commitment's 32-byte seed, for the known-answer tables
// that prover/tests/coins_known_maker.cpp prints. The programs cp_commit and cp_prove derive theirs
// from other inputs (prover/common/coin_keys.h; prover/second_writing/schedule_keys.h is its second
// implementation). A process key is the ChaCha20 key from which one run's private random values
// (`coins` in identifiers) are derived. Nothing here erases a key or checks a digest's length.
//
// The anchor is a commitment's first weight table. A leaf table (`leaf`; not a Merkle leaf) is a
// further table of the same weights with its own masks; a LINK proof (`link`) ties it to the
// anchor. h_card is the digest of a table's commitment record, h_0 that of the record without its
// last digest. A digest has DW words.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "chacha_block.h"
#include "format_hash.h"

namespace coins {

constexpr Word kWordCPSD = 0x44535043;  // "CPSD"
constexpr Word kWordCPLF = 0x464C5043;  // "CPLF"
constexpr Word kWordCPBK = 0x4B425043;  // "CPBK"
constexpr Word kWordCPKW = 0x574B5043;  // "CPKW"

// Absorbs words into a key, four at a time: key = chacha_key(key, four words). chacha_key()
// (chacha_block.h) gives words 0 to 7 of a ChaCha20 block. An incomplete last group is padded with
// zero words.
inline Key absorb_in_fours(Key key, const Word* words, std::size_t count) {
  for (std::size_t at = 0; at < count; at += 4) {
    Word w[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < 4 && at + i < count; ++i) w[i] = words[at + i];
    key = chacha_key(key, w[0], w[1], w[2], w[3]);
  }
  return key;
}
inline Key absorb_in_fours(const Key& key, const std::vector<Word>& words) {
  return absorb_in_fours(key, words.data(), words.size());
}

// A key derived directly from the seed. `kind`: 1 anchor's masks, 5 LINK proof, 6 batch, 7 leaf
// table's masks, 8 one-time proofs. `k`: a leaf table's number, or 0.
inline Key seed_child(const Key& seed, Word kind, Word k) {
  return chacha_key(seed, kWordCPSD, kind, k, 0);
}

// --- A batch -----------------------------------------------------------------

// Kbat, a batch's first key: Kh absorbs rho, the batch's fresh bytes, and then D (`d`), a digest
// that the caller supplies; coins_known_maker.cpp passes fixed test words. rho, like every `fresh`
// below, is 32 bytes as eight words (key_from_bytes()).
inline Key batch_base_key(const Key& seed, const Key& rho, const std::vector<Word>& d) {
  Key x = seed_child(seed, 6, 0);  // Kh
  x = absorb_in_fours(x, rho.data(), rho.size());
  return absorb_in_fours(x, d);
}

// The batch's leaf table k, sampled by rejection among the live (not retired) ones, first .. m - 1
// of the m leaf tables: the rule of cp::draw_leaf() (prover/common/cp_files.h), which returns
// k - first.
inline Word draw_leaf(const Key& kbat, Word m, Word first) {
  if (first >= m) throw std::invalid_argument("draw_leaf: no live leaf");
  const std::uint64_t live = m - first;
  // 2^64 mod live, in 64-bit arithmetic.
  const std::uint64_t excess = (~std::uint64_t(0) % live + 1) % live;
  for (Word t = 0;; ++t) {
    const Block o = chacha_block(kbat, kWordCPLF, t, 0, 0);
    const std::uint64_t v = o[0] | std::uint64_t(o[1]) << 32;
    // v < 2^64 - excess
    if (excess == 0 || v < std::uint64_t(0) - excess) return first + Word(v % live);
    // The rule does not bound t. Two implementations of it need to agree only for t below 2^32,
    // where this one stops.
    if (t == ~Word(0)) throw std::runtime_error("draw_leaf: no try is left");
  }
}

// The batch's process key, from Kbat and h_card of the sampled leaf table.
inline Key batch_process_key(const Key& kbat, const std::vector<Word>& h_card) {
  return absorb_in_fours(chacha_key(kbat, kWordCPBK, 0, 0, 0), h_card);
}

// A word for comparing two copies of a process key. No private random value's label leads to this
// input of chacha_block(): under this key, context_key() (coin_generator.h) passes 0 as the third
// word.
inline Word key_check_word(const Key& process_key) {
  return chacha_block(process_key, 0, 0, kWordCPKW, 0)[0];
}

// --- The committer -----------------------------------------------------------

// The process key of the committer's one-time proofs about the anchor, the range file among them;
// `h_0` is the anchor's. `version` is the format's version word, not the range file's format word,
// which is in the labels of these proofs' private random values.
inline Key one_time_process_key(const Key& seed, const std::vector<Word>& h_0, Word version) {
  std::vector<Word> w = h_0;
  w.push_back(version);
  return absorb_in_fours(seed_child(seed, 8, 0), w);
}

// The process key of one run of leaf table k's LINK proof. The two digests and the run's 32 fresh
// bytes are absorbed as one list of 2 DW + 8 words, not each padded separately.
inline Key link_process_key(const Key& seed, Word k, const std::vector<Word>& anchor_h_card,
                            const std::vector<Word>& leaf_h_0, const Key& fresh) {
  std::vector<Word> w = anchor_h_card;
  w.insert(w.end(), leaf_h_0.begin(), leaf_h_0.end());
  w.insert(w.end(), fresh.begin(), fresh.end());
  return absorb_in_fours(seed_child(seed, 5, k), w);
}

// The mask seeds are keys for prover/second_writing/mask_generator.h, not process keys.
inline Key anchor_mask_seed(const Key& seed) { return seed_child(seed, 1, 0); }
inline Key leaf_mask_seed(const Key& seed, Word k, const Key& fresh) {
  return absorb_in_fours(seed_child(seed, 7, k), fresh.data(), fresh.size());
}

// --- Keys that are public, for tests ------------------------------------------

// One step of SplitMix64 from the state z (splitmix64.c by Sebastiano Vigna).
constexpr std::uint64_t mix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Test seed n < 256 (not checked): SEED_TEST_n of prover/hash_commitment/hc_format.h. Nothing is
// hidden under it.
inline Key test_seed(unsigned n) {
  Key k;
  for (Word i = 0; i < 8; ++i)
    k[i] = n == 0 ? (4 * i) | (4 * i + 1) << 8 | (4 * i + 2) << 16 | (4 * i + 3) << 24
                  : Word(mix64(0x484B4559ull + 16 * n + i) >> 16);
  return k;
}

}  // namespace coins
