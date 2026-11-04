// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives the masks of a weight table's rows (random entries encoded with each row), the salts of
// its Merkle tree's leaves and its rows' pads from the table's
// commit-run key (`mask_seed`, a ChaCha20 key).
// A second implementation, against which the committer program cp_commit checks values that
// prover/hash_commitment/hf_mask.cuh and hf_pads.cuh derive.
// The rule is defined in docs/format/COIN-KEYS.md, "2a. From a key to a random value".
// Any copy of this file in prover/tests/coins_second_text/ must stay byte-identical to the one in
// prover/second_writing/. The reason is in prover/second_writing/README.md.
//
// A table's commit-run key must key that table's masks, salts and pads and nothing else. None of
// the three has a label, unlike the values that `BasicCoins`
// (prover/second_writing/coin_generator.h) derives from a process key.
// Nothing is wiped here: masks, blocks and, in chacha_block()'s local arrays, copies of the table's
// commit-run key may stay on the stack. prover/common/stack_scrub.h states what a caller must do.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "chacha_block.h"
#include "coin_generator.h"

namespace coins {

// "MASK", "SALT" and "PADS" in ASCII.
constexpr Word kWordMASK = 0x4D41534B;
constexpr Word kWordSALT = 0x53414C54;
constexpr Word kWordPADS = 0x50414453;

// The prime p of the table's field, and the rule's threshold: the largest multiple of p not above
// 2^64, which is p itself. `kP`, which `BasicCoins` uses, is the same prime, not read here.
// What depends on the field here, in prover/hash_commitment/hf_mask_ref.h and in toy/mask_rules.py
// rests on one reading of the rule's wording: an error in that reading would be in all three, and
// a comparison among them would not show it. The three are separate from
// prover/hash_commitment/hf_mask.cuh, not from one another.
constexpr std::uint64_t kTableP = 0xFFFFFFFF00000001;  // 2^64 - 2^32 + 1
constexpr std::uint64_t kTableThreshold = kTableP;     // once p
static_assert(~std::uint64_t(0) - kTableThreshold < kTableP - 1 && kTableThreshold % kTableP == 0,
              "one more p would pass 2^64");

inline std::uint64_t block_value(const Block& o, unsigned s) {
  return o[2 * s] | std::uint64_t(o[2 * s + 1]) << 32;
}

// blk(r, g, try) in the rule's notation.
inline Block mask_block(const Key& mask_seed, Word r, Word g, Word attempt) {
  return chacha_block(mask_seed, r, g, attempt, kWordMASK);
}

// Masks 8 g .. 8 g + 7 of row r, by rejection sampling: the values, modulo p, of the first block
// whose eight 64-bit values are all below `threshold`.
// A `threshold` below the default is only for tests that must reach a second try.
inline std::array<std::uint64_t, 8> masks(const Key& mask_seed, Word r, Word g,
                                          std::uint64_t threshold = kTableThreshold) {
  for (Word attempt = 0;; ++attempt) {
    const Block o = mask_block(mask_seed, r, g, attempt);
    bool good = true;
    for (unsigned s = 0; s < 8; ++s) good = good && block_value(o, s) < threshold;
    if (good) {
      std::array<std::uint64_t, 8> m;
      for (unsigned s = 0; s < 8; ++s) m[s] = block_value(o, s) % kTableP;
      return m;
    }
    if (attempt == ~Word(0)) throw std::runtime_error("masks: no try below 2^32 is good");
  }
}

// The pads 8 g .. 8 g + 7 of row r of a table: the masks' rule with the word "PADS".
// It shares no code with prover/hash_commitment/hf_pads.cuh, but the two have the same authors.
inline std::array<std::uint64_t, 8> pads(const Key& mask_seed, Word r, Word g,
                                         std::uint64_t threshold = kTableThreshold) {
  for (Word attempt = 0;; ++attempt) {
    const Block o = chacha_block(mask_seed, r, g, attempt, kWordPADS);
    unsigned below = 0;
    for (unsigned s = 0; s < 8; ++s) below += block_value(o, s) < threshold ? 1 : 0;
    if (below == 8) {
      std::array<std::uint64_t, 8> p;
      for (unsigned s = 0; s < 8; ++s) p[s] = block_value(o, s) % kTableP;
      return p;
    }
    if (attempt == ~Word(0)) throw std::runtime_error("pads: no try below 2^32 is good");
  }
}

// The salt of the tree leaf at `position`: the first five values of one block, modulo p. With no
// second try the residues are not equally likely, on purpose: a salt only has to be hard to guess.
// At p = 2^64 - 2^32 + 1 the residues below 2^32 - 1 are twice as likely as the others.
inline std::array<std::uint64_t, 5> table_salt(const Key& mask_seed, std::uint64_t position) {
  const Block o = chacha_block(mask_seed, Word(position), Word(position >> 32), 0, kWordSALT);
  std::array<std::uint64_t, 5> s;
  for (unsigned x = 0; x < 5; ++x) s[x] = block_value(o, x) % kTableP;
  return s;
}

}  // namespace coins
