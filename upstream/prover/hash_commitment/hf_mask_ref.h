// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The CPU reference version of prover/hash_commitment/hf_mask.cuh and hf_pads.cuh: a weight table's
// masks (random entries encoded with each row), its Merkle tree leaves' salts and its rows' pads,
// computed from the table's 32-byte making key, a ChaCha20 key, by the
// rules defined in docs/format/COIN-KEYS.md, "2a. From a key to a random value". It is slow. Tests
// include it, and so do run-time self-checks of `cp_commit` and `cp_prove` (hf_mask_own.cuh,
// hf_open_side.cuh and hf_pads.cuh in prover/hash_commitment/), which recompute masks, salts and
// pads with it, from test keys or a second copy of the table's key.
// It includes neither hf_mask.cuh nor prover/hash_commitment/hc_chacha.cuh (hf_mask.cuh's block
// function), but block() is close to hc_chacha.cuh's text: both are adapted from one public-domain
// file (chacha-merged.c; see block()) and keep its form. Their agreement therefore cannot show an
// error taken from chacha-merged.c, or one made alike in adapting it. The self-check in
// hf_mask_own.cuh also compares block() with a test vector of RFC 8439.
// Limit: this is a second implementation of ours, not an independent one, block() included; for
// the field-dependent parts see the comment above P.
#pragma once
#if defined(BLK_FIELD) && BLK_FIELD != 2
#error "BLK_FIELD, if it is given, must be 2 (p = 2^64 - 2^32 + 1)"
#endif
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace hf_mask_ref {
// The field has p = 2^64 - 2^32 + 1 elements.
// Limit: the field-dependent parts of this file, toy/mask_rules.py and
// prover/second_writing/mask_generator.h were written together, from the rules' wording: these
// three are separate from prover/hash_commitment/hf_mask.cuh, not from one another.
// FULL: the masks' rejection threshold, the largest multiple of p that is at most 2^64.
constexpr uint64_t P = 18446744069414584321ull; // p = 2^64 - 2^32 + 1
constexpr uint64_t FULL = P;                    // 2 p is above 2^64
static_assert(P == 0ull - 4294967296ull + 1 && P > 9223372036854775808ull, "the number as it is written, and more than 2^63");
constexpr uint32_t WORD_MASK = 0x4D41534Bu; // "MASK"
constexpr uint32_t WORD_SALT = 0x53414C54u; // "SALT"
constexpr uint32_t WORD_PADS = 0x50414453u; // "PADS"
// The threshold that a known answer's k selects (`HfKnownMasks::k`, hf_mask_known.h; 1 to 8, not
// checked). Below 8, retries are frequent.
constexpr uint64_t limit(int k) {
  return k == 8 ? FULL : (uint64_t)k * (FULL / 8);
}

// ChaCha20's block function (RFC 8439, 2.3), with `last_four` in place of the RFC's block counter
// and nonce. How the key's bytes and the constant become words, the loop of ten double rounds and
// the final addition of the initial state are adapted from the key setup and the encryption
// function of the file that is named below.
inline void block(const uint8_t key[32], const uint32_t last_four[4], uint32_t out[16]) {
  const auto word = [](const uint8_t *p) { // four bytes as a little-endian word
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
  };
  const auto rotate = [](uint32_t v, int c) { return v << c | v >> (32 - c); };
  // Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version
  // 20080118, public domain).
  // RFC 8439, section 2.1, specifies the same function.
  const auto quarter_round = [&](uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    a += b, d = rotate(d ^ a, 16);
    c += d, b = rotate(b ^ c, 12);
    a += b, d = rotate(d ^ a, 8);
    c += d, b = rotate(b ^ c, 7);
  };
  const uint8_t constants[17] = "expand 32-byte k";
  const uint32_t j0 = word(constants + 0), j1 = word(constants + 4), j2 = word(constants + 8),
                 j3 = word(constants + 12);
  const uint32_t j4 = word(key + 0), j5 = word(key + 4), j6 = word(key + 8), j7 = word(key + 12);
  const uint32_t j8 = word(key + 16), j9 = word(key + 20), j10 = word(key + 24), j11 = word(key + 28);
  const uint32_t j12 = last_four[0], j13 = last_four[1], j14 = last_four[2], j15 = last_four[3];
  uint32_t x0 = j0, x1 = j1, x2 = j2, x3 = j3, x4 = j4, x5 = j5, x6 = j6, x7 = j7;
  uint32_t x8 = j8, x9 = j9, x10 = j10, x11 = j11, x12 = j12, x13 = j13, x14 = j14, x15 = j15;
  // The order of the eight quarter rounds is that of the round loop in D. J. Bernstein's
  // chacha-merged.c.
  for (int i = 20; i > 0; i -= 2) {
    quarter_round(x0, x4, x8, x12);
    quarter_round(x1, x5, x9, x13);
    quarter_round(x2, x6, x10, x14);
    quarter_round(x3, x7, x11, x15);
    quarter_round(x0, x5, x10, x15);
    quarter_round(x1, x6, x11, x12);
    quarter_round(x2, x7, x8, x13);
    quarter_round(x3, x4, x9, x14);
  }
  out[0] = x0 + j0, out[1] = x1 + j1, out[2] = x2 + j2, out[3] = x3 + j3;
  out[4] = x4 + j4, out[5] = x5 + j5, out[6] = x6 + j6, out[7] = x7 + j7;
  out[8] = x8 + j8, out[9] = x9 + j9, out[10] = x10 + j10, out[11] = x11 + j11;
  out[12] = x12 + j12, out[13] = x13 + j13, out[14] = x14 + j14, out[15] = x15 + j15;
}
// 64-bit value sigma (0 to 7, not checked) of block `o`
inline uint64_t value(const uint32_t o[16], int sigma) {
  return (uint64_t)o[2 * sigma] + 4294967296ull * o[2 * sigma + 1];
}
// Value i of weight-table row r in the family of inputs whose fourth word is `word`: value i mod 8,
// reduced mod p, of the first block with `last_four` = (r, floor(i / 8), t, word), t = 0, 1, ..,
// whose eight values are all below `limit`. `tries`, if not null, receives that t; if no t below
// 2^32 gives such a block, of_family() calls abort(). Not checked: that i is below 2^35.
inline uint64_t of_family(uint32_t word, const uint8_t *key, uint32_t r, uint64_t i, uint64_t limit, uint32_t *tries) {
  for (uint32_t t = 0;; t++) {
    const uint32_t input[4] = {r, (uint32_t)(i / 8), t, word};
    uint32_t o[16];
    block(key, input, o);
    int below = 0;
    for (int sigma = 0; sigma < 8; sigma++) below += value(o, sigma) < limit ? 1 : 0;
    if (below == 8) {
      if (tries) *tries = t;
      return value(o, (int)(i % 8)) % P;
    }
    if (t == 4294967295u) abort();
  }
}
// Mask i of row r. Without a key there are no masks: abort(). Unlike hf_mask8() (hf_mask.cuh),
// mask() uses `limit` in every build. Not checked: that the row has a mask i.
inline uint64_t mask(const uint8_t *key, uint32_t r, uint64_t i, uint64_t limit, uint32_t *tries = nullptr) {
  if (!key) abort();
  return of_family(WORD_MASK, key, r, i, limit, tries);
}
// Pad j of row r of a table. Without a key there are no pads: abort().
inline uint64_t pad(const uint8_t *key, uint32_t r, uint64_t j, uint64_t limit, uint32_t *tries = nullptr) {
  if (!key) abort();
  return of_family(WORD_PADS, key, r, j, limit, tries);
}
// Salt symbol sigma (0 to 4, not checked) of tree leaf I: value sigma, reduced mod p, of the one
// block with `last_four` = (I mod 2^32, floor(I / 2^32), 0, WORD_SALT). No threshold, no retry.
// Without a key there are no salts: abort().
inline uint64_t salt(const uint8_t *key, uint64_t I, int sigma) {
  if (!key) abort();
  const uint32_t input[4] = {(uint32_t)(I % 4294967296ull), (uint32_t)(I / 4294967296ull), 0, WORD_SALT};
  uint32_t o[16];
  block(key, input, o);
  return value(o, sigma) % P;
}
} // namespace hf_mask_ref
