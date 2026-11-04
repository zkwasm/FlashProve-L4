// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

/*
   BLAKE2 reference source code package - reference C implementations

   Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
   terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
   your option.  The terms of these licenses can be found at:

   - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
   - OpenSSL license   : https://www.openssl.org/source/license.html
   - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0

   More information about the BLAKE2 hash function can be found at
   https://blake2.net.
*/
// Adapted from blake2b-ref.c and the two headers that it includes, blake2.h and
// blake2-impl.h, of the BLAKE2 reference source code package (reference C
// implementations); taken under CC0 1.0 Universal (SPDX: CC0-1.0).

// The format's hash, defined in verifier/HASHES.md: compression function C, hash H, tree hash TH.

// coin_generator.h does not call this hash. TH is for D = TH(list), the digest of a
// batch's requests that batch_base_key() in process_keys.h takes. No other file
// calls hash_value() or tree_hash().
//
// C: BLAKE2b's compression function F (RFC 7693, section 3.2). H departs
// from BLAKE2b-512 in one point: the high half of the RFC's counter t holds the
// domain-separation number and the length of the whole list.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace coins {

using HashValue = std::array<std::uint32_t, 16>;  // a value: SW = 16 words

namespace detail {

// Adapted from the three files named above: everything that this file puts in
// namespace detail, the body of compress(), and the parameter word that
// hash_start() uses. load64, store64 and rotr64 come from blake2-impl.h, the
// fields of the state and of the parameter block from blake2.h, and the rest
// from blake2b-ref.c; all three carry the notice above. Ours are HashValue, the
// signature of compress(), the rest of hash_start(), hash_value() and
// tree_hash().
//
// Our changes to the adapted text:
//  - It is C++. G and ROUND are functions, not macros: both take m as an
//    argument, and ROUND takes v too.
//  - blake2b_IV is named kBlakeIV, and blake2b_compress is named compress.
//  - There is no state that lasts between calls. compress() gets the
//    reference's state fields h (as cv), t[0] (as c), t[1] (as num and l) and
//    f[0] (as last) from its caller, and returns the new h.
//  - In the reference t[1] counts the overflows of t[0]. Here it holds the
//    format's two additional words: num + 2^32 l.
//  - f[1], the last-node flag, is always zero.
//  - load64 and store64 work on two words of 32 bits, not on eight bytes.
//  - Of the parameter block, only the first eight bytes are written out, as one
//    constant, for a digest of 64 bytes and no key.
//  - Of the reference's other functions, three things are taken over: the value
//    of the set flag (blake2b_set_lastblock), what the parameter word needs
//    (the init functions), and the loop that writes h out (blake2b_final). All
//    else is left out, among it the counter's stepping, blake2b_update, keyed
//    hashing, the one-call blake2b() and blake2(), and the self-test.

inline constexpr std::uint64_t kBlakeIV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

inline constexpr std::uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

// The first eight bytes of the reference's parameter block, with the values
// that blake2b_init() gives them for a digest of 64 bytes, read as one word the
// way blake2b_init_param() reads them. The block's other 56 bytes are zero
// there.
inline constexpr std::uint64_t kParamWord0 = (std::uint64_t(64) << 0) |  // digest_length
                                             (std::uint64_t(0) << 8) |   // key_length
                                             (std::uint64_t(1) << 16) |  // fanout
                                             (std::uint64_t(1) << 24) |  // depth
                                             (std::uint64_t(0) << 32);   // leaf_length

// A 64-bit word is stored as two 32-bit words, the low half first.
inline std::uint64_t load64(const std::uint32_t* p) {
  return (std::uint64_t(p[0]) << 0) | (std::uint64_t(p[1]) << 32);
}

inline void store64(std::uint32_t* p, const std::uint64_t w) {
  p[0] = std::uint32_t(w >> 0);
  p[1] = std::uint32_t(w >> 32);
}

// c must be between 1 and 63.
inline std::uint64_t rotr64(const std::uint64_t w, const unsigned c) {
  return (w >> c) | (w << (64 - c));
}

inline void G(const std::uint64_t* m, const int r, const int i, std::uint64_t& a, std::uint64_t& b,
              std::uint64_t& c, std::uint64_t& d) {
  a = a + b + m[blake2b_sigma[r][2 * i + 0]];
  d = rotr64(d ^ a, 32);
  c = c + d;
  b = rotr64(b ^ c, 24);
  a = a + b + m[blake2b_sigma[r][2 * i + 1]];
  d = rotr64(d ^ a, 16);
  c = c + d;
  b = rotr64(b ^ c, 63);
}

inline void ROUND(const std::uint64_t* m, std::uint64_t* v, const int r) {
  G(m, r, 0, v[0], v[4], v[8], v[12]);
  G(m, r, 1, v[1], v[5], v[9], v[13]);
  G(m, r, 2, v[2], v[6], v[10], v[14]);
  G(m, r, 3, v[3], v[7], v[11], v[15]);
  G(m, r, 4, v[0], v[5], v[10], v[15]);
  G(m, r, 5, v[1], v[6], v[11], v[12]);
  G(m, r, 6, v[2], v[7], v[8], v[13]);
  G(m, r, 7, v[3], v[4], v[9], v[14]);
}

}  // namespace detail

// C(cv, m, c, l, num, last): the compression function F of RFC 7693, section 3.2, with the format's
// words l and num in the high half of its counter. `block` points to 32 words, each four message
// bytes with the low byte first. c counts the message bytes fed so far, including this block's but
// not its zero padding.
inline HashValue compress(const HashValue& cv, const std::uint32_t* block, std::uint64_t c,
                          std::uint32_t l, std::uint32_t num, bool last) {
  using detail::kBlakeIV;
  using detail::ROUND;

  // What the reference keeps in its state S comes from the caller here.
  std::uint64_t h[8];
  const std::uint64_t t[2] = {c, num | std::uint64_t(l) << 32};
  const std::uint64_t f[2] = {last ? std::uint64_t(-1) : std::uint64_t(0), 0};
  std::uint64_t m[16];
  std::uint64_t v[16];
  std::size_t i;

  for (i = 0; i < 8; ++i) {
    h[i] = detail::load64(cv.data() + 2 * i);
  }

  for (i = 0; i < 16; ++i) {
    m[i] = detail::load64(block + 2 * i);
  }

  for (i = 0; i < 8; ++i) {
    v[i] = h[i];
  }

  v[8] = kBlakeIV[0];
  v[9] = kBlakeIV[1];
  v[10] = kBlakeIV[2];
  v[11] = kBlakeIV[3];
  v[12] = kBlakeIV[4] ^ t[0];
  v[13] = kBlakeIV[5] ^ t[1];
  v[14] = kBlakeIV[6] ^ f[0];
  v[15] = kBlakeIV[7] ^ f[1];

  ROUND(m, v, 0);
  ROUND(m, v, 1);
  ROUND(m, v, 2);
  ROUND(m, v, 3);
  ROUND(m, v, 4);
  ROUND(m, v, 5);
  ROUND(m, v, 6);
  ROUND(m, v, 7);
  ROUND(m, v, 8);
  ROUND(m, v, 9);
  ROUND(m, v, 10);
  ROUND(m, v, 11);

  for (i = 0; i < 8; ++i) {
    h[i] = h[i] ^ v[i] ^ v[i + 8];
  }

  HashValue next;
  for (i = 0; i < 8; ++i) {
    detail::store64(next.data() + 2 * i, h[i]);
  }
  return next;
}

// IV', the state from which BLAKE2b-512 starts.
inline HashValue hash_start() {
  HashValue s;
  for (int i = 0; i < 8; ++i) {
    const std::uint64_t x = detail::kBlakeIV[i] ^ (i == 0 ? detail::kParamWord0 : 0u);
    s[2 * i] = std::uint32_t(x);
    s[2 * i + 1] = std::uint32_t(x >> 32);
  }
  return s;
}

// H^v(num; w): the value of the hash of the list w[0 .. length - 1] under the
// domain-separation number num.
inline HashValue hash_value(std::uint32_t num, const std::uint32_t* w, std::size_t length) {
  if (length >> 32) throw std::invalid_argument("hash_value: the list has 2^32 words or more");
  const std::uint32_t total = std::uint32_t(length);
  const std::size_t blocks = length == 0 ? 1 : (length + 31) / 32;
  HashValue cv = hash_start();
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint32_t block[32] = {};
    for (std::size_t i = 0; i < 32 && 32 * b + i < length; ++i) block[i] = w[32 * b + i];
    const bool last = b + 1 == blocks;
    const std::uint64_t fed = last ? 4 * std::uint64_t(length) : 128 * std::uint64_t(b + 1);
    cv = compress(cv, block, fed, total, num, last);
  }
  return cv;
}

// TH(w): the tree hash of a list of at least dw words. A digest is a value's first
// dw words (8 in the format); dw must be 1 to 16 (unchecked). Level 0 is hashed under
// the number 10 and every level above it under 11, in pieces of 16 dw words.
inline std::vector<std::uint32_t> tree_hash(const std::vector<std::uint32_t>& w, unsigned dw) {
  if (w.size() < dw) throw std::invalid_argument("tree_hash: the list is shorter than a digest");
  const std::size_t piece = 16 * std::size_t(dw);
  std::vector<std::uint32_t> level = w;
  std::uint32_t num = 10;
  do {
    std::vector<std::uint32_t> next;
    for (std::size_t at = 0; at < level.size(); at += piece) {
      const std::size_t len = level.size() - at < piece ? level.size() - at : piece;
      const HashValue v = hash_value(num, level.data() + at, len);
      next.insert(next.end(), v.begin(), v.begin() + dw);
    }
    level.swap(next);
    num = 11;
  } while (level.size() > dw);
  return level;
}

}  // namespace coins
