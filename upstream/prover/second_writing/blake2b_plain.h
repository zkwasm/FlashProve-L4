// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

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

// BLAKE2b with a digest of 32 bytes, no key, no salt and no personalization, on a
// whole string in memory: blake2b_256(). Python's
// hashlib.blake2b(string, digest_size=32).digest() gives the same 32 bytes.
// BLAKE2's published specification is RFC 7693, "The BLAKE2 Cryptographic Hash
// and Message Authentication Code (MAC)" (https://www.rfc-editor.org/rfc/rfc7693);
// this is the hash of its section 3.3 with nn = 32 and kk = 0.
//
// schedule_keys.h uses this hash to derive the programs' ChaCha20 keys again, and a program
// compares both derivations. This file includes no file of the programs. The hash of the first
// derivation is adapted from the same reference code (blocks/hash/blake2b_f.h), its feeding and
// finishing (HasherT, blocks/hash/hash.h) from blake2b_update and blake2b_final as this file has them.
// So the comparison can show an error made in adapting that code, but not an error of that code, nor
// an error in adapting it that both hashes have: the two hashes are not independent. Other limits of
// the comparison: prover/second_writing/README.md, "How far it is independent of the programs".
//
// The file is stored twice, byte for byte; only the copy in
// prover/second_writing/ is part of this repository.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace coins {

// Overwrites bytes. The writes go through a volatile pointer so that they are not removed as dead stores.
inline void wipe_octets(std::uint8_t* bytes, std::size_t n) {
  volatile std::uint8_t* p = bytes;
  for (std::size_t i = 0; i < n; ++i) p[i] = 0;
}

namespace blake2b_detail {

// Everything in this namespace but kDigestBytes, and the body of blake2b_256()
// below it, is adapted from the BLAKE2 reference implementation (see the head of
// this file).
//
// Our changes to the adapted text:
//  - Taken from blake2b-ref.c: the tables blake2b_IV (named kIV here) and
//    blake2b_sigma; G and ROUND; the functions blake2b_set_lastblock,
//    blake2b_increment_counter, blake2b_init0, blake2b_init_param, blake2b_init,
//    blake2b_compress, blake2b_update, blake2b_final and blake2b (named
//    blake2b_256 here). From blake2.h: the types blake2b_constant, blake2b_state
//    and blake2b_param. From blake2-impl.h: rotr64, load64, store32 and store64.
//  - It is C++. G and ROUND are functions, not macros: both take m as an
//    argument, and ROUND takes v too. The structs have no typedef. blake2b_param
//    has no BLAKE2_PACKED, because its fields fill 64 bytes without gaps; a
//    static_assert replaces BLAKE2_DUMMY_2.
//  - The digest always has kDigestBytes = 32 bytes. So the arguments outlen,
//    S->outlen and the temporary buffer of blake2b_final are left out:
//    blake2b_init puts 32 into the parameter block, and blake2b_final stores the
//    first four words of h straight to out.
//  - Left out: the key (blake2b_init_key, BLAKE2B_KEYBYTES, blake2b's arguments
//    key and keylen); the tests of arguments, with blake2b_is_lastblock, and the
//    status results (no function here returns a status); tree mode
//    (blake2b_set_lastnode, S->last_node, S->f[1], which counts as 0); the alias
//    blake2, crypto_hash and the self-test.
//  - secure_zero_memory is left out. In its place blake2b_256() overwrites its
//    state, which holds the end of the string, with wipe_octets(). The copies of a
//    block in blake2b_compress's m and v are not overwritten.
//  - load64, store32 and store64 are the portable branches only, so no byte order
//    is assumed.
//  - Some loops count with int where the reference has size_t, and the spacing is
//    this repository's.

constexpr int kDigestBytes = 32;

inline constexpr std::uint64_t kIV[8] = {
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

// blake2b_constant of blake2.h, without BLAKE2B_KEYBYTES.
enum blake2b_constant {
  BLAKE2B_BLOCKBYTES = 128,
  BLAKE2B_OUTBYTES = 64,
  BLAKE2B_SALTBYTES = 16,
  BLAKE2B_PERSONALBYTES = 16
};
// blake2b_state of blake2.h, without outlen, last_node and the second word of f. h is the chaining
// value. t is the number of message bytes compressed so far, including the block passed to
// blake2b_compress, low word first. f[0] is the last-block flag. buf[0 .. buflen) contains the
// bytes not yet compressed; the bytes from buf[buflen] on are stale, and are overwritten before buf
// is compressed.
struct blake2b_state {
  std::uint64_t h[8];
  std::uint64_t t[2];
  std::uint64_t f[1];
  std::uint8_t buf[BLAKE2B_BLOCKBYTES];
  std::size_t buflen;
};
// blake2b_param of blake2.h: the parameter block of RFC 7693, section 2.5. Each number is the byte
// offset where its field ends.
struct blake2b_param {
  std::uint8_t digest_length;                   // 1
  std::uint8_t key_length;                      // 2
  std::uint8_t fanout;                          // 3
  std::uint8_t depth;                           // 4
  std::uint32_t leaf_length;                    // 8
  std::uint32_t node_offset;                    // 12
  std::uint32_t xof_length;                     // 16
  std::uint8_t node_depth;                      // 17
  std::uint8_t inner_length;                    // 18
  std::uint8_t reserved[14];                    // 32
  std::uint8_t salt[BLAKE2B_SALTBYTES];         // 48
  std::uint8_t personal[BLAKE2B_PERSONALBYTES]; // 64
};
// Padded structs result in a compile-time error, as by BLAKE2_DUMMY_2 of blake2.h.
static_assert(sizeof(blake2b_param) == BLAKE2B_OUTBYTES, "blake2b_param must have no padding");
// load64 of blake2-impl.h, its portable branch: reads eight bytes as a little-endian number.
inline std::uint64_t load64(const void *src) {
  const std::uint8_t *p = (const std::uint8_t *)src;
  return ((std::uint64_t)(p[0]) << 0) |
         ((std::uint64_t)(p[1]) << 8) |
         ((std::uint64_t)(p[2]) << 16) |
         ((std::uint64_t)(p[3]) << 24) |
         ((std::uint64_t)(p[4]) << 32) |
         ((std::uint64_t)(p[5]) << 40) |
         ((std::uint64_t)(p[6]) << 48) |
         ((std::uint64_t)(p[7]) << 56);
}
// store32 of blake2-impl.h, its portable branch.
inline void store32(void *dst, std::uint32_t w) {
  std::uint8_t *p = (std::uint8_t *)dst;
  p[0] = (std::uint8_t)(w >> 0);
  p[1] = (std::uint8_t)(w >> 8);
  p[2] = (std::uint8_t)(w >> 16);
  p[3] = (std::uint8_t)(w >> 24);
}
// store64 of blake2-impl.h, its portable branch.
inline void store64(void *dst, std::uint64_t w) {
  std::uint8_t *p = (std::uint8_t *)dst;
  p[0] = (std::uint8_t)(w >> 0);
  p[1] = (std::uint8_t)(w >> 8);
  p[2] = (std::uint8_t)(w >> 16);
  p[3] = (std::uint8_t)(w >> 24);
  p[4] = (std::uint8_t)(w >> 32);
  p[5] = (std::uint8_t)(w >> 40);
  p[6] = (std::uint8_t)(w >> 48);
  p[7] = (std::uint8_t)(w >> 56);
}
// blake2b_set_lastblock of blake2b-ref.c, without the last-node flag.
inline void blake2b_set_lastblock(blake2b_state *S) {
  S->f[0] = (std::uint64_t)-1;
}
// blake2b_increment_counter of blake2b-ref.c.
inline void blake2b_increment_counter(blake2b_state *S, const std::uint64_t inc) {
  S->t[0] += inc;
  S->t[1] += (S->t[0] < inc);
}
// blake2b_init0 of blake2b-ref.c.
inline void blake2b_init0(blake2b_state *S) {
  std::memset(S, 0, sizeof(blake2b_state));
  for (int i = 0; i < 8; ++i) {
    S->h[i] = kIV[i];
  }
}
// blake2b_init_param of blake2b-ref.c: init xors IV with input parameter block.
inline void blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
  const std::uint8_t *p = (const std::uint8_t *)(P);
  blake2b_init0(S);
  // IV XOR ParamBlock
  for (int i = 0; i < 8; ++i) {
    S->h[i] ^= load64(p + sizeof(S->h[i]) * i);
  }
}
// blake2b_init of blake2b-ref.c, for a digest of kDigestBytes bytes. The parameter block is the
// bytes 20 00 01 01 (hex), then sixty zero bytes, so the initial chaining value is kIV with
// 0x01010020 xored into h[0].
inline void blake2b_init(blake2b_state *S) {
  blake2b_param P[1];
  P->digest_length = (std::uint8_t)kDigestBytes;
  P->key_length = 0;
  P->fanout = 1;
  P->depth = 1;
  store32(&P->leaf_length, 0);
  store32(&P->node_offset, 0);
  store32(&P->xof_length, 0);
  P->node_depth = 0;
  P->inner_length = 0;
  std::memset(P->reserved, 0, sizeof(P->reserved));
  std::memset(P->salt, 0, sizeof(P->salt));
  std::memset(P->personal, 0, sizeof(P->personal));
  blake2b_init_param(S, P);
}
// blake2b_compress of blake2b-ref.c.
inline void blake2b_compress(blake2b_state *S, const std::uint8_t block[BLAKE2B_BLOCKBYTES]) {
  std::uint64_t m[16];
  std::uint64_t v[16];
  std::size_t i;

  for (i = 0; i < 16; ++i) {
    m[i] = load64(block + i * sizeof(m[i]));
  }

  for (i = 0; i < 8; ++i) {
    v[i] = S->h[i];
  }

  v[8] = kIV[0];
  v[9] = kIV[1];
  v[10] = kIV[2];
  v[11] = kIV[3];
  v[12] = kIV[4] ^ S->t[0];
  v[13] = kIV[5] ^ S->t[1];
  v[14] = kIV[6] ^ S->f[0];
  v[15] = kIV[7];  // S->f[1] is 0

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
    S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
  }
}
// blake2b_update of blake2b-ref.c: appends inlen bytes to the message. pin may be null if inlen is
// 0. A buffer that has just become full is not compressed yet, because the message may end there
// and its last block is compressed with the last-block flag. So after a call with inlen > 0, buflen
// is 1 to 128.
inline void blake2b_update(blake2b_state *S, const void *pin, std::size_t inlen) {
  const unsigned char *in = (const unsigned char *)pin;
  if (inlen > 0) {
    std::size_t left = S->buflen;
    std::size_t fill = BLAKE2B_BLOCKBYTES - left;
    if (inlen > fill) {
      S->buflen = 0;
      std::memcpy(S->buf + left, in, fill); // Fill buffer
      blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
      blake2b_compress(S, S->buf); // Compress
      in += fill;
      inlen -= fill;
      while (inlen > BLAKE2B_BLOCKBYTES) {
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress(S, in);
        in += BLAKE2B_BLOCKBYTES;
        inlen -= BLAKE2B_BLOCKBYTES;
      }
    }
    std::memcpy(S->buf + S->buflen, in, inlen);
    S->buflen += inlen;
  }
}
// blake2b_final of blake2b-ref.c: pads and compresses the last block and writes the digest, the
// first kDigestBytes bytes of h. It must be called only once on a state, and the state must not be
// used afterwards. Neither is checked.
inline void blake2b_final(blake2b_state *S, std::uint8_t out[kDigestBytes]) {
  blake2b_increment_counter(S, S->buflen);
  blake2b_set_lastblock(S);
  std::memset(S->buf + S->buflen, 0, BLAKE2B_BLOCKBYTES - S->buflen); // Padding
  blake2b_compress(S, S->buf);
  for (int i = 0; i < kDigestBytes / 8; ++i) {
    store64(out + sizeof(S->h[i]) * i, S->h[i]);
  }
}

}  // namespace blake2b_detail

// blake2b of blake2b-ref.c, with outlen 32 and no key: the digest of the string `in`, which may be
// empty.
inline std::array<std::uint8_t, 32> blake2b_256(const std::vector<std::uint8_t>& in) {
  using namespace blake2b_detail;
  static_assert(kDigestBytes == 32, "the result has 32 bytes");
  blake2b_state S[1];
  std::array<std::uint8_t, 32> out;
  blake2b_init(S);
  blake2b_update(S, in.data(), in.size());
  blake2b_final(S, out.data());
  wipe_octets(reinterpret_cast<std::uint8_t*>(S), sizeof S);
  return out;
}

}  // namespace coins
