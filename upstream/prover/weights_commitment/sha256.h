// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// SHA-256 (FIPS 180-4) of a byte stream. Host code. It uses the x86 SHA instructions if the
// compiler has them enabled (__SHA__ and __SSE4_1__ are defined), and plain C++ otherwise.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#if defined(__SHA__) && defined(__SSE4_1__)
#include <immintrin.h>
#endif

// Feed bytes with update(), then call hex() or digest() once, as the last call on the object.
// The object keeps the state and up to 64 input bytes; a caller that hashes a secret must zero the
// object.
class Sha256 {
public:
  void update(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    total_ += n;
    if (fill_) {
      const size_t take = n < 64 - fill_ ? n : 64 - fill_;
      memcpy(buf_ + fill_, p, take);
      fill_ += take, p += take, n -= take;
      if (fill_ < 64) return;
      blocks(buf_, 1);
      fill_ = 0;
    }
    if (n >= 64) {
      blocks(p, n / 64);
      p += n / 64 * 64, n %= 64;
    }
    memcpy(buf_, p, n);
    fill_ = n;
  }
  void digest(uint8_t out[32]) {
    close();
    for (int i = 0; i < 32; i++) out[i] = (uint8_t)(h_[i / 4] >> (24 - 8 * (i % 4)));
  }
  // The digest as 64 lower-case hex digits.
  std::string hex() {
    close();
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 8; i++)
      for (int b = 28; b >= 0; b -= 4) s.push_back(d[(h_[i] >> b) & 15]);
    return s;
  }

private:
  // Appends the padding and the length in bits; h_ is then the digest.
  void close() {
    const uint64_t bits = total_ * 8;
    uint8_t pad[72] = {0x80};
    const size_t padn = (fill_ < 56 ? 56 : 120) - fill_;
    for (int i = 0; i < 8; i++) pad[padn + i] = (uint8_t)(bits >> (56 - 8 * i));
    update(pad, padn + 8);
  }
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf_[64];
  size_t fill_ = 0;
  uint64_t total_ = 0;

  // Round constants; only the plain C++ blocks() uses them.
  static const uint32_t *K() {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    return k;
  }

#if defined(__SHA__) && defined(__SSE4_1__)
  // Compresses the n 64-byte blocks at data into h_; FIPS 180-4, section 6.2.2, specifies the
  // function. Neither data nor h_ needs to be aligned.
  // Adapted from sha256-x86.c by Jeffrey Walton (public domain), which is based on code from Intel
  // and by Sean Gulley for the miTLS project.
  // See also: S. Gulley and others, "Intel SHA Extensions", July 2013, which published the pattern
  // for these instructions: four rounds per step, the state held as A, B, E, F and C, D, G, H, and
  // shuffled into this arrangement and back out of it outside the main loop. The eight shuffles
  // here are the same operations, with the same constants and in the same order, as in RustCrypto's
  // sha2 (src/sha256/x86.rs; MIT OR Apache-2.0), which is written in Rust.
  void blocks(const uint8_t *data, size_t n) {
    __m128i state0, state1;
    __m128i msg, tmp;
    __m128i msg0, msg1, msg2, msg3;
    __m128i abef_save, cdgh_save;
    // Reverses the bytes of each 32-bit lane: the words of a block are big-endian.
    const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    // Loads h_ and reorders it as sha256rnds2 expects. The trailing notes name FIPS 180-4's working
    // variables a to h, highest 32-bit lane first.
    tmp = _mm_loadu_si128((const __m128i *)&h_[0]);
    state1 = _mm_loadu_si128((const __m128i *)&h_[4]);

    tmp = _mm_shuffle_epi32(tmp, 0xb1);           // CDAB
    state1 = _mm_shuffle_epi32(state1, 0x1b);     // EFGH
    state0 = _mm_alignr_epi8(tmp, state1, 8);     // ABEF
    state1 = _mm_blend_epi16(state1, tmp, 0xf0);  // CDGH

    while (n > 0) {
      abef_save = state0;
      cdgh_save = state1;

      // Rounds 0-3. msg is four message words plus their round constants. sha256rnds2 does two
      // rounds with the two low lanes of msg; the shuffle by 0x0e brings down the two high lanes.
      msg = _mm_loadu_si128((const __m128i *)(data + 0));
      msg0 = _mm_shuffle_epi8(msg, mask);
      msg = _mm_add_epi32(msg0, _mm_set_epi64x(0xe9b5dba5b5c0fbcfULL, 0x71374491428a2f98ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

      // Rounds 4-7
      msg1 = _mm_loadu_si128((const __m128i *)(data + 16));
      msg1 = _mm_shuffle_epi8(msg1, mask);
      msg = _mm_add_epi32(msg1, _mm_set_epi64x(0xab1c5ed5923f82a4ULL, 0x59f111f13956c25bULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 8-11
      msg2 = _mm_loadu_si128((const __m128i *)(data + 32));
      msg2 = _mm_shuffle_epi8(msg2, mask);
      msg = _mm_add_epi32(msg2, _mm_set_epi64x(0x550c7dc3243185beULL, 0x12835b01d807aa98ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 12-15. Words 16 to 63 of the message schedule are computed four at a time:
      // sha256msg1 (first in rounds 4-7), the addition of tmp (the term W[t-7]), then sha256msg2.
      msg3 = _mm_loadu_si128((const __m128i *)(data + 48));
      msg3 = _mm_shuffle_epi8(msg3, mask);
      msg = _mm_add_epi32(msg3, _mm_set_epi64x(0xc19bf1749bdc06a7ULL, 0x80deb1fe72be5d74ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg3, msg2, 4);
      msg0 = _mm_add_epi32(msg0, tmp);
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 16-19
      msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x240ca1cc0fc19dc6ULL, 0xefbe4786e49b69c1ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg0, msg3, 4);
      msg1 = _mm_add_epi32(msg1, tmp);
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 20-23
      msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x76f988da5cb0a9dcULL, 0x4a7484aa2de92c6fULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg1, msg0, 4);
      msg2 = _mm_add_epi32(msg2, tmp);
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 24-27
      msg = _mm_add_epi32(msg2, _mm_set_epi64x(0xbf597fc7b00327c8ULL, 0xa831c66d983e5152ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg2, msg1, 4);
      msg3 = _mm_add_epi32(msg3, tmp);
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 28-31
      msg = _mm_add_epi32(msg3, _mm_set_epi64x(0x1429296706ca6351ULL, 0xd5a79147c6e00bf3ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg3, msg2, 4);
      msg0 = _mm_add_epi32(msg0, tmp);
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 32-35
      msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x53380d134d2c6dfcULL, 0x2e1b213827b70a85ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg0, msg3, 4);
      msg1 = _mm_add_epi32(msg1, tmp);
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 36-39
      msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x92722c8581c2c92eULL, 0x766a0abb650a7354ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg1, msg0, 4);
      msg2 = _mm_add_epi32(msg2, tmp);
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg0 = _mm_sha256msg1_epu32(msg0, msg1);

      // Rounds 40-43
      msg = _mm_add_epi32(msg2, _mm_set_epi64x(0xc76c51a3c24b8b70ULL, 0xa81a664ba2bfe8a1ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg2, msg1, 4);
      msg3 = _mm_add_epi32(msg3, tmp);
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg1 = _mm_sha256msg1_epu32(msg1, msg2);

      // Rounds 44-47
      msg = _mm_add_epi32(msg3, _mm_set_epi64x(0x106aa070f40e3585ULL, 0xd6990624d192e819ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg3, msg2, 4);
      msg0 = _mm_add_epi32(msg0, tmp);
      msg0 = _mm_sha256msg2_epu32(msg0, msg3);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg2 = _mm_sha256msg1_epu32(msg2, msg3);

      // Rounds 48-51
      msg = _mm_add_epi32(msg0, _mm_set_epi64x(0x34b0bcb52748774cULL, 0x1e376c0819a4c116ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg0, msg3, 4);
      msg1 = _mm_add_epi32(msg1, tmp);
      msg1 = _mm_sha256msg2_epu32(msg1, msg0);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
      msg3 = _mm_sha256msg1_epu32(msg3, msg0);

      // Rounds 52-55
      msg = _mm_add_epi32(msg1, _mm_set_epi64x(0x682e6ff35b9cca4fULL, 0x4ed8aa4a391c0cb3ULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg1, msg0, 4);
      msg2 = _mm_add_epi32(msg2, tmp);
      msg2 = _mm_sha256msg2_epu32(msg2, msg1);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

      // Rounds 56-59
      msg = _mm_add_epi32(msg2, _mm_set_epi64x(0x8cc7020884c87814ULL, 0x78a5636f748f82eeULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      tmp = _mm_alignr_epi8(msg2, msg1, 4);
      msg3 = _mm_add_epi32(msg3, tmp);
      msg3 = _mm_sha256msg2_epu32(msg3, msg2);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

      // Rounds 60-63
      msg = _mm_add_epi32(msg3, _mm_set_epi64x(0xc67178f2bef9a3f7ULL, 0xa4506ceb90befffaULL));
      state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
      msg = _mm_shuffle_epi32(msg, 0x0e);
      state0 = _mm_sha256rnds2_epu32(state0, state1, msg);

      state0 = _mm_add_epi32(state0, abef_save);
      state1 = _mm_add_epi32(state1, cdgh_save);

      data += 64;
      n -= 1;
    }

    tmp = _mm_shuffle_epi32(state0, 0x1b);        // FEBA
    state1 = _mm_shuffle_epi32(state1, 0xb1);     // DCHG
    state0 = _mm_blend_epi16(tmp, state1, 0xf0);  // DCBA
    state1 = _mm_alignr_epi8(state1, tmp, 8);     // HGFE

    _mm_storeu_si128((__m128i *)&h_[0], state0);
    _mm_storeu_si128((__m128i *)&h_[4], state1);

    // Zeroes the ten variables that depend on the input: the input can be a key or a nonce. The
    // empty asm statement keeps the compiler from removing the assignments. Copies that the
    // compiler keeps in other registers or in stack slots are not zeroed.
    msg = tmp = msg0 = msg1 = msg2 = msg3 = _mm_set_epi64x(0, 0);
    state0 = state1 = abef_save = cdgh_save = _mm_set_epi64x(0, 0);
    asm volatile("" : "+x"(msg), "+x"(tmp), "+x"(msg0), "+x"(msg1), "+x"(msg2), "+x"(msg3),
                      "+x"(state0), "+x"(state1), "+x"(abef_save), "+x"(cdgh_save));
  }
#else
  static uint32_t rot(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  // Computes the same as the x86 blocks() (FIPS 180-4, section 6.2.2), in plain C++.
  void blocks(const uint8_t *p, size_t n) {
    const uint32_t *k = K();
    for (; n; n--, p += 64) {
      uint32_t w[64], v[8];
      for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
      for (int i = 16; i < 64; i++)
        w[i] = w[i - 16] + (rot(w[i - 15], 7) ^ rot(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (rot(w[i - 2], 17) ^ rot(w[i - 2], 19) ^ (w[i - 2] >> 10));
      memcpy(v, h_, 32);
      for (int i = 0; i < 64; i++) {
        const uint32_t t1 =
            v[7] + (rot(v[4], 6) ^ rot(v[4], 11) ^ rot(v[4], 25)) + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
        const uint32_t t2 =
            (rot(v[0], 2) ^ rot(v[0], 13) ^ rot(v[0], 22)) + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
        for (int q = 7; q > 0; q--) v[q] = v[q - 1];
        v[4] += t1, v[0] = t1 + t2;
      }
      for (int i = 0; i < 8; i++) h_[i] += v[i];
      // Zeroes w and v, which depend on the input. t1, t2 and copies that the compiler keeps in
      // registers or in other stack slots are not zeroed.
      explicit_bzero(w, sizeof w);
      explicit_bzero(v, sizeof v);
    }
  }
#endif
};
