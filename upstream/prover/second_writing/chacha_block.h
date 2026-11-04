// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// The ChaCha20 block function on 32-bit words.
//
// Specified in RFC 8439, "ChaCha20 and Poly1305 for IETF Protocols", section
// 2.3, which also gives a test vector. Comments write the function as
// f(K; x0, x1, x2, x3): K is the key, x0 takes the place of the block counter
// (state word 12) and x1 to x3 the place of the nonce (state words 13 to 15).
// The result is sixteen words: the RFC's serialization into bytes is not used.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace coins {

using Word = std::uint32_t;
using Key = std::array<Word, 8>;
using Block = std::array<Word, 16>;

namespace detail {

// The state's constants: "expand 32-byte k" in ASCII, little-endian.
constexpr Word kSigma[4] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u};

constexpr Word rotl(Word x, int n) { return (x << n) | (x >> (32 - n)); }  // requires 0 < n < 32

// The ChaCha quarter round. The arguments must be four different objects.
// Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code
// (chacha-merged.c, version 20080118, public domain).
// The source is the copy that OpenSSH 9.6p1 carries as chacha.c. quarter_round()
// has the macro's four lines, with its macros PLUS and XOR written out as + and ^
// and rotl() in place of ROTATE, as a function on four references. RFC 8439,
// section 2.1, specifies the same function.
inline void quarter_round(Word& a, Word& b, Word& c, Word& d) {
  a = a + b; d = rotl(d ^ a, 16);
  c = c + d; b = rotl(b ^ c, 12);
  a = a + b; d = rotl(d ^ a, 8);
  c = c + d; b = rotl(b ^ c, 7);
}

}  // namespace detail

// f(K; x0, x1, x2, x3), the ChaCha20 block function of RFC 8439, section 2.3:
// twenty rounds on the state (constants, K, x0, x1, x2, x3), then the initial
// state is added word by word. Each pass of the loop is two rounds: a column
// round, then a diagonal round. No counter is advanced: the caller sets x0 to x3
// for every block. The copy of K in `j` is not erased.
// The order of the eight quarter rounds is that of the round loop in the same
// file of D. J. Bernstein's (chacha-merged.c). chacha_block() follows these parts
// of the function chacha_encrypt_bytes() in OpenSSH 9.6p1's chacha.c, which carries
// D. J. Bernstein's ChaCha code: the copy of the initial state into the working
// state, the round loop, and the additions after the loop, here as one loop. The
// arrays `j` and `x` replace 32 separate variables of the source: sixteen named j
// and sixteen named x, each numbered from 0 to 15.
inline Block chacha_block(const Key& key, Word x0, Word x1, Word x2, Word x3) {
  const Block j = {detail::kSigma[0], detail::kSigma[1], detail::kSigma[2],
                   detail::kSigma[3], key[0], key[1], key[2], key[3],
                   key[4], key[5], key[6], key[7], x0, x1, x2, x3};
  Block x = j;
  for (int i = 20; i > 0; i -= 2) {
    detail::quarter_round(x[0], x[4], x[8], x[12]);
    detail::quarter_round(x[1], x[5], x[9], x[13]);
    detail::quarter_round(x[2], x[6], x[10], x[14]);
    detail::quarter_round(x[3], x[7], x[11], x[15]);
    detail::quarter_round(x[0], x[5], x[10], x[15]);
    detail::quarter_round(x[1], x[6], x[11], x[12]);
    detail::quarter_round(x[2], x[7], x[8], x[13]);
    detail::quarter_round(x[3], x[4], x[9], x[14]);
  }
  for (int i = 0; i < 16; ++i) x[i] = x[i] + j[i];
  return x;
}

// key(K; a, b, c, d), a derived key: words 0 to 7 of f(K; a, b, c, d). `o` is not erased.
inline Key chacha_key(const Key& key, Word a, Word b, Word c, Word d) {
  const Block o = chacha_block(key, a, b, c, d);
  Key k;
  for (int i = 0; i < 8; ++i) k[i] = o[i];
  return k;
}

// W blocks under one key, the same values as W calls of chacha_block():
// out[w][i] becomes word w of f(K; x0[i], x1, x2, x3[i]). Each loop over i can be
// vectorized: no iteration uses the result of another. `key`, x0 and x3 are read
// again after the rounds, so `out` must not overlap them.
// The order of the eight quarter rounds is that of the round loop in the same
// file of D. J. Bernstein's (chacha-merged.c). The loop over r counts from 20 down
// by 2, as that loop does in the function chacha_encrypt_bytes() in OpenSSH
// 9.6p1's chacha.c, which carries D. J. Bernstein's ChaCha code. The source computes
// one block after the other; here each step is a loop over the W blocks, and `out`
// holds the working state.
template <std::size_t W>
inline void chacha_blocks(const Key& key, const Word (&x0)[W], Word x1, Word x2,
                          const Word (&x3)[W], Word (&out)[16][W]) {
  for (int w = 0; w < 4; ++w) {
    for (std::size_t i = 0; i < W; ++i) out[w][i] = detail::kSigma[w];
  }
  for (int w = 0; w < 8; ++w) {
    const Word k = key[w];
    for (std::size_t i = 0; i < W; ++i) out[4 + w][i] = k;
  }
  for (std::size_t i = 0; i < W; ++i) {
    out[12][i] = x0[i];
    out[13][i] = x1;
    out[14][i] = x2;
    out[15][i] = x3[i];
  }
  for (int r = 20; r > 0; r -= 2) {
    for (std::size_t i = 0; i < W; ++i) {
      detail::quarter_round(out[0][i], out[4][i], out[8][i], out[12][i]);
      detail::quarter_round(out[1][i], out[5][i], out[9][i], out[13][i]);
      detail::quarter_round(out[2][i], out[6][i], out[10][i], out[14][i]);
      detail::quarter_round(out[3][i], out[7][i], out[11][i], out[15][i]);
      detail::quarter_round(out[0][i], out[5][i], out[10][i], out[15][i]);
      detail::quarter_round(out[1][i], out[6][i], out[11][i], out[12][i]);
      detail::quarter_round(out[2][i], out[7][i], out[8][i], out[13][i]);
      detail::quarter_round(out[3][i], out[4][i], out[9][i], out[14][i]);
    }
  }
  for (int w = 0; w < 4; ++w) {
    for (std::size_t i = 0; i < W; ++i)
      out[w][i] = out[w][i] + detail::kSigma[w];
  }
  for (int w = 0; w < 8; ++w) {
    const Word k = key[w];
    for (std::size_t i = 0; i < W; ++i) out[4 + w][i] = out[4 + w][i] + k;
  }
  for (std::size_t i = 0; i < W; ++i) {
    out[12][i] = out[12][i] + x0[i];
    out[13][i] = out[13][i] + x1;
    out[14][i] = out[14][i] + x2;
    out[15][i] = out[15][i] + x3[i];
  }
}

// Reads a key from 32 bytes: eight words, each little-endian (RFC 8439, section
// 2.3).
inline Key key_from_bytes(const std::uint8_t (&bytes)[32]) {
  Key k;
  for (int i = 0; i < 8; ++i)
    k[i] = Word(bytes[4 * i]) | Word(bytes[4 * i + 1]) << 8 |
           Word(bytes[4 * i + 2]) << 16 | Word(bytes[4 * i + 3]) << 24;
  return k;
}

// Overwrites a key with zeros. The volatile pointer keeps the compiler from
// removing the writes as dead stores. Copies that the compiler made in registers
// or on the stack are not erased.
inline void wipe(Key& key) {
  volatile Word* p = key.data();
  for (int i = 0; i < 8; ++i) p[i] = 0;
}

}  // namespace coins
