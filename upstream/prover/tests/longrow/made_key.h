// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// For GPU tests of the prover's private randomness (`coin` in identifiers). namespace second is a
// second CPU implementation of the generator in prover/common/coin_chacha.h, written from the rule
// at the top of that file. Its keys and labels (sixteen words that select a block; words 4 to 15
// name the context) follow that rule. block(), context_key(), a_try() and cell() call none of that
// file's functions.
// Limit: second::block() is close to coin_chacha.h's block function: both are adapted from one
// public-domain file (chacha-merged.c; see second::block()) and keep its form. Their agreement
// therefore cannot show an error taken from chacha-merged.c, or one made alike in adapting it.
// No test vector of RFC 8439 is checked here: second::differs_from_the_reference() compares
// `second` with prover/longrow/reference.h, which calls coin_chacha.h's functions, and
// prover/tests/coins_rfc8439_test.cpp checks coin_chacha.h's chacha20_block() on the RFC's vectors.
// second::cell() takes a context's key, like a kernel; lr::cell() derives it from the process key.
// made_key() makes a synthetic process key and context in the reference's types, lr::Key and
// lr::Params; card_key_of() gives their lr::CoinKey. Only differs_from_the_reference() uses them.
#pragma once
#include "longrow/reference.h"

namespace lr {
// A context's key with the context's nine label words. Only this file uses it.
struct CoinKey {
  u32 key[COINKEYW];
  u32 ctr_lo, ctr_hi, nonce[NONCEW], working_word, obj, mem;
};
} // namespace lr
// The lr::CoinKey for k and p.working_word (the version word). `card` in identifiers means the GPU.
static inline lr::CoinKey card_key_of(const lr::Key &k, const lr::Params &p = {}) {
  lr::CoinKey c;
  lr::context_key(c.key, k, p);
  c.ctr_lo = (u32)k.ctr, c.ctr_hi = (u32)(k.ctr >> 32);
  for (int i = 0; i < NONCEW; i++) c.nonce[i] = k.nonce[i];
  c.working_word = p.working_word, c.obj = k.obj, c.mem = k.mem;
  return c;
}
struct MadeKey {
  lr::Key host;
  lr::Params params;
  lr::CoinKey card() const { return card_key_of(host, params); }
};
// A synthetic process key and context, derived from `seed`. None of their words is zero.
static inline MadeKey made_key(u64 seed) {
  MadeKey m;
  for (int i = 0; i < COINKEYW; i++) m.host.key[i] = (u32)mix64(seed + i) | 1;
  m.host.ctr = ((u32)mix64(seed + 20) | 1) | (u64)(1 + seed % 3) << 32;
  for (int i = 0; i < NONCEW; i++) m.host.nonce[i] = (u32)mix64(seed + 30 + i) | 1;
  m.params.working_word = 2135742722u, m.host.obj = seed & 1 ? 3 : 1, m.host.mem = 1 + (u32)(seed % 24);
  return m;
}

namespace second {
// The ChaCha20 block for a key of eight words and the state's last four words (RFC 8439, 2.3).
// The state's first four words, the loop of ten double rounds and the final addition of the initial
// state are adapted from the key setup and the encryption function of the file that is named below.
static inline void block(u32 out[16], const u32 key[8], const u32 last[4]) {
  const auto word = [](const unsigned char *p) { // four bytes as a little-endian word
    return (u32)p[0] | (u32)p[1] << 8 | (u32)p[2] << 16 | (u32)p[3] << 24;
  };
  const auto rotate = [](u32 v, int c) { return v << c | v >> (32 - c); };
  // Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version
  // 20080118, public domain).
  // RFC 8439, section 2.1, specifies the same function.
  const auto quarter_round = [&](u32 &a, u32 &b, u32 &c, u32 &d) {
    a += b, d = rotate(d ^ a, 16);
    c += d, b = rotate(b ^ c, 12);
    a += b, d = rotate(d ^ a, 8);
    c += d, b = rotate(b ^ c, 7);
  };
  const unsigned char constants[17] = "expand 32-byte k";
  const u32 j0 = word(constants + 0), j1 = word(constants + 4), j2 = word(constants + 8), j3 = word(constants + 12);
  const u32 j4 = key[0], j5 = key[1], j6 = key[2], j7 = key[3];
  const u32 j8 = key[4], j9 = key[5], j10 = key[6], j11 = key[7];
  const u32 j12 = last[0], j13 = last[1], j14 = last[2], j15 = last[3];
  u32 x0 = j0, x1 = j1, x2 = j2, x3 = j3, x4 = j4, x5 = j5, x6 = j6, x7 = j7;
  u32 x8 = j8, x9 = j9, x10 = j10, x11 = j11, x12 = j12, x13 = j13, x14 = j14, x15 = j15;
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
// The context's key: the label's words 4 to 15 go in four at a time, and words 0 to 7 of an output
// are the next key.
static inline void context_key(u32 out[8], const lr::Key &k, const lr::Params &p) {
  const u32 w[12] = {(u32)k.ctr, (u32)(k.ctr >> 32), k.nonce[0], k.nonce[1], k.nonce[2], k.nonce[3], p.working_word,
                     k.obj,      k.mem,              0,          0,          0};
  for (int i = 0; i < 8; i++) out[i] = k.key[i];
  for (int part = 0; part < 3; part++) {
    u32 o[16];
    block(o, out, w + 4 * part);
    for (int i = 0; i < 8; i++) out[i] = o[i];
  }
}
// The value that try number tr gives for cell i of a stream, under a context's key. A block gives
// five cells (field elements) if the stream's bits 14 and 15 are 2 (lrw::SALT), otherwise four.
static inline u64 a_try(const u32 key[8], u32 stream, u64 i, u32 tr) {
  const u32 per = ((stream >> 14) & 3) == 2 ? 5 : 4, q = (u32)(i % per);
  const u64 bk = i / per;
  const u32 last[4] = {(u32)bk, stream, tr ? 8 * tr + q : 0, (u32)(bk >> 32)};
  u32 o[16];
  block(o, key, last);
  const int w = tr ? 0 : 2 * q;
  // As many low bits as the field's modulus, lr::P, has: all 64, so nothing is masked. The length
  // is fixed here, not taken from the programs' constants.
  return o[w] | (u64)o[w + 1] << 32;
}
// Cell i: the value of the first try below lr::P that is neither `rejected` nor `rejected_too`.
static inline u64 cell(const u32 key[8], u32 stream, u64 i, u64 rejected = lr::P, u64 rejected_too = lr::P) {
  for (u32 tr = 0;; tr++) {
    const u64 v = a_try(key, stream, i, tr);
    if (v < lr::P && v != rejected && v != rejected_too) return v;
  }
}
// Counts the cells that differ from the reference's lr::cell() (40 of each kind of stream for each
// of four synthetic keys) and the context keys' differing words. One test fails unless the count
// is 0, so that matching `second` also means matching the reference:
// prover/tests/coin_contexts_gpu_test.cu. A rejected try is rare, so a_try() with tr > 0 is most
// likely not compared.
static inline size_t differs_from_the_reference() {
  size_t off = 0;
  for (u64 seed = 1; seed <= 4; seed++) {
    const MadeKey m = made_key(seed);
    const lr::CoinKey c = m.card();
    u32 own[8];
    second::context_key(own, m.host, m.params);
    for (int i = 0; i < 8; i++) off += own[i] != c.key[i];
    for (u32 kind = 0; kind < 4; kind++)
      for (u64 i = 0; i < 40; i++) {
        const u32 st = lrw::stream(5 + (u32)seed, (lrw::Kind)kind, kind == 2 ? 0 : 3);
        off += second::cell(own, st, i) != lr::cell(m.host, st, i, m.params);
      }
  }
  return off;
}
} // namespace second
