// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A generator of the prover's private randomness (`coin` in identifiers): the random values that
// do the hiding. It is built only from the ChaCha20 block function f (chacha20_block()) and shares
// nothing with the hash of the transcript and the trees (prover/common/hash.h): no function, no
// constant, no key. No verifier and no proof sees it, and soundness assumes nothing of it.
//
// Each output block has a label of sixteen words: a context (such as a request's proof) and a call.
//   call:    w0 = call_lo, w1 = stream, w2 = retry, w3 = call_hi
//   context: w4 = counter_lo, w5 = counter_hi, w6 .. w9 = nonce, w10 = the format's version word,
//            w11 = kind of object, w12 = a request's number in its batch (`member`), w13 .. w15 = 0
// The label goes in four words at a time, starting from the process key K_proc:
//   K1    = f(K_proc; w4, w5, w6, w7)       words 0 to 7
//   K2    = f(K1;     w8, w9, w10, w11)     words 0 to 7
//   K_ctx = f(K2;     w12, w13, w14, w15)   words 0 to 7     coin_context_key()
//   block = f(K_ctx;  w0, w1, w2, w3)       all 16 words     coin_output()
// Callers must keep two rules on labels: rule 1 at coin_context_key(), rule 2 at coin_key_word().
#pragma once
#include "common/end_name.h"
#include "common/field_host.h" // HD
#include "common/setting.h"    // COINKEYW, NONCEW
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/random.h>

constexpr int COINOUTW = 16; // the words of a block, which is the output of one call
static_assert(COINKEYW == 8 && NONCEW == 4, "ChaCha20 has a key of 256 bits; the nonce is the label's words 6 to 9");

namespace coin_chacha {
// Rotates left by n, 0 < n < 32.
HD uint32_t rotl(uint32_t x, int n) {
#if defined(__CUDA_ARCH__)
  return __funnelshift_l(x, x, n);
#else
  return (x << n) | (x >> (32 - n));
#endif
}
// The ChaCha quarter round: replaces the four words by its four results. The arguments must be four
// different objects. The words are secret, so only additions, exclusive ors and rotations by fixed
// counts touch them. Each rotation is a call of rotl(), the only rotation in this file.
// Adapted from the macro QUARTERROUND of D. J. Bernstein's ChaCha code (chacha-merged.c, version 20080118, public domain).
// RFC 8439, section 2.1, specifies the same function.
HD void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  a = a + b, d = rotl(d ^ a, 16);
  c = c + d, b = rotl(b ^ c, 12);
  a = a + b, d = rotl(d ^ a, 8);
  c = c + d, b = rotl(b ^ c, 7);
}
} // namespace coin_chacha

// One column round and one diagonal round (RFC 8439, 2.3). The order of the eight quarter rounds is that of the round
// loop in D. J. Bernstein's chacha-merged.c.
#define COIN_DOUBLE_ROUND()                                                                                            \
  do {                                                                                                                 \
    coin_chacha::quarter_round(x0, x4, x8, x12);                                                                       \
    coin_chacha::quarter_round(x1, x5, x9, x13);                                                                       \
    coin_chacha::quarter_round(x2, x6, x10, x14);                                                                      \
    coin_chacha::quarter_round(x3, x7, x11, x15);                                                                      \
    coin_chacha::quarter_round(x0, x5, x10, x15);                                                                      \
    coin_chacha::quarter_round(x1, x6, x11, x12);                                                                      \
    coin_chacha::quarter_round(x2, x7, x8, x13);                                                                       \
    coin_chacha::quarter_round(x3, x4, x9, x14);                                                                       \
  } while (0)

// out = f(key; i12, i13, i14, i15), the block function of RFC 8439, section 2.3. The state's words
// 0 to 3 are the RFC's constants and words 4 to 11 the key. Word 12, where the RFC has its block
// counter, is i12; words 13 to 15, where it has its nonce, are i13 to i15. After twenty rounds the
// first state is added. out may not overlap key.
// Host and GPU run the same source of the functions marked HD (rotl() apart), and g++ compiles it
// as plain C++. No HD function of this file loops over rounds or indexes an array by a variable, so
// that in a kernel all words can stay in registers. That is the compiler's choice, and no test of
// this repository checks it (docs/limits-all-rows.tsv, row H39.3). No test at all checks it for
// coin_again(), which kernels of prover/hiding/trees.cuh call: see there, at k_fill().
// prover/tests/coins_rfc8439_test.cpp compares this function with RFC 8439's test vectors.
// cp_commit and cp_prove repeat the comparison, on the host and in a kernel, in coins_own_test()
// (prover/hiding/coins_own.cuh).
// No test of this repository compares coin_key() or cp_key() with a test vector of RFC 8439.
HD void chacha20_block(uint32_t out[COINOUTW], const uint32_t key[COINKEYW], uint32_t i12, uint32_t i13, uint32_t i14,
                       uint32_t i15) {
  const uint32_t c0 = 0x61707865u, c1 = 0x3320646eu, c2 = 0x79622d32u, c3 = 0x6b206574u;
  uint32_t x0 = c0, x1 = c1, x2 = c2, x3 = c3;
  uint32_t x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
  uint32_t x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7];
  uint32_t x12 = i12, x13 = i13, x14 = i14, x15 = i15;

  COIN_DOUBLE_ROUND(); //  1
  COIN_DOUBLE_ROUND(); //  2
  COIN_DOUBLE_ROUND(); //  3
  COIN_DOUBLE_ROUND(); //  4
  COIN_DOUBLE_ROUND(); //  5
  COIN_DOUBLE_ROUND(); //  6
  COIN_DOUBLE_ROUND(); //  7
  COIN_DOUBLE_ROUND(); //  8
  COIN_DOUBLE_ROUND(); //  9
  COIN_DOUBLE_ROUND(); // 10

  out[0] = x0 + c0, out[1] = x1 + c1, out[2] = x2 + c2, out[3] = x3 + c3;
  out[4] = x4 + key[0], out[5] = x5 + key[1], out[6] = x6 + key[2], out[7] = x7 + key[3];
  out[8] = x8 + key[4], out[9] = x9 + key[5], out[10] = x10 + key[6], out[11] = x11 + key[7];
  out[12] = x12 + i12, out[13] = x13 + i13, out[14] = x14 + i14, out[15] = x15 + i15;
}

#undef COIN_DOUBLE_ROUND

// The block of one call: call number 'call' of a stream, at this retry (0 for the first try), under a context's key.
HD void coin_output(uint32_t out[COINOUTW], const uint32_t context_key[COINKEYW], uint32_t stream, uint64_t call,
                   uint32_t retry) {
  chacha20_block(out, context_key, (uint32_t)call, stream, retry, (uint32_t)(call >> 32));
}

// Host only. context_key = K_ctx for this process key and the label's words w4 to w12. K1, K2 and
// K_ctx are as secret as a key: anyone who has one can derive every block below it. So the local
// blocks are zeroed. K_ctx must not be stored, because a stale or wrongly shared K_ctx would repeat
// one-time values in proofs that still verify: a host function that launches kernels which draw
// derives it again, keeps it in a local variable for its own launches and zeroes it before it
// returns.
// process_key comes from the key schedule in prover/common/coin_keys.h: from the commitment's seed,
// 32 fresh bytes from the system and a digest of what the run works on (zeros stand for the 32
// bytes in the anchor's making, and in all of cp_commit's makings under a test seed or
// -public-weights, where nothing is hidden; a test build's -test-coins names a file in the place
// of a batch's). The schedule is defined in
// docs/format/COIN-KEYS.md, "2. The schedule". The programs always pass a nonce of zeros.
// Rule 1 of the labels, objects. Under a batch's K_proc two generators draw: this one, through
//    prover/hiding/trees.cuh (only the one-time mask entries of the sub-proofs' message slots:
//    `slot_pads` in identifiers, Tree::slot_pads()), and that of blocks/coins/coins.h, created in
//    prover/hiding/engine_coins.cuh (all other values). Their counters may coincide, and neither
//    sees the other's record of draws. That is safe only because w11, the kind of object, differs:
//    1 here (3 in a unit test) against 2 (a request) and 4 (the batch) there, and 5 and 10 there
//    under cp_commit's keys (prover/hiding/long_rows_words.h has the static_asserts). No code may
//    draw with the other generator's object.
static inline void coin_context_key(uint32_t context_key[COINKEYW], const uint32_t process_key[COINKEYW],
                                    uint32_t counter_lo, uint32_t counter_hi, uint32_t version, uint32_t object,
                                    const uint32_t nonce[NONCEW], uint32_t member) {
  uint32_t a[COINOUTW], b[COINOUTW];
  chacha20_block(a, process_key, counter_lo, counter_hi, nonce[0], nonce[1]);
  chacha20_block(b, a, nonce[2], nonce[3], version, object);
  chacha20_block(a, b, member, 0, 0, 0);
  for (int i = 0; i < COINKEYW; i++) context_key[i] = a[i];
  volatile uint32_t *va = a, *vb = b;
  for (int i = 0; i < COINOUTW; i++) va[i] = vb[i] = 0;
}

// ---- Steps of the key schedule, and fresh bytes. Host only. ----
// Tags in the first input word. Under the seed: CP_SD, for a child key. Under a batch's first key:
// CP_LF, for the choice of its leaf table (draw_leaf() in prover/common/cp_files.h), and CP_BK, for
// its process key. The programs take only CP_LF from here: for the other two tags they use the
// equal constants COIN_SEED_WORD (blocks/coins/coins.h) and BATCH_WORD (prover/common/coin_keys.h).
constexpr uint32_t CP_SD = 0x44535043, CP_LF = 0x464C5043, CP_BK = 0x4B425043; // "CPSD", "CPLF", "CPBK"

// The schedule's key(K; a, b, c, d): out = words 0 to 7 of f(K; a, b, c, d). out may be K.
static inline void coin_key(uint32_t out[COINKEYW], const uint32_t K[COINKEYW], uint32_t a, uint32_t b, uint32_t c,
                            uint32_t d) {
  uint32_t block[COINOUTW];
  chacha20_block(block, K, a, b, c, d);
  for (int i = 0; i < COINKEYW; i++) out[i] = block[i];
  volatile uint32_t *v = block;
  for (int i = 0; i < COINOUTW; i++) v[i] = 0;
}
// The comparison word of a process key [own-test-key]: word 0 of f(K_proc; 0, 0, "CPKW", 0). It
// serves to check that two copies of a key agree: key32_word(), key32_word_of() in
// prover/hiding/trees.cuh.
// Rule 2 of the labels, the comparison word. No other call has that input. Every other call under
//    K_proc, in either generator of rule 1, derives K1 from (w4, w5, w6, w7), and w6 is a nonce
//    word, which the programs always pass as zero. A format that gave the nonce words a use would
//    have to keep w6 away from "CPKW". The word is compared inside the process and never leaves it:
//    it goes into no file, no message and no printed line.
constexpr uint32_t CP_KW = 0x574B5043; // "CPKW"
static inline uint32_t coin_key_word(const uint32_t process_key[COINKEYW]) {
  uint32_t out[COINKEYW];
  coin_key(out, process_key, 0, 0, CP_KW, 0);
  const uint32_t word = out[0];
  volatile uint32_t *v = out;
  for (int i = 0; i < COINKEYW; i++) v[i] = 0;
  return word;
}
// coin_key() on bytes: a key's 32 bytes are its eight words, little-endian. out may be K.
static inline void cp_key(uint8_t out[4 * COINKEYW], const uint8_t K[4 * COINKEYW], uint32_t a, uint32_t b, uint32_t c,
                          uint32_t d) {
  uint32_t w[COINKEYW];
  for (int i = 0; i < COINKEYW; i++)
    w[i] = (uint32_t)K[4 * i] | (uint32_t)K[4 * i + 1] << 8 | (uint32_t)K[4 * i + 2] << 16 | (uint32_t)K[4 * i + 3] << 24;
  coin_key(w, w, a, b, c, d);
  for (int i = 0; i < 4 * COINKEYW; i++) out[i] = (uint8_t)(w[i / 4] >> 8 * (i % 4));
  volatile uint32_t *v = w;
  for (int i = 0; i < COINKEYW; i++) v[i] = 0;
}
// Fills `out` with 32 fresh bytes by one call of getrandom(): in cp_prove for each batch; in
// cp_commit for each leaf table and each time it makes the key file or a leaf file. A call that a
// signal interrupted before it gave a byte is repeated. If the call fails, gives fewer bytes or
// gives zeros only, as a dead source does (a sound one with probability 2^-256), this function
// calls exit(2), which cp_prove's exit handler turns into exit code 1 (end_on_exit() in
// prover/prover/ends.h).
static inline void cp_fresh32(uint8_t out[32]) {
  ssize_t given;
  do given = ::getrandom(out, 32, 0);
  while (given < 0 && errno == EINTR);
  uint8_t any = 0;
  if (given == 32)
    for (int i = 0; i < 32; i++) any |= out[i];
  if (any) return;
  volatile uint8_t *v = out;
  for (int i = 0; i < 32; i++) v[i] = 0;
  if (given == 32) // for cp_prove's last line
    name_the_end("source-zero");
  else
    name_the_end("source-short");
  if (given == 32)
    fprintf(stderr, "source-zero: the system's random generator (getrandom) gave 32 zero bytes, which a sound generator "
                    "does not do; the program ends. Check the generator before starting again\n");
  else
    fprintf(stderr, "source-short: the system's random generator (getrandom) failed or gave fewer than 32 bytes; "
                    "the program ends\n");
  exit(2);
}
