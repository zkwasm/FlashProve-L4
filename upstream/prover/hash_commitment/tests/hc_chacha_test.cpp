// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// CPU test of prover/hash_commitment/hc_chacha.cuh: the ChaCha20 block function, hc_block_value()
// and hc_reduce_for_tests(). No GPU, no CUDA.
//   from prover/:  OUT=<dir> ./build.sh hc_host_test tests/hc_chacha_test
//                  <dir>/hc/test_build/tests/hc_chacha_test
// It prints "all N tests passed" and exits with code 0, or lists the failed tests and exits with 1.
//
// Second use, to compare with OpenSSL; no script or test of this repository runs it:
//   ./hc_chacha_test --block KEY_AS_64_HEX_DIGITS x12 x13 x14 x15
// prints the block's 64 bytes (each word low byte first, as RFC 8439 serialises it) in hex.
// "Key" is the RFC's word for the 32-byte mask seed (`hc_seed32`); masks are random values encoded
// with the weights. The inputs x12 .. x15 are state words 12 to 15: the RFC's counter and nonce.
//
// Expected values. The quarter round, RFC_232 and ZERO_BLOCK are from RFC 8439, 2.1.1, 2.3.2 and
// appendix A.1, test vector 1; "openssl enc -chacha20" (3.0.13) gives both blocks too. BLOCK_5_7
// and BLOCK_FAR are its output on 64 zero bytes, with 16-byte iv = x12 .. x15, each low byte first.
// This test does not cover how blocks become masks (inputs, rejected blocks);
// prover/tests/masking_cells_test.cpp covers that.
#include "hash_commitment/hc_chacha.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int n_run = 0, n_bad = 0;

static void ok(bool cond, const char *what) {
  ++n_run;
  if (!cond) {
    ++n_bad;
    std::printf("FAILED: %s\n", what);
  }
}

static bool same(const uint32_t *x, const uint32_t *y, int n) {
  for (int i = 0; i < n; ++i)
    if (x[i] != y[i]) return false;
  return true;
}

// v mod p the slow way, to compare with hc_reduce_for_tests.
static uint64_t slow_mod_p(uint64_t v) {
  return v % HC_P;
}

static void rfc_key(hc_seed32 &key) {
  uint8_t kb[32];
  for (int i = 0; i < 32; ++i) kb[i] = uint8_t(i);
  hc_seed_from_bytes(kb, key);
}

static const uint32_t RFC_232[16] = {
    0xe4e7f110u, 0x15593bd1u, 0x1fdd0f50u, 0xc47120a3u, 0xc7f4d1c7u, 0x0368c033u, 0x9aaa2204u, 0x4e6cd4c3u,
    0x466482d2u, 0x09aa9f07u, 0x05d7c214u, 0xa2028bd9u, 0xd19c12b5u, 0xb94e16deu, 0xe883d0cbu, 0x4e3c50a2u,
};

// RFC 8439, A.1, test vector 1: key zero, all four inputs zero.
static const uint32_t ZERO_BLOCK[16] = {
    0xade0b876u, 0x903df1a0u, 0xe56a5d40u, 0x28bd8653u, 0xb819d2bdu, 0x1aed8da0u, 0xccef36a8u, 0xc70d778bu,
    0x7c5941dau, 0x8d485751u, 0x3fe02477u, 0x374ad8b8u, 0xf4b8436au, 0x1ca11815u, 0x69b687c3u, 0x8665eeb2u,
};

// Key 00..1f: the block at the inputs (5, 7, 0, 0).
static const uint32_t BLOCK_5_7[16] = {
    0x01e8b527u, 0x0c2cc867u, 0xa64e8130u, 0xbcf2f782u, 0x226bcbffu, 0x94834a0bu, 0x246a63eau, 0xf5ddd1eeu,
    0xcf14081au, 0x5f9a9d7au, 0xd7ce50fdu, 0x430e9cf3u, 0x0405e062u, 0x2859dd39u, 0xf3e0a328u, 0xda3d3e75u,
};

// Key 00..1f: the block at the inputs (2^21 - 1, 2^22 - 1, 0, 0).
static const uint32_t BLOCK_FAR[16] = {
    0x5aa2e44eu, 0xd150caa3u, 0x374f98d2u, 0x0814759bu, 0x6d9a2f2fu, 0xca38eef7u, 0x4033249cu, 0xc3a7bcb8u,
    0x648acb86u, 0x466d74d2u, 0x0a37bca3u, 0x97751333u, 0x1c77e030u, 0xed53f7c4u, 0x89612d3cu, 0xffffa2c0u,
};

static void test_cipher() {
  // RFC 8439, 2.1.1.
  uint32_t a = 0x11111111u, b = 0x01020304u, c = 0x9b8d6f43u, d = 0x01234567u;
  hc_quarter_round(a, b, c, d);
  ok(a == 0xea2a92f4u && b == 0xcb1cf8ceu && c == 0x4581472eu && d == 0x5881c4bbu,
     "the quarter round of RFC 8439, 2.1.1");

  hc_seed32 key;
  uint32_t o[16];
  rfc_key(key);
  const uint32_t key_words[8] = {0x03020100u, 0x07060504u, 0x0b0a0908u, 0x0f0e0d0cu,
                                 0x13121110u, 0x17161514u, 0x1b1a1918u, 0x1f1e1d1cu};
  ok(same(key.w, key_words, 8), "the key's bytes are read low byte first (RFC 8439, 2.3.2, initial state)");

  // Counter 1 is x12. The nonce 00 00 00 09 | 00 00 00 4a | 00 00 00 00 is x13, x14, x15,
  // low byte first.
  chacha20_block(key, 1u, 0x09000000u, 0x4a000000u, 0u, o);
  ok(same(o, RFC_232, 16), "the block of RFC 8439, 2.3.2");

  const hc_seed32 zero_key = {};
  chacha20_block(zero_key, 0u, 0u, 0u, 0u, o);
  ok(same(o, ZERO_BLOCK, 16), "the block of RFC 8439, A.1, test vector 1");

  chacha20_block(key, 5u, 7u, 0u, 0u, o);
  ok(same(o, BLOCK_5_7, 16), "the block at (5, 7, 0, 0), against openssl");
  chacha20_block(key, (1u << 21) - 1, (1u << 22) - 1, 0u, 0u, o);
  ok(same(o, BLOCK_FAR, 16), "the block at (2^21 - 1, 2^22 - 1, 0, 0), against openssl");

  // The first input is x12 and the second x13, not the other way round.
  uint32_t o2[16];
  chacha20_block(key, 7u, 5u, 0u, 0u, o2);
  ok(!same(o2, BLOCK_5_7, 16), "the inputs (5, 7, 0, 0) and (7, 5, 0, 0) have different blocks");
}

// p = 2^64 - 2^32 + 1. "The threshold" in the messages is HC_MASK_LIMIT, the greatest multiple of
// p below 2^64, which is p itself.
static void test_reduce() {
  struct {
    uint64_t v, want;
    const char *what;
  } cases[] = {
      {0, 0, "0 mod p = 0"},
      {1, 1, "1 mod p = 1"},
      {HC_P - 1, HC_P - 1, "(p - 1) mod p = p - 1"},
      {HC_P, 0, "p mod p = 0"},
      {HC_P + 1, 1, "(p + 1) mod p = 1"},
      {HC_MASK_LIMIT - 1, HC_P - 1, "(the threshold - 1) mod p = p - 1"},
      {HC_MASK_LIMIT, 0, "the threshold mod p = 0"},
      {HC_MASK_LIMIT + 1, 1, "(the threshold + 1) mod p = 1"},
      {uint64_t(1) << 32, uint64_t(1) << 32, "2^32 mod p = 2^32"},
      {uint64_t(1) << 63, uint64_t(1) << 63, "2^63 mod p = 2^63"},
      {0xFFFFFFFF00000000ull, 0xFFFFFFFF00000000ull, "(2^64 - 2^32) mod p = p - 1"},
      {0xFFFFFFFF00000001ull, 0, "(2^64 - 2^32 + 1) mod p = 0"},
      {~uint64_t(0), 0xFFFFFFFEull, "(2^64 - 1) mod p = 2^32 - 2"},
  };
  for (auto &c : cases) {
    ok(hc_reduce_for_tests(c.v) == c.want, c.what);
    ok(slow_mod_p(c.v) == c.want, c.what); // the expected value itself
  }
  // Near every multiple of p below 2^64, then scattered values, against slow_mod_p().
  bool all = true, below = true;
  for (uint64_t m = 0; m <= 8; ++m)
    for (int64_t e = -9; e <= 9; ++e) {
      uint64_t v = m * HC_P + uint64_t(e); // can wrap, which is fine: any v is a case
      uint64_t r = hc_reduce_for_tests(v);
      all &= r == slow_mod_p(v);
      below &= r < HC_P;
    }
  // The generator below is xorshift64: G. Marsaglia, "Xorshift RNGs", Journal of Statistical Software 8 (14), 2003.
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < 1000000; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    uint64_t r = hc_reduce_for_tests(s);
    all &= r == slow_mod_p(s);
    below &= r < HC_P;
  }
  ok(all, "hc_reduce_for_tests equals v % p near every multiple of p and on 10^6 scattered values");
  ok(below, "hc_reduce_for_tests is always below p");
}

static void test_value() {
  uint32_t o[16];
  // The low word comes first: value sigma = o[2 sigma] + 2^32 o[2 sigma + 1].
  for (int i = 0; i < 16; ++i) o[i] = 0;
  o[6] = 0x89ABCDEFu;
  o[7] = 0x01234567u;
  ok(hc_block_value(o, 3) == 0x0123456789ABCDEFull, "value 3 of a block is o[6] + 2^32 o[7]");
}

static int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int print_block(int argc, char **argv) {
  if (argc != 7 || std::strlen(argv[2]) != 64) {
    std::fprintf(stderr, "usage: hc_chacha_test --block KEY_AS_64_HEX_DIGITS x12 x13 x14 x15\n");
    return 2;
  }
  uint8_t kb[32];
  for (int i = 0; i < 32; ++i) {
    int h = hex_digit(argv[2][2 * i]), l = hex_digit(argv[2][2 * i + 1]);
    if (h < 0 || l < 0) {
      std::fprintf(stderr, "the key is not hex\n");
      return 2;
    }
    kb[i] = uint8_t(16 * h + l);
  }
  hc_seed32 key;
  uint32_t o[16], x[4];
  hc_seed_from_bytes(kb, key);
  for (int i = 0; i < 4; ++i) x[i] = uint32_t(std::strtoul(argv[3 + i], nullptr, 0));
  chacha20_block(key, x[0], x[1], x[2], x[3], o);
  for (int i = 0; i < 16; ++i)
    std::printf("%02x%02x%02x%02x", o[i] & 255u, (o[i] >> 8) & 255u, (o[i] >> 16) & 255u, o[i] >> 24);
  std::printf("\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "--block") == 0) return print_block(argc, argv);
  test_cipher();
  test_reduce();
  test_value();
  if (n_bad) {
    std::printf("%d of %d tests FAILED\n", n_bad, n_run);
    return 1;
  }
  std::printf("all %d tests passed\n", n_run);
  return 0;
}
