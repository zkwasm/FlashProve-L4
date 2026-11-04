// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// HMAC-SHA-256 of a message under a 32-byte key. Host code.
// Public source: RFC 2104, "HMAC: Keyed-Hashing for Message Authentication", section 2:
// H(K xor opad, H(K xor ipad, text)), where text is the message, K the key zero-padded to
// SHA-256's 64-byte block, ipad 64 bytes of 0x36 and opad 64 bytes of 0x5c.
// RFC 4231 has test cases; prover/tests/hmac_sha256_test.cpp uses four.
#pragma once
#include "weights_commitment/sha256.h"
#include <cstdint>
#include <cstring>

// Feed the message with update(), then call finish() once (not checked; a second result is wrong).
// The destructor zeroes the object, which contains K xor opad.
class HmacSha256 {
public:
  explicit HmacSha256(const uint8_t key[32]) {
    uint8_t inner_block[64];
    for (int i = 0; i < 64; i++) {
      const uint8_t k = i < 32 ? key[i] : 0;
      inner_block[i] = k ^ 0x36, outer_block[i] = k ^ 0x5c;
    }
    inner.update(inner_block, 64);
    explicit_bzero(inner_block, 64);
  }
  ~HmacSha256() { explicit_bzero((void *)this, sizeof *this); }
  HmacSha256(const HmacSha256 &) = delete;
  HmacSha256 &operator=(const HmacSha256 &) = delete;
  void update(const void *bytes, size_t n) { inner.update(bytes, n); }
  void finish(uint8_t out[32]) {
    uint8_t first[32];
    inner.digest(first);
    Sha256 outer;
    outer.update(outer_block, 64);
    outer.update(first, 32);
    outer.digest(out);
    explicit_bzero(&outer, sizeof outer);
    explicit_bzero(first, sizeof first);
  }

private:
  Sha256 inner;
  uint8_t outer_block[64];
};
