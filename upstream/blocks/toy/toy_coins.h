// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Private randomness (`Coins`, blocks/engine/table.h) for tests: the hashes test_coins(key_, i),
// i = 0, 1, ..., under a 32-byte key. A fixed key, a 64-bit number padded with zeros, makes a run
// repeatable; fresh() takes all 32 bytes from the operating system. The toy programs of this
// directory draw only their synthetic witnesses from it.
#pragma once
#include <cstdio>
#include <cstring>

#include "engine/table.h"
#include "hash/hash.h"

namespace blk {

class ToyCoins : public Coins {
public:
  explicit ToyCoins(u64 key) {
    u8 b[DIGEST_BYTES] = {};
    word_to_bytes(key, b);
    digest_from_bytes(b, key_);
  }
  static ToyCoins fresh() {
    ToyCoins c(0);
    u8 b[DIGEST_BYTES];
    std::FILE *f = std::fopen("/dev/urandom", "rb");
    BLK_REQUIRE(f && std::fread(b, 1, sizeof b, f) == sizeof b, "coins: the operating system gives no random bytes");
    std::fclose(f);
    digest_from_bytes(b, c.key_);
    return c;
  }
  void bytes(u8 *out, size_t n) override {
    while (n) {
      if (have_ == 0) {
        digest_to_bytes(test_coins(key_, counter_++), block_);
        have_ = DIGEST_BYTES;
      }
      const size_t take = n < have_ ? n : have_;
      std::memcpy(out, block_ + (DIGEST_BYTES - have_), take);
      out += take, n -= take, have_ -= take;
    }
  }
  void fill(F *out, size_t n) override {
    for (size_t i = 0; i < n; i++) {
      u8 w[8];
      do bytes(w, 8);
      while (!F_from_word(word_from_bytes(w), out[i]));
    }
  }

private:
  Digest key_;
  u64 counter_ = 0;
  u8 block_[DIGEST_BYTES];
  size_t have_ = 0;
};

} // namespace blk
