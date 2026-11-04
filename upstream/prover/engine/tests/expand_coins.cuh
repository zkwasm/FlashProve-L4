// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Deterministic private randomness (`Coins` in identifiers) for tests and benchmarks of the GPU
// engine (prover/engine/engine_gpu.cuh). Not for a prover: the key is a number that the test
// chooses, so nothing made with these values is hidden. A value depends only on the key, its call
// number (each method call takes the next) and its index in the call, so host and GPU (`_card`)
// methods agree and CPU- and GPU-engine runs can be compared byte for byte.
// Call numbers start at 1: call 0 would repeat ToyCoins (blocks/toy/toy_coins.h) for the same key.
#pragma once
#include "engine/card_coins.cuh"
#include "hash/hash.h"

namespace blk {

// Counter for test_coins() (blocks/hash/hash.h: 32 bytes per counter): call number, attempt,
// index of the 32-byte block within the call. ExpandCoins::next() keeps call and block within
// their bit fields.
BLK_HD inline u64 expand_coins_counter(u64 call, u64 attempt, u64 block) { return call << 44 | attempt << 40 | block; }
BLK_FN u64 expand_coins_word(const Digest &key, u64 call, u64 attempt, u64 i) {
  const Digest d = test_coins(key, expand_coins_counter(call, attempt, i / 4));
  return d.w[2 * (i % 4)] | (u64)d.w[2 * (i % 4) + 1] << 32;
}
// Element i of a call: the first of 16 attempts that F_from_word() accepts, or 0 (chance < 2^-512).
BLK_FN F expand_coins_element(const Digest &key, u64 call, u64 i) {
  F x = F::zero();
  for (u64 attempt = 0; attempt < 16; attempt++)
    if (F_from_word(expand_coins_word(key, call, attempt, i), x)) break;
  return x;
}
static __global__ void k_expand_coins_fill(Digest key, u64 call, F *out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expand_coins_element(key, call, i);
}
// Writes n8 64-bit words; each thread writes the words of one 32-byte block, at most four.
static __global__ void k_expand_coins_words(Digest key, u64 call, u64 *out, size_t n8) {
  const size_t block = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (4 * block >= n8) return;
  const Digest d = test_coins(key, expand_coins_counter(call, 0, block));
  for (size_t i = 0; i < 4 && 4 * block + i < n8; i++) out[4 * block + i] = d.w[2 * i] | (u64)d.w[2 * i + 1] << 32;
}

class ExpandCoins : public CardCoins {
public:
  explicit ExpandCoins(u64 key) {
    u8 b[DIGEST_BYTES] = {};
    word_to_bytes(key, b);
    digest_from_bytes(b, key_);
  }
  void fill(F *out, size_t n) override {
    const u64 call = next(n / 4);
    for (size_t i = 0; i < n; i++) out[i] = expand_coins_element(key_, call, i);
  }
  void bytes(u8 *out, size_t n) override {
    const u64 call = next(n / 32);
    for (size_t i = 0; i < n; i++) out[i] = (u8)(expand_coins_word(key_, call, 0, i / 8) >> 8 * (i % 8));
  }
  void fill_card(F *out, size_t n, cudaStream_t stream) override {
    const u64 call = next(n / 4);
    if (!n) return;
    k_expand_coins_fill<<<blocks_for(n, 256), 256, 0, stream>>>(key_, call, out, n);
    ENGINE_CK(cudaGetLastError());
  }
  // `out` must be 8-byte aligned: not checked.
  void bytes_card(u8 *out, size_t n, cudaStream_t stream) override {
    BLK_REQUIRE(n % 8 == 0, "coins: bytes for the GPU come in words of 8");
    const u64 call = next(n / 32);
    if (!n) return;
    k_expand_coins_words<<<blocks_for((n / 8 + 3) / 4, 256), 256, 0, stream>>>(key_, call, reinterpret_cast<u64 *>(out), n / 8);
    ENGINE_CK(cudaGetLastError());
  }

private:
  u64 next(u64 blocks) {
    BLK_REQUIRE(calls_ < (1u << 18) && blocks < ((u64)1 << 40), "coins: this test generator has run out of calls or of room");
    return calls_++;
  }
  Digest key_;
  u64 calls_ = 1;
};

} // namespace blk
