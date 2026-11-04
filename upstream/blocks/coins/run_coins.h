// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes the generator of a prover's private randomness (`coins` in identifiers) for one run of a
// program. The run's process key (a ChaCha20 key) is derived from a child key of the prover's
// seed, 32 fresh bytes from the operating system, and a digest (CoinInputs) that must
// cover all that the run's messages depend on: public files, statement, witness. So if the system's
// bytes repeat, different inputs still get different randomness; and the same randomness on the
// same inputs gives the same proof (coins.h states what hiding requires). Before it has a seed, a
// program must run the self-test coins_own_test() or, with a GPU, coins_own_test_card()
// (prover/engine/stream_coins.cuh). This file enforces neither the coverage nor the self-test.
#pragma once
#include <memory>
#include <string>
#include <type_traits>

#include "coins/coins.h"

namespace blk {

// Hashes a run's inputs as named items, each as the name's length, the name, the bytes' length, the
// bytes (a length is 8 bytes, lowest first), so the hashed string can be parsed as a list in only
// one way. `Hasher` (blocks/hash/hash.h) aborts before the string reaches 2^32 bytes.
// The digest is as secret as the most secret item: the destructor wipes the hash state; the caller
// must wipe the digest.
class CoinInputs {
public:
  CoinInputs() : h_(Domain::COIN_INPUTS, 0) {}
  ~CoinInputs() { // the last block's bytes and the chaining value
    static_assert(std::is_trivially_destructible<Hasher>::value && std::is_trivially_copyable<Hasher>::value, "a Hasher holds all that it holds in itself");
    wipe(&h_, sizeof h_);
  }
  CoinInputs(const CoinInputs &) = delete;
  CoinInputs &operator=(const CoinInputs &) = delete;
  // An item with a name and no bytes: a list's title, or a case without data.
  CoinInputs &title(const std::string &name) { return add(name, static_cast<const void *>(""), 0); }
  // A number: 8 bytes, lowest first.
  CoinInputs &number(const std::string &name, u64 value) {
    u8 b[8];
    word_to_bytes(value, b);
    return add(name, static_cast<const void *>(b), sizeof b);
  }
  CoinInputs &add(const std::string &name, const void *bytes, size_t n) {
    h_.push_u64(name.size()), h_.push(reinterpret_cast<const u8 *>(name.data()), name.size());
    h_.push_u64(n), h_.push(static_cast<const u8 *>(bytes), n);
    return *this;
  }
  CoinInputs &add(const std::string &name, const std::string &text) { return add(name, text.data(), text.size()); }
  CoinInputs &add(const std::string &name, const Digest &d) {
    u8 b[DIGEST_BYTES];
    digest_to_bytes(d, b);
    return add(name, b, sizeof b);
  }
  CoinInputs &add(const std::string &name, const F *elements, size_t n) {
    static_assert(sizeof(F) == 8, "an element lies in memory as one word");
    return add(name, static_cast<const void *>(elements), 8 * n);
  }
  Digest finish() { return h_.finish(); }

private:
  Hasher h_;
};

// Generator: StreamCoins, or a class with the same constructor arguments, such as CardStreamCoins
// (prover/engine/stream_coins.cuh), which also writes into GPU memory. kind: the number of the
// seed's child key, which must serve runs of this kind only. context: the proof or message that the
// generator serves. The programs cp_commit and cp_prove call neither this template nor the next;
// `Kind` in prover/common/coin_keys.h lists their numbers.
template <class Generator = StreamCoins>
inline std::unique_ptr<Generator> run_coins(const CoinKey &seed, u32 kind, const Digest &inputs, const CoinContext &context = CoinContext{}) {
  u8 fresh[32];
  fresh32(fresh); // ends the process if the system's source fails
  CoinKey child = seed_child(seed, kind, 0), process = process_key(child, fresh, inputs);
  auto coins = std::make_unique<Generator>(process, context);
  wipe(fresh, sizeof fresh), wipe(child), wipe(process);
  return coins;
}
// For a program that keeps no seed between runs (blocks/toy): the seed is also 32 fresh bytes,
// from a separate call of fresh32().
template <class Generator = StreamCoins> inline std::unique_ptr<Generator> run_coins_without_a_seed(const Digest &inputs, const CoinContext &context = CoinContext{}) {
  u8 bytes[32];
  fresh32(bytes);
  CoinKey seed = coin_key_from_bytes(bytes);
  auto coins = run_coins<Generator>(seed, 0, inputs, context);
  wipe(bytes, sizeof bytes), wipe(seed);
  return coins;
}
} // namespace blk
