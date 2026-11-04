// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// [own-test-coins] (printed on failure): the key-derivation part of this self-test, which the
// programs cp_commit and cp_prove run before reading or drawing a seed; on failure they do
// neither. It is a known-answer test of prover/common/coin_keys.h, which derives the keys of
// their private randomness (`coin` in identifiers) from the seed, and of draw_leaf() in
// prover/common/cp_files.h, which picks the leaf table (`copy_table` in identifiers: a copy of
// the weight table) for a batch of requests. No verifier sees a key, so a fault in either shows
// in no proof: a key that absorbed fewer inputs than it should would still give random values,
// and two batches, or two of cp_commit's tables (`first_table`, `copy_table`) or files
// (`key_file`, `entry`), could share them.
//
// The inputs are synthetic and public, so nothing is zeroed here. At run time cp_commit's keys, or
// values drawn under them, and a batch's two keys are compared with a second implementation's:
// prover/hash_commitment/hf_commit_derivation_check.cuh, prover/prover/batch_keys_held.h.
//
// Host code only.
#pragma once
#include "common/coin_keys.h"
#include "common/cp_files.h"
#include <cstdint>
#include <vector>

namespace cp {
// The expected values come from a third implementation, prover/tests/coin_keys_values.py (hashlib's
// BLAKE2b; ChaCha20's block function as specified in RFC 8439, sections 2.1 to 2.3), which imports
// none of the programs' files. prover/tests/coin_keys_test.cpp compares KEYS_KNOWN and
// KEYS_KNOWN_DRAW, not KEYS_KNOWN_LIVE, with the script's stored output,
// prover/tests/coin_keys_values.txt. keys_own_test() can catch a changed function, a
// miscompilation or a faulty machine, not a misreading of the derivation that script and
// functions share.
struct KeysKnown {
  const char *name, *hex; // a key's 32 bytes: its eight words, each with its lowest byte first
};
constexpr KeysKnown KEYS_KNOWN[7] = {
    {"first_table", "542fa235a5d33a32a8987dee2a0607cdb46d65ead8e8a97421203ae5503433e2"},
    {"copy_table", "4e0e3825d2c60ef42263ba2606845ac58f5efeb55f0634b4648c7a28025f1aa7"},
    {"key_file", "e6adf32c290bd9ab6a4eaa7cd46fb7ca5bcf2f69aa5108bd61223f679eee7a2a"},
    {"entry", "bdd41ca962dba11d0d718b93adaed5b4b4c3bfb7cb9b7e0ac32aa2833953aaa8"},
    {"entry_with_answers", "166c78667509b067c3e5be016ae37b53e66d526ffe4d64628156bdc54fe0564e"},
    {"batch_first", "d3e5a4408fda6fa67f005d3773c1bc000e8f14389eb70a30f89a184ad138677d"},
    {"batch_process", "9757de81186811a08387aace9eebeb47aff67bf13f0d84b2e68ffb877d5bf184"},
};
constexpr uint32_t KEYS_KNOWN_LIVE = 1000, KEYS_KNOWN_DRAW = 918; // live leaf tables; expected draw

// Returns true if all seven keys and the draw, made under the key `batch_first`, are as expected.
static inline bool keys_own_test() {
  auto pattern = [](unsigned a, unsigned b, size_t n) { // byte i is a i + b
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t)(a * i + b);
    return v;
  };
  const std::vector<uint8_t> seed_bytes = pattern(7, 1, 32), system_a = pattern(5, 3, 32), system_b = pattern(11, 9, 32), name = pattern(9, 1, 112), answers = pattern(17, 6, 24);
  const coin_keys::Shape shape = {33, ((uint64_t)1 << 32) + 5, 44040191, 11, 0, 40};
  const std::vector<coin_keys::Slot> slots = {{true, {1, 2, 3, 0xFFFFFFFFu}}, {false, {}}, {true, {7, 9}}};
  blk::Digest c, id;
  blk::digest_from_bytes(pattern(3, 2, 32).data(), c), blk::digest_from_bytes(pattern(13, 4, 32).data(), id);
  const blk::CoinKey seed = blk::coin_key_from_bytes(seed_bytes.data());
  const blk::CoinKey first = coin_keys::batch_first_key(seed, system_b.data(), c, 5, 2, slots);
  const blk::CoinKey got[7] = {coin_keys::first_table_key(seed, c, shape),
                               coin_keys::copy_table_key(seed, 3, system_a.data(), c, shape),
                               coin_keys::key_file_key(seed, system_b.data(), id),
                               coin_keys::entry_key(seed, 3, system_a.data(), id, name, {}),
                               coin_keys::entry_key(seed, 3, system_a.data(), id, name, answers),
                               first,
                               coin_keys::batch_process_key(first, name, {})};
  uint32_t differs = 0;
  auto digit = [](char x) { return (uint32_t)(x <= '9' ? x - '0' : x - 'a' + 10); };
  for (int k = 0; k < 7; k++) {
    uint8_t bytes[32];
    coin_keys::key_bytes(bytes, got[k]);
    for (int i = 0; i < 32; i++) differs |= bytes[i] ^ (digit(KEYS_KNOWN[k].hex[2 * i]) << 4 | digit(KEYS_KNOWN[k].hex[2 * i + 1]));
  }
  uint8_t first_bytes[32];
  coin_keys::key_bytes(first_bytes, first);
  differs |= draw_leaf(first_bytes, KEYS_KNOWN_LIVE) ^ KEYS_KNOWN_DRAW;
  return differs == 0;
}
} // namespace cp
