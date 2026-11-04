// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives the keys of cp_commit's and cp_prove's private randomness (`coins`) from the provider's
// seed: a second implementation, by which both programs and tests check prover/common/coin_keys.h.
//
// This implementation follows the table in the comment at the top of that file and includes no file
// of prover/common/ or blocks/. Comparing the two implementations checks the schedule: which inputs
// enter each key, in what encoding and order.
// An error in that table that both follow, or a misreading of it that both share, goes unseen. So
// does a fault that both share in the ChaCha20 block function or in BLAKE2b: for each function,
// both follow the same published source. prover/tests/coins_rfc8439_test.cpp checks the
// byte-identical copy of chacha_block() in prover/tests/coins_second_text/ against RFC 8439's test
// vectors. blake2b_256() is tested only through keys: prover/tests/coin_keys_test.cpp requires them
// to equal values computed with Python's hashlib.
//
// Every key is words 0 to 7 of a block under the previous key. No key, intermediate or returned, is
// wiped here, and the intermediate ones stay on the stack: prover/common/stack_scrub.h states what
// a caller must do.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "blake2b_plain.h"
#include "chacha_block.h"

namespace coins {
namespace schedule {

using Octets = std::vector<std::uint8_t>;

// --- A list of items -----------------------------------------------------------

// A list of facts (named items) goes into a key as the eight words of its digest. The digest is
// BLAKE2b-256 of: the numbers 176 and 0, the format's label for this hash; then item by item the
// length of the item's name, the name, the length of its bytes, the bytes. Lengths and numbers are
// eight bytes, lowest byte first.
class Items {
 public:
  Items() {
    eight(176);
    eight(0);
  }
  ~Items() { wipe_octets(all_.data(), all_.size()); }
  Items(const Items&) = delete;
  Items& operator=(const Items&) = delete;

  void item(const std::string& name, const Octets& bytes) {
    eight(name.size());
    all_.insert(all_.end(), name.begin(), name.end());
    eight(bytes.size());
    all_.insert(all_.end(), bytes.begin(), bytes.end());
  }
  void name_alone(const std::string& name) { item(name, Octets()); }
  void number(const std::string& name, std::uint64_t value) {
    Octets b;
    for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(value >> (8 * i)));
    item(name, b);
  }
  // Four bytes a word, lowest first.
  std::array<Word, 8> words() const {
    const std::array<std::uint8_t, 32> d = blake2b_256(all_);
    std::array<Word, 8> w;
    for (int i = 0; i < 8; ++i)
      w[i] = Word(d[4 * i]) | Word(d[4 * i + 1]) << 8 | Word(d[4 * i + 2]) << 16 | Word(d[4 * i + 3]) << 24;
    return w;
  }

 private:
  void eight(std::uint64_t x) {
    for (int i = 0; i < 8; ++i) all_.push_back(std::uint8_t(x >> (8 * i)));
  }
  Octets all_;
};

// --- The facts -------------------------------------------------------------------

inline void put_configuration(Items& l, const Octets& c) { l.item("the configuration", c); }
inline void put_shape(Items& l, const std::vector<std::uint64_t>& shape) {
  Octets b;
  for (std::uint64_t word : shape)
    for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(word >> (8 * i)));
  l.item("the shape", b);
}
inline void put_table(Items& l, const Octets& name, const Octets& answers) {
  l.item("the copy's name", name);
  l.item("the copy's answers", answers);
}
// Whoever makes a change after which cp_prove, on equal inputs, sends another byte or draws its
// random values in another order or number (a change of what a proof's chains hash is one), while
// the format's version word stays, raises this number by one in the same change, wherever it is
// written. Nothing in the programs or the tests checks that. (docs/format/COIN-KEYS.md, "The
// prover's revision".)
constexpr std::uint64_t kProverRevision = 1;

// --- The common form ---------------------------------------------------------------

constexpr Word kSeedWord = 0x44535043;   // "CPSD"
constexpr Word kBatchWord = 0x4B425043;  // "CPBK"

inline Key take_eight(Key key, const std::array<Word, 8>& w) {
  key = chacha_key(key, w[0], w[1], w[2], w[3]);
  return chacha_key(key, w[4], w[5], w[6], w[7]);
}
// All keys but batch_process()'s: seed, kind, number; then the 32 bytes; then the list's digest.
inline Key form(const Key& seed, Word kind, Word number, const Key& thirty_two_bytes, const Items& list) {
  Key k = chacha_key(seed, kSeedWord, kind, number, 0);
  k = take_eight(k, thirty_two_bytes);
  return take_eight(k, list.words());
}

// --- The six derivations -------------------------------------------------------------

inline Key first_table(const Key& seed, const Octets& c, const std::vector<std::uint64_t>& shape) {
  Items l;
  l.name_alone("a stored table's making");
  put_configuration(l, c);
  put_shape(l, shape);
  l.name_alone("the first table");
  return form(seed, 1, 0, Key{}, l);
}
// `kept`, `of_the_system`: form()'s 32 bytes, from getrandom() outside test runs. `kept` is stored
// with leaf table (`copy`) k's secret files.
inline Key copy_table(const Key& seed, Word k, const Key& kept, const Octets& c,
                      const std::vector<std::uint64_t>& shape) {
  Items l;
  l.name_alone("a stored table's making");
  put_configuration(l, c);
  put_shape(l, shape);
  l.number("copy", k);
  return form(seed, 7, k, kept, l);
}
inline Key key_file(const Key& seed, const Key& of_the_system, const Octets& id) {
  Items l;
  l.name_alone("the key file's making");
  l.item("the published ID", id);
  return form(seed, 8, 0, of_the_system, l);
}
inline Key entry(const Key& seed, Word k, const Key& of_the_system, const Octets& id, const Octets& name,
                 const Octets& answers) {
  Items l;
  l.name_alone("an entry's making");
  l.item("the published ID", id);
  put_table(l, name, answers);
  return form(seed, 5, k, of_the_system, l);
}
struct SlotWords {
  bool is_request = false;
  std::vector<Word> words;
};
inline Key batch_first(const Key& seed, const Key& of_the_system, const Octets& c, Word copies, Word first_live,
                       const std::vector<SlotWords>& slots) {
  Items l;
  l.name_alone("a batch");
  put_configuration(l, c);
  l.number("the prover's revision", kProverRevision);
  l.number("copies", copies);
  l.number("the first live copy", first_live);
  l.number("slots", slots.size());
  for (const SlotWords& s : slots) {
    if (!s.is_request) {
      l.name_alone("no request");
      continue;
    }
    Octets b;
    for (Word w : s.words)
      for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(w >> (8 * i)));
    l.item("request", b);
    wipe_octets(b.data(), b.size());
  }
  return form(seed, 6, 0, of_the_system, l);
}
inline Key batch_process(const Key& first_key, const Octets& name, const Octets& answers) {
  Items l;
  l.name_alone("the batch's stored copy");
  put_table(l, name, answers);
  return take_eight(chacha_key(first_key, kBatchWord, 0, 0, 0), l.words());
}

}  // namespace schedule
}  // namespace coins
