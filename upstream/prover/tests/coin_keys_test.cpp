// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests the keys of the committer's and prover's private randomness (`coin` in identifiers).
// prover/common/coin_keys.h derives them from the seed, system bytes and input lists. A key that
// reads too few inputs still yields random values, and no proof shows the fault.
//  1. Known answers from a third implementation (prover/tests/coin_keys_values.py: hashlib, a
//     ChaCha20 as specified in RFC 8439, sections 2.1 to 2.3) for the programs' functions, the
//     second implementation (prover/second_writing/schedule_keys.h) and the table of the programs'
//     self-test (cp::keys_own_test(), prover/common/cp_keys_own.h), which must pass.
//  2. twin(): when inputs change, exactly the keys that read them must change, and the two C++
//     implementations must agree.
//  3. Keys for different uses differ on the same inputs. Two tables' input lists (`ck::lists`) are
//     compared by digest: through blk::seed_child() the keys would differ even with equal lists.
//  4. There is no negative control:
//     no run of this test shows that a check fails where an item
//     is left out.
// All three implementations were written from the same description of the key rule, so agreement
// shows that the routines follow that description, not that the description is right, and
// agreement with the C++ implementation would not show a fault in the code they share.
// All inputs are synthetic. Build and run from prover/ (one command):
//   g++ -std=c++17 -O2 -Wall -Wextra -fno-ipa-modref -DBLK_FIELD=2 -DBLK_HASH=1 -I . -I ../blocks
//       tests/coin_keys_test.cpp -o <dir>/coin_keys_test && <dir>/coin_keys_test
#include "common/coin_keys.h"
#include "common/cp_keys_own.h"
#include "second_writing/schedule_keys.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace ck = coin_keys;
namespace sw = coins::schedule;
static int good = 0, other = 0;
static void expect(bool ok, const std::string &what) {
  if (ok)
    good++;
  else
    other++, printf("OTHERWISE: %s\n", what.c_str());
}
static std::vector<uint8_t> pattern(unsigned a, unsigned b, size_t n = 32) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = (uint8_t)((a * i + b) % 256);
  return v;
}
static std::string hex(const uint32_t *w) {
  std::string s;
  for (int i = 0; i < 32; i++) {
    const unsigned x = (w[i / 4] >> 8 * (i % 4)) & 255;
    s += "0123456789abcdef"[x >> 4], s += "0123456789abcdef"[x & 15];
  }
  return s;
}

// In names and messages, `first_table` is the anchor (a commitment's first weight table), `copy` a
// leaf table (a further one), `entry` a leaf file, `writing` an implementation and `published ID`
// the commitment ID (`id`).
// `c`: configuration digest. `k`: a leaf table's number. `system_a`, `system_b`: system bytes (in
// the programs, 32 from the operating system). `answers`: a table's record, if any.
struct Inputs {
  std::vector<uint8_t> seed = pattern(7, 1), system_a = pattern(5, 3), system_b = pattern(11, 9), c = pattern(3, 2), id = pattern(13, 4);
  std::vector<uint64_t> shape = {33, ((uint64_t)1 << 32) + 5, 44040191, 11, 0, 40};
  std::vector<uint8_t> name = pattern(9, 1, 112), answers;
  std::vector<std::pair<bool, std::vector<uint32_t>>> slots = {{true, {1, 2, 3, 0xFFFFFFFFu}}, {false, {}}, {true, {7, 9}}};
  uint32_t copies = 5, first_live = 2, k = 3;
};
// The six keys from the programs' functions.
static std::map<std::string, std::string> first_writing(const Inputs &in) {
  const blk::CoinKey seed = blk::coin_key_from_bytes(in.seed.data());
  blk::Digest c, id;
  blk::digest_from_bytes(in.c.data(), c), blk::digest_from_bytes(in.id.data(), id);
  std::vector<ck::Slot> slots;
  for (const auto &s : in.slots) slots.push_back(ck::Slot{s.first, s.second});
  std::map<std::string, std::string> out;
  out["first_table"] = hex(ck::first_table_key(seed, c, in.shape).w);
  out["copy_table"] = hex(ck::copy_table_key(seed, in.k, in.system_a.data(), c, in.shape).w);
  out["key_file"] = hex(ck::key_file_key(seed, in.system_b.data(), id).w);
  out["entry"] = hex(ck::entry_key(seed, in.k, in.system_a.data(), id, in.name, in.answers).w);
  const blk::CoinKey first = ck::batch_first_key(seed, in.system_b.data(), c, in.copies, in.first_live, slots);
  out["batch_first"] = hex(first.w);
  out["batch_process"] = hex(ck::batch_process_key(first, in.name, in.answers).w);
  return out;
}
static coins::Key as_key(const std::vector<uint8_t> &b) {
  coins::Key k;
  for (int i = 0; i < 8; i++) k[i] = b[4 * i] | (uint32_t)b[4 * i + 1] << 8 | (uint32_t)b[4 * i + 2] << 16 | (uint32_t)b[4 * i + 3] << 24;
  return k;
}
static std::map<std::string, std::string> second_writing(const Inputs &in) {
  const coins::Key seed = as_key(in.seed);
  std::vector<sw::SlotWords> slots;
  for (const auto &s : in.slots) slots.push_back(sw::SlotWords{s.first, s.second});
  std::map<std::string, std::string> out;
  out["first_table"] = hex(sw::first_table(seed, in.c, in.shape).data());
  out["copy_table"] = hex(sw::copy_table(seed, in.k, as_key(in.system_a), in.c, in.shape).data());
  out["key_file"] = hex(sw::key_file(seed, as_key(in.system_b), in.id).data());
  out["entry"] = hex(sw::entry(seed, in.k, as_key(in.system_a), in.id, in.name, in.answers).data());
  const coins::Key first = sw::batch_first(seed, as_key(in.system_b), in.c, in.copies, in.first_live, slots);
  out["batch_first"] = hex(first.data());
  out["batch_process"] = hex(sw::batch_process(first, in.name, in.answers).data());
  return out;
}

// `moves`: the keys that must change; every other key must stay. `from`: applied first to both
// `Inputs`.
static void twin(const std::string &what, const std::function<void(Inputs &)> &change, const std::set<std::string> &moves, const std::function<void(Inputs &)> &from = [](Inputs &) {}) {
  Inputs base, changed;
  from(base), from(changed);
  change(changed);
  const auto before = first_writing(base), after = first_writing(changed), again = second_writing(changed);
  expect(after == again, what + ": the two writings agree on the changed input");
  for (const auto &k : before) {
    const bool moved = after.at(k.first) != k.second;
    expect(moved == (moves.count(k.first) != 0), what + ": " + k.first + (moves.count(k.first) ? " must change" : " must stay"));
  }
}
int main() {
  // The random value generator's self-test: ends the process on failure.
  blk::coins_own_test();
  // 1. Known answers (the script's stored output). `draw`: draw_leaf()'s choice of a leaf table
  // (prover/common/cp_files.h).
  std::map<std::string, std::string> known;
  std::ifstream file("tests/coin_keys_values.txt");
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    std::string name, value;
    s >> name >> value;
    known[name] = value;
  }
  expect(known.size() == 8, "tests/coin_keys_values.txt has seven keys and the draw (run the test from prover/)");
  for (const cp::KeysKnown &k : cp::KEYS_KNOWN) expect(known.count(k.name) && known[k.name] == k.hex, std::string(k.name) + ": the own test's table has the Python text's value");
  expect(known["draw"] == std::to_string(cp::KEYS_KNOWN_DRAW), "the own test's table has the Python text's draw");
  expect(cp::keys_own_test(), "the programs' own test of the keys at a start passes");
  const Inputs base;
  const auto one = first_writing(base), two = second_writing(base);
  for (const auto &k : one) {
    expect(known.count(k.first) && known[k.first] == k.second, k.first + ": the programs' routine meets the Python text's value");
    expect(two.at(k.first) == k.second, k.first + ": the second writing gives the same");
  }
  {
    Inputs with;
    with.answers = pattern(17, 6, 24);
    expect(first_writing(with).at("entry") == known["entry_with_answers"] && second_writing(with).at("entry") == known["entry_with_answers"], "an entry's key with answers");
  }
  // 2. Changes to every field of `Inputs`.
  const std::set<std::string> ALL = {"first_table", "copy_table", "key_file", "entry", "batch_first", "batch_process"};
  twin("the seed", [](Inputs &i) { i.seed[31] ^= 1; }, ALL);
  twin("the configuration", [](Inputs &i) { i.c[0] ^= 1; }, {"first_table", "copy_table", "batch_first", "batch_process"});
  for (size_t word = 0; word < 6; word++)
    twin("word " + std::to_string(word) + " of the shape", [word](Inputs &i) { i.shape[word] ^= (uint64_t)1 << 40; }, {"first_table", "copy_table"});
  twin("a word more in the shape", [](Inputs &i) { i.shape.push_back(0); }, {"first_table", "copy_table"});
  twin("the copy's number", [](Inputs &i) { i.k = 4; }, {"copy_table", "entry"});
  twin("the system's bytes of a copy's and an entry's making", [](Inputs &i) { i.system_a[5] ^= 1; }, {"copy_table", "entry"});
  twin("the system's bytes of the key file's making and of a batch", [](Inputs &i) { i.system_b[5] ^= 1; }, {"key_file", "batch_first", "batch_process"});
  twin("the published ID", [](Inputs &i) { i.id[9] ^= 1; }, {"key_file", "entry"});
  twin("the copy's name", [](Inputs &i) { i.name[111] ^= 1; }, {"entry", "batch_process"});
  twin("the copy's answers", [](Inputs &i) { i.answers = {0}; }, {"entry", "batch_process"});
  twin("a byte from the name's end to the answers' start", [](Inputs &i) { i.answers = {i.name.back()}, i.name.pop_back(); }, {"entry", "batch_process"});
  twin("the number of copies", [](Inputs &i) { i.copies = 6; }, {"batch_first", "batch_process"});
  twin("the first live copy", [](Inputs &i) { i.first_live = 1; }, {"batch_first", "batch_process"});
  // The item `the prover's revision` is no input: each implementation holds the number itself.
  expect(ck::PROVER_REVISION == sw::kProverRevision, "the two C++ implementations hold one number for the item `the prover's revision`");
  twin("a slot more", [](Inputs &i) { i.slots.push_back({false, {}}); }, {"batch_first", "batch_process"});
  twin("a word of a request", [](Inputs &i) { i.slots[0].second[2] ^= 1; }, {"batch_first", "batch_process"});
  // A request of 1,026 words (1,024 tokens, then its range r, e): first, last, inner words matter.
  for (size_t word : {0, 7, 8, 9, 512, 1024, 1025})
    twin("word " + std::to_string(word) + " of a request of 1,026 words", [word](Inputs &i) { i.slots[2].second[word] ^= 1; }, {"batch_first", "batch_process"}, [](Inputs &i) {
      i.slots[2].second.clear();
      for (uint32_t t = 0; t < 1026; t++) i.slots[2].second.push_back(2654435761u * (t + 1) >> 7);
    });
  twin("a request moved to another slot", [](Inputs &i) { std::swap(i.slots[1], i.slots[2]); }, {"batch_first", "batch_process"});
  twin("a request of no words against no request", [](Inputs &i) { i.slots[1] = {true, {}}; }, {"batch_first", "batch_process"});
  twin("a word from one request's end to the next one's start", [](Inputs &i) { i.slots = {{true, {1, 2, 3}}, {false, {}}, {true, {0xFFFFFFFFu, 7, 9}}}; }, {"batch_first", "batch_process"});
  // 3. Keys for different uses differ.
  {
    std::set<std::string> all;
    for (const auto &k : one) all.insert(k.second);
    expect(all.size() == one.size(), "the six keys differ from each other");
    // Leaf table 0 with the anchor's `c`, `shape` and system bytes (zeros): only the use differs.
    const blk::CoinKey seed = blk::coin_key_from_bytes(base.seed.data());
    blk::Digest c;
    blk::digest_from_bytes(base.c.data(), c);
    const uint8_t zeros[32] = {};
    expect(hex(ck::first_table_key(seed, c, base.shape).w) != hex(ck::copy_table_key(seed, 0, zeros, c, base.shape).w), "the first table is no copy 0");
    expect(hex(ck::lists::copy_table(c, base.shape, 3).w) != hex(ck::lists::copy_table(c, base.shape, 4).w), "the list of copy 3's making is not that of copy 4's");
    expect(hex(ck::lists::first_table(c, base.shape).w) != hex(ck::lists::copy_table(c, base.shape, 0).w), "the list of the first table's making is not that of copy 0's");
  }
  printf("coin_keys_test: %d checks as expected, %d otherwise\n", good, other);
  return other != 0;
}
