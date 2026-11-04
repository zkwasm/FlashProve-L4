// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives, from the provider's secret seed, the keys of the private randomness (`coin` in names) of
// the committer program cp_commit and the prover program cp_prove: random tails (the random
// coefficients encoded with each row), mask rows, pads, hiding polynomials, salts. No verifier
// derives these keys, and the file format does not depend on how they are derived. Hiding rests on
// them, and no verifier, honest run or ordinary test can see a fault in them. What hiding rests on,
// and a known limit, are stated at first_table_key(). The full rule is in
// docs/format/COIN-KEYS.md.
//
//     K = take(take(child(kind, number), the 8 words of 32 bytes), the 8 words of I)
//     I = H(176, 0, 0; item, item, ..)
// child() is blk::seed_child() and take() is blk::coin_key_take() (blocks/coins/coins.h): each
// derives a ChaCha20 key from the key before it. I is the digest of a list of named items
// (blk::CoinInputs in blocks/coins/run_coins.h, which defines how an item is hashed). H is the
// format's hash; its arguments are domain number 176 (blk::Domain::COIN_INPUTS), tree 0, index 0.
// The first item is the key's title.
// The anchor (`first_table` in names) and its leaf tables (`copy`) each contain all the weights.
// The key file and leaf file k (`entry`) contain cp_commit's proofs about them. A commit run
// (`making`) is one creation of such a table or file. System bytes (`of_the_system`) are 32 bytes
// that the caller must draw for one commit run or batch from the system's random generator. Under a
// test seed, and under the option -public-weights (prover/common/public_weights.h), nothing is
// hidden, and cp_commit passes zeros for them. The table uses the words in backquotes.
//
//   key                        child    the 32 bytes                              the items behind the title
//   the first table's making   (1, 0)   zeros, a constant of the rule: the seed   the configuration; the shape; «the first table»
//                                       is drawn in this making, and in no other
//   copy k's making            (7, k)   of the system, drawn for this making,     the configuration; the shape; «copy» k
//                                       kept with the copy's secret files
//   the key file's making      (8, 0)   of the system, kept nowhere (*)           the published ID
//   entry k's making           (5, k)   of the system, kept nowhere (*)           the published ID; copy k's name; its answers
//                                       (*) but under cp_commit's -keep-for-the-audit, until the audit's end
//   a batch's first key        (6, 0)   of the system, kept nowhere (**)          the configuration; the prover's revision; the copies' number; the first live copy; the slots' number; slot by
//                                                                                 slot «request» with its words, or «no request»
//                                       (**) but under cp_prove's -keep-for-the-audit, for one batch, until the audit's verdict
//   a batch's process key = take( key(the first key; "CPBK", 0, 0, 0), the 8 words of I )    the drawn copy's name; its answers
//   On (*): the bytes are kept in files until the audit (tools/audit/audit.py) ends in YES.
//   On (**): they are kept in a file until that batch's audit reaches a verdict and erases them
//   (with --rehearsal it erases nothing; when else it does not: tools/audit/audit_batch.sh).
//   key() is blk::coin_key(). The items: see "the facts" below.
//
// A second implementation, prover/second_writing/schedule_keys.h, includes no file of this
// directory or of blocks/. It is not independent of this file: it follows the table above. Its
// BLAKE2b and its ChaCha20 quarter round are adapted from the same reference code as the programs',
// so they are not independent either. How far the rest is independent has not been checked
// (prover/second_writing/README.md, "How far it is independent of the programs").
// Known answers come from a third, in Python: prover/tests/coin_keys_values.py.
// prover/tests/coin_keys_test.cpp compares the three.
#pragma once
#include <string>
#include <vector>

#include "coins/run_coins.h" // ../blocks

// No proof and no verifier can see whether lists::first_table() and lists::copy_table()
// leave an item out. Test builds that are not part of this repository leave one out on
// purpose; no code in this repository reads their switches. blocks/fence.h's list of test macros sees only
// what is defined before the first include of blocks/base.h: a macro that a source defines later gets past
// it. A list of the same kind, of fault-injecting switches, is not in this repository, nor is the code
// under them: a switch that a source defined later got past that list (seen once, in a test harness).

namespace coin_keys {
using blk::CoinInputs;
using blk::CoinKey;
using blk::Digest;

// `kind` of blk::seed_child(): one value for each use of the seed.
enum Kind : uint32_t { FIRST_TABLE = 1, ENTRY = 5, BATCH = 6, COPY_TABLE = 7, KEY_FILE = 8 }; // 2, 3, 4 are not used
constexpr uint32_t BATCH_WORD = 0x4B425043; // "CPBK"
// The item `the prover's revision` of a batch's first key (docs/format/COIN-KEYS.md, "The prover's
// revision"). Whoever makes a change after which cp_prove, on equal inputs (files, seed, system
// bytes, batch, options), sends another byte or draws its random values in another order or number
// (a change of what a proof's chains hash is one), while the format's version word stays, raises
// this number by one in the same change, wherever it is written. One number stands for one
// behaviour of cp_prove: several such changes may share one raise only if no build made between
// them has served a secret key directory that matters. Nothing in the programs or the tests checks
// that. Without the raise two versions of cp_prove on one secret key directory, given equal system
// bytes and equal batches, draw equal random values and send them under different challenges.
constexpr uint64_t PROVER_REVISION = 1;

// ---- the facts ----
// `c` is the configuration digest, a hash of the public configuration files
// (docs/format/FORMAT.md, R4.9). `id` is the commitment ID (R4.11), which the hashed labels
// call the published ID. R4.10 defines the name and shape of a weight table, which it calls a
// stored table. A table's answers are its record: stored answers to samples of its rows. `name` and
// `answers` hold a table's name and answers in docs/format/FORMAT.md's encoding. Pads are random
// values that mask the answers. The configurations: prover/examples/TARGET-KITS.md.
using Shape = std::vector<uint64_t>;  // S: the shape words of a table's name, in the name's order
using Bytes = std::vector<uint8_t>;   // a table's name, or its answers
struct Slot {                         // one place in a batch: a request or none
  bool is_request = false;
  std::vector<uint32_t> words;        // the request's words as cp_prove read them: client data
};

// ---- the lists: one function for each, which the programs and the tests call ----
namespace lists {
// An item that is its name alone; a number as 8 bytes, lowest first.
inline void title(CoinInputs &in, const std::string &name) { in.add(name, static_cast<const void *>(""), 0); }
inline void number(CoinInputs &in, const std::string &name, uint64_t value) {
  uint8_t b[8];
  blk::word_to_bytes(value, b);
  in.add(name, static_cast<const void *>(b), sizeof b);
}
inline void configuration(CoinInputs &in, const Digest &c) { in.add("the configuration", c); }
inline void shape(CoinInputs &in, const Shape &S) {
  Bytes b(8 * S.size());
  for (size_t i = 0; i < S.size(); i++) blk::word_to_bytes(S[i], &b[8 * i]);
  in.add("the shape", static_cast<const void *>(b.data()), b.size());
}
inline void table(CoinInputs &in, const Bytes &name, const Bytes &answers) {
  in.add("the copy's name", static_cast<const void *>(name.data()), name.size());
  in.add("the copy's answers", static_cast<const void *>(answers.data()), answers.size());
}
// The items with which first_table() and copy_table() begin.
inline void making(CoinInputs &in, const Digest &c, const Shape &S) {
  title(in, "a stored table's making");
  configuration(in, c);
  shape(in, S);
  (void)c, (void)S;
}
inline Digest first_table(const Digest &c, const Shape &S) {
  CoinInputs in;
  making(in, c, S);
  title(in, "the first table");
  return in.finish();
}
inline Digest copy_table(const Digest &c, const Shape &S, uint32_t k) {
  CoinInputs in;
  making(in, c, S);
  number(in, "copy", k);
  (void)k;
  return in.finish();
}
inline Digest key_file(const Digest &id) {
  CoinInputs in;
  title(in, "the key file's making"), in.add("the published ID", id);
  return in.finish();
}
inline Digest entry(const Digest &id, const Bytes &name, const Bytes &answers) {
  CoinInputs in;
  title(in, "an entry's making"), in.add("the published ID", id), table(in, name, answers);
  return in.finish();
}
// Leaf tables are numbered 0 to `copies` - 1. `first_live`: the lowest number that batches can use.
inline Digest batch(const Digest &c, uint32_t copies, uint32_t first_live, const std::vector<Slot> &slots) {
  CoinInputs in;
  title(in, "a batch"), configuration(in, c), number(in, "the prover's revision", PROVER_REVISION), number(in, "copies", copies), number(in, "the first live copy", first_live);
  number(in, "slots", slots.size());
  for (const Slot &s : slots) {
    if (!s.is_request) {
      title(in, "no request");
      continue;
    }
    Bytes b(4 * s.words.size());
    for (size_t i = 0; i < s.words.size(); i++)
      for (int j = 0; j < 4; j++) b[4 * i + j] = (uint8_t)(s.words[i] >> 8 * j);
    in.add("request", static_cast<const void *>(b.data()), b.size());
    blk::wipe(b.data(), b.size());
  }
  return in.finish();
}
inline Digest batch_copy(const Bytes &name, const Bytes &answers) {
  CoinInputs in;
  title(in, "the batch's stored copy"), table(in, name, answers);
  return in.finish();
}
} // namespace lists

// ---- the keys. The caller wipes the secrets that it passes in and the key that it gets. ----
// Writes K's eight words as 32 bytes, lowest byte first; blk::coin_key_from_bytes() is the inverse.
inline void key_bytes(uint8_t out[32], const CoinKey &K) {
  for (int i = 0; i < 32; i++) out[i] = (uint8_t)(K.w[i / 4] >> 8 * (i % 4));
}
// K of the formula at the top of this file.
inline CoinKey from(const CoinKey &seed, Kind kind, uint32_t number, const uint8_t bytes[32], Digest I) {
  CoinKey child = blk::seed_child(seed, kind, number);
  const CoinKey K = blk::process_key(child, bytes, I);
  blk::wipe(child), blk::wipe(I.w, sizeof I.w);
  return K;
}
// A table's commit-run key is the result of first_table_key() or copy_table_key(). From it come the
// table's random tails, its tree leaves' salts and its pads.
// What hiding rests on: the same random tails, salts or pads must never be used under two Merkle
// roots. So each table's commit-run key depends on bytes that are drawn fresh for that run: the
// seed for the anchor, 32 system bytes for a leaf table. That the key also reads the
// configuration, the shape and which table it is guards against a broken or repeating system
// generator, and nothing more: with equal bytes, another configuration, shape or table still gets
// other random values.
// Known limit, the operator's duty: a machine image that is started twice and whose generator
// repeats. With the same weights it makes one table on two machines, each with a separate count of
// opened columns. With other weights under the same public files it makes two tables whose random
// tails are equal. No table's commit-run key reads anything of the weights. Of the weights the
// other keys read only tables' names, the commitment ID and answers, which follow from the weights
// and are the commitment's public values (docs/limits-all-rows.tsv, row H4.2).
// Sources for the danger of a machine image that is started twice:
// Ristenpart, Yilek, "When Good Randomness Goes Bad: Virtual Machine Reset Vulnerabilities and Hedging Deployed
// Cryptography", NDSS Symposium 2010, section 1;
// Garfinkel, Rosenblum, "When Virtual is Harder than Real: Security Challenges in Virtual Machine Based Computing
// Environments", HotOS-X, June 2005;
// Everspaugh et al., "Not-So-Random Numbers in Virtualized Linux and the Whirlwind RNG".
inline CoinKey first_table_key(const CoinKey &seed, const Digest &c, const Shape &S) {
  const uint8_t zeros[32] = {};
  return from(seed, FIRST_TABLE, 0, zeros, lists::first_table(c, S));
}
// `kept`: leaf table k's system bytes.
inline CoinKey copy_table_key(const CoinKey &seed, uint32_t k, const uint8_t kept[32], const Digest &c, const Shape &S) {
  return from(seed, COPY_TABLE, k, kept, lists::copy_table(c, S, k));
}
inline CoinKey key_file_key(const CoinKey &seed, const uint8_t of_the_system[32], const Digest &id) {
  return from(seed, KEY_FILE, 0, of_the_system, lists::key_file(id));
}
inline CoinKey entry_key(const CoinKey &seed, uint32_t k, const uint8_t of_the_system[32], const Digest &id, const Bytes &name, const Bytes &answers) {
  return from(seed, ENTRY, k, of_the_system, lists::entry(id, name, answers));
}
inline CoinKey batch_first_key(const CoinKey &seed, const uint8_t of_the_system[32], const Digest &c, uint32_t copies, uint32_t first_live,
                               const std::vector<Slot> &slots) {
  return from(seed, BATCH, 0, of_the_system, lists::batch(c, copies, first_live, slots));
}
// `name`, `answers`: of the batch's leaf table, which cp_prove draws under the first key.
inline CoinKey batch_process_key(const CoinKey &first_key, const Bytes &name, const Bytes &answers) {
  CoinKey Y = blk::coin_key(first_key, BATCH_WORD, 0, 0, 0);
  Digest I = lists::batch_copy(name, answers);
  blk::coin_key_take(Y, I.w, 8);
  blk::wipe(I.w, sizeof I.w);
  return Y;
}
} // namespace coin_keys
