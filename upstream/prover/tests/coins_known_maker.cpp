// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Prints a table of known answers (prover/hiding/coins_known_f2.h) for namespace hz's generator of
// private random values (`coins` in identifiers). No verifier can tell whether these values follow
// the generator's rule, so the programs cp_prove and cp_commit compare that generator with part of
// the table when they start: hz::coins_own_test() (prover/hiding/coins_own.cuh) [own-test-coins].
// The strings kHead to kWholeRowHead, printed into the table, say what is computed.
// prover/tests/coins_known_test.sh fails unless the stored table is this program's output.
// The generator's values depend on the field, p = 2^64 - 2^32 + 1 (`kP`,
// prover/tests/coins_second_text/coin_generator.h); the table's keys do not.
//
// The answers are computed with a second implementation of the generator, the header library in
// prover/tests/coins_second_text/, except for five conventions marked "Written here" below. This
// file includes no file of the programs and calls none of their functions, so a fault in the
// programs' generator cannot reach the table through code.
// Limits. The library is not independent of the programs: both follow one written rule, some points
// of which were settled to match what both did. Nor are the five conventions independent of the
// programs' code: an error in them that the programs also have is not detected. An error here or in
// the library that the programs lack makes a correct program fail the comparison. Further limits:
// prover/second_writing/README.md, "How far it is independent of the programs".
// From prover/: g++ -std=c++17 -O2 -fno-ipa-modref -o coins_known_maker tests/coins_known_maker.cpp
#include "coins_second_text/coin_generator.h"
#include "coins_second_text/coin_streams.h"
#include "coins_second_text/format_hash.h"
#include "coins_second_text/process_keys.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace coins;
using Cells = std::vector<std::uint64_t>;

namespace {

// Terms. A cell is one value of the generator, below the field's prime. A blinding context
// (`Context`: words 4, 5 and 10 to 12 of a cell's sixteen-word label) has a key, K_ctx, derived
// from a process key. Under K_ctx a stream is a numbered sequence of cells, four or five to a call
// of the generator's block function. The key file is `ivk/key.bin`.
constexpr Word kVersion = 530690;    // word 10 of the labels under a batch's key and a link's
constexpr Word kRangeWord = 530691;  // word 10 of the labels under the key of the key file's proofs
constexpr Word kLiveLeaves = 1000, kLink = 3, kLeafIndices = 64;
constexpr int kChains = 2, kContexts = 12, kLongContexts = 4, kFillGroups = 3, kLongGroups = 5;
constexpr Word kTrees[kLongContexts] = {320, 8, 8, 8};
// Per context: ns, how many slots a sub-proof (not a batch) has in fill_list()'s group 0, and how
// many cells group 1 takes from each stream.
constexpr std::uint64_t kSlots[kContexts] = {6603, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127};
constexpr std::uint64_t kStarts[kContexts] = {256, 256, 256, 256, 16384, 256, 256, 256, 256, 256, 256, 256};
constexpr std::uint64_t kRunFrom = 16384;  // kStarts' largest: groups 1 and 2 share no cell
constexpr int kWholeStreamContext = 6;
constexpr std::uint64_t kWholeStreamCells = 3112508;
constexpr std::uint64_t kWholeRowCells = std::uint64_t(1) << 22;

// The digest of a list of cells: the first 32 bytes of BLAKE2b-512 of its bytes, eight per cell,
// little-endian.
// Written here: the library has the hash's compression function and first state, but not this loop.
// Its hash_value() takes these steps with the list's length l and a number num in the high half of
// compress()'s counter. With that half forced to 0, the same steps are BLAKE2b-512 of the list's
// bytes: compress() is then F of RFC 7693 with the counter t = the bytes fed so far.
// coins_known_test.sh compares four of the table's digests with Python's BLAKE2b.
std::vector<Word> digest(const Cells& cells) {
  const std::size_t words = 2 * cells.size(), blocks = words == 0 ? 1 : (words + 31) / 32;
  HashValue cv = hash_start();
  for (std::size_t b = 0; b < blocks; ++b) {
    Word block[32] = {};
    for (std::size_t i = 0; i < 32 && 32 * b + i < words; ++i) {
      const std::uint64_t cell = cells[(32 * b + i) / 2];
      block[i] = i % 2 ? Word(cell >> 32) : Word(cell);
    }
    const bool last = b + 1 == blocks;
    cv = compress(cv, block, last ? 4 * std::uint64_t(words) : 128 * std::uint64_t(b + 1), 0, 0, last);
  }
  return std::vector<Word>(cv.begin(), cv.begin() + 8);
}

// The keys derived from one test seed; kHead describes each.
// Limit: nothing in this tree checks the steps from the seed to these keys. Of this struct, the
// table's readers (prover/hiding/coins_own.cuh, prover/tests/coins_known_host_test.cpp) recompute
// only `leaf` and `key_word`, from `kbat` and `kproc` as given. The programs derive their keys by
// another rule (prover/common/coin_keys.h).
struct Chain {
  int n;
  Key seed, rho;
  std::vector<Word> words;  // 20
  Key kbat, kproc, once, link, leaf_masks, anchor_masks;
  Word leaf, key_word;
};

Chain chain(int s) {
  Chain c;
  c.n = s == 0 ? 8 : 10;
  c.seed = test_seed(unsigned(s) + 2);
  for (Word j = 0; j < 8; ++j) c.rho[j] = Word(mix64(1000 + 16 * std::uint64_t(s) + j) >> 16);
  for (Word j = 0; j < 20; ++j) c.words.push_back(Word(mix64(2000 + 32 * std::uint64_t(s) + j) >> 16));
  const std::vector<Word> first(c.words.begin(), c.words.begin() + c.n);
  const std::vector<Word> second(c.words.begin() + c.n, c.words.begin() + 2 * c.n);
  c.kbat = batch_base_key(c.seed, c.rho, first);
  c.leaf = draw_leaf(c.kbat, kLiveLeaves, 0);
  c.kproc = batch_process_key(c.kbat, second);
  c.key_word = key_check_word(c.kproc);
  c.once = one_time_process_key(c.seed, first, kVersion);
  c.link = link_process_key(c.seed, kLink, first, second, c.rho);
  c.leaf_masks = leaf_mask_seed(c.seed, kLink, c.rho);
  c.anchor_masks = anchor_mask_seed(c.seed);
  return c;
}

// The contexts.
struct Named {
  int chain;
  int key;  // 0: Kproc, 1: once, 2: link
  Context context;
  const char* what;
};

const Named kNamed[kContexts] = {
    {0, 0, request_proof_context(kVersion, 0), "a request's proof, slot 0"},
    {0, 0, request_proof_context(kVersion, 23), "a request's proof, slot 23"},
    {1, 0, request_proof_context(kVersion, 1), "a request's proof, slot 1"},
    {1, 0, Context{(std::uint64_t(5) << 32) + 7, kVersion, 3, 65537},
     "made up: neither a program nor the second implementation has this context"},
    {0, 0, shared_message_context(kVersion, kLeafIndices), "the shared opening message of a batch with 64 leaf indices"},
    {0, 0, first_message_context(kVersion, 23), "the first opening message of slot 23"},
    {0, 0, second_message_context(kVersion, kLeafIndices, 23), "the second opening message of slot 23, 64 leaf indices"},
    {0, 1, range_file_context(kRangeWord), "the range file"},
    {0, 1, once_first_message_context(kRangeWord), "the one-time files' first message"},
    {0, 1, once_shared_message_context(kRangeWord), "the one-time files' shared message"},
    {0, 1, once_second_message_context(kRangeWord), "the one-time files' second message"},
    {0, 2, link_context(kVersion), "a link"},
};

// Written here, in the second column of kNamed: which process key a context uses. The library's
// four contexts for the key file's proofs (7 to 10) use `once`, a link's uses `link`, all others
// `kproc`, a batch's.
const Key& process_key(const Chain* chains, const Named& x) {
  const Chain& c = chains[x.chain];
  return x.key == 0 ? c.kproc : x.key == 1 ? c.once : c.link;
}

// The lists.
void append(Cells& list, const Coins& coins, Word stream, std::uint64_t first, std::uint64_t count) {
  const std::size_t at = list.size();
  list.resize(at + count);
  coins.cells(stream, first, count, list.data() + at);
}

// A fill group's list under context x (see kFillHead): streams numbered from a base, four cells a
// call.
Cells fill_list(const Coins& coins, int x, int group) {
  Cells list;
  if (group == 0) {
    for (Word gid = 0; gid < 60; ++gid)
      for (unsigned kappa = 0; kappa < 3; ++kappa)
        append(list, coins, slot_pad_stream(gid), slot_pad_cell(kSlots[x], kappa, 0), kSlots[x]);
  } else if (group == 1) {
    for (Word stream : {coordinates_stream(kBaseCP), rows_stream(kBaseCP), salts_stream(kBaseCP)})
      append(list, coins, stream, 0, kStarts[x]);
    for (Word i = 0; i < 96; ++i) append(list, coins, kBaseOpening + i, 0, kStarts[x]);
  } else {
    append(list, coins, rows_stream(kBaseCP), kRunFrom, 16384);
    append(list, coins, salts_stream(kBaseZU), (std::uint64_t(1) << 35) + 3, 8190);
  }
  return list;
}
// A whole stream of context 6 but for its start, which fill group 1 has. Under that context's
// object, 8, kBaseZV is the base of the tree ZF_i, not of ZV.
Cells whole_stream_list(const Coins& coins) {
  Cells list;
  append(list, coins, rows_stream(kBaseZV), kStarts[kWholeStreamContext], kWholeStreamCells);
  return list;
}

// I(v) of kLongHead.
std::uint64_t position(unsigned v) { return v < 56 ? v : (std::uint64_t(1) << (20 + 3 * (v - 56))) + (v - 56); }

// A long group's list (see kLongHead): streams formed from a commitment's serial number, a kind and
// a row. Only tests read these lists' digests and the whole row's.
Cells long_list(const Coins& coins, int group, Word trees) {
  Cells list;
  if (group == 0) {
    for (Word num = 0; num < trees; ++num)
      for (Word row : {0u, 9u, 300u}) append(list, coins, serial_stream(num, kKindBlindingCoordinates, row), 0, 1024);
  } else if (group == 1) {
    for (Word num = 0; num < trees; ++num) append(list, coins, serial_stream(num, kKindBlindingRow, 17), 0, 1024);
  } else if (group == 2) {
    append(list, coins, serial_stream(kNumPM, kKindBlindingRow, 0), 0, 16383);
    append(list, coins, serial_stream(num_of_segment_commitment(1, 1), kKindBlindingRow, 40), 0, 16383);
    append(list, coins, serial_stream(kNumP, kKindDerivedRow, 0), 0, 16383);
    append(list, coins, serial_stream(kNumP, kKindDerivedRow, 3), 0, 16383);
  } else if (group == 3) {
    // Written here: salt element x of position I is cell 5 I + x of the stream of kind 2, row 0.
    // The library's BasicRequestCoins::leaf_salt() has the same product but serves only a request's
    // context.
    for (Word num = 0; num < trees; ++num)
      for (unsigned v = 0; v < 64; ++v)
        append(list, coins, serial_stream(num, kKindSalts, 0), kSaltCells * position(v), kSaltCells);
  } else {
    // Written here: the library has no rule for a tree leaf. With R rows and 2^ln columns:
    //   leaf_I = dig C(S_sl, block, sl, 0, 104 + 2^16 NUM),   sl = ceil(R / 16),
    //   block = salt_I as 10 words, R, ln, NUM, I mod 2^32, floor(I / 2^32), then 17 zero words,
    // where C is compress() with `last` false (the 0 is l), and dig keeps the first eight
    // words. S_sl is the hash state after the column's symbols. Here it is sixteen zero words, not
    // a column's state: the case tests only the salt.
    constexpr Word kRows = 32, kLn = 6, kLeafNumber = 104;
    for (Word num = 0; num < trees; ++num)
      for (Word column = 0; column < (Word(1) << kLn); ++column) {
        std::uint64_t salt[kSaltCells];
        coins.cells(serial_stream(num, kKindSalts, 0), kSaltCells * std::uint64_t(column), kSaltCells, salt);
        Word block[32] = {};
        for (unsigned x = 0; x < kSaltCells; ++x) {
          block[2 * x] = Word(salt[x]);
          block[2 * x + 1] = Word(salt[x] >> 32);
        }
        block[10] = kRows, block[11] = kLn, block[12] = num, block[13] = column, block[14] = 0;
        const HashValue leaf = compress(HashValue{}, block, (kRows + 15) / 16, 0, kLeafNumber + (num << 16), false);
        for (int i = 0; i < 4; ++i) list.push_back(leaf[2 * i] | std::uint64_t(leaf[2 * i + 1]) << 32);
      }
  }
  return list;
}

// Printing.
// Written here: a key's 32 bytes are its eight words, each little-endian.
void print_bytes(const char* name, const Key& key) {
  std::printf("     /* %-12s */ {", name);
  for (int i = 0; i < 32; ++i) std::printf("%s0x%02x", i == 0 ? "" : i == 16 ? ",\n                          " : ", ", unsigned(key[i / 4] >> (8 * (i % 4)) & 255));
  std::printf("},\n");
}
void print_words(const Word* w, int count, const char* indent) {
  for (int i = 0; i < count; ++i)
    std::printf("%s0x%08xu", i == 0 ? "" : i % 8 == 0 ? (std::string(",\n") + indent).c_str() : ", ", unsigned(w[i]));
}

// Parts of kHead. The flag has no effect: the library only rejects other values of BLK_FIELD.
#define KNOWN_FLAG " -DBLK_FIELD=2"
#define KNOWN_FILE "hiding/coins_known_f2.h"
#define KNOWN_PRIME "2^64 - 2^32 + 1"
const char* const kHead = R"(// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Known answers of the generator of the prover's coins (common/coin_chacha.h), for the test
// that cp_prove and cp_commit make of that generator when they start [own-test-coins]
// (hiding/coins_own.cuh), and for tests.
//
// tests/coins_known_maker.cpp has printed this file, and it is changed in no other way. From prover/:
//   g++ -std=c++17 -O2 -fno-ipa-modref)" KNOWN_FLAG R"( -o coins_known_maker tests/coins_known_maker.cpp
//   ./coins_known_maker > )" KNOWN_FILE R"(
// tests/coins_known_test.sh makes the file again and compares. The maker computes the answers with a second
// implementation of the coins' rule, the header library in tests/coins_second_text/, but for the conventions that the
// maker's own comments name. That library and the maker include no file of the generator that is tested, so no routine
// of it has a part in a value of this file. The programs do compile copies of some of the library's headers, for
// comparisons at run time (second_writing/README.md, which also says how far the library is independent of them).
//
// The cases were chosen by writers who knew a list of faults that the authors had planted in copies of these sources. So
// that the test stops those faults is no blind trial of it.
//
// NOTHING HERE IS SECRET. Every key below follows from two seeds that are written here, so nothing is hidden under them.
//
// A DIGEST stands for a list of cells. A cell is a coin, a number below p = )" KNOWN_PRIME R"(. The digest is the first 32 bytes of
// BLAKE2b-512 (RFC 7693; no key) of the list's bytes, eight bytes a cell, the lowest first. Word w of a digest is its
// bytes 4 w to 4 w + 3, the lowest first.
#pragma once
#include <cstdint>

constexpr int COINS_KNOWN_CHAINS = 2, COINS_KNOWN_CONTEXTS = 12, COINS_KNOWN_LONG_CONTEXTS = 4;
constexpr int COINS_KNOWN_FILL_GROUPS = 3, COINS_KNOWN_LONG_GROUPS = 5, COINS_KNOWN_DIGEST_WORDS = 8;

// FROM A SEED TO THE PROCESS KEYS. The programs derive theirs from other inputs (common/coin_keys.h).
// Chain 0 has digests of n = 8 words, chain 1 of n = 10, whatever the build's own number is. The inputs: seed (test seed
// number 2 or 3 of hash_commitment/hc_format.h), rho, and 20 public words. D and h_0 are words[0 .. n - 1], h_card is
// words[n .. 2 n - 1], and the start of a link is words[0 .. 2 n - 1]. The answers (only leaf's and key_word's routines exist):
//   Kbat          cp::hedged_key(): from the seed, rho and D
//   leaf          cp::draw_leaf(): the draw under Kbat among COINS_KNOWN_LIVE_LEAVES live leaves
//   Kproc         cp::batch_key(): a batch's process key, from Kbat and h_card
//   key_word      coin_key_word(): the word by which two holders of Kproc compare it
//   once          cp::once_key(): the one-time files' process key, from the seed, h_0 and COINS_KNOWN_VERSION
//   link          cp::link_key(): the process key of link COINS_KNOWN_LINK, with rho as the making's fresh bytes
//   leaf_masks    cp::leaf_mask_key(): the mask key of leaf COINS_KNOWN_LINK, with rho as the making's fresh bytes
//   anchor_masks  cp::seed_key() with ANCHOR_MASKS: the anchor's mask key
// A key's 32 bytes are its eight words, each with its lowest byte first.
constexpr uint32_t COINS_KNOWN_VERSION = 530690, COINS_KNOWN_LIVE_LEAVES = 1000, COINS_KNOWN_LINK = 3;
struct CoinsKnownChain {
  int n;
  uint8_t seed[32], rho[32];
  uint32_t words[20];
  uint8_t Kbat[32];
  uint32_t leaf;
  uint8_t Kproc[32];
  uint32_t key_word;
  uint8_t once[32], link[32], leaf_masks[32], anchor_masks[32];
};
)";

const char* const kContextsHead = R"(
// THE BLINDING CONTEXTS.
// A context's process key is one of a chain's three, and its key K_ctx follows from that key and the context's counter,
// version word, object and member (coin_context_key(); the nonce is zero). But for context 3, the numbers are those that
// the second implementation gives to the context that is named above the row. Apart from the version word, only
// contexts 0 to 2 and, of the lists below, the slot pads match what cp_prove draws; outside the test at the start no
// program uses the other contexts' objects or the other groups' streams (hiding/coins_own.cuh).
// Word 0 of K_ctx is odd in one of the contexts 0, 1 and 2 and even in another, so that a fault which strikes under one
// value of that bit alone meets a context with it. The bit was chosen because the authors had planted such a fault in a
// copy of these sources; no other property of a key is covered. The maker prints no table otherwise.
enum CoinsKnownKey : int { COINS_KNOWN_KPROC = 0, COINS_KNOWN_ONCE = 1, COINS_KNOWN_LINK_KEY = 2 };
struct CoinsKnownContext {
  int chain;
  CoinsKnownKey key;
  uint64_t counter;
  uint32_t version, object, member;
  uint32_t K_ctx[8];
};
)";

const char* const kFillHead = R"(
// THE STREAMS THAT ARE NUMBERED FROM A BASE: four cells a call (hz::coin4(), hiding/trees.cuh).
// COINS_KNOWN_FILL[context][group] is the digest of the group's list under that context's key. A list holds its cells in
// the order in which they are named here.
//   group 0, slot pads: for gid = 0 to 59, the cells 0 to 3 ns - 1 of stream 4 (4096 + 64 gid) + 1, with
//     ns = COINS_KNOWN_SLOTS[context]. They are the pads of a sub-proof with ns slots: coordinate kappa < 3 of slot y
//     is cell ns kappa + y. Under context 0, ns is above 6584, the largest number of slots in the reference parameter
//     set of a specification that is not in this repository; no ns has been compared with the number of slots at an
//     example configuration (hiding/coins_own.cuh). Every ns is 3 mod 4, so that coordinates 1 and 2 begin at a call's
//     fourth and third cell.
//   group 1, many streams: the first COINS_KNOWN_STARTS[context] cells of the streams 16392, 16393 and 16394 (the
//     crossing-pad commitment's), then of the 96 streams from 31744 on (the opening's).
//   group 2, long runs: the cells 16384 to 32767 of stream 16393, then the 8190 cells from 2^35 + 3 on of stream 31834,
//     which begin at a call's fourth cell and so lie in one call more than 8190 / 4, rounded up.
// No cell of a context is in two of its lists, that of the single draw below among them: a program may draw them all in
// one context, whose ledger lets no cell be drawn twice.
)";

const char* const kWholeStreamHead = R"(
// A single long draw, under context 6: COINS_KNOWN_WHOLE_STREAM is the digest of the COINS_KNOWN_WHOLE_STREAM_CELLS
// cells from cell COINS_KNOWN_STARTS[6] = 256 on of stream 31761. With the 256 cells of group 1 they are
// the 3112764 cells of the rows of coins of a request's second opening message in the reference parameter set of a
// specification that is not in this repository. That text does not print the number, and nothing here computes it. No
// proof uses this context's object (hiding/coins_own.cuh). In the same set some streams of the shared opening message
// are longer, the longest 27 times: none of them is drawn whole here.
)";

const char* const kLongHead = R"(
// THE STREAMS OF A REQUEST'S ROW COMMITMENTS.
// Stream (NUM, kind, row) has the number 2^30 + 2^16 NUM + 2^14 kind + row. A call gives five cells in a stream of kind
// 2 and four in every other. COINS_KNOWN_LONG[context][group] is the digest of the group's list under the key of that
// context, one of the first four, with T = COINS_KNOWN_TREES[context].
//   group 0, blinding coordinates: for NUM = 0 to T - 1 and row = 0, 9, 300: the cells 0 to 1023 of (NUM, 0, row).
//   group 1, rows of coins: for NUM = 0 to T - 1: the cells 0 to 1023 of (NUM, 1, 17).
//   group 2, long runs: the cells 0 to 16382 of (2, 1, 0), of (72, 1, 40), of (1, 3, 0) and of (1, 3, 3). Each ends
//     inside a call.
//   group 3, salts: for NUM = 0 to T - 1 and v = 0 to 63: the five cells from 5 I(v) on of (NUM, 2, 0), the salt of
//     position I(v). I(v) = v for v < 56, and I(56 + j) = 2^(20 + 3 j) + j for j < 8.
//   group 4, leaves: for NUM = 0 to T - 1 and I = 0 to 63: the leaf of column I of a tree with 32 rows and 2^6 columns
//     whose columns' hash chains have left zeros. It is the output of the hash's call C(sixteen zero
//     words, [the salt of position I of (NUM, 2, 0) | 32, 6 | NUM, I | zeros], c0 = 2, c1 = 0, l = 0,
//     num = 104 + 2^16 NUM, last = false), where a pair a, b is the 64-bit word a + 2^32 b. The list takes the output's
//     first eight 32-bit words, as four 64-bit words. These are no cells; they show the salts that the leaves took in.
)";

const char* const kWholeRowHead = R"(
// A whole row of coins of 2^22 cells, as many as a long row has coordinates (K = 2^22) in the reference parameter set
// of a specification that is not in this repository: the digest of the cells 0 to COINS_KNOWN_WHOLE_ROW_CELLS - 1 of
// (72, 1, 40) under the key of context 0. Only a test reads it (tests/coins_known_host_test.cpp).
)";

// The texts above state some of this file's constants a second time, as literals. Builds each
// statement listed here from the constants with which the answers are computed, looks for it in the
// text named beside it, and returns the first that is not found, or "" if all are found. Other
// literals, in the texts' prose, are not compared. coins_known_test.sh would not detect a difference:
// it compares the stored table with this program's output, and both would hold the same contradiction.
std::string missing_statement() {
  const auto n = [](std::uint64_t value) { return std::to_string(value); };
  const std::pair<const char*, std::string> stated[] = {
      {kHead, "COINS_KNOWN_CHAINS = " + n(kChains) + ", COINS_KNOWN_CONTEXTS = " + n(kContexts) +
                  ", COINS_KNOWN_LONG_CONTEXTS = " + n(kLongContexts) + ";"},
      {kHead, "COINS_KNOWN_FILL_GROUPS = " + n(kFillGroups) + ", COINS_KNOWN_LONG_GROUPS = " + n(kLongGroups) + ","},
      {kHead, "COINS_KNOWN_VERSION = " + n(kVersion) + ", COINS_KNOWN_LIVE_LEAVES = " + n(kLiveLeaves) +
                  ", COINS_KNOWN_LINK = " + n(kLink) + ";"},
      {kWholeStreamHead, "under context " + n(kWholeStreamContext) + ":"},
      {kWholeStreamHead, "COINS_KNOWN_STARTS[" + n(kWholeStreamContext) + "] = " + n(kStarts[kWholeStreamContext]) + " on"},
      {kWholeStreamHead, "the " + n(kStarts[kWholeStreamContext] + kWholeStreamCells) + " cells"},
  };
  for (const auto& s : stated)
    if (!std::strstr(s.first, s.second.c_str())) return s.second;
  return "";
}

// A context's or a group's number on the command line: one or two decimal digits. -1 for any other
// text.
int small_number(const char* text) {
  const std::size_t length = std::strlen(text);
  return length >= 1 && length <= 2 && std::strspn(text, "0123456789") == length ? std::atoi(text) : -1;
}

int usage() {
  std::fprintf(stderr, "usage: coins_known_maker [cells fill|long <context> <group>]\n");
  return 2;
}

}  // namespace

// Usage:
//   coins_known_maker                                the table, on standard output
//   coins_known_maker cells fill <context> <group>   fill_list(): one value per line, 16 hex digits
//   coins_known_maker cells long <context> <group>   the same for long_list()
// The `cells` forms are for coins_known_test.sh. Exit code 0; 2 for bad usage; 1, and no table, if
// word 0 of K_ctx has the same parity in contexts 0, 1 and 2 (see kContextsHead), or if
// missing_statement() finds that a statement is missing.
int main(int argc, char** argv) {
  const Chain chains[kChains] = {chain(0), chain(1)};

  if (argc == 5 && !std::strcmp(argv[1], "cells")) {
    const bool fill = !std::strcmp(argv[2], "fill");
    const int x = small_number(argv[3]), group = small_number(argv[4]);
    if ((!fill && std::strcmp(argv[2], "long")) || x < 0 || x >= (fill ? kContexts : kLongContexts) || group < 0 ||
        group >= (fill ? kFillGroups : kLongGroups))
      return usage();
    const Coins coins(process_key(chains, kNamed[x]), kNamed[x].context);
    for (std::uint64_t cell : fill ? fill_list(coins, x, group) : long_list(coins, group, kTrees[x]))
      std::printf("%016llx\n", (unsigned long long)cell);
    return 0;
  }
  if (argc != 1) return usage();
  const std::string missing = missing_statement();
  if (!missing.empty()) {
    std::fprintf(stderr, "coins_known_maker: the text printed into the table (kHead, kWholeStreamHead) lacks \"%s\", which follows "
                         "from this program's constants: no table is printed. Change the constant and the text together\n",
                 missing.c_str());
    return 1;
  }

  Key k_ctx[kContexts];
  for (int x = 0; x < kContexts; ++x) k_ctx[x] = context_key(process_key(chains, kNamed[x]), kNamed[x].context);
  const unsigned odd = (k_ctx[0][0] & 1) + (k_ctx[1][0] & 1) + (k_ctx[2][0] & 1);
  if (odd == 0 || odd == 3) {
    std::fprintf(stderr, "coins_known_maker: word 0 of K_ctx is %s in all of the contexts 0, 1 and 2: no table is printed. "
                         "Take other slots or seeds for these contexts\n", odd ? "odd" : "even");
    return 1;
  }

  std::fputs(kHead, stdout);
  std::printf("constexpr CoinsKnownChain COINS_KNOWN_CHAIN[COINS_KNOWN_CHAINS] = {\n");
  for (const Chain& c : chains) {
    std::printf("    {%d,\n", c.n);
    print_bytes("seed", c.seed);
    print_bytes("rho", c.rho);
    std::printf("     /* %-12s */ {", "words");
    print_words(c.words.data(), 20, "                          ");
    std::printf("},\n");
    print_bytes("Kbat", c.kbat);
    std::printf("     /* %-12s */ %uu,\n", "leaf", unsigned(c.leaf));
    print_bytes("Kproc", c.kproc);
    std::printf("     /* %-12s */ 0x%08xu,\n", "key_word", unsigned(c.key_word));
    print_bytes("once", c.once);
    print_bytes("link", c.link);
    print_bytes("leaf_masks", c.leaf_masks);
    print_bytes("anchor_masks", c.anchor_masks);
    std::printf("    },\n");
  }
  std::printf("};\n");

  std::fputs(kContextsHead, stdout);
  std::printf("constexpr CoinsKnownContext COINS_KNOWN_CONTEXT[COINS_KNOWN_CONTEXTS] = {\n");
  for (int x = 0; x < kContexts; ++x) {
    const Named& n = kNamed[x];
    std::printf("    // %d: %s\n", x, n.what);
    std::printf("    {%d, %s, %lluull, %uu, %uu, %uu,\n     {", n.chain,
                n.key == 0 ? "COINS_KNOWN_KPROC" : n.key == 1 ? "COINS_KNOWN_ONCE" : "COINS_KNOWN_LINK_KEY",
                (unsigned long long)n.context.counter, unsigned(n.context.version), unsigned(n.context.object),
                unsigned(n.context.member));
    print_words(k_ctx[x].data(), 8, "");
    std::printf("}},\n");
  }
  std::printf("};\n");

  std::fputs(kFillHead, stdout);
  for (const auto& sizes : {std::make_pair("COINS_KNOWN_SLOTS", kSlots), std::make_pair("COINS_KNOWN_STARTS", kStarts)}) {
    std::printf("constexpr uint32_t %s[COINS_KNOWN_CONTEXTS] = {", sizes.first);
    for (int x = 0; x < kContexts; ++x) std::printf("%s%llu", x ? ", " : "", (unsigned long long)sizes.second[x]);
    std::printf("};\n");
  }
  std::printf("constexpr uint32_t COINS_KNOWN_FILL[COINS_KNOWN_CONTEXTS][COINS_KNOWN_FILL_GROUPS][COINS_KNOWN_DIGEST_WORDS] = {\n");
  for (int x = 0; x < kContexts; ++x) {
    const Coins coins(process_key(chains, kNamed[x]), kNamed[x].context);
    std::printf("    {\n");
    for (int group = 0; group < kFillGroups; ++group) {
      std::printf("        {");
      print_words(digest(fill_list(coins, x, group)).data(), 8, "");
      std::printf("},\n");
    }
    std::printf("    },\n");
  }
  std::printf("};\n");

  std::fputs(kWholeStreamHead, stdout);
  {
    const Coins whole(process_key(chains, kNamed[kWholeStreamContext]), kNamed[kWholeStreamContext].context);
    std::printf("constexpr uint64_t COINS_KNOWN_WHOLE_STREAM_CELLS = %lluull;\n", (unsigned long long)kWholeStreamCells);
    std::printf("constexpr uint32_t COINS_KNOWN_WHOLE_STREAM[COINS_KNOWN_DIGEST_WORDS] = {\n    ");
    print_words(digest(whole_stream_list(whole)).data(), 8, "");
    std::printf("};\n");
  }

  std::fputs(kLongHead, stdout);
  std::printf("constexpr uint32_t COINS_KNOWN_TREES[COINS_KNOWN_LONG_CONTEXTS] = {%u, %u, %u, %u};\n", unsigned(kTrees[0]),
              unsigned(kTrees[1]), unsigned(kTrees[2]), unsigned(kTrees[3]));
  std::printf("constexpr uint32_t COINS_KNOWN_LONG[COINS_KNOWN_LONG_CONTEXTS][COINS_KNOWN_LONG_GROUPS][COINS_KNOWN_DIGEST_WORDS] = {\n");
  for (int x = 0; x < kLongContexts; ++x) {
    const Coins coins(process_key(chains, kNamed[x]), kNamed[x].context);
    std::printf("    {\n");
    for (int group = 0; group < kLongGroups; ++group) {
      std::printf("        {");
      print_words(digest(long_list(coins, group, kTrees[x])).data(), 8, "");
      std::printf("},\n");
    }
    std::printf("    },\n");
  }
  std::printf("};\n");

  std::fputs(kWholeRowHead, stdout);
  {
    const Coins coins(process_key(chains, kNamed[0]), kNamed[0].context);
    Cells row;
    append(row, coins, serial_stream(num_of_segment_commitment(1, 1), kKindBlindingRow, 40), 0, kWholeRowCells);
    std::printf("constexpr uint64_t COINS_KNOWN_WHOLE_ROW_CELLS = %lluull;\n", (unsigned long long)kWholeRowCells);
    std::printf("constexpr uint32_t COINS_KNOWN_WHOLE_ROW[COINS_KNOWN_DIGEST_WORDS] = {\n    ");
    print_words(digest(row).data(), 8, "");
    std::printf("};\n");
  }
  return 0;
}
