// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Checks host functions for the keys and values of `cp_commit`'s and `cp_prove`'s private
// randomness (`coin` in identifiers) against a table of known answers
// (prover/hiding/coins_known_f2.h), which prover/tests/coins_known_maker.cpp printed from a second
// implementation (prover/tests/coins_second_text/).
// Both programs check part of the table when they start, on the host and in kernels
// [own-test-coins] (hz::coins_own_test(), prover/hiding/coins_own.cuh). This test needs no GPU and,
// unlike them, reports which answer differs.
// Terms. Chain: the keys derived from one seed. Context: one of a chain's three process keys
// (ChaCha20 keys: `Kproc`, `once`, `link`) and a `counter`, `version`, `object` and `member`;
// together they give the key K_ctx. Cell: one field element of a numbered stream under K_ctx. A
// list of cells is stored as its digest.
// Not covered, because they need nvcc: the kernels and rf::hf_coin_on_the_host()
// (prover/hiding/coins_by_value.cuh).
// Built and run by prover/tests/unit.sh.
#include "common/cp_files.h"
#include "common/hash.h"
#include "hiding/coins_known_f2.h"
#include "longrow/reference.h"
#include <cstdio>
#include <cstring>
#include <vector>

static int failed = 0;
// what: the condition that should hold. of, x: "chain" or "context" and its index in the table, or
// null. group: the list's group, or -1.
static void expect(bool held, const char *what, const char *of = nullptr, int x = 0, int group = -1) {
  if (held) return;
  printf("FAILED: %s", what);
  if (of) printf(" (%s %d", of, x);
  if (group >= 0) printf(", group %d", group);
  printf(of ? ")\n" : "\n");
  failed++;
}

// True if want is the first 32 bytes of BLAKE2b-512 of the cells, eight bytes each, little-endian
// (by f2b::C(), prover/common/blake2b_f.h).
static bool digest_is(const std::vector<u64> &cells, const uint32_t want[COINS_KNOWN_DIGEST_WORDS]) {
  const u64 words = 2 * cells.size(), blocks = words ? (words + BLKW - 1) / BLKW : 1;
  u32 cv[SW];
  f2b::first_state(cv);
  for (u64 b = 0; b < blocks; b++) {
    u32 m[BLKW];
    for (u64 i = 0; i < BLKW; i++) {
      const u64 j = b * BLKW + i;
      m[i] = j < words ? (u32)(cells[j / 2] >> 32 * (j % 2)) : 0;
    }
    const bool last = b + 1 == blocks;
    const u64 fed = last ? 4 * words : 4 * (u64)BLKW * (b + 1);
    f2b::C(cv, m, (u32)fed, (u32)(fed >> 32), 0, 0, last);
  }
  return !memcmp(cv, want, 4 * COINS_KNOWN_DIGEST_WORDS);
}

// Context x of the table, as lr::cell(), a CPU reference (prover/longrow/reference.h), takes it.
// flip: the bit of the process key's word 0 to flip, or -1.
struct Context {
  lr::Key key;
  lr::Params params;
  std::vector<u64> list;
  Context(int x, int flip) {
    const CoinsKnownContext &c = COINS_KNOWN_CONTEXT[x];
    const CoinsKnownChain &chain = COINS_KNOWN_CHAIN[c.chain];
    const uint8_t *bytes = c.key == COINS_KNOWN_KPROC ? chain.Kproc : c.key == COINS_KNOWN_ONCE ? chain.once : chain.link;
    for (int i = 0; i < 32; i++) key.key[i / 4] |= (u32)bytes[i] << 8 * (i % 4);
    if (flip >= 0) key.key[0] ^= 1u << flip;
    key.ctr = c.counter, key.obj = c.object, key.mem = c.member;
    params.working_word = c.version;
  }
  void cells(u32 stream, u64 first, u64 count) {
    for (u64 i = 0; i < count; i++) list.push_back(lr::cell(key, stream, first + i, params));
  }
};
// Fills c.list with the cells whose digest is COINS_KNOWN_FILL[x][group]; the groups are defined at
// that array.
static void fill_list(Context &c, int x, int group) {
  c.list.clear();
  if (group == 0) {
    for (u32 gid = 0; gid < 60; gid++) c.cells(4 * (4096 + 64 * gid) + 1, 0, 3 * (u64)COINS_KNOWN_SLOTS[x]);
  } else if (group == 1) {
    for (u32 stream : {16392u, 16393u, 16394u}) c.cells(stream, 0, COINS_KNOWN_STARTS[x]);
    for (u32 i = 0; i < 96; i++) c.cells(31744 + i, 0, COINS_KNOWN_STARTS[x]);
  } else {
    c.cells(16393, 16384, 16384);
    c.cells(31834, ((u64)1 << 35) + 3, 8190);
  }
}
// The same for COINS_KNOWN_LONG[x][group]. Group 4 lists not cells but tree leaves (trees of 32
// rows, 2^6 columns), each hashed by earlier_hash_bare_c() (prover/common/hash.h) from one salt.
static void long_list(Context &c, int group, u32 trees) {
  c.list.clear();
  for (u32 NUM = 0; NUM < trees && group != 2; NUM++) {
    if (group == 0)
      for (u32 row : {0u, 9u, 300u}) c.cells(lrw::stream(NUM, lrw::ROW_MASKS, row), 0, 1024);
    if (group == 1) c.cells(lrw::stream(NUM, lrw::MASK_ROW, 17), 0, 1024);
    if (group == 3)
      for (u32 v = 0; v < 64; v++)
        c.cells(lrw::stream(NUM, lrw::SALT, 0), lrw::SALT_CELLS * (v < 56 ? (u64)v : ((u64)1 << (20 + 3 * (v - 56))) + (v - 56)),
                lrw::SALT_CELLS);
    if (group == 4)
      for (u32 I = 0; I < 64; I++) {
        const lr::Salt salt = lr::salt(c.key, NUM, I, c.params);
        u32 cv[SW] = {0}, m[BLKW] = {0};
        for (int x = 0; x < lrw::SALT_CELLS; x++) m[2 * x] = (u32)salt[x], m[2 * x + 1] = (u32)(salt[x] >> 32);
        m[2 * lrw::SALT_CELLS] = 32, m[2 * lrw::SALT_CELLS + 1] = 6, m[2 * lrw::SALT_CELLS + 2] = NUM, m[2 * lrw::SALT_CELLS + 3] = I;
        earlier_hash_bare_c(cv, m, 2, 0, tree_f(lrw::LEAF_END, NUM));
        for (int i = 0; i < 4; i++) c.list.push_back(cv[2 * i] | (u64)cv[2 * i + 1] << 32);
      }
  }
  if (group == 2) {
    c.cells(lrw::stream(2, lrw::MASK_ROW, 0), 0, 16383);
    c.cells(lrw::stream(72, lrw::MASK_ROW, 40), 0, 16383);
    c.cells(lrw::stream(1, lrw::KEY_ROW, 0), 0, 16383);
    c.cells(lrw::stream(1, lrw::KEY_ROW, 3), 0, 16383);
  }
}

int main() {
  // Checks cp::draw_leaf() (prover/common/cp_files.h), which picks a batch's leaf table among M
  // live ones, and coin_key_word() (prover/common/coin_chacha.h), a 32-bit fingerprint of a process
  // key. The table's keys are not re-derived from its seeds: the programs derive their keys by
  // another rule (prover/common/coin_keys.h), which prover/tests/coin_keys_test.cpp tests.
  // Nothing reads the table's `leaf_masks` and `anchor_masks`.
  for (int s = 0; s < COINS_KNOWN_CHAINS; s++) {
    const CoinsKnownChain &c = COINS_KNOWN_CHAIN[s];
    expect(cp::draw_leaf(c.Kbat, COINS_KNOWN_LIVE_LEAVES) == c.leaf, "cp::draw_leaf() gives the table's leaf", "chain", s);
    u32 words[COINKEYW] = {0};
    for (int i = 0; i < 32; i++) words[i / 4] |= (u32)c.Kproc[i] << 8 * (i % 4);
    expect(coin_key_word(words) == c.key_word, "coin_key_word() gives the table's word", "chain", s);
  }
  { // A draw that rejects its first value (`passed_over`).
    // With M below 2^32 a value is rejected with probability below 2^-32, too rarely for a run or
    // the table's cases to reach that branch of cp::draw_leaf(). This key was found by search:
    // its first value, 18446744071704135846, is above the largest multiple of M = 4294901761
    // below 2^64.
    const uint8_t passed_over[32] = {0x24, 0xe3, 0xb9, 0xe2, 0x02};
    expect(cp::draw_leaf(passed_over, 4294901761u) == 2351731775u, "cp::draw_leaf() passes a first value above the last multiple of M over and takes the second");
  }
  // flip = -1: each context's K_ctx (lr::context_key() calls coin_context_key()) and each list's
  // digest must equal the table's. flip = 9, a negative control: with that bit of the process key
  // flipped, each must differ (`COINS_KNOWN_WHOLE_ROW` and `COINS_KNOWN_WHOLE_STREAM` are left
  // out for their length).
  for (int flip : {-1, 9})
    for (int x = 0; x < COINS_KNOWN_CONTEXTS; x++) {
      Context c(x, flip);
      const bool same = flip < 0;
      u32 K_ctx[COINKEYW];
      lr::context_key(K_ctx, c.key, c.params);
      expect(!memcmp(K_ctx, COINS_KNOWN_CONTEXT[x].K_ctx, sizeof K_ctx) == same,
             same ? "coin_context_key() gives the table's K_ctx" : "the control: another process key gives another K_ctx", "context", x);
      for (int group = 0; group < COINS_KNOWN_FILL_GROUPS; group++) {
        fill_list(c, x, group);
        expect(digest_is(c.list, COINS_KNOWN_FILL[x][group]) == same,
               same ? "the list's digest is COINS_KNOWN_FILL's" : "the control: another key misses COINS_KNOWN_FILL's digest", "context", x, group);
      }
      for (int group = 0; x < COINS_KNOWN_LONG_CONTEXTS && group < COINS_KNOWN_LONG_GROUPS; group++) {
        long_list(c, group, COINS_KNOWN_TREES[x]);
        expect(digest_is(c.list, COINS_KNOWN_LONG[x][group]) == same,
               same ? "the list's digest is COINS_KNOWN_LONG's" : "the control: another key misses COINS_KNOWN_LONG's digest", "context", x, group);
      }
      if (x == 0 && same) {
        c.list.clear();
        c.cells(lrw::stream(72, lrw::MASK_ROW, 40), 0, COINS_KNOWN_WHOLE_ROW_CELLS);
        expect(digest_is(c.list, COINS_KNOWN_WHOLE_ROW), "the whole row's digest is COINS_KNOWN_WHOLE_ROW");
      }
      if (x == 6 && same) {
        c.list.clear();
        c.cells(31761, COINS_KNOWN_STARTS[x], COINS_KNOWN_WHOLE_STREAM_CELLS);
        expect(digest_is(c.list, COINS_KNOWN_WHOLE_STREAM), "the whole stream's digest is COINS_KNOWN_WHOLE_STREAM");
      }
    }
  printf(failed ? "coins_known_host_test: %d checks FAILED\n" : "coins_known_host_test: all checks hold\n", failed);
  return failed ? 1 : 0;
}
