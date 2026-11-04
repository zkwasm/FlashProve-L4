// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The Merkle tree over a weight table T in plain CPU code, for tests and toy sizes: whole() keeps
// every node in host memory. It is a stored tree, defined in
// docs/format/FORMAT.md, "6. Trees and openings" (rules R7.x; a stored tree: R7.4).
// Its hash H is defined in docs/format/FORMAT.md, "3.1 The hash" (rules R2.x).
// prover/hash_commitment/tests/hf_commit_table_test.cu compares leaf() and whole() with the fast
// implementation, prover/hash_commitment/hf_tree.cuh; a test that compares up() with it is not part
// of this repository. This file shares with it
// only BLAKE2b's compression function F and its initialization vector (../blocks/hash/blake2b_f.h); BLAKE2b itself is written out below: a walk over
// the blocks of the whole string, following RFC 7693, section 3.3, with the parameter block of the BLAKE2 specification
// (https://www.blake2.net/blake2.pdf, section 2.8).
// So the comparison can detect a fault in the bytes hashed, the parameter block or the calls of F,
// but none in the vector or in F, except in the GPU-only form of its rotations.
//
//   H(number, tree, index, start; bytes) = the first 32 bytes of BLAKE2b, digest length 32, no key, salt = start's bytes 0 to 15, personalization = its
//       bytes 16 to 31, of the string: the word number + 2^16 tree, the word index, bytes. A word is 8 bytes, lowest first.
//   leaf_I = H(174, num + 2^16 R + 2^40 5 (the leaf's shape: its symbols, its salt's words),
//       2^ln + I, the configuration's digest; the R symbols T[0][I] .. T[R - 1][I], a word each;
//       zero bytes up to the string's next multiple of 128 if the salt would cross one; the salt's
//       five words);
//   node(0, I) = leaf_I;  node(l, y) = H(166, 0, 0, the configuration's digest; node(l - 1, 2 y),
//       node(l - 1, 2 y + 1));  the root RT_F = node(ln, 0).
// Leaves and inner nodes are hashed under different numbers. The idea is public: RFC 6962 (Laurie,
// Langley, Kasper, "Certificate Transparency", 2013), section 2.1. A further public source on what
// a tree's hash calls take in: RFC 8391 (Huelsing, Butin, Gazdag, Rijneveld, Mohaisen, "XMSS:
// eXtended Merkle Signature Scheme", 2018), section 2.5, where each call depends on the node's
// height and index. Here a leaf's call takes in its index; a node's call takes in neither.
//
// Top layer: the 2^c nodes of level ln - c, where up() stops. prover/hash_commitment/hf_tree.cuh
// gives them once for a whole opening. (hf_blocks_opening() in prover/hash_commitment/hf_keeper.h
// converts such an opening to the form of
// docs/format/FORMAT.md, R7.8, which has no top layer.) The idea is public: Aumasson, Endignoux,
// "Improving Stateless Hash-Based Signatures", Cryptology ePrint Archive 2017/933, section 2.1 with Figure 1: all nodes
// of one level are included, and the authentication paths end at that level. A related form is the Merkle cap of
// Chiesa, Yogev, "Subquadratic SNARGs in the Random Oracle Model", Cryptology ePrint Archive 2021/281, section 2.2.
#pragma once
#include "common/setting.h" // DW
#include "hash/blake2b_f.h" // ../blocks
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hf_tree_ref {
static_assert(DW == 8, "a digest has 32 bytes");
// start: 32 bytes as eight words of 32 bits, lowest byte first.
inline void H(uint32_t out[DW], uint64_t number, uint64_t tree, uint64_t index, const uint32_t start[8], const std::vector<uint8_t> &bytes) {
  uint32_t any = 0;
  for (int i = 0; i < 8; i++) any |= start[i];
  if (!any) { // as ../blocks' tags do: whoever forgot the configuration's digest hears of it here, not from a verifier
    fprintf(stderr, "internal: a hash of T's tree starts from zeros: the configuration's digest was not set\n");
    abort();
  }
  std::vector<uint8_t> all;
  for (int k = 0; k < 8; k++) all.push_back((uint8_t)((number + 65536 * tree) >> (8 * k)));
  for (int k = 0; k < 8; k++) all.push_back((uint8_t)(index >> (8 * k)));
  all.insert(all.end(), bytes.begin(), bytes.end());
  uint8_t block[64] = {32, 0, 1, 1}; // digest length, key length, fanout, depth; then zeros; salt and personalization from byte 32 on
  for (int i = 0; i < 32; i++) block[32 + i] = (uint8_t)(start[i / 4] >> (8 * (i % 4)));
  const uint64_t IV[8] = {blk::blake2b_IV0, blk::blake2b_IV1, blk::blake2b_IV2, blk::blake2b_IV3,
                          blk::blake2b_IV4, blk::blake2b_IV5, blk::blake2b_IV6, blk::blake2b_IV7};
  blk::u64 h[8], m[16];
  for (int i = 0; i < 8; i++) {
    h[i] = IV[i];
    for (int k = 0; k < 8; k++) h[i] ^= (uint64_t)block[8 * i + k] << (8 * k);
  }
  const size_t n = all.size(), blocks = (n + 127) / 128;
  for (size_t b = 0; b < blocks; b++) {
    for (int i = 0; i < 16; i++) {
      m[i] = 0;
      for (int k = 0; k < 8; k++)
        if (128 * b + 8 * i + k < n) m[i] |= (uint64_t)all[128 * b + 8 * i + k] << (8 * k);
    }
    blk::Blake2bF::F(h, m, b + 1 < blocks ? 128 * (b + 1) : n, 0, b + 1 == blocks);
  }
  for (int w = 0; w < DW; w++) out[w] = (uint32_t)(h[w / 2] >> (32 * (w % 2)));
}
// column: canonical symbols. num: the tree's number, HF_NUM_F (0) for a weight table.
// Not checked: that the symbols are canonical, num < 2^16, R < 2^24, I < 2^ln.
inline void leaf(uint32_t out[DW], const uint64_t *column, uint32_t R, const uint64_t salt[5], int ln, uint32_t num, uint64_t I,
                 const uint32_t configuration[8]) {
  std::vector<uint8_t> bytes;
  for (uint32_t r = 0; r < R; r++)
    for (int k = 0; k < 8; k++) bytes.push_back((uint8_t)(column[r] >> (8 * k)));
  if ((16 + bytes.size()) % 128 + 40 > 128)
    while ((16 + bytes.size()) % 128) bytes.push_back(0);
  for (int x = 0; x < 5; x++)
    for (int k = 0; k < 8; k++) bytes.push_back((uint8_t)(salt[x] >> (8 * k)));
  H(out, 174, num + ((uint64_t)R << 16) + ((uint64_t)5 << 40), ((uint64_t)1 << ln) + I, configuration, bytes); // the shape stands above the tree's number
}
inline void node(uint32_t out[DW], const uint32_t *left, const uint32_t *right, const uint32_t configuration[8]) {
  std::vector<uint8_t> bytes;
  for (const uint32_t *child : {left, right})
    for (int i = 0; i < 32; i++) bytes.push_back((uint8_t)(child[i / 4] >> (8 * (i % 4))));
  H(out, 166, 0, 0, configuration, bytes);
}
// All nodes of the tree over the 2^ln leaves: level[l] holds node(l, 0), node(l, 1), .., DW words each.
inline std::vector<std::vector<uint32_t>> whole(const std::vector<uint32_t> &leaves, int ln, const uint32_t configuration[8]) {
  std::vector<std::vector<uint32_t>> level(ln + 1);
  level[0] = leaves;
  for (int l = 1; l <= ln; l++) {
    level[l].resize(((size_t)1 << (ln - l)) * DW);
    for (uint64_t y = 0; y < (uint64_t)1 << (ln - l); y++)
      node(&level[l][y * DW], &level[l - 1][2 * y * DW], &level[l - 1][(2 * y + 1) * DW], configuration);
  }
  return level;
}
// Computes node(ln - c, I >> (ln - c)) from leaf I and its ln - c siblings, lowest level first.
// Not checked: 0 <= c <= ln.
inline void up(uint32_t out[DW], const uint32_t *leaf_I, const uint32_t *siblings, int ln, int c, uint64_t I, const uint32_t configuration[8]) {
  uint32_t now[DW];
  for (int w = 0; w < DW; w++) now[w] = leaf_I[w];
  for (int l = 1; l <= ln - c; l++) {
    const uint32_t *other = siblings + (size_t)(l - 1) * DW;
    const bool right = (I >> (l - 1)) & 1;
    uint32_t next[DW];
    node(next, right ? other : now, right ? now : other, configuration);
    for (int w = 0; w < DW; w++) now[w] = next[w];
  }
  for (int w = 0; w < DW; w++) out[w] = now[w];
}
} // namespace hf_tree_ref
