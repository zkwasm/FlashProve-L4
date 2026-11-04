// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Hash modes of the older proof format, byte for byte: calls of F, BLAKE2b's compression function,
// with a domain-separation number `num` in the upper half of F's counter, where BLAKE2b has 0. New
// code uses the current format's hash, blocks/hash/hash.h. Only tests call this file's functions.
// prover/common/hash.h has the prover's own copy of this hash.
// Known-answer tests: blocks/hash/tests/older_test.cpp. What soundness asks of this hash,
// of which other copies make digests in a proof's first query: docs/format/FORMAT.md, R15.3, and
// blocks/hash/MODE.md, "2. What is assumed". cp_verify has a copy of this hash: see tree_hash().
//
// H(num; w) hashes a list w of 32-bit words as BLAKE2b-512 hashes their little-endian bytes, except
// that every call's Tweak (blocks/hash/tweak.h) has num and l = w's length in words.
// C(cv; block; c0, c1; num) is a bare call: l = 0, `last` = false. num's upper 16 bits may name a
// tree. Not checked: that num != 0, and that no number is used for both H and C.
// H, C and the tree hash are restated from prover/common/hash.h and prover/common/host_common.h.
// Source for modelling F as a random function: Luykx, Mennink, Neves, "Security Analysis of
// BLAKE2's Modes of Operation" (ePrint 2016/827), which does not analyse H, C or the tree hash.
#pragma once
#include "hash/hash.h"
#include <vector>

namespace blk {
struct Older {
  typedef Blake2bF Fn;
  static constexpr int SW = 16, BLKW = 32, DW = 8;
  static constexpr int SLAB = 16;       // the symbols that one step of a leaf's chain takes in
  static constexpr int SALT_CELLS = 5;  // a leaf's salt, in symbols
  static constexpr int PIECE = 16 * DW; // the words of a piece of the tree hash
  enum : u32 { TH_LIST = 10, TH_UPPER = 11, TABLE_STEP = 130, TABLE_END = 131, TABLE_NODE = 132 };

  // H and C call F, the compression function of BLAKE2b, which is specified in RFC 7693
  // (M-J. Saarinen, J-P. Aumasson, "The BLAKE2 Cryptographic Hash and Message Authentication Code
  // (MAC)", November 2015, https://www.rfc-editor.org/rfc/rfc7693), section 3.2, and implemented in
  // blocks/hash/blake2b_f.h. A call's Tweak fills F's counter t and flag f:
  // t mod 2^64 = c0 + 2^32 c1, t >> 64 = num + 2^32 l, f = `last`, the flag.
  // Fn::first() gives the first state of BLAKE2b-512 (RFC 7693, section 3.3).
  BLK_FN static void bare(u32 cv[SW], const u32 m[BLKW], u32 c0, u32 c1, u32 num) {
    BLK_HASH_REQUIRE((num & 0xFFFF) < 160, "the numbers from 160 on are not the older format's");
    Fn::compress(cv, m, Tweak{c0, c1, 0, num, false});
  }
  BLK_FN static Digest digest_of(const u32 state[SW]) {
    Digest d;
    BLK_UNROLL
    for (int i = 0; i < DW; i++) d.w[i] = state[i];
    return d;
  }

  // ---- lists ----
  // H(num; w[0 .. L - 1]) as a state: SW words of 32 bits, of which the first DW are the digest.
  BLK_FN static void list_state(u32 out[SW], u32 num, const u32 *w, u64 L) {
    BLK_HASH_REQUIRE(!(L >> 32), "a list has fewer than 2^32 words");
    BLK_HASH_REQUIRE((num & 0xFFFF) < 160, "the numbers from 160 on are not the older format's");
    Fn::first(out);
    const u64 blocks = L ? (L + BLKW - 1) / BLKW : 1;
    for (u64 b = 0; b < blocks; b++) {
      u32 m[BLKW];
      for (int i = 0; i < BLKW; i++) m[i] = b * BLKW + i < L ? w[b * BLKW + i] : 0;
      const bool last = b + 1 == blocks;
      const u64 fed = last ? 4 * L : 4 * (u64)BLKW * (b + 1);
      Fn::compress(out, m, Tweak{(u32)fed, (u32)(fed >> 32), (u32)L, num, last});
    }
  }
  BLK_FN static Digest list_digest(u32 num, const u32 *w, u64 L) {
    u32 s[SW];
    list_state(s, num, w, L);
    return digest_of(s);
  }
  // The tree hash TH(w): each piece of PIECE words (the last may be shorter) is replaced by its
  // digest, under TH_LIST at first and TH_UPPER on lists of digests, until one is left. Host only.
  // cp_verify computes the same hash with TH() (verifier/src/vf_common.h), on the configuration
  // files, a request's tokens and the statement's words.
  // Source for the lowest level and the levels above it under different domain-separation numbers: RFC 6962 (Laurie,
  // Langley, Kasper, "Certificate Transparency", 2013), section 2.1.
  // The rule is kept in: RFC 9162 (Laurie, Messeri, Stradling, "Certificate Transparency Version 2.0", 2021), section
  // 2.1.1, which obsoletes RFC 6962.
  static Digest tree_hash(const u32 *w, size_t n) {
    BLK_REQUIRE(n >= (size_t)DW, "the tree hash is defined for 8 words or more");
    std::vector<u32> now, next;
    for (u32 num = TH_LIST;; num = TH_UPPER) {
      const u32 *from = num == TH_LIST ? w : now.data();
      const size_t have = num == TH_LIST ? n : now.size(), pieces = (have + PIECE - 1) / PIECE;
      next.resize(pieces * DW);
      for (size_t c = 0; c < pieces; c++) {
        const Digest d = list_digest(num, from + c * PIECE, have - c * PIECE < (size_t)PIECE ? have - c * PIECE : PIECE);
        for (int i = 0; i < DW; i++) next[c * DW + i] = d.w[i];
      }
      now.swap(next);
      if (pieces == 1) break;
    }
    Digest d;
    for (int i = 0; i < DW; i++) d.w[i] = now[i];
    return d;
  }

  // ---- leaf chains and nodes ----
  // One step: S := C(S; 16 symbols as 32 words, low word first; c0 = slab, c1 = 0; num).
  BLK_FN static void chain_step(u32 S[SW], const u64 sym[SLAB], u32 slab, u32 num) {
    u32 m[BLKW];
    BLK_UNROLL
    for (int q = 0; q < SLAB; q++) m[2 * q] = (u32)sym[q], m[2 * q + 1] = (u32)(sym[q] >> 32);
    bare(S, m, slab, 0, num);
  }
  // Zeros pad the last slab. Not checked: rows <= 2^32 - 16; beyond it the loop never ends.
  BLK_FN static void chain(u32 S[SW], const u64 *sym, u32 rows, u32 num) {
    Fn::first(S);
    for (u32 b = 0; b * SLAB < rows; b++) {
      u64 slab[SLAB];
      for (u32 q = 0; q < (u32)SLAB; q++) slab[q] = b * SLAB + q < rows ? sym[b * SLAB + q] : 0;
      chain_step(S, slab, b, num);
    }
  }
  BLK_FN static u32 slabs(u32 R) { return (R + SLAB - 1) / SLAB; }
  // Leaf I of 2^ln, a column of R symbols: the digest of C(chain over the column under `step`; the
  // salt's five symbols, R, ln, tree, I in two words, zeros; c0 = slabs(R), c1 = 0; end).
  BLK_FN static Digest leaf(const u64 *sym, u32 R, const u64 salt[SALT_CELLS], u32 ln, u32 tree, u64 I, u32 step, u32 end) {
    u32 S[SW], m[BLKW];
    chain(S, sym, R, step);
    BLK_UNROLL
    for (int i = 0; i < BLKW; i++) m[i] = 0;
    BLK_UNROLL
    for (int x = 0; x < SALT_CELLS; x++) m[2 * x] = (u32)salt[x], m[2 * x + 1] = (u32)(salt[x] >> 32);
    m[10] = R, m[11] = ln, m[12] = tree, m[13] = (u32)I, m[14] = (u32)(I >> 32);
    bare(S, m, slabs(R), 0, end);
    return digest_of(S);
  }
  // Node y = `place` of `level`, from nodes 2 y and 2 y + 1 of the level below; leaves are level 0.
  BLK_FN static Digest node(const Digest &left, const Digest &right, u32 level, u64 place, u32 num) {
    BLK_HASH_REQUIRE(level >= 1 && !(place >> 32), "a node's level is at least 1 and its place below 2^32");
    u32 S[SW], m[BLKW];
    Fn::first(S);
    BLK_UNROLL
    for (int i = 0; i < BLKW; i++) m[i] = i < DW ? left.w[i] : i < 2 * DW ? right.w[i - DW] : 0;
    bare(S, m, (u32)place, level, num);
    return digest_of(S);
  }
  // The older format's weight table; `tree` is 16,384 there.
  BLK_FN static Digest table_leaf(const u64 *sym, u32 R, const u64 salt[SALT_CELLS], u32 ln, u32 tree, u64 I) {
    return leaf(sym, R, salt, ln, tree, I, TABLE_STEP, TABLE_END);
  }
  BLK_FN static Digest table_node(const Digest &left, const Digest &right, u32 level, u64 place) {
    return node(left, right, level, place, TABLE_NODE);
  }
};
} // namespace blk
