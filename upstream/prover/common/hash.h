// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// This header hashes sequences of u32 values; its functions compile for the CPU and for CUDA kernels
// alike. Its one primitive, earlier_hash_c(), calls BLAKE2b's compression function F (RFC 7693, section 3.2;
// prover/common/blake2b_f.h) and puts a use's number and a length into the upper half of F's counter t.
// Two 32-bit words make a 64-bit word of F, the low word first. An output has SW words and is
// called a state; its first DW words are a digest.
// This hash is not the proof format's own, but some digests in a proof's first query (the hash
// input that fixes the statement) are made with it. They are listed in
// docs/format/FORMAT.md, "12.6 The earlier hash"; rule R15.3 of that document defines
// earlier_hash_h_digest() at DW = 8 under the name K. What soundness asks of this hash, and of the tree hash that
// is built with it (k_th(), prover/hiding/product_test.cuh), is stated in that rule and in
// blocks/hash/MODE.md, "2. What is assumed".
// Source for modelling F itself as a random function, with all its inputs free:
// Luykx, Mennink, Neves, "Security Analysis of BLAKE2's Modes of Operation", IACR Transactions on
// Symmetric Cryptology 2016(1), Cryptology ePrint Archive 2016/827 (version of August 31, 2016),
// section 2.2 and Theorem 1. By Theorem 1, F is indifferentiable from a random function up to
// about 2^256 queries if the block cipher inside F is a weakly ideal cipher, that is, ideal except
// for one structural property of BLAKE2's cipher. Because it idealizes the cipher, the theorem
// supports the model and does not prove it.
#pragma once
#include "common/field_host.h"
#include "common/setting.h" // DW, SW, BLKW: the widths of the hash
#include "common/blake2b_f.h"
#include <cstdio>
#include <cstdlib>
static_assert(SW == 16 && BLKW == 32 && DW <= SW, "F has a state of eight and a block of sixteen 64-bit words");
// In BLAKE2b, F's 128-bit counter t is the number of bytes fed. Here:
//   h = cv; m = the block; t mod 2^64 = c0 + 2^32 c1; t >> 64 = num + 2^32 l; f = last.
// num tells the hash's uses apart (domain separation); l is the hashed list's length in words,
// or 0. Replaces cv, the chaining value, by the output.
// before_compress() and after_compress() are hooks for tests and only read their arguments; a
// header that is not part of this repository declares them.
HD void earlier_hash_c(u32 cv[SW], const u32 m[BLKW], u32 c0, u32 c1, u32 l, u32 num, bool last) {
#ifdef TEST_HOOKS
  test_hooks::before_compress(cv, m, c0, c1, l, num, last); // cv is still the input
#endif
  f2b::C(cv, m, c0, c1, l, num, last);
#ifdef TEST_HOOKS
  test_hooks::after_compress(cv, m, c0, c1, l, num, last); // cv holds the output
#endif
}
// BLAKE2b-512's first state (RFC 7693, section 3.3: 64 bytes of output, no key).
HD void earlier_hash_initial_state(u32 cv[SW]) {
  f2b::first_state(cv);
}
// A bare call: l = 0, and the flag `last` is not set. Every call that earlier_hash_h_state() makes for L below
// 2^32 has l > 0 or the flag, so it never gives F the inputs of a bare call.
HD void earlier_hash_bare_c(u32 cv[SW], const u32 m[BLKW], u32 c0, u32 c1, u32 num) {
  earlier_hash_c(cv, m, c0, c1, 0, num, false);
}
// Hashes a list of L words, word j at w[j * stride]. The last block is filled with zero words;
// L = 0 takes one block of zeros. In each call c0 + 2^32 c1 is the number
// of the list's bytes fed so far. With num = 0 and l = 0 the result is BLAKE2b-512 of the list's
// bytes, each word lowest byte first; only prover/tests/hash_test.cpp calls it in this way.
// Programs call it only through earlier_hash_h_state().
HD void earlier_hash_h_state_with_length(u32 out[SW], u32 num, u32 l, const u32 *w, u64 L, u64 stride) {
  earlier_hash_initial_state(out);
  u64 blocks = (L + BLKW - 1) / BLKW;
  if (blocks == 0) blocks = 1;
  for (u64 b = 0; b < blocks; b++) {
    u32 m[BLKW];
    for (int i = 0; i < BLKW; i++) {
      const u64 j = b * BLKW + i;
      m[i] = j < L ? w[j * stride] : 0;
    }
    const bool last = b + 1 == blocks;
    const u64 fed = last ? 4 * L : 4 * (u64)BLKW * (b + 1);
    earlier_hash_c(out, m, (u32)fed, (u32)(fed >> 32), l, num, last);
  }
}
// H(num; w) as a state. L must be below 2^32, which is not checked: l is one word.
HD void earlier_hash_h_state(u32 out[SW], u32 num, const u32 *w, u64 L, u64 stride = 1) {
  earlier_hash_h_state_with_length(out, num, (u32)L, w, L, stride);
}
// The same hash as a digest.
HD void earlier_hash_h_digest(u32 out[DW], u32 num, const u32 *w, u64 L, u64 stride = 1) {
  u32 state[SW];
  earlier_hash_h_state(state, num, w, L, stride);
  for (int i = 0; i < DW; i++) out[i] = state[i];
}
// The word num of a call that goes into a Merkle tree: `number` in the low half and t, which
// identifies the tree, in the upper half. Both must be below 2^16, which is not checked here.
// For `number` 64 to 66, t is the number of the tree's kind. For
// 103 to 105 and 133 (prover/hiding/long_rows_words.h), t is the tree's own number, NUM there.
HD constexpr u32 tree_f(u32 number, u32 t) {
  return number + (t << 16);
}
// The values of t for `number` 64 to 66. Only the static_assert below uses TREE_UP: no code
// compares a tree's t with it. TREE_NONE means "not set"; it is the default of `tnum`, the field
// for t in the struct `Tree` (prover/hiding/trees.cuh).
// Outside tests, a tree's t is TREE_CP (the tree `CP` in prover/prover/inputs.cuh) or TREE_NONE: no
// program has a tree with t >= TREE_UP. No other file's code uses the remaining names.
constexpr u32 TREE_UW = 0, TREE_RA = 1, TREE_BW = 32, TREE_ZV = 48, TREE_ZF = 49, TREE_ZU = 96, TREE_ZUA = 97,
              TREE_ZUC = 112, TREE_S = 255, TREE_CP = 256, TREE_UP = 1u << 15, TREE_NONE = ~0u;
static_assert(TREE_UW < TREE_RA && TREE_RA < TREE_BW && TREE_BW < TREE_ZV && TREE_ZV < TREE_ZF && TREE_ZF < TREE_ZU &&
                  TREE_ZU < TREE_ZUA && TREE_ZUA < TREE_ZUC && TREE_ZUC < TREE_S && TREE_S < TREE_CP &&
                  TREE_CP < TREE_UP,
              "the kinds' numbers differ, and a salted kind's is below TREE_UP");
