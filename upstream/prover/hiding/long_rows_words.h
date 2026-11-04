// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// `Object` separates the `coins` (private random values) of the generators in cp_prove and
// cp_commit. It is word 11 of a coin's label: the 16 words that, with a key, determine a
// generator's output (prover/common/coin_chacha.h). The rest defines a row commitment
// ("long tree"): a Merkle tree over the columns of encoded rows. No file that includes this one
// builds it; tests draw its coins (prover/longrow/reference.h). prover/hiding/coins_long.cuh
// repeats `Object`, and a static_assert there compares the two lists.
#pragma once
#include "common/setting.h"
#include <cstdint>
#include <initializer_list>
namespace lrw {
constexpr uint32_t WORKING_WORD = 530690; // lr::Params's default version: word 10 of a coin's label
// Domain-separation numbers of calls C(cv, block, t0, t1, f) of the compression function: earlier_hash_bare_c()
// of prover/common/hash.h. IV is the initial state, and a block is filled up with zero words. Tree
// NUM has R rows and 2^ln columns, and leaf I is column I's. A node is the digest (DW words)
// node(l, y) = C(IV, [node(l - 1, 2 y) | node(l - 1, 2 y + 1)], t0 = y, t1 = l, NODE),
// with node(0, I) = leaf I and root = node(ln, 0). Where NODE, LEAF_CHAIN, LEAF_END or LEAF_FOLD is
// written as f, the word is that number + 2^16 NUM: tree_f() of prover/common/hash.h.
constexpr uint32_t LEAF_CHAIN = 103, LEAF_END = 104, NODE = 105;
constexpr uint32_t POSITION = 9; // f of the call that derives query positions; unused in C++
// A column's hash chain: S = IV; for slab b (ROWS_A_SLAB rows), S = C(S, the column's symbols in
// the slab's rows, in row order, low word first, t0 = b, t1 = 0, LEAF_CHAIN). The last slab may be
// half full. The leaf is the digest of C(S, [the salt's cells | R | ln | NUM | I], t0 = the number
// of slabs, t1 = 0, LEAF_END). R includes the zero rows.
constexpr int ROWS_A_SLAB = SLABR, ROWS_STEP = 8, SALT_CELLS = SALTW / 2, MASK_ROWS = 4, DIGEST_BYTES = 4 * DW;
// The folded leaf, a second form of the leaf; only tests/s8_model.py (Python) builds it. Let Rr be the number
// of real rows (all but the zero rows) and rl = Rr - ROWS_A_SLAB (slabs - 1) the number of those in
// the last slab. If rl <= FOLD_ROWS, the chain stops before the last slab, and the leaf is the
// digest of C(S, B, t0 = slabs - 1, t1 = 0, LEAF_FOLD). Cells (64 bits) 0 .. rl - 1 of the block B
// are that slab's real symbols; zeros follow; the last 7, the trailer, are
// [the salt's cells | R | ln | NUM | I].
constexpr uint32_t LEAF_FOLD = 133;
constexpr int FOLD_ROWS = ROWS_A_SLAB - SALT_CELLS - 2; // the trailer begins at this cell of the block
static_assert(2 * FOLD_ROWS + SALTW + 4 == 2 * ROWS_A_SLAB && FOLD_ROWS == 9,
              "nine symbols and the trailer fill a block");
#ifdef __CUDACC__
__host__ __device__
#endif
    constexpr bool
    folds(int real_rows) { // whether the leaves of a tree of this many real rows can be folded
  return real_rows > 0 && (real_rows - 1) % ROWS_A_SLAB < FOLD_ROWS;
}
// Rows come in groups (each has a class, a number of rows, `Kd` and `a`), in the order of this enum; zero rows
// follow, up to a multiple of ROWS_STEP.
// VEIL: Dalal, Hemo, Rabinovich, Rothblum, "VEIL: Lightweight Zero-Knowledge for Hash-Based
// Multilinear Proof Systems", Cryptology ePrint Archive 2026/683 (version of April 13, 2026).
// Its blinding row (section 1.2, "The Proximity Test"; step 3 of Figures 8 and 9; Definition 2.20)
// is one uniformly random vector, which enters the verifier's random combination of the committed
// vectors with a non-zero coefficient. Here rows are over the base field and coefficients in its
// extension of degree EDEG = 3: MASK_ROWS mask rows replace that vector, and their coefficients
// must span the extension. No code in prover/longrow/ counts mask rows or forms the combination.
enum Class : int {
  N, // narrow data, mod p: a bit, or an integer of at most 16 bits, signed or unsigned (-2^15 <= v < 2^16)
  B, // narrow data that a later segment of the request's proof reads again, in a copy (class C)
  W, // wide data: arbitrary field elements
  C, // a copy of a class B or C row of the previous segment, blinding coordinates included; no coins
  U, // a component of a combined row (Kd = K), or one row with all 3 under a shorter code; no coins
  M  // a mask row: uniform on all K coordinates (VEIL also has a blinding row, in another form)
};
// A row with data has K coordinates: data at j < Kd, zeros up to Ks (Kd rounded up to a multiple of
// pow2ceil(mk)), mk = masks(a, t) blinding coordinates at Ks .. Kc - 1, zeros up to K. a counts the
// sets of t query positions at which the row is ever opened. The blinding coordinates follow VEIL
// (section 1.2, "Trace Queries"; step 2 of Figures 1 and 8): before a vector is encoded, at least
// as many uniformly random elements are appended as the test makes queries. Here, if a > 0, mk
// exceeds a t, the positions of all the sets. VEIL defines the property of the code that this
// relies on (Definition 2.1; a criterion: Remark 2.2) and states it for a Reed-Solomon code whose
// message is the coefficient vector (Example 2.9).
constexpr uint64_t masks(uint64_t a, uint64_t t) {
  return a ? (a * t + 5) & ~(uint64_t)1 : 0;
}
constexpr uint64_t pow2ceil(uint64_t x) {
  uint64_t p = 1;
  while (p < x) p <<= 1;
  return p;
}
// Coins ("cells") come in streams, one per (NUM, kind, row): stream(), which checks neither NUM_BOUND nor ROWS_BOUND.
// NUM is the tree's number; 0 is reserved for the salt of the request's transcript (not checked).
enum Kind : uint32_t {
  ROW_MASKS = 0, // cell c is the blinding value at coordinate Ks + c, c < mk
  MASK_ROW = 1,  // cell j is the coefficient at coordinate j, j < K
  SALT = 2,      // row 0; cell SALT_CELLS I + x = salt cell x of column I
  KEY_ROW = 3    // a class W row made of coins: cell j is the coefficient at coordinate j, j < Kd
};
// Under a batch's process key cp_prove runs two generators. That of prover/hiding/trees.cuh, under
// REQUEST (only tests use FILLER), gives the sub-proofs' one-time pads: hz::Tree::slot_pads(),
// which builds no tree. That of blocks/coins/coins.h runs under ENGINE for a request's proof and
// under ENGINE_BATCH for what a batch's requests share. Both number their streams from 16,384 up,
// not by stream(), and neither sees the other's draws. Under separate keys, cp_commit runs the
// second: ENGINE_KEY_FILE for the key file, ENGINE_ENTRY for a leaf file (`leaf/<k>.bin`).
// Word 12 of a label is the request's slot in its batch, or 0 for coins of no one request.
enum Object : uint32_t { REQUEST = 1, ENGINE = 2, FILLER = 3, ENGINE_BATCH = 4, ENGINE_KEY_FILE = 5, ENGINE_ENTRY = 10 };
// Under one key, only `Object` keeps the two generators' coins apart: equal labels give equal
// coins. It is not proved that under one process key no two blinding values come from the same
// label and the same words of the output. Returns whether `object` differs from 6 to 8, 22 to 24
// and 38, "the opening's" object numbers in prover/hash_commitment/hf_message_format.h. Outside
// static_asserts, no code uses those constants.
constexpr bool no_object_of_the_opening(uint32_t object) {
  for (uint32_t base : {HF_COINS_OPEN, HF_COINS_PART1, HF_COINS_PART2})
    if (object == base || object == hf_coins_object(base, true)) return false;
  return object != HF_LINK_OBJECT;
}
static_assert(no_object_of_the_opening(REQUEST) && no_object_of_the_opening(FILLER),
              "a request's object numbers must differ from every object number of the opening");
static_assert(no_object_of_the_opening(ENGINE) && ENGINE != REQUEST && ENGINE != FILLER && no_object_of_the_opening(ENGINE_BATCH) &&
                  ENGINE_BATCH != REQUEST && ENGINE_BATCH != FILLER && ENGINE_BATCH != ENGINE,
              "the engine's coins have an object number that no other coins under any of the programs' keys have");
static_assert(no_object_of_the_opening(ENGINE_KEY_FILE) && no_object_of_the_opening(ENGINE_ENTRY) &&
                  ENGINE_KEY_FILE > ENGINE_BATCH && ENGINE_ENTRY > ENGINE_KEY_FILE,
              "so have the engine's coins in cp_commit");
constexpr uint32_t NUM_BOUND = 1u << 14, ROWS_BOUND = 1u << 14; // they keep `stream` one-to-one
static_assert(NUM_BOUND <= (1u << 16), "a tree's number fills the upper half of a call's last word");
constexpr uint32_t stream(uint32_t NUM, Kind kind, uint32_t row) {
  return (1u << 30) + (NUM << 16) + ((uint32_t)kind << 14) + row;
}
constexpr int CELLS_A_CALL = 4;
// One block of a stream is one generator call's output: CELLS_A_CALL cells (words 0 to 7) or, in a
// SALT stream, SALT_CELLS, so that a leaf's salt is one call.
#ifdef __CUDACC__
__host__ __device__
#endif
    constexpr uint32_t
    cells_a_block(uint32_t stream) {
  return ((stream >> 14) & 3) == (uint32_t)SALT ? SALT_CELLS : CELLS_A_CALL;
}
// The opening of a tree: for each distinct query position, in ascending order, the real rows'
// symbols in row order and then the salt's cells. The pruned paths follow, leaf level first: for
// each node(l, y) on a queried leaf's path, in ascending order of y, node(l, y xor 1) is sent
// unless it is on such a path. Positions and zero rows are not sent.

// Returns whether f is below 2^16 and differs from every number in the list; it is also false if
// a listed number is 2^16 or more. Domain-separation numbers stay below 2^16: tree_f() adds
// 2^16 NUM. The list is written out here; nothing compares it with the numbers that other files
// use.
constexpr bool differs_from_all(uint32_t f) {
  if (f >> 16) return false;
  for (uint32_t g : {3u,  4u,  5u,  6u,  7u,  8u,   9u,   10u,  11u,  12u,  13u,  14u,  15u,  16u,
                     64u, 65u, 66u, 67u, 72u, 73u,  75u,  83u,  84u,  90u,  91u,  92u,  93u,  94u,
                     95u, 96u, 97u, 98u, 99u, 100u, 101u, 102u, 106u, 120u, 121u, 122u, 123u, 124u,
                     109u, // out of use; it is given no other use
                     // the block of numbers set aside for prover/hash_commitment/
                     130u, 131u, 132u, 133u, 134u, 135u, 136u, 137u, 138u, 139u})
    if (f == g || g >> 16) return false;
  return true;
}
static_assert(differs_from_all(LEAF_CHAIN) && differs_from_all(LEAF_END) && differs_from_all(NODE) &&
                  LEAF_CHAIN != LEAF_END && LEAF_END != NODE && LEAF_CHAIN != NODE,
              "LEAF_CHAIN, LEAF_END and NODE must differ from each other and from the other "
              "domain-separation numbers listed above");
// Only LEAF_FOLD takes one of 133 to 139.
static_assert(!differs_from_all(LEAF_FOLD) && LEAF_FOLD >= 133 && LEAF_FOLD <= 139 && LEAF_FOLD != LEAF_CHAIN &&
                  LEAF_FOLD != LEAF_END && LEAF_FOLD != NODE,
              "LEAF_FOLD must be one of the listed numbers 133 to 139");
} // namespace lrw
