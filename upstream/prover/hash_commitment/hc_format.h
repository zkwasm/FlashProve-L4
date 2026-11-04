// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Constants of the generator that makes the masks, salts and pads of a weight table
// (the committed, encoded weights), and the fixed test seeds. Masks are encoded with each row,
// salts hashed into each Merkle leaf. The generator is the ChaCha20 block function (hc_chacha.cuh)
// under the table's own key, which is derived from the commitment's seed.
// hf_format.h has the weight commitment's other constants and repeats some of these as HF_ names.
// No CUDA and no namespace: every name begins with HC_ or hc_, the prefix of the files for the
// generator (hc_chacha.cuh) and the seed (hc_seed.h). Namespace hc holds what hc_seed.h and
// hc_create_file.h (creating a file) declare. Unrelated code outside this directory is in the same
// namespace: hc::TieH, a linear claim of the approximate range check.
#pragma once
#include "common/field_host.h"
#include "field/field_names.h"
#include <cstddef>
#include <cstdint>

constexpr uint64_t HC_P = FIELD_P;                 // p, the number of elements of F
constexpr int HC_SEED_WORDS = 8;                   // the seed, 32 bytes
constexpr size_t HC_SEED_BYTES = 4 * HC_SEED_WORDS;
constexpr int HC_BLOCK_WORDS = 16;                 // words in a ChaCha20 block; nothing reads it
// A block gives eight masks: its eight 64-bit values (hc_block_value() in hc_chacha.cuh) modulo p.
// hf_mask8() (hf_mask.cuh) accepts a block only if all eight values are below this limit, and
// otherwise tries the next block: below a multiple of p, every residue modulo p occurs equally
// often. The limit is p itself: p = 2^64 - 2^32 + 1, and 2 p does not fit into 64 bits.
// hf_mask8() uses HF_MASK_LIMIT, kept equal by a static_assert there. It takes a lower limit only
// if compiled with HC_MASK_LOW_LIMIT=1, which a production build (SERVING_BUILD) does not allow;
// tests do so to make second tries frequent.
constexpr uint64_t HC_MASK_LIMIT = FIELD_WORD_LIMIT; // the greatest multiple of p below 2^64
// The fixed test seeds SEED_TEST_n, n < HC_TEST_SEEDS: anyone can read them here. SEED_TEST_0 is
// the bytes 00 01 .. 1f (little-endian words). Only the static_assert below reads
// HC_OTHER_TEST_SEED.
//
// seed_draw() (prover/hash_commitment/hc_seed.h) rejects a sampled seed equal to a test seed. The
// prover program cp_prove accepts one from seed.bin in the secret key directory (`isk`) only with
// -test-coins (otherwise the error [hc-key-test]): read_the_seed() in prover/prover/isk.cuh. It
// accepts -test-coins only if that directory's isk.txt has the line "seed test" [cp-test-coins]:
// read_isk() in the same file. With -test-coins the 32 bytes that each batch otherwise takes from
// the operating system are read from a file, and nothing is hidden from anyone who has that file
// and the seed. With -test-seed (below) the seed is read from a file too, and nothing is hidden.
//
// The committer program cp_commit writes "seed test" into a commitment made with -test-seed.
// Nothing authenticates the line: anyone who can write isk.txt can add it (cut_isk_text() in
// prover/common/cp_files.h; docs/limits-all-rows.tsv, row H9.3). A production build has neither
// option, and its cp_prove rejects a commitment with the line [cp-isk-seed-test].
constexpr uint32_t HC_TEST_KEY_WORD = 0x484B4559u; // "HKEY"
constexpr int HC_TEST_SEEDS = 256, HC_OTHER_TEST_SEED = 255;
constexpr uint64_t hc_mix64(uint64_t z) { // public source: splitmix64.c by Sebastiano Vigna, one step
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
constexpr uint32_t hc_test_seed_word(int n, int i) { // word i < HC_SEED_WORDS of SEED_TEST_n
  return n == 0 ? 0x03020100u + 0x04040404u * (uint32_t)i
                : (uint32_t)(hc_mix64((uint64_t)HC_TEST_KEY_WORD + 16 * (uint64_t)n + (uint64_t)i) >> 16);
}
static_assert(hc_test_seed_word(0, 0) == 0x03020100u && hc_test_seed_word(0, 7) == 0x1f1e1d1cu &&
                  hc_test_seed_word(1, 0) == 0x935EF63Cu && hc_test_seed_word(1, 7) == 0xC5EB768Du &&
                  hc_test_seed_word(HC_OTHER_TEST_SEED, 0) == 0x1DDE391Du &&
                  hc_test_seed_word(HC_OTHER_TEST_SEED, 7) == 0x389D04E5u,
              "the known answers for the test keys");
// Error name from seed_stop() (hc_seed.h) for a sampled seed that is a test seed or all zero.
constexpr const char *HC_STOP_KEY_TEST = "hc-key-test";
