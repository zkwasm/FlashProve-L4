// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Constants of the weight commitment's format for the GPU programs `cp_commit` and `cp_prove`,
// first half: what belongs to the format as a whole and to a weight table. The second half,
// hf_message_format.h, has what is generated per batch, per LINK proof (between two weight tables)
// and once per commitment; hf_format.h includes both. Section numbers run across both halves: this
// half has no sections 6 to 8.
// Terms. A weight table (T) is all the weights as a matrix of long rows, each with random masks,
// encoded, with one Merkle tree over its columns; RT_F (T_F in identifiers) is the tree's root. The
// anchor is a commitment's first weight table. A leaf table ("copy" in HF_RECORD_COPY_WORD) is a
// further table of the same weights with its own masks; it is not a leaf of a Merkle tree.
// Terms: docs/glossary.md.
#pragma once
#include "frame/version.h" // blocks/: the one version word
#include "field/field_names.h"
#include <cstddef>
#include <cstdint>
#include <initializer_list>

// ---- 0. The digest's width.
// DIGEST_WORDS is a digest's width in 32-bit words: a compile-time constant, 8 (32 bytes), never
// read from a file. An expression "HF_DW == 8 ? a : b" always takes a.
#ifndef DIGEST_WORDS
#define DIGEST_WORDS 8
#endif
static_assert(DIGEST_WORDS == 8, "DIGEST_WORDS is 8");

// ---- 1. Fixed sizes and limits.
// Here a "line" is a number from the parameter file, policy.txt, under a short name; the file
// itself does not use these names. From its line `table`: fb and fln, log2 of a table's
// rows and columns; fi0 and fnm, the numbers of a row's message entries and masks; lkW, log2 of the
// weight matrix W's columns. From its line `sec`: hb and fhb.
constexpr uint64_t HF_P = FIELD_P;                 // p, the number of elements of F
constexpr int HF_DW = DIGEST_WORDS;                // DW of prover/common/setting.h
constexpr int HF_FB_LEAST = 3, HF_FB_MOST = 13;    // line fb: 2^fb table rows, a multiple of 8
constexpr int HF_LN_LEAST = 2, HF_LN_MOST = 33;    // line fln; tests use 33, blk::read_parameters() accepts at most 32
constexpr int HF_COLUMNS_MOST = 65536;             // keep_level_most() leaves room for this many columns
constexpr int HF_SLAB_SYMBOLS = 16;                // 8-byte symbols in one 128-byte hash block
constexpr int HF_SALT_SYMBOLS = 5;                 // a tree leaf's salt, at any digest width
// The number of zero words before the last digest of a hashed list, `off` words into the list, so
// that the digest is within one block of 32 words.
constexpr int hf_pad(int off) { return off % 32 + HF_DW <= 32 ? 0 : 32 - off % 32; }
// The lines hb and fhb in HfParams::full() and in some parameter sets of HfParams::toy_shape()
// (hf_opening_params.h); otherwise only tests use these two constants.
constexpr int HF_HB_REFERENCE = HF_DW == 8 ? 139 : 150, HF_FHB_REFERENCE = HF_DW == 8 ? 131 : 150;
// The target in bits that every round's error bound must reach, and the default for the bits that
// the programs request: `BAR` in prover/common/setting.h.
constexpr int HF_BAR = HF_DW == 8 ? 128 : 136;

// ---- 2. Tag, version and tier words.
// A build accepts one version word: four characters that name the format, the field and the hash
// (`H821`; the field is Goldilocks, p = 2^64 - 2^32 + 1). `cp_verify` too takes it from
// blocks/frame/version.h. blk::read_parameters() (blocks/frame/parameters.h) rejects a parameter
// file with another word; s8::set() in prover/common/state8.h compares the word again.
constexpr uint32_t HF_VERSION_WORD = blk::SERVED_VERSION;
constexpr bool hf_version_admitted(uint32_t word) { return word == HF_VERSION_WORD; }
// True for "HG01" to "HG06", first character in the highest byte (HF_VERSION_WORD has it in the
// lowest): version words of formats that these programs cannot read. s8::set() rejects them as
// `version-word-other-format`, but blk::read_parameters() rejects a file's `HG01` earlier, as
// `version-unknown`.
constexpr bool hf_version_older(uint32_t word) { return word >= 0x48473031u && word <= 0x48473036u; }
constexpr const char *HF_VERSION_OLDER_TEXT = "the parameter file's line version has a word of another format; these programs cannot read any file of that format, and the commitment must be made again with these programs";
constexpr uint32_t hf_version_word() { return HF_VERSION_WORD; }
// The first word of a table's commitment record (`card` in h_card and wcard.bin; section 5).
constexpr uint32_t HF_RECORD_MAGIC = 0x48435243u; // "HCRC": the anchor's
constexpr uint32_t HF_RECORD_COPY_WORD = 0x4843524Cu; // "HCRL": a leaf table's
// A table's level: 0 for the anchor, 1 for a leaf table. No level is above 1, because each leaf
// table is tied to the anchor itself.
constexpr int HF_LEVEL_MOST = 1;
constexpr int hf_record_level(uint32_t first_word) {
  return first_word == HF_RECORD_MAGIC ? 0 : first_word == HF_RECORD_COPY_WORD ? 1 : -1;
}
// Tier 1: the table has a key (section 3).
constexpr uint32_t HF_TIER_HIDDEN = 1; // the only tier word that a record may contain
static_assert(HF_DW == 8, "this format has digests of 8 words");

// ---- 3. The generator of masks and salts: ChaCha20's block function under a table's key kM.
// kM (HF_SEED_BYTES bytes) is derived from the commitment's seed (seed.bin) and, for a leaf table,
// the 32 bytes of its fresh.bin.
// The function's last four input words, x12 .. x15:
//   masks                (r, g, try, HF_MASK_WORD): row r, block g of eight masks, try 0, 1, ...
//                        A block is accepted if all eight 64-bit values are below HF_MASK_LIMIT,
//                        the greatest multiple of p in 64 bits, which is p itself.
//   salt of tree leaf I  (I mod 2^32, floor(I / 2^32), 0, HF_SALT_WORD). The salt is the block's
//                        first five 64-bit values mod p; there is no second try.
// Code: hf_mask.cuh.
constexpr uint32_t HF_MASK_WORD = 0x4D41534Bu; // "MASK", 1,296,126,795
constexpr uint32_t HF_SALT_WORD = 0x53414C54u; // "SALT", 1,396,788,308
constexpr uint32_t HF_PAD_WORD = 0x50414453u;  // "PADS", 1,346,454,611: a record's pads (hf_pads.cuh)
constexpr uint64_t HF_MASK_LIMIT = FIELD_WORD_LIMIT;
constexpr int HF_SEED_WORDS = 8, HF_BLOCK_MASKS = 8;
constexpr size_t HF_SEED_BYTES = 4 * HF_SEED_WORDS;

// ---- 4. Domain-separation numbers.
// No code uses these three: the hash calls of a table's tree take their numbers from the library,
// blocks/ (TreeTags in hf_tree.cuh).
constexpr uint32_t HF_NUM_LEAF_STEP = 130, HF_NUM_LEAF_END = 131, HF_NUM_NODE = 132;
// The tree's number in the label of T's tree leaves: 0 for every weight table
// (docs/format/FORMAT.md, R7.4).
constexpr uint32_t HF_NUM_F = 0;
// 257 and 9 in a rule for the a-th column index: I_a = (o_0 + 2^32 o_1) mod 2^fln, where o is the
// hash's compression function of a chaining value and the block (257, a) under number 9. Nothing
// uses HF_POS_T or HF_NUM_POSITION: the programs take column indices from the library
// (blk::PositionList, hf_keeper.h).
constexpr uint32_t HF_POS_T = 257, HF_NUM_POSITION = 9;

// ---- 5. The commitment record and its file.
// A commitment record describes one weight table: nine header words, then four digests. Its file,
// wcard.bin, is the record followed by RT_F. rf::HfRecord (hf_record.h) builds, reads and checks
// both. TH() is the tree hash of a list of words.
//   h_0    = TH(the first HF_REC0_WORDS words)
//   h_card = TH(the record)
// `cp_commit` builds a record in memory, checks its tier word and prints h_card; it does not write
// wcard.bin and derives no key from the record. In cp_commit the fourth digest, h_once, stays zero.
constexpr int HF_TF_WORDS = HF_DW;
enum HfRecordAt : int {
  HF_REC_MAGIC = 0,   // HF_RECORD_MAGIC or HF_RECORD_COPY_WORD
  HF_REC_VERSION = 1, // hf_version_word()
  HF_REC_TIER = 2,    // HF_TIER_HIDDEN
  HF_REC_LK = 3,      // line lkW
  HF_REC_M0 = 4,      // m_0, the number of W's rows
  HF_REC_FB = 5,
  HF_REC_FLN = 6,
  HF_REC_FI0 = 7,
  HF_REC_FNM = 8,
  HF_REC_UNIT_LIST = 9,                    // TH(unit list); a unit: one or two tensors' rows in W
  HF_REC_H_PAR = HF_REC_UNIT_LIST + HF_DW, // the digest of the three configuration files
  HF_REC_TH_TF = HF_REC_H_PAR + HF_DW,     // TH(T_F), over the DW words of RT_F
  HF_REC0_WORDS = HF_REC_TH_TF + HF_DW,    // h_0's input ends here
  HF_REC_H_ONCE = HF_REC0_WORDS,
  HF_RECORD_WORDS = HF_REC_H_ONCE + HF_DW, // h_card's input ends here, and RT_F begins in the file
  HF_RECORD_FILE_WORDS = HF_RECORD_WORDS + HF_TF_WORDS
};
static_assert(HF_REC0_WORDS == 9 + 3 * HF_DW && HF_RECORD_WORDS == 9 + 4 * HF_DW && HF_RECORD_FILE_WORDS == 9 + 5 * HF_DW,
              "REC0W, RECW, the file");
static_assert(HF_DW == 8 ? HF_REC0_WORDS == 33 && HF_RECORD_WORDS == 41 && HF_RECORD_FILE_WORDS == 49
                         : HF_REC0_WORDS == 39 && HF_RECORD_WORDS == 49 && HF_RECORD_FILE_WORDS == 59,
              "F93's and FL11's numbers");

// ---- 9. File names.
// HF_RECORD_FILE's only use: foreign_name() (hf_commit_making.h) accepts it in a table's directory.
constexpr const char *HF_RECORD_FILE = "wcard.bin";
// In a table's directory in `isk`, the secret key directory: the 2^(fln - s) digests of one stored
// level s (`keep` in names) of the table's Merkle tree (level 0: the tree leaves), in order,
// without a header.
constexpr const char *HF_NODES_FILE = "hcnodes.bin";

// ---- 10. Error names: where `cp_commit` or `cp_prove` stops or withholds its output.
// seed.bin must be a regular file, no symbolic link, of 32 bytes, not all zero, without permissions
// for group and others.
constexpr const char *HF_STOP_KEY_FILE = "hc-key-file";
// seed.bin contains a test seed from the source, and the call has no test option that accepts it;
// or a table's key about to be used is missing or all zero.
constexpr const char *HF_STOP_KEY_TEST = "hc-key-test";
// hcnodes.bin is unreadable, has the wrong length or does not hash to the table's root.
constexpr const char *HF_STOP_NODES_ROOT = "hc-nodes-root";
// A recomputed node of the stored level differs from the stored one.
constexpr const char *HF_OWN_TEST_NODE = "hc-own-node";
// The stored level s is negative or above fln - c, the level of an opening's top layer of 2^c
// nodes. `cp_commit` takes the largest c of three types of opening.
constexpr const char *HF_STOP_KEEP_LEVEL = "hc-keep-level";
// An output directory (-ivk, -isk) or its working name with .making already exists: `cp_commit`
// makes a commitment only in two new directories.
constexpr const char *HF_STOP_FILE_THERE = "file-there-already";
// One opening would show more columns of a table than a row has masks (line fnm).
constexpr const char *HF_STOP_MASK_BUDGET = "hc-mask-budget";
// No try number below 2^32 gives an accepted block of masks.
constexpr const char *HF_STOP_MASK_TRIES = "internal-mask-tries";
// Self-checks ("own test" in names). They must not be removed as redundant. Three of them recompute
// masks or salts under the key's second copy, `key_again` in hf_mask_own.cuh.
constexpr const char *HF_OWN_TEST_GENERATOR = "own-test-generator"; // the generator's known answers, before the key is read
constexpr const char *HF_OWN_TEST_MASKS = "own-test-masks"; // also: a key without its second copy
// The masks' part of u, the folded rows, at sampled indices.
constexpr const char *HF_OWN_TEST_FOLD = "own-test-fold";
// Only in `cp_prove`'s last line: the key's second copy is missing or all zero. The messages
// printed before that line say own-test-fold or own-test-masks instead.
constexpr const char *HF_OWN_TEST_KEY_COPY = "own-test-key-copy";
constexpr const char *HF_OWN_TEST_SALT = "own-test-salt";
// The record's tier word says that there is a key and the process has none, or vice versa.
constexpr const char *HF_OWN_TEST_HIDDEN = "own-test-hidden";
// A pad of a table, in host memory or in its copy in GPU memory, is not the value
// that a second generation gives.
constexpr const char *HF_OWN_TEST_PADS = "own-test-pads";
