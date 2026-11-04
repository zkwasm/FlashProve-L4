// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Parameters of the weight table T, for the committer program `cp_commit` and the prover program
// `cp_prove`: the numbers that the parameter file (`policy.txt`) gives for T, and what follows
// from them and from the number of weight rows.
// Programs use HfParams (prover/hash_commitment/hf_opening_params.h), which extends HfTableParams.
// After a failed check the derived fields must not be used. The verifier program `cp_verify` does
// not use this file: it takes the same numbers from blocks/frame/parameters.h, so the checks here
// limit only what the committer and the prover accept.
//
// The weights W (m0 rows of k = 2^lkW columns) are read as a matrix M of R = 2^fb long rows. A
// row's message is its l data entries, then zeros, then its pads: i0 entries. The message and nm
// random entries, the masks, are encoded by blk::Code{fln, i0, nm} (blocks/code/code.h), of dimension
// KD = i0 + nm and length n = 2^fln: the masks are the first nm coefficients, then the message.
// The R codewords are T. A column of T, one symbol of each row, is a tree leaf of one Merkle tree.
// Terms: docs/glossary.md.
#pragma once
#include "common/setting.h"
#include "hash_commitment/hf_counting_rules.h"
#include "hash_commitment/hf_table_format.h"
#include <cstddef>
#include <cstdint>
#include <string>

// A member function that the GPU's code may call too.
#ifdef __CUDACC__
#define HF_HD __host__ __device__
#else
#define HF_HD
#endif

namespace hf {

static_assert(HF_DW == DW, "hf_table_format.h and common/setting.h must agree on a digest's words");

// The layout, in 32-bit words, in which the GPU code passes tcols opened columns of T (`shown` in
// identifiers) to hf_blocks_opening() (hf_keeper.h), which converts them to a blk::Opening. First
// come the 2^c digests at depth c of T's tree (the top layer), then the columns, each with the
// sibling digests on its path up to that layer. hf_shown() (hf_keeper.h) fills this struct for any
// tcols. Only tests read sy_words.
struct HfShown {
  int tcols = 0;   // the number of opened columns of T
  int c = 0;       // the least c that minimizes shown_words: `top_layer` in hf_counting_rules.h
  uint64_t column_words = 0; // R symbols and 5 salt symbols, 2 words each, then fln - c siblings
  uint64_t sy_words = 0;     // all columns and salts, without siblings: tcols (2 R + 10)
  uint64_t shown_words = 0;  // DW 2^c + tcols column_words
};

struct HfTableParams {
  // ---- Inputs. The names lkW, fb and so on, here and in the messages below, are not lines of
  // `policy.txt`. They are short names for the numbers on three lines:
  // `table` (fb fln fi0 fnm 0 lkW), `sec` (hb, fhb), `work positions` (pow pows powc).
  int lk = 0;    // lkW
  size_t m0 = 0; // W's rows, from the model's shape: `layoutW` in prover/common/weight_layout.h
  int fhb = 0;   // fhb: the target, in bits
  // Bits of proof of work before positions are drawn.
  int pows = 0;  // pows: in a batch's shared hash chain, which all proofs of the batch repeat
  int powc = 0;  // powc: in the key file and the leaf files, which `cp_commit` writes
  int b = 0;     // fb
  uint64_t i0 = 0, nm = 0; // fi0, even; fnm, odd
  int ln = 0;              // fln
  // How many answers the table's record stores for each row: policy.txt's line `answers`.
  int answers = 0;
  // The configuration digest, a hash of the public configuration files
  // (docs/format/FORMAT.md, R4.9), as eight words: every hash of T's tree starts from it. The
  // code that reads the public files sets it. While it is all zero, no tag for the tree's hashes
  // can be built: the program stops (TagT in blocks/hash/hash.h).
  uint32_t configuration[8] = {};
  // ---- Derived by table_derive(), except walks, which stays 2. Only tests read lead, slabs and walks.
  size_t k = 0, R = 0;
  int ly = 0;      // ceil(log2 m0): the bits of a row number y of W
  int lcl = 0;     // lk - b >= 1: the low bits of a column number c of W, which stay in the entry's
                   // index (see `row_of`)
  int ld = 0;      // ly + lk - b: the bits of a data entry's index
  int lc = 0;      // the bit length of KD - 1, at least ld: the bits of a coefficient index
  int lead = 0;    // lc - ld
  uint64_t l = 0;  // m0 2^lcl: a row's data entries, at the message indices 0 .. l - 1
  uint64_t KD = 0; // i0 + nm: a row's coefficients, the dimension of T's code; odd
  uint64_t n = 0;  // 2^ln
  int slabs = 0;   // ceil(R / 16): a column is hashed in groups of 16 symbols (HF_SLAB_SYMBOLS)
  // A commitment's first table is the anchor, which the committer tests once (the one-time test, in
  // the key file). Batches are proved against leaf tables: further tables of the same weights, each
  // tied to the anchor by the LINK proof in its leaf file (`link` in identifiers and messages).
  // The programs take the numbers of columns that they open from lin_sized() and link_sized()
  // (blocks/frame/lin_frame.h, blocks/frame/link_frame.h). The limit that the programs enforce
  // counts columns: blk::Configured::table_columns() in blocks/frame/configure.h.
  int walks = 2;   // passes over the weights in a batch's opening: the folded row u; the columns

  // ---- The map from W's address to M's, and back: M[r][i] = W[y][c], where r is the high fb bits
  // of c and i = y 2^lcl + the low lcl bits of c. y < m0, c < k; r < R, i < l. Only tests call
  // these three functions.
  HF_HD uint32_t row_of(uint64_t /* y */, uint64_t c) const { return (uint32_t)(c >> lcl); }
  HF_HD uint64_t coef_of(uint64_t y, uint64_t c) const { return (y << lcl) + (c & (((uint64_t)1 << lcl) - 1)); }
  HF_HD void cell_of(uint32_t r, uint64_t i, uint64_t &y, uint64_t &c) const {
    y = i >> lcl, c = ((uint64_t)r << lcl) + (i & (((uint64_t)1 << lcl) - 1));
  }
  // A row's pads (docs/format/09-key-and-leaf-files.md, R13.8): three for each stored answer, in
  // the last entries of the row's message of i0 entries. pad_entry(j) is pad j's index in the
  // message. table_derive() requires l + pads() <= i0, so the pads come after the data.
  HF_HD uint64_t pads() const { return 3 * (uint64_t)answers; }
  HF_HD uint64_t pad_entry(uint64_t j) const { return i0 - pads() + j; }
  // The largest value allowed for s, the level of T's tree that `cp_commit` stores (`-keep`; the
  // leaves are level 0): an opening's paths end at its top layer, so s <= ln - c. By default s is
  // the smaller of lcl and this value. It allows for HF_COLUMNS_MOST columns;
  // Unit::plan() in hf_keeper.cu checks s against the top layer for the number of columns that a
  // batch opens.
  // Why lcl: hf_side_make_columns() (hf_open_side.cuh) computes each opened column within a run of
  // 2^lcl consecutive columns, so for s <= lcl one run holds all 2^s columns under a stored node;
  // a higher s takes 2^(s - lcl) runs for each opened column.
  int keep_level_most() const { return ln - top_layer(ln, HF_COLUMNS_MOST); }

  // ---- The checks, in three functions; the third also fills the derived fields.
  // HfParams::derive() (hf_opening_params.h) calls them in this order, with its own checks between
  // them; the order fixes the name under which a file with several faults is rejected. Each returns
  // the empty string, or the reason for the rejection, which begins with the check's name.
  static std::string number(uint64_t x) { return std::to_string(x); }
  // The ranges of lkW, fb and fln.
  std::string table_lines_first() const {
    if (lk < ELKW_LEAST || lk > ELKW_MOST)
      return "w-shape-lk: lkW must be from " + number(ELKW_LEAST) + " to " + number(ELKW_MOST);
    if (b < HF_FB_LEAST || b > HF_FB_MOST || b >= lk)
      return "policy-rows: fb must be from " + number(HF_FB_LEAST) + " to " + number(HF_FB_MOST) + " and below lkW";
    if (ln < HF_LN_LEAST || ln > HF_LN_MOST) return "policy-fln: fln must be from 2 to " + number(HF_LN_MOST);
    return "";
  }
  // fhb, pows and powc. hb: HfParams::hb, the target in bits.
  std::string table_lines_second(int hb) const {
    if (fhb < 1 || fhb > hb) return "policy-fhb: fhb must be from 1 to hb";
    if (pows < 0 || pows > 32 || pows >= fhb) return "policy-pows: pows must be at most 32 and below fhb";
    if (powc < 0 || powc > 32 || powc >= fhb) return "policy-powc: powc must be at most 32 and below fhb";
    return "";
  }
  // Checks m0 and the row.
  std::string table_derive() {
    if (!m0 || m0 > ((size_t)1 << ELMW)) return "w-shape-mW: W must have at least 1 and at most 2^" + number(ELMW) + " rows";
    k = (size_t)1 << lk;
    // The row.
    ly = log2_ceil(m0);
    if (b > ly) return "policy-rows: fb must be at most the number of bits of a row number of W";
    R = (size_t)1 << b;
    slabs = (int)((R + HF_SLAB_SYMBOLS - 1) / HF_SLAB_SYMBOLS);
    lcl = lk - b;
    ld = ly + lcl;
    n = (uint64_t)1 << ln;
    l = (uint64_t)m0 << lcl;
    if (i0 >> 32 || nm >> 32) return "internal-hc-derive: a line's value is below 2^32";
    if (l > i0) return "policy-mask-place: fi0 must be at least the number of a row's data coefficients, " + number(l);
    if (!(nm & 1)) return "policy-mask-shape: fnm must be odd";
    if (i0 % power_of_two_at_least(nm))
      return "policy-mask-place: fi0 must be a multiple of the least power of two that is at least fnm";
    if (i0 & 1) return "policy-mask-place: fi0 must be even";
    // A row's message, fi0 entries, is a whole number of runs of 2^(lkW - fb) entries.
    if (i0 & (((uint64_t)1 << lcl) - 1)) return "policy-mask-place: fi0 must be a multiple of 2^(lkW - fb)";
    KD = i0 + nm;
    lc = log2_ceil(KD); // the bit length of KD - 1
    if (lc < ld) return "internal-hc-derive: the top coefficient index has fewer bits than a data index"; // cannot happen once l <= fi0
    lead = lc - ld;
    if (lc + 1 > ln) return "policy-fln: 2^fln must be at least 2^" + number((uint64_t)lc + 1);
    if (answers < 1 || answers > 4) return "policy-answers: the line answers must be from 1 to 4";
    if (l + pads() > i0)
      return "policy-pad-place: fi0, the third number of the line table, must be at least " + number(l + pads()) + ": a row's data coefficients and three pads for each of its answers";
    return "";
  }
};

} // namespace hf
