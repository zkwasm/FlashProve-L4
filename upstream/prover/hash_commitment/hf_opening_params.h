// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Parameters of the weight commitment for the GPU programs (the committer cp_commit, the prover
// cp_prove), second of two headers. HfParams extends HfTableParams (hf_table_params.h: the weight
// table T, its code, its opened columns) with the parameters of what a prover sends with one
// opening of T. from_policy() fills it from a parsed parameter file; shape(), full() and toy()
// fill it for tests. After a failed from_policy() or derive() the derived fields must not be used.
//
// An opening folds T's rows into one vector u and commits to it (ZU); commits to ZV, which holds
// for each row of the weight matrix W that row's multilinear extension at one random point; and
// opens tcols columns of T. cp_prove makes one for each batch. cp_commit makes one for the one-time
// test of the anchor (a commitment's first weight table) and, in part, for each LINK proof, which
// ties a leaf table (a further table of the same weights) to the anchor.
//
// derive() checks the values read and derives T's numbers only. No program sets the derived fields
// declared in HfParams (see there): blocks/frame/ shapes what cp_commit and cp_prove send. The
// verifier program cp_verify does not include this file: the checks here protect the prover only.
// Terms: docs/glossary.md.
#pragma once
#include "common/setting.h"
#include "hash_commitment/hf_message_format.h"
#include "hash_commitment/hf_table_params.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace hf {

static_assert(HF_START_WORDS == S120W && HF_START_PAD == S120PAD, "hf_message_format.h and common/setting.h must agree on the list under 120 (F38, F93)");
static_assert(HF_FUN_MOST == ELNO, "a row commitment of the opening has a code length of at most 2^ELNO");

// ---- The shapes of the opening's row commitments, on the host. A row commitment encodes the rows
// of a matrix to length n = 2^ln and hashes the n columns into one Merkle tree, of which t query
// positions are opened. The rules are those of shapeF(), shape_flat() and pad_tree_shape() in
// prover/hiding/trees.cuh, repeated here so that lengths and memory can be planned without CUDA.
// prover/hash_commitment/tests/hf_params_test.cpp checks a few shapes against fixed numbers, not
// against trees.cuh. Only that test uses these functions.
//
// mk: a code row's blinding coordinates. Ks: K, the number of pads, rounded up to a multiple of the
// power of two at or above mk. Kc = Ks + mk: a code row's coordinates.
struct PadShape { // a pad commitment (S, S_i)
  int ln = 0, t = 0, mk = 0;
  size_t n = 0, Ks = 0, Kc = 0;
};
// The shape for K pads (random elements for masking what rounds send: see ns_c, ns_i) that `a` sets
// of positions read, at a target of `bits` bits: the least power of two n >= 2048 at which t and
// mk = (a t + 5) & ~1 reach a fixed point with Kc 2^irM <= n. irM: line oiS.
inline PadShape pad_shape(size_t K, int a, int bits, int irM) {
  size_t n = power_of_two_at_least(K) << irM;
  if (n < 2048) n = 2048;
  for (;; n *= 2) {
    int t = 1;
    bool ok = true;
    PadShape s;
    for (;;) {
      const int mk = (a * t + 5) & ~1;
      const size_t q = power_of_two_at_least((uint64_t)mk), Ks = (K + q - 1) / q * q, Kc = Ks + (size_t)mk;
      if ((Kc << irM) > n) {
        ok = false;
        break;
      }
      s.ln = log2_ceil(n);
      const int t2 = (int)least_count(n + Kc + 1, s.ln + 1, bits);
      s.t = t, s.mk = mk, s.n = n, s.Ks = Ks, s.Kc = Kc;
      if (t2 == t) break;
      t = t2;
    }
    if (ok) return s;
  }
}
// A table has R rows of K cells; a cell is ec elements of F (3 for a cell in E). A code row has
// Kp = 2^w - P2 data coordinates and mk blinding coordinates, Kc in all (P2: the power of two at or
// above mk). It interleaves 2^x table rows, Kpp = Kp / 2^x cells of each, so each of the
// R2 = R / 2^x groups of table rows takes NHr code rows (NH: NHr rounded up to a power of two).
// Real rows are rows over F, ec for each code row: Rr in all, Rd of them dummy rows that bring the
// number up to 8.
struct TableShape { // a table's row commitment (ZV, UW_i)
  bool ok = false;
  int w = 0, ln = 0, t = 0, mk = 0, x = 0;
  size_t n = 0, P2 = 0, Kp = 0, Kc = 0, Kpp = 0, R2 = 0, NHr = 0, NH = 0, Rd = 0, Rr = 0;
  int cube() const { return w + log2_ceil(R2 * NH); } // variables of its cube (padded index space)
};
// The shape at a given w (ln = w + ir), with the x that gives the fewest real data rows, the
// smaller x at a tie. Returns false if this w has no shape.
inline bool table_shape_at(size_t R, size_t K, int ec, int a, int hb, int ir, int w, TableShape &s) {
  s = TableShape();
  s.w = w, s.ln = w + ir;
  if (s.ln + 1 > 32 || s.ln + hb > EBITS || s.ln > ELNO) return false;
  s.n = (size_t)1 << s.ln;
  s.t = (int)least_count((((uint64_t)1 << ir) + 1) * ((uint64_t)1 << w) + 1, w + ir + 1, hb);
  s.mk = (a * s.t + 5) & ~1;
  s.P2 = power_of_two_at_least((uint64_t)s.mk);
  if (2 * s.P2 > ((size_t)1 << w)) return false;
  s.Kp = ((size_t)1 << w) - s.P2;
  s.Kc = s.Kp + (size_t)s.mk;
  size_t least = 0;
  bool any = false;
  for (int x = 0; ((size_t)1 << x) <= s.P2 && ((size_t)1 << x) <= R; x++) {
    if (R % ((size_t)1 << x)) continue;
    const size_t Kpp = s.Kp >> x, NHr = (K + Kpp - 1) / Kpp, dr = (size_t)ec * (R >> x) * NHr;
    if (any && dr >= least) continue;
    any = true, least = dr;
    s.x = x, s.Kpp = Kpp, s.R2 = R >> x, s.NHr = NHr, s.NH = power_of_two_at_least(NHr);
  }
  if (!any || least + 8 > ((size_t)1 << 32)) return false;
  s.Rd = least >= 8 ? 0 : 8 - least;
  s.Rr = least + s.Rd;
  s.ok = true;
  return true;
}
// Chooses w: the least w that has a shape with at most rw real data rows (Rr - Rd) or, if there is
// none, the least w that gives the fewest. rw: line orw. ir, log2 of the inverse rate: line oiU
// (cells in E) or oiD (cells in F). a, hb: pad_shape()'s `a`, `bits`.
inline TableShape table_shape(size_t R, size_t K, int ec, int a, int hb, int ir, size_t rw) {
  TableShape s, fewest;
  for (int w = 10; w <= EBITS - 1 - hb && w <= 29; w++) {
    if (!table_shape_at(R, K, ec, a, hb, ir, w, s)) continue;
    if (s.Rr - s.Rd <= rw) return s;
    if (!fewest.ok || s.Rr - s.Rd < fewest.Rr - fewest.Rd) fewest = s;
  }
  return fewest; // .ok is false if no w does
}

// The parameters of one opening, in one of two modes: a batch's, or (`once`) the one-time test's.
// cp_commit and cp_prove use objects of their own, in which they set only `once` (true for the
// committer's openings) and the HfShown part; its tcols comes from blocks/frame/ (see derive()).
// Nothing sets or reads the other fields. They describe the shared message: the part of an opening
// that a batch's requests have in common. The closing is its last part.
struct HfOpen : HfShown { // HfShown (hf_table_params.h): tcols, c and the lengths of T's columns
  bool once = false;
  int leafd = 0;   // depth of the binding tree (Merkle tree over the requests; its root: ROOT)
  int members = 0; // its leaves, 2^leafd: the most requests of a batch; 1 for the one-time test
  int sets = 0;    // the sets of positions for which ZV has blinding: `members` + 1
  int claims = 0;  // the closing's claims (C.0) .. (C.tcols - 1) and (Z): tcols + 1, in both modes
  // Offsets in the shared message, in 32-bit words (hf_message_format.h, section 6). at_shown: T's
  // top layer, where the shown part begins; at_columns: the first column; at_closing: the closing's
  // rounds. nonce_words: 2 if positions are drawn after proof of work (pows, or powc, is not 0),
  // else 0; the nonce follows root ZU and, in the closing, root X_2.
  // The closing: lambda_e rounds (HfParams::lambda_c or lambda_once) of 2 elements of E, 6 words
  // each; beta, one element; root X_2 and its nonce; the first level (ZU, ZV, S); further levels.
  uint64_t nonce_words = 0;
  uint64_t at_ROOT = 0, at_root_ZU = 0, at_shown = 0, at_columns = 0, at_root_S = 0, at_root_ZV = 0, at_closing = 0;
};

struct HfParams : HfTableParams {
  // ---- From the parameter file.
  // from_policy() takes these from the lines `sec`, `slots`, `work positions` and `group folded`.
  int hb = 0;    // target, in bits, for the row commitments' numbers of query positions
  // Bits of proof of work before positions are drawn: a list of positions is sized for its target
  // less these bits. pows and powc (HfTableParams) are the counterparts for a batch's shared rounds
  // and for the committer's rounds.
  int pow = 0;   // a request's own files
  int lNmax = 0; // an upper bound on leafd
  int leafd = 0; // depth of the binding tree: a batch has at most 2^leafd requests
  int lZU = 0, lnZU = 0; // fuk, fun: log2 of a ZU code row's data coordinates, and of its code length
  // ---- Derived.
  // No code in this repository sets the
  // fields from here to `link_anchor`.
  // The notes say what each field is for.
  //
  // ZU. Code row hi holds u's coefficients hi 2^fuk .. (hi + 1) 2^fuk - 1. A code row in which all
  // of these are unused (the gap between data and masks, or above KD - 1) does not exist.
  // The notes on ZU's code rows, here and at toy(), assume that in a row of T the
  // data come before the masks. blk::Code (blocks/code/code.h) puts the masks first,
  // from coefficient 0, so a position in these notes is not a position in that code.
  size_t ZU_K = 0, ZU_n = 0; // 2^fuk data coordinates of a code row; code length 2^fun
  // The part of ZU's shape that depends on the target of its list of positions: `ZU_batch` for a
  // batch (hb - pows), `ZU_c` for the committer's files (hb - powc).
  struct ZuCounts {
    int t = 0, mk = 0; // query positions; blinding coordinates, at 2^fuk .. 2^fuk + mk - 1
    size_t Kc = 0;     // 2^fuk + mk: above 2^fuk, so an index in a blinded row has fuk + 1 bits
  };
  ZuCounts ZU_batch, ZU_c;
  int ZU_hi_bits = 0;        // lc - fuk: the variables of hi
  int ZU_hi_data = 0;        // code rows below this hold data (the last perhaps gap and masks too)
  int ZU_hi_from = 0, ZU_hi_to = 0; // further code rows, from .. to, hold masks; none if from > to
  int ZU_rows = 0;           // the number of code rows that exist
  int ZU_Rd = 0, ZU_Rr = 0;  // dummy real rows, up to a floor of 8; real rows, 3 ZU_rows + ZU_Rd
  int ZU_column_symbols = 0; // ZU_Rr + 6 + 5
  // The other row commitments' shapes, for planning. S_i (`Si`) and UW_i (`UW`) are request i's
  // own. omega[y]: the multilinear extension of W's row y at the opening's random point.
  // k = 2^lk: W's columns.
  TableShape ZV, ZV_once; // ZV[y] = omega[y], m0 cells over E, read by 1 + 2^leafd or by 2 sets of positions
  TableShape UW;          // UW_i in a batch: k cells over E, one set of positions
  PadShape S, S_once, Si; // the shared closing's pads, and a request's in a batch
  int lambda_c = 0, lambda_once = 0; // the shared closing's rounds: max(lc, ZV's cube)
  int lambda_i = 0;                  // rounds of a request's second message: max(UW's cube, ZV's)
  int ns_c = 0, ns_once = 0;         // S's pads: 2 lambda (no message slots)
  int ns_i = 0;                      // S_i's: 2 lk (the merge), then 2 lambda_i
  HfOpen batch, once;
  // The anchor's side of a LINK proof: like `batch`, but with powc. Only its HfShown part applies
  // to a LINK proof, not the offsets.
  HfOpen link_anchor;

  // g_*: a file's bits of proof of work. hb_*: the target less these bits (see `pow`). The one-time
  // test's files and a LINK proof are the committer's (powc). In a batch both the shared message
  // and the second messages of the requests (`member` in identifiers) read ZV, so ZV takes the
  // smaller of pow and pows.
  int g_open(bool one_time) const { return one_time ? powc : pows; }  // the shared message
  int g_member(bool one_time) const { return one_time ? powc : pow; } // a request's second message
  int hb_open(bool one_time) const { return hb - g_open(one_time); }
  int hb_member(bool one_time) const { return hb - g_member(one_time); }
  int hb_zv(bool one_time) const { return hb - (one_time ? powc : pow < pows ? pow : pows); }
  int hb_c() const { return hb - powc; }

  const HfOpen &open(bool one_time) const { return one_time ? once : batch; }
  const HfOpen &mode(bool one_time) const { return open(one_time); } // the same as open()

  // ---- ZU's code rows have two numbers. hi < 2^ZU_hi_bits, the high part of a coefficient's
  // index, also counts rows that do not exist. The rank counts only the rows that exist, and orders
  // symbols, real rows and streams. A caller must not pass one for the other. Nothing in this
  // repository calls these three functions.
  HF_HD bool ZU_exists(int hi) const { return hi < ZU_hi_data || (hi >= ZU_hi_from && hi <= ZU_hi_to); }
  HF_HD int ZU_rank(int hi) const { return hi < ZU_hi_data ? hi : ZU_hi_data + hi - ZU_hi_from; } // of an existing row
  HF_HD int ZU_hi_of(int rank) const { return rank < ZU_hi_data ? rank : rank - ZU_hi_data + ZU_hi_from; }

  // Checks the fields read from the parameter file; derives HfTableParams' other fields, no more.
  // Returns the empty string, or the reason why the values are not allowed, which begins with the
  // failed check's name ("policy-leafd: .."). Most of these names are not rejection names of
  // cp_verify. Not checked here: the mask budget (the programs count opened columns against
  // blk::Configured::table_columns()), and anything that depends on the bits that a verifier's user
  // requests.
  std::string derive() {
    std::string why;
    // First the values as read, without m0.
    if (!(why = table_lines_first()).empty()) return why;
    if (hb < 1 || hb > 4096) return "hid-hb: hb must be from 1 to 4096";
    if (pow < 0 || pow > 32 || pow >= hb) return "policy-pow: pow must be at most 32 and below hb";
    if (lNmax < 0 || lNmax > HF_LNMAX_MOST) return "w-policy-Nmax: lNmax must be at most " + number(HF_LNMAX_MOST);
    if (leafd < 0 || leafd > HF_LEAFD_MOST || leafd > lNmax)
      return "policy-leafd: leafd must be at most " + number(HF_LEAFD_MOST) + " and at most lNmax";
    if (!(why = table_lines_second(hb)).empty()) return why;
    if (lZU < 1 || lZU >= lnZU || lnZU > HF_FUN_MOST)
      return "policy-zu-lines: fuk must be at least 1 and below fun, and fun at most " + number(HF_FUN_MOST);
    // m0; then T's row.
    if (!(why = table_derive()).empty()) return why;
    // The shapes of an opening's oracle messages and the number of columns that it opens are not
    // derived here: blocks/frame/lin_frame.h computes them from the parameter file (a LINK proof's:
    // blocks/frame/link_frame.h), as for cp_verify.
    return "";
  }

  // Fills this object from `frame`, the frame's lines as parse() (prover/common/parameter_file.h)
  // accepted them (a blk::Parameters: see s8::set(), prover/common/state8.h), and `rows`, the number
  // of W's rows. Returns 0, or 2 after one line on stderr. First s8::set() checks that the version
  // word is this build's and fills the global s8::now; every other error comes from derive().
  template <class Frame> int from_policy(const Frame &frame, size_t rows) {
    *this = HfParams();
    m0 = rows, answers = (int)frame.answers;
    const std::string refused = s8::set(frame);
    if (!refused.empty()) {
      fprintf(stderr, "the parameter file is not admitted: %s\n", refused.c_str());
      return 2;
    }
    hb = s8::now.hb, fhb = s8::now.fhb, pow = s8::now.pow, pows = s8::now.pows, powc = s8::now.powc;
    lk = (int)frame.block_log, b = (int)frame.table_rows_log, ln = (int)frame.table_log_n;
    i0 = (uint32_t)frame.table_data, nm = (uint32_t)frame.table_random;
    while ((uint64_t)1 << leafd < frame.slots) leafd++; // `slots` is a power of two
    lNmax = leafd;
    lZU = (int)frame.folded.log_ell, lnZU = (int)frame.folded.log_n;
    const std::string why = derive();
    if (!why.empty()) {
      fprintf(stderr, "the parameter file is not admitted: %s\n", why.c_str());
      return 2;
    }
    return 0;
  }

  // Prints one or two lines for a program's log. Every number printed is public.
  void print(FILE *f = stdout) const {
    fprintf(f,
            "hf: lk=%d m0=%zu ly=%d R=%zu lcl=%d l=%llu i0=%llu nm=%llu KD=%llu ld=%d lc=%d ln=%d hb=%d fhb=%d | leafd=%d\n",
            lk, m0, ly, R, lcl, (unsigned long long)l, (unsigned long long)i0, (unsigned long long)nm,
            (unsigned long long)KD, ld, lc, ln, hb, fhb, leafd);
    if (pow || pows || powc)
      fprintf(f, "hf: work bits: pow=%d pows=%d powc=%d\n", pow, pows, powc);
  }

  // ---- Named shapes, for tests. shape() exits with code 2 if derive() rejects the values: the
  // caller's error. pow, pows and powc stay 0: no proof of work. HfTableParams::answers is 1.
  static HfParams shape(int lk, size_t m0, int hb, int fhb, int b, uint64_t i0, uint64_t nm, int ln, int leafd, int lZU, int lnZU) {
    HfParams p;
    p.lk = lk, p.m0 = m0, p.hb = hb, p.fhb = fhb, p.b = b, p.i0 = i0, p.nm = nm, p.ln = ln, p.answers = 1, p.leafd = leafd;
    p.lZU = lZU, p.lnZU = lnZU, p.lNmax = leafd;
    const std::string why = p.derive();
    if (!why.empty()) {
      fprintf(stderr, "internal error: a named shape is not admitted: %s\n", why.c_str());
      exit(2);
    }
    return p;
  }
  // Full size: Llama 4 Maverick's 3,076,096 rows of 2^18 columns. which = 2 gives leafd 8, not 6:
  // sized for one opening shared by a compute node's 192 requests (at most 2^8 = 256); a batch of cp_prove
  // has at most 64 (SLOTS_MOST, prover/prover/slot.h). The full-size example's policy.txt (see
  // prover/examples/TARGET-KITS.md) differs in fln, hb, fhb (here 33, 139, 131) and in the bits of
  // proof of work (here 0).
  static HfParams full(int which = 0) {
    return shape(18, 3076096, HF_HB_REFERENCE, HF_FHB_REFERENCE, 11, 402653184ull, 44040191ull, 33, which == 2 ? 8 : 6, 22, 24);
  }
  // Small sets for unit tests. Sets 0 to 6: W has 4,768 rows of 2^8
  // columns; sets 7 to 9: 59,392 rows of 2^11 columns. The notes on sets 0 and 2 assume that data
  // come before masks (see ZU_K).
  //   0 plain             R = 8: half of the 16 symbols (HF_SLAB_SYMBOLS, a slab) that one step of
  //                       a tree leaf's hash takes; unused coefficients between data and masks;
  //                       ZU's code rows 10, 11, 14, 15 do not exist
  //   1 128 rows          fb = lkW - 1, so lcl = 1; n = 2^15
  //   2 small_gap         fi0 = l + 256, the least that holds a row's pads: data and masks share a
  //                       code row of ZU
  //   3 extra_index_bit   lc = ld + 1, so lead = 1
  //   4 masks_to_the_top  KD = 2^lc - 1, the largest for its lc
  //   5 sixty-four rows   6 one_place: leafd = 0   9 1,024 rows
  //   7, 8   T's fb, fln, fi0, fnm and lkW as in prover/examples/target_small_f2/policy.txt, line
  //          `table`; set 7 has fhb 16
  static constexpr int TOYS = 10;
  static HfParams toy(int which) {
    HfParams P = toy_shape(which);
    for (uint32_t i = 0; i < 8; i++) P.configuration[i] = 0x00796F74u + (i << 24); // made up: a toy has no public files
    return P;
  }
  static HfParams toy_shape(int which) {
    constexpr int T = HF_DW == 8 ? 132 : 144, B = HF_HB_REFERENCE, FB = HF_FHB_REFERENCE;
    switch (which) {
    case 0: return shape(8, 4768, T, 16, 3, 196608, 32767, 20, 2, 14, 16);
    case 1: return shape(8, 4768, T, 16, 7, 10240, 2047, 15, 2, 10, 13);
    case 2: return shape(8, 4768, T, 16, 3, 152832, 255, 20, 2, 14, 16);
    case 3: return shape(8, 4768, T, 16, 3, 262144, 4095, 20, 2, 14, 16);
    case 4: return shape(8, 4768, T, 16, 3, 196608, 65535, 20, 2, 14, 16);
    case 5: return shape(8, 4768, T, 16, 6, 24576, 4095, 17, 2, 11, 13);
    case 6: return shape(8, 4768, T, 16, 3, 196608, 32767, 20, 0, 14, 16);
    case 7: return shape(11, 59392, B, 16, 8, 524288, 65535, 23, 6, 15, 17);
    case 8: return shape(11, 59392, B, FB, 8, 524288, 65535, 23, 6, 15, 17);
    default: return shape(11, 59392, B, FB, 10, 131072, 32767, 21, 6, 13, 15);
    }
  }
};

typedef HfOpen HfMode;
} // namespace hf
using hf::HfMode;
using hf::HfOpen;
using hf::HfParams;
