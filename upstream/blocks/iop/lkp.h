// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// LKP, the lookup argument by product trees: every entry of a key vector occurs in a public table.
// RNG, the approximate range check (blocks/iop/rng.h), and WF, the committer's range proof
// (blocks/iop/wf.h), call it, through lkp_prove() and lkp_verify(). In the program cp_verify only
// WF calls it, to check a commitment's key file; for the lookups in a request's sub-proofs
// cp_verify has a second implementation (verifier/src/vf_lookup_argument.h).
//
// The idea: for a random challenge alpha,
//     prod_i (alpha - K[i])  =  a * prod_t (alpha - tau(t))^(m_t),
// for keys K[i], table rows tau(t), multiplicities m_t (the number of keys counted at row t) and a
// public factor a for padding (LkpPadding). Inputs, outputs, steps, sources: "the protocol" below.
//
// The caller's duties: LkpPublic is fixed before the first challenge; and, since K is never sent,
// the caller turns every claim on K that LKP returns (LkpKeys) into a claim on what it has sent.
// LKP checks that each such claim is taken (LkpKeyClaim). It does not check what becomes of it,
// which messages it then reads (LkpPublic::read_from), or which entries the caller's callback
// reads as the multiplicities (LkpPlanesAt).
#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "engine/sumcheck_cpu.h"
#include "iop/context.h"
#include "iop/line.h"
#include "iop/psc.h"
#include "iop/psc_rows.h"
#include "iop/relation.h"
#include "numbers/round_names.h"

namespace blk {

// ---- the protocol ----
// From: a key vector K of 2^v entries, which is never sent (the caller knows how to read it from
//       sent vectors); a public table tau of 2^d rows in E, the extension field (rows may repeat);
//       the multiplicities m_t (each key is counted at one row that it equals) in bits, one bit an
//       entry: b[j, t] = b[j + 2^mu t] = bit j of m_t, j < w = 2^mu (`plane` j: bit j of every
//       m_t), in an oracle message (encoded rows sent as Merkle roots: relation.h) whose
//       out-of-domain sample (a random linear form of the message; SEND, send.h) has been answered.
// To:   linear claims (`Row` in identifiers) with public values (`target` in identifiers) on b and
//       on mask entries (`MaskSlot` in identifiers; random elements of E, each added to one sent
//       scalar: mask.h), which LKP adds to the caller's group (the oracle messages that one
//       proximity test reads); and a claim on K's multilinear extension at a point (LkpKeys; with
//       keys in parts, one for each part).
//
// Both sides of the idea's equation are computed by trees of products, which are not sent.
//   key side    KL_v[i] = alpha - K[i];  KL_k[i] = KL_(k+1)[i] * KL_(k+1)[i + 2^k];  Z = KL_0.
//   table side  TL_d[j + 2^mu t] = 1 + b[j, t] (alpha - 1 - tau(t));
//               TL_l[i] = TL_(l+1)[i] * TL_(l+1)[i + 2^(mu + l)];
//               G_j = TL_0[j]: the product of alpha - tau(t) over the rows whose bit j is set.
//   the tie     Z = a * prod_j G_j^(2^j), by squaring and multiplying:
//               P_(w-1) = G_(w-1), P_j = P_(j+1)^2 G_j, Z = a P_0.
// Messages. V (the verifier): theta, if LkpPublic::shifted. V: alpha. Then steps 1 to 7. Every
// sumcheck is a PSC (psc.h: the sumcheck with a hiding polynomial), and every value that a step
// sends is sent under a mask entry.
//   1  the tie: w - 1 sumchecks without variables. Step 0: 0 = Z - a P_1^2 G_0; sends Z, P_1, G_0.
//      Step j = 1 .. w - 2: the claim on P_j from the step before; sends P_(j+1), G_j.
//      Left over: a claim on Z and one on each G_j.
//   1b only with keys in parts (LkpPublic::parts): from the claim on Z to one on each part's root.
//   2  V: a point r of mu coordinates. The w claims on the G_j, weighted by eq(r, j) (eq: the
//      multilinear equality polynomial), are one claim on TL_0 at r.
//   3  key side, k = 0 .. v - 1: the claim on KL_k at a point, summand y_0 y_1 (the halves of
//      KL_(k+1)) under equality weights; LINE (line.h; V: one more coordinate) joins the two end
//      claims into one on KL_(k+1).
//   4  table side, l = 0 .. d - 1: the same, from the claim of 2.
//   5  the leaves: the claim on TL_d; the summand 1 + b (alpha - 1 - tau~), tau~ the table's
//      multilinear extension, is affine in b, so no value is sent; its linear claim reads b.
//   6  V: a point of mu + d coordinates. The sumcheck of sum eq(point, .) (b^2 - b) = 0 tests that
//      every entry of b is 0 or 1. Sends the value of b at the point of the sumcheck's challenges.
//   7  no message: the linear claim  sum 2^j b[j, t] = 2^v - (the number of padding entries), which
//      gives both sides of the idea's equation the same degree in alpha.
// Errors: (2^v - 1) / |E| for alpha, mu / |E| for r, (mu + d) / |E| for the point of 6, and those
// of PSC and LINE (theta: see LkpPublic).
// Needs 2^v < 2^w and 2^d (2^w - 1) < p (the base field's size), so that the claim of 7 is an
// equation between integers.
//
// Sources. What the labels mean, and each work cited in full: PUBLISHED-WORKS.md,
// "Published methods of the proof system".
//   * Source for the idea's equation, without the factor a: Papini, Haböck, "Improving logarithmic
//     derivative lookups using GKR", Cryptology ePrint Archive 2023/1284 (version of
//     September 18, 2023), section 2, equation (1).
//   * A related technique for the multiplicities, bits that select a term's power, is in Haböck,
//     "Multivariate lookups based on logarithmic derivatives", Cryptology ePrint Archive 2022/1530
//     (version of March 31, 2023), section 4.3, equation (35). That section describes it as a
//     technique of the Polygon Miden VM, and as an application to Plookup of the multiplicity
//     technique of Bootle, Cerulli, Groth, Jakobsen, Maller, "Arya: Nearly linear-time
//     zero-knowledge proofs for correct program execution", ASIACRYPT 2018.
//   * Source for the argument in this form, the equation included: Bootle, Cerulli, Groth,
//     Jakobsen, Maller (just cited), in the full version, Cryptology ePrint Archive 2018/380,
//     appendix B.5. There a digit e selects its power through the entry e v + 1 - e of a further
//     committed matrix; the leaves TL_d have the same form. Its three tasks (to show that the
//     digits are 0 or 1, to show that the multiplicities have the right sum, and to square and
//     multiply) fall here, in the same order, to steps 6 and 7 and the tie. In that paper the
//     powers (x - b)^(2^l) of every table entry b are committed.
//     Here the factors are grouped by the bit j, and only the roots G_j of the trees are squared
//     (the tie).
//   * Steps 3 and 4 are the protocol of Goldwasser, Kalai and Rothblum for a binary tree of
//     multiplications, in Thaler's form, with PSC in place of the plain sumcheck. Source:
//     Goldwasser, Kalai, Rothblum, "Delegating Computation: Interactive Proofs for Muggles",
//     STOC 2008, in the full version, Electronic Colloquium on Computational Complexity,
//     Report TR17-108, sections 1.8 and 3.2. Source: Thaler, "Time-Optimal Interactive Proofs for
//     Circuit Evaluation", arXiv:1304.3812, section 5.3.1, Proposition 2 and the paragraph after
//     Remark 2. Thaler's tree multiplies entries that differ in the least significant bit; the
//     trees here multiply entries that differ in the most significant remaining bit.
//   * Source for committing to nothing further for each key: Qu, Guo, Xuan, Liu, Zhang, "Celer: A
//     Lookup Argument for Large-Scale Queries", CRYPTO 2026, Cryptology ePrint Archive 2026/1453,
//     abstract. How Celer reaches this has not been compared with this file's argument. So Celer is
//     cited for that aim only; as to the method it is related work, no more. For the argument by
//     logarithmic derivatives, Papini and Haböck (cited above; their abstract) reach the same end
//     with the protocol of Goldwasser, Kalai and Rothblum; that form, like LKP, commits to no
//     inverses. There the multiplicities are committed as field elements and the circuit adds
//     fractions. Here they are committed as bits and the trees multiply.
//   * Source for a key vector that is never sent, and a claim on it that the caller turns into a
//     claim on what it has sent: Diamond, Posen, "Succinct Arguments over Towers of Binary Fields",
//     Cryptology ePrint Archive 2023/1784, section 4.1 ("Definitions and Notions": virtual
//     polynomials) and section 4.4 ("Binary-Field Lasso": a lookup on virtual polynomials).
//   * Step 6 is a zero-check of b^2 - b. Source: Chen, Bünz, Boneh, Zhang, "HyperPlonk: Plonk with
//     Linear-Time Prover and High-Degree Custom Gates", Cryptology ePrint Archive 2022/1355
//     (version of January 9, 2023), section 3.2. That section takes the reduction of a zero-check
//     to a sumcheck from Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted
//     setup", CRYPTO 2020.
//   * A related technique for the tie: Setty, Lee, "Quarks: Quadruple-efficient transparent
//     zkSNARKs", Cryptology ePrint Archive 2020/1275 (last revised October 24, 2020), section 5
//     (Lemma 5.1) and section 6. There the prover commits to partial products of a grand product.
//     Here the trees are proved level by level throughout (steps 3 and 4), and no partial product
//     is committed: the P_j, which combine the trees' roots, are sent under mask entries.

// Padding. Entries of K may be declared as padding: such an entry holds a row of the table, named
// by its number, and is not counted in the multiplicities; a is the product of alpha - (that row)
// over all padding entries. LKP has accepted a key in no row, in its own test, when the count of
// padding entries was a separate public input and the caller kept the list of entries: the list
// had one entry fewer than the count. Hence two safeguards:
//   * The padding is written once in the public inputs, as key numbers (indices into K): runs
//     (LkpPadding) and whole key blocks (LkpKeyBlock). lkp_padding_count() counts the padding
//     entries from these, and LKP itself takes them out of the claim that it returns (LkpKeys).
//   * A padding value is a row of the table. Argued, not proved:
//     the right side of the idea's equation is then a product of factors
//     alpha - (a row), whatever the number of padding entries, so a false number can make an
//     honest proof fail, or force more keys to be that row, but lets in no key that is in no row.
//     What a test shows (one that is not part of this repository): with one surplus padding
//     entry injected in prover and verifier, a key in no row is accepted if the entry's value is
//     free (the key itself), and rejected if it is a row.
// A second test that is not part of this repository (`LKP_TWO_PADDINGS`). From its head comment:
//   The claim on the keys follows a second description of the padding, which differs from the one
//   that gives the tie's factor and the number of keys, so that one real key is not read. LKP has
//   accepted such a key in two cases where its public inputs stated one fact twice (the number of
//   key variables beside the list of parts; a padding's function beside its indices' key block) and
//   the caller made the claim.
//
// One run of padding among the keys of a function (a table of its own and the keys to be found in
// it; LkpPublic, `shifted`): the key numbers first + 2^log_step * i, i < length, of part `part`
// hold row `row` of the table (after the shift). With a tree's number in a key number's low g bits
// (LkpKeyBlock) a run stays in one tree: log_step >= g.
struct LkpPadding {
  unsigned part = 0;
  u64 first = 0;
  unsigned log_step = 0;
  u64 length = 0;
  u64 row = 0;
};
// The key number is y + 2^g x, y < 2^g. The keys that share y form tree number y, a subtree of the
// key side's tree. A function has the trees first .. first + 2^log_trees - 1 (first a multiple of
// 2^log_trees). Its own key vector has the index (y - first) + 2^log_trees x.
struct LkpKeyBlock {
  u64 first = 0;
  unsigned log_trees = 0;
  int function = -1;    // -1: trees of padding only,
  u64 padding_row = 0;  //     whose entries all hold this row of the table (after the shift)
};
// One function's table inside the table. With LkpPublic::h = 0: the rows first .. first +
// 2^log_rows - 1 (first a multiple of 2^log_rows). With h > 0 the low h bits of a row's number are
// its table part's number (a fast prover stores the rows in this order): table part `first` has the
// rows first + 2^h t, t < 2^(d - h), and row first + 2^h t is row t mod 2^log_rows of the
// function's table, which is thus repeated.
struct LkpTablePart {
  u64 first = 0;
  unsigned log_rows = 0;
  unsigned function = 0;
  std::vector<E> rows;  // written out; empty: the range table 0, 1, .., 2^log_rows - 1
};

// Everything public about one lookup, as plain data; prover and verifier take it from here only.
// It must be fixed before the challenges: lkp_prove() and lkp_verify() take it as a
// Public<LkpPublic>, a value that is in the first query of this hash chain (the Fiat-Shamir
// transcript; this query is hashed before the prover speaks) or of an earlier chain that this
// chain follows, or is computed from such values (blocks/chain/public.h). A prover that can still
// choose a row of the table when it knows alpha picks it so that the two products agree, for keys
// that are in no row.
//
// `live`: fewer planes than plane slots. The number of bits that a multiplicity needs is seldom a
// power of two. Of the w = 2^mu plane slots the first `live` are real; the others are public zeros:
// nothing is sent for them, their roots G_j are the public 1, the tie has live - 1 steps, and the
// conditions read 2^v < 2^live, 2^d (2^live - 1) < p.
//
// `parts`: keys in parts, for a prover that cannot hold all keys at once. The keys come as k
// vectors of 2^(v_i) entries, each with its own tree, rounds and LkpKeys, proved one after the
// other under one alpha, one table side and one vector of multiplicities. Z is then the product of
// the parts' roots Z_i, by k - 1 more sumchecks without variables after the tie: Q_0 = Z,
// Q_i = Z_i Q_(i+1), Q_(k-1) = Z_(k-1). In errors and conditions the number of all keys
// (all_keys(), computed from `parts`) replaces 2^v.
//
// `shifted`: several functions in one table. Before alpha the verifier draws theta, function f's
// keys and rows are moved by f * theta, and the lookup runs on all keys and on the union of the
// moved tables. A key of function f that is in no row of its own table passes only if
// key + f theta = row + f2 theta for a row of another function f2: one value of theta for each such
// row. Error of the shift: (rows of the union) / |E| for the first such key, which is fixed before
// theta.
struct LkpPublic {
  std::vector<unsigned> parts;  // v_i: part i of the keys has 2^(v_i) entries; at least one part
  // The table has 2^d rows; each multiplicity has w = 2^mu bits, one in each plane slot.
  unsigned d = 0, mu = 0;
  unsigned live = 0;            // how many of the 2^mu plane slots are real; 0: all
  unsigned h = 0;        // low h bits of a row's number: its table part's number (LkpTablePart)
  unsigned g = 0;        // the low g bits of a key's number are its tree's number
  bool shifted = false;  // function f's keys and rows are moved by f * theta
  // All oracle messages from which the keys and the multiplicities are read. They must be fixed
  // before theta and alpha. LKP checks that the sample of each has been answered and that its own
  // linear claims read main rows of these messages only. It cannot check what is the caller's duty:
  // that the caller's claim on the keys reads no other message, and that a message whose sample was
  // answered in another chain has its root in this chain's first query.
  std::vector<OracleId> read_from;
  std::vector<LkpKeyBlock> keys;      // they tile the 2^g tree numbers
  std::vector<LkpTablePart> tables;   // they tile the 2^d rows (a table may repeat to fill them)
  std::vector<LkpPadding> padding;    // strictly increasing by (part, first); no two runs overlap
  // A sumcheck's rounds bind the variables from the highest down. In the last key level of every
  // part the variables listed here are bound first, in this order, and then the others from the
  // highest down: a fast prover needs an order of its own. The order is part of the proof format,
  // because the challenges are drawn in the order of the rounds. A point's coordinates are indexed
  // by variable, whatever the order of the rounds.
  std::vector<unsigned> last_key_level_order;

  unsigned planes() const { return live ? live : 1u << mu; }
  const std::vector<unsigned> &part_bits() const { return parts; }
  u64 all_keys() const {  // after lkp_public_check: no wrap-around
    u64 n = 0;
    for (unsigned x : part_bits()) n += u64(1) << x;
    return n;
  }
};

// The bytes by which the first query fixes these structs (blocks/chain/public.h): every field, in
// the order of its declaration.
inline void public_bytes(Writer &w, const LkpPadding &p) {
  public_bytes(w, p.part), public_bytes(w, p.first), public_bytes(w, p.log_step), public_bytes(w, p.length), public_bytes(w, p.row);
}
inline void public_bytes(Writer &w, const LkpKeyBlock &b) {
  public_bytes(w, b.first), public_bytes(w, b.log_trees), public_bytes(w, b.function), public_bytes(w, b.padding_row);
}
inline void public_bytes(Writer &w, const LkpTablePart &t) {
  public_bytes(w, t.first), public_bytes(w, t.log_rows), public_bytes(w, t.function), public_bytes(w, t.rows);
}
inline void public_bytes(Writer &w, const LkpPublic &p) {
  public_bytes(w, p.parts), public_bytes(w, p.d), public_bytes(w, p.mu), public_bytes(w, p.live), public_bytes(w, p.h), public_bytes(w, p.g), public_bytes(w, p.shifted);
  public_bytes(w, p.read_from);
  public_bytes(w, p.keys), public_bytes(w, p.tables), public_bytes(w, p.padding), public_bytes(w, p.last_key_level_order);
}

// What lkp_run() works with: lkp_shape() makes it from an LkpPublic and theta.
struct LkpShape {
  unsigned d = 0, mu = 0, planes = 0;
  std::vector<unsigned> part_bits;
  u64 all_keys = 0;
  std::vector<std::pair<u64, E>> padding;  // (how many entries, the value that they hold)
  // The extension of the table at a point of d coordinates.
  std::function<E(const std::vector<E> &point)> table_at;
  std::vector<unsigned> last_key_level_order;  // as in LkpPublic
};

// True if the whole lookup, a part of the keys or a function's block of trees is padding only.
// S has passed lkp_fault()'s checks on blocks and runs: no two runs overlap, so their lengths add
// up to the number of padding entries.
inline bool lkp_only_padding(const LkpPublic &S) {
  bool a_function = false;
  for (const LkpKeyBlock &b : S.keys) a_function = a_function || b.function >= 0;
  if (!a_function) return true;
  for (size_t part = 0; part < S.part_bits().size(); part++) {
    u64 padded = 0;
    for (const LkpPadding &p : S.padding) padded += p.part == part ? p.length : 0;
    if (padded >> S.part_bits()[part]) return true;
  }
  if (S.parts.size() == 1)  // (tree numbers and key blocks are used with one part only)
    for (const LkpKeyBlock &b : S.keys) {
      if (b.function < 0) continue;
      u64 padded = 0;
      for (const LkpPadding &p : S.padding) padded += (p.first & ((u64(1) << S.g) - 1)) >> b.log_trees == b.first >> b.log_trees ? p.length : 0;
      if (padded >> (S.part_bits()[0] - S.g + b.log_trees)) return true;
    }
  return false;
}
// Whether the public inputs are well-formed and consistent: nullptr, or the fault in words.
inline const char *lkp_fault(const LkpPublic &S) {
  if (S.read_from.empty()) return "LKP: it is not said from which oracle messages keys and multiplicities are read";
  if (S.parts.empty()) return "LKP: no part of keys";
  if (!(S.mu >= 1 && S.mu <= 6 && S.live < (1u << S.mu) && S.planes() >= 2))  // «all are live» has one spelling: 0
    return "LKP: the plane slots are not 2, 4, .. or 64, or fewer than two of them are live, or as many as there are or more (all: 0)";
  u64 keys = 0;  // of all parts; below 2^planes <= 2^63 at every step, so the sum cannot wrap
  if (!(S.planes() < 64)) return "LKP: the sum of all multiplicities could wrap around in the field";
  for (unsigned x : S.part_bits()) {
    if (!(x >= 1 && x < S.planes() && keys + (u64(1) << x) < u64(1) << S.planes())) return "LKP: no key in a part, or the multiplicities have too few bits for the number of keys";
    keys += u64(1) << x;
  }
  if (!(S.d < F_BITS && S.d + S.planes() < F_BITS)) return "LKP: the sum of all multiplicities could wrap around in the field";
  if (S.parts.size() > 1) if (!(S.g == 0 && !S.shifted)) return "LKP: keys in several parts are served for one function and without trees' numbers only";
  if (!(S.g <= S.part_bits()[0])) return "LKP: a tree's number has more bits than a key's";
  const unsigned v = S.part_bits()[0];  // tree numbers and key blocks are used with one part only
  unsigned least = v;                   // the smallest v_i
  for (unsigned x : S.part_bits()) least = std::min(least, x);
  // Aligned blocks whose sizes are powers of two: two overlap only if one contains the other;
  // blocks that do not overlap and whose sizes add up to the whole tile it.
  auto meet = [](u64 a, unsigned a_bits, u64 b, unsigned b_bits) { return a >> std::max(a_bits, b_bits) == b >> std::max(a_bits, b_bits); };
  u64 trees = 0, rows = 0;
  for (size_t i = 0; i < S.keys.size(); i++) {
    const LkpKeyBlock &b = S.keys[i];
    if (!(b.log_trees <= S.g && (b.first >> b.log_trees << b.log_trees) == b.first && (b.first >> S.g) == 0)) return "LKP: a function's trees are no aligned block of tree numbers";
    for (size_t j = 0; j < i; j++) if (!(!meet(b.first, b.log_trees, S.keys[j].first, S.keys[j].log_trees))) return "LKP: two blocks of tree numbers overlap";
    trees += u64(1) << b.log_trees;
    if (b.function < 0 && b.padding_row >> S.d) return "LKP: a padding row is no row of the table";
  }
  if (!(trees == u64(1) << S.g)) return "LKP: the blocks do not cover all tree numbers";
  if (!S.shifted) {  // unshifted, a key of one function could be found in another's table
    for (const LkpKeyBlock &b : S.keys) if (!(b.function <= 0)) return "LKP: several functions share a table without the shift";
    for (const LkpTablePart &t : S.tables) if (!(t.function == 0)) return "LKP: several functions share a table without the shift";
  }
  if (!(S.h <= S.d)) return "LKP: a table part's number has more bits than a row's";
  for (size_t i = 0; i < S.tables.size(); i++) {
    const LkpTablePart &t = S.tables[i];
    if (S.h) {  // one part for each number, in the order of the numbers
      if (!(t.first == i && t.log_rows <= S.d - S.h)) return "LKP: the table parts are not numbered 0, 1, .., or a part's table has more rows than a part";
      if (!(t.rows.empty() || t.rows.size() == size_t(1) << t.log_rows)) return "LKP: a written-out table has not the announced number of rows";
      rows += u64(1) << (S.d - S.h);
      continue;
    }
    if (!(t.log_rows <= S.d && (t.first >> t.log_rows << t.log_rows) == t.first && (t.first >> S.d) == 0)) return "LKP: a table is no aligned block of the rows";
    if (!(t.rows.empty() || t.rows.size() == size_t(1) << t.log_rows)) return "LKP: a written-out table has not the announced number of rows";
    for (size_t j = 0; j < i; j++) if (!(!meet(t.first, t.log_rows, S.tables[j].first, S.tables[j].log_rows))) return "LKP: two tables overlap in the rows";
    rows += u64(1) << t.log_rows;
  }
  if (!(rows == u64(1) << S.d)) return "LKP: the tables do not cover all rows";
  if (S.padding.size() > 4096) return "LKP: more than 4096 runs of padding";
  for (size_t i = 0; i < S.padding.size(); i++) {
    const LkpPadding &p = S.padding[i];
    if (p.part >= S.part_bits().size()) return "LKP: a run of padding lies in a part that is not there";
    const unsigned bits = S.part_bits()[p.part];
    if (p.length < 1 || p.log_step >= bits || p.first >> bits || p.length > u64(1) << (bits - p.log_step) || (p.first + ((p.length - 1) << p.log_step)) >> bits)
      return "LKP: a run of padding is empty or does not lie in its part";
    if (p.row >> S.d) return "LKP: a padding row is no row of the table";
    if (S.g && p.log_step < S.g) return "LKP: a run of padding leaves its tree";
    for (const LkpKeyBlock &b : S.keys)
      if ((p.first & ((u64(1) << S.g) - 1)) >> b.log_trees == b.first >> b.log_trees && b.function < 0) return "LKP: a run of padding lies in a tree of padding alone";
    if (i && !(S.padding[i - 1].part < p.part || (S.padding[i - 1].part == p.part && S.padding[i - 1].first < p.first))) return "LKP: the runs of padding are not in increasing order";
    for (size_t j = 0; j < i; j++) {  // two runs with steps that are powers of two
      const LkpPadding &fine = S.padding[j].log_step <= p.log_step ? S.padding[j] : p, &coarse = S.padding[j].log_step <= p.log_step ? p : S.padding[j];
      if (fine.part != coarse.part || ((fine.first ^ coarse.first) & ((u64(1) << fine.log_step) - 1))) continue;  // another part, or another residue modulo the finer step
      const u64 low = fine.first, high = fine.first + ((fine.length - 1) << fine.log_step);
      const u64 skip = coarse.first >= low ? 0 : (low - coarse.first + (u64(1) << coarse.log_step) - 1) >> coarse.log_step;  // index in the coarse run of its first key number >= low
      if (skip < coarse.length && coarse.first + (skip << coarse.log_step) <= high) return "LKP: two runs of padding meet";
    }
  }
  // A padding entry is not read: if every entry of the lookup, of a part or of a function's trees
  // were padding, an accepted proof would say nothing about what was sent there. (Whether the right
  // entries are padding no check of form can tell: a caller derives them.)
  if (lkp_only_padding(S)) return "LKP: no key is left that is no padding: in the look-up, in a part, or in a function's trees";
  if (!S.last_key_level_order.empty()) {
    std::vector<bool> seen(least - 1, false);
    for (unsigned j : S.last_key_level_order) {
      if (!(j < least - 1 && !seen[j])) return "LKP: the last key level's order names a variable twice, or one that a part has not";
      seen[j] = true;
    }
  }
  return nullptr;
}
// blocks/chain/public.h calls public_fault() on each value that it puts into a first query; a fault
// ends the process.
inline const char *public_fault(const LkpPublic &S) { return lkp_fault(S); }
inline const char *public_fault(const LkpPadding &) { return nullptr; }  // checked in `lkp_fault`
inline const char *public_fault(const LkpKeyBlock &) { return nullptr; }
inline const char *public_fault(const LkpTablePart &) { return nullptr; }
// lkp_fault(), and the check that needs the statement. A fault ends the process.
inline void lkp_public_check(const Statement &st, const LkpPublic &S) {
  const char *fault = lkp_fault(S);
  BLK_REQUIRE(!fault, fault);
  for (OracleId o : S.read_from) BLK_REQUIRE(st.oracle(o).sampled, "LKP: an oracle message that keys or multiplicities are read from has no answered sample");
}

// The key block whose trees include this key number's tree.
inline const LkpKeyBlock &lkp_block_of(const LkpPublic &S, u64 key_number) {
  const u64 y = key_number & ((u64(1) << S.g) - 1);
  for (const LkpKeyBlock &b : S.keys)
    if (y >> b.log_trees == b.first >> b.log_trees) return b;
  BLK_REQUIRE(false, "LKP: a tree number that belongs to no block");
  return S.keys[0];
}
// Row t of the table, after the shift.
inline E lkp_row(const LkpPublic &S, const E &theta, u64 t) {
  for (const LkpTablePart &p : S.tables) {
    if (S.h ? (t & ((u64(1) << S.h) - 1)) != p.first : t >> p.log_rows != p.first >> p.log_rows) continue;
    const u64 own = (S.h ? t >> S.h : t) & ((u64(1) << p.log_rows) - 1);  // the table's own row; with h > 0 the table is repeated
    return (p.rows.empty() ? E::from(from_u64(own)) : p.rows[own]) + E::from(from_u64(p.function)) * theta;
  }
  BLK_REQUIRE(false, "LKP: a row that belongs to no part of the table");
  return E::zero();
}
// Whether key number i of a part is in a run of LkpPublic::padding; key blocks are not looked at.
inline bool lkp_is_padding(const LkpPublic &S, unsigned part, u64 i) {
  for (const LkpPadding &p : S.padding)
    if (p.part == part && i >= p.first && ((i - p.first) & ((u64(1) << p.log_step) - 1)) == 0 && (i - p.first) >> p.log_step < p.length) return true;
  return false;
}
// The number of all padding entries, in runs and in key blocks.
inline u64 lkp_padding_count(const LkpPublic &S) {
  u64 n = 0;
  for (const LkpPadding &p : S.padding) n += p.length;
  for (const LkpKeyBlock &b : S.keys)
    if (b.function < 0) n += u64(1) << (S.part_bits()[0] - S.g + b.log_trees);
  return n;
}
// The key numbers i of a part with (i & mask) == value.
struct LkpCube {
  u64 mask = 0, value = 0;
};
// A run as disjoint cubes: at most two for each bit of a key number.
inline std::vector<LkpCube> lkp_cubes(const LkpPadding &p, unsigned bits) {
  std::vector<LkpCube> cubes;
  const u64 all = (u64(1) << bits) - 1, low = (u64(1) << p.log_step) - 1;
  for (u64 x = p.first >> p.log_step, left = p.length; left;) {
    unsigned b = 0;  // the largest aligned block that begins at x and fits
    while (b < 62 && (x & ((u64(2) << b) - 1)) == 0 && (u64(2) << b) <= left) b++;
    cubes.push_back({all & ~(((u64(1) << b) - 1) << p.log_step), (p.first & low) | (x << p.log_step)});
    x += u64(1) << b, left -= u64(1) << b;
  }
  return cubes;
}
// The trees of a key block as a cube.
inline LkpCube lkp_cube(const LkpPublic &S, const LkpKeyBlock &b) { return {((u64(1) << S.g) - 1) & ~((u64(1) << b.log_trees) - 1), b.first}; }
// lkp_cube_weight(): the sum of eq(z, i) over a cube. lkp_cube_tensor(): scalar * eq(z, .) on the
// cube, 0 elsewhere, as a product form (`Tensor`, tensor.h).
inline E lkp_cube_weight(const LkpCube &c, const std::vector<E> &z) {
  E w = E::one();
  for (unsigned t = 0; t < z.size(); t++)
    if ((c.mask >> t) & 1) w = w * (((c.value >> t) & 1) ? z[t] : E::one() - z[t]);
  return w;
}
inline Tensor lkp_cube_tensor(const LkpCube &c, const std::vector<E> &z, const E &scalar) {
  Tensor v;
  v.scalar = scalar;
  for (unsigned t = 0; t < z.size(); t++) {
    const bool picked = (c.mask >> t) & 1, one = (c.value >> t) & 1;
    v.push(picked && one ? E::zero() : E::one() - z[t], picked && !one ? E::zero() : z[t]);
  }
  return v;
}
// All rows of the table, after the shift.
inline std::vector<E> lkp_rows(const LkpPublic &S, const E &theta) {
  std::vector<E> rows(size_t(1) << S.d);
  for (const LkpTablePart &t : S.tables)
    for (u64 i = 0; i < u64(1) << (S.h ? S.d - S.h : t.log_rows); i++) {
      const u64 own = i & ((u64(1) << t.log_rows) - 1);  // the table's own row; with h > 0 the table is repeated
      rows[S.h ? t.first + (i << S.h) : t.first + i] = (t.rows.empty() ? E::from(from_u64(own)) : t.rows[own]) + E::from(from_u64(t.function)) * theta;
    }
  return rows;
}
// The table's multilinear extension at a point: each table part's own extension (a range table's in
// closed form) times the factor that selects that part: by its block of rows or, with h > 0, by
// its number in the low h coordinates. A repeated table's extension does not depend on the
// coordinates above its own.
inline E lkp_table_at(const LkpPublic &S, const E &theta, const std::vector<E> &point) {
  E sum = E::zero();
  for (const LkpTablePart &t : S.tables) {
    E weight = E::one(), own = E::zero();
    for (unsigned j = S.h ? 0 : t.log_rows; j < (S.h ? S.h : S.d); j++) weight = weight * (((t.first >> j) & 1) ? point[j] : E::one() - point[j]);
    const E *z = point.data() + S.h;  // the coordinates of the table's own row
    if (t.rows.empty()) {
      E two_to = E::one();
      for (unsigned j = 0; j < t.log_rows; j++, two_to = two_to + two_to) own += two_to * z[j];
    } else {
      std::vector<E> fold = t.rows;
      for (unsigned j = 0; j < t.log_rows; j++) {
        for (size_t i = 0; i < fold.size() / 2; i++) fold[i] = fold[2 * i] + z[j] * (fold[2 * i + 1] - fold[2 * i]);
        fold.resize(fold.size() / 2);
      }
      own = fold[0];
    }
    sum += weight * (own + E::from(from_u64(t.function)) * theta);
  }
  return sum;
}
inline LkpShape lkp_shape(const LkpPublic &fixed, const E &theta) {
  LkpShape S;
  S.d = fixed.d, S.mu = fixed.mu, S.planes = fixed.planes(), S.part_bits = fixed.part_bits(), S.all_keys = fixed.all_keys();
  S.last_key_level_order = fixed.last_key_level_order;
  for (const LkpPadding &p : fixed.padding) S.padding.push_back({p.length, lkp_row(fixed, theta, p.row)});
  for (const LkpKeyBlock &b : fixed.keys)
    if (b.function < 0) S.padding.push_back({u64(1) << (S.part_bits[0] - fixed.g + b.log_trees), lkp_row(fixed, theta, b.padding_row)});
  S.table_at = [fixed, theta](const std::vector<E> &point) { return lkp_table_at(fixed, theta, point); };
  return S;
}

// From the highest variable down.
inline std::vector<unsigned> lkp_default_order(unsigned variables) {
  std::vector<unsigned> order;
  for (unsigned j = variables; j-- > 0;) order.push_back(j);
  return order;
}
// The named variables first, the others from the highest down.
inline std::vector<unsigned> lkp_order_beginning(const std::vector<unsigned> &first, unsigned variables) {
  std::vector<unsigned> order = first;
  for (unsigned j = variables; j-- > 0;)
    if (std::find(first.begin(), first.end(), j) == first.end()) order.push_back(j);
  return order;
}
// An order of rounds for a level whose index has a tree's number in its low g bits. Of the bits
// above these: 2 1 0, then 5 4 3, then 8 7 6, then the others from the highest down; last the low
// g bits from the highest down. Needs at least g + 9 variables. Only tests call it.
inline std::vector<unsigned> lkp_range_order(unsigned variables, unsigned g) {
  BLK_REQUIRE(variables >= g + 9, "LKP: the range order needs nine variables above the tree's number");
  std::vector<unsigned> order;
  for (unsigned three = 0; three < 9; three += 3)
    for (unsigned i = 3; i-- > 0;) order.push_back(g + three + i);
  for (unsigned j = variables; j-- > g + 9;) order.push_back(j);
  for (unsigned j = g; j-- > 0;) order.push_back(j);
  return order;
}
// The table with its index bits moved: bit j of the new index is bit order[j] of the old one.
inline std::vector<E> lkp_in_order(const std::vector<E> &table, const std::vector<unsigned> &order) {
  std::vector<E> moved(table.size());
  for (size_t i = 0; i < table.size(); i++) {
    size_t old = 0;
    for (size_t j = 0; j < order.size(); j++) old |= ((i >> j) & 1) << order[j];
    moved[i] = table[old];
  }
  return moved;
}
// A point given round by round, with each coordinate put back at its variable's place.
inline std::vector<E> lkp_by_variable(const std::vector<E> &by_round, const std::vector<unsigned> &order) {
  std::vector<E> point(by_round.size());
  for (size_t j = 0; j < order.size(); j++) point[order[j]] = by_round[j];
  return point;
}

// Steps 3, 4, 5 and 6 of the protocol.
enum class LkpStep { KEYS, TABLE, LEAVES, BITS };

// The prover's expensive part (`heavy` in identifiers): the trees of the key side and the table
// side, and the sumchecks over them without masks (PSC adds the masks). Without a shift, begin()
// gets theta = 0.
class LkpProver {
 public:
  virtual ~LkpProver() = default;
  virtual void begin(const E &theta, const E &alpha) = 0;  // applies the shift, builds the trees
  virtual std::vector<E> key_roots() = 0;    // one for each part of the keys
  virtual std::vector<E> plane_roots() = 0;  // G_0 .. G_(w-1)
  // The sumcheck of one level. `vars` are in the order of the rounds; round j binds variable
  // order[j]. KEYS (of part `part`) and TABLE, level k or l: the maps are the two halves of level
  // k + 1 or l + 1. LEAVES and BITS: the one map is b.
  virtual std::unique_ptr<SumcheckProver<E>> sumcheck(LkpStep step, unsigned part, unsigned level, const Summand<E> &summand,
                                                      const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) = 0;
};

// Adds a product-form vector on the multiplicities b (mu + d variables, the plane in the low ones)
// to a linear claim. The caller's callback must read the live planes only (lkp_live_blocks() cuts
// them into aligned blocks); LKP does not check this.
using LkpPlanesAt = std::function<void(Row &row, Tensor v)>;

// What LKP leaves to its caller, one for each part of the keys:
//     sum_i weights[i] * key[i] + <masks> = target,    key[i] = key i of the part before the shift.
// weights[i] = eq(point, i) at a real key and 0 at every padding entry. LKP has moved the padding's
// share (eq(point, i) times the row that the entry holds) and the shift's share (theta * f * the
// weight of function f's real keys) into `target`, from LkpPublic only. The caller applies the
// weights to what it has sent and adds nothing public.
struct LkpKeys {
  std::vector<E> point;
  Form weights;  // on the part's key numbers: eq(point, .) less one product form per padding cube
  std::vector<MaskTerm<E>> masks;
  E target = E::zero();
};
// A lookup's LkpKeys. Unless the caller makes linear claims of them, the lookup says nothing about
// what was sent, and no honest run would show the omission. So every claim must be taken: the
// destructor ends the process if one is left, unless the channel (`Rounds`, blocks/chain/rounds.h)
// has rejected the proof meanwhile. What the caller does with a taken claim is not checked.
class LkpKeyClaim {
 public:
  LkpKeyClaim() = default;
  LkpKeyClaim(const LkpKeyClaim &) = delete;
  LkpKeyClaim &operator=(const LkpKeyClaim &) = delete;
  ~LkpKeyClaim() { BLK_REQUIRE(left_ == 0 || !channel_->ok(), "LKP: the claim on the key vector was dropped: the caller has made no row of it"); }
  void put(LkpKeys &&claim, const Rounds &channel) {  // the channel must outlive this
    claims_.push_back(std::move(claim)), full_.push_back(true), left_++, channel_ = &channel;
  }
  size_t parts() const { return claims_.size(); }
  LkpKeys take(size_t part = 0) {
    BLK_REQUIRE(part < claims_.size() && full_[part], "LKP: there is no such claim on a key vector, or it has been taken");
    full_[part] = false, left_--;
    return std::move(claims_[part]);
  }

 private:
  std::vector<LkpKeys> claims_;
  std::vector<bool> full_;
  size_t left_ = 0;
  const Rounds *channel_ = nullptr;
};
// What lkp_prove() and lkp_verify() fill in for the caller.
struct LkpOutput {
  E theta = E::zero(), alpha = E::zero();
  LkpKeyClaim keys;
  std::vector<Row> rows;  // copies of the claims that LKP added to the group, for tests and reports
};

// ---- the summands ----
// Summand (summand.h): the end values of `shown` maps are sent; `kept` maps stay as terms of the
// sumcheck's linear claim.
//
// y_0 y_1, for steps 1b, 3 and 4.
inline Summand<E> lkp_summand_product() {
  Summand<E> s;
  s.shown = 2, s.degree = 2;
  s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] * y[1]}; };
  return s;
}
// Step 1, on (Z, P_1, G_0) if `first`, else on (P_(j+1), G_j).
inline Summand<E> lkp_summand_tie(bool first, const E &a) {
  Summand<E> s;
  s.shown = first ? 3 : 2, s.degree = 3;
  if (first)
    s.at = [a](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] - a * y[1] * y[1] * y[2]}; };
  else
    s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] * y[0] * y[1]}; };
  return s;
}
// Step 5, affine in the kept map b. Summand::at gets its point in the order of the rounds. The
// table's variables are the d variables above the planes'.
inline Summand<E> lkp_summand_leaves(const LkpShape &S, const E &alpha, const std::vector<unsigned> &order) {
  Summand<E> s;
  s.kept = 1;
  const unsigned mu = S.mu;
  const std::function<E(const std::vector<E> &)> table_at = S.table_at;
  s.at = [alpha, order, mu, table_at](const std::vector<E> &by_round, const std::vector<E> &) {
    const std::vector<E> point = lkp_by_variable(by_round, order);
    return std::vector<E>{E::one(), alpha - E::one() - table_at(std::vector<E>(point.begin() + mu, point.end()))};
  };
  return s;
}
// Step 6.
inline Summand<E> lkp_summand_bits() {
  Summand<E> s;
  s.shown = 1, s.degree = 2;
  s.at = [](const std::vector<E> &, const std::vector<E> &y) { return std::vector<E>{y[0] * y[0] - y[0]}; };
  return s;
}

// lkp_masks(): the number of mask entries that a lookup uses, one for each scalar that it sends;
// term by term, psc_masks() (psc.h) of steps 1, 1b, 3, 4, 5, 6. All come from one oracle message's
// mask rows, except that, if the caller names a second message (`key_masks`), lkp_key_masks() of
// them come from there: the two inner masks (PSC's mask entries for its end values) of each part's
// last key level, under which the two halves of alpha - K are sent and which LINE joins into the
// claim on K. A caller whose key vector is linear in a committed table (WF) needs the claim on K to
// name entries of a message that only the table's own test reads.
inline size_t lkp_key_masks(const LkpPublic &S) { return 2 * S.part_bits().size(); }
inline size_t lkp_masks(const LkpPublic &S) {
  size_t n = 7 + 6 * (size_t(S.planes()) - 2);
  n += 5 * (S.part_bits().size() - 1);
  for (unsigned v : S.part_bits())
    for (unsigned k = 0; k < v; k++) n += 2 * k + 5;
  for (unsigned l = 0; l < S.d; l++) n += 2 * (S.mu + l) + 5;
  n += 1 + S.mu + 2 * S.d;            // the leaves: no round of PSC's variable Y, no end value
  n += 1 + 2 * (S.mu + S.d) + 2 + 1;  // the bits
  return n;
}

// ---- one implementation for both sides ----
// Side: LkpProverSide or LkpVerifierSide. Its tie(), split() and psc() each run one PSC; the
// verifier's return false if the channel has rejected the proof. note(): the verifier's
// intermediate values, for comparing a second verifier (context.h).
//
// Steps 3 and 4: takes `claim` from a tree's top level down to its leaves. Level l has `more` + l
// variables.
template <class Side>
bool lkp_level_chain(Side &side, LkpStep step, unsigned part, unsigned levels, unsigned more, const LkpShape &S, PointClaim<E> &claim,
                     const std::function<void(Row)> &leave) {
  const std::string name = step == LkpStep::TABLE ? "LKP table, level " : S.part_bits.size() > 1 ? "LKP keys, part " + std::to_string(part) + ", level " : "LKP keys, level ";
  const Summand<E> product = lkp_summand_product();
  for (unsigned l = 0; l < levels; l++) {
    const unsigned variables = more + l;
    std::vector<unsigned> order = lkp_default_order(variables);
    const bool last_of_keys = step == LkpStep::KEYS && l + 1 == levels;
    if (last_of_keys) order = lkp_order_beginning(S.last_key_level_order, variables);
    BLK_REQUIRE(order.size() == variables && claim.point.size() == variables, "LKP: a level's order or point has not one entry for each variable");
    SumClaim<E> sum;
    for (unsigned j = 0; j < variables; j++) sum.vars.push_back(Weight<E>::equality(claim.point[order[j]], 3));
    sum.masks = std::move(claim.masks), sum.target = claim.value;
    PscOutput<E> out;
    if (!side.psc(std::move(sum), product, step, part, l, order, last_of_keys, out)) return false;
    leave(psc_row_u(out, {}, name + std::to_string(l)));
    for (PointClaim<E> &e : out.ends) e.point = lkp_by_variable(e.point, order);
    claim = line<E>(side.ch(), std::move(out.ends[0]), std::move(out.ends[1]));
  }
  return true;
}

// The whole protocol; the numbered comments are its steps.
template <class Side>
bool lkp_run(Side &side, const LkpPublic &fixed, const LkpPlanesAt &planes, GroupId group, LkpOutput &out) {
  lkp_public_check(side.statement(), fixed);
  // Every linear claim of LKP's own goes into the group here, so that no caller can lose one.
  const std::function<void(Row)> leave = [&side, &fixed, &out, group](Row row) {
    auto declared = [&fixed](OracleId o) { return std::find(fixed.read_from.begin(), fixed.read_from.end(), o) != fixed.read_from.end(); };
    bool all = true;
    for (const Term &t : row.terms) all = all && (t.part != Part::MAIN || declared(t.oracle));
    for (const Unit &u : row.units) all = all && (u.part != Part::MAIN || declared(u.oracle));
    for (const Run &r : row.runs) all = all && (r.part != Part::MAIN || declared(r.oracle));
    for (const MainForm &f : row.forms) all = all && declared(f.oracle);
    BLK_REQUIRE(all, "LKP: a row reads the main table of an oracle message that is not among those declared as read");
    out.rows.push_back(row);
    side.statement().add_row(group, std::move(row));
  };
  out.theta = fixed.shifted ? side.ch().close_algebraic(R_LKP_SHIFT).draw() : E::zero();
  const LkpShape S = lkp_shape(fixed, out.theta);
  const size_t w = size_t(1) << S.mu;
  const E alpha = out.alpha = side.ch().close_algebraic(R_LKP).draw();
  side.begin(out.theta, alpha);
  E a = E::one();
  u64 padding = 0;
  for (const std::pair<u64, E> &p : S.padding) {
    E power = alpha - p.second;  // (alpha - value)^count by squaring
    for (u64 e = p.first; e; e >>= 1, power = power * power)
      if (e & 1) a = a * power;
    padding += p.first;
  }
  BLK_REQUIRE(padding == lkp_padding_count(fixed), "LKP: the padding's factor and its count differ");
  side.tie_factor(a);
  side.note("LKP: alpha", alpha);
  side.note("LKP: the public factor a", a);

  // 1: the tie
  PointClaim<E> on_root, on_power;
  std::vector<PointClaim<E>> on_plane;
  for (size_t j = 0; j + 1 < S.planes; j++) {
    SumClaim<E> sum;
    sum.target = E::zero();
    if (j) sum = SumClaim<E>::at_point(std::move(on_power), {});
    PscOutput<E> step;
    if (!side.tie(std::move(sum), lkp_summand_tie(j == 0, a), j, step)) return false;
    leave(psc_row_u(step, {}, "LKP tie, step " + std::to_string(j)));
    size_t at = 0;
    if (j == 0) on_root = std::move(step.ends[at++]);
    on_power = std::move(step.ends[at++]);
    on_plane.push_back(std::move(step.ends[at]));
  }
  on_plane.push_back(std::move(on_power));  // the last power is the last live plane's root

  // 1b: from the product of the parts' roots to each root
  std::vector<PointClaim<E>> on_roots;
  for (size_t i = 0; i + 1 < S.part_bits.size(); i++) {
    PscOutput<E> step;
    if (!side.split(SumClaim<E>::at_point(std::move(on_root), {}), lkp_summand_product(), i, step)) return false;
    leave(psc_row_u(step, {}, "LKP parts, step " + std::to_string(i)));
    on_roots.push_back(std::move(step.ends[0]));
    on_root = std::move(step.ends[1]);
  }
  on_roots.push_back(std::move(on_root));

  // 2: one claim on the table side's top level
  PointClaim<E> on_table;
  {
    AlgebraicDraw from = side.ch().close_algebraic(R_LKP_PLANES);
    for (unsigned t = 0; t < S.mu; t++) on_table.point.push_back(from.draw());
  }
  on_table.value = E::zero();
  for (size_t j = 0; j < w; j++) {
    E weight = E::one();
    for (unsigned t = 0; t < S.mu; t++) weight = weight * (((j >> t) & 1) ? on_table.point[t] : E::one() - on_table.point[t]);
    if (j >= S.planes) {  // a slot of public zeros: its leaves are 1, and so is its root
      on_table.value += weight;
      continue;
    }
    for (MaskTerm<E> &m : on_plane[j].masks) on_table.masks.push_back({std::move(m.slot), weight * m.coeff});
    on_table.value += weight * on_plane[j].value;
  }
  side.note("LKP: the table side's first claim", on_table.value);

  // 3, 4: the trees of the key side and of the table side
  for (unsigned part = 0; part < S.part_bits.size(); part++)
    if (!lkp_level_chain(side, LkpStep::KEYS, part, S.part_bits[part], 0, S, on_roots[part], leave)) return false;
  if (!lkp_level_chain(side, LkpStep::TABLE, 0, S.d, S.mu, S, on_table, leave)) return false;

  // 5: the leaves of the table side
  const unsigned all = S.mu + S.d;
  const std::vector<unsigned> order = lkp_default_order(all);
  const MapAt bits_at = [&planes](Row &row, const std::vector<E> &point, const E &coeff) {
    planes(row, Tensor::eq(point.data(), (unsigned)point.size(), coeff));
  };
  {
    SumClaim<E> sum;
    for (unsigned j = 0; j < all; j++) sum.vars.push_back(Weight<E>::equality(on_table.point[order[j]], order[j] < S.mu ? 2 : 3));
    sum.masks = std::move(on_table.masks), sum.target = on_table.value;
    PscOutput<E> leaves;
    if (!side.psc(std::move(sum), lkp_summand_leaves(S, alpha, order), LkpStep::LEAVES, 0, 0, order, false, leaves)) return false;
    leaves.point = lkp_by_variable(leaves.point, order);
    side.note("LKP: the leaves' factor", alpha - E::one() - S.table_at(std::vector<E>(leaves.point.begin() + S.mu, leaves.point.end())));
    leave(psc_row_u(leaves, {bits_at}, "LKP leaves"));
  }

  // 6: the bits are bits
  {
    std::vector<E> point(all);
    {
      AlgebraicDraw from = side.ch().close_algebraic(R_ZERO);
      for (E &x : point) x = from.draw();
    }
    SumClaim<E> sum;
    for (unsigned j = 0; j < all; j++) sum.vars.push_back(Weight<E>::equality(point[order[j]], 3));
    sum.target = E::zero();
    PscOutput<E> bits;
    if (!side.psc(std::move(sum), lkp_summand_bits(), LkpStep::BITS, 0, 0, order, false, bits)) return false;
    leave(psc_row_u(bits, {}, "LKP bits"));
    bits.ends[0].point = lkp_by_variable(bits.ends[0].point, order);
    leave(point_row(std::move(bits.ends[0]), bits_at, "LKP bits at the point"));
  }

  // 7: the multiplicities sum to the number of keys that are not padding
  {
    Row sum;
    sum.name = "LKP number of keys";
    Tensor v;
    E two_to = E::one() + E::one();  // 2^(2^t)
    for (unsigned t = 0; t < S.mu; t++, two_to = two_to * two_to) v.push(E::one(), two_to);
    for (unsigned t = 0; t < S.d; t++) v.push(E::one(), E::one());
    planes(sum, std::move(v));
    sum.target = E::from(from_u64(S.all_keys - padding));
    leave(std::move(sum));
  }

  // Turns each part's claim on its leaves, alpha - (shifted key, or padding row), into a claim on
  // the real keys before the shift (LkpKeys).
  for (unsigned part = 0; part < on_roots.size(); part++) {
    PointClaim<E> &on_leaves = on_roots[part];
    LkpKeys k;
    k.point = std::move(on_leaves.point);
    for (MaskTerm<E> &m : on_leaves.masks) k.masks.push_back({std::move(m.slot), -m.coeff});
    k.target = alpha - on_leaves.value;
    k.weights.add(Tensor::eq(k.point.data(), (unsigned)k.point.size()));
    auto take_out = [&](const LkpCube &cube, u64 row) {  // a cube of padding entries holding `row`
      const E weight = lkp_cube_weight(cube, k.point);
      k.target = k.target - weight * lkp_row(fixed, out.theta, row);
      k.weights.add(lkp_cube_tensor(cube, k.point, -E::one()));
      return weight;
    };
    std::vector<E> real(fixed.keys.size(), E::zero());  // the weight of each key block's real keys
    for (size_t b = 0; b < fixed.keys.size(); b++) {
      if (fixed.keys[b].function < 0)
        take_out(lkp_cube(fixed, fixed.keys[b]), fixed.keys[b].padding_row);
      else
        real[b] = lkp_cube_weight(lkp_cube(fixed, fixed.keys[b]), k.point);
    }
    const std::vector<LkpPadding> &runs = fixed.padding;
    for (const LkpPadding &p : runs) {
      if (p.part != part) continue;
      const size_t b = size_t(&lkp_block_of(fixed, p.first) - fixed.keys.data());
      for (const LkpCube &cube : lkp_cubes(p, (unsigned)k.point.size())) real[b] = real[b] - take_out(cube, p.row);
    }
    for (size_t b = 0; b < fixed.keys.size(); b++)
      if (fixed.keys[b].function > 0) k.target = k.target - out.theta * E::from(from_u64((u64)fixed.keys[b].function)) * real[b];
    side.note("LKP: a part's key claim", k.target);
    out.keys.put(std::move(k), side.ch());
  }
  return true;
}

// ---- the two sides ----
// The n mask entries of one PSC (`MaskLedger`, mask.h, gives out each once): the hiding
// polynomial's first, then the inner masks. If `keys`, the two inner masks come from key_ledger.
inline std::vector<MaskSlot> lkp_fresh(MaskLedger &ledger, MaskLedger &key_ledger, size_t n, bool keys) {
  if (!keys) return ledger.take(n);
  std::vector<MaskSlot> fresh = ledger.take(n - 2);
  for (MaskSlot &s : key_ledger.take(2)) fresh.push_back(std::move(s));
  return fresh;
}
struct LkpProverSide {
  ProverContext &P;
  MaskLedger &ledger, &key_ledger;
  LkpProver &heavy;
  std::vector<E> power;  // P_j; `begin` fills all four
  std::vector<E> plane;  // G_j
  std::vector<E> root;   // Z_i, the parts' roots
  std::vector<E> rest;   // Q_i = Z_i Z_(i+1) ..

  E a = E::one();

  ProverRounds &ch() { return P.ch; }
  Statement &statement() { return P.st; }
  void tie_factor(const E &factor) {
    a = factor;
  }
  void note(const char *, const E &) {}
  MaskValues<E> values() {
    ProverContext *p = &P;
    return [p](u32 oracle, u32 index) { return p->behind[oracle].mask.entry(index); };
  }
  void begin(const E &theta, const E &alpha) {
    heavy.begin(theta, alpha);
    plane = heavy.plane_roots();  // all 2^mu slots; a slot of public zeros has G_j = 1 and P_j = 1
    power = plane;
    for (size_t j = plane.size() - 1; j-- > 0;) power[j] = power[j + 1] * power[j + 1] * plane[j];
    root = rest = heavy.key_roots();
    for (size_t i = root.size() - 1; i-- > 0;) rest[i] = root[i] * rest[i + 1];
  }
  bool tie(SumClaim<E> &&claim, const Summand<E> &S, size_t j, PscOutput<E> &out) {
    std::vector<std::vector<E>> maps;
    if (j == 0) maps.push_back({rest[0]});
    maps.push_back({power[j + 1]}), maps.push_back({plane[j]});
    SumcheckCpu<E> inner(S, claim.vars, std::move(maps));
    std::vector<MaskSlot> fresh = ledger.take(psc_masks(claim, S));
    out = psc_prove<E>(P.ch, std::move(claim), S, inner, std::move(fresh), values());
    return true;
  }
  bool split(SumClaim<E> &&claim, const Summand<E> &S, size_t i, PscOutput<E> &out) {
    SumcheckCpu<E> inner(S, claim.vars, {{root[i]}, {rest[i + 1]}});
    std::vector<MaskSlot> fresh = ledger.take(psc_masks(claim, S));
    out = psc_prove<E>(P.ch, std::move(claim), S, inner, std::move(fresh), values());
    return true;
  }
  bool psc(SumClaim<E> &&claim, const Summand<E> &S, LkpStep step, unsigned part, unsigned level, const std::vector<unsigned> &order, bool last_of_keys,
           PscOutput<E> &out) {
    const std::unique_ptr<SumcheckProver<E>> inner = heavy.sumcheck(step, part, level, S, claim.vars, order);
    std::vector<MaskSlot> fresh = lkp_fresh(ledger, key_ledger, psc_masks(claim, S), last_of_keys);
    out = psc_prove<E>(P.ch, std::move(claim), S, *inner, std::move(fresh), values());
    return true;
  }
};
struct LkpVerifierSide {
  VerifierContext &V;
  MaskLedger &ledger, &key_ledger;

  VerifierRounds &ch() { return V.ch; }
  Statement &statement() { return V.st; }
  void tie_factor(const E &) {}
  void begin(const E &, const E &) {}
  void note(const char *what, const E &value) { V.note(what, value); }  // for a second verifier
  bool tie(SumClaim<E> &&claim, const Summand<E> &S, size_t, PscOutput<E> &out) {
    std::vector<MaskSlot> fresh = ledger.take(psc_masks(claim, S));
    return psc_verify<E>(V.ch, std::move(claim), S, std::move(fresh), out);
  }
  bool split(SumClaim<E> &&claim, const Summand<E> &S, size_t, PscOutput<E> &out) {
    std::vector<MaskSlot> fresh = ledger.take(psc_masks(claim, S));
    return psc_verify<E>(V.ch, std::move(claim), S, std::move(fresh), out);
  }
  bool psc(SumClaim<E> &&claim, const Summand<E> &S, LkpStep, unsigned, unsigned, const std::vector<unsigned> &, bool last_of_keys, PscOutput<E> &out) {
    std::vector<MaskSlot> fresh = lkp_fresh(ledger, key_ledger, psc_masks(claim, S), last_of_keys);
    return psc_verify<E>(V.ch, std::move(claim), S, std::move(fresh), out);
  }
};

// The entry points.
//   masks      the oracle message whose mask rows give the lkp_masks(fixed) mask entries
//   group      where LKP adds its linear claims; `masks` and the messages of fixed.read_from must
//              belong to it already: a claim that names another message ends the process
//              (Statement::add_row(), relation.h)
//   key_masks  if given, the oracle message that gives lkp_key_masks(fixed) of these entries
//              (which ones: see lkp_key_masks()). It need not belong to the group: only the claims
//              on the keys name its entries.
// Both mask messages must be fixed before the first challenge under which one of their entries
// masks a value: the process ends unless the sample of each has been answered.
inline void lkp_mask_messages_check(const Statement &st, OracleId masks, OracleId key_masks) {
  BLK_REQUIRE(st.oracle(masks).sampled && (key_masks == NONE_ID || st.oracle(key_masks).sampled), "LKP: an oracle message that gives mask entries has no answered sample");
}
inline void lkp_prove(ProverContext &P, const Public<LkpPublic> &fixed, LkpProver &heavy, const LkpPlanesAt &planes, OracleId masks, GroupId group, LkpOutput &out,
                      OracleId key_masks = NONE_ID) {
  lkp_mask_messages_check(P.st, masks, key_masks);
  LkpProverSide side{P, P.st.masks(masks), P.st.masks(key_masks == NONE_ID ? masks : key_masks), heavy, {}, {}, {}, {}};
  lkp_run(side, P.ch.get(fixed), planes, group, out);
}
// False: the channel has rejected the proof. True alone decides nothing: LKP compares no value from
// the proof; its claims, in `group` and in out.keys, are still to be tested. The caller's contract:
// see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool lkp_verify(VerifierContext &V, const Public<LkpPublic> &fixed, const LkpPlanesAt &planes, OracleId masks, GroupId group, LkpOutput &out,
                                     OracleId key_masks = NONE_ID) {
  lkp_mask_messages_check(V.st, masks, key_masks);
  LkpVerifierSide side{V, V.st.masks(masks), V.st.masks(key_masks == NONE_ID ? masks : key_masks)};
  return lkp_run(side, V.ch.get(fixed), planes, group, out) && V.ch.ok();
}

// ---- LkpProver on the CPU, straight from the definitions ----
class LkpCpu : public LkpProver {
 public:
  // keys: before the shift, by key number, part after part; a padding entry's value is ignored.
  // bits: 2^(mu + d) entries, the plane in the low bits of the index; 0 in the planes that are not
  //       live (not checked).
  LkpCpu(const LkpPublic &fixed, std::vector<E> keys, std::vector<E> bits) : fixed_(fixed), keys_(std::move(keys)), bits_(std::move(bits)) {
    BLK_REQUIRE(keys_.size() == fixed.all_keys() && bits_.size() == size_t(1) << (fixed.mu + fixed.d), "LKP: keys or bits have not the announced number");
  }
  void begin(const E &theta, const E &alpha) override {
    key_levels_.clear();
    size_t first = 0;
    for (unsigned part = 0; part < fixed_.part_bits().size(); part++) {
      const unsigned v = fixed_.part_bits()[part];
      std::vector<E> leaves(size_t(1) << v);
      for (size_t i = 0; i < leaves.size(); i++) {
        const LkpKeyBlock &b = lkp_block_of(fixed_, i);
        leaves[i] = alpha - (b.function < 0 ? lkp_row(fixed_, theta, b.padding_row) : keys_[first + i] + E::from(from_u64((u64)b.function)) * theta);
      }
      for (const LkpPadding &p : fixed_.padding)
        if (p.part == part)
          for (u64 i = 0; i < p.length; i++) leaves[p.first + (i << p.log_step)] = alpha - lkp_row(fixed_, theta, p.row);
      key_levels_.push_back(tree(std::move(leaves), 1));
      first += size_t(1) << v;
    }
    const std::vector<E> table = lkp_rows(fixed_, theta);
    const size_t w = size_t(1) << fixed_.mu;
    std::vector<E> table_leaves(bits_.size());
    for (size_t i = 0; i < bits_.size(); i++) table_leaves[i] = E::one() + bits_[i] * (alpha - E::one() - table[i / w]);
    table_levels_ = tree(std::move(table_leaves), w);
  }
  std::vector<E> key_roots() override {
    std::vector<E> roots;
    for (const auto &levels : key_levels_) roots.push_back(levels[0][0]);
    return roots;
  }
  std::vector<E> plane_roots() override { return table_levels_[0]; }
  std::unique_ptr<SumcheckProver<E>> sumcheck(LkpStep step, unsigned part, unsigned level, const Summand<E> &summand, const std::vector<Weight<E>> &vars,
                                              const std::vector<unsigned> &order) override {
    std::vector<std::vector<E>> maps;
    if (step == LkpStep::KEYS || step == LkpStep::TABLE) {
      const std::vector<E> &below = (step == LkpStep::KEYS ? key_levels_[part] : table_levels_)[level + 1];
      const size_t half = below.size() / 2;
      maps.push_back(lkp_in_order(std::vector<E>(below.begin(), below.begin() + half), order));
      maps.push_back(lkp_in_order(std::vector<E>(below.begin() + half, below.end()), order));
    } else {
      maps.push_back(lkp_in_order(bits_, order));
    }
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(summand, vars, std::move(maps)));
  }

 private:
  // levels[l] from the top down; levels[0] has `top` entries.
  static std::vector<std::vector<E>> tree(std::vector<E> leaves, size_t top) {
    std::vector<std::vector<E>> up;
    up.push_back(std::move(leaves));
    while (up.back().size() > top) {
      const std::vector<E> &below = up.back();
      std::vector<E> level(below.size() / 2);
      for (size_t i = 0; i < level.size(); i++) level[i] = below[i] * below[i + level.size()];
      up.push_back(std::move(level));
    }
    return std::vector<std::vector<E>>(up.rbegin(), up.rend());
  }
  LkpPublic fixed_;
  std::vector<E> keys_, bits_;
  std::vector<std::vector<std::vector<E>>> key_levels_;  // by part
  std::vector<std::vector<E>> table_levels_;
};

// Writes the honest multiplicities, which do not depend on the shift, to `bits`, in LkpCpu's
// layout. A key (before the shift) is counted at the first row equal to it among its function's
// table parts, in the order of S.tables; a padding entry is not counted. keys: part after part,
// S.all_keys() of them (not checked). Returns false if a counted key is in no row of its function's
// table. left_out: further keys to leave uncounted; no caller in this repository passes any.
inline bool lkp_count(const LkpPublic &S, const std::vector<E> &keys, std::vector<E> &bits, const std::vector<bool> &left_out = {}) {
  std::vector<u64> count(size_t(1) << S.d, 0);
  bool all = true;
  unsigned part = 0;
  u64 part_first = 0;
  for (size_t i = 0; i < keys.size(); i++) {
    while (i - part_first >= u64(1) << S.part_bits()[part]) part_first += u64(1) << S.part_bits()[part], part++;
    const LkpKeyBlock &b = lkp_block_of(S, i - part_first);
    if (b.function < 0 || lkp_is_padding(S, part, i - part_first) || (!left_out.empty() && left_out[i])) continue;
    bool found = false;
    for (const LkpTablePart &t : S.tables) {
      if ((int)t.function != b.function) continue;
      for (u64 r = 0; r < u64(1) << t.log_rows && !found; r++)
        if ((t.rows.empty() ? E::from(from_u64(r)) : t.rows[r]) == keys[i]) count[S.h ? t.first + (r << S.h) : t.first + r]++, found = true;
    }
    all = all && found;
  }
  bits.assign(count.size() << S.mu, E::zero());
  for (size_t t = 0; t < count.size(); t++)
    for (size_t j = 0; j < size_t(1) << S.mu; j++)
      if ((count[t] >> j) & 1) bits[j + (t << S.mu)] = E::one();
  return all;
}

// The live planes 0 .. live - 1 as aligned blocks of plane numbers, the largest first: (first
// plane, log2 of the number of planes).
inline std::vector<std::pair<unsigned, unsigned>> lkp_live_blocks(unsigned live) {
  std::vector<std::pair<unsigned, unsigned>> blocks;
  for (unsigned first = 0; first < live;) {
    unsigned bits = 0;
    while (first + (2u << bits) <= live && (first & ((2u << bits) - 1)) == 0) bits++;
    blocks.push_back({first, bits});
    first += 1u << bits;
  }
  return blocks;
}
// Restricts a product-form vector v (mu plane variables, then others) to one such block: the
// block's own plane variables stay; each higher one is set to its bit of block.first, and its
// factor goes into the scalar.
inline Tensor lkp_on_block(const Tensor &v, unsigned mu, const std::pair<unsigned, unsigned> &block) {
  BLK_REQUIRE(!v.has_dense(), "LKP: a vector on the multiplicities is a product form");
  Tensor part;
  part.scalar = v.scalar;
  for (unsigned t = 0; t < v.vars(); t++) {
    if (t >= block.second && t < mu)
      part.scalar = part.scalar * (((block.first >> t) & 1) ? v.hi(t) : v.lo(t));
    else
      part.push(v.lo(t), v.hi(t));
  }
  return part;
}

// LkpKeys::weights restricted to one function's own key vector (its index: LkpKeyBlock): the
// tree-number bits above the block's own are set to those of the block's `first`, and their factors
// go into the scalars.
inline Form lkp_weights_of_block(const LkpPublic &S, const Form &weights, size_t block) {
  BLK_REQUIRE(block < S.keys.size() && S.keys[block].function >= 0 && weights.entries.empty(), "LKP: no such function's block, or weights that LKP has not made");
  const LkpKeyBlock &b = S.keys[block];
  Form own;
  for (const Tensor &v : weights.tensors) {
    BLK_REQUIRE(!v.has_dense() && v.vars() == S.part_bits()[0], "LKP: weights that LKP has not made");
    Tensor part;
    part.scalar = v.scalar;
    for (unsigned t = 0; t < v.vars(); t++) {
      if (t >= b.log_trees && t < S.g)
        part.scalar = part.scalar * (((b.first >> t) & 1) ? v.hi(t) : v.lo(t));
      else
        part.push(v.lo(t), v.hi(t));
    }
    own.add(std::move(part));
  }
  return own;
}

}  // namespace blk
