// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Here a sub-proof (the argument for one part of the forward pass) lists committed arrays ("trees":
// hz::TreeN, R rows by K columns) with their coefficients in its combined linear claim: tree X adds
// <M_X, X>, M_X[r][c] = sum_i cv_i[r] L_i[c]. add_eqs() (door.cuh) reads Tn, num, cv, L, c0, c1 and
// vecE. Nothing reads Ent's other fields or implements `Stream`; their comments describe a
// sumcheck that is not in this repository. Include transcript.cuh (for T) and trees.cuh first.
#pragma once
#include <vector>
#include <string>
namespace he { // a sub-proof's list of trees for its equation (the combined linear claim above)
/* Runs the first rounds of phase H, the rounds over the lh bits of hi (column number = hi Kpp + cm;
   letters as in `hz::Shape2::place`, trees.cuh), for a tree of one long row whose coefficients
   ("weights") and cells do not fit as full arrays (Ent::stream). It computes each round's partial
   sums in passes over the cells, forming the weights where they are used, then writes the pair of
   arrays folded at these rounds' challenges. The messages are the same as with full arrays. */
struct Stream {
  int rounds = 0; // how many rounds of phase H it takes: 1 <= rounds <= lh - 1
  int parts = 0;  // how many pairs of partial sums round() writes
  virtual ~Stream() {}
  /* Round t < rounds. On the GPU: re, this entry's t earlier challenges (re[0] bound the top bit of
     hi), and the output part, 2 x parts elements: pairs that sum to the entry's part of g(0) and of
     g's X^2 coefficient (g: round polynomial). It checks the launch status but does not wait. */
  virtual void round(int t, const T *re, T *part) = 0;
  /* After the last of these rounds. Writes all of A (the weights) and B (the cells), NH >> rounds rows of 2^w elements
     each: entry (h << w) + c is the sum over q < 2^rounds of eq(re, q) times entry (q (NH >> rounds) + h, c) of the
     full array (for B: the table's one row, without blinding coordinates, as NH rows of stride 2^w, 0 wherever no
     element is placed). */
  virtual void lay(const T *re, T *A, T *B) = 0;
};
// One tree's entry. num: add_eqs() names a tree of field elements "U<num + 1>". vecE: on the GPU,
// that tree's one row of K elements; null for 16-bit cells, whose address goes to plain_rows().
struct Ent {
  hz::TreeN *Tn = nullptr;
  u32 num = 0, tag = 0;
  hz::Tree *zf = nullptr;
  int b = 0, lr = 0, lh = 0, lc = 0, w = 0, x = 0, V = 0;
  size_t n1 = 0, Rp = 0; // nothing sets or reads the fields from zf to Rp
  std::vector<const T *> cv;
  std::vector<T *> L;
  // On the GPU: cv[i] has R elements, L[i] has K. cv, L, c0 and c1 need equal lengths; add_eqs()
  // does not check. L[i] counts as zero outside columns [c0[i], c1[i]). Public source: P. Sam
  // Johnson's lecture slides "Exploiting Structure" (NITK, course MA222) sum an inner product with a vector that is
  // known to be zero outside an index range over that range only, and call it an abbreviated inner product (there for
  // the product of two triangular matrices). Here each vector carries its own column range, and k_rowdot()
  // (trees.cuh) sums over it only.
  std::vector<size_t> c0, c1;
  std::vector<int> sv;
  // Finer than c0, c1: L[sv[s]] is zero outside its segments [s0[s], s1[s]). TreeN::rowdotS()
  // (trees.cuh) takes segments in this form. Only the attention sub-proof fills sv, s0 and s1.
  // Public source: G. Wang, J. Zeng, X. Xiao, S. Wu, J. Yang, L. Zheng, Z. Chen, J. Bian, D. Yu and H. Wang,
  // "FlashMask: Efficient and Rich Mask Extension of FlashAttention", arXiv 2410.01359, 2024, section 3 "Observation"
  // and section 4: https://arxiv.org/pdf/2410.01359. FlashMask describes the masked part of each column of an attention
  // score matrix by at most two row-index intervals; here the list holds the vectors' nonzero segments.
  std::vector<size_t> s0, s1;
  const T *vecE = nullptr;
  /* one_pair: after the rounds over rhi the entry keeps a single pair of arrays over (hi, cm, rlo),
     of 2^(lh + w) entries each, not one pair for each of its V vectors. The sums and messages are
     the same. It is meant for a tree with many vectors, each 0 outside a column range [c0, c1):
     L[i] is used only there, so vectors with disjoint ranges may share one buffer. */
  // Public source: Gebremedhin, Manne and Pothen ("What Color Is Your Jacobian?", SIAM Review 47, 2005, section 1.2)
  // pack the nonzeros of matrix columns that have no nonzero in a common row into one column of a compressed Jacobian,
  // in order to compute them together; here the vectors share one buffer to save memory.
  bool one_pair = false;
  // If set, it runs the entry's first stream->rounds rounds of phase H. The entry then has one
  // vector (V = 1), cv = {1}; L[0] is not read and may be null; A and B begin at NH >> rounds rows.
  Stream *stream = nullptr;
  size_t *dc01 = nullptr; // on the device: c0 of all vectors, then c1
  T *CV = nullptr, *W = nullptr, *A[2] = {nullptr, nullptr}, *B[2] = {nullptr, nullptr}, *e = nullptr, *om = nullptr,
    *DA[2] = {nullptr, nullptr}, *DB[2] = {nullptr, nullptr}, *ce = nullptr, *u = nullptr, *g = nullptr;
  const T **dL = nullptr;
  int last = 0;
  size_t lenH = 0;
  // The entry's arrays that hold values made from the tree's cells, with their lengths in bytes, to
  // be erased. Always empty: no function in this repository allocates, lists or erases them.
  std::vector<std::pair<void *, size_t>> hidden;
};
// A sub-proof fills the list at set-up, then calls add_eqs(), which copies the pointers and reads
// the trees' R, K, ec and nm: set up every Tn first. Later changes to the list have no effect.
struct Eqs {
  std::vector<Ent> list;
};
} // namespace he
