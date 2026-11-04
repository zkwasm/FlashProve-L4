// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Which rows of the weight matrix W each lookup call of the committer's range proof reads. The
// range proof (the block WF, blocks/iop/wf.h) is to show that every cell of W is a signed byte and
// that every weight stored as two digits is in [-4096, 4095]. A lookup call ("call" from here on)
// is to show that its keys are in a public table. A plan has N_F calls, each of 2^(lwr + lk) keys:
//   call f = j < NB, "B_j":  up to 2^lwr rows of W; a key is cell + 128; the table has 2^8 rows;
//   call f = NB + j, "W_j":  up to 2^lwr pairs of rows, the two digits lo and hi of a weight; a key
//                            is lo + 256 hi + 4,096; the table has 2^13 rows.
// The index of a key is pl 2^lk + c: pl < 2^lwr is its place, c < 2^lk its column of W.
// A call is one part (blk::WfPart) of one of WF's two lookups: the committer program `cp_commit`
// takes the calls B of a plan with lwr = L and the calls W of a plan with lwr = L - 1, where L is
// the line `ranges-part` of the parameter file (`policy.txt`; `policy` in error names); see
// once_make(), prover/hash_commitment/hf_commit_once.cuh.
//
// Everything here follows from lk, lwr, identity_bits and the layout of W. Nothing is secret. The
// file is C++ for the host; HfRangeView's members also compile for the GPU.
#pragma once
#include "common/field_host.h"
#include "hash_commitment/hf_range_format.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define HF_RANGE_BOTH __host__ __device__ __forceinline__
#else
#define HF_RANGE_BOTH inline
#endif

namespace hf {

// A unit of W's layout (the rows of one or two tensors) as pair_map() needs it: rows off ..
// off + rows - 1. It stores rows / digits rows of numbers, one row of W for each digit: digit q of
// row i of the numbers is in row off + q (rows / digits) + i.
struct HfRangeUnit {
  size_t off = 0, rows = 0;
  int digits = 2; // 2, or 4 in the embedding; `pair_map` rejects every other number
};
// The units of a LayW (prover/common/weight_layout.h) or of any type with its fields. The embedding
// (X = 12) must have four digits and every other unit two. A unit that does not gets 0 digits,
// which pair_map() rejects.
template <class Layout> std::vector<HfRangeUnit> range_units(const Layout &Y) {
  std::vector<HfRangeUnit> units;
  for (const auto &u : Y.U) units.push_back({u.off, u.rows, u.nlw == (u.X == 12 ? 4 : 2) ? u.nlw : 0});
  return units;
}

// m_e, the first row after all units.
inline size_t range_end(const std::vector<HfRangeUnit> &units) {
  size_t end = 0;
  for (const HfRangeUnit &u : units)
    if (u.off + u.rows > end) end = u.off + u.rows;
  return end;
}

// The plan's sizes and counts with the pair map (each pair's two rows of W): all that rows_at()
// needs. A kernel takes it by value, with ymap pointing to a copy of the map in GPU memory
// (HfRangePlan::view_with()).
struct HfRangeView {
  int lk = 0, lwr = 0;
  int n = 0;                        // lwr + lk: every call has 2^n keys
  int iota = 0;                     // a call has 2^iota identities; a key's identity is its column mod 2^iota
  uint32_t NB = 0, NWc = 0, NF = 0; // calls on bytes, calls on two-digit weights, N_F = NB + NWc
  uint32_t m0 = 0;                  // W's rows
  uint32_t mR = 0;                  // the ranged rows are those below mR (HfRangePlan::make())
  uint32_t mP = 0;                  // the pairs of ranged rows, mR / 2
  uint32_t m2 = 0;                  // the pairs below m2 are of class N: the two digits of a weight
  const uint32_t *ymap = nullptr;   // 2 mP row numbers: ymap[d mP + psi] = y(psi, d)
  // Set only in a view of pieces (HfRangePlan::pieces_of()), by which WfWindows (hf_wf_keys.cuh)
  // computes a call's keys one piece at a time. whole_lwr is the whole plan's lwr (0: not such a
  // view); lwr, n, NB, NWc and NF above are then those of pieces in place of calls. "Call" f is
  // piece q = f mod 2^(whole_lwr - lwr) of the whole plan's call f div 2^(whole_lwr - lwr). Its
  // place (g | p), where g is the top `closed` bits, is that call's place (g | q | p).
  int whole_lwr = 0, closed = 0;

  // The pair map: the row of W that is row d < 2 of pair psi < mP.
  HF_RANGE_BOTH uint32_t y(uint32_t psi, uint32_t d) const { return ymap[(size_t)d * mP + psi]; }
  // Whether row y < m0 is ranged, that is, covered by the calls.
  HF_RANGE_BOTH bool ranged(uint32_t y) const { return y < mR; }
  HF_RANGE_BOTH uint32_t kind(uint32_t f) const { return f < NB ? HF_RANGE_B : HF_RANGE_W; }
  HF_RANGE_BOTH uint32_t places() const { return (uint32_t)1 << lwr; }

  // For f < NF and pl < 2^lwr: writes the rows of W at place pl of call f and their coefficients wt
  // ("weights" in messages), and returns their number.
  //   B_j: place (d | p), d the top bit, has row d of pair j 2^(lwr - 1) + p, if that is below mP;
  //   W_j: place pl has both rows of pair j 2^lwr + pl, if that is below m2.
  // Every other place is not covered: the result is 0.
  //   key_f[pl 2^lk + c] = off_f + sum_i wt[i] W[row[i]][c]  at a covered place,
  //                        0, the padding key,               at every other.
  // In this file only rows_at() and gamma_closed() decide which rows of W are at which place.
  HF_RANGE_BOTH int rows_at(uint32_t f, uint32_t pl, uint32_t row[2], uint32_t wt[2]) const {
    int lwr = this->lwr; // from here on the whole plan's call, place and numbers
    uint32_t NB = this->NB;
    if (whole_lwr) {
      const int more = whole_lwr - lwr, low = lwr - closed;
      pl = ((((pl >> low) << more) | (f & (((uint32_t)1 << more) - 1))) << low) | (pl & (((uint32_t)1 << low) - 1));
      f >>= more, NB >>= more, lwr = whole_lwr;
    }
    if (f < NB) {
      const uint32_t half = (uint32_t)1 << (lwr - 1);
      const uint64_t psi = (uint64_t)f * half + (pl & (half - 1));
      if (psi >= mP) return 0;
      row[0] = y((uint32_t)psi, pl >> (lwr - 1)), wt[0] = 1;
      return 1;
    }
    const uint64_t psi = ((uint64_t)(f - NB) << lwr) + pl;
    if (psi >= m2) return 0;
    row[0] = y((uint32_t)psi, 0), wt[0] = 1;
    row[1] = y((uint32_t)psi, 1), wt[1] = HF_RANGE_SECOND_DIGIT;
    return 2;
  }
};

// table[i] = eq(z, i) for i < 2^l, with coordinate 0 of z at the top bit of i: the product over v < l of z[v] where bit
// l - 1 - v of i is 1 and of 1 - z[v] where it is 0.
template <class El> void range_eq_table(const El *z, int l, El *table) {
  table[0] = El(1, 0, 0);
  for (int v = 0; v < l; v++)
    for (size_t t = (size_t)1 << v; t-- > 0;) {
      const El one = emul(table[t], z[v]);
      table[2 * t] = esub(table[t], one);
      table[2 * t + 1] = one;
    }
}

// The plan: the view, the pair map that it points to, and one record for each call. Outside this
// file and tests, only kind, k, off, pub and ob of a record are read (hf_range_windows.cuh). WF has
// no identities; that file reads iota and pub only to count keys. The record's fields from e on,
// msg, nrel, nop, raLen, bwK, claims and slots() describe a range file (`ranges.bin`), as do the
// names pad slot, S_R, RA, BW, O1 to O3 and K1' to K3. No program in this repository writes or
// reads such a file.
struct HfRangePlan : HfRangeView {
  struct One {
    uint32_t kind = 0;       // HF_RANGE_B or HF_RANGE_W
    uint32_t j = 0;          // B_j or W_j
    int k = 0;               // k_f: the table has 2^k rows
    uint32_t off = 0;        // off_f
    uint32_t cov = 0;        // covered places
    uint32_t pub = 0;        // padding keys in each identity, (2^lwr - cov) 2^(lk - iota)
    uint64_t e = 0;          // e_f, its masked elements in the file: roots, key side, table side
    uint64_t base = 0;       // its first pad slot, which is also the number of elements of the calls before it
    uint64_t sv = 0;         // the pad slot of vL_f; vR_f has sv + 1
    uint64_t rt = 0;         // the first pad slot of the 2 iota rounds of its root test
    uint64_t ob = 0, Kb = 0; // its columns of BW: ob .. ob + Kb - 1
    uint64_t ra = 0;         // the first of its 62 x 2^iota entries of RA
    uint64_t o1 = 0;         // the number of the first of the 3 (n - iota + k + 2) + 2 claims of its layers (O1)
    uint64_t o2 = 0;         // the number of the first of the 95 claims of its root test (O2)
    uint64_t o3 = 0;         // the number of K1'_f; K2.(f, 0), K2.(f, 1) and K3.f have the next three
  };
  std::vector<One> c;
  int identity = 0;                    // an identity has at most 2^identity keys: iota = max(0, n - identity)
  uint64_t msg = 0, nrel = 0, nop = 0; // masked elements; product relations; openings
  uint64_t raLen = 0, bwK = 0;         // RA's entries; BW's columns
  uint64_t claims = 0;                 // O1, O2 and O3 of all calls
  std::vector<uint32_t> map;           // what ymap points to
  std::vector<uint32_t> pair_of;       // the map's inverse: pair_of[y] = 2 psi + d for y < mR

  HfRangePlan() = default;
  HfRangePlan(const HfRangePlan &) = delete; // ymap points into map
  HfRangePlan &operator=(const HfRangePlan &) = delete;

  // ns, the pad slots of S_R. lambda_e is the greater of b_RA and b_BW, numbers of the shapes of RA
  // and BW.
  uint64_t slots(int lambda_e) const { return msg + 2 * nrel + nop + 2 * (uint64_t)lambda_e; }
  // The view with the pair map at another address: a copy of map, 2 mP words, on the GPU.
  HfRangeView view_with(const uint32_t *map_elsewhere) const {
    HfRangeView v = *this;
    v.ymap = map_elsewhere;
    return v;
  }

  // Builds the plan for a matrix W of `rows` rows and 2^lk_ columns, with 2^lwr_ places in a call.
  // All rows are ranged (hf_ranged_rows(), hf_range_format.h).
  // lwr_ is "lwr" in messages, except in the two policy-range-calls messages, which name the line
  // itself; `cp_commit` derives it from the parameter file's line `ranges-part`.
  // identity_bits becomes `identity`; `cp_commit` passes 30.
  // Checks the conditions marked R0 (a) to (d) below, in that order, with the checks on the layout
  // before (c) and the pair map's before (d), and then runs the coverage test. Returns the empty
  // string, or an error message that begins with an error name. After an error the plan must not
  // be used.
  std::string make(int lk_, uint32_t lwr_, size_t rows, const std::vector<HfRangeUnit> &units,
                   int identity_bits = HF_RANGE_IDENTITY_BITS) {
    static_cast<HfRangeView &>(*this) = HfRangeView();
    c.clear(), map.clear(), pair_of.clear();
    msg = nrel = nop = raLen = bwK = claims = 0;
    identity = identity_bits;
    auto number = [](uint64_t x) {
      return std::to_string(x);
    };
    // R0 (a): first lwr alone, before it is cast to int.
    if (lwr_ < (uint32_t)HF_LWR_LEAST || lwr_ > (uint32_t)HF_LWR_MOST)
      return "policy-cert-words: lwr, log2 of the places in a lookup call (the line ranges-part, or one less), must be from " + number(HF_LWR_LEAST) + " to " +
             number(HF_LWR_MOST) + "; it is " + number(lwr_);
    if (lk_ < 0 || lk_ > HF_RANGE_N_MOST || (int)lwr_ + lk_ < HF_RANGE_N_LEAST || (int)lwr_ + lk_ > HF_RANGE_N_MOST)
      return "policy-cert-words: lwr + lk (lwr: the line ranges-part, or one less; lk: the last number of the line table) must be from " + number(HF_RANGE_N_LEAST) + " to " +
             number(HF_RANGE_N_MOST);
    lk = lk_, lwr = (int)lwr_, n = lwr + lk;
    // R0 (b): iota is at most lk, and an identity has at least 2^4 keys.
    iota = n > identity ? n - identity : 0;
    if (iota > lk || iota > n - 4)
      return "policy-range-iota: lwr + lk (lwr: the line ranges-part, or one less; lk: the last number of the line table) - " + std::to_string(identity) +
             " must be at most lk and at most lwr + lk - 4";
    // The ranged rows.
    if (!rows || rows % 2 || rows >> 31)
      return "internal-cert-pairs: internal error: W must have an even number of rows, at least 2 and below 2^31";
    m0 = (uint32_t)rows;
    const size_t end = range_end(units);
    if (end > rows || !end || end % 2)
      return "internal-cert-pairs: internal error: W must have units, and they must end at an even row, within W";
    mR = (uint32_t)hf_ranged_rows(m0, end);
    if (mR % 2 || mR > m0 || end > mR)
      return "internal-cert-pairs: internal error: the ranged rows must be an even number, hold all units and lie "
             "within W";
    mP = mR / 2;
    const uint64_t half = (uint64_t)1 << (lwr - 1), full = (uint64_t)1 << lwr;
    // R0 (c): lwr is no larger than W's rows need.
    const uint64_t calls_B = (mP + half - 1) / half;
    if (calls_B * half >= 2 * (uint64_t)mP)
      return "policy-cert-words: lwr is too large: the calls on bytes would have at least twice as many places as W "
             "has rows to show";
    const std::string why = pair_map(units);
    if (!why.empty()) return why;
    const uint64_t calls_W = (m2 + full - 1) / full;
    // R0 (d): the number of calls is from HF_RANGE_CALLS_LEAST to HF_RANGE_CALLS_MOST. `cp_commit`
    // applies both bounds, and no verifier code uses them.
    if (calls_B + calls_W < (uint64_t)HF_RANGE_CALLS_LEAST)
      return "policy-range-calls: the parameter file's line ranges-part is too large for this matrix: it gives one "
             "lookup call, fewer than " +
             number(HF_RANGE_CALLS_LEAST);
    if (calls_B + calls_W > (uint64_t)HF_RANGE_CALLS_MOST)
      return "policy-range-calls: the parameter file's line ranges-part is too small for this matrix: it gives " +
             number(calls_B + calls_W) + " lookup calls, more than " + number(HF_RANGE_CALLS_MOST);
    NB = (uint32_t)calls_B, NWc = (uint32_t)calls_W, NF = NB + NWc;
    c.assign(NF, One());
    for (uint32_t f = 0; f < NF; f++) {
      One &o = c[f];
      o.kind = kind(f), o.j = f < NB ? f : f - NB;
      o.k = hf_range_k(o.kind), o.off = hf_range_off(o.kind);
      const uint64_t left =
          o.kind == HF_RANGE_B ? mP - o.j * half : m2 - o.j * full; // pairs of this and later calls
      o.cov = (uint32_t)(o.kind == HF_RANGE_B ? 2 * (left < half ? left : half) : left < full ? left : full);
      o.pub = (uint32_t)((full - o.cov) << (lk - iota)); // at most 2^(n - iota) <= 2^identity
      o.e = hf_range_elements(n, iota, o.k);
      o.base = msg, o.sv = o.base + hf_range_sv(n, iota);
      msg += o.e;
      nrel += hf_range_nrel(n, iota, o.k), nop += hf_range_nop(n, iota, o.k);
      o.ob = bwK, o.Kb = hf_range_bw_columns(iota, o.k), bwK += o.Kb;
      o.ra = raLen, raLen += hf_range_ra_entries(iota);
      o.o1 = claims, claims += 3 * (uint64_t)(n - iota + o.k + 2) + 2;
    }
    // After all calls' elements come the root tests' rounds, call after call. After all of O1 come O2, then O3.
    for (One &o : c) o.rt = msg, msg += 2 * (uint64_t)iota;
    for (One &o : c) o.o2 = claims, claims += 95;
    for (One &o : c) o.o3 = claims, claims += 4;
    return cover_test();
  }
  // Makes this plan a view of `whole` in pieces of 2^piece_lwr places (see HfRangeView::whole_lwr).
  // With a place, a piece has every place that differs from it only in the top closed_bits bits, as
  // prover/engine/lkp_pieces.cuh needs. c[f] has the piece's cov and pub but its call's columns of
  // BW (ob, Kb), so that counts of a piece's keys are added to its call's. msg, nrel, nop, raLen
  // and claims are 0.
  // Not checked: make() has succeeded on `whole`, and closed_bits <= piece_lwr <= whole.lwr.
  void pieces_of(const HfRangePlan &whole, int piece_lwr, int closed_bits) {
    static_cast<HfRangeView &>(*this) = whole;
    map = whole.map, pair_of = whole.pair_of, ymap = map.data();
    identity = whole.identity, bwK = whole.bwK;
    msg = nrel = nop = raLen = claims = 0;
    const int more = whole.lwr - piece_lwr;
    whole_lwr = whole.lwr, closed = closed_bits, lwr = piece_lwr, n = lwr + lk;
    NB = whole.NB << more, NWc = whole.NWc << more, NF = NB + NWc;
    c.assign(NF, One());
    for (uint32_t f = 0; f < NF; f++) {
      One &o = c[f];
      const One &of = whole.c[f >> more];
      o.kind = of.kind, o.j = f < NB ? f : f - NB, o.k = of.k, o.off = of.off, o.ob = of.ob, o.Kb = of.Kb;
      uint32_t row[2], wt[2];
      for (uint32_t pl = 0; pl < places(); pl++) o.cov += rows_at(f, pl, row, wt) != 0;
      o.pub = (uint32_t)((uint64_t)(places() - o.cov) << (lk - iota));
    }
  }
  // Builds the pair map. The pairs, in order: for each unit of two digits, digit 0 with digit 1 of
  // each number (these are the pairs below m2); for each unit of four digits (the embedding),
  // digits 0 with 1 of each number, then 2 with 3; the rows from the last unit's end up to mR, in
  // pairs of neighbours. Needs m0, mR and mP; sets map, ymap, pair_of and m2.
  std::string pair_map(const std::vector<HfRangeUnit> &units) {
    const char *fault =
        "internal-cert-pairs: internal error: the row pairs do not cover every ranged row of W exactly once";
    map.assign(2 * (size_t)mP, 0);
    ymap = map.data();
    size_t psi = 0, end = 0;
    auto pair = [&](size_t y0, size_t y1) {
      if (psi < mP) map[psi] = (uint32_t)y0, map[mP + psi] = (uint32_t)y1;
      psi++;
    };
    for (const HfRangeUnit &u : units) {
      if ((u.digits != 2 && u.digits != 4) || u.rows % (size_t)u.digits || u.off + u.rows > mR) return fault;
      if (u.off + u.rows > end) end = u.off + u.rows;
      for (size_t h = u.rows / 2, i = 0; u.digits == 2 && i < h; i++) pair(u.off + i, u.off + h + i);
    }
    if (psi > mP) return fault;
    m2 = (uint32_t)psi;
    for (const HfRangeUnit &u : units)
      for (size_t h = u.rows / 4, q = 0; u.digits == 4 && q < 2; q++)
        for (size_t i = 0; i < h; i++) pair(u.off + 2 * q * h + i, u.off + (2 * q + 1) * h + i);
    for (size_t y = end; y + 1 < mR; y += 2) pair(y, y + 1);
    if (psi != mP) return fault;
    const uint32_t none = ~(uint32_t)0;
    pair_of.assign(m0, none);
    for (size_t i = 0; i < map.size(); i++) {
      if (map[i] >= mR || pair_of[map[i]] != none) return fault;
      pair_of[map[i]] = (uint32_t)(2 * (i % mP) + i / mP);
    }
    return "";
  }

  // The coverage test: the four numbered steps of cover_test_with(). It guards against bugs in this
  // program: no input should make it fail. Of the pair map it checks only that each ranged row
  // occurs once (step 1), not which rows are paired. That the right rows are paired is tested only
  // by the known answers of a test that is not part of this repository. `cp_commit` also
  // compares the pairing with the configuration's (once_parts_agree(),
  // prover/hash_commitment/hf_commit_once.cuh): this shows that the two agree, not that either is
  // right. Returns a string as make() does.
  std::string cover_test() const {
    return cover_test_with(
        [this](uint32_t f, uint32_t pl, uint32_t row[2], uint32_t wt[2]) { return rows_at(f, pl, row, wt); });
  }
  // The same with rows_of in place of rows_at(), for tests of the coverage test.
  template <class Rows> std::string cover_test_with(Rows rows_of) const {
    const std::string pairs = "internal-cert-pairs: internal error: ", cover = "internal-range-cover: internal error: ";
    auto str = [](uint64_t x) {
      return std::to_string(x);
    };
    // 1. The pair map contains every ranged row exactly once, and nothing else.
    const uint32_t none = ~(uint32_t)0;
    if (!mR || 2 * (uint64_t)mP != mR || mR > m0 || m2 > mP)
      return pairs + "the numbers of rows and of pairs do not agree";
    if (map.size() != 2 * (size_t)mP || ymap != map.data())
      return pairs + "the pair map is not the plan's own, of 2 mP rows";
    std::vector<uint32_t> pair_here(m0, none);
    for (size_t i = 0; i < map.size(); i++) {
      if (map[i] >= mR || pair_here[map[i]] != none)
        return pairs + "the row pairs do not cover every ranged row of W exactly once";
      pair_here[map[i]] = (uint32_t)(2 * (i % mP) + i / mP);
    }
    // 2. n, iota, NB, NWc and NF, recomputed here.
    if (lwr < HF_LWR_LEAST || lwr > HF_LWR_MOST || lk < 0 || n != lwr + lk ||
        iota != (n > identity ? n - identity : 0) || iota > lk)
      return cover + "n and iota are not those of lwr and lk";
    const uint64_t half = (uint64_t)1 << (lwr - 1), full = (uint64_t)1 << lwr;
    if (NB != (mP + half - 1) / half || NWc != (m2 + full - 1) / full || NF != NB + NWc || c.size() != NF)
      return cover + "the numbers of calls are not those of lwr and of the numbers of pairs";
    // 3. Every call's record (all B first), then each of its places.
    std::vector<uint8_t> cB(m0, 0), cW(m0, 0);
    for (uint32_t f = 0; f < NF; f++) {
      const One &o = c[f];
      if (o.kind != kind(f) || o.j != (f < NB ? f : f - NB) || o.k != hf_range_k(o.kind) ||
          o.off != hf_range_off(o.kind))
        return cover + "call " + str(f) + " has not the kind, the number, the table and the offset that belong to it";
      uint64_t covered = 0;
      for (uint64_t pl = 0; pl < full; pl++) {
        uint32_t row[2] = {none, none}, wt[2] = {0, 0};
        const int rows = rows_of(f, (uint32_t)pl, row, wt);
        if (!rows) continue;
        covered++;
        auto at = [&] {
          return "call " + str(f) + ", place " + str(pl) + ": ";
        };
        if (o.kind == HF_RANGE_B) {
          if (rows != 1 || wt[0] != 1 || row[0] >= m0)
            return cover + at() + "a call on bytes must have one row of W there, with the weight 1";
          if (cB[row[0]]++)
            return cover + at() + "row " + str(row[0]) + " stands at an earlier place of a call on bytes too";
        } else {
          if (rows != 2 || wt[0] != 1 || wt[1] != 256 || row[0] >= m0 || row[1] >= m0 || pair_here[row[0]] % 2 ||
              pair_here[row[1]] != pair_here[row[0]] + 1 || pair_here[row[0]] / 2 >= m2)
            return cover + at() +
                   "a call on weights must have the two rows of one pair of class N there, with the weights 1 and 256";
          if (cW[row[0]]++ || cW[row[1]]++)
            return cover + at() + "the pair stands at an earlier place of a call on weights too";
        }
      }
      if (covered != o.cov || o.pub != (full - covered) << (lk - iota))
        return cover + "call " + str(f) + " covers " + str(covered) + " places; its record says " + str(o.cov) +
               ", with " + str(o.pub) + " padding keys in an identity";
    }
    // 4. Every ranged row once in a call B; the rows of the pairs of class N once in a call W; no
    //    other row anywhere.
    for (uint32_t y = 0; y < m0; y++) {
      const int bytes = ranged(y) ? 1 : 0, weights = ranged(y) && pair_here[y] / 2 < m2 ? 1 : 0;
      if (cB[y] != bytes)
        return cover + "row " + str(y) + " stands in " + str(cB[y]) + " calls on bytes; it must stand in " + str(bytes);
      if (cW[y] != weights)
        return cover + "row " + str(y) + " stands in " + str(cW[y]) + " calls on weights; it must stand in " +
               str(weights);
    }
    return "";
  }

  // ---- On field elements. Only tests call these functions and range_eq_table(). The element type
  // El must have eadd, esub, emul, esc and the constructor El(a, b, c), as
  // prover/common/host_common.h declares them for E, the cubic extension field's element. A point's
  // coordinate 0 belongs to the top bit of an index. zk_f = (zr_f, zc_f) is call f's point: the
  // first lwr coordinates are the place's, the last lk the column's.

  // SG_f = the sum of eq(zr_f, pl) over the covered places, from the table eq_f[pl] = eq(zr_f, pl) (range_eq_table()).
  template <class El> El sg_from_table(uint32_t f, const El *eq_f) const {
    El sum(0, 0, 0);
    uint32_t row[2], wt[2];
    for (uint32_t pl = 0; pl < places(); pl++)
      if (rows_at(f, pl, row, wt)) sum = eadd(sum, eq_f[pl]);
    return sum;
  }
  // The same from the point: zr = the first lwr coordinates of zk_f.
  template <class El> El sg(uint32_t f, const El *zr) const {
    std::vector<El> eq_f(places(), El(0, 0, 0));
    range_eq_table(zr, lwr, eq_f.data());
    return sg_from_table(f, eq_f.data());
  }
  // const_f = off_f SG_f.
  template <class El> El constant(uint32_t f, El sg_f) const { return esc(sg_f, (uint64_t)c[f].off); }
  // Adds co G_f to the m0 entries of gamma: G_f[y] = the sum over the places pl that have row y of
  // wt eq(zr_f, pl). With all calls and co = co_f the sum is Gamma_R.
  template <class El> void add_g(uint32_t f, El co, const El *eq_f, El *gamma) const {
    uint32_t row[2], wt[2];
    for (uint32_t pl = 0; pl < places(); pl++) {
      const int rows = rows_at(f, pl, row, wt);
      if (!rows) continue;
      const El term = emul(co, eq_f[pl]);
      for (int i = 0; i < rows; i++) gamma[row[i]] = eadd(gamma[row[i]], esc(term, (uint64_t)wt[i]));
    }
  }
  // Gamma_R[y], y < m0, by a closed formula, which does not use rows_at() and so is a check on it.
  // co: N_F coefficients. eq: the N_F tables one after another, eq[f 2^lwr + pl] = eq(zr_f, pl).
  template <class El> El gamma_closed(size_t y, const El *co, const El *eq) const {
    if (y >= mR) return El(0, 0, 0);
    const size_t psi = pair_of[y] >> 1, d = pair_of[y] & 1, half = (size_t)1 << (lwr - 1), full = (size_t)1 << lwr;
    const size_t fB = psi / half, plB = d * half + psi % half;
    El g = emul(co[fB], eq[fB * full + plB]);
    if (psi < m2) {
      const size_t fW = NB + psi / full, plW = psi % full;
      const El w = emul(co[fW], eq[fW * full + plW]);
      g = eadd(g, d ? esc(w, (uint64_t)256) : w);
    }
    return g;
  }

  //
  // Prints the plan.
  void print(FILE *to) const {
    fprintf(to, "lk=%d lwr=%d identity=%d k=%llu m0=%u mP=%u m2=%u n=%d iota=%d NB=%u NWc=%u NF=%u\n", lk, lwr,
            identity, 1ull << lk, m0, mP, m2, n, iota, NB, NWc, NF);
    for (uint32_t f = 0; f < NF; f++)
      fprintf(to, "f %4u  %c_%-4u n %2d iota %d k_f %2d off_f %4u cov_f %6u pub_b %u\n", f,
              c[f].kind == HF_RANGE_B ? 'B' : 'W', c[f].j, n, iota, c[f].k, c[f].off, c[f].cov, c[f].pub);
  }
};

} // namespace hf
using hf::HfRangePlan;
using hf::HfRangeView;
