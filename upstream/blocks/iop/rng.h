// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// RNG: the approximate range check on committed cells (small integers), with packing and reads.
//
// A prover's cells form integer matrices Y, which it does not send. It sends packed entries, with
// several cells of a column in each (RngMatrix), and reads, weighted partial sums through which
// other protocol blocks use the cells (RngRead). RNG is to show that one integer matrix Y with
// |entries| < 2 D, a looser bound than honest cells meet, has this packing and these reads.
// D and the messages: at RngShape. The error of the round that draws Theta, the verifier's random
// matrix, and a union bound: above rng_theta(). The equations and their numbering: above rng_row().
//
// The caller sends packed entries and reads with an out-of-domain sample of each (a random linear
// form of the whole message: send_prove(), send_verify()) and fixes the reads' weights. It then
// calls rng_prove() or rng_verify(), then rng_row(ctx, ..), and adds the returned linear claim
// (`Row` in identifiers) to the group that its proximity test reads: without it RNG shows nothing.
//
// The programs cp_prove and cp_verify do not include this file: they have a second implementation
// of the check. Only tests under blocks/ run this one.
//
// The check is the approximate range check of Lyubashevsky, Nguyen, Seiler, section 2.7
// (combinations with coefficients 0, +1, -1). It is used against wrap-around as in Gentry, Halevi,
// Lyubashevsky, section 3.4 (a protocol). Both works are cited in full at rng_theta().
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/context.h"
#include "iop/laid.h"
#include "iop/lkp.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "numbers/blocks.h"
#include "numbers/round_names.h"

namespace blk {

// One matrix Y of cells, m rows by k columns, and where its packing is: in an oracle message
// (encoded rows, sent as Merkle roots: `OracleShape` in blocks/iop/relation.h). With F the field
// and E its extension (blocks/field/field.h), the packed matrix, over F:
//   Pack(Y)[i'][c] = sum_(l < lanes) radix^l Y[lanes i' + l][c]
struct RngMatrix {
  u64 rows = 0, columns = 0;    // of the cells
  OracleId packed = NONE_ID;    // the oracle message that holds the packed matrix
  std::vector<LaidBlock> laid;  // the packed matrix (i = packed row, c = column) in that message's main table
};
// One read of the shape's matrix number `matrix`, under a public weight vector phi of length L.
// Its table, over E:
//   Tab[c (m / L) + g] = sum_(i < L) phi[i] Y[g L + i][c]
// Other blocks need reads because no linear form gets a single cell back from a packed entry.
struct RngRead {
  size_t matrix = 0;
  u64 length = 0;  // L, its weight vector's length: divides the matrix's number of rows
};  // (Its table begins in the reads' vector where the one before it ends: RngShape::read_first.)
// The weights are not part of the shape: as a rule a run computes them from challenges. Not
// checked, the caller's duty: they are functions only of values in the hash chain's first query
// (the Fiat-Shamir transcript's first hash input), received messages and draws of this channel
// (`Rounds` in blocks/chain/rounds.h).
using RngWeights = std::vector<std::vector<E>>;  // phi, one vector for each read
// What is public about one check, the weights apart. N = `combinations`, the number of random
// combinations of a matrix's columns (the columns of Psi, below); b0, b1 = digit_bits;
// 2 D = 2^(b0 + b1), where the entries of an honest prover's combinations must be in [-D, D);
// radix = 2^log_radix; `lanes` = the cells in one packed entry. The messages of a run (V: the
// verifier's, P: the prover's):
//   V    Theta, with entries in {-1, 0, 1}: N for each column of each matrix (rng_theta()).
//   P    the digits of Psi = Y Theta (a row for each cell row, N combinations), one digit an entry:
//        Psi + D = lo + 2^b0 hi, lo < 2^b0, hi < 2^b1. In the same prover message, the
//        multiplicities of the lookup on the digits. Both are sent with a sample (send_prove()).
//   LKP  the lookup argument (blocks/iop/lkp.h), run by RNG itself: every digit is in its range.
// The whole shape must be in the first query. So the entry points take it as a Public<RngShape>: a
// value that a chain's Binder wrote into that query (Rounds::get() also accepts the first query of
// an earlier chain that this chain follows), or one derived from such values
// (blocks/chain/public.h).
struct RngShape {
  unsigned lanes = 1;
  unsigned log_radix = 0;
  unsigned digit_bits[2] = {0, 0};
  unsigned combinations = 0;
  std::vector<RngMatrix> matrices;
  std::vector<RngRead> reads;
  OracleId reads_in = NONE_ID;        // the oracle message that holds the reads' vector (none if there are no reads)
  std::vector<LaidBlock> reads_laid;  // that vector as a matrix with one row (i = 0, c = the entry's number)
  // The digits' matrix has i = the combination u and c = the column: the low digit of cell row j
  // (counted through all matrices) at c = j, the high digit at c = digit_columns + j. digit_columns
  // is a power of two, at least cell_rows().
  u64 digit_columns = 0;
  OracleNotice digits;                 // the shape of the oracle message that RNG sends
  std::vector<LaidBlock> digits_laid;  // in its main table
  OracleNotice multiplicities;         // of the lookup on the digits: sent with the digits; `rng_planes_at` fixes its layout
  OracleId digits_in = NONE_ID;        // the number that the digits' message gets on the channel; the multiplicities' message gets the next
  OracleId masks_from = NONE_ID;       // the oracle message whose mask rows give the lookup's mask entries

  u64 cell_rows() const {
    u64 n = 0;
    for (const RngMatrix &m : matrices) n += m.rows;
    return n;
  }
  u64 all_columns() const {
    u64 n = 0;
    for (const RngMatrix &m : matrices) n += m.columns;
    return n;
  }
  u64 packed_rows(size_t mu) const { return (matrices[mu].rows + lanes - 1) / lanes; }
  u64 read_entries(size_t q) const { return matrices[reads[q].matrix].columns * (matrices[reads[q].matrix].rows / reads[q].length); }
  u64 read_first(size_t q) const {  // read q's first entry in the reads' vector; q = `reads.size()`: that vector's length
    u64 n = 0;
    for (size_t i = 0; i < q; i++) n += read_entries(i);
    return n;
  }
  u64 step() const { return u64(1) << ceil_log2(combinations); }  // Np: smallest power of two >= N
  // How many equation numbers there are, unused ones included (numbering: rng_row()): the batching
  // by powers is one of that many claims.
  u64 equations() const {
    u64 n = 0;
    for (size_t mu = 0; mu < matrices.size(); mu++) n += step() << ceil_log2(packed_rows(mu));
    for (const RngRead &r : reads) n += step() * (matrices[r.matrix].rows / r.length);
    return n;
  }
};

// The bytes by which a shape enters the first query: every field, in declaration order.
inline void public_bytes(Writer &w, const RngMatrix &m) {
  public_bytes(w, m.rows), public_bytes(w, m.columns), public_bytes(w, m.packed), public_bytes(w, m.laid);
}
inline void public_bytes(Writer &w, const RngRead &r) { public_bytes(w, r.matrix), public_bytes(w, r.length); }
inline void public_bytes(Writer &w, const RngShape &S) {
  public_bytes(w, S.lanes), public_bytes(w, S.log_radix), public_bytes(w, S.digit_bits[0]), public_bytes(w, S.digit_bits[1]), public_bytes(w, S.combinations);
  public_bytes(w, S.matrices), public_bytes(w, S.reads), public_bytes(w, S.reads_in), public_bytes(w, S.reads_laid);
  public_bytes(w, S.digit_columns), public_bytes(w, S.digits), public_bytes(w, S.digits_laid);
  public_bytes(w, S.multiplicities), public_bytes(w, S.digits_in), public_bytes(w, S.masks_from);
}

// Whether the shape is valid and consistent, as far as the object alone shows: nullptr, or the
// fault in words. A Binder asks this through public_fault() and ends the process on a fault. RNG
// needs 4 D - 2 < radix and (4 D - 2)(1 + radix + .. + radix^(lanes - 1)) < p: then a packed entry
// has at most one unpacking into integers in (-2 D, 2 D). The test on `window` implies both.
inline const char *rng_fault(const RngShape &S) {
  if (!(S.lanes >= 1 && S.combinations >= 1 && !S.matrices.empty())) return "RNG: no lane, no combination or no matrix";
  if (!(S.lanes <= 64 && S.log_radix < 64 && S.digit_bits[0] < 64 && S.digit_bits[1] < 64)) return "RNG: a number of the shape is out of range";
  const unsigned window = S.digit_bits[0] + S.digit_bits[1];  // 2 D = 2^window
  if (!(S.digit_bits[0] >= 1 && S.digit_bits[1] >= 1 && window + 1 <= S.log_radix && S.log_radix * (S.lanes - 1) + window + 2 < F_BITS))
    return "RNG: the radix is too small for the window of the combinations, or a packed entry could wrap around in the field";
  // Caps under which no product or sum of the shape's numbers wraps around a u64 in this file: 2^24
  // rows, columns or combinations, 2^12 matrices or reads, 2^36 digit columns. The largest are then
  // equations(), which adds two sums, each at most 2^12 x 2^24 x 2^24 = 2^60, and the number of the
  // lookup's keys, 2 x digit columns x Np <= 2 x 2^36 x 2^24 = 2^61.
  const u64 most = u64(1) << 24;
  if (!(S.combinations <= most && S.matrices.size() <= 4096 && S.reads.size() <= 4096 && S.digit_columns <= u64(1) << 36)) return "RNG: a number of the shape is out of range";
  for (const RngMatrix &m : S.matrices)
    if (!(m.rows >= 1 && m.columns >= 1 && m.rows <= most && m.columns <= most)) return "RNG: an empty matrix, or one with too many rows or columns";
  if (!(S.cell_rows() <= S.digit_columns && (S.digit_columns & (S.digit_columns - 1)) == 0)) return "RNG: the digits' matrix has too few columns, or their number is no power of two";
  for (const RngMatrix &m : S.matrices)
    if (!S.reads.empty() && m.packed == S.reads_in) return "RNG: packed cells and reads in one oracle message are not served";
  for (const RngRead &r : S.reads)
    if (!(r.matrix < S.matrices.size() && r.length >= 1 && S.matrices[r.matrix].rows % r.length == 0)) return "RNG: a read of no matrix, or weights whose number does not divide the matrix's rows";
  if (S.digits_in == NONE_ID || S.masks_from == NONE_ID || S.masks_from >= S.digits_in) return "RNG: the digits' message has no number, or the look-up's masks come from no earlier message";
  for (const RngMatrix &m : S.matrices)
    if (m.packed >= S.digits_in) return "RNG: packed cells in a message that comes no earlier than the digits";
  if (!S.reads.empty() && S.reads_in >= S.digits_in) return "RNG: the reads in a message that comes no earlier than the digits";
  return nullptr;
}
inline const char *public_fault(const RngMatrix &) { return nullptr; }  // a matrix or a read is checked with the whole shape
inline const char *public_fault(const RngRead &) { return nullptr; }
inline const char *public_fault(const RngShape &S) { return rng_fault(S); }

// The lookup on the digits. Its public inputs (rng_lookup()), the layout of its multiplicities
// (rng_planes_at()) and the claim on its keys (rng_key_row()) are functions of the shape, and
// digits_laid alone says where the digits are: with a second description of that layout in a
// caller's hands, a cell of 1000 has been accepted under the bound 2 D = 32 (the control of a
// test that is not part of this repository).
// LKP serves two functions, each with its own table: function 0 is the low digits, range 2^b0;
// function 1 the high digits, range 2^b1. With DC = digit_columns, the digit of combination u in
// column j < DC (y = 0 low, 1 high: the matrix's column j + DC y) is key number y + 2 (j + DC u).
// The low bit y is LKP's tree number (LkpPublic::g = 1), and tree y belongs to function y. Key
// numbers with N <= u < Np name no digit: they are padding, row 0 of the function's table.
// A column beyond the cell rows is part of the digits' matrix: its digits are real keys, on which
// RNG's claim has no weight.
inline LkpPublic rng_lookup(const RngShape &S) {
  LkpPublic L;
  const u64 DC = S.digit_columns, N = S.combinations, Np = S.step();
  L.parts = {1 + ceil_log2(DC) + ceil_log2(Np)};
  L.d = std::max(S.digit_bits[0], S.digit_bits[1]) + 1;
  const unsigned planes = L.parts[0] + 1;  // the smallest number with 2^v < 2^planes, for 2^v keys
  L.mu = std::max(1u, ceil_log2(planes));
  L.live = planes == 1u << L.mu ? 0 : planes;
  L.g = 1, L.shifted = true;
  L.keys = {{0, 0, 0, 0}, {1, 0, 1, 0}};
  const u64 half = u64(1) << (L.d - 1);  // each function's table, repeated, fills half of the rows
  for (unsigned f = 0; f < 2; f++)
    for (u64 first = f * half; first < (f + 1) * half; first += u64(1) << S.digit_bits[f]) L.tables.push_back({first, S.digit_bits[f], f, {}});
  if (N < Np)
    for (u64 y = 0; y < 2; y++) L.padding.push_back({0, y + 2 * DC * N, 1, DC * (Np - N), y * half});
  L.read_from = {S.digits_in, S.digits_in + 1};
  return L;
}
// Where the multiplicities are: only the live planes (LkpPublic::live; plane j holds bit j of every
// multiplicity), in row 0 of their message's main table. A place is an index within a row. The
// block of planes (first, bits) of lkp_live_blocks() starts at place first * 2^d; within it the
// index is (plane - first) + 2^bits * (row of the lookup's table).
inline u64 rng_planes_places(const LkpPublic &L) { return (u64)L.planes() << L.d; }
inline LkpPlanesAt rng_planes_at(const Statement &st, const RngShape &S, const LkpPublic &L) {
  const OracleId o = S.digits_in + 1;
  const unsigned row_bits = st.row_bits(o, Part::MAIN), place_bits = st.table_bits(o, Part::MAIN) - row_bits, mu = L.mu, d = L.d, planes = L.planes();
  return [o, row_bits, place_bits, mu, d, planes](Row &row, Tensor v) {
    for (const std::pair<unsigned, unsigned> &block : lkp_live_blocks(planes)) {
      Tensor part = lkp_on_block(v, mu, block);
      const unsigned own = part.vars();
      part.insert_choice(own, place_bits - own, ((u64)block.first << d) >> own);
      part.insert_choice(0, row_bits, 0);
      row.add(o, Part::MAIN, std::move(part));
    }
  };
}
// The claim that LKP leaves on its keys, as a linear claim on the sent digits. LKP's weights are
// product forms over the bits (y | j | u) of a key's number; digits_laid places them.
inline Row rng_key_row(const Statement &st, const RngShape &S, LkpKeys &&keys) {
  const unsigned bits_j = ceil_log2(S.digit_columns), bits_u = ceil_log2(S.step());
  const unsigned row_bits = st.row_bits(S.digits_in, Part::MAIN), place_bits = st.table_bits(S.digits_in, Part::MAIN) - row_bits;
  Row row;
  row.name = "the digits at the point";
  BLK_REQUIRE(keys.weights.entries.empty(), "RNG: weights that LKP has not made");
  for (const Tensor &v : keys.weights.tensors) {
    BLK_REQUIRE(!v.has_dense() && v.vars() == 1 + bits_j + bits_u, "RNG: weights that LKP has not made");
    IndexWeights on_u, on_column;
    for (unsigned t = 0; t < bits_j; t++) on_column.lo.push_back(v.lo(1 + t)), on_column.hi.push_back(v.hi(1 + t));
    on_column.lo.push_back(v.lo(0)), on_column.hi.push_back(v.hi(0));  // the matrix's column is j + DC y
    for (unsigned t = 0; t < bits_u; t++) on_u.lo.push_back(v.lo(1 + bits_j + t)), on_u.hi.push_back(v.hi(1 + bits_j + t));
    row.add_form(S.digits_in, laid_form(S.digits_laid, row_bits, place_bits, on_u, on_column, v.scalar));
  }
  row.target = keys.target;
  for (MaskTerm<E> &m : keys.masks) row.add_mask(std::move(m));
  return row;
}

// The checks on the shape that need the statement (`Statement` in blocks/iop/relation.h) and the
// codes. Both entry points call it before Theta is drawn.
inline void rng_shape_check(const Statement &st, const RngShape &S) {
  const char *fault = rng_fault(S);
  BLK_REQUIRE(!fault, fault);
  for (const RngMatrix &m : S.matrices) BLK_REQUIRE(st.oracle(m.packed).sampled, "RNG: an oracle message with packed cells has no answered sample");
  // The layouts: every element in exactly one block, no entry of a table with two elements.
  auto bits_of = [&st](OracleId o, unsigned &rows, unsigned &places) { rows = st.row_bits(o, Part::MAIN), places = st.table_bits(o, Part::MAIN) - rows; };
  auto in_sent_rows = [](const std::vector<LaidBlock> &blocks, const OracleShape &s) {  // a table has 2^ceil(log rows) row slots: the last may be rows of zeros that nobody sent
    for (const LaidBlock &b : blocks)
      if (b.first_row + (u64(1) << b.row_bits) > s.main_rows) return false;
    return true;
  };
  unsigned row_bits, place_bits;
  for (size_t mu = 0; mu < S.matrices.size(); mu++) {  // all matrices of one oracle message together, when its first one comes
    bool first = true;
    for (size_t nu = 0; nu < mu; nu++) first = first && S.matrices[nu].packed != S.matrices[mu].packed;
    if (!first) continue;
    std::vector<LaidMatrix> together;
    for (size_t nu = mu; nu < S.matrices.size(); nu++)
      if (S.matrices[nu].packed == S.matrices[mu].packed) together.push_back(LaidMatrix{S.packed_rows(nu), S.matrices[nu].columns, S.matrices[nu].laid});
    bits_of(S.matrices[mu].packed, row_bits, place_bits);
    BLK_REQUIRE(laid_ok(together, row_bits, place_bits), "RNG: the packed matrices are not laid lawfully into their oracle message's table");
    for (const LaidMatrix &x : together) BLK_REQUIRE(in_sent_rows(x.blocks, st.oracle(S.matrices[mu].packed).shape), "RNG: a block of a layout lies in a row that is not sent");
  }
  if (!S.reads.empty()) {
    BLK_REQUIRE(st.oracle(S.reads_in).sampled, "RNG: the oracle message with the reads has no answered sample");
    bits_of(S.reads_in, row_bits, place_bits);
    BLK_REQUIRE(laid_ok({LaidMatrix{1, S.read_first(S.reads.size()), S.reads_laid}}, row_bits, place_bits), "RNG: the reads are not laid lawfully into their oracle message's table");
    BLK_REQUIRE(in_sent_rows(S.reads_laid, st.oracle(S.reads_in).shape), "RNG: a block of a layout lies in a row that is not sent");
  }
  BLK_REQUIRE(st.oracle(S.masks_from).sampled, "RNG: the oracle message that gives the look-up's mask entries has no answered sample");
  const OracleShape &d = S.digits.shape, &m = S.multiplicities.shape;
  BLK_REQUIRE(d.main_rows >= 1 && d.code < st.codes().base.size() &&
                  laid_ok({LaidMatrix{S.combinations, 2 * S.digit_columns, S.digits_laid}}, ceil_log2(d.main_rows), st.codes().base[d.code].log_ell()),
              "RNG: the digits are not laid lawfully into their oracle message's table");
  BLK_REQUIRE(in_sent_rows(S.digits_laid, d), "RNG: a block of a layout lies in a row that is not sent");
  const LkpPublic L = rng_lookup(S);
  fault = lkp_fault(L);
  BLK_REQUIRE(!fault, fault);
  BLK_REQUIRE(m.main_rows >= 1 && m.code < st.codes().base.size() && rng_planes_places(L) <= st.codes().base[m.code].ell, "RNG: the multiplicities do not fit row 0 of their oracle message's table");
  // Digits and multiplicities cover no place of their message's pad, the random entries that mask a
  // sample's answer. (Without mask rows the pad is in the main rows: Send::pad() in
  // blocks/iop/send.h.)
  if (d.answers && !d.mask_rows)
    for (const PadPlace &p : Send::pad(st.codes(), d)) {
      const unsigned bits = ceil_log2(d.main_rows);
      const u64 row = p.index & ((u64(1) << bits) - 1), place = p.index >> bits;
      for (const LaidBlock &b : S.digits_laid)
        BLK_REQUIRE(!(row >= b.first_row && row < b.first_row + (u64(1) << b.row_bits) && place >= b.first_place && place < b.first_place + (u64(1) << b.place_bits)),
                    "RNG: the digits cover a place of their message's pad");
    }
  if (m.answers && !m.mask_rows)
    for (const PadPlace &p : Send::pad(st.codes(), m)) {
      const unsigned bits = ceil_log2(m.main_rows);
      BLK_REQUIRE((p.index & ((u64(1) << bits) - 1)) != 0 || p.index >> bits >= rng_planes_places(L), "RNG: the multiplicities cover a place of their message's pad");
    }
}

inline void rng_weights_check(const RngShape &S, const RngWeights &phi) {
  BLK_REQUIRE(phi.size() == S.reads.size(), "RNG: not one weight vector for each read");
  for (size_t q = 0; q < phi.size(); q++) BLK_REQUIRE(phi[q].size() == S.reads[q].length, "RNG: a read's weight vector has not the announced length");
}

// N is both in the shape, by which the block runs, and in the plan (the list of all rounds with
// their errors: blocks/numbers/plan.h), which computes the error 2^-N of Theta's round from it and
// stores it beside the round (PlanRound::combinations). Of the plan's N, the chain's first query
// contains only the product columns x N, the number of Theta's entries. So both sides pass the
// shape's N to the channel where Theta is drawn (Rounds::require_combinations()): another N than
// that round's, or a plan that stores none, ends the process. The channel's count of the entries
// (neither more nor fewer than planned) then ties the number of columns.
// Without these ties, 33 columns of 1 combination, run under a plan for 1 column of 33, have been
// accepted with a cell far beyond the bound: that happens whenever the cell's one coefficient is 0.
// (Tested in blocks/iop/tests/rng_test.cpp: rng_plan_fault() must report such inputs, and a run
// under them must end the process on both sides.)
// N is never recomputed from the plan's setting at run time: a comparison with a number recomputed
// from the setting let the same run through, because the caller had changed the setting after
// planning.
// rng_plan_fault() asks the same before a run, of every round of the plan named R_RNG, for a caller
// that gets a shape from elsewhere: nullptr, or what is wrong. A plan whose range checks differ in
// N fits no shape.
inline const char *rng_plan_fault(const RngShape &S, const Planned &p) {
  bool planned = false;
  for (const PlanRound &r : p->rounds_) {
    if (r.name != R_RNG) continue;
    planned = true;
    if (S.combinations != r.combinations) return "RNG: the shape has another number of combinations than the plan prices the matrix's round with";
  }
  return planned ? nullptr : "RNG: the plan has no round in which a matrix is drawn";
}

// Draws the verifier's matrix Theta, on both sides. Entry (column c of matrix mu, combination u) is
// at (first column of mu + c) * N + u. An entry is 0 with probability 1/2, and 1 or -1 with
// probability 1/4 each. The programs' second implementation draws Theta in draw_matrix()
// (prover/hiding/tail_f.cuh) and chainTrits() (verifier/src/vf_transcript_channel.h), with the same call of
// require_combinations().
//
// Order. Theta may be drawn only when every oracle message with packed cells or reads of this check
// has had its sample answered and the weights phi are fixed: a prover that knows Theta can alter a
// read by a vector on which all N combinations vanish. rng_shape_check() ends the process unless
// all those samples have been answered. The entry points copy phi before they draw Theta, and
// rng_row() uses the copy.
//
// Error of this round: 2^-N, for this distribution. Take a column's packed entries and its reads
// together, and call the column explained if one integer column with entries in (-2 D, 2 D) has
// this packing and these reads. Packed cells, reads and phi are fixed before Theta, each as one
// object (beyond unique decoding this rests on the answered samples, whose round has its own
// error), so the first column that is not explained is fixed too. With all other entries of Theta
// fixed, no two neighbouring values of that column's entry Theta[c][u] let combination u pass: the
// difference of the two equations would explain the column. So the passing values are in {0} or in
// {-1, 1}, with probability at most 1/2: that is why 0 has probability 1/2. Each combination must
// pass by itself, and the N combinations depend on disjoint entries of Theta: 2^-N. Nothing is
// assumed about Psi: a lane of Y Theta may overflow and carry.
// (The count is that of this protocol's technical report, argued there. This step was argued on
// paper for this project. By one comparison, part of it rests on Lemma 2.5 of the first
// work cited below, its argument has the shape of the proof of that work's Lemma A.1, and what it
// adds (the last sentences below say for what) is not in that work. A form that takes
// packing and reads apart has a hole; this one has been tried with exhaustive
// searches at toy sizes; it is argued on paper and not refereed.)
// print_accounting() (blocks/numbers/accounting.h; it prints "smallness test" for this check) also
// prints a union bound over all columns: their number times the round's error in the plan, which at
// 2^-N is N - log2(columns) bits. That is the same event and the same prover, counted more
// coarsely. It does not cover a prover that is not bound to one object, it is no knowledge error,
// and no verdict depends on it.
//
// Source for the distribution of Theta's entries: Lyubashevsky, Nguyen, Seiler, "Shorter
// Lattice-Based Zero-Knowledge Proofs via One-Time Commitments", Cryptology ePrint Archive
// 2020/1448, section 2.7 and Lemma A.1 (for any probability of 0, with +1 and -1 equally likely).
// Source for the distribution of Theta's entries: Gentry, Halevi, Lyubashevsky, "Practical
// Non-interactive Publicly Verifiable Secret Sharing with Thousands of Parties", Cryptology ePrint
// Archive 2021/1397, Lemma 3.4 (the probabilities 1/2, 1/4, 1/4, for 128 combinations).
// The count 2^-N is in the line of Lemma 2.5 of Lyubashevsky, Nguyen, Seiler, ePrint 2020/1448:
// this protocol's technical report states it as adapted from that lemma and argues it anew, for
// cells that are sent packed and through reads, the two taken together, with the combinations
// written in digits that a lookup checks. The argument above ("Error of this round") has not been
// compared with Lemma 3.4 of ePrint 2021/1397; one reading of that paper finds the step, at least in part, in its section 3.4.
using RngTheta = std::vector<signed char>;
template <class Ch>
RngTheta rng_theta(Ch &ch, const RngShape &S) {
  auto from = ch.close_bits(R_RNG);
  ch.require_combinations(S.combinations);  // before an entry is drawn
  return from.trits(S.all_columns() * S.combinations);
}

// The linear claim that combines all of RNG's equations, equation number j under
// x^(first_power + j). There is one equation for each packed row i', or each part g of a read, and
// each combination u:
//   sum_c Theta[c][u] Pack(Y)[i'][c] = sum_l radix^l Psi[lanes i' + l][u]
//   sum_c Theta[c][u] Tab[c (m / L) + g] = sum_i phi[i] Psi[g L + i][u]
// Numbering, with Np = step(). Matrix mu has the numbers first_mu + i' Np + u, where first_mu is
// the sum over the earlier matrices of Np * (their packed rows rounded up to a power of two). After
// all matrices, the parts of all reads are counted through, s = 0, 1, .. (the reads in the order of
// the list, g ascending within a read): part s has the numbers first_reads + s Np + u. Numbers that
// name no equation stay unused.
// The equations are never written out: the claim's weights are products. On a packed matrix: powers
// of x^Np over i' times gx over c, where gx[c] = sum_u x^u Theta[c][u]. On the digits: powers of x
// over u times a vector over the columns. On the reads: one vector.
inline Row rng_row(const Statement &st, const RngShape &S, const RngWeights &phi, const RngTheta &theta, OracleId digits, const E &x, u64 first_power = 0) {
  const u64 N = S.combinations, Np = S.step();
  auto power = [&x](u64 e) {
    E r = E::one(), s = x;
    for (; e; e >>= 1, s = s * s)
      if (e & 1) r = r * s;
    return r;
  };
  auto bits_of = [&st](OracleId o, unsigned &rows, unsigned &places) { rows = st.row_bits(o, Part::MAIN), places = st.table_bits(o, Part::MAIN) - rows; };
  const E x_step = power(Np);
  std::vector<E> x_to(N);  // x^u
  x_to[0] = E::one();
  for (u64 u = 1; u < N; u++) x_to[u] = x_to[u - 1] * x;
  E sum_x = E::zero();
  for (const E &p : x_to) sum_x += p;

  Row row;
  row.name = "RNG";
  auto on_digits = std::make_shared<std::vector<E>>(2 * S.digit_columns, E::zero());  // over the columns; low digits first
  std::vector<std::shared_ptr<const std::vector<E>>> gx;                              // for each matrix, over its columns
  std::vector<u64> first_cell_row;
  u64 number = first_power, column = 0, cell_row = 0;
  unsigned row_bits, place_bits;
  for (size_t mu = 0; mu < S.matrices.size(); mu++) {
    const RngMatrix &m = S.matrices[mu];
    auto g = std::make_shared<std::vector<E>>(m.columns, E::zero());
    for (u64 c = 0; c < m.columns; c++)
      for (u64 u = 0; u < N; u++) {
        const signed char t = theta[(column + c) * N + u];
        if (t > 0) (*g)[c] += x_to[u];
        if (t < 0) (*g)[c] = (*g)[c] - x_to[u];
      }
    gx.push_back(g);
    const E lead = power(number);
    bits_of(m.packed, row_bits, place_bits);
    row.add_form(m.packed, laid_form(m.laid, row_bits, place_bits, IndexWeights::powers(x_step, ceil_log2(S.packed_rows(mu))), IndexWeights::written(g), lead));
    E of_packed_row = lead;
    for (u64 i = 0; i < m.rows; i++) {
      if (i && i % S.lanes == 0) of_packed_row = of_packed_row * x_step;
      (*on_digits)[cell_row + i] = (*on_digits)[cell_row + i] - of_packed_row * E::from(from_u64(u64(1) << (S.log_radix * (i % S.lanes))));
    }
    first_cell_row.push_back(cell_row);
    number += Np << ceil_log2(S.packed_rows(mu)), column += m.columns, cell_row += m.rows;
  }
  if (!S.reads.empty()) {
    auto on_reads = std::make_shared<std::vector<E>>(S.read_first(S.reads.size()), E::zero());
    E lead = power(number);
    for (size_t q = 0; q < S.reads.size(); q++) {
      const RngRead &r = S.reads[q];
      const RngMatrix &m = S.matrices[r.matrix];
      const u64 L = r.length, parts = m.rows / L;
      for (u64 g = 0; g < parts; g++, lead = lead * x_step) {
        for (u64 c = 0; c < m.columns; c++) (*on_reads)[S.read_first(q) + c * parts + g] += lead * (*gx[r.matrix])[c];
        for (u64 i = 0; i < L; i++) {
          E &w = (*on_digits)[first_cell_row[r.matrix] + g * L + i];
          w = w - lead * phi[q][i];
        }
      }
    }
    bits_of(S.reads_in, row_bits, place_bits);
    IndexWeights one;  // the one row of the reads' vector
    row.add_form(S.reads_in, laid_form(S.reads_laid, row_bits, place_bits, one, IndexWeights::written(on_reads), E::one()));
  }
  // Psi = lo + 2^b0 hi - D: the high digits get the low digits' weights times 2^b0, and D goes to
  // the right side.
  E all = E::zero();
  const E high = E::from(from_u64(u64(1) << S.digit_bits[0]));
  for (u64 j = 0; j < S.digit_columns; j++) {
    all += (*on_digits)[j];
    (*on_digits)[S.digit_columns + j] = high * (*on_digits)[j];
  }
  bits_of(digits, row_bits, place_bits);
  row.add_form(digits, laid_form(S.digits_laid, row_bits, place_bits, IndexWeights::powers(x, ceil_log2(N)), IndexWeights::written(on_digits), E::one()));
  row.target = E::from(from_u64(u64(1) << (S.digit_bits[0] + S.digit_bits[1] - 1))) * sum_x * all;
  return row;
}

// What RNG leaves for rng_row(). No honest run would reveal a missing claim, so the process ends if
// an output is dropped before rng_row() was called on it (`row_owed`), unless the channel has
// rejected the proof meanwhile. Whether the caller adds the claim that rng_row() returns to its
// group is not checked.
struct RngOutput {
  RngTheta theta;
  RngWeights phi;  // as they were when Theta was drawn
  OracleId digits = NONE_ID;
  E shift = E::zero(), alpha = E::zero();  // of the lookup on the digits
  std::vector<Row> rows;                   // copies of the claims that RNG added to the group (the lookup's, the one on its keys), for tests and reports
  mutable bool row_owed = false;
  const Rounds *channel = nullptr;  // must outlive this

  RngOutput() = default;
  RngOutput(const RngOutput &) = delete;
  RngOutput &operator=(const RngOutput &) = delete;
  ~RngOutput() { BLK_REQUIRE(!row_owed || !channel->ok(), "RNG: its output was dropped: the caller has made no row of its equations"); }
};

// Writes a matrix's element (i, c) into a table, at the place that the layout's blocks give it.
// Ends the process if no block contains the element.
inline void laid_set(Table &table, const std::vector<LaidBlock> &blocks, u64 i, u64 c, const E &value) {
  for (const LaidBlock &b : blocks) {
    if (i >> b.bits_i != b.first_i >> b.bits_i || c >> b.bits_c != b.first_c >> b.bits_c) continue;
    const u64 ri = i - b.first_i, rc = c - b.first_c, x = b.i_is_low ? (rc << b.bits_i) + ri : (ri << b.bits_c) + rc;
    table.set(b.first_row + (x >> b.place_bits), b.first_place + (x & ((u64(1) << b.place_bits) - 1)), value);
    return;
  }
  BLK_REQUIRE(false, "laid: an element that no block holds");
}

// The entry points. group: RNG adds the lookup's claims and the claim on its keys to it and lets
// its own two messages join; the messages with packed cells, reads and masks must be members
// already. (Statement::add_row() ends the process otherwise: for the masks here, for packed cells
// and reads only when the caller adds rng_row()'s claim.) A faulty shape, weights of the wrong size
// or a call out of order ends the process (BLK_REQUIRE in blocks/base.h).
// cells[mu][i * columns + c]; not checked: that the sums of a row's cells fit an i64.
// rng_prove() returns false, with nothing sent after Theta, if a combination leaves [-D, D). That
// failure depends on the witness and on Theta. With D >= (columns) x (the cells' bound) it never
// happens; with a tighter D its probability is part of the zero-knowledge error (range_check_bits()
// in blocks/frame/hiding_conditions.h). Nothing here tries again; a caller that does must draw all
// its private randomness afresh.
inline bool rng_prove(ProverContext &P, const Public<RngShape> &fixed, const RngWeights &phi, const std::vector<std::vector<i64>> &cells, GroupId group, RngOutput &out) {
  const RngShape &S = P.ch.get(fixed);
  rng_shape_check(P.st, S), rng_weights_check(S, phi);
  const Public<LkpPublic> look = fixed.derive(rng_lookup);
  const LkpPublic &L = P.ch.get(look);
  out.phi = phi;
  out.theta = rng_theta(P.ch, S);
  const u64 N = S.combinations, DC = S.digit_columns;
  const i64 D = i64(1) << (S.digit_bits[0] + S.digit_bits[1] - 1);
  Table digits(S.digits.shape.main_deg, S.digits.shape.main_rows, P.st.codes().base[S.digits.shape.code].ell);
  std::vector<E> keys(L.all_keys(), E::zero());
  u64 column = 0, cell_row = 0;
  for (size_t mu = 0; mu < S.matrices.size(); mu++) {
    const RngMatrix &m = S.matrices[mu];
    BLK_REQUIRE(cells[mu].size() == m.rows * m.columns, "RNG: the cells have not the matrix's size");
    for (u64 i = 0; i < m.rows; i++)
      for (u64 u = 0; u < N; u++) {
        i64 psi = 0;
        for (u64 c = 0; c < m.columns; c++) psi += cells[mu][i * m.columns + c] * out.theta[(column + c) * N + u];
        if (psi < -D || psi >= D) return false;
        // (A malicious prover's hi may leave its range: it makes lo + 2^b0 hi - D = psi.)
        const i64 shifted = psi + D, lo = shifted & ((i64(1) << S.digit_bits[0]) - 1), hi = (shifted - lo) >> S.digit_bits[0];
        laid_set(digits, S.digits_laid, u, cell_row + i, E::from(from_i64(lo)));
        laid_set(digits, S.digits_laid, u, DC + cell_row + i, E::from(from_i64(hi)));
        keys[0 + 2 * (cell_row + i + DC * u)] = E::from(from_i64(lo)), keys[1 + 2 * (cell_row + i + DC * u)] = E::from(from_i64(hi));
      }
    column += m.columns, cell_row += m.rows;
  }
  std::vector<E> bits;
  BLK_REQUIRE(lkp_count(L, keys, bits), "RNG: a digit outside its range");
  Table planes(S.multiplicities.shape.main_deg, S.multiplicities.shape.main_rows, P.st.codes().base[S.multiplicities.shape.code].ell);
  for (const std::pair<unsigned, unsigned> &block : lkp_live_blocks(L.planes()))
    for (u64 t = 0; t < u64(1) << L.d; t++)
      for (u64 j = 0; j < u64(1) << block.second; j++) planes.set(0, ((u64)block.first << L.d) + j + (t << block.second), bits[block.first + j + (t << L.mu)]);
  std::vector<OracleDraft> drafts(2);
  drafts[0].shape = S.digits.shape, drafts[0].leaf_class = S.digits.leaf_class, drafts[0].main = P.engine.put(digits);
  drafts[1].shape = S.multiplicities.shape, drafts[1].leaf_class = S.multiplicities.leaf_class, drafts[1].main = P.engine.put(planes);
  for (OracleDraft &d : drafts)
    if (d.shape.mask_rows) d.mask = Send::fresh_mask(P.st.codes(), d.shape, P.coins);
  const std::vector<OracleId> ids = send_prove(P, std::move(drafts));
  BLK_REQUIRE(ids[0] == S.digits_in && ids[1] == S.digits_in + 1, "RNG: the digits' message gets another number on the channel than the shape says");
  out.digits = ids[0];
  P.st.join(group, ids[0]), P.st.join(group, ids[1]);
  out.row_owed = true, out.channel = &P.ch;
  LkpCpu heavy(L, std::move(keys), std::move(bits));
  LkpOutput left;
  lkp_prove(P, look, heavy, rng_planes_at(P.st, S, L), S.masks_from, group, left);
  out.shift = left.theta, out.alpha = left.alpha, out.rows = std::move(left.rows);
  Row on_keys = rng_key_row(P.st, S, left.keys.take());
  out.rows.push_back(on_keys);
  P.st.add_row(group, std::move(on_keys));
  return true;
}
// False: the channel has rejected the proof. True alone decides nothing: RNG compares no value from
// the proof; its claims in `group`, and the one that rng_row() returns, are still to be tested. The
// caller's contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool rng_verify(VerifierContext &V, const Public<RngShape> &fixed, const RngWeights &phi, GroupId group, RngOutput &out) {
  const RngShape &S = V.ch.get(fixed);
  rng_shape_check(V.st, S), rng_weights_check(S, phi);
  const Public<LkpPublic> look = fixed.derive(rng_lookup);
  const LkpPublic &L = V.ch.get(look);
  out.phi = phi;
  out.theta = rng_theta(V.ch, S);
  std::vector<OracleId> ids;
  if (!send_verify(V, std::vector<OracleNotice>{S.digits, S.multiplicities}, ids)) return false;
  BLK_REQUIRE(ids[0] == S.digits_in && ids[1] == S.digits_in + 1, "RNG: the digits' message gets another number on the channel than the shape says");
  out.digits = ids[0];
  V.st.join(group, ids[0]), V.st.join(group, ids[1]);
  out.row_owed = true, out.channel = &V.ch;
  LkpOutput left;
  if (!lkp_verify(V, look, rng_planes_at(V.st, S, L), S.masks_from, group, left)) return false;
  out.shift = left.theta, out.alpha = left.alpha, out.rows = std::move(left.rows);
  Row on_keys = rng_key_row(V.st, S, left.keys.take());
  out.rows.push_back(on_keys);
  V.st.add_row(group, std::move(on_keys));
  return true;
}
// The batching of the equations. RNG draws the batching element x itself, so that it comes after
// the digits' sample and the lookup, and has the channel compare the number of batched claims with
// the number that the round's planned error counts. rows_before: claims of the caller's own under
// the lower powers 1, x, .., x^(rows_before - 1) of the same x, which is returned for them.
template <class Context>
Row rng_row(Context &ctx, const Public<RngShape> &S, const RngOutput &out, E &x, u64 rows_before = 0) {
  BLK_REQUIRE(out.row_owed, "RNG: its row has been made already, or RNG has not run");
  auto from = ctx.ch.close_algebraic(R_BAT_SIGMA);
  from.require_rows(rows_before + ctx.ch.get(S).equations());
  x = from.draw();
  out.row_owed = false;
  return rng_row(ctx.st, ctx.ch.get(S), out.phi, out.theta, out.digits, x, rows_before);
}

// The honest packed matrix and reads, written into tables that the caller sends.
inline void rng_lay_packed(Table &table, const RngShape &S, size_t mu, const std::vector<i64> &cells) {
  const RngMatrix &m = S.matrices[mu];
  for (u64 p = 0; p < S.packed_rows(mu); p++)
    for (u64 c = 0; c < m.columns; c++) {
      F sum = F::zero();
      for (unsigned l = 0; l < S.lanes && p * S.lanes + l < m.rows; l++)
        sum = sum + from_u64(u64(1) << (S.log_radix * l)) * from_i64(cells[(p * S.lanes + l) * m.columns + c]);
      laid_set(table, m.laid, p, c, E::from(sum));
    }
}
inline void rng_lay_reads(Table &table, const RngShape &S, const RngWeights &phi, const std::vector<std::vector<i64>> &cells) {
  for (size_t q = 0; q < S.reads.size(); q++) {
    const RngRead &r = S.reads[q];
    const RngMatrix &m = S.matrices[r.matrix];
    const u64 L = r.length, parts = m.rows / L;
    for (u64 c = 0; c < m.columns; c++)
      for (u64 g = 0; g < parts; g++) {
        E sum = E::zero();
        for (u64 i = 0; i < L; i++) sum += phi[q][i] * E::from(from_i64(cells[r.matrix][(g * L + i) * m.columns + c]));
        laid_set(table, S.reads_laid, 0, S.read_first(q) + c * parts + g, sum);
      }
  }
}

}  // namespace blk
