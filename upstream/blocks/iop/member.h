// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// A request's own part of LIN (`member` in names). LIN proves linear claims of a batch's requests
// on the committed table (blocks/iop/lin.h). A request's sub-proofs leave J >= 1 pending claims on
// the table's matrix M (`TensorRow`, blocks/iop/batt.h; "pending row" in names and messages):
//   < g (x) eq(psi, .), M > + (mask entries) = value (`target`); g weighs rows, eq(psi, .) columns.
// On the request's first hash chain (a Fiat-Shamir transcript), member_first_prove() and
// member_first_verify() batch them into one with the block BATT if J > 1 and run LIN's first part,
// which sends W = g^T M. The caller then seals the chain (ends it in a digest on which later chains
// build) and calls LIN's shared and second parts itself; the second ends in the request's own
// proximity test or tests. Checks, errors, hiding: BATT's and LIN's.
// The claims' mask entries (random field elements) and BATT's (batt_masks() of them) are all in the
// request's index-mask message, an oracle message of mask rows only, sent earlier on that chain.
// The caller's duties, not checked: everything in a pending claim follows from what the chain has
// absorbed before the call; and `LinFirst`, a plain struct that keeps the claim, reaches LIN's
// second part unchanged. Tests, J = 1 too: blocks/iop/tests/member_test.cpp.
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/batt.h"
#include "iop/lin.h"

namespace blk {

// A pending claim in LIN's form. Consumes the claim's mask entries: each is named by a `MaskSlot`
// (blocks/iop/mask.h), which can be moved but not copied.
inline LinClaim member_claim(TensorRow &&row) {
  LinClaim c;
  c.g = std::move(row.g), c.psi = std::move(row.psi), c.target = row.target;
  for (MaskTerm<E> &m : row.masks) c.add_mask(std::move(m.slot), m.coeff);
  row.masks.clear();
  return c;
}

// Row weights given as a vector on a range of rows (`stretch` in names and messages; the `cube` is
// the indices below 2^k_row): values[i] on row first_row + i, 0 on the rest of the cube. The range
// is cut into aligned blocks of 2^d rows, at most two of each size. A block of two or more rows is
// one `Tensor` whose dense factor is a slice of the shared vector, so nothing is copied; a block of
// one row has its value read here and stored as the Tensor's scalar. (As a vector over all rows,
// each sub-proof's weights would take 2^k_row elements.)
// Callers use member_weights_on_rows(): this function does not know how many rows the matrix has.
// Tested directly by blocks/iop/tests/member_rows_test.cpp.
namespace member_inner {
inline std::vector<Tensor> stretch_in_the_cube(unsigned k_row, u64 first_row, std::shared_ptr<const std::vector<E>> values) {
  BLK_REQUIRE(values && k_row < 63 && first_row + values->size() <= (u64(1) << k_row), "member: the stretch of rows does not lie in the cube");
  std::vector<Tensor> g;
  for (u64 at = first_row, end = first_row + values->size(); at < end;) {
    unsigned d = 0;  // the largest aligned block that begins at 'at' and ends by 'end'
    while (d < k_row && (at & ((u64(2) << d) - 1)) == 0 && at + (u64(2) << d) <= end) d++;
    Tensor t;
    if (d == 0) {
      t.scalar = (*values)[at - first_row];
      t.insert_choice(0, k_row, at);
    } else {
      t.push_dense(d);
      t.insert_choice(d, k_row - d, at >> d);
      t.set_values(values, at - first_row);
    }
    g.push_back(std::move(t));
    at += u64(1) << d;
  }
  return g;
}
}  // namespace member_inner

// stretch_in_the_cube() for callers: ends the process unless the range is inside the matrix. LIN
// silently ignores weights from row matrix_rows on (rightly for weights in product form, which
// extend over all 2^k_row indices), so a caller whose own layout of the rows has more of them than
// `LinShape` counts would lose its claims on the surplus rows.
// Not guarded: first_row + values->size() can overflow 64 bits. The wrapped sum can pass this check
// and stretch_in_the_cube()'s, and the result is then empty: no weights, and no message.
inline std::vector<Tensor> member_weights_on_rows(const LinShape &S, u64 first_row, std::shared_ptr<const std::vector<E>> values) {
  BLK_REQUIRE(values && first_row + values->size() <= S.matrix_rows, "member: a stretch of row weights does not lie inside the matrix");
  return member_inner::stretch_in_the_cube(S.k_row(), first_row, std::move(values));
}

// The pending claim that a sub-proof leaves (claim (E) of batt_shown.h, for a program that
// implements that block itself): < g (x) eq(psi, .), M > + beta c = v.
// c: the entry of the index-mask message that the sub-proof has taken; it is consumed.
inline TensorRow member_pending_row(std::vector<Tensor> g, std::vector<E> psi, MaskSlot &&c, const E &beta, const E &v) {
  TensorRow r;
  r.g = std::move(g), r.psi = std::move(psi), r.target = v;
  r.masks.push_back({std::move(c), beta});
  return r;
}

// The prover's access to the matrix, for one request only. Called once each, in this order:
// begin(), sumcheck() if J > 1, combine().
// The vectors W_j = g_j^T M, their copies in the sumcheck and the combination are hidden values:
// an implementation for real weights erases its own copies at combine() and when it is destroyed;
// the combination is the caller's (it goes into LIN's oracle message W).
class MemberHeavy {
 public:
  virtual ~MemberHeavy() = default;
  // An implementation may compute the W_j in begin(), the only call in member_first_prove() that
  // may read an entry of a dense weight vector (`Tensor::values`): the function itself copies the
  // Tensors, multiplies their scalars and reads sizes. So if the implementation has the W_j from
  // elsewhere and no range has a block of one row (whose value is read when its Tensor is made), a
  // program may write these entries on other threads until LIN's second part reads them.
  virtual void begin(const std::vector<std::vector<Tensor>> &g) = 0;  // g[j]: weights of claim j
  // The prover of BATT's sumcheck over the W_j, without masks (`BattSumcheck` in batt.h).
  virtual std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &, const std::vector<Weight<E>> &, const std::vector<E> &chi,
                                                      const std::vector<std::vector<E>> &psi) = 0;
  virtual std::vector<E> combine(const std::vector<E> &a) = 0;  // sum_j a_j W_j, one entry for each column
};

// Plans BATT's rounds for J = rows claims (none if J = 1). LIN's first part is not planned here:
// blocks/frame/files.h plans it next, with lin_plan_first().
inline void plan_member_rows(Plan &p, const std::string &where, u64 rows, unsigned k_col) {
  if (rows > 1) plan_batt(p, where, rows, k_col);
}

// Checks the pending claims and the index-mask message against `LinShape`. Both sides run it first;
// a mismatch ends the process. (BATT compares the claims with each other.) Of the message that the
// caller names, only the shape is checked (`answers` > 0 means sampled: the block SEND takes an
// out-of-domain sample of the message). Every mask entry of every claim must be in it because an
// entry of another message would draw that message into the request's own group (`LinFirst::own`,
// tested in LIN's second part). In a request's proof the index-mask message is sent in the first
// prover message and is in no segment's group (`ReqShape::at_start`, blocks/iop/req.h; a segment
// is one sub-proof's part of the chain).
inline void member_rows_fit(const Statement &st, const LinShape &S, OracleId index_masks, const std::vector<TensorRow> &pending) {
  const OracleShape &o = st.oracle(index_masks).shape;
  BLK_REQUIRE(o.main_rows == 0 && o.mask_rows > 0 && o.answers > 0 && o.mask_code == S.own_mask_code,
              "member: the index-mask message is no sampled message of mask rows alone under the own group's mask code");
  for (const TensorRow &r : pending)
    for (const MaskTerm<E> &m : r.masks) BLK_REQUIRE(m.slot.oracle() == index_masks, "member: a pending row names a mask entry outside the index-mask message");
  BLK_REQUIRE(!pending.empty(), "member: no pending row");
  BLK_REQUIRE(batt_k_col(pending) == S.k_col(), "member: the pending rows' points do not fit the matrix's columns");
  for (const TensorRow &r : pending)
    for (const Tensor &t : r.g) BLK_REQUIRE(t.vars() == S.k_row(), "member: a pending row's weights do not fit the matrix's rows");
}

// The prover's side of the first part. x: LIN's public inputs as bound to this chain (lin_bind() or
// its two halves, blocks/iop/lin.h). In a proof file's first chain the call follows the request's
// segments and the proximity test that ends them. The committer's one-time test in the key file is
// a batch of one: make_key_under() (blocks/frame/check_key.h) calls this function on its range
// proof's claims, and check_key_under() calls the verifier's side.
inline LinFirst member_first_prove(ProverContext &P, const LinPublic &x, OracleId index_masks, std::vector<TensorRow> pending, MemberHeavy &heavy) {
  member_rows_fit(P.st, P.ch.get(x.shape), index_masks, pending);
  std::vector<std::vector<Tensor>> g;
  for (const TensorRow &r : pending) g.push_back(r.g);
  heavy.begin(g);
  std::vector<E> a{E::one()};
  TensorRow one;
  if (pending.size() > 1) {
    const ProverContext *p = &P;
    ProverRounds &channel = P.ch;
    one = batt_prove(
        channel, std::move(pending), P.st.masks(index_masks), [p](u32 oracle, u32 index) { return p->behind[oracle].mask.entry(index); },
        [&heavy](const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &chi, const std::vector<std::vector<E>> &psi) {
          return heavy.sumcheck(s, vars, chi, psi);
        },
        &a);
  } else {
    one = std::move(pending[0]);
  }
  std::vector<E> w = heavy.combine(a);
  return lin_first_prove(P, x, member_claim(std::move(one)), std::move(w));
}

// The verifier's side: false if BATT or LIN's first part rejects. No parameter carries data of
// another request of the batch.
[[nodiscard]] inline bool member_first_verify(VerifierContext &V, const LinPublic &x, OracleId index_masks, std::vector<TensorRow> pending, LinFirst &out) {
  member_rows_fit(V.st, V.ch.get(x.shape), index_masks, pending);
  TensorRow one;
  if (pending.size() > 1) {
    if (!batt_verify(V.ch, std::move(pending), V.st.masks(index_masks), one)) return false;
    if (V.notes) {  // to compare another verifier with this one block by block; changes nothing
      V.note("BATT: target", one.target), V.note("BATT: gamma_0", one.psi.front()), V.note("BATT: the first weight's scalar", one.g.front().scalar);
      V.note("BATT: coefficient of its first mask entry", one.masks.front().coeff), V.note("BATT: coefficient of the last mask term", one.masks.back().coeff);
    }
  } else {
    one = std::move(pending[0]);
  }
  V.note("member: the claim's target", one.target);
  return lin_first_verify(V, x, member_claim(std::move(one)), out);
}

}  // namespace blk
