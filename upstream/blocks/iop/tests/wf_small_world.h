// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Two test fixtures (`world` in identifiers) that run WF (blocks/iop/wf.h), the committer's range
// proof by lookups on the matrix in a committed table (`Index` in identifiers), at a very small
// size, for tests that run the prover very often: the exact test of hiding
// (blocks/tests/hiding/exact.h) needs a run for each private random field element.
//
// Under wf_small_world(), the block SEND (blocks/iop/send.h) sends and samples the index-mask
// message (mask rows only, to mask claims on the matrix), then WF's multiplicities; the lookup
// follows, and the run ends: no proximity test, no batching of the claims, no test on the table
// (wf_small_whole_world() has all three).
// Such a run proves nothing. It sends the hash chain's salt, two Merkle roots, the samples'
// answers and the look-up's scalars. The two pending claims on the matrix (`TensorRow`; `rows` in
// wf_rows_prove()) that WF leaves are computed by both sides and never sent; each uses two entries
// of the index-mask message.
#pragma once
#ifndef BLK_WF_ROWS_FOR_A_TEST
#error "say #define BLK_WF_ROWS_FOR_A_TEST 1 before the first include: this world runs WF without its proximity test"
#endif
#include "chain/closed.h"
#include "iop/tests/wf_world.h"

namespace blk {

// `lin.input` is the index-mask message; WF reads the matrix's size from `lin.S`.
struct WfSmall {
  LinWorld lin{TableTest::ONE_TIME};  // only `codes`, `S`, `input` and `configuration` are used
  WfPublic wf;
  Planned plan;
  IndexName table;  // made up: no table is committed
};
// The matrix: 4 rows, 2^(v + k_lo) = 4 columns. Fields: `LinCase` (blocks/iop/tests/lin_world.h).
//                                        name     v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws ahead
static const LinCase WF_SMALL_LIN = {"WF alone, small", 1, 1,   4,   1,   1,  1,  1,  2,   16,  8,  8, true,  1, 0,     1, true};
// The same case with two mask rows in the index-mask message. The last two values are
// `tight_tails` and `index_mask_rows`.
static const LinCase WF_SMALL_LIN_TWO_MASK_ROWS = {"WF alone, small, two mask rows", 1, 1, 4, 1, 1, 1, 1, 2, 16, 8, 8, true, 1, 0, 1, true, false, 2};

// The one lookup, its oracle message and the proximity test's stages. Appends their codes to
// `lin.codes`. A key is an entry (a toy byte, -2 .. 1) + 2, in the lookup table 0 .. 3. A
// multiplicity has 6 bits (`live`) in 2^mu = 8 slots. Two parts of 4 places: part 0 has matrix
// rows 0, 1, 2 and padding; part 1 has padding, row 3, padding, padding. The padding key is 1:
// with 0, the padding's share, which the lookup subtracts from a pending claim's value, would be 0.
// mask_length: the message length of the oracle message's mask code. A shorter code gives the
// message more mask rows: 3 at 64, 9 at 16.
inline WfPublic wf_small_public(LinWorld &lin, size_t mask_length = 64) {
  const WfMatrix m = wf_matrix_of(lin.S);
  WfLookup bytes;
  bytes.name = "bytes", bytes.d = 2, bytes.mu = 3, bytes.live = 6;
  bytes.digit_weight = {F::one()}, bytes.offset = from_u64(2), bytes.padding_key = from_u64(1);
  bytes.parts = {WfPart{2, {WfRun{0, 0, 3, 0}}}, WfPart{2, {WfRun{1, 3, 1, 0}}}};
  WfPublic x;
  x.lookups = {bytes};
  const unsigned draws = WF_SMALL_LIN.draws;
  lin.codes.mask.push_back(toy_code(mask_length, tail_for_positions(draws), 0));
  lin.codes.base.push_back(toy_code(32, tail_for_positions(draws), lin.codes.mask.back().log_n));
  OracleShape s;
  s.name = "multiplicities", s.code = (CodeId)(lin.codes.base.size() - 1), s.mask_code = (CodeId)(lin.codes.mask.size() - 1);
  s.main_deg = 1, s.answers = 1;
  s.main_rows = (size_t)wf_layout(x, lin.codes.base[s.code].ell).back().row + 1;
  s.mask_rows = Send::mask_rows_for(wf_own_masks(x, m), lin.codes.mask[s.mask_code].ell, s.answers);
  x.message = OracleNotice{s, 0};
  x.message_id = 1;  // after the index-mask message
  x.stages = lin_stages(lin.codes, s.code, s.mask_code, {s}, draws);
  x.choice.masks_ride = false;
  return x;
}

inline WfSmall wf_small_world() {
  WfSmall w;
  w.lin = lin_world(WF_SMALL_LIN, TableTest::ONE_TIME);
  const WfMatrix m = wf_matrix_of(w.lin.S);
  const WfPublic &x = w.wf = wf_small_public(w.lin);  // `stages` is bound too, but never run
  Plan plan = toy_empty_plan();
  lin_plan_send(plan, w.lin.codes, {w.lin.input});
  lin_plan_send(plan, w.lin.codes, wf_message(x));
  const LkpPublic l = wf_lookups(x, m)[0];
  plan_lookup(plan, x.lookups[0].name, LookupSize{0, l.d, l.mu, false, l.live, l.parts});
  w.plan = plan;
  w.table.commitment.assign(DIGEST_BYTES, 7);
  w.table.record.y.assign(w.lin.S.fold.index.rows() * w.lin.S.fold.index.answers, E::one());
  return w;
}

// The same lookup and matrix in the committer's entire one-time run, for a very small key file:
// index-mask message; WF with its proximity test; BATT, which batches the pending claims into one;
// LIN, the test on the table, for a batch of one. For wf_session() and wf_verify_view()
// (blocks/iop/tests/wf_world.h; no test passes them this fixture) and, as
// KeyPlans{lin.first, lin.shared, lin.second}, for make_key_under() and check_key_under()
// (blocks/frame/check_key.h).
// two_mask_rows: the index-mask message has two mask rows instead of one, and the multiplicities'
// message nine instead of three (a case of blocks/tests/hiding/key_exact_test.cpp).
inline WfWorld wf_small_whole_world(bool two_mask_rows = false) {
  WfWorld w;
  w.lin = lin_world(two_mask_rows ? WF_SMALL_LIN_TWO_MASK_ROWS : WF_SMALL_LIN, TableTest::ONE_TIME);
  w.wf = wf_small_public(w.lin, two_mask_rows ? 16 : 64);
  for (const WfLookup &l : w.wf.lookups) w.pending += l.parts.size();
  Plan first = toy_empty_plan();
  lin_plan_send(first, w.lin.codes, {w.lin.input});
  wf_toy_plan(first, w.lin.codes, w.wf, wf_matrix_of(w.lin.S), WF_SMALL_LIN.draws);
  plan_member_rows(first, "rows on the table", w.pending, w.lin.S.k_col());
  lin_plan_send(first, w.lin.codes, lin_message_w(w.lin.S));
  lin_plan_seal(first);
  w.lin.first = first;
  return w;
}

// A witness: every entry is in -2 .. 1, derived from the seed `key`. Two seeds give two witnesses
// of one public statement.
inline LinMatrix wf_small_matrix(const LinShape &S, u64 key) {
  LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
  M.a.resize(M.rows * M.columns);
  TestCoins data(key);
  for (F &cell : M.a) {
    const long long v = (long long)(data.word() % 4) - 2;
    cell = v < 0 ? -from_u64((u64)-v) : from_u64((u64)v);
  }
  return M;
}
inline LinMatrix wf_small_matrix(const WfSmall &w, u64 key) { return wf_small_matrix(w.lin.S, key); }

struct WfSmallBound {
  Public<OracleNotice> input;
  Public<WfPublic> wf;
  LinPublic lin;
};
inline WfSmallBound wf_small_bind(Binder &b, const WfSmall &w) {
  WfSmallBound x;
  b.configuration(w.lin.configuration);
  x.input = b.parameter("the index-mask message", w.lin.input);
  x.wf = b.parameter("the proof of the ranges", w.wf);
  x.lin = lin_bind(b, w.lin.codes, w.lin.S, w.table);  // last: it closes the parameters
  return x;
}
constexpr u64 WF_SMALL_LABEL = 1;

// All private randomness, the chain's salt included, comes from `coins`; every byte sent goes to
// `out`. Returns false if an entry of M is out of range; `out` then has no complete run.
// pads: if not null, receives each sample's stored pad, for pads_fault() (blocks/iop/send.h).
inline bool wf_small_prove(const WfSmall &w, const LinMatrix &M, Engine &engine, Coins &coins, Writer &out, std::vector<PadLook> *pads = nullptr) {
  Binder b(WF_SMALL_LABEL, w.plan);
  const WfSmallBound x = wf_small_bind(b, w);
  ProverRounds ch(b, random_digest(coins), out);
  ProverContext P(x.lin.codes, ch, engine, coins);
  P.pads = pads;
  std::vector<OracleDraft> drafts;
  drafts.push_back(lin_draft(P, ch.get(x.input).shape, false, ch.get(x.input).leaf_class, {}));
  const OracleId masks = send_prove(P, std::move(drafts))[0];
  const LinMatrix *m = &M;
  WfCpu heavy(w.wf, wf_matrix_of(w.lin.S), [m](u64 row, u64 column) { return m->at(row, column); });
  WfOutput left;
  FoldMask none;
  if (!wf_rows_prove(WfRowsAlone(), P, x.wf, x.lin, masks, heavy, left, none)) return false;
  (void)left.pending.take();  // not tested here, but taken: otherwise `WfPending` ends the process
  ch.finish();
  return true;
}
// The verifier's side, for run_verifier() (blocks/chain/closed.h), which also rejects leftover
// bytes. True means that the run is well formed, and nothing else: no claim is tested. notes: if
// not null, receives each pending claim's value (`target`).
inline bool wf_small_decide(VerifierRounds &ch, const WfSmallBound &x, Notes *notes) {
  VerifierContext V(x.lin.codes, ch);
  std::vector<OracleId> ids;
  if (!send_verify(V, std::vector<OracleNotice>{ch.get(x.input)}, ids)) return false;
  WfOutput left;
  FoldMask none;
  if (!wf_rows_verify(WfRowsAlone(), V, x.wf, x.lin, ids[0], left, none)) return false;
  for (const TensorRow &r : left.pending.take())
    if (notes) notes->emplace_back("a pending row's target", r.target);
  return true;
}
inline bool wf_small_well_formed(const WfSmall &w, const std::vector<u8> &bytes, Notes *notes = nullptr) {
  return run_verifier(WF_SMALL_LABEL, w.plan, w, wf_small_bind, wf_small_decide, bytes.data(), bytes.size(), nullptr, notes);
}

}  // namespace blk
