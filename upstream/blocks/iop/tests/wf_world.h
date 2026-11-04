// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture (`world` in identifiers): the committer's one-time proof at toy size. It is meant to
// show that every entry of the matrix encoded in a committed table (`Index` in identifiers) is in
// its range. Its steps: the index-mask message (masks for the claims on the table), WF (the range
// proof by lookups), BATT (joins the pending claims that WF leaves: `TensorRow` in identifiers,
// "rows on the table" in strings) and LIN (checks the joined claim against the table) for a batch
// of one. Hash chains, trees and proximity tests (`iopp` in identifiers) are not mocked; the plans
// of rounds are hand-written. Before any include, define BLK_PLANS_BY_HAND_FOR_A_TEST and
// BLK_TESTED_TABLE_FOR_A_TEST (for on_a_tests_word()).
// The toy matrix has the layout of blocks/frame/wf_units.h in small: 22 rows of 2^4 columns; every
// entry is a toy byte in -4 .. 3; a weight in -16 .. 15 is two of them, low + 8 high. A unit stores
// its weights' low digits, then their high digits: rows 0 .. 11 (six rows of weights) and 12 .. 17
// (three). Rows 18 .. 21 are bytes only.
#pragma once
#include "iop/tests/member_world.h"
#include "iop/wf.h"

namespace blk {
// lin: WF_CASES' columns are `LinCase`'s fields (lin_world.h). The matrix has 2^(v + k_lo) = 2^4
// columns in both cases. col: the table's columns that a batch opens. bat: the batches that the
// table's random tails cover. req = 1: the batch of one. ans: the answers per row of the table in
// its record (stored answers to samples of the table's rows).
// masks_ride: the masks of WF's proximity test go out with earlier prover messages (`IoppChoice`).
struct WfCase {
  LinCase lin;
  bool masks_ride;
};
// Row i < 9 of weights: the matrix row of its low (digit 0) or high (digit 1) digits.
inline u64 wf_toy_row(u64 i, unsigned digit) { return i < 6 ? i + 6 * digit : 12 + (i - 6) + 3 * digit; }
constexpr u64 WF_TOY_WEIGHT_ROWS = 9, WF_TOY_FIRST_BYTES_ONLY = 18;
static const WfCase WF_CASES[] = {
    //  name                               v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws ahead
    {{"masks ahead",                       2, 2,  22,   1,   1,  4,  4,  2,   16,  8,  8, true,  1, 0,     3, true}, true},
    {{"masks in their own messages",         2, 2, 22,  0,   1,  3,  4,  3,   16,  8,  8, true,  1, 0,     2, true}, false},
};
// pending: the number of pending claims that WF leaves, one for each part of each lookup.
struct WfWorld {
  LinWorld lin{TableTest::ONE_TIME};
  WfPublic wf;
  size_t pending = 0;
};

// Two lookups (a place is a row of keys, one per column):
//   bytes    entry + 4 is in 0 .. 7, for every entry. Parts of 16, 4 and 4 places. The top bit of a
//            place selects the first or second row of a pair (low and high digits, or neighbours
//            among rows 18 .. 21), so any padding ends each half of a part. 9 multiplicity bits
//            (`live`) in 2^mu = 16 slots. `first_bound`: the sumcheck on a part's last key
//            level binds variables 1 and 0 first, in this order (`WfLookup`, blocks/iop/wf.h).
//   weights  low + 8 high + 16 is in 0 .. 31. Parts of 8 and 2 places. The first covers the first
//            unit's six rows of weights and two of the second unit's, so a part crosses a unit's
//            border; the second has the ninth row and a place of padding. 8 multiplicity bits
//            (2^mu).
inline WfPublic wf_toy_public(const WfCase &t, LinWorld &w) {
  WfPublic x;
  WfLookup bytes, weights;
  bytes.name = "bytes", bytes.d = 3, bytes.mu = 4, bytes.live = 9;
  bytes.digit_weight = {F::one()}, bytes.offset = from_u64(4), bytes.padding_key = from_u64(3);  // not 0, so that the padding's share of a claim is not 0
  // `WfRun`: first place, first row, places, row stride between digits.
  // An "entry" below is a pair of rows: 0 .. 8 by wf_toy_row(); 9 is (18, 19); 10 is (20, 21).
  bytes.parts = {WfPart{4, {WfRun{0, 0, 6, 0}, WfRun{6, 12, 2, 0}, WfRun{8, 6, 6, 0}, WfRun{14, 15, 2, 0}}},  // entries 0 .. 7
                 WfPart{2, {WfRun{0, 14, 1, 0}, WfRun{1, 18, 1, 0}, WfRun{2, 17, 1, 0}, WfRun{3, 19, 1, 0}}},  // entries 8, 9
                 WfPart{2, {WfRun{0, 20, 1, 0}, WfRun{2, 21, 1, 0}}}};                                         // entry 10; places 1 and 3 are padding
  bytes.first_bound = {1, 0};
  weights.name = "weights", weights.d = 5, weights.mu = 3;
  weights.digit_weight = {F::one(), from_u64(8)}, weights.offset = from_u64(16), weights.padding_key = from_u64(17);
  weights.parts = {WfPart{3, {WfRun{0, 0, 6, 6}, WfRun{6, 12, 2, 3}}}, WfPart{1, {WfRun{0, 14, 1, 3}}}};
  x.lookups = {bytes, weights};

  const unsigned draws = t.lin.draws;
  w.codes.mask.push_back(toy_code(256, tail_for_positions(draws), 0));
  w.codes.base.push_back(toy_code(256, tail_for_positions(draws), w.codes.mask.back().log_n));
  OracleShape s;
  s.name = "multiplicities", s.code = (CodeId)(w.codes.base.size() - 1), s.mask_code = (CodeId)(w.codes.mask.size() - 1);
  s.main_deg = 1, s.main_rows = 2, s.mask_rows = 4, s.answers = 1;
  x.message = OracleNotice{s, 0};
  x.message_id = 1;  // after the index-mask message
  x.stages = lin_stages(w.codes, s.code, s.mask_code, {s}, draws);
  x.choice.masks_ride = t.masks_ride;
  return x;
}

// Plans WF's rounds. The rounds that sample a sent message and those of the proximity test are
// planned by hand with error 0 (iopp_support.h).
inline void wf_toy_plan(Plan &p, const Codes &codes, const WfPublic &x, const WfMatrix &m, unsigned draws) {
  lin_plan_send(p, codes, wf_message(x));
  for (size_t i = 0; i < x.lookups.size(); i++) {
    const LkpPublic l = wf_lookups(x, m)[i];
    plan_lookup(p, x.lookups[i].name, LookupSize{0, l.d, l.mu, false, l.live, l.parts});
  }
  toy_plan_iopp(p, codes, x.stages, draws, x.choice, x.choice.masks_ride, wf_rows_batched(x, m));
}
// lin_world() plans the shared chain and the second chain; the first chain's plan is replaced here.
// The three chains are introduced above wf_session().
inline WfWorld wf_world(const WfCase &t) {
  WfWorld w;
  w.lin = lin_world(t.lin, TableTest::ONE_TIME);  // the committer's one-time test on its table
  w.wf = wf_toy_public(t, w.lin);
  for (const WfLookup &l : w.wf.lookups) w.pending += l.parts.size();
  Plan first = toy_empty_plan();
  lin_plan_send(first, w.lin.codes, {w.lin.input});
  wf_toy_plan(first, w.lin.codes, w.wf, wf_matrix_of(w.lin.S), t.lin.draws);
  plan_member_rows(first, "rows on the table", w.pending, w.lin.S.k_col());
  lin_plan_send(first, w.lin.codes, lin_message_w(w.lin.S));
  lin_plan_seal(first);
  w.lin.first = first;
  return w;
}

// wf_matrix(): every entry is in its range, and both ends of both ranges occur. Helpers come first.
inline F wf_signed(long long v) { return v < 0 ? -from_u64((u64)-v) : from_u64((u64)v); }
inline void wf_put_digits(LinMatrix &M, u64 i, u64 column, long long low, long long high) {
  M.a[wf_toy_row(i, 0) * M.columns + column] = wf_signed(low), M.a[wf_toy_row(i, 1) * M.columns + column] = wf_signed(high);
}
inline void wf_put_weight(LinMatrix &M, u64 i, u64 column, long long weight) {
  const long long low = ((weight + 4) % 8 + 8) % 8 - 4;
  wf_put_digits(M, i, column, low, (weight - low) / 8);
}
inline LinMatrix wf_matrix(const LinShape &S, u64 seed) {
  LinMatrix M{S.matrix_rows, u64(1) << S.k_col(), {}};
  M.a.resize(M.rows * M.columns);
  TestCoins data(seed);
  for (u64 c = 0; c < M.columns; c++) {
    for (u64 i = 0; i < WF_TOY_WEIGHT_ROWS; i++) wf_put_weight(M, i, c, (long long)(data.word() % 32) - 16);
    for (u64 row = WF_TOY_FIRST_BYTES_ONLY; row < M.rows; row++) M.a[row * M.columns + c] = wf_signed((long long)(data.word() % 8) - 4);
  }
  wf_put_weight(M, 0, 0, -16), wf_put_weight(M, WF_TOY_WEIGHT_ROWS - 1, M.columns - 1, 15);
  M.a[(M.rows - 1) * M.columns] = wf_signed(-4), M.a[(M.rows - 1) * M.columns + 1] = wf_signed(3);
  return M;
}

// What wf_bind_first() puts into the first chain's first query: parameters and one statement item.
struct WfBound {
  Public<WfPublic> wf;
  LinPublic lin;
  Public<OracleNotice> input;
  Public<LinBindingShape> binding;
};
inline WfBound wf_bind_first(Binder &b, const WfWorld &w, const IndexName &table) {
  WfBound x;
  b.configuration(w.lin.configuration);
  x.input = b.parameter("the index-mask message", w.lin.input), x.binding = b.parameter("the binding tree", w.lin.binding);
  x.wf = b.parameter("the proof of the ranges", w.wf);
  x.lin = lin_bind(b, w.lin.codes, w.lin.S, table);  // last: its item `table` closes the parameters
  return x;
}

// What a run notes for its test. targets, psi: each pending claim's value and point, on both sides.
// pads: the random entries that send_prove() writes into a sampled message to mask the sample's
// answers; pads_fault() (both in blocks/iop/send.h) compares two runs' pads.
struct WfSeen {
  std::vector<E> targets;
  std::vector<std::vector<E>> psi;
  u32 own_masks = 0, index_masks = 0;  // index-mask entries that WF takes; own_masks is never set
  bool served = true;                  // false: prover gave up (e.g. key out of range); empty view
  std::vector<PadLook> pads;           // prover only: each sampled message's pads, for `pads_fault`
};

// Makes the `WfHeavy` (the lookups' large computations) for one run on M, which must outlive it.
using WfHeavyMaker = std::function<std::unique_ptr<WfHeavy>(const WfPublic &x, const WfMatrix &size, const LinMatrix &M)>;
inline std::unique_ptr<WfHeavy> wf_heavy_cpu(const WfPublic &x, const WfMatrix &size, const LinMatrix &M) {
  const LinMatrix *m = &M;
  return std::unique_ptr<WfHeavy>(new WfCpu(x, size, [m](u64 row, u64 column) { return m->at(row, column); }));
}

// LIN's three parts run on three hash chains (Fiat-Shamir transcripts; blocks/iop/lin.h): the
// first, which here has the index-mask message, WF and BATT before LIN's first part, then the
// shared and the second. A first query is the hash input that starts a chain; a seal is the digest
// that ends one. The binding tree is a Merkle tree over the first chains' seals (here one seal);
// its root goes into the shared chain's first query, and the first and shared chains' seals into
// the second's. The table, the binding tree and the shared and second parts are as in
// blocks/iop/tests/lin_world.h and blocks/iop/tests/member_world.h.
//
// The committer's run on the committed table `index`. spoken_of: the matrix from which the prover
// computes WF, BATT and LIN's message W (honest: the committed one). Engine and makers: CPU or GPU.
inline LinView wf_session(const WfWorld &w, IndexHolder &index, const IndexRecord &rec, const LinMatrix &spoken_of, u64 coin_seed, Engine &engine, WfSeen &seen,
                          const WfHeavyMaker &ranges_maker = wf_heavy_cpu, const MemberHeavyMaker &products_maker = member_heavy_cpu) {
  const LinShape &S = w.lin.S;
  TestCoins coins(coin_seed);
  const IndexName table{index.commitment(), rec};
  LinView view;
  LinBinding binding(w.lin.binding, w.lin.configuration, coins);
  std::unique_ptr<LinBegun> begun;
  BLK_REQUIRE(lin_begin(index, S, begun), "the keeper does not serve");

  Writer first_bytes, second_bytes, shared_bytes;
  Binder b(1, w.lin.first);
  const WfBound bound = wf_bind_first(b, w, table);
  ProverRounds ch(b, toy_random_bytes(coin_seed + 100), first_bytes);
  ProverContext P(bound.lin.codes, ch, engine, coins);
  P.pads = &seen.pads;
  std::vector<OracleDraft> drafts;
  drafts.push_back(lin_draft(P, ch.get(bound.input).shape, false, ch.get(bound.input).leaf_class, {}));
  const OracleId masks = send_prove(P, std::move(drafts))[0];

  const std::unique_ptr<WfHeavy> ranges = ranges_maker(w.wf, wf_matrix_of(S), spoken_of);
  WfOutput left;
  const u32 index_before = P.st.masks(masks).left();
  if (!wf_prove(P, bound.wf, bound.lin, masks, *ranges, left)) {
    seen.served = false;
    return view;
  }
  seen.index_masks = index_before - P.st.masks(masks).left();
  for (const TensorRow &r : left.pending.look()) seen.targets.push_back(r.target), seen.psi.push_back(r.psi);

  const std::unique_ptr<MemberHeavy> products = products_maker(S, spoken_of);
  LinFirst lin = member_first_prove(P, bound.lin, masks, left.pending.take(), *products);
  Sealed seal;
  BLK_REQUIRE(ch.seal(R_SEAL, seal), "a prover's channel does not refuse");
  ch.finish();  // every chain's end is run, earlier chains first
  binding.put(0, seal.digest());

  Binder shared_b(LIN_LABEL_SHARED, w.lin.shared);
  shared_b.configuration(w.lin.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.lin.codes, S, table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, coins);
  shared_P.pads = &seen.pads;
  LinShared shared;
  lin_shared_prove(shared_P, shared_x, w.lin.test, index, std::move(begun), shared);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish();

  Binder second_b(LIN_LABEL_SECOND, w.lin.second);
  second_b.configuration(w.lin.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  ProverRounds second_ch(second_b, toy_random_bytes(coin_seed + 200), second_bytes);
  ProverContext P2 = lin_go_on(std::move(P), second_ch);
  P2.pads = &seen.pads;
  const LinSecond second = lin_second_prove(P2, bound.lin, lin, shared_P, shared);
  lin_second_tests_prove(P2, bound.lin, lin, second);
  second_ch.finish();
  P2.st.require_all_tested();
  Writer path;
  binding.write_path(path, 0);
  lin_release(shared_P, shared);
  view.first = first_bytes.bytes(), view.path = path.bytes(), view.shared = shared_bytes.bytes(), view.second = second_bytes.bytes();
  return view;
}

// *why: the step of an early rejection, else ""; wf_test.cpp only prints it.
inline bool wf_verify_view(const WfWorld &w, const IndexKey &key, const LinView &view, WfSeen &seen, const char **why = nullptr) {
  const IndexName table{key.commitment(), key.rec};
  const char *unused;
  const char *&reason = why ? *why : unused;
  reason = "";
  Reader first_bytes(view.first.data(), view.first.size());
  Binder first_b(1, w.lin.first);
  const WfBound x = wf_bind_first(first_b, w, table);
  VerifierSession session;  // one verdict on the three chains
  VerifierRounds first_ch(first_b, first_bytes, session);
  VerifierContext first_V(x.lin.codes, first_ch);
  std::vector<OracleId> ids;
  if (!send_verify(first_V, std::vector<OracleNotice>{first_ch.get(x.input)}, ids)) return reason = "first part: the index-mask message", false;
  WfOutput left;
  const u32 index_before = first_V.st.masks(ids[0]).left();
  if (!wf_verify(first_V, x.wf, x.lin, ids[0], left)) return reason = "first part: the proof of the ranges", false;
  seen.index_masks = index_before - first_V.st.masks(ids[0]).left();
  for (const TensorRow &r : left.pending.look()) seen.targets.push_back(r.target), seen.psi.push_back(r.psi);
  LinFirst first;
  if (!member_first_verify(first_V, x.lin, ids[0], left.pending.take(), first)) return reason = "first part: the rows on the table", false;
  Sealed seal;
  if (!first_ch.seal(R_SEAL, seal) || !first_ch.finish()) return reason = "first part: its end", false;

  Reader path(view.path.data(), view.path.size());
  LinBindingRoot binding_root;
  if (!lin_binding_root(path, first_ch.get(x.binding), first_ch.configuration(), seal.digest(), binding_root) || !path.at_end()) return reason = "path", false;

  Reader shared_bytes(view.shared.data(), view.shared.size());
  Binder shared_b(LIN_LABEL_SHARED, w.lin.shared);
  shared_b.configuration(w.lin.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.lin.codes, w.lin.S, table, binding_root);
  VerifierRounds shared_ch(shared_b, shared_bytes, session);
  VerifierContext shared_V(shared_x.codes, shared_ch);
  LinShared shared;
  // wf_world() sets ONE_TIME, so `key` is read as an untested table. In the other arm the test
  // vouches for the table, as in lin_verify_view() (lin_world.h).
  if (!lin_shared_verify(shared_V, shared_x, w.lin.test, w.lin.test == TableTest::ONE_TIME ? TableRead(key) : TableRead::on_a_tests_word(key, w.lin.shared->set), shared)) return reason = "shared part", false;
  if (!lin_shared_tests_verify(shared_V, shared_x, shared)) return reason = "shared part: a proximity test", false;
  Sealed shared_seal;
  if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish()) return reason = "shared part: its end", false;

  Reader second_bytes(view.second.data(), view.second.size());
  Binder second_b(LIN_LABEL_SECOND, w.lin.second);
  second_b.configuration(w.lin.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  VerifierRounds second_ch(second_b, second_bytes, session);
  VerifierContext V = lin_go_on(std::move(first_V), second_ch);
  LinSecond second;
  if (!lin_second_verify(V, x.lin, first, shared_V, shared, second)) return reason = "second part", false;
  if (!lin_second_tests_verify(V, x.lin, first, second)) return reason = "second part: a proximity test", false;
  if (!second_ch.finish()) return reason = "second part: its end", false;
  V.st.require_all_tested();
  return session.accepted();
}

}  // namespace blk
