// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests WF (blocks/iop/wf.h), the committer's range proof on the entries of the matrix in a
// committed table (`Index` in identifiers), in entire one-time sessions at toy size. WF is meant to
// show by lookups (LKP, blocks/iop/lkp.h) that keys, affine in the entries, occur in range tables.
// For each part of a lookup's keys it leaves a pending claim (`pending row` in names and messages),
// a linear claim on the matrix. A session (wf_session(), wf_verify_view()) is the index-mask
// message (masks for claims on the matrix), WF, then BATT and LIN, which check the pending claims
// against the table. blocks/iop/tests/wf_world.h defines the session and the toy matrix: every
// entry is a toy byte in -4 .. 3, and a weight, low + 8 * high of two entries (its digits), is in
// -16 .. 15.
//
// In messages, `the proof of the ranges` is WF and `refused` means rejected; `first part`,
// `shared part` and `second part` are the bytes of a session's three hash chains (Fiat-Shamir
// transcripts), in order.
//
// No malicious prover runs in this test. In this program no
// verifier gets a session on a matrix with an entry out of its range: the honest prover serves none
// (see main()), and the sessions that must be rejected are honest ones with one bit flipped.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `wf_world` builds plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets test code mark a table as tested (blocks/iop/foldeq.h)
#include <functional>

#include "chain/tests/death.h"
#include "iop/tests/wf_world.h"

using namespace blk;

struct Committed {
  std::unique_ptr<IndexCpu> index;
  IndexKey key;
};
static Committed commit(const WfWorld &w, const LinMatrix &M) {
  Committed c;
  TestCoins coins(21);
  c.index.reset(new IndexCpu(w.lin.S.fold.index, lin_table_rows(w.lin.S, M), coins));
  c.key = c.index->key();
  return c;
}

// Checks wf_pending_row() against the definition of the keys. For each part it builds, at an
// arbitrary point, the true claim on the part's keys in the form that LKP outputs (`LkpKeys`),
// without masks, and checks that the resulting pending claim (`TensorRow`) holds for M:
// < g (x) eq(psi, .), M > = target.
// A session's own pending claims are not checked this way: the test does not know their mask
// entries.
static void pending_row_by_definition(const WfWorld &w, const LinMatrix &M) {
  TestCoins data(5);
  const LinMatrix *m = &M;
  const WfMatrix size = wf_matrix_of(w.lin.S);
  const WfCpu heavy(w.wf, size, [m](u64 row, u64 column) { return m->at(row, column); });
  for (size_t i = 0; i < w.wf.lookups.size(); i++) {
    const WfLookup &l = w.wf.lookups[i];
    std::vector<bool> padding;  // as the runs define it
    const std::vector<E> keys = heavy.keys(i, &padding);
    const LkpPublic told = wf_lookups(w.wf, size)[i];  // what LKP is told
    size_t first = 0;
    for (unsigned number = 0; number < l.parts.size(); number++) {
      const WfPart &p = l.parts[number];
      const unsigned v = p.log_places + size.k_col;
      LkpKeys claim;  // the weight eq(point, k) at a covered place, 0 at a place of padding
      claim.point.resize(v);
      for (E &z : claim.point) z = data.e();
      for (u64 k = 0; k < u64(1) << v; k++) {
        CHECK(lkp_is_padding(told, number, k) == padding[first + k], "LKP is told another padding than the runs leave");
        if (lkp_is_padding(told, number, k)) continue;
        claim.weights.add_entry(k, eq_index(k, claim.point.data(), v));
        claim.target += eq_index(k, claim.point.data(), v) * keys[first + k];
      }
      const TensorRow row = wf_pending_row(size, l, p, std::move(claim));
      LinClaim form;
      form.g = row.g;
      const std::vector<E> on_columns = lin_w(w.lin.S, form, M);
      CHECK(mle_eval(on_columns.data(), on_columns.size(), row.psi.data(), size.k_col) == row.target, "a pending row is not what the key map says");
      first += size_t(1) << v;
    }
  }
}
// This test: honest sessions, damaged proofs, invalid parameters, misuse of WF. No case gives
// WF and LIN different sizes of the matrix: the size is stored only in `LinShape` (see `WfMatrix`
// in blocks/iop/wf.h). No bit is flipped in `LinView::shared` or `LinView::path`.
int main() {
  EngineCpu engine;
  size_t cases = 0, flips = 0, bytes = 0;
  for (const WfCase &t : WF_CASES) {
    const WfWorld w = wf_world(t);
    const WfMatrix size = wf_matrix_of(w.lin.S);
    const char *fault = wf_fault(w.lin.codes, w.wf, size);
    if (fault) std::printf("%s: %s\n", t.lin.name, fault);
    CHECK(!fault && !lin_fault(w.lin.codes, w.lin.S), "the case's parameters are refused");
    CHECK(!w.lin.first->refused(), "the plan of the first chain is refused");
    CHECK(wf_covers_all_rows(w.wf, 0, size) && !wf_covers_all_rows(w.wf, 1, size), "which rows the look-ups cover");
    CHECK(!wf_covers_all_rows(w.wf, 0, WfMatrix{size.k_col, size.rows + 1}), "a matrix with a row more counts as covered");
    {
      const size_t ell = w.lin.codes.base[w.wf.message.shape.code].ell;
      std::vector<int> used(w.wf.message.shape.main_rows * ell, 0);
      for (const WfBlock &b : wf_layout(w.wf, ell)) {
        CHECK(b.first_place % (u64(1) << b.bits) == 0 && b.first_place + (u64(1) << b.bits) <= ell, "a block is not aligned, or leaves its row");
        for (u64 i = 0; i < u64(1) << b.bits; i++) used[b.row * ell + b.first_place + i]++;
      }
      CHECK(*std::max_element(used.begin(), used.end()) == 1, "two blocks of planes overlap");
    }
    const LinMatrix M = wf_matrix(w.lin.S, 11);
    pending_row_by_definition(w, M);
    Committed c = commit(w, M);
    WfSeen p, v;
    const LinView view = wf_session(w, *c.index, c.key.rec, M, 41, engine, p);
    const char *why;
    const bool yes = wf_verify_view(w, c.key, view, v, &why);
    if (!yes) std::printf("%s: refused (%s)\n", t.lin.name, why);
    CHECK(p.served && yes, "an honest session is refused");
    CHECK(p.targets == v.targets && p.psi == v.psi && p.targets.size() == w.pending, "the two sides leave WF with different pending rows");
    CHECK(p.index_masks == wf_index_masks(w.wf, size) && v.index_masks == p.index_masks, "WF takes another number of entries of the index-mask message than announced");
    bytes = view.first.size();
    cases++;
    {  // the pads in the prover's memory: same table and matrix, other private randomness (`coins`)
      WfSeen again;
      wf_session(w, *c.index, c.key.rec, M, 42, engine, again);
      const char *pads = pads_fault(p.pads, again.pads);
      if (pads) std::printf("%s: %s\n", t.lin.name, pads);
      // A pad masks the answer to an out-of-domain sample. No pad may be 0 or equal another, in one
      // run or across both. The bound 7 is one pad for each of: the index-mask message, the
      // multiplicities, the two code switches of WF's own test, and LIN's W, folded row and Omega.
      CHECK(!pads && p.pads.size() >= 7, "a pad of the session is 0, stands twice, or too few places were looked at");
    }

    for (int which = 0; which < 2; which++) {  // the honest prover fails on a value out of range
      LinMatrix bad = M;
      if (which == 0) wf_put_digits(bad, 3, 5, 0, 2);
      if (which == 1) bad.a[(bad.rows - 2) * bad.columns + 7] = wf_signed(4);
      Committed cb = commit(w, bad);
      WfSeen s;
      wf_session(w, *cb.index, cb.key.rec, bad, 41, engine, s);
      CHECK(!s.served, "the honest prover proves the ranges of a matrix that is out of them");
    }
    TestCoins where(9);
    for (int i = 0; i < 40; i++) {
      LinView bent = view;
      std::vector<u8> &part = i % 4 == 3 ? bent.second : bent.first;
      part[where.word() % part.size()] ^= (u8)(1u << (where.word() % 8));
      WfSeen s;
      const bool taken = wf_verify_view(w, c.key, bent, s);
      if (taken) std::printf("%s: a view with a flipped bit is ACCEPTED\n", t.lin.name);
      CHECK(!taken, "a view with a flipped bit is accepted");
      flips++;
    }
  }
  {  // `wf_fault` rejects each invalid `WfPublic` with the expected message
    const WfWorld w = wf_world(WF_CASES[0]);
    auto fault_of = [&](const std::function<void(WfPublic &)> &bend) {
      WfPublic x = w.wf;
      bend(x);
      const char *f = wf_fault(w.lin.codes, x, wf_matrix_of(w.lin.S));  // `public_fault` first, then the fit to matrix and codes
      return std::string(f ? f : "");
    };
    CHECK(fault_of([](WfPublic &) {}) == "", "the honest parameters are refused");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].parts[0].runs[1].first_place = 5; }) == "a part's runs overlap, are not in rising order, or begin beyond the part", "runs that overlap");
    CHECK(fault_of([](WfPublic &x) { std::swap(x.lookups[0].parts[0].runs[0], x.lookups[0].parts[0].runs[1]); }) == "a part's runs overlap, are not in rising order, or begin beyond the part", "runs that fall");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].parts[0].runs[3].places = 3; }) == "a run reaches beyond its part", "a run beyond its part");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].parts[1].runs[0].places = 0; }) == "a run covers no place", "a run without a place");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].parts[1].runs.clear(); }) == "a part covers no place", "a part without a run");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].parts[2].runs[0].stride = 1; }) == "a run has a stride where a key has one digit", "a stride that nobody reads");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].padding_key = from_u64(8); }) == "a place of padding holds no row of the table", "a padding key outside the table");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].parts[0].runs[1].stride = 9; }) == "a run's places reach beyond the matrix", "a digit's row beyond the matrix");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].first_bound = {1, 1}; }) == "the order of the last key level names a variable twice", "an order with a variable twice");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].first_bound = {5}; }) == "the order of the last key level names a variable that a part has not", "an order with a variable that a part has not");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].live = 8; }) == "a part has no key, or more keys than a multiplicity can count", "a part with more keys than the planes count");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].parts.push_back(x.lookups[1].parts[0]); }) == "a look-up has more keys than a multiplicity can count", "more keys than the planes count");
    // With these, acceptance would mean less than "every entry is in its range":
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].digit_weight[0] = F::zero(); }) == "a digit of a look-up's key has the weight 0", "the covering look-up's entries enter no key");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].digit_weight[1] = F::zero(); }) == "a digit of a look-up's key has the weight 0", "a high digit that enters no key");
    CHECK(fault_of([](WfPublic &x) { x.lookups[0].digit_weight[0] = from_u64(2); }) == "no look-up of the ranges covers every row of the matrix", "the covering look-up's key is not offset + entry");
    CHECK(fault_of([](WfPublic &x) { x.lookups[1].parts[0].runs[0].stride = 0; }) == "the digits of a key lie in one row", "two digits in one row");
  }
  {  // misuse by a caller: the process aborts with the expected message
    const WfCase &t = WF_CASES[0];
    const WfWorld w = wf_world(t);
    const LinMatrix M = wf_matrix(w.lin.S, 11);
    Committed c = commit(w, M);
    const IndexName table{c.index->commitment(), c.key.rec};
    auto run = [&](bool sampled, bool own_binder, bool rows_taken = true) {
      TestCoins coins(1);
      Writer bytes_out;
      Binder b(1, w.lin.first), other(1, w.lin.first);
      const WfBound bound = wf_bind_first(b, w, table), foreign = wf_bind_first(other, w, table);
      ProverRounds ch(b, toy_random_bytes(3), bytes_out);
      ProverContext P(bound.lin.codes, ch, engine, coins);
      std::vector<OracleDraft> drafts;
      drafts.push_back(lin_draft(P, ch.get(bound.input).shape, false, ch.get(bound.input).leaf_class, {}));
      const OracleId masks = sampled ? send_prove(P, std::move(drafts))[0] : send_oracles(P, std::move(drafts))[0];
      const LinMatrix *m = &M;
      WfCpu ranges(w.wf, wf_matrix_of(w.lin.S), [m](u64 row, u64 column) { return m->at(row, column); });
      WfOutput left;
      (void)wf_prove(P, own_binder ? bound.wf : foreign.wf, bound.lin, masks, ranges, left);
      if (rows_taken) (void)left.pending.take();
    };
    CHECK(runs_through([&] { run(true, true); }), "the control of the stops stops");
    CHECK(ends_with("WF: the index-mask message is a sampled message of mask rows alone", [&] { run(false, true); }), "WF runs on an index-mask message whose sample is not answered");
    CHECK(ends_with("a public input that does not stand in this chain's first query", [&] { run(true, false); }), "WF runs on parameters that stand in another chain's first query");
    CHECK(ends_with("WF: the rows on the matrix were dropped", [&] { run(true, true, false); }), "a caller may drop the rows that WF leaves on the matrix");
  }
  std::printf("%s: %zu cases, %zu flipped bits, last first part %zu bytes\n", failures ? "FAILED" : "ok", cases, flips, bytes);
  return failures != 0;
}
