// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests blocks/iop/member.h, a request's own part of LIN (the proof of a batch's linear claims on
// the committed matrix), in whole batches at toy size with real hash chains (Fiat-Shamir
// transcripts), trees and proximity tests. BATT batches a request's pending claims on the matrix
// (`TensorRow`) into one on its first chain; LIN then sends W, that claim's row weights applied to
// the matrix. member_session() (blocks/iop/tests/member_world.h) proves one batch.
//
// In place of the rest of a request's proof, its first chain begins with its index-mask message
// (mask rows only, with an out-of-domain sample) and the claims' values, each masked by one entry
// of that message. Both sides derive the claims' weights and points from a seed and the request's
// number, which are in the chain's first query (the hash input that fixes its public inputs). A
// view (`LinView`) is what one request's verifier reads: first chain, path in the binding tree (a
// Merkle tree over the seals, the digests that end the batch's first chains), shared chain, second
// chain. The messages call the chains "parts", a claim a "row", its value a "target" and the
// table's holder (`IndexCpu`) "the keeper", and say "refused" for rejected.
//
// This program is the honest test. Six malicious provers, two of which attack the order of the
// three chains, belong to a test that is not part of this repository. That test has no control in
// which a weakened verifier accepts one of these provers: that needs a changed blocks/iop/lin.h.
// The committer's one-time test of the committed table, which a batch's error bound
// presupposes, is not run.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `member_world` builds plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets test code mark a table as tested (blocks/iop/foldeq.h)
#include "chain/tests/death.h"
#include "iop/tests/member_world.h"

using namespace blk;

// The honest test: every view of an honest batch must be accepted, by a verifier that gets nothing
// of the other requests, and rejected under another request's claims.
int main() {
  EngineCpu engine;
  [[maybe_unused]] size_t first_bytes = 0;
  size_t cases = 0;
  for (const LinCase &t : MEMBER_CASES) {
    const size_t pending = 3;
    const LinWorld w = member_world(t, pending);
    CHECK(!lin_fault(w.codes, w.S), "the case's shape is refused");
    const LinMatrix M = lin_matrix(w.S, 11), other = lin_matrix(w.S, 12);
    TestCoins commit_coins(21);
    IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), commit_coins);
    const IndexKey key = index.key();
    cases++;
    {  // weights, points or a message number that do not fit `LinShape` must abort the program
      const LinShape &S = w.S;
      CHECK(ends_with("does not lie inside the matrix", [&S] { member_weights_on_rows(S, 1, std::make_shared<std::vector<E>>(S.matrix_rows)); }),
            "a stretch of row weights that ends one row beyond the matrix is taken");
      CHECK(runs_through([&S] { member_weights_on_rows(S, 0, std::make_shared<std::vector<E>>(S.matrix_rows)); }), "the stretch of all the matrix's rows is not taken");
      std::vector<TensorRow> high(1), wide(1);
      const std::vector<E> zeros(S.k_row() + S.k_col() + 1, E::zero());
      high[0].g.push_back(Tensor::eq(zeros.data(), S.k_row() + 1)), high[0].psi.assign(S.k_col(), E::zero());
      wide[0].g.push_back(Tensor::eq(zeros.data(), S.k_row())), wide[0].psi.assign(S.k_col() + 1, E::zero());
      Statement st(w.codes);
      const OracleId masks = st.sent(w.input.shape), other_message = st.sent(S.w);
      CHECK(ends_with("do not fit the matrix's rows", [&] { member_rows_fit(st, S, masks, high); }), "weights with a row variable too many are taken");
      CHECK(ends_with("do not fit the matrix's columns", [&] { member_rows_fit(st, S, masks, wide); }), "a point with a column variable too many is taken");
      CHECK(ends_with("no sampled message of mask rows alone", [&] { member_rows_fit(st, S, other_message, wide); }), "W's own number is taken for the index-mask message's");
    }

    std::vector<PadLook> pads_a, pads_b;
    MemberOptions look_a, look_b;
    look_a.pads = &pads_a, look_b.pads = &pads_b;
    const std::vector<LinView> a = member_session(t, w, pending, index, key.rec, M, other, 31, 41, engine, look_a);
    CHECK(a.size() == t.requests, "the keeper does not serve");
    for (unsigned c = 0; c < a.size(); c++) {
      const char *why;
      const bool yes = member_verify_view(w, pending, key, 31, c, a[c], &why);
      const bool wanted = true;
      if (yes != wanted) std::printf("%s: request %u is %s (%s)\n", t.name, c, yes ? "accepted" : "refused", why);
      CHECK(yes == wanted, "a view does not come out as it should");
    }
    first_bytes = a[0].first.size();
    for (unsigned c = 0; c + 1 < a.size(); c++) CHECK(!member_verify_view(w, pending, key, 31, c + 1, a[c]), "a view is ACCEPTED for another request's rows");

    const std::vector<LinView> b = member_session(t, w, pending, index, key.rec, M, other, 31, 77, engine, look_b);  // the same claims, other private randomness
    CHECK(b.size() == t.requests && member_verify_view(w, pending, key, 31, 0, b[0]), "the second batch is refused");
    // A verifier cannot tell whether a pad (a sample answer's mask) was drawn, so both batches'
    // pads are read from the prover's memory: none may be 0 or occur twice.
    {  // `pads_fault` is in blocks/iop/send.h
      const char *fault = pads_fault(pads_a, pads_b);
      if (fault) std::printf("%s: %s\n", t.name, fault);
      CHECK(!fault && !pads_a.empty(), "a pad is not drawn afresh");
    }
    // All fourteen mixtures of request 0's views in batches a and b must be rejected. A coarse
    // check: twelve fail at the shared chain, whose first query has the binding tree's root. The
    // two orders of chains that matter are left to a test that is not part of this repository.
    for (unsigned mix = 1; mix < 15; mix++) {
      LinView v = a[0];
      if (mix & 1) v.first = b[0].first;
      if (mix & 2) v.path = b[0].path;
      if (mix & 4) v.shared = b[0].shared;
      if (mix & 8) v.second = b[0].second;
      if (member_verify_view(w, pending, key, 31, 0, v)) std::printf("%s: the mixed view %u is ACCEPTED\n", t.name, mix);
      CHECK(!member_verify_view(w, pending, key, 31, 0, v), "a view mixed from two batches is ACCEPTED");
    }

    // Outside MemberHeavy::begin(), member_first_prove() must read no entry of a written-out weight
    // (`Tensor::values`), so that a program may write these entries on other threads meanwhile.
    // Here they are zero until each first chain's seal, and every view must equal batch a's.
    {  // the same seeds and table as batch a
      MemberOptions written_late;
      written_late.weights_written_late = true;
      TestCoins same_coins(21);
      IndexCpu again(w.S.fold.index, lin_table_rows(w.S, M), same_coins);
      const std::vector<LinView> l = member_session(t, w, pending, again, key.rec, M, other, 31, 41, engine, written_late);
      CHECK(l.size() == a.size(), "the keeper does not serve");
      for (unsigned c = 0; c < l.size() && c < a.size(); c++)
        CHECK(l[c].first == a[c].first && l[c].path == a[c].path && l[c].shared == a[c].shared && l[c].second == a[c].second, "the first part has read an entry of a written-out weight: the bytes differ");
    }

    {  // an option for hiding tests (`query_record`): its views must be rejected
      const IndexRecord made_up{std::vector<E>(key.rec.y.size(), E::one())};
      MemberOptions door;
      door.query_record = &made_up;
      TestCoins same_coins(21);
      IndexCpu again(w.S.fold.index, lin_table_rows(w.S, M), same_coins);
      const std::vector<LinView> d = member_session(t, w, pending, again, key.rec, M, other, 31, 41, engine, door);
      CHECK(d.size() == t.requests, "the keeper does not serve");
      for (unsigned c = 0; c < d.size(); c++) CHECK(!member_verify_view(w, pending, key, 31, c, d[c]), "a session whose first queries hold a made-up record: a view is accepted");
    }

    const LinWorld w1 = member_world(t, 1);  // one pending claim: no BATT
    const std::vector<LinView> one = member_session(t, w1, 1, index, key.rec, M, other, 35, 45, engine);
    CHECK(one.size() == t.requests, "the keeper does not serve");
    for (unsigned c = 0; c < one.size(); c++) CHECK(member_verify_view(w1, 1, key, 35, c, one[c]), "a request with one pending row is refused");
    CHECK(one[0].first.size() + (batt_masks(w.S.k_col()) + pending - 1) * E_BYTES == first_bytes, "BATT costs other bytes than its mask entries");
  }
  std::printf("%s: %zu cases, last first part %zu bytes\n", failures ? "FAILED" : "ok", cases, first_bytes);
  return failures != 0;
}
