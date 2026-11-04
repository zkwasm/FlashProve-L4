// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests BatchOpening (prover/prover/batch_opening.h), whose calls from begin() to end() take a
// batch through LIN, the proof of its requests' pending claims (`TensorRow`) on the weight table
// (`Index` in identifiers). CPU only, at toy size, with honest provers, on the fixtures of
// blocks/iop/tests/member_world.h. main() builds each first chain (a request's own hash chain, up
// to its seal) as member_session() there does. member_verify_view(), the fixtures' verifier, must
// accept each `LinView` (one request's proof) and reject it when given another request's number;
// no two seals may share a leaf (`place`) of the binding tree. The table's one-time test, on which
// a batch's error bound relies, is not run: member_verify_view() marks the table as tested.
// Build, from prover/: g++ -std=c++17 -O2 -fno-ipa-modref -DDIGEST_WORDS=8 -DBLK_FIELD=2
//   -DBLK_HASH=1 -I . -I ../blocks tests/batch_opening_test.cpp
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // the fixtures build plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // lets test code mark a table as tested (blocks/iop/foldeq.h)
#include <cstdint>
struct K3 {  // the GPU's element of the cubic extension field E, for a build without GPU headers
  uint64_t k0, k1, k2;
};
typedef K3 Ext;
#include "prover/batch_opening.h"

#include <cstdio>
#include <set>

#include "engine/engine_cpu.h"
#include "iop/tests/member_world.h"
using namespace blk;

int main() {
  int bad = 0, views = 0;
  // pending claims per request
  const size_t pending = 3;
  for (const LinCase &t : MEMBER_CASES) {
    const LinWorld w = member_world(t, pending);
    const u64 seed = 41, coin_seed = 77;
    const LinMatrix M = lin_matrix(w.S, 11);
    TestCoins commit_coins(21);
    IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), commit_coins);
    const IndexKey key = index.key();
    EngineCpu engine;
    TestCoins coins(coin_seed);
    hf::HfLinSetup setup;
    setup.codes = w.codes, setup.sized.S = w.S, setup.plan = w.shared;
    const IndexName name{index.commitment(), key.rec};
    const BatchBinds binds{hf::hf_lin_bind_alone(setup, name), [](Binder &b, const Sealed &own, const Sealed &shared) {
                             b.configuration(toy_configuration());  // every chain names its configuration first
                             lin_bind_second(b, own, shared);
                           }};
    BatchOpening opening(setup, w.binding, toy_configuration(), w.second, binds, index, engine, coins);
    if (!opening.begin()) {
      bad++, std::printf("the toy table covers no batch: %s\n", t.name);
      continue;
    }
    std::vector<RequestPart> part(t.requests);
    for (unsigned c = 0; c < t.requests; c++) {
      RequestPart &r = part[c];
      r.binder.reset(new Binder(1, w.first));
      const LinFirstBound bound = member_bind_first(*r.binder, w, name, seed, c, pending).lin;  // the verifier binds the same public inputs through this function
      r.lin = bound.lin;
      r.first.reset(new Writer());
      r.channel.reset(new ProverRounds(*r.binder, toy_random_bytes(coin_seed + 100 + c), *r.first));
      r.context.reset(new ProverContext(r.lin.codes, *r.channel, engine, coins));
      std::vector<OracleDraft> drafts;
      drafts.push_back(lin_draft(*r.context, r.channel->get(bound.input).shape, false, r.channel->get(bound.input).leaf_class, {}));
      const OracleId masks = send_prove(*r.context, std::move(drafts))[0];
      std::vector<TensorRow> rows;
      for (size_t j = 0; j < pending; j++) {
        TensorRow row = member_row_form(w.S, seed, c, j);
        LinClaim f;
        f.g = row.g;
        const std::vector<E> wj = lin_w(w.S, f, M);
        MaskSlot slot = r.context->st.masks(masks).take();
        row.target = mle_eval(wj.data(), wj.size(), row.psi.data(), w.S.k_col()) + r.context->mask_value(slot);
        row.masks.push_back({std::move(slot), E::one()});
        r.channel->send(row.target);
        rows.push_back(std::move(row));
      }
      HeavyCpu heavy(w.S, M);
      r.lin_first = member_first_prove(*r.context, r.lin, masks, std::move(rows), heavy);
      BLK_REQUIRE(r.channel->seal(R_SEAL, r.seal), "a prover's channel does not refuse");
      r.channel->finish();
      opening.put(r);
    }
    opening.open();
    for (RequestPart &r : part) opening.second(r);
    std::set<u64> places;
    for (unsigned c = 0; c < t.requests; c++) {
      LinView v;
      v.first = part[c].first->bytes(), v.path = part[c].path, v.shared = opening.shared(), v.second = part[c].second;
      places.insert(part[c].place);
      const char *why = "";
      views++;
      if (!member_verify_view(w, pending, key, seed, c, v, &why, nullptr, LABEL_PROOF_SHARED_CHAIN, LABEL_PROOF_SECOND_CHAIN)) bad++, std::printf("an honest view is refused: %s, request %u: %s\n", t.name, c, why);
      if (t.requests > 1 && member_verify_view(w, pending, key, seed, (c + 1) % t.requests, v, nullptr, nullptr, LABEL_PROOF_SHARED_CHAIN, LABEL_PROOF_SECOND_CHAIN)) bad++, std::printf("A VIEW IS ACCEPTED UNDER ANOTHER REQUEST'S ROWS: %s\n", t.name);
    }
    if (places.size() != t.requests) bad++, std::printf("two requests stand at one place: %s\n", t.name);
    opening.end();
  }
  std::printf("batch_opening_test: %s (%d views)\n", bad ? "FAILED" : "ok", views);
  return bad ? 1 : 0;
}
