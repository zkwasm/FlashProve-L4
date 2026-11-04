// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Test fixture (`world` in identifiers) at toy size: proves and checks batches under LIN, the
// protocol block for one linear claim per request on the committed table (`Index` in identifiers),
// which stores the matrix M. A request brings one or more pending claims on M (`TensorRow`;
// "pending rows" in messages); BATT batches several into one (blocks/iop/member.h).
// A request's proof has a first, a shared and a second chain ("part" in messages).
// The prover is honest unless `MemberOptions::add_to_w` is set; no test in this repository sets it.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include "engine/sumcheck_cpu.h"
#include "iop/member.h"
#include "iop/tests/lin_world.h"

namespace blk {

// Columns: `LinCase`'s fields (lin_world.h).
static const LinCase MEMBER_CASES[] = {
    // name                                   v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws ahead
    {"one code, rows low",                    2, 2,   3,   1,   1,  4,  4,  2,    8,  8,  8, false, 3, 2,    3, false},
    {"three codes, rows high",                3, 1,   5,   0,   1,  3,  4,  3,   16,  4,  8, true,  2, 2,    2, false},
    {"G_u's code apart, masks ahead",         2, 2,   3,   1,   1,  4,  4,  2,   16,  8,  8, true,  3, 2,    3, true},
    {"one request",                           1, 2,   4,   2,   1,  3,  4,  2,    8,  8,  4, false, 1, 0,    2, false},
};

// lin_world() with the first chain planned for `pending` pending claims per request.
inline LinWorld member_world(const LinCase &t, size_t pending) {
  LinWorld w = lin_world(t, TableTest::LATER);  // a batch's test, not the committer's one-time test
  Plan first = toy_empty_plan();
  lin_plan_send(first, w.codes, {w.input});
  first.send_scalars("the pending rows' targets", pending);
  plan_member_rows(first, "rows on the table", pending, w.S.k_col());
  lin_plan_send(first, w.codes, lin_message_w(w.S));
  lin_plan_seal(first);
  w.first = first;
  return w;
}
// Pending claim j of request c without masks and value (`target`).
// Both sides derive it from the seed.
inline TensorRow member_row_form(const LinShape &S, u64 seed, unsigned c, size_t j) {
  LinClaim f = lin_claim_form(S, seed * 16 + j, c);
  TensorRow r;
  r.g = std::move(f.g), r.psi = std::move(f.psi);
  // Adds weights written out on a range of rows, like a sub-proof's.
  TestCoins data(seed * 77 + 5 * c + j);
  const u64 first = (seed + c + j) % S.matrix_rows;
  auto values = std::make_shared<std::vector<E>>(S.matrix_rows - first);
  for (E &x : *values) x = data.e();
  for (Tensor &t : member_weights_on_rows(S, first, std::move(values))) r.g.push_back(std::move(t));
  return r;
}

// `MemberHeavy` (the prover's access to the matrix for one request) on the CPU. W_j = g_j^T M for
// the weights g_j of pending claim j; W, which LIN sends, is a combination of the W_j.
class HeavyCpu : public MemberHeavy {
 public:
  HeavyCpu(const LinShape &S, const LinMatrix &M) : S_(S), M_(M) {}
  void begin(const std::vector<std::vector<Tensor>> &g) override {
    w_.clear();
    for (const std::vector<Tensor> &x : g) {
      LinClaim c;
      c.g = x;
      w_.push_back(lin_w(S_, c, M_));
    }
  }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &,
                                              const std::vector<std::vector<E>> &) override {
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(s, vars, w_));
  }
  std::vector<E> combine(const std::vector<E> &a) override {
    std::vector<E> w(M_.columns, E::zero());
    for (size_t j = 0; j < a.size(); j++)
      for (u64 c = 0; c < M_.columns; c++) w[c] += a[j] * w_[j][c];
    return w;
  }

 private:
  const LinShape &S_;
  const LinMatrix &M_;
  std::vector<std::vector<E>> w_;
};

using MemberHeavyMaker = std::function<std::unique_ptr<MemberHeavy>(const LinShape &S, const LinMatrix &M)>;
inline std::unique_ptr<MemberHeavy> member_heavy_cpu(const LinShape &S, const LinMatrix &M) { return std::unique_ptr<MemberHeavy>(new HeavyCpu(S, M)); }
// A malicious prover's `MemberHeavy`: adds `add` to the first entries of W.
class HeavyThatAdds : public MemberHeavy {
 public:
  HeavyThatAdds(std::unique_ptr<MemberHeavy> inner, std::vector<E> add) : inner_(std::move(inner)), add_(std::move(add)) {}
  void begin(const std::vector<std::vector<Tensor>> &g) override { inner_->begin(g); }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &chi,
                                              const std::vector<std::vector<E>> &psi) override {
    return inner_->sumcheck(s, vars, chi, psi);
  }
  std::vector<E> combine(const std::vector<E> &a) override {
    std::vector<E> w = inner_->combine(a);
    for (size_t c = 0; c < add_.size(); c++) w[c] += add_[c];
    return w;
  }

 private:
  std::unique_ptr<MemberHeavy> inner_;
  std::vector<E> add_;
};

// Models a program that has the W_j from elsewhere: the constructor calls begin() on the true
// weights; the later begin(), from member_first_prove(), reads nothing.
class HeavyBegunBefore : public MemberHeavy {
 public:
  HeavyBegunBefore(std::unique_ptr<MemberHeavy> inner, const std::vector<TensorRow> &rows) : inner_(std::move(inner)) {
    std::vector<std::vector<Tensor>> g;
    for (const TensorRow &r : rows) g.push_back(r.g);
    inner_->begin(g);
  }
  void begin(const std::vector<std::vector<Tensor>> &) override {}
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &chi,
                                              const std::vector<std::vector<E>> &psi) override {
    return inner_->sumcheck(s, vars, chi, psi);
  }
  std::vector<E> combine(const std::vector<E> &a) override { return inner_->combine(a); }

 private:
  std::unique_ptr<MemberHeavy> inner_;
};

struct MemberOptions {
  // The pending claims enter member_first_prove() with every entry of `Tensor::values` zero; the
  // true entries are written after the first chain's seal, as by a program that writes them on
  // other threads meanwhile (when that is allowed: MemberHeavy::begin(), blocks/iop/member.h). The
  // views must equal those without this option.
  bool weights_written_late = false;
  u64 tests_coin_seed = 0;  // not 0: seeds the shared chain's proximity tests' private randomness
  std::vector<E> add_to_w;  // added to the first entries of request 0's W
  std::vector<PadLook> *pads = nullptr;  // collects every sampled message's pad, for `pads_fault`
  // For hiding tests over whole batches (blocks/tests/hiding/). `coins`, if set, replaces
  // TestCoins(coin_seed), but not the first and second chains' 32 random bytes, which stay
  // toy_random_bytes(coin_seed + ...). With `look`, every Writer lists its items and is passed to
  // `look` when complete, as "first", "shared", "second" or "path", with the request's number (0
  // for the shared chain).
  Coins *coins = nullptr;
  std::function<void(const char *chain, unsigned request, const Writer &)> look;
  // For a hiding test, as LinSessionOptions::query_record (lin_world.h): if set, the first queries
  // of the first and shared chains contain *query_record, while the record's claims use the real
  // record (`rec`). member_verify_view() rejects such a session.
  const IndexRecord *query_record = nullptr;
};
// The first query (the hash input that fixes the public inputs) of a request's first chain:
// lin_bind_first()'s items, then the number of pending claims. One function for both sides and
// any test that builds such a chain itself: a prover whose first query omits an item is rejected
// only on the shared chain, when the table's columns are checked.
struct MemberFirstBound {
  LinFirstBound lin;
  Public<u64> pending;
};
inline MemberFirstBound member_bind_first(Binder &b, const LinWorld &w, const IndexName &table, u64 seed, unsigned c, size_t pending) {
  MemberFirstBound x;
  x.lin = lin_bind_first(b, w, table, seed, c);
  x.pending = b.statement("the number of pending rows", u64(pending));
  return x;
}
// Proves one batch (`session` in the function's name). Returns each request's view (`LinView`:
// first chain; path in the binding tree, the Merkle tree over the first chains' seals; shared
// chain; second chain), or nothing if lin_begin() allows no further batch on the table. The
// index-mask message (an oracle message of masks) and the pending claims' masked values stand in
// for a request's sub-proofs. `pending`: as passed to member_world(). M: the committed matrix.
// `other`: a second matrix, which this function does not read. `seed`: of the public claims;
// `coin_seed`: of private randomness. `engine`, `maker`: the CPU's or a GPU's.
inline std::vector<LinView> member_session(const LinCase &t, const LinWorld &w, size_t pending, IndexHolder &index, const IndexRecord &rec, const LinMatrix &M,
                                           [[maybe_unused]] const LinMatrix &other, u64 seed, u64 coin_seed, Engine &engine,
                                           const MemberOptions &opt = MemberOptions(), const MemberHeavyMaker &maker = member_heavy_cpu) {
  const LinShape &S = w.S;
  TestCoins own_coins(coin_seed);
  Coins &coins = opt.coins ? *opt.coins : own_coins;
  struct Request {
    Writer first, second;
    std::unique_ptr<ProverRounds> ch;
    std::unique_ptr<ProverContext> P;
    LinFirstBound bound;
    LinFirst lin;
    Sealed seal;
  };
  const IndexName table{index.commitment(), opt.query_record ? *opt.query_record : rec};
  std::vector<Request> req(t.requests);
  std::vector<LinView> views(t.requests);
  LinBinding binding(w.binding, w.configuration, coins);
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(index, S, begun)) return {};

  for (unsigned c = 0; c < t.requests; c++) {
    Request &r = req[c];
    if (opt.look) r.first.keep_list(), r.second.keep_list();  // before the first byte, for `look`
    Binder b(1, w.first);
    const MemberFirstBound bound = member_bind_first(b, w, table, seed, c, pending);
    r.bound = bound.lin;
    const Public<u64> rows_bound = bound.pending;
    r.ch.reset(new ProverRounds(b, toy_random_bytes(coin_seed + 100 + c), r.first));
    r.P.reset(new ProverContext(r.bound.lin.codes, *r.ch, engine, coins));
    r.P->pads = opt.pads;
    std::vector<OracleDraft> drafts;
    drafts.push_back(lin_draft(*r.P, r.ch->get(r.bound.input).shape, false, r.ch->get(r.bound.input).leaf_class, {}));
    const OracleId masks = send_prove(*r.P, std::move(drafts))[0];
    const LinMatrix *mine = &M;
    std::vector<TensorRow> rows;
    BLK_REQUIRE(r.ch->get(rows_bound) == pending, "the world's own number");
    for (size_t j = 0; j < pending; j++) {
      TensorRow row = member_row_form(S, r.ch->get(r.bound.seed), r.ch->get(r.bound.request), j);
      LinClaim f;
      f.g = row.g;
      const std::vector<E> wj = lin_w(S, f, *mine);
      MaskSlot slot = r.P->st.masks(masks).take();
      row.target = mle_eval(wj.data(), wj.size(), row.psi.data(), S.k_col()) + r.P->mask_value(slot);
      row.masks.push_back({std::move(slot), E::one()});
      r.ch->send(row.target);
      rows.push_back(std::move(row));
    }
    std::unique_ptr<MemberHeavy> made = maker(S, *mine);
    if (c == 0 && !opt.add_to_w.empty()) made.reset(new HeavyThatAdds(std::move(made), opt.add_to_w));
    std::vector<std::pair<std::shared_ptr<std::vector<E>>, std::shared_ptr<const std::vector<E>>>> late;  // pairs: the zero vector now in the claims; its true entries
    if (opt.weights_written_late) {
      made.reset(new HeavyBegunBefore(std::move(made), rows));
      std::map<const std::vector<E> *, std::shared_ptr<std::vector<E>>> zeros;
      for (TensorRow &row : rows)
        for (Tensor &x : row.g) {
          if (!x.values) continue;
          std::shared_ptr<std::vector<E>> &z = zeros[x.values.get()];
          if (!z) z = std::make_shared<std::vector<E>>(x.values->size(), E::zero()), late.push_back({z, x.values});
          x.values = z;
        }
    }
    MemberHeavy &heavy = *made;
    r.lin = member_first_prove(*r.P, r.bound.lin, masks, std::move(rows), heavy);
    BLK_REQUIRE(r.ch->seal(R_SEAL, r.seal), "a prover's channel does not refuse");
    for (auto &[zeros, truth] : late) *zeros = *truth;
    r.ch->finish();  // otherwise the second chain's `finish` ends the process
    if (opt.look) opt.look("first", c, r.first);
    binding.put(c, r.seal.digest());
  }

  Writer shared_bytes;
  if (opt.look) shared_bytes.keep_list();
  Binder shared_b(LIN_LABEL_SHARED, w.shared);
  shared_b.configuration(w.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.codes, w.S, table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  TestCoins other_coins(opt.tests_coin_seed);
  RelayCoins shared_coins(coins);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, shared_coins);
  shared_P.pads = opt.pads;
  LinShared shared;
  if (opt.query_record) {  // not `lin_shared_prove`, which takes the record from the first query
    shared.bound = shared_x;
    lin_shared_prove_on_shape(shared_P, shared_ch.get(shared_x.shape), shared_x.places, w.test, index, rec, std::move(begun), shared);
  } else {
    lin_shared_prove(shared_P, shared_x, w.test, index, std::move(begun), shared);
  }
  if (opt.tests_coin_seed) shared_coins.take_from(other_coins);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish();
  if (opt.look) opt.look("shared", 0, shared_bytes);

  for (unsigned c = 0; c < t.requests; c++) {
    Request &r = req[c];
    Binder b(LIN_LABEL_SECOND, w.second);
    b.configuration(w.configuration);
    lin_bind_second(b, r.seal, shared_seal);
    ProverRounds ch(b, toy_random_bytes(coin_seed + 200 + c), r.second);
    ProverContext P = lin_go_on(std::move(*r.P), ch);
    P.pads = opt.pads;
    const LinSecond second = lin_second_prove(P, r.bound.lin, r.lin, shared_P, shared);
    lin_second_tests_prove(P, r.bound.lin, r.lin, second);
    ch.finish();
    if (opt.look) opt.look("second", c, r.second);
    P.st.require_all_tested();
    Writer path;
    if (opt.look) path.keep_list();
    binding.write_path(path, c);
    if (opt.look) opt.look("path", c, path);  // no field element
    views[c].first = r.first.bytes(), views[c].path = path.bytes(), views[c].shared = shared_bytes.bytes(), views[c].second = r.second.bytes();
  }
  lin_release(shared_P, shared);
  return views;
}

// What a malicious prover learns by running the verifier's steps on a view of its own.
// `gamma`: the point of the claim that enters LIN. `shared_point`: LIN's first challenge on the
// shared chain. No caller in this repository passes one to member_verify_view().
struct MemberPeek {
  std::vector<E> gamma, shared_point;
};
// The verifier of request c: the table's key, the seed of the public claims and the request's
// view; nothing of the other requests. `why`: the step that rejected, if any. The two labels: those
// of the shared and the second chain; a caller whose prover takes others hands these in.
inline bool member_verify_view(const LinWorld &w, size_t pending, const IndexKey &key, u64 seed, unsigned c, const LinView &view, const char **why = nullptr,
                        MemberPeek *peek = nullptr, u64 shared_label = LIN_LABEL_SHARED, u64 second_label = LIN_LABEL_SECOND) {
  const IndexName table{key.commitment(), key.rec};
  const char *unused;
  const char *&reason = why ? *why : unused;
  reason = "";
  Reader first_bytes(view.first.data(), view.first.size());
  Binder first_b(1, w.first);
  const MemberFirstBound first_bound = member_bind_first(first_b, w, table, seed, c, pending);
  const LinFirstBound &x = first_bound.lin;
  const Public<u64> rows_bound = first_bound.pending;
  VerifierSession session;  // one verdict: its three chains share Merkle-root records and outcomes
  VerifierRounds first_ch(first_b, first_bytes, session);
  VerifierContext first_V(x.lin.codes, first_ch);
  std::vector<OracleId> ids;
  if (!send_verify(first_V, std::vector<OracleNotice>{first_ch.get(x.input)}, ids)) return reason = "first part: the index-mask message", false;
  std::vector<TensorRow> rows;
  for (size_t j = 0; j < first_ch.get(rows_bound); j++) {
    TensorRow row = member_row_form(first_ch.get(x.lin.shape), first_ch.get(x.seed), first_ch.get(x.request), j);
    if (!first_ch.recv(row.target)) return reason = "first part: a target", false;
    row.masks.push_back({first_V.st.masks(ids[0]).take(), E::one()});
    rows.push_back(std::move(row));
  }
  LinFirst first;
  if (!member_first_verify(first_V, x.lin, ids[0], std::move(rows), first)) return reason = "first part", false;
  Sealed seal;
  if (!first_ch.seal(R_SEAL, seal) || !first_ch.finish()) return reason = "first part: its end", false;

  Reader path(view.path.data(), view.path.size());
  LinBindingRoot binding_root;
  if (!lin_binding_root(path, first_ch.get(x.binding), first_ch.configuration(), seal.digest(), binding_root) || !path.at_end()) return reason = "path", false;

  Reader shared_bytes(view.shared.data(), view.shared.size());
  Binder shared_b(shared_label, w.shared);
  shared_b.configuration(w.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, w.codes, w.S, table, binding_root);
  VerifierRounds shared_ch(shared_b, shared_bytes, session);
  VerifierContext shared_V(shared_x.codes, shared_ch);
  LinShared shared;
  // The fixture runs no one-time test: it vouches that the table has passed one, as a batch's test
  // requires (enforced through `TableRead`, blocks/iop/foldeq.h).
  if (!lin_shared_verify(shared_V, shared_x, w.test, w.test == TableTest::ONE_TIME ? TableRead(key) : TableRead::on_a_tests_word(key, w.shared->set), shared)) return reason = "shared part", false;
  if (peek) peek->gamma = first.claim.psi, peek->shared_point = shared.psi;
  if (!lin_shared_tests_verify(shared_V, shared_x, shared)) return reason = "shared part: a proximity test", false;
  Sealed shared_seal;
  if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish()) return reason = "shared part: its end", false;

  Reader second_bytes(view.second.data(), view.second.size());
  Binder second_b(second_label, w.second);
  second_b.configuration(w.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  VerifierRounds second_ch(second_b, second_bytes, session);
  VerifierContext V = lin_go_on(std::move(first_V), second_ch);
  LinSecond second;
  if (!lin_second_verify(V, x.lin, first, shared_V, shared, second)) return reason = "second part", false;
  if (!lin_second_tests_verify(V, x.lin, first, second)) return reason = "second part: a proximity test", false;
  if (!second_ch.finish()) return reason = "second part: its end", false;
  V.st.require_all_tested();
  return session.accepted();  // all three chains have finished; this reads their outcomes
}

}  // namespace blk
