// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests of LIN (blocks/iop/lin.h) and FOLDEQ at toy size. LIN proves one linear claim per request
// of a batch (row weights g, point psi, public value `target`) on a matrix M in a committed table
// (`Index` in identifiers), with one fold of the table. FOLDEQ turns claims on the fold into claims
// on a fresh oracle message, G_u. LIN's hash chains (Fiat-Shamir transcripts; a first query fixes a
// chain's public inputs) and their linear claims (`Row` in identifiers):
//   first, per request:  W = g^T M. (Eval): W at psi, plus mask entries, equals `target`.
//   shared, per batch:   the shared point psi*, G_u, table columns, Omega = M eq(psi*, .), zeta.
//                        (Fold): Omega at zeta is what G_u gives.
//   second, per request: (Dot): <g, Omega> = W at psi*.
// lin_session() in blocks/iop/tests/lin_world.h is a batch's prover; lin_verify_view() is the
// verifier, given only one request's view (`LinView`). This test checks that honest views are
// accepted, false targets and altered views are rejected, and a caller's mistakes abort. No
// malicious ("cheating") prover runs in this test. The only false claims here are false values handed to the honest prover's code.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // tests build and alter plans by hand (blocks/numbers/plan.h)
#define BLK_TESTED_TABLE_FOR_A_TEST   // tests vouch that tables are tested (blocks/iop/foldeq.h)
#include "chain/tests/death.h"
#include "iop/tests/lin_world.h"

using namespace blk;

// The cases cover one to three codes for G_u, Omega and W, both
// vector layouts (`rows_high`) and fresh fold masks sent with earlier prover messages (`ahead`). A
// record is a table's stored answers to out-of-domain samples of its rows; a pad is the random
// element that masks one sample's answer. A claim that reads two codes needs the block SPLIT, which
// makes it one claim on each. Columns: `LinCase`'s fields (lin_world.h).
static const LinCase CASES[] = {
    // name                                   v k_lo rows spare ans col bat rate  u  om  w  high req leaves draws
    {"one code, rows low",                    2, 2,   3,   1,   1,  4,  3,  2,    8,  8,  8, false, 3, 2,    3, false},
    {"one code, rows high",                   2, 2,   3,   0,   1,  4,  3,  2,    8,  8,  8, true,  3, 2,    3, false},
    {"three codes, rows low",                 2, 1,   5,   3,   1,  3,  2,  2,   16,  4,  8, false, 2, 1,    3, false},
    {"three codes, rows high",                3, 1,   5,   0,   1,  3,  2,  3,   16,  4,  8, true,  2, 2,    2, false},
    {"two answers, W's code apart",           1, 2,   4,   2,   2,  3,  2,  2,    8,  8,  4, false, 1, 0,    2, false},
    {"one row of the table, one of the matrix", 0, 2, 1,   0,   1,  3,  2,  2,    4,  4,  4, true,  2, 1,    2, false},
    {"G_u's code apart, masks ahead",         2, 2,   3,   1,   1,  4,  3,  2,   16,  8,  8, true,  3, 2,    3, true},
    {"one code, masks ahead",                 2, 2,   3,   0,   1,  4,  3,  2,    8,  8,  8, false, 2, 1,    3, true},
    // `tight_tails` (last field): a fresh code's random tail covers just its own opened positions
    {"tight tails",                           2, 2,   3,   1,   1,  4,  3,  2,    8,  8,  8, false, 3, 2,    3, false, true},
    {"tight tails, masks ahead",              2, 1,   5,   0,   1,  3,  2,  2,   16,  4,  8, true,  2, 1,    3, true,  true},
    {"many columns, some drawn twice",        1, 1,   2,   0,   1, 40,  2,  2,   32, 32, 32, true,  1, 0,    2, false},  // must stay last: a loop below runs it again until a column is drawn twice
};
static const size_t ALTERED[] = {4, 2};  // the cases whose views are altered byte by byte: one request; two requests, a path, both splits

// An IndexHolder (blocks/iop/foldeq.h; "keeper" in messages) holds a table for the prover. This one
// forwards every call to a real one except configuration(), so the base class answers zeros: no
// configuration digest.
class NamesNoConfiguration : public IndexHolder {
 public:
  explicit NamesNoConfiguration(IndexHolder &real) : real_(real) {}
  const IndexShape &shape() const override { return real_.shape(); }
  std::vector<u8> commitment() const override { return real_.commitment(); }
  std::vector<u8> name() const override { return real_.name(); }
  IndexRecord record() override { return real_.record(); }
  bool begin(u64 columns) override { return real_.begin(columns); }
  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override { real_.fold(gamma, message, tail, blocks); }
  TableId fold_table(const E *gamma, const FoldedWant &want, BlockSums *blocks) override { return real_.fold_table(gamma, want, blocks); }
  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override { real_.open(at, file, columns); }
  void forget() override { real_.forget(); }

 private:
  IndexHolder &real_;
};

struct Committed {
  std::unique_ptr<IndexCpu> index;
  IndexKey key;
};
[[maybe_unused]] static Committed commit(const LinWorld &w, const LinMatrix &M, u64 seed) {
  Committed c;
  TestCoins coins(seed);
  c.index.reset(new IndexCpu(w.S.fold.index, lin_table_rows(w.S, M), coins));
  c.key = c.index->key();
  CHECK(!index_key_fault(c.key), "an honest key is refused");
  CHECK(record_bytes(c.index->record()) == record_bytes(c.key.rec), "a table has two records");
  return c;
}

// Verifies every view of a batch. True if view c is accepted exactly when wanted[c] and, if `by` is
// given, every rejection has that reason.
[[maybe_unused]] static bool views_are(const LinWorld &w, const IndexKey &key, u64 seed, const LinSession &s, const std::vector<bool> &wanted,
                                       const char *text, const char *by = nullptr) {
  bool good = true;
  for (unsigned c = 0; c < s.views.size(); c++) {
    const char *why;
    const bool yes = lin_verify_view(w, key, seed, c, s.views[c], nullptr, &why);
    if (yes != wanted[c] || (!yes && by && std::string(why) != by))
      good = false, std::printf("%s: request %u is %s (%s)\n", text, c, yes ? "accepted" : "refused", why);
  }
  return good;
}

// ---------------------------------------------------------------------------------------------------------------------
// This test.
static void test_helpers() {
  TestCoins data(5);
  for (unsigned bits = 0; bits <= 5; bits++) {
    std::vector<E> z(bits + 2);
    for (E &x : z) x = data.e();
    const Tensor v = Tensor::eq(z.data(), bits + 2, data.e());
    for (u64 len = 0; len <= (u64(1) << bits); len++) {
      std::vector<Tensor> cut;
      tensor_below(v, 1, bits, len, cut);
      std::vector<E> sum(size_t(1) << (bits + 2), E::zero());
      for (const Tensor &t : cut) t.add_to(sum.data());
      for (u64 j = 0; j < sum.size(); j++)
        CHECK(sum[j] == ((((j >> 1) & ((u64(1) << bits) - 1)) < len) ? v.at(j) : E::zero()), "tensor_below differs from the vector written out");
    }
  }
  for (u64 len = 0; len <= 8; len++) {  // the cut bits (1 to 3) index a written-out factor
    auto values = std::make_shared<std::vector<E>>(8);
    for (E &x : *values) x = data.e();
    Tensor v = Tensor::eq(values->data(), 1);
    v.push_dense(3);
    v.set_values(values, 0);
    std::vector<Tensor> cut;
    tensor_below(v, 1, 3, len, cut);
    E point[4], sum = E::zero(), direct = E::zero();
    for (E &x : point) x = data.e();
    for (const Tensor &t : cut) sum += t.mle(point);
    for (u64 j = 0; j < 16; j++)
      if ((j >> 1) < len) direct += eq_index(j, point, 4) * v.at(j);
    CHECK(sum == direct, "tensor_below differs from the vector written out, with a written-out factor");
  }
  for (int high = 0; high < 2; high++) {
    const VectorLay lay{2, 3, high == 1};
    std::vector<E> z(4);
    for (E &x : z) x = data.e();
    Tensor v = Tensor::eq(z.data(), 2);
    v.push_dense(2);
    v.set_values(std::make_shared<std::vector<E>>(z), 0);
    const Tensor on = lay.on_table(v);
    for (u64 j = 0; j < 32; j++) {
      CHECK(lay.entry_at(lay.index(j)) == j, "a lay's two maps are not inverse");
      CHECK(on.at(lay.index(j)) == (j < 16 ? v.at(j) : E::zero()), "a structured vector moves to other entries than the vector");
    }
  }
}


int main() {
  test_helpers();
  size_t bytes_of_a_view = 0;
  bool twice = false;
  for (const LinCase &t : CASES) {
    const LinWorld w = lin_world(t, TableTest::LATER);
    CHECK(!lin_fault(w.codes, w.S), "the case's shape is refused");
    const LinMatrix M = lin_matrix(w.S, 11);
    Committed c = commit(w, M, 21);
    const std::vector<bool> all(t.requests, true);

    const LinSession s = lin_session(t, w, *c.index, c.key.rec, M, 31, 41);
    CHECK(s.served, "the keeper does not serve");
    CHECK(views_are(w, c.key, 31, s, all, t.name), "an honest view is refused");
    CHECK(s.column_rows == t.columns, "not one row for each draw of a column");
    // A batch's test (TableTest::LATER, "a later test" in messages) needs a table that passed the
    // committer's one-time test under the run's `Setting`, or foldeq_columns_verify() aborts. The
    // view is honest; `handed` replaces the TableRead by which lin_verify_view() vouches for the
    // table.
    if (&t == &CASES[0] || &t == &CASES[1]) {
      const TableRead untested(c.key);
      Setting weaker = w.shared->set, more_work = w.shared->set;
      weaker.target--, more_work.work[3][2]++;
      const TableRead at_a_lower_target = TableRead::on_a_tests_word(c.key, weaker), with_other_work = TableRead::on_a_tests_word(c.key, more_work);
      const TestedTable none_yet;
      const TableRead none(none_yet);
      auto view_on = [&](const TableRead &handed) { (void)lin_verify_view(w, c.key, 31, 0, s.views[0], nullptr, nullptr, nullptr, false, &handed); };
      CHECK(ends_with("this reader has passed none", [&] { view_on(untested); }), "a later test runs on a reader that its table's keeper has handed out");
      CHECK(ends_with("tested under another setting", [&] { view_on(at_a_lower_target); }), "a later test runs on a table that was tested at a lower target");
      CHECK(ends_with("tested under another setting", [&] { view_on(with_other_work); }), "a later test runs on a table that was tested with other work bits");
      CHECK(!none.tested_under() && !none_yet.tested_under(), "a table that nobody has made counts as tested");
    }
    twice = twice || s.different_column_rows < s.column_rows;
    for (unsigned r = 0; r + 1 < t.requests; r++) {
      CHECK(s.views[r].shared == s.views[r + 1].shared, "two views of a batch differ in the shared part");
      CHECK(!lin_verify_view(w, c.key, 31, r + 1, s.views[r]), "a view is accepted for another request's claim");
    }
    bytes_of_a_view = s.views[0].first.size() + s.views[0].path.size() + s.views[0].shared.size() + s.views[0].second.size();

    LinSessionOptions off;
    off.off_target = E::one();
    const LinSession f = lin_session(t, w, *c.index, c.key.rec, M, 32, 42, off);
    std::vector<bool> wanted = all;
    wanted[0] = false;
    CHECK(f.served && views_are(w, c.key, 32, f, wanted, t.name), "a false target: the request is accepted, or another one is refused");

    // The table's random tails (random entries encoded with each row) cover `batches` batches; the
    // IndexHolder then serves no more.
    for (unsigned b = 2; b < t.batches; b++) CHECK(lin_session(t, w, *c.index, c.key.rec, M, 33, 43 + b).served, "the keeper refuses too early");
    CHECK(!lin_session(t, w, *c.index, c.key.rec, M, 34, 50).served, "the keeper serves more batches than the random tails bear");

    LinShape bad = w.S;  // W's layout fills its rows: no room for the pad
    bad.w.main_rows = (size_t)((u64(1) << bad.k_col()) / w.codes.base[bad.w.code].ell);
    CHECK(lin_fault(w.codes, bad) != nullptr, "a layout that runs into a pad is admitted");
    {  // Two aborts that protect hiding. No verifier sees what they guard.
      // (1) Omega's random tail covers 1 + N lists of opened positions: the shared test's and one
      //     for each of N requests. A batch with one request more than the codes were sized for
      //     aborts. Tried only where the binding tree, the salted Merkle tree over the seals (final
      //     digests) of a batch's first chains, has a free place (leaf).
      if ((1u << t.log_leaves) > t.requests) {
        LinCase more = t;
        more.requests++;
        CHECK(ends_with("main rows are opened at more positions than their random tail hides", [&] {
                Committed fresh = commit(w, M, 21);
                (void)lin_session(more, w, *fresh.index, fresh.key.rec, M, 31, 41);
              }),
              "a batch with a request more than Omega's random tail covers runs through");
      }
      // (2) Without the LinBegun of lin_begin(), which counts the batch's columns against the
      //     table, the shared part aborts before it folds or opens the table.
      LinSessionOptions uncounted;
      uncounted.not_counted = true;
      CHECK(ends_with("the batch has not been counted against the table's copy", [&] {
              Committed fresh = commit(w, M, 21);
              (void)lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41, uncounted);
            }),
            "a batch that was not counted is served");
    }
    {  // an option for hiding tests (`query_record`): its views must be rejected
      const IndexRecord made_up{std::vector<E>(c.key.rec.y.size(), E::one())};
      LinSessionOptions door;
      door.query_record = &made_up;
      Committed fresh = commit(w, M, 21);
      const LinSession s = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41, door);
      CHECK(s.served && views_are(w, fresh.key, 31, s, std::vector<bool>(t.requests, false), t.name), "a session whose first queries hold a made-up record: a view is accepted");
    }
    // A plan (`Plan`) lists a chain's rounds with their error bounds. At the rounds that draw the
    // shared point (R_LIN_POINT) and the table's columns (R_LOC), it states the table for which the
    // bounds were computed ("priced" in messages). In an altered plan (`bent`), one stated number
    // is raised by 1 (message entries, random-tail entries, row variables, answers), or no table
    // is stated (`said`).
    for (int bend = 0; bend < 10; bend++) {  // altered plans of the shared chain: each side aborts
      const bool at_the_point = bend >= 5;  // else at R_LOC; each round's check aborts by itself
      const int how = bend % 5;
      LinWorld other = w;
      Plan bent(w.shared);
      for (PlanRound &r : bent.rounds_)
        if (r.name == (at_the_point ? R_LIN_POINT : R_LOC) && r.table.said) r.table.code.ell += how == 0, r.table.code.r += how == 1, r.table.v += how == 2, r.table.answers += how == 3, r.table.said = how != 4;
      other.shared = bent;
      const char *const words = "rounds: the round draws columns of another table than its planned error is priced for, or the plan names none";
      Committed fresh = commit(w, M, 21);
      const LinSession honest = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41);
      CHECK(honest.served && ends_with(words, [&] { (void)lin_verify_view(other, fresh.key, 31, 0, honest.views[0]); }), "a verifier runs under a plan that is priced for another table");
      CHECK(ends_with(words, [&] {
              Committed again = commit(w, M, 21);
              (void)lin_session(t, other, *again.index, again.key.rec, M, 31, 41);
            }),
            "a prover runs under a plan that is priced for another table");
    }
    {  // The plan's error bounds are for a batch's test (TableTest::LATER): a side that runs the
       // one-time test aborts.
      LinWorld other = w;
      other.test = TableTest::ONE_TIME;
      const char *const words = "rounds: the round draws columns of another table than its planned error is priced for, or the plan names none";
      Committed fresh = commit(w, M, 21);
      const LinSession honest = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41);
      CHECK(honest.served && ends_with(words, [&] { (void)lin_verify_view(other, fresh.key, 31, 0, honest.views[0]); }), "a verifier runs the one-time test under a plan that is priced for a later one");
      CHECK(ends_with(words, [&] {
              Committed again = commit(w, M, 21);
              (void)lin_session(t, other, *again.index, again.key.rec, M, 31, 41);
            }),
            "a prover runs the one-time test under a plan that is priced for a later one");
      other.test = TableTest::LINK;
      CHECK(ends_with("LIN: a test on the table is the one-time test or a later one", [&] { (void)lin_verify_view(other, fresh.key, 31, 0, honest.views[0]); }), "LIN runs as a link's test");
    }
    // The plan states twice as many requests at the shared point as the binding tree has places,
    // half as many (or one more), or no number: each side aborts.
    for (int bend = 0; bend < 3; bend++) {
      LinWorld other = w;
      const u64 places = u64(1) << w.binding.log_leaves;
      Plan bent(w.shared);
      for (PlanRound &r : bent.rounds_)
        if (r.name == R_LIN_POINT) r.sharing = bend == 0 ? 2 * places : bend == 1 ? (places > 1 ? places / 2 : places + 1) : 0;
      other.shared = bent;
      const char *const words = "rounds: the round draws a point for another number of requests than its planned error is priced for, or the plan names none";
      Committed fresh = commit(w, M, 21);
      const LinSession honest = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41);
      CHECK(honest.served && ends_with(words, [&] { (void)lin_verify_view(other, fresh.key, 31, 0, honest.views[0]); }), "a verifier runs under a plan that counts another number of requests at the point than the tree has places");
      CHECK(ends_with(words, [&] {
              Committed again = commit(w, M, 21);
              (void)lin_session(t, other, *again.index, again.key.rec, M, 31, 41);
            }),
            "a prover runs under a plan that counts another number of requests at the point than the tree has places");
    }
    {  // A binding tree's root fixes its depth (LinBindingRoot): the same seal, salt and siblings,
       // read as a path of another depth, do not lead to that root.
      TestCoins coins(77);
      LinBindingShape deep;
      deep.log_leaves = 3;
      LinBinding tree(deep, w.configuration, coins);
      Digest seal{};
      seal.w[0] = 5;
      tree.put(2, seal);
      const LinBindingRoot root = tree.root();
      Writer path;
      tree.write_path(path, 2);
      CHECK(root.places() == 8, "a tree of depth 3 has 8 places");
      for (unsigned depth = 0; depth <= 5; depth++) {
        LinBindingShape read_as = deep;
        read_as.log_leaves = depth;
        std::vector<u8> bytes = path.bytes();
        bytes.resize(bytes.size() + 2 * DIGEST_BYTES, 0);  // room for a longer path
        Reader r(bytes.data(), bytes.size());
        LinBindingRoot got;
        const bool read = lin_binding_root(r, read_as, w.configuration, seal, got);
        if (depth == 3)
          CHECK(read && got.digest() == root.digest() && got.places() == 8, "control: the path leads to the root, which comes with the tree's places");
        else
          CHECK(!read || !(got.digest() == root.digest()), "a path read at another depth leads to the same root");
      }
    }
    {  // `lin_weights_of_the_prover` changes no byte of a proof: the same batch with a claim's row
       // weights added up into one written-out Tensor, and with none added up.
      const size_t dial = lin_written_out_from;
      std::vector<LinView> views[2];
      for (int always = 0; always < 2; always++) {
        lin_written_out_from = always ? 0 : ~size_t(0);
        Committed fresh = commit(w, M, 21);
        views[always] = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41).views;
      }
      lin_written_out_from = dial;
      bool same = views[0].size() == t.requests && views[1].size() == t.requests;
      for (unsigned c = 0; same && c < t.requests; c++)
        same = views[0][c].first == views[1][c].first && views[0][c].path == views[1][c].path && views[0][c].shared == views[1][c].shared && views[0][c].second == views[1][c].second;
      CHECK(same, "the proof's bytes hang on how the prover writes a claim's row weights");
    }
    {  // Two batches on the same table, matrix and claims with different private randomness: the
       // pad entries of every sampled oracle message (the requests' masks, W, G_u, Omega, the
       // splits' and the code switches' messages), read back from the prover's memory, are never 0
       // and never repeat.
      std::vector<PadLook> one, other;
      LinSessionOptions look;
      Committed first_table = commit(w, M, 21), second_table = commit(w, M, 21);
      look.pads = &one;
      CHECK(lin_session(t, w, *first_table.index, first_table.key.rec, M, 31, 41, look).served, "the keeper does not serve");
      look.pads = &other;
      CHECK(lin_session(t, w, *second_table.index, second_table.key.rec, M, 31, 4141, look).served, "the keeper does not serve");
      const char *fault = pads_fault(one, other);
      CHECK(!fault && one.size() >= 2 + 3 * (size_t)t.requests, fault ? fault : "fewer pads were looked at than the session has sampled messages");
      // The pads of a table, which its committer draws: three entries of F for each
      // row and answer. The answers that they mask stay in the key for the table's life. Two tables
      // committed with different randomness: no pad is 0 and no value repeats, within a table or
      // across both.
      const IndexShape &I = w.S.fold.index;
      Committed another = commit(w, M, 22);
      std::vector<F> all;
      for (const IndexCpu *table : {first_table.index.get(), another.index.get()})
        for (u64 r = 0; r < I.rows(); r++)
          for (size_t p = I.plain(); p < I.code.ell; p++) all.push_back(table->message_entry(r, p));
      // The record reads these entries as its pads: record_value() on the IndexCpu's stored entries
      // gives the record's answers.
      const std::vector<SampleSeed> seeds = record_seeds(I, first_table.key.name());
      for (u64 r = 0; r < I.rows(); r++) {
        std::vector<F> message(I.code.ell);
        for (size_t p = 0; p < I.code.ell; p++) message[p] = first_table.index->message_entry(r, p);
        for (unsigned i = 0; i < I.answers; i++) CHECK(record_value(I, seeds[i], i, message.data()) == first_table.key.rec.y[r * I.answers + i], "the record does not read the keeper's entries");
      }
      std::sort(all.begin(), all.end(), [](const F &a, const F &b) { return a.v < b.v; });
      for (size_t i = 0; i < all.size(); i++) CHECK(!is_zero(all[i]) && (i == 0 || all[i] != all[i - 1]), "a pad of the table is 0, or one value stands at two of its pads");
    }
    {  // `public_fault` (blocks/chain/public.h): a Binder's check on each value that it binds
      CHECK(!public_fault(w.S) && !public_fault(w.binding) && !public_fault(IndexName{c.key.commitment(), c.key.rec}), "an honest public value has a fault");
      LinBindingShape b = w.binding;
      b.salt_bytes = 12;
      CHECK(public_fault(b) != nullptr, "a binding tree with short salts is admitted");
      IndexShape unanswered = w.S.fold.index;
      unanswered.answers = 0;
      CHECK(public_fault(unanswered) != nullptr, "a table's number of answers below 1 is admitted");
      IndexRecord r = c.key.rec;
      r.y.push_back(E::one());
      CHECK(record_fault(w.S.fold.index, r) != nullptr, "a record with an answer too many is admitted");
      IndexKey k = c.key;
      k.rec = r;
      CHECK(k.fault() != nullptr, "a key with an answer too many is admitted");
      k = c.key, k.made_under = Digest{};
      CHECK(k.fault() != nullptr, "a key that names no configuration is admitted");
      CHECK(public_fault(IndexName{{}, c.key.rec}) != nullptr, "a table without a commitment is admitted");
      CHECK(c.key.commitment().size() == DIGEST_BYTES && c.key.name().size() == 2 * DIGEST_BYTES + 6 * 8, "a table's commitment is its root, its name 112 bytes");
      LinShape s = w.S;
      s.fold.columns = 0;
      CHECK(public_fault(s) != nullptr, "a test that draws no column is admitted");
    }
    {  // A table whose tree is hashed under another configuration digest than the chains name: a
       // bug in the calling program; either side aborts.
      Digest other = w.configuration;
      other.w[1] ^= 1;
      TestCoins coins(21);
      IndexCpu index(w.S.fold.index, lin_table_rows(w.S, M), coins, other);
      const IndexKey key = index.key();
      const char *const words = "starts from another configuration's digest";
      CHECK(ends_with(words, [&] { (void)lin_session(t, w, index, key.rec, M, 31, 41); }), "the prover runs on a table of another configuration");
      Committed fresh = commit(w, M, 21);
      const LinSession good = lin_session(t, w, *fresh.index, fresh.key.rec, M, 31, 41);
      CHECK(ends_with(words, [&] { (void)lin_verify_view(w, key, 31, 0, good.views[0]); }), "the verifier runs on a key of another configuration");
      // Likewise with an IndexHolder or a key that states no configuration digest (zeros).
      Committed silent = commit(w, M, 21);
      NamesNoConfiguration keeper(*silent.index);
      CHECK(ends_with(words, [&] { (void)lin_session(t, w, keeper, silent.key.rec, M, 31, 41); }), "the prover runs on a keeper that names no configuration");
      IndexKey mute = fresh.key;
      mute.made_under = Digest{};
      CHECK(ends_with(words, [&] { (void)lin_verify_view(w, mute, 31, 0, good.views[0]); }), "the verifier runs on a key that names no configuration");
      // A key with another leaf-salt length than the shape in the first query: the verifier aborts.
      IndexKey salted = fresh.key;
      salted.table.salt_bytes += 8;
      CHECK(ends_with("another table's key", [&] { (void)lin_verify_view(w, salted, 31, 0, good.views[0]); }), "the verifier runs on a key with another salt's length than the first query says");
      // The second part takes the shared point, Omega's root and its sample claim from the shared
      // chain's context: that chain's seal must be in the second chain's first query.
      CHECK(ends_with("does not hold the seal of the shared chain", [&] { (void)lin_verify_view(w, fresh.key, 31, 0, good.views[0], nullptr, nullptr, nullptr, true); }),
            "the second part runs on the context of another shared chain");
    }
    {  // A wrong layout: the fresh mask of the first fold of
       // the shared test on Omega is sent with Omega under Omega's leaf class, so it shares Omega's
       // tree leaves. lin_alone_in_its_tree(), which both sides call once Omega is sent, returns
       // false for it and true for LIN's own message.
      const std::vector<IoppStage> &test = w.S.shared_split_needed() ? w.S.test_omega : w.S.test_u;
      for (int planted = 0; planted < 2; planted++) {
        Statement st(w.codes);
        TreeBook book(40, toy_configuration(), toy_configuration());
        std::vector<OracleNotice> message = lin_message_omega(w.S);
        if (planted) message.push_back(fold_mask_notice(lin_fold_shape(test, w.S.omega_leaf_class)));
        std::vector<OracleId> ids;
        book.enter(st, message, ids);
        CHECK(lin_alone_in_its_tree(book, ids[0]) == !planted, planted ? "a mask that rides in Omega's leaves is admitted" : "Omega is not alone in its tree");
      }
    }
    {  // (Fold) has no weight on G_u beyond the matrix's entries: none on spare entries, pads, the
       // unused entries before the tail, or the tail.
      TestCoins points(77);
      std::vector<E> psi(w.S.k_col()), zeta(w.S.k_row());
      for (E &x : psi) x = points.e();
      for (E &x : zeta) x = points.e();
      const FoldedRow u{0, vector_lay(w.codes, w.S.fold.u, w.S.fold.rows_high)};
      const Row row = lin_fold_row(w.S, psi, zeta, u, 1, vector_lay(w.codes, w.S.omega, w.S.omega_rows_high));
      bool none = true, some = false;
      for (u64 j = 0; j < w.S.fold.entries(); j++) {
        const bool weight = !is_zero(row.coefficient(0, Part::MAIN, u.lay.index(j)));
        (j < (w.S.matrix_rows << w.S.k_lo) ? some : none) = j < (w.S.matrix_rows << w.S.k_lo) ? some || weight : none && !weight;
      }
      CHECK(some && none, "the row between Omega and the folded row has weight beyond the matrix's entries");
    }
    bad = w.S, bad.shared_split_leaf_class = bad.omega_leaf_class;
    CHECK((lin_fault(w.codes, bad) != nullptr) == w.S.shared_split_needed(), "Omega and the shared split's mask message in one leaf class");
    bad = w.S, bad.own_split_leaf_class = bad.w_leaf_class;
    CHECK((lin_fault(w.codes, bad) != nullptr) == w.S.own_split_needed(), "W and a request's split's mask message in one leaf class");
    bad = w.S, bad.k_lo = 63;
    CHECK(lin_fault(w.codes, bad) != nullptr, "a number of low column bits that makes a shift wrap is admitted");
    bad = w.S, bad.matrix_rows = ~u64(0);
    CHECK(lin_fault(w.codes, bad) != nullptr, "a number of rows that makes a shift wrap is admitted");
  }

  for (u64 again = 0; !twice && again < 50; again++) {  // until some batch draws a column twice
    const LinCase &t = CASES[sizeof CASES / sizeof CASES[0] - 1];
    const LinWorld w = lin_world(t, TableTest::LATER);
    const LinMatrix M = lin_matrix(w.S, 11);
    Committed c = commit(w, M, 100 + again);
    const LinSession s = lin_session(t, w, *c.index, c.key.rec, M, 31, 500 + again);
    CHECK(s.column_rows == t.columns && views_are(w, c.key, 31, s, std::vector<bool>(t.requests, true), t.name), "an honest view is refused");
    twice = s.different_column_rows < s.column_rows;
  }
  CHECK(twice, "no batch draws a column twice: the rule of one row for each draw is not tested");

  // Request 0's view with one bit flipped (each byte in turn), or with one of its four parts a
  // zero byte longer or a byte shorter, must be rejected. Its `target` is not altered.
  for (size_t which : ALTERED) {
    const LinCase &t = CASES[which];
    const LinWorld w = lin_world(t, TableTest::LATER);
    const LinMatrix M = lin_matrix(w.S, 11);
    Committed c = commit(w, M, 21);
    const LinSession s = lin_session(t, w, *c.index, c.key.rec, M, 31, 41);
    size_t accepted = 0, tried = 0;
    for (int part = 0; part < 4; part++) {
      LinView v = s.views[0];
      std::vector<u8> &b = part == 0 ? v.first : part == 1 ? v.path : part == 2 ? v.shared : v.second;
      for (size_t i = 0; i < b.size(); i++) {
        b[i] ^= (u8)(1u << (i % 8));
        accepted += lin_verify_view(w, c.key, 31, 0, v), tried++;
        b[i] ^= (u8)(1u << (i % 8));
      }
      b.push_back(0);
      accepted += lin_verify_view(w, c.key, 31, 0, v), tried++;
      b.pop_back(), b.pop_back();
      accepted += lin_verify_view(w, c.key, 31, 0, v), tried++;
    }
    if (accepted) std::printf("%zu of %zu altered views are accepted\n", accepted, tried);
    CHECK(accepted == 0, "an altered view is accepted");
  }
  std::printf("%s: %zu cases, last view %zu bytes\n", failures ? "FAILED" : "ok", sizeof CASES / sizeof CASES[0], bytes_of_a_view);
  return failures != 0;
}
