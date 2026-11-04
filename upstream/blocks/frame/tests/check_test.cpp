// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Runs whole proof files through check_proof() at toy message lengths and the target of 128 bits.
   configure() derives sizes and the plans (lists of rounds) of a proof's three hash chains
   (Fiat-Shamir transcripts) from the public files; a batch of two requests is proved on these plans
   with real hash chains, Merkle trees and proximity tests. In place of a model, each segment of a
   request (`shape.link`, `toy_links`) multiplies two hidden vectors element by element
   (blocks/toy/segments.h); each request leaves a few linear claims on the committed matrix
   (pending claims, `TensorRow`). The reply is one scalar; the toy segments prove nothing about it.

   This test checks that honest files are accepted and no longer than planned, and that these
   are rejected (`refused` in identifiers and output): a file under another request, reply,
   configuration or leaf table number (`entry`; printed: stored copy); a header (`head` in
   identifiers) with another tag or version word; damage (blocks/tests/damage.h) at a sample of
   about 400 places; any mix of the parts of three honest files that is not an honest file.
   No malicious prover runs in this test.
   */
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // blocks/iop/tests/member_world.h builds a Plan by hand
#define BLK_TESTED_TABLE_FOR_A_TEST   // for `on_a_tests_word` (blocks/frame/configure.h) in `check`
#include <cstdio>
#include <sstream>

#include "engine/engine_cpu.h"
#include "frame/check.h"
#include "frame/tests/toy_shapes.h"
#include "iop/tests/member_world.h"
#include "tests/damage.h"
#include "toy/segments.h"

using namespace blk;

// Seed of a request's pending claims (member_row_form()): a public function of the request's bytes.
static u64 seed_of(const std::vector<u8> &request) {
  u64 x = 7;
  for (u8 b : request) x = x * 131 + b;
  return x % 1000;
}

// Proves one batch: a proof file per request, or none if lin_begin() fails. `parts`, if given, gets
// each file's five parts: header, first chain, place in the batch, shared chain, second chain.
using Parts = std::vector<std::vector<u8>>;
static std::vector<std::vector<u8>> prove_batch(const Configured &w, IndexHolder &index, const IndexName &table, const LinMatrix &M, u32 entry,
                                                const std::vector<std::vector<u8>> &requests, const std::vector<E> &replies, u64 coin_seed, std::vector<Parts> *parts = nullptr) {
  const LinShape &S = w.fixed().lin;
  TestCoins coins(coin_seed);
  EngineCpu engine;
  struct Request {
    Writer first, second;
    std::unique_ptr<ProverRounds> ch;
    std::unique_ptr<ProverContext> P;
    RequestPublic x;
    LinFirst lin;
    Sealed seal;
  };
  std::vector<Request> req(requests.size());
  LinBinding binding(w.fixed().binding, w.fixed().configuration, coins);
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(index, S, begun)) return {};
  const Summand<E> summand = toy_summand();
  for (unsigned c = 0; c < requests.size(); c++) {
    Request &r = req[c];
    Binder b(LABEL_FIRST_CHAIN, w.proof().chains[0].plan);
    r.x = bind_request_parameters(b, w);
    bind_request_statement(b, r.x, table, entry, requests[c], requests[c]);  // in the toy a request is its own file
    r.ch.reset(new ProverRounds(b, toy_random_bytes(coin_seed + 100 + c), r.first));
    r.P.reset(new ProverContext(r.x.lin.codes, *r.ch, engine, coins));
    ProverContext &P = *r.P;
    ReqProver frame(P, r.x.shape);
    const Codes &codes = P.st.codes();
    const ReqShape &shape = r.ch->get(r.x.shape);
    const unsigned vars = codes.base[shape.code].log_ell() - 1;
    const size_t len = size_t(1) << vars, ell = codes.base[shape.code].ell;
    std::vector<F> x(len);
    coins.fill(x.data(), len);
    for (size_t l = 0; l < shape.link.size(); l++) {
      std::vector<F> bl(len), cl(len);
      coins.fill(bl.data(), len);
      for (size_t i = 0; i < len; i++) cl[i] = x[i] * bl[i];
      Table own(1, l == 0 ? 2 : 1, ell), product(1, 1, ell);
      for (size_t i = 0; i < len; i++) {
        if (l == 0) own.set(0, i, E::from(x[i]));
        own.set(l == 0 ? 1 : 0, i, E::from(bl[i]));
        product.set(0, i, E::from(cl[i]));
      }
      std::vector<OracleDraft> at_start;
      if (l == 0) at_start.push_back(lin_draft(P, shape.at_start[0].shape, false, shape.at_start[0].leaf_class, {}));
      std::vector<E> plain;
      if (shape.link[l].plain) plain.push_back(replies[c]);
      const std::vector<OracleId> ids = frame.first({engine.put(own), engine.put(product)}, std::move(at_start), Table(), plain);
      const SegmentPlaces at = segments_places(frame, ids);
      std::vector<E> zeta(vars);
      {
        AlgebraicDraw from = r.ch->close_algebraic(R_ZERO);
        for (E &v : zeta) v = from.draw();
      }
      SumClaim<E> claim = toy_claim(zeta);
      std::vector<std::vector<E>> maps(3, std::vector<E>(len));
      for (size_t i = 0; i < len; i++) maps[0][i] = E::from(x[i]), maps[1][i] = E::from(bl[i]), maps[2][i] = E::from(cl[i]);
      SumcheckCpu<E> inner(summand, claim.vars, std::move(maps));
      std::vector<MaskSlot> fresh = P.st.masks(ids[0]).take(psc_masks(claim, summand));
      const ProverContext *p = &P;
      PscOutput<E> out = psc_prove<E>(*r.ch, std::move(claim), summand, inner, std::move(fresh), [p](u32 oracle, u32 i) { return p->behind[oracle].mask.entry(i); });
      segments_rows(P.st, frame.group(), at, std::move(out));
      frame.close();
      x = std::move(cl);
    }
    frame.finish();
    const OracleId masks = frame.at_start()[0];
    std::vector<TensorRow> rows;
    for (size_t j = 0; j < r.ch->get(r.x.pending_rows); j++) {
      TensorRow row = member_row_form(S, seed_of(requests[c]), 0, j);
      LinClaim f;
      f.g = row.g;
      const std::vector<E> wj = lin_w(S, f, M);
      MaskSlot slot = P.st.masks(masks).take();
      row.target = mle_eval(wj.data(), wj.size(), row.psi.data(), S.k_col()) + P.mask_value(slot);
      row.masks.push_back({std::move(slot), E::one()});
      r.ch->send(row.target);
      rows.push_back(std::move(row));
    }
    HeavyCpu heavy(S, M);
    r.lin = member_first_prove(P, r.x.lin, masks, std::move(rows), heavy);
    BLK_REQUIRE(r.ch->seal(R_SEAL, r.seal), "a prover's channel does not refuse");
    r.ch->finish_inside_a_file();
    binding.put(c, r.seal.digest());
  }
  Writer shared_bytes;
  Binder shared_b(LABEL_PROOF_SHARED_CHAIN, w.proof().chains[1].plan);
  const LinPublic shared_x = bind_shared_chain(shared_b, w, table, binding.root());
  ProverRounds shared_ch(shared_b, Digest{}, shared_bytes);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, coins);
  LinShared shared;
  lin_shared_prove(shared_P, shared_x, TableTest::LATER, index, std::move(begun), shared);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish_inside_a_file();
  std::vector<std::vector<u8>> files;
  for (unsigned c = 0; c < requests.size(); c++) {
    Request &r = req[c];
    Binder b(LABEL_PROOF_SECOND_CHAIN, w.proof().chains[2].plan);
    bind_second_chain(b, w, r.seal, shared_seal);
    ProverRounds ch(b, toy_random_bytes(coin_seed + 200 + c), r.second);
    ProverContext P = lin_go_on(std::move(*r.P), ch);
    const LinSecond second = lin_second_prove(P, r.x.lin, r.lin, shared_P, shared);
    lin_second_tests_prove(P, r.x.lin, r.lin, second);
    ch.finish();
    P.st.require_all_tested();
    Writer head, place, file;
    write_head(head, TAG_PROOF, entry);
    binding.write_path(place, c);
    const Parts five = {head.bytes(), r.first.bytes(), place.bytes(), shared_bytes.bytes(), r.second.bytes()};
    for (const std::vector<u8> &part : five) file.put_bytes(part.data(), part.size());
    files.push_back(file.bytes());
    if (parts) parts->push_back(five);
  }
  lin_release(shared_P, shared);
  return files;
}

// Callback for check_proof(): verifies the toy segments, passes on the received reply
// (`out.reply`), then receives each pending claim's value (`target`).
static bool toy_links(VerifierContext &V, const RequestPublic &x, LinksResult &out) {
  VerifierRounds &ch = V.ch;
  ReqVerifier frame(V, x.shape);
  const ReqShape &shape = ch.get(x.shape);
  const unsigned vars = V.st.codes().base[shape.code].log_ell() - 1;
  const Summand<E> summand = toy_summand();
  for (size_t l = 0; l < shape.link.size(); l++) {
    std::vector<OracleId> ids;
    if (!frame.first(ids)) return false;
    if (shape.link[l].plain) out.reply = frame.plain()[0], out.replied = true;
    const SegmentPlaces at = segments_places(frame, ids);
    std::vector<E> zeta(vars);
    {
      AlgebraicDraw from = ch.close_algebraic(R_ZERO);
      for (E &v : zeta) v = from.draw();
    }
    SumClaim<E> claim = toy_claim(zeta);
    std::vector<MaskSlot> fresh = V.st.masks(ids[0]).take(psc_masks(claim, summand));
    PscOutput<E> psc;
    if (!psc_verify<E>(ch, std::move(claim), summand, std::move(fresh), psc)) return false;
    segments_rows(V.st, frame.group(), at, std::move(psc));
    if (!frame.close()) return false;
  }
  if (!frame.finish()) return false;
  out.index_masks = frame.at_start()[0];
  for (size_t j = 0; j < ch.get(x.pending_rows); j++) {
    TensorRow row = member_row_form(ch.get(x.lin.shape), seed_of(ch.get(x.request)), 0, j);
    if (!ch.recv(row.target)) return false;
    row.masks.push_back({V.st.masks(out.index_masks).take(), E::one()});
    out.pending.push_back(std::move(row));
  }
  return true;
}

// Left by the last check(): a rejection's reason and step; an accepted request's place in its batch
static const char *last_why = "", *last_at = "";
static u64 last_place = 0;
// other_words: null, or what the caller hands in as the request file's words in the request's place.
static bool check(const Configured &w, const IndexKey &key, u32 entry, const std::vector<u8> &request, const E &reply, const std::vector<u8> &bytes, const std::vector<u8> *other_words = nullptr) {
  Reader file(bytes.data(), bytes.size());
  const char *why = nullptr, *at = "not written";
  ProofFacts facts;
  // check_proof() takes a TestedTable, which check_key() (key file) or check_entry() (leaf file)
  // yields. This test makes neither file and runs no one-time test (the committer's test of its
  // first table): on_a_tests_word() vouches for the table instead.
  const ProofVerdict verdict = check_proof(w, on_a_tests_word(w, key), entry, request, other_words ? *other_words : request, reply, file, toy_links, nullptr, why, &facts, &at);
  const bool yes = verdict != ProofVerdict::NO;
  if (yes != !at) failures++, std::printf("FAILED: where a refusal fell is said behind a yes, or not behind a no\n");
  last_at = at ? at : "";
  if (yes && (verdict == ProofVerdict::YES_WITH_WORK) != w.needs_work()) failures++, std::printf("FAILED: the verdict does not say whether work was needed\n");
  last_why = why ? why : "";
  last_place = yes ? facts.place : ~(u64)0;
  return yes;
}

int main() {
  const std::string text = toy_file();
  const Configured w = configure({text, toy_model()}, toy_shapes);
  if (!w.refused().empty()) return std::printf("FAILED: %s\n", w.refused().c_str()), 1;
  // What the cases below rely on: by its lists alone this configuration has fewer than 128 bits, and 4 to 8 bits of proof of work make up the rest; so an accepted proof gets `YES_WITH_WORK`, never `YES`.
  CHECK(w.needs_work() && w.bits_without_work() < 128 && w.bits_without_work() + w.work() >= 128 && w.work() >= 4 && w.work() <= 8, "the toy file has work on its lists, and the verdict must say so");
  {  // what a table's limit of opened columns (`bears`) must fit
    // Each leaf table has a LINK proof (`link_on_the_first`, `link_on_the_further`), which ties it
    // to the commitment's first table (the anchor) and opens columns of both. The limit must fit a
    // leaf table's own LINK proof or the anchor's one-time test, plus `batches` batches; with the
    // larger of the two, not one batch more.
    const Configured::TableColumns t = w.table_columns();
    CHECK(t.bears && t.one_time && t.link_on_the_first && t.link_on_the_further && t.a_batch && t.batches, "a number of columns is 0");
    CHECK(t.link_on_the_further + t.batches * t.a_batch <= t.bears && t.one_time + t.batches * t.a_batch <= t.bears, "a stored copy does not bear its batches");
    CHECK(std::max(t.link_on_the_further, t.one_time) + (t.batches + 1) * t.a_batch > t.bears, "a stored copy would bear a batch more");
    // The check below must accept the most LINK proofs on the anchor (`links_on_the_first`) that
    // fit with its one-time test and `batches` batches, and reject one more.
    TableUse use;
    use.links_on_the_first = (t.bears - t.one_time - t.batches * t.a_batch) / t.link_on_the_first;
    CHECK(hiding_conditions_missed_on_the_table(w.sized(), w.fixed().pending_rows, use).empty(), "the gate counts other columns");
    use.links_on_the_first += 1;
    CHECK(!hiding_conditions_missed_on_the_table(w.sized(), w.fixed().pending_rows, use).compare(0, 6, "table:"), "the gate counts other columns");
  }
  const LinMatrix M = lin_matrix(w.fixed().lin, 11);
  TestCoins commit_coins(21);
  IndexCpu index(w.fixed().lin.fold.index, lin_table_rows(w.fixed().lin, M), commit_coins, w.fixed().configuration);
  const IndexKey key = index.key();
  const IndexName table{key.commitment(), key.rec};
  {  // the commitment ID (`published_id`) changes with the table and with the public files
    const Digest id = published_id(w, key);
    TestCoins other_coins(22);
    IndexCpu again(w.fixed().lin.fold.index, lin_table_rows(w.fixed().lin, M), other_coins, w.fixed().configuration);
    CHECK(id == published_id(w, index.key()) && !(id == published_id(w, again.key())), "the commitment ID");
    const Configured longer_files = configure({text, toy_model() + "\n"}, toy_shapes);  // one more byte, the same shapes: under these files the same root has another name and commitment ID
    const IndexKey under_them(key.table, longer_files.configuration(), key.root, key.rec);
    CHECK(!(id == published_id(longer_files, under_them)), "the commitment ID does not hang on the public files");
  }

  const std::vector<std::vector<u8>> requests = {{1, 2, 3, 4}, {5, 6, 7}};
  const std::vector<E> replies = {E::from(from_u64(1001)), E::from(from_u64(1002))};
  const u32 entry = 5;
  std::vector<Parts> parts;
  const std::vector<std::vector<u8>> files = prove_batch(w, index, table, M, entry, requests, replies, 41, &parts);
  CHECK(files.size() == 2, "the keeper does not serve");
  if (failures) return 1;
  const PartSize planned = merged(w.sized().d.set, {w.proof()})->size_of(0);
  for (unsigned c = 0; c < 2; c++) {
    CHECK(check(w, key, entry, requests[c], replies[c], files[c]) && last_place == c, "an honest file is refused");
    if (*last_why) std::printf("why: %s\n", last_why);
    CHECK(files[c].size() <= planned.bytes, "a file is longer than planned");
  }
  const std::vector<u8> &good = files[0];
  auto refused = [&](const std::vector<u8> &bytes, const char *text_) { CHECK(!check(w, key, entry, requests[0], replies[0], bytes), text_); };
  CHECK(!check(w, key, entry, requests[1], replies[0], good), "a file is accepted under another request");
  CHECK(!check(w, key, entry, requests[0], replies[0], good, &requests[1]), "a file is accepted under another request file's words, the item `request` being the same");
  CHECK(!check(w, key, entry, requests[0], replies[1], good) && std::string(last_why) == "the proof is about another reply" && std::string(last_at) == "first chain: the reply", "another reply");
  CHECK(!check(w, key, entry + 1, requests[0], replies[0], good) && std::string(last_at) == "before the chains", "a file is accepted for another stored copy");
  {  // the first chain also hashes `entry`: it rejects a file even if header and call change alike
    std::vector<u8> b = good;
    b[8] ^= 1;
    CHECK(!check(w, key, entry ^ 1, requests[0], replies[0], b) && std::string(last_why) != "the proof was made on another stored copy of the table", "the stored copy's number is a free word");
  }
  {  // other public files: one more byte but the same shapes; other identifiers; rejected files
    const Configured longer = configure({text, toy_model() + "\n"}, toy_shapes), others = configure({text, toy_model("1 9 3")}, toy_shapes);
    CHECK(longer.refused().empty() && others.refused().empty(), "the other files are not admitted");
    CHECK(!check(longer, key, entry, requests[0], replies[0], good), "a file is accepted under another configuration");
    CHECK(!check(others, key, entry, requests[0], replies[0], good), "a file is accepted under other identifiers of the links");
    CHECK(!configure({text, toy_model("1 2")}, toy_shapes).refused().empty() && !configure({text, toy_model("1 2 3 4")}, toy_shapes).refused().empty(),
          "identifiers that are not one for each link are admitted");
    CHECK(!configure({text}, toy_shapes).refused().empty() && !configure({}, toy_shapes).refused().empty() && !configure({text + "x", toy_model()}, toy_shapes).refused().empty(),
          "files that are missing or not in their form are admitted");
    {  // files that miss a hiding condition are rejected: a mask row too short for its sub-proof
      std::string narrow_masks = text;
      const size_t at = narrow_masks.find("group link 1 13 9 13 5");
      CHECK(at != std::string::npos, "the toy file has another line for its links");
      narrow_masks.replace(at, 22, "group link 1 13 9 13 3");
      const Configured leaky = configure({narrow_masks, toy_model()}, toy_shapes);
      CHECK(!leaky.refused().compare(0, 8, "hiding: "), "files that miss a condition of hiding are admitted");
    }
    CHECK(!check(configure({text}, toy_shapes), key, entry, requests[0], replies[0], good) && std::string(last_why) == "the public files are not admitted" && std::string(last_at) == "before the chains", "refused files serve");
  }
  for (size_t at : {size_t(0), size_t(5), size_t(6), size_t(7)}) {  // tag; format, field, hash
    std::vector<u8> b = good;
    b[at] ^= 3;
    refused(b, "a file with another head is accepted");
  }
  // At each place: a bit flipped, a word at or above the modulus p, a cut; also bytes after the
  // end. The environment variable BLK_TEST_DAMAGE ("bits stride") replaces the sample;
  // tests/new/run_new.py, stage `bytes`, asks for one bit of every 7th byte by default.
  const DamageChoice how = damage_choice(good.size(), 400);
  const DamageCount damaged = damage("a proof file", good, how, [&](const u8 *b, size_t n) { return check(w, key, entry, requests[0], replies[0], std::vector<u8>(b, b + n)); });
  CHECK(!damaged.bad, "a damaged file is accepted");
  size_t joined = 0;
  {  // each of parts 1 to 4 is taken from request 0, request 1, or request 0 of a second batch
    std::vector<Parts> other;
    const std::vector<std::vector<u8>> again = prove_batch(w, index, table, M, entry, requests, replies, 43, &other);
    CHECK(again.size() == 2 && check(w, key, entry, requests[0], replies[0], again[0]), "a second honest batch is refused");
    const Parts *from[3] = {&parts[0], &parts[1], &other[0]};
    size_t honest = 0;
    for (unsigned pick = 0; pick < 81; pick++) {
      std::vector<u8> b = parts[0][0];
      for (unsigned part = 1, rest = pick; part <= 4; part++, rest /= 3) b.insert(b.end(), (*from[rest % 3])[part].begin(), (*from[rest % 3])[part].end());
      if (b == good || b == again[0]) {
        honest++;
        continue;
      }
      refused(b, "parts of several honest files are accepted as one file");
      joined++;
    }
    CHECK(honest == 3, "three of the 81 joinings are honest files: each batch's own, and the first with its neighbour's copy of the shared chain, which is the same bytes");
  }
  std::printf("a file: %zu bytes (planned: at most %llu, expected %.0f); refused: %s; %zu joinings of honest files' parts\n", good.size(), (unsigned long long)planned.bytes, planned.expected, damaged.text(how).c_str(), joined);
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
