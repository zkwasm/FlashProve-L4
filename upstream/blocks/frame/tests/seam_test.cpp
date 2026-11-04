// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Tests, on whole proof files, how linear claims on the committed matrix M leave a segment
   (blocks/iop/batt_shown.h). A segment is one sub-proof's part of a request's first hash chain
   (Fiat-Shamir transcript). Identifiers and messages say `row` for a claim, `link` for a segment,
   `Index` for the committed table, which holds M, and `entry` for that table's number.
   `prove_batch` (blocks/frame/tests/seam_world.h) proves two requests with real hash chains,
   Merkle trees and proximity tests, sized from a parameter file; `check_proof`
   (blocks/frame/check.h) checks each request's file: its first chain, the batch's shared chain,
   its second chain.

   No model is involved. Each segment multiplies two hidden vectors entry by entry
   (blocks/toy/segments.h) and makes `CLAIMS` claims on M. Their forms follow from the request's
   bytes; their values are sent masked by entries of the segment's mask row (random values).
   `batt_shown` reduces them to two claims. (U), on the segment's mask row only, is tested with the
   segment's group (the messages that one proximity test reads). (E), on M and one entry c of the
   request's index-mask message (its masks for claims on M), goes on to blocks/iop/member.h and is
   tested in the request's proximity test on its second chain.

   Honest files must be accepted and be no longer than planned, and a file under another request
   must be rejected. Every prover here is honest: this test does not show that `check_proof`
   rejects a false (U) or (E), or a batch proved for another matrix. Three stored false files of
   these kinds, made by a test prover that is not part of this repository, must be rejected by
   `check_proof` in examples/protocol-toy-false-files/stored_false_test.py. */
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // a test may build or alter plans: blocks/numbers/plan.h
#define BLK_TESTED_TABLE_FOR_A_TEST   // a test may vouch for a table: blocks/iop/foldeq.h
#include "frame/tests/seam_world.h"

// Set by check(): a rejection's reason and step (empty after a yes); an accepted request's place
// in its batch.
static const char *last_why = "", *last_at = "";
static u64 last_place = 0;
static bool check(const Configured &w, const IndexKey &key, u32 entry, const std::vector<u8> &request, const E &reply, const std::vector<u8> &bytes) {
  Reader file(bytes.data(), bytes.size());
  const char *why = nullptr, *at = "not written";
  ProofFacts facts;
  // check_proof()'s error bound assumes a table that check_key() or check_entry() has accepted, so
  // it takes a `TestedTable` (blocks/iop/foldeq.h). No key file is made here: the test vouches for
  // the table.
  const ProofVerdict verdict = check_proof(w, on_a_tests_word(w, key), entry, request, request, reply, file, toy_links, nullptr, why, &facts, &at);
  const bool yes = verdict != ProofVerdict::NO;
  if (yes != !at) failures++, std::printf("FAILED: where a refusal fell is said behind a yes, or not behind a no\n");
  last_at = at ? at : "";
  if (yes && (verdict == ProofVerdict::YES_WITH_WORK) != w.needs_work()) failures++, std::printf("FAILED: the verdict does not say whether work was needed\n");
  last_why = why ? why : "";
  last_place = yes ? facts.place : ~(u64)0;
  return yes;
}

int main() {
  const Configured w = configure({seam_file(), toy_model()}, toy_shapes);
  if (!w.refused().empty()) return std::printf("FAILED: %s\n", w.refused().c_str()), 1;
  const LinMatrix M = lin_matrix(w.fixed().lin, 11);
  TestCoins commit_coins(21);
  IndexCpu index(w.fixed().lin.fold.index, lin_table_rows(w.fixed().lin, M), commit_coins, w.fixed().configuration);
  const IndexKey key = index.key();
  const IndexName table{key.commitment(), key.rec};

  const std::vector<std::vector<u8>> requests = {{1, 2, 3, 4}, {5, 6, 7}};
  const std::vector<E> replies = {E::from(from_u64(1001)), E::from(from_u64(1002))};
  const u32 entry = 5;
  std::vector<PadLook> pads_one, pads_other;
  // LIN (blocks/iop/lin.h) proves a batch's claims on the table. Its messages: W, one request's
  // combination of M's rows; the folded row, a combination of the table's rows, which is compared
  // with opened columns; Omega, M's columns combined at a shared point. A keeper (`IndexHolder`)
  // gives the prover a table's root, folded row, sums for Omega and opened columns. Here the keeper
  // is `index`, the committed table, and the batch is proved for M.
  //
  // Not tried here: a batch in which only the table's item (`IndexName`: root and record) in the
  // first queries, the hash inputs that start the chains, is the committed table's. All else is
  // another table's, proved honestly, so every claim is true. What must reject is the one check
  // that ties a proof to the key's root: that of the opened columns' paths (IndexKey::read(),
  // blocks/iop/foldeq.h). One stored false file is of this kind, and the verdict listed for it is
  // that check's message (examples/protocol-toy-false-files/LIST.tsv).
  const std::vector<std::vector<u8>> files = prove_batch(w, index, table, M, entry, requests, replies, 41, &pads_one);
  CHECK(files.size() == 2, "the keeper does not serve");
  if (failures) return 1;
  // `batt_shown` ends with a challenge beta, then its sumcheck's end value, masked by beta * c.
  // A prover that could choose c after beta could prove a false claim: hence the index-mask message
  // is sent in the request's first prover message.
  const PartSize planned = merged(w.sized().d.set, {w.proof()})->size_of(0);
  for (unsigned c = 0; c < 2; c++) {
    CHECK(check(w, key, entry, requests[c], replies[c], files[c]) && last_place == c, "an honest file is refused");
    if (*last_why) std::printf("why: %s\n", last_why);
    CHECK(first_chain_passes(w, table, entry, requests[c], files[c]), "the first chain alone is refused");
    CHECK(files[c].size() <= planned.bytes, "a file is longer than planned");
  }
  CHECK(!check(w, key, entry, requests[1], replies[0], files[0]), "a file is ACCEPTED under another request");
  // No verifier can see a pad (the random mask on a sample's answer, blocks/iop/send.h) that was
  // never drawn. So the batch is proved again with other private randomness (`coins` in
  // identifiers), and every sampled message's pad is read from the prover's memory: of both runs'
  // pads none may be 0 and no two equal (pads_fault()).
  {
    CHECK(prove_batch(w, index, table, M, entry, requests, replies, 43, &pads_other).size() == 2, "the keeper does not serve a second batch");
    const char *fault = pads_fault(pads_one, pads_other);
    if (fault) std::printf("%s\n", fault);
    CHECK(!fault && !pads_one.empty(), "a pad is not drawn afresh");
  }
  CHECK(!configure({seam_file(), "links 3\npending 2\nmatrix 100\nidentifiers 1 2 3\n"}, toy_shapes).refused().empty(), "public files that leave fewer rows on the table than there are links are admitted");
  std::printf("a file: %zu bytes (planned: at most %llu)\n", files[0].size(), (unsigned long long)planned.bytes);
  std::printf("%s\n", failures ? "FAILED" : "ok");
  return failures != 0;
}
