// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Checks a whole proof file with the library's verifiers (`check_proof`) and builds the first
   queries of its three hash chains (Fiat-Shamir transcripts), in functions that prover and verifier
   both call. A first query is the hash input that fixes a chain's public inputs before the prover
   speaks. Both sides take a `Configured` (blocks/frame/configure.h), made from the public files,
   in part by the calling program's `ShapesOf` function; nothing checks that this function depends
   on those files only (see `Shapes` there).

   A proof file is
     header (`head` in identifiers) | first chain | place | shared chain | second chain
   (described in blocks/frame/FORMAT.md, "4. The files"); no part has a length field. All proofs of
   a batch repeat the shared chain; the other two chains are the request's own. The place is the
   request's leaf number, leaf salt and path in the binding tree, the Merkle tree over the seals
   (final digests) of the batch's first chains.

   `check_proof` knows no model. After the chain's salt (32 bytes drawn by the prover, see
   `RequestPublic`), the first chain begins with the request's segments (`link` in identifiers and
   messages), each holding one sub-proof: the argument for one part of a model's computation. The
   caller's callback checks them. The committed table (`Index` in identifiers) must have passed
   `check_key` or `check_entry`: in this process, or see `check_proof_if_tested`. */
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "chain/rounds.h"
#include "frame/configure.h"
#include "frame/heads.h"
#include "iop/member.h"
#include "iop/req_rows.h"

namespace blk {

/* The public inputs of a request's first chain, as its first query binds them in two lists:
   parameters, then statement. The parameters are fixed before a request is known; `Binder`
   (blocks/chain/public.h) writes the format's version and the plan (the chain's list of rounds)
   before them. The committed table's `IndexName` is its commitment and its record: answers to
   out-of-domain samples of its rows, given once, right after it was committed. It comes from a key
   file or a leaf file (`entry` in identifiers),
   which a committer wrote, so it is in the statement: such a file sets no parameter. The `reply`
   (for a model, the next token) is in neither list: it is one unmasked scalar in a round of its
   own, right before the last segment's first prover message. Nor is the chain's salt: the chain's
   first 32 bytes, which the prover draws for each run. */
struct RequestPublic {
  Public<Digest> configuration;
  LinPublic lin;  // error-correcting codes (`Codes`), shape of the test on the table, `IndexName`
  Public<ReqShape> shape;
  Public<ReqLayout> layout;
  Public<std::vector<u64>> identifiers;
  Public<u64> pending_rows;
  Public<LinBindingShape> binding;
  Public<u64> entry;
  Public<std::vector<u8>> request;
  Public<std::vector<u8>> file_words;  // the caller's request itself, byte for byte: `request` may hold it only as a digest
};
inline RequestPublic bind_request_parameters(Binder &b, const Configured &c) {
  const RequestFixed &f = c.fixed();
  RequestPublic x;
  x.configuration = b.configuration(f.configuration);
  lin_bind_parameters(b, f.codes, f.lin, x.lin);
  x.shape = b.parameter("request's shape", f.shape);
  x.layout = b.parameter("layout", f.layout);
  x.identifiers = b.parameter("links' identifiers", f.identifiers);
  x.pending_rows = b.parameter("pending rows", f.pending_rows);
  x.binding = b.parameter("binding tree", f.binding);
  return x;
}
// The first item of the statement closes the parameters: bind every parameter before this call.
inline void bind_request_statement(Binder &b, RequestPublic &x, const IndexName &table, u32 entry, const std::vector<u8> &request, const std::vector<u8> &file_words) {
  lin_bind_table(b, table, x.lin);
  x.entry = b.statement("stored copy", (u64)entry);
  x.request = b.statement("request", request);
  x.file_words = b.statement("the request file's words", file_words);
}
// Every chain binds the configuration digest (a hash of the public files): the hashes of its trees'
// vertices start from it.
inline LinPublic bind_shared_chain(Binder &b, const Configured &c, const IndexName &table, const LinBindingRoot &binding_root) {
  b.configuration(c.fixed().configuration);
  return lin_bind_shared(b, c.fixed().codes, c.fixed().lin, table, binding_root);
}
inline void bind_second_chain(Binder &b, const Configured &c, const Sealed &own, const Sealed &shared) {
  b.configuration(c.fixed().configuration);
  lin_bind_second(b, own, shared);
}

/* Filled in by the caller's callback (`LinksCheck`), which must check all the segments, the
   proximity test after the last, and any rounds planned by `Subproofs::after_links`
   (blocks/frame/files.h).
   `index_masks`: the oracle message with the masks for the request's linear claims on the
   committed table. `pending`: those claims ("rows on the table" in messages), left for LIN, the
   test on the table. `check_proof` ends the process if the callback accepts without setting
   `replied`, with the expected reply but another number of claims than the parameter
   `pending_rows`, or after it or `more` (`MoreParameters`) has changed the `key_public_bytes` of
   `check_proof`'s `Configured`. `check_proof` looks for no other change to that object. */
struct LinksResult {
  OracleId index_masks = NONE_ID;
  std::vector<TensorRow> pending;
  bool replied = false;  // set where the reply is taken off the channel (`VerifierRounds`)
  E reply = E::zero();
};
using LinksCheck = std::function<bool(VerifierContext &, const RequestPublic &, LinksResult &)>;
using MoreParameters = std::function<void(Binder &)>;
// What a caller may want to print of an accepted proof:
// the first chain's last value, which differs from run to run because the prover draws a new chain
// salt for each, and the request's place in its batch.
struct ProofFacts {
  Digest seal{};
  u64 place = 0;
};

/* A verdict other than NO means that the file was accepted, under the configuration `c`, as a proof
   that `reply` is the reply to `request` from the weights in `tested`, but only as far as `links`
   checks a model's computation: the callers in this directory check only toy segments.
   YES_WITH_WORK: under `c`, some round of a proof file's chains meets the error target (2^-target,
   `Setting::target`, for each round) only with its proof of work.
   Only the test that each batch runs (TableTest::LATER) runs on the table. Its error bound relies
   on the committer's one-time test, in the key file, and for a leaf table ("stored copy" in labels
   and messages; a further table of the same weights) also on the proof in its leaf file that ties
   it to the tested table (the protocol LINK, blocks/iop/link.h: not the `link` of identifiers,
   which is a segment).
   tested: the table, as `check_key` or, for a leaf table, `check_entry` filled it in on acceptance
     (`TestedTable`, defined in blocks/iop/foldeq.h). In a build with BLK_TESTED_TABLE_FOR_A_TEST,
     `on_a_tests_word` makes one on the caller's word, unchecked. The process ends if the table's
     key was checked under other public values (`key_public_bytes`) than `c`'s.
   entry: `tested`'s number; compared with the file's header, not with `tested`.
   file_words: the caller's request itself, byte for byte (for a model: the tokens and the range).
     It is an item of the first query beside `request`, so that every challenge hangs on it
     through this library's hash alone, whatever digests of it `request` holds.
   file: the whole proof file and no more, positioned at its first byte.
   reply: compared with the scalar that `links` received; nothing else compares the two.
   more: null, or binds further parameters of the first chain: public data that the caller's
     sub-proof verifiers read, such as lookup tables.
   why: the first reason for a NO, otherwise null. facts: filled unless the verdict is NO.
   at: null, or receives where in the file's order the rejection came (null on acceptance). For
     logs, since `why` need not show it: where a chain's verifier rejects, `why` is the text of
     `VerifierRounds::why` if it has one, and those texts are the same in each chain.
     This file only writes `*at`: no decision reads it. (`ProofFacts::place` is another thing:
     the request's place in its batch.) */
[[nodiscard]] inline ProofVerdict check_proof(const Configured &c, const TestedTable &tested, u32 entry, const std::vector<u8> &request, const std::vector<u8> &file_words, const E &reply, Reader &file, const LinksCheck &links,
                                      const MoreParameters &more, const char *&why, ProofFacts *facts = nullptr, const char **at = nullptr) {
  const IndexReader &table = tested.reader();
  auto stands = [&](const char *where) {
    if (at) *at = where;
  };
  auto no = [&](const VerifierRounds &ch, const char *where) { return stands(where), why = ch.why() ? ch.why() : where, ProofVerdict::NO; };
  stands("before the chains");
  if (!c.refused().empty()) return why = "the public files are not admitted", ProofVerdict::NO;
  // NO, not the end of the process: `check_key` and `check_entry` fill their table only on
  // acceptance, so a default-constructed `tested` is still empty after a rejection, which a file's
  // bytes can cause. The BLK_REQUIRE below depends only on the caller's objects.
  if (!tested.tested_under()) return why = "the table has passed no check: its key or entry was refused, or was never checked", ProofVerdict::NO;
  auto same_key_values = [&] { return tested.key_public() == key_public_bytes(key_fixed(c.sized(), c.configuration())); };
  BLK_REQUIRE(same_key_values(),
              "check_proof: the table's key was checked under other public values (ranges, rows, codes) than this configuration's: key, entry and proof must be checked under one Configured");
  const FilePlan &plans = c.proof();
  const IndexName name{table.commitment(), table.record()};
  u32 made_on = 0;
  if ((why = read_head(file, TAG_PROOF, made_on))) return ProofVerdict::NO;
  if (made_on != entry) return why = "the proof was made on another stored copy of the table", ProofVerdict::NO;

  Binder first_b(LABEL_FIRST_CHAIN, plans.chains[0].plan);
  RequestPublic x = bind_request_parameters(first_b, c);
  if (more) more(first_b);
  bind_request_statement(first_b, x, name, entry, request, file_words);
  VerifierSession session;  // one per verdict: all three chains share it
  VerifierRounds first_ch(first_b, file, session);
  VerifierContext first_V(x.lin.codes, first_ch);
  LinksResult r;
  if (!links(first_V, x, r)) return no(first_ch, "first chain: the links");
  BLK_REQUIRE(same_key_values(), "check_proof: the caller's routine for the links has changed the configuration under which this proof is being checked");  // c is the caller's object, taken by reference
  BLK_REQUIRE(r.replied, "check_proof: the links hand out no reply");
  if (!(r.reply == reply)) return stands("first chain: the reply"), why = "the proof is about another reply", ProofVerdict::NO;
  BLK_REQUIRE(r.pending.size() == first_ch.get(x.pending_rows), "check_proof: the links leave another number of rows on the table than the first query says");
  LinFirst first;
  if (!member_first_verify(first_V, x.lin, r.index_masks, std::move(r.pending), first)) return no(first_ch, "first chain: the rows on the table");
  Sealed seal;
  if (!first_ch.seal(R_SEAL, seal) || !first_ch.finish_inside_a_file()) return no(first_ch, "first chain: its end");

  stands("between the first chain and the shared chain");
  LinBindingRoot binding_root;
  u64 place = 0;
  {
    Reader ahead = file;  // the place's first item, read once more for the caller
    (void)ahead.get_u64(place);
  }
  if (!lin_binding_root(file, first_ch.get(x.binding), first_ch.configuration(), seal.digest(), binding_root)) return why = "the request's place in its batch", ProofVerdict::NO;

  Binder shared_b(LABEL_PROOF_SHARED_CHAIN, plans.chains[1].plan);
  const LinPublic shared_x = bind_shared_chain(shared_b, c, name, binding_root);
  VerifierRounds shared_ch(shared_b, file, session);
  VerifierContext shared_V(shared_x.codes, shared_ch);
  LinShared shared;
  if (!lin_shared_verify(shared_V, shared_x, TableTest::LATER, tested, shared)) return no(shared_ch, "shared chain");
  if (!lin_shared_tests_verify(shared_V, shared_x, shared)) return no(shared_ch, "shared chain: a proximity test");
  Sealed shared_seal;
  if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish_inside_a_file()) return no(shared_ch, "shared chain: its end");
  shared_V.st.require_all_tested();

  Binder second_b(LABEL_PROOF_SECOND_CHAIN, plans.chains[2].plan);
  bind_second_chain(second_b, c, seal, shared_seal);
  VerifierRounds second_ch(second_b, file, session);
  VerifierContext V = lin_go_on(std::move(first_V), second_ch);
  LinSecond second;
  if (!lin_second_verify(V, x.lin, first, shared_V, shared, second)) return no(second_ch, "second chain");
  if (!lin_second_tests_verify(V, x.lin, first, second)) return no(second_ch, "second chain: a proximity test");
  if (!second_ch.finish()) return no(second_ch, "second chain: its end");  // also: no byte is left over
  V.st.require_all_tested();
  stands("behind the chains");
  if (!session.accepted()) return why = "a chain of the proof was refused", ProofVerdict::NO;  // false if a chain rejected; else ends the process if one is unfinished
  if (facts) facts->seal = seal.digest(), facts->place = place;
  why = nullptr, stands(nullptr);
  return c.needs_work() ? ProofVerdict::YES_WITH_WORK : ProofVerdict::YES;
}
/* `check_proof` for a program that checks the key file and the leaf file in other processes, as
   the verifier program cp_verify (verifier/src/cp_verify.cpp) does. `table` was read from its file
   without the proofs that follow it there (`read_key_table_unproved`,
   `read_entry_table_unproved`). A result other than NO is worth a yes only if those checks have
   passed on the same bytes; nothing here checks that. It comes with IF_TESTED_WHY in `why` and
   `*at`. */
[[nodiscard]] inline IfTested check_proof_if_tested(const Configured &c, const UnprovedTable &table, u32 entry, const std::vector<u8> &request, const std::vector<u8> &file_words, const E &reply, Reader &file, const LinksCheck &links,
                                                    const MoreParameters &more, const char *&why, ProofFacts *facts = nullptr, const char **at = nullptr) {
  const ProofVerdict verdict = check_proof(c, table.as_if_tested(), entry, request, file_words, reply, file, links, more, why, facts, at);
  if (verdict == ProofVerdict::NO) return IfTested::NO;
  why = IF_TESTED_WHY;  // never null here, so that "no reason was given" cannot be taken for a yes
  if (at) *at = IF_TESTED_WHY;
  return verdict == ProofVerdict::YES ? IfTested::YES_IF_THE_TABLE_IS_TESTED : IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED;
}

}  // namespace blk
