// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// LINK: proves that a further committed table has the same entries as the first one.
//
// A committed table (`Index` in identifiers, foldeq.h) is 2^v encoded rows under a Merkle root. The
// rows' random tails hide only a bounded number of opened columns, so a committer makes further
// tables of the same rows with fresh randomness (`coins` in identifiers): a commitment's anchor is
// the first table, each leaf table a further one. What was shown once about the first table, that
// its entries are well formed, then holds for a further table if both have the same entries.
//
// Assumed: the first table has passed its one-time test, or a LINK proof that ties it to a table
// that has, so its key (`IndexReader`) determines one table. Inside one process Link::verify()
// checks this: given a first table that is no `TestedTable` (blocks/iop/foldeq.h), it rejects or
// ends the process. Across processes no code in blocks/ checks it: cp_verify, which checks the
// three files in three processes, says yes only if all three have passed, on the same
// bytes (see run_verifier() in blocks/chain/closed.h). Nothing is assumed of the further table. An
// accepted LINK proof (`link` in identifiers and messages; req.h's `link` is unrelated) is meant to
// show that the further key determines a table too, whose rows agree with the first table's in
// every entry before the cut (see LinkShape). Shape, codes and plan (`Planned`: the rounds and
// their error bounds) are configuration: a caller that reads them from a file calls link_fault()
// first, because link_prove() and link_verify() end the process on a fault. Protocol: class Link.
//
// What a link is to prove follows further work that this protocol builds on
// (see THIRD_PARTY.md).
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "chain/public.h"
#include "chain/rounds.h"
#include "field/field.h"
#include "iop/context.h"
#include "iop/foldeq.h"
#include "iop/iopp.h"
#include "iop/relation.h"
#include "iop/send.h"
#include "iop/tensor.h"
#include "numbers/round_names.h"

namespace blk {

// Label of LINK's hash chain (the Fiat-Shamir transcript of one proof): "LNK" in ASCII, then 01.
constexpr u64 LINK_LABEL = 0x4c4e4b01;

// ---------------------------------------------------------------------------------------------------------------------
// The public shape
// ---------------------------------------------------------------------------------------------------------------------

// The public parameters of a LINK proof, each stored once. For each table LINK uses the steps of
// FOLDEQ (blocks/iop/foldeq.h): the table's rows, folded at one challenge point, are sent as a
// fresh oracle message (encoded rows under a Merkle root), here called the folded row. The two
// FOLDEQ shapes, first() and further(), differ only in their oracle message's name and in the
// number of drawn columns.
//
// The cut, cut(), is the first index of a row's message at which two tables of the same rows may
// differ: where the pads begin. A record
// (`IndexRecord`) holds stored answers to out-of-domain samples (random linear forms of a whole
// message, blocks/iop/send.h) of a table's rows; the pads are random entries that mask these
// answers. The cut follows from the tables' shape and is not a parameter. All entries before it are
// compared, also those that no matrix uses and that honest tables fill with zeros: this covers
// whatever a later test on the table can read, and every entry of a table is then either compared
// or fresh randomness. With a shorter cut, a further table could differ in an entry that later
// proofs read.
struct LinkShape {
  IndexShape table;                            // of both tables
  u64 columns_first = 0, columns_further = 0;  // how many columns of each table a LINK proof draws
  // A folded row's oracle message: `rows` main rows over E (the extension field,
  // blocks/field/field.h) under `code`, no mask rows, and as many answers to its sample as the
  // plan's Setting::samples. rows_high: see VectorLay (blocks/iop/foldeq.h). The two messages form
  // one group: the proximity test (`iopp` in identifiers) reads a group's oracle messages at one
  // list of positions.
  CodeId code = 0, mask_code = 0;  // of the one group
  size_t rows = 0;
  bool rows_high = false;
  unsigned leaf_class = 0;  // of both, and of the mask sent with them: all are read at one list
  // If true, the mask for the test's first fold is sent in the same prover message as the two
  // folded rows and shares their tree.
  bool masks_ahead = false;
  std::vector<IoppStage> later;  // the proximity test's stages after the first
  IoppChoice choice;

  unsigned v() const { return table.v; }
  unsigned entry_bits() const { return table.cube_bits(); }
  u64 cut() const { return table.plain(); }  // number of leading message entries compared
  FoldeqShape side(bool of_the_further_table, unsigned samples) const {
    FoldeqShape f;
    f.index = table;
    f.u.name = of_the_further_table ? "the further table's folded row" : "the first table's folded row";
    f.u.code = code, f.u.main_rows = rows, f.u.main_deg = E_DEGREE, f.u.answers = samples;
    f.rows_high = rows_high, f.leaf_class = leaf_class;
    f.columns = of_the_further_table ? columns_further : columns_first;
    return f;
  }
  FoldeqShape first(unsigned samples) const { return side(false, samples); }
  FoldeqShape further(unsigned samples) const { return side(true, samples); }
  // The test's first stage is the group of the two folded rows: each takes an aligned block of
  // 2^ceil_log2(rows) rows; no mask rows except the fold's.
  std::vector<IoppStage> test() const {
    IoppStage group;
    group.code = code, group.mask_code = mask_code, group.k = 1 + ceil_log2(rows), group.h = 0;
    std::vector<IoppStage> st{group};
    st.insert(st.end(), later.begin(), later.end());
    return st;
  }
};

// Returns null, or why S is no valid shape, as far as S shows it without the codes.
inline const char *public_fault(const LinkShape &S) {
  if (const char *f = public_fault(S.table)) return f;
  if (S.columns_first < 1 || S.columns_further < 1) return "no column is drawn";
  if (S.rows < 1) return "a folded row's oracle message has no rows";
  if (S.later.empty()) return "the proximity test has one stage";
  if (const char *f = public_fault(S.later)) return f;
  return public_fault(S.choice);
}

// Returns null, or what is wrong with S, by itself or against the codes that it names. For
// parameters read from a file. samples: the plan's Setting::samples.
inline const char *link_fault(const Codes &codes, const LinkShape &S, unsigned samples) {
  if (const char *f = public_fault(S)) return f;
  if (const char *f = foldeq_fault(codes, S.first(samples))) return f;  // the further side differs only in name and number of columns
  if (S.mask_code >= codes.mask.size()) return "the group names a mask code that is not there";
  return iopp_fault(codes, S.test(), samples);  // the stages' codes and sizes fit together
}

// The test's first fold, as far as its fresh mask needs it: the mask's size follows from the
// stage's k, h and mask code, so the layout's members stay empty.
inline FoldShape link_fold_shape(const LinkShape &S) {
  FoldShape F = iopp_fold_shape(S.test()[0], GroupLayout());
  F.mask_leaf_class = S.leaf_class;
  return F;
}
// The oracle messages that LINK sends before its proximity test, all in one prover message, in
// order. steps_verify() and link_plan() (blocks/frame/link_frame.h) take them from here;
// steps_prove() drafts them in the same order.
inline std::vector<OracleNotice> link_message(const LinkShape &S, unsigned samples) {
  std::vector<OracleNotice> m{OracleNotice{S.first(samples).u, S.leaf_class}, OracleNotice{S.further(samples).u, S.leaf_class}};
  if (S.masks_ahead) m.push_back(fold_mask_notice(link_fold_shape(S)));
  return m;
}

inline void public_bytes(Writer &w, const LinkShape &S) {
  public_bytes(w, S.table), public_bytes(w, S.columns_first), public_bytes(w, S.columns_further);
  public_bytes(w, S.code), public_bytes(w, S.mask_code), public_bytes(w, S.rows), public_bytes(w, S.rows_high), public_bytes(w, S.leaf_class), public_bytes(w, S.masks_ahead);
  public_bytes(w, S.later), public_bytes(w, S.choice);
}

// Handles (`Public`, blocks/chain/public.h) to what link_bind() writes into the chain's first
// query, the hash input that fixes the public inputs before the prover speaks: configuration
// digest, codes and shape as parameters; the two tables, each by commitment and record, as the
// statement. LINK makes its linear claims (`Row` in identifiers; a claim's value is `target`) from
// these, from challenges, from opened symbols and from the samples' answers.
struct LinkPublic {
  Public<Digest> configuration;
  Public<Codes> codes;
  Public<LinkShape> shape;
  Public<IndexName> first, further;
};
inline LinkPublic link_bind(Binder &b, const Digest &configuration, const Codes &codes, const LinkShape &S, const IndexName &first, const IndexName &further) {
  LinkPublic x;
  x.configuration = b.configuration(configuration);  // the first item after the plan
  x.codes = b.parameter("codes", codes), x.shape = b.parameter("the link", S);
  x.first = b.statement("the first table", first), x.further = b.statement("the further table", further);
  return x;
}

// Compares the members that the first query holds (public_bytes()). IndexShape::distances is not
// one: foldeq_columns_verify() (foldeq.h) compares it where a table's columns are read, and ends
// the process if it differs.
inline bool link_same_table_shape(const IndexShape &a, const IndexShape &b) {
  return a.code == b.code && a.v == b.v && a.answers == b.answers && a.salt_bytes == b.salt_bytes;
}
// The hashes of a table's tree start from the digest of the configuration under which the table was
// made; it must equal the chain's. Zeros, the default of IndexReader::configuration() and
// IndexHolder::configuration(), are compared like any other digest.
inline bool link_same_configuration(const Digest &of_the_table, const Digest &of_the_chain) { return of_the_table == of_the_chain; }
// Returns null, or what is wrong with `key`, by itself or against this table shape (the record's
// size too) and configuration. Keys may come from a committer's files: verify() rejects (`refused`
// in identifiers) on a fault, before it reads the proof.
inline const char *link_key_fault(const Digest &configuration, const IndexShape &table, const IndexReader &key) {
  if (const char *f = key.fault()) return f;
  if (!link_same_table_shape(key.shape(), table)) return "a key has another shape than the link's table";
  if (!link_same_configuration(key.configuration(), configuration)) return "a key's tree starts from another configuration's digest than the chain names";
  return record_fault(table, key.record());
}

// ---------------------------------------------------------------------------------------------------------------------
// The claims and the draws
// ---------------------------------------------------------------------------------------------------------------------

// The equality claim:
//   < eq(phi, .) before the cut, first folded message - further folded message > = 0.
// It stops at the cut. The pads and random tails of two tables are independent, so a claim that ran
// on would be false for honest tables, and it would tie the pads of two tables together, which
// nothing else does.
// cut: S.cut(); a parameter only so that a test can use another value.
inline Row link_equality_row(const LinkShape &S, const std::vector<E> &phi, const FoldedRow &first, const FoldedRow &further, u64 cut) {
  BLK_REQUIRE(phi.size() == S.entry_bits(), "LINK: the point of comparison has one coordinate for each bit of an entry's number");
  Row row;
  row.name = "LINK: the two folded rows agree before the cut";
  for (int side = 0; side < 2; side++) {
    const FoldedRow &at = side ? further : first;
    std::vector<Tensor> pieces;
    tensor_below(Tensor::eq(phi.data(), S.entry_bits(), side ? -E::one() : E::one()), 0, S.entry_bits(), cut, pieces);
    for (Tensor &t : pieces) row.add(at.oracle, Part::MAIN, at.lay.on_table(std::move(t)));
  }
  return row;
}

template <class Channel> inline std::vector<E> link_draw(Channel &ch, const char *round, unsigned count) {
  auto from = ch.close_algebraic(round);
  std::vector<E> x(count);
  for (E &e : x) e = from.draw();
  return x;
}

// The positions of the round R_LINK, one verifier message, split in two: the first columns_first
// draws are columns of the first table, the others of the further one. plan_link()
// (blocks/numbers/blocks.h) sizes each list by itself, for its table's distance, but of the plan's
// two lengths the first query has only the sum, which link_lists() compares with the shape's. So
// both sides also compare the first list's length with the plan's where the positions are drawn
// (require_first_list()); a difference ends the process.
struct LinkLists {
  PositionList first, further;
};
inline LinkLists link_lists(const LinkShape &S, const PositionList &drawn) {
  BLK_REQUIRE(drawn.draws().size() == S.columns_first + S.columns_further && drawn.log_n() == S.table.code.log_n,
              "LINK: the plan draws other columns than the shape says");
  LinkLists l;
  l.first = drawn.part(0, S.columns_first), l.further = drawn.part(S.columns_first, S.columns_further);  // each keeps the mark of the chain that drew the list
  return l;
}

// ---------------------------------------------------------------------------------------------------------------------
// The block
// ---------------------------------------------------------------------------------------------------------------------

// What the steps before the proximity test leave for it. The test uses it up.
struct LinkRun {
  std::vector<E> gamma, phi;
  FoldedRow first, further;
  GroupId group = NONE_ID;
  FoldMask mask;  // of the test's first fold, if it was sent with the folded rows (`masks_ahead`)
};

// The values of LinkWhy::step: the part of verify() that rejected.
inline constexpr const char *LINK_REFUSED_KEYS = "the keys", *LINK_REFUSED_STEPS = "before the test", *LINK_REFUSED_TEST = "the proximity test", *LINK_REFUSED_END = "the end";
// Why a LINK proof was rejected. reason: text for a log: link_key_fault()'s, or the channel's
// (`VerifierRounds::why()`) for the first failure that it met (a nonce, an opening, the proximity
// test's last check ..), or, if the channel has none, the step's name. Both are "" after an
// accepted proof.
struct LinkWhy {
  const char *step = "", *reason = "";
};

struct LinkAccess;  // defined only by tests, for provers and verifiers that are wrong in one thing

// The protocol, on one hash chain whose first query holds both tables' commitments and records.
// G_0 and G_j are the folded rows: fresh oracle messages that hold the
// first and the further table's fold at gamma, sum_r eq(r, gamma) row_r, whole (message, pads,
// random tail), as in FOLDEQ (blocks/iop/foldeq.h). SEND (blocks/iop/send.h) sends them and takes
// an out-of-domain sample of each.
//   verifier   gamma, v elements of E: one fold point for both tables
//   prover     G_0 and G_j, in one prover message
//   verifier   the seed of their out-of-domain samples;   prover: the answers
//   verifier   positions, in one message (LinkLists)
//   prover     opens these columns of both tables
//   verifier   phi, one element of E for each bit of a message entry's index
//   then       the proximity test on the one group that holds G_0 and G_j
// The verifier compares the opened columns with the two roots. All else is linear claims on the
// group, left to the proximity test, after which none is open: SEND's sample claims, FOLDEQ's
// claims for each table (one for each draw of a column and IndexShape::answers
// more) and the equality claim. The prover names no value but the samples' answers: every other one
// is 0, a fold of a key's record, or a fold of opened symbols.
//
// Round errors, beside SEND's and the test's (formulas: plan_link() in blocks/numbers/blocks.h):
//   gamma      v / |E|: two tables that differ before the cut have folds that differ there, unless
//              a nonzero multilinear polynomial vanishes at gamma. Added: a proximity term for the
//              further table and for the first.
//              The term bounds correlated agreement of the 2^v rows, proved only up to a third of
//              the minimum distance; what is assumed beyond it: `CountSource` in blocks/numbers/codes.h.
//   positions  the larger of the two lists' errors, not their sum: which of G_0 and G_j holds
//              another row than its table's fold is fixed before the draw, and that table's own
//              list must miss it.
//   phi        (bits of an entry's index) / |E|: two different vectors agree under eq(phi, .) with
//              at most this probability.
//
// Hiding is as in FOLDEQ, for each table (blocks/iop/foldeq.h, at IndexHolder::record() and
// begin(), record_value() and `FoldeqShape`): begin() counts the opened columns against the
// table's random tails, G_0 and G_j each have a fresh pad for their sample, and their unused
// entries are zero. A table has the same name, seeds and record in either role: only its place in
// the first query says whether it is the first or the further table.
//
// Orders that must hold:
//   - Both commitments and both records before gamma: a committer that knows gamma first makes a
//     further table whose difference from the first folds to zero at gamma. prove() and verify()
//     make the first query themselves, so no caller can pass a chain that was begun on anything
//     else.
//   - G_0, G_j, the seed and the answers before the positions (SEND does these four in one call): a
//     prover that knows the positions first fits G_j's random tail to the drawn columns and gives
//     G_j any message it likes.
//   - phi after G_0 and G_j: otherwise two different rows are chosen to agree at phi.
class Link {
 public:
  // Writes the whole LINK proof to `proof`. Returns false, with nothing written, if the random
  // tails of one of the tables no longer cover the columns that the proof draws
  // (IndexHolder::begin()).
  // configuration: the configuration digest, under which both tables were made.
  static bool prove(const Planned &plan, const Digest &configuration, const Codes &codes, const LinkShape &S, IndexHolder &first, IndexHolder &further,
                    Engine &engine, Coins &coins, Writer &proof, NonceSearch search = nullptr) {
    const char *fault = link_fault(codes, S, plan->set.samples);
    BLK_REQUIRE(!fault, fault ? fault : "");
    BLK_REQUIRE(link_same_table_shape(first.shape(), S.table) && link_same_table_shape(further.shape(), S.table),
                "LINK: a keeper holds a table of another shape than its side of the link");
    BLK_REQUIRE(link_same_configuration(first.configuration(), configuration) && link_same_configuration(further.configuration(), configuration),
                "LINK: a table's tree starts from another configuration's digest than the chain names");
    // further.begin() comes first: a fresh table is unlikely to return false, and a false from the
    // second call leaves the columns of the first call spent.
    if (!further.begin(S.columns_further) || !first.begin(S.columns_first)) return false;
    Binder binder(LINK_LABEL, plan);
    const LinkPublic x =
        link_bind(binder, configuration, codes, S, IndexName{first.commitment(), first.record()}, IndexName{further.commitment(), further.record()});
    u8 random_bytes[DIGEST_BYTES];
    coins.bytes(random_bytes, sizeof random_bytes);
    Digest random;
    digest_from_bytes(random_bytes, random);
    ProverRounds ch(binder, random, proof, search);
    ProverContext P(x.codes, ch, engine, coins);
    LinkRun run;
    steps_prove(P, ch.get(x.shape), first, ch.get(x.first).record, further, ch.get(x.further).record, run);
    test_prove(P, ch.get(x.shape), run);
    ch.finish();
    P.st.require_all_tested();
    return true;
  }

  // first_table: a `TestedTable` (blocks/iop/foldeq.h), tested under the plan's `Setting`: LINK's
  // error bound needs a first table that has passed its test. Otherwise the process ends where the
  // table's columns are read, unless verify() rejects earlier.
  // further: the table under test.
  // session: the chains of one verdict (blocks/chain/session.h). LINK proofs that are checked for
  // one verdict share it, and with it what was seen under each table's root.
  // why (may be null): on a rejection, where and why.
  // notes (may be null): intermediate values, for comparing a second verifier with this one; no
  // decision reads them.
  // The caller's contract, and what no code checks (for one, anything across processes or files):
  // see run_verifier() in blocks/chain/closed.h.
  [[nodiscard]] static bool verify(const Planned &plan, const Digest &configuration, const Codes &codes, const LinkShape &S, const TableRead &first_table,
                                   const IndexReader &further, Reader &proof, VerifierSession &session, LinkWhy *why = nullptr,
                                   std::vector<std::pair<std::string, E>> *notes = nullptr) {
    const IndexReader &first = first_table.reader();
    LinkWhy unused;
    LinkWhy &said = why ? *why : unused;
    said = LinkWhy();
    const char *fault = link_fault(codes, S, plan->set.samples);
    BLK_REQUIRE(!fault, fault ? fault : "");
    if ((fault = link_key_fault(configuration, S.table, first)) || (fault = link_key_fault(configuration, S.table, further))) return said = LinkWhy{LINK_REFUSED_KEYS, fault}, false;
    Binder binder(LINK_LABEL, plan);
    const LinkPublic x =
        link_bind(binder, configuration, codes, S, IndexName{first.commitment(), first.record()}, IndexName{further.commitment(), further.record()});
    VerifierRounds ch(binder, proof, session);
    VerifierContext V(x.codes, ch);
    V.notes = notes;
    LinkRun run;
    auto refused_in = [&](const char *step) { return said = LinkWhy{step, ch.why() ? ch.why() : step}, false; };
    if (!steps_verify(V, ch.get(x.shape), first_table, ch.get(x.first).record, further, ch.get(x.further).record, ch.get(x.shape).cut(), run))
      return refused_in(LINK_REFUSED_STEPS);
    if (!test_verify(V, ch.get(x.shape), run)) return refused_in(LINK_REFUSED_TEST);
    if (!ch.finish()) return refused_in(LINK_REFUSED_END);
    V.st.require_all_tested();
    return true;
  }

 private:
  friend struct LinkAccess;

  // From gamma to the equality claim, on a chain whose first query holds this shape and these two
  // tables (link_bind()). require_table() ends the process unless the plan computed the error of
  // the round just closed from S.table's parameters; gamma's error has them in its proximity terms.
  static void steps_prove(ProverContext &P, const LinkShape &S, IndexHolder &first, const IndexRecord &first_record, IndexHolder &further,
                          const IndexRecord &further_record, LinkRun &out) {
    out.gamma = link_draw(P.ch, R_LINK_POINT, S.v());
    P.ch.require_table(priced_table(S.table, TableTest::LINK));  // gamma's error is for this table
    const unsigned samples = P.ch.plan()->set.samples;
    const FoldeqShape side_first = S.first(samples), side_further = S.further(samples);
    const std::vector<E> eq_gamma = foldeq_weights(side_first, out.gamma.data());
    out.group = P.st.new_group("LINK: the two folded rows", S.code, S.mask_code);

    std::vector<OracleDraft> drafts;
    drafts.push_back(foldeq_draft(P, first, first_record, side_first, out.gamma.data(), nullptr));
    drafts.push_back(foldeq_draft(P, further, further_record, side_further, out.gamma.data(), nullptr));
    if (S.masks_ahead) {
      const FoldShape F = link_fold_shape(S);
      out.mask = fold_mask_make(F, P);
      drafts.push_back(fold_mask_draft(P.st.codes(), F, out.mask));
    }
    const std::vector<OracleId> ids = send_prove(P, std::move(drafts));
    if (S.masks_ahead) out.mask.sent_as(ids[2]);
    out.first = foldeq_joined(P.st, side_first, first.name(), first_record, eq_gamma, ids[0], out.group);
    out.further = foldeq_joined(P.st, side_further, further.name(), further_record, eq_gamma, ids[1], out.group);

    auto from = P.ch.close_positions(R_LINK);
    const LinkLists lists = link_lists(S, from.positions(S.table.code.log_n));
    from.require_first_list(S.columns_first);
    from.require_table(priced_table(S.table, TableTest::LINK));
    foldeq_columns_prove(P, first, side_first, eq_gamma, lists.first, out.first, out.group);
    foldeq_columns_prove(P, further, side_further, eq_gamma, lists.further, out.further, out.group);

    out.phi = link_draw(P.ch, R_LINK_PHI, S.entry_bits());
    P.st.add_row(out.group, link_equality_row(S, out.phi, out.first, out.further, S.cut()));
  }

  // cut: S.cut(); an argument only so that a test's scratch verifier can be wrong in it. The
  // records are those in the first query. Of each reader it uses name(), which gives the record's
  // seeds, and, in foldeq_columns_verify(), shape().distances and read(), which checks the columns
  // against the root; of first_table also tested_under().
  static bool steps_verify(VerifierContext &V, const LinkShape &S, const TableRead &first_table, const IndexRecord &first_record, const IndexReader &further,
                           const IndexRecord &further_record, u64 cut, LinkRun &out) {
    const IndexReader &first = first_table.reader();
    out.gamma = link_draw(V.ch, R_LINK_POINT, S.v());
    V.ch.require_table(priced_table(S.table, TableTest::LINK));  // as the prover's side
    const unsigned samples = V.ch.plan()->set.samples;
    const FoldeqShape side_first = S.first(samples), side_further = S.further(samples);
    const std::vector<E> eq_gamma = foldeq_weights(side_first, out.gamma.data());
    out.group = V.st.new_group("LINK: the two folded rows", S.code, S.mask_code);

    std::vector<OracleId> ids;
    if (!send_verify(V, link_message(S, samples), ids)) return false;
    if (S.masks_ahead) out.mask.sent_as(ids[2]);
    out.first = foldeq_joined(V.st, side_first, first.name(), first_record, eq_gamma, ids[0], out.group);
    out.further = foldeq_joined(V.st, side_further, further.name(), further_record, eq_gamma, ids[1], out.group);
    // `rows` so far: `answers` claims per record, the first table's first
    V.note("LINK: the target of the first table's record's first row", V.st.group(out.group).rows.at(0).target);
    V.note("LINK: the target of the further table's record's first row", V.st.group(out.group).rows.at(S.table.answers).target);

    auto from = V.ch.close_positions(R_LINK);
    const PositionList drawn = from.positions(S.table.code.log_n);
    if (!V.ch.ok()) return false;
    from.require_first_list(S.columns_first);  // the plan's split point: see `LinkLists`
    from.require_table(priced_table(S.table, TableTest::LINK));
    const LinkLists lists = link_lists(S, drawn);
    const size_t rows_before = V.st.group(out.group).rows.size();
    if (!foldeq_columns_verify(V, first_table, TableTest::LINK, TableStanding::ALREADY_TESTED, side_first, eq_gamma, lists.first, out.first, out.group)) return false;
    if (!foldeq_columns_verify(V, further, TableTest::LINK, TableStanding::UNDER_TEST, side_further, eq_gamma, lists.further, out.further, out.group)) return false;
    V.note("LINK: the fold of the first table's column at its first draw", V.st.group(out.group).rows[rows_before].target);
    V.note("LINK: the fold of the further table's column at its first draw", V.st.group(out.group).rows[rows_before + S.columns_first].target);

    out.phi = link_draw(V.ch, R_LINK_PHI, S.entry_bits());
    V.st.add_row(out.group, link_equality_row(S, out.phi, out.first, out.further, cut));
    {
      E at_entry_0 = E::one();  // eq(phi, 0)
      for (const E &x : out.phi) at_entry_0 *= E::one() - x;
      V.note("LINK: the equality row's coefficient at entry 0 of the first table's folded row", at_entry_0);
    }
    return V.ch.ok();
  }

  static void test_prove(ProverContext &P, const LinkShape &S, LinkRun &x) {
    iopp_prove(P, x.group, S.test(), S.choice, std::move(x.mask));
    release_oracles(P, {x.first.oracle, x.further.oracle});  // nothing reads the two folded rows again
  }
  static bool test_verify(VerifierContext &V, const LinkShape &S, LinkRun &x) { return iopp_verify(V, x.group, S.test(), S.choice, std::move(x.mask)); }
};

inline bool link_prove(const Planned &plan, const Digest &configuration, const Codes &codes, const LinkShape &S, IndexHolder &first, IndexHolder &further,
                       Engine &engine, Coins &coins, Writer &proof, NonceSearch search = nullptr) {
  return Link::prove(plan, configuration, codes, S, first, further, engine, coins, proof, search);
}
// The caller's contract: see run_verifier() in blocks/chain/closed.h.
[[nodiscard]] inline bool link_verify(const Planned &plan, const Digest &configuration, const Codes &codes, const LinkShape &S, const TableRead &first,
                                      const IndexReader &further, Reader &proof, VerifierSession &session, LinkWhy *why = nullptr,
                                      std::vector<std::pair<std::string, E>> *notes = nullptr) {
  return Link::verify(plan, configuration, codes, S, first, further, proof, session, why, notes);
}

}  // namespace blk
