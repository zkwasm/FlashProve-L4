// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Toy-size test fixture (`world` in identifiers) for LINK (`link` in identifiers,
// blocks/iop/link.h). A LINK proof is to show that a further committed table (`Index` in
// identifiers), a leaf table, agrees with the first (the anchor) in every row's message before the
// cut: where the pads masking its record (stored sample answers) begin.
// LINK folds each table into one row, sent as a fresh oracle message. Hash chain, trees, CPU
// engine and proximity test are the library's; the plan is hand-written, with 0 as every round's
// error. Before any include, define BLK_PLANS_BY_HAND_FOR_A_TEST and BLK_TESTED_TABLE_FOR_A_TEST.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "iop/index_cpu.h"
#include "iop/link.h"
#include "iop/tests/lin_world.h"

namespace blk {

// v: a table has 2^v rows. columns_first, columns_further: columns of each table that one LINK
// proof draws. rows_high: the folded rows' layout (`VectorLay`, blocks/iop/foldeq.h).
struct LinkCase {
  const char *name;
  unsigned v;
  size_t data, spare;     // a row's message entries: holding data; zero in honest tables
  unsigned answers;       // of each table's record, for each row
  unsigned columns_first, columns_further;
  unsigned links;         // a table's random tails allow at least this many LINK proofs
  unsigned log_rate;      // the tables' code has at most the rate 2^-log_rate
  size_t ell_u;           // message length of the code of the two fresh oracle messages
  bool rows_high;
  unsigned draws;         // positions drawn at every stage of the proximity test
  bool ahead;             // fresh masks are sent with earlier prover messages where possible
};

struct LinkWorld {
  Digest configuration = toy_configuration();  // synthetic; bound by the chain and both tables
  Codes codes;
  LinkShape S;
  Planned plan;
};

inline LinkWorld link_world(const LinkCase &t) {
  LinkWorld w;
  Plan plan = toy_empty_plan();
  LinkShape &S = w.S;
  IndexShape I;
  I.v = t.v, I.answers = t.answers, I.salt_bytes = plan.set.salt_bytes;
  I.code.ell = t.data + t.spare + I.pads();
  I.code.r = tail_for_positions((u64)std::max(t.columns_first, t.columns_further) * t.links);
  I.code.log_n = I.code.log_dim() + t.log_rate;
  S.table = I;
  S.columns_first = t.columns_first, S.columns_further = t.columns_further;
  S.rows_high = t.rows_high;
  S.masks_ahead = S.choice.masks_ride = t.ahead;

  w.codes.mask.push_back(toy_code(16, tail_for_positions(t.draws), 0));
  w.codes.base.push_back(toy_code(t.ell_u, tail_for_positions(t.draws), w.codes.mask.back().log_n));
  S.code = 0, S.mask_code = 0;
  const unsigned samples = plan.set.samples;
  S.rows = lin_rows_for(S.first(samples).entries() + samples, t.ell_u, t.rows_high);
  {
    const std::vector<IoppStage> test = lin_stages(w.codes, S.code, S.mask_code, {S.first(samples).u, S.further(samples).u}, t.draws);
    S.later.assign(test.begin() + 1, test.end());
    CHECK(test[0].k == S.test()[0].k && test[0].h == S.test()[0].h, "the group's numbers of row variables, by the shape and by the tests' own routine");
  }

  plan.close(R_LINK_POINT, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = S.v();
  plan.rounds_.back().table = priced_table(I, TableTest::LINK);  // whose rows the point folds
  lin_plan_send(plan, w.codes, link_message(S, samples));
  toy_plan_positions(plan, R_LINK, t.columns_first + t.columns_further, I.code.log_n, 0);
  plan.rounds_.back().first_list = t.columns_first, plan.rounds_.back().table = priced_table(I, TableTest::LINK);  // where the one list is split, and whose columns it opens (blocks/numbers/plan.h)
  plan.close(R_LINK_PHI, RoundKind::ALGEBRAIC, ErrorKind::FIELD, Bound((u64)0), Priced::NOTHING).draw_E = S.entry_bits();
  // Linear claims (`Row`) that the proximity test batches: both records', one per column draw,
  // the equality claim, and one sample claim per folded row.
  toy_plan_iopp(plan, w.codes, S.test(), t.draws, S.choice, S.masks_ahead, 2 * (u64)t.answers + t.columns_first + t.columns_further + 1 + 2);
  w.plan = plan;
  return w;
}

// The rows of a table, without pads: synthetic data, then zeros.
inline Table link_rows(const LinkCase &t, const LinkWorld &w, u64 seed) {
  Table rows(1, w.S.table.rows(), w.S.table.plain());
  TestCoins data(seed);
  for (u64 r = 0; r < rows.rows; r++)
    for (size_t j = 0; j < t.data; j++) {
      F x;
      data.fill(&x, 1);
      rows.set(r, j, E::from(x));
    }
  return rows;
}
inline void link_change(Table &rows, u64 r, size_t j) { rows.set(r, j, rows.at(r, j) + E::one()); }

// A committed table's prover side (`index`) and verifier side (`key`).
struct LinkTable {
  std::unique_ptr<IndexCpu> index;
  IndexKey key;
};
inline LinkTable link_commit(const LinkWorld &w, const Table &rows, u64 coin_seed) {
  LinkTable c;
  TestCoins coins(coin_seed);
  c.index.reset(new IndexCpu(w.S.table, rows, coins, w.configuration));
  c.key = c.index->key();
  CHECK(!index_key_fault(c.key), "an honest key is refused");
  return c;
}

// The coefficient of a claim at one entry; counts `terms` and `units` only.
inline E link_coefficient(const Row &row, OracleId o, Part part, u64 index) {
  E sum = E::zero();
  for (const Term &t : row.terms)
    if (t.oracle == o && t.part == part) sum += t.v.at(index);
  for (const Unit &u : row.units)
    if (u.oracle == o && u.part == part && u.index == index) sum += u.coeff;
  return sum;
}
// An oracle message's pad masks one sent scalar, its sample's answer. A claim that read it too
// would tie that answer to something else.
inline void link_check_pads(const Statement &st, GroupId g) {
  for (const Row &row : st.group(g).rows) {
    CHECK(row.runs.empty() && row.forms.empty(), "a row of LINK has a piece of a kind that this check does not read");
    for (OracleId o : st.group(g).members)
      for (const PadPlace &p : Send::pad(st.codes(), st.oracle(o).shape)) CHECK(is_zero(link_coefficient(row, o, p.part, p.index)), "a row reads a pad");
  }
}

// Deliberate faults of the tests' prover and verifier. With none, link_check() calls link_verify().
struct LinkScratch {
  u64 cut_short_by = 0;                // the verifier's equality claim ends this many entries early
  bool query_without_further = false;  // both sides' first query: all-zero commitment, empty record
  bool any() const { return cut_short_by || query_without_further; }
};

// made: false if a table's random tails cannot cover the proof's columns.
struct LinkProof {
  bool made = false;
  std::vector<u8> bytes;
  // Observed by LinkAccess::prove() before the proximity test:
  bool long_row_holds = false;  // the equality claim, extended to the message's end, holds
  size_t column_rows[2] = {0, 0}, false_column_rows[2] = {0, 0}, other_false_rows = 0;  // [0] on the first table's folded row, [1] on the further one's
  std::vector<E> pads;  // the pads of the two folded rows' oracle messages, the first table's first
};

// The tests' access to the private steps of `Link`.
struct LinkAccess {
  static LinkPublic bind(Binder &b, const LinkWorld &w, const IndexName &first, const IndexName &further, const LinkScratch &scratch) {
    return link_bind(b, w.configuration, w.codes, w.S, first, scratch.query_without_further ? IndexName{std::vector<u8>(DIGEST_BYTES, 0), IndexRecord()} : further);
  }

  // Link::prove() step by step, without its checks of shape and configuration; examines the claims.
  static LinkProof prove(const LinkWorld &w, LinkTable &first, LinkTable &further, u64 coin_seed, const LinkScratch &scratch) {
    const LinkShape &S = w.S;
    LinkProof out;
    TestCoins coins(coin_seed);
    EngineCpu engine;
    Writer bytes;
    if (!further.index->begin(S.columns_further) || !first.index->begin(S.columns_first)) return out;
    Binder binder(LINK_LABEL, w.plan);
    const LinkPublic x = bind(binder, w, IndexName{first.index->commitment(), first.key.rec}, IndexName{further.index->commitment(), further.key.rec}, scratch);
    u8 random_bytes[DIGEST_BYTES];
    coins.bytes(random_bytes, sizeof random_bytes);
    Digest random;
    digest_from_bytes(random_bytes, random);
    ProverRounds ch(binder, random, bytes);
    ProverContext P(x.codes, ch, engine, coins);
    LinkRun run;
    Link::steps_prove(P, S, *first.index, first.key.rec, *further.index, further.key.rec, run);

    link_check_pads(P.st, run.group);
    for (OracleId o : {run.first.oracle, run.further.oracle}) {
      const Table main = P.engine.get(P.behind[o].main);
      for (const PadPlace &p : Send::pad(P.st.codes(), P.st.oracle(o).shape)) out.pads.push_back(p.part == Part::MAIN ? main.entry(p.index) : P.behind[o].mask.entry(p.index));
    }
    const Row equality = link_equality_row(S, run.phi, run.first, run.further, S.cut());
    // The equality claim must read nothing from the cut on:
    for (const FoldedRow *at : {&run.first, &run.further})  // no pad, random tail or unused entry
      for (u64 j = S.cut(); j < (u64(1) << at->lay.bits()); j++)
        CHECK(is_zero(link_coefficient(equality, at->oracle, Part::MAIN, at->lay.index(j))), "the equality row reads an entry behind the cut");
    out.long_row_holds = is_zero(row_value(P, link_equality_row(S, run.phi, run.first, run.further, S.table.code.ell)));
    for (const Row &row : P.st.group(run.group).rows) {
      const bool holds = row_value(P, row) == row.target;
      if (row.kind != RowKind::GENERATOR) {
        out.other_false_rows += !holds;
        continue;
      }
      const int side = row.terms.at(0).oracle == run.further.oracle;
      out.column_rows[side]++, out.false_column_rows[side] += !holds;
    }

    Link::test_prove(P, S, run);
    ch.finish();
    P.st.require_all_tested();
    out.made = true, out.bytes = bytes.bytes();
    return out;
  }

  // `first` counts as tested: see link_check().
  static bool verify(const LinkWorld &w, const IndexKey &first, const IndexKey &further, Reader &bytes, const char *&reason, const LinkScratch &scratch) {
    const LinkShape &S = w.S;
    Binder binder(LINK_LABEL, w.plan);
    const LinkPublic x = bind(binder, w, IndexName{first.commitment(), first.rec}, IndexName{further.commitment(), further.rec}, scratch);
    VerifierSession session;
    VerifierRounds ch(binder, bytes, session);
    VerifierContext V(x.codes, ch);
    LinkRun run;
    if (!Link::steps_verify(V, S, TableRead::on_a_tests_word(first, w.plan->set), first.rec, further, further.rec, S.cut() - scratch.cut_short_by, run)) return reason = LINK_REFUSED_STEPS, false;
    if (!Link::test_verify(V, S, run)) return reason = LINK_REFUSED_TEST, false;
    if (!ch.finish()) return reason = LINK_REFUSED_END, false;
    V.st.require_all_tested();
    return true;
  }
};

// Without faults and with the same private randomness (`coins` in identifiers), the proof's bytes
// equal link_prove()'s (link_test.cpp compares them).
inline LinkProof link_make(const LinkWorld &w, LinkTable &first, LinkTable &further, u64 coin_seed, const LinkScratch &scratch = LinkScratch()) {
  return LinkAccess::prove(w, first, further, coin_seed, scratch);
}

inline bool link_check(const LinkWorld &w, const IndexKey &first, const IndexKey &further, const std::vector<u8> &proof, const char **why = nullptr,
                       const LinkScratch &scratch = LinkScratch()) {
  const char *unused;
  const char *&reason = why ? *why : unused;
  reason = "";
  Reader bytes(proof.data(), proof.size());
  if (!scratch.any()) {  // why: a key's fault in words, else the step's name
    VerifierSession session;
    LinkWhy said;
    // LINK's error bound needs a first table that has passed its test: otherwise
    // foldeq_columns_verify() (blocks/iop/foldeq.h) ends the process. This fixture runs only LINK
    // and declares `first` tested, under the plan's setting, without testing it.
    const bool yes = link_verify(w.plan, w.configuration, w.codes, w.S, TableRead::on_a_tests_word(first, w.plan->set), further, bytes, session, &said);
    CHECK(std::string(said.step) == LINK_REFUSED_KEYS || yes == session.accepted(), "the link's answer is not its session's");  // a key that does not fit: no chain was begun
    CHECK(yes == !*said.step && yes == !*said.reason, "a refusal without a reason, or a reason without a refusal");
    reason = std::string(said.step) == LINK_REFUSED_KEYS ? said.reason : said.step;
    return yes;
  }
  return LinkAccess::verify(w, first, further, bytes, reason, scratch);
}

}  // namespace blk
