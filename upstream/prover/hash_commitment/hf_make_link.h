// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Makes and checks a LINK proof (blocks/iop/link.h) for the weight table T (`Index` in
// identifiers; the weights in rows encoded with masks, under a Merkle tree): one hash chain (a
// Fiat-Shamir transcript). The proof ties a leaf table (`further`: the same weights under its own
// masks) to the anchor (`first`), the commitment's first table. LINK compares a row's message
// before the cut, the first index where two tables of one commitment may differ; here that is where
// a row's pads begin (HfTableParams::pad_entry(0)). So it compares the data cells and the unused
// entries after them, but no pads and no
// masks: each table has its own pads, as it has its own masks.
//   hf_link_setup()  once, into a new HfLinkSetup: shapes, codes, column counts, the chain's plan
//   hf_make_link()   writes the proof, in the format of a leaf file after header, root and record
//   hf_check_link()  for tests: checks it; the verifier program `cp_verify` calls
//                    blk::check_entry_if_tested()
// Only prover/hash_commitment/tests/hf_link_side_test.cu uses this file, at toy and small
// shapes on random cells: the committer program `cp_commit` makes its leaf files
// (`ivk/leaf/<k>.bin`) by hf::entry_make(), which calls blk::make_entry(). Plain C++, no CUDA.
#pragma once
#include <memory>
#include <string>

#include "frame/link_frame.h"

#include "hash_commitment/hf_keeper.h"

namespace hf {

struct HfLinkSetup {
  std::string refused;  // empty, or why these lines give no LINK proof
  blk::Digest configuration{};  // digest of the public files: an item of the chain's first query
  blk::Setting set;
  blk::Codes codes;
  blk::LinkSized sized;
  blk::Planned plan;  // the LINK proof's chain
  HfShown shown_first, shown_further;  // layout of each table's opened columns, for its `HfKeeper`
};

// lines: blk::link_lines() of the parameter file (`policy.txt`), or values that a test sets.
// distances: the tables', from blk::derive() or blk::link_table(). If out.refused is then not
// empty, the other two functions end the process.
inline void hf_link_setup(const HfTableParams &P, const blk::Digest &configuration, const blk::Setting &set, const blk::LinkLines &lines,
                          const blk::TableSize &distances, HfLinkSetup &out) {
  out.configuration = configuration, out.set = set;
  out.sized = blk::link_sized(out.set, lines, distances, out.codes);
  if (!out.sized.refused.empty()) return void(out.refused = out.sized.refused);
  const blk::LinkShape &S = out.sized.S;
  const blk::IndexShape shape = hf_index_shape(P);
  if (!blk::link_same_table_shape(S.table, shape) || S.cut() != P.pad_entry(0))
    return void(out.refused = "the lines describe a link between other tables than T's copies");
  out.shown_first = hf_shown(P, S.columns_first), out.shown_further = hf_shown(P, S.columns_further);
  blk::Plan plan(out.set);
  plan.begin_scope(blk::Scope::ONCE);
  blk::TreeSize table;  // sized as an opening of a stored tree in blocks/, which `HfKeeper` writes
  table.name = "T", table.log_n = (unsigned)P.ln, table.leaf_bytes = 8 * P.R;
  blk::plan_chain_start(plan, "link's chain", true);
  blk::link_plan(plan, out.codes, out.sized, table);
  out.plan = plan.finish();
  if (!out.plan->holds()) out.refused = "the plan of the link's chain does not hold: " + out.plan->why_not();
}

// first, further: an `HfKeeper` (hf_keeper.h) built on s.shown_first and one built on
// s.shown_further; another column count ends the process. search, for proof-of-work nonces, may be
// null. Returns false, with nothing written, if either table's masks do not cover another LINK
// proof's columns; further.begin() is called first, and its columns stay counted if first.begin()
// returns false.
// Takes at least four passes over the weights: each `HfKeeper`
// folds its table in one, at the fold point, and opens its
// columns in one or more, after the positions are drawn.
[[nodiscard]] inline bool hf_make_link(const HfLinkSetup &s, blk::IndexHolder &first, blk::IndexHolder &further, blk::Engine &engine, blk::Coins &coins,
                                       blk::NonceSearch search, blk::Writer &bytes) {
  BLK_REQUIRE(s.refused.empty(), "a link under lines that were refused");
  return blk::link_prove(s.plan, s.configuration, s.codes, s.sized.S, first, further, engine, coins, bytes, std::move(search));
}

// Checks one proof in its own blk::VerifierSession (the chains of one verdict). first, further:
// readers built on s.sized.S.table; reading columns under other IndexShape::distances ends the
// process. first: also a blk::TestedTable (blocks/iop/foldeq.h) tested under s.plan->set, or the
// process ends where its columns are read. The anchor's one-time test (in the key file) is not run
// here: prover/hash_commitment/tests/hf_link_side_test.cu instead vouches for it, by
// blk::TableRead::on_a_tests_word().
// why, words (each may be null): after a rejection, the step (blk::LINK_REFUSED_..) and a reason.
[[nodiscard]] inline bool hf_check_link(const HfLinkSetup &s, const blk::TableRead &first, const blk::IndexReader &further, const uint8_t *bytes, size_t count,
                                        const char **why = nullptr, const char **words = nullptr) {
  BLK_REQUIRE(s.refused.empty(), "a check under lines that were refused");
  blk::Reader file(bytes, count);
  blk::VerifierSession session;
  blk::LinkWhy said;
  const bool yes = blk::link_verify(s.plan, s.configuration, s.codes, s.sized.S, first, further, file, session, &said);
  if (why) *why = said.step;
  if (words) *words = said.reason;
  return yes;
}

}  // namespace hf
