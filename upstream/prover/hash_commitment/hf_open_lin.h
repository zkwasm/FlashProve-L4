// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the shared chain of LIN (blocks/iop/lin.h), the block for proving one linear claim per
// request on the weight table T (`Index` in identifiers) with one fold of T, a random combination
// of its rows, per batch. The shared chain is the hash chain (a Fiat-Shamir transcript) that runs
// once per batch and is repeated in each of its proofs.
// Plain C++: T, engine, private randomness (`coins` in identifiers) and nonce search are arguments.
// Order per batch, as in prover/prover/batch_opening.h:
//   blk::lin_begin()    before any first chain starts (unchecked); counts the batch's columns of T;
//                       false if T's masks (random entries of each row) no longer cover them
//   first chains        one per request (blocks/iop/member.h); their seals, the digests that end
//                       them, go into the leaves of the binding tree (`binding_root`)
//   hf_open_lin()       the shared chain (HfLinOpened::bytes); `keeper` folds T, then opens columns
//   second chains       one per request; they read HfLinOpened::context, `shared` and `seal`
//   hf_open_lin_end()   after the last second chain; releases Omega, the oracle message they read
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "frame/heads.h" // the labels of a proof's chains
#include "frame/lin_frame.h"

#include "hash_commitment/hf_keeper.h"

namespace hf {

// The same for every batch: shapes, codes, numbers of opened columns and positions, the plan (the
// chain's list of rounds). The committer's one-time test (LinLines::one_time) needs a separate
// HfLinSetup. LINK proofs, which tie two weight tables, open columns of T too.
struct HfLinSetup {
  std::string refused;  // empty, or why these lines give no opening
  blk::Setting set;
  blk::Codes codes;
  blk::LinSized sized;
  blk::Planned plan;                // the shared chain's
  HfShown shown;                    // layout of T's opened columns as HfKeeper receives them
  uint64_t batches() const { return sized.batches; }  // batches a table allows, before LINK proofs
};

// lines: what blk::lin_lines() derives from the parameter file; table_size: as blk::Derived::table.
// Pass a fresh `out`: LIN's codes are appended to out.codes, and out.refused is not cleared.
// For tests: the prover program `cp_prove` fills HfLinSetup in Prover::set_up_the_shared_chain().
inline void hf_lin_setup(const HfTableParams &P, const blk::Setting &set, const blk::LinLines &lines, const blk::TableSize &table_size, HfLinSetup &out) {
  out.set = set;
  out.sized = blk::lin_sized(out.set, lines, table_size, out.codes);
  if (!out.sized.refused.empty()) return void(out.refused = out.sized.refused);
  if (const char *why = hf_lin_misfit(P, out.sized.S)) return void(out.refused = why);
  out.shown = hf_shown(P, out.sized.S.fold.columns);
  blk::Plan plan(out.set);
  plan.begin_scope(lines.one_time ? blk::Scope::ONCE : blk::Scope::SHARED);
  blk::TreeSize table;  // T's Merkle tree, from which the plan computes the opening's size
  table.name = "T", table.log_n = (unsigned)P.ln, table.leaf_bytes = 8 * P.R;
  blk::lin_plan_shared(plan, out.codes, out.sized, table);
  blk::plan_seal(plan, "shared chain");
  out.plan = plan.finish();
  if (!out.plan->holds()) out.refused = "the shared chain's plan does not hold: " + out.plan->why_not();
}

// Adds the items of the shared chain's first query (its first hash input): the configuration
// digest (a hash of the public files), then LIN's. Returns LIN's bound public inputs.
// `cp_prove` uses blk::bind_shared_chain() (blocks/frame/check.h).
typedef std::function<blk::LinPublic(blk::Binder &, const blk::LinBindingRoot &binding_root)> HfLinBind;
// For tests: binds `made_up` as the configuration digest, then LIN's items. `made_up` must equal
// the table's configuration(), or the process ends; zero means blk::toy_configuration().
// `s` must outlive the result.
inline HfLinBind hf_lin_bind_alone(const HfLinSetup &s, blk::IndexName name, blk::Digest made_up = blk::Digest{}) {
  if (made_up == blk::Digest{}) made_up.w[0] = 0x00796f74;
  return [&s, name = std::move(name), made_up](blk::Binder &b, const blk::LinBindingRoot &binding_root) {
    b.configuration(made_up);
    return blk::lin_bind_shared(b, s.codes, s.sized.S, name, binding_root);
  };
}

// hf_open_lin()'s result. It must not move, and `engine` and `coins` must outlive it: `ch` and
// `context` keep references.
struct HfLinOpened {
  blk::Writer bytes;
  blk::LinPublic bound;
  std::unique_ptr<blk::ProverRounds> ch;
  std::unique_ptr<blk::ProverContext> context;
  blk::LinShared shared;
  blk::Sealed seal;
};

// test (blocks/numbers/plan.h): ONE_TIME, the committer's one-time test (only tests pass it);
// LATER, a batch's. The process ends if s.plan was made with the other LinLines::one_time or if
// `begun` (from blk::lin_begin()) is null. `bind` must make the binding tree's root the chain's
// salt, as blk::lin_bind_shared() does: the `blk::Digest{}` below is then not read.
inline void hf_open_lin(const HfLinSetup &s, blk::TableTest test, blk::IndexHolder &keeper, std::unique_ptr<blk::LinBegun> begun, const blk::LinBindingRoot &binding_root, const HfLinBind &bind,
                        blk::Engine &engine, blk::Coins &coins, blk::NonceSearch search, HfLinOpened &out) {
  BLK_REQUIRE(s.refused.empty(), "an opening under lines that were refused");
  blk::Binder b(blk::LABEL_PROOF_SHARED_CHAIN, s.plan);
  out.bound = bind(b, binding_root);
  out.ch.reset(new blk::ProverRounds(b, blk::Digest{}, out.bytes, std::move(search)));
  out.context.reset(new blk::ProverContext(out.bound.codes, *out.ch, engine, coins));
  blk::lin_shared_prove(*out.context, out.bound, test, keeper, std::move(begun), out.shared);
  blk::lin_shared_tests_prove(*out.context, out.bound, out.shared);
  BLK_REQUIRE(out.ch->seal(blk::R_SEAL, out.seal), "a prover's channel does not refuse");
  out.ch->finish();
}
inline void hf_open_lin_end(HfLinOpened &o) { blk::lin_release(*o.context, o.shared); }

// The verifier's side of hf_open_lin(), for tests: the verifier program `cp_verify` calls
// blk::check_proof_if_tested().
struct HfLinChecked {
  blk::LinPublic bound;
  std::unique_ptr<blk::Reader> file;
  std::unique_ptr<blk::VerifierSession> session;  // this chain's; a whole proof's chains share one
  std::unique_ptr<blk::VerifierRounds> ch;
  std::unique_ptr<blk::VerifierContext> context;
  blk::LinShared shared;
  blk::Sealed seal;
  const char *why = "";  // after `false`: the step that failed
};
// `bytes`: the shared chain and nothing after it; they must outlive `out`.
// `key`: ONE_TIME accepts an untested blk::IndexReader. A LATER test's error bounds assume a table
// that has passed its one-time test, so LATER needs a blk::TestedTable (blocks/iop/foldeq.h): with
// an untested reader the process ends where the columns are read. A test can instead vouch for the
// table with blk::TableRead::on_a_tests_word(), which checks nothing.
[[nodiscard]] inline bool hf_check_lin(const HfLinSetup &s, blk::TableTest test, const blk::TableRead &key, const blk::LinBindingRoot &binding_root, const HfLinBind &bind, const uint8_t *bytes,
                                       size_t count, HfLinChecked &out) {
  BLK_REQUIRE(s.refused.empty(), "a check under lines that were refused");
  blk::Binder b(blk::LABEL_PROOF_SHARED_CHAIN, s.plan);
  out.bound = bind(b, binding_root);
  out.file.reset(new blk::Reader(bytes, count));
  out.context.reset(), out.ch.reset();  // first: `~VerifierRounds` uses the old session
  out.session.reset(new blk::VerifierSession());
  out.ch.reset(new blk::VerifierRounds(b, *out.file, *out.session));
  out.context.reset(new blk::VerifierContext(out.bound.codes, *out.ch));
  if (!blk::lin_shared_verify(*out.context, out.bound, test, key, out.shared)) return out.why = "before the tests", false;
  if (!blk::lin_shared_tests_verify(*out.context, out.bound, out.shared)) return out.why = "a proximity test", false;
  if (!out.ch->seal(blk::R_SEAL, out.seal) || !out.ch->finish()) return out.why = "the end", false;
  return true;
}

}  // namespace hf
