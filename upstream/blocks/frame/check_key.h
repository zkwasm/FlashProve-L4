// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Writes and checks a key file: `make_key` is the committer's side, `check_key` the verifier's. A
   key file asserts two things about the committed table (`Index` in identifiers) under its Merkle
   root: its matrix entries are in their ranges (the range proof WF, blocks/iop/wf.h), and the table
   is within the one-time distance of one codeword (the one-time test: LIN, blocks/iop/lin.h, for a
   batch of one). Both sides take a `Configured`. Neither their caller nor the file supplies plans,
   codes, shapes or the configuration digest: `configure` (blocks/frame/configure.h) made them from
   the public files, some through the calling program's `ShapesOf` function, which must depend on
   those files only; nothing checks that (see `Shapes` there).

   Layout (no length fields): header (`head` in identifiers) | root | record's answers | one-time
   chain | place | shared chain | second chain. Chains: Fiat-Shamir hash chains. One-time chain: the
   index-mask message (mask rows only), sent and sampled | WF | the linear claims ("rows" in
   messages) that WF leaves on the matrix, batched into one if several (BATT), and LIN's first part
   | seal (the chain's digest, not stored). Place: the seal's leaf in a one-leaf binding tree: leaf
   number 0 and its salt. Shared and second chain: LIN's, as in a proof file (blocks/frame/check.h).
   Root and record's answers (to samples of the table's rows, `IndexRecord`): bytes of the file, not
   a chain's messages. They are the first statement item in the first query (the hash input that
   starts a chain) of the one-time and shared chains, so a changed root changes every challenge.

   A failed run is never resumed: a second call draws all its private randomness (`coins`) again,
   and the table columns that `lin_begin` counted stay spent. */
#pragma once
#include <string>
#include <vector>

#include "chain/closed.h"  // Notes
#include "chain/rounds.h"
#include "frame/configure.h"
#include "frame/heads.h"
#include "frame/key_fixed.h"
#include "iop/member.h"
#include "iop/wf.h"

namespace blk {

// A key file's public values, as the one-time chain's first query contains them.
struct KeyPublic {
  LinPublic lin;  // codes, the shape of the test on the table, the table's root and record
  Public<OracleNotice> index_masks;
  Public<WfPublic> ranges;
  Public<LinBindingShape> binding;
};
inline KeyPublic bind_one_time_chain(Binder &b, const KeyFixed &f, const IndexName &table) {
  KeyPublic x;
  b.configuration(f.configuration);
  x.index_masks = b.parameter("index-mask message", f.index_masks);
  x.ranges = b.parameter("ranges", f.ranges);
  x.binding = b.parameter("binding tree", f.binding);
  x.lin = lin_bind(b, f.codes, f.lin, table);  // last: its statement item closes the parameters
  return x;
}

struct KeyPlans {
  Planned one_time, shared, second;
};
// The plans and values of a run. Only key_under() makes one, or a test (BLK_KEY_UNDER_FOR_A_TEST).
class KeyUnder {
 public:
  const KeyPlans plans;
  const KeyFixed fixed;
#ifdef BLK_KEY_UNDER_FOR_A_TEST
  KeyUnder(const KeyPlans &p, const KeyFixed &f) : plans(p), fixed(f) {}
#endif
 private:
  struct FromConfigured {};
  KeyUnder(const KeyPlans &p, KeyFixed f, FromConfigured) : plans(p), fixed(std::move(f)) {}
  friend KeyUnder key_under(const Configured &c);
};
#ifdef BLK_KEY_UNDER_FOR_A_TEST
BLK_MARK_NOT_FOR_SERVING();  // blocks/fence.h does not see a define made after its first include
#endif

/* The committer's run under given plans and values; programs call `make_key`. True: `file` holds a
   key file. False, and `file` is empty: `table` may open no more columns (`lin_begin`), or
   `wf_prove` fails, e.g. on an entry out of its range. `table` owns the committed table; `ranges`
   (WF) and `products` (BATT, LIN) compute on its matrix, `engine` on this run's oracle messages. */
[[nodiscard]] inline bool make_key_under(const KeyUnder &under, IndexHolder &table, WfHeavy &ranges, MemberHeavy &products,
                                         Engine &engine, Coins &coins, const NonceSearch &search, Writer &file) {
  const KeyPlans &plans = under.plans;
  const KeyFixed &fixed = under.fixed;
  const char *fault = key_fault(fixed);
  BLK_REQUIRE(!fault, fault ? fault : "");  // an internal check: `configure` rejects such files
  BLK_REQUIRE(file.bytes().empty(), "make_key: the file comes empty");
  const IndexName name{table.commitment(), table.record()};
  std::unique_ptr<LinBegun> begun;
  if (!lin_begin(table, fixed.lin, begun)) return false;

  Writer &out = file;  // in place, so that a `file` set to `keep_list` records each item's kind
  write_head(out, TAG_KEY);
  BLK_REQUIRE(name.commitment.size() == DIGEST_BYTES, "make_key: what a table's keeper gives as its commitment is a root");
  Digest root;
  digest_from_bytes(name.commitment.data(), root);
  out.put(root);
  for (const E &y : name.record.y) out.put(y);

  Binder b(LABEL_ONE_TIME_CHAIN, plans.one_time);
  const KeyPublic x = bind_one_time_chain(b, fixed, name);
  ProverRounds ch(b, random_digest(coins), out, search);
  ProverContext P(x.lin.codes, ch, engine, coins);
  std::vector<OracleDraft> drafts(1);
  drafts[0].shape = ch.get(x.index_masks).shape, drafts[0].leaf_class = ch.get(x.index_masks).leaf_class;
  drafts[0].mask = Send::fresh_mask(P.st.codes(), drafts[0].shape, coins), drafts[0].mask_draw = coins.last_draw();
  const OracleId masks = send_prove(P, std::move(drafts))[0];
  // Self-check after sending: a key file stays public; its masks cover sums linear in the weights.
  BLK_REQUIRE_NAMED(not_as_drawn(P, masks) == 0, "own-test-index-mask", "internal error (own-test-index-mask): values of the key file's index-mask message are not the random values that were drawn for it. The program or the machine has a fault; nothing of this run may be used");
  WfOutput left;
  if (!wf_prove(P, x.ranges, x.lin, masks, ranges, left)) return file.clear(), false;
  LinFirst first = member_first_prove(P, x.lin, masks, left.pending.take(), products);
  Sealed seal;
  BLK_REQUIRE(ch.seal(R_SEAL, seal), "a prover's channel does not refuse");
  ch.finish_inside_a_file();  // every chain's end is run, earlier chains first

  LinBinding binding(fixed.binding, ch.configuration(), coins);
  binding.put(0, seal.digest());
  const LinBindingRoot binding_root = binding.root();
  binding.write_path(out, 0);

  Binder shared_b(LIN_LABEL_SHARED, plans.shared);
  shared_b.configuration(fixed.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, fixed.codes, fixed.lin, name, binding_root);
  ProverRounds shared_ch(shared_b, Digest{}, out, search);
  ProverContext shared_P(shared_x.codes, shared_ch, engine, coins);
  LinShared shared;
  lin_shared_prove(shared_P, shared_x, TableTest::ONE_TIME, table, std::move(begun), shared);
  lin_shared_tests_prove(shared_P, shared_x, shared);
  Sealed shared_seal;
  BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
  shared_ch.finish_inside_a_file();
  shared_P.st.require_all_tested();

  Binder second_b(LIN_LABEL_SECOND, plans.second);
  second_b.configuration(fixed.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  ProverRounds second_ch(second_b, random_digest(coins), out, search);
  ProverContext P2 = lin_go_on(std::move(P), second_ch);
  const LinSecond second = lin_second_prove(P2, x.lin, first, shared_P, shared);
  lin_second_tests_prove(P2, x.lin, first, second);
  second_ch.finish();
  P2.st.require_all_tested();
  lin_release(shared_P, shared);
  return true;
}

/* Reads header | root | record's answers into `table`, without the file's proofs, and leaves `file`
   after these bytes. Only the form is checked: tag, version word, canonical elements,
   `index_key_fault`. True says nothing about the table. A caller other than `check_key_under`
   relies on a separate `check_key` on the same bytes, and must accept nothing unless it passes. */
[[nodiscard]] inline bool read_key_table_unproved(const KeyFixed &fixed, Reader &file, IndexKey &table, const char *&why) {
  if ((why = read_head(file, TAG_KEY))) return false;
  const IndexShape &shape = fixed.lin.fold.index;
  IndexKey read(shape, fixed.configuration, Digest{}, IndexRecord());
  if (!file.get(read.root)) return why = "the table's root", false;
  read.rec.y.resize(shape.rows() * shape.answers);
  for (E &y : read.rec.y)
    if (!file.get(y)) return why = "the record's answers", false;
  if ((why = index_key_fault(read))) return false;
  table = read;
  return true;
}
/* The verifier's run under given plans and values; programs call `check_key`. True: `key` is what a
   verifier keeps of the table for later proofs, with the `Setting` under which this test passed and
   the key's public values (`TestedTable`, blocks/iop/foldeq.h). For logs only: `why`, the first
   reason for a rejection, and `place` (not the file's part of that name), its chain and step (null
   after true): `why` can read alike in every chain. `notes`: see `check_key`. */
[[nodiscard]] inline bool check_key_under(const KeyUnder &under, Reader &file, TestedTable &key, const char *&why, Notes *notes = nullptr, const char **place = nullptr) {
  const KeyPlans &plans = under.plans;
  const KeyFixed &fixed = under.fixed;
  if (place) *place = "before the chains";
  auto no = [&](const VerifierRounds &ch, const char *where) {
    if (place) *place = where;
    return why = ch.why() ? ch.why() : where, false;
  };
  if ((why = key_fault(fixed))) return false;
  for (const Planned *p : {&plans.one_time, &plans.shared, &plans.second})
    if (!(*p)->holds()) return why = does_not_hold(*p, "check_key"), false;  // no key is worth anything under such parameters
  BLK_REQUIRE(Plan::same_setting(plans.one_time->set, plans.shared->set) && Plan::same_setting(plans.shared->set, plans.second->set), "check_key: a key's three chains are planned under one setting");
  IndexKey read;
  if (!read_key_table_unproved(fixed, file, read, why)) return false;
  const IndexName name{read.commitment(), read.rec};

  Binder first_b(LABEL_ONE_TIME_CHAIN, plans.one_time);
  const KeyPublic x = bind_one_time_chain(first_b, fixed, name);
  VerifierSession session;  // three chains, one verdict; one record of vertex values per root
  VerifierRounds first_ch(first_b, file, session);
  VerifierContext first_V(x.lin.codes, first_ch);
  first_V.notes = notes;
  std::vector<OracleId> ids;
  if (!send_verify(first_V, std::vector<OracleNotice>{first_ch.get(x.index_masks)}, ids)) return no(first_ch, "one-time chain: the index-mask message");
  WfOutput left;
  if (!wf_verify(first_V, x.ranges, x.lin, ids[0], left)) return no(first_ch, "one-time chain: the ranges");
  LinFirst first;
  if (!member_first_verify(first_V, x.lin, ids[0], left.pending.take(), first)) return no(first_ch, "one-time chain: the rows on the table");
  Sealed seal;
  if (!first_ch.seal(R_SEAL, seal) || !first_ch.finish_inside_a_file()) return no(first_ch, "one-time chain: its end");

  LinBindingRoot binding_root;
  if (place) *place = "between the one-time chain and the shared chain";
  if (!lin_binding_root(file, first_ch.get(x.binding), first_ch.configuration(), seal.digest(), binding_root)) return why = "the seal's place", false;

  Binder shared_b(LIN_LABEL_SHARED, plans.shared);
  shared_b.configuration(fixed.configuration);
  const LinPublic shared_x = lin_bind_shared(shared_b, fixed.codes, fixed.lin, name, binding_root);
  VerifierRounds shared_ch(shared_b, file, session);
  VerifierContext shared_V(shared_x.codes, shared_ch);
  shared_V.notes = notes;
  LinShared shared;
  if (!lin_shared_verify(shared_V, shared_x, TableTest::ONE_TIME, read, shared)) return no(shared_ch, "shared chain");
  if (!lin_shared_tests_verify(shared_V, shared_x, shared)) return no(shared_ch, "shared chain: a proximity test");
  Sealed shared_seal;
  if (!shared_ch.seal(R_SEAL, shared_seal) || !shared_ch.finish_inside_a_file()) return no(shared_ch, "shared chain: its end");
  shared_V.st.require_all_tested();

  Binder second_b(LIN_LABEL_SECOND, plans.second);
  second_b.configuration(fixed.configuration);
  lin_bind_second(second_b, seal, shared_seal);
  VerifierRounds second_ch(second_b, file, session);
  VerifierContext V = lin_go_on(std::move(first_V), second_ch);
  V.notes = notes;
  LinSecond second;
  if (!lin_second_verify(V, x.lin, first, shared_V, shared, second)) return no(second_ch, "second chain");
  if (!lin_second_tests_verify(V, x.lin, first, second)) return no(second_ch, "second chain: a proximity test");
  if (!second_ch.finish()) return no(second_ch, "second chain: its end");  // also: no byte is left over
  V.st.require_all_tested();
  if (place) *place = "behind the chains";
  if (!session.accepted()) return why = "a chain of the key has refused", false;  // the verdict: false if a chain rejected, even if a result above was dropped; else ends the process if one is unfinished
  if (place) *place = nullptr;
  key = TestedTable(std::move(read), plans.shared->set, key_public_bytes(fixed));
  why = nullptr;
  return true;
}

// ---- what a program calls: all public values come from one `Configured`, made by configure() ----

inline KeyFixed key_fixed(const Configured &c) { return key_fixed(c.sized(), c.configuration()); }
inline KeyUnder key_under(const Configured &c) {  // plans and values all come from c, as copies
  BLK_REQUIRE(c.refused().empty() && c.key().chains.size() == 3, "a key has three chains");
  return KeyUnder(KeyPlans{c.key().chains[0].plan, c.key().chains[1].plan, c.key().chains[2].plan}, key_fixed(c), KeyUnder::FromConfigured{});
}
// See make_key_under(). `IndexHolder`, `WfHeavy`, `MemberHeavy` and `Engine` do the committer's
// heavy work; each has a CPU implementation for toy sizes.
// The self-check in make_key_under() compares the index-mask message's mask table and random tail,
// as the prover's context keeps them, with what was drawn: not the pads, which send_prove() checks,
// and not the copies given to the engine for encoding. It compares nothing if `coins` numbers no
// draws (NO_DRAW): a program that relies on it must require coins that do.
// See not_as_drawn() (blocks/iop/send.h) for what the self-check does not see.
[[nodiscard]] inline bool make_key(const Configured &c, IndexHolder &table, WfHeavy &ranges, MemberHeavy &products, Engine &engine, Coins &coins,
                                   const NonceSearch &search, Writer &file) {
  BLK_REQUIRE(c.refused().empty(), "make_key: the public files are not admitted");
  return make_key_under(key_under(c), table, ranges, products, engine, coins, search, file);
}
// read_key_table_unproved() under a `Configured`: no proof is checked (see the first overload). The
// result's type, `UnprovedTable` (blocks/iop/foldeq.h), limits what it can be used for.
[[nodiscard]] inline bool read_key_table_unproved(const Configured &c, Reader &file, UnprovedTable &table, const char *&why) {
  if (!c.refused().empty()) return why = "the public files are not admitted", false;
  IndexKey read;
  if (!read_key_table_unproved(key_fixed(c), file, read, why)) return false;
  table = UnprovedTable(std::move(read), c.sized().d.set, key_public_bytes(key_fixed(c)));
  return true;
}
/* A yes means: both assertions at the top of this file hold, under these public files, up to the
   error that the key's plans count at the files' target and given the protocol's analysis, which no
   program here proves. YES_WITH_WORK: only with the proofs of work counted; the caller decides from
   `c.work_of_the_key`. The verdict does not convert to bool implicitly (enum class `ProofVerdict`).
   A yes says nothing about whose weights they are: no item of a chain names a model, as no step
   could check it; a publisher's word on that is outside the protocol.
   Some of the key's public values, such as the ranges and the matrix's number of rows, come from
   the files through the calling program's `ShapesOf` function (see the top of this file), and the
   configuration digest covers only the files (see `key_public_bytes`, blocks/frame/key_fixed.h).
   NO leaves `key` unchanged. `notes`: values the blocks note, for a second verifier; no check
   reads them. */
[[nodiscard]] inline ProofVerdict check_key(const Configured &c, Reader &file, TestedTable &key, const char *&why, Notes *notes = nullptr, const char **place = nullptr) {
  if (place) *place = "before the chains";
  if (!c.refused().empty()) return why = "the public files are not admitted", ProofVerdict::NO;
  if (!check_key_under(key_under(c), file, key, why, notes, place)) return ProofVerdict::NO;
  return c.work_of_the_key().needed ? ProofVerdict::YES_WITH_WORK : ProofVerdict::YES;
}

}  // namespace blk
