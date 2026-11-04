// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Turns a configuration's public files (the parameter file and any others, such as a layout of
   arrays or a model's public constants) into what committer, prover and verifier run on.
   `configure` is the only function that makes a `Configured`. From the same bytes it computes the
   configuration digest and reads the parameters, and it passes these bytes to the caller's
   function (`ShapesOf`), which returns the shapes; sizes and plans follow from the parameters and
   the shapes. What is hashed and what is run are meant to have one source; they have it only if
   that function keeps to these bytes, which nothing checks. A hash chain is one protocol's
   Fiat-Shamir transcript; a plan is a chain's list of rounds; the digest is in every chain's first
   query (hashed before the prover speaks).

   What a caller can get wrong: the function that it passes to `configure` must depend on the
   public files only, and nothing checks that (see `Shapes`). */
#pragma once
#include <string>
#include <vector>

#include "frame/files.h"
#include "frame/heads.h"
#include "frame/hiding_conditions.h"
#include "frame/key_fixed.h"
#include "iop/req_rows.h"
#include "numbers/accounting.h"

namespace blk {

// The eight labels differ: those of the seven kinds of chain in key, leaf and proof files, then
// LABEL_RECORD (blocks/chain/chain.h). They are listed in rising order.
static_assert(LABEL_FIRST_CHAIN < LABEL_ONE_TIME_CHAIN && LABEL_ONE_TIME_CHAIN < LABEL_PROOF_SHARED_CHAIN &&
                  LABEL_PROOF_SHARED_CHAIN < LABEL_PROOF_SECOND_CHAIN && LABEL_PROOF_SECOND_CHAIN < LIN_LABEL_SHARED &&
                  LIN_LABEL_SHARED < LIN_LABEL_SECOND && LIN_LABEL_SECOND < LINK_LABEL && LINK_LABEL < LABEL_RECORD,
              "two kinds of hash chain have the same label, or this list is no longer in rising order");

/* The verdict of `check_proof`, `check_key` or `check_entry` (check.h, check_key.h, check_entry.h)
   on one file; each takes a `Configured`. A proof file holds one request's proof. The key file is
   about the anchor (`first` or `first_table` in identifiers), a commitment's first committed table
   (`Index` in identifiers). A leaf file (`entry` in identifiers) is about a leaf table, a further
   table that the file's LINK proof ties to the anchor.
   YES_WITH_WORK: some round's error meets the target (the parameter file's line `sec`) only because
   its proof of work is counted. The author of the public files chose the target and the bits of
   proof of work, and is not the party that relies on the file. That party reads `bits_without` in
   the file's `Configured::Work` and decides whether so many bits are enough. Each file has its own
   verdict; a caller that wants one on key, entry and proof together combines the three. No verdict
   converts to bool implicitly (enum class), so that YES_WITH_WORK cannot pass for a yes in a bare
   `if`. */
enum class ProofVerdict { NO, YES, YES_WITH_WORK };

// What is fixed before a request is known. `lin`: the shape of LIN, a batch's test on the committed
// table; `binding`: the shape of the Merkle tree over the seals (final digests) of a batch's first
// chains. A proof file has three chains: the first (the request's own, with its segments), the
// shared chain (one for the whole batch) and the second (the request's own again).
struct RequestFixed {
  Digest configuration{};
  Codes codes;
  LinShape lin;
  ReqShape shape;
  ReqLayout layout;
  std::vector<u64> identifiers;  // one per segment (`link`): a sub-proof's part of the first chain
  u64 pending_rows = 0;          // linear claims (`rows`) on the committed table left by segments
  LinBindingShape binding;
};

/* What the caller's function (`ShapesOf`) reads from the public files. `read_parameters`
   (blocks/frame/parameters.h) stops after the lines that it knows. The rest of the parameter file
   (`Parameters::rest`) and all further files enter the configuration digest whole, but only this
   function reads them: it must use every line or set `refused`. It may accept several spellings of
   a file: `layout_read` (blocks/frame/req_frame.h) skips blank lines and lines that begin with '#'
   and accepts signs and leading zeros. Two spellings of one layout are two configurations, with
   two commitment IDs (`published_id` in identifiers).

   Not checked: this function and the functions in `sub` (`plan`, `after_links`) must depend on the
   public files only. A function pointer carries no state but can read a global, and those of the
   programs `cp_verify` and `cp_prove` do: `stageShapesOf` (verifier/src/cp_verify.cpp) and
   `chain_shapes_of` (prover/prover/chain_frame.cuh) read a file-scope pointer that is set around
   the call. What the pointer (`stageShapesMade`, `chain_shapes_from`) leads to is built by the
   program before the call, so there the contract rests on that code, and nothing here checks it.
   The functions in `sub` are std::function objects, which can also capture state. Of each round
   that they add to a plan, the name, sizes, draws (what the verifier draws in it) and bits of
   proof of work are hashed (the plan is among a first query's parameters); the round's error is
   not. */
struct Shapes {
  Subproofs sub;
  ReqLayout layout;
  std::vector<u64> identifiers;
  std::vector<RangeCheck> range_checks;  // of one request; they follow from the sub-proofs' shapes
  std::string refused;  // not empty: why `configure` is to reject the files
};
using ShapesOf = Shapes (*)(const std::vector<std::string> &files, const Parameters &par);

/* A plain value: it keeps no function or pointer of the caller of `configure`, so it may be copied
   and kept. If `refused` is not empty, which other members are filled depends on where `configure`
   stopped. */
class Configured {
 public:
  const std::string &refused() const { return refused_; }  // not empty: rejected; check first
  const Parameters &parameters() const { return sized_.par; }
  const Sized &sized() const { return sized_; }
  // chains[i].plan: the plan of chain i. Proof file and key file: three chains; leaf file: one.
  const FilePlan &proof() const { return proof_; }
  const FilePlan &key() const { return key_; }
  const FilePlan &entry() const { return entry_; }
  const Digest &configuration() const { return fixed_.configuration; }
  /* Column counts of one committed table, the anchor or a leaf table (`further` in identifiers),
     for the code that keeps the table's running count of opened columns.
     `bears`: how many columns of one table may be opened over its life: as many positions as a
     row's random tail (the random entries encoded with it) hides. The pads that mask a table's
     record (`IndexRecord`: stored sample answers) are not part of the tail and use up none of it.
     `one_time` to `a_batch`: the columns that one opening opens, counted when it is begun: the key
     file's one-time test (on the anchor), a LINK proof (on each of its two tables), a batch (on
     its leaf table).
     `batches`: how many batches a leaf table allows after its own LINK proof. `configure` rejects
     files unless max(one_time, link_on_the_further) + batches * a_batch <= bears: one bound for
     the anchor and for a leaf table (a leaf table's own LINK proof takes the place of the one-time
     test). It leaves out the LINK proofs on the anchor (`link_on_the_first`). */
  struct TableColumns {
    u64 bears = 0, one_time = 0, link_on_the_first = 0, link_on_the_further = 0, a_batch = 0, batches = 0;
  };
  TableColumns table_columns() const {
    TableColumns t;
    t.bears = hiding_positions(sized_.batch.S.fold.index.code);
    t.one_time = sized_.once.S.fold.columns, t.link_on_the_first = sized_.link.S.columns_first, t.link_on_the_further = sized_.link.S.columns_further;
    t.a_batch = sized_.batch.S.fold.columns, t.batches = sized_.batch.batches;
    return t;
  }
  const RequestFixed &fixed() const { return fixed_; }
  /* One request's approximate range checks (`RangeCheck`, blocks/frame/hiding_conditions.h: random
     combinations of a row's cells must lie in a range) as `Shapes::range_checks` gave them, once
     `hiding_conditions_missed` has accepted them (empty if `configure` stopped earlier). No chain's
     first query contains them, so no proof is rejected because prover and verifier gave different
     lists; `cp_prove` compares its list with the verifier's (`chain_make` in
     prover/prover/chain_frame.cuh). */
  const std::vector<RangeCheck> &range_checks() const { return range_checks_; }
  // For one file's chains. needed: some round meets the target only with its proof of work.
  // bits_without: -log2 of the largest round error with the work left out; bits: that round's bits
  // of proof of work. needs_work(), bits_without_work() and work() return the proof file's
  // fields only, not the key file's or a leaf file's.
  struct Work {
    bool needed = false;
    double bits_without = 0;
    unsigned bits = 0;
  };
  const Work &work_of_the_key() const { return work_[1]; }
  const Work &work_of_an_entry() const { return work_[2]; }
  bool needs_work() const { return work_[0].needed; }
  double bits_without_work() const { return work_[0].bits_without; }
  unsigned work() const { return work_[0].bits; }

 private:
  Configured() = default;
  friend Configured configure(const std::vector<std::string> &files, ShapesOf shapes);
  std::string refused_;
  Sized sized_;
  FilePlan proof_, key_, entry_;
  RequestFixed fixed_;
  std::vector<RangeCheck> range_checks_;
  Work work_[3];
};

// files: the public files' bytes, the parameter file first. A rejection's reason, refused(), is a
// text for a log; every number in it follows from the files if the caller's function keeps to them
// (`Shapes`: not checked).
inline Configured configure(const std::vector<std::string> &files, ShapesOf shapes) {
  Configured c;
  Parameters par;
  if (files.empty()) return c.refused_ = "no parameter file", c;
  if (!read_parameters(files[0], par, c.refused_)) return c;
  Shapes s = shapes(files, par);
  if (!(c.refused_ = s.refused).empty()) return c;
  const Subproofs &sub = s.sub;  // its functions are called here and stored nowhere
  c.sized_ = sized(par, sub);
  if (!(c.refused_ = c.sized_.refused).empty()) return c;
  c.proof_ = file_proof(c.sized_, sub, false), c.key_ = file_key(c.sized_, false), c.entry_ = file_entry(c.sized_, false);
  int file = 0;
  for (const FilePlan *f : {&c.proof_, &c.key_, &c.entry_}) {
    for (const ChainPlan &chain : f->chains) {
      const std::string where = f->name + ", " + chain.name;
      if (!chain.plan->holds()) return c.refused_ = where + ": " + chain.plan->why_not(), c;  // no file is worth anything under such parameters
      if (!chain.plan->names_of_the_format()) return c.refused_ = where + ": a round has a name that is none of the format's", c;
    }
    const Planned whole = merged(c.sized_.d.set, {*f});
    if (whole->refused()) return c.refused_ = f->name + ": " + whole->why_not(), c;  // a chain's errors were computed under a `Setting` other than the file's
    const Accounting a = account(whole);
    c.work_[file++] = Configured::Work{a.needs_work, a.lists.bits, a.lists.work};
  }
  /* Committer, prover and verifier alike reject files that miss a condition for hiding, before a
     table is made. A verifier cannot see a missed condition in a proof; but here anyone who relies
     on a key file learns that its committer's files are outside the conditions under which hiding
     is argued. How often the one-time test and the LINK proofs were begun on a table is not in the
     files (this call takes the default `TableUse`): the code that stores the table counts that. */
  {
    const std::string missed = hiding_conditions_missed(c.sized_, sub, s.range_checks);
    if (!missed.empty()) return c.refused_ = "hiding: " + missed, c;
    c.range_checks_ = s.range_checks;
  }
  RequestFixed &f = c.fixed_;
  f.configuration = configuration_digest(files);
  if (const char *fault = key_fault(key_fixed(c.sized_, f.configuration))) return c.refused_ = std::string("key: ") + fault, c;  // rejected here, before a table is committed, with the reason from `key_fault`
  f.codes = c.sized_.proof_codes, f.lin = c.sized_.batch.S, f.shape = c.sized_.request;
  f.layout = std::move(s.layout), f.identifiers = std::move(s.identifiers), f.pending_rows = sub.pending_rows;
  f.binding.log_leaves = ceil_log2(par.slots), f.binding.salt_bytes = c.sized_.d.set.salt_bytes;
  // Conditions on several of these values together.
  if (f.identifiers.size() != f.shape.link.size()) return c.refused_ = "the links' identifiers are not one for each link", c;
  if (f.shape.at_start.size() != 1) return c.refused_ = "a request's first message holds one message of masks for its rows on the table", c;
  if (!f.pending_rows) return c.refused_ = "a request leaves no row on the table", c;
  for (size_t l = 0; l < f.shape.link.size(); ++l)
    if (f.shape.link[l].plain != (l + 1 == f.shape.link.size() ? 1u : 0u)) return c.refused_ = "the reply is one plain scalar of the last link, and no other link has one", c;
  if (const char *fault = req_layout_fault(f.codes, f.shape, f.layout)) return c.refused_ = fault, c;
  return c;
}

// The commitment ID (heads.h) of the commitment whose anchor `first_table` reads (not checked: that
// it is the anchor). The reader's configuration digest is the caller's own, not a file's: a
// mismatch with c's, a fault() or a rejected c is a bug, and the process ends.
inline Digest published_id(const Configured &c, const IndexReader &first_table) {
  BLK_REQUIRE(c.refused().empty(), "published_id: the public files are not admitted");
  BLK_REQUIRE(!first_table.fault() && first_table.configuration() == c.configuration(), "published_id: the table's reader was formed under other public files, or contradicts itself");
  return published_id(c.configuration(), first_table.name(), record_seeds(first_table.shape(), first_table.name()), first_table.record().y);
}

#ifdef BLK_TESTED_TABLE_FOR_A_TEST
// For a test or a toy tool that checks a leaf file or a proof file under c without a key file.
// Checks nothing: the caller vouches for `key`'s table as if `check_key` or `check_entry` had
// produced it (see `TestedTable`, blocks/iop/foldeq.h).
inline TestedTable on_a_tests_word(const Configured &c, IndexKey key) {
  return TestedTable::on_a_tests_word(std::move(key), c.sized().d.set, key_public_bytes(key_fixed(c.sized(), c.configuration())));
}
#endif

}  // namespace blk
