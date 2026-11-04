// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Writes and checks a leaf file (`entry` in identifiers): `make_entry` is the committer's side,
   `check_entry` the verifier's, or `check_entry_if_tested` if another process checks the key file.
   The anchor (`first`; "first table" in messages) is a commitment's first committed table (`Index`
   in identifiers). A leaf table (`further`; "copy" in messages) is another table with the same
   entries and fresh private randomness (`coins` in identifiers).

   A leaf file holds a leaf table's root, its record (`IndexRecord`: stored answers to samples of
   the table's rows) and a LINK proof, whose messages
   blocks/iop/link.h lists. The proof is meant to show that root and record determine one table of
   codewords close to the committed rows, and that before the cut that table's message entries
   equal the anchor's. The cut (`LinkShape::cut`) is the first index of a row's message at which
   two tables of one commitment may differ. So what a key file asserts about the anchor
   (blocks/frame/check_key.h) is meant to hold of the leaf table too; requests are proved against
   leaf tables. */
#pragma once
#include <string>
#include <vector>

#include "chain/closed.h"  // Notes
#include "frame/configure.h"
#include "frame/heads.h"
#include "frame/link_frame.h"
#include "iop/link.h"

namespace blk {

/* The public parameters of a LINK proof, fixed before a leaf table is committed. Both sides derive
   them from the public files (those given to `configure`), never from the leaf file. Both tables
   have the same shape and configuration digest.
   configuration: the configuration digest, a hash of all the public files.
   link: the tables' shape, each table's count of drawn columns, the proximity test's stages.
   codes: those that `link` refers to. */
struct EntryFixed {
  Digest configuration{};
  Codes codes;
  LinkShape link;
};

// entry_codes, link: as link_sized() fills them (`Sized::entry_codes`, `Sized::link`).
inline EntryFixed entry_fixed(const Digest &configuration, const Codes &entry_codes, const LinkSized &link) {
  EntryFixed f;
  f.configuration = configuration, f.codes = entry_codes, f.link = link.S;
  return f;
}

// Returns null, or what is wrong with the configuration. On a fault `make_entry_under` aborts the
// process and `check_entry_under` rejects; both ask before anything else.
// plan (`Planned`, blocks/numbers/plan.h): the chain's rounds with their error bounds.
inline const char *entry_fault(const Planned &plan, const EntryFixed &f) { return link_fault(f.codes, f.link, plan->set.samples); }

/* A plan with the values that it was made for: the `_under` functions take them only in this form.
   Only `entry_under` constructs it, unless the including file first defines
   BLK_ENTRY_UNDER_FOR_A_TEST, which does not compile under SERVING_BUILD (blocks/fence.h).
   Reason: the shape splits the positions of the round `LINK` between the two tables, and the plan
   computed that round's error bound from its own two counts. Under a plan made for other values,
   `check_entry_under` could accept on error bounds that do not apply to this shape. `configure`
   derives both from the same files, and `link_plan` (blocks/frame/link_frame.h) aborts the process
   if the counts differ. For a pair from elsewhere, the channel (blocks/chain/rounds.h: each side's
   object for sending, receiving and drawing challenges) aborts the process if split or table
   differs from the plan's (`require_first_list`, `require_table`), but it compares only the
   numbers that a plan records with a round, never an error bound itself. Under a plan edited by
   hand that records this shape's counts but a bound computed from larger counts, an acceptance
   rests on fewer columns of the leaf table than the bound assumes.
   key_public: a `KeyFixed` (configuration values under which a key file is checked) as bytes,
   without the table's test distances (`key_public_bytes`, blocks/frame/key_fixed.h). The anchor's
   key file must have been checked under the same bytes. */
class EntryUnder {
 public:
  const Planned plan;
  const EntryFixed fixed;  // stored by value: `entry_under` computes it; no caller can change it
  const std::vector<u8> key_public;  // empty only after the test constructor, and then not compared
#ifdef BLK_ENTRY_UNDER_FOR_A_TEST
  EntryUnder(const Planned &p, const EntryFixed &f) : plan(p), fixed(f) {}
#endif
 private:
  struct FromConfigured {};
  EntryUnder(const Planned &p, EntryFixed f, std::vector<u8> k, FromConfigured) : plan(p), fixed(std::move(f)), key_public(std::move(k)) {}
  friend EntryUnder entry_under(const Configured &c);
};
#ifdef BLK_ENTRY_UNDER_FOR_A_TEST
BLK_MARK_NOT_FOR_SERVING();  // blocks/fence.h misses a macro defined after its first include
#endif

/* The committer's run on a given plan and values; programs call `make_entry`, below.
   Layout of a leaf file: header (`head` in identifiers: tag `TAG_ENTRY`, version word) | root |
   the record's answers | LINK proof. No part has a length field. The LINK proof is one hash chain,
   that is, one Fiat-Shamir transcript, and must end at the file's last byte. The file holds no
   table number: leaf tables are distinguished by their roots.
   Returns true after appending a complete leaf file to `file`; false, with nothing appended, if a
   table has too few columns left to open (a column: all rows' symbols at one position): its random
   tails (random coefficients encoded with each row) cover a bounded number.
   A run cannot be resumed: every call draws new private randomness from `coins`, and a table
   counts the run's columns as spent from `IndexHolder::begin` on, whether or not the run finishes.
   If the anchor is the table with too few columns left, the leaf table's columns are spent all the
   same: `link_prove` calls the leaf table's `begin` first.
   first, further: the anchor's and the leaf table's `IndexHolder` (blocks/iop/foldeq.h: the
   committer's access to a committed table). search: finds a proof-of-work nonce
   (`Engine::search_nonce`); may be empty. */
[[nodiscard]] inline bool make_entry_under(const EntryUnder &under, IndexHolder &first, IndexHolder &further, Engine &engine, Coins &coins, const NonceSearch &search, Writer &file) {
  const Planned &plan = under.plan;
  const EntryFixed &fixed = under.fixed;
  const char *fault = entry_fault(plan, fixed);
  BLK_REQUIRE(!fault, fault ? fault : "");
  const IndexName name{further.commitment(), further.record()};
  Writer out;  // copied to `file` only after the whole run has succeeded
  write_head(out, TAG_ENTRY);
  BLK_REQUIRE(name.commitment.size() == DIGEST_BYTES, "make_entry: a table of this kind is committed to by its root");
  out.put_bytes(name.commitment.data(), DIGEST_BYTES);
  for (const E &y : name.record.y) out.put(y);
  if (!link_prove(plan, fixed.configuration, fixed.codes, fixed.link, first, further, engine, coins, out, search)) return false;
  file.put_bytes(out.bytes().data(), out.bytes().size());
  return true;
}

/* Reads a leaf file up to its LINK proof and fills `entry` with the leaf table's key (`IndexKey`,
   what a verifier keeps of one committed table: shape, configuration digest, root, answers). The
   LINK proof is not read, so nothing is proved about the table: a caller must accept nothing unless
   `check_entry` also passes on the same bytes.
   Checked: tag, version word, canonical field elements, `index_key_fault`; the count of answers
   read is the shape's. `entry` is written only on success, and `file` is then at the LINK proof's
   first byte. `check_entry_under` parses this part with the same function.
   Root and answers are bytes of the file, not messages of the chain: `link_verify` puts both
   tables' roots and records into the chain's first query (the hash input that starts the chain),
   so a changed root changes every challenge (docs/limits-all-rows.tsv, row T7.1.8). */
[[nodiscard]] inline bool read_entry_table_unproved(const EntryFixed &fixed, Reader &file, IndexKey &entry, const char *&why) {
  if ((why = read_head(file, TAG_ENTRY))) return false;
  const IndexShape &shape = fixed.link.table;
  IndexKey read(shape, fixed.configuration, Digest{}, IndexRecord());  // shape and configuration are the verifier's own; root and answers the file's
  if (!file.get(read.root)) return why = "the copy's root", false;
  read.rec.y.resize(shape.rows() * shape.answers);
  for (E &y : read.rec.y)
    if (!file.get(y)) return why = "the record's answers", false;
  if ((why = index_key_fault(read))) return false;
  entry = std::move(read);
  return true;
}

/* The verifier's check on a given plan and values; programs call `check_entry` or
   `check_entry_if_tested`, below. Returns true if `file` is a leaf file under this configuration
   and its LINK proof ties its table to the table of `first`.
   first: the anchor. It must come from `check_key` (blocks/frame/check_key.h) on an accepted key
     file, never from the leaf file; `check_entry_if_tested` instead passes a table read from the
     key file without its proofs. That it is the anchor is not checked: a leaf table returned by an
     earlier `check_entry` is accepted as `first` too. A `TestedTable` (blocks/iop/foldeq.h) holds
     a table's key, the `Setting` (field, security target, proof-of-work bits, ...) under which its
     check passed, and the `key_public` bytes of the key file's check. A default-constructed
     `first` is rejected; other bytes than a non-empty `under.key_public` abort the process.
   entry: on true, the leaf table, for every proof made against it: its key, the plan's `Setting`
     and the `key_public` bytes of `first`.
   why: the first reason for a rejection, for a log; null after true.
   notes (may be null): intermediate values; no decision reads them.
   place (may be null): where in the file the rejection happened, since one reason (a nonce, an
     opening) can come from several places: one of ENTRY_PLACE_.. or, from `link_verify`, of
     LINK_REFUSED_..; "" after true. */
inline constexpr const char *ENTRY_PLACE_PARAMETERS = "the parameters", *ENTRY_PLACE_TABLE = "head, root and answers";
[[nodiscard]] inline bool check_entry_under(const EntryUnder &under, const TestedTable &first, Reader &file, TestedTable &entry, const char *&why, Notes *notes = nullptr,
                                            const char **place = nullptr) {
  const Planned &plan = under.plan;
  const EntryFixed &fixed = under.fixed;
  const char *unused;
  const char *&where = place ? *place : unused;
  where = ENTRY_PLACE_PARAMETERS;
  if ((why = entry_fault(plan, fixed))) return false;
  if (!plan->holds()) return why = does_not_hold(plan, "check_entry"), false;  // no leaf file is worth anything under such parameters
  // A rejection, not an abort: this is the case after `check_key` returned NO, for example, which
  // can depend on a key file's bytes. The BLK_REQUIRE below depends only on the caller's objects.
  if (!first.tested_under()) return why = "the first table has passed no check: its key was refused, or was never checked", false;
  BLK_REQUIRE(under.key_public.empty() || first.key_public() == under.key_public,
              "check_entry: the first table's key was checked under other public values (ranges, rows, codes) than this configuration's: key, entry and proof must be checked under one Configured");
  where = ENTRY_PLACE_TABLE;
  IndexKey read;
  if (!read_entry_table_unproved(fixed, file, read, why)) return false;
  VerifierSession session;  // one session for each verdict; a leaf file's check is one verdict
  LinkWhy said;
  if (!link_verify(plan, fixed.configuration, fixed.codes, fixed.link, first, read, file, session, &said, notes)) return where = said.step, why = said.reason, false;  // also: no byte is left over
  where = "";
  entry = TestedTable(std::move(read), plan->set, first.key_public());
  why = nullptr;
  return true;
}

/* The forms that programs call; arguments as in the `_under` forms, but `read_entry_table_unproved`
   fills an `UnprovedTable` (blocks/iop/foldeq.h): the key with c's `Setting` and `key_public`
   bytes. Plan, codes, shape and digest all come from one `Configured`, that is, from the public
   files only. If `configure` rejected those files (`Configured::refused` is not empty),
   `entry_under` and `make_entry` abort the process, `read_entry_table_unproved` returns false and
   `check_entry` NO. `check_entry` has three verdicts, like a proof file's check (`ProofVerdict`,
   blocks/frame/configure.h): YES_WITH_WORK if some round's error meets the target (the bits that
   each round's error bound must reach) only with its proof of work counted; the caller then decides
   from `Configured::work_of_an_entry`. */
inline EntryFixed entry_fixed(const Configured &c) { return entry_fixed(c.configuration(), c.sized().entry_codes, c.sized().link); }
inline EntryUnder entry_under(const Configured &c) {  // plan and values are both from c
  BLK_REQUIRE(c.refused().empty() && c.entry().chains.size() == 1, "an entry has one chain");
  return EntryUnder(c.entry().chains[0].plan, entry_fixed(c), key_public_bytes(key_fixed(c.sized(), c.configuration())), EntryUnder::FromConfigured{});
}
[[nodiscard]] inline bool make_entry(const Configured &c, IndexHolder &first, IndexHolder &further, Engine &engine, Coins &coins, const NonceSearch &search, Writer &file) {
  BLK_REQUIRE(c.refused().empty(), "make_entry: the public files are not admitted");
  return make_entry_under(entry_under(c), first, further, engine, coins, search, file);
}
[[nodiscard]] inline bool read_entry_table_unproved(const Configured &c, Reader &file, UnprovedTable &entry, const char *&why) {
  if (!c.refused().empty()) return why = "the public files are not admitted", false;
  IndexKey read;
  if (!read_entry_table_unproved(entry_fixed(c), file, read, why)) return false;
  entry = UnprovedTable(std::move(read), c.sized().d.set, key_public_bytes(key_fixed(c.sized(), c.configuration())));
  return true;
}
[[nodiscard]] inline ProofVerdict check_entry(const Configured &c, const TestedTable &first, Reader &file, TestedTable &entry, const char *&why, Notes *notes = nullptr,
                                              const char **place = nullptr) {
  if (place) *place = ENTRY_PLACE_PARAMETERS;
  if (!c.refused().empty()) return why = "the public files are not admitted", ProofVerdict::NO;
  if (!check_entry_under(entry_under(c), first, file, entry, why, notes, place)) return ProofVerdict::NO;
  return c.work_of_an_entry().needed ? ProofVerdict::YES_WITH_WORK : ProofVerdict::YES;
}
/* `check_entry` for a program that checks the key file in another process on the same bytes, as
   `cp_verify` does. first: read from the key file without its proofs (`read_key_table_unproved`,
   blocks/frame/check_key.h). A result other than NO is worth a yes only if that other run of
   `check_key` has passed, which nothing here checks. The leaf table is not returned: a caller that
   goes on to a proof file reads it with `read_entry_table_unproved`. Unless the result is NO, `why`
   and `*place` are set to IF_TESTED_WHY. */
[[nodiscard]] inline IfTested check_entry_if_tested(const Configured &c, const UnprovedTable &first, Reader &file, const char *&why, Notes *notes = nullptr, const char **place = nullptr) {
  TestedTable unused;
  const ProofVerdict verdict = check_entry(c, first.as_if_tested(), file, unused, why, notes, place);
  if (verdict == ProofVerdict::NO) return IfTested::NO;
  why = IF_TESTED_WHY;  // never null here, so that "no reason was given" cannot be taken for a yes
  if (place) *place = IF_TESTED_WHY;
  return verdict == ProofVerdict::YES ? IfTested::YES_IF_THE_TABLE_IS_TESTED : IfTested::YES_WITH_WORK_IF_THE_TABLE_IS_TESTED;
}

}  // namespace blk
