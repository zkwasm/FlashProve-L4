// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// run_verifier() runs the verifier's side of one hash chain (one protocol's Fiat-Shamir
// transcript). Of its caller's `inputs`, the code that decides sees only what is in the chain's
// first query, the hash input that fixes the public inputs before the prover's first message. The
// toy protocols in blocks/toy/ and tests call it; the verifier program `cp_verify` does not. The
// comment above it lists the exceptions and the contract for callers of the library's verifying
// functions.
//
// The types of blocks/chain/public.h guard what enters a protocol block (one step of the proof
// system), not what the block's caller reads when it builds on the block's claims. Toy verifiers
// have several times accepted false claims in this way: their tests passed them a list of
// positions, a count or a shape from outside the first query.
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "chain/public.h"
#include "chain/rounds.h"
#include "chain/session.h"

namespace blk {

// Named intermediate values, for comparison with a second verifier. The library's verifiers only
// append to them. Not enforced: `decide` must not read them.
using Notes = std::vector<std::pair<std::string, E>>; // (name, value)

// run_verifier(label, plan, inputs, bind, decide, proof, bytes, why, notes):
//   `bind`    writes the public inputs into the first query and returns handles (`Public<..>`).
//   `decide`  is the verifier: it gets the channel (`VerifierRounds`) and the handles,
//             not the inputs.
// Both are function pointers, to which a lambda with captures does not convert. So, apart from
// global state, what `*notes` already contains and the plan, `decide` reads only `Public<..>`
// values of this chain, received messages, challenges and constants. The channel gives it the
// caller's whole plan (`Rounds::plan()`). Of a plan, `Binder` writes each round's name, counts and
// bits of proof of work into the first query, but neither the target in bits nor any error bound
// (see the contract below).
// Not enforced: `Bound` must contain only `Public<..>` handles. A plain copy of an input in it
// reaches `decide` without being in the first query, as does a field that a type's public_bytes()
// leaves out.
// `*why` gets the first reason for a rejection (`refuse` in the code). Not enforced: `decide` must
// pass `VerifierRounds::refuse()` string literals only, because the pointer is kept. A plan whose
// holds() is false is rejected before `bind` runs; the cause goes to stderr.
//
// Contract for callers of this function and of every verifying function of the library that takes
// a channel or a plan (`Planned`: the list of all rounds, each with its error bound).
//
// The plan must come from one of the library's composite planners, each of which plans several
// blocks (req_plan(), the .._plan() functions of blocks/frame/, a toy's). The planner must be
// called on one object of sizes (such as `Sized` in blocks/frame/files.h), and the codes and shapes
// written into the first query must be that object's.
//   Checked at run time: each round's name, kind, counts and bits of proof of work; and, where a
//   block asks (require_..() in blocks/chain/rounds.h), those inputs of the round's error bound
//   that the plan records (`Priced` in blocks/numbers/plan.h).
//   Never checked against the run: the error bound itself. It is in no chain's first query.
//   `Plan` and `Planned` (blocks/numbers/plan.h) keep a caller from writing or altering a round by
//   mistake, not a caller who sets out to get round them. A plan built with single blocks' planners
//   (the plan_..() functions of blocks/numbers/blocks.h) from sizes of the caller's choice is
//   trusted wherever no block asks.
//
// Across calls, key file, leaf file (`entry` in identifiers) and batch (the requests that are
// proved together) must be sized from one configuration. A batch's test on a committed table
// (`Index` in identifiers, blocks/iop/foldeq.h) is a later test; a leaf file's LINK proof ties its
// table to the key file's. The error bounds of both are true only of a table whose one-time test
// (the committer's own, as in a key file) ran at the same distances, target (the bits that each
// round's error bound must reach) and proof of work. Test code has got false claims accepted, at
// toy size, in six ways, all of which the second and third checks below stop. The six are listed,
// with figures, in blocks/README.md, "If you write a verifier on this library: a known limit".
//   Checked: plan, run and the table's reader (`IndexReader`: what a verifier has of a table) give
//   the same four numbers for the distances or, unless the proof is already rejected, the process
//   ends. The numbers are two radii and two integers m
//   that stand for radii (`IndexShape::distances`; in no first query). The next check covers the
//   `Setting`. In the programs `cp_commit`,
//   `cp_prove` and `cp_verify` the compared values are copies of one `Configured`'s numbers
//   (blocks/frame/configure.h), so no file can trigger this check there; it guards a caller that
//   computes sizes twice.
//   Checked, within one process (of no effect in `cp_verify`: first point under "Not checked"):
//   that the table has passed its one-time test, or a LINK proof to a table that has, at these
//   distances and under this `Setting` (target, bits of proof of work). A later test
//   must be given a `TestedTable` (blocks/iop/foldeq.h), and so must a LINK proof for its first
//   side, the key file's table. Outside tests, two toy tools (second point under "Not checked") and
//   the stand-in of the first point, only check_key_under() and check_entry_under(), which
//   check_key() and check_entry() call, create one that counts as tested, after they have accepted
//   the whole file; it records the `Setting` of that check and has no setter. Anything else gets no
//   yes: a reader filled in by hand, a reader from a prover's holder of the table (`IndexHolder`;
//   `IndexCpu::key()` in blocks/iop/index_cpu.h), or a table tested under another `Setting`. Unless
//   the proof is rejected earlier, the process then ends where the columns are read
//   (foldeq_columns_verify()).
//   Checked likewise: that the table's key file was checked under the same public values as the
//   later check's own. Some of what a key file states (the ranges of the matrix's entries, its
//   rows) is derived from the public files by a function of the calling program (`ShapesOf` in
//   blocks/frame/configure.h), and the configuration digest covers the files only. So two such
//   functions can give two different configurations with one digest, one `Setting` and one
//   commitment ID (published_id()). A `TestedTable` carries the key file's public values
//   (key_public_bytes() in blocks/frame/key_fixed.h); a leaf file's table carries those of the key
//   file to which it is tied. check_entry_under() and check_proof() end the process if these values
//   are not their own configuration's (tested by blocks/frame/tests/tested_table_test.cpp). No
//   later check relies on anything else that such a function adds. The rest (a request's segments,
//   layout, identifiers and rows on the table) is read by check_proof() only, under one
//   `Configured`, and no check builds on its result. A leaf file's public values do not depend on
//   the function.
//   Not checked:
//   - Anything across processes or files. A program that checks key file, leaf file and proof in
//     separate processes, as `cp_verify` does, reads the tables without their proofs
//     (`UnprovedTable`) and calls check_entry_if_tested() and check_proof_if_tested(). Inside these
//     two functions the `UnprovedTable` stands in for a `TestedTable`, with the `Setting` and
//     public values of the `Configured` under which it was read. With one `Configured`, as in each
//     process of `cp_verify`, the last two checks above compare it with itself and show nothing.
//     The results are of type `IfTested`, which has no plain yes. That all its processes passed, on
//     the same bytes, is that program's own logic.
//   - What a test vouches for. A test or toy tool that defines BLK_TESTED_TABLE_FOR_A_TEST can
//     declare a table tested. Two toy tools define it: blocks/frame/handover_files.cpp and
//     blocks/frame/handover_proof.cpp. A test that defines BLK_KEY_UNDER_FOR_A_TEST or
//     BLK_ENTRY_UNDER_FOR_A_TEST chooses the plans under which a file is checked, and an
//     `EntryUnder` made that way skips the comparison of public values. None of the three macros
//     compiles together with `SERVING_BUILD` (blocks/fence.h).
//   - A caller that composes blocks below the checkers of whole files (check_key(), check_entry(),
//     check_proof()): it is the author of a verifier. If it writes out a block's steps from the
//     public pieces (foldeq_columns_verify(), the channel's calls) instead of calling link_verify()
//     or lin_shared_verify(), it says itself which side of a LINK proof is under test, and checks
//     what it chooses to check. Only check_entry_under() and check_proof() compare the key file's
//     public values: below them, foldeq_columns_verify() compares distances and `Setting` only.
//   - A caller that reads the outcome from something other than the `IfTested` result. What
//     check_proof_if_tested() fills in beside its result (`ProofFacts`) and the position of the
//     `Reader` in the file also show whether the function rejected.
//   - What the calling program supplies; all of it is trusted. Its `ShapesOf` function: the ranges
//     and lookups that it derives from the files are whatever it returns, and the comparison of
//     public values only stops one process from mixing two such functions. The `links` and `more`
//     arguments of check_proof(); `bind` and `decide`; plans built from sizes of its choice. A
//     reader class of its own: `IndexReader` can be derived from, and on a side under test (the
//     one-time test; a LINK proof's further side, the leaf file's table) such a class answers
//     read() as it likes. The checkers of whole files make their own readers, so this does not
//     reach them.
//   - A caller who sets out to defeat the types. Everything above is claimed of code whose
//     behaviour the standard defines, naming no private member, declaring nothing in blk or std,
//     and nothing at global scope before these headers, operator new kept, no fence.h macro set. Outside it,
//     the comment on `Plan` and `Planned` in blocks/numbers/plan.h has the one list of the ways
//     found so far, among them macros, casts and what does a cast's work, a reference kept beyond
//     its object's lifetime, an explicit instantiation or specialisation over a private name,
//     declarations in namespace blk. All of them apply to `TestedTable` and `UnprovedTable`. Two of
//     the last kind: a source file that defines one of their friend functions itself, instead of
//     including its header, makes such tables as it likes; one that defines the struct
//     `LinkAccess`, which blocks/iop/link.h leaves to the tests, can call the private steps of
//     `Link` (a LINK proof's prover and verifier) one by one and leave out the proximity test.
//
// A program keeps the contract by passing its checkers one `Configured` and no plan, size or shape
// from elsewhere. `cp_verify` does so, apart from its `links` callback and its `ShapesOf` function.
// Both work on one set of sub-proof plans and shapes, which the program builds before it calls
// configure() and which the library trusts (see "What the calling program supplies"). Each of the
// three stages of `cp_verify` (key, leaf, request) calls configure() once, makes no
// `IndexHolder`, and reads its tables only with read_key_table_unproved() and
// read_entry_table_unproved() on that `Configured`. The tables go to check_entry_if_tested() (stage
// leaf) and check_proof_if_tested() (stage request); the stage key calls check_key() and drops the
// `TestedTable` that it gets. The stages are forked after the files were read, so all configure
// from the same bytes. A key or leaf file made under another parameter file fails every opening:
// every hash in a table's tree starts from the configuration digest, which covers the parameter
// file.
//
// That all stages judge under the same configuration files is cp_verify's own logic, which the
// library does not check: a whole call reads each file once and runs all three stages on those
// bytes; its part-calls (-key, -leaf) never say YES.
template <class Inputs, class Bound>
[[nodiscard]] bool run_verifier(u64 label, const Planned &plan, const Inputs &inputs, Bound (*bind)(Binder &, const Inputs &),
                                bool (*decide)(VerifierRounds &, const Bound &, Notes *), const u8 *proof, size_t bytes,
                                const char **why = nullptr, Notes *notes = nullptr) {
  if (!plan->holds()) { // no proof is worth anything under such parameters
    const char *words = does_not_hold(plan, "run_verifier");
    if (why) *why = words;
    return false;
  }
  Reader reader(proof, bytes);
  Binder binder(label, plan);
  const Bound bound = bind(binder, inputs);
  VerifierSession session; // one chain, one session: the verdict is read from it
  VerifierRounds ch(binder, reader, session);
  if (!decide(ch, bound, notes) && ch.ok()) ch.refuse("a block's verifier has refused without giving its reason");
  (void)ch.finish(); // ends the chain, as `accepted` requires, and rejects leftover bytes
  const bool ok = session.accepted();
  if (why) *why = ok ? nullptr : (ch.why() ? ch.why() : "a block's verifier has refused without giving its reason");
  return ok;
}

// The same for a verifier's side that returns something: a block's linear claims, or a received
// value that the caller compares with its own. `out` is reset first, so data only flow out through
// it; the caller reads it after the verdict. The contract above applies.
template <class Inputs, class Bound, class Out>
[[nodiscard]] bool run_verifier(u64 label, const Planned &plan, const Inputs &inputs, Bound (*bind)(Binder &, const Inputs &),
                                bool (*decide)(VerifierRounds &, const Bound &, Out &, Notes *), const u8 *proof, size_t bytes, Out &out,
                                const char **why = nullptr, Notes *notes = nullptr) {
  out = Out();
  if (!plan->holds()) {
    const char *words = does_not_hold(plan, "run_verifier");
    if (why) *why = words;
    return false;
  }
  Reader reader(proof, bytes);
  Binder binder(label, plan);
  const Bound bound = bind(binder, inputs);
  VerifierSession session; // one chain, one session: the verdict is read from it
  VerifierRounds ch(binder, reader, session);
  if (!decide(ch, bound, out, notes) && ch.ok()) ch.refuse("a block's verifier has refused without giving its reason");
  (void)ch.finish(); // ends the chain, as `accepted` requires, and rejects leftover bytes
  const bool ok = session.accepted();
  if (why) *why = ok ? nullptr : (ch.why() ? ch.why() : "a block's verifier has refused without giving its reason");
  return ok;
}

} // namespace blk
