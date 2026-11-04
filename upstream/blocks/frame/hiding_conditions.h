// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Conditions that hiding puts on a configuration (the parameter file and the other public files):
   chiefly, enough random values for everything that proofs open or send. A function that tests
   returns "" or the first condition missed, as "<name>: <what>"; the numbers in that text follow
   from the public files and from `TableUse`. Some returns bear a condition's name but say that the
   count could not be made (plan and shape disagree, a case is not handled, the input cannot be
   used), and hiding_conditions_missed_by_the_file() can also return "parameters-not-admitted: ..",
   which is none of the names below. The programs ask only whether the result is empty, so each counts
   as a condition missed.

   All three programs test these conditions on the parameter files: `cp_verify` too refuses files
   whose numbers miss one (`configure`, blocks/frame/configure.h, which committer, prover and
   verifier all run). No verifier can test what is no fact of the files: how often a stored table
   has been used, or whether the prover drew and laid its random values as the rule says: a proof
   does not show it. So a serving prover or committer must also call in time:
   `hiding_conditions_missed_by_the_file` as soon as the parameter file is parsed, and
   `hiding_conditions_missed` (a) before a column of a committed table is counted or opened, (b)
   before a request is accepted, (c) before a byte that depends on a seed, a weight or private
   randomness is written to a file, a cache or a socket.

   A proof made under parameters that miss a condition passes every check on the proof itself and
   may reveal something about the weights. `configure` tests with the default `TableUse`, not with
   how often openings were really begun on a table. The default counts no column of a commitment's
   first table that a LINK proof opens (it ties a further table to the first): see `TableUse`. The
   sixteen conditions that are not arithmetic on parameters, listed at `hiding_conditions_missed`,
   are tested by no function here, and so not by `configure`. What an empty result means: see
   `hiding_conditions_missed`. */
#pragma once
#include <string>
#include <vector>

#include "code/code.h"
#include "frame/files.h"

namespace blk {

/* Words; in brackets, the names in the code and in its messages.
     chain, plan    the Fiat-Shamir transcript of one protocol (blocks/chain/chain.h); its rounds
                    and messages (`Planned`, blocks/numbers/plan.h)
     oracle message main rows and mask rows, each a codeword, sent as Merkle roots (`OracleNotice`)
     random tail    a row's r random coefficients (`Code::r`, "random entries"), which are to hide
                    up to `hiding_positions` opened positions (blocks/code/code.h)
     mask entry     one element of a mask row (a row of random values only)
     list           the positions drawn in one round
     proximity test is to show that the rows of a group (oracle messages read at one list) are close
                    to codewords (one stage: `IoppStage`, blocks/iop/iopp.h)
     table          a committed table ("stored table" above): the weights as a matrix of encoded
                    rows under a Merkle tree
     commitment     an anchor and its leaf tables ("key" in messages)
     anchor         a commitment's first committed table (`first`, "first table")
     leaf table     a further committed table of the same weights (`copy`, `further`, "stored copy";
                    in `first_copy`, `copies` and one refusal these words take in the anchor too)
     LINK proof     ties a leaf table to the anchor (`z.link`, "link of two tables")
     sub-proof      the argument for one part of the model's computation (`Subproofs`,
                    blocks/frame/files.h)
     segment        one sub-proof's part of a request's first chain (`ReqLink`, `z.request.link`)
     LIN            the test on the committed table, for a batch or as the committer's one-time
                    test. Among its oracle messages (blocks/iop/lin.h): the folded row ("folded
                    table"), Omega ("second oracle"), each request's W ("own vector") and index-mask
                    message (mask rows only, for its pending claims)
     pending claim  a linear claim ("row") on the table that a sub-proof leaves for LIN
                    (`pending_rows`)
     pad            the entries ("padding-entries" in messages) that mask the answers to an oracle
                    message's out-of-domain sample (a random linear form of the whole message,
                    blocks/iop/send.h)

   Counting. The sizes under test are derived in blocks/frame/derive.h, blocks/frame/lin_frame.h and
   blocks/frame/link_frame.h. The functions here compare them with a separate count, taken from the
   plans' draws of positions, the shapes that prover and verifier run on, `hiding_positions` and the
   scalars that a sub-proof's own plan sends. Draws count with their repeats, and the lists at which
   one encoding of a row is opened add up. A kept row (a main row that later segments read) is sent
   again, with a fresh random tail, in every segment that reads it, so each encoding is read at one
   list.

   The conditions, by their reported names.
     main-rows      Every main row's random tail hides all draws of all lists at which that encoding
                    of the row is opened.
     mask-rows      The same for every mask row, under its mask code.
     mask-room      Every oracle message from which protocol blocks take mask entries has room for
                    them, each taken once, and for its own pad.
     fold-room      The hiding polynomial of a fold over k row variables, h of them shared with the
                    mask rows, fits a mask row: 2k + 1 - h entries.
     table          A committed table's random tail hides the columns of the one-time test, of the
                    LINK proofs on that table and of all batches allowed on it (`batches`).
                    `TableUse` says how many tests and LINK proofs.
     padding-entries
                    The oracle messages of the folded row, Omega and W have room for their data and
                    their pads. W's pad is in its mask row if it has one. That row's room for the
                    pad is not tested here: a run aborts in `Send::pad` (blocks/iop/send.h) if the
                    row is too short, and the row's `MaskLedger` gives out no entry of the pad. For
                    the record of stored sample answers (the line `record`), a row of the table has
                    three places outside the weights' layout for each answer.
     range-checks   An honest prover fails an approximate range check (RNG, blocks/iop/rng.h) when a
                    combination of cells leaves the digits' range, and the chance of that is part of
                    the zero-knowledge error. Summed over a request's checks, times the requests
                    that a commitment can serve, it is at most 2^-target (the line `sec`).
     salt           A tree leaf's salt has at least 40 bytes.
     leaves         Oracle messages of one prover message with the same leaf class share a tree's
                    leaves, and opening a leaf reveals all of it: so only messages read at exactly
                    the same lists share a class. What a request's first prover message carries
                    outside every segment (`at_start`) shares a class with no segment's message.
                    Omega, which every request's own test reads, shares a class with nothing else in
                    its prover message: no mask of the batch's test on Omega may be in Omega's
                    leaves.
     batch          LIN is sized for as many requests as a batch may hold (the line `slots`). */

// The number of positions that each POSITIONS round of a chain's plan draws, in order.
inline std::vector<u64> lists_of(const Planned &p) {
  std::vector<u64> t;
  for (const PlanRound &r : p->rounds_)
    if (r.kind == RoundKind::POSITIONS) t.push_back(r.draw_positions);
  return t;
}

/* One approximate range check of a request. rows, columns: of the integer matrix of unpacked cells,
   not of the oracle message's rows. bound: every cell is below it in absolute value, whatever
   request a client sends (not checked here). combinations: of each row's cells, with coefficients
   0 (probability 1/2), 1 and -1 (1/4 each). A combination must lie in [-range, range), which its
   two digits, of digit_bits[] bits, must represent exactly. */
struct RangeCheck {
  u64 rows = 0, columns = 0, combinations = 0, bound = 0, range = 0;
  unsigned digit_bits[2] = {0, 0};
};
/* How often a commitment's one-time test and LINK proofs have been begun. The caller counts; every
   caller outside the tests passes the defaults. Under the defaults the condition `table` counts no
   column that LINK proofs open of the anchor. `cp_commit` limits those as it runs:
   `count_first_table_columns` (prover/common/cp_files.h) adds the columns of each opening begun on
   the anchor to a count file, and an opening that would take the count past `hiding_positions` is
   not made. The file counts for its own directory only: a copy of the directory has its own count.

   A batch, a LINK proof and a one-time test count in full once they are begun on a table, whether
   or not a column was opened or sent: also a run that stopped, failed, was withheld or was started
   again, and a test or trial run under the production secrets, on every process and machine that
   has them. The count must be kept where a crash cannot set it back, and of two concurrent batches
   each must reserve its columns before it draws. */
struct TableUse {
  u64 one_time_tests = 1;       // on the anchor
  u64 links_on_the_first = 0;   // LINK proofs of leaf tables: each opens columns of the anchor
  u64 links_on_a_copy = 1;      // on one leaf table: its own LINK proof, each time it was begun
};
/* A lower bound on -log2 of the chance that an honest prover fails the check; negative if none is
   shown. With C = columns, B = bound, D = range: a coefficient times a cell a has the moment
   generating function (1 + cosh(x a)) / 2 = cosh(x a / 2)^2 <= exp(x^2 a^2 / 4), so a combination S
   of C cells has P(|S| >= D) <= 2 exp(-D^2 / (C (B - 1)^2)); a union bound over rows and
   combinations. 1.4426 < log2(e). */
inline long range_check_bits(const RangeCheck &c) {
  if (!c.rows || !c.columns || !c.combinations || c.bound < 2 || !c.range) return -1;
  if (c.range >> 40 || c.bound >> 32 || c.columns >> 32 || c.rows >> 40 || c.combinations >> 16) return -1;
  if (c.digit_bits[0] + c.digit_bits[1] > 41 || ((u64)1 << (c.digit_bits[0] + c.digit_bits[1])) != 2 * c.range) return -1;  // else the bound below is not about what is tested
  const u128 above = (u128)c.range * c.range * 14426, below = (u128)c.columns * (c.bound - 1) * (c.bound - 1) * 10000;
  const u128 bits = above / below;
  if (bits >> 30) return (long)1 << 30;
  return (long)bits - 1 - (long)ceil_log2(c.rows) - (long)ceil_log2(c.combinations);
}

inline std::string hiding_missed(const char *name, const std::string &what, u64 have, u64 need) {
  return std::string(name) + ": " + what + ": " + std::to_string(have) + " where " + std::to_string(need) + " are needed";
}
// One proximity test: its list i, lists[at + i], reads the rows of its stage i and nothing
// else. more_main, more_mask: further draws on the first stage's main and mask rows. Not checked:
// that `lists` has at + stages.size() entries; the callers compare the sizes first.
inline std::string hiding_missed_in_test(const std::string &what, const Codes &codes, const std::vector<IoppStage> &stages, const std::vector<u64> &lists, size_t at, u64 more_main,
                                         u64 more_mask) {
  for (size_t i = 0; i < stages.size(); ++i) {
    const std::string where = what + ", stage " + std::to_string(i + 1);
    const u64 t = lists[at + i], main = hiding_positions(codes.base[stages[i].code]), mask = hiding_positions(codes.mask[stages[i].mask_code]);
    if (main < t + (i ? 0 : more_main)) return hiding_missed("main-rows", where, main, t + (i ? 0 : more_main));
    if (mask < t + (i ? 0 : more_mask)) return hiding_missed("mask-rows", where, mask, t + (i ? 0 : more_mask));
    if (codes.mask[stages[i].mask_code].ell < 2 * (u64)stages[i].k + 1 - stages[i].h)
      return hiding_missed("fold-room", where, codes.mask[stages[i].mask_code].ell, 2 * (u64)stages[i].k + 1 - stages[i].h);
  }
  return "";
}

// An oracle message and whose lists read it, one bit each (OWN: one list of each request's test).
struct ReadAt {
  OracleNotice notice;
  unsigned lists = 0;
};
// The condition `leaves` within one prover message.
inline std::string hiding_missed_in_leaves(const std::string &where, const std::vector<ReadAt> &message) {
  for (size_t i = 0; i < message.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (message[i].notice.leaf_class == message[j].notice.leaf_class && message[i].lists != message[j].lists)
        return "leaves: " + where + ": " + message[i].notice.shape.name + " shares leaves with " + message[j].notice.shape.name + ", which is read at other lists";
  return "";
}
// The condition `leaves` in LIN's three prover messages with oracle messages. The bits say whose
// lists read an oracle message: those of the test on the folded row's group (FOLDED), on Omega's
// group if that is another (SECOND), of a request's own test (OWN).
inline std::string hiding_missed_in_the_tables_leaves(const std::string &where, const LinShape &S) {
  enum : unsigned { FOLDED = 1, SECOND = 2, OWN = 4 };
  const bool split = S.shared_split_needed();
  if (S.own_split_needed()) return "leaves: " + where + ": a request's own vector under another code than the second oracle's is not reckoned with here";
  std::vector<ReadAt> u, omega, w;
  for (const OracleNotice &n : lin_message_u(S)) u.push_back(ReadAt{n, FOLDED});  // the folded row; if `masks_ahead`, the fresh mask of its test's first fold
  const std::vector<OracleNotice> second = lin_message_omega(S);
  for (size_t i = 0; i < second.size(); ++i)  // Omega; the mask-only message that parts a claim between two groups; any other: a mask of the batch's test on Omega
    omega.push_back(ReadAt{second[i], i == 0 ? (split ? SECOND : FOLDED) | OWN : split && i == 1 ? FOLDED | SECOND : split ? SECOND : FOLDED});
  for (const OracleNotice &n : lin_message_w(S)) w.push_back(ReadAt{n, OWN});  // a request's W; if `masks_ahead`, the fresh mask of its own test's first fold
  for (const std::vector<ReadAt> *m : {&u, &omega, &w}) {
    const std::string why = hiding_missed_in_leaves(where, *m);
    if (!why.empty()) return why;
  }
  return "";
}

/* The conditions on the committed table, on LIN and on the LINK proof. z: from `sized`, or from
   `sized_table` where no request's shape is known yet (blocks/frame/files.h).
   pending_rows: one request's pending claims. */
inline std::string hiding_conditions_missed_on_the_table(const Sized &z, u64 pending_rows, const TableUse &use = TableUse()) {
  const Setting &s = z.d.set;
  if (s.salt_bytes < 40) return hiding_missed("salt", "bytes of a leaf's salt", s.salt_bytes, 40);
  for (int once = 0; once < 2; ++once) {  // before anything is planned: a planner takes the shapes' leaf classes as they are
    const std::string why = hiding_missed_in_the_tables_leaves(once ? "one-time test" : "batch", (once ? z.once : z.batch).S);
    if (!why.empty()) return why;
  }
  FilePlan batch, one_time_test;
  add_table_chains(batch, s, z.par, z.proof_codes, z.batch, false), add_table_chains(one_time_test, s, z.par, z.key_codes, z.once, false);
  const FilePlan entry = file_entry(z, false);
  for (int once = 0; once < 2; ++once) {
    const LinSized &x = once ? z.once : z.batch;
    const Codes &codes = once ? z.key_codes : z.proof_codes;
    const FilePlan &file = once ? one_time_test : batch;
    const LinShape &S = x.S;
    const u64 requests = once ? 1 : z.par.slots;
    if (x.lines.requests != requests)
      return hiding_missed("batch", once ? "requests in the one-time test" : "requests that the test on the table is sized for", x.lines.requests, requests);
    const std::vector<u64> shared = lists_of(file.chains[0].plan), own = lists_of(file.chains[1].plan);
    const bool split = S.shared_split_needed();
    if (shared.size() != 1 + S.test_u.size() + (split ? S.test_omega.size() : 0) || own.size() != S.test_own.size() || own.empty())
      return "main-rows: the chains of the test on the table have other lists than the columns and one for each stage";
    {
      const u64 answers = z.par.answers, in_a_row = z.par.table_data + z.par.table_random;
      const u64 room_u = (u64)S.fold.u.main_rows * codes.base[S.fold.u.code].ell, room_omega = (u64)S.omega.main_rows * codes.base[S.omega.code].ell;
      const u64 room_w = (u64)S.w.main_rows * codes.base[S.w.code].ell, columns = (u64)1 << S.k_col();
      const u64 weights_in_a_row = S.matrix_rows << S.k_lo;
      if (z.par.table_data < weights_in_a_row + 3 * answers)
        return hiding_missed("padding-entries", "places of a row of the table", z.par.table_data, weights_in_a_row + 3 * answers);
      if (room_u < in_a_row + answers) return hiding_missed("padding-entries", "entries of the folded table's oracle message", room_u, in_a_row + answers);
      if (room_omega < S.matrix_rows + answers) return hiding_missed("padding-entries", "entries of the second oracle's message", room_omega, S.matrix_rows + answers);
      if (room_w < columns + (S.w.mask_rows ? 0 : answers))
        return hiding_missed("padding-entries", "entries of a request's own vector's message", room_w, columns + (S.w.mask_rows ? 0 : answers));
    }
    {  /* The index-mask message needs one entry for each pending claim; if there are several, the
          hiding polynomial of their batching, 1 + 2 for each column variable; its pad. In the
          one-time test the claims come from the committer's range proof (WF, blocks/iop/wf.h;
          `ranges` in identifiers): one for each part of each lookup, and each takes a second entry
          there. Where the range proof is not sized (`sized_table`): one claim. */
      u64 parts = 0;
      for (const WfLookup &l : z.ranges.x.lookups) parts += l.parts.size();
      const u64 pending = !once ? pending_rows : parts ? parts : 1;
      const u64 need = pending + (once ? parts : 0) + (pending > 1 ? 1 + 2 * (u64)S.k_col() : 0) + z.par.answers;
      const u64 room = (u64)x.input.shape.mask_rows * codes.mask[x.input.shape.mask_code].ell;
      if (room < need) return hiding_missed("mask-room", once ? "the masks of the committer's rows on the table" : "the masks of a request's rows on the table", room, need);
      if (split) {
        const u64 apart = (u64)S.shared_split.mask_rows * codes.mask[S.shared_split.mask_code].ell;
        if (apart < 1 + z.par.answers) return hiding_missed("mask-room", "the masks that part a row between two groups", apart, 1 + z.par.answers);
      }
    }
    const u64 t_u = shared[1], t_omega = split ? shared[1 + S.test_u.size()] : t_u, t_own = own[0];
    /* Omega is read by the batch's test and by every request's own. The mask-only message by which
       SPLIT parts a claim between two groups is read at both groups' lists. */
    const std::string where = once ? "one-time test" : "batch";
    std::string why = hiding_missed_in_test(where + ", folded table", codes, S.test_u, shared, 1, split ? 0 : requests * t_own, split ? t_omega : 0);
    if (why.empty() && split) why = hiding_missed_in_test(where + ", second oracle", codes, S.test_omega, shared, 1 + S.test_u.size(), requests * t_own, t_u);
    if (why.empty()) why = hiding_missed_in_test(where + ", own test", codes, S.test_own, own, 0, t_omega + (requests - 1) * t_own, 0);
    if (!why.empty()) return why;
  }
  const std::vector<u64> link = lists_of(entry.chains[0].plan);
  if (link.size() != 1 + z.link.S.test().size()) return "main-rows: a link's chain has other lists than the columns and one for each stage";
  {
    const std::string why = hiding_missed_in_test("link of two tables", z.entry_codes, z.link.S.test(), link, 1, 0, 0);
    if (!why.empty()) return why;
  }
  // The table's own random tail, over the life of one committed table.
  const Code table{z.par.table_log_n, (size_t)z.par.table_data, (size_t)z.par.table_random};
  const u64 a_batch = lists_of(batch.chains[0].plan)[0], one_time = lists_of(one_time_test.chains[0].plan)[0];
  if (link[0] != z.link.S.columns_first + z.link.S.columns_further) return "table: a link draws other numbers of columns than its shape says";  // of both tables together
  // A leaf table's own LINK proof takes the place of a one-time test.
  const u64 first_copy = use.one_time_tests * one_time + use.links_on_the_first * z.link.S.columns_first, further_copy = use.links_on_a_copy * z.link.S.columns_further;
  const u64 need = std::max(first_copy, further_copy) + z.batch.batches * a_batch;
  if (hiding_positions(table) < need) return hiding_missed("table", "columns that a stored copy's random entries hide", hiding_positions(table), need);
  return "";
}

/* What can be tested on the parameter file alone, before any shape is known: the conditions on the
   table, with the largest matrix that the table's rows can hold and one pending claim. It does not
   replace `hiding_conditions_missed`. If `sized_table` rejects the parameters, the result begins
   "parameters-not-admitted: ". */
inline std::string hiding_conditions_missed_by_the_file(const Parameters &par, const TableUse &use = TableUse()) {
  const unsigned k_lo = par.block_log - par.table_rows_log;
  const u64 pads = 3 * (u64)par.answers;
  const Sized z = sized_table(par, (par.table_data - std::min(par.table_data, pads)) >> k_lo);
  if (!z.refused.empty()) return "parameters-not-admitted: " + z.refused;
  return hiding_conditions_missed_on_the_table(z, 1, use);
}

// The conditions on the committer's range proof. z: from sized(par, sub).
inline std::string hiding_conditions_missed_in_the_ranges(const Sized &z) {
  const Setting &s = z.d.set;
  const WfPublic &x = z.ranges.x;
  const FilePlan key = file_key(z, false);
  const std::vector<u64> lists = lists_of(key.chains[0].plan);
  if (lists.size() != x.stages.size()) return "main-rows: the one-time chain has other lists than one for each stage of the ranges' test";
  {
    const std::string why = hiding_missed_in_test("ranges", z.key_codes, x.stages, lists, 0, 0, 0);
    if (!why.empty()) return why;
  }
  /* Every scalar that a lookup's sumchecks send is masked by one entry. Two for each part, which
     are linear in the table's entries, take entries of the index-mask message (counted in
     `hiding_conditions_missed_on_the_table`); all others take entries of the range proof's own mask
     rows. Each such row gives up as many places as the pad has; all rows but one have a further
     free entry. */
  u64 entries = 0, parts = 0;
  for (const WfLookup &l : x.lookups) parts += l.parts.size();
  for (const PlanMessage &m : key.chains[0].plan->messages_)
    for (const WfLookup &l : x.lookups)
      if (m.kind == MessageKind::SCALARS && m.in_chain && !m.where.compare(0, 8 + l.name.size(), "ranges: " + l.name)) entries += m.count * (m.bytes / s.e_bytes);
  if (entries < 2 * parts) return "mask-room: the ranges' look-ups send fewer scalars than two for each part";
  entries -= 2 * parts;
  const u64 rows = x.message.shape.mask_rows, in_a_row = z.key_codes.mask[x.message.shape.mask_code].ell;
  const u64 room = rows && in_a_row > z.par.answers ? rows * (in_a_row - z.par.answers) + (rows - 1) : 0;
  if (room < entries) return hiding_missed("mask-room", "the ranges' multiplicities", room, entries);
  return "";
}

// The conditions on a request's segments and on the last proximity test; also `salt`, and `leaves`
// in LIN's three prover messages (batch and one-time test). z: from sized(par, sub).
inline std::string hiding_conditions_missed_in_the_links(const Sized &z, const Subproofs &sub) {
  const Setting &s = z.d.set;
  if (s.salt_bytes < 40) return hiding_missed("salt", "bytes of a leaf's salt", s.salt_bytes, 40);
  for (const OracleNotice &n : z.request.at_start)
    for (const ReqLink &l : z.request.link)
      for (const ReqMessage &m : l.messages)
        if (m.leaf_class == n.leaf_class) return "leaves: " + n.shape.name + " shares a leaf class with " + m.shape.name + ", which is read at another list";
  for (int once = 0; once < 2; ++once) {
    const std::string why = hiding_missed_in_the_tables_leaves(once ? "one-time test" : "batch", (once ? z.once : z.batch).S);
    if (!why.empty()) return why;
  }

  // ---- a request's first chain: one list for each segment, then the last test's ----
  {
    const FilePlan proof = file_proof(z, sub, false);
    const std::vector<u64> lists = lists_of(proof.chains[0].plan);
    const std::vector<ReqGroup> groups = req_groups(z.request);
    const size_t links = z.request.link.size();
    if (lists.size() != links + z.request.last.size()) return "main-rows: a request's first chain has other lists than one for each link and each stage of the last test";
    const Code &main = z.proof_codes.base[z.request.code], &mask = z.proof_codes.mask[z.request.mask_code];
    for (size_t l = 0; l < links; ++l) {
      const std::string &where = z.request.link[l].name;
      if (hiding_positions(main) < lists[l]) return hiding_missed("main-rows", where, hiding_positions(main), lists[l]);
      if (hiding_positions(mask) < lists[l]) return hiding_missed("mask-rows", where, hiding_positions(mask), lists[l]);
      if (mask.ell < 2 * (u64)groups[l].k + 1 - groups[l].h) return hiding_missed("fold-room", where, mask.ell, 2 * (u64)groups[l].k + 1 - groups[l].h);
      // Each scalar of a masked sumcheck is masked by one entry of the segment's mask row.
      Plan own(s);
      for (unsigned moment = 0; moment < z.request.link[l].moments(); ++moment) sub.plan(own, l, moment);
      u64 entries = 0;
      const Planned laid = own.finish();  // read for its messages alone
      for (const PlanMessage &m : laid->messages_)
        if (m.kind == MessageKind::SCALARS && m.in_chain) entries += m.count * (m.bytes / s.e_bytes);
      entries += z.par.answers;  // the mask row's own pad
      if (mask.ell < entries) return hiding_missed("mask-room", where, mask.ell, entries);
    }
    return hiding_missed_in_test("last test", z.proof_codes, z.request.last, lists, links, 0, 0);
  }
}

/* The complete test. z: from `sized`. checks: the range checks of one request.

   What an empty result means. The written arguments for hiding, which are not in this repository,
   assume public files that miss no condition; an empty result proves nothing by itself. The
   arguments are about observers who see no position of a table beyond those that proofs open: a
   column count that stops exactly at the limit leaves no margin for positions learned in any other
   way. The unopened columns are to be protected by their leaves' salts.

   What is not arithmetic on parameters. No function here tests the following conditions; tests of
   the code must.
     - Every entry of a random tail, mask entry and pad is exactly uniform in its field (in the
       extension field E: coordinate by coordinate): the sampler redraws a value that is not below
       the prime p = 2^64 - 2^32 + 1, and which values are redrawn depends on the seed alone. (The
       parser, blocks/frame/parameters.h, rejects a version word that names another field.)
     - A row's random tail is where blocks/code/code.h puts it: the lowest coefficients of a plain
       Reed-Solomon code. `hiding_positions` is about that place alone.
     - Nothing is ever encoded again without a fresh random tail. The mask of a proximity test's
       base case has the input's codes, so a random tail as long as the input's.
     - p is prime and E = F[X] / (X^3 - c) is a field (otherwise a nonzero factor can cancel a pad).
       Every multiplicity of a lookup fits its bits (a lookup's planner rejects other sizes). A pad
       is at its own place, which no claim but its sample's reads.
     - The random tails of different tables are independent. A request's chain is not started again
       by itself: each further own test is one more list on the batch's Omega, which is sized for as
       many own tests as a batch has slots.
     - A challenge that multiplies a mask is never 0.
     - A mask entry masks one sent scalar of one block. (It is read again in the claims that later
       steps prove.)
     - From the sending of a mask row to the last block that takes an entry from it, the verifier
       reads nothing of that row: every entry is used before the first proximity test on any group
       that contains its oracle message. A sumcheck's opening scalar is not sent before the message
       that fixes its weights.
     - The explicit instance of a session (the chains of one verdict) is the request, the next token
       and the record. Every claim's value is 0, a symbol that was read, a public function of
       record, challenges and sent values, or a value masked by an entry sent in the same session.
       Nothing but the tables' secrets passes from one run to the next.
     - A sample's linear form has weight 0 on pad places, a LINK proof's equality
       claim leaves the pads out, and pads of different tables are independent.
     - One secret makes one table: one matrix, one layout, one role (anchor or leaf table). The code
       that derives a table's private randomness reads configuration, shape and role, but nothing
       of the matrix; no program turns a leaf table into the anchor of a new commitment. (A table
       regenerated under its secret after the weights or the layout have changed reveals the
       unmasked encoding of the difference wherever both are read; its new root gives new seeds: two
       records on one set of pads.) A table regenerated from its secret is the same
       table, with the same running column count.
     - A claim on a weight has weight 0 on a row's pad places.
     - A run that is started again draws all its private randomness afresh. So a second run of a
       request's segment never reuses the index-mask message of an earlier run (two sent values
       under one mask entry give an unmasked linear functional of the table). Nothing that is sent
       reads an entry of that message unmasked.
     - Whatever builds a table takes its pads, its rows' random tails and its leaves' salts either
       all from that build's single secret or all from fresh randomness. A run's key contains fresh
       bytes from the operating system. No private random value's label recurs with other inputs.
     - The round at which a run stops is decided by what the verifier has seen and by randomness
       that does not depend on the table's pads and random tails. What a prover does when a search
       for a proof-of-work nonce reaches its cap depends on that fact alone. After a failed range
       check none of that request's proof is handed out; of its computed values its batch takes in only a salted digest.
     - Unused entries of a re-encoded row are zeros, or values that do not depend on a secret. */
inline std::string hiding_conditions_missed(const Sized &z, const Subproofs &sub, const std::vector<RangeCheck> &checks, const TableUse &use = TableUse()) {
  {
    std::string why = hiding_conditions_missed_in_the_links(z, sub);
    if (why.empty()) why = hiding_conditions_missed_in_the_ranges(z);
    if (why.empty()) why = hiding_conditions_missed_on_the_table(z, sub.pending_rows, use);
    if (!why.empty()) return why;
  }
  // ---- the range checks: all of one request's, times the requests under a commitment ----
  if (checks.empty()) return "range-checks: the sub-proofs name none";
  long least = (long)1 << 30;
  for (const RangeCheck &c : checks) {
    const long bits = range_check_bits(c);
    if (bits < 0) return "range-checks: a check fails too often, its digits do not represent its range exactly, or its numbers are out of range";
    least = std::min(least, bits);
  }
  /* The requests under a commitment: a batch's, times a table's batches, times the tables. A leaf
     table is tied to the anchor and to no other table, by a LINK proof that opens columns of the
     anchor: so the anchor's random tail bounds the number of leaf tables, whatever `TableUse` says
     has been begun so far. Each factor's logarithm is rounded up by itself. */
  const Code table{z.par.table_log_n, (size_t)z.par.table_data, (size_t)z.par.table_random};
  const u64 bears = hiding_positions(table), one_time = z.once.S.fold.columns;
  const u64 copies = 1 + (bears - std::min(bears, one_time)) / std::max<u64>(1, z.link.S.columns_first);
  const long have = least - (long)ceil_log2(checks.size()) - (long)ceil_log2(z.par.slots) - (long)ceil_log2(std::max<u64>(1, z.batch.batches)) - (long)ceil_log2(copies);
  if (have < (long)z.par.sec) return hiding_missed("range-checks", "bits of the bound on the chance that some honest range check under a key fails", have < 0 ? 0 : (u64)have, z.par.sec);
  return "";
}

}  // namespace blk
