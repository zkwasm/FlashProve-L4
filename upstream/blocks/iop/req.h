// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The frame of a request's proof (`Req` in names): a chain of segments (`link` in identifiers and
// messages), each closed into the next, and one proximity test (`iopp` in identifiers) at the end.
// In `cp_prove` and `cp_verify` it runs on the first of a proof's three Fiat-Shamir hash chains.
// This file fixes the order of the frame's messages, listed at ReqFrame, for the plan (req_plan()),
// the prover (ReqProver) and the verifier (ReqVerifier). A call that is not due ends the process.
//
// A segment is the part of the proof for one sub-proof (the argument for one part of the
// computation), with one group of oracle messages. An oracle message (`OracleShape` in relation.h)
// is a set of encoded rows, sent as a Merkle root; a group is read at one list of positions. The
// segment's oracle messages are sent at a few commit points (`moment`: numbered prover messages of
// the segment), and the block SEND takes an out-of-domain sample of each right after its prover
// message. The sub-proof ends in linear claims with public values (`Row`; "rows" with "targets" in
// names and messages) on the messages that these oracle messages encode.
// Closing a segment: the block BAT batches its claims into one, FOLD reduces that one to a claim on
// single rows, and the code switch CS re-encodes these rows as a new oracle message, the carrying
// oracle message, sent in the next segment's first prover message.
#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base.h"
#include "iop/bat.h"
#include "iop/context.h"
#include "iop/cs.h"
#include "iop/fold.h"
#include "iop/iopp.h"
#include "iop/send.h"
#include "numbers/blocks.h"

namespace blk {

// ---------------------------------------------------------------------------------------------------------------------
// The public configuration
// ---------------------------------------------------------------------------------------------------------------------

// One oracle message of a segment. Its `shape` gives the numbers of main rows and of mask rows
// (rows of random values). `leaf_class`: oracle messages of one prover message share a Merkle tree
// only within one class (lay_trees() in context.h).
// `life` > 0: later segments read its main rows, and segment l + life is the last that does (l: its
// own segment). In segments l to l + life - 1 they are kept rows of the group: FOLD leaves them as
// they are, and the code switch sends them again. In segment l + life they are ordinary rows.
struct ReqMessage {
  OracleShape shape;
  unsigned leaf_class = 0;
  unsigned moment = 0;  // the prover message of its segment in which it is sent; 0 is the first
  unsigned life = 0;    // 0: no later segment reads its main rows
};
// One segment.
struct ReqLink {
  std::string name;
  std::vector<ReqMessage> messages;  // by moment
  // The number of plain scalars: public values that this segment's claims depend on and that the
  // prover can name only now (such as a next token, called reply in `R_REPLY`, that it has just
  // computed), so they are not in the first query of the hash chain (the Fiat-Shamir transcript),
  // which fixes the statement before the prover speaks. They are sent unmasked right before the
  // segment's first prover message, in a round whose verifier's message is empty. So they are fixed
  // before the seed of that prover message's samples and before the positions of the previous
  // segment's group.
  // The frame does not check them: their form, and that they are what a client was given, is for
  // the sub-verifier to check.
  unsigned plain = 0;
  unsigned moments() const { return messages.empty() ? 0 : messages.back().moment + 1; }
};
// The frame's public configuration. ReqProver and ReqVerifier take it as `Public<ReqShape>`: a
// value that a Binder (blocks/chain/public.h) has written into the first query.
struct ReqShape {
  CodeId code = 0, mask_code = 0;  // of every segment's group
  std::vector<ReqLink> link;
  // Oracle messages, of mask rows only, that are sent in the very first prover message and belong
  // to no group of the frame. They are read at other lists of positions than the segments' own, so
  // their leaf classes differ.
  std::vector<OracleNotice> at_start;
  unsigned answers = 1;        // to the sample of a carrying oracle message
  std::vector<IoppStage> last;  // the stages of the last test; the first is the group of the last carrying oracle message
  IoppChoice choice;
};

// The shape's bytes in the first query hold each public fact once, except the number of answers of
// an `at_start` message, which public_fault(ReqShape) compares with the shape's. They leave out
// what the structs repeat for convenience: a `ReqMessage`'s two codes and number of answers, which
// are the shape's, and the last test's first stage, which is the group of the last carrying oracle
// message. public_fault(ReqShape) reports a copy that differs, and the Binder then ends the
// process. ReqFrame's constructor compares the number of answers with the plan's (`set.samples`).
inline void public_bytes(Writer &w, const ReqMessage &m) {
  for (u64 x : {(u64)m.shape.main_rows, (u64)(m.shape.main_rows ? m.shape.main_deg : 1), (u64)m.shape.mask_rows, (u64)m.leaf_class, (u64)m.moment, (u64)m.life}) w.put_u64(x);
}
inline void public_bytes(Writer &w, const ReqLink &l) { w.put_u64(l.plain), public_bytes(w, l.messages); }  // the name is for prints alone
inline void public_bytes(Writer &w, const ReqShape &s) {
  w.put_u64(s.code), w.put_u64(s.mask_code), public_bytes(w, s.link), public_bytes(w, s.at_start);
  w.put_u64(s.answers), public_bytes(w, std::vector<IoppStage>(s.last.begin() + (s.last.empty() ? 0 : 1), s.last.end())), public_bytes(w, s.choice);
}

// What follows from the configuration for one group. req_groups() returns the segments' groups,
// then the last test's.
// k, h: the main rows that are not kept fit in 2^k rows and the mask rows in 2^h, each oracle
// message's rows rounded up to a power of two; k >= h, k >= 1.
// The carrying oracle message has two main rows over the extension field E and no mask rows: the
// first holds the folded main message, the second everything else that a closing carries on.
struct ReqGroup {
  unsigned k = 0, h = 0;
  SwitchTarget target;         // of the switch into this group; not used for the first
  std::vector<std::pair<size_t, size_t>> born;  // target.kept[i]: first sent as (segment, index)
};

inline std::vector<ReqGroup> req_groups(const ReqShape &s) {
  struct Alive {
    OracleShape shape;
    unsigned leaf_class;  // a message that is sent again keeps its leaf class
    unsigned left;
    std::pair<size_t, size_t> born;
  };
  auto block = [](size_t rows) { return rows ? u64(1) << ceil_log2(rows) : 0; };
  std::vector<Alive> alive;
  std::vector<ReqGroup> groups(s.link.size() + 1);
  for (size_t l = 0; l < groups.size(); l++) {
    ReqGroup &G = groups[l];
    u64 main = 0, mask = 0;
    if (l > 0) {
      OracleShape &c = G.target.shape;
      c.name = "carried", c.code = s.code, c.main_rows = 2, c.main_deg = E_DEGREE, c.answers = s.answers;
      G.target.main = Cut{Part::MAIN, 0, 0};
      G.target.rest = Cut{Part::MAIN, 0, 1};
      main += 2;
    }
    std::vector<Alive> still;
    for (const Alive &a : alive) {
      KeptTarget k;
      k.shape = a.shape, k.leaf_class = a.leaf_class, k.kept_again = a.left > 1;
      G.target.kept.push_back(k);
      G.born.push_back(a.born);
      if (k.kept_again)
        still.push_back(Alive{a.shape, a.leaf_class, a.left - 1, a.born});
      else
        main += block(a.shape.main_rows);
    }
    alive = std::move(still);
    if (l < s.link.size())
      for (size_t i = 0; i < s.link[l].messages.size(); i++) {
        const ReqMessage &m = s.link[l].messages[i];
        if (m.life > 0)
          alive.push_back(Alive{m.shape, m.leaf_class, m.life, {l, i}});
        else
          main += block(m.shape.main_rows);
        mask += block(m.shape.mask_rows);
      }
    G.h = ceil_log2(mask);
    G.k = std::max(std::max(ceil_log2(main), G.h), 1u);
  }
  return groups;
}

// The sizes on which the choice of one distance for all segments' groups depends: the maximum of
// each number over all groups. Of the codes they need only the two message lengths, so they can be
// computed before the lengths of the codes' random tails (`r`) are known.
struct ReqLoad {
  unsigned k = 0;         // row variables of a group
  u64 slots = 1;          // 1 + kept rows, rounded up to a power of two
  unsigned cube_bits = 0; // variables of a sample's cube
  u64 rows = 0, rows_a_draw = 0;  // claims in a closing <= rows + rows_a_draw * (position draws)
};
// log_ell, log_ell_zk: log2 of the base code's and the mask code's message length.
// open_rows[l]: the number of claims that segment l's sub-proof leaves.
inline ReqLoad req_load(const ReqShape &s, unsigned log_ell, unsigned log_ell_zk, const std::vector<u64> &open_rows) {
  BLK_REQUIRE(open_rows.size() == s.link.size(), "request: one number of open rows for each link");
  auto cube = [&](const OracleShape &o) {
    const unsigned main = o.main_rows ? ceil_log2(o.main_rows) + log_ell : 0, mask = o.mask_rows ? ceil_log2(o.mask_rows) + log_ell_zk : 0;
    return o.main_rows && o.mask_rows ? std::max(main, mask) + 1 : main + mask;
  };
  const std::vector<ReqGroup> groups = req_groups(s);
  ReqLoad load;
  for (const OracleNotice &n : s.at_start) load.cube_bits = std::max(load.cube_bits, cube(n.shape));
  for (size_t l = 0; l < groups.size(); l++) {
    u64 rows = l < s.link.size() ? open_rows[l] : 0, kept_before = 0, kept_here = 0;
    if (l > 0) rows += 1 + s.answers, load.cube_bits = std::max(load.cube_bits, cube(groups[l].target.shape));
    for (const KeptTarget &k : groups[l].target.kept) rows += 2 * (u64)k.shape.answers, kept_before += k.shape.main_rows;  // its old sample rows and its new ones
    if (l < s.link.size())
      for (const ReqMessage &m : s.link[l].messages) rows += m.shape.answers, load.cube_bits = std::max(load.cube_bits, cube(m.shape));
    if (l + 1 < groups.size())
      for (const KeptTarget &k : groups[l + 1].target.kept) kept_here += k.shape.main_rows;
    load.k = std::max(load.k, groups[l].k);
    load.slots = std::max(load.slots, u64(1) << ceil_log2(1 + kept_here));
    load.rows = std::max(load.rows, rows);
    if (l > 0) load.rows_a_draw = std::max(load.rows_a_draw, 3 + kept_before);
  }
  return load;
}

// public_fault() checks what the shape shows by itself; the Binder calls it on whatever it binds.
// req_fault() also checks the shape against the codes; ReqFrame's constructor calls it. Both return
// null or a description of the fault and never end the process, so numbers from a file can be
// checked first.
constexpr size_t REQ_MOST_ROWS = size_t(1) << 20;  // of an oracle message; far above any use, and sums of them stay far below 2^63
inline const char *public_fault(const ReqMessage &m) {
  if (m.shape.main_rows == 0 && m.shape.mask_rows == 0) return "an oracle message of the link has no rows";
  if (m.shape.main_rows > REQ_MOST_ROWS || m.shape.mask_rows > REQ_MOST_ROWS) return "an oracle message of the link has more than 2^20 rows";
  if (m.shape.main_rows && m.shape.main_deg != 1 && m.shape.main_deg != E_DEGREE) return "main rows of the link are neither over F nor over E";
  if (const char *fault = public_fault(OracleNotice{m.shape, m.leaf_class})) return fault;
  if (m.life > 0 && (m.shape.main_rows == 0 || m.shape.mask_rows > 0)) return "a kept oracle message has no main rows, or has mask rows";
  // The argument that ties a kept message to its later copies (blocks/iop/cs.h, above
  // switch_rows()) does not need the next rule. It is a rule of the design: what a later segment
  // reads is sent, sampled and answered before the sub-proof of the segment that sends it gets its
  // first challenge. The frame has been tested in this form only.
  if (m.life > 0 && m.moment != 0) return "a kept oracle message does not go in its link's first prover message";
  return nullptr;
}
inline const char *public_fault(const ReqLink &l) {
  if (l.messages.empty() || l.messages[0].moment != 0) return "the link has no oracle message at its first moment";
  unsigned moment = 0;
  for (const ReqMessage &m : l.messages) {
    if (m.moment != moment && m.moment != ++moment) return "the link's oracle messages do not come by moment, or a moment is empty";
    if (const char *fault = public_fault(m)) return fault;
  }
  return nullptr;
}
// `link`: set to the segment with the fault, counted from 0 (the number of segments if in none).
inline const char *req_shape_fault(const ReqShape &s, size_t *link = nullptr) {
  size_t unused;
  size_t &at = link ? *link : unused;
  at = s.link.size();
  if (s.link.empty()) return "the request has no link";
  // The soundness argument needs this check and the comparison of every message's number of answers
  // below. The error of a sub-proof's round that tests what the segment's oracle messages encode
  // has no factor for the list size (the number of codewords within the distance), because each of
  // these oracle messages has its sample answered. SEND samples only a message whose shape has
  // answers, and their number is a line of the parameter file: these two checks, and the file's own
  // range (blocks/frame/parameters.h), keep a file from switching a sample off. After them comes
  // one more stop: Statement::join() (blocks/iop/relation.h), which both sides run, ends the
  // process if an unsampled oracle message joins a group. (Observed on the prover's side only, with
  // the second check removed.)
  if (s.answers < 1) return "no answer to a sample: a carrying oracle message would not be sampled";
  unsigned most_class = 0;
  for (size_t l = 0; l < s.link.size(); l++) {
    at = l;
    if (const char *fault = public_fault(s.link[l])) return fault;
    for (const ReqMessage &m : s.link[l].messages) {
      if (m.shape.answers != s.answers) return "an oracle message of the link has another number of answers to its sample than the request";
      if (m.shape.main_rows && m.shape.code != s.code) return "main rows of the link are under another code than the group's";
      if (m.shape.mask_rows && m.shape.mask_code != s.mask_code) return "mask rows of the link are under another code than the group's mask code";
      if (l + m.life >= s.link.size()) return "an oracle message is kept beyond the last link";
      most_class = std::max(most_class, m.leaf_class);
    }
    // A closing folds main rows that are not kept. From the second segment on the carrying oracle
    // message has such rows; the first segment's group has only its own.
    if (l == 0 && std::none_of(s.link[0].messages.begin(), s.link[0].messages.end(), [](const ReqMessage &m) { return m.shape.main_rows > 0 && m.life == 0; }))
      return "the first link has no main row that its closing could fold";
  }
  at = s.link.size();
  for (const OracleNotice &n : s.at_start) {  // shares no leaf with the segments' oracle messages
    if (n.shape.main_rows || !n.shape.mask_rows || n.shape.mask_rows > REQ_MOST_ROWS) return "what rides at the start is not an oracle message of mask rows alone";
    if (const char *fault = public_fault(n)) return fault;
    if (n.shape.answers != s.answers) return "what rides at the start has another number of answers than the request";
    if (n.leaf_class <= most_class) return "what rides at the start is in a leaf class of the links' oracle messages";
  }
  const std::vector<ReqGroup> groups = req_groups(s);
  for (size_t l = 0; l + 1 < groups.size(); l++)
    if (groups[l].k > CODE_MOST_LOG_N) return at = l, "the link's group has too many rows";
  if (s.last.size() < 2) return "a proximity test has fewer than two stages";
  if (const char *fault = public_fault(s.last)) return fault;
  if (const char *fault = public_fault(s.choice)) return fault;
  const IoppStage &first = s.last[0];
  if (first.code != s.code || first.mask_code != s.mask_code || first.k != groups.back().k || first.h != groups.back().h)
    return "the last test's first stage is not the group of the last carrying oracle message";
  return nullptr;
}
inline const char *public_fault(const ReqShape &s) { return req_shape_fault(s); }

inline const char *req_fault(const Codes &codes, const ReqShape &s, size_t *link = nullptr) {
  if (const char *fault = req_shape_fault(s, link)) return fault;
  size_t unused;
  size_t &at = link ? *link : unused;
  at = s.link.size();
  if (s.code >= codes.base.size() || s.mask_code >= codes.mask.size()) return "the links' group names a code that is not there";
  const Code &c = codes.base[s.code], &zk = codes.mask[s.mask_code];
  if (s.answers >= c.ell || s.answers >= zk.ell) return "more answers to a sample than a row has places";
  for (const OracleNotice &n : s.at_start) {
    if (n.shape.mask_code >= codes.mask.size()) return "what rides at the start names a mask code that is not there";
    if (n.shape.answers >= codes.mask[n.shape.mask_code].ell) return "what rides at the start has more answers than a row has places";
  }
  const std::vector<ReqGroup> groups = req_groups(s);
  for (size_t l = 0; l + 1 < groups.size(); l++) {
    at = l;
    if (zk.ell < fold_mask_columns(groups[l].k, groups[l].h)) return "the mask rows are too short for the hiding polynomial of the link's closing";
    u64 carried = 2 * (u64)zk.ell + c.r + 2 * (u64)zk.r + s.answers;
    for (const KeptTarget &k : groups[l + 1].target.kept)
      carried += (u64)k.shape.main_rows * c.r + (u64)k.shape.answers * (k.shape.main_deg == 1 ? E_DEGREE : 1);
    if (carried > c.ell) return "what the link's closing carries on does not fit one row";
  }
  at = s.link.size();
  return iopp_fault(codes, s.last, s.answers);
}

// ---------------------------------------------------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------------------------------------------------

// What the plan needs beyond the configuration.
struct ReqSizes {
  GroupSize group;          // the segments' group: codes, rule, m; `req_plan` sets k, h, slots
  std::vector<Stage> last;  // the solved stages of the last test
  std::vector<u64> rows;    // for each segment: the number of claims that its sub-proof leaves
  // The group in which the `at_start` oracle messages are tested, which is none of the frame's: a
  // sample's error grows with the list size of its oracle message's group. Not read if there are
  // none; otherwise it must have a code.
  GroupSize at_start;
};
// Appends the rounds of a segment's sub-proof that follow the SEND of one of its commit points.
using ReqSubPlan = std::function<void(Plan &, size_t link, unsigned moment)>;

// Appends every round and message of the frame to the plan `p` (`Plan` in blocks/numbers/plan.h:
// the list of all rounds, each with its sizes and error), in the order in which ReqProver and
// ReqVerifier run them. Not checked here: that req_fault(codes, s) is null; the caller must have
// checked it.
inline void req_plan(Plan &p, const Codes &codes, const ReqShape &s, const ReqSizes &z, const ReqSubPlan &sub) {
  BLK_REQUIRE(z.rows.size() == s.link.size() && p.set.samples == s.answers, "request: the sizes do not fit the configuration");
  // The round errors are computed from z (to "price" a round is to compute its error), the
  // messages' sizes from `codes` and s: both must describe the same codes. Compared here, for the
  // segments' two codes and each stage's in the last test (iopp_sized_fault()): log_n, ell and r,
  // the codes' numbers. Distances and lists are stored nowhere: the planners reckon them from z's
  // groups. That is, analysis() (blocks/numbers/blocks.h) computes a group's distance and list size
  // from its codes, its radius rule (`rule`) and its parameter `m` wherever a round's error is
  // computed. Compared at run time: the run's codes with the plan's (require_group(),
  // require_codes() in blocks/chain/rounds.h).
  // z.at_start is compared with nothing, here or at run time: the caller is trusted.
  BLK_REQUIRE(z.group.main.log_n == codes.base[s.code].log_n && z.group.main.ell == codes.base[s.code].ell && z.group.main.r == codes.base[s.code].r &&
                  z.group.mask.log_n == codes.mask[s.mask_code].log_n && z.group.mask.ell == codes.mask[s.mask_code].ell && z.group.mask.r == codes.mask[s.mask_code].r,
              "request: the links' group is sized for other codes than the configuration names");
  BLK_REQUIRE(!z.last.empty(), "request: the last test is sized for no stage");
  const char *const last_fault = iopp_sized_fault(codes, s.last, z.last);
  BLK_REQUIRE(!last_fault, last_fault ? last_fault : "");
  const std::vector<ReqGroup> groups = req_groups(s);
  BLK_REQUIRE(s.at_start.empty() || z.at_start.main.ell, "request: something rides at the start, and the sizes do not say with which list its sample is priced");
  // `group`: the one that these oracle messages join.
  // `riders`: the last `riders` entries of `sent` only ride: they are priced with their own group's
  // list size. (To ride is to join no group of the frame: these are the `at_start` messages, and
  // their samples' errors use the list size of z.at_start.)
  // own_round: a round of SEND's own closes the answers; otherwise the positions' round of the
  // code switch does.
  auto send = [&](const std::string &where, const std::vector<OracleNotice> &sent, const GroupSize &group, size_t riders, bool own_round) {
    std::vector<SampleSize> sampled;
    for (size_t i = 0; i < sent.size(); i++)
      if (sent[i].shape.answers > 0) sampled.push_back(SampleSize{Send::cube(codes, sent[i].shape).bits, {i + riders < sent.size() ? group : z.at_start}});
    plan_send(p, where, sampled, (unsigned)tree_sizes(codes, sent).size(), own_round);
  };
  // A sub-proof's planner may change the plan's scope (`Scope` in blocks/numbers/setting.h: how
  // often a round's proof of work is paid: per request, per batch or once); after each call to it
  // the frame's scope is restored.
  const Scope scope = p.scope();
  auto sub_proof = [&](size_t l, unsigned moment) { sub(p, l, moment), p.begin_scope(scope); };
  GroupSize g = z.group;      // of the segment that was closed last
  std::vector<TreeSize> trees;  // that hold the present group's rows
  std::string before;
  u64 carried_rows = 0;  // the rows that the last switch leaves on the present group
  for (size_t l = 0; l < groups.size(); l++) {
    const bool is_link = l < s.link.size();
    const std::string where = is_link ? s.link[l].name : "last group";
    std::vector<OracleNotice> sent;
    if (l > 0) sent.push_back(OracleNotice{groups[l].target.shape, groups[l].target.leaf_class});
    for (const KeptTarget &k : groups[l].target.kept) sent.push_back(OracleNotice{k.shape, k.leaf_class});
    if (is_link)
      for (const ReqMessage &m : s.link[l].messages)
        if (m.moment == 0) sent.push_back(OracleNotice{m.shape, m.leaf_class});
    sent.push_back(fold_mask_notice(iopp_fold_shape(IoppStage{s.code, s.mask_code, groups[l].k, groups[l].h}, GroupLayout())));
    const std::vector<TreeSize> first_trees = tree_sizes(codes, sent);  // without `at_start`
    u64 sample_rows = 0;  // of the group's members
    for (const OracleNotice &n : sent) sample_rows += n.shape.answers;
    if (l == 0) sent.insert(sent.end(), s.at_start.begin(), s.at_start.end());
    if (is_link && s.link[l].plain) plan_scalars(p, "plain scalars", where, s.link[l].plain), plan_none(p, R_REPLY, where);
    send(where, sent, is_link ? z.group : z.last.front().g, l == 0 ? s.at_start.size() : 0, l == 0);
    if (l > 0) {
      const u64 t = plan_loc(p, before, g, trees);
      size_t kept_rows = 0, old_samples = 0;
      for (const KeptTarget &k : groups[l].target.kept) kept_rows += k.shape.main_rows, old_samples += k.shape.answers;
      carried_rows = 1 + old_samples + t * (3 + kept_rows);  // the carried claim, the kept messages' old sample rows, the generator rows
    }
    trees = first_trees;
    if (!is_link) {
      plan_iopp(p, where, z.last, trees, carried_rows + sample_rows, 0, s.choice.masks_ride);
      break;
    }
    sub_proof(l, 0);
    for (unsigned moment = 1; moment < s.link[l].moments(); moment++) {
      sent.clear();
      for (const ReqMessage &m : s.link[l].messages)
        if (m.moment == moment) sent.push_back(OracleNotice{m.shape, m.leaf_class});
      for (const OracleNotice &n : sent) sample_rows += n.shape.answers;
      send(where, sent, z.group, 0, true);
      for (const TreeSize &t : tree_sizes(codes, sent)) trees.push_back(t);
      sub_proof(l, moment);
    }
    size_t kept_rows = 0;  // of this group: they are folded side by side with each main row
    for (const KeptTarget &k : groups[l + 1].target.kept) kept_rows += k.shape.main_rows;
    g = z.group, g.name = where, g.k = groups[l].k, g.h = groups[l].h, g.slots = u64(1) << ceil_log2(1 + kept_rows);
    const u64 rows = z.rows[l] + carried_rows + sample_rows;
    plan_bat(p, where, {g}, rows);
    plan_fold(p, where, g, 0);
    before = where;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// What both sides keep
// ---------------------------------------------------------------------------------------------------------------------

// The calls, the same on both sides, and their messages, for segments l = 0 .. L - 1. A `|`
// separates a prover's message from a verifier's:
//   first()   if the segment has plain scalars: these | an empty verifier's message;
//             one prover message: the carrying oracle message of segment l - 1; the kept oracle
//             messages, sent again; the segment's oracle messages of commit point 0, its mask rows
//             among them; the fresh mask of the segment's own closing; if l = 0, `at_start`
//             | seed | answers | positions of segment l - 1's group      (for l = 0: SEND only)
//   later()   for each further commit point, within the sub-proof: its oracle messages
//             | seed | answers
//   close()   BAT and FOLD on the segment's group
//   finish()  one prover message: the carrying oracle message of the last segment; the fresh mask
//             of the last test's first FOLD
//             | seed | answers | positions of the last segment's group
//             | the whole proximity test, on that carrying oracle message only
// So a group's positions are drawn only after the oracle message that re-encodes its fold has been
// sent, sampled and answered. The next segment's first oracle messages share that prover message,
// its seed and, by leaf class, its trees.
// Between these calls the sub-proof uses the same context, takes mask entries from the segment's
// mask rows and adds its claims to group(); the frame knows nothing else of it. Everything public
// that enters a claim, its value above all, must come from the hash chain's first query, from what
// the prover has sent, or from the verifier's challenges. Nothing here checks that.
class ReqFrame {
 public:
  const ReqShape &shape() const { return shape_; }
  GroupId group() const { return group_; }  // of the present segment
  size_t link() const { return link_; }     // the present segment, counted from 0
  // The kept oracle messages of earlier segments as sent again in the present one, in the order of
  // sending.
  const std::vector<OracleId> &kept_before() const { return kept_before_; }
  const std::vector<OracleId> &at_start() const { return at_start_; }
  // The present segment's plain scalars (ReqLink::plain), until the next first().
  const std::vector<E> &plain() const { return plain_; }
  // The oracle message that was sent as number `message` of segment `born`, as the present segment
  // has it: itself, or the one that took its rows at the last code switch. The process ends if
  // the present segment cannot read it or its commit point has not come.
  OracleId where(size_t born, size_t message) const {
    BLK_REQUIRE(next_ == Next::LATER_OR_CLOSE, "request: no link is under way: its first prover message has not gone, or it is closed");
    if (born == link_) {
      BLK_REQUIRE(message < own_.size(), "request: this oracle message of the link has not been sent yet");
      return own_[message];
    }
    const std::vector<std::pair<size_t, size_t>> &list = groups_[link_].born;
    for (size_t i = 0; i < list.size(); i++)
      if (list[i] == std::make_pair(born, message)) return kept_before_[i];
    BLK_REQUIRE(false, "request: the present link does not read this oracle message");
    return NONE_ID;
  }

 protected:
  // `samples`: the number of answers to a sample in the chain's plan (`set.samples`).
  ReqFrame(const Codes &codes, const ReqShape &shape, unsigned samples) : shape_(shape), groups_(req_groups(shape)) {
    const char *fault = req_fault(codes, shape);
    BLK_REQUIRE(!fault, fault ? fault : "");
    BLK_REQUIRE(shape.answers == samples, "request: the shape has another number of answers to a sample than the plan's setting");
  }
  enum class Next { FIRST, LATER_OR_CLOSE, DONE };
  FoldShape fold_shape(size_t l, GroupLayout layout) const {
    return iopp_fold_shape(IoppStage{shape_.code, shape_.mask_code, groups_[l].k, groups_[l].h}, std::move(layout));
  }
  std::vector<const ReqMessage *> at_moment(unsigned moment) const {
    std::vector<const ReqMessage *> m;
    for (const ReqMessage &x : shape_.link[link_].messages)
      if (x.moment == moment) m.push_back(&x);
    return m;
  }
  // Adds one commit point's oracle messages to the segment's group, those with `life` > 0 as kept.
  void join(Statement &S, const std::vector<const ReqMessage *> &sent, const std::vector<OracleId> &ids) {
    for (size_t i = 0; i < sent.size(); i++) S.join(group_, ids[i], sent[i]->life > 0), own_.push_back(ids[i]);
  }
  void begin_link(Statement &S) {
    group_ = S.new_group(link_ < shape_.link.size() ? shape_.link[link_].name : "last group", shape_.code, shape_.mask_code);
    moment_ = 1;
    own_.clear();
  }

  const ReqShape shape_;
  const std::vector<ReqGroup> groups_;
  Next next_ = Next::FIRST;
  size_t link_ = 0;
  unsigned moment_ = 0;  // the next commit point of the present segment
  GroupId group_ = NONE_ID;
  std::vector<OracleId> kept_before_, at_start_, own_;
  std::vector<E> plain_;
  FoldMask mask_;    // of the present segment's closing
  SingleRows rows_;  // what the last closing has left
};

// ---------------------------------------------------------------------------------------------------------------------
// The prover
// ---------------------------------------------------------------------------------------------------------------------

class ReqProver : public ReqFrame {
 public:
  ReqProver(ProverContext &P, const Public<ReqShape> &shape) : ReqFrame(P.st.codes(), P.ch.get(shape), P.ch.plan()->set.samples), P_(P) {}

  // Sends the next segment's first prover message; returns the ids of the segment's oracle messages
  // of commit point 0.
  // `main`: the main tables of these oracle messages, in their order (NO_ID for one without main
  // rows). The entries reserved for a pad (the random element that masks a sample's answer) must be
  // zero: SEND draws every pad and ends the process if one of these entries is not zero.
  // mask_row: the segment's mask rows, if the caller has drawn them itself: uniform and fresh (not
  // checked), the pad's entries zero. They go to the first of these oracle messages that has mask
  // rows; the mask rows of the others are drawn here. drafts() draws a table for that first one
  // too, and `mask_row` replaces it. Anyone who computes the prover's private random values
  // (`coins` in identifiers) again must count this unused draw: each draw takes the next stream
  // (blocks/coins/coins.h). at_start: with the first segment only.
  // `plain`: the segment's plain scalars (ReqLink::plain).
  std::vector<OracleId> first(const std::vector<TableId> &main, std::vector<OracleDraft> at_start = {}, Table mask_row = Table(),
                              std::vector<E> plain = {}) {
    BLK_REQUIRE(next_ == Next::FIRST && link_ < shape_.link.size(), "request: a link's first prover message is not due");
    BLK_REQUIRE(at_start.size() == (link_ == 0 ? shape_.at_start.size() : 0), "request: what rides at the start is fixed");
    BLK_REQUIRE(plain.size() == shape_.link[link_].plain, "request: the number of a link's plain scalars is fixed");
    const std::vector<const ReqMessage *> own = at_moment(0);
    std::vector<OracleDraft> with = drafts(own, main);
    if (mask_row.rows)
      for (OracleDraft &d : with)
        if (d.shape.mask_rows) {
          BLK_REQUIRE(mask_row.rows == d.mask.rows && mask_row.len == d.mask.len && mask_row.deg == d.mask.deg, "request: the mask row has another shape");
          d.mask = std::move(mask_row);
          break;
        }
    mask_ = fold_mask_make(fold_shape(link_, GroupLayout()), P_);
    with.push_back(fold_mask_draft(P_.st.codes(), fold_shape(link_, GroupLayout()), mask_));
    for (OracleDraft &d : at_start) with.push_back(std::move(d));
    begin_link(P_.st);
    for (const E &x : plain) P_.ch.send(x);
    if (!plain.empty()) P_.ch.close_empty(R_REPLY);
    plain_ = std::move(plain);
    std::vector<OracleId> ids;
    if (link_ == 0) {
      ids = send_prove(P_, std::move(with));
    } else {
      const Switched s = cs_sigma_prove(P_, rows_, witness_, groups_[link_].target, group_, std::move(with));
      let_go();
      kept_before_ = s.kept;
      ids = s.with;
    }
    mask_.sent_as(ids[own.size()]);
    at_start_.insert(at_start_.end(), ids.begin() + own.size() + 1, ids.end());
    ids.resize(own.size());
    join(P_.st, own, ids);
    next_ = Next::LATER_OR_CLOSE;
    return ids;
  }

  // Sends the oracle messages of the segment's next commit point. `main`: as for first().
  std::vector<OracleId> later(const std::vector<TableId> &main) {
    BLK_REQUIRE(next_ == Next::LATER_OR_CLOSE && moment_ < shape_.link[link_].moments(), "request: the link has no further moment");
    const std::vector<const ReqMessage *> own = at_moment(moment_++);
    const std::vector<OracleId> ids = send_prove(P_, drafts(own, main));
    join(P_.st, own, ids);
    return ids;
  }

  // Closes the segment, once the sub-proof is over and all its claims are in group().
  // targets_known = false: some added claim has an arbitrary value, because the caller has not
  // evaluated its own equation on the committed tables (a sub-proof whose kernels never form the
  // claims' constants). FOLD then takes its running claim from its own first round
  // (FoldShape::target_from_the_rounds in fold.h), which spares that evaluation and changes nothing
  // that is sent. The verifier computes the values itself from what was sent, so a claim that does
  // not hold makes the verifier reject by the end of the last proximity test (finish()), except
  // with the errors that the plan lists for the rounds from this closing to that end.
  void close(bool targets_known = true) {
    BLK_REQUIRE(next_ == Next::LATER_OR_CLOSE && moment_ == shape_.link[link_].moments(), "request: moments of the link are missing");
    RunInput in = P_.st.begin_run(group_);  // first: from here on nothing can be added to the group
    BLK_REQUIRE(in.layout.k == groups_[link_].k && in.layout.h == groups_[link_].h, "request: the group is not the planned one");
    FoldShape F = fold_shape(link_, in.layout);
    // The next FOLD meets this claim again, so the engine (blocks/engine/engine.h: it stores the
    // prover's main tables) keeps its coefficient vector, and the prover stores it in no other
    // form (`EngineTerm` in blocks/iop/tensor.h): the structured form, which only a verifier needs,
    // would grow with every segment.
    F.keep_folded_coefficients = F.folded_coefficients_alone = true;
    F.target_from_the_rounds = !targets_known;
    const E theta = P_.ch.close_algebraic(R_BAT).draw();
    P_.ch.require_rows(in.rows().size());
    fold_prove(P_, F, Bat::combine(P_.st, group_, std::move(in), theta), std::move(mask_), rows_, witness_);
    forget_coefficients();  // of the previous segment: this fold was the last to read them
    coefficients_ = fold_kept_coefficients(rows_.c);
    mask_ = FoldMask();
    link_++;
    next_ = Next::FIRST;
  }

  // After a close(), for tests and a prover's self-checks: whether the claim that the closing has
  // left holds on the prover's tables. No byte of a proof depends on the prover's copy of that
  // claim: a round's coefficients are computed from tables and masks only.
  bool carried_claim_holds() {
    BLK_REQUIRE(next_ == Next::FIRST && link_ > 0, "request: no closing has left a claim");
    return single_rows_hold(P_, rows_, witness_);
  }

  // After the last segment's close().
  void finish() {
    BLK_REQUIRE(next_ == Next::FIRST && link_ == shape_.link.size(), "request: links are missing");
    const FoldShape F = fold_shape(link_, GroupLayout());
    mask_ = fold_mask_make(F, P_);
    std::vector<OracleDraft> with;
    with.push_back(fold_mask_draft(P_.st.codes(), F, mask_));
    begin_link(P_.st);
    const Switched s = cs_sigma_prove(P_, rows_, witness_, groups_[link_].target, group_, std::move(with));
    let_go();
    mask_.sent_as(s.with[0]);
    iopp_prove(P_, group_, shape_.last, shape_.choice, std::move(mask_));
    forget_coefficients();
    release_oracles(P_, {s.g, s.with[0]});
    next_ = Next::DONE;
  }

 private:
  // Releases the previous segment's tables and trees, once its group has been opened at its one
  // list of positions, so that the prover's memory need not hold the whole request's.
  void let_go() {
    P_.engine.drop(witness_.main);
    release_oracles(P_, single_rows_oracles(rows_));
  }
  void forget_coefficients() {
    if (coefficients_ != NO_ID) P_.engine.drop(coefficients_);
    coefficients_ = NO_ID;
  }
  std::vector<OracleDraft> drafts(const std::vector<const ReqMessage *> &own, const std::vector<TableId> &main) {
    BLK_REQUIRE(main.size() == own.size(), "request: one main table for each oracle message of the moment");
    std::vector<OracleDraft> d(own.size());
    for (size_t i = 0; i < own.size(); i++) {
      d[i].shape = own[i]->shape, d[i].leaf_class = own[i]->leaf_class, d[i].main = main[i];
      if (d[i].shape.mask_rows) d[i].mask = Send::fresh_mask(P_.st.codes(), d[i].shape, P_.coins);
    }
    return d;
  }
  ProverContext &P_;
  SingleRowsWitness witness_;
  TableId coefficients_ = NO_ID;  // the last closing's folded coefficient vector, an engine table
};

// ---------------------------------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------------------------------

// The same calls in the same order. Each returns false if it rejects the proof; first() and later()
// set `ids` to what the prover's calls return.
// A true finish() says nothing about bytes that are left over in the proof: the caller ends with
// the channel's finish(). Nor about the `at_start` oracle messages, which belong to no group of the
// frame, or about a group that a sub-proof has made itself: testing those is the caller's task.
// The contract for callers (in short: plan, codes and shape come from one object of sizes) is
// stated at run_verifier() in blocks/chain/closed.h.
class ReqVerifier : public ReqFrame {
 public:
  ReqVerifier(VerifierContext &V, const Public<ReqShape> &shape) : ReqFrame(V.st.codes(), V.ch.get(shape), V.ch.plan()->set.samples), V_(V) {}

  bool first(std::vector<OracleId> &ids) {
    BLK_REQUIRE(next_ == Next::FIRST && link_ < shape_.link.size(), "request: a link's first prover message is not due");
    const std::vector<const ReqMessage *> own = at_moment(0);
    std::vector<OracleNotice> with = notices(own);
    with.push_back(fold_mask_notice(fold_shape(link_, GroupLayout())));
    if (link_ == 0) with.insert(with.end(), shape_.at_start.begin(), shape_.at_start.end());
    begin_link(V_.st);
    plain_.assign(shape_.link[link_].plain, E::zero());
    for (E &x : plain_)
      if (!V_.ch.recv(x)) return false;
    if (!plain_.empty()) V_.ch.close_empty(R_REPLY);
    if (link_ == 0) {
      if (!send_verify(V_, with, ids)) return false;
    } else {
      Switched s;
      if (!cs_sigma_verify(V_, rows_, groups_[link_].target, group_, with, s)) return false;
      kept_before_ = s.kept;
      ids = s.with;
    }
    mask_ = FoldMask();
    mask_.sent_as(ids[own.size()]);
    at_start_.insert(at_start_.end(), ids.begin() + own.size() + 1, ids.end());
    ids.resize(own.size());
    join(V_.st, own, ids);
    next_ = Next::LATER_OR_CLOSE;
    return true;
  }

  bool later(std::vector<OracleId> &ids) {
    BLK_REQUIRE(next_ == Next::LATER_OR_CLOSE && moment_ < shape_.link[link_].moments(), "request: the link has no further moment");
    const std::vector<const ReqMessage *> own = at_moment(moment_++);
    if (!send_verify(V_, notices(own), ids)) return false;
    join(V_.st, own, ids);
    return true;
  }

  bool close() {
    BLK_REQUIRE(next_ == Next::LATER_OR_CLOSE && moment_ == shape_.link[link_].moments(), "request: moments of the link are missing");
    RunInput in = V_.st.begin_run(group_);
    BLK_REQUIRE(in.layout.k == groups_[link_].k && in.layout.h == groups_[link_].h, "request: the group is not the planned one");
    const FoldShape F = fold_shape(link_, in.layout);
    const E theta = V_.ch.close_algebraic(R_BAT).draw();
    V_.ch.require_rows(in.rows().size());
    GroupRow row = Bat::combine(V_.st, group_, std::move(in), theta);
    V_.note("REQ: the group row's target", row.target);  // for comparing verifiers; changes nothing
    if (!fold_verify(V_, F, std::move(row), std::move(mask_), rows_)) return false;
    V_.note("REQ: FOLD's target", rows_.target);
    mask_ = FoldMask();
    link_++;
    next_ = Next::FIRST;
    return V_.ch.ok();
  }

  bool finish() {
    BLK_REQUIRE(next_ == Next::FIRST && link_ == shape_.link.size(), "request: links are missing");
    begin_link(V_.st);
    Switched s;
    if (!cs_sigma_verify(V_, rows_, groups_[link_].target, group_, {fold_mask_notice(fold_shape(link_, GroupLayout()))}, s)) return false;
    mask_ = FoldMask();
    mask_.sent_as(s.with[0]);
    next_ = Next::DONE;
    return iopp_verify(V_, group_, shape_.last, shape_.choice, std::move(mask_));
  }

 private:
  static std::vector<OracleNotice> notices(const std::vector<const ReqMessage *> &own) {
    std::vector<OracleNotice> n;
    for (const ReqMessage *m : own) n.push_back(OracleNotice{m->shape, m->leaf_class});
    return n;
  }
  VerifierContext &V_;
};

// ---------------------------------------------------------------------------------------------------------------------
// A whole request
// ---------------------------------------------------------------------------------------------------------------------

// Only tests call req_prove() and req_verify(). `cp_prove` and `cp_verify` make the same calls on
// ReqProver and ReqVerifier one by one, in ChainRun (prover/prover/chain_run.cuh) and
// mb::ChainCheck (verifier/src/vf_request_chain.h).
// The sub-proofs as req_prove() and req_verify() call them: run() follows the SEND of each commit
// point of each segment; ReqSubVerifier::run() returns false to reject the proof. What a sub-proof
// may do there: see ReqFrame.
class ReqSubProver {
 public:
  virtual ~ReqSubProver() = default;
  // The main tables of a commit point's oracle messages, as ReqProver::first() takes them.
  virtual std::vector<TableId> tables(size_t link, unsigned moment) = 0;
  virtual std::vector<OracleDraft> at_start() { return {}; }
  // The segment's mask rows, if the sub-proof draws them itself: uniform and fresh, the pad's
  // entries zero (ReqProver::first()). An empty table: the frame draws them.
  virtual Table mask_row(size_t) { return Table(); }
  virtual std::vector<E> plain(size_t) { return {}; }
  // False: the segment's claims come without their true values (ReqProver::close()).
  virtual bool targets_known(size_t) { return true; }
  // `sent`: the oracle messages of this commit point.
  virtual void run(ProverContext &P, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &sent) = 0;
};
class ReqSubVerifier {
 public:
  virtual ~ReqSubVerifier() = default;
  virtual bool run(VerifierContext &V, const ReqFrame &frame, unsigned moment, const std::vector<OracleId> &sent) = 0;
};

// behind_close: null, or a test's callback, called after each segment's close().
inline void req_prove(ProverContext &P, const Public<ReqShape> &bound, ReqSubProver &sub, const std::function<void(ReqProver &)> &behind_close = nullptr) {
  ReqProver frame(P, bound);
  const ReqShape &shape = frame.shape();
  for (size_t l = 0; l < shape.link.size(); l++) {
    sub.run(P, frame, 0, frame.first(sub.tables(l, 0), l == 0 ? sub.at_start() : std::vector<OracleDraft>(), sub.mask_row(l), sub.plain(l)));
    for (unsigned moment = 1; moment < shape.link[l].moments(); moment++) sub.run(P, frame, moment, frame.later(sub.tables(l, moment)));
    frame.close(sub.targets_known(l));
    if (behind_close) behind_close(frame);
  }
  frame.finish();
}

// For tests: `cp_verify` does not call it (see above ReqSubProver). What a true result does not
// cover, and the contract for callers: see ReqVerifier.
[[nodiscard]] inline bool req_verify(VerifierContext &V, const Public<ReqShape> &bound, ReqSubVerifier &sub) {
  ReqVerifier frame(V, bound);
  const ReqShape &shape = frame.shape();
  std::vector<OracleId> sent;
  for (size_t l = 0; l < shape.link.size(); l++) {
    if (!frame.first(sent) || !sub.run(V, frame, 0, sent)) return false;
    for (unsigned moment = 1; moment < shape.link[l].moments(); moment++)
      if (!frame.later(sent) || !sub.run(V, frame, moment, sent)) return false;
    if (!frame.close()) return false;
  }
  return frame.finish();
}

}  // namespace blk
