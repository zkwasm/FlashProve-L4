// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Estimates the GPU memory that one request's own proof takes in the GPU engine (EngineGpu,
// prover/engine/engine_gpu.cuh; `card` in names: GPU), from the request's public shape, before
// anything runs. cp_prove only prints the estimate (Prover::chain_make()). A segment (`link` in
// identifiers and printed lines) is one sub-proof's part of the request's proof. A step of
// card_plan() has three sums, in bytes:
//   tables  main tables: the unencoded main rows of oracle messages (rows sent as Merkle roots).
//           EngineGpu::open() computes columns from the rows, so they stay until their segment is
//           opened, after the next segment's first prover message;
//   trees   what the segment's trees keep until then: digests, salts, random tails, mask rows;
//   work    the step's work space.
// Not counted: the encoder's transform tables (8 n bytes per code length n, per engine); the work
// space of the fold and of the opened columns (prover/engine/fold.cuh, prover/engine/columns.cuh,
// prover/engine/products.cuh); what the sub-proofs' kernels hold; the last proximity test beyond
// its first message; anything about the weights. Only card_tree_bytes() is checked against the
// engine, by a test that is not part of this repository. Host code only: no CUDA header is needed.
#pragma once
#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "iop/req.h"

namespace blk {
// nodes: see card_tree_bytes(). salts: see card_salts(). work: if true, the engine keeps the
// encoder's buffers after their first use, also while the request waits; if false, they return to
// the device's memory pool at each step's end.
struct CardKeep { // TreeKeep's choices, for code that includes no CUDA header
  bool nodes = true, salts = true, work = true;
};
// Must equal TREE_LOWER_LEVELS; a test that is not part of this repository has the static_assert.
constexpr unsigned CARD_LOWER_LEVELS = 10; // engine/tree.cuh: TREE_LOWER_LEVELS
// The bytes of a tree's kept digests and salts: the value of the static EngineGpu::tree_bytes().
// Without k.nodes, a tree of more than 2^10 leaves drops the nine levels just above its leaves.
inline size_t card_tree_bytes(const OracleTreeShape &s, const CardKeep &k) {
  const TreeShape t = s.tree();
  const unsigned a = t.log_n;
  size_t digests = (size_t)2 << a;
  if (!k.nodes && a > CARD_LOWER_LEVELS) digests = ((size_t)1 << a) + ((size_t)2 << (a - CARD_LOWER_LEVELS));
  return DIGEST_BYTES * digests + (k.salts ? s.salt_bytes << s.log_n : 0);
}
// A tree's salts: work space of its commitment if not kept. The engine drops them only if commit()
// gets CardStreamCoins (prover/engine/stream_coins.cuh); otherwise the estimate is too low.
inline size_t card_salts(const OracleTreeShape &s) { return s.salt_bytes << s.log_n; }
// Work space to commit to a tree: 16 codewords; the columns' chaining values, if a leaf's hash
// input exceeds 16 words; the mask rows' codewords, if the tree has main rows too.
inline size_t card_commit_work(const OracleTreeShape &s) {
  const size_t n = (size_t)1 << s.log_n;
  const bool slabs = leaf_bytes(s.width(), s.salt_bytes) > 16 * F_BYTES;
  return 16 * F_BYTES * n + (slabs ? 4 * HashFn::CVW * n : 0) + (s.has_main() && s.has_mask() ? (s.mask_width() * F_BYTES) << s.log_n_zk : 0);
}
// An opening computes dropped levels again: work space of 2^10 digests for each opened position.
inline size_t card_open_work(const OracleTreeShape &s, const CardKeep &k, u64 positions) {
  return !k.nodes && s.tree().log_n > CARD_LOWER_LEVELS ? (DIGEST_BYTES << CARD_LOWER_LEVELS) * positions : 0;
}
// One step of card_plan(), which follows blocks/iop/req.h's order of messages. The non-static
// EngineGpu::tree_bytes() is the engine's count of `trees`.
struct CardStep {
  std::string what;
  size_t tables = 0, trees = 0, work = 0;
  size_t all() const { return tables + trees + work; }
};
struct CardPlan {
  std::vector<CardStep> steps;
  size_t peak = 0, peak_at = 0;     // the greatest all(), and its step
  size_t closed_most = 0;           // the greatest sum just after a segment is closed
  void add(const CardStep &s) {
    if (s.all() > peak) peak = s.all(), peak_at = steps.size();
    steps.push_back(s);
  }
};
// table_bytes(link, m, shape) returns the bytes of the main table of s.link[link].messages[m], as
// the prover stores it in GPU memory; shape is that message's. Cells may be narrower than a field
// element there: CardTables::table_bytes() in prover/engine/lay.cuh gives the count for its main
// tables, and F_BYTES per cell overstates a request by far. Counted in full field elements: the
// message that re-encodes the previous segment's fold, kept rows (sent again for later segments)
// and s.at_start. positions: codeword positions opened per segment.
using CardTableBytes = std::function<size_t(size_t, size_t, const OracleShape &)>;
inline CardPlan card_plan(const Codes &codes, const ReqShape &s, const CardKeep &keep, size_t salt_bytes, u64 positions, const CardTableBytes &table_bytes) {
  const std::vector<ReqGroup> groups = req_groups(s);
  CardPlan plan;
  struct Held {
    size_t tables = 0, trees = 0;
    std::vector<OracleTreeShape> shapes;
  } before, now;
  size_t idle = 0; // work space that the engine keeps between steps: 0 without keep.work
  // own[i]: the index of sent[i] in s.link[link].messages, or NOT_OWN.
  constexpr size_t NOT_OWN = ~(size_t)0;
  auto send = [&](const std::string &where, size_t link, const std::vector<OracleNotice> &sent, const std::vector<size_t> &own) {
    for (size_t i = 0; i < sent.size(); i++) {
      const OracleShape &o = sent[i].shape;
      if (o.main_rows) now.tables += own[i] != NOT_OWN ? table_bytes(link, own[i], o) : o.main_rows * o.main_deg * codes.base[o.code].ell * F_BYTES;
      if (o.main_rows) now.trees += o.main_rows * o.main_deg * codes.base[o.code].r * F_BYTES;
      if (o.mask_rows) now.trees += o.mask_rows * E_BYTES * (codes.mask[o.mask_code].ell + codes.mask[o.mask_code].r);
    }
    for (const OracleTreeShape &t : lay_trees(codes, sent, salt_bytes).shapes) {
      now.trees += card_tree_bytes(t, keep);
      now.shapes.push_back(t);
      plan.add(CardStep{where + ": a tree of 2^" + std::to_string(t.has_main() ? t.log_n : t.log_n_zk) + " leaves of " +
                            std::to_string(t.has_main() ? t.main_width() : t.mask_width()) + " symbols is committed",
                        before.tables + now.tables, before.trees + now.trees,
                        std::max(idle, card_commit_work(t)) + (keep.salts ? 0 : card_salts(t))});
      if (keep.work) idle = std::max(idle, card_commit_work(t) - (t.has_main() && t.has_mask() ? DIGEST_BYTES << t.log_n_zk : 0));
    }
  };
  for (size_t l = 0; l < groups.size(); l++) {
    const bool is_link = l < s.link.size();
    const std::string where = is_link ? s.link[l].name : "last group";
    std::vector<OracleNotice> sent;
    if (l > 0) sent.push_back(OracleNotice{groups[l].target.shape, groups[l].target.leaf_class});
    for (const KeptTarget &k : groups[l].target.kept) sent.push_back(OracleNotice{k.shape, k.leaf_class});
    std::vector<size_t> own(sent.size(), NOT_OWN);
    if (is_link)
      for (size_t m = 0; m < s.link[l].messages.size(); m++)
        if (s.link[l].messages[m].moment == 0) sent.push_back(OracleNotice{s.link[l].messages[m].shape, s.link[l].messages[m].leaf_class}), own.push_back(m);
    sent.push_back(fold_mask_notice(iopp_fold_shape(IoppStage{s.code, s.mask_code, groups[l].k, groups[l].h}, GroupLayout())));
    if (l == 0) sent.insert(sent.end(), s.at_start.begin(), s.at_start.end());
    own.resize(sent.size(), NOT_OWN);
    send(where, l, sent, own);
    if (l > 0) {
      size_t work = 0;
      for (const OracleTreeShape &t : before.shapes) work += card_open_work(t, keep, positions);
      plan.add(CardStep{where + ": the link before is opened", before.tables + now.tables, before.trees + now.trees, idle + work});
      before = Held();
    }
    if (!is_link) break; // the last test's later stages have smaller main tables: not counted
    for (unsigned moment = 1; moment < s.link[l].moments(); moment++) {
      sent.clear(), own.clear();
      for (size_t m = 0; m < s.link[l].messages.size(); m++)
        if (s.link[l].messages[m].moment == moment) sent.push_back(OracleNotice{s.link[l].messages[m].shape, s.link[l].messages[m].leaf_class}), own.push_back(m);
      send(where, l, sent, own);
    }
    plan.add(CardStep{where + ": closed", now.tables, now.trees, idle});
    plan.closed_most = std::max(plan.closed_most, now.tables + now.trees + idle);
    before = now, now = Held();
  }
  return plan;
}

// The same, for a callback that needs only the segment and the message's shape.
inline CardPlan card_plan(const Codes &codes, const ReqShape &s, const CardKeep &keep, size_t salt_bytes, u64 positions,
                          const std::function<size_t(size_t, const OracleShape &)> &table_bytes) {
  return card_plan(codes, s, keep, salt_bytes, positions, CardTableBytes([&](size_t link, size_t, const OracleShape &o) { return table_bytes(link, o); }));
}
// Prints the peak's step, closed_most, and an estimate for in_step requests proved together: one
// at its peak, the others, waiting for their turns, at closed_most. Not checked: p has a step,
// in_step > 0.
inline void card_plan_print(std::FILE *to, const CardPlan &p, unsigned in_step) {
  const CardStep &top = p.steps[p.peak_at];
  std::fprintf(to, "GPU memory of a request's own proof through the engine, reckoned from its shape (GB = 10^9 bytes):\n");
  std::fprintf(to, "  peak %.2f GB at \"%s\": tables %.2f, trees %.2f, work space %.2f\n", p.peak / 1e9, top.what.c_str(), top.tables / 1e9, top.trees / 1e9,
               top.work / 1e9);
  std::fprintf(to, "  after a link is closed, at most %.2f GB\n", p.closed_most / 1e9);
  std::fprintf(to, "  %u requests in step, one at its peak and the others closed: %.2f GB\n", in_step, (p.peak + (in_step - 1) * (double)p.closed_most) / 1e9);
}

} // namespace blk
