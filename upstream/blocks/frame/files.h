// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Plans of three files that a verifier reads: the key file, a leaf file (`entry` in identifiers)
   and a proof file. A file starts with a header (`head`) and has one or more hash chains
   (Fiat-Shamir transcripts). The key file and a leaf file have a table's root (`plan_table_name`)
   before the first chain. After the first chain, a proof file has the request's place in its batch,
   and the key file a place in a binding tree of one leaf (see `FilePlan`). Each chain has its own
   `Plan` (blocks/numbers/plan.h: its rounds, messages, sizes and error bounds), which its prover
   and verifier go through as they run. The order of the protocol blocks in a chain is defined here
   and described in blocks/frame/FORMAT.md.

   No file contains a length: a file's bytes are its plan's messages in order. What enters a hash
   chain has a fixed length. The answer to a draw of positions varies in length, because equal
   positions are answered once and Merkle paths are pruned.

   `link` has two senses in identifiers: a segment (`ReqLink`), one sub-proof's part of a request's
   first chain, and the LINK proof (`LinkSized`), which ties a leaf table (a further table of the
   same entries) to the anchor (a commitment's first table). */
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "frame/derive.h"
#include "frame/lin_frame.h"
#include "frame/link_frame.h"
#include "frame/wf_frame.h"
#include "iop/member.h"
#include "iop/req.h"

namespace blk {

/* The caller's data on a request's sub-proofs (one for each part of the forward pass) and on the
   committer's range proof (WF; `wf` in identifiers). A sub-proof leaves linear claims (`rows` in
   identifiers): open claims on the oracle messages (encoded rows, sent as Merkle roots) of its
   segment's group, and pending claims on the committed table (`index` in identifiers), for LIN,
   the test on that table (blocks/frame/lin_frame.h). */
struct Subproofs {
  /* The segments with their oracle messages (shape, commit point `moment`, leaf class, `life`).
     `sized` sets the shape's own `code`, `mask_code`, `answers`, `last` and `choice`, and appends
     to `at_start` the index-mask message: the request's masks for its pending claims. The caller
     sets each message's own `code`, `mask_code` (both 0) and `answers` (the parameter file's);
     `req_fault` checks them. */
  ReqShape request;
  std::vector<u64> open_rows;               // for each segment: the number of its open claims
  ReqSubPlan plan;                          // appends a sub-proof's rounds after one commit point
  u64 pending_rows = 1;                     // the number of a request's pending claims
  std::function<void(Plan &)> after_links;  // may be empty: rounds after the segments' last test
  /* What the committer's range proof looks up, entry by entry (blocks/frame/wf_frame.h; from
     a layout: blocks/frame/wf_units.h). `ranges.matrix_rows`, the number of rows of the matrix
     that the committed table stores, is LIN's source too. `ranges.k_col` (the matrix has 2^k_col
     columns) is not read: `sized` takes the last number of the parameter file's line `table`. */
  WfShape ranges;
};

/* A placeholder for planning without a model's sub-proofs (blocks/frame/cp_frame.cpp, tests). Its
   3,076,096 matrix rows are from the public shape of Llama 4 Maverick (no weight); of its other
   numbers nothing is claimed. 25 segments, each with 77 rows over the field F at commit point 0, 7
   read by the next segment; one row over the extension E, the packed cells' partial evaluations; 31
   rows of digits and multiplicities. `sized` rejects it unless the file's `answers` is 1. */
inline Subproofs stand_in_subproofs() {
  Subproofs s;
  auto message = [](const char *name, size_t rows, unsigned deg, size_t mask_rows, unsigned moment, unsigned life) {
    ReqMessage m;
    m.shape.name = name, m.shape.code = 0, m.shape.main_rows = rows, m.shape.main_deg = deg, m.shape.mask_rows = mask_rows, m.shape.mask_code = 0;
    m.moment = moment, m.life = life;
    return m;
  };
  for (unsigned l = 0; l < 25; ++l) {
    ReqLink link;
    link.name = "link " + std::to_string(l + 1);
    link.messages = {message("cells", 63, 1, 1, 0, 0), message("multiplicities", 7, 1, 0, 0, 0), message("cells that the next link reads", 7, 1, 0, 0, l + 1 < 25 ? 1 : 0),
                     message("reads", 1, 3, 0, 1, 0), message("digits", 27, 1, 0, 2, 0), message("digits' multiplicities", 4, 1, 0, 2, 0)};
    link.plain = l + 1 == 25 ? 1 : 0;  // the next token: one unmasked scalar, in the last segment
    s.request.link.push_back(link);
  }
  s.open_rows.assign(25, (u64)1 << 13);
  {  // 3,076,096 rows of bytes; 1,527,800 pairs of them, in one run, are the two digits of weights
    s.ranges.matrix_rows = 3076096;
    WfShapeLookup bytes, weights;
    bytes.lookup.name = "bytes", bytes.lookup.d = 8, bytes.lookup.digit_weight = {F::one()}, bytes.lookup.offset = from_u64(128);
    bytes.lookup.first_bound = {2, 1, 0, 5, 4, 3, 8, 7, 6}, bytes.log_rows_apart = 1;
    weights.lookup.name = "weights", weights.lookup.d = 13, weights.lookup.digit_weight = {F::one(), from_u64(256)};
    weights.lookup.offset = from_u64(4096), weights.lookup.first_bound = bytes.lookup.first_bound;
    weights.entries = {WfRun{0, 0, 1527800, 1527800}};
    bytes.entries = {weights.entries[0], WfRun{0, 3055600, 10248, 10248}};
    s.ranges.lookups = {bytes, weights};
  }
  s.plan = [](Plan &p, size_t, unsigned moment) {
    if (moment == 0) {
      for (int i = 0; i < 2; ++i) {
        plan_shift(p, "look-up", (u64)1 << 21);
        plan_lkp(p, "look-up", 28);
        plan_product_tree(p, "look-up, keys", 28, 0);
        plan_product_tree(p, "look-up, table", 16, 5);
      }
      for (int i = 0; i < 4; ++i) {
        PscSize x;
        x.degree.assign(24, 4), x.degree_Y = 3, x.values = 3;
        plan_zero(p, "relations", 24);
        plan_psc(p, "relations", x);
      }
    } else if (moment == 1) {
      plan_rng(p, "smallness", (u64)1 << 16);
    } else {
      plan_lkp(p, "digits' look-up", 26);
      plan_product_tree(p, "digits' look-up", 26, 0);
      plan_batt(p, "rows on the table", 14, 22);
      plan_scalars(p, "scalars", "stand-in", 560);
      plan_bat_sigma(p, "smallness", (u64)1 << 14);
    }
  };
  s.pending_rows = 13;
  return s;
}

// What derive() needs from the sub-proofs' shapes. Of the segments' two codes only the message
// lengths are used: derive() has yet to fix their random tails. 4096 is assumed, not checked, to
// be at least the number of positions drawn on a segment's group, which derive() fixes too.
// req_plan() uses the true number, and a plan that misses its target cannot be run.
inline FrameShape frame_shape_of(const Parameters &par, const Subproofs &sub) {
  ReqShape shape = sub.request;
  shape.answers = par.answers;
  const ReqLoad load = req_load(shape, par.link.log_ell, par.link.mask_log_ell, sub.open_rows);
  FrameShape f;
  f.link_rows_log = load.k, f.link_slots = load.slots, f.link_cube_bits = load.cube_bits;
  f.link_batched_rows = load.rows + load.rows_a_draw * 4096;  // an assumed bound: see above
  f.last_rows_log = req_groups(shape).back().k;
  for (const ReqLink &l : shape.link) f.link_messages = std::max<unsigned>(f.link_messages, (unsigned)(2 * l.messages.size() + 2));
  f.index_rows = sub.ranges.matrix_rows;
  return f;
}

// How the committed table's columns are hashed: a tree leaf contains one symbol of every row.
inline TreeSize table_tree_of(const Setting &s, const Parameters &par) {
  TreeSize t;
  t.name = "table", t.log_n = par.table_log_n, t.leaf_bytes = ((u64)s.f_bytes) << par.table_rows_log;
  return t;
}

struct ChainPlan {
  std::string name;
  Planned plan;
};
// `has_place` (proof file, key file): after the first chain comes the place in the binding tree,
// a Merkle tree over the seals of a batch's first chains (seal: the digest that ends such a chain).
struct FilePlan {
  std::string name;
  unsigned head_words = 0;  // of 4 bytes: tag, version and, in a proof file, the leaf table number
  std::vector<ChainPlan> chains;
  unsigned place_depth = 0;  // the binding tree's depth
  bool has_place = false;
};
// scope: the chain is made once per request, batch or commitment; its rounds' proof of work
// depends on it.
// compact: equal rounds may be planned once, with a count; such a plan gives errors and sizes but
// cannot be run.
inline Plan new_chain(const Setting &s, Scope scope, bool compact) {
  Plan p(s);
  p.compact = compact;
  p.begin_scope(scope);
  return p;
}
inline void add_chain(FilePlan &f, const std::string &name, const Plan &p) { f.chains.push_back(ChainPlan{name, p.finish()}); }

/* What the three files' plans are made from: the parameter file's lines, what is derived from
   them, and each file's codes by number. If `refused` is not empty, it says why sizing failed, and
   not all fields are filled. */
struct Sized {
  Parameters par;
  Derived d;
  Codes proof_codes, key_codes, entry_codes;
  ReqShape request;      // `Subproofs::request`, completed by `sized`
  LinSized batch, once;  // LIN for a batch, and as the committer's one-time test
  WfSized ranges;        // the committer's proof that the table's entries are in their ranges
  LinkSized link;
  std::string refused;
};
inline Sized sized(const Parameters &par, const Subproofs &sub) {
  Sized z;
  z.par = par;
  const FrameShape shape = frame_shape_of(par, sub);
  z.d = derive(par, shape);
  if (!(z.refused = z.d.refused).empty()) return z;
  const Setting &s = z.d.set;
  z.request = sub.request;
  z.request.last = iopp_stages(z.d.last.stages, z.proof_codes);  // adds the segments' two codes first: they get number 0
  z.request.code = z.request.mask_code = 0, z.request.answers = par.answers;
  z.request.choice.masks_ride = true;
  {
    LinLines lines = lin_lines(par, shape, false);
    /* No parameter line gives the index-mask message's number of mask rows. It follows from the
       mask entries used: one for each pending claim and, if there are several, those of BATT, which
       batches them into one. */
    lines.input_mask_rows = Send::mask_rows_for(sub.pending_rows + (sub.pending_rows > 1 ? batt_masks(par.block_log) : 0), (size_t)1 << par.own.mask_log_ell, par.answers);
    if (!lines.input_mask_rows) return z.refused = "own: no number of mask rows holds the entries that a request's rows on the table take", z;
    z.batch = lin_sized(s, lines, z.d.table, z.proof_codes);
  }
  if (!(z.refused = z.batch.refused).empty()) return z;
  /* The index-mask message is sent in the request's first prover message. It is opened at the
     positions of the second chain's proximity test (`iopp` in identifiers) and of no segment's, so
     it gets its own leaf class and thus its own Merkle tree. */
  OracleNotice masks = z.batch.input;
  masks.leaf_class = 0;
  for (const ReqLink &l : z.request.link)
    for (const ReqMessage &m : l.messages) masks.leaf_class = std::max(masks.leaf_class, m.leaf_class + 1);
  z.request.at_start.push_back(masks);
  if (const char *fault = req_fault(z.proof_codes, z.request)) return z.refused = fault, z;
  {
    WfShape ranges = sub.ranges;
    ranges.k_col = par.block_log;
    const WfLines of_the_ranges = wf_lines(ranges, par.once_ranges, par.ranges_part);
    WfPublic look_ups;  // of its fields, the count below depends only on `lookups` and their parts
    look_ups.lookups = of_the_ranges.lookups;
    LinLines lines = lin_lines(par, shape, true);
    lines.input_mask_rows = Send::mask_rows_for(wf_index_mask_entries(look_ups, WfMatrix{of_the_ranges.k_col, of_the_ranges.matrix_rows}), (size_t)1 << par.once_own.mask_log_ell, par.answers);
    if (!lines.input_mask_rows) return z.refused = "once-own: no number of mask rows holds the entries that the ranges and the joining take", z;
    z.once = lin_sized(s, lines, z.d.table, z.key_codes);
    if (!(z.refused = z.once.refused).empty()) return z;
    z.ranges = wf_sized(s, of_the_ranges, z.key_codes);
    if (!(z.refused = z.ranges.refused).empty()) return z;
  }
  z.link = link_sized(s, link_lines(par), z.d.table, z.entry_codes);
  z.refused = z.link.refused;
  return z;
}

/* `sized` without a request's shape: LIN for a batch and as the one-time test, and the LINK proof.
   For checking a parameter file before the sub-proofs' shapes are known. index_rows: the committed
   matrix's number of rows. `request` and `ranges` stay empty, each index-mask message is taken to
   have one mask row, and the code numbers are not those of a proof file's chains. */
inline Sized sized_table(const Parameters &par, u64 index_rows) {
  Sized z;
  z.par = par, z.d.set = setting_of(par);
  derive_table(par, z.d);
  if (!(z.refused = z.d.refused).empty()) return z;
  FrameShape shape;
  shape.index_rows = index_rows;
  z.batch = lin_sized(z.d.set, lin_lines(par, shape, false), z.d.table, z.proof_codes);
  if (!(z.refused = z.batch.refused).empty()) return z;
  z.once = lin_sized(z.d.set, lin_lines(par, shape, true), z.d.table, z.key_codes);
  if (!(z.refused = z.once.refused).empty()) return z;
  z.link = link_sized(z.d.set, link_lines(par), z.d.table, z.entry_codes);
  z.refused = z.link.refused;
  return z;
}

// LIN's shared chain and second chain, after the chain that leaves LIN's claim.
// x: LIN for a batch, or as the committer's one-time test.
// In this file only the shared chain starts without a salt.
inline void add_table_chains(FilePlan &f, const Setting &s, const Parameters &par, const Codes &codes, const LinSized &x, bool compact) {
  {
    Plan p = new_chain(s, x.lines.one_time ? Scope::ONCE : Scope::SHARED, compact);
    plan_chain_start(p, "shared chain", false);
    lin_plan_shared(p, codes, x, table_tree_of(s, par));
    plan_seal(p, "shared chain");
    add_chain(f, "shared chain", p);
  }
  {
    Plan p = new_chain(s, x.lines.one_time ? Scope::ONCE : Scope::REQUEST, compact);
    plan_chain_start(p, "second chain", true);
    lin_plan_second(p, codes, x, tree_sizes(codes, {x.input}));
    add_chain(f, "second chain", p);
  }
}

// If z was rejected (`z.refused` is not empty), gives f a rejected plan for each named chain and
// returns true. Such a plan does not meet its target and cannot be run.
inline bool refused_file(FilePlan &f, const Sized &z, std::initializer_list<const char *> chains) {
  if (z.refused.empty()) return false;
  for (const char *name : chains) {
    Plan p = new_chain(z.d.set, Scope::REQUEST, true);
    p.refuse(z.refused);
    add_chain(f, name, p);
  }
  return true;
}

// A proof file: header | first chain (the request's own) | its place in the batch |
// shared chain (one for the whole batch) | second chain (the request's own again).
// sub must be the `Subproofs` that z was sized from; file_proof() does not compare them.
// req_plan() (blocks/iop/req.h) orders the segments' rounds and messages.
inline FilePlan file_proof(const Sized &z, const Subproofs &sub, bool compact) {
  const Setting &s = z.d.set;
  FilePlan f;
  f.name = "proof", f.head_words = 3;
  if (refused_file(f, z, {"first chain", "shared chain", "second chain"})) return f;
  {
    Plan p = new_chain(s, Scope::REQUEST, compact);
    plan_chain_start(p, "first chain", true);
    req_plan(p, z.proof_codes, z.request, ReqSizes{z.d.link.group, z.d.last.stages, sub.open_rows, z.batch.own.stages[0].g}, sub.plan);  // `at_start` is tested in LIN's `own` group
    if (sub.after_links) sub.after_links(p), p.begin_scope(Scope::REQUEST);  // it may have changed the scope
    plan_member_rows(p, "rows on the table", sub.pending_rows, z.batch.S.k_col());  // BATT, if several: the pending claims into one
    lin_plan_first(p, z.proof_codes, z.batch);
    plan_seal(p, "first chain");
    add_chain(f, "first chain", p);
  }
  add_table_chains(f, s, z.par, z.proof_codes, z.batch, compact);
  f.has_place = true, f.place_depth = ceil_log2(z.par.slots);
  return f;
}

/* A committed table's root and its record (stored answers to samples of its rows). They are items
   of the file before the chain, not messages of the chain: a reader
   takes them first and puts them, as an `IndexName` (blocks/iop/foldeq.h), into the chain's first
   query, the hash input before the prover's first message. */
inline void plan_table_name(Plan &p, const std::string &where, const TableSize &T) {
  plan_file_item(p, "root", where, MessageKind::ROOT, p.set.digest_bytes, p.set.digest_bytes, 0);
  plan_record(p, where, T);
}

// The key file: header | the anchor's root and record | one-time chain: index-mask message, range
// proof (WF), LIN's first message (the committer's vector W) | place, in a binding tree of one
// leaf | shared chain and second chain: the one-time test, LIN for a batch of one.
inline FilePlan file_key(const Sized &z, bool compact) {
  const Setting &s = z.d.set;
  FilePlan f;
  f.name = "key", f.head_words = 2;
  if (refused_file(f, z, {"one-time chain", "shared chain", "second chain"})) return f;
  {
    Plan p = new_chain(s, Scope::ONCE, compact);
    plan_table_name(p, "table", z.once.table);
    plan_chain_start(p, "one-time chain", true);
    plan_send(p, "masks of the committer's rows on the table", {SampleSize{Send::cube(z.key_codes, z.once.input.shape).bits, {z.once.own.stages[0].g}}}, 1);
    wf_plan(p, z.key_codes, z.ranges);  // with BATT on the pending claims that it leaves
    lin_plan_first(p, z.key_codes, z.once);
    plan_seal(p, "one-time chain");
    add_chain(f, "one-time chain", p);
  }
  add_table_chains(f, s, z.par, z.key_codes, z.once, compact);
  f.has_place = true, f.place_depth = 0;  // the binding tree of a batch of one
  return f;
}

// A leaf file: header | the leaf table's root and record | one chain: the LINK proof
// (blocks/frame/link_frame.h), with the proximity test on the two folded rows.
inline FilePlan file_entry(const Sized &z, bool compact) {
  const Setting &s = z.d.set;
  FilePlan f;
  f.name = "entry", f.head_words = 2;
  if (refused_file(f, z, {"link's chain"})) return f;
  Plan p = new_chain(s, Scope::ONCE, compact);
  plan_table_name(p, "further table", z.link.table);
  plan_chain_start(p, "link's chain", true);
  link_plan(p, z.entry_codes, z.link, table_tree_of(s, z.par));
  add_chain(f, "link's chain", p);
  return f;
}

// Joins several files into one plan, for the error accounting, the lengths and the list of fields.
// A chain that was rejected, or whose errors were computed under a `Setting` other than s, causes
// the joined plan to be rejected.
inline Planned merged(const Setting &s, const std::vector<FilePlan> &files) {
  Plan p(s, files.empty() ? "proof" : files[0].name);
  for (const FilePlan &f : files) {
    p.begin_part(f.name);
    plan_file_item(p, "head", "", MessageKind::HEAD, 4 * f.head_words, 4.0 * f.head_words, 0);
    for (size_t i = 0; i < f.chains.size(); ++i) {
      plan_chain(p, f.chains[i].plan);
      if (i || !f.has_place) continue;
      // With the first chain's seal, the next three items give the binding tree's root, which is in
      // the shared chain's first query.
      plan_file_item(p, "place in the batch", "", MessageKind::HEAD, 8, 8, 0);
      plan_file_item(p, "salt of the place's leaf", "", MessageKind::SALTS, s.salt_bytes, s.salt_bytes, 1);
      plan_file_item(p, "path in the batch", "", MessageKind::PATHS, (u64)f.place_depth * s.digest_bytes, (double)f.place_depth * s.digest_bytes, f.place_depth);
    }
  }
  return p.finish();
}

}  // namespace blk
