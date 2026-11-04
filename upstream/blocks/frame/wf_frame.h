// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Derives the sizes of WF, the committer's range proof on the committed matrix (blocks/iop/wf.h):
// the parts and planes of each lookup, the shape of WF's oracle message (encoded rows, sent as
// Merkle roots), the codes and stages of its proximity test (`iopp` in identifiers), and its rounds
// in the plan of the key file's one-time chain. Call wf_lines(), wf_sized(), wf_plan() in turn;
// check `WfSized::refused` before wf_plan().
// Terms: a lookup shows that every key occurs in a public table, here 0 .. 2^d - 1. A key is a
// constant plus a weighted sum of digits, entries of one matrix column. A place gives one key for
// each column and reads one row for each digit; a part is 2^log_places places. Planes are the bits
// of a multiplicity (how often a table row occurs among the keys). Pending claims are linear claims
// on the matrix that WF leaves to its caller; their mask entries are in the index-mask message,
// which the caller sends before WF.
#pragma once
#include <string>
#include <vector>

#include "frame/derive.h"
#include "frame/lin_frame.h"
#include "iop/wf.h"

namespace blk {

// One lookup before it is cut into parts. An entry is a set of rows of the matrix, `stride` apart.
// If they are the digits of one key, an entry takes one place. If each is a key by itself
// (log_rows_apart > 0), each takes a place, and a part has all rows of the same entries, so a
// prover that unpacks an entry gets all its keys at once. A `WfRun` in `entries` is `places`
// entries with consecutive first rows, from `first_row` on.
struct WfShapeLookup {
  WfLookup lookup;              // `parts` is not read; `mu`, `live` must be 0: `wf_lines` sets them
  std::vector<WfRun> entries;   // all entries, in order; `first_place` is not read
  unsigned log_rows_apart = 0;  // if > 0: an entry has 2^log_rows_apart rows, each a one-digit key
};
// What WF looks up; wf_shape_of_layout() (blocks/frame/wf_units.h) makes it from a model's layout.
// The matrix has matrix_rows rows and 2^k_col columns.
struct WfShape {
  unsigned k_col = 0;
  u64 matrix_rows = 0;
  std::vector<WfShapeLookup> lookups;
};

// Cuts a list of entries into parts of 2^log_places places. A part has
// 2^(log_places - log_rows_apart) consecutive entries; a run that crosses a part's end is split
// there; the last part may be partly padding (places that read no row). Row q of a part's entry i
// is at place q * 2^(log_places - log_rows_apart) + i.
inline std::vector<WfPart> wf_cut(const std::vector<WfRun> &entries, unsigned log_places, unsigned log_rows_apart) {
  BLK_REQUIRE(log_rows_apart <= log_places && log_places < 63, "wf_cut: a part is smaller than an entry");
  const u64 most = u64(1) << (log_places - log_rows_apart);
  std::vector<std::vector<WfRun>> cut;  // by part: first_place = the entry's number in the part
  u64 room = 0;
  for (WfRun r : entries)
    while (r.places) {
      if (!room) cut.emplace_back(), room = most;
      const u64 take = std::min(room, r.places);
      cut.back().push_back(WfRun{most - room, r.first_row, take, r.stride});
      r.first_row += take, r.places -= take, room -= take;
    }
  std::vector<WfPart> parts;
  for (const std::vector<WfRun> &one : cut) {
    WfPart p{log_places, {}};
    for (u64 q = 0; q < u64(1) << log_rows_apart; q++)
      for (const WfRun &r : one) p.runs.push_back(WfRun{q * most + r.first_place, r.first_row + q * r.stride, r.places, log_rows_apart ? 0 : r.stride});
    parts.push_back(std::move(p));
  }
  return parts;
}

// The largest number of parts of one lookup. Each part is a separate product tree in the plan, in
// the key file and in the committer's work, and nothing later stops a large number of them: the
// sizes that depend on it are derived to fit. `ranges-part` is in a public file, and a verifier
// plans from that file before anything ties it to a commitment. So wf_lines() counts a lookup's
// parts before cutting it, and sets `fault` if there are more than this.
// The limit has no other basis than to keep a planning call within seconds and megabytes: time and
// memory grow with the number of parts, and a verification call plans several times. Measured with
// the program `cp_frame -plan key` on blocks/frame/examples/record.f2.txt, only
// `ranges-part` changed (12 and 9 give more parts than the limit allows):
//   18 (the file's own value): 12 parts a lookup, 0.22 s, 27 MB
//   14: 192 parts, 1.4 s, 278 MB
//   12: 768 parts, 4.9 s, 975 MB
//    9: 6,144 parts, 36.6 s, 6.4 GB
// 192, 768 and 6,144 are upper bounds, 12 * 2^(18 - `ranges-part`). Worked out, not run:
// wf_parts_of() on the two lookups of stand_in_subproofs() (blocks/frame/files.h), the shape that
// `cp_frame` plans, gives 187 and 188, 746 and 751, 5,968 and 6,008 parts.
// A verifier's planning call (`cp_verify -plan accounting` on prover/examples/target_full_f2 and
// target_small_f2) on a model's real shapes, at its parameter file's own value and at the smallest
// value that the limit allows:
//   full size: 1.8 s, 0.18 GB and 6.3 s, 0.56 GB
//   a small model: 0.5 s, 0.04 GB and 2.8 s, 0.24 GB
// At the smallest value the call costs about three and a half times (full size) and six times
// (small model) what it costs at the file's own value, which every caller pays anyway: a small factor.
// The limit is about twenty times the example files' number of parts, so other files have room.
constexpr u64 WF_MOST_PARTS = 256;
// How many parts wf_cut() would make.
inline u64 wf_parts_of(const std::vector<WfRun> &entries, unsigned log_places, unsigned log_rows_apart) {
  BLK_REQUIRE(log_rows_apart <= log_places && log_places < 63, "wf_cut: a part is smaller than an entry");
  const u64 most = u64(1) << (log_places - log_rows_apart);
  u64 whole = 0, left = 0;  // full parts; entries in the unfilled part (a sum of all could wrap)
  for (const WfRun &r : entries) {
    whole += r.places / most, left += r.places % most;
    whole += left / most, left %= most;
  }
  return whole + (left ? 1 : 0);
}

struct WfLines {
  unsigned k_col = 0;
  u64 matrix_rows = 0;
  std::vector<WfLookup> lookups;
  GroupLine group;         // of the oracle message with the multiplicities
  // if true, the mask of the proximity test's first fold is sent with the multiplicities
  bool masks_ride = true;
  OracleId message_id = 1; // after the index-mask message
  const char *fault = nullptr;  // not null: `wf_sized` rejects the shape with this message
};
// The number of planes of a lookup with these parts: the smallest number, at least 2, that exceeds
// log_places + k_col of every part and with which a multiplicity can count all the lookup's keys,
// padding included (a part has 2^(log_places + k_col) keys). wf_fault() rejects fewer; it also
// rejects 63, where the search stops.
inline unsigned wf_planes_of(const std::vector<WfPart> &parts, unsigned k_col) {
  unsigned planes = 2;
  u64 keys = 0;
  for (const WfPart &p : parts) planes = std::max(planes, p.log_places + k_col + 1), keys += u64(1) << std::min(p.log_places + k_col, 62u);
  while (planes < 63 && keys >> planes) planes++;
  return planes;
}
// Cuts each lookup into parts of 2^(part_log - digits) places (`digits`: log2 of a key's number of
// digits, rounded up) and sets its planes: `live` is their number (0: all 2^mu), and 2^mu the
// smallest power of two that holds them.
//   part_log  the parameter file's line `ranges-part`: a part reads up to 2^part_log rows of the
//             matrix, so all parts need about the same memory in a prover
//   group     the parameter file's lines `group once-ranges` and `stages once-ranges`
inline WfLines wf_lines(const WfShape &shape, const GroupLine &group, unsigned part_log) {
  WfLines L;
  L.k_col = shape.k_col, L.matrix_rows = shape.matrix_rows, L.group = group;
  // A part of more rows than the matrix has, rounded up to a power of two, reads no more rows and
  // costs more: in wf_pending_row() (blocks/iop/wf.h) prover and verifier each write one element
  // for each place of a part, 2^36 elements at part_log = 36.
  if (part_log > ceil_log2(std::max<u64>(shape.matrix_rows, 1))) L.fault = "a part would read more rows than the matrix has: the line ranges-part is too large for this matrix";
  for (const WfShapeLookup &s : shape.lookups) {
    WfLookup l = s.lookup;
    const unsigned digits = ceil_log2(std::max<size_t>(l.digit_weight.size(), 1));
    l.parts.clear();  // stays empty if part_log is too small: `wf_sized` then rejects the shape
    if (!L.fault && part_log >= digits && part_log - digits >= s.log_rows_apart && part_log < 63) {
      if (wf_parts_of(s.entries, part_log - digits, s.log_rows_apart) > WF_MOST_PARTS)
        L.fault = "a look-up would have more than 256 parts: the line ranges-part is too small for this matrix";
      else
        l.parts = wf_cut(s.entries, part_log - digits, s.log_rows_apart);
    }
    if (l.mu || l.live) L.fault = "a look-up's planes follow from its parts: whoever gives the shape leaves them 0";
    const unsigned planes = wf_planes_of(l.parts, shape.k_col);
    l.mu = std::max(1u, ceil_log2(planes)), l.live = planes == (1u << l.mu) ? 0 : planes;
    L.lookups.push_back(std::move(l));
  }
  return L;
}

// `x`: WF's public parameters; `test`: its proximity test's sizes.
struct WfSized {
  WfLines lines;
  // The matrix's size as `lines` has it; it sizes the mask rows and the plan. A run does not read
  // it: both sides take the size from the `LinShape` in the chain's first query (wf_matrix_of(),
  // blocks/iop/wf.h) and form the lookups from that, so every column of a row that a place reads
  // gives a key. Nothing compares the two sizes. sized() (blocks/frame/files.h) makes both from the
  // model's number of rows and the parameter file's number of column variables. key_fault()
  // (blocks/frame/key_fixed.h) calls wf_fault() on these lookups with the `LinShape`'s size:
  // wf_fault() rejects parts that reach beyond that matrix's rows, and requires one lookup that
  // reads every row. It has no such check on k_col, which enters it only through the counts of keys
  // and of masks. A run under another k_col has a different sequence of rounds from this plan's,
  // and `Cursor` (blocks/numbers/plan.h) stops the program.
  WfMatrix matrix;
  WfPublic x;
  TestSizes test;
  size_t pending = 0;   // pending claims on the matrix that WF leaves: one for each part
  std::string refused;  // not empty: why these inputs give no valid sizes
};

// codes: the caller's list; the codes of WF's proximity test are added to it where missing.
inline WfSized wf_sized(const Setting &s, const WfLines &L, Codes &codes) {
  WfSized z;
  z.lines = L;
  std::string &why = z.refused;
  WfPublic &x = z.x;
  if (L.fault) return why = std::string("ranges: ") + L.fault, z;
  const WfMatrix &m = z.matrix = WfMatrix{L.k_col, L.matrix_rows};
  x.lookups = L.lookups, x.message_id = L.message_id;
  x.choice.masks_ride = L.masks_ride;
  if (L.lookups.empty()) return why = "ranges: no look-up", z;
  for (const WfLookup &l : L.lookups) {
    if (l.mu < 1 || l.mu > 6 || l.live > (1u << l.mu) || l.d < 1 || l.d > 40 || l.parts.empty()) return why = "ranges: a look-up's table, planes or parts are not admitted", z;
    z.pending += l.parts.size();
  }
  const u64 ell = u64(1) << L.group.log_ell, ell_zk = u64(1) << L.group.mask_log_ell;
  OracleShape &o = x.message.shape;
  o.name = "multiplicities of the ranges' look-ups", o.main_deg = 1, o.answers = s.samples;
  for (const WfBlock &b : wf_layout(x, ell))
    if ((u64(1) << b.bits) > ell) return why = "once-ranges: a block of planes is longer than a main row", z;
  o.main_rows = (size_t)wf_layout(x, ell).back().row + 1;
  if (ell_zk <= s.samples) return why = "once-ranges: a mask row is shorter than the pad", z;
  o.mask_rows = Send::mask_rows_for(wf_own_masks(x, m), ell_zk, s.samples);
  if (!o.mask_rows) return why = "once-ranges: no number of mask rows has room for the look-ups' masks", z;

  unsigned k, h;
  lin_group_bits({&o}, k, h);
  z.test.group = shape_of("once-ranges", L.group, k, false);
  z.test.group.h = h;
  GroupLoad load;
  load.cube_bits = stage_cube_bits(z.test.group), load.messages = 1, load.batched_rows = wf_rows_batched(x, m);
  if (!solve_group(s, Scope::ONCE, load, [](u64 t) { return t; }, z.test, why)) return z;
  if (!solve_test(s, Scope::ONCE, L.group, z.test, why)) return z;
  x.stages = iopp_stages(z.test.stages, codes);
  o.code = x.stages[0].code, o.mask_code = x.stages[0].mask_code;
  if (const char *f = wf_fault(codes, x, m)) return why = std::string("ranges: ") + f, z;
  return z;
}

// Entries that the one-time chain needs in the index-mask message: two for each part,
// 1 + 2 k_col if there are several parts (for BATT, which batches the pending claims into one), and
// `samples` pads, which mask its samples' answers. Called only by tests.
inline size_t wf_index_mask_room(const WfSized &z, unsigned samples) { return wf_index_mask_entries(z.x, z.matrix) + samples; }

// WF's rounds in the one-time chain, after the index-mask message is sent and sampled: WF, then, if
// WF leaves more than one pending claim, BATT. z.refused must be empty: this is not checked.
inline void wf_plan(Plan &p, const Codes &codes, const WfSized &z) {
  const WfPublic &x = z.x;
  const std::vector<TreeSize> trees = tree_sizes(codes, wf_message(x));
  plan_send(p, "ranges: multiplicities", {SampleSize{Send::cube(codes, x.message.shape).bits, {z.test.stages[0].g}}}, (unsigned)trees.size());  // the copy of the group from which `plan_iopp` computes the errors
  const std::vector<LkpPublic> lookups = wf_lookups(x, z.matrix);
  for (size_t i = 0; i < lookups.size(); i++)
    plan_lookup(p, "ranges: " + x.lookups[i].name, LookupSize{0, lookups[i].d, lookups[i].mu, false, lookups[i].live, lookups[i].parts});
  if (const char *fault = iopp_sized_fault(codes, x.stages, z.test.stages)) return p.refuse(std::string("ranges: ") + fault);  // is `codes` the list that `wf_sized` filled?
  plan_iopp(p, "ranges", z.test.stages, trees, wf_rows_batched(x, z.matrix), x.choice.masks_ride ? 0 : 1, x.choice.masks_ride);
  plan_member_rows(p, "ranges: rows on the table", z.pending, z.matrix.k_col);
}

}  // namespace blk
