// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Fixture for tests that run RNG, the approximate range check on committed cells (small integers):
// it is to show |cell| < 2 D (blocks/iop/rng.h). The namespace `rng_world` lets one program
// include several fixtures. A run's steps (SEND sends oracle messages, that is encoded main and
// mask rows under Merkle roots, and takes an out-of-domain sample of each):
//   SEND   the packed cells of the matrices (`lanes` cells in an entry); the mask row (random
//          values) of this message supplies the lookup's mask entries;
//   SEND   the reads: tables of weighted partial sums of the cells, over the extension field E;
//   RNG    the verifier's matrix; SEND of the digits of the combinations (each row of cells times
//          each column of that matrix) and of the multiplicities of the digits' lookup; the
//          lookup (LKP); RNG derives all of this from its one RngShape;
//   one batching element, which combines all of RNG's equations into one linear claim (`Row`);
//   if the case is `whole`, the proximity test (`iopp` in identifiers) on the group of all four
//   oracle messages.
// A program sets rng_world::the_case, sizes a run with size_run() and calls prove() and verify().
#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "chain/closed.h"
#include "engine/engine_cpu.h"
#include "iop/bat.h"
#include "iop/iopp.h"
#include "iop/lkp.h"
#include "iop/rng.h"
#include "iop/row_value.h"
#include "iop/send.h"
#include "numbers/accounting.h"
#include "numbers/blocks.h"
#include "toy/toy_coins.h"

namespace rng_world {
using namespace blk;

static E e_of(u64 x) { return E::from(from_u64(x)); }

// ---- the case: the free parameters of the fixture ----
struct RngCase {
  const char *name = "rng_test's own";
  // Main rows are this long so that at no stage of the proximity test the mask code has the larger
  // rate.
  unsigned log_mask = 9, log_len = 13;  // a mask row has 2^9 entries, a main row 2^13
  unsigned b0 = 3, b1 = 2;              // the digits' bits: D = 2^(b0 + b1 - 1) = 16
  u64 digit_columns = 16;               // a power of two, at least the number of all cell rows
  // cells in one packed entry; log2 of the packing radix
  unsigned lanes = 3, log_radix = 7;
  std::vector<std::pair<u64, u64>> matrices = {{8, 5}, {4, 3}};       // rows, columns
  std::vector<std::pair<size_t, u64>> reads = {{0, 8}, {0, 4}, {1, 2}};  // matrix, length of the weight vector
  // (The largest number of columns of a matrix) x cell_bound < D, so that no honest combination
  // leaves its window [-D, D); not checked.
  i64 cell_bound = 3;    // cells are in -cell_bound .. cell_bound
  // 33 is no power of two, so that some key numbers of the digits' lookup name no digit (padding).
  unsigned target = 33;  // error target in bits (`Setting::target`) = the number of combinations
  // If true, the digits' and the multiplicities' messages have no mask row, as in a request's
  // segments (blocks/frame/req_frame.h), where only a segment's first message has one; here the
  // reads keep theirs. The pads that mask the answers to the two messages' samples are then in
  // their main rows, in tables that RNG builds and that no caller can reach before they are sent.
  bool reference = false;
  // If false, the plan (the list of a run's rounds) and the run stop after the batching element of
  // RNG's claim: no proximity test, codes written by hand, and a plan whose errors mean nothing.
  // For tests of what the prover sends (blocks/tests/hiding/). No verifier's verdict is worth
  // anything on such a run.
  bool whole = true;
  size_t tail = 2;  // whole == false: random entries of a row's encoding
  // Not 0: public inputs that contradict each other, for the test that RNG aborts under them. The
  // shape in the hash chain's first query (which fixes the public inputs) gets
  // `shape_combinations` combinations; the plan's round for the verifier's matrix is planned for
  // `planned_columns` columns of cells instead of RngShape::all_columns(). (The round fixes the
  // product, the number of the matrix's entries; its error is computed from the plan's own number
  // of combinations.) Read only in size_run_at().
  unsigned shape_combinations = 0;
  u64 planned_columns = 0;
};
inline RngCase the_case;  // read by functions below; a program sets it before it sizes a run
// A small case that stops after RNG.
inline RngCase rng_small_case(bool reference) {
  RngCase c;
  c.name = reference ? "small, the reference shape" : "small";
  c.log_mask = 7, c.log_len = 7, c.b0 = 2, c.b1 = 1, c.digit_columns = 4, c.lanes = 3, c.log_radix = 4;  // (a mask row is no longer than a main row)
  c.matrices = {{3, 2}}, c.reads = {{0, 3}}, c.cell_bound = 1, c.target = 2, c.reference = reference, c.whole = false;
  return c;
}

// Cuts [0, count) into aligned ranges of power-of-two lengths: (first, log2 of the length).
static std::vector<std::pair<u64, unsigned>> pieces(u64 count) {
  std::vector<std::pair<u64, unsigned>> p;
  for (u64 first = 0; first < count;) {
    unsigned bits = 0;
    while (first + (u64(2) << bits) <= count && (first & ((u64(2) << bits) - 1)) == 0) bits++;
    p.push_back({first, bits});
    first += u64(1) << bits;
  }
  return p;
}
// Places the blocks in one row, the largest first, so that every block's range is aligned.
static void place_in_row(std::vector<LaidBlock *> blocks, u64 row) {
  std::stable_sort(blocks.begin(), blocks.end(), [](const LaidBlock *a, const LaidBlock *b) { return a->bits_i + a->bits_c > b->bits_i + b->bits_c; });
  u64 place = 0;
  for (LaidBlock *b : blocks) {
    b->first_row = row, b->row_bits = 0, b->first_place = place, b->place_bits = b->bits_i + b->bits_c, b->i_is_low = true;
    place += u64(1) << b->place_bits;
  }
}
static std::vector<LaidBlock> cut(u64 rows, u64 columns) {
  std::vector<LaidBlock> blocks;
  for (const auto &i : pieces(rows))
    for (const auto &c : pieces(columns)) {
      LaidBlock b;
      b.first_i = i.first, b.bits_i = i.second, b.first_c = c.first, b.bits_c = c.second;
      blocks.push_back(b);
    }
  return blocks;
}
static OracleShape message(const char *name, size_t rows, unsigned deg, const Setting &set, bool mask_row = true) {
  OracleShape s;
  s.name = name, s.code = 0, s.main_rows = rows, s.main_deg = deg, s.mask_rows = mask_row ? 1 : 0, s.answers = set.samples;
  return s;
}
// Oracle messages, numbered as sent: 0 packed cells, 1 reads, 2 digits, 3 multiplicities.
static RngShape rng_shape(const Setting &set, unsigned combinations) {
  RngShape S;
  S.lanes = the_case.lanes, S.log_radix = the_case.log_radix, S.digit_bits[0] = the_case.b0, S.digit_bits[1] = the_case.b1, S.combinations = combinations;
  S.matrices.resize(the_case.matrices.size());
  for (size_t mu = 0; mu < S.matrices.size(); mu++) S.matrices[mu].rows = the_case.matrices[mu].first, S.matrices[mu].columns = the_case.matrices[mu].second;
  std::vector<LaidBlock *> all;
  for (size_t mu = 0; mu < S.matrices.size(); mu++) {
    S.matrices[mu].packed = 0;
    S.matrices[mu].laid = cut(S.packed_rows(mu), S.matrices[mu].columns);
    for (LaidBlock &b : S.matrices[mu].laid) all.push_back(&b);
  }
  place_in_row(all, 0);
  u64 first = 0;
  for (const std::pair<size_t, u64> &r : the_case.reads) {
    RngRead read;
    read.matrix = r.first, read.length = r.second;
    first += S.matrices[r.first].columns * (S.matrices[r.first].rows / r.second);
    S.reads.push_back(read);
  }
  S.reads_in = 1;
  S.reads_laid = cut(1, first);
  all.clear();
  for (LaidBlock &b : S.reads_laid) all.push_back(&b);
  place_in_row(all, 0);
  // The digits: combination u of column c at place c * Np + u, where Np = S.step(); low digits in
  // row 0, high digits in row 1.
  S.digit_columns = the_case.digit_columns;
  S.digits = OracleNotice{message("digits", 2, 1, set, !the_case.reference), 0};
  for (u64 c = 0; c < 2 * the_case.digit_columns; c++)
    for (const auto &u : pieces(combinations)) {
      LaidBlock b;
      b.first_i = u.first, b.bits_i = u.second, b.first_c = c, b.bits_c = 0, b.i_is_low = true;
      b.first_row = c / the_case.digit_columns, b.row_bits = 0, b.first_place = (c % the_case.digit_columns) * S.step() + u.first, b.place_bits = u.second;
      S.digits_laid.push_back(b);
    }
  S.multiplicities = OracleNotice{message("multiplicities", 1, 1, set, !the_case.reference), 0};
  S.digits_in = 2, S.masks_from = 0;
  return S;
}
// The reads' weight vectors. RNG lets a caller derive them from challenges; here they are
// constants, fixed in the first query like the shape.
static RngWeights weights(const RngShape &S) {
  ToyCoins fixed(5);
  RngWeights phi;
  for (const RngRead &r : S.reads) {
    phi.emplace_back();
    for (u64 i = 0; i < r.length; i++) phi.back().push_back(fixed.e());
  }
  return phi;
}
// The number of linear claims that RNG leaves, that of rng_row() included.
static u64 rows_left(const RngShape &R) {  // LKP's, the key row, RNG's
  const LkpPublic L = rng_lookup(R);
  return (L.planes() - 1) + L.parts[0] + L.d + 4 + 1 + 1;
}

struct RangeRun {
  Setting set;
  Codes codes;
  Planned plan;
  std::vector<IoppStage> stages;
  unsigned combinations = 0;
  explicit RangeRun(const Setting &s) : set(s) {}
};
static std::vector<std::vector<OracleNotice>> prover_messages(const Setting &set) {
  return {{{message("packed cells", 1, 1, set), 0}}, {{message("reads", 1, E_DEGREE, set), 0}}};  // RNG's own two are in its shape
}
// Sizes a `whole` run, with main codewords of 2^(log_len + up) symbols. May return a refused()
// plan.
static RangeRun size_run_at(unsigned up) {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = the_case.target;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  RangeRun run(set);
  Plan plan(set);
  run.combinations = the_case.shape_combinations ? the_case.shape_combinations : rng_combinations(plan);
  const RngShape R = rng_shape(set, run.combinations);
  const u64 rows = rows_left(R) + 4 * set.samples;
  GroupSize g;
  g.name = "cells, reads, digits, multiplicities";
  g.k = 3, g.h = the_case.reference ? 1 : 2;  // 1 + 1 + 2 + 1 main rows; four mask rows, or two
  g.main.name = "rows", g.main.ell = u64(1) << the_case.log_len, g.main.log_n = the_case.log_len + up;
  g.mask.name = "mask rows", g.mask.over_E = true, g.mask.ell = u64(1) << the_case.log_mask, g.mask.log_n = the_case.log_mask + up + 1;
  const LkpPublic L = rng_lookup(R);
  if (lkp_masks(L) + set.samples > g.mask.ell) plan.refuse("the mask row is too short for the look-up");
  u64 t = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, t);
    GroupLoad load;
    load.cube_bits = stage_cube_bits(g), load.messages = 2, load.batched_rows = rows;
    g = size_group(set, plan.scope(), g, load);
    if (!g.m) {
      plan.refuse("no slack holds the first stage's rounds at the target");
      return run.plan = plan.finish(), run;
    }
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= t) break;
    t = more;
  }
  const std::vector<Stage> stages = solve_stages(plan, g, t, {StageChoice{1, 1, g.main.log_n, the_case.log_mask + 3 + up}, StageChoice{0, 0, g.main.log_n, g.main.log_n}});
  if (plan.refused()) return run.plan = plan.finish(), run;
  run.stages = iopp_stages(stages, run.codes);
  plan_chain_start(plan, "range check run", true);
  std::vector<TreeSize> trees;
  std::vector<std::vector<OracleNotice>> sent = prover_messages(set);
  sent.push_back({R.digits, R.multiplicities});
  for (size_t i = 0; i < sent.size(); i++) {
    if (i == 2) plan_rng(plan, "range check", the_case.planned_columns ? the_case.planned_columns : R.all_columns());
    std::vector<SampleSize> samples;
    for (const OracleNotice &n : sent[i]) samples.push_back(SampleSize{Send::cube(run.codes, n.shape).bits, {g}});
    const std::vector<TreeSize> own = tree_sizes(run.codes, sent[i]);
    plan_send(plan, sent[i][0].shape.name, samples, (unsigned)own.size());
    trees.insert(trees.end(), own.begin(), own.end());
  }
  plan_lookup(plan, "digits", LookupSize{L.parts[0], L.d, L.mu, true, L.live});
  plan_bat_sigma(plan, "range check", R.equations());
  plan_iopp(plan, g.name, stages, trees, rows, 1);
  return run.plan = plan.finish(), run;
}
// Sizes a run for `whole == false` (see RngCase::whole).
static RangeRun block_run() {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = the_case.target;
  RangeRun run(set);
  Plan plan(set);
  run.combinations = rng_combinations(plan);
  const RngShape R = rng_shape(set, run.combinations);
  const LkpPublic L = rng_lookup(R);
  BLK_REQUIRE(lkp_masks(L) + set.samples <= u64(1) << the_case.log_mask, "rng_world: the mask row is too short for the look-up");
  run.codes.base.push_back(Code{the_case.log_len + 1, size_t(1) << the_case.log_len, the_case.tail});
  run.codes.mask.push_back(Code{the_case.log_mask + 1, size_t(1) << the_case.log_mask, the_case.tail});
  plan_chain_start(plan, "range check alone", true);
  std::vector<std::vector<OracleNotice>> sent = prover_messages(set);
  sent.push_back({R.digits, R.multiplicities});
  for (size_t i = 0; i < sent.size(); i++) {
    if (i == 2) plan_rng(plan, "range check", R.all_columns());
    std::vector<SampleSize> samples;
    for (const OracleNotice &n : sent[i]) samples.push_back(SampleSize{Send::cube(run.codes, n.shape).bits, no_sized_group()});
    plan_send(plan, sent[i][0].shape.name, samples, (unsigned)tree_sizes(run.codes, sent[i]).size());
  }
  plan_lookup(plan, "digits", LookupSize{L.parts[0], L.d, L.mu, true, L.live});
  plan_bat_sigma(plan, "range check", R.equations());
  return run.plan = plan.finish(), run;
}
// For a `whole` case, tries `up` from 1 to 6. If every plan is refused(), returns the last run:
// check plan->refused().
static RangeRun size_run() {
  if (!the_case.whole) return block_run();
  for (unsigned up = 1; up < 6; up++) {
    RangeRun run = size_run_at(up);
    if (!run.plan->refused()) return run;
  }
  return size_run_at(6);
}
static IoppChoice choice(const Setting &) { return IoppChoice(); }  // the defaults
// The public inputs go into the chain's first query, through the `Binder` (blocks/chain/public.h)
// from which both sides start their channel.
static const RngShape *handed_in = nullptr;  // to test RNG's aborts: replaces the shape in `bind`
struct Inputs {  // handles of what the first query holds, and nothing else
  Public<Codes> codes;
  Public<std::vector<std::vector<OracleNotice>>> sent;  // the prover messages before RNG's own
  Public<RngShape> range;
  Public<RngWeights> weights;
  Public<std::vector<IoppStage>> stages;
  Public<IoppChoice> choice;
};
// Every chain's first query contains a digest of the instance's public files; here it is made up.
static Digest made_up_configuration() {
  Digest d;
  for (u32 i = 0; i < 8; i++) d.w[i] = 0x5EED0000u + i;
  return d;
}
static Inputs bind(Binder &b, const RangeRun &run) {
  Inputs in;
  b.configuration(made_up_configuration());
  in.codes = b.parameter("codes", run.codes);
  in.sent = b.parameter("oracle messages", prover_messages(run.set));
  const RngShape R = rng_shape(run.set, run.combinations);
  in.range = b.parameter("range check", handed_in ? *handed_in : R);
  in.weights = b.parameter("weights of the reads", weights(R));
  in.stages = b.parameter("the stages of the proximity test", run.stages);
  in.choice = b.parameter("the choices of the proximity test", choice(run.set));
  return in;
}

// What one side of a run leaves for the test.
struct Seen {
  std::vector<PadLook> pads;  // the prover's side: what is at the pads' places of all samples
  std::vector<Row> rows;
  // the prover's side, from row_is_the_sum()
  bool structured_row_is_the_sum = true;
  std::vector<Table> main, mask;  // the prover's side: what it has sent, by oracle message
};
static E left_side(const Row &row, const Seen &sent) {
  E sum = E::zero();
  for (const Term &t : row.terms) sum += table_dot((t.part == Part::MAIN ? sent.main : sent.mask)[t.oracle], t.v);
  for (const Unit &u : row.units) sum += u.coeff * (u.part == Part::MAIN ? sent.main : sent.mask)[u.oracle].entry(u.index);
  for (const MainForm &f : row.forms) {
    const Table &t = sent.main[f.oracle];
    std::vector<E> w(t.entries(), E::zero());
    f.c.add_to(w.data(), ceil_log2(t.entries()));
    for (u64 j = 0; j < t.entries(); j++) sum += w[j] * t.entry(j);
  }
  if (!row.runs.empty()) std::abort();  // no claim made here has runs
  return sum;
}
// The names of the verifier's claims that are false on what the prover has sent.
static std::string false_rows(const Seen &prover, const Seen &verifier) {
  std::string names;
  for (const Row &r : verifier.rows)
    if (left_side(r, prover) != r.target) names += r.name + "; ";
  return names;
}
// Compares values (`target`) and left sides on what `a` has sent, not coefficients.
[[maybe_unused]] static bool same_rows(const Seen &a, const Seen &b) {
  if (a.rows.size() != b.rows.size()) return false;
  for (size_t i = 0; i < a.rows.size(); i++)
    if (a.rows[i].target != b.rows[i].target || left_side(a.rows[i], a) != left_side(b.rows[i], a)) return false;
  return true;
}
static OracleDraft draft_of(ProverContext &pr, const OracleShape &shape, const Table &main) {
  OracleDraft d;
  d.shape = shape;
  d.main = pr.engine.put(main);
  d.mask = Send::fresh_mask(pr.st.codes(), shape, pr.coins);
  return d;
}

// Writes out RNG's equations one by one as coefficient vectors on the main tables of messages 0 to
// 2, sums them weighted by the powers of x and compares the sum with `row`. Psi[j][u]: combination
// u of cell row j.
[[maybe_unused]] static bool row_is_the_sum(const RngShape &S, const RngWeights &phi, const RngTheta &theta, const Row &row, const E &x) {
  const size_t len = size_t(1) << the_case.log_len;
  std::vector<std::vector<E>> want = {std::vector<E>(len, E::zero()), std::vector<E>(len, E::zero()), std::vector<E>(2 * len, E::zero())};
  E target = E::zero();
  auto at = [&](size_t o, const std::vector<LaidBlock> &blocks, u64 i, u64 c) -> E & {
    Table probe(1, o == 2 ? 2 : 1, len);
    laid_set(probe, blocks, i, c, E::one());
    for (u64 j = 0; j < probe.entries(); j++)
      if (probe.entry(j) == E::one()) return want[o][j];
    std::abort();
  };
  const u64 N = S.combinations, Np = S.step();
  const E D = e_of(u64(1) << (the_case.b0 + the_case.b1 - 1)), high = e_of(u64(1) << the_case.b0);
  std::vector<E> power(S.equations());
  power[0] = E::one();
  for (size_t j = 1; j < power.size(); j++) power[j] = power[j - 1] * x;
  auto psi = [&](u64 cell_row, u64 u, const E &w) {  // adds w * Psi[cell_row][u] to the left side
    at(2, S.digits_laid, u, cell_row) += w, at(2, S.digits_laid, u, S.digit_columns + cell_row) += w * high;
    target += w * D;
  };
  u64 number = 0, column = 0, cell_row = 0;
  std::vector<u64> first_row, first_column;
  for (size_t mu = 0; mu < S.matrices.size(); mu++) {
    const RngMatrix &m = S.matrices[mu];
    for (u64 p = 0; p < S.packed_rows(mu); p++)
      for (u64 u = 0; u < N; u++) {
        const E w = power[number + p * Np + u];
        for (u64 c = 0; c < m.columns; c++) at(0, m.laid, p, c) += w * E::from(from_i64(theta[(column + c) * N + u]));
        for (unsigned l = 0; l < S.lanes && p * S.lanes + l < m.rows; l++) psi(cell_row + p * S.lanes + l, u, -w * e_of(u64(1) << (S.log_radix * l)));
      }
    first_row.push_back(cell_row), first_column.push_back(column);
    number += Np << ceil_log2(S.packed_rows(mu)), column += m.columns, cell_row += m.rows;
  }
  for (size_t q = 0; q < S.reads.size(); q++) {
    const RngRead &r = S.reads[q];
    const RngMatrix &m = S.matrices[r.matrix];
    const u64 L = r.length, parts = m.rows / L;
    for (u64 g = 0; g < parts; g++, number += Np)
      for (u64 u = 0; u < N; u++) {
        const E w = power[number + u];
        for (u64 c = 0; c < m.columns; c++) at(1, S.reads_laid, 0, S.read_first(q) + c * parts + g) += w * E::from(from_i64(theta[(first_column[r.matrix] + c) * N + u]));
        for (u64 i = 0; i < L; i++) psi(first_row[r.matrix] + g * L + i, u, -w * phi[q][i]);
      }
  }
  if (row.target != target) return false;
  for (OracleId o = 0; o < 3; o++)
    for (u64 j = 0; j < want[o].size(); j++)
      if (row.coefficient(o, Part::MAIN, j) != want[o][j]) return false;
  return true;
}

// The witness: the cells, matrix by matrix, row by row. prove() computes the honest prover's packed
// entries, reads, digits and multiplicities from them. All that is public (sizes, bounds, layouts,
// the reads' weights) is the same for every witness; no sum or other output of the cells is public.
// number < 10: random within the bound, each number from its own key; 10: all cells 0; 11: all at
// the bound; 12: witness 0 with one cell changed.
using RngWitness = std::vector<std::vector<i64>>;
static RngWitness rng_witness(unsigned number) {
  const i64 bound = the_case.cell_bound;
  ToyCoins data(99 + (number < 10 ? number : 0));
  RngWitness cells(the_case.matrices.size());
  for (size_t mu = 0; mu < cells.size(); mu++)
    for (u64 i = 0; i < the_case.matrices[mu].first * the_case.matrices[mu].second; i++) {
      u8 b;
      data.bytes(&b, 1);
      cells[mu].push_back(number == 10 ? 0 : number == 11 ? bound : (i64)(b % (2 * bound + 1)) - bound);
    }
  if (number == 12) cells[0][0] = cells[0][0] == bound ? bound - 1 : cells[0][0] + 1;
  BLK_REQUIRE(number <= 12, "rng_world: no such witness");
  return cells;
}
// coins: all of the prover's private randomness.
// proof: the caller's, so that it can call keep_list() (the `Writer` then lists its items) before
// the first byte.
// order_broken: the reads are sent without a sample (for the test that RNG then aborts).
// gave_up: set if a combination leaves its window [-D, D): RNG's prover then sends nothing more
// and the run ends. If null, such a run aborts.
// before_rng: the run ends before RNG; its bytes, those of the two SENDs, bring a verifier as far
// as RNG.
static void prove(const RangeRun &run, RngWitness cells, Coins &coins, Writer &proof, Seen &seen, bool order_broken = false, bool *gave_up = nullptr, bool before_rng = false) {
  const RngShape R = rng_shape(run.set, run.combinations);
  EngineCpu engine;
  Binder binder(8, run.plan);
  const Inputs in = bind(binder, run);
  u8 salt[DIGEST_BYTES];
  coins.bytes(salt, sizeof salt);
  Digest random_bytes;
  digest_from_bytes(salt, random_bytes);
  ProverRounds ch(binder, random_bytes, proof, [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); });
  ProverContext pr(in.codes, ch, engine, coins);
  pr.pads = &seen.pads;
  const size_t len = run.codes.base[0].ell;

  Table packed(1, 1, len), reads(E_DEGREE, 1, len);
  for (size_t mu = 0; mu < cells.size(); mu++) rng_lay_packed(packed, R, mu, cells[mu]);
  std::vector<std::vector<i64>> read_from = cells;
  rng_lay_reads(reads, R, weights(R), read_from);
  {
    std::vector<OracleDraft> d;
    d.push_back(draft_of(pr, message("packed cells", 1, 1, run.set), packed));
    send_prove(pr, std::move(d));
  }
  if (order_broken) {
    std::vector<OracleDraft> d;
    d.push_back(draft_of(pr, message("reads", 1, E_DEGREE, run.set), reads));
    send_oracles(pr, std::move(d));  // sent, but no sample
  } else {
    std::vector<OracleDraft> d;
    d.push_back(draft_of(pr, message("reads", 1, E_DEGREE, run.set), reads));
    send_prove(pr, std::move(d));
  }

  const GroupId g = pr.st.new_group("cells, reads, digits, multiplicities", 0);
  pr.st.join(g, 0);
  if (!order_broken) pr.st.join(g, 1);  // `join` would abort too, before RNG's own check
  if (before_rng) return;
  RngOutput out;
  if (!rng_prove(pr, in.range, pr.ch.get(in.weights), cells, g, out)) {
    BLK_REQUIRE(gave_up, "rng_world: an honest combination leaves its window");
    *gave_up = true;
    return;
  }
  E x;
  std::vector<Row> rows = out.rows;
  rows.push_back(rng_row(pr, in.range, out, x));
  seen.structured_row_is_the_sum = row_is_the_sum(R, weights(R), out.theta, rows.back(), x);
  seen.rows = rows;
  for (OracleId o = 0; o < 4; o++) seen.main.push_back(engine.get(pr.behind[o].main)), seen.mask.push_back(pr.behind[o].mask);
  pr.st.add_row(g, std::move(rows.back()));  // RNG has added the others
  if (!the_case.whole) {  // the run stops after RNG: nothing tests the group
    ch.finish();
    return;
  }
  iopp_prove(pr, g, in.stages, in.choice);
  ch.finish();
  pr.st.require_all_tested();
}
// Witness 0 and ToyCoins(key): two runs differ only in the prover's private randomness.
[[maybe_unused]] static std::vector<u8> prove(const RangeRun &run, u64 key, Seen &seen, bool order_broken = false) {
  ToyCoins coins(key);
  Writer proof;
  prove(run, rng_witness(0), coins, proof, seen, order_broken);
  return proof.bytes();
}

// The verifier's side, run by run_verifier() (blocks/chain/closed.h): decide() gets the channel and
// the handles of what the first query contains, not the run. One global of the test reaches it,
// `verifier_leaves`, which is no public input: decide() stores its claims there for the test.
static Seen *verifier_leaves = nullptr;
static bool decide(VerifierRounds &ch, const Inputs &in, Notes *) {
  VerifierContext vf(in.codes, ch);
  const std::vector<std::vector<OracleNotice>> &sent = vf.ch.get(in.sent);
  std::vector<OracleId> ids;
  if (!send_verify(vf, sent[0], ids) || !send_verify(vf, sent[1], ids)) return false;
  const GroupId g = vf.st.new_group("cells, reads, digits, multiplicities", 0);
  vf.st.join(g, 0), vf.st.join(g, 1);
  RngOutput out;
  if (!rng_verify(vf, in.range, vf.ch.get(in.weights), g, out)) return false;
  E x;
  verifier_leaves->rows = out.rows;
  verifier_leaves->rows.push_back(rng_row(vf, in.range, out, x));
  vf.st.add_row(g, verifier_leaves->rows.back());
  if (!iopp_verify(vf, g, in.stages, in.choice)) return false;
  vf.st.require_all_tested();
  return true;
}
// For `whole` runs: decide() does not skip the proximity test.
static bool verify(const RangeRun &run, const std::vector<u8> &proof, Seen &seen, const char **why = nullptr) {
  verifier_leaves = &seen;
  const char *words = nullptr;
  const bool ok = run_verifier(8, run.plan, run, bind, decide, proof.data(), proof.size(), &words);
  if (why) *why = words ? words : "";
  return ok;
}

}  // namespace rng_world
