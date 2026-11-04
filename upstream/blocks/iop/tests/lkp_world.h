// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Fixtures (`world` in identifiers), a prover and a verifier to test LKP, the lookup argument
// (blocks/iop/lkp.h): every key occurs in a public table. Runs are planned for a target of 32 bits
// (`Setting::target`; default: 128). A run:
//   SEND  sends one oracle message (encoded rows under Merkle roots) and takes its out-of-domain
//         sample. Main row 0 has the keys, main row 1 the bit planes of the multiplicities (the
//         number of keys counted at each table row; plane j: bit j of each); the one mask row
//         supplies every mask entry (a random element added to one sent scalar).
//   LKP   the lookup; then one linear claim (`Row`) on the sent keys for each part of the keys.
//   Unless the run stops after LKP: the proximity test (`iopp` in identifiers) on the one group
//   (the oracle messages that it reads).
// To use: size_run() plans a run for a fixture; then call prove() and verify(). block_run() plans
// one that stops after LKP: call prove() only.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "chain/closed.h"
#include "engine/engine_cpu.h"
#include "iop/bat.h"
#include "iop/iopp.h"
#include "iop/lkp.h"
#include "iop/row_value.h"
#include "iop/send.h"
#include "numbers/blocks.h"
#include "toy/toy_coins.h"

namespace lkp_world {
using namespace blk;

static E e_of(u64 x) { return E::from(from_u64(x)); }
static std::vector<E> written(std::vector<u64> rows) {
  std::vector<E> r;
  for (u64 x : rows) r.push_back(e_of(x));
  return r;
}

struct World {
  const char *name;
  LkpPublic fixed;
};
static unsigned LOG_LEN = 0;  // log2 of a main row's message length, for the current run

// One part of 2^v keys, 2^d table rows, 2^mu bit-plane slots, 2^g trees of keys (key i: tree
// i mod 2^g). shifted: LKP moves the keys and rows of function f (a table and its keys) by
// f * theta, a challenge: "the shift".
static LkpPublic sizes(unsigned v, unsigned d, unsigned mu, unsigned g, bool shifted) {
  LkpPublic p;
  p.parts = {v}, p.d = d, p.mu = mu, p.g = g, p.shifted = shifted;
  p.read_from = {0};
  return p;
}
// lkp_test.cpp refers to these fixtures by index: append new ones at the end. Fields:
//   keys     LkpKeyBlock {first tree, log2 trees, function (-1: padding only), the padding's row}
//   tables   LkpTablePart {first row (h > 0: part number), log2 rows, function, rows ({}: 0, 1 ..)}
//   padding  LkpPadding {part, first key, log2 step, length, row}: keys that LKP takes to hold a
//            table row and does not count
[[maybe_unused]] static std::vector<World> worlds() {
  std::vector<World> w;
  w.push_back({"one range table", sizes(5, 3, 3, 0, false)});
  w.back().fixed.keys = {{0, 0, 0, 0}}, w.back().fixed.tables = {{0, 3, 0, {}}};
  w.push_back({"two functions under the shift", sizes(5, 3, 3, 2, true)});
  w.back().fixed.keys = {{0, 1, 0, 0}, {2, 0, 1, 0}, {3, 0, -1, 2}};  // tree 3: padding only; it holds row 2 (of function 0's table)
  w.back().fixed.tables = {{0, 2, 0, {}}, {4, 2, 1, written({20, 31, 42, 53})}};
  w.back().fixed.padding = {{0, 26, 2, 2, 7}};  // in tree 2: keys 26, 30 hold row 7 (53, shifted)
  w.push_back({"many keys, range order", sizes(10, 3, 4, 0, false)});
  w.back().fixed.keys = {{0, 0, 0, 0}}, w.back().fixed.tables = {{0, 3, 0, {}}};
  w.back().fixed.last_key_level_order = lkp_range_order(9, 0);
  w.push_back({"padding inside a tree, a table laid twice", sizes(5, 3, 3, 0, false)});
  w.back().fixed.keys = {{0, 0, 0, 0}};
  w.back().fixed.tables = {{0, 2, 0, written({7, 8, 9, 10})}, {4, 2, 0, written({7, 8, 9, 10})}};
  w.back().fixed.padding = {{0, 5, 0, 1, 1}, {0, 29, 0, 3, 6}};  // key 5 holds row 1 (8); keys 29, 30, 31 hold row 6 (9)
  w.push_back({"six live planes of eight slots", sizes(5, 3, 3, 0, false)});
  w.back().fixed.keys = {{0, 0, 0, 0}}, w.back().fixed.tables = {{0, 3, 0, {}}}, w.back().fixed.live = 6;
  w.push_back({"keys in three parts, padding in two of them", sizes(4, 3, 3, 0, false)});
  w.back().fixed.keys = {{0, 0, 0, 0}}, w.back().fixed.tables = {{0, 3, 0, {}}}, w.back().fixed.parts = {4, 3, 5};
  w.back().fixed.padding = {{0, 9, 1, 3, 2}, {2, 22, 0, 10, 5}};  // part 0: keys 9, 11, 13 hold row 2; part 2: keys 22 .. 31 hold row 5
  w.back().fixed.last_key_level_order = {0, 1};  // variables bound first in every part's last key level; the shortest part has only these two
  w.push_back({"the table part's number in a row's low bit, a short table repeated", sizes(5, 3, 3, 1, true)});
  w.back().fixed.h = 1;
  w.back().fixed.keys = {{0, 0, 0, 0}, {1, 0, 1, 0}};
  w.back().fixed.tables = {{0, 2, 0, {}}, {1, 1, 1, written({70, 81})}};
  return w;
}
[[maybe_unused]] static World without_shift(World w) {
  // Makes both tables parts of function 0's table: without the shift LKP accepts only one function.
  w.name = "the two tables as one function's, WITHOUT the shift", w.fixed.shifted = false;
  for (LkpKeyBlock &b : w.fixed.keys) b.function = std::min(b.function, 0);
  for (LkpTablePart &t : w.fixed.tables) t.function = 0;
  return w;
}
// The first part of a function's table, and a row of a part before the shift.
static const LkpTablePart &table_of(const LkpPublic &p, int function) {
  for (const LkpTablePart &t : p.tables)
    if ((int)t.function == function) return t;
  std::abort();
}
static E row_of(const LkpTablePart &t, u64 r) { return t.rows.empty() ? e_of(r) : t.rows[r]; }

static OracleShape oracle_shape(const Setting &set) {
  OracleShape s;
  s.name = "keys, multiplicities";
  s.code = 0, s.main_rows = 2, s.main_deg = E_DEGREE, s.mask_rows = 1, s.answers = set.samples;
  return s;
}
// Key part p starts at index 64 p of main row 0: with several parts, each may have at most 64 keys
// (not checked).
static u64 part_first(unsigned part) { return u64(part) << 6; }
// LKP's callback: adds a linear form of the multiplicities to a claim. Only the live bit planes are
// sent: each aligned block (first, bits) of lkp_live_blocks() starts at index first * 2^d of main
// row 1, with plane j of table row t at offset (j - first) + 2^bits * t.
static LkpPlanesAt planes_at(const LkpPublic &p, OracleId o) {
  return [p, o](Row &row, Tensor v) {
    for (const std::pair<unsigned, unsigned> &block : lkp_live_blocks(p.planes())) {
      Tensor part = lkp_on_block(v, p.mu, block);
      const unsigned own = part.vars();
      part.insert_choice(own, LOG_LEN - own, ((u64)block.first << p.d) >> own);
      part.insert_choice(0, 1, 1);
      row.add(o, Part::MAIN, std::move(part));
    }
  };
}
static void lay_planes(Table &main, const LkpPublic &p, const std::vector<E> &bits) {
  for (const std::pair<unsigned, unsigned> &block : lkp_live_blocks(p.planes()))
    for (u64 t = 0; t < u64(1) << p.d; t++)
      for (u64 j = 0; j < u64(1) << block.second; j++) main.set(1, ((u64)block.first << p.d) + j + (t << block.second), bits[block.first + j + (t << p.mu)]);
}
// Turns LKP's claim on one part of the keys into a linear claim on the sent keys, which start at
// index part_first(part) of main row 0. Everything else in the claim comes from LKP.
static Row key_row(bool several_parts, OracleId o, LkpKeys &&keys, unsigned part = 0) {
  Row row;
  row.name = several_parts ? "the keys of part " + std::to_string(part) + " at the point" : "the keys at the point";
  const unsigned own = (unsigned)keys.point.size();
  keys.weights.insert_choice(own, LOG_LEN - own, part_first(part) >> own);
  keys.weights.insert_choice(0, 1, 0);
  row.add_form(o, std::move(keys.weights));
  row.target = keys.target;
  for (MaskTerm<E> &m : keys.masks) row.add_mask(std::move(m));
  return row;
}
// For key i, counted through all parts in order: its part and its index within that part.
static std::pair<unsigned, u64> place_of(const LkpPublic &p, u64 i) {
  unsigned part = 0;
  while (i >= u64(1) << p.part_bits()[part]) i -= u64(1) << p.part_bits()[part], part++;
  return {part, i};
}
static bool is_padding(const LkpPublic &p, u64 i) { return lkp_is_padding(p, place_of(p, i).first, place_of(p, i).second); }
static u64 rows_left(const World &w) {  // the number of claims that LKP and `key_row` make
  u64 levels = 0;
  for (unsigned v : w.fixed.part_bits()) levels += v;
  const u64 parts = w.fixed.part_bits().size();
  return (w.fixed.planes() - 1) + (parts - 1) + levels + w.fixed.d + 1 + 2 + 1 + parts;
}

struct LookupRun {
  World world;
  Setting set;
  Codes codes;
  Planned plan;
  std::vector<IoppStage> stages;
  // false: plan and run stop after the claims on the keys: no proximity test, and codes set by
  // hand, not sized for the target. For tests of what the prover sends (blocks/tests/hiding/). A
  // verifier's verdict on such a run is worth nothing.
  bool whole = true;
  // Only with whole == false: a second oracle message with the same main rows is sent and sampled,
  // and the multiplicities are read from it. LKP aborts unless read_from lists this message.
  bool second_message = false;
  LookupRun(const World &w, const Setting &s) : world(w), set(s) {}
};
// Plans a run with its proximity test. up: log2 of the main code's length over its message's
// length; the mask code has up + 1.
static LookupRun size_run_at(const World &w, unsigned up) {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 32;
  set.set_work(ErrorKind::QUERY, Scope::REQUEST, 8);
  LookupRun run(w, set);
  Plan plan(set);
  const u64 rows = rows_left(w) + set.samples;
  GroupSize g;
  g.name = "keys, multiplicities";
  g.k = 1, g.h = 0;
  g.mask.name = "mask row", g.mask.over_E = true;
  g.mask.ell = pow2_at_least(lkp_masks(w.fixed) + set.samples);
  // Mask rows grow from stage to stage (each takes in two folded mask rows and the random tails);
  // main rows shrink. The main rows' log2 length, L + 4, is meant to keep the mask code's rate no
  // larger at any stage.
  const unsigned L = ceil_log2(g.mask.ell);
  LOG_LEN = L + 4;
  g.main.name = "rows", g.main.ell = u64(1) << LOG_LEN;
  g.main.log_n = LOG_LEN + up, g.mask.log_n = L + up + 1;
  u64 t = 0;
  for (int turn = 0; turn < 16; turn++) {
    g.main.r = g.mask.r = tail_for(set, t);
    GroupLoad load;
    load.cube_bits = stage_cube_bits(g);
    load.batched_rows = rows;
    g = size_group(set, plan.scope(), g, load);
    if (!g.m) {
      plan.refuse("no slack holds the first stage's rounds at the target");
      return run.plan = plan.finish(), run;
    }
    const u64 more = plan_positions(plan, analysis(set, g).at.agree, Bound(Real::two_to(-(long)set.target - 30)));
    if (more <= t) break;
    t = more;
  }
  const std::vector<Stage> stages = solve_stages(plan, g, t, {StageChoice{1, 1, g.main.log_n, L + 3 + up}, StageChoice{0, 0, g.main.log_n, g.main.log_n}});
  if (plan.refused()) return run.plan = plan.finish(), run;
  run.stages = iopp_stages(stages, run.codes);
  const std::vector<OracleNotice> sent = {{oracle_shape(set), 0}};
  plan_chain_start(plan, "look-up run", true);
  plan_send(plan, "keys, multiplicities", {SampleSize{Send::cube(run.codes, oracle_shape(set)).bits, {g}}}, (unsigned)tree_sizes(run.codes, sent).size());
  plan_lookup(plan, "look-up", LookupSize{w.fixed.parts[0], w.fixed.d, w.fixed.mu, w.fixed.shifted, w.fixed.live, w.fixed.parts});
  plan_iopp(plan, g.name, stages, tree_sizes(run.codes, sent), rows, 1);
  return run.plan = plan.finish(), run;
}
// Plans a run that stops after LKP. Main rows have 2^log_len message entries, the mask row
// 2^log_mask; each row has `tail` random entries.
[[maybe_unused]] static LookupRun block_run(const World &w, unsigned log_len, unsigned log_mask, size_t tail = 2, bool second_message = false) {
  Setting set = setting_goldilocks();
  set.hash_block_bytes = HASH_BLOCK_BYTES;
  set.target = 32;
  LookupRun run(w, set);
  Plan plan(set);
  run.whole = false, run.second_message = second_message;
  BLK_REQUIRE(log_mask <= log_len && lkp_masks(w.fixed) + set.samples <= u64(1) << log_mask, "lkp_world: the mask row is longer than a main row, or too short for the look-up");
  LOG_LEN = log_len;
  run.codes.base.push_back(Code{log_len + 1, size_t(1) << log_len, tail});
  run.codes.mask.push_back(Code{log_mask + 1, size_t(1) << log_mask, tail});
  const std::vector<OracleNotice> sent = {{oracle_shape(set), 0}};
  plan_chain_start(plan, "look-up alone", true);
  for (int message = 0; message < (second_message ? 2 : 1); message++)
    plan_send(plan, "keys, multiplicities", {SampleSize{Send::cube(run.codes, oracle_shape(set)).bits, no_sized_group()}}, (unsigned)tree_sizes(run.codes, sent).size());
  plan_lookup(plan, "look-up", LookupSize{w.fixed.parts[0], w.fixed.d, w.fixed.mu, w.fixed.shifted, w.fixed.live, w.fixed.parts});
  return run.plan = plan.finish(), run;
}
// The smallest fixture: one function, one part of keys, no shift, four bit planes.
[[maybe_unused]] static World small_world() {
  World w{"small: eight keys, two of padding, four rows", sizes(3, 2, 2, 0, false)};
  w.fixed.keys = {{0, 0, 0, 0}}, w.fixed.tables = {{0, 2, 0, {}}};
  w.fixed.padding = {{0, 6, 0, 2, 1}};  // keys 6, 7 hold row 1
  return w;
}
// The first of up = 1 .. 5 whose plan is not rejected (`refused()`), else up = 6, rejected or not.
static LookupRun size_run(const World &w) {
  for (unsigned up = 1; up < 6; up++) {
    LookupRun run = size_run_at(w, up);
    if (!run.plan->refused()) return run;
    std::printf("  (%s, lengths doubled %u times: %s)\n", w.name, up, run.plan->why_not().c_str());
  }
  return size_run_at(w, 6);
}
static IoppChoice choice(const Setting &) { return IoppChoice(); }  // defaults; `Setting::samples` sets the answers per sample
// bind() puts the public inputs into the hash chain's first query (the hash input fixed before the
// prover speaks), through the `Binder` from which both sides start their channel.
// With table_bound false, of `LkpPublic` only three sizes are bound, and prove() calls lkp_run()
// directly: LKP's entry points, lkp_prove() and lkp_verify(), take only bound inputs. Nothing in
// this repository sets it false. A test outside it did: a table chosen after alpha was accepted.
static bool table_bound = true;
static const LkpPublic *handed_in = nullptr;  // abort tests: `bind` binds it, not `run.world.fixed`
struct Inputs {  // handles to what the first query contains, and nothing else
  Public<Codes> codes;
  Public<OracleNotice> oracle;
  Public<LkpPublic> fixed;
  Public<std::vector<IoppStage>> stages;
  Public<IoppChoice> choice;
};
// Every chain's first query contains the configuration digest (a hash of the public files); the
// tests use a constant for it.
static Digest made_up_configuration() {
  Digest d;
  for (u32 i = 0; i < 8; i++) d.w[i] = 0x5EED0000u + i;
  return d;
}
static Inputs bind(Binder &b, const LookupRun &run) {
  Inputs in;
  b.configuration(made_up_configuration());
  in.codes = b.parameter("codes", run.codes);
  in.oracle = b.parameter("oracle message", OracleNotice{oracle_shape(run.set), 0});
  if (table_bound)
    in.fixed = b.parameter("look-up", handed_in ? *handed_in : run.world.fixed);
  else
    b.parameter("sizes of the look-up", std::vector<u64>{run.world.fixed.parts[0], run.world.fixed.d, run.world.fixed.mu});
  in.stages = b.parameter("the stages of the proximity test", run.stages);
  in.choice = b.parameter("the choices of the proximity test", choice(run.set));
  return in;
}

// What prove() records for the test; verify() fills in `rows` only.
struct Seen {
  std::vector<Row> rows;  // LKP's and the caller's
  u32 masks_used = 0;
  E theta = E::zero(), alpha = E::zero();
  std::vector<E> key_point;
  Table main, mask;  // the prover's side: what it has sent
};
// A claim's linear form, evaluated on the given main and mask rows.
static E left_side(const Row &row, const Table &main, const Table &mask) {
  E sum = E::zero();
  for (const Term &t : row.terms) sum += table_dot(t.part == Part::MAIN ? main : mask, t.v);
  for (const Unit &u : row.units) sum += u.coeff * (u.part == Part::MAIN ? main : mask).entry(u.index);
  for (const MainForm &f : row.forms) {  // the claims on the keys: product forms only
    if (!f.c.entries.empty()) std::abort();
    for (const Tensor &v : f.c.tensors) sum += table_dot(main, v);
  }
  if (!row.runs.empty()) std::abort();  // LKP and this test make none
  return sum;
}
// The names of the verifier's claims that are false on what the prover has sent.
static std::string false_rows(const Seen &prover, const Seen &verifier) {
  std::string names;
  for (const Row &r : verifier.rows)
    if (left_side(r, prover.main, prover.mask) != r.target) names += r.name + "; ";
  return names;
}
[[maybe_unused]] static bool same_rows(const Seen &a, const Seen &b) {
  if (a.rows.size() != b.rows.size()) return false;
  for (size_t i = 0; i < a.rows.size(); i++)
    if (a.rows[i].target != b.rows[i].target || left_side(a.rows[i], a.main, a.mask) != left_side(b.rows[i], a.main, a.mask)) return false;
  return true;
}
static bool key_masks_from_second = false;  // message 1 gives key masks; LKP aborts if unsampled
[[maybe_unused]] static const E *learnt_alpha = nullptr;  // nothing sets or reads it
// A witness is one byte per key, from which prove() picks a row of that key's table; it then
// computes the multiplicities. The public inputs are the same for every witness. number < 1000:
// bytes generated from the number; 1000: every key is row 0; 1001: the last row that prove() uses
// (in tables of 2, 4, 16 or 256 rows); 1002: witness 0 with the byte of key 2 raised by one.
// Exceptions in every witness: padding, and keys 0 and 2^g, which prove() sets to row 1 of
// function 0.
using LkpWitness = std::vector<u8>;
[[maybe_unused]] static LkpWitness lkp_witness(const LkpPublic &fixed, u64 number) {
  LkpWitness picks(fixed.all_keys(), 0);
  ToyCoins data(7000 + (number < 1000 ? number : 0));
  data.bytes(picks.data(), picks.size());
  if (number == 1000) picks.assign(picks.size(), 0);
  if (number == 1001) picks.assign(picks.size(), 254);  // 254 mod (2^log_rows - 1) = 2^log_rows - 2 if log_rows is 1, 2, 4 or 8; else another row
  if (number == 1002) picks[2]++;
  BLK_REQUIRE(number <= 1002, "lkp_world: no such witness");
  return picks;
}
// The honest prover. picks: the witness (`LkpWitness`), from which it takes the keys; it counts
// the multiplicities from these keys.
// coins: the source of all the prover's private randomness (not of the witness). proof: passed in
// so that a caller can call keep_list() on it before the first byte.
// second_unsampled: a second oracle message is sent without a sample, for the tests that LKP aborts
// if it is to use such a message.
static void prove(const LookupRun &run, const LkpWitness &picks, Coins &coins, Writer &proof, Seen &seen, bool second_unsampled = false) {
  const World &w = run.world;
  const LkpPublic &fixed = w.fixed;
  LOG_LEN = run.codes.base[0].log_ell();
  BLK_REQUIRE(picks.size() == fixed.all_keys(), "lkp_world: a witness of another world");
  // Keys before the shift: each a row of its function's table, except at padding (77, 88: sent,
  // not looked up).
  std::vector<E> keys(fixed.all_keys(), E::zero());
  for (size_t i = 0; i < keys.size(); i++) {
    const LkpKeyBlock &b = lkp_block_of(fixed, place_of(fixed, i).second);
    const u8 pick = picks[i];
    if (b.function < 0)
      keys[i] = e_of(77);
    else if (is_padding(fixed, i))
      keys[i] = e_of(88);  // used neither by the prover's trees nor by a claim on the keys
    else
      keys[i] = row_of(table_of(fixed, b.function), pick % ((u64(1) << table_of(fixed, b.function).log_rows) - 1));  // each table's last row stays unused
  }
  keys[0] = keys[u64(1) << fixed.g] = row_of(table_of(fixed, 0), 1);  // row 1 of function 0 is used at least twice
  std::vector<E> counted = keys;  // the values from which the multiplicities are counted
  LkpPublic provers_own = fixed;  // the public inputs with which the prover's trees are built
  const std::vector<E> sent = keys;
  std::vector<E> bits;
  BLK_REQUIRE(lkp_count(fixed, counted, bits), "lkp_world: the test's own keys are not all in their tables");
  const std::vector<E> &really_sent = sent;
  EngineCpu engine;
  Binder binder(7, run.plan);
  const Inputs in = bind(binder, run);
  u8 salt[DIGEST_BYTES];
  coins.bytes(salt, sizeof salt);
  Digest random_bytes;
  digest_from_bytes(salt, random_bytes);
  ProverRounds ch(binder, random_bytes, proof, [&engine](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); });
  ProverContext pr(in.codes, ch, engine, coins);

  Table main(E_DEGREE, 2, run.codes.base[0].ell);
  {
    size_t i = 0;
    for (unsigned part = 0; part < fixed.part_bits().size(); part++)
      for (size_t x = 0; x < size_t(1) << fixed.part_bits()[part]; x++, i++)
        main.set(0, part_first(part) + x, really_sent[i]);
  }
  lay_planes(main, fixed, bits);
  OracleDraft draft;
  draft.shape = oracle_shape(run.set);
  draft.main = engine.put(main);
  draft.mask = Send::fresh_mask(run.codes, draft.shape, coins);
  std::vector<OracleDraft> drafts;
  drafts.push_back(std::move(draft));
  const OracleId o = send_prove(pr, std::move(drafts))[0];
  if (second_unsampled) {
    OracleDraft second;
    second.shape = oracle_shape(run.set);
    second.main = engine.put(main);
    second.mask = Send::fresh_mask(run.codes, second.shape, coins);
    std::vector<OracleDraft> more;
    more.push_back(std::move(second));
    send_oracles(pr, std::move(more));
  }
  OracleId planes_in = o;
  if (run.second_message) {
    OracleDraft second;
    second.shape = oracle_shape(run.set);
    second.main = engine.put(main);
    second.mask = Send::fresh_mask(run.codes, second.shape, coins);
    std::vector<OracleDraft> more;
    more.push_back(std::move(second));
    planes_in = send_prove(pr, std::move(more))[0];
  }
  const GroupId g = pr.st.new_group("keys, multiplicities", 0);
  pr.st.join(g, o);
  if (run.second_message) pr.st.join(g, planes_in);

  LkpCpu heavy(provers_own, keys, bits);
  const u32 before = pr.st.masks(o).left();
  LkpOutput out;
  const LkpPlanesAt planes = planes_at(fixed, planes_in);
  if (table_bound) {
    lkp_prove(pr, in.fixed, heavy, planes, o, g, out, key_masks_from_second ? 1 : NONE_ID);
  } else {
    LkpProverSide side{pr, pr.st.masks(o), pr.st.masks(o), heavy, {}, {}, {}, {}};
    lkp_run(side, fixed, planes, g, out);
  }
  seen.masks_used = before - pr.st.masks(o).left();
  seen.alpha = out.alpha, seen.theta = out.theta;
  seen.main = main, seen.mask = pr.behind[o].mask;
  seen.rows = out.rows;
  for (unsigned part = 0; part < out.keys.parts(); part++) {
    LkpKeys on_keys = out.keys.take(part);
    seen.key_point = on_keys.point;
    Row k = key_row(out.keys.parts() > 1, o, std::move(on_keys), part);
    seen.rows.push_back(k);
    pr.st.add_row(g, std::move(k));
  }
  if (!run.whole) {  // the run stops after LKP: nothing tests the group
    ch.finish();
    return;
  }
  iopp_prove(pr, g, in.stages, in.choice);
  ch.finish();
  pr.st.require_all_tested();
}
// The short form: `key` seeds the prover's randomness, and the witness is number key % 1000.
[[maybe_unused]] static std::vector<u8> prove(const LookupRun &run, u64 key, Seen &seen, bool second_unsampled = false) {
  ToyCoins coins(key);
  Writer proof;
  prove(run, lkp_witness(run.world.fixed, key % 1000), coins, proof, seen, second_unsampled);
  return proof.bytes();
}

// The verifier's side, confined by run_verifier() (blocks/chain/closed.h): decide() gets the
// channel and handles to what the first query contains, not the fixture or the LookupRun. Two
// globals reach it, neither a public input: where it stores its claims for the test, and a
// switch that makes it a faulty caller. It sets `LOG_LEN` from the bound codes.
static Seen *verifier_leaves = nullptr;
static bool claim_is_dropped = false;  // `decide` drops LKP's claim on the keys; LKP must abort
static bool decide(VerifierRounds &ch, const Inputs &in, Notes *) {
  VerifierContext vf(in.codes, ch);
  LOG_LEN = vf.st.codes().base[0].log_ell();
  std::vector<OracleId> ids;
  if (!send_verify(vf, std::vector<OracleNotice>{vf.ch.get(in.oracle)}, ids)) return false;
  const OracleId o = ids[0];
  const GroupId g = vf.st.new_group("keys, multiplicities", 0);
  vf.st.join(g, o);
  LkpOutput out;
  if (!lkp_verify(vf, in.fixed, planes_at(vf.ch.get(in.fixed), o), o, g, out))
    return false;
  if (claim_is_dropped) return true;
  verifier_leaves->rows = out.rows;
  for (unsigned part = 0; part < out.keys.parts(); part++) {
    Row k = key_row(out.keys.parts() > 1, o, out.keys.take(part), part);
    verifier_leaves->rows.push_back(k);
    vf.st.add_row(g, std::move(k));
  }
  if (!iopp_verify(vf, g, in.stages, in.choice)) return false;
  vf.st.require_all_tested();
  return true;
}
static bool verify(const LookupRun &run, const std::vector<u8> &proof, Seen &seen, const char **why = nullptr, bool claim_dropped = false) {
  verifier_leaves = &seen, claim_is_dropped = claim_dropped;
  const char *words = nullptr;
  const bool ok = run_verifier(7, run.plan, run, bind, decide, proof.data(), proof.size(), &words);
  if (why) *why = words ? words : "";
  return ok;
}

}  // namespace lkp_world
