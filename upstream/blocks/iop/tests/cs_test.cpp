// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Tests CS^sigma, the code switch without its batching step (blocks/iop/cs.h): it re-encodes folded
// rows as a new oracle message (encoded rows, sent as Merkle roots), which joins a further group
// (the oracle messages that one proximity test reads). A run has two or three groups. Each but the
// last is closed into the next by BAT (batches its claims into one), FOLD (reduces that claim to
// one on single rows) and CS^sigma; the last gets a proximity test. Both sides run at toy size with
// the real hash chain, Merkle trees and CPU engine. A "claim" is a linear claim on oracle messages
// (`Row` in identifiers, "row" in messages), `Row::target` its value. This test checks that:
//   - honest proofs are accepted, and the claims that a switch leaves hold on the prover's tables;
//   - a false value in any one group is rejected (`refused` in identifiers and messages); so is,
//     for two cases, the proof with one bit flipped in any one byte; misuse (`breach_..`) aborts;
//   - the "free room" after "the rest" (see `Lay`) is 0; no oracle message is opened at more than
//     `draws` positions; pads (random masks of samples' answers) are nonzero and never repeat;
//   - each type of claim on a kept oracle message (`Case::kept`) sees a change the others miss;
//   - switch_rows(), Bat::combine, SwitchPlaces and SwitchBox agree with a second implementation.
// No cheating (malicious) prover runs in this test. The only false claims here are false values handed to the honest prover's code.
#define BLK_PLANS_BY_HAND_FOR_A_TEST  // `toy_plan` builds its plan by hand (blocks/numbers/plan.h)
#include <map>
#include "chain/closed.h"
#include "chain/tests/death.h"
#include "iop/tests/iopp_support.h"

using namespace blk;

// The types of claim ("kind of row" in messages) that a switch leaves, as bits. OLD: the carried
// claim. OLD_SAMPLE: a kept oracle message's sample claim ("this form of the message has the
// sample's answer") from before the switch. SAMPLE: the sample claims of the new oracle message and
// of the re-sent kept ones.
// GEN_..: generator claims (a symbol opened in an old row, as a linear form of the new messages) on
// the single rows: the folded main row, a kept row, the old mask row, the fresh mask row.
enum : unsigned { OLD = 1, SAMPLE = 2, GEN_MAIN = 4, GEN_KEPT = 8, GEN_MASK = 16, GEN_FRESH = 32, OLD_SAMPLE = 64 };

// Where a switch puts a group's folded rows in the new oracle message. `SwitchTarget` has a `Cut`
// (a block of rows) for each of the two parts:
//   MAIN_ROW    the folded main row's message in one main row, all else that is carried ("the
//               rest") in a second main row; no mask rows
//   MASK_ROWS   the rest in mask rows, as between the stages of a proximity test
//   FREE_ROOM   as MAIN_ROW, but the folded message is shorter than the row that takes it
// Entries of a `Cut` that its part leaves unused are "free room": in CS^sigma the honest prover has
// 0 there, and no claim reads them but the new oracle message's own sample claims (type SAMPLE).
enum class Lay { MAIN_ROW, MASK_ROWS, FREE_ROOM };
// draws: the number of positions drawn each time. answers: the number of out-of-domain samples
// (random linear forms of a whole message), each with one answer, of each sampled oracle message.
// kept: the first group has a kept oracle message ("kept member" in messages), of two rows over the
// field F (E is its cubic extension). Its rows are kept rows: FOLD leaves them as sent, and the
// switch sends them again. In the last group they are ordinary rows.
struct Case {
  const char *name;
  Lay lay;
  unsigned draws, answers;
  unsigned groups;  // 2 or 3
  bool kept;
};

constexpr size_t KEPT_ROWS = 2;

struct Link {  // a group; its layout has 2^k main rows, not counting kept rows, and 2^h mask rows
  CodeId code = 0, mask_code = 0;
  unsigned k = 0, h = 0;
  OracleNotice fresh;   // its fresh oracle message
  bool keeps = false;   // the kept rows are kept rows here; else ordinary rows (if there are any)
  SwitchTarget target;  // of the switch into this group; not used in the first
};
// The test fixture of a `Case` (toy_world()).
struct World {
  Codes codes;
  std::vector<Link> link;
  OracleNotice born;              // the oracle message whose rows are kept
  std::vector<IoppStage> stages;  // of the proximity test (`iopp` in identifiers) on the last group
  IoppChoice choice;
};

static World toy_world(const Case &t) {
  World w;
  w.choice.masks_ride = true;
  // `carried`: the length of the rest: two mask rows' messages, the random tails, the old pads.
  const size_t r = tail_for_positions(t.draws), ell_zk = 8;
  const u64 carried = 2 * ell_zk + 3 * r + (t.kept ? KEPT_ROWS * r + E_DEGREE * t.answers : 0);
  const size_t ell = (size_t)pow2_from(carried + t.answers);  // a row fits the rest and the pad
  w.codes.mask.push_back(toy_code(ell_zk, r, 0));
  w.codes.base.push_back(toy_code(ell, r, w.codes.mask[0].log_n));
  w.link.resize(t.groups);
  if (t.lay == Lay::MASK_ROWS) {  // two main rows of half the length, two mask rows
    w.codes.mask.push_back(toy_code((size_t)pow2_from((carried + t.answers + 1) / 2), r, 0));
    w.codes.base.push_back(toy_code(ell / 2, r, w.codes.mask[1].log_n));
  }
  if (t.lay == Lay::FREE_ROOM) w.codes.base.push_back(toy_code(ell / 4, r, w.codes.mask[0].log_n)), w.link[0].code = 1;
  for (unsigned i = 0; i < t.groups; i++) {
    Link &l = w.link[i];
    l.keeps = t.kept && i + 1 < t.groups;
    if (i > 0 && t.lay == Lay::MASK_ROWS) {
      l.code = l.mask_code = 1;
      l.target = switch_target_stage("carried", 1, 1, 1, 1, t.answers);
    } else if (i > 0) {
      l.target.shape.name = "carried", l.target.shape.main_deg = E_DEGREE, l.target.shape.answers = t.answers;
      l.target.shape.code = 0, l.target.shape.main_rows = 2;
      l.target.main = Cut{Part::MAIN, 0, 0};
      l.target.rest = Cut{Part::MAIN, 0, 1};
    }
    OracleShape &f = l.fresh.shape;
    f.name = "fresh", f.code = l.code, f.main_rows = i ? 3 : 4, f.main_deg = 1, f.mask_rows = 1, f.mask_code = l.mask_code;
    f.answers = t.answers;
    // 2^k main rows: a block of 4 for the fresh oracle message, 2 for the carried one, 2 for kept
    // rows that have become ordinary rows.
    l.k = i == 0 ? 2 : 3;
    l.h = i > 0 && t.lay == Lay::MASK_ROWS ? 2 : 0;  // mask rows: 1 (fresh), 2 (carried)
  }
  OracleShape &b = w.born.shape;
  b.name = "kept", b.code = w.link[0].code, b.main_rows = KEPT_ROWS, b.main_deg = 1, b.answers = t.answers;
  if (t.kept)
    for (unsigned i = 1; i < t.groups; i++) {
      KeptTarget k;
      k.shape = b, k.shape.code = w.link[i].code, k.kept_again = w.link[i].keeps;
      w.link[i].target.kept.push_back(k);
    }
  const Link &last = w.link.back();
  w.stages.push_back(IoppStage{last.code, last.mask_code, last.k, last.h});
  toy_later_stages(w.codes, w.stages, {{2, 1}, {0, 0}}, t.draws, t.answers);
  return w;
}

// The rounds of a run. The prover sends the values of each group's three own claims (toy_rows()). A
// switch's prover message has the carried oracle message, any re-sent kept one, the next group's
// fresh one and the mask of that group's fold. `rows` counts the present group's claims: its own,
// the sample claims, and those that a switch leaves.
static Plan toy_plan(const Case &t, const World &w) {
  Plan p = toy_empty_plan(t.answers);
  p.send_root("first");
  std::vector<OracleShape> sampled = {w.link[0].fresh.shape};
  if (t.kept) sampled.push_back(w.born.shape);
  toy_plan_sample(p, w.codes, sampled, t.answers);
  p.send_scalars("targets", 3);
  u64 rows = 3 + (u64)t.answers * sampled.size();
  for (unsigned i = 0; i + 1 < t.groups; i++) {
    toy_plan_bat(p, rows);
    toy_plan_fold(p, group_as_run(w.codes, w.link[i].code, w.link[i].mask_code, true, w.link[i].k, w.link[i].h, w.link[i].keeps ? u64(1) << ceil_log2(1 + KEPT_ROWS) : 1), i == 0);
    p.send_root("carried, kept, fresh, fold mask");
    sampled = {w.link[i + 1].target.shape};
    for (const KeptTarget &k : w.link[i + 1].target.kept) sampled.push_back(k.shape);
    sampled.push_back(w.link[i + 1].fresh.shape);
    toy_plan_sample(p, w.codes, sampled, t.answers, false);
    toy_plan_positions(p, R_LOC, t.draws, w.codes.base[w.link[i].code].log_n, 0, group_as_run(w.codes, w.link[i].code, w.link[i].mask_code, true, 0, 0, 1));
    p.send_scalars("targets", 3);
    rows = 3 + (u64)t.answers * sampled.size() + 1 + (u64)t.answers * w.link[i + 1].target.kept.size() + (u64)t.draws * (3 + (w.link[i].keeps ? KEPT_ROWS : 0));
  }
  toy_plan_iopp(p, w.codes, w.stages, t.draws, w.choice, true, rows);
  return p;
}

static FoldShape link_shape(const Link &l, GroupLayout layout) {
  return iopp_fold_shape(IoppStage{l.code, l.mask_code, l.k, l.h}, std::move(layout));
}

struct Proved {
  std::vector<u8> proof;
  size_t trees = 0;
  std::vector<unsigned> broken;  // per switch between groups: types false on the prover's tables
  std::vector<std::pair<unsigned, unsigned>> tie;  // per `tie_probe` call: `blind`, the result
  bool tables_fine = true;       // free room is 0, pads are not, openings within `draws`
  bool rows_hold() const {
    for (unsigned b : broken)
      if (b) return false;
    return true;
  }
};

// A run's public inputs, bound in the hash chain's first query. Shapes, cuts and switch
// targets follow from `the_case` (toy_world()), the coefficients of the groups' own claims from
// `seed`.
struct Bound_ {
  Public<std::vector<u64>> the_case;
  Public<Codes> codes;
  Public<std::vector<IoppStage>> stages;
  Public<IoppChoice> choice;
  Public<u64> seed;
};
static Bound_ bind(Binder &b, const Case &t, const World &w, u64 seed) {
  Bound_ p;
  b.configuration(toy_configuration());
  p.the_case = b.parameter("case", std::vector<u64>{(u64)t.lay, t.draws, t.answers, t.groups, (u64)t.kept});
  p.codes = b.parameter("codes", w.codes), p.stages = b.parameter("stages", w.stages), p.choice = b.parameter("choice", w.choice);
  p.seed = b.statement("seed of the rows", seed);
  return p;
}

// Injected faults (`planted` in messages) for the death tests in main(): with each, the process
// must abort with a given message (ends_with()).
// After the first switch whose rest is in main rows, the prover adds a claim whose coefficient
// vector only its engine knows (`EngineTerm`, blocks/iop/tensor.h), placed by one of the switch
// target's two cuts.
// 1: the rest's cut. Its rows contain the new oracle message's pad, so Statement::add_row(), which
//    checks that no claim reads a pad, asks for a coefficient inside the term. The process must
//    abort there, not invent an answer.
// 2, the control: the main row's cut, as for every carried claim. The pad is outside the term,
//    where the coefficient is 0 by definition: the claim is accepted.
static int breach_engine_term = 0;
// The value that `SingleRows::masks`, the number of mask rows, gets before a switch, on either
// side. 2 is what a fold leaves: no fault.
static unsigned breach_mask_rows = 2;
// After the first switch whose rest is in mask rows, the prover asks the new oracle message's mask
// rows for a free mask entry. They hold what the switch carries, so none is free.
static bool breach_takes_a_mask = false;
// A switch's target describes the kept oracle messages otherwise than the old group has them, on
// either side. 0: no fault. 1: one `KeptTarget` fewer. 2: a mask row in one. 3: a main row more.
static int breach_kept_target = 0;
static SwitchTarget breached(SwitchTarget T) {
  if (breach_kept_target == 1) T.kept.pop_back();
  if (breach_kept_target == 2) T.kept[0].shape.mask_rows = 1;
  if (breach_kept_target == 3) T.kept[0].shape.main_rows++;
  return T;
}

// Shows what each type of claim on a kept oracle message is needed for (the argument is in
// blocks/iop/cs.h). After a switch, on the prover's tables only, it adds to row 0 of the re-sent
// oracle message `o` a vector over F that the claims of the types in `blind` do not see, and
// returns the types of the claims that are then false. `all`: every claim on the new oracle
// message and the re-sent kept ones, each with its type.
// No prover could make all of these changes: one that the generator claims do not see needs the
// positions in advance. That lucky prover is what the old sample claim is for, and no run can
// produce it: the chance that the generator claims miss it falls exponentially in `draws`.
// The vector is a kernel vector, found by elimination. A claim's coefficient at an entry is
// measured as the change in row_value() when 1 is added there, whatever form cs.h gives its claims.
static unsigned tie_probe(ProverContext &P, Engine &engine, const Codes &codes, const std::vector<std::pair<const Row *, unsigned>> &all, OracleId o, unsigned blind) {
  const TableId honest = P.behind[o].main;
  const Table table = engine.get(honest);
  std::vector<u64> at;  // the entries that may change: row 0, off the pad
  for (u64 place = 0; place < table.len; place++) {
    const u64 index = place << table.log_rows;
    bool pad = false;
    for (const PadPlace &p : Send::pad(codes, P.st.oracle(o).shape)) pad = pad || p.index == index;
    if (!pad) at.push_back(index);
  }
  std::vector<const Row *> unseen;
  for (const auto &r : all)
    if (r.second & blind) unseen.push_back(r.first);
  // A generator claim has coefficients in F and gives one condition over F, each other claim three
  // (2 * 8 allows for eight of those): so this many entries are enough for a kernel vector.
  at.resize(std::min(at.size(), unseen.size() + 2 * 8 + 1));
  std::vector<E> before;
  for (const Row *r : unseen) before.push_back(row_value(P, *r));
  // One line for each coordinate over F of each claim's value, one column for each entry.
  std::vector<std::vector<F>> lines(E_DEGREE * unseen.size(), std::vector<F>(at.size(), F::zero()));
  for (size_t j = 0; j < at.size(); j++) {
    P.behind[o].main = engine.with_entries(honest, {{at[j], table.entry(at[j]) + E::one()}});
    for (size_t q = 0; q < unseen.size(); q++) {
      const E c = row_value(P, *unseen[q]) - before[q];
      for (unsigned k = 0; k < E_DEGREE; k++) lines[E_DEGREE * q + k][j] = c.c[k];
    }
    engine.drop(P.behind[o].main);
  }
  std::vector<size_t> pivot_of;  // the columns with a pivot, in the order of their lines
  size_t line = 0;
  for (size_t j = 0; j < at.size() && line < lines.size(); j++) {
    size_t with = line;
    while (with < lines.size() && is_zero(lines[with][j])) with++;
    if (with == lines.size()) continue;
    std::swap(lines[line], lines[with]);
    const F scale = inv(lines[line][j]);
    for (F &x : lines[line]) x *= scale;
    for (size_t other = 0; other < lines.size(); other++)
      if (other != line && !is_zero(lines[other][j])) {
        const F times = lines[other][j];
        for (size_t c = 0; c < at.size(); c++) lines[other][c] -= times * lines[line][c];
      }
    pivot_of.push_back(j), line++;
  }
  size_t free_column = 0;  // the first column without a pivot: the kernel vector is 1 there
  while (free_column < at.size() && std::find(pivot_of.begin(), pivot_of.end(), free_column) != pivot_of.end()) free_column++;
  BLK_REQUIRE(free_column < at.size(), "a test: the rows see every change of the message's row: it is too short for this probe");
  std::vector<std::pair<u64, E>> changed = {{at[free_column], table.entry(at[free_column]) + E::one()}};
  for (size_t l = 0; l < pivot_of.size(); l++)
    if (!is_zero(lines[l][free_column])) changed.push_back({at[pivot_of[l]], table.entry(at[pivot_of[l]]) - E::from(lines[l][free_column])});
  P.behind[o].main = engine.with_entries(honest, changed);
  unsigned broken = 0;
  for (const auto &r : all)
    if (row_value(P, *r.first) != r.first->target) broken |= r.second;
  engine.drop(P.behind[o].main);
  P.behind[o].main = honest;
  return broken;
}


// ========== A second implementation of what a switch leaves and of what BAT combines ==========
// Both sides build a switch's claims with one function, switch_rows() (blocks/iop/cs.h), and a run
// pins only coefficients that meet a random value. It cannot see a coefficient on an entry that is
// 0 in honest runs (free room), a claim left out on both sides, or two claims batched under one
// power. The first was seen: with every generator claim on the main row also reading the first free
// entry, on both sides, this test without the `my_` functions passed; they found 180 differences.
// The `my_` functions compare a switch's claims, on every entry of every table, with claims written
// out densely (generator matrices by encoding unit vectors, indices by separate arithmetic), and
// Bat::combine with the plain sum of powers, on the groups' claims and on synthetic ones (`made-up`
// in messages). They call no switch_..() function and run only in honest proofs (`my_on`).
// Messages call them `MY` and `the second text`.
// No test of this repository builds this test from a copy of blocks/ with that free-room fault, or
// with every claim under the power 1 in Bat::combine, to show that it then fails.
// `RunInputCheat`, a friend of `RunInput`, lets my_batch_check() batch any list of claims.
namespace blk { struct RunInputCheat { static RunInput make(GroupLayout L, std::vector<Row> rows) { RunInput in; in.layout = std::move(L); in.rows_ = std::move(rows); return in; } }; }
static bool my_on = false;
static size_t my_rows_checked = 0, my_entries_checked = 0, my_free_checked = 0, my_batches = 0, my_bad = 0;
// A claim written out: (oracle message, 0 = main table or 1 = mask table) -> its coefficients.
using Dense = std::map<std::pair<OracleId, int>, std::vector<E>>;
// A code's generator matrix, by encoding unit vectors: j < ell is a message entry, then the tail.
static std::vector<std::vector<F>> my_matrix(const Code &c) {  // [position x][coefficient j]
  std::vector<std::vector<F>> G(c.n(), std::vector<F>(c.ell + c.r, F::zero()));
  for (size_t j = 0; j < c.ell + c.r; j++) {
    std::vector<F> m(c.ell, F::zero()), t(c.r, F::zero()), w(c.n());
    (j < c.ell ? m[j] : t[j - c.ell]) = F::one();
    encode(c, m.data(), t.data(), w.data());
    for (u64 x = 0; x < c.n(); x++) G[x][j] = w[x];
  }
  return G;
}
static void my_dense_check(const Statement &S, const Codes &codes, const SingleRows &in, const SwitchTarget &T, const Switched &s, GroupId g) {
  const std::vector<Row> &left = S.group(g).rows;
  const Code &code = codes.base[in.code], &zk = codes.mask[in.mask_code];
  const auto G = my_matrix(code), Z = my_matrix(zk);
  auto blank = [&]() {
    Dense d;
    d[{s.g, 0}].assign(size_t(1) << S.table_bits(s.g, Part::MAIN), E::zero());
    if (S.has(s.g, Part::MASK)) d[{s.g, 1}].assign(size_t(1) << S.table_bits(s.g, Part::MASK), E::zero());
    for (OracleId o : s.kept) d[{o, 0}].assign(size_t(1) << S.table_bits(o, Part::MAIN), E::zero());
    return d;
  };
  // Where each part is stored. `place`: the rest's entries in order, pads left out.
  const unsigned rb_main = ceil_log2(T.shape.main_rows);
  auto main_index = [&](u64 j) { return (T.main.first_row + j % (u64(1) << T.main.bits)) + ((j >> T.main.bits) << rb_main); };
  const int rp = T.rest.part == Part::MAIN ? 0 : 1;
  const unsigned rb_rest = ceil_log2(rp ? T.shape.mask_rows : T.shape.main_rows);
  const u64 len_rest = rp ? codes.mask[T.shape.mask_code].ell : codes.base[T.shape.code].ell;
  std::vector<u64> place;
  for (u64 j = 0; j < (len_rest << T.rest.bits); j++) {
    const u64 index = (T.rest.first_row + j % (u64(1) << T.rest.bits)) + ((j >> T.rest.bits) << rb_rest);
    bool pad = false;
    for (const auto &p : S.oracle(s.g).pad) pad = pad || (p.first == T.rest.part && p.second == index);
    if (!pad) place.push_back(index);
  }
  std::vector<OracleId> old_kept;
  for (size_t i = 0; i < in.layout.members.size(); i++) if (in.layout.kept[i]) old_kept.push_back(in.layout.members[i]);
  // Offsets in the rest: mask rows' messages; tails of the main, kept and mask rows; old pads.
  size_t at = 0;
  const size_t mask_at[2] = {0, zk.ell};
  at = 2 * zk.ell;
  const size_t main_tail = at; at += code.r;
  std::vector<size_t> kept_tail;
  for (OracleId o : old_kept) for (size_t b = 0; b < S.oracle(o).shape.main_rows; b++) kept_tail.push_back(at), at += code.r;
  const size_t mask_tail[2] = {at, at + zk.r};
  at += 2 * zk.r;
  std::vector<size_t> old_pad;
  for (OracleId o : old_kept) old_pad.push_back(at), at += S.oracle(o).pad.size();
  const size_t used = at;
  // Adds a dense vector on an old kept table to d; its pad entries go to the rest.
  auto carry_kept = [&](Dense &d, size_t i, const std::vector<E> &v) {
    const auto &pad = S.oracle(old_kept[i]).pad;
    for (u64 j = 0; j < v.size(); j++) {
      size_t is_pad = pad.size();
      for (size_t q = 0; q < pad.size(); q++) if (pad[q].second == j) is_pad = q;
      if (is_pad == pad.size()) d[{s.kept[i], 0}][j] += v[j];
      else d[{s.g, rp}][place[old_pad[i] + is_pad]] += v[j];
    }
  };
  // In the order of switch_rows(): carried claim, old sample claims, each draw's generator claims.
  std::vector<Dense> want;
  {
    Dense d = blank();
    std::vector<E> c(code.ell, E::zero());
    in.c.add_to(c.data(), code.log_ell());
    for (u64 j = 0; j < code.ell; j++) d[{s.g, 0}][main_index(j)] += c[j];
    for (const KeptForm &f : in.kept) {
      size_t i = 0; while (old_kept[i] != f.oracle) i++;
      std::vector<E> v(size_t(1) << S.table_bits(f.oracle, Part::MAIN), E::zero());
      f.c.add_to(v.data(), S.table_bits(f.oracle, Part::MAIN));
      carry_kept(d, i, v);
    }
    for (unsigned j = 0; j < 2; j++) for (size_t q = 0; q < in.d[j].size(); q++) d[{s.g, rp}][place[mask_at[j] + q]] += in.d[j][q];
    CHECK(in.d[0].size() == zk.ell && in.d[1].size() == zk.ell, "MY: d has another length than a mask row's message");
    want.push_back(std::move(d));
  }
  for (size_t i = 0; i < old_kept.size(); i++)
    for (const Row &sample : S.oracle(old_kept[i]).sample_rows) {
      Dense d = blank();
      std::vector<E> v(size_t(1) << S.table_bits(old_kept[i], Part::MAIN));
      for (u64 j = 0; j < v.size(); j++) v[j] = sample.coefficient(old_kept[i], Part::MAIN, j);
      carry_kept(d, i, v);
      want.push_back(std::move(d));
    }
  for (u64 x : s.at.draws()) {
    {
      Dense d = blank();
      for (u64 j = 0; j < code.ell; j++) d[{s.g, 0}][main_index(j)] += E::from(G[x][j]);
      for (u64 j = 0; j < code.r; j++) d[{s.g, rp}][place[main_tail + j]] += E::from(G[x][code.ell + j]);
      want.push_back(std::move(d));
    }
    size_t row = 0;
    for (size_t i = 0; i < old_kept.size(); i++) {
      const unsigned rb = ceil_log2(S.oracle(old_kept[i]).shape.main_rows);
      for (size_t b = 0; b < S.oracle(old_kept[i]).shape.main_rows; b++, row++) {
        Dense d = blank();
        std::vector<E> v(size_t(1) << S.table_bits(old_kept[i], Part::MAIN), E::zero());
        for (u64 j = 0; j < code.ell; j++) v[b + (j << rb)] = E::from(G[x][j]);
        carry_kept(d, i, v);
        for (u64 j = 0; j < code.r; j++) d[{s.g, rp}][place[kept_tail[row] + j]] += E::from(G[x][code.ell + j]);
        want.push_back(std::move(d));
      }
    }
    for (unsigned m = 0; m < 2; m++) {
      Dense d = blank();
      const u64 xz = x % zk.n();
      for (u64 j = 0; j < zk.ell; j++) d[{s.g, rp}][place[mask_at[m] + j]] += E::from(Z[xz][j]);
      for (u64 j = 0; j < zk.r; j++) d[{s.g, rp}][place[mask_tail[m] + j]] += E::from(Z[xz][zk.ell + j]);
      want.push_back(std::move(d));
    }
  }
  CHECK(want.size() == left.size(), "MY: another number of rows than my list has");
  for (size_t q = 0; q < want.size() && q < left.size(); q++) {
    const Row &r = left[q];
    auto known = [&](OracleId o, Part p) { if (!want[q].count({o, p == Part::MAIN ? 0 : 1})) my_bad++, std::printf("MY: row %zu names a table outside g and the kept messages\n", q); };
    for (const Term &t : r.terms) known(t.oracle, t.part);
    for (const Unit &u : r.units) known(u.oracle, u.part);
    for (const Run &u : r.runs) known(u.oracle, u.part);
    for (const MainForm &f : r.forms) known(f.oracle, Part::MAIN);
    for (const auto &kv : want[q])
      for (u64 j = 0; j < kv.second.size(); j++) {
        my_entries_checked++;
        if (!(r.coefficient(kv.first.first, kv.first.second ? Part::MASK : Part::MAIN, j) == kv.second[j])) {
          if (my_bad++ < 10) std::printf("MY: row %zu (%s), oracle %u part %d entry %llu differs\n", q, r.name.c_str(), kv.first.first, kv.first.second, (unsigned long long)j);
        }
      }
    for (size_t f = used; f < place.size(); f++) my_free_checked++;
    my_rows_checked++;
  }
}
// Compares Bat::combine with sum_i theta^i rows[i], computed densely on the group's tables.
static void my_batch_check(const Statement &S, GroupId g, std::vector<Row> rows, const E &theta, const char *what) {
  const GroupLayout L = S.layout(g);
  std::vector<E> c(size_t(1) << (L.k + L.log_ell), E::zero()), d(size_t(1) << (L.h + L.log_ell_zk), E::zero());
  std::map<OracleId, std::vector<E>> kept;
  E target = E::zero(), power = E::one();
  for (const Row &r : rows) {
    for (size_t i = 0; i < L.members.size(); i++) {
      const OracleId o = L.members[i];
      if (S.has(o, Part::MAIN)) {
        const unsigned a = S.row_bits(o, Part::MAIN);
        const u64 n = u64(1) << S.table_bits(o, Part::MAIN);
        if (L.kept[i] && !kept.count(o)) kept[o].assign(n, E::zero());
        for (u64 j = 0; j < n; j++) {
          const E v = power * r.coefficient(o, Part::MAIN, j);
          if (L.kept[i]) kept[o][j] += v;
          else c[(L.main_first[i] + (j & ((u64(1) << a) - 1))) + ((j >> a) << L.k)] += v;
        }
      }
      if (S.has(o, Part::MASK)) {
        const unsigned a = S.row_bits(o, Part::MASK);
        for (u64 j = 0; j < u64(1) << S.table_bits(o, Part::MASK); j++)
          d[(L.mask_first[i] + (j & ((u64(1) << a) - 1))) + ((j >> a) << L.h)] += power * r.coefficient(o, Part::MASK, j);
      }
    }
    target += power * r.target;
    power *= theta;
  }
  const GroupRow out = Bat::combine(S, g, RunInputCheat::make(L, std::move(rows)), theta);
  std::vector<E> got(c.size(), E::zero());
  out.c.add_to(got.data(), L.k + L.log_ell);
  bool same = got == c && out.d == d && out.target == target;
  for (const KeptForm &f : out.kept) {
    std::vector<E> v(size_t(1) << S.table_bits(f.oracle, Part::MAIN), E::zero());
    f.c.add_to(v.data(), S.table_bits(f.oracle, Part::MAIN));
    same = same && kept.count(f.oracle) && kept[f.oracle] == v;
    kept.erase(f.oracle);
  }
  for (const auto &kv : kept) for (const E &e : kv.second) same = same && is_zero(e);
  // the same vector fixed at a point, as a fold does it
  {
    TestCoins pt(77);
    std::vector<E> gam(L.k);
    for (E &x : gam) x = pt.e();
    Form f = out.c;
    const E eps = pt.e();
    f.fix_low(gam.data(), L.k, eps);
    std::vector<E> low(size_t(1) << L.log_ell, E::zero()), mine(low.size(), E::zero());
    f.add_to(low.data(), L.log_ell);
    for (u64 j = 0; j < c.size(); j++) {
      E w = eps;
      for (unsigned t = 0; t < L.k; t++) w = w * (((j >> t) & 1) ? gam[t] : E::one() - gam[t]);
      mine[j >> L.k] += w * c[j];
    }
    same = same && low == mine;
  }
  my_batches++;
  if (!same) my_bad++, std::printf("MY: Bat::combine differs from the plain sum: %s\n", what);
}
// Synthetic claims on a group's members. `Run`s (a vector on entries first, first + step, ..) that
// share one vector, have the same start and step, overlap, or have one entry; several `Unit`s (one
// coefficient each) on one entry; `Tensor`s and `Form`s; empty and repeated claims.
static std::vector<Row> my_random_rows(const Statement &S, GroupId g, u64 seed, size_t count) {
  TestCoins data(seed);
  const std::vector<OracleId> members = S.group(g).members;
  std::vector<Row> rows;
  auto shared = std::make_shared<std::vector<F>>(23);
  data.fill(shared->data(), shared->size());
  for (size_t q = 0; q < count; q++) {
    Row r;
    r.target = data.e();
    const unsigned pieces = (unsigned)(data.word() % 6);  // 0: an empty claim
    for (unsigned p = 0; p < pieces; p++) {
      const OracleId o = members[data.word() % members.size()];
      const Part part = S.has(o, Part::MASK) && (!S.has(o, Part::MAIN) || data.word() % 2) ? Part::MASK : Part::MAIN;
      const u64 n = u64(1) << S.table_bits(o, part);
      switch (data.word() % 7) {
        case 0: case 1: {  // a range of the shared vector, with one of a few starts and steps
          const u64 step = u64(1) << (data.word() % 3), first = data.word() % 3;
          size_t len = 1 + data.word() % 7, from = data.word() % 4;
          while (first + step * (len - 1) >= n) len--;
          r.add_run(o, part, first, step, shared, from, len);
          break;
        }
        case 2: {
          const u64 step = 1 + data.word() % 3, first = data.word() % 5;
          size_t len = 1 + data.word() % 7;
          while (first + step * (len - 1) >= n) len--;
          std::vector<E> own(len);
          for (E &x : own) x = data.word() % 3 ? data.e() : E::from(F::one());
          r.add_run(o, part, first, step, own);
          break;
        }
        case 3: r.add_unit(o, part, data.word() % std::min<u64>(n, 4), data.e()); break;
        case 4: r.add_unit(o, part, data.word() % n, data.word() % 2 ? E::zero() : data.e()); break;
        case 5: {
          std::vector<E> z(S.table_bits(o, part));
          for (E &x : z) x = data.word() % 4 ? data.e() : (data.word() % 2 ? E::one() : E::zero());
          r.add(o, part, Tensor::eq(z.data(), (unsigned)z.size(), data.e()));
          break;
        }
        default: {
          if (!S.has(o, Part::MAIN)) break;
          Form f;
          std::vector<F> z(S.table_bits(o, Part::MAIN));
          data.fill(z.data(), z.size());
          f.add(Tensor::product(z.data(), (unsigned)z.size(), data.e()));
          f.add_entry(data.word() % (u64(1) << S.table_bits(o, Part::MAIN)), data.e());
          f.add_entry(3, data.e());
          if (data.word() % 2) f.freeze(S.table_bits(o, Part::MAIN));
          r.add_form(o, f);
        }
      }
    }
    rows.push_back(r);
    if (data.word() % 5 == 0) rows.push_back(rows[data.word() % rows.size()]);  // a claim once more
  }
  return rows;
}
// The group's claims in the order of Statement::begin_run(): plain, sample, generator.
static std::vector<Row> my_run_order(const Statement &S, GroupId g) {
  std::vector<Row> rows;
  for (const Row &r : S.group(g).rows) if (r.kind == RowKind::PLAIN) rows.push_back(r);
  for (OracleId o : S.group(g).members) for (const Row &r : S.oracle(o).sample_rows) rows.push_back(r);
  for (const Row &r : S.group(g).rows) if (r.kind == RowKind::GENERATOR) rows.push_back(r);
  return rows;
}
static void my_batches_on(const Statement &S, GroupId g, u64 seed) {
  TestCoins th(seed);
  my_batch_check(S, g, my_run_order(S, g), th.e(), "the group's own rows");
  my_batch_check(S, g, {}, th.e(), "no rows");
  my_batch_check(S, g, my_random_rows(S, g, seed + 1, 1), th.e(), "one made-up row");
  for (u64 i = 0; i < 3; i++) my_batch_check(S, g, my_random_rows(S, g, seed + 2 + i, 40), th.e(), "forty made-up rows");
  std::vector<Row> both = my_run_order(S, g);
  for (Row &r : my_random_rows(S, g, seed + 9, 30)) both.push_back(r);
  my_batch_check(S, g, both, th.e(), "the group's rows and thirty made-up ones");
  my_batch_check(S, g, both, E::zero(), "theta = 0");
  my_batch_check(S, g, both, E::one(), "theta = 1");
}
// ===================== end of the second implementation =====================

// The prover. The first toy_rows() claim on group false_at (from 0) gets a false value; ~0u: none.
static Proved prove(const Case &t, u64 seed, unsigned false_at) {
  const World w = toy_world(t);
  const Plan plan = toy_plan(t, w);
  TestCoins coins(toy_coins_seed(seed)), data(seed + 1);
  EngineCpu engine;
  Writer proof;
  Binder binder(1, plan);
  const Bound_ bound = bind(binder, t, w, seed);
  ProverRounds ch(binder, toy_random_bytes(seed + 2), proof);
  ProverContext P(bound.codes, ch, engine, coins);
  P.pads = toy_pads;
  Proved out;
  auto add_rows = [&](unsigned i, GroupId g, const std::vector<OracleId> &on) {
    bool head = true;
    for (Row &row : toy_rows(P.st, on, ch.get(bound.seed) + 10 + i)) {
      row.target = row_value(P, row) + (head && i == false_at ? E::one() : E::zero());
      head = false;
      P.ch.send(row.target);  // the hash chain binds each value before the claims are batched
      P.st.add_row(g, std::move(row));
    }
  };

  std::vector<OracleDraft> drafts;
  drafts.push_back(toy_draft(w.codes, w.link[0].fresh, engine, coins, data));
  if (t.kept) drafts.push_back(toy_draft(w.codes, w.born, engine, coins, data));
  std::vector<OracleId> on = send_prove(P, std::move(drafts));
  GroupId g = P.st.new_group("group 1", w.link[0].code, w.link[0].mask_code);
  P.st.join(g, on[0]);
  if (t.kept) P.st.join(g, on[1], true);
  add_rows(0, g, on);

  FoldMask mask;
  for (unsigned i = 0; i + 1 < t.groups; i++) {
    const Link &next = w.link[i + 1];
    GroupLayout L;
    if (my_on) my_batches_on(P.st, g, seed + 1000 + i);
    const GroupRow row = bat_prove(P, g, &L);
    const FoldShape F = link_shape(w.link[i], L);
    if (i == 0) mask = fold_mask_make(F, coins);
    SingleRows rows;
    SingleRowsWitness W;
    fold_prove(P, F, row, std::move(mask), rows, W);
    if (breach_mask_rows != 2) rows.masks = breach_mask_rows;

    g = P.st.new_group("next group", next.code, next.mask_code);
    const FoldShape N = link_shape(next, GroupLayout());
    mask = fold_mask_make(N, coins);
    std::vector<OracleDraft> with;
    with.push_back(toy_draft(w.codes, next.fresh, engine, coins, data));
    with.push_back(fold_mask_draft(w.codes, N, mask));
    const Switched s = cs_sigma_prove(P, rows, W, breach_kept_target ? breached(next.target) : next.target, g, std::move(with));
    if (my_on) my_dense_check(P.st, w.codes, rows, next.target, s, g);
    if (breach_takes_a_mask && next.target.rest.part == Part::MASK) {
      (void)P.st.masks(s.g).take();
      BLK_REQUIRE(false, "a test: an entry was handed out");
    }
    if (breach_engine_term && next.target.rest.part == Part::MAIN) {
      const Cut &cut = breach_engine_term == 1 ? next.target.rest : next.target.main;
      const unsigned row_bits = switch_row_bits(next.target, Part::MAIN), log_len = w.codes.base[next.target.shape.code].log_ell();
      Form c;
      // Any table will do: the process aborts before the term is evaluated.
      c.engine_terms.push_back(EngineTerm{P.behind[s.g].main, E::one(), 0, 0});
      Row laid;
      laid.name = "a claim that only the engine knows";
      laid.add_form(s.g, switch_lift(std::move(c), log_len + cut.bits, cut, row_bits, row_bits + log_len));  // it fills the cut's block
      P.st.add_row(g, std::move(laid));
      BLK_REQUIRE(false, "the planted row was taken");
    }
    mask.sent_as(s.with[1]);
    engine.drop(W.main);
    release_oracles(P, single_rows_oracles(rows));
    unsigned broken = 0;
    std::vector<std::pair<const Row *, unsigned>> all;  // claims, each with its type
    // What the switch leaves: the carried claim; the kept members' old sample claims; per draw, one
    // generator claim per single row.
    const std::vector<Row> &left = P.st.group(g).rows;
    const size_t old_samples = (size_t)t.answers * s.kept.size();
    CHECK(left.size() == 1 + old_samples + (size_t)t.draws * rows.rows(), "a switch leaves another number of rows than one claim, the kept messages' old sample rows and the generator rows");
    for (size_t q = 0; q < left.size(); q++) {
      const size_t row = q > old_samples ? (q - 1 - old_samples) % rows.rows() : 0;
      all.push_back({&left[q], q == 0                      ? OLD
                               : q <= old_samples          ? OLD_SAMPLE
                               : row == 0                  ? GEN_MAIN
                               : row < rows.mask_row(0)    ? GEN_KEPT
                               : row == rows.mask_row(0)   ? GEN_MASK
                                                           : GEN_FRESH});
    }
    std::vector<OracleId> sampled = s.kept;
    sampled.push_back(s.g);
    for (OracleId o : sampled)
      for (const Row &r : P.st.oracle(o).sample_rows) all.push_back({&r, SAMPLE});
    for (const auto &r : all)
      if (row_value(P, *r.first) != r.first->target) broken |= r.second;
    out.broken.push_back(broken);
    if (t.kept && i == 0)
      for (unsigned seen : {GEN_KEPT, OLD_SAMPLE, OLD, SAMPLE}) {
        const unsigned blind = (OLD | OLD_SAMPLE | GEN_KEPT | SAMPLE) & ~seen;
        out.tie.push_back({blind, tie_probe(P, engine, w.codes, all, s.kept[0], blind)});
      }
    // `tables_fine`: unused entries of the rest's `Cut` are 0, the entries of s.g's pad are not.
    {
      const bool in_main = next.target.rest.part == Part::MAIN;
      const Table table = in_main ? engine.get(P.behind[s.g].main) : P.behind[s.g].mask;
      const SwitchPlaces places(w.codes, next.target);
      for (size_t q = switch_parts(P.st, rows).length; q < places.size(); q++)
        out.tables_fine = out.tables_fine && is_zero(table.entry(places[q]));
      for (const PadPlace &p : Send::pad(w.codes, next.target.shape)) out.tables_fine = out.tables_fine && !is_zero(table.entry(p.index));
    }
    P.st.join(g, s.with[0]);
    on = {s.with[0]};
    on.insert(on.end(), s.kept.begin(), s.kept.end());
    add_rows(i + 1, g, on);
  }
  if (my_on) my_batches_on(P.st, g, seed + 2000);
  iopp_prove(P, g, bound.stages, bound.choice, std::move(mask));
  ch.finish();
  // one list for each oracle message
  for (const ProverOracle &o : P.behind) out.tables_fine = out.tables_fine && o.opened <= t.draws;
  out.proof = proof.bytes();
  out.trees = P.book.trees();
  return out;
}

// The verifier, run by run_verifier() (blocks/chain/closed.h): decide() is passed the channel and
// the handles of the first query's values, not the test's `World`. It rebuilds shapes, cuts and
// switch targets from the case's five numbers as the first query has them.
struct Inputs {
  const Case &t;
  const World &w;
  u64 seed;
};
static Bound_ bind_inputs(Binder &b, const Inputs &in) { return bind(b, in.t, in.w, in.seed); }
static bool decide(VerifierRounds &ch, const Bound_ &bound, Notes *) {
  const std::vector<u64> &c = ch.get(bound.the_case);
  const Case t = {"", (Lay)c.at(0), (unsigned)c.at(1), (unsigned)c.at(2), (unsigned)c.at(3), c.at(4) != 0};
  const World w = toy_world(t);
  VerifierContext V(bound.codes, ch);
  auto add_rows = [&](unsigned i, GroupId g, const std::vector<OracleId> &on) {
    for (Row &row : toy_rows(V.st, on, ch.get(bound.seed) + 10 + i)) {
      if (!V.ch.recv(row.target)) return false;
      V.st.add_row(g, std::move(row));
    }
    return true;
  };

  std::vector<OracleNotice> notices = {w.link[0].fresh};
  if (t.kept) notices.push_back(w.born);
  std::vector<OracleId> on;
  if (!send_verify(V, notices, on)) return false;
  GroupId g = V.st.new_group("group 1", w.link[0].code, w.link[0].mask_code);
  V.st.join(g, on[0]);
  if (t.kept) V.st.join(g, on[1], true);
  if (!add_rows(0, g, on)) return false;

  FoldMask mask;
  for (unsigned i = 0; i + 1 < t.groups; i++) {
    const Link &next = w.link[i + 1];
    GroupLayout L;
    GroupRow row;
    if (!bat_verify(V, g, row, &L)) return false;
    SingleRows rows;
    if (!fold_verify(V, link_shape(w.link[i], L), row, std::move(mask), rows)) return false;
    if (breach_mask_rows != 2) rows.masks = breach_mask_rows;

    g = V.st.new_group("next group", next.code, next.mask_code);
    Switched s;
    if (!cs_sigma_verify(V, rows, breach_kept_target ? breached(next.target) : next.target, g, {next.fresh, fold_mask_notice(link_shape(next, GroupLayout()))}, s)) return false;
    mask = FoldMask();
    mask.sent_as(s.with[1]);
    V.st.join(g, s.with[0]);
    on = {s.with[0]};
    on.insert(on.end(), s.kept.begin(), s.kept.end());
    if (!add_rows(i + 1, g, on)) return false;
  }
  return iopp_verify(V, g, bound.stages, bound.choice, std::move(mask));
}
static bool verify(const Case &t, u64 seed, const std::vector<u8> &proof) {
  const World w = toy_world(t);
  const Plan plan = toy_plan(t, w);
  return run_verifier(1, plan, Inputs{t, w, seed}, bind_inputs, decide, proof.data(), proof.size());
}

// Compares SwitchPlaces, which computes each index on demand, with the explicit list: every entry
// of the rest's `Cut` in fill order (cut_index()), except the pad's. Both call cut_index(): only
// the skipping of the pad's entries is compared.
static void places_as_written(const World &w) {
  for (size_t i = 1; i < w.link.size(); i++) {
    const SwitchTarget &T = w.link[i].target;
    const unsigned a = switch_row_bits(T, T.rest.part);
    const u64 room = (u64)switch_row_length(w.codes, T, T.rest.part) << T.rest.bits;
    std::vector<u64> written;
    for (u64 j = 0; j < room; j++) {
      const u64 index = cut_index(T.rest, a, j);
      bool is_pad = false;
      for (const PadPlace &p : Send::pad(w.codes, T.shape)) is_pad = is_pad || (p.part == T.rest.part && p.index == index);
      if (!is_pad) written.push_back(index);
    }
    const SwitchPlaces places(w.codes, T);
    CHECK(places.size() == written.size(), "the rest's places: another number than the written-out list has");
    bool same = places.size() == written.size();
    for (size_t q = 0; same && q < written.size(); q++) same = places[q] == written[q];
    CHECK(same, "the rest's places: an entry differs from the written-out list");
  }
}

// A row of a mask code's generator matrix as one `Tensor` (blocks/iop/tensor.h) on a box of a
// table's index (`SwitchBox`: 2^vars entries, aligned, a power of two apart), compared entry by
// entry with the explicit row. switch_rows() builds such a `Tensor` on both sides, so no run of the
// protocol would notice a wrong one.
static void boxes() {
  const Code zk{8, 64, 8};
  const unsigned vars = 6;
  for (unsigned shift : {0u, 1u, 3u})
    for (u64 row : {u64(0), (u64(1) << shift) - 1})
      for (size_t first : {size_t(0), size_t(64), size_t(128)}) {
        const unsigned table_bits = shift + 9;
        std::vector<u64> places;  // one of 2^shift rows, but for its last three entries (a pad's)
        for (u64 place = 0; place + 3 < (u64(1) << 9); place++) places.push_back(row + (place << shift));
        const SwitchBox box = switch_box(places, first, vars);
        CHECK(box.is, "an aligned stretch of a row is not taken for a box");
        if (!box.is) continue;
        for (u64 x : {u64(0), u64(1), u64(77), u64(255)}) {
          std::vector<F> written(zk.ell);
          generator_row(zk, x, written.data(), nullptr);
          const GeneratorRow at = generator_at(zk, x);
          const Tensor v = switch_on_box(box, Tensor::product(at.f, vars, E::from(at.message_scale)), vars, table_bits);
          CHECK(v.vars() == table_bits, "the tensor has another number of variables than the table's index");
          std::vector<E> wanted(size_t(1) << table_bits, E::zero());
          for (size_t i = 0; i < zk.ell; i++) wanted[places[first + i]] = E::from(written[i]);
          bool equal = true;
          for (u64 j = 0; j < wanted.size(); j++) equal = equal && v.at(j) == wanted[j];
          CHECK(equal, "the tensor differs from the written-out generator row, or has weight outside the box");
        }
      }
  std::vector<u64> places;
  for (u64 place = 0; place < 200; place++) places.push_back(place);
  CHECK(!switch_box(places, 3, vars).is && !switch_box(places, 32, vars).is, "a stretch that does not start at a multiple of its length is taken for a box");
  CHECK(!switch_box(places, 192, vars).is, "a stretch that runs beyond the rest is taken for a box");
  places[70] = 300, places.resize(128);
  CHECK(!switch_box(places, 64, vars).is, "a stretch with a gap is taken for a box");
  for (u64 place = 0; place < 128; place++) places[place] = 3 * place;
  CHECK(!switch_box(places, 0, vars).is, "a stretch at distances that are no power of two is taken for a box");
}

int main() {
  boxes();
  const unsigned D = 6;
  const std::vector<Case> cases = {
      {"rest in a main row", Lay::MAIN_ROW, D, 1, 2, false},
      {"rest in mask rows", Lay::MASK_ROWS, D, 1, 2, false},
      {"free room", Lay::FREE_ROOM, D, 1, 2, false},
      {"rest in a main row, two answers", Lay::MAIN_ROW, D, 2, 2, false},
      {"rest in mask rows, two answers", Lay::MASK_ROWS, D, 2, 2, false},
      {"three groups", Lay::MAIN_ROW, D, 1, 3, false},
      {"kept rows, two groups", Lay::MAIN_ROW, D, 1, 2, true},
      {"kept rows, three groups", Lay::MAIN_ROW, D, 1, 3, true},
      {"kept rows, three groups, two answers", Lay::MAIN_ROW, D, 2, 3, true},
  };
  [[maybe_unused]] const unsigned nowhere = ~0u;
  size_t runs = 0, refused = 0;
  for (const Case &t : cases) {
    const World w = toy_world(t);
    const char *fault = iopp_fault(w.codes, w.stages, t.answers);
    if (fault) std::printf("%s: %s\n", t.name, fault);
    CHECK(!fault, "the sizes of the stages do not close");
    if (fault) continue;
    places_as_written(w);
    if (&t == &cases[0]) {  // `breach_mask_rows`: CS aborts on fewer mask rows than FOLD leaves
      const std::vector<u8> honest = prove(t, 200, ~0u).proof;
      for (unsigned rows : {0u, 1u}) {
        CHECK(ends_with("not those that a fold leaves", [&] {
                breach_mask_rows = rows;
                (void)prove(t, 200, ~0u);
              }),
              "the prover's switch takes single rows with fewer than two mask rows");
        CHECK(ends_with("not those that a fold leaves", [&] {
                breach_mask_rows = rows;
                (void)verify(t, 200, honest);
              }),
              "the verifier's switch takes single rows with fewer than two mask rows");
      }
    }
    if (t.kept) {  // `tie_probe`: each of four claim types catches a change that the others miss
      const Proved p = prove(t, 200, ~0u);
      CHECK(p.tie.size() == 4, "the probes of a kept message's tie were not made");
      for (const auto &probe : p.tie) {
        const unsigned seen = (OLD | OLD_SAMPLE | GEN_KEPT | SAMPLE) & ~probe.first;
        if (probe.second != seen) std::printf("%s: hidden from the kinds %u: false are the kinds %u, expected %u\n", t.name, probe.first, probe.second, seen);
        CHECK(probe.second == seen, "a change of a message sent anew that three kinds of rows do not see is not caught by the fourth kind alone");
      }
    }
    if (t.kept) {  // `breach_kept_target`: each fault aborts with its own message, on either side
      const std::vector<u8> honest = prove(t, 200, ~0u).proof;
      const char *const says[] = {"", "CS: one new oracle message for each kept member of the old group", "CS: a kept oracle message has no mask rows", "CS: a kept oracle message is sent anew in the same shape"};
      for (int kind : {1, 2, 3}) {
        CHECK(ends_with(says[kind], [&] {
                breach_kept_target = kind;
                (void)prove(t, 200, ~0u);
              }),
              "the prover's switch takes a target that speaks of the kept members otherwise than the old group");
        CHECK(ends_with(says[kind], [&] {
                breach_kept_target = kind;
                (void)verify(t, 200, honest);
              }),
              "the verifier's switch takes a target that speaks of the kept members otherwise than the old group");
      }
    }
    if (t.lay == Lay::MASK_ROWS)
      CHECK(ends_with("MaskLedger: a proximity test on this oracle message has begun", [&] {
              breach_takes_a_mask = true;
              (void)prove(t, 200, ~0u);
            }),
            "mask rows that hold what a switch carries hand out an entry as a mask");
    if (t.lay == Lay::MAIN_ROW) {
      CHECK(ends_with("a term that only the engine knows", [&] {
              breach_engine_term = 1;
              (void)prove(t, 200, ~0u);
            }),
            "a claim that only the engine knows is laid over the pad of a sample, and the statement takes it");
      CHECK(ends_with("the planted row was taken", [&] {
              breach_engine_term = 2;
              (void)prove(t, 200, ~0u);
            }),
            "control: a claim that only the engine knows, laid beside the pad as every switch lays it, is not taken");
    }
    const u64 seeds = 3;
    for (u64 seed = 200; seed < 200 + seeds; seed++) {
      my_on = true;
      const Proved p = prove(t, seed, nowhere);
      my_on = false;
      CHECK(p.rows_hold(), "a row that a switch leaves does not hold for the honest prover");
      CHECK(p.tables_fine, "free room that is not zero, a pad that is, or an oracle opened at a second list");
      CHECK(verify(t, seed, p.proof), "an honest proof is refused");
      if (const char *pads = toy_pads_fault([&] { (void)prove(t, seed, nowhere); })) {
        std::printf("%s: %s\n", t.name, pads);
        CHECK(false, "a pad is not drawn, or is drawn once for two places or two runs");
      }
      for (unsigned i = 0; i < t.groups; i++) {
        const Proved f = prove(t, seed, i);
        CHECK(f.rows_hold() == (i + 1 == t.groups), "a false target on a closed group must give a false row on the next, and only that");
        CHECK(!verify(t, seed, f.proof), "a false target is accepted");
      }
      if (seed == 200 && (&t == &cases[0] || &t == &cases[7])) {
        for (size_t i = 0; i < p.proof.size(); i++) {
          std::vector<u8> changed = p.proof;
          changed[i] ^= (u8)(1u << (i % 8));
          runs++;
          if (!verify(t, seed, changed)) refused++;
          else std::printf("ACCEPTED a proof with byte %zu changed: %s\n", i, t.name);
        }
      }
    }
  }
  CHECK(refused == runs, "something that had to be refused was accepted");
  std::printf("the second text: %zu rows of switches held entry by entry (%zu entries; %zu free places among them), %zu batches held against the plain sum; differences: %zu\n", my_rows_checked, my_entries_checked, my_free_checked,
              my_batches, my_bad);
  CHECK(my_rows_checked && my_free_checked && my_batches && !my_bad, "what a switch leaves, or Bat::combine, differs from the second text, or the second text has looked at nothing");
  std::printf("cs_test: %s (%zu of %zu refused)\n", failures ? "FAILED" : "ok", refused, runs);
  return failures ? 1 : 0;
}
