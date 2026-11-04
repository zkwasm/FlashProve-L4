// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Toy-size cases for the proximity test (blocks/iop/iopp.h), with a prover that runs on any
// `Engine` and the verifier. The prover sends a group's oracle messages (SEND), makes three linear
// claims (`Row`) on them and runs the entire test. With equal seeds two engines must give the same
// proof, byte for byte. The including file must define BLK_PLANS_BY_HAND_FOR_A_TEST first.
#pragma once
#include "chain/closed.h"
#include "iop/tests/iopp_support.h"

namespace blk {

// One oracle message of the group. `main_deg`: 1 for main rows over F, 3 over E.
struct IoppMember {
  size_t main_rows;
  unsigned main_deg;
  size_t mask_rows;
};
// `ride`: each later stage's masks are sent with the code switch (CS; `cs_`, `Switch` in
// identifiers) that creates the stage (`masks_ride`).
struct IoppCase {
  const char *name;
  std::vector<IoppMember> members;
  size_t ell, ell_zk;        // message lengths of the group's two codes
  std::vector<Later> later;  // the last one is {0, 0}
  unsigned draws;            // at every stage
  unsigned answers;          // to every out-of-domain sample
  bool ride;
};

// Public data derived from a case. The group has 2^k main and 2^h mask rows, padding included.
struct IoppWorld {
  Codes codes;
  std::vector<IoppStage> stages;
  IoppChoice choice;
  std::vector<OracleNotice> notices;
  unsigned k = 0, h = 0;
};

inline IoppWorld iopp_toy_world(const IoppCase &t) {
  IoppWorld w;
  w.choice.masks_ride = t.ride;
  u64 main = 0, mask = 0;
  for (const IoppMember &m : t.members) {
    OracleNotice n;
    n.shape.name = "member", n.shape.code = 0, n.shape.main_rows = m.main_rows, n.shape.main_deg = m.main_deg;
    n.shape.mask_rows = m.mask_rows, n.shape.mask_code = 0, n.shape.answers = t.answers;
    w.notices.push_back(n);
    main += m.main_rows ? pow2_from(m.main_rows) : 0, mask += m.mask_rows ? pow2_from(m.mask_rows) : 0;
  }
  w.h = ceil_log2(mask), w.k = std::max(std::max(ceil_log2(main), w.h), 1u);
  const size_t r = tail_for_positions(t.draws);
  w.codes.mask.push_back(toy_code(t.ell_zk, r, 0));
  w.codes.base.push_back(toy_code(t.ell, r, w.codes.mask[0].log_n));
  w.stages.push_back(IoppStage{0, 0, w.k, w.h});
  toy_later_stages(w.codes, w.stages, t.later, t.draws, t.answers);
  return w;
}

inline Plan iopp_toy_plan(const IoppCase &t, const IoppWorld &w) {
  Plan p = toy_empty_plan(t.answers);
  p.send_root("members");
  std::vector<OracleShape> shapes;
  for (const OracleNotice &n : w.notices) shapes.push_back(n.shape);
  toy_plan_sample(p, w.codes, shapes, t.answers);
  p.send_scalars("targets", 3);
  toy_plan_iopp(p, w.codes, w.stages, t.draws, w.choice, false, 3 + (u64)t.answers * w.notices.size());
  return p;
}

// A run's public inputs, bound in the first query of the hash chain (the Fiat-Shamir transcript).
// The claims' coefficients are derived from `seed`.
struct IoppPublic {
  Public<Codes> codes;
  Public<std::vector<OracleNotice>> notices;
  Public<std::vector<IoppStage>> stages;
  Public<IoppChoice> choice;
  Public<u64> seed;
};
inline IoppPublic iopp_toy_bind(Binder &b, const IoppWorld &w, u64 seed) {
  IoppPublic p;
  b.configuration(toy_configuration());
  p.codes = b.parameter("codes", w.codes), p.notices = b.parameter("oracle messages", w.notices);
  p.stages = b.parameter("stages", w.stages), p.choice = b.parameter("choice", w.choice);
  p.seed = b.statement("seed of the rows", seed);
  return p;
}

struct IoppProved {
  std::vector<u8> proof;
  size_t trees = 0;
  bool tables_fine = true;  // see `iopp_toy_prove`
};

// The prover sends each claim's true value (`target`), the first one plus `off_target`.
// `tables_fine`: no oracle message was opened at more than t.draws positions, the most in one list;
// more means a second list.
// With `look`, the run stops after the first code switch and returns no proof; `tables_fine` then
// means: in the new oracle message's mask rows every pad (it masks a sample's answer) is nonzero,
// and the free room (entries with neither carried values nor a pad), if any, is not all zero.
inline IoppProved iopp_toy_prove(const IoppCase &t, u64 seed, const E &off_target, Engine &engine, bool look = false) {
  const IoppWorld w = iopp_toy_world(t);
  const Plan plan = iopp_toy_plan(t, w);
  TestCoins coins(toy_coins_seed(seed)), data(seed + 1);
  Writer proof;
  Binder binder(1, plan);
  const IoppPublic bound = iopp_toy_bind(binder, w, seed);
  ProverRounds ch(binder, toy_random_bytes(seed + 2), proof);
  ProverContext P(bound.codes, ch, engine, coins);
  P.pads = toy_pads;

  std::vector<OracleDraft> drafts;
  for (const OracleNotice &n : ch.get(bound.notices)) drafts.push_back(toy_draft(w.codes, n, engine, coins, data));
  const std::vector<OracleId> ids = send_prove(P, std::move(drafts));
  const GroupId g = P.st.new_group("group", 0, 0);
  for (OracleId o : ids) P.st.join(g, o);
  IoppProved out;
  E off = off_target;
  for (Row &row : toy_rows(P.st, ids, ch.get(bound.seed) + 3)) {
    row.target = row_value(P, row) + off;
    off = E::zero();
    P.ch.send(row.target);  // the hash chain binds each claim's value before the claims are batched
    P.st.add_row(g, std::move(row));
  }
  if (look) {
    GroupLayout L;
    GroupRow row = bat_prove(P, g, &L);
    const FoldShape F = iopp_fold_shape(w.stages[0], L);
    SingleRows rows;
    SingleRowsWitness W;
    fold_prove(P, F, row, fold_mask_make(F, coins), rows, W);
    const IoppStage &n = w.stages[1];
    const SwitchTarget T = switch_target_stage("stage 2", n.code, n.mask_code, n.k, n.h, t.answers);
    const Switched s = cs_prove(P, rows, W, T, P.st.new_group("stage 2", n.code, n.mask_code), {}, row, L);
    const Table &mask = P.behind[s.g].mask;
    const SwitchPlaces places(w.codes, T);
    const size_t carried = switch_parts(P.st, rows).length;  // the free room follows these entries
    bool all_zero = places.size() > carried;
    for (size_t q = carried; q < places.size(); q++) all_zero = all_zero && is_zero(mask.entry(places[q]));
    out.tables_fine = !all_zero;
    for (const PadPlace &p : Send::pad(w.codes, T.shape)) out.tables_fine = out.tables_fine && !is_zero(mask.entry(p.index));
    return out;
  }
  iopp_prove(P, g, bound.stages, bound.choice);
  ch.finish();
  for (const ProverOracle &o : P.behind) out.tables_fine = out.tables_fine && o.opened <= t.draws;
  out.proof = proof.bytes();
  out.trees = P.book.trees();
  return out;
}

// run_verifier() (blocks/chain/closed.h) gives iopp_toy_decide() the channel and handles to the
// first query's values, not the `IoppWorld`, to keep the test's unbound data from the verifier.
struct IoppInputs {
  const IoppWorld &w;
  u64 seed;
};
inline IoppPublic iopp_toy_bind_inputs(Binder &b, const IoppInputs &in) { return iopp_toy_bind(b, in.w, in.seed); }
inline bool iopp_toy_decide(VerifierRounds &ch, const IoppPublic &bound, Notes *) {
  VerifierContext V(bound.codes, ch);
  std::vector<OracleId> ids;
  if (!send_verify(V, ch.get(bound.notices), ids)) return false;
  const GroupId g = V.st.new_group("group", 0, 0);
  for (OracleId o : ids) V.st.join(g, o);
  for (Row &row : toy_rows(V.st, ids, ch.get(bound.seed) + 3)) {
    if (!V.ch.recv(row.target)) return false;
    V.st.add_row(g, std::move(row));
  }
  return iopp_verify(V, g, bound.stages, bound.choice);
}
inline bool iopp_toy_verify(const IoppCase &t, u64 seed, const std::vector<u8> &proof) {
  const IoppWorld w = iopp_toy_world(t);
  const Plan plan = iopp_toy_plan(t, w);
  return run_verifier(1, plan, IoppInputs{w, seed}, iopp_toy_bind_inputs, iopp_toy_decide, proof.data(), proof.size());
}

// D: positions drawn at every stage.
inline std::vector<IoppCase> iopp_toy_cases(unsigned D) {
  return {
      {"two stages", {{2, 1, 1}}, 8, 8, {{0, 0}}, D, 1, false},
      {"three stages, two members", {{3, 1, 1}, {1, 3, 1}}, 16, 8, {{2, 1}, {0, 0}}, D, 1, false},
      {"three stages, two members, masks ride", {{3, 1, 1}, {1, 3, 1}}, 16, 8, {{2, 1}, {0, 0}}, D, 1, true},
      {"four stages, h = 0", {{4, 1, 1}}, 64, 8, {{2, 0}, {1, 0}, {0, 0}}, D, 1, false},
      {"four stages, h = k, masks ride", {{4, 3, 4}}, 64, 8, {{2, 2}, {1, 1}, {0, 0}}, D, 1, true},
      {"two answers", {{3, 1, 2}, {1, 1, 0}}, 16, 8, {{2, 2}, {0, 0}}, D, 2, false},
      {"a member of mask rows alone", {{2, 1, 0}, {0, 1, 2}}, 16, 8, {{1, 1}, {0, 0}}, D, 1, true},
  };
}

}  // namespace blk
