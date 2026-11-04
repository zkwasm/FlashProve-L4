// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Runs the toy proof's first steps (blocks/toy/product.h) in one honest run, prover against
// verifier through the proof's bytes: SEND (sends the oracle message), ZERO and PSC (sumcheck),
// BAT (batches the resulting linear claims into one `GroupRow`, blocks/iop/relation.h).
// Of the proximity test only BAT runs. In place of the rest, both sides' claims must agree
// (`target` and `d`; of `c` only the number of tensors), and the verifier's claim must hold on the
// prover's tables.
#include <cstdio>

#include "engine/engine_cpu.h"
#include "toy/product.h"
#include "toy/toy_coins.h"

using namespace blk;

static bool run(size_t &bytes) {
  ToyShape shape;
  shape.set = setting_goldilocks();
  shape.codes.base = {Code{7, 16, 9}};
  shape.codes.mask = {Code{6, 16, 9}};
  Plan plan(shape.set);
  toy_plan_preamble(plan, shape, {}, no_sized_group());
  plan_bat(plan, "a, b, c", no_sized_group(), toy_rows_batched(shape.set));

  ToyCoins coins(11);
  Table abc(1, 3, shape.codes.base[0].ell);
  coins.fill(abc.a.data(), 2 * abc.len);
  for (size_t i = 0; i < abc.len; i++) abc.set(2, i, abc.at(0, i) * abc.at(1, i));

  EngineCpu engine;
  Writer proof;
  Binder prover_binder(TOY_LABEL, plan.finish());
  const ToyPublic mine = toy_bind(prover_binder, shape, {});
  ProverRounds pch(prover_binder, random_digest(coins), proof);
  ProverContext pr(mine.codes, pch, engine, coins);
  GroupRow prow;
  GroupLayout layout;
  FoldMask none; // no proximity test follows: no fold, no mask
  const GroupId pg = toy_prove_preamble(pr, mine, abc, none);
  prow = bat_prove(pr, pg, &layout);
  pch.finish();
  bytes = proof.size();

  Reader reader(proof.data(), proof.size());
  Binder verifier_binder(TOY_LABEL, plan.finish());
  const ToyPublic theirs = toy_bind(verifier_binder, shape, {});
  VerifierSession session;
  VerifierRounds vch(verifier_binder, reader, session);
  VerifierContext vf(theirs.codes, vch);
  GroupId vg = 0;
  GroupRow vrow;
  FoldMask none_either;
  if (!toy_verify_preamble(vf, theirs, vg, none_either) || !bat_verify(vf, vg, vrow) || !vch.finish()) return false;
  if (vrow.target != prow.target || vrow.d != prow.d || vrow.c.tensors.size() != prow.c.tensors.size()) return false;

  // <c, abc> + <d, mask> = target
  E left = E::zero();
  std::vector<E> c(abc.len << layout.k, E::zero());
  vrow.c.add_to(c.data(), layout.k + layout.log_ell);
  for (u64 j = 0; j < c.size(); j++) left += c[j] * abc.at(j & ((u64(1) << layout.k) - 1), j >> layout.k);
  const Table &mask = pr.behind[0].mask;
  for (u64 j = 0; j < vrow.d.size(); j++) left += vrow.d[j] * mask.entry(j);
  return left == vrow.target;
}

int main() {
  size_t bytes = 0;
  if (!run(bytes)) {
    std::printf("FAILED: the honest run\n");
    return 1;
  }
  std::printf("preamble: ok, %zu bytes\n", bytes);
  return 0;
}
