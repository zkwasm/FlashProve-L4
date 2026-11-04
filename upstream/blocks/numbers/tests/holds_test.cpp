// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* A `Plan` (blocks/numbers/plan.h: rounds with their errors) from derived parameters (for a
   segment, its closing, `plan_iopp`'s proximity test) must hold at the error target in each field,
   with and without proof of work. There is no negative control here: no run derives one position
   fewer per draw, one step less slack or one combination fewer in the approximate range check
   (`plan_rng`, `ErrorKind::SMALL`). On samples, numbers_test.cpp finds positions() minimal. */
#include <cstdio>

#include "numbers/blocks.h"

using namespace blk;

static bool plan_holds(Setting s, unsigned work) {
  s.set_work(ErrorKind::QUERY, Scope::REQUEST, work);
  s.set_work(ErrorKind::CLOSENESS, Scope::REQUEST, work);
  s.set_work(ErrorKind::SMALL, Scope::REQUEST, work);
  Plan p(s);
  GroupSize g;
  g.name = "group";
  g.main.name = "main", g.main.log_n = 23, g.main.ell = (u64)1 << 22, g.main.r = 301;
  g.mask.name = "mask", g.mask.log_n = 17, g.mask.ell = (u64)1 << 13, g.mask.r = 301, g.mask.over_E = true;
  g.k = 8;
  g = size_group(s, Scope::REQUEST, g, GroupLoad());
  TreeSize tree;
  tree.name = "tree", tree.log_n = 23, tree.leaf_bytes = 1048;
  plan_send(p, "cells", {SampleSize{31, {g}}}, 1);
  plan_rng(p, "cells", 1 << 10);
  plan_bat(p, "closing", {g}, 1000);
  plan_fold(p, "closing", g, 0);
  plan_send(p, "carried", {SampleSize{24, {g}}}, 1);
  const u64 t = plan_loc(p, "closing", g, {tree});
  auto stages = solve_stages(p, g, t, {StageChoice{5, 3, 20, 16}, StageChoice{5, 3, 16, 16}, StageChoice{0, 0, 16, 16}});
  plan_iopp(p, "test", stages, {tree}, 100, 1);
  return p.finish()->holds();
}

int main() {
  bool all = true, none = true;
  for (unsigned work : {0u, 20u})
    for (const Setting &s : {setting_m61(), setting_goldilocks()}) {
      const bool h = plan_holds(s, work);
      all = all && h, none = none && !h;
    }
  std::printf(all ? "ok\n" : "AN HONEST PLAN DOES NOT HOLD\n");
  return all ? 0 : 1;
}
