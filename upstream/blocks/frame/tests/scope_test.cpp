// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* Plans the first hash chain of a request's proof file twice, at toy size, the second time with
   sub-proof planners that leave `Scope::ONCE` set. A round's scope selects its bits of proof of
   work (`ONCE`: per commitment). `req_plan` (blocks/iop/req.h) and `file_proof`
   (blocks/frame/files.h) must restore the scope after each planner, so both plans must have the
   same rounds, all with `Scope::REQUEST`. */
#include <cstdio>

#include "frame/tests/toy_shapes.h"

using namespace blk;

int main() {
  const std::vector<std::string> files = {toy_file(), toy_model()};
  Parameters par;
  std::string why;
  if (!read_parameters(files[0], par, why)) return std::printf("FAILED: the toy's parameter file is not read: %s\n", why.c_str()), 1;
  const Shapes shapes = toy_shapes(files, par);
  const Subproofs &sub = shapes.sub;
  const Sized z = sized(par, sub);
  if (!shapes.refused.empty() || !z.refused.empty() || !sub.after_links) return std::printf("FAILED: the toy's files are not admitted, or its sub-proofs lay nothing behind the links\n"), 1;
  Subproofs leaves_a_scope = sub;
  leaves_a_scope.plan = [&](Plan &p, size_t link, unsigned moment) { sub.plan(p, link, moment), p.begin_scope(Scope::ONCE); };
  leaves_a_scope.after_links = [&](Plan &p) { sub.after_links(p), p.begin_scope(Scope::ONCE); };
  const Planned a = file_proof(z, sub, false).chains[0].plan, b = file_proof(z, leaves_a_scope, false).chains[0].plan;
  size_t apart = a->holds() && a->rounds_.size() == b->rounds_.size() ? 0 : 1;
  for (size_t i = 0; i < a->rounds_.size() && i < b->rounds_.size(); ++i) {
    const PlanRound &x = a->rounds_[i], &y = b->rounds_[i];
    if (x.name != y.name || x.scope != Scope::REQUEST || y.scope != Scope::REQUEST || x.work_bits != y.work_bits || x.draw_positions != y.draw_positions || Real::cmp(x.error, y.error) != 0) ++apart;
  }
  std::printf(apart ? "FAILED: %zu of %zu rounds are laid otherwise behind a planner that leaves another scope\n" : "ok (%zu of %zu rounds apart)\n", apart, a->rounds_.size());
  return apart ? 1 : 0;
}
