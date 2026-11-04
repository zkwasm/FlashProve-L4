// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

/* What a planning call (`-plan`; checks no proof) prints: the sizes of the groups (rows tested
   together), how many columns of committed tables are drawn, the error accounting, each file's
   length and what its proximity terms (printed: `closeness`) rest on. */
#pragma once
#include <cstdio>
#include <string>
#include <vector>

#include "frame/fields.h"
#include "frame/files.h"
#include "numbers/accounting.h"

namespace blk {

inline void print_test(const Setting &set, const char *name, const TestSizes &x) {
  const GroupSize &g = x.group;
  const Analysis a = analysis(set, g);
  std::printf("group: %s: code 2^%u, message %llu + %llu random, 2^%u rows; mask code 2^%u, %llu + %llu; distance %.6f, ", name, g.main.log_n,
              (unsigned long long)g.main.ell, (unsigned long long)g.main.r, g.k, g.mask.log_n, (unsigned long long)g.mask.ell, (unsigned long long)g.mask.r,
              a.at.delta.lo.value());
  if (g.m) std::printf("m = %llu, ", (unsigned long long)g.m);
  std::printf("list %s, %llu positions; rule: %s; its closeness terms rest on %s\n", a.list.dec().c_str(), (unsigned long long)x.t, name_of(g.rule),
              rests_on(weakest_source(a)));
  for (size_t i = 1; i < x.stages.size(); ++i) {
    const GroupSize &s = x.stages[i].g;
    std::printf("group:   stage %zu: 2^%u rows of %llu + %llu in 2^%u; 2^%u mask rows of %llu + %llu in 2^%u; list %s, %llu positions\n", i + 1, s.k,
                (unsigned long long)s.main.ell, (unsigned long long)s.main.r, s.main.log_n, s.h, (unsigned long long)s.mask.ell, (unsigned long long)s.mask.r,
                s.mask.log_n, analysis(set, s).list.dec().c_str(), (unsigned long long)x.stages[i].t);
  }
}

inline void print_size(const Planned &p, const char *part) {
  const int i = p->part_index(part);
  if (i < 0) return;
  const PartSize z = p->size_of((unsigned)i);
  std::printf("length: %s: at most %llu bytes, %.0f on average over the drawn positions\n", part, (unsigned long long)z.bytes, z.expected);
}

/* what: "key", "leaf" (printed: `entry`) or "request" (all three files: "key, entry and proof").
   z.refused must be empty (unchecked). rounds, fields: also list these.
   `d.link`: a request's segments; `z.link`: the LINK proof between two tables.
   True: each round meets the target, proof of work counted. */
inline bool print_plan(const std::string &what, const Sized &z, const Subproofs &sub, bool rounds, bool fields) {
  const Derived &d = z.d;
  std::vector<FilePlan> files;
  if (what == "key" || what == "request") files.push_back(file_key(z, !fields));
  if (what == "leaf" || what == "request") files.push_back(file_entry(z, !fields));
  if (what == "request") files.push_back(file_proof(z, sub, !fields));
  const Planned p = merged(d.set, files);
  if (what == "request") print_test(d.set, "link", d.link), print_test(d.set, "last test", d.last);
  for (const LinSized *x : {&z.batch, &z.once}) {
    if (what == "leaf" || (what == "key" && x == &z.batch)) continue;
    std::printf("group: %s\n", x == &z.once ? "in the committer's files:" : "in a batch:");
    print_test(d.set, "folded table", x->folded);
    if (x->S.shared_split_needed()) print_test(d.set, "Omega", x->omega);
    print_test(d.set, "own test", x->own);
    if (x == &z.once) print_test(d.set, "ranges' multiplicities", z.ranges.test);  // the committer's range proof has its own proximity test
  }
  if (what != "key") print_test(d.set, "folded tables of a link", z.link.linked);
  if (what != "leaf") std::printf("table: %llu columns in the one-time test\n", (unsigned long long)z.once.S.fold.columns);
  if (what != "key")
    std::printf("table: %llu columns of the first table and %llu of the further one in a link\n", (unsigned long long)z.link.S.columns_first,
                (unsigned long long)z.link.S.columns_further);
  if (what == "request") {
    std::printf("table: %llu columns a batch; a stored copy bears %llu batches\n", (unsigned long long)z.batch.S.fold.columns, (unsigned long long)z.batch.batches);
    std::printf("smallness test: %u combinations\n", rng_combinations(d.set, Scope::REQUEST));
  }
  const Accounting a = account(p);
  if (rounds) print_rounds(a);
  print_accounting(a, what == "request" ? "key, entry and proof" : what == "key" ? "the key" : "an entry", rounds);
  for (unsigned i = 0; i < p->parts.size(); ++i) {  // which files rest on the theorem on codes?
    const Accounting f = account(p, (int)i);
    std::printf("footing: %s: of its rounds with a closeness term %llu rest on a lemma of the paper that this protocol builds on (see THIRD_PARTY.md), proved there, %llu on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given", p->parts[i].c_str(),
                (unsigned long long)f.closeness[0], (unsigned long long)f.closeness[1]);
    std::printf("\n");
  }
  for (const char *part : {"key", "entry", "proof"}) print_size(p, part);
  if (fields)
    for (unsigned i = 0; i < p->parts.size(); ++i) print_fields(p, i);
  return a.holds;
}

}  // namespace blk
